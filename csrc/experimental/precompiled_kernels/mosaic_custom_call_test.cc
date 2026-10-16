/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "csrc/experimental/precompiled_kernels/mosaic_custom_call.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/experimental/precompiled_kernels/kernel_test_utils.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "gtest/gtest.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/hlo_runner.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"

namespace torch_tpu {
namespace {

TEST(DynamicMosaic, CopiesDifferentShapesWithOneExecutable) {
  TT_ASSERT_OK_AND_ASSIGN(auto client, CreateKernelTestClient());
  xla::HloRunner runner(std::move(client));
  // Copy from src to dst, with runtime-varying dimensions.
  const kiln::Shape dynamic_2d = kiln::f32(kiln::kDynamic, kiln::kDynamic);
  TT_ASSERT_OK_AND_ASSIGN(
      kiln::Kernel kernel,
      kiln::CreateKernel(
          kiln::GridSpec{.name = "main"}, kiln::InSpec::Hbm(dynamic_2d),
          kiln::InSpec::Hbm(dynamic_2d), kiln::OutSpec::Hbm(dynamic_2d),
          [](kiln::Ref src, kiln::Ref /*destination*/, kiln::Ref dst) {
            kiln::Ref tile = kiln::AllocaVmem(kiln::f32(8, 128));
            kiln::Semaphore sem = kiln::AllocaDmaSemaphore();
            kiln::For(0, src.Dim(0), 8, [&](kiln::Scalar row) {
              kiln::For(0, src.Dim(1), 128, [&](kiln::Scalar col) {
                kiln::Scalar r = kiln::MultipleOf(row, 8);
                kiln::Scalar c = kiln::MultipleOf(col, 128);
                kiln::AsyncCopy(src(kiln::Slice(r, 8), kiln::Slice(c, 128)),
                                tile, sem)
                    .Wait();
                kiln::AsyncCopy(
                    tile, dst(kiln::Slice(r, 8), kiln::Slice(c, 128)), sem)
                    .Wait();
              });
            });
          }));
  TT_ASSERT_OK_AND_ASSIGN(const std::string serialized_kernel,
                          std::move(kernel).Serialize());
  auto payload = xla::ShapeUtil::MakeShape(xla::F32, {8, 128});
  payload.mutable_layout()->add_tiles()->add_dimensions(8).add_dimensions(128);
  auto dynamic = payload;
  for (int d = 0; d < 2; ++d) {
    dynamic.set_dynamic_dimension(d, true);
    dynamic.set_dimensions(d, xla::Shape::kUnboundedSize);
  }
  auto metadata = xla::ShapeUtil::MakeShape(xla::S32, {2});
  metadata.mutable_layout()->add_tiles()->add_dimensions(128);
  TT_ASSERT_OK_AND_ASSIGN(
      auto hlo, BuildMosaicCustomCall(
                    serialized_kernel, {dynamic, dynamic, metadata, metadata},
                    dynamic, {2, 3, -1, -1}, /*aliased_parameter=*/1));
  TT_ASSERT_OK_AND_ASSIGN(
      auto executable,
      runner.CreateExecutable(std::move(hlo), /*run_hlo_passes=*/false));
  for (const std::vector<int64_t>& dims :  // INT_VEC_OK
       {std::vector<int64_t>{8, 128},      // INT_VEC_OK
        {16, 256},
        {24, 384},
        {3, 7},
        {9, 129},
        {0, 128},
        {8, 0}}) {
    TT_ASSERT_OK_AND_ASSIGN(
        auto input,
        xla::Literal::Make(xla::ShapeUtil::MakeShape(xla::F32, dims)));
    // Fill with a repeating pattern.
    for (int64_t i = 0; i < input.element_count(); ++i) {
      input.data<float>()[i] = static_cast<float>(i % 19) - 9.f;
    }
    auto extents = xla::LiteralUtil::CreateR1<int32_t>(
        {static_cast<int32_t>(dims[0]), static_cast<int32_t>(dims[1])});
    // Transfer a concrete, prefix-free destination; only operand 1 is donated.
    auto concrete = input.shape();
    *concrete.mutable_layout() = payload.layout();
    auto destination = xla::LiteralUtil::CreateFull<float>(dims, -123.f);
    std::vector<xla::ShapeLayout> layouts = {
        xla::ShapeLayout(concrete), xla::ShapeLayout(concrete),
        xla::ShapeLayout(metadata), xla::ShapeLayout(metadata)};
    TT_ASSERT_OK_AND_ASSIGN(
        auto buffers, runner.TransferLiteralsToDefaultDevice(
                          layouts, {&input, &destination, &extents, &extents}));
    xla::ExecuteOptions options;
    options.strict_shape_checking = false;
    options.non_donatable_input_indices = {0, 2, 3};
    TT_ASSERT_OK_AND_ASSIGN(
        auto outputs,
        runner.ExecuteWithDeviceBuffers(executable.get(), buffers, &options));
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_TRUE(outputs[0]->on_device_shape().is_static());
    EXPECT_TRUE(xla::ShapeUtil::Compatible(outputs[0]->on_device_shape(),
                                           input.shape()));
    TT_ASSERT_OK_AND_ASSIGN(auto result, outputs[0]->ToLiteral().Await());
    EXPECT_EQ(*result, input);
  }
}

}  // namespace
}  // namespace torch_tpu
