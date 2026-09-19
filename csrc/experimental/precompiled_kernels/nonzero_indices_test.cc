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

#include "csrc/experimental/precompiled_kernels/nonzero_indices.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/experimental/precompiled_kernels/kernel_test_utils.h"
#include "csrc/experimental/precompiled_kernels/mosaic_custom_call.h"
#include "csrc/internal/mosaic/op_builders.h"
#include "gtest/gtest.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

// Spread exactly K nonzeros across the input, including its last element.
// Exercise NaN, infinity, negatives and subnormals; only signed zeros are zero.
xla::Literal MakeInput(int64_t rows, int64_t cols, int64_t k) {
  xla::Literal input(xla::ShapeUtil::MakeShape(xla::F32, {rows, cols}));
  const float nonzeros[] = {INFINITY, -2.f, 1e-40f};
  const int64_t size = rows * cols;
  for (int64_t i = 0; i < size; ++i) {
    bool selected = (i + 1) * k / size != i * k / size;
    input.data<float>()[i] = selected ? nonzeros[i % 3] : (i % 2 ? -0.f : 0.f);
  }
  if (k) input.data<float>()[size - 1] = NAN;
  return input;
}

// Compute coordinates independently on the CPU, in logical row-major order.
xla::Literal ReferenceIndices(const xla::Literal& input) {
  int64_t count = 0;
  for (int64_t i = 0; i < input.element_count(); ++i) {
    count += input.data<float>()[i] != 0.f;
  }
  xla::Literal result(xla::ShapeUtil::MakeShape(xla::S64, {count, 2}));
  const int64_t cols = input.shape().dimensions(1);
  for (int64_t i = 0, out = 0; i < input.element_count(); ++i) {
    if (input.data<float>()[i] != 0.f) {
      result.Set<int64_t>({out, 0}, i / cols);
      result.Set<int64_t>({out, 1}, i % cols);
      ++out;
    }
  }
  return result;
}

class NonzeroIndicesTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (runner_ != nullptr) return;
    TT_ASSERT_OK_AND_ASSIGN(auto owned_client, CreateKernelTestClient());
    client_ = owned_client.get();
    runner_ = new xla::HloRunner(std::move(owned_client));

    auto make_unbounded = [](xla::PrimitiveType type, xla::Tile tile) {
      xla::Shape s = xla::ShapeUtil::MakeShape(
          type, {xla::Shape::kUnboundedSize, xla::Shape::kUnboundedSize},
          {true, true});
      *s.mutable_layout() = xla::Layout({1, 0}, {tile});
      return s;
    };
    xla::Shape input = make_unbounded(xla::F32, xla::Tile({8, 128}));
    xla::Shape output_view = make_unbounded(xla::S32, xla::Tile({1, 128}));
    xla::Shape dims = xla::ShapeUtil::MakeShape(xla::S32, {2});
    *dims.mutable_layout() = xla::Layout({0}, {xla::Tile({128})});
    parameters_ = new std::vector<xla::Shape>{input, output_view, dims, dims};

    auto context = mlir::torch_tpu::CreateMlirContextWithDialects();
    auto module = BuildNonzeroIndices(*context);
    TT_ASSERT_OK_AND_ASSIGN(
        auto hlo,
        BuildMosaicCustomCall(*module, *parameters_, output_view,
                              {2, 3, -1, -1}, /*aliased_parameter=*/1));
    TT_ASSERT_OK_AND_ASSIGN(
        auto exec,
        runner_->CreateExecutable(std::move(hlo), /*run_hlo_passes=*/false));
    executable_ = exec.release();
  }

  static absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> AllocateIndexOutput(
      const xla::Shape& output) {
    TT_ASSIGN_OR_RETURN(
        auto* memory,
        client_->addressable_devices()[0]->default_memory_space());
    TT_ASSIGN_OR_RETURN(auto buffer,
                        client_->CreateUninitializedBuffer(output, memory));
    TT_ASSIGN_OR_RETURN(auto bytes, buffer->GetOnDeviceSizeInBytes());
    return buffer->Bitcast(
        xla::S32, {static_cast<int64_t>(bytes / (128 * sizeof(int32_t))), 128},
        &(*parameters_)[1].layout());
  }

  static absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
  PrepareOperands(const xla::Literal& input, const xla::Shape& output) {
    TT_ASSIGN_OR_RETURN(auto destination, AllocateIndexOutput(output));
    auto input_dims = xla::LiteralUtil::CreateR1<int32_t>(
        {static_cast<int32_t>(input.shape().dimensions(0)),
         static_cast<int32_t>(input.shape().dimensions(1))});
    auto output_dims = xla::LiteralUtil::CreateR1<int32_t>(
        {static_cast<int32_t>(destination->on_device_shape().dimensions(0)),
         128});
    auto concrete = input.shape();
    *concrete.mutable_layout() = (*parameters_)[0].layout();
    TT_ASSIGN_OR_RETURN(auto buffers, runner_->TransferLiteralsToDefaultDevice(
                                          {xla::ShapeLayout(concrete),
                                           xla::ShapeLayout((*parameters_)[2]),
                                           xla::ShapeLayout((*parameters_)[3])},
                                          {&input, &input_dims, &output_dims}));
    for (const auto& buffer : buffers) {
      TT_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
    }
    buffers.insert(buffers.begin() + 1, std::move(destination));
    return buffers;
  }

  void CheckMosaicKernelCorrectness(int64_t rows, int64_t cols, int64_t k,
                                    xla::Literal* out = nullptr) {
    SCOPED_TRACE(testing::Message() << rows << "x" << cols << " K=" << k);
    ASSERT_GT(k, 2);
    auto input = MakeInput(rows, cols, k);
    auto expected = ReferenceIndices(input);
    auto output_shape = expected.shape();
    TT_ASSERT_OK_AND_ASSIGN(auto layout,
                            client_->GetDefaultLayout(xla::S64, {k, 2}));
    // XLA default layout for S64[K,2] is {0,1:T(2,128)} only when K > 2
    // (for K <= 2 it is {1,0:T(2,128)}, which the kernel does not support).
    EXPECT_EQ(layout, xla::Layout({0, 1}, {xla::Tile({2, 128})}));
    *output_shape.mutable_layout() = layout;
    TT_ASSERT_OK_AND_ASSIGN(auto buffers, PrepareOperands(input, output_shape));

    xla::ExecuteOptions options;
    options.strict_shape_checking = false;
    TT_ASSERT_OK_AND_ASSIGN(auto outputs, runner_->ExecuteWithDeviceBuffers(
                                              executable_, buffers, &options));
    TT_ASSERT_OK_AND_ASSIGN(
        auto native,
        outputs[0]->Bitcast(xla::S64, {k, 2}, &output_shape.layout()));
    TT_ASSERT_OK_AND_ASSIGN(auto actual, native->ToLiteral().Await());
    EXPECT_EQ(*actual, expected);
    TT_ASSERT_OK_AND_ASSIGN(auto unchanged, buffers[0]->ToLiteral().Await());
    if (input.element_count()) {
      EXPECT_EQ(std::memcmp(input.untyped_data(), unchanged->untyped_data(),
                            input.element_count() * sizeof(float)),
                0);
    }
    if (out != nullptr) *out = std::move(*actual);
  }

  static inline xla::PjRtClient* client_ = nullptr;
  static inline xla::HloRunner* runner_ = nullptr;
  static inline xla::OpaqueExecutable* executable_ = nullptr;
  static inline std::vector<xla::Shape>* parameters_ = nullptr;
};

TEST_F(NonzeroIndicesTest, ReusesExecutableAcrossShapesAndCounts) {
  const std::pair<int64_t, int64_t> shapes[] = {
      {1, 3}, {3, 1}, {3, 7}, {9, 129}, {17, 257}, {1, 16385}};
  for (auto [rows, cols] : shapes) {
    const int64_t size = rows * cols;
    const int64_t ks[] = {3, 127, 128, 129, 1023, 1024, 1025, size};
    for (int64_t k : ks) {
      if (k <= size) CheckMosaicKernelCorrectness(rows, cols, k);
    }
  }
}

}  // namespace
}  // namespace torch_tpu
