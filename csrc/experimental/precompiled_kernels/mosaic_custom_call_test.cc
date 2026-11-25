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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/experimental/precompiled_kernels/kernel_test_utils.h"
#include "csrc/experimental/precompiled_kernels/kernel_utils.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "csrc/pjrt/runtime_shape_metadata_pool.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/hlo_runner.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

// Returns a kernel that copies from src to dst, with runtime-varying
// dimensions.
absl::StatusOr<kiln::Kernel> BuildCopyKernel() {
  const kiln::Shape dynamic_2d = kiln::f32(kiln::kDynamic, kiln::kDynamic);
  return kiln::CreateKernel(
      kiln::GridSpec{.name = "main"}, kiln::InSpec::Hbm(dynamic_2d),
      kiln::InSpec::Hbm(dynamic_2d), kiln::OutSpec::Hbm(dynamic_2d),
      [](kiln::HbmRef src, kiln::HbmRef /*destination*/, kiln::HbmRef dst) {
        kiln::VmemRef tile = kiln::AllocaVmem(kiln::f32(8, 128));
        kiln::Semaphore sem = kiln::AllocaDmaSemaphore();
        kiln::For(0, src.Dim(0), 8, [&](kiln::Scalar row) {
          kiln::For(0, src.Dim(1), 128, [&](kiln::Scalar col) {
            kiln::Scalar r = kiln::MultipleOf(row, 8);
            kiln::Scalar c = kiln::MultipleOf(col, 128);
            kiln::AsyncCopy(src(kiln::Slice(r, 8), kiln::Slice(c, 128)), tile,
                            sem)
                .Wait();
            kiln::AsyncCopy(tile, dst(kiln::Slice(r, 8), kiln::Slice(c, 128)),
                            sem)
                .Wait();
          });
        });
      });
}

TEST(DynamicMosaic, CopiesDifferentShapesWithOneExecutable) {
  TT_ASSERT_OK_AND_ASSIGN(auto client, CreateKernelTestClient());
  xla::HloRunner runner(std::move(client));
  TT_ASSERT_OK_AND_ASSIGN(kiln::Kernel kernel, BuildCopyKernel());
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

TEST(DynamicMosaic, CopiesHostMemoryMetadataToDevice) {
  TT_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<xla::PjRtClient> client,
                          CreateKernelTestClient());
  TT_ASSERT_OK_AND_ASSIGN(kiln::Kernel kernel, BuildCopyKernel());
  const xla::Layout layout({1, 0}, {xla::Tile({8, 128})});
  const xla::Shape dynamic =
      MakeUnboundedShape(xla::F32, /*rank=*/2, layout.tiles());
  // The dimensions of the operands are in pinned host memory.
  const xla::Shape metadata = MakeRuntimeDimsShape(/*rank=*/2);
  TT_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      CompileMosaicKernel(*client, std::make_unique<xla::CompileOptions>(),
                          std::move(kernel),
                          {dynamic, dynamic, metadata, metadata}, dynamic,
                          {2, 3, -1, -1}, /*aliased_parameter=*/1));
  EXPECT_THAT(
      executable->GetParameterMemoryKinds(),
      absl_testing::IsOkAndHolds(testing::ElementsAre(testing::ElementsAre(
          "device", "device", "pinned_host", "pinned_host"))));

  xla::PjRtDevice* const device = client->addressable_devices()[0];
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtMemorySpace* const memory_space,
                          device->default_memory_space());
  // Each execution overwrites the two metadata buffers of the one before.
  const std::shared_ptr<RuntimeShapeMetadataPool> pool =
      RuntimeShapeMetadataPool::Create();
  std::set<const xla::PjRtBuffer*> seen_metadata_buffers;
  for (const std::vector<int64_t>& dims :  // INT_VEC_OK
       {std::vector<int64_t>{8, 128},      // INT_VEC_OK
        {16, 256},
        {3, 7},
        {9, 129},
        {0, 128}}) {
    TT_ASSERT_OK_AND_ASSIGN(
        auto input,
        xla::Literal::Make(xla::ShapeUtil::MakeShape(xla::F32, dims)));
    for (int64_t i = 0; i < input.element_count(); ++i) {
      input.data<float>()[i] = static_cast<float>(i % 19) - 9.f;
    }
    const auto destination = xla::LiteralUtil::CreateFull<float>(dims, -123.f);
    const auto to_device = [&](const xla::Literal& literal) {
      return client->BufferFromHostBuffer(
          literal.untyped_data(), xla::F32, dims, /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, memory_space, &layout);
    };
    TT_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<xla::PjRtBuffer> input_buffer,
                            to_device(input));
    TT_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<xla::PjRtBuffer> destination_buffer,
        to_device(destination));
    const std::vector<int32_t> extents(dims.begin(), dims.end());
    TT_ASSERT_OK_AND_ASSIGN(
        const std::shared_ptr<xla::PjRtBuffer> input_extents,
        pool->Acquire(extents, metadata, *device));
    TT_ASSERT_OK_AND_ASSIGN(
        const std::shared_ptr<xla::PjRtBuffer> destination_extents,
        pool->Acquire(extents, metadata, *device));
    seen_metadata_buffers.insert(input_extents.get());
    seen_metadata_buffers.insert(destination_extents.get());

    const std::vector<xla::PjRtBuffer*> arguments = {
        input_buffer.get(), destination_buffer.get(), input_extents.get(),
        destination_extents.get()};
    xla::ExecuteOptions options;
    options.strict_shape_checking = false;
    // Only the aliased destination (param 1) is donated.
    options.non_donatable_input_indices = {0, 2, 3};
    TT_ASSERT_OK_AND_ASSIGN(
        auto outputs,
        executable->Execute(absl::MakeConstSpan(&arguments, 1), options));
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].size(), 1);
    // The execution is complete once its result is read.
    TT_ASSERT_OK_AND_ASSIGN(auto result, outputs[0][0]->ToLiteral().Await());
    EXPECT_EQ(*result, input);
  }
  EXPECT_EQ(seen_metadata_buffers.size(), 2);
}

TEST(DynamicMosaic, StartsEveryHostMemoryCopyBeforeItWaitsForOne) {
  const xla::Shape tensor = xla::ShapeUtil::MakeShape(xla::F32, {8, 128});
  const xla::Shape metadata = MakeRuntimeDimsShape(/*rank=*/2);
  TT_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<xla::HloModule> hlo,
      BuildMosaicCustomCall("", {tensor, tensor, metadata, metadata}, tensor,
                            {2, 3, -1, -1}, /*aliased_parameter=*/-1));

  std::vector<xla::HloOpcode> opcodes;
  for (const xla::HloInstruction* const instruction :
       hlo->schedule().sequence(hlo->entry_computation()).instructions()) {
    opcodes.push_back(instruction->opcode());
  }
  // The copies overlap only if both start before the wait for the first.
  const auto first_wait = absl::c_find(opcodes, xla::HloOpcode::kCopyDone);
  EXPECT_EQ(std::count(opcodes.begin(), first_wait, xla::HloOpcode::kCopyStart),
            2);
  EXPECT_EQ(absl::c_count(opcodes, xla::HloOpcode::kCopyDone), 2);
  EXPECT_EQ(opcodes.back(), xla::HloOpcode::kCustomCall);
}

TEST(DynamicMosaic, BitcastsPredOperandToU8) {
  TT_ASSERT_OK_AND_ASSIGN(auto client, CreateKernelTestClient());
  xla::HloRunner runner(std::move(client));
  // Copy from a u8 src to a u8 dst, with runtime-varying dimensions.
  const kiln::Shape dynamic_2d = kiln::u8(kiln::kDynamic, kiln::kDynamic);
  TT_ASSERT_OK_AND_ASSIGN(
      kiln::Kernel kernel,
      kiln::CreateKernel(
          kiln::GridSpec{.name = "main"}, kiln::InSpec::Hbm(dynamic_2d),
          kiln::InSpec::Hbm(dynamic_2d), kiln::OutSpec::Hbm(dynamic_2d),
          [](kiln::HbmRef src, kiln::HbmRef /*destination*/, kiln::HbmRef dst) {
            kiln::VmemRef tile = kiln::AllocaVmem(kiln::u8(8, 128));
            kiln::Semaphore sem = kiln::AllocaDmaSemaphore();
            kiln::For(
                0, src.Dim(0), 8, [&src, &dst, &tile, &sem](kiln::Scalar row) {
                  kiln::For(0, src.Dim(1), 128,
                            [&src, &dst, &tile, &sem, row](kiln::Scalar col) {
                              kiln::Scalar r = kiln::MultipleOf(row, 8);
                              kiln::Scalar c = kiln::MultipleOf(col, 128);
                              kiln::AsyncCopy(
                                  src(kiln::Slice(r, 8), kiln::Slice(c, 128)),
                                  tile, sem)
                                  .Wait();
                              kiln::AsyncCopy(
                                  tile,
                                  dst(kiln::Slice(r, 8), kiln::Slice(c, 128)),
                                  sem)
                                  .Wait();
                            });
                });
          }));
  TT_ASSERT_OK_AND_ASSIGN(const std::string serialized_kernel,
                          std::move(kernel).Serialize());
  auto make_byte_packed_2d = [](xla::PrimitiveType type) {
    xla::Shape shape = xla::ShapeUtil::MakeShape(type, {8, 128});
    shape.mutable_layout()->add_tiles()->add_dimensions(8).add_dimensions(128);
    shape.mutable_layout()->add_tiles()->add_dimensions(4).add_dimensions(1);
    return shape;
  };
  auto make_unbounded = [](xla::Shape shape) {
    for (int d = 0; d < 2; ++d) {
      shape.set_dynamic_dimension(d, true);
      shape.set_dimensions(d, xla::Shape::kUnboundedSize);
    }
    return shape;
  };
  const xla::Shape pred_payload = make_byte_packed_2d(xla::PRED);
  const xla::Shape u8_payload = make_byte_packed_2d(xla::U8);
  const xla::Shape dynamic_pred = make_unbounded(pred_payload);
  const xla::Shape dynamic_u8 = make_unbounded(u8_payload);
  xla::Shape metadata = xla::ShapeUtil::MakeShape(xla::S32, {2});
  metadata.mutable_layout()->add_tiles()->add_dimensions(128);

  TT_ASSERT_OK_AND_ASSIGN(
      auto hlo,
      BuildMosaicCustomCall(
          serialized_kernel, {dynamic_pred, dynamic_u8, metadata, metadata},
          dynamic_u8, {2, 3, -1, -1}, /*aliased_parameter=*/1,
          [&dynamic_u8](xla::HloComputation::Builder& builder,
                        absl::Span<xla::HloInstruction*> operands) {
            operands[0] = builder.AddInstruction(
                xla::HloInstruction::CreateBitcast(dynamic_u8, operands[0]));
          }));
  TT_ASSERT_OK_AND_ASSIGN(
      auto executable,
      runner.CreateExecutable(std::move(hlo), /*run_hlo_passes=*/false));
  for (const std::vector<int64_t>& dims :  // INT_VEC_OK
       {std::vector<int64_t>{8, 128},      // INT_VEC_OK
        {16, 256},
        {3, 7},
        {9, 129},
        {0, 128}}) {
    TT_ASSERT_OK_AND_ASSIGN(
        auto input,
        xla::Literal::Make(xla::ShapeUtil::MakeShape(xla::PRED, dims)));
    TT_ASSERT_OK_AND_ASSIGN(
        auto expected,
        xla::Literal::Make(xla::ShapeUtil::MakeShape(xla::U8, dims)));
    for (int64_t i = 0; i < input.element_count(); ++i) {
      const bool value = (i % 7 == 0) || (i % 3 == 1);
      input.data<bool>()[i] = value;
      expected.data<uint8_t>()[i] = value ? uint8_t{1} : uint8_t{0};
    }
    xla::Literal extents = xla::LiteralUtil::CreateR1<int32_t>(
        {static_cast<int32_t>(dims[0]), static_cast<int32_t>(dims[1])});
    xla::Shape concrete_pred = input.shape();
    *concrete_pred.mutable_layout() = pred_payload.layout();
    xla::Shape concrete_u8 = expected.shape();
    *concrete_u8.mutable_layout() = u8_payload.layout();
    xla::Literal destination = xla::LiteralUtil::CreateFull<uint8_t>(dims, 165);
    std::vector<xla::ShapeLayout> layouts = {
        xla::ShapeLayout(concrete_pred), xla::ShapeLayout(concrete_u8),
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
                                           expected.shape()));
    TT_ASSERT_OK_AND_ASSIGN(auto result, outputs[0]->ToLiteral().Await());
    EXPECT_EQ(*result, expected);
    EXPECT_FALSE(buffers[0]->IsDeleted());
    TT_ASSERT_OK_AND_ASSIGN(auto preserved_input,
                            buffers[0]->ToLiteral().Await());
    EXPECT_EQ(*preserved_input, input);
  }
}

TEST(DynamicMosaic, RejectsOutOfRangeAliasedParameter) {
  const xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {8, 128});
  EXPECT_THAT(BuildMosaicCustomCall("", {shape, shape}, shape, {-1, -1},
                                    /*aliased_parameter=*/2),
              absl_testing::StatusIs(
                  error::kInvalidArgument,
                  testing::HasSubstr("aliased parameter 2 is out of range")));
}

}  // namespace
}  // namespace torch_tpu
