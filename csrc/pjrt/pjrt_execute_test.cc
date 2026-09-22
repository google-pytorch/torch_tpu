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

#include "csrc/pjrt/pjrt_execute.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "csrc/common/compilation.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

using testing::ElementsAre;

class PjrtExecuteTest : public testing::Test {
 protected:
  void SetUp() override {
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    client_ = PjrtBackend::GetInstance().GetClient();
    ASSERT_NE(client_, nullptr);
    ASSERT_FALSE(client_->addressable_devices().empty());
    TT_ASSERT_OK_AND_ASSIGN(
        memory_space_,
        client_->addressable_devices()[0]->default_memory_space());
  }

  void TearDown() override { PjrtBackend::GetInstance().Shutdown(); }

  xla::PjRtClient* client_;
  xla::PjRtMemorySpace* memory_space_ = nullptr;
};

TEST_F(PjrtExecuteTest,
       Execute_WithDynamicExecutionMetadata_RoutesInputsOutputsAndMetadata) {
  // Build a 4-parameter XLA computation with 1 logical input (N = 1) and
  // 1 logical output (M = 1):
  //   param 0: f32[2, 3] (logical input 0, logical_idx = 0)
  //   param 1: f32[2, 3] (pre-allocated donated output 0, logical_idx = 1)
  //   param 2: s32[2]    (disjoint metadata for logical_idx = 0 -> input 0)
  //   param 3: s32[2]    (disjoint metadata for logical_idx = 1 -> output 0)
  // Computation returns:
  //   param 0 + dim0(param 2) + dim1(param 3)
  xla::XlaBuilder builder("dynamic_metadata_test");
  xla::Shape data_shape =
      xla::ShapeUtil::MakeShape(xla::F32, /*dimensions=*/{2, 3});
  xla::LayoutUtil::SetToDefaultLayout(&data_shape);
  xla::Shape meta_shape =
      xla::ShapeUtil::MakeShape(xla::S32, /*dimensions=*/{2});
  xla::LayoutUtil::SetToDefaultLayout(&meta_shape);

  xla::XlaOp p0 = xla::Parameter(&builder, 0, data_shape, "input_0");
  xla::XlaOp p1 = xla::Parameter(&builder, 1, data_shape, "donated_out_0");
  xla::XlaOp p2 = xla::Parameter(&builder, 2, meta_shape, "meta_in0_s32");
  xla::XlaOp p3 = xla::Parameter(&builder, 3, meta_shape, "meta_out0_s32");

  // Extract dim 0 from p2 (value = 2) and dim 1 from p3 (value = 3).
  // Insert into p1.
  xla::XlaOp dim0_s32 = xla::Reshape(xla::Slice(p2, {0}, {1}, {1}), {});
  xla::XlaOp dim0_f32 = xla::ConvertElementType(dim0_s32, xla::F32);
  xla::XlaOp dim1_s32 = xla::Reshape(xla::Slice(p3, {1}, {2}, {1}), {});
  xla::XlaOp dim1_f32 = xla::ConvertElementType(dim1_s32, xla::F32);
  xla::XlaOp c0 = xla::ConstantR0<int32_t>(&builder, 0);
  xla::XlaOp sum_dims = xla::Add(dim0_f32, dim1_f32);
  xla::XlaOp result = xla::DynamicUpdateSlice(
      p1, xla::Add(p0, xla::Broadcast(sum_dims, {2, 3})), {c0, c0});

  TT_ASSERT_OK_AND_ASSIGN(xla::XlaComputation computation,
                          builder.Build(result));

  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> loaded_exec,
      client_->CompileAndLoad(computation, xla::CompileOptions()));

  UnboundedDynamicExecutionMetadata dynamic_meta;
  dynamic_meta.logical_input_to_pjrt_param = {0};
  dynamic_meta.donated_output_params = {
      UnboundedDynamicExecutionMetadata::DonatedOutputParam{
          .logical_output_idx = 0,
          .pjrt_param_idx = 1,
      }};
  dynamic_meta.metadata_params = {
      UnboundedDynamicExecutionMetadata::MetadataParamBinding{
          .logical_idx = 0,  // logical input 0
          .pjrt_param_idx = 2,
      },
      UnboundedDynamicExecutionMetadata::MetadataParamBinding{
          .logical_idx = 1,  // logical output 0 (num_inputs + 0)
          .pjrt_param_idx = 3,
      }};
  dynamic_meta.pjrt_param_shapes = {
      data_shape,
      data_shape,
      meta_shape,
      meta_shape,
  };

  TT_ASSERT_OK_AND_ASSIGN(
      SharedLoadedExecutableWithMetadata executable,
      LoadedExecutableWithMetadata::MakeShared(std::move(loaded_exec),
                                               /*argument_indices=*/{},
                                               std::move(dynamic_meta)));

  // Create logical input buffer on device with all 10.0f values.
  const std::vector<float> host_input(6, 10.0f);
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> input_buf,
      client_->BufferFromHostBuffer(
          host_input.data(), xla::F32, /*dims=*/{2, 3},
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, memory_space_,
          /*device_layout=*/nullptr));

  const Shape logical_shape({2, 3}, mlir::ElementType::F32);
  TT_ASSERT_OK_AND_ASSIGN(PjRtBufferPointers outputs,
                          Execute(executable, {input_buf.get()},
                                  /*logical_output_shapes=*/{logical_shape}));

  ASSERT_EQ(outputs.size(), 1);
  xla::PjRtBuffer* output_buf = outputs[0].get();
  ASSERT_NE(output_buf, nullptr);

  TT_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> out_literal,
                          output_buf->ToLiteral().Await());

  // Each output element should be input (10.0f) + dim_0 (2.0f) + dim_1 (3.0f)
  // = 15.0f.
  EXPECT_THAT(out_literal->data<float>(),
              ElementsAre(15.0f, 15.0f, 15.0f, 15.0f, 15.0f, 15.0f));
}

}  // namespace
}  // namespace torch_tpu
