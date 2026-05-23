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

#include "torch_tpu/common/dynamism_utils.h"

#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "xla/tsl/platform/statusor.h"

namespace torch_tpu {
namespace {

class DynamismOpsBuilder {
 public:
  DynamismOpsBuilder()
      : context_(), module_builder_(context_, mlir::unknownLoc(context_)) {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::ModuleBuilder& get() { return module_builder_; }
  mlir::MLIRContext& getContext() { return context_; }

 private:
  mlir::MLIRContext context_;
  mlir::ModuleBuilder module_builder_;
};

TEST(DynamismOpsTest, GetTraversalOutputDimensionsNoBoundedInput) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  DynamismOpsBuilder ops_builder;

  // Create Input DeviceBufferRefs.
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef input1,
      DeviceBufferList::MakePlaceholder({5, 10}, mlir::ElementType::F32));

  TF_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef input2,
      DeviceBufferList::MakePlaceholder({5, 10}, mlir::ElementType::F32));

  // Create a Deferred Op that uses the inputs.
  std::vector<DeviceBufferRef> add_inputs = {input1, input2};
  Shape add_output_shape(Dimensions{5, 10}, mlir::ElementType::F32);
  auto builder = [](mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    EXPECT_EQ(inputs.size(), 2);
    TT_ASSIGN_OR_RETURN(auto output, BuildAddShlo(inputs[0], inputs[1]));
    return DynamicMlirOpResults{output};
  };
  TF_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> add_deferred_refs,
                          DeviceBufferList::CreateDeferred(
                              OpName::kAdd, std::move(builder), add_inputs,
                              OpParamCacheKeys::Empty(), {add_output_shape}));
  ASSERT_EQ(add_deferred_refs.size(), 1);
  SharedDeviceBufferList add_node = add_deferred_refs[0].device_buffer_list();
  DeviceBufferRef add_output = add_deferred_refs[0];

  std::vector<DeviceBufferRef> outputs = {add_output};

  TF_ASSERT_OK_AND_ASSIGN(auto traversal,
                          Traversal::Create(outputs, /*stopping_points=*/{}));

  // Call GetTraversalOutputDimensions.
  TF_ASSERT_OK_AND_ASSIGN(
      auto outputs_to_dims,
      GetTraversalOutputDimensions(ops_builder.getContext(), *traversal));

  ASSERT_EQ(outputs_to_dims.size(), 1);
  auto output_to_dims = outputs_to_dims[0];

  EXPECT_EQ(output_to_dims.dims.size(), 2);
  EXPECT_EQ(output_to_dims.dims[0].size, 5);
  EXPECT_EQ(output_to_dims.dims[0].boundOpDim, -1);
  EXPECT_EQ(output_to_dims.dims[1].size, 10);
  EXPECT_EQ(output_to_dims.dims[1].boundOpDim, -1);
}

TEST(DynamismOpsTest, GetTraversalOutputDimensionsWithBoundedInput) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  DynamismOpsBuilder ops_builder;

  // Create Input DeviceBufferRefs.
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef input1,
      DeviceBufferList::MakePlaceholder({5, 10}, mlir::ElementType::F32));
  ASSERT_EQ(input1.MarkDynamic(/*dimension=*/1, /*lower_bound=*/10,
                               /*upper_bound=*/100),
            absl::OkStatus());
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef input2,
      DeviceBufferList::MakePlaceholder({5, 10}, mlir::ElementType::F32));
  ASSERT_EQ(input2.MarkDynamic(/*dimension=*/1, /*lower_bound=*/2,
                               /*upper_bound=*/100),
            absl::OkStatus());

  // Create a Deferred Op that uses the inputs.
  std::vector<DeviceBufferRef> add_inputs = {input1, input2};
  Shape add_output_shape(Dimensions{5, 10}, mlir::ElementType::F32);
  auto builder = [](mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    EXPECT_EQ(inputs.size(), 2);
    TT_ASSIGN_OR_RETURN(auto output, BuildAddShlo(inputs[0], inputs[1]));
    return DynamicMlirOpResults{output};
  };
  TF_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> add_deferred_refs,
                          DeviceBufferList::CreateDeferred(
                              OpName::kAdd, std::move(builder), add_inputs,
                              OpParamCacheKeys::Empty(), {add_output_shape}));
  ASSERT_EQ(add_deferred_refs.size(), 1);
  SharedDeviceBufferList add_node = add_deferred_refs[0].device_buffer_list();
  DeviceBufferRef add_output = add_deferred_refs[0];

  std::vector<DeviceBufferRef> outputs = {add_output};

  TF_ASSERT_OK_AND_ASSIGN(auto traversal,
                          Traversal::Create(outputs, /*stopping_points=*/{}));

  // Call GetTraversalOutputDimensions.
  TF_ASSERT_OK_AND_ASSIGN(
      auto outputs_to_dims,
      GetTraversalOutputDimensions(ops_builder.getContext(), *traversal));
  ASSERT_EQ(outputs_to_dims.size(), 1);
  auto output_to_dims = outputs_to_dims[0];

  EXPECT_EQ(output_to_dims.dims.size(), 2);
  EXPECT_EQ(output_to_dims.dims[0].size, 5);
  EXPECT_EQ(output_to_dims.dims[1].size, 100);
}

}  // namespace
}  // namespace torch_tpu
