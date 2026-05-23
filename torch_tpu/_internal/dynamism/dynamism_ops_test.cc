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

#include "torch_tpu/_internal/dynamism/dynamism_ops.h"

#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/shape.h"
#include "xla/tsl/platform/statusor.h"

namespace torch_tpu {
namespace {

class DynamismOpsBuilder {
 public:
  DynamismOpsBuilder() : context_() {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::MLIRContext& getContext() { return context_; }

 private:
  mlir::MLIRContext context_;
};

TEST(DynamismOpsTest, GetPadModuleDocstringExample) {
  DynamismOpsBuilder ops_builder;
  mlir::MLIRContext& context = ops_builder.getContext();

  std::vector<Shape> shapes;

  // Shape 1: [3, 5 ; dim=0, <=10]
  Dimensions dims1 = {3, 5};
  absl::InlinedVector<BoundedDynamicDimension, 1> dynamic_dims1 = {
      {.dimension = 0, .lower_bound = 0, .upper_bound = 10}};
  shapes.push_back(Shape(dims1, mlir::ElementType::F32, dynamic_dims1));

  // Shape 2: []
  Dimensions dims2 = {};
  shapes.push_back(Shape(dims2, mlir::ElementType::F32, {}));

  // Shape 3: [8, 2, 2 ; dim1, <=5, dim2, <=7]
  Dimensions dims3 = {8, 2, 2};
  absl::InlinedVector<BoundedDynamicDimension, 1> dynamic_dims3 = {
      {.dimension = 1, .lower_bound = 0, .upper_bound = 5},
      {.dimension = 2, .lower_bound = 0, .upper_bound = 7}};
  shapes.push_back(Shape(dims3, mlir::ElementType::F32, dynamic_dims3));

  // Shape 4: [6, 0 ; dim0, <=10]
  Dimensions dims4 = {6, 0};
  absl::InlinedVector<BoundedDynamicDimension, 1> dynamic_dims4 = {
      {.dimension = 0, .lower_bound = 0, .upper_bound = 10}};
  shapes.push_back(Shape(dims4, mlir::ElementType::F32, dynamic_dims4));

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetPadModule(context, shapes));
  ASSERT_TRUE(module);

  std::string module_str;
  llvm::raw_string_ostream os(module_str);
  module->print(os);

  // Verify expected function signature substring
  EXPECT_TRUE(
      absl::StrContains(module_str,
                        "-> (tensor<10x5xf32>, tensor<i32>, tensor<f32>, "
                        "tensor<8x5x7xf32>, tensor<i32>, tensor<i32>)"));
}

TEST(DynamismOpsTest, GetSliceModuleDocstringExample) {
  DynamismOpsBuilder ops_builder;
  mlir::MLIRContext& context = ops_builder.getContext();

  std::vector<Dimensions> dimensions_vec = {{3, 5}, {8, 2, 2}};
  std::vector<Dimensions> padded_dimensions_vec = {{10, 5}, {8, 5, 7}};
  std::vector<mlir::ElementType> input_dtypes = {mlir::ElementType::F32,
                                                 mlir::ElementType::F32};

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetSliceModule(context, dimensions_vec,
                                         padded_dimensions_vec, input_dtypes));
  ASSERT_TRUE(module);

  std::string module_str;
  llvm::raw_string_ostream os(module_str);
  module->print(os);

  // Verify expected function signature returning two sliced tensors
  EXPECT_TRUE(
      absl::StrContains(module_str, "-> (tensor<3x5xf32>, tensor<8x2x2xf32>)"));
}

}  // namespace
}  // namespace torch_tpu
