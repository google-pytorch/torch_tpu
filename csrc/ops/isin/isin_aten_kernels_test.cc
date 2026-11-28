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

// ANY_OP_IMPL_FILENAME_OK=unit test file
#include "csrc/ops/isin/isin_aten_kernels.h"

#include <cstdint>

#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

namespace torch_tpu {
namespace {

using ::testing::ElementsAre;

class TestModuleBuilder {
 public:
  TestModuleBuilder()
      : context_(), module_builder_(context_, mlir::unknownLoc(context_)) {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::ModuleBuilder& get() { return module_builder_; }

 private:
  mlir::MLIRContext context_;
  mlir::ModuleBuilder module_builder_;
};

mlir::MlirOp MakeBoundedI32Arg(mlir::func::FunctionBuilder& fb,
                               const Dimensions& shape,
                               absl::Span<const int64_t> bound_dims) {
  mlir::OpBuilder& op_builder = fb.getOpBuilder();
  mlir::MlirOp op = mlir::func::Argument(
      fb, mlir::RankedTensorType::get(shape, op_builder.getI32Type()));
  for (const int64_t dim : bound_dims) {
    mlir::MlirOp size = mlir::func::Argument(
        fb, mlir::RankedTensorType::get({}, op_builder.getI32Type()));
    op = mlir::stablehlo::SetDimensionSize(op, size, dim);
  }
  return op;
}

TEST(BuildIsInShloTest, StaticInputsFlattenMultiDimTestElementsAndReduce) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp elements = MakeBoundedI32Arg(fb, {3, 4}, {});
  mlir::MlirOp test_elements = MakeBoundedI32Arg(fb, {2, 3}, {});

  TT_ASSERT_OK_AND_ASSIGN(
      mlir::MlirOp result,
      BuildIsInShlo(elements, test_elements, IsUnique::kNo, IsInverted::kNo));
  mlir::func::Return(fb, result);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  const mlir::RankedTensorType result_type = GetTensorTypeOrDie(result);
  EXPECT_TRUE(result_type.hasStaticShape());
  EXPECT_THAT(result_type.getShape(), ElementsAre(3, 4));
  EXPECT_TRUE(result_type.getElementType().isInteger(1));

  int num_reshapes = 0;
  int num_reduces = 0;
  int num_nots = 0;
  module->walk([&](mlir::stablehlo::ReshapeOp) { ++num_reshapes; });
  module->walk([&](mlir::stablehlo::ReduceOp reduce) {
    ++num_reduces;
    EXPECT_THAT(reduce.getDimensions(), ElementsAre(2));
  });
  module->walk([&](mlir::stablehlo::NotOp) { ++num_nots; });
  EXPECT_EQ(num_reshapes, 1);
  EXPECT_EQ(num_reduces, 1);
  EXPECT_EQ(num_nots, 0);
}

TEST(BuildIsInShloTest,
     BoundedElementsAndTestElementsAvoidReshapeAndPreserveBounds) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp elements = MakeBoundedI32Arg(fb, {6, 4}, {0, 1});
  mlir::MlirOp test_elements = MakeBoundedI32Arg(fb, {5, 3}, {0});

  TT_ASSERT_OK_AND_ASSIGN(
      mlir::MlirOp result,
      BuildIsInShlo(elements, test_elements, IsUnique::kYes, IsInverted::kNo));
  mlir::func::Return(fb, result);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  const mlir::stablehlo::Dimensions dims = GetDimensions(result);
  ASSERT_EQ(dims.size(), 2);
  EXPECT_EQ(dims[0].size, 6);
  EXPECT_TRUE(dims[0].boundOp.has_value());
  EXPECT_EQ(dims[1].size, 4);
  EXPECT_TRUE(dims[1].boundOp.has_value());
  EXPECT_TRUE(GetTensorTypeOrDie(result).getElementType().isInteger(1));

  int num_reshapes = 0;
  int num_reduces = 0;
  module->walk([&](mlir::stablehlo::ReshapeOp) { ++num_reshapes; });
  module->walk([&](mlir::stablehlo::ReduceOp reduce) {
    ++num_reduces;
    EXPECT_THAT(reduce.getDimensions(), ElementsAre(2, 3));
  });
  EXPECT_EQ(num_reshapes, 0);
  EXPECT_EQ(num_reduces, 1);
}

TEST(BuildIsInShloTest, ScalarTestElementsSkipsReduce) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp elements = MakeBoundedI32Arg(fb, {6, 4}, {0});
  mlir::MlirOp test_elements = MakeBoundedI32Arg(fb, {}, {});

  TT_ASSERT_OK_AND_ASSIGN(
      mlir::MlirOp result,
      BuildIsInShlo(elements, test_elements, IsUnique::kNo, IsInverted::kNo));
  mlir::func::Return(fb, result);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  const mlir::stablehlo::Dimensions dims = GetDimensions(result);
  ASSERT_EQ(dims.size(), 2);
  EXPECT_EQ(dims[0].size, 6);
  EXPECT_TRUE(dims[0].boundOp.has_value());
  EXPECT_EQ(dims[1].size, 4);
  EXPECT_FALSE(dims[1].boundOp.has_value());

  int num_compares = 0;
  int num_reduces = 0;
  module->walk([&](mlir::stablehlo::CompareOp) { ++num_compares; });
  module->walk([&](mlir::stablehlo::ReduceOp) { ++num_reduces; });
  EXPECT_EQ(num_compares, 1);
  EXPECT_EQ(num_reduces, 0);
}

TEST(BuildIsInShloTest, InvertedFlagEmitsLogicalNot) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp elements = MakeBoundedI32Arg(fb, {6}, {0});
  mlir::MlirOp test_elements = MakeBoundedI32Arg(fb, {4}, {0});

  TT_ASSERT_OK_AND_ASSIGN(
      mlir::MlirOp result,
      BuildIsInShlo(elements, test_elements, IsUnique::kNo, IsInverted::kYes));
  mlir::func::Return(fb, result);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  const mlir::stablehlo::Dimensions dims = GetDimensions(result);
  ASSERT_EQ(dims.size(), 1);
  EXPECT_EQ(dims[0].size, 6);
  EXPECT_TRUE(dims[0].boundOp.has_value());

  int num_nots = 0;
  module->walk([&](mlir::stablehlo::NotOp) { ++num_nots; });
  EXPECT_EQ(num_nots, 1);
}

}  // namespace
}  // namespace torch_tpu
