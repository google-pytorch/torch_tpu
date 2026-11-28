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
#include "csrc/ops/topk/topk.h"

#include <cstdint>

#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
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
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

namespace torch_tpu {
namespace {

using ::absl_testing::StatusIs;
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

mlir::MlirOp MakeBoundedArg(mlir::func::FunctionBuilder& fb,
                            const Dimensions& shape, mlir::Type element_type,
                            absl::Span<const int64_t> bound_dims) {
  mlir::OpBuilder& op_builder = fb.getOpBuilder();
  mlir::MlirOp op = mlir::func::Argument(
      fb, mlir::RankedTensorType::get(shape, element_type));
  for (const int64_t dim : bound_dims) {
    mlir::MlirOp size = mlir::func::Argument(
        fb, mlir::RankedTensorType::get({}, op_builder.getI32Type()));
    op = mlir::stablehlo::SetDimensionSize(op, size, dim);
  }
  return op;
}

TEST(BuildTopKShloTest, BoundedInputLowersViaSortAndPreservesOtherBoundedDims) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp input =
      MakeBoundedArg(fb, {6, 10}, fb.getOpBuilder().getF32Type(), {0, 1});

  TT_ASSERT_OK_AND_ASSIGN(
      TopKOutputs outputs,
      BuildTopKShlo(input, /*k=*/3, /*dim=*/1, TopKMode::kLargest,
                    TopKStableMode::kStable));
  mlir::func::Return(fb, {outputs.values, outputs.indices});
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  int num_chlo_topk = 0;
  int num_sorts = 0;
  module->walk([&](mlir::chlo::TopKOp) { ++num_chlo_topk; });
  module->walk([&](mlir::stablehlo::SortOp sort) {
    ++num_sorts;
    EXPECT_EQ(sort.getDimension(), 1);
    EXPECT_TRUE(sort.getIsStable());
  });
  EXPECT_EQ(num_chlo_topk, 0);
  EXPECT_EQ(num_sorts, 1);

  for (mlir::MlirOp out : {outputs.values, outputs.indices}) {
    const mlir::stablehlo::Dimensions dims = GetDimensions(out);
    ASSERT_EQ(dims.size(), 2);
    EXPECT_EQ(dims[0].size, 6);
    EXPECT_TRUE(dims[0].boundOp.has_value());
    EXPECT_EQ(dims[1].size, 3);
    EXPECT_FALSE(dims[1].boundOp.has_value());
  }
  EXPECT_TRUE(GetTensorTypeOrDie(outputs.values).getElementType().isF32());
  EXPECT_TRUE(GetTensorTypeOrDie(outputs.indices)
                  .getElementType()
                  .isSignlessInteger(64));
}

TEST(BuildTopKShloTest, BoundedInputSmallestAndNegativeDim) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp input =
      MakeBoundedArg(fb, {8, 5}, fb.getOpBuilder().getF32Type(), {0});

  TT_ASSERT_OK_AND_ASSIGN(
      TopKOutputs outputs,
      BuildTopKShlo(input, /*k=*/2, /*dim=*/-2, TopKMode::kSmallest,
                    TopKStableMode::kUnstable));
  mlir::func::Return(fb, {outputs.values, outputs.indices});
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  int num_chlo_topk = 0;
  int num_sorts = 0;
  module->walk([&](mlir::chlo::TopKOp) { ++num_chlo_topk; });
  module->walk([&](mlir::stablehlo::SortOp sort) {
    ++num_sorts;
    EXPECT_EQ(sort.getDimension(), 0);
    EXPECT_FALSE(sort.getIsStable());
  });
  EXPECT_EQ(num_chlo_topk, 0);
  EXPECT_EQ(num_sorts, 1);

  const mlir::RankedTensorType values_type = GetTensorTypeOrDie(outputs.values);
  const mlir::RankedTensorType indices_type =
      GetTensorTypeOrDie(outputs.indices);
  EXPECT_TRUE(values_type.hasStaticShape());
  EXPECT_THAT(values_type.getShape(), ElementsAre(2, 5));
  EXPECT_TRUE(indices_type.hasStaticShape());
  EXPECT_THAT(indices_type.getShape(), ElementsAre(2, 5));
}

TEST(BuildTopKShloTest, BoundedInputZeroK) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp input =
      MakeBoundedArg(fb, {6, 10}, fb.getOpBuilder().getF32Type(), {0, 1});

  TT_ASSERT_OK_AND_ASSIGN(
      TopKOutputs outputs,
      BuildTopKShlo(input, /*k=*/0, /*dim=*/1, TopKMode::kLargest));
  mlir::func::Return(fb, {outputs.values, outputs.indices});
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  const mlir::stablehlo::Dimensions dims = GetDimensions(outputs.values);
  ASSERT_EQ(dims.size(), 2);
  EXPECT_EQ(dims[0].size, 6);
  EXPECT_TRUE(dims[0].boundOp.has_value());
  EXPECT_EQ(dims[1].size, 0);
  EXPECT_FALSE(dims[1].boundOp.has_value());
}

TEST(BuildTopKShloTest, StaticInputUsesChloTopKAndTransposesNonInnermostDim) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp input =
      MakeBoundedArg(fb, {4, 6}, fb.getOpBuilder().getF32Type(), {});

  TT_ASSERT_OK_AND_ASSIGN(
      TopKOutputs outputs,
      BuildTopKShlo(input, /*k=*/2, /*dim=*/0, TopKMode::kSmallest));
  mlir::func::Return(fb, {outputs.values, outputs.indices});
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  int num_chlo_topk = 0;
  int num_sorts = 0;
  module->walk([&](mlir::chlo::TopKOp) { ++num_chlo_topk; });
  module->walk([&](mlir::stablehlo::SortOp) { ++num_sorts; });
  EXPECT_EQ(num_chlo_topk, 1);
  EXPECT_EQ(num_sorts, 0);

  const mlir::RankedTensorType values_type = GetTensorTypeOrDie(outputs.values);
  const mlir::RankedTensorType indices_type =
      GetTensorTypeOrDie(outputs.indices);
  EXPECT_TRUE(values_type.hasStaticShape());
  EXPECT_THAT(values_type.getShape(), ElementsAre(2, 6));
  EXPECT_TRUE(values_type.getElementType().isF32());
  EXPECT_TRUE(indices_type.hasStaticShape());
  EXPECT_THAT(indices_type.getShape(), ElementsAre(2, 6));
  EXPECT_TRUE(indices_type.getElementType().isSignlessInteger(64));
}

TEST(BuildTopKShloTest, StaticIntegerInputSmallestUsesBitwiseNot) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp input =
      MakeBoundedArg(fb, {4, 6}, fb.getOpBuilder().getI32Type(), {});

  TT_ASSERT_OK_AND_ASSIGN(
      TopKOutputs outputs,
      BuildTopKShlo(input, /*k=*/2, /*dim=*/1, TopKMode::kSmallest));
  mlir::func::Return(fb, {outputs.values, outputs.indices});
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  int num_chlo_topk = 0;
  int num_nots = 0;
  module->walk([&](mlir::chlo::TopKOp) { ++num_chlo_topk; });
  module->walk([&](mlir::stablehlo::NotOp) { ++num_nots; });
  EXPECT_EQ(num_chlo_topk, 1);
  EXPECT_EQ(num_nots, 2);
}

TEST(BuildTopKShloTest, RejectsInvalidDimAndKOnBoundedInput) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp input =
      MakeBoundedArg(fb, {6, 10}, fb.getOpBuilder().getF32Type(), {0, 1});
  mlir::func::Return(fb, input);

  EXPECT_THAT(
      BuildTopKShlo(input, /*k=*/2, /*dim=*/2, TopKMode::kLargest).status(),
      StatusIs(error::kIndexError));
  EXPECT_THAT(
      BuildTopKShlo(input, /*k=*/11, /*dim=*/1, TopKMode::kLargest).status(),
      StatusIs(error::kInvalidArgument));
}

}  // namespace
}  // namespace torch_tpu
