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
#include "csrc/ops/softmax/softmax.h"

#include <cstdint>

#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

namespace torch_tpu {
namespace {

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

// Returns an f32 function argument of `shape` with `bound_dims` made bounded
// dynamic via `set_dimension_size` with runtime size arguments.
mlir::MlirOp MakeBoundedArg(mlir::func::FunctionBuilder& fb,
                            const Dimensions& shape,
                            absl::Span<const int64_t> bound_dims) {
  mlir::OpBuilder& op_builder = fb.getOpBuilder();
  mlir::MlirOp op = mlir::func::Argument(
      fb, mlir::RankedTensorType::get(shape, op_builder.getF32Type()));
  for (const int64_t dim : bound_dims) {
    mlir::MlirOp size = mlir::func::Argument(
        fb, mlir::RankedTensorType::get({}, op_builder.getI32Type()));
    op = mlir::stablehlo::SetDimensionSize(op, size, dim);
  }
  return op;
}

bool IsNegInfConstant(mlir::Value value) {
  if (auto bcast = value.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    value = bcast.getOperand();
  }
  auto const_op = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto fp_attr = mlir::dyn_cast<mlir::DenseFPElementsAttr>(const_op.getValue());
  if (!fp_attr || !fp_attr.isSplat()) {
    return false;
  }
  const llvm::APFloat splat = fp_attr.getSplatValue<llvm::APFloat>();
  return splat.isInfinity() && splat.isNegative();
}

bool ContainsMaxOp(mlir::Region& region) {
  bool found = false;
  region.walk([&](mlir::stablehlo::MaxOp) { found = true; });
  return found;
}

TEST(SoftmaxTest, MaxReductionUsesNegInfInit) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp input = MakeBoundedArg(fb, {2, 4}, /*bound_dims=*/{});

  TT_ASSERT_OK_AND_ASSIGN(
      mlir::MlirOp result,
      BuildSoftmaxShlo(input, /*dim=*/1, SoftmaxMode::kSoftmax));
  mlir::func::Return(fb, result);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  int num_max_reduces = 0;
  int num_selects = 0;
  module->walk([&](mlir::stablehlo::ReduceOp reduce) {
    if (ContainsMaxOp(reduce.getBody())) {
      ++num_max_reduces;
      ASSERT_EQ(reduce.getInitValues().size(), 1);
      EXPECT_TRUE(IsNegInfConstant(reduce.getInitValues()[0]))
          << DebugString(module.get());
    }
  });
  module->walk([&](mlir::stablehlo::SelectOp) { ++num_selects; });
  EXPECT_EQ(num_max_reduces, 1);
  EXPECT_EQ(num_selects, 0);
}

TEST(SoftmaxTest, DynamicReductionDimMasksPaddedElementsWithNegInf) {
  for (const SoftmaxMode mode :
       {SoftmaxMode::kSoftmax, SoftmaxMode::kLogSoftmax}) {
    TestModuleBuilder test_builder;
    mlir::ModuleBuilder& mb = test_builder.get();
    mlir::func::FunctionBuilder fb(mb, "main");
    mlir::MlirOp input = MakeBoundedArg(fb, {2, 4}, /*bound_dims=*/{1});

    TT_ASSERT_OK_AND_ASSIGN(mlir::MlirOp result,
                            BuildSoftmaxShlo(input, /*dim=*/1, mode));
    mlir::func::Return(fb, result);
    mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
        << DebugString(module.get());

    const mlir::stablehlo::Dimensions dims = GetDimensions(result);
    ASSERT_EQ(dims.size(), 2);
    EXPECT_EQ(dims[0].size, 2);
    EXPECT_FALSE(dims[0].boundOp.has_value());
    EXPECT_EQ(dims[1].size, 4);
    EXPECT_TRUE(dims[1].boundOp.has_value());

    int num_neg_inf_selects = 0;
    module->walk([&](mlir::stablehlo::SelectOp select) {
      if (IsNegInfConstant(select.getOnFalse())) {
        ++num_neg_inf_selects;
      }
    });
    EXPECT_EQ(num_neg_inf_selects, 1) << DebugString(module.get());

    int num_max_reduces = 0;
    module->walk([&](mlir::stablehlo::ReduceOp reduce) {
      if (ContainsMaxOp(reduce.getBody())) {
        ++num_max_reduces;
        ASSERT_EQ(reduce.getInitValues().size(), 1);
        EXPECT_TRUE(IsNegInfConstant(reduce.getInitValues()[0]))
            << DebugString(module.get());
      }
    });
    EXPECT_EQ(num_max_reduces, 1);
  }
}

TEST(SoftmaxTest, DynamicNonReductionDimSkipsPaddingMask) {
  TestModuleBuilder test_builder;
  mlir::ModuleBuilder& mb = test_builder.get();
  mlir::func::FunctionBuilder fb(mb, "main");
  mlir::MlirOp input = MakeBoundedArg(fb, {2, 4}, /*bound_dims=*/{0});

  TT_ASSERT_OK_AND_ASSIGN(
      mlir::MlirOp result,
      BuildSoftmaxShlo(input, /*dim=*/1, SoftmaxMode::kSoftmax));
  mlir::func::Return(fb, result);
  mlir::OwningOpRef<mlir::ModuleOp> module = mb.build();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module.get())))
      << DebugString(module.get());

  const mlir::stablehlo::Dimensions dims = GetDimensions(result);
  ASSERT_EQ(dims.size(), 2);
  EXPECT_EQ(dims[0].size, 2);
  EXPECT_TRUE(dims[0].boundOp.has_value());
  EXPECT_EQ(dims[1].size, 4);
  EXPECT_FALSE(dims[1].boundOp.has_value());

  int num_selects = 0;
  module->walk([&](mlir::stablehlo::SelectOp) { ++num_selects; });
  EXPECT_EQ(num_selects, 0);
}

}  // namespace
}  // namespace torch_tpu
