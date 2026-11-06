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

#include "csrc/mlir/passes/canonicalize_minor_interleave/canonicalize_minor_interleave.h"  // NOLINT

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"  // NOLINT
#include "mlir/Dialect/Func/IR/FuncOps.h"      // NOLINT
#include "mlir/IR/BuiltinAttributes.h"         // NOLINT
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"  // NOLINT
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"

namespace torch_tpu {
namespace {

void InitContext(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::chlo::ChloDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

template <typename OpTy>
int CountOps(mlir::ModuleOp module) {
  int count = 0;
  module.walk([&count](OpTy) { ++count; });
  return count;
}

mlir::DenseElementsAttr MakeSequentialF32Tensor(mlir::MLIRContext& ctx,
                                                llvm::ArrayRef<int64_t> shape,
                                                float base = 0.5f) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = base + static_cast<float>((i * 7 + 1) % 53) * 0.1f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

void ExpectAttrsClose(mlir::DenseElementsAttr actual,
                      mlir::DenseElementsAttr expected, float tol = 1e-4f) {
  ASSERT_EQ(actual.getType(), expected.getType());
  auto act_vals = actual.getValues<float>();
  auto exp_vals = expected.getValues<float>();
  auto act_it = act_vals.begin();
  auto exp_it = exp_vals.begin();
  for (; exp_it != exp_vals.end(); ++exp_it, ++act_it) {
    EXPECT_NEAR(*act_it, *exp_it, tol);
  }
}

// -----------------------------------------------------------------------------
// Rule A Tests: Interleave [..., K, 2] -> [..., 2K] to DotGeneral sum
// -----------------------------------------------------------------------------

TEST(CanonicalizeMinorInterleaveTest, RuleAFoldsConcatInterleaveToDot) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%a: tensor<2x4x4xf32>, %b: tensor<2x4x4xf32>)
      -> tensor<2x4x8xf32> {
    %a_exp = stablehlo.reshape %a : (tensor<2x4x4xf32>) -> tensor<2x4x4x1xf32>
    %b_exp = stablehlo.reshape %b : (tensor<2x4x4xf32>) -> tensor<2x4x4x1xf32>
    %cat = stablehlo.concatenate %a_exp, %b_exp, dim = 3 : (tensor<2x4x4x1xf32>, tensor<2x4x4x1xf32>) -> tensor<2x4x4x2xf32>  // NOLINT
    %out = stablehlo.reshape %cat : (tensor<2x4x4x2xf32>) -> tensor<2x4x8xf32>
    return %out : tensor<2x4x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  auto ref_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  auto opt_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeMinorInterleavePass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_GE(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_module), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_module), 0);

  auto in_a = MakeSequentialF32Tensor(context, {2, 4, 4}, 0.2f);
  auto in_b = MakeSequentialF32Tensor(context, {2, 4, 4}, 0.7f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {in_a, in_b}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_a, in_b}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeMinorInterleaveTest, RuleAFoldsDusInterleaveToDot) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%a: tensor<2x4x4xf32>, %b: tensor<2x4x4xf32>)
      -> tensor<2x4x8xf32> {
    %init = stablehlo.constant dense<0.0> : tensor<2x4x4x2xf32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %a_exp = stablehlo.reshape %a : (tensor<2x4x4xf32>) -> tensor<2x4x4x1xf32>
    %b_exp = stablehlo.reshape %b : (tensor<2x4x4xf32>) -> tensor<2x4x4x1xf32>
    %dus0 = stablehlo.dynamic_update_slice %init, %a_exp, %c0, %c0, %c0, %c0 : (tensor<2x4x4x2xf32>, tensor<2x4x4x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x4x4x2xf32>  // NOLINT
    %dus1 = stablehlo.dynamic_update_slice %dus0, %b_exp, %c0, %c0, %c0, %c1 : (tensor<2x4x4x2xf32>, tensor<2x4x4x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x4x4x2xf32>  // NOLINT
    %out = stablehlo.reshape %dus1 : (tensor<2x4x4x2xf32>) -> tensor<2x4x8xf32>
    return %out : tensor<2x4x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  auto ref_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  auto opt_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeMinorInterleavePass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_GE(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_module), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::DynamicUpdateSliceOp>(*opt_module), 0);

  auto in_a = MakeSequentialF32Tensor(context, {2, 4, 4}, 0.3f);
  auto in_b = MakeSequentialF32Tensor(context, {2, 4, 4}, 0.8f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {in_a, in_b}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_a, in_b}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

// -----------------------------------------------------------------------------
// Rule B Tests: Stride-2 Slices -> x @ I_even^T / x @ I_odd^T
// -----------------------------------------------------------------------------

TEST(CanonicalizeMinorInterleaveTest, RuleBFoldsStride2SlicesToDot) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4x8xf32>) -> tensor<2x4x4xf32> {
    %even = stablehlo.slice %x [0:2, 0:4, 0:8:2] : (tensor<2x4x8xf32>) -> tensor<2x4x4xf32>  // NOLINT
    %odd = stablehlo.slice %x [0:2, 0:4, 1:8:2] : (tensor<2x4x8xf32>) -> tensor<2x4x4xf32>  // NOLINT
    %add = stablehlo.add %even, %odd : tensor<2x4x4xf32>
    return %add : tensor<2x4x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  auto ref_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  auto opt_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeMinorInterleavePass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_GE(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_module), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 0);

  auto in_x = MakeSequentialF32Tensor(context, {2, 4, 8}, 0.5f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_x}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_x}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

// -----------------------------------------------------------------------------
// Rule C Tests: Concat(0, A) + Concat(B, 0) and Unary(Concat(u, u))
// -----------------------------------------------------------------------------

TEST(CanonicalizeMinorInterleaveTest, RuleCFoldsZeroPaddedConcatSums) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> tensor<2x8xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<2x4xf32>
    %cat0 = stablehlo.concatenate %z, %a, dim = 1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>  // NOLINT
    %cat1 = stablehlo.concatenate %b, %z, dim = 1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>  // NOLINT
    %sum = stablehlo.add %cat0, %cat1 : tensor<2x8xf32>
    return %sum : tensor<2x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  auto ref_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  auto opt_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeMinorInterleavePass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::AddOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_module), 1);

  auto in_a = MakeSequentialF32Tensor(context, {2, 4}, 0.2f);
  auto in_b = MakeSequentialF32Tensor(context, {2, 4}, 0.8f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {in_a, in_b}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_a, in_b}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeMinorInterleaveTest, RuleCFoldsUnaryOverDuplicatedConcat) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%u: tensor<2x4xf32>) -> tensor<2x8xf32> {
    %cat = stablehlo.concatenate %u, %u, dim = 1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>  // NOLINT
    %sin = stablehlo.sine %cat : tensor<2x8xf32>
    return %sin : tensor<2x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  auto ref_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  auto opt_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeMinorInterleavePass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::SineOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_module), 1);

  auto in_u = MakeSequentialF32Tensor(context, {2, 4}, 0.4f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_u}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_u}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

// -----------------------------------------------------------------------------
// Negative Tests: Half-split, partial RoPE, large K, missing pair
// -----------------------------------------------------------------------------

TEST(CanonicalizeMinorInterleaveTest, LeavesHalfSplitUntouched) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x8xf32>) -> tensor<2x4xf32> {
    %lo = stablehlo.slice %x [0:2, 0:4] : (tensor<2x8xf32>) -> tensor<2x4xf32>
    %hi = stablehlo.slice %x [0:2, 4:8] : (tensor<2x8xf32>) -> tensor<2x4xf32>
    %add = stablehlo.add %lo, %hi : tensor<2x4xf32>
    return %add : tensor<2x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  auto opt_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeMinorInterleavePass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 2);
}

TEST(CanonicalizeMinorInterleaveTest, LeavesPartialRoPEUntouched) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4x16xf32>) -> tensor<2x4x2xf32> {
    %even = stablehlo.slice %x [0:2, 0:4, 0:4:2] : (tensor<2x4x16xf32>) -> tensor<2x4x2xf32>  // NOLINT
    %odd = stablehlo.slice %x [0:2, 0:4, 1:4:2] : (tensor<2x4x16xf32>) -> tensor<2x4x2xf32>  // NOLINT
    %add = stablehlo.add %even, %odd : tensor<2x4x2xf32>
    return %add : tensor<2x4x2xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  auto opt_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeMinorInterleavePass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 2);
}

TEST(CanonicalizeMinorInterleaveTest, LeavesLargeKUntouched) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x512xf32>) -> tensor<2x256xf32> {
    %even = stablehlo.slice %x [0:2, 0:512:2] : (tensor<2x512xf32>) -> tensor<2x256xf32>  // NOLINT
    %odd = stablehlo.slice %x [0:2, 1:512:2] : (tensor<2x512xf32>) -> tensor<2x256xf32>  // NOLINT
    %add = stablehlo.add %even, %odd : tensor<2x256xf32>
    return %add : tensor<2x256xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  auto opt_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeMinorInterleavePass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 2);
}

TEST(CanonicalizeMinorInterleaveTest, LeavesSingleStride2SliceUntouched) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4x8xf32>) -> tensor<2x4x4xf32> {
    %even = stablehlo.slice %x [0:2, 0:4, 0:8:2] : (tensor<2x4x8xf32>) -> tensor<2x4x4xf32>  // NOLINT
    return %even : tensor<2x4x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  auto opt_module = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeMinorInterleavePass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);
}

TEST(CanonicalizeMinorInterleaveTest, NestedStride2SlicesRuleBASanRegression) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @nested_slices(%x: tensor<2x16xf32>)
    -> (tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
  %even_1 = "stablehlo.slice"(%x) <{
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 2, 16>,
    strides = array<i64: 1, 2>
  }> : (tensor<2x16xf32>) -> tensor<2x8xf32>
  %odd_1 = "stablehlo.slice"(%x) <{
    start_indices = array<i64: 0, 1>,
    limit_indices = array<i64: 2, 16>,
    strides = array<i64: 1, 2>
  }> : (tensor<2x16xf32>) -> tensor<2x8xf32>
  %even_2 = "stablehlo.slice"(%even_1) <{
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 2, 8>,
    strides = array<i64: 1, 2>
  }> : (tensor<2x8xf32>) -> tensor<2x4xf32>
  %odd_2 = "stablehlo.slice"(%even_1) <{
    start_indices = array<i64: 0, 1>,
    limit_indices = array<i64: 2, 8>,
    strides = array<i64: 1, 2>
  }> : (tensor<2x8xf32>) -> tensor<2x4xf32>
  return %even_1, %odd_1, %even_2, %odd_2
      : tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x4xf32>, tensor<2x4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_module && opt_module);

  mlir::DenseElementsAttr x = MakeSequentialF32Tensor(context, {2, 16});
  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_module, {x}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeMinorInterleavePass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  auto actual = mlir::stablehlo::evalModule(*opt_module, {x}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  for (size_t res_idx = 0; res_idx < expected->size(); ++res_idx) {
    auto exp_it = (*expected)[res_idx].value_begin<float>();
    auto act_it = (*actual)[res_idx].value_begin<float>();
    for (; exp_it != (*expected)[res_idx].value_end<float>();
         ++exp_it, ++act_it) {
      EXPECT_FLOAT_EQ(*exp_it, *act_it);
    }
  }
}

}  // namespace
}  // namespace torch_tpu
