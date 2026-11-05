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

#include "csrc/mlir/passes/canonicalize_reductions/canonicalize_reductions.h"  // NOLINT

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
    vals[i] = base + static_cast<float>((i * 11 + 3) % 97) * 0.05f;
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
// Mechanism 1 Tests: Hoist monotonic elementwise chains from reduce_max
// -----------------------------------------------------------------------------

TEST(CanonicalizeReductionsTest, HoistsMonotonicChainFromMaxReduce) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4x8xf32>) -> tensor<2x4xf32> {
    %tanh = stablehlo.tanh %x : tensor<2x4x8xf32>
    %c_scale = stablehlo.constant dense<2.0> : tensor<2x4x8xf32>
    %mul = stablehlo.multiply %tanh, %c_scale : tensor<2x4x8xf32>
    %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %red = stablehlo.reduce(%mul init: %init)
        applies stablehlo.maximum across dimensions = [2]
        : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x4xf32>
    return %red : tensor<2x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // The reduce op now directly consumes %x, and tanh + mul are hoisted after.
  opt_module->walk([](mlir::stablehlo::ReduceOp red) {
    EXPECT_EQ(red.getInputs()[0].getDefiningOp(), nullptr);
  });

  auto in_val = MakeSequentialF32Tensor(context, {2, 4, 8});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeReductionsTest, NegativeScaleDoesNotHoist) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4x8xf32>) -> tensor<2x4xf32> {
    %c_neg = stablehlo.constant dense<-2.0> : tensor<2x4x8xf32>
    %mul = stablehlo.multiply %x, %c_neg : tensor<2x4x8xf32>
    %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %red = stablehlo.reduce(%mul init: %init)
        applies stablehlo.maximum across dimensions = [2]
        : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x4xf32>
    return %red : tensor<2x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  module->walk([](mlir::stablehlo::ReduceOp red) {
    EXPECT_NE(red.getInputs()[0].getDefiningOp(), nullptr);
    EXPECT_TRUE(
        mlir::isa<mlir::stablehlo::MulOp>(red.getInputs()[0].getDefiningOp()));
  });
}

// -----------------------------------------------------------------------------
// Mechanism 4 Tests: Fold redundant elementwise reshape roundtrips
// -----------------------------------------------------------------------------

TEST(CanonicalizeReductionsTest, FoldsRedundantReshapeRoundtrip) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4xf32>) -> tensor<2xf32> {
    %r_in = stablehlo.reshape %x : (tensor<2x4xf32>) -> tensor<1x2x4x1xf32>
    %c1 = stablehlo.constant dense<1.0> : tensor<1x2x4x1xf32>
    %add = stablehlo.add %r_in, %c1 : tensor<1x2x4x1xf32>
    %sub = stablehlo.subtract %add, %c1 : tensor<1x2x4x1xf32>
    %r_out = stablehlo.reshape %sub : (tensor<1x2x4x1xf32>) -> tensor<2x4xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %red = stablehlo.reduce(%r_out init: %zero)
        applies stablehlo.add across dimensions = [1]
        : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %red : tensor<2xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*opt_module), 0);

  auto in_val = MakeSequentialF32Tensor(context, {2, 4});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeReductionsTest,
     FoldsReshapeRoundtripMultipleEntriesWithScalarBroadcast) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<2x4xf32>,
      %y: tensor<2x4xf32>) -> tensor<2xf32> {
    %rx = stablehlo.reshape %x : (tensor<2x4xf32>) -> tensor<1x2x4x1xf32>
    %ry = stablehlo.reshape %y : (tensor<2x4xf32>) -> tensor<1x2x4x1xf32>
    %add = stablehlo.add %rx, %ry : tensor<1x2x4x1xf32>
    %s = stablehlo.constant dense<2.0> : tensor<f32>
    %bcast = stablehlo.broadcast_in_dim %s, dims = []
        : (tensor<f32>) -> tensor<1x2x4x1xf32>
    %mul = stablehlo.multiply %add, %bcast : tensor<1x2x4x1xf32>
    %r_out = stablehlo.reshape %mul : (tensor<1x2x4x1xf32>) -> tensor<2x4xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %red = stablehlo.reduce(%r_out init: %zero)
        applies stablehlo.add across dimensions = [1]
        : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %red : tensor<2xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // Both entry reshapes and exit reshape must be folded.
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*opt_module), 1);

  auto in_x = MakeSequentialF32Tensor(context, {2, 4});
  auto in_y = MakeSequentialF32Tensor(context, {2, 4}, 1.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {in_x, in_y}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_x, in_y}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

// -----------------------------------------------------------------------------
// Mechanism 5 Tests: Fold unit unreduced axes into variadic reductions
// -----------------------------------------------------------------------------

TEST(CanonicalizeReductionsTest, FoldsUnitUnreducedAxesVariadic) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x4x1x8xf32>)
      -> (tensor<1x4x1xf32>, tensor<1x4x1xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %0:2 = "stablehlo.reduce"(%x, %x, %zero, %zero) ({
    ^bb0(%a0: tensor<f32>, %a1: tensor<f32>,
         %b0: tensor<f32>, %b1: tensor<f32>):
      %s0 = stablehlo.add %a0, %b0 : tensor<f32>
      %s1 = stablehlo.add %a1, %b1 : tensor<f32>
      "stablehlo.return"(%s0, %s1) : (tensor<f32>, tensor<f32>) -> ()
    }) {dimensions = array<i64: 3>}
        : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>,
           tensor<f32>, tensor<f32>) -> (tensor<1x4x1xf32>, tensor<1x4x1xf32>)
    return %0#0, %0#1 : tensor<1x4x1xf32>, tensor<1x4x1xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  opt_module->walk([](mlir::stablehlo::ReduceOp red) {
    EXPECT_EQ(red.getDimensions().size(), 3);
  });
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*opt_module), 2);

  auto in_val = MakeSequentialF32Tensor(context, {1, 4, 1, 8});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
  ExpectAttrsClose((*actual)[1], (*expected)[1]);
}

// -----------------------------------------------------------------------------
// CancelScaleInDiv Tests
// -----------------------------------------------------------------------------

TEST(CanonicalizeReductionsTest, CancelScaleInDivDirectNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %c = stablehlo.constant dense<2.5> : tensor<4x8xf32>
    %mul = stablehlo.multiply %x, %c : tensor<4x8xf32>
    %div = stablehlo.divide %mul, %c : tensor<4x8xf32>
    return %div : tensor<4x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::DivOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::MulOp>(*opt_module), 0);

  auto in_val = MakeSequentialF32Tensor(context, {4, 8});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeReductionsTest, CancelScaleInDivNestedNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8xf32>, %y: tensor<4x8xf32>)
      -> tensor<4x8xf32> {
    %c = stablehlo.constant dense<0.125> : tensor<4x8xf32>
    %mul1 = stablehlo.multiply %x, %c : tensor<4x8xf32>
    %mul2 = stablehlo.multiply %mul1, %y : tensor<4x8xf32>
    %div = stablehlo.divide %mul2, %c : tensor<4x8xf32>
    return %div : tensor<4x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(ref_module);
  ASSERT_TRUE(opt_module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::DivOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::MulOp>(*opt_module), 1);

  auto in_x = MakeSequentialF32Tensor(context, {4, 8}, 1.0f);
  auto in_y = MakeSequentialF32Tensor(context, {4, 8}, 2.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {in_x, in_y}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_x, in_y}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

// -----------------------------------------------------------------------------
// Regression Tests for Audit Bugs
// -----------------------------------------------------------------------------

// Audit Bug 1: Linear-only chain on multi-use reduce NOT hoisted.
TEST(CanonicalizeReductionsTest, LinearOnlyChainOnMultiUseReduceNotHoisted) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4x8xf32>)
      -> (tensor<2x4xf32>, tensor<2x4x8xf32>) {
    %c_scale = stablehlo.constant dense<2.0> : tensor<2x4x8xf32>
    %mul = stablehlo.multiply %x, %c_scale : tensor<2x4x8xf32>
    %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %red = stablehlo.reduce(%mul init: %init)
        applies stablehlo.maximum across dimensions = [2]
        : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x4xf32>
    return %red, %mul : tensor<2x4xf32>, tensor<2x4x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // %mul has multiple uses and is linear-only: must NOT be hoisted.
  module->walk([](mlir::stablehlo::ReduceOp red) {
    EXPECT_NE(red.getInputs()[0].getDefiningOp(), nullptr);
    EXPECT_TRUE(
        mlir::isa<mlir::stablehlo::MulOp>(red.getInputs()[0].getDefiningOp()));
  });
}

// Audit Bug 2: Masked-softmax with -FLT_MAX init NOT rewritten.
TEST(CanonicalizeReductionsTest, MaskedSoftmaxWithFltMaxInitNotRewritten) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4x8xf32>) -> tensor<2x4xf32> {
    %tanh = stablehlo.tanh %x : tensor<2x4x8xf32>
    %c_scale = stablehlo.constant dense<2.0> : tensor<2x4x8xf32>
    %mul = stablehlo.multiply %tanh, %c_scale : tensor<2x4x8xf32>
    // -FLT_MAX = 0xFF7FFFFF (largest finite negative float, not -inf)
    %init = stablehlo.constant dense<0xFF7FFFFF> : tensor<f32>
    %red = stablehlo.reduce(%mul init: %init)
        applies stablehlo.maximum across dimensions = [2]
        : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x4xf32>
    return %red : tensor<2x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Must NOT hoist when init is -FLT_MAX instead of -inf.
  module->walk([](mlir::stablehlo::ReduceOp red) {
    EXPECT_NE(red.getInputs()[0].getDefiningOp(), nullptr);
  });
}

// Audit Bug 3: -inf broadcast operand NOT treated as invariant.
TEST(CanonicalizeReductionsTest, NegInfBroadcastOperandNotTreatedAsInvariant) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4x8xf32>) -> tensor<2x4xf32> {
    // -inf constant broadcasted across non-reduced dimensions
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<1xf32>
    %bcast = stablehlo.broadcast_in_dim %neg_inf, dims = [0]
        : (tensor<1xf32>) -> tensor<2x4x8xf32>
    %add = stablehlo.add %x, %bcast : tensor<2x4x8xf32>
    %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %red = stablehlo.reduce(%add init: %init)
        applies stablehlo.maximum across dimensions = [2]
        : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x4xf32>
    return %red : tensor<2x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // -inf broadcast operand must NOT be hoisted past reduction.
  module->walk([](mlir::stablehlo::ReduceOp red) {
    EXPECT_NE(red.getInputs()[0].getDefiningOp(), nullptr);
  });
}

TEST(CanonicalizeReductionsTest, Size0ReducedSliceNotHoisted) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x0x8xf32>) -> tensor<2x8xf32> {
    %tanh = stablehlo.tanh %x : tensor<2x0x8xf32>
    %c_scale = stablehlo.constant dense<2.0> : tensor<2x0x8xf32>
    %mul = stablehlo.multiply %tanh, %c_scale : tensor<2x0x8xf32>
    %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %red = stablehlo.reduce(%mul init: %init)
        applies stablehlo.maximum across dimensions = [1]
        : (tensor<2x0x8xf32>, tensor<f32>) -> tensor<2x8xf32>
    return %red : tensor<2x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Size 0 reduced slice must NOT be hoisted.
  module->walk([](mlir::stablehlo::ReduceOp red) {
    EXPECT_NE(red.getInputs()[0].getDefiningOp(), nullptr);
  });
}

TEST(CanonicalizeReductionsTest, F8E4M3FNConvertBeforeReduceMaxNotHoisted) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4x8xf8E4M3FN>) -> tensor<2x4xf32> {
    %cvt = stablehlo.convert %x : (tensor<2x4x8xf8E4M3FN>) -> tensor<2x4x8xf32>
    %c_scale = stablehlo.constant dense<2.0> : tensor<2x4x8xf32>
    %mul = stablehlo.multiply %cvt, %c_scale : tensor<2x4x8xf32>
    %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %red = stablehlo.reduce(%mul init: %init)
        applies stablehlo.maximum across dimensions = [2]
        : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x4xf32>
    return %red : tensor<2x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // f8E4M3FN has no infinity; convert must NOT be hoisted past reduce_max.
  module->walk([](mlir::stablehlo::ReduceOp red) {
    EXPECT_NE(red.getInputs()[0].getDefiningOp(), nullptr);
  });
}

TEST(CanonicalizeReductionsTest,
     NonSplatNegInfBroadcastOperandNotTreatedAsInvariant) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4x8xf32>) -> tensor<2x4xf32> {
    // Non-splat constant containing -inf (0xFF800000)
    %mask = stablehlo.constant dense<[0.0, 0xFF800000]> : tensor<2xf32>
    %bcast = stablehlo.broadcast_in_dim %mask, dims = [0]
        : (tensor<2xf32>) -> tensor<2x4x8xf32>
    %add = stablehlo.add %x, %bcast : tensor<2x4x8xf32>
    %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %red = stablehlo.reduce(%add init: %init)
        applies stablehlo.maximum across dimensions = [2]
        : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x4xf32>
    return %red : tensor<2x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Non-splat broadcast constant with -inf must NOT be hoisted.
  module->walk([](mlir::stablehlo::ReduceOp red) {
    EXPECT_NE(red.getInputs()[0].getDefiningOp(), nullptr);
  });
}

TEST(CanonicalizeReductionsTest, MultiUseWideningConvertTanhHoisted) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4x8xbf16>)
      -> (tensor<2x4xf32>, tensor<2x4x8xf32>) {
    %tanh = stablehlo.tanh %x : tensor<2x4x8xbf16>
    %cvt = stablehlo.convert %tanh : (tensor<2x4x8xbf16>) -> tensor<2x4x8xf32>
    %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %red = stablehlo.reduce(%cvt init: %init)
        applies stablehlo.maximum across dimensions = [2]
        : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x4xf32>
    %other = stablehlo.negate %cvt : tensor<2x4x8xf32>
    return %red, %other : tensor<2x4xf32>, tensor<2x4x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Widening convert + tanh must be hoisted even though cvt has multiple uses.
  int reduce_count = 0;
  module->walk([&](mlir::stablehlo::ReduceOp red) {
    ++reduce_count;
    EXPECT_EQ(red.getInputs()[0].getDefiningOp(), nullptr);
  });
  EXPECT_EQ(reduce_count, 1);
}

TEST(CanonicalizeReductionsTest, HoistsMonotonicChainFromMinReduce) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4x8xf32>) -> tensor<2x4xf32> {
    %tanh = stablehlo.tanh %x : tensor<2x4x8xf32>
    %c_scale = stablehlo.constant dense<2.0> : tensor<2x4x8xf32>
    %mul = stablehlo.multiply %tanh, %c_scale : tensor<2x4x8xf32>
    %init = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %red = stablehlo.reduce(%mul init: %init)
        applies stablehlo.minimum across dimensions = [2]
        : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x4xf32>
    return %red : tensor<2x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Monotonic chain hoisted out of reduce_min.
  int reduce_count = 0;
  module->walk([&](mlir::stablehlo::ReduceOp red) {
    ++reduce_count;
    EXPECT_EQ(red.getInputs()[0].getDefiningOp(), nullptr);
  });
  EXPECT_EQ(reduce_count, 1);
}

TEST(CanonicalizeReductionsTest, ShiftsBroadcastDimsInAddInvariant) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4x8xf32>, %bias: tensor<2xf32>)
      -> tensor<2x8xf32> {
    %bcast = stablehlo.broadcast_in_dim %bias, dims = [0]
        : (tensor<2xf32>) -> tensor<2x4x8xf32>
    %add = stablehlo.add %x, %bcast : tensor<2x4x8xf32>
    %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %red = stablehlo.reduce(%add init: %init)
        applies stablehlo.maximum across dimensions = [1]
        : (tensor<2x4x8xf32>, tensor<f32>) -> tensor<2x8xf32>
    return %red : tensor<2x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Add invariant hoisted past reduction; broadcast dims shifted to [0].
  int reduce_count = 0;
  module->walk([&](mlir::stablehlo::ReduceOp red) {
    ++reduce_count;
    EXPECT_EQ(red.getInputs()[0].getDefiningOp(), nullptr);
  });
  EXPECT_EQ(reduce_count, 1);
}

TEST(CanonicalizeReductionsTest, NegativeOrEmptyDimsOrDynamicShapeNotHoisted) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @empty_dims(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %tanh = stablehlo.tanh %x : tensor<2x4xf32>
    %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %red = stablehlo.reduce(%tanh init: %init)
        applies stablehlo.maximum across dimensions = []
        : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    return %red : tensor<2x4xf32>
  }
  func.func @dynamic_shape(%x: tensor<?x4xf32>) -> tensor<?xf32> {
    %tanh = stablehlo.tanh %x : tensor<?x4xf32>
    %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %red = stablehlo.reduce(%tanh init: %init)
        applies stablehlo.maximum across dimensions = [1]
        : (tensor<?x4xf32>, tensor<f32>) -> tensor<?xf32>
    return %red : tensor<?xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Empty dimensions or dynamic shapes must not be hoisted.
  int reduce_count = 0;
  module->walk([&](mlir::stablehlo::ReduceOp red) {
    ++reduce_count;
    EXPECT_NE(red.getInputs()[0].getDefiningOp(), nullptr);
  });
  EXPECT_EQ(reduce_count, 2);
}

TEST(CanonicalizeReductionsTest, SingleResultNotFolded) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x4x1xf32>) -> tensor<1x1xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %red = stablehlo.reduce(%x init: %zero)
        applies stablehlo.add across dimensions = [1]
        : (tensor<1x4x1xf32>, tensor<f32>) -> tensor<1x1xf32>
    return %red : tensor<1x1xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Single-result reductions are not folded into reshape.
  module->walk([](mlir::stablehlo::ReduceOp red) {
    EXPECT_EQ(red.getDimensions().size(), 1);
  });
}

TEST(CanonicalizeReductionsTest, NonIdentityCombinerNotFolded) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x4x1xf32>, %y: tensor<1x4x1xf32>)
      -> (tensor<1x1xf32>, tensor<1x1xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %0:2 = "stablehlo.reduce"(%x, %y, %zero, %zero) ({
    ^bb0(%a0: tensor<f32>, %a1: tensor<f32>,
         %b0: tensor<f32>, %b1: tensor<f32>):
      %sub = stablehlo.subtract %a0, %b0 : tensor<f32>
      "stablehlo.return"(%sub, %b1) : (tensor<f32>, tensor<f32>) -> ()
    }) {dimensions = array<i64: 1>}
        : (tensor<1x4x1xf32>, tensor<1x4x1xf32>,
           tensor<f32>, tensor<f32>) -> (tensor<1x1xf32>, tensor<1x1xf32>)
    return %0#0, %0#1 : tensor<1x1xf32>, tensor<1x1xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Non-identity combiner must not be folded.
  module->walk([](mlir::stablehlo::ReduceOp red) {
    EXPECT_EQ(red.getDimensions().size(), 1);
  });
}

TEST(CanonicalizeReductionsTest, AllUnitScalarOutputFolded) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x4x1xf32>, %y: tensor<1x4x1xf32>)
      -> (tensor<1x1xf32>, tensor<1x1xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %0:2 = "stablehlo.reduce"(%x, %y, %zero, %zero) ({
    ^bb0(%a0: tensor<f32>, %a1: tensor<f32>,
         %b0: tensor<f32>, %b1: tensor<f32>):
      %s0 = stablehlo.add %a0, %b0 : tensor<f32>
      %s1 = stablehlo.add %a1, %b1 : tensor<f32>
      "stablehlo.return"(%s0, %s1) : (tensor<f32>, tensor<f32>) -> ()
    }) {dimensions = array<i64: 1>}
        : (tensor<1x4x1xf32>, tensor<1x4x1xf32>,
           tensor<f32>, tensor<f32>) -> (tensor<1x1xf32>, tensor<1x1xf32>)
    return %0#0, %0#1 : tensor<1x1xf32>, tensor<1x1xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Unit unreduced axes [0, 2] folded into reduction; reshapes restore shape.
  int reduce_count = 0;
  module->walk([&](mlir::stablehlo::ReduceOp red) {
    ++reduce_count;
    EXPECT_EQ(red.getDimensions().size(), 3);
  });
  EXPECT_EQ(reduce_count, 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*module), 2);
}

TEST(CanonicalizeReductionsTest, DynamicVariadicInputNotFolded) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<?x4x1xf32>, %y: tensor<?x4x1xf32>)
      -> (tensor<?x1xf32>, tensor<?x1xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %0:2 = "stablehlo.reduce"(%x, %y, %zero, %zero) ({
    ^bb0(%a0: tensor<f32>, %a1: tensor<f32>,
         %b0: tensor<f32>, %b1: tensor<f32>):
      %s0 = stablehlo.add %a0, %b0 : tensor<f32>
      %s1 = stablehlo.add %a1, %b1 : tensor<f32>
      "stablehlo.return"(%s0, %s1) : (tensor<f32>, tensor<f32>) -> ()
    }) {dimensions = array<i64: 1>}
        : (tensor<?x4x1xf32>, tensor<?x4x1xf32>,
           tensor<f32>, tensor<f32>) -> (tensor<?x1xf32>, tensor<?x1xf32>)
    return %0#0, %0#1 : tensor<?x1xf32>, tensor<?x1xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Dynamic variadic input must NOT be folded into static reshape.
  module->walk([](mlir::stablehlo::ReduceOp red) {
    EXPECT_EQ(red.getDimensions().size(), 1);
  });
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*module), 0);
}

TEST(CanonicalizeReductionsTest, NonReduceConsumerLeftUntouched) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %r1 = stablehlo.reshape %x : (tensor<2x4xf32>) -> tensor<8xf32>
    %exp = stablehlo.exponential %r1 : tensor<8xf32>
    %r2 = stablehlo.reshape %exp : (tensor<8xf32>) -> tensor<2x4xf32>
    return %r2 : tensor<2x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Reshapes not feeding a reduction must remain untouched.
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*module), 2);
}

TEST(CanonicalizeReductionsTest, MultiUseNumMulNotCanceled) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %c = stablehlo.constant dense<2.0> : tensor<4xf32>
    %mul = stablehlo.multiply %x, %c : tensor<4xf32>
    %div = stablehlo.divide %mul, %c : tensor<4xf32>
    return %div, %mul : tensor<4xf32>, tensor<4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Multi-use multiplication must not be canceled.
  EXPECT_EQ(CountOps<mlir::stablehlo::DivOp>(*module), 1);
}

TEST(CanonicalizeReductionsTest, ZeroMismatchedOrLossySplatNotCanceled) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @zero_scale(%x: tensor<4xf32>) -> tensor<4xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<4xf32>
    %mul = stablehlo.multiply %x, %zero : tensor<4xf32>
    %div = stablehlo.divide %mul, %zero : tensor<4xf32>
    return %div : tensor<4xf32>
  }
  func.func @mismatched_scale(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c2 = stablehlo.constant dense<2.0> : tensor<4xf32>
    %c3 = stablehlo.constant dense<3.0> : tensor<4xf32>
    %mul = stablehlo.multiply %x, %c2 : tensor<4xf32>
    %div = stablehlo.divide %mul, %c3 : tensor<4xf32>
    return %div : tensor<4xf32>
  }
  func.func @lossy_convert_scale(%x: tensor<4xbf16>) -> tensor<4xbf16> {
    %c_f32 = stablehlo.constant dense<1.000100e+00> : tensor<4xf32>
    %c_bf16 = stablehlo.convert %c_f32 : (tensor<4xf32>) -> tensor<4xbf16>
    %mul = stablehlo.multiply %x, %c_bf16 : tensor<4xbf16>
    %div = stablehlo.divide %mul, %c_bf16 : tensor<4xbf16>
    return %div : tensor<4xbf16>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Zero scale, mismatched scale, or lossy splat must NOT be canceled.
  EXPECT_EQ(CountOps<mlir::stablehlo::DivOp>(*module), 3);
}

}  // namespace
}  // namespace torch_tpu
