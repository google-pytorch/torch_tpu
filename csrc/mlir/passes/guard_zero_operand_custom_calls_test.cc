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

#include "csrc/mlir/passes/guard_zero_operand_custom_calls.h"

#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// Two `lax.empty`-style allocations of the same type, each consumed by a
// distinct computation. Distinct allocations must stay distinct.
constexpr char kTwoAllocationsIr[] = R"mlir(
module {
  func.func @main(%x: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
    %a = stablehlo.custom_call @AllocateBuffer() {has_side_effect = false} : () -> tensor<8xf32>
    %b = stablehlo.custom_call @AllocateBuffer() {has_side_effect = false} : () -> tensor<8xf32>
    %ra = stablehlo.add %x, %a : tensor<8xf32>
    %rb = stablehlo.add %x, %b : tensor<8xf32>
    return %ra, %rb : tensor<8xf32>, tensor<8xf32>
  }
}
)mlir";

class GuardZeroOperandCustomCallsTest : public testing::Test {
 protected:
  GuardZeroOperandCustomCallsTest() {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    registry.insert<mlir::func::FuncDialect>();
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> Parse(llvm::StringRef ir) {
    return mlir::parseSourceString<mlir::ModuleOp>(ir, &context_);
  }

  static llvm::SmallVector<mlir::stablehlo::CustomCallOp> CustomCalls(
      mlir::ModuleOp module, llvm::StringRef target) {
    llvm::SmallVector<mlir::stablehlo::CustomCallOp> calls;
    module.walk([&](mlir::stablehlo::CustomCallOp op) {
      if (op.getCallTargetName() == target) {
        calls.push_back(op);
      }
    });
    return calls;
  }

  mlir::MLIRContext context_;
};

// Documents the underlying problem: without the guard, MLIR CSE merges
// distinct zero-operand, side-effect-free custom calls.
TEST_F(GuardZeroOperandCustomCallsTest, CseMergesUnguardedAllocations) {
  mlir::OwningOpRef<mlir::ModuleOp> module = Parse(kTwoAllocationsIr);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context_);
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  EXPECT_EQ(CustomCalls(*module, "AllocateBuffer").size(), 1);
}

TEST_F(GuardZeroOperandCustomCallsTest, GuardMarksAllocationsSideEffecting) {
  mlir::OwningOpRef<mlir::ModuleOp> module = Parse(kTwoAllocationsIr);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context_);
  pm.addNestedPass<mlir::func::FuncOp>(createGuardZeroOperandCustomCallsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  auto calls = CustomCalls(*module, "AllocateBuffer");
  ASSERT_EQ(calls.size(), 2);
  for (mlir::stablehlo::CustomCallOp op : calls) {
    EXPECT_TRUE(op.getHasSideEffect());
    EXPECT_FALSE(mlir::isMemoryEffectFree(op));
    EXPECT_TRUE(op->hasAttr(kZeroOperandCustomCallGuardAttr));
  }
}

TEST_F(GuardZeroOperandCustomCallsTest,
       GuardCseUnguardKeepsAllocationsDistinctAndRestoresAttrs) {
  mlir::OwningOpRef<mlir::ModuleOp> module = Parse(kTwoAllocationsIr);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context_);
  pm.addNestedPass<mlir::func::FuncOp>(createGuardZeroOperandCustomCallsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      createUnguardZeroOperandCustomCallsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  auto calls = CustomCalls(*module, "AllocateBuffer");
  ASSERT_EQ(calls.size(), 2);
  EXPECT_NE(calls[0].getResult(0), calls[1].getResult(0));
  for (mlir::stablehlo::CustomCallOp op : calls) {
    // The explicit `has_side_effect = false` attribute is restored.
    ASSERT_TRUE(op.getHasSideEffectAttr());
    EXPECT_FALSE(op.getHasSideEffect());
    EXPECT_TRUE(mlir::isMemoryEffectFree(op));
    EXPECT_FALSE(op->hasAttr(kZeroOperandCustomCallGuardAttr));
  }

  int add_count = 0;
  module->walk([&](mlir::stablehlo::AddOp) { ++add_count; });
  EXPECT_EQ(add_count, 2);
}

TEST_F(GuardZeroOperandCustomCallsTest, RestoresAbsentHasSideEffectAttr) {
  constexpr char kIr[] = R"mlir(
module {
  func.func @main() -> tensor<8xf32> {
    %a = stablehlo.custom_call @AllocateBuffer() : () -> tensor<8xf32>
    return %a : tensor<8xf32>
  }
}
)mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module = Parse(kIr);
  ASSERT_TRUE(module);
  ASSERT_FALSE(
      CustomCalls(*module, "AllocateBuffer")[0].getHasSideEffectAttr());

  mlir::PassManager pm(&context_);
  pm.addNestedPass<mlir::func::FuncOp>(createGuardZeroOperandCustomCallsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      createUnguardZeroOperandCustomCallsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  auto calls = CustomCalls(*module, "AllocateBuffer");
  ASSERT_EQ(calls.size(), 1);
  EXPECT_FALSE(calls[0].getHasSideEffectAttr());
  EXPECT_FALSE(calls[0].getHasSideEffect());
  EXPECT_FALSE(calls[0]->hasAttr(kZeroOperandCustomCallGuardAttr));
}

TEST_F(GuardZeroOperandCustomCallsTest, MatchesAnyZeroOperandTargetName) {
  constexpr char kIr[] = R"mlir(
module {
  func.func @main() -> (tensor<8xf32>, tensor<8xf32>) {
    %a = stablehlo.custom_call @SomeOtherFreshValue() : () -> tensor<8xf32>
    %b = stablehlo.custom_call @SomeOtherFreshValue() : () -> tensor<8xf32>
    return %a, %b : tensor<8xf32>, tensor<8xf32>
  }
}
)mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module = Parse(kIr);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context_);
  pm.addNestedPass<mlir::func::FuncOp>(createGuardZeroOperandCustomCallsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      createUnguardZeroOperandCustomCallsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  EXPECT_EQ(CustomCalls(*module, "SomeOtherFreshValue").size(), 2);
}

TEST_F(GuardZeroOperandCustomCallsTest, LeavesCustomCallsWithOperandsAlone) {
  constexpr char kIr[] = R"mlir(
module {
  func.func @main(%x: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
    %a = stablehlo.custom_call @pure_fn(%x) {has_side_effect = false} : (tensor<8xf32>) -> tensor<8xf32>
    %b = stablehlo.custom_call @pure_fn(%x) {has_side_effect = false} : (tensor<8xf32>) -> tensor<8xf32>
    return %a, %b : tensor<8xf32>, tensor<8xf32>
  }
}
)mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module = Parse(kIr);
  ASSERT_TRUE(module);

  {
    mlir::PassManager pm(&context_);
    pm.addNestedPass<mlir::func::FuncOp>(
        createGuardZeroOperandCustomCallsPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
  }
  for (mlir::stablehlo::CustomCallOp op : CustomCalls(*module, "pure_fn")) {
    EXPECT_FALSE(op.getHasSideEffect());
    EXPECT_FALSE(op->hasAttr(kZeroOperandCustomCallGuardAttr));
  }

  // Pure custom calls with identical operands are still CSE'd.
  mlir::PassManager pm(&context_);
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      createUnguardZeroOperandCustomCallsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
  EXPECT_EQ(CustomCalls(*module, "pure_fn").size(), 1);
}

TEST_F(GuardZeroOperandCustomCallsTest,
       LeavesAlreadySideEffectingCustomCallsAlone) {
  constexpr char kIr[] = R"mlir(
module {
  func.func @main() -> tensor<8xf32> {
    %a = stablehlo.custom_call @effectful() {has_side_effect = true} : () -> tensor<8xf32>
    return %a : tensor<8xf32>
  }
}
)mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module = Parse(kIr);
  ASSERT_TRUE(module);

  {
    mlir::PassManager pm(&context_);
    pm.addNestedPass<mlir::func::FuncOp>(
        createGuardZeroOperandCustomCallsPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
  }
  auto calls = CustomCalls(*module, "effectful");
  ASSERT_EQ(calls.size(), 1);
  EXPECT_FALSE(calls[0]->hasAttr(kZeroOperandCustomCallGuardAttr));

  mlir::PassManager pm(&context_);
  pm.addNestedPass<mlir::func::FuncOp>(
      createUnguardZeroOperandCustomCallsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
  EXPECT_TRUE(CustomCalls(*module, "effectful")[0].getHasSideEffect());
}

}  // namespace
}  // namespace torch_tpu
