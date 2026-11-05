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

#include "csrc/mlir/passes/cancel_inverse_view_transposes/cancel_inverse_view_transposes.h"  // NOLINT

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
#include "mlir/Support/LogicalResult.h"  // NOLINT
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
    vals[i] = base + static_cast<float>((i * 11 + 3) % 97) * 0.1f;
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

TEST(CancelInverseViewTransposesTest,
     CancelsInverseTransposeAroundSingleSlice) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x4x8x16xf32>) -> tensor<1x4x8x8xf32> {
    %t1 = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<1x4x8x16xf32>) -> tensor<8x4x16x1xf32>
    %sl = stablehlo.slice %t1 [0:8, 0:4, 8:16, 0:1]
        : (tensor<8x4x16x1xf32>) -> tensor<8x4x8x1xf32>
    %t2 = stablehlo.transpose %sl, dims = [3, 1, 0, 2]
        : (tensor<8x4x8x1xf32>) -> tensor<1x4x8x8xf32>
    return %t2 : tensor<1x4x8x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 1);
}

TEST(CancelInverseViewTransposesTest,
     CancelsInverseTransposeAroundMultiDimSlice) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x4x8x16xf32>) -> tensor<1x2x8x8xf32> {
    %t1 = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<1x4x8x16xf32>) -> tensor<8x4x16x1xf32>
    %sl = stablehlo.slice %t1 [0:8, 1:3, 4:12, 0:1]
        : (tensor<8x4x16x1xf32>) -> tensor<8x2x8x1xf32>
    %t2 = stablehlo.transpose %sl, dims = [3, 1, 0, 2]
        : (tensor<8x2x8x1xf32>) -> tensor<1x2x8x8xf32>
    return %t2 : tensor<1x2x8x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto in_val = MakeSequentialF32Tensor(context, {1, 4, 8, 16});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CancelInverseViewTransposesTest,
     CancelsInverseTransposeAroundChainedUnitStrideSlices) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x4x8x16xf32>) -> tensor<1x4x8x4xf32> {
    %t1 = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<1x4x8x16xf32>) -> tensor<8x4x16x1xf32>
    %sl1 = stablehlo.slice %t1 [0:8, 0:4, 2:14, 0:1]
        : (tensor<8x4x16x1xf32>) -> tensor<8x4x12x1xf32>
    %sl2 = stablehlo.slice %sl1 [0:8, 0:4, 1:5, 0:1]
        : (tensor<8x4x12x1xf32>) -> tensor<8x4x4x1xf32>
    %t2 = stablehlo.transpose %sl2, dims = [3, 1, 0, 2]
        : (tensor<8x4x4x1xf32>) -> tensor<1x4x8x4xf32>
    return %t2 : tensor<1x4x8x4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto in_val = MakeSequentialF32Tensor(context, {1, 4, 8, 16});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CancelInverseViewTransposesTest,
     CancelsInverseTransposeAroundUnitDimReshapes) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x4x8x16xf32>) -> tensor<1x1x4x8x16xf32> {
    %t1 = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<1x4x8x16xf32>) -> tensor<8x4x16x1xf32>
    %r = stablehlo.reshape %t1
        : (tensor<8x4x16x1xf32>) -> tensor<1x8x4x16x1xf32>
    %t2 = stablehlo.transpose %r, dims = [0, 4, 2, 1, 3]
        : (tensor<1x8x4x16x1xf32>) -> tensor<1x1x4x8x16xf32>
    return %t2 : tensor<1x1x4x8x16xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*opt_module), 1);

  auto in_val = MakeSequentialF32Tensor(context, {1, 4, 8, 16});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CancelInverseViewTransposesTest, CancelsRank5InverseTransposeAroundSlice) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<2x3x4x5x6xf32>) -> tensor<2x3x2x5x6xf32> {
    %t1 = stablehlo.transpose %x, dims = [4, 1, 2, 3, 0]
        : (tensor<2x3x4x5x6xf32>) -> tensor<6x3x4x5x2xf32>
    %sl = stablehlo.slice %t1 [0:6, 0:3, 1:3, 0:5, 0:2]
        : (tensor<6x3x4x5x2xf32>) -> tensor<6x3x2x5x2xf32>
    %t2 = stablehlo.transpose %sl, dims = [4, 1, 2, 3, 0]
        : (tensor<6x3x2x5x2xf32>) -> tensor<2x3x2x5x6xf32>
    return %t2 : tensor<2x3x2x5x6xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 1);
}

TEST(CancelInverseViewTransposesTest,
     CancelsRank3InverseTransposeAroundReshape) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x4x8xf32>) -> tensor<4x8x1x1xf32> {
    %t1 = stablehlo.transpose %x, dims = [2, 1, 0]
        : (tensor<1x4x8xf32>) -> tensor<8x4x1xf32>
    %r = stablehlo.reshape %t1
        : (tensor<8x4x1xf32>) -> tensor<1x8x4x1xf32>
    %t2 = stablehlo.transpose %r, dims = [2, 1, 0, 3]
        : (tensor<1x8x4x1xf32>) -> tensor<4x8x1x1xf32>
    return %t2 : tensor<4x8x1x1xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*module), 1);
}

TEST(CancelInverseViewTransposesTest, DoesNotCancelNonInverseTranspose) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x4x8x16xf32>) -> tensor<8x1x4x8xf32> {
    %t1 = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<1x4x8x16xf32>) -> tensor<8x4x16x1xf32>
    %sl = stablehlo.slice %t1 [0:8, 0:4, 0:8, 0:1]
        : (tensor<8x4x16x1xf32>) -> tensor<8x4x8x1xf32>
    %t2 = stablehlo.transpose %sl, dims = [0, 3, 1, 2]
        : (tensor<8x4x8x1xf32>) -> tensor<8x1x4x8xf32>
    return %t2 : tensor<8x1x4x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Non-inverse composition: transposes must NOT be cancelled.
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module), 2);
}

TEST(CancelInverseViewTransposesTest,
     DoesNotCancelTransposeAroundNonUnitDimReshape) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8x16xf32>) -> tensor<16x8x4xf32> {
    %t1 = stablehlo.transpose %x, dims = [2, 0, 1]
        : (tensor<4x8x16xf32>) -> tensor<16x4x8xf32>
    %r = stablehlo.reshape %t1
        : (tensor<16x4x8xf32>) -> tensor<4x16x8xf32>
    %t2 = stablehlo.transpose %r, dims = [1, 2, 0]
        : (tensor<4x16x8xf32>) -> tensor<16x8x4xf32>
    return %t2 : tensor<16x8x4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // Non-unit-dim reshape alters non-unit dimensions (16x4x8 -> 4x16x8):
  // transposes must NOT cancel across it.
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_module), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*opt_module), 1);

  auto in_val = MakeSequentialF32Tensor(context, {4, 8, 16});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CancelInverseViewTransposesTest,
     DoesNotCancelTransposeAroundNonUnitDimReshapeThenSlice) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8x16xf32>) -> tensor<16x4x4xf32> {
    %t1 = stablehlo.transpose %x, dims = [2, 0, 1]
        : (tensor<4x8x16xf32>) -> tensor<16x4x8xf32>
    %r = stablehlo.reshape %t1
        : (tensor<16x4x8xf32>) -> tensor<4x16x8xf32>
    %sl = stablehlo.slice %r [0:4, 0:16, 0:4]
        : (tensor<4x16x8xf32>) -> tensor<4x16x4xf32>
    %t2 = stablehlo.transpose %sl, dims = [1, 2, 0]
        : (tensor<4x16x4xf32>) -> tensor<16x4x4xf32>
    return %t2 : tensor<16x4x4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // Non-unit-dim reshape followed by slice: transposes must NOT cancel.
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_module), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*opt_module), 1);

  auto in_val = MakeSequentialF32Tensor(context, {4, 8, 16});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CancelInverseViewTransposesTest,
     CancelsInverseTransposeAroundMultiDimStridedSlice) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x4x16x8xf32>) -> tensor<1x2x5x8xf32> {
    %t1 = stablehlo.transpose %x, dims = [3, 1, 2, 0]
        : (tensor<1x4x16x8xf32>) -> tensor<8x4x16x1xf32>
    %sl = stablehlo.slice %t1 [0:8, 0:4:2, 1:16:3, 0:1]
        : (tensor<8x4x16x1xf32>) -> tensor<8x2x5x1xf32>
    %t2 = stablehlo.transpose %sl, dims = [3, 1, 2, 0]
        : (tensor<8x2x5x1xf32>) -> tensor<1x2x5x8xf32>
    return %t2 : tensor<1x2x5x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto in_val = MakeSequentialF32Tensor(context, {1, 4, 16, 8});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CancelInverseViewTransposesTest, DoesNotCancelChainedStridedSlice) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8x16xf32>) -> tensor<4x4x8xf32> {
    %t1 = stablehlo.transpose %x, dims = [1, 2, 0]
        : (tensor<4x8x16xf32>) -> tensor<8x16x4xf32>
    %sl1 = stablehlo.slice %t1 [0:8:2, 0:16, 0:4]
        : (tensor<8x16x4xf32>) -> tensor<4x16x4xf32>
    %sl2 = stablehlo.slice %sl1 [0:4, 0:8, 0:4]
        : (tensor<4x16x4xf32>) -> tensor<4x8x4xf32>
    %t2 = stablehlo.transpose %sl2, dims = [2, 0, 1]
        : (tensor<4x8x4xf32>) -> tensor<4x4x8xf32>
    return %t2 : tensor<4x4x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Chained strided slice must not cancel.
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 2);
}

TEST(CancelInverseViewTransposesTest,
     CancelsSqueezedInverseTransposeAroundSlice) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x4x8x16xf32>) -> tensor<4x1x8x8xf32> {
    %t1 = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<1x4x8x16xf32>) -> tensor<8x4x16x1xf32>
    %sl = stablehlo.slice %t1 [0:8, 0:4, 0:8, 0:1]
        : (tensor<8x4x16x1xf32>) -> tensor<8x4x8x1xf32>
    %t2 = stablehlo.transpose %sl, dims = [1, 3, 0, 2]
        : (tensor<8x4x8x1xf32>) -> tensor<4x1x8x8xf32>
    return %t2 : tensor<4x1x8x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*opt_module), 1);

  auto in_val = MakeSequentialF32Tensor(context, {1, 4, 8, 16});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CancelInverseViewTransposesTest,
     CancelsInverseTransposeAroundSliceThenUnitDimReshape) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8x16xf32>) -> tensor<1x4x8x8xf32> {
    %t1 = stablehlo.transpose %x, dims = [1, 2, 0]
        : (tensor<4x8x16xf32>) -> tensor<8x16x4xf32>
    %sl = stablehlo.slice %t1 [0:8, 0:8, 0:4]
        : (tensor<8x16x4xf32>) -> tensor<8x8x4xf32>
    %r = stablehlo.reshape %sl
        : (tensor<8x8x4xf32>) -> tensor<1x8x8x4xf32>
    %t2 = stablehlo.transpose %r, dims = [0, 3, 1, 2]
        : (tensor<1x8x8x4xf32>) -> tensor<1x4x8x8xf32>
    return %t2 : tensor<1x4x8x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto in_val = MakeSequentialF32Tensor(context, {4, 8, 16});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CancelInverseViewTransposesTest,
     CancelsInverseTransposeAroundUnitDimReshapeThenSlice) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8x16xf32>) -> tensor<1x4x8x8xf32> {
    %t1 = stablehlo.transpose %x, dims = [1, 2, 0]
        : (tensor<4x8x16xf32>) -> tensor<8x16x4xf32>
    %r = stablehlo.reshape %t1
        : (tensor<8x16x4xf32>) -> tensor<1x8x16x4xf32>
    %sl = stablehlo.slice %r [0:1, 0:8, 0:8, 0:4]
        : (tensor<1x8x16x4xf32>) -> tensor<1x8x8x4xf32>
    %t2 = stablehlo.transpose %sl, dims = [0, 3, 1, 2]
        : (tensor<1x8x8x4xf32>) -> tensor<1x4x8x8xf32>
    return %t2 : tensor<1x4x8x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto in_val = MakeSequentialF32Tensor(context, {4, 8, 16});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CancelInverseViewTransposesTest, CancelsMultiUseRopeHalfSplit) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8x16xf32>)
      -> (tensor<4x8x8xf32>, tensor<4x8x8xf32>) {
    %t1 = stablehlo.transpose %x, dims = [1, 0, 2]
        : (tensor<4x8x16xf32>) -> tensor<8x4x16xf32>
    %sl1 = stablehlo.slice %t1 [0:8, 0:4, 0:8]
        : (tensor<8x4x16xf32>) -> tensor<8x4x8xf32>
    %t2a = stablehlo.transpose %sl1, dims = [1, 0, 2]
        : (tensor<8x4x8xf32>) -> tensor<4x8x8xf32>
    %sl2 = stablehlo.slice %t1 [0:8, 0:4, 8:16]
        : (tensor<8x4x16xf32>) -> tensor<8x4x8xf32>
    %t2b = stablehlo.transpose %sl2, dims = [1, 0, 2]
        : (tensor<8x4x8xf32>) -> tensor<4x8x8xf32>
    return %t2a, %t2b : tensor<4x8x8xf32>, tensor<4x8x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Both inverse transposes cancel; t1 has no remaining users, so 0 remain.
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 2);
}

TEST(CancelInverseViewTransposesTest,
     CancelsWhenTransposeHasExtraNonMatchingUser) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8x16xf32>)
      -> (tensor<4x8x8xf32>, tensor<8x4x16xf32>) {
    %t1 = stablehlo.transpose %x, dims = [1, 0, 2]
        : (tensor<4x8x16xf32>) -> tensor<8x4x16xf32>
    %sl1 = stablehlo.slice %t1 [0:8, 0:4, 0:8]
        : (tensor<8x4x16xf32>) -> tensor<8x4x8xf32>
    %t2 = stablehlo.transpose %sl1, dims = [1, 0, 2]
        : (tensor<8x4x8xf32>) -> tensor<4x8x8xf32>
    return %t2, %t1 : tensor<4x8x8xf32>, tensor<8x4x16xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // t2 cancels with a slice from %x. %t1 remains because of the return use.
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 1);
}

TEST(CancelInverseViewTransposesTest, BailsGracefullyOnDynamicShapes) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<?x4x8x16xf32>) -> tensor<?x4x8x8xf32> {
    %t1 = stablehlo.transpose %x, dims = [2, 1, 3, 0]
        : (tensor<?x4x8x16xf32>) -> tensor<8x4x16x?xf32>
    %sl = stablehlo.slice %t1 [0:8, 0:4, 0:8, 0:1]
        : (tensor<8x4x16x?xf32>) -> tensor<8x4x8x?xf32>
    %t2 = stablehlo.transpose %sl, dims = [3, 1, 0, 2]
        : (tensor<8x4x8x?xf32>) -> tensor<?x4x8x8xf32>
    return %t2 : tensor<?x4x8x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  // Must execute successfully without crash.
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*module), 2);
}

TEST(CancelInverseViewTransposesTest,
     BailsCleanlyWhenSliceShrinksNonUnitDimToUnitDim) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8x16xf32>) -> tensor<1x16x4xf32> {
    %t1 = stablehlo.transpose %x, dims = [1, 0, 2]
        : (tensor<4x8x16xf32>) -> tensor<8x4x16xf32>
    %sl = stablehlo.slice %t1 [0:1, 0:4, 0:16]
        : (tensor<8x4x16xf32>) -> tensor<1x4x16xf32>
    %t2 = stablehlo.transpose %sl, dims = [0, 2, 1]
        : (tensor<1x4x16xf32>) -> tensor<1x16x4xf32>
    return %t2 : tensor<1x16x4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // Squeezed perm size changes (3 vs 2), so cancellation bails out cleanly.
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_module), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto in_val = MakeSequentialF32Tensor(context, {4, 8, 16});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {in_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {in_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CancelInverseViewTransposesTest,
     BailsCleanlyOnZeroElementTensorsAndZeroBounds) {
  // 1. Zero dimension in input tensor: exercises d > 1 filtering.
  static constexpr const char* kZeroDimIr = R"mlir(
module {
  func.func @main(%x: tensor<0x4x8xf32>) -> tensor<0x4x8xf32> {
    %t1 = stablehlo.transpose %x, dims = [1, 0, 2]
        : (tensor<0x4x8xf32>) -> tensor<4x0x8xf32>
    %sl = stablehlo.slice %t1 [0:4, 0:0, 0:8]
        : (tensor<4x0x8xf32>) -> tensor<4x0x8xf32>
    %t2 = stablehlo.transpose %sl, dims = [1, 0, 2]
        : (tensor<4x0x8xf32>) -> tensor<0x4x8xf32>
    return %t2 : tensor<0x4x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_mod1 =
      mlir::parseSourceString<mlir::ModuleOp>(kZeroDimIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod1 =
      mlir::parseSourceString<mlir::ModuleOp>(kZeroDimIr, &context);
  ASSERT_TRUE(ref_mod1 && opt_mod1);

  mlir::PassManager pm1(&context);
  pm1.addNestedPass<mlir::func::FuncOp>(
      createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm1.run(*opt_mod1)));

  // Zero-dim bails out in d > 1 filtering and slice bound checks.
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_mod1), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_mod1), 1);

  auto in_val1 = MakeSequentialF32Tensor(context, {0, 4, 8});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected1 = mlir::stablehlo::evalModule(*ref_mod1, {in_val1}, config);
  auto actual1 = mlir::stablehlo::evalModule(*opt_mod1, {in_val1}, config);
  ASSERT_TRUE(mlir::succeeded(expected1));
  ASSERT_TRUE(mlir::succeeded(actual1));
  ExpectAttrsClose((*actual1)[0], (*expected1)[0]);

  // 2. Slice bound with 0 limit on a unit dimension (exercises line 248).
  static constexpr const char* kZeroSliceBoundIr = R"mlir(
module {
  func.func @main(%x: tensor<1x4x8xf32>) -> tensor<0x4x8xf32> {
    %t1 = stablehlo.transpose %x, dims = [1, 0, 2]
        : (tensor<1x4x8xf32>) -> tensor<4x1x8xf32>
    %sl = stablehlo.slice %t1 [0:4, 0:0, 0:8]
        : (tensor<4x1x8xf32>) -> tensor<4x0x8xf32>
    %t2 = stablehlo.transpose %sl, dims = [1, 0, 2]
        : (tensor<4x0x8xf32>) -> tensor<0x4x8xf32>
    return %t2 : tensor<0x4x8xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod2 =
      mlir::parseSourceString<mlir::ModuleOp>(kZeroSliceBoundIr, &context);
  ASSERT_TRUE(mod2);
  mlir::PassManager pm2(&context);
  pm2.addNestedPass<mlir::func::FuncOp>(
      createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm2.run(*mod2)));
  // Slice bound check rejected cancellation on line 248 (cum_limits != 1).
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*mod2), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*mod2), 1);
}

TEST(CancelInverseViewTransposesTest, HandlesAllUnitAndRank0Shapes) {
  // 1. All unit dimensions: tensor<1x1x1xf32>.
  static constexpr const char* kAllUnitIr = R"mlir(
module {
  func.func @main(%x: tensor<1x1x1xf32>) -> tensor<1x1x1xf32> {
    %t1 = stablehlo.transpose %x, dims = [2, 1, 0]
        : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %sl = stablehlo.slice %t1 [0:1, 0:1, 0:1]
        : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %t2 = stablehlo.transpose %sl, dims = [2, 1, 0]
        : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    return %t2 : tensor<1x1x1xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> ref_unit =
      mlir::parseSourceString<mlir::ModuleOp>(kAllUnitIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_unit =
      mlir::parseSourceString<mlir::ModuleOp>(kAllUnitIr, &context);
  ASSERT_TRUE(ref_unit && opt_unit);

  mlir::PassManager pm_unit(&context);
  pm_unit.addNestedPass<mlir::func::FuncOp>(
      createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm_unit.run(*opt_unit)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_unit), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_unit), 1);

  auto in_unit = MakeSequentialF32Tensor(context, {1, 1, 1});
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected_unit =
      mlir::stablehlo::evalModule(*ref_unit, {in_unit}, config);
  auto actual_unit = mlir::stablehlo::evalModule(*opt_unit, {in_unit}, config);
  ASSERT_TRUE(mlir::succeeded(expected_unit));
  ASSERT_TRUE(mlir::succeeded(actual_unit));
  ExpectAttrsClose((*actual_unit)[0], (*expected_unit)[0]);

  // 2. Rank 0 tensor: tensor<f32>.
  static constexpr const char* kRank0Ir = R"mlir(
module {
  func.func @main(%x: tensor<f32>) -> tensor<f32> {
    %c = stablehlo.constant dense<3.5> : tensor<f32>
    %add = stablehlo.add %x, %c : tensor<f32>
    return %add : tensor<f32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mod_rank0 =
      mlir::parseSourceString<mlir::ModuleOp>(kRank0Ir, &context);
  ASSERT_TRUE(mod_rank0);
  mlir::PassManager pm_rank0(&context);
  pm_rank0.addNestedPass<mlir::func::FuncOp>(
      createCancelInverseViewTransposesPass());
  ASSERT_TRUE(mlir::succeeded(pm_rank0.run(*mod_rank0)));
}

}  // namespace
}  // namespace torch_tpu
