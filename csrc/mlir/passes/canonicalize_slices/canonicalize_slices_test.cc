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

#include "csrc/mlir/passes/canonicalize_slices/canonicalize_slices.h"  // NOLINT

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
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"
#include "stablehlo/transforms/Passes.h"

namespace torch_tpu {
namespace {

void InitContext(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect>();
  mlir::stablehlo::registerAllDialects(registry);
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

TEST(CanonicalizeSlicesTest, HoistsSliceAcrossAddWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<8x16xf32>, %y: tensor<8x16xf32>) -> tensor<4x8xf32> {
    %add = stablehlo.add %x, %y : tensor<8x16xf32>
    %sl = stablehlo.slice %add [0:4, 0:8]
        : (tensor<8x16xf32>) -> tensor<4x8xf32>
    return %sl : tensor<4x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // Slices hoisted onto %x and %y; Add operates on sliced tensors.
  EXPECT_EQ(CountOps<mlir::stablehlo::AddOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 2);

  auto x_val = MakeSequentialF32Tensor(context, {8, 16}, 1.0f);
  auto y_val = MakeSequentialF32Tensor(context, {8, 16}, 2.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {x_val, y_val}, config);
  auto actual =
      mlir::stablehlo::evalModule(*opt_module, {x_val, y_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest, HoistsSliceAcrossReverseWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<16x8xf32>) -> tensor<4x8xf32> {
    %rev = stablehlo.reverse %x, dims = [0]
        : (tensor<16x8xf32>) -> tensor<16x8xf32>
    %sl = stablehlo.slice %rev [2:6, 0:8]
        : (tensor<16x8xf32>) -> tensor<4x8xf32>
    return %sl : tensor<4x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ReverseOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto x_val = MakeSequentialF32Tensor(context, {16, 8}, 1.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {x_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {x_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest, HoistsSliceAcrossTransposeWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<8x16xf32>) -> tensor<4x2xf32> {
    %tr = stablehlo.transpose %x, dims = [1, 0]
        : (tensor<8x16xf32>) -> tensor<16x8xf32>
    %sl = stablehlo.slice %tr [0:4, 0:2] : (tensor<16x8xf32>) -> tensor<4x2xf32>
    return %sl : tensor<4x2xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto x_val = MakeSequentialF32Tensor(context, {8, 16}, 0.5f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {x_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {x_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest, HoistsSliceAcrossConcatSingleInputWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<4x8xf32>, %y: tensor<4x8xf32>) -> tensor<4x4xf32> {
    %cat = stablehlo.concatenate %x, %y, dim = 1
        : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x16xf32>
    %sl = stablehlo.slice %cat [0:4, 0:4]
        : (tensor<4x16xf32>) -> tensor<4x4xf32>
    return %sl : tensor<4x4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // ConcatenateOp eliminated entirely.
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto x_val = MakeSequentialF32Tensor(context, {4, 8}, 1.0f);
  auto y_val = MakeSequentialF32Tensor(context, {4, 8}, 5.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {x_val, y_val}, config);
  auto actual =
      mlir::stablehlo::evalModule(*opt_module, {x_val, y_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest, FoldsSliceOfDynamicUpdateSliceWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %base: tensor<8x8xf32>, %upd: tensor<4x8xf32>) -> tensor<2x8xf32> {
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %dus = stablehlo.dynamic_update_slice %base, %upd, %c2, %c0
        : (tensor<8x8xf32>, tensor<4x8xf32>, tensor<i32>, tensor<i32>)
        -> tensor<8x8xf32>
    %sl = stablehlo.slice %dus [2:4, 0:8]
        : (tensor<8x8xf32>) -> tensor<2x8xf32>
    return %sl : tensor<2x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // DUS folded: only slice of update remains.
  EXPECT_EQ(CountOps<mlir::stablehlo::DynamicUpdateSliceOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto base_val = MakeSequentialF32Tensor(context, {8, 8}, 0.1f);
  auto upd_val = MakeSequentialF32Tensor(context, {4, 8}, 10.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {base_val, upd_val}, config);
  auto actual =
      mlir::stablehlo::evalModule(*opt_module, {base_val, upd_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest, MergesContiguousConcatSlicesWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %sl1 = stablehlo.slice %x [0:4, 0:8] : (tensor<8x8xf32>) -> tensor<4x8xf32>
    %sl2 = stablehlo.slice %x [4:8, 0:8] : (tensor<8x8xf32>) -> tensor<4x8xf32>
    %cat = stablehlo.concatenate %sl1, %sl2, dim = 0
        : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<8x8xf32>
    return %cat : tensor<8x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  // ConcatenateOp eliminated: merged into single full slice.
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_module), 0);

  auto x_val = MakeSequentialF32Tensor(context, {8, 8}, 1.5f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {x_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {x_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest, FactorsMultiSliceSharedIntervalsWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x16xf32>) -> (tensor<2x4xf32>, tensor<2x8xf32>) {
    %sl1 = stablehlo.slice %x [0:2, 0:4] : (tensor<4x16xf32>) -> tensor<2x4xf32>
    %sl2 = stablehlo.slice %x [0:2, 0:8] : (tensor<4x16xf32>) -> tensor<2x8xf32>
    return %sl1, %sl2 : tensor<2x4xf32>, tensor<2x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 3);

  auto x_val = MakeSequentialF32Tensor(context, {4, 16}, 3.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {x_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {x_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
  ExpectAttrsClose((*actual)[1], (*expected)[1]);
}

TEST(CanonicalizeSlicesTest,
     FactorsMultiSliceSharedIntervalsOnOpResultWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<4x16xf32>, %y: tensor<4x16xf32>)
      -> (tensor<2x4xf32>, tensor<2x8xf32>) {
    %add = stablehlo.add %x, %y : tensor<4x16xf32>
    %sl1 = stablehlo.slice %add [0:2, 0:4]
        : (tensor<4x16xf32>) -> tensor<2x4xf32>
    %sl2 = stablehlo.slice %add [0:2, 0:8]
        : (tensor<4x16xf32>) -> tensor<2x8xf32>
    return %sl1, %sl2 : tensor<2x4xf32>, tensor<2x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 4);
  EXPECT_EQ(CountOps<mlir::stablehlo::AddOp>(*opt_module), 1);

  auto x_val = MakeSequentialF32Tensor(context, {4, 16}, 1.0f);
  auto y_val = MakeSequentialF32Tensor(context, {4, 16}, 2.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {x_val, y_val}, config);
  auto actual =
      mlir::stablehlo::evalModule(*opt_module, {x_val, y_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
  ExpectAttrsClose((*actual)[1], (*expected)[1]);
}

TEST(CanonicalizeSlicesTest, HoistsSliceAcrossBroadcastWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<8xf32>) -> tensor<4x4xf32> {
    %bcast = stablehlo.broadcast_in_dim %x, dims = [1]
        : (tensor<8xf32>) -> tensor<16x8xf32>
    %sl = stablehlo.slice %bcast [0:4, 0:4]
        : (tensor<16x8xf32>) -> tensor<4x4xf32>
    return %sl : tensor<4x4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::BroadcastInDimOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto x_val = MakeSequentialF32Tensor(context, {8}, 1.2f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {x_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {x_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest,
     HoistsSliceAcrossReshapePreservedAxesWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8xf32>) -> tensor<1x2x4xf32> {
    %r = stablehlo.reshape %x : (tensor<4x8xf32>) -> tensor<1x4x8xf32>
    %sl = stablehlo.slice %r [0:1, 0:2, 0:4]
        : (tensor<1x4x8xf32>) -> tensor<1x2x4xf32>
    return %sl : tensor<1x2x4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto x_val = MakeSequentialF32Tensor(context, {4, 8}, 2.1f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {x_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {x_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest, HoistsSliceAcrossDotGeneralWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %lhs: tensor<2x8x16xf32>,
      %rhs: tensor<2x16x32xf32>) -> tensor<1x4x8xf32> {
    %dot = stablehlo.dot_general %lhs, %rhs,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<2x8x16xf32>, tensor<2x16x32xf32>) -> tensor<2x8x32xf32>
    %sl = stablehlo.slice %dot [0:1, 0:4, 0:8]
        : (tensor<2x8x32xf32>) -> tensor<1x4x8xf32>
    return %sl : tensor<1x4x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::DotGeneralOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 2);

  auto lhs_val = MakeSequentialF32Tensor(context, {2, 8, 16}, 0.2f);
  auto rhs_val = MakeSequentialF32Tensor(context, {2, 16, 32}, 0.3f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {lhs_val, rhs_val}, config);
  auto actual =
      mlir::stablehlo::evalModule(*opt_module, {lhs_val, rhs_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest,
     PushesSliceAcrossCrossAxisReshapeCaseAWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8xf32>) -> tensor<4xf32> {
    %r = stablehlo.reshape %x : (tensor<4x8xf32>) -> tensor<32xf32>
    %sl = stablehlo.slice %r [0:4] : (tensor<32xf32>) -> tensor<4xf32>
    return %sl : tensor<4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto x_val = MakeSequentialF32Tensor(context, {4, 8}, 1.1f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {x_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {x_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest,
     PushesSliceAcrossCrossAxisReshapeCaseBWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8xf32>) -> tensor<16xf32> {
    %r = stablehlo.reshape %x : (tensor<4x8xf32>) -> tensor<32xf32>
    %sl = stablehlo.slice %r [8:24] : (tensor<32xf32>) -> tensor<16xf32>
    return %sl : tensor<16xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto x_val = MakeSequentialF32Tensor(context, {4, 8}, 2.2f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {x_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {x_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest,
     HoistsSliceAcrossConcatSpanningMultipleInputsWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<4x8xf32>, %y: tensor<4x8xf32>) -> tensor<4x4xf32> {
    %cat = stablehlo.concatenate %x, %y, dim = 1
        : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x16xf32>
    %sl = stablehlo.slice %cat [0:4, 6:10]
        : (tensor<4x16xf32>) -> tensor<4x4xf32>
    return %sl : tensor<4x4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 2);

  auto x_val = MakeSequentialF32Tensor(context, {4, 8}, 1.0f);
  auto y_val = MakeSequentialF32Tensor(context, {4, 8}, 5.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {x_val, y_val}, config);
  auto actual =
      mlir::stablehlo::evalModule(*opt_module, {x_val, y_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest,
     HoistsSliceAcrossSelectWithScalarPredWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %pred: tensor<i1>,
      %x: tensor<8x8xf32>,
      %y: tensor<8x8xf32>) -> tensor<4x4xf32> {
    %sel = stablehlo.select %pred, %x, %y
        : (tensor<i1>, tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    %sl = stablehlo.slice %sel [0:4, 0:4]
        : (tensor<8x8xf32>) -> tensor<4x4xf32>
    return %sl : tensor<4x4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::SelectOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 2);

  auto pred_type =
      mlir::RankedTensorType::get({}, mlir::IntegerType::get(&context, 1));
  auto pred_val = mlir::DenseElementsAttr::get(
      pred_type, mlir::BoolAttr::get(&context, true));
  auto x_val = MakeSequentialF32Tensor(context, {8, 8}, 1.0f);
  auto y_val = MakeSequentialF32Tensor(context, {8, 8}, 10.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module,
                                              {pred_val, x_val, y_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module,
                                            {pred_val, x_val, y_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest, HoistsMultiUseIdenticalSlicesWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<8x16xf32>,
      %y: tensor<8x16xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    %add = stablehlo.add %x, %y : tensor<8x16xf32>
    %sl1 = stablehlo.slice %add [0:4, 0:8]
        : (tensor<8x16xf32>) -> tensor<4x8xf32>
    %sl2 = stablehlo.slice %add [0:4, 0:8]
        : (tensor<8x16xf32>) -> tensor<4x8xf32>
    return %sl1, %sl2 : tensor<4x8xf32>, tensor<4x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::AddOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 2);

  auto x_val = MakeSequentialF32Tensor(context, {8, 16}, 1.0f);
  auto y_val = MakeSequentialF32Tensor(context, {8, 16}, 2.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {x_val, y_val}, config);
  auto actual =
      mlir::stablehlo::evalModule(*opt_module, {x_val, y_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
  ExpectAttrsClose((*actual)[1], (*expected)[1]);
}

TEST(CanonicalizeSlicesTest, FoldsSliceOfPadWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<6x8xf32>) -> tensor<2x4xf32> {
    %c_val = stablehlo.constant dense<4.200000e+01> : tensor<f32>
    %pad = stablehlo.pad %x, %c_val, low = [3, 1], high = [2, 3],
                         interior = [0, 0]
        : (tensor<6x8xf32>, tensor<f32>) -> tensor<11x12xf32>
    %sl = stablehlo.slice %pad [4:6, 3:7]
        : (tensor<11x12xf32>) -> tensor<2x4xf32>
    return %sl : tensor<2x4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::PadOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);

  auto x_val = MakeSequentialF32Tensor(context, {6, 8}, 0.7f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {x_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {x_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest, FoldsSliceOfPadWithSecondarySlicedAxis) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8xf32>) -> tensor<4x4xf32> {
    %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %pad = stablehlo.pad %x, %c0, low = [2, 0], high = [2, 0],
                         interior = [0, 0]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<8x8xf32>
    %neg = stablehlo.negate %pad : tensor<8x8xf32>
    %sl = stablehlo.slice %neg [2:6, 0:4]
        : (tensor<8x8xf32>) -> tensor<4x4xf32>
    return %sl : tensor<4x4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::PadOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::NegOp>(*opt_module), 1);

  auto x_val = MakeSequentialF32Tensor(context, {4, 8}, 1.5f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {x_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {x_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest, NegativePadWithInteriorPaddingNotFolded) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8xf32>) -> tensor<2x4xf32> {
    %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %pad = stablehlo.pad %x, %c0, low = [2, 0], high = [2, 0],
                         interior = [1, 0]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<11x8xf32>
    %sl = stablehlo.slice %pad [2:4, 0:4]
        : (tensor<11x8xf32>) -> tensor<2x4xf32>
    return %sl : tensor<2x4xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::PadOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 1);
}

TEST(CanonicalizeSlicesTest, NegativeSliceExceedingUnpaddedBoundsNotFolded) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %pad = stablehlo.pad %x, %c0, low = [2, 0], high = [2, 0],
                         interior = [0, 0]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<8x8xf32>
    %sl = stablehlo.slice %pad [1:5, 0:8]
        : (tensor<8x8xf32>) -> tensor<4x8xf32>
    return %sl : tensor<4x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::PadOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 1);
}

TEST(CanonicalizeSlicesTest, NegativeNonSliceUseOnProducerNotHoisted) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<8x16xf32>,
      %y: tensor<8x16xf32>) -> (tensor<4x8xf32>, tensor<8x16xf32>) {
    %add = stablehlo.add %x, %y : tensor<8x16xf32>
    %sl = stablehlo.slice %add [0:4, 0:8]
        : (tensor<8x16xf32>) -> tensor<4x8xf32>
    return %sl, %add : tensor<4x8xf32>, tensor<8x16xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::AddOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 1);
}

TEST(CanonicalizeSlicesTest, BailsGracefullyOnDynamicShapes) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<?x16xf32>, %y: tensor<?x16xf32>) -> tensor<?x8xf32> {
    %add = stablehlo.add %x, %y : tensor<?x16xf32>
    %sl = stablehlo.slice %add [0:4, 0:8]
        : (tensor<?x16xf32>) -> tensor<?x8xf32>
    return %sl : tensor<?x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
  EXPECT_EQ(CountOps<mlir::stablehlo::AddOp>(*module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 1);
}

TEST(CanonicalizeSlicesTest,
     HoistsSliceAcrossAddWithZeroSizedDimensionNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<0x16xf32>, %y: tensor<0x16xf32>) -> tensor<0x8xf32> {
    %add = stablehlo.add %x, %y : tensor<0x16xf32>
    %sl = stablehlo.slice %add [0:0, 0:8]
        : (tensor<0x16xf32>) -> tensor<0x8xf32>
    return %sl : tensor<0x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::AddOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 2);

  auto x_val = MakeSequentialF32Tensor(context, {0, 16}, 1.0f);
  auto y_val = MakeSequentialF32Tensor(context, {0, 16}, 2.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {x_val, y_val}, config);
  auto actual =
      mlir::stablehlo::evalModule(*opt_module, {x_val, y_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest, HoistsSliceAcrossAddWithZeroLengthWindowNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<8x16xf32>, %y: tensor<8x16xf32>) -> tensor<0x8xf32> {
    %add = stablehlo.add %x, %y : tensor<8x16xf32>
    %sl = stablehlo.slice %add [2:2, 0:8]
        : (tensor<8x16xf32>) -> tensor<0x8xf32>
    return %sl : tensor<0x8xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::AddOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 2);

  auto x_val = MakeSequentialF32Tensor(context, {8, 16}, 1.0f);
  auto y_val = MakeSequentialF32Tensor(context, {8, 16}, 2.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {x_val, y_val}, config);
  auto actual =
      mlir::stablehlo::evalModule(*opt_module, {x_val, y_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest, FoldsRank0ScalarSliceNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<f32>) -> tensor<f32> {
    %sl = stablehlo.slice %x [] : (tensor<f32>) -> tensor<f32>
    return %sl : tensor<f32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 0);

  auto x_val = MakeSequentialF32Tensor(context, {}, 3.14f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected = mlir::stablehlo::evalModule(*ref_module, {x_val}, config);
  auto actual = mlir::stablehlo::evalModule(*opt_module, {x_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest,
     HandlesLargeUi64StartIndexWithoutNegativeWrapping) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %base: tensor<8x8xf32>, %upd: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %c_large = stablehlo.constant dense<18446744073709551615> : tensor<ui64>
    %dus = stablehlo.dynamic_update_slice %base, %upd, %c_large, %c_large
        : (tensor<8x8xf32>, tensor<2x2xf32>, tensor<ui64>, tensor<ui64>)
          -> tensor<8x8xf32>
    %sl = stablehlo.slice %dus [6:8, 6:8]
        : (tensor<8x8xf32>) -> tensor<2x2xf32>
    return %sl : tensor<2x2xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::DynamicUpdateSliceOp>(*module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 0);
}

TEST(CanonicalizeSlicesTest, PreservesDynamicInputWhenFoldingParentSlice) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<?x16xf32>) -> tensor<4x8xf32> {
    %s1 = stablehlo.slice %x [0:8, 0:16]
        : (tensor<?x16xf32>) -> tensor<8x16xf32>
    %s2 = stablehlo.slice %s1 [0:4, 0:8]
        : (tensor<8x16xf32>) -> tensor<4x8xf32>
    return %s2 : tensor<4x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*module), 2);
}

TEST(CanonicalizeSlicesTest, BailsGracefullyOnDynamicConcatSliceInputs) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<?x8xf32>) -> tensor<8x8xf32> {
    %s1 = stablehlo.slice %x [0:4, 0:8]
        : (tensor<?x8xf32>) -> tensor<4x8xf32>
    %s2 = stablehlo.slice %x [4:8, 0:8]
        : (tensor<?x8xf32>) -> tensor<4x8xf32>
    %cat = stablehlo.concatenate %s1, %s2, dim = 0
        : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<8x8xf32>
    return %cat : tensor<8x8xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*module), 1);
}

TEST(CanonicalizeSlicesTest, MultiStepCompositionCleansDeadSlicesWithNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<8x8xf32>, %y: tensor<8x8xf32>) -> tensor<2x4xf32> {
    %add = stablehlo.add %x, %y : tensor<8x8xf32>
    %sl1 = stablehlo.slice %add [0:4, 0:4]
        : (tensor<8x8xf32>) -> tensor<4x4xf32>
    %sl2 = stablehlo.slice %add [4:8, 0:4]
        : (tensor<8x8xf32>) -> tensor<4x4xf32>
    %cat = stablehlo.concatenate %sl1, %sl2, dim = 0
        : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
    %sl3 = stablehlo.slice %cat [0:2, 0:4]
        : (tensor<8x4xf32>) -> tensor<2x4xf32>
    return %sl3 : tensor<2x4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_module), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::AddOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 2);

  auto x_val = MakeSequentialF32Tensor(context, {8, 8}, 1.0f);
  auto y_val = MakeSequentialF32Tensor(context, {8, 8}, 2.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {x_val, y_val}, config);
  auto actual =
      mlir::stablehlo::evalModule(*opt_module, {x_val, y_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest,
     HoistsSliceAcrossChloBroadcastAddWithSize1OperandNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<4x8xf32>, %y: tensor<1x8xf32>) -> tensor<2x4xf32> {
    %add = chlo.broadcast_add %x, %y
        : (tensor<4x8xf32>, tensor<1x8xf32>) -> tensor<4x8xf32>
    %sl = stablehlo.slice %add [0:2, 0:4]
        : (tensor<4x8xf32>) -> tensor<2x4xf32>
    return %sl : tensor<2x4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::chlo::BroadcastAddOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 2);

  mlir::PassManager pm_legalize(&context);
  pm_legalize.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createChloLegalizeToStablehloPass());
  ASSERT_TRUE(mlir::succeeded(pm_legalize.run(*ref_module)));
  ASSERT_TRUE(mlir::succeeded(pm_legalize.run(*opt_module)));

  auto x_val = MakeSequentialF32Tensor(context, {4, 8}, 1.0f);
  auto y_val = MakeSequentialF32Tensor(context, {1, 8}, 2.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {x_val, y_val}, config);
  auto actual =
      mlir::stablehlo::evalModule(*opt_module, {x_val, y_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

TEST(CanonicalizeSlicesTest,
     HoistsSliceAcrossChloBroadcastAddWithLowerRankOperandNumerics) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<4x8xf32>, %y: tensor<8xf32>) -> tensor<2x4xf32> {
    %add = chlo.broadcast_add %x, %y
        : (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4x8xf32>
    %sl = stablehlo.slice %add [0:2, 0:4]
        : (tensor<4x8xf32>) -> tensor<2x4xf32>
    return %sl : tensor<2x4xf32>
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
  pm.addNestedPass<mlir::func::FuncOp>(createCanonicalizeSlicesPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*opt_module)));

  EXPECT_EQ(CountOps<mlir::chlo::BroadcastAddOp>(*opt_module), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_module), 2);

  mlir::PassManager pm_legalize(&context);
  pm_legalize.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createChloLegalizeToStablehloPass());
  ASSERT_TRUE(mlir::succeeded(pm_legalize.run(*ref_module)));
  ASSERT_TRUE(mlir::succeeded(pm_legalize.run(*opt_module)));

  auto x_val = MakeSequentialF32Tensor(context, {4, 8}, 1.0f);
  auto y_val = MakeSequentialF32Tensor(context, {8}, 2.0f);
  mlir::stablehlo::InterpreterConfiguration config;
  auto expected =
      mlir::stablehlo::evalModule(*ref_module, {x_val, y_val}, config);
  auto actual =
      mlir::stablehlo::evalModule(*opt_module, {x_val, y_val}, config);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectAttrsClose((*actual)[0], (*expected)[0]);
}

}  // namespace
}  // namespace torch_tpu
