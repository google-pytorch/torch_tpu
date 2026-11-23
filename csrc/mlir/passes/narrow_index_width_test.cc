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

#include "csrc/mlir/passes/narrow_index_width.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"

namespace torch_tpu {
namespace {

using testing::HasSubstr;
using testing::Not;

void InitContext(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

std::string PrintModule(mlir::ModuleOp module) {
  std::string out;
  llvm::raw_string_ostream os(out);
  module.print(os);
  return out;
}

mlir::LogicalResult RunPass(mlir::ModuleOp module, mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  pm.addNestedPass<mlir::func::FuncOp>(createNarrowIndexWidthPass());
  if (mlir::failed(pm.run(module))) {
    return mlir::failure();
  }
  return mlir::verify(module);
}

mlir::DenseElementsAttr MakeI32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      llvm::ArrayRef<int32_t> vals) {
  auto type =
      mlir::RankedTensorType::get(shape, mlir::IntegerType::get(&ctx, 32));
  return mlir::DenseElementsAttr::get(type, vals);
}

mlir::DenseElementsAttr MakeF32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      llvm::ArrayRef<float> vals) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  return mlir::DenseElementsAttr::get(type, vals);
}

void ExpectExactF32Match(const mlir::DenseElementsAttr& expected,
                         const mlir::DenseElementsAttr& actual) {
  ASSERT_EQ(expected.getType(), actual.getType());
  auto exp_it = expected.value_begin<llvm::APFloat>();
  auto act_it = actual.value_begin<llvm::APFloat>();
  for (; exp_it != expected.value_end<llvm::APFloat>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ((*exp_it).convertToFloat(), (*act_it).convertToFloat());
  }
}

void ExpectExactIntMatch(const mlir::DenseElementsAttr& expected,
                         const mlir::DenseElementsAttr& actual) {
  ASSERT_EQ(expected.getType(), actual.getType());
  auto exp_it = expected.value_begin<llvm::APInt>();
  auto act_it = actual.value_begin<llvm::APInt>();
  for (; exp_it != expected.value_end<llvm::APInt>(); ++exp_it, ++act_it) {
    EXPECT_EQ(*exp_it, *act_it);
  }
}

TEST(NarrowIndexWidthTest, EliminatesRoundTripConvert) {
  mlir::MLIRContext context;
  InitContext(context);
  const char* kIr = R"mlir(
func.func @main(%x: tensor<8xi32>) -> tensor<8xi32> {
  %wide = stablehlo.convert %x : (tensor<8xi32>) -> tensor<8xi64>
  %back = stablehlo.convert %wide : (tensor<8xi64>) -> tensor<8xi32>
  return %back : tensor<8xi32>
}
)mlir";
  auto mod = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*mod, &context)));
  EXPECT_THAT(PrintModule(*mod), Not(HasSubstr("stablehlo.convert")));
}

TEST(NarrowIndexWidthTest, MoeRoutingChainParityAndIR) {
  mlir::MLIRContext context;
  InitContext(context);
  // Models topk_idx(i32) -> i64 -> flatten/reshape -> argsort(keys, iota)
  // -> gather + scatter_add, preserving i64 ABI if escaping.
  const char* kIr = R"mlir(
func.func @main(
    %tokens: tensor<4x2xf32>,
    %expert_ids_i32: tensor<2x2xi32>,
    %base_out: tensor<4x2xf32>) -> (tensor<4x2xf32>, tensor<4xi64>) {
  %expert_ids = stablehlo.convert %expert_ids_i32
      : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %flat_ids = stablehlo.reshape %expert_ids
      : (tensor<2x2xi64>) -> tensor<4xi64>
  %iota = stablehlo.iota dim = 0 : tensor<4xi64>
  %sorted:2 = "stablehlo.sort"(%flat_ids, %iota) ({
  ^bb0(%a0: tensor<i64>, %a1: tensor<i64>, %b0: tensor<i64>, %b1: tensor<i64>):
    %cmp = stablehlo.compare LT, %a0, %a1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true}
      : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
  %gather_idx = stablehlo.reshape %sorted#1 : (tensor<4xi64>) -> tensor<4x1xi64>
  %gathered = "stablehlo.gather"(%tokens, %gather_idx) <{
      dimension_numbers = #stablehlo.gather<
          offset_dims = [1],
          collapsed_slice_dims = [0],
          start_index_map = [0],
          index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 2>
  }> : (tensor<4x2xf32>, tensor<4x1xi64>) -> tensor<4x2xf32>
  %scattered = "stablehlo.scatter"(%base_out, %gather_idx, %gathered) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) {
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
          update_window_dims = [1],
          inserted_window_dims = [0],
          scatter_dims_to_operand_dims = [0],
          index_vector_dim = 1>,
      unique_indices = false
  } : (tensor<4x2xf32>, tensor<4x1xi64>, tensor<4x2xf32>) -> tensor<4x2xf32>
  return %scattered, %sorted#0 : tensor<4x2xf32>, tensor<4xi64>
}
)mlir";
  auto orig_mod = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  auto opt_mod = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  auto tokens = MakeF32Tensor(context, {4, 2},
                              {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  auto expert_ids = MakeI32Tensor(context, {2, 2}, {2, 0, 2, 1});
  auto base_out = MakeF32Tensor(
      context, {4, 2}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});

  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(
      *orig_mod, {tokens, expert_ids, base_out}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  // Verify sort, iota, gather, and scatter all run natively in i32.
  opt_mod->walk([](mlir::stablehlo::IotaOp op) {
    auto t = mlir::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_TRUE(t.getElementType().isSignlessInteger(32));
  });
  opt_mod->walk([](mlir::stablehlo::SortOp op) {
    for (mlir::Value in : op.getInputs()) {
      auto t = mlir::cast<mlir::RankedTensorType>(in.getType());
      EXPECT_TRUE(t.getElementType().isSignlessInteger(32));
    }
  });
  opt_mod->walk([](mlir::stablehlo::GatherOp op) {
    auto t = mlir::cast<mlir::RankedTensorType>(op.getStartIndices().getType());
    EXPECT_TRUE(t.getElementType().isSignlessInteger(32));
  });
  opt_mod->walk([](mlir::stablehlo::ScatterOp op) {
    auto t =
        mlir::cast<mlir::RankedTensorType>(op.getScatterIndices().getType());
    EXPECT_TRUE(t.getElementType().isSignlessInteger(32));
  });

  auto actual = mlir::stablehlo::evalModule(
      *opt_mod, {tokens, expert_ids, base_out}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectExactF32Match((*expected)[0], (*actual)[0]);
  ExpectExactIntMatch((*expected)[1], (*actual)[1]);
}

TEST(NarrowIndexWidthTest, NegativeAndOutOfBoundsGatherClampingMatches) {
  mlir::MLIRContext context;
  InitContext(context);
  const char* kIr = R"mlir(
func.func @main(%table: tensor<5xf32>, %idx_i32: tensor<4x1xi32>)
    -> tensor<4xf32> {
  %idx_i64 = stablehlo.convert %idx_i32 : (tensor<4x1xi32>) -> tensor<4x1xi64>
  %out = "stablehlo.gather"(%table, %idx_i64) <{
      dimension_numbers = #stablehlo.gather<
          collapsed_slice_dims = [0],
          start_index_map = [0],
          index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1>
  }> : (tensor<5xf32>, tensor<4x1xi64>) -> tensor<4xf32>
  return %out : tensor<4xf32>
}
)mlir";
  auto orig_mod = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  auto opt_mod = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  auto table = MakeF32Tensor(context, {5}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f});
  auto idx = MakeI32Tensor(context, {4, 1},
                           {std::numeric_limits<int32_t>::min(), -3, 2,
                            std::numeric_limits<int32_t>::max()});

  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {table, idx}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(mlir::succeeded(RunPass(*opt_mod, &context)));
  opt_mod->walk([](mlir::stablehlo::GatherOp op) {
    auto t = mlir::cast<mlir::RankedTensorType>(op.getStartIndices().getType());
    EXPECT_TRUE(t.getElementType().isSignlessInteger(32));
  });

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {table, idx}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectExactF32Match((*expected)[0], (*actual)[0]);
}

TEST(NarrowIndexWidthTest, ViewOpsAndCompareAgainstInRangeConstant) {
  mlir::MLIRContext context;
  InitContext(context);
  const char* kIr = R"mlir(
func.func @main(%a: tensor<2x2xi32>, %pred: tensor<2x4xi1>) -> tensor<2x4xi1> {
  %wide = stablehlo.convert %a : (tensor<2x2xi32>) -> tensor<2x2xi64>
  %tr = stablehlo.transpose %wide, dims = [1, 0]
      : (tensor<2x2xi64>) -> tensor<2x2xi64>
  %rev = stablehlo.reverse %tr, dims = [0]
      : (tensor<2x2xi64>) -> tensor<2x2xi64>
  %cst_pad = stablehlo.constant dense<-1> : tensor<i64>
  %pad = stablehlo.pad %rev, %cst_pad, low = [0, 0], high = [0, 2],
      interior = [0, 0] : (tensor<2x2xi64>, tensor<i64>) -> tensor<2x4xi64>
  %cst_zero = stablehlo.constant dense<0> : tensor<2x4xi64>
  %sel = stablehlo.select %pred, %pad, %cst_zero
      : (tensor<2x4xi1>, tensor<2x4xi64>, tensor<2x4xi64>) -> tensor<2x4xi64>
  %cst_lim = stablehlo.constant dense<2> : tensor<2x4xi64>
  %cmp = stablehlo.compare LT, %sel, %cst_lim
      : (tensor<2x4xi64>, tensor<2x4xi64>) -> tensor<2x4xi1>
  return %cmp : tensor<2x4xi1>
}
)mlir";
  auto orig_mod = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  auto opt_mod = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  auto a = MakeI32Tensor(context, {2, 2}, {0, 5, -2, 3});
  auto pred_type =
      mlir::RankedTensorType::get({2, 4}, mlir::IntegerType::get(&context, 1));
  auto pred = mlir::DenseElementsAttr::get(
      pred_type,
      llvm::ArrayRef<bool>{true, false, true, true, false, true, true, false});

  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {a, pred}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(mlir::succeeded(RunPass(*opt_mod, &context)));
  opt_mod->walk([](mlir::stablehlo::CompareOp cmp) {
    auto t = mlir::cast<mlir::RankedTensorType>(cmp.getLhs().getType());
    EXPECT_TRUE(t.getElementType().isSignlessInteger(32));
  });
  EXPECT_THAT(PrintModule(*opt_mod), Not(HasSubstr("stablehlo.convert")));

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {a, pred}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectExactIntMatch((*expected)[0], (*actual)[0]);
}

TEST(NarrowIndexWidthTest, DynamicSliceAndDynamicUpdateSliceParity) {
  mlir::MLIRContext context;
  InitContext(context);
  const char* kIr = R"mlir(
func.func @main(%x: tensor<4x4xf32>, %row: tensor<i32>, %upd: tensor<2x2xf32>)
    -> tensor<4x4xf32> {
  %row_i64 = stablehlo.convert %row : (tensor<i32>) -> tensor<i64>
  %c0 = stablehlo.constant dense<1> : tensor<i64>
  %sl = stablehlo.dynamic_slice %x, %row_i64, %c0, sizes = [2, 2]
      : (tensor<4x4xf32>, tensor<i64>, tensor<i64>) -> tensor<2x2xf32>
  %sum = stablehlo.add %sl, %upd : tensor<2x2xf32>
  %out = stablehlo.dynamic_update_slice %x, %sum, %row_i64, %c0
      : (tensor<4x4xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>)
      -> tensor<4x4xf32>
  return %out : tensor<4x4xf32>
}
)mlir";
  auto orig_mod = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  auto opt_mod = mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  std::vector<float> x_vals(16);
  for (int i = 0; i < 16; ++i) x_vals[i] = static_cast<float>(i);
  auto x = MakeF32Tensor(context, {4, 4}, x_vals);
  auto row = MakeI32Tensor(context, {}, {1});
  auto upd = MakeF32Tensor(context, {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});

  mlir::stablehlo::InterpreterConfiguration cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {x, row, upd}, cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_TRUE(mlir::succeeded(RunPass(*opt_mod, &context)));
  opt_mod->walk([](mlir::stablehlo::DynamicSliceOp op) {
    for (mlir::Value idx : op.getStartIndices()) {
      auto t = mlir::cast<mlir::RankedTensorType>(idx.getType());
      EXPECT_TRUE(t.getElementType().isSignlessInteger(32));
    }
  });
  opt_mod->walk([](mlir::stablehlo::DynamicUpdateSliceOp op) {
    for (mlir::Value idx : op.getStartIndices()) {
      auto t = mlir::cast<mlir::RankedTensorType>(idx.getType());
      EXPECT_TRUE(t.getElementType().isSignlessInteger(32));
    }
  });

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {x, row, upd}, cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ExpectExactF32Match((*expected)[0], (*actual)[0]);
}

TEST(NarrowIndexWidthTest, OverflowAndDynamicSafetyGuardrails) {
  mlir::MLIRContext context;
  InitContext(context);

  // 1. Out-of-range i64 constant (INT32_MAX + 1) must NOT be narrowed.
  const char* kOverflowCmp = R"mlir(
func.func @main(%x: tensor<4xi32>) -> tensor<4xi1> {
  %wide = stablehlo.convert %x : (tensor<4xi32>) -> tensor<4xi64>
  %big = stablehlo.constant dense<2147483648> : tensor<4xi64>
  %cmp = stablehlo.compare LT, %wide, %big
      : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}
)mlir";
  auto cmp_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kOverflowCmp, &context);
  ASSERT_TRUE(cmp_mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*cmp_mod, &context)));
  cmp_mod->walk([](mlir::stablehlo::CompareOp op) {
    auto t = mlir::cast<mlir::RankedTensorType>(op.getLhs().getType());
    EXPECT_TRUE(t.getElementType().isSignlessInteger(64));
  });

  // 2. Iota axis > INT32_MAX must remain i64.
  const char* kBigIota = R"mlir(
func.func @main() -> tensor<2147483648xi1> {
  %iota = stablehlo.iota dim = 0 : tensor<2147483648xi64>
  %z = stablehlo.constant dense<0> : tensor<2147483648xi64>
  %cmp = stablehlo.compare EQ, %iota, %z
      : (tensor<2147483648xi64>, tensor<2147483648xi64>) -> tensor<2147483648xi1>
  return %cmp : tensor<2147483648xi1>
}
)mlir";
  auto iota_mod = mlir::parseSourceString<mlir::ModuleOp>(kBigIota, &context);
  ASSERT_TRUE(iota_mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*iota_mod, &context)));
  iota_mod->walk([](mlir::stablehlo::IotaOp op) {
    auto t = mlir::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_TRUE(t.getElementType().isSignlessInteger(64));
  });

  // 3. Unbounded dynamic gather operand stays i64; bounded dynamic <= INT32_MAX
  // narrows safely to i32.
  const char* kUnboundedGather = R"mlir(
func.func @main(%table: tensor<?xf32>, %idx_i32: tensor<4x1xi32>)
    -> tensor<4xf32> {
  %idx = stablehlo.convert %idx_i32 : (tensor<4x1xi32>) -> tensor<4x1xi64>
  %out = "stablehlo.gather"(%table, %idx) <{
      dimension_numbers = #stablehlo.gather<
          collapsed_slice_dims = [0],
          start_index_map = [0],
          index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1>
  }> : (tensor<?xf32>, tensor<4x1xi64>) -> tensor<4xf32>
  return %out : tensor<4xf32>
}
)mlir";
  auto unb_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kUnboundedGather, &context);
  ASSERT_TRUE(unb_mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*unb_mod, &context)));
  unb_mod->walk([](mlir::stablehlo::GatherOp op) {
    auto t = mlir::cast<mlir::RankedTensorType>(op.getStartIndices().getType());
    EXPECT_TRUE(t.getElementType().isSignlessInteger(64));
  });

  const char* kBoundedGather = R"mlir(
func.func @main(
    %table: tensor<?xf32, #stablehlo.type_extensions<bounds = [64]>>,
    %idx_i32: tensor<4x1xi32>) -> tensor<4xf32> {
  %idx = stablehlo.convert %idx_i32 : (tensor<4x1xi32>) -> tensor<4x1xi64>
  %out = "stablehlo.gather"(%table, %idx) <{
      dimension_numbers = #stablehlo.gather<
          collapsed_slice_dims = [0],
          start_index_map = [0],
          index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1>
  }> : (tensor<?xf32, #stablehlo.type_extensions<bounds = [64]>>,
        tensor<4x1xi64>) -> tensor<4xf32>
  return %out : tensor<4xf32>
}
)mlir";
  auto bnd_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kBoundedGather, &context);
  ASSERT_TRUE(bnd_mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*bnd_mod, &context)));
  bnd_mod->walk([](mlir::stablehlo::GatherOp op) {
    auto t = mlir::cast<mlir::RankedTensorType>(op.getStartIndices().getType());
    EXPECT_TRUE(t.getElementType().isSignlessInteger(32));
  });

  // 4. Raw i64 function input (not originating from convert i32->i64) stays
  // i64.
  const char* kRawArg = R"mlir(
func.func @main(%table: tensor<8xf32>, %idx: tensor<4x1xi64>) -> tensor<4xf32> {
  %out = "stablehlo.gather"(%table, %idx) <{
      dimension_numbers = #stablehlo.gather<
          collapsed_slice_dims = [0],
          start_index_map = [0],
          index_vector_dim = 1>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1>
  }> : (tensor<8xf32>, tensor<4x1xi64>) -> tensor<4xf32>
  return %out : tensor<4xf32>
}
)mlir";
  auto raw_mod = mlir::parseSourceString<mlir::ModuleOp>(kRawArg, &context);
  ASSERT_TRUE(raw_mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*raw_mod, &context)));
  raw_mod->walk([](mlir::stablehlo::GatherOp op) {
    auto t = mlir::cast<mlir::RankedTensorType>(op.getStartIndices().getType());
    EXPECT_TRUE(t.getElementType().isSignlessInteger(64));
  });

  // 5. Unsigned comparison is NOT narrowed.
  const char* kUnsignedCmp = R"mlir(
func.func @main(%x: tensor<4xi32>, %y: tensor<4xi32>) -> tensor<4xi1> {
  %wx = stablehlo.convert %x : (tensor<4xi32>) -> tensor<4xi64>
  %wy = stablehlo.convert %y : (tensor<4xi32>) -> tensor<4xi64>
  %cmp = stablehlo.compare LT, %wx, %wy, UNSIGNED
      : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}
)mlir";
  auto ucmp_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kUnsignedCmp, &context);
  ASSERT_TRUE(ucmp_mod);
  ASSERT_TRUE(mlir::succeeded(RunPass(*ucmp_mod, &context)));
  ucmp_mod->walk([](mlir::stablehlo::CompareOp op) {
    auto t = mlir::cast<mlir::RankedTensorType>(op.getLhs().getType());
    EXPECT_TRUE(t.getElementType().isSignlessInteger(64));
  });
}

}  // namespace
}  // namespace torch_tpu
