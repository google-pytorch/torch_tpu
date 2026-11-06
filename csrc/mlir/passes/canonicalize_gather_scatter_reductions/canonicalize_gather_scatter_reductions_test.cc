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

#include "csrc/mlir/passes/canonicalize_gather_scatter_reductions/canonicalize_gather_scatter_reductions.h"  // NOLINT(whitespace/line_length)

#include <cstddef>
#include <cstdint>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"  // NOLINT(whitespace/line_length)
#include "mlir/Dialect/Func/IR/FuncOps.h"      // NOLINT(whitespace/line_length)
#include "mlir/IR/BuiltinAttributes.h"         // NOLINT(whitespace/line_length)
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"  // NOLINT(whitespace/line_length)
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

mlir::DenseElementsAttr MakeBF16Tensor(mlir::MLIRContext& ctx,
                                       llvm::ArrayRef<int64_t> shape) {
  auto type = mlir::RankedTensorType::get(shape, mlir::BFloat16Type::get(&ctx));
  const int64_t n = type.getNumElements();
  llvm::SmallVector<llvm::APFloat> vals;
  vals.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    float f = static_cast<float>((i * 11 + 3) % 32) * 0.25f - 3.5f;
    llvm::APFloat apf(f);
    bool loses_info = false;
    apf.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                &loses_info);
    vals.push_back(apf);
  }
  return mlir::DenseElementsAttr::get(type, vals);
}

mlir::DenseElementsAttr MakeF32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  llvm::SmallVector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = static_cast<float>((i * 7 + 1) % 19) * 0.125f - 1.0f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

mlir::DenseElementsAttr MakeI64Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      llvm::ArrayRef<int64_t> vals) {
  auto type =
      mlir::RankedTensorType::get(shape, mlir::IntegerType::get(&ctx, 64));
  return mlir::DenseElementsAttr::get(type, vals);
}

mlir::DenseElementsAttr MakeI8Tensor(mlir::MLIRContext& ctx,
                                     llvm::ArrayRef<int64_t> shape,
                                     llvm::ArrayRef<int8_t> vals) {
  auto type =
      mlir::RankedTensorType::get(shape, mlir::IntegerType::get(&ctx, 8));
  return mlir::DenseElementsAttr::get(type, vals);
}

void CheckNumerics(mlir::ModuleOp before_mod, mlir::ModuleOp after_mod,
                   llvm::ArrayRef<mlir::DenseElementsAttr> inputs) {
  mlir::stablehlo::InterpreterConfiguration ref_cfg;
  auto expected = mlir::stablehlo::evalModule(before_mod, inputs, ref_cfg);
  auto actual = mlir::stablehlo::evalModule(after_mod, inputs, ref_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ(expected->size(), actual->size());
  for (size_t i = 0; i < expected->size(); ++i) {
    auto exp_vals = (*expected)[i].getValues<llvm::APFloat>();
    auto act_vals = (*actual)[i].getValues<llvm::APFloat>();
    auto exp_it = exp_vals.begin();
    auto act_it = act_vals.begin();
    for (; exp_it != exp_vals.end(); ++exp_it, ++act_it) {
      EXPECT_FLOAT_EQ((*act_it).convertToFloat(), (*exp_it).convertToFloat());
    }
  }
}

TEST(CanonicalizeGatherScatterReductionsTest,
     HoistsUnitGatherAcrossReshapeConvertAndLogSoftmaxSubtracts) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %mm: tensor<4x8xbf16>,
        %idx: tensor<4x1x1xi64>) -> tensor<4xf32> {
    %min_f32 = stablehlo.constant dense<0xFF7FFFFF> : tensor<f32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %r3 = stablehlo.reshape %mm : (tensor<4x8xbf16>) -> tensor<1x4x8xbf16>
    %c3 = stablehlo.convert %r3 : (tensor<1x4x8xbf16>) -> tensor<1x4x8xf32>
    %logits = stablehlo.reshape %c3 : (tensor<1x4x8xf32>) -> tensor<4x8xf32>
    %max = stablehlo.reduce(%logits init: %min_f32)
        across dimensions = [1]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %bcast_max = stablehlo.broadcast_in_dim %max, dims = [0]
        : (tensor<4xf32>) -> tensor<4x8xf32>
    %shifted = stablehlo.subtract %logits, %bcast_max : tensor<4x8xf32>
    %exp = stablehlo.exponential %shifted : tensor<4x8xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32)
        across dimensions = [1]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %lse = stablehlo.log %sum : tensor<4xf32>
    %bcast_lse = stablehlo.broadcast_in_dim %lse, dims = [0]
        : (tensor<4xf32>) -> tensor<4x8xf32>
    %log_probs = stablehlo.subtract %shifted, %bcast_lse : tensor<4x8xf32>
    %gathered = "stablehlo.gather"(%log_probs, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    %out = stablehlo.reshape %gathered : (tensor<4x1xf32>) -> tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  int gather_count = 0;
  bool gather_on_bf16 = false;
  after_mod->walk([&](mlir::stablehlo::GatherOp op) {
    ++gather_count;
    auto in_type =
        mlir::cast<mlir::RankedTensorType>(op.getOperand().getType());
    gather_on_bf16 = in_type.getElementType().isBF16();
  });
  EXPECT_EQ(gather_count, 1);
  EXPECT_TRUE(gather_on_bf16);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeBF16Tensor(context, {4, 8}),
      MakeI64Tensor(context, {4, 1, 1}, {2, 0, 7, 5}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest,
     HoistsUnitGatherWithConcatenatedIotaIndices) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %mm: tensor<4x8xbf16>,
        %idx_col: tensor<4x1x1xi64>) -> tensor<4xf32> {
    %min_f32 = stablehlo.constant dense<0xFF7FFFFF> : tensor<f32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %logits = stablehlo.convert %mm : (tensor<4x8xbf16>) -> tensor<4x8xf32>
    %max = stablehlo.reduce(%logits init: %min_f32)
        across dimensions = [1]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %bcast_max = stablehlo.broadcast_in_dim %max, dims = [0]
        : (tensor<4xf32>) -> tensor<4x8xf32>
    %shifted = stablehlo.subtract %logits, %bcast_max : tensor<4x8xf32>
    %exp = stablehlo.exponential %shifted : tensor<4x8xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32)
        across dimensions = [1]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %lse = stablehlo.log %sum : tensor<4xf32>
    %bcast_lse = stablehlo.broadcast_in_dim %lse, dims = [0]
        : (tensor<4xf32>) -> tensor<4x8xf32>
    %log_probs = stablehlo.subtract %shifted, %bcast_lse : tensor<4x8xf32>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %bcast_iota = stablehlo.broadcast_in_dim %iota, dims = [0]
        : (tensor<4xi64>) -> tensor<4x1xi64>
    %row_iota = stablehlo.reshape %bcast_iota
        : (tensor<4x1xi64>) -> tensor<4x1x1xi64>
    %concat_indices = stablehlo.concatenate %row_iota, %idx_col, dim = 2
        : (tensor<4x1x1xi64>, tensor<4x1x1xi64>) -> tensor<4x1x2xi64>
    %gathered = "stablehlo.gather"(%log_probs, %concat_indices) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [0, 1],
        operand_batching_dims = [],
        start_indices_batching_dims = [],
        start_index_map = [0, 1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x2xi64>) -> tensor<4x1xf32>
    %out = stablehlo.reshape %gathered : (tensor<4x1xf32>) -> tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  int gather_count = 0;
  bool gather_on_bf16 = false;
  after_mod->walk([&](mlir::stablehlo::GatherOp op) {
    ++gather_count;
    auto in_type =
        mlir::cast<mlir::RankedTensorType>(op.getOperand().getType());
    gather_on_bf16 = in_type.getElementType().isBF16();
  });
  EXPECT_EQ(gather_count, 1);
  EXPECT_TRUE(gather_on_bf16);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeBF16Tensor(context, {4, 8}),
      MakeI64Tensor(context, {4, 1, 1}, {2, 0, 7, 5}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest, SinksAcrossMul) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %tbl: tensor<4x8xf32>,
        %scale: tensor<4xf32>,
        %idx: tensor<4x1x1xi64>) -> tensor<4x1xf32> {
    %bcast = stablehlo.broadcast_in_dim %scale, dims = [0]
        : (tensor<4xf32>) -> tensor<4x8xf32>
    %mul = stablehlo.multiply %tbl, %bcast : tensor<4x8xf32>
    %out = "stablehlo.gather"(%mul, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    return %out : tensor<4x1xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  int gather_count = 0;
  after_mod->walk([&](mlir::stablehlo::GatherOp op) {
    ++gather_count;
    EXPECT_EQ(
        op.getOperand(),
        after_mod->lookupSymbol<mlir::func::FuncOp>("main").getArgument(0));
  });
  EXPECT_EQ(gather_count, 1);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {4, 8}),
      MakeF32Tensor(context, {4}),
      MakeI64Tensor(context, {4, 1, 1}, {1, 3, 5, 7}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest, SinksAcrossDivBothSides) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %tbl: tensor<4x8xf32>,
        %scale: tensor<4xf32>,
        %idx: tensor<4x1x1xi64>) -> (tensor<4x1xf32>, tensor<4x1xf32>) {
    %bcast = stablehlo.broadcast_in_dim %scale, dims = [0]
        : (tensor<4xf32>) -> tensor<4x8xf32>
    %div_table_by_bcast = stablehlo.divide %tbl, %bcast : tensor<4x8xf32>
    %div_bcast_by_table = stablehlo.divide %bcast, %tbl : tensor<4x8xf32>
    %out1 = "stablehlo.gather"(%div_table_by_bcast, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    %out2 = "stablehlo.gather"(%div_bcast_by_table, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    return %out1, %out2 : tensor<4x1xf32>, tensor<4x1xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  int gather_count = 0;
  after_mod->walk([&](mlir::stablehlo::GatherOp op) {
    ++gather_count;
    EXPECT_EQ(
        op.getOperand(),
        after_mod->lookupSymbol<mlir::func::FuncOp>("main").getArgument(0));
  });
  EXPECT_EQ(gather_count, 1);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {4, 8}),
      MakeF32Tensor(context, {4}),
      MakeI64Tensor(context, {4, 1, 1}, {0, 2, 4, 6}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest, SinksAcrossNegAndAbs) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %tbl: tensor<4x8xf32>,
        %idx: tensor<4x1x1xi64>) -> (tensor<4x1xf32>, tensor<4x1xf32>) {
    %neg = stablehlo.negate %tbl : tensor<4x8xf32>
    %abs = stablehlo.abs %tbl : tensor<4x8xf32>
    %out_neg = "stablehlo.gather"(%neg, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    %out_abs = "stablehlo.gather"(%abs, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    return %out_neg, %out_abs : tensor<4x1xf32>, tensor<4x1xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  after_mod->walk([&](mlir::stablehlo::GatherOp op) {
    EXPECT_EQ(
        op.getOperand(),
        after_mod->lookupSymbol<mlir::func::FuncOp>("main").getArgument(0));
  });

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {4, 8}),
      MakeI64Tensor(context, {4, 1, 1}, {3, 1, 6, 2}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest, SinksAcrossAddAndSubLhs) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %tbl: tensor<4x8xf32>,
        %offset: tensor<4xf32>,
        %idx: tensor<4x1x1xi64>) -> (tensor<4x1xf32>, tensor<4x1xf32>) {
    %bcast = stablehlo.broadcast_in_dim %offset, dims = [0]
        : (tensor<4xf32>) -> tensor<4x8xf32>
    %add = stablehlo.add %tbl, %bcast : tensor<4x8xf32>
    %sub_lhs = stablehlo.subtract %bcast, %tbl : tensor<4x8xf32>
    %out_add = "stablehlo.gather"(%add, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    %out_sub = "stablehlo.gather"(%sub_lhs, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    return %out_add, %out_sub : tensor<4x1xf32>, tensor<4x1xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  after_mod->walk([&](mlir::stablehlo::GatherOp op) {
    EXPECT_EQ(
        op.getOperand(),
        after_mod->lookupSymbol<mlir::func::FuncOp>("main").getArgument(0));
  });

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {4, 8}),
      MakeF32Tensor(context, {4}),
      MakeI64Tensor(context, {4, 1, 1}, {0, 3, 5, 2}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest, SinksAcrossNeedsBaseReshape) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %mm: tensor<1x4x8xbf16>,
        %idx: tensor<4x1x1xi64>) -> tensor<4x1xf32> {
    %r = stablehlo.reshape %mm : (tensor<1x4x8xbf16>) -> tensor<4x8xbf16>
    %c = stablehlo.convert %r : (tensor<4x8xbf16>) -> tensor<4x8xf32>
    %gathered = "stablehlo.gather"(%c, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    return %gathered : tensor<4x1xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  int gather_count = 0;
  after_mod->walk([&](mlir::stablehlo::GatherOp op) {
    ++gather_count;
    auto in_type =
        mlir::cast<mlir::RankedTensorType>(op.getOperand().getType());
    EXPECT_TRUE(in_type.getElementType().isBF16());
  });
  EXPECT_EQ(gather_count, 1);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeBF16Tensor(context, {1, 4, 8}),
      MakeI64Tensor(context, {4, 1, 1}, {2, 0, 7, 5}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest, MultiIndexK2Gather) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %mm: tensor<4x8xbf16>,
        %bias: tensor<4xf32>,
        %idx: tensor<4x2x1xi64>) -> tensor<4x2xf32> {
    %c = stablehlo.convert %mm : (tensor<4x8xbf16>) -> tensor<4x8xf32>
    %bcast = stablehlo.broadcast_in_dim %bias, dims = [0]
        : (tensor<4xf32>) -> tensor<4x8xf32>
    %sub = stablehlo.subtract %c, %bcast : tensor<4x8xf32>
    %gathered = "stablehlo.gather"(%sub, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x2x1xi64>) -> tensor<4x2xf32>
    return %gathered : tensor<4x2xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  int gather_count = 0;
  after_mod->walk([&](mlir::stablehlo::GatherOp op) {
    ++gather_count;
    auto in_type =
        mlir::cast<mlir::RankedTensorType>(op.getOperand().getType());
    EXPECT_TRUE(in_type.getElementType().isBF16());
    EXPECT_EQ(op.getType(), mlir::RankedTensorType::get(
                                {4, 2}, mlir::BFloat16Type::get(&context)));
  });
  EXPECT_EQ(gather_count, 1);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeBF16Tensor(context, {4, 8}),
      MakeF32Tensor(context, {4}),
      MakeI64Tensor(context, {4, 2, 1}, {2, 3, 0, 1, 7, 6, 5, 4}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest, SinksAcrossSplatConstant) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %tbl: tensor<4x8xf32>,
        %idx: tensor<4x1x1xi64>) -> tensor<4x1xf32> {
    %half = stablehlo.constant dense<5.000000e-01> : tensor<4x8xf32>
    %mul = stablehlo.multiply %tbl, %half : tensor<4x8xf32>
    %out = "stablehlo.gather"(%mul, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    return %out : tensor<4x1xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  int gather_count = 0;
  after_mod->walk([&](mlir::stablehlo::GatherOp op) {
    ++gather_count;
    EXPECT_EQ(
        op.getOperand(),
        after_mod->lookupSymbol<mlir::func::FuncOp>("main").getArgument(0));
  });
  EXPECT_EQ(gather_count, 1);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {4, 8}),
      MakeI64Tensor(context, {4, 1, 1}, {1, 4, 2, 6}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest,
     SinksAcrossBothOperandsFromTable) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %tbl1: tensor<4x8xf32>,
        %tbl2: tensor<4x8xf32>,
        %idx: tensor<4x1x1xi64>) -> tensor<4x1xf32> {
    %sum = stablehlo.add %tbl1, %tbl2 : tensor<4x8xf32>
    %out = "stablehlo.gather"(%sum, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    return %out : tensor<4x1xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  int gather_count = 0;
  after_mod->walk([&](mlir::stablehlo::GatherOp) { ++gather_count; });
  EXPECT_EQ(gather_count, 2);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {4, 8}),
      MakeF32Tensor(context, {4, 8}),
      MakeI64Tensor(context, {4, 1, 1}, {0, 1, 2, 3}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest, PreservesMultiUseLogProbs) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %tbl: tensor<4x8xf32>,
        %offset: tensor<4xf32>,
        %idx: tensor<4x1x1xi64>) -> (tensor<4x1xf32>, tensor<4x8xf32>) {
    %bcast = stablehlo.broadcast_in_dim %offset, dims = [0]
        : (tensor<4xf32>) -> tensor<4x8xf32>
    %log_probs = stablehlo.subtract %tbl, %bcast : tensor<4x8xf32>
    %gathered = "stablehlo.gather"(%log_probs, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    return %gathered, %log_probs : tensor<4x1xf32>, tensor<4x8xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  int gather_count = 0;
  after_mod->walk([&](mlir::stablehlo::GatherOp op) {
    ++gather_count;
    EXPECT_EQ(
        op.getOperand(),
        after_mod->lookupSymbol<mlir::func::FuncOp>("main").getArgument(0));
  });
  EXPECT_EQ(gather_count, 1);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {4, 8}),
      MakeF32Tensor(context, {4}),
      MakeI64Tensor(context, {4, 1, 1}, {2, 1, 4, 3}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest,
     FoldsAdditiveReduceOverZeroBaseUnitScatter) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %idx: tensor<4x1x1xi64>,
        %updates: tensor<4x1xf32>) -> tensor<4xf32> {
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %base = stablehlo.broadcast_in_dim %zero_f32, dims = []
        : (tensor<f32>) -> tensor<4x8xf32>
    %scattered = "stablehlo.scatter"(%base, %idx, %updates) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [],
        inserted_window_dims = [1],
        input_batching_dims = [0],
        scatter_indices_batching_dims = [0],
        scatter_dims_to_operand_dims = [1],
        index_vector_dim = 2>,
      unique_indices = false
    }> ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    })
        : (tensor<4x8xf32>, tensor<4x1x1xi64>,
           tensor<4x1xf32>) -> tensor<4x8xf32>
    %c_bf16 = stablehlo.convert %scattered
        : (tensor<4x8xf32>) -> tensor<4x8xbf16>
    %c_f32 = stablehlo.convert %c_bf16 : (tensor<4x8xbf16>) -> tensor<4x8xf32>
    %reduced = stablehlo.reduce(%c_f32 init: %zero_f32)
        across dimensions = [1]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    return %reduced : tensor<4xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  int reduce_count = 0;
  int scatter_count = 0;
  after_mod->walk([&](mlir::stablehlo::ReduceOp) { ++reduce_count; });
  after_mod->walk([&](mlir::stablehlo::ScatterOp) { ++scatter_count; });
  EXPECT_EQ(reduce_count, 0);
  EXPECT_EQ(scatter_count, 0);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeI64Tensor(context, {4, 1, 1}, {3, -1, 7, 9}),
      MakeF32Tensor(context, {4, 1}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest,
     FoldsScatterReduceWithConcatenatedIotaIndices) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %idx_col: tensor<4x1x1xi64>,
        %updates: tensor<4x1xf32>) -> tensor<4xf32> {
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %base = stablehlo.broadcast_in_dim %zero_f32, dims = []
        : (tensor<f32>) -> tensor<4x8xf32>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %bcast_iota = stablehlo.broadcast_in_dim %iota, dims = [0]
        : (tensor<4xi64>) -> tensor<4x1xi64>
    %row_iota = stablehlo.reshape %bcast_iota
        : (tensor<4x1xi64>) -> tensor<4x1x1xi64>
    %concat_indices = stablehlo.concatenate %row_iota, %idx_col, dim = 2
        : (tensor<4x1x1xi64>, tensor<4x1x1xi64>) -> tensor<4x1x2xi64>
    %scattered = "stablehlo.scatter"(%base, %concat_indices, %updates) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [],
        inserted_window_dims = [0, 1],
        input_batching_dims = [],
        scatter_indices_batching_dims = [],
        scatter_dims_to_operand_dims = [0, 1],
        index_vector_dim = 2>,
      unique_indices = false
    }> ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    })
        : (tensor<4x8xf32>, tensor<4x1x2xi64>,
           tensor<4x1xf32>) -> tensor<4x8xf32>
    %reduced = stablehlo.reduce(%scattered init: %zero_f32)
        across dimensions = [1]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    return %reduced : tensor<4xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  int reduce_count = 0;
  int scatter_count = 0;
  after_mod->walk([&](mlir::stablehlo::ReduceOp) { ++reduce_count; });
  after_mod->walk([&](mlir::stablehlo::ScatterOp) { ++scatter_count; });
  EXPECT_EQ(reduce_count, 0);
  EXPECT_EQ(scatter_count, 0);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeI64Tensor(context, {4, 1, 1}, {3, -1, 7, 9}),
      MakeF32Tensor(context, {4, 1}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest,
     FoldsScatterReduceWithNarrowIntegerIndexOverflow) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %idx: tensor<4x1x1xi8>,
        %updates: tensor<4x1xf32>) -> tensor<4xf32> {
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %base = stablehlo.broadcast_in_dim %zero_f32, dims = []
        : (tensor<f32>) -> tensor<4x128xf32>
    %scattered = "stablehlo.scatter"(%base, %idx, %updates) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [],
        inserted_window_dims = [1],
        input_batching_dims = [0],
        scatter_indices_batching_dims = [0],
        scatter_dims_to_operand_dims = [1],
        index_vector_dim = 2>,
      unique_indices = false
    }> ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    })
        : (tensor<4x128xf32>, tensor<4x1x1xi8>,
           tensor<4x1xf32>) -> tensor<4x128xf32>
    %reduced = stablehlo.reduce(%scattered init: %zero_f32)
        across dimensions = [1]
        : (tensor<4x128xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    return %reduced : tensor<4xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  int reduce_count = 0;
  int scatter_count = 0;
  after_mod->walk([&](mlir::stablehlo::ReduceOp) { ++reduce_count; });
  after_mod->walk([&](mlir::stablehlo::ScatterOp) { ++scatter_count; });
  EXPECT_EQ(reduce_count, 0);
  EXPECT_EQ(scatter_count, 0);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeI8Tensor(context, {4, 1, 1}, {10, -1, 127, 0}),
      MakeF32Tensor(context, {4, 1}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

// ----------------- Negative Tests (Gates Kept) -----------------

void ExpectPassLeavesReduceAndScatter(mlir::MLIRContext& context,
                                      llvm::StringRef ir) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(ir, &context);
  ASSERT_TRUE(module);
  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
              absl_testing::IsOk());
  int reduce_count = 0;
  int scatter_count = 0;
  module->walk([&](mlir::stablehlo::ReduceOp) { ++reduce_count; });
  module->walk([&](mlir::stablehlo::ScatterOp) { ++scatter_count; });
  EXPECT_EQ(reduce_count, 1);
  EXPECT_EQ(scatter_count, 1);
}

TEST(CanonicalizeGatherScatterReductionsTest,
     ScatterReductionNegativeGatesRetained) {
  mlir::MLIRContext context;
  InitContext(context);

  auto check_no_fold = [&](const char* name, llvm::StringRef ir) {
    SCOPED_TRACE(name);
    ExpectPassLeavesReduceAndScatter(context, ir);
  };

  // 1. Scatter base is non-zero
  check_no_fold("ScatterBaseIsNonZero", R"mlir(
module {
  func.func @main(
      %idx: tensor<4x1x1xi64>,
      %updates: tensor<4x1xf32>) -> tensor<4xf32> {
    %one_f32 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %base = stablehlo.broadcast_in_dim %one_f32, dims = []
        : (tensor<f32>) -> tensor<4x8xf32>
    %scattered = "stablehlo.scatter"(%base, %idx, %updates) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [], inserted_window_dims = [1],
        input_batching_dims = [0], scatter_indices_batching_dims = [0],
        scatter_dims_to_operand_dims = [1], index_vector_dim = 2>,
      unique_indices = false
    }> ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) : (tensor<4x8xf32>, tensor<4x1x1xi64>,
          tensor<4x1xf32>) -> tensor<4x8xf32>
    %reduced = stablehlo.reduce(%scattered init: %zero_f32)
        across dimensions = [1]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    return %reduced : tensor<4xf32>
  }
}
)mlir");

  // 2. Updates dimension != 1
  check_no_fold("UpdatesDimNotOne", R"mlir(
module {
  func.func @main(
      %idx: tensor<4x2x1xi64>,
      %updates: tensor<4x2xf32>) -> tensor<4xf32> {
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %base = stablehlo.broadcast_in_dim %zero_f32, dims = []
        : (tensor<f32>) -> tensor<4x8xf32>
    %scattered = "stablehlo.scatter"(%base, %idx, %updates) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [], inserted_window_dims = [1],
        input_batching_dims = [0], scatter_indices_batching_dims = [0],
        scatter_dims_to_operand_dims = [1], index_vector_dim = 2>,
      unique_indices = false
    }> ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) : (tensor<4x8xf32>, tensor<4x2x1xi64>,
          tensor<4x2xf32>) -> tensor<4x8xf32>
    %reduced = stablehlo.reduce(%scattered init: %zero_f32)
        across dimensions = [1]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    return %reduced : tensor<4xf32>
  }
}
)mlir");

  // 3. Scatter region is mul (only add supported)
  check_no_fold("ScatterRegionIsMul", R"mlir(
module {
  func.func @main(
      %idx: tensor<4x1x1xi64>,
      %updates: tensor<4x1xf32>) -> tensor<4xf32> {
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %base = stablehlo.broadcast_in_dim %zero_f32, dims = []
        : (tensor<f32>) -> tensor<4x8xf32>
    %scattered = "stablehlo.scatter"(%base, %idx, %updates) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [], inserted_window_dims = [1],
        input_batching_dims = [0], scatter_indices_batching_dims = [0],
        scatter_dims_to_operand_dims = [1], index_vector_dim = 2>,
      unique_indices = false
    }> ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %mul = stablehlo.multiply %lhs, %rhs : tensor<f32>
      stablehlo.return %mul : tensor<f32>
    }) : (tensor<4x8xf32>, tensor<4x1x1xi64>,
          tensor<4x1xf32>) -> tensor<4x8xf32>
    %reduced = stablehlo.reduce(%scattered init: %zero_f32)
        across dimensions = [1]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    return %reduced : tensor<4xf32>
  }
}
)mlir");

  // 4. Reduce dimension is not the scatter dimension
  check_no_fold("ReduceDimNotScatterDim", R"mlir(
module {
  func.func @main(
      %idx: tensor<4x1x1xi64>,
      %updates: tensor<4x1xf32>) -> tensor<8xf32> {
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %base = stablehlo.broadcast_in_dim %zero_f32, dims = []
        : (tensor<f32>) -> tensor<4x8xf32>
    %scattered = "stablehlo.scatter"(%base, %idx, %updates) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [], inserted_window_dims = [1],
        input_batching_dims = [0], scatter_indices_batching_dims = [0],
        scatter_dims_to_operand_dims = [1], index_vector_dim = 2>,
      unique_indices = false
    }> ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) : (tensor<4x8xf32>, tensor<4x1x1xi64>,
          tensor<4x1xf32>) -> tensor<4x8xf32>
    %reduced = stablehlo.reduce(%scattered init: %zero_f32)
        across dimensions = [0]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    return %reduced : tensor<8xf32>
  }
}
)mlir");

  // 5. Batch indices are non-iota
  check_no_fold("BatchIndicesNonIota", R"mlir(
module {
  func.func @main(
      %idx_col: tensor<4x1x1xi64>,
      %updates: tensor<4x1xf32>) -> tensor<4xf32> {
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %base = stablehlo.broadcast_in_dim %zero_f32, dims = []
        : (tensor<f32>) -> tensor<4x8xf32>
    %const_row = stablehlo.constant dense<0> : tensor<4x1x1xi64>
    %concat_indices = stablehlo.concatenate %const_row, %idx_col, dim = 2
        : (tensor<4x1x1xi64>, tensor<4x1x1xi64>) -> tensor<4x1x2xi64>
    %scattered = "stablehlo.scatter"(%base, %concat_indices, %updates) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [], inserted_window_dims = [0, 1],
        input_batching_dims = [], scatter_indices_batching_dims = [],
        scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>,
      unique_indices = false
    }> ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) : (tensor<4x8xf32>, tensor<4x1x2xi64>,
          tensor<4x1xf32>) -> tensor<4x8xf32>
    %reduced = stablehlo.reduce(%scattered init: %zero_f32)
        across dimensions = [1]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    return %reduced : tensor<4xf32>
  }
}
)mlir");

  // 6. Non-value-preserving convert on iota indices
  check_no_fold("NonValuePreservingConvert", R"mlir(
module {
  func.func @main(
      %idx_col: tensor<256x1x1xi8>,
      %updates: tensor<256x1xf32>) -> tensor<256xf32> {
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %base = stablehlo.broadcast_in_dim %zero_f32, dims = []
        : (tensor<f32>) -> tensor<256x8xf32>
    %iota = stablehlo.iota dim = 0 : tensor<256xi64>
    %iota_i8 = stablehlo.convert %iota
        : (tensor<256xi64>) -> tensor<256xi8>
    %bcast_iota = stablehlo.broadcast_in_dim %iota_i8, dims = [0]
        : (tensor<256xi8>) -> tensor<256x1xi8>
    %row_iota = stablehlo.reshape %bcast_iota
        : (tensor<256x1xi8>) -> tensor<256x1x1xi8>
    %concat_indices = stablehlo.concatenate %row_iota, %idx_col, dim = 2
        : (tensor<256x1x1xi8>, tensor<256x1x1xi8>) -> tensor<256x1x2xi8>
    %scattered = "stablehlo.scatter"(%base, %concat_indices, %updates) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [], inserted_window_dims = [0, 1],
        input_batching_dims = [], scatter_indices_batching_dims = [],
        scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>,
      unique_indices = false
    }> ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) : (tensor<256x8xf32>, tensor<256x1x2xi8>,
          tensor<256x1xf32>) -> tensor<256x8xf32>
    %reduced = stablehlo.reduce(%scattered init: %zero_f32)
        across dimensions = [1]
        : (tensor<256x8xf32>, tensor<f32>) -> tensor<256xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    return %reduced : tensor<256xf32>
  }
}
)mlir");

  // 7. Concat input dim size != 1
  check_no_fold("ConcatInputDimSizeNotOne", R"mlir(
module {
  func.func @main(
      %idx: tensor<4x1x2xi64>,
      %updates: tensor<4x1xf32>) -> tensor<4xf32> {
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %zero_i64 = stablehlo.constant dense<0> : tensor<i64>
    %base = stablehlo.broadcast_in_dim %zero_f32, dims = []
        : (tensor<f32>) -> tensor<4x8xf32>
    %empty_iota = stablehlo.broadcast_in_dim %zero_i64, dims = []
        : (tensor<i64>) -> tensor<4x1x0xi64>
    %concat_indices = stablehlo.concatenate %empty_iota, %idx, dim = 2
        : (tensor<4x1x0xi64>, tensor<4x1x2xi64>) -> tensor<4x1x2xi64>
    %scattered = "stablehlo.scatter"(%base, %concat_indices, %updates) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [], inserted_window_dims = [0, 1],
        input_batching_dims = [], scatter_indices_batching_dims = [],
        scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>,
      unique_indices = false
    }> ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) : (tensor<4x8xf32>, tensor<4x1x2xi64>,
          tensor<4x1xf32>) -> tensor<4x8xf32>
    %reduced = stablehlo.reduce(%scattered init: %zero_f32)
        across dimensions = [1]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    return %reduced : tensor<4xf32>
  }
}
)mlir");
}

TEST(CanonicalizeGatherScatterReductionsTest, GatherNegativeGatesRetained) {
  mlir::MLIRContext context;
  InitContext(context);

  auto check_no_hoist = [&](const char* name, llvm::StringRef ir) {
    SCOPED_TRACE(name);
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(ir, &context);
    ASSERT_TRUE(module);
    ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context),
                absl_testing::IsOk());
    int gather_count = 0;
    module->walk([&](mlir::stablehlo::GatherOp op) {
      ++gather_count;
      EXPECT_NE(
          op.getOperand(),
          module->lookupSymbol<mlir::func::FuncOp>("main").getArgument(0));
    });
    EXPECT_EQ(gather_count, 1);
  };

  // 1. Variant broadcast (broadcast along non-gather dim)
  check_no_hoist("VariantBroadcast", R"mlir(
module {
  func.func @main(
        %arg0: tensor<4x8xf32>,
        %idx: tensor<4x1x1xi64>,
        %col_vec: tensor<8xf32>) -> tensor<4x1xf32> {
    %bcast_variant = stablehlo.broadcast_in_dim %col_vec, dims = [1]
        : (tensor<8xf32>) -> tensor<4x8xf32>
    %sub = stablehlo.subtract %arg0, %bcast_variant : tensor<4x8xf32>
    %gathered = "stablehlo.gather"(%sub, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [], collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0], start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false, slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    return %gathered : tensor<4x1xf32>
  }
}
)mlir");

  // 2. Non-trailing index_vector_dim
  check_no_hoist("NonTrailingIndexVectorDim", R"mlir(
module {
  func.func @main(
        %tbl: tensor<4x8xf32>,
        %bias: tensor<4xf32>,
        %idx: tensor<1x4x1xi64>) -> tensor<4x1xf32> {
    %bcast = stablehlo.broadcast_in_dim %bias, dims = [0]
        : (tensor<4xf32>) -> tensor<4x8xf32>
    %sub = stablehlo.subtract %tbl, %bcast : tensor<4x8xf32>
    %res = "stablehlo.gather"(%sub, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [], collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [1], start_index_map = [1],
        index_vector_dim = 0>,
      indices_are_sorted = false, slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<1x4x1xi64>) -> tensor<4x1xf32>
    return %res : tensor<4x1xf32>
  }
}
)mlir");
}

TEST(CanonicalizeGatherScatterReductionsTest,
     HoistsGatherWithKeepdimBroadcast) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %tbl: tensor<4x8xf32>,
        %lse: tensor<4x1xf32>,
        %idx: tensor<4x1x1xi64>) -> tensor<4x1xf32> {
    %bcast = stablehlo.broadcast_in_dim %lse, dims = [0, 1]
        : (tensor<4x1xf32>) -> tensor<4x8xf32>
    %sub = stablehlo.subtract %tbl, %bcast : tensor<4x8xf32>
    %res = "stablehlo.gather"(%sub, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    return %res : tensor<4x1xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*after_mod, &context),
              absl_testing::IsOk());

  int gather_count = 0;
  after_mod->walk([&](mlir::stablehlo::GatherOp op) {
    ++gather_count;
    EXPECT_EQ(
        op.getOperand(),
        after_mod->lookupSymbol<mlir::func::FuncOp>("main").getArgument(0));
  });
  EXPECT_EQ(gather_count, 1);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {4, 8}),
      MakeF32Tensor(context, {4, 1}),
      MakeI64Tensor(context, {4, 1, 1}, {1, 3, 5, 7}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest,
     DoesNotRewriteReshapeConvertWhenConvertHasMultipleUses) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
      %x: tensor<4x8xbf16>) -> (tensor<1x4x8xf32>, tensor<32xf32>) {
    %conv = stablehlo.convert %x : (tensor<4x8xbf16>) -> tensor<4x8xf32>
    %r1 = stablehlo.reshape %conv : (tensor<4x8xf32>) -> tensor<1x4x8xf32>
    %r2 = stablehlo.reshape %conv : (tensor<4x8xf32>) -> tensor<32xf32>
    return %r1, %r2 : tensor<1x4x8xf32>, tensor<32xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeGatherScatterReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  int convert_count = 0;
  module->walk([&](mlir::stablehlo::ConvertOp) { ++convert_count; });
  EXPECT_EQ(convert_count, 1);
}

TEST(CanonicalizeGatherScatterReductionsTest, SinksAcrossRepeatedOperands) {
  mlir::MLIRContext context;
  InitContext(context);

  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %tbl: tensor<4x8xf32>,
        %idx: tensor<4x1x1xi64>) -> tensor<4x1xf32> {
    %mul = stablehlo.multiply %tbl, %tbl : tensor<4x8xf32>
    %res = "stablehlo.gather"(%mul, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x8xf32>, tensor<4x1x1xi64>) -> tensor<4x1xf32>
    return %res : tensor<4x1xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> before_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> after_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(before_mod);
  ASSERT_TRUE(after_mod);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeGatherScatterReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*after_mod)));

  int gather_count = 0;
  after_mod->walk([&](mlir::stablehlo::GatherOp op) {
    ++gather_count;
    EXPECT_EQ(
        op.getOperand(),
        after_mod->lookupSymbol<mlir::func::FuncOp>("main").getArgument(0));
  });
  // Must dedupe: exactly one gather created, not two.
  EXPECT_EQ(gather_count, 1);

  llvm::SmallVector<mlir::DenseElementsAttr> inputs = {
      MakeF32Tensor(context, {4, 8}),
      MakeI64Tensor(context, {4, 1, 1}, {0, 3, 2, 6}),
  };
  CheckNumerics(*before_mod, *after_mod, inputs);
}

TEST(CanonicalizeGatherScatterReductionsTest,
     RejectsExpandingBucketTableGather) {
  mlir::MLIRContext context;
  InitContext(context);

  // Table is 4x32, gathered result is 4x256 (result_dim > table_dim).
  // Sinking elementwise exponential through the gather must be rejected.
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(
        %tbl: tensor<4x32xf32>,
        %idx: tensor<4x256x1xi64>) -> tensor<4x256xf32> {
    %exp = stablehlo.exponential %tbl : tensor<4x32xf32>
    %res = "stablehlo.gather"(%exp, %idx) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [0],
        start_index_map = [1],
        index_vector_dim = 2>,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x32xf32>, tensor<4x256x1xi64>) -> tensor<4x256xf32>
    return %res : tensor<4x256xf32>
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeGatherScatterReductionsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Sinking is rejected, so GatherOp still takes %exp as its operand.
  int gather_count = 0;
  module->walk([&gather_count](mlir::stablehlo::GatherOp op) {
    ++gather_count;
    EXPECT_TRUE(op.getOperand().getDefiningOp<mlir::stablehlo::ExpOp>());
  });
  EXPECT_EQ(gather_count, 1);

  int exp_count = 0;
  module->walk([&exp_count](mlir::stablehlo::ExpOp op) {
    ++exp_count;
    auto type = mlir::cast<mlir::RankedTensorType>(op.getType());
    EXPECT_EQ(type.getDimSize(1), 32);
  });
  EXPECT_EQ(exp_count, 1);
}

}  // namespace
}  // namespace torch_tpu
