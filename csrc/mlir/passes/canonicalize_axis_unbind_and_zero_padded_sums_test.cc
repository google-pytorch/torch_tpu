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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"

namespace torch_tpu {
namespace {

// Registers required dialects and initializes `context` for testing.
void InitContext(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

// Counts operations of type `OpTy` inside `module`.
template <typename OpTy>
int CountOps(mlir::ModuleOp module) {
  int count = 0;
  module.walk([&](OpTy) { ++count; });
  return count;
}

// Generates a deterministically populated floating-point test tensor.
mlir::DenseElementsAttr MakeF32Tensor(mlir::MLIRContext& ctx,
                                      llvm::ArrayRef<int64_t> shape,
                                      float base) {
  auto type = mlir::RankedTensorType::get(shape, mlir::Float32Type::get(&ctx));
  const int64_t n = type.getNumElements();
  std::vector<float> vals(n);
  for (int64_t i = 0; i < n; ++i) {
    vals[i] = base + static_cast<float>((i * 17 + 5) % 251) * 0.125f -
              static_cast<float>((i * 7) % 13) * 0.5f;
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(vals));
}

TEST(CanonicalizeAxisUnbindAndZeroPaddedSumsTest,
     ForwardPartitionedUnbindFlattensToMinorAxisSlices) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %norm: tensor<2x4x3x128xf32>,
    %embed: tensor<2x4x384xf32>)
    -> (tensor<2x4x128xf32>, tensor<2x4x128xf32>, tensor<2x4x128xf32>) {
  %scale = stablehlo.constant dense<0.5> : tensor<f32>
  %embed_4d = stablehlo.reshape %embed
      : (tensor<2x4x384xf32>) -> tensor<2x4x3x128xf32>
  %sum = stablehlo.add %norm, %embed_4d : tensor<2x4x3x128xf32>
  %bcast_scale = stablehlo.broadcast_in_dim %scale, dims = []
      : (tensor<f32>) -> tensor<2x4x3x128xf32>
  %y = stablehlo.multiply %sum, %bcast_scale : tensor<2x4x3x128xf32>
  %s0 = stablehlo.slice %y [0:2, 0:4, 0:1, 0:128]
      : (tensor<2x4x3x128xf32>) -> tensor<2x4x1x128xf32>
  %r0 = stablehlo.reshape %s0
      : (tensor<2x4x1x128xf32>) -> tensor<2x4x128xf32>
  %s1 = stablehlo.slice %y [0:2, 0:4, 1:2, 0:128]
      : (tensor<2x4x3x128xf32>) -> tensor<2x4x1x128xf32>
  %r1 = stablehlo.reshape %s1
      : (tensor<2x4x1x128xf32>) -> tensor<2x4x128xf32>
  %s2 = stablehlo.slice %y [0:2, 0:4, 2:3, 0:128]
      : (tensor<2x4x3x128xf32>) -> tensor<2x4x1x128xf32>
  %r2 = stablehlo.reshape %s2
      : (tensor<2x4x1x128xf32>) -> tensor<2x4x128xf32>
  return %r0, %r1, %r2
      : tensor<2x4x128xf32>, tensor<2x4x128xf32>, tensor<2x4x128xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr norm = MakeF32Tensor(context, {2, 4, 3, 128}, 1.0f);
  mlir::DenseElementsAttr embed = MakeF32Tensor(context, {2, 4, 384}, -0.5f);
  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_mod, {norm, embed}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  // 4 reshapes (1 on embed + 3 on slices) reduced to 1 reshape on norm.
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_mod), 3);

  auto actual =
      mlir::stablehlo::evalModule(*opt_mod, {norm, embed}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  for (int out_idx = 0; out_idx < 3; ++out_idx) {
    ASSERT_EQ((*actual)[out_idx].getNumElements(),
              (*expected)[out_idx].getNumElements());
    auto exp_it = (*expected)[out_idx].value_begin<float>();
    auto act_it = (*actual)[out_idx].value_begin<float>();
    for (; exp_it != (*expected)[out_idx].value_end<float>();
         ++exp_it, ++act_it) {
      EXPECT_FLOAT_EQ(*act_it, *exp_it);
    }
  }
}

TEST(CanonicalizeAxisUnbindAndZeroPaddedSumsTest,
     ForwardQkvTransposeUnbindEliminatesTransposeAndFlattensSlices) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(%qkv_2d: tensor<8x384xf32>)
    -> (tensor<2x4x2x64xf32>, tensor<2x4x2x64xf32>, tensor<2x4x2x64xf32>) {
  %qkv_4d = stablehlo.reshape %qkv_2d
      : (tensor<8x384xf32>) -> tensor<2x4x3x128xf32>
  %trans = stablehlo.transpose %qkv_4d, dims = [2, 0, 1, 3]
      : (tensor<2x4x3x128xf32>) -> tensor<3x2x4x128xf32>
  %s0 = stablehlo.slice %trans [0:1, 0:2, 0:4, 0:128]
      : (tensor<3x2x4x128xf32>) -> tensor<1x2x4x128xf32>
  %q = stablehlo.reshape %s0
      : (tensor<1x2x4x128xf32>) -> tensor<2x4x2x64xf32>
  %s1 = stablehlo.slice %trans [1:2, 0:2, 0:4, 0:128]
      : (tensor<3x2x4x128xf32>) -> tensor<1x2x4x128xf32>
  %k = stablehlo.reshape %s1
      : (tensor<1x2x4x128xf32>) -> tensor<2x4x2x64xf32>
  %s2 = stablehlo.slice %trans [2:3, 0:2, 0:4, 0:128]
      : (tensor<3x2x4x128xf32>) -> tensor<1x2x4x128xf32>
  %v = stablehlo.reshape %s2
      : (tensor<1x2x4x128xf32>) -> tensor<2x4x2x64xf32>
  return %q, %k, %v
      : tensor<2x4x2x64xf32>, tensor<2x4x2x64xf32>, tensor<2x4x2x64xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr qkv = MakeF32Tensor(context, {8, 384}, 0.25f);
  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(*orig_mod, {qkv}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::SliceOp>(*opt_mod), 3);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {qkv}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  for (int out_idx = 0; out_idx < 3; ++out_idx) {
    ASSERT_EQ((*actual)[out_idx].getNumElements(),
              (*expected)[out_idx].getNumElements());
    auto exp_it = (*expected)[out_idx].value_begin<float>();
    auto act_it = (*actual)[out_idx].value_begin<float>();
    for (; exp_it != (*expected)[out_idx].value_end<float>();
         ++exp_it, ++act_it) {
      EXPECT_FLOAT_EQ(*act_it, *exp_it);
    }
  }
}

TEST(CanonicalizeAxisUnbindAndZeroPaddedSumsTest,
     BackwardSelectPadSumFoldsToMinorAxisConcatAndSinksReshape) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %g0: tensor<2x4x128xf32>,
    %g1: tensor<2x4x128xf32>,
    %g2: tensor<2x4x128xf32>)
    -> (tensor<2x4x384xf32>, tensor<2x4x3x128xf32>) {
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %scale = stablehlo.constant dense<0.5> : tensor<f32>
  %u0 = stablehlo.reshape %g0 : (tensor<2x4x128xf32>) -> tensor<2x4x1x128xf32>
  %p0 = stablehlo.pad %u0, %zero,
      low = [0, 0, 0, 0], high = [0, 0, 2, 0], interior = [0, 0, 0, 0]
      : (tensor<2x4x1x128xf32>, tensor<f32>) -> tensor<2x4x3x128xf32>
  %u1 = stablehlo.reshape %g1 : (tensor<2x4x128xf32>) -> tensor<2x4x1x128xf32>
  %p1 = stablehlo.pad %u1, %zero,
      low = [0, 0, 1, 0], high = [0, 0, 1, 0], interior = [0, 0, 0, 0]
      : (tensor<2x4x1x128xf32>, tensor<f32>) -> tensor<2x4x3x128xf32>
  %u2 = stablehlo.reshape %g2 : (tensor<2x4x128xf32>) -> tensor<2x4x1x128xf32>
  %p2 = stablehlo.pad %u2, %zero,
      low = [0, 0, 2, 0], high = [0, 0, 0, 0], interior = [0, 0, 0, 0]
      : (tensor<2x4x1x128xf32>, tensor<f32>) -> tensor<2x4x3x128xf32>
  %a01 = stablehlo.add %p0, %p1 : tensor<2x4x3x128xf32>
  %a012 = stablehlo.add %a01, %p2 : tensor<2x4x3x128xf32>
  %bcast_scale = stablehlo.broadcast_in_dim %scale, dims = []
      : (tensor<f32>) -> tensor<2x4x3x128xf32>
  %scaled = stablehlo.multiply %a012, %bcast_scale : tensor<2x4x3x128xf32>
  %flat_grad = stablehlo.reshape %scaled
      : (tensor<2x4x3x128xf32>) -> tensor<2x4x384xf32>
  return %flat_grad, %scaled : tensor<2x4x384xf32>, tensor<2x4x3x128xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr g0 = MakeF32Tensor(context, {2, 4, 128}, 1.0f);
  mlir::DenseElementsAttr g1 = MakeF32Tensor(context, {2, 4, 128}, 2.0f);
  mlir::DenseElementsAttr g2 = MakeF32Tensor(context, {2, 4, 128}, 3.0f);
  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_mod, {g0, g1, g2}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::PadOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::AddOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_mod), 1);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*opt_mod), 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {g0, g1, g2}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  for (int out_idx = 0; out_idx < 2; ++out_idx) {
    ASSERT_EQ((*actual)[out_idx].getNumElements(),
              (*expected)[out_idx].getNumElements());
    auto exp_it = (*expected)[out_idx].value_begin<float>();
    auto act_it = (*actual)[out_idx].value_begin<float>();
    for (; exp_it != (*expected)[out_idx].value_end<float>();
         ++exp_it, ++act_it) {
      EXPECT_FLOAT_EQ(*act_it, *exp_it);
    }
  }
}

TEST(CanonicalizeAxisUnbindAndZeroPaddedSumsTest,
     BackwardQkvPadSumTransposeEliminatesPadsAddsAndTranspose) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %gq: tensor<2x4x2x64xf32>,
    %gk: tensor<2x4x2x64xf32>,
    %gv: tensor<2x4x2x64xf32>) -> tensor<8x384xf32> {
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %uq = stablehlo.reshape %gq : (tensor<2x4x2x64xf32>) -> tensor<1x2x4x128xf32>
  %pq = stablehlo.pad %uq, %zero,
      low = [0, 0, 0, 0], high = [2, 0, 0, 0], interior = [0, 0, 0, 0]
      : (tensor<1x2x4x128xf32>, tensor<f32>) -> tensor<3x2x4x128xf32>
  %uk = stablehlo.reshape %gk : (tensor<2x4x2x64xf32>) -> tensor<1x2x4x128xf32>
  %pk = stablehlo.pad %uk, %zero,
      low = [1, 0, 0, 0], high = [1, 0, 0, 0], interior = [0, 0, 0, 0]
      : (tensor<1x2x4x128xf32>, tensor<f32>) -> tensor<3x2x4x128xf32>
  %uv = stablehlo.reshape %gv : (tensor<2x4x2x64xf32>) -> tensor<1x2x4x128xf32>
  %pv = stablehlo.pad %uv, %zero,
      low = [2, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 0, 0]
      : (tensor<1x2x4x128xf32>, tensor<f32>) -> tensor<3x2x4x128xf32>
  %s0 = stablehlo.add %pv, %pk : tensor<3x2x4x128xf32>
  %s1 = stablehlo.add %s0, %pq : tensor<3x2x4x128xf32>
  %trans = stablehlo.transpose %s1, dims = [1, 2, 0, 3]
      : (tensor<3x2x4x128xf32>) -> tensor<2x4x3x128xf32>
  %out = stablehlo.reshape %trans
      : (tensor<2x4x3x128xf32>) -> tensor<8x384xf32>
  return %out : tensor<8x384xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr gq = MakeF32Tensor(context, {2, 4, 2, 64}, 0.5f);
  mlir::DenseElementsAttr gk = MakeF32Tensor(context, {2, 4, 2, 64}, 1.5f);
  mlir::DenseElementsAttr gv = MakeF32Tensor(context, {2, 4, 2, 64}, -1.0f);
  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected =
      mlir::stablehlo::evalModule(*orig_mod, {gq, gk, gv}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::PadOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::AddOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::TransposeOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_mod), 1);

  auto actual = mlir::stablehlo::evalModule(*opt_mod, {gq, gk, gv}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  ASSERT_EQ((*actual)[0].getNumElements(), (*expected)[0].getNumElements());
  auto exp_it = (*expected)[0].value_begin<float>();
  auto act_it = (*actual)[0].value_begin<float>();
  for (; exp_it != (*expected)[0].value_end<float>(); ++exp_it, ++act_it) {
    EXPECT_FLOAT_EQ(*act_it, *exp_it);
  }
}

TEST(CanonicalizeAxisUnbindAndZeroPaddedSumsTest,
     BackwardStridedSliceReshapedConcatAndDusConcatFoldCleanly) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %even_grad: tensor<4x6xf32>,
    %odd_grad: tensor<4x6xf32>,
    %patch_grad: tensor<2x7x8xf32>,
    %cls_grad: tensor<2x1x8xf32>)
    -> (tensor<4x12xf32>, tensor<2x8x8xf32>) {
  %zero_f32 = stablehlo.constant dense<0.0> : tensor<f32>
  %z_4x6x1 = stablehlo.broadcast_in_dim %zero_f32, dims = []
      : (tensor<f32>) -> tensor<4x6x1xf32>
  %e_exp = stablehlo.reshape %even_grad
      : (tensor<4x6xf32>) -> tensor<4x6x1xf32>
  %c_even = stablehlo.concatenate %e_exp, %z_4x6x1, dim = 2
      : (tensor<4x6x1xf32>, tensor<4x6x1xf32>) -> tensor<4x6x2xf32>
  %r_even = stablehlo.reshape %c_even : (tensor<4x6x2xf32>) -> tensor<4x12xf32>

  %o_exp = stablehlo.reshape %odd_grad
      : (tensor<4x6xf32>) -> tensor<4x6x1xf32>
  %c_odd = stablehlo.concatenate %z_4x6x1, %o_exp, dim = 2
      : (tensor<4x6x1xf32>, tensor<4x6x1xf32>) -> tensor<4x6x2xf32>
  %r_odd = stablehlo.reshape %c_odd : (tensor<4x6x2xf32>) -> tensor<4x12xf32>
  %interleaved = stablehlo.add %r_odd, %r_even : tensor<4x12xf32>

  %z_2x8x8 = stablehlo.broadcast_in_dim %zero_f32, dims = []
      : (tensor<f32>) -> tensor<2x8x8xf32>
  %z_2x1x8 = stablehlo.broadcast_in_dim %zero_f32, dims = []
      : (tensor<f32>) -> tensor<2x1x8xf32>
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c7 = stablehlo.constant dense<7> : tensor<i64>
  %dus_cls = stablehlo.dynamic_update_slice %z_2x8x8, %cls_grad, %c0, %c7, %c0
      : (tensor<2x8x8xf32>, tensor<2x1x8xf32>,
         tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x8x8xf32>
  %cat_patch = stablehlo.concatenate %patch_grad, %z_2x1x8, dim = 1
      : (tensor<2x7x8xf32>, tensor<2x1x8xf32>) -> tensor<2x8x8xf32>
  %combined_seq = stablehlo.add %dus_cls, %cat_patch : tensor<2x8x8xf32>
  return %interleaved, %combined_seq : tensor<4x12xf32>, tensor<2x8x8xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> orig_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(orig_mod && opt_mod);

  mlir::DenseElementsAttr even_g = MakeF32Tensor(context, {4, 6}, 1.0f);
  mlir::DenseElementsAttr odd_g = MakeF32Tensor(context, {4, 6}, 2.0f);
  mlir::DenseElementsAttr patch_g = MakeF32Tensor(context, {2, 7, 8}, 0.5f);
  mlir::DenseElementsAttr cls_g = MakeF32Tensor(context, {2, 1, 8}, -1.5f);
  mlir::stablehlo::InterpreterConfiguration interp_cfg;
  auto expected = mlir::stablehlo::evalModule(
      *orig_mod, {even_g, odd_g, patch_g, cls_g}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(expected));

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  EXPECT_EQ(CountOps<mlir::stablehlo::DynamicUpdateSliceOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::AddOp>(*opt_mod), 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ConcatenateOp>(*opt_mod), 2);

  auto actual = mlir::stablehlo::evalModule(
      *opt_mod, {even_g, odd_g, patch_g, cls_g}, interp_cfg);
  ASSERT_TRUE(mlir::succeeded(actual));
  for (int out_idx = 0; out_idx < 2; ++out_idx) {
    ASSERT_EQ((*actual)[out_idx].getNumElements(),
              (*expected)[out_idx].getNumElements());
    auto exp_it = (*expected)[out_idx].value_begin<float>();
    auto act_it = (*actual)[out_idx].value_begin<float>();
    for (; exp_it != (*expected)[out_idx].value_end<float>();
         ++exp_it, ++act_it) {
      EXPECT_FLOAT_EQ(*act_it, *exp_it);
    }
  }
}

TEST(CanonicalizeAxisUnbindAndZeroPaddedSumsTest,
     PreservesUnalignedMinorAxisAndOverlappingIntervals) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @main(
    %unaligned: tensor<2x4x3x64xf32>,
    %u0: tensor<2x3x4xf32>,
    %u1: tensor<2x3x4xf32>) -> (tensor<2x4x64xf32>, tensor<2x5x4xf32>) {
  %s0 = stablehlo.slice %unaligned [0:2, 0:4, 0:1, 0:64]
      : (tensor<2x4x3x64xf32>) -> tensor<2x4x1x64xf32>
  %r0 = stablehlo.reshape %s0 : (tensor<2x4x1x64xf32>) -> tensor<2x4x64xf32>
  %s1 = stablehlo.slice %unaligned [0:2, 0:4, 1:2, 0:64]
      : (tensor<2x4x3x64xf32>) -> tensor<2x4x1x64xf32>
  %r1 = stablehlo.reshape %s1 : (tensor<2x4x1x64xf32>) -> tensor<2x4x64xf32>
  %sum_s = stablehlo.add %r0, %r1 : tensor<2x4x64xf32>

  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %p0 = stablehlo.pad %u0, %zero,
      low = [0, 0, 0], high = [0, 2, 0], interior = [0, 0, 0]
      : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x5x4xf32>
  %p1 = stablehlo.pad %u1, %zero,
      low = [0, 2, 0], high = [0, 0, 0], interior = [0, 0, 0]
      : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x5x4xf32>
  %overlap_sum = stablehlo.add %p0, %p1 : tensor<2x5x4xf32>
  return %sum_s, %overlap_sum : tensor<2x4x64xf32>, tensor<2x5x4xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> opt_mod =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(opt_mod);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*opt_mod, &context),
              absl_testing::IsOk());

  // Unaligned 64 (< 128) keeps the 2 reshape ops, and overlapping [0:3) + [2:5)
  // keeps the 2 pad ops.
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*opt_mod), 2);
  EXPECT_EQ(CountOps<mlir::stablehlo::PadOp>(*opt_mod), 2);
}

}  // namespace
}  // namespace torch_tpu
