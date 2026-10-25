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

#include "csrc/mlir/passes/canonicalize_batched_log_softmax_and_dual_dot_grad.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

template <typename OpTy>
int CountOps(mlir::ModuleOp module) {
  int count = 0;
  module.walk([&](OpTy) { ++count; });
  return count;
}

bool RunPass(mlir::ModuleOp module, mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  pm.addNestedPass<mlir::func::FuncOp>(
      createCanonicalizeBatchedLogSoftmaxAndDualDotGradPass());
  return mlir::succeeded(pm.run(module));
}

void InitContext(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

TEST(CanonicalizeBatchedLogSoftmaxAndDualDotGradTest,
     FoldsMultiEntryReshapeDetour) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<8x32xf32>, %y: tensor<8x32xf32>) -> tensor<8x32xf32> {
    %x_3d = stablehlo.reshape %x : (tensor<8x32xf32>) -> tensor<2x4x32xf32>
    %y_3d = stablehlo.reshape %y : (tensor<8x32xf32>) -> tensor<2x4x32xf32>
    %add = stablehlo.add %x_3d, %y_3d : tensor<2x4x32xf32>
    %out = stablehlo.reshape %add : (tensor<2x4x32xf32>) -> tensor<8x32xf32>
    return %out : tensor<8x32xf32>
  }
}
)mlir";
  mlir::MLIRContext context;
  InitContext(context);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(RunPass(*module, &context));

  int rank3_count = 0;
  module->walk([&](mlir::Operation* op) {
    for (mlir::Value res : op->getResults()) {
      if (auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(res.getType())) {
        if (rtt.getRank() == 3) {
          ++rank3_count;
        }
      }
    }
  });
  EXPECT_EQ(rank3_count, 0);
  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*module), 0);
}

TEST(CanonicalizeBatchedLogSoftmaxAndDualDotGradTest,
     FoldsReshapeDetourWithSharedConstant) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<8x32xf32>) -> (tensor<8x32xf32>, tensor<2x4x32xf32>) {
    %c = stablehlo.constant dense<2.000000e+00> : tensor<2x4x32xf32>
    %x_3d = stablehlo.reshape %x : (tensor<8x32xf32>) -> tensor<2x4x32xf32>
    %mul = stablehlo.multiply %x_3d, %c : tensor<2x4x32xf32>
    %out = stablehlo.reshape %mul : (tensor<2x4x32xf32>) -> tensor<8x32xf32>
    %other = stablehlo.abs %c : tensor<2x4x32xf32>
    return %out, %other : tensor<8x32xf32>, tensor<2x4x32xf32>
  }
}
)mlir";
  mlir::MLIRContext context;
  InitContext(context);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(RunPass(*module, &context));

  EXPECT_EQ(CountOps<mlir::stablehlo::ReshapeOp>(*module), 0);
}

TEST(CanonicalizeBatchedLogSoftmaxAndDualDotGradTest,
     CancelsSoftcapBackwardScaleWithSquareAndConverts) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%dy: tensor<8x32xbf16>, %x: tensor<8x32xbf16>) -> tensor<8x32xbf16> {
    %c30 = stablehlo.constant dense<3.000000e+01> : tensor<8x32xbf16>
    %c1 = stablehlo.constant dense<1.000000e+00> : tensor<8x32xbf16>
    %scaled = stablehlo.multiply %dy, %c30 : tensor<8x32xbf16>
    %tanh = stablehlo.tanh %x : tensor<8x32xbf16>
    %conv1 = stablehlo.convert %tanh : (tensor<8x32xbf16>) -> tensor<8x32xf32>
    %conv2 = stablehlo.convert %tanh : (tensor<8x32xbf16>) -> tensor<8x32xf32>
    %sq_f32 = stablehlo.multiply %conv1, %conv2 : tensor<8x32xf32>
    %sq = stablehlo.convert %sq_f32 : (tensor<8x32xf32>) -> tensor<8x32xbf16>
    %one_minus = stablehlo.subtract %c1, %sq : tensor<8x32xbf16>
    %bwd = stablehlo.multiply %scaled, %one_minus : tensor<8x32xbf16>
    %unscaled = stablehlo.divide %bwd, %c30 : tensor<8x32xbf16>
    return %unscaled : tensor<8x32xbf16>
  }
}
)mlir";
  mlir::MLIRContext context;
  InitContext(context);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(RunPass(*module, &context));

  EXPECT_EQ(CountOps<mlir::stablehlo::DivOp>(*module), 0);
}

TEST(CanonicalizeBatchedLogSoftmaxAndDualDotGradTest,
     HoistsMonotonicChainFromReduceMin) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<8x32xf32>) -> tensor<8xf32> {
    %inf = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %exp = stablehlo.exponential %x : tensor<8x32xf32>
    %min = stablehlo.reduce(%exp init: %inf) across dimensions = [1] :
        (tensor<8x32xf32>, tensor<f32>) -> tensor<8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.minimum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    return %min : tensor<8xf32>
  }
}
)mlir";
  mlir::MLIRContext context;
  InitContext(context);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(RunPass(*module, &context));

  int exp_after_reduce = 0;
  module->walk([&](mlir::stablehlo::ReduceOp red) {
    EXPECT_EQ(red.getInputs()[0].getDefiningOp(), nullptr);
    for (mlir::Operation* user : red.getResult(0).getUsers()) {
      if (llvm::isa<mlir::stablehlo::ExpOp>(user)) {
        ++exp_after_reduce;
      }
    }
  });
  EXPECT_EQ(exp_after_reduce, 1);
}

TEST(CanonicalizeBatchedLogSoftmaxAndDualDotGradTest,
     HoistsInvariantAddAndSubtractProjections) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<8x32xf32>, %b: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %bcast = stablehlo.broadcast_in_dim %b, dims = [0] :
        (tensor<8xf32>) -> tensor<8x32xf32>
    %add_lhs = stablehlo.add %bcast, %x : tensor<8x32xf32>
    %max1 = stablehlo.reduce(%add_lhs init: %neg_inf) across dimensions = [1] :
        (tensor<8x32xf32>, tensor<f32>) -> tensor<8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %sub_lhs = stablehlo.subtract %x, %bcast : tensor<8x32xf32>
    %max2 = stablehlo.reduce(%sub_lhs init: %neg_inf) across dimensions = [1] :
        (tensor<8x32xf32>, tensor<f32>) -> tensor<8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    return %max1, %max2 : tensor<8xf32>, tensor<8xf32>
  }
}
)mlir";
  mlir::MLIRContext context;
  InitContext(context);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(RunPass(*module, &context));

  module->walk([&](mlir::stablehlo::ReduceOp red) {
    EXPECT_EQ(red.getInputs()[0].getDefiningOp(), nullptr);
  });
}

TEST(CanonicalizeBatchedLogSoftmaxAndDualDotGradTest,
     HoistsNonlinearOperatorsLogisticAndErf) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<8x32xf32>) -> tensor<8xf32> {
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %erf = chlo.erf %x : tensor<8x32xf32> -> tensor<8x32xf32>
    %logistic = stablehlo.logistic %erf : tensor<8x32xf32>
    %max = stablehlo.reduce(%logistic init: %neg_inf) across dimensions = [1] :
        (tensor<8x32xf32>, tensor<f32>) -> tensor<8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    return %max : tensor<8xf32>
  }
}
)mlir";
  mlir::MLIRContext context;
  InitContext(context);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(RunPass(*module, &context));

  module->walk([&](mlir::stablehlo::ReduceOp red) {
    EXPECT_EQ(red.getInputs()[0].getDefiningOp(), nullptr);
  });
}

TEST(CanonicalizeBatchedLogSoftmaxAndDualDotGradTest,
     BarrierNarrowedMultiDotGradientPositiveAndNegative) {
  static constexpr const char* kIrNegSingleTranscendental = R"mlir(
module {
  func.func @main(%x: tensor<8x32xf32>, %w1: tensor<32x16xbf16>,
                  %w2: tensor<8x16xbf16>) -> (tensor<8x16xbf16>, tensor<32x16xbf16>) {
    %t = stablehlo.tanh %x : tensor<8x32xf32>
    %narrow = stablehlo.convert %t : (tensor<8x32xf32>) -> tensor<8x32xbf16>
    %d1 = stablehlo.dot_general %narrow, %w1, contracting_dims = [1] x [0] :
        (tensor<8x32xbf16>, tensor<32x16xbf16>) -> tensor<8x16xbf16>
    %narrow_t = stablehlo.transpose %narrow, dims = [1, 0] :
        (tensor<8x32xbf16>) -> tensor<32x8xbf16>
    %d2 = stablehlo.dot_general %narrow_t, %w2, contracting_dims = [1] x [0] :
        (tensor<32x8xbf16>, tensor<8x16xbf16>) -> tensor<32x16xbf16>
    return %d1, %d2 : tensor<8x16xbf16>, tensor<32x16xbf16>
  }
}
)mlir";

  static constexpr const char* kIrNegIdenticalAxes = R"mlir(
module {
  func.func @main(%x: tensor<8x32xf32>, %w1: tensor<32x16xbf16>,
                  %w2: tensor<32x16xbf16>) -> (tensor<8x16xbf16>, tensor<8x16xbf16>) {
    %t = stablehlo.tanh %x : tensor<8x32xf32>
    %e = stablehlo.exponential %t : tensor<8x32xf32>
    %narrow = stablehlo.convert %e : (tensor<8x32xf32>) -> tensor<8x32xbf16>
    %d1 = stablehlo.dot_general %narrow, %w1, contracting_dims = [1] x [0] :
        (tensor<8x32xbf16>, tensor<32x16xbf16>) -> tensor<8x16xbf16>
    %d2 = stablehlo.dot_general %narrow, %w2, contracting_dims = [1] x [0] :
        (tensor<8x32xbf16>, tensor<32x16xbf16>) -> tensor<8x16xbf16>
    return %d1, %d2 : tensor<8x16xbf16>, tensor<8x16xbf16>
  }
}
)mlir";

  static constexpr const char* kIrPositive = R"mlir(
module {
  func.func @main(%x: tensor<8x32xf32>, %w1: tensor<32x16xbf16>,
                  %w2: tensor<8x16xbf16>) -> (tensor<8x16xbf16>, tensor<32x16xbf16>) {
    %t = stablehlo.tanh %x : tensor<8x32xf32>
    %e = stablehlo.exponential %t : tensor<8x32xf32>
    %narrow = stablehlo.convert %e : (tensor<8x32xf32>) -> tensor<8x32xbf16>
    %d1 = stablehlo.dot_general %narrow, %w1, contracting_dims = [1] x [0] :
        (tensor<8x32xbf16>, tensor<32x16xbf16>) -> tensor<8x16xbf16>
    %narrow_t = stablehlo.transpose %narrow, dims = [1, 0] :
        (tensor<8x32xbf16>) -> tensor<32x8xbf16>
    %d2 = stablehlo.dot_general %narrow_t, %w2, contracting_dims = [1] x [0] :
        (tensor<32x8xbf16>, tensor<8x16xbf16>) -> tensor<32x16xbf16>
    return %d1, %d2 : tensor<8x16xbf16>, tensor<32x16xbf16>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);

  auto m_neg1 = mlir::parseSourceString<mlir::ModuleOp>(
      kIrNegSingleTranscendental, &context);
  ASSERT_TRUE(m_neg1);
  ASSERT_TRUE(RunPass(*m_neg1, &context));
  EXPECT_EQ(CountOps<mlir::stablehlo::OptimizationBarrierOp>(*m_neg1), 0);

  auto m_neg2 =
      mlir::parseSourceString<mlir::ModuleOp>(kIrNegIdenticalAxes, &context);
  ASSERT_TRUE(m_neg2);
  ASSERT_TRUE(RunPass(*m_neg2, &context));
  EXPECT_EQ(CountOps<mlir::stablehlo::OptimizationBarrierOp>(*m_neg2), 0);

  auto m_pos = mlir::parseSourceString<mlir::ModuleOp>(kIrPositive, &context);
  ASSERT_TRUE(m_pos);
  ASSERT_TRUE(RunPass(*m_pos, &context));
  EXPECT_EQ(CountOps<mlir::stablehlo::OptimizationBarrierOp>(*m_pos), 1);
}

}  // namespace
}  // namespace torch_tpu
