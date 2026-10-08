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

#include "csrc/mlir/passes/reassociate_norm_weights.h"

#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// Skips broadcast_in_dim ops to find the underlying value.
mlir::Value SkipBroadcasts(mlir::Value v) {
  while (v) {
    if (auto bcast = v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      v = bcast.getOperand();
    } else {
      break;
    }
  }
  return v;
}

// Checks if a value is a constant 1.0 (float) or 1 (integer) splat.
bool IsOneSplat(mlir::Value val) {
  val = SkipBroadcasts(val);
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) return false;
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr || attr.empty() || !attr.isSplat()) return false;
  if (mlir::isa<mlir::FloatType>(attr.getElementType())) {
    return attr.getSplatValue<llvm::APFloat>().isExactlyValue(1.0);
  }
  if (mlir::isa<mlir::IntegerType>(attr.getElementType())) {
    return attr.getSplatValue<llvm::APInt>() == 1;
  }
  return false;
}

struct MatchedWeight {
  mlir::Value raw_weight;
  mlir::stablehlo::BroadcastInDimOp outer_bcast;
};

// Matches:
// 1. (1.0 + weight) or (weight + 1.0)
// 2. broadcast_in_dim(1.0 + weight) or broadcast_in_dim(weight + 1.0)
//
// Returns matched weight metadata if matched, without mutating the IR.
std::optional<MatchedWeight> MatchOnePlusWeight(mlir::Value val) {
  mlir::stablehlo::BroadcastInDimOp outer_bcast =
      val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  mlir::Value inner_val = outer_bcast ? outer_bcast.getOperand() : val;

  auto add_op = inner_val.getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add_op) return std::nullopt;

  mlir::Value raw_weight;
  if (IsOneSplat(add_op.getLhs())) {
    raw_weight = add_op.getRhs();
  } else if (IsOneSplat(add_op.getRhs())) {
    raw_weight = add_op.getLhs();
  } else {
    return std::nullopt;
  }

  return MatchedWeight{raw_weight, outer_bcast};
}

// Rewrites:
//   mul(norm_x, 1.0 + weight)  OR  mul(1.0 + weight, norm_x)
// Into:
//   add(norm_x, mul(norm_x, weight))
struct ReassociateMulNormPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::MulOp> {
  using OpRewritePattern<mlir::stablehlo::MulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::MulOp mul_op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Value lhs = mul_op.getLhs();
    mlir::Value rhs = mul_op.getRhs();

    mlir::Location loc = mul_op.getLoc();
    mlir::Value norm_x;
    std::optional<MatchedWeight> matched;

    // Case 1: mul(norm_x, 1.0 + weight)
    if (auto m = MatchOnePlusWeight(rhs)) {
      norm_x = lhs;
      matched = m;
    }
    // Case 2: mul(1.0 + weight, norm_x)
    else if (auto m = MatchOnePlusWeight(lhs)) {
      norm_x = rhs;
      matched = m;
    } else {
      return mlir::failure();
    }

    // Weight tensor rank check: weight should be rank <= 1 (1D feature weight,
    // optionally broadcasted to match norm_x's shape).
    mlir::Value raw_weight = SkipBroadcasts(matched->raw_weight);
    auto raw_weight_type =
        mlir::dyn_cast<mlir::RankedTensorType>(raw_weight.getType());
    if (!raw_weight_type || raw_weight_type.getRank() > 1) {
      return mlir::failure();
    }

    // If the add_op was wrapped in a broadcast_in_dim, reconstruct the
    // broadcast on weight now that matching and constraints have succeeded.
    mlir::Value weight = matched->raw_weight;
    if (matched->outer_bcast) {
      weight = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, matched->outer_bcast.getType(), weight,
          matched->outer_bcast.getBroadcastDimensionsAttr());
    }

    auto new_mul = rewriter.create<mlir::stablehlo::MulOp>(
        loc, mul_op.getType(), norm_x, weight);
    auto new_add = rewriter.create<mlir::stablehlo::AddOp>(
        loc, mul_op.getType(), norm_x, new_mul);

    rewriter.replaceOp(mul_op, new_add);
    return mlir::success();
  }
};

struct ReassociateNormWeightsPass
    : public mlir::PassWrapper<ReassociateNormWeightsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReassociateNormWeightsPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-reassociate-norm-weights";
  }
  llvm::StringRef getDescription() const override {
    return "Reassociates norm_x * (1.0 + weight) into norm_x + (norm_x * "
           "weight).";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<ReassociateMulNormPattern>(context);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createReassociateNormWeightsPass() {
  return std::make_unique<ReassociateNormWeightsPass>();
}

}  // namespace torch_tpu
