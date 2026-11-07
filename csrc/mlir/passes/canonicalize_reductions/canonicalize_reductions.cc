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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // NOLINT
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"  // NOLINT
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"  // NOLINT
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // NOLINT
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

[[nodiscard]] mlir::RankedTensorType GetStaticRankedTensorType(
    mlir::Value val) {
  if (!val) {
    return nullptr;
  }
  auto ty = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!ty || !ty.hasStaticShape()) {
    return nullptr;
  }
  return ty;
}

[[nodiscard]] mlir::RankedTensorType GetStaticFloatTensorType(mlir::Value val) {
  auto ty = GetStaticRankedTensorType(val);
  if (!ty || !mlir::isa<mlir::FloatType>(ty.getElementType())) {
    return nullptr;
  }
  return ty;
}

[[nodiscard]] std::optional<mlir::Value> UnwrapFloatSplatSource(
    mlir::Value val, llvm::SmallVectorImpl<mlir::FloatType>& convert_types) {
  while (true) {
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      val = bcast.getOperand();
      continue;
    }
    if (auto reshape = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      val = reshape.getOperand();
      continue;
    }
    if (auto convert = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      auto in_ty = GetStaticFloatTensorType(convert.getOperand());
      auto out_ty = GetStaticFloatTensorType(convert.getResult());
      if (!in_ty || !out_ty) {
        return std::nullopt;
      }
      convert_types.push_back(
          mlir::cast<mlir::FloatType>(out_ty.getElementType()));
      val = convert.getOperand();
      continue;
    }
    return val;
  }
}

[[nodiscard]] std::optional<double> ExtractFloatSplatFromConstantOp(
    mlir::stablehlo::ConstantOp const_op,
    llvm::ArrayRef<mlir::FloatType> convert_types) {
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense || !dense.isSplat()) {
    return std::nullopt;
  }
  auto fl_ty = mlir::dyn_cast<mlir::FloatType>(dense.getElementType());
  if (!fl_ty) {
    return std::nullopt;
  }
  llvm::APFloat ap_val = dense.getSplatValue<llvm::APFloat>();
  for (mlir::FloatType f_ty : llvm::reverse(convert_types)) {
    bool loses_info = false;
    ap_val.convert(f_ty.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
                   &loses_info);
    if (loses_info) {
      return std::nullopt;
    }
  }
  if (!ap_val.isFinite()) {
    return std::nullopt;
  }
  return ap_val.convertToDouble();
}

[[nodiscard]] std::optional<double> ExtractSplatFloatConstant(mlir::Value val) {
  llvm::SmallVector<mlir::FloatType, 4> convert_types;
  std::optional<mlir::Value> unwrapped =
      UnwrapFloatSplatSource(val, convert_types);
  if (!unwrapped.has_value()) {
    return std::nullopt;
  }
  auto const_op = unwrapped->getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  return ExtractFloatSplatFromConstantOp(const_op, convert_types);
}

[[nodiscard]] bool HasNonFiniteConstantSource(mlir::Value val) {
  llvm::SmallVector<mlir::FloatType, 4> convert_types;
  std::optional<mlir::Value> unwrapped =
      UnwrapFloatSplatSource(val, convert_types);
  if (!unwrapped.has_value()) {
    return false;
  }
  auto const_op = unwrapped->getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense) {
    return false;
  }
  auto fl_ty = mlir::dyn_cast<mlir::FloatType>(dense.getElementType());
  if (!fl_ty) {
    return false;
  }
  auto is_non_finite = [&convert_types](llvm::APFloat ap_val) {
    for (mlir::FloatType f_ty : llvm::reverse(convert_types)) {
      bool loses_info = false;
      ap_val.convert(f_ty.getFloatSemantics(),
                     llvm::APFloat::rmNearestTiesToEven, &loses_info);
    }
    return !ap_val.isFinite();
  };
  if (dense.isSplat()) {
    return is_non_finite(dense.getSplatValue<llvm::APFloat>());
  }
  for (llvm::APFloat ap_val : dense.getValues<llvm::APFloat>()) {
    if (is_non_finite(ap_val)) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool SameNonZeroSplatConstant(mlir::Value a, mlir::Value b) {
  const std::optional<double> va = ExtractSplatFloatConstant(a);
  const std::optional<double> vb = ExtractSplatFloatConstant(b);
  if (!va.has_value() || !vb.has_value() || *va == 0.0 || *vb == 0.0) {
    return false;
  }
  return *va == *vb;
}

[[nodiscard]] mlir::Value CreateSplatFloatConstant(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::RankedTensorType tensor_type, double val) {
  auto fl_elem = mlir::cast<mlir::FloatType>(tensor_type.getElementType());
  llvm::APFloat ap_val(val);
  bool loses_info = false;
  ap_val.convert(fl_elem.getFloatSemantics(),
                 llvm::APFloat::rmNearestTiesToEven, &loses_info);
  auto dense = mlir::DenseElementsAttr::get(tensor_type, ap_val);
  return builder.create<mlir::stablehlo::ConstantOp>(loc, tensor_type, dense);
}

[[nodiscard]] bool IsReductionInvariantOperand(
    mlir::Value val, const llvm::DenseSet<int64_t>& red_dims) {
  if (HasNonFiniteConstantSource(val)) {
    return false;
  }
  if (ExtractSplatFloatConstant(val).has_value()) {
    return true;
  }
  auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast || !GetStaticRankedTensorType(bcast.getOperand())) {
    return false;
  }
  if (HasNonFiniteConstantSource(bcast.getOperand())) {
    return false;
  }
  for (const int64_t d : bcast.getBroadcastDimensions()) {
    if (red_dims.contains(d)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] mlir::Value ShiftBroadcastDims(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val,
    mlir::RankedTensorType reduced_type,
    const llvm::DenseSet<int64_t>& red_dims) {
  if (std::optional<double> c = ExtractSplatFloatConstant(val)) {
    return CreateSplatFloatConstant(builder, loc, reduced_type, *c);
  }
  auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast) {
    return nullptr;
  }
  llvm::SmallVector<int64_t> new_dims;
  for (const int64_t d : bcast.getBroadcastDimensions()) {
    int64_t shift = 0;
    for (const int64_t r : red_dims) {
      if (r < d) {
        ++shift;
      }
    }
    new_dims.push_back(d - shift);
  }
  mlir::Value bcast_in = bcast.getOperand();
  auto bcast_in_ty = GetStaticRankedTensorType(bcast_in);
  if (!bcast_in_ty ||
      bcast_in_ty.getElementType() != reduced_type.getElementType()) {
    return nullptr;
  }
  if (bcast_in.getType() == reduced_type &&
      new_dims.size() == static_cast<size_t>(reduced_type.getRank())) {
    bool is_identity = true;
    for (int64_t i = 0; i < reduced_type.getRank(); ++i) {
      if (new_dims[i] != i) {
        is_identity = false;
        break;
      }
    }
    if (is_identity) {
      return bcast_in;
    }
  }
  return builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, reduced_type, bcast_in, builder.getDenseI64ArrayAttr(new_dims));
}

// -----------------------------------------------------------------------------
// Mechanism 1: Hoist monotonic scalar scale/convert chains from reduce_max
// -----------------------------------------------------------------------------

enum class ReduceKind { kMax, kMin };

struct ReduceMatch {
  ReduceKind kind;
  bool is_inf = false;
};

[[nodiscard]] std::optional<ReduceKind> MatchReduceCombiner(mlir::Block& body) {
  if (body.getOperations().size() != 2) {
    return std::nullopt;
  }
  auto ret = mlir::dyn_cast<mlir::stablehlo::ReturnOp>(body.getTerminator());
  if (!ret || ret.getNumOperands() != 1) {
    return std::nullopt;
  }
  mlir::Operation* bin = &body.front();
  if (bin->getNumOperands() != 2 || bin->getResult(0) != ret.getOperand(0)) {
    return std::nullopt;
  }
  mlir::Value a0 = body.getArgument(0);
  mlir::Value a1 = body.getArgument(1);
  if (!((bin->getOperand(0) == a0 && bin->getOperand(1) == a1) ||
        (bin->getOperand(0) == a1 && bin->getOperand(1) == a0))) {
    return std::nullopt;
  }
  if (mlir::isa<mlir::stablehlo::MaxOp>(bin)) {
    return ReduceKind::kMax;
  }
  if (mlir::isa<mlir::stablehlo::MinOp>(bin)) {
    return ReduceKind::kMin;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<bool> ValidateReduceInitValue(mlir::Value init_val,
                                                          ReduceKind kind) {
  auto const_op = init_val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense || !dense.isSplat()) {
    return std::nullopt;
  }
  auto fl_ty = mlir::dyn_cast<mlir::FloatType>(dense.getElementType());
  if (!fl_ty) {
    return std::nullopt;
  }
  llvm::APFloat ap = dense.getSplatValue<llvm::APFloat>();
  const bool is_neg = ap.isNegative();
  const bool is_inf = ap.isInfinity();
  // Reduce-max init must be exactly -inf, never -FLT_MAX.
  // Reduce-min init must be exactly +inf, never +FLT_MAX.
  if (kind == ReduceKind::kMax) {
    if (!is_neg || !is_inf) {
      return std::nullopt;
    }
  } else {
    if (is_neg || !is_inf) {
      return std::nullopt;
    }
  }
  return is_inf;
}

[[nodiscard]] std::optional<ReduceMatch> MatchMaxOrMinReduce(
    mlir::stablehlo::ReduceOp reduce_op) {
  if (reduce_op.getInputs().size() != 1 ||
      reduce_op.getInitValues().size() != 1 || reduce_op.getNumResults() != 1) {
    return std::nullopt;
  }
  mlir::Region& region = reduce_op.getBody();
  if (!region.hasOneBlock()) {
    return std::nullopt;
  }
  mlir::Block& body = region.front();
  if (body.getNumArguments() != 2) {
    return std::nullopt;
  }
  std::optional<ReduceKind> kind = MatchReduceCombiner(body);
  if (!kind.has_value()) {
    return std::nullopt;
  }
  std::optional<bool> is_inf =
      ValidateReduceInitValue(reduce_op.getInitValues()[0], *kind);
  if (!is_inf.has_value()) {
    return std::nullopt;
  }
  return ReduceMatch{*kind, *is_inf};
}

enum class StepKind {
  kConvert,
  kExp,
  kTanh,
  kNeg,
  kAbs,
  kMulScalar,
  kDivScalar,
  kAddInvariant,
  kSubInvariant,
};

struct MonotonicStep {
  StepKind kind;
  mlir::Operation* op = nullptr;
  mlir::Value input_val;
  mlir::Value other_opnd;
  bool is_lhs = true;
};

[[nodiscard]] std::optional<MonotonicStep> MatchMonotonicStep(
    mlir::Value val, const llvm::DenseSet<int64_t>& red_dims) {
  mlir::Operation* def_op = val.getDefiningOp();
  if (!def_op) {
    return std::nullopt;
  }
  if (auto cvt = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(def_op)) {
    auto in_ty = GetStaticFloatTensorType(cvt.getOperand());
    auto out_ty = GetStaticFloatTensorType(cvt.getResult());
    if (in_ty && out_ty) {
      auto in_fl = mlir::cast<mlir::FloatType>(in_ty.getElementType());
      auto out_fl = mlir::cast<mlir::FloatType>(out_ty.getElementType());
      if (llvm::APFloat::semanticsHasInf(in_fl.getFloatSemantics()) &&
          llvm::APFloat::semanticsHasInf(out_fl.getFloatSemantics())) {
        return MonotonicStep{StepKind::kConvert, def_op, cvt.getOperand(),
                             nullptr, true};
      }
    }
    return std::nullopt;
  }
  if (auto exp = mlir::dyn_cast<mlir::stablehlo::ExpOp>(def_op)) {
    return MonotonicStep{StepKind::kExp, def_op, exp.getOperand(), nullptr,
                         true};
  }
  if (auto tanh = mlir::dyn_cast<mlir::stablehlo::TanhOp>(def_op)) {
    return MonotonicStep{StepKind::kTanh, def_op, tanh.getOperand(), nullptr,
                         true};
  }
  if (auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(def_op)) {
    std::optional<double> c0 = ExtractSplatFloatConstant(mul.getLhs());
    std::optional<double> c1 = ExtractSplatFloatConstant(mul.getRhs());
    if (c1.has_value() && *c1 > 0.0) {
      return MonotonicStep{StepKind::kMulScalar, def_op, mul.getLhs(),
                           mul.getRhs(), true};
    }
    if (c0.has_value() && *c0 > 0.0) {
      return MonotonicStep{StepKind::kMulScalar, def_op, mul.getRhs(),
                           mul.getLhs(), false};
    }
    return std::nullopt;
  }
  if (auto div = mlir::dyn_cast<mlir::stablehlo::DivOp>(def_op)) {
    std::optional<double> c = ExtractSplatFloatConstant(div.getRhs());
    if (c.has_value() && *c > 0.0) {
      return MonotonicStep{StepKind::kDivScalar, def_op, div.getLhs(),
                           div.getRhs(), true};
    }
    return std::nullopt;
  }
  if (auto add = mlir::dyn_cast<mlir::stablehlo::AddOp>(def_op)) {
    if (IsReductionInvariantOperand(add.getRhs(), red_dims)) {
      return MonotonicStep{StepKind::kAddInvariant, def_op, add.getLhs(),
                           add.getRhs(), true};
    }
    if (IsReductionInvariantOperand(add.getLhs(), red_dims)) {
      return MonotonicStep{StepKind::kAddInvariant, def_op, add.getRhs(),
                           add.getLhs(), false};
    }
    return std::nullopt;
  }
  if (auto sub = mlir::dyn_cast<mlir::stablehlo::SubtractOp>(def_op)) {
    if (IsReductionInvariantOperand(sub.getRhs(), red_dims)) {
      return MonotonicStep{StepKind::kSubInvariant, def_op, sub.getLhs(),
                           sub.getRhs(), true};
    }
    return std::nullopt;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<mlir::Value> CollectMonotonicChain(
    mlir::Value input, const llvm::DenseSet<int64_t>& red_dims,
    llvm::SmallVectorImpl<MonotonicStep>& rev_steps) {
  mlir::Value cur = input;
  bool all_single_use = true;
  bool seen_widening_convert = false;
  bool seen_non_convert = false;
  bool seen_nonlinear = false;
  size_t best_prefix_len = 0;

  while (true) {
    if (rev_steps.size() >= 8) {
      break;
    }
    std::optional<MonotonicStep> step = MatchMonotonicStep(cur, red_dims);
    if (!step.has_value()) {
      break;
    }
    if (cur != input) {
      all_single_use = all_single_use && cur.hasOneUse();
    }
    if (step->kind == StepKind::kConvert) {
      auto in_ty = GetStaticFloatTensorType(step->input_val);
      auto out_ty = GetStaticFloatTensorType(step->op->getResult(0));
      if (in_ty && out_ty &&
          in_ty.getElementType().getIntOrFloatBitWidth() <
              out_ty.getElementType().getIntOrFloatBitWidth()) {
        seen_widening_convert = true;
      }
    } else {
      seen_non_convert = true;
    }
    const bool is_nonlinear =
        (step->kind == StepKind::kExp || step->kind == StepKind::kTanh);
    if (is_nonlinear) {
      seen_nonlinear = true;
    }
    rev_steps.push_back(*step);

    if (all_single_use && seen_non_convert &&
        (input.hasOneUse() || (seen_widening_convert && seen_nonlinear))) {
      best_prefix_len = rev_steps.size();
    }
    cur = step->input_val;
  }

  if (rev_steps.empty()) {
    return std::nullopt;
  }
  bool has_non_convert = false;
  for (const MonotonicStep& s : rev_steps) {
    if (s.kind != StepKind::kConvert) {
      has_non_convert = true;
      break;
    }
  }
  if (!has_non_convert) {
    return std::nullopt;
  }
  // Linear-only chains on multi-use inputs must not be hoisted to avoid
  // duplicating compute; only widening converts with nonlinear steps (such
  // as softcapped logits) justify hoisting when input has multiple uses.
  const bool can_use_full_chain =
      all_single_use &&
      (input.hasOneUse() || (seen_widening_convert && seen_nonlinear));
  if (!can_use_full_chain) {
    if (best_prefix_len == 0) {
      return std::nullopt;
    }
    rev_steps.truncate(best_prefix_len);
    cur = rev_steps.back().input_val;
  }
  return cur;
}

[[nodiscard]] mlir::Value CreateHoistedReduceOp(
    mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value base_input,
    mlir::RankedTensorType out_type, mlir::DenseI64ArrayAttr dims_attr,
    ReduceKind red_kind, bool init_is_inf) {
  auto base_ty = GetStaticFloatTensorType(base_input);
  if (!base_ty) {
    return nullptr;
  }
  auto base_fl = mlir::cast<mlir::FloatType>(base_ty.getElementType());
  if (init_is_inf &&
      !llvm::APFloat::semanticsHasInf(base_fl.getFloatSemantics())) {
    return nullptr;
  }
  auto scalar_base_ty = mlir::RankedTensorType::get({}, base_fl);
  const bool neg_init = (red_kind == ReduceKind::kMax);
  llvm::APFloat init_ap =
      init_is_inf
          ? llvm::APFloat::getInf(base_fl.getFloatSemantics(), neg_init)
          : llvm::APFloat::getLargest(base_fl.getFloatSemantics(), neg_init);
  mlir::Value new_init = rewriter.create<mlir::stablehlo::ConstantOp>(
      loc, scalar_base_ty,
      mlir::DenseElementsAttr::get(scalar_base_ty, init_ap));

  auto red_base_ty = mlir::RankedTensorType::get(out_type.getShape(), base_fl);
  auto new_reduce = rewriter.create<mlir::stablehlo::ReduceOp>(
      loc, mlir::TypeRange{red_base_ty}, mlir::ValueRange{base_input},
      mlir::ValueRange{new_init}, dims_attr);

  mlir::Block* body =
      rewriter.createBlock(&new_reduce.getBody(), new_reduce.getBody().end(),
                           {scalar_base_ty, scalar_base_ty}, {loc, loc});
  rewriter.setInsertionPointToStart(body);
  mlir::Value r0 = body->getArgument(0);
  mlir::Value r1 = body->getArgument(1);
  mlir::Value comb;
  if (red_kind == ReduceKind::kMax) {
    comb = rewriter.create<mlir::stablehlo::MaxOp>(loc, r0, r1);
  } else {
    comb = rewriter.create<mlir::stablehlo::MinOp>(loc, r0, r1);
  }
  rewriter.create<mlir::stablehlo::ReturnOp>(loc, mlir::ValueRange{comb});
  rewriter.setInsertionPointAfter(new_reduce);
  return new_reduce.getResult(0);
}

[[nodiscard]] mlir::Value ApplyMonotonicStep(
    mlir::PatternRewriter& rewriter, mlir::Location loc,
    const MonotonicStep& step, mlir::Value cur_val,
    const llvm::DenseSet<int64_t>& red_dims) {
  auto cur_ty = GetStaticRankedTensorType(cur_val);
  switch (step.kind) {
    case StepKind::kConvert: {
      auto orig_out_ty = GetStaticRankedTensorType(step.op->getResult(0));
      auto target_ty = mlir::RankedTensorType::get(
          cur_ty.getShape(), orig_out_ty.getElementType());
      return rewriter.create<mlir::stablehlo::ConvertOp>(loc, target_ty,
                                                         cur_val);
    }
    case StepKind::kExp:
      return rewriter.create<mlir::stablehlo::ExpOp>(loc, cur_val);
    case StepKind::kTanh:
      return rewriter.create<mlir::stablehlo::TanhOp>(loc, cur_val);
    case StepKind::kMulScalar: {
      mlir::Value shifted_other =
          ShiftBroadcastDims(rewriter, loc, step.other_opnd, cur_ty, red_dims);
      if (!shifted_other) return nullptr;
      mlir::Value lhs = step.is_lhs ? cur_val : shifted_other;
      mlir::Value rhs = step.is_lhs ? shifted_other : cur_val;
      return rewriter.create<mlir::stablehlo::MulOp>(loc, lhs, rhs);
    }
    case StepKind::kDivScalar: {
      mlir::Value shifted_other =
          ShiftBroadcastDims(rewriter, loc, step.other_opnd, cur_ty, red_dims);
      if (!shifted_other) return nullptr;
      return rewriter.create<mlir::stablehlo::DivOp>(loc, cur_val,
                                                     shifted_other);
    }
    case StepKind::kAddInvariant: {
      mlir::Value shifted_other =
          ShiftBroadcastDims(rewriter, loc, step.other_opnd, cur_ty, red_dims);
      if (!shifted_other) return nullptr;
      mlir::Value lhs = step.is_lhs ? cur_val : shifted_other;
      mlir::Value rhs = step.is_lhs ? shifted_other : cur_val;
      return rewriter.create<mlir::stablehlo::AddOp>(loc, lhs, rhs);
    }
    case StepKind::kSubInvariant: {
      mlir::Value shifted_other =
          ShiftBroadcastDims(rewriter, loc, step.other_opnd, cur_ty, red_dims);
      if (!shifted_other) return nullptr;
      return rewriter.create<mlir::stablehlo::SubtractOp>(loc, cur_val,
                                                          shifted_other);
    }
    default:
      return nullptr;
  }
}

struct HoistMonotonicReduceChainPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern<mlir::stablehlo::ReduceOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReduceOp reduce_op,
      mlir::PatternRewriter& rewriter) const override {
    std::optional<ReduceMatch> match = MatchMaxOrMinReduce(reduce_op);
    if (!match.has_value()) {
      return mlir::failure();
    }
    mlir::Value input = reduce_op.getInputs()[0];
    auto in_ty = GetStaticRankedTensorType(input);
    auto out_ty = GetStaticRankedTensorType(reduce_op.getResults()[0]);
    if (!in_ty || !out_ty) {
      return mlir::failure();
    }
    llvm::ArrayRef<int64_t> dims = reduce_op.getDimensions();
    if (dims.empty()) {
      return mlir::failure();
    }
    for (int64_t d : dims) {
      if (d < 0 || d >= in_ty.getRank() || in_ty.getDimSize(d) <= 0) {
        return mlir::failure();
      }
    }
    llvm::DenseSet<int64_t> red_dims(dims.begin(), dims.end());
    llvm::SmallVector<MonotonicStep, 8> rev_steps;
    std::optional<mlir::Value> base_val =
        CollectMonotonicChain(input, red_dims, rev_steps);
    if (!base_val.has_value() || rev_steps.empty()) {
      return mlir::failure();
    }
    auto base_ty = GetStaticFloatTensorType(*base_val);
    if (!base_ty) {
      return mlir::failure();
    }
    auto base_fl = mlir::cast<mlir::FloatType>(base_ty.getElementType());
    if (match->is_inf &&
        !llvm::APFloat::semanticsHasInf(base_fl.getFloatSemantics())) {
      return mlir::failure();
    }

    mlir::Location loc = reduce_op.getLoc();
    mlir::Value hoisted_reduce = CreateHoistedReduceOp(
        rewriter, loc, *base_val, out_ty, reduce_op.getDimensionsAttr(),
        match->kind, match->is_inf);
    if (!hoisted_reduce) {
      return mlir::failure();
    }

    mlir::Value cur = hoisted_reduce;
    for (const MonotonicStep& step : llvm::reverse(rev_steps)) {
      cur = ApplyMonotonicStep(rewriter, loc, step, cur, red_dims);
      if (!cur) {
        return mlir::failure();
      }
    }
    rewriter.replaceOp(reduce_op, cur);
    return mlir::success();
  }
};

// -----------------------------------------------------------------------------
// Mechanism 5: Fold unit unreduced axes into variadic reductions
// -----------------------------------------------------------------------------

[[nodiscard]] bool IsSingleInputIdentityReduction(mlir::Value a0,
                                                  mlir::Value a1,
                                                  mlir::Value ret_val,
                                                  mlir::Value init_val) {
  mlir::Operation* bin = ret_val.getDefiningOp();
  if (!bin || bin->getNumOperands() != 2) {
    return false;
  }
  if (!((bin->getOperand(0) == a0 && bin->getOperand(1) == a1) ||
        (bin->getOperand(0) == a1 && bin->getOperand(1) == a0))) {
    return false;
  }
  if (mlir::isa<mlir::stablehlo::AddOp>(bin)) {
    return ExtractSplatFloatConstant(init_val) == 0.0;
  }
  if (mlir::isa<mlir::stablehlo::MulOp>(bin)) {
    return ExtractSplatFloatConstant(init_val) == 1.0;
  }
  if (mlir::isa<mlir::stablehlo::MaxOp>(bin)) {
    return ValidateReduceInitValue(init_val, ReduceKind::kMax).has_value();
  }
  if (mlir::isa<mlir::stablehlo::MinOp>(bin)) {
    return ValidateReduceInitValue(init_val, ReduceKind::kMin).has_value();
  }
  return false;
}

[[nodiscard]] bool IsIdentityReduction(mlir::stablehlo::ReduceOp reduce_op) {
  if (reduce_op.getInputs().size() != reduce_op.getInitValues().size()) {
    return false;
  }
  mlir::Region& region = reduce_op.getBody();
  if (!region.hasOneBlock()) {
    return false;
  }
  mlir::Block& body = region.front();
  const size_t num_inputs = reduce_op.getInputs().size();
  if (body.getNumArguments() != 2 * num_inputs) {
    return false;
  }
  auto ret = mlir::dyn_cast<mlir::stablehlo::ReturnOp>(body.getTerminator());
  if (!ret || ret.getNumOperands() != num_inputs) {
    return false;
  }
  for (size_t i = 0; i < num_inputs; ++i) {
    if (!IsSingleInputIdentityReduction(
            body.getArgument(i), body.getArgument(i + num_inputs),
            ret.getOperand(i), reduce_op.getInitValues()[i])) {
      return false;
    }
  }
  return true;
}

struct FoldUnitUnreducedAxesPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern<mlir::stablehlo::ReduceOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReduceOp reduce_op,
      mlir::PatternRewriter& rewriter) const override {
    if (reduce_op.getNumResults() <= 1) {
      return mlir::failure();
    }
    for (mlir::Value input : reduce_op.getInputs()) {
      if (!GetStaticRankedTensorType(input)) {
        return mlir::failure();
      }
    }
    for (mlir::Value res : reduce_op.getResults()) {
      if (!GetStaticRankedTensorType(res)) {
        return mlir::failure();
      }
    }
    auto in_ty = GetStaticRankedTensorType(reduce_op.getInputs()[0]);
    if (!in_ty) {
      return mlir::failure();
    }
    llvm::ArrayRef<int64_t> dims = reduce_op.getDimensions();
    if (dims.empty()) {
      return mlir::failure();
    }
    for (int64_t d : dims) {
      if (d < 0 || d >= in_ty.getRank()) {
        return mlir::failure();
      }
    }
    llvm::DenseSet<int64_t> red_dims(dims.begin(), dims.end());
    llvm::SmallVector<int64_t> unit_unreduced;
    for (int64_t a = 0; a < in_ty.getRank(); ++a) {
      if (!red_dims.contains(a) && in_ty.getDimSize(a) == 1) {
        unit_unreduced.push_back(a);
      }
    }
    if (unit_unreduced.empty()) {
      return mlir::failure();
    }
    if (!IsIdentityReduction(reduce_op)) {
      return mlir::failure();
    }

    llvm::SmallVector<int64_t> new_dims(dims.begin(), dims.end());
    new_dims.append(unit_unreduced.begin(), unit_unreduced.end());
    llvm::sort(new_dims);
    llvm::DenseSet<int64_t> new_dims_set(new_dims.begin(), new_dims.end());

    llvm::SmallVector<int64_t> new_shape;
    for (int64_t a = 0; a < in_ty.getRank(); ++a) {
      if (!new_dims_set.contains(a)) {
        new_shape.push_back(in_ty.getDimSize(a));
      }
    }

    llvm::SmallVector<mlir::Type> new_result_types;
    for (size_t i = 0; i < reduce_op.getNumResults(); ++i) {
      auto old_res_ty =
          mlir::cast<mlir::RankedTensorType>(reduce_op.getResult(i).getType());
      new_result_types.push_back(
          mlir::RankedTensorType::get(new_shape, old_res_ty.getElementType()));
    }

    mlir::Location loc = reduce_op.getLoc();
    auto new_reduce = rewriter.create<mlir::stablehlo::ReduceOp>(
        loc, new_result_types, reduce_op.getInputs(), reduce_op.getInitValues(),
        rewriter.getDenseI64ArrayAttr(new_dims));

    mlir::IRMapping mapping;
    reduce_op.getBody().cloneInto(&new_reduce.getBody(), mapping);

    llvm::SmallVector<mlir::Value> replacements;
    for (size_t i = 0; i < reduce_op.getNumResults(); ++i) {
      replacements.push_back(rewriter.create<mlir::stablehlo::ReshapeOp>(
          loc, reduce_op.getResult(i).getType(), new_reduce.getResult(i)));
    }
    rewriter.replaceOp(reduce_op, replacements);
    return mlir::success();
  }
};

// -----------------------------------------------------------------------------
// Mechanism 4: Fold redundant elementwise reshape roundtrips
// -----------------------------------------------------------------------------

[[nodiscard]] bool IsShapeAgnosticElementwiseOp(mlir::Operation* op) {
  return mlir::isa<
      mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
      mlir::stablehlo::MulOp, mlir::stablehlo::DivOp, mlir::stablehlo::NegOp,
      mlir::stablehlo::AbsOp, mlir::stablehlo::ConvertOp,
      mlir::stablehlo::TanhOp, mlir::stablehlo::ExpOp, mlir::stablehlo::LogOp,
      mlir::stablehlo::SqrtOp, mlir::stablehlo::LogisticOp, mlir::chlo::ErfOp>(
      op);
}

[[nodiscard]] bool IsSplatOrScalarBroadcastOfShape(
    mlir::Operation* op, llvm::ArrayRef<int64_t> inner_shape) {
  if (auto const_op = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    return dense && dense.isSplat();
  }
  if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
    auto in_ty = GetStaticRankedTensorType(bcast.getOperand());
    auto out_ty = GetStaticRankedTensorType(bcast.getResult());
    return in_ty && out_ty && in_ty.getNumElements() == 1 &&
           out_ty.getShape() == inner_shape;
  }
  return false;
}

[[nodiscard]] bool ProcessReshapeInComponent(
    mlir::stablehlo::ReshapeOp resh, llvm::ArrayRef<int64_t> outer_shape,
    llvm::ArrayRef<int64_t> inner_shape,
    llvm::DenseSet<mlir::Operation*>& entry_reshapes,
    llvm::DenseSet<mlir::Operation*>& exit_reshapes,
    llvm::SmallVectorImpl<mlir::Operation*>& worklist) {
  auto in_ty = GetStaticRankedTensorType(resh.getOperand());
  auto out_ty = GetStaticRankedTensorType(resh.getResult());
  if (!in_ty || !out_ty) {
    return false;
  }
  if (in_ty.getShape() == outer_shape && out_ty.getShape() == inner_shape) {
    entry_reshapes.insert(resh.getOperation());
    for (mlir::Operation* user : resh.getResult().getUsers()) {
      worklist.push_back(user);
    }
    return true;
  }
  if (in_ty.getShape() == inner_shape && out_ty.getShape() == outer_shape) {
    exit_reshapes.insert(resh.getOperation());
    if (mlir::Operation* def = resh.getOperand().getDefiningOp()) {
      worklist.push_back(def);
    }
    return true;
  }
  return false;
}

[[nodiscard]] bool ProcessElementwiseInComponent(
    mlir::Operation* op, llvm::ArrayRef<int64_t> inner_shape,
    llvm::SmallVectorImpl<mlir::Operation*>& worklist) {
  auto res_ty = GetStaticRankedTensorType(op->getResult(0));
  if (!res_ty || res_ty.getShape() != inner_shape) {
    return false;
  }
  for (mlir::Value opnd : op->getOperands()) {
    if (mlir::Operation* def = opnd.getDefiningOp()) {
      worklist.push_back(def);
    } else {
      return false;
    }
  }
  for (mlir::Operation* user : op->getResult(0).getUsers()) {
    worklist.push_back(user);
  }
  return true;
}

[[nodiscard]] bool ProcessComponentOperation(
    mlir::Operation* cur, mlir::Block& block,
    llvm::ArrayRef<int64_t> outer_shape, llvm::ArrayRef<int64_t> inner_shape,
    llvm::DenseSet<mlir::Operation*>& entry_reshapes,
    llvm::DenseSet<mlir::Operation*>& exit_reshapes,
    llvm::SmallVectorImpl<mlir::Operation*>& worklist, bool& has_elementwise) {
  if (cur->getBlock() != &block || cur->getNumResults() != 1) {
    return false;
  }
  if (auto resh = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(cur)) {
    return ProcessReshapeInComponent(resh, outer_shape, inner_shape,
                                     entry_reshapes, exit_reshapes, worklist);
  }
  if (IsSplatOrScalarBroadcastOfShape(cur, inner_shape)) {
    return true;
  }
  if (IsShapeAgnosticElementwiseOp(cur)) {
    if (!ProcessElementwiseInComponent(cur, inner_shape, worklist)) {
      return false;
    }
    has_elementwise = true;
    return true;
  }
  return false;
}

[[nodiscard]] bool CollectReshapeBoundedComponent(
    mlir::Block& block, mlir::stablehlo::ReshapeOp seed_entry,
    llvm::ArrayRef<int64_t> outer_shape, llvm::ArrayRef<int64_t> inner_shape,
    llvm::DenseSet<mlir::Operation*>& comp_ops,
    llvm::DenseSet<mlir::Operation*>& entry_reshapes,
    llvm::DenseSet<mlir::Operation*>& exit_reshapes) {
  llvm::SmallVector<mlir::Operation*, 16> worklist = {
      seed_entry.getOperation()};
  bool has_elementwise = false;

  while (!worklist.empty()) {
    mlir::Operation* cur = worklist.pop_back_val();
    if (!comp_ops.insert(cur).second) {
      continue;
    }
    if (!ProcessComponentOperation(cur, block, outer_shape, inner_shape,
                                   entry_reshapes, exit_reshapes, worklist,
                                   has_elementwise)) {
      return false;
    }
  }
  return has_elementwise;
}

void CloneOpAtOuterRank(mlir::Operation* op,
                        llvm::ArrayRef<int64_t> outer_shape,
                        llvm::DenseMap<mlir::Value, mlir::Value>& rewritten) {
  mlir::OpBuilder builder(op);
  mlir::Location loc = op->getLoc();
  auto old_res_ty = GetStaticRankedTensorType(op->getResult(0));
  auto new_res_ty =
      mlir::RankedTensorType::get(outer_shape, old_res_ty.getElementType());

  if (auto const_op = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
    auto dense = mlir::cast<mlir::DenseElementsAttr>(const_op.getValue());
    auto new_attr = mlir::DenseElementsAttr::get(
        new_res_ty, dense.getSplatValue<mlir::Attribute>());
    mlir::Value new_c =
        builder.create<mlir::stablehlo::ConstantOp>(loc, new_res_ty, new_attr);
    rewritten[const_op.getResult()] = new_c;
    return;
  }
  if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
    mlir::Value bcast_opnd = bcast.getOperand();
    auto bcast_opnd_ty = GetStaticRankedTensorType(bcast_opnd);
    if (bcast_opnd_ty && bcast_opnd_ty.getRank() > 0) {
      auto scalar_ty =
          mlir::RankedTensorType::get({}, bcast_opnd_ty.getElementType());
      bcast_opnd = builder.create<mlir::stablehlo::ReshapeOp>(loc, scalar_ty,
                                                              bcast_opnd);
    }
    mlir::Value new_b = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, new_res_ty, bcast_opnd, builder.getDenseI64ArrayAttr({}));
    rewritten[bcast.getResult()] = new_b;
    return;
  }
  mlir::IRMapping mapping;
  for (mlir::Value opnd : op->getOperands()) {
    mapping.map(opnd, rewritten.lookup(opnd));
  }
  mlir::Operation* cloned = builder.clone(*op, mapping);
  cloned->getResult(0).setType(new_res_ty);
  rewritten[op->getResult(0)] = cloned->getResult(0);
}

void RewriteComponentAtOuterRank(
    mlir::Block& block, llvm::ArrayRef<int64_t> outer_shape,
    const llvm::DenseSet<mlir::Operation*>& comp_ops,
    const llvm::DenseSet<mlir::Operation*>& entry_reshapes,
    const llvm::DenseSet<mlir::Operation*>& exit_reshapes) {
  llvm::SmallVector<mlir::Operation*, 16> topo_ops;
  for (mlir::Operation& op : block) {
    if (comp_ops.contains(&op)) {
      topo_ops.push_back(&op);
    }
  }

  llvm::DenseMap<mlir::Value, mlir::Value> rewritten;
  for (mlir::Operation* op : topo_ops) {
    if (entry_reshapes.contains(op)) {
      auto resh = mlir::cast<mlir::stablehlo::ReshapeOp>(op);
      rewritten[resh.getResult()] = resh.getOperand();
      continue;
    }
    if (exit_reshapes.contains(op)) {
      continue;
    }
    CloneOpAtOuterRank(op, outer_shape, rewritten);
  }

  for (mlir::Operation* op : topo_ops) {
    if (exit_reshapes.contains(op)) {
      auto resh = mlir::cast<mlir::stablehlo::ReshapeOp>(op);
      mlir::Value repl = rewritten.lookup(resh.getOperand());
      resh.getResult().replaceAllUsesWith(repl);
    }
  }

  for (mlir::Operation* op : llvm::reverse(topo_ops)) {
    if (op->use_empty()) {
      op->erase();
    }
  }
}

[[nodiscard]] bool HasExitFeedingReduce(
    const llvm::DenseSet<mlir::Operation*>& exit_reshapes) {
  for (mlir::Operation* op : exit_reshapes) {
    for (mlir::Operation* user : op->getResult(0).getUsers()) {
      if (mlir::isa<mlir::stablehlo::ReduceOp>(user)) {
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] llvm::SmallVector<mlir::stablehlo::ReshapeOp, 16>
FindCandidateReshapes(mlir::Block& block) {
  llvm::SmallVector<mlir::stablehlo::ReshapeOp, 16> candidates;
  for (mlir::Operation& op : block) {
    if (auto resh = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(&op)) {
      auto in_ty = GetStaticRankedTensorType(resh.getOperand());
      auto out_ty = GetStaticRankedTensorType(resh.getResult());
      if (in_ty && out_ty && in_ty.getRank() < out_ty.getRank()) {
        candidates.push_back(resh);
      }
    }
  }
  return candidates;
}

void TryFoldReshapeBoundedComponentAtSeed(
    mlir::Block& block, mlir::stablehlo::ReshapeOp seed_entry,
    llvm::DenseSet<mlir::Operation*>& processed_entries) {
  if (!seed_entry || processed_entries.contains(seed_entry.getOperation())) {
    return;
  }
  auto outer_sample_ty = GetStaticRankedTensorType(seed_entry.getOperand());
  auto inner_sample_ty = GetStaticRankedTensorType(seed_entry.getResult());
  if (!outer_sample_ty || !inner_sample_ty) {
    return;
  }
  llvm::ArrayRef<int64_t> outer_shape = outer_sample_ty.getShape();
  llvm::ArrayRef<int64_t> inner_shape = inner_sample_ty.getShape();

  llvm::DenseSet<mlir::Operation*> comp_ops;
  llvm::DenseSet<mlir::Operation*> entry_reshapes;
  llvm::DenseSet<mlir::Operation*> exit_reshapes;
  const bool has_elementwise = CollectReshapeBoundedComponent(
      block, seed_entry, outer_shape, inner_shape, comp_ops, entry_reshapes,
      exit_reshapes);

  for (mlir::Operation* e : entry_reshapes) {
    processed_entries.insert(e);
  }
  if (!has_elementwise || exit_reshapes.empty() ||
      !HasExitFeedingReduce(exit_reshapes)) {
    return;
  }

  RewriteComponentAtOuterRank(block, outer_shape, comp_ops, entry_reshapes,
                              exit_reshapes);
}

void FoldRedundantElementwiseReshapeRoundtrips(mlir::func::FuncOp func) {
  llvm::DenseSet<mlir::Operation*> processed_entries;

  for (mlir::Block& block : func.getBody()) {
    llvm::SmallVector<mlir::stablehlo::ReshapeOp, 16> candidate_entries =
        FindCandidateReshapes(block);

    for (mlir::stablehlo::ReshapeOp seed_entry : candidate_entries) {
      TryFoldReshapeBoundedComponentAtSeed(block, seed_entry,
                                           processed_entries);
    }
  }
}

// -----------------------------------------------------------------------------
// CancelScaleInDiv: ((X * c) * Y) / c -> X * Y and (X * c) / c -> X
// -----------------------------------------------------------------------------

struct CancelScaleInDivPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::DivOp> {
  using OpRewritePattern<mlir::stablehlo::DivOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::DivOp div_op,
      mlir::PatternRewriter& rewriter) const override {
    const auto tensor_ty = GetStaticFloatTensorType(div_op.getResult());
    if (!tensor_ty) {
      return mlir::failure();
    }
    auto num_mul = div_op.getLhs().getDefiningOp<mlir::stablehlo::MulOp>();
    if (!num_mul || !num_mul.getResult().hasOneUse()) {
      return mlir::failure();
    }
    const mlir::Value denom = div_op.getRhs();

    // Direct case: (X * c) / c -> X
    if (SameNonZeroSplatConstant(num_mul.getRhs(), denom)) {
      rewriter.replaceOp(div_op, num_mul.getLhs());
      return mlir::success();
    }
    if (SameNonZeroSplatConstant(num_mul.getLhs(), denom)) {
      rewriter.replaceOp(div_op, num_mul.getRhs());
      return mlir::success();
    }

    // Nested case: ((X * c) * Y) / c -> X * Y or (Y * (X * c)) / c -> Y * X
    mlir::Value x = nullptr;
    mlir::Value y = nullptr;

    if (auto lhs_mul = num_mul.getLhs().getDefiningOp<mlir::stablehlo::MulOp>();
        lhs_mul && lhs_mul.getResult().hasOneUse()) {
      if (SameNonZeroSplatConstant(lhs_mul.getRhs(), denom)) {
        x = lhs_mul.getLhs();
        y = num_mul.getRhs();
      } else if (SameNonZeroSplatConstant(lhs_mul.getLhs(), denom)) {
        x = lhs_mul.getRhs();
        y = num_mul.getRhs();
      }
    }

    if (!x) {
      if (auto rhs_mul =
              num_mul.getRhs().getDefiningOp<mlir::stablehlo::MulOp>();
          rhs_mul && rhs_mul.getResult().hasOneUse()) {
        if (SameNonZeroSplatConstant(rhs_mul.getRhs(), denom)) {
          x = num_mul.getLhs();
          y = rhs_mul.getLhs();
        } else if (SameNonZeroSplatConstant(rhs_mul.getLhs(), denom)) {
          x = num_mul.getLhs();
          y = rhs_mul.getRhs();
        }
      }
    }

    if (!x || !y) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::stablehlo::MulOp>(div_op,
                                                        div_op.getType(), x, y);
    return mlir::success();
  }
};

// -----------------------------------------------------------------------------
// Pass definition
// -----------------------------------------------------------------------------

struct CanonicalizeReductionsPass
    : public mlir::PassWrapper<CanonicalizeReductionsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeReductionsPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-canonicalize-reductions";
  }
  llvm::StringRef getDescription() const override {
    return "Canonicalizes StableHLO reductions: hoists monotonic scalar "
           "scale/convert chains from reduce_max, folds unit unreduced axes "
           "into variadic reductions, folds redundant elementwise reshape "
           "roundtrips, and cancels scale multiplications in division.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    FoldRedundantElementwiseReshapeRoundtrips(func);

    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<FoldUnitUnreducedAxesPattern>(context);
    patterns.add<HoistMonotonicReduceChainPattern>(context);
    patterns.add<CancelScaleInDivPattern>(context);

    mlir::GreedyRewriteConfig config;
    config.setMaxIterations(10);
    if (mlir::failed(
            mlir::applyPatternsGreedily(func, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCanonicalizeReductionsPass() {
  return std::make_unique<CanonicalizeReductionsPass>();
}

}  // namespace torch_tpu
