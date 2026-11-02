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

#include "csrc/mlir/passes/canonicalize_reductions/canonicalize_reductions.h"  // NOLINT(whitespace/line_length)

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // NOLINT(whitespace/line_length)
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"  // NOLINT(whitespace/line_length)
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"  // NOLINT(whitespace/line_length)
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // NOLINT(whitespace/line_length)
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
  }
  const double v = ap_val.convertToDouble();
  if (!std::isfinite(v)) {
    return std::nullopt;
  }
  return v;
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

[[nodiscard]] bool IsFiniteNonZeroFloat(double value,
                                        mlir::FloatType float_type) {
  if (value == 0.0) {
    return false;
  }
  llvm::APFloat ap_val(value);
  bool loses_info = false;
  ap_val.convert(float_type.getFloatSemantics(),
                 llvm::APFloat::rmNearestTiesToEven, &loses_info);
  return ap_val.isFinite() && !ap_val.isZero();
}

[[nodiscard]] bool IsSplatConstantValue(mlir::Value val, double expected_val) {
  std::optional<double> c = ExtractSplatFloatConstant(val);
  return c.has_value() && *c == expected_val;
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
  if (ExtractSplatFloatConstant(val).has_value()) {
    return true;
  }
  auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast || !GetStaticRankedTensorType(bcast.getOperand())) {
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
      llvm::equal(new_dims, llvm::seq<int64_t>(0, reduced_type.getRank()))) {
    return bcast_in;
  }
  return builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, reduced_type, bcast_in, builder.getDenseI64ArrayAttr(new_dims));
}

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
  const bool is_largest = ap.isLargest();
  if (kind == ReduceKind::kMax) {
    if (!is_neg || (!is_inf && !is_largest)) {
      return std::nullopt;
    }
  } else {
    if (is_neg || (!is_inf && !is_largest)) {
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
  kTanh,
  kExp,
  kLogistic,
  kErf,
  kMulPosConst,
  kDivPosConst,
  kAddOnLhs,
  kAddOnRhs,
  kSubOnLhs,
};

struct MonotonicStep {
  StepKind kind;
  mlir::Operation* op;
  mlir::Value input_val;
  mlir::Type result_elem_type;
  mlir::Value invariant_operand;
  double pos_const_val = 0.0;
};

[[nodiscard]] bool IsNonlinearStep(StepKind kind) {
  switch (kind) {
    case StepKind::kTanh:
    case StepKind::kExp:
    case StepKind::kLogistic:
    case StepKind::kErf:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] std::optional<MonotonicStep> MatchConvertMonotonicStep(
    mlir::Operation* def, mlir::RankedTensorType in_type,
    mlir::RankedTensorType cur_ty) {
  auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(def);
  if (!conv) {
    return std::nullopt;
  }
  auto src_ty = GetStaticFloatTensorType(conv.getOperand());
  if (!src_ty || src_ty.getShape() != in_type.getShape()) {
    return std::nullopt;
  }
  auto src_fl = mlir::cast<mlir::FloatType>(src_ty.getElementType());
  auto dst_fl = mlir::cast<mlir::FloatType>(cur_ty.getElementType());
  // Strictly widening converts into at least 32-bit float are strictly
  // order-preserving under IEEE-754 semantics. Narrowing or equal-width
  // converts are excluded to preserve intermediate precision.
  const bool is_widening =
      (src_fl.getWidth() < dst_fl.getWidth() && dst_fl.getWidth() >= 32);
  if (!is_widening) {
    return std::nullopt;
  }
  return MonotonicStep{StepKind::kConvert,      conv,    conv.getOperand(),
                       cur_ty.getElementType(), nullptr, 0.0};
}

[[nodiscard]] std::optional<MonotonicStep> MatchUnaryMonotonicStep(
    mlir::Operation* def, mlir::Type elem_type) {
  if (mlir::isa<mlir::stablehlo::TanhOp>(def)) {
    return MonotonicStep{StepKind::kTanh, def,     def->getOperand(0),
                         elem_type,       nullptr, 0.0};
  }
  if (mlir::isa<mlir::stablehlo::ExpOp>(def)) {
    return MonotonicStep{StepKind::kExp, def,     def->getOperand(0),
                         elem_type,      nullptr, 0.0};
  }
  if (mlir::isa<mlir::stablehlo::LogisticOp>(def)) {
    return MonotonicStep{StepKind::kLogistic, def,     def->getOperand(0),
                         elem_type,           nullptr, 0.0};
  }
  if (mlir::isa<mlir::chlo::ErfOp>(def)) {
    return MonotonicStep{StepKind::kErf, def,     def->getOperand(0),
                         elem_type,      nullptr, 0.0};
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MonotonicStep> MatchScaleMonotonicStep(
    mlir::Operation* def, mlir::Type elem_type) {
  if (auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(def)) {
    if (std::optional<double> c = ExtractSplatFloatConstant(mul.getRhs());
        c.has_value() && *c > 0.0) {
      return MonotonicStep{
          StepKind::kMulPosConst, mul, mul.getLhs(), elem_type, nullptr, *c};
    }
    if (std::optional<double> c = ExtractSplatFloatConstant(mul.getLhs());
        c.has_value() && *c > 0.0) {
      return MonotonicStep{
          StepKind::kMulPosConst, mul, mul.getRhs(), elem_type, nullptr, *c};
    }
  }
  if (auto div = mlir::dyn_cast<mlir::stablehlo::DivOp>(def)) {
    if (std::optional<double> c = ExtractSplatFloatConstant(div.getRhs());
        c.has_value() && *c > 0.0) {
      return MonotonicStep{
          StepKind::kDivPosConst, div, div.getLhs(), elem_type, nullptr, *c};
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MonotonicStep> MatchShiftMonotonicStep(
    mlir::Operation* def, mlir::Type elem_type,
    const llvm::DenseSet<int64_t>& red_dims) {
  if (auto add = mlir::dyn_cast<mlir::stablehlo::AddOp>(def)) {
    if (IsReductionInvariantOperand(add.getRhs(), red_dims)) {
      return MonotonicStep{StepKind::kAddOnLhs, add,          add.getLhs(),
                           elem_type,           add.getRhs(), 0.0};
    }
    if (IsReductionInvariantOperand(add.getLhs(), red_dims)) {
      return MonotonicStep{StepKind::kAddOnRhs, add,          add.getRhs(),
                           elem_type,           add.getLhs(), 0.0};
    }
  }
  if (auto sub = mlir::dyn_cast<mlir::stablehlo::SubtractOp>(def)) {
    if (IsReductionInvariantOperand(sub.getRhs(), red_dims)) {
      return MonotonicStep{StepKind::kSubOnLhs, sub,          sub.getLhs(),
                           elem_type,           sub.getRhs(), 0.0};
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<MonotonicStep> MatchSingleMonotonicStep(
    mlir::Operation* def, mlir::RankedTensorType in_type,
    mlir::RankedTensorType cur_ty, const llvm::DenseSet<int64_t>& red_dims) {
  if (auto step = MatchConvertMonotonicStep(def, in_type, cur_ty)) {
    return step;
  }
  if (auto step = MatchUnaryMonotonicStep(def, cur_ty.getElementType())) {
    return step;
  }
  if (auto step = MatchScaleMonotonicStep(def, cur_ty.getElementType())) {
    return step;
  }
  return MatchShiftMonotonicStep(def, cur_ty.getElementType(), red_dims);
}

void EmitSingleHoistedStep(mlir::PatternRewriter& rewriter,
                           mlir::RankedTensorType out_type,
                           const MonotonicStep& step,
                           const llvm::DenseSet<int64_t>& red_dims,
                           mlir::Value& cur_red) {
  auto step_red_ty =
      mlir::RankedTensorType::get(out_type.getShape(), step.result_elem_type);
  mlir::Location loc = step.op->getLoc();
  switch (step.kind) {
    case StepKind::kConvert:
      cur_red = rewriter.create<mlir::stablehlo::ConvertOp>(loc, step_red_ty,
                                                            cur_red);
      break;
    case StepKind::kTanh:
      cur_red =
          rewriter.create<mlir::stablehlo::TanhOp>(loc, step_red_ty, cur_red);
      break;
    case StepKind::kExp:
      cur_red =
          rewriter.create<mlir::stablehlo::ExpOp>(loc, step_red_ty, cur_red);
      break;
    case StepKind::kLogistic:
      cur_red = rewriter.create<mlir::stablehlo::LogisticOp>(loc, step_red_ty,
                                                             cur_red);
      break;
    case StepKind::kErf:
      cur_red = rewriter.create<mlir::chlo::ErfOp>(loc, step_red_ty, cur_red);
      break;
    case StepKind::kMulPosConst: {
      mlir::Value c_red = CreateSplatFloatConstant(rewriter, loc, step_red_ty,
                                                   step.pos_const_val);
      cur_red = rewriter.create<mlir::stablehlo::MulOp>(loc, step_red_ty,
                                                        cur_red, c_red);
      break;
    }
    case StepKind::kDivPosConst: {
      mlir::Value c_red = CreateSplatFloatConstant(rewriter, loc, step_red_ty,
                                                   step.pos_const_val);
      cur_red = rewriter.create<mlir::stablehlo::DivOp>(loc, step_red_ty,
                                                        cur_red, c_red);
      break;
    }
    case StepKind::kAddOnLhs:
    case StepKind::kAddOnRhs: {
      mlir::Value inv_red = ShiftBroadcastDims(
          rewriter, loc, step.invariant_operand, step_red_ty, red_dims);
      if (!inv_red) {
        return;
      }
      cur_red = rewriter.create<mlir::stablehlo::AddOp>(loc, step_red_ty,
                                                        cur_red, inv_red);
      break;
    }
    case StepKind::kSubOnLhs: {
      mlir::Value inv_red = ShiftBroadcastDims(
          rewriter, loc, step.invariant_operand, step_red_ty, red_dims);
      if (!inv_red) {
        return;
      }
      cur_red = rewriter.create<mlir::stablehlo::SubtractOp>(loc, step_red_ty,
                                                             cur_red, inv_red);
      break;
    }
  }
}

// -----------------------------------------------------------------------------
// Mechanism 1: Hoist monotonic elementwise chains from reduce_max / reduce_min
// -----------------------------------------------------------------------------

[[nodiscard]] bool IsStepWideningConvert(const MonotonicStep& step,
                                         mlir::RankedTensorType cur_ty) {
  if (step.kind != StepKind::kConvert) {
    return false;
  }
  auto src_ty = GetStaticFloatTensorType(step.input_val);
  if (!src_ty) {
    return false;
  }
  auto src_fl = mlir::cast<mlir::FloatType>(src_ty.getElementType());
  auto dst_fl = mlir::cast<mlir::FloatType>(cur_ty.getElementType());
  return src_fl.getWidth() < dst_fl.getWidth();
}

[[nodiscard]] bool IsNextOpNonlinear(mlir::Value input_val) {
  mlir::Operation* def = input_val.getDefiningOp();
  return def != nullptr &&
         mlir::isa<mlir::stablehlo::TanhOp, mlir::stablehlo::ExpOp,
                   mlir::stablehlo::LogisticOp, mlir::chlo::ErfOp>(def);
}

[[nodiscard]] std::optional<mlir::Value> CollectMonotonicStepChain(
    mlir::Value input, mlir::RankedTensorType in_type,
    const llvm::DenseSet<int64_t>& red_dims,
    llvm::SmallVectorImpl<MonotonicStep>& rev_steps) {
  mlir::Value cur = input;
  bool all_single_use = true;
  bool seen_non_convert = false;
  bool seen_nonlinear = false;
  bool seen_widening_convert = false;
  size_t best_prefix_len = 0;

  while (mlir::Operation* def = cur.getDefiningOp()) {
    auto cur_ty = GetStaticFloatTensorType(cur);
    if (!cur_ty || cur_ty.getShape() != in_type.getShape()) {
      break;
    }
    const bool step_single_use = (cur == input) || cur.hasOneUse();
    std::optional<MonotonicStep> step =
        MatchSingleMonotonicStep(def, in_type, cur_ty, red_dims);
    if (!step.has_value()) {
      break;
    }

    rev_steps.push_back(*step);
    all_single_use = all_single_use && step_single_use;
    if (step->kind != StepKind::kConvert) {
      seen_non_convert = true;
    }
    if (IsNonlinearStep(step->kind)) {
      seen_nonlinear = true;
    }
    if (IsStepWideningConvert(*step, cur_ty)) {
      seen_widening_convert = true;
    }
    const bool next_is_nonlinear = IsNextOpNonlinear(step->input_val);
    // Profitability heuristic: hoisting purely linear ops (such as mul/div by
    // positive constant) past a reduction across dimensions does not save FLOPs
    // unless combined with a widening convert or nonlinear activation where
    // evaluation at a smaller rank yields significant compute reduction.
    if (all_single_use && seen_non_convert &&
        (input.hasOneUse() ||
         (seen_widening_convert && (seen_nonlinear || next_is_nonlinear)))) {
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
  const bool can_use_full_chain =
      all_single_use && (input.hasOneUse() || seen_widening_convert);
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
  auto base_fl = mlir::cast<mlir::FloatType>(base_ty.getElementType());
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
  mlir::Value comb_res =
      (red_kind == ReduceKind::kMax)
          ? rewriter
                .create<mlir::stablehlo::MaxOp>(loc, body->getArgument(0),
                                                body->getArgument(1))
                .getResult()
          : rewriter
                .create<mlir::stablehlo::MinOp>(loc, body->getArgument(0),
                                                body->getArgument(1))
                .getResult();
  rewriter.create<mlir::stablehlo::ReturnOp>(loc, mlir::ValueRange{comb_res});
  rewriter.setInsertionPointAfter(new_reduce);
  return new_reduce.getResult(0);
}

struct HoistMonotonicReduceChainPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern<mlir::stablehlo::ReduceOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReduceOp reduce_op,
      mlir::PatternRewriter& rewriter) const override {
    auto match = MatchMaxOrMinReduce(reduce_op);
    if (!match.has_value()) {
      return mlir::failure();
    }
    const auto [red_kind, init_is_inf] = *match;
    mlir::Value input = reduce_op.getInputs()[0];
    auto in_type = GetStaticFloatTensorType(input);
    auto out_type = GetStaticFloatTensorType(reduce_op.getResult(0));
    if (!in_type || !out_type) {
      return mlir::failure();
    }
    llvm::DenseSet<int64_t> red_dims;
    for (const int64_t d : reduce_op.getDimensions()) {
      if (in_type.getDimSize(d) <= 1) {
        return mlir::failure();
      }
      red_dims.insert(d);
    }
    if (red_dims.empty()) {
      return mlir::failure();
    }

    llvm::SmallVector<MonotonicStep, 8> rev_steps;
    std::optional<mlir::Value> base_input =
        CollectMonotonicStepChain(input, in_type, red_dims, rev_steps);
    if (!base_input.has_value()) {
      return mlir::failure();
    }

    mlir::Location loc = reduce_op.getLoc();
    mlir::Value cur_red = CreateHoistedReduceOp(
        rewriter, loc, *base_input, out_type, reduce_op.getDimensionsAttr(),
        red_kind, init_is_inf);

    for (const MonotonicStep& step : llvm::reverse(rev_steps)) {
      EmitSingleHoistedStep(rewriter, out_type, step, red_dims, cur_red);
    }

    rewriter.replaceOp(reduce_op, cur_red);
    return mlir::success();
  }
};

// -----------------------------------------------------------------------------
// Mechanism 2: Fuse chained additive reductions
// -----------------------------------------------------------------------------

[[nodiscard]] bool IsAdditiveSingleFloatReduce(
    mlir::stablehlo::ReduceOp reduce_op) {
  if (reduce_op.getDimensions().empty() || reduce_op.getInputs().size() != 1 ||
      reduce_op.getInitValues().size() != 1 || reduce_op.getNumResults() != 1) {
    return false;
  }
  auto in_ty = GetStaticFloatTensorType(reduce_op.getInputs()[0]);
  if (!in_ty) {
    return false;
  }
  auto fl_ty = mlir::cast<mlir::FloatType>(in_ty.getElementType());
  // Accumulation width must be at least 32 bits (f32 or f64). F16/BF16
  // mean-of-means accumulates catastrophic rounding error if chained divisions
  // are merged into a single multi-reduction without promoting accumulation.
  if (fl_ty.getWidth() < 32) {
    return false;
  }
  mlir::Region& region = reduce_op.getBody();
  if (!region.hasOneBlock()) {
    return false;
  }
  mlir::Block& body = region.front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2) {
    return false;
  }
  auto ret = mlir::dyn_cast<mlir::stablehlo::ReturnOp>(body.getTerminator());
  if (!ret || ret.getNumOperands() != 1) {
    return false;
  }
  mlir::Operation* bin = &body.front();
  if (!mlir::isa<mlir::stablehlo::AddOp>(bin) ||
      bin->getResult(0) != ret.getOperand(0)) {
    return false;
  }
  mlir::Value a0 = body.getArgument(0);
  mlir::Value a1 = body.getArgument(1);
  if (!((bin->getOperand(0) == a0 && bin->getOperand(1) == a1) ||
        (bin->getOperand(0) == a1 && bin->getOperand(1) == a0))) {
    return false;
  }
  std::optional<double> init_val =
      ExtractSplatFloatConstant(reduce_op.getInitValues()[0]);
  return init_val.has_value() && *init_val == 0.0;
}

[[nodiscard]] bool HasKeepDimReducedShape(
    mlir::RankedTensorType x_type, const llvm::DenseSet<int64_t>& dims1_set,
    mlir::RankedTensorType r2_in_type) {
  if (x_type.getRank() != r2_in_type.getRank()) {
    return false;
  }
  for (int64_t a = 0; a < x_type.getRank(); ++a) {
    const int64_t expected = dims1_set.contains(a) ? 1 : x_type.getDimSize(a);
    if (r2_in_type.getDimSize(a) != expected) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CollectDropDimSecondReductionAxes(
    llvm::ArrayRef<int64_t> non_reduced1, llvm::ArrayRef<int64_t> dims2,
    llvm::DenseSet<int64_t>& combined_set) {
  for (const int64_t d2 : dims2) {
    if (d2 < 0 || d2 >= static_cast<int64_t>(non_reduced1.size())) {
      return false;
    }
    combined_set.insert(non_reduced1[d2]);
  }
  return true;
}

[[nodiscard]] bool CollectKeepDimSecondReductionAxes(
    int64_t rank, llvm::ArrayRef<int64_t> dims2,
    llvm::DenseSet<int64_t>& combined_set) {
  for (const int64_t d2 : dims2) {
    if (d2 < 0 || d2 >= rank) {
      return false;
    }
    combined_set.insert(d2);
  }
  return true;
}

[[nodiscard]] std::optional<llvm::SmallVector<int64_t>> MapChainedReductionDims(
    mlir::RankedTensorType x_type, llvm::ArrayRef<int64_t> dims1,
    mlir::RankedTensorType r1_out_type, mlir::RankedTensorType r2_in_type,
    llvm::ArrayRef<int64_t> dims2) {
  llvm::DenseSet<int64_t> dims1_set(dims1.begin(), dims1.end());
  llvm::SmallVector<int64_t> non_reduced1;
  for (int64_t a = 0; a < x_type.getRank(); ++a) {
    if (!dims1_set.contains(a)) {
      non_reduced1.push_back(a);
    }
  }

  llvm::DenseSet<int64_t> combined_set(dims1.begin(), dims1.end());
  if (r2_in_type.getShape() == r1_out_type.getShape()) {
    if (!CollectDropDimSecondReductionAxes(non_reduced1, dims2, combined_set)) {
      return std::nullopt;
    }
  } else if (HasKeepDimReducedShape(x_type, dims1_set, r2_in_type)) {
    if (!CollectKeepDimSecondReductionAxes(x_type.getRank(), dims2,
                                           combined_set)) {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }

  if (combined_set.size() <= dims1_set.size()) {
    return std::nullopt;
  }
  llvm::SmallVector<int64_t> combined_dims(combined_set.begin(),
                                           combined_set.end());
  llvm::sort(combined_dims);
  for (const int64_t d : combined_dims) {
    if (x_type.getDimSize(d) <= 0) {
      return std::nullopt;
    }
  }
  return combined_dims;
}

struct ChainedReductionChain {
  mlir::stablehlo::ReduceOp reduce1;
  double div_scale = 1.0;
  double mul_scale = 1.0;
  bool has_div_scale = false;
  bool has_mul_scale = false;
};

[[nodiscard]] std::optional<mlir::Value> TryStepThroughFloatConvert(
    mlir::stablehlo::ConvertOp conv_outer, mlir::FloatType elem_type) {
  mlir::Value mid = conv_outer.getOperand();
  auto mid_ty = GetStaticFloatTensorType(mid);
  if (!mid_ty) {
    return std::nullopt;
  }
  auto mid_fl = mlir::cast<mlir::FloatType>(mid_ty.getElementType());
  if (mid_fl == elem_type) {
    return mid;
  }
  auto conv_inner = mid.getDefiningOp<mlir::stablehlo::ConvertOp>();
  if (!conv_inner || !mid.hasOneUse()) {
    return std::nullopt;
  }
  auto src_ty = GetStaticFloatTensorType(conv_inner.getOperand());
  if (!src_ty || src_ty.getElementType() != elem_type) {
    return std::nullopt;
  }
  // Middle type must be strictly narrower than elem_type AND must be bf16 or
  // f16. Stepping through non-bf16/f16 narrowing convert roundtrips (e.g.
  // f64 -> f32 -> f64) alters precision semantics.
  if (mid_fl.getWidth() >= elem_type.getWidth() ||
      !(mid_fl.isBF16() || mid_fl.isF16())) {
    return std::nullopt;
  }
  return conv_inner.getOperand();
}

[[nodiscard]] std::optional<mlir::Value> StepBackwardChainedReductionLink(
    mlir::Value cur, mlir::FloatType elem_type, ChainedReductionChain& chain) {
  if (auto reshape_op = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    return reshape_op.getOperand();
  }
  if (auto conv_outer = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    return TryStepThroughFloatConvert(conv_outer, elem_type);
  }
  if (auto div_op = cur.getDefiningOp<mlir::stablehlo::DivOp>()) {
    std::optional<double> c = ExtractSplatFloatConstant(div_op.getRhs());
    if (!c.has_value() || *c == 0.0) {
      return std::nullopt;
    }
    chain.div_scale *= *c;
    chain.has_div_scale = true;
    return div_op.getLhs();
  }
  if (auto mul_op = cur.getDefiningOp<mlir::stablehlo::MulOp>()) {
    if (std::optional<double> c = ExtractSplatFloatConstant(mul_op.getRhs());
        c.has_value() && *c != 0.0) {
      chain.mul_scale *= *c;
      chain.has_mul_scale = true;
      return mul_op.getLhs();
    }
    if (std::optional<double> c = ExtractSplatFloatConstant(mul_op.getLhs());
        c.has_value() && *c != 0.0) {
      chain.mul_scale *= *c;
      chain.has_mul_scale = true;
      return mul_op.getRhs();
    }
    return std::nullopt;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<ChainedReductionChain>
MatchChainedReductionProducer(mlir::Value r2_in, mlir::FloatType elem_type) {
  ChainedReductionChain chain;
  mlir::Value cur = r2_in;
  while (cur.hasOneUse() && !cur.getDefiningOp<mlir::stablehlo::ReduceOp>()) {
    std::optional<mlir::Value> next =
        StepBackwardChainedReductionLink(cur, elem_type, chain);
    if (!next.has_value()) {
      return std::nullopt;
    }
    cur = *next;
  }
  auto reduce1 = cur.getDefiningOp<mlir::stablehlo::ReduceOp>();
  if (!reduce1 || !reduce1.getResult(0).hasOneUse() ||
      !IsAdditiveSingleFloatReduce(reduce1)) {
    return std::nullopt;
  }
  chain.reduce1 = reduce1;
  return chain;
}

struct FuseChainedAdditiveReductionsPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern<mlir::stablehlo::ReduceOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReduceOp reduce2,
      mlir::PatternRewriter& rewriter) const override {
    if (!IsAdditiveSingleFloatReduce(reduce2)) {
      return mlir::failure();
    }
    mlir::Value r2_in = reduce2.getInputs()[0];
    auto r2_in_type = GetStaticFloatTensorType(r2_in);
    if (!r2_in_type) {
      return mlir::failure();
    }
    auto elem_type = mlir::cast<mlir::FloatType>(r2_in_type.getElementType());

    std::optional<ChainedReductionChain> chain =
        MatchChainedReductionProducer(r2_in, elem_type);
    if (!chain.has_value()) {
      return mlir::failure();
    }
    if ((chain->has_div_scale &&
         !IsFiniteNonZeroFloat(chain->div_scale, elem_type)) ||
        (chain->has_mul_scale &&
         !IsFiniteNonZeroFloat(chain->mul_scale, elem_type))) {
      return mlir::failure();
    }
    mlir::stablehlo::ReduceOp reduce1 = chain->reduce1;
    mlir::Value x = reduce1.getInputs()[0];
    auto x_type = GetStaticFloatTensorType(x);
    auto r1_out_type = GetStaticFloatTensorType(reduce1.getResult(0));
    if (!x_type || !r1_out_type || x_type.getElementType() != elem_type) {
      return mlir::failure();
    }

    std::optional<llvm::SmallVector<int64_t>> combined_dims =
        MapChainedReductionDims(x_type, reduce1.getDimensions(), r1_out_type,
                                r2_in_type, reduce2.getDimensions());
    if (!combined_dims.has_value()) {
      return mlir::failure();
    }

    llvm::DenseSet<int64_t> combined_set(combined_dims->begin(),
                                         combined_dims->end());
    llvm::SmallVector<int64_t> reduced_shape;
    for (int64_t a = 0; a < x_type.getRank(); ++a) {
      if (!combined_set.contains(a)) {
        reduced_shape.push_back(x_type.getDimSize(a));
      }
    }
    auto reduced_type = mlir::RankedTensorType::get(reduced_shape, elem_type);
    mlir::Location loc = reduce2.getLoc();

    auto new_reduce = rewriter.create<mlir::stablehlo::ReduceOp>(
        loc, mlir::TypeRange{reduced_type}, mlir::ValueRange{x},
        mlir::ValueRange{reduce1.getInitValues()[0]},
        rewriter.getDenseI64ArrayAttr(*combined_dims));

    mlir::Block* body =
        rewriter.createBlock(&new_reduce.getBody(), new_reduce.getBody().end(),
                             {mlir::RankedTensorType::get({}, elem_type),
                              mlir::RankedTensorType::get({}, elem_type)},
                             {loc, loc});
    rewriter.setInsertionPointToStart(body);
    mlir::Value add_res = rewriter.create<mlir::stablehlo::AddOp>(
        loc, body->getArgument(0), body->getArgument(1));
    rewriter.create<mlir::stablehlo::ReturnOp>(loc, mlir::ValueRange{add_res});

    rewriter.setInsertionPointAfter(new_reduce);
    mlir::Value result_val = new_reduce.getResult(0);
    auto target_type =
        mlir::cast<mlir::RankedTensorType>(reduce2.getResult(0).getType());
    if (result_val.getType() != target_type) {
      result_val = rewriter.create<mlir::stablehlo::ReshapeOp>(loc, target_type,
                                                               result_val);
    }

    mlir::Operation* replace_target = reduce2.getOperation();
    double div_scale = chain->div_scale;
    bool has_div_scale = chain->has_div_scale;
    // Check if reduce2 has a single user that is a downstream DivOp by a splat
    // constant, so we can merge chained divisions (e.g. mean-of-means:
    // /8 then /4 -> /32).
    if (reduce2.getResult(0).hasOneUse()) {
      auto user_div = mlir::dyn_cast<mlir::stablehlo::DivOp>(
          *reduce2.getResult(0).user_begin());
      if (user_div && user_div.getLhs() == reduce2.getResult(0)) {
        if (std::optional<double> c =
                ExtractSplatFloatConstant(user_div.getRhs());
            c.has_value() && IsFiniteNonZeroFloat(*c, elem_type)) {
          const double merged_scale = div_scale * (*c);
          if (IsFiniteNonZeroFloat(merged_scale, elem_type)) {
            div_scale = merged_scale;
            has_div_scale = true;
            replace_target = user_div.getOperation();
            target_type = mlir::cast<mlir::RankedTensorType>(
                replace_target->getResult(0).getType());
          }
        }
      }
    }

    // Mechanism 2 precision notice: merging scales (div/mul) changes rounding
    // behavior slightly because dividing once by (C1 * C2) is not bit-exact
    // compared to (x / C1) / C2 under IEEE-754 floating point arithmetic.
    if (has_div_scale) {
      mlir::Value div_cst =
          CreateSplatFloatConstant(rewriter, loc, target_type, div_scale);
      result_val = rewriter.create<mlir::stablehlo::DivOp>(loc, target_type,
                                                           result_val, div_cst);
    }
    if (chain->has_mul_scale) {
      mlir::Value mul_cst = CreateSplatFloatConstant(rewriter, loc, target_type,
                                                     chain->mul_scale);
      result_val = rewriter.create<mlir::stablehlo::MulOp>(loc, target_type,
                                                           result_val, mul_cst);
    }

    rewriter.replaceOp(replace_target, result_val);
    return mlir::success();
  }
};

// -----------------------------------------------------------------------------
// Mechanism 3: Canonicalize unit dimension reductions to reshape
// -----------------------------------------------------------------------------

[[nodiscard]] bool IsSingleInputIdentityReduction(mlir::Value a0,
                                                  mlir::Value a1,
                                                  mlir::Value res,
                                                  mlir::Value init_val) {
  mlir::Operation* bin = res.getDefiningOp();
  if (!bin || bin->getNumOperands() != 2) {
    return false;
  }
  if (!((bin->getOperand(0) == a0 && bin->getOperand(1) == a1) ||
        (bin->getOperand(0) == a1 && bin->getOperand(1) == a0))) {
    return false;
  }
  if (mlir::isa<mlir::stablehlo::AddOp>(bin)) {
    return IsSplatConstantValue(init_val, 0.0);
  }
  if (mlir::isa<mlir::stablehlo::MulOp>(bin)) {
    return IsSplatConstantValue(init_val, 1.0);
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

struct CanonicalizeUnitDimensionReductionsPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern<mlir::stablehlo::ReduceOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReduceOp reduce_op,
      mlir::PatternRewriter& rewriter) const override {
    llvm::ArrayRef<int64_t> dims = reduce_op.getDimensions();
    if (dims.empty()) {
      return mlir::failure();
    }
    auto in_ty = GetStaticRankedTensorType(reduce_op.getInputs()[0]);
    if (!in_ty) {
      return mlir::failure();
    }
    for (const int64_t d : dims) {
      if (d < 0 || d >= in_ty.getRank() || in_ty.getDimSize(d) != 1) {
        return mlir::failure();
      }
    }
    if (!IsIdentityReduction(reduce_op)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value> replacements;
    for (size_t i = 0; i < reduce_op.getNumResults(); ++i) {
      replacements.push_back(rewriter.create<mlir::stablehlo::ReshapeOp>(
          reduce_op.getLoc(), reduce_op.getResult(i).getType(),
          reduce_op.getInputs()[i]));
    }
    rewriter.replaceOp(reduce_op, replacements);
    return mlir::success();
  }
};

// -----------------------------------------------------------------------------
// Mechanism 5: Fold unit unreduced axes into variadic reductions
// -----------------------------------------------------------------------------

struct FoldUnitUnreducedAxesPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern<mlir::stablehlo::ReduceOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReduceOp reduce_op,
      mlir::PatternRewriter& rewriter) const override {
    if (reduce_op.getNumResults() <= 1) {
      return mlir::failure();
    }
    auto in_ty = GetStaticRankedTensorType(reduce_op.getInputs()[0]);
    if (!in_ty) {
      return mlir::failure();
    }
    llvm::ArrayRef<int64_t> dims = reduce_op.getDimensions();
    if (dims.empty()) {
      return mlir::failure();
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
  auto r_in = GetStaticRankedTensorType(resh.getOperand());
  auto r_out = GetStaticRankedTensorType(resh.getResult());
  if (r_in && r_out && r_in.getShape() == outer_shape &&
      r_out.getShape() == inner_shape) {
    entry_reshapes.insert(resh.getOperation());
    for (mlir::Operation* user : resh.getResult().getUsers()) {
      worklist.push_back(user);
    }
    return true;
  }
  if (r_in && r_out && r_in.getShape() == inner_shape &&
      r_out.getShape() == outer_shape) {
    exit_reshapes.insert(resh.getOperation());
    if (mlir::Operation* in_def = resh.getOperand().getDefiningOp()) {
      worklist.push_back(in_def);
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool ProcessElementwiseInComponent(
    mlir::Operation* cur, llvm::ArrayRef<int64_t> inner_shape,
    llvm::SmallVectorImpl<mlir::Operation*>& worklist) {
  auto res_ty = GetStaticRankedTensorType(cur->getResult(0));
  if (!res_ty || res_ty.getShape() != inner_shape) {
    return false;
  }
  for (mlir::Value opnd : cur->getOperands()) {
    auto opnd_ty = GetStaticRankedTensorType(opnd);
    mlir::Operation* opnd_def = opnd.getDefiningOp();
    if (!opnd_ty || opnd_ty.getShape() != inner_shape || !opnd_def) {
      return false;
    }
    worklist.push_back(opnd_def);
  }
  for (mlir::Operation* user : cur->getResult(0).getUsers()) {
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
    return "Canonicalizes StableHLO reductions: hoists monotonic elementwise "
           "chains from max/min reductions, fuses chained additive reductions, "
           "simplifies unit dimension reductions to reshapes, folds "
           "elementwise "
           "reshape roundtrips, and folds unit unreduced axes into variadic "
           "reductions.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    FoldRedundantElementwiseReshapeRoundtrips(func);

    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<CanonicalizeUnitDimensionReductionsPattern>(context);
    patterns.add<FoldUnitUnreducedAxesPattern>(context);
    patterns.add<HoistMonotonicReduceChainPattern>(context);
    patterns.add<FuseChainedAdditiveReductionsPattern>(context);

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
