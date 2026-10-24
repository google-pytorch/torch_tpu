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

#include "csrc/mlir/passes/hoist_monotonic_reduce_and_barrier_multi_dot_grad.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

enum class ReduceKind {
  kMax,
  kMin,
};

enum class StepKind {
  kConvert,
  kTanh,
  kExp,
  kLogistic,
  kErf,
  kAddOnLhs,
  kAddOnRhs,
  kSubOnLhs,
  kMulPosConst,
  kDivPosConst,
};

struct MonotonicStep {
  StepKind kind = StepKind::kConvert;
  mlir::Operation* op = nullptr;
  mlir::Type result_elem_type;
  mlir::Value invariant_other;
  double pos_const_val = 0.0;
};

// Returns a static ranked float tensor type if `val` has one, else nullptr.
[[nodiscard]] mlir::RankedTensorType GetStaticFloatTensorType(mlir::Value val) {
  auto type = llvm::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!type || !type.hasStaticShape() ||
      !llvm::isa<mlir::FloatType>(type.getElementType())) {
    return nullptr;
  }
  return type;
}

// Extracts a splat finite float constant from `val` looking through broadcasts,
// reshapes, and float conversions.
[[nodiscard]] std::optional<double> ExtractSplatFloatConstant(mlir::Value val) {
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
      val = convert.getOperand();
      continue;
    }
    break;
  }
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  auto dense = llvm::dyn_cast<mlir::DenseFPElementsAttr>(const_op.getValue());
  if (!dense || !dense.isSplat()) {
    return std::nullopt;
  }
  const double v = dense.getSplatValue<llvm::APFloat>().convertToDouble();
  if (!std::isfinite(v)) {
    return std::nullopt;
  }
  return v;
}

// Builds a splat float ConstantOp of `tensor_type` with value `val`.
[[nodiscard]] mlir::Value CreateSplatFloatConstant(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::RankedTensorType tensor_type, double val) {
  auto float_ty = llvm::cast<mlir::FloatType>(tensor_type.getElementType());
  llvm::APFloat ap_val(val);
  bool loses_info = false;
  ap_val.convert(float_ty.getFloatSemantics(),
                 llvm::APFloat::rmNearestTiesToEven, &loses_info);
  auto attr = mlir::DenseElementsAttr::get(tensor_type, ap_val);
  return mlir::stablehlo::ConstantOp::create(builder, loc, tensor_type, attr);
}

// Folds consecutive single-use positive splat constant MulOp/DivOp pairs on
// f32/f64 tensors: `(X * c1) / c2 -> X * (c1 / c2)`,
// `(X / c1) * c2 -> X * (c2 / c1)`, `(X * c1) * c2 -> X * (c1 * c2)`, and
// `(X / c1) / c2 -> X / (c1 * c2)`.
bool FoldConsecutiveConstantMulDiv(mlir::func::FuncOp func) {
  bool changed = false;
  for (mlir::Block& block : func.getBody()) {
    for (mlir::Operation& op : llvm::make_early_inc_range(block)) {
      const bool outer_is_mul = llvm::isa<mlir::stablehlo::MulOp>(&op);
      const bool outer_is_div = llvm::isa<mlir::stablehlo::DivOp>(&op);
      if (!outer_is_mul && !outer_is_div) {
        continue;
      }
      auto tensor_ty = GetStaticFloatTensorType(op.getResult(0));
      if (!tensor_ty) {
        continue;
      }
      auto float_ty = llvm::cast<mlir::FloatType>(tensor_ty.getElementType());
      if (float_ty.getWidth() < 32) {
        continue;
      }

      mlir::Value outer_lhs = op.getOperand(0);
      mlir::Value outer_rhs = op.getOperand(1);
      std::optional<double> c2 = ExtractSplatFloatConstant(outer_rhs);
      mlir::Value inner_val = outer_lhs;
      if (!c2.has_value() && outer_is_mul) {
        c2 = ExtractSplatFloatConstant(outer_lhs);
        inner_val = outer_rhs;
      }
      if (!c2.has_value() || *c2 <= 0.0 || !inner_val.hasOneUse()) {
        continue;
      }

      mlir::Operation* inner_op = inner_val.getDefiningOp();
      if (!inner_op) {
        continue;
      }
      const bool inner_is_mul = llvm::isa<mlir::stablehlo::MulOp>(inner_op);
      const bool inner_is_div = llvm::isa<mlir::stablehlo::DivOp>(inner_op);
      if (!inner_is_mul && !inner_is_div) {
        continue;
      }
      mlir::Value inner_lhs = inner_op->getOperand(0);
      mlir::Value inner_rhs = inner_op->getOperand(1);
      std::optional<double> c1 = ExtractSplatFloatConstant(inner_rhs);
      mlir::Value base_x = inner_lhs;
      if (!c1.has_value() && inner_is_mul) {
        c1 = ExtractSplatFloatConstant(inner_lhs);
        base_x = inner_rhs;
      }
      if (!c1.has_value() || *c1 <= 0.0) {
        continue;
      }

      double combined = 0.0;
      bool emit_div = false;
      if (inner_is_mul && outer_is_mul) {
        combined = (*c1) * (*c2);
      } else if (inner_is_mul && outer_is_div) {
        combined = (*c1) / (*c2);
      } else if (inner_is_div && outer_is_mul) {
        combined = (*c2) / (*c1);
      } else {
        combined = (*c1) * (*c2);
        emit_div = true;
      }
      if (!std::isfinite(combined) || combined <= 0.0) {
        continue;
      }

      mlir::OpBuilder builder(&op);
      mlir::Value const_tensor =
          CreateSplatFloatConstant(builder, op.getLoc(), tensor_ty, combined);
      mlir::Value repl =
          emit_div ? mlir::stablehlo::DivOp::create(
                         builder, op.getLoc(), tensor_ty, base_x, const_tensor)
                         .getResult()
                   : mlir::stablehlo::MulOp::create(
                         builder, op.getLoc(), tensor_ty, base_x, const_tensor)
                         .getResult();
      op.getResult(0).replaceAllUsesWith(repl);
      op.erase();
      if (inner_op->use_empty()) {
        inner_op->erase();
      }
      changed = true;
    }
  }
  return changed;
}

// Matches a single-input scalar MaxOp or MinOp reduction body and verifies that
// the init value is a valid identity (-inf/min for MaxOp, +inf/max for MinOp).
[[nodiscard]] std::optional<std::pair<ReduceKind, bool>> MatchMaxOrMinReduce(
    mlir::stablehlo::ReduceOp reduce_op) {
  // GENERIC: semantic single-input single-result reduction check
  if (reduce_op.getInputs().size() != 1 ||
      reduce_op.getInitValues().size() != 1 || reduce_op.getNumResults() != 1) {
    return std::nullopt;
  }
  mlir::Region& region = reduce_op.getBody();
  // GENERIC: semantic single-block 2-arg scalar reduction body
  if (!region.hasOneBlock()) {
    return std::nullopt;
  }
  mlir::Block& body = region.front();
  if (body.getNumArguments() != 2) {
    return std::nullopt;
  }
  auto ret_op = llvm::dyn_cast<mlir::stablehlo::ReturnOp>(body.getTerminator());
  if (!ret_op || ret_op.getResults().size() != 1) {
    return std::nullopt;
  }
  mlir::Operation* comb = ret_op.getResults()[0].getDefiningOp();
  if (!comb || comb->getNumOperands() != 2) {
    return std::nullopt;
  }
  mlir::Value a0 = body.getArgument(0);
  mlir::Value a1 = body.getArgument(1);
  if (!((comb->getOperand(0) == a0 && comb->getOperand(1) == a1) ||
        (comb->getOperand(0) == a1 && comb->getOperand(1) == a0))) {
    return std::nullopt;
  }

  ReduceKind kind;
  if (llvm::isa<mlir::stablehlo::MaxOp>(comb)) {
    kind = ReduceKind::kMax;
  } else if (llvm::isa<mlir::stablehlo::MinOp>(comb)) {
    kind = ReduceKind::kMin;
  } else {
    return std::nullopt;
  }

  auto init_const =
      reduce_op.getInitValues()[0].getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!init_const) {
    return std::nullopt;
  }
  auto dense = llvm::dyn_cast<mlir::DenseFPElementsAttr>(init_const.getValue());
  if (!dense || !dense.isSplat()) {
    return std::nullopt;
  }
  llvm::APFloat init_val = dense.getSplatValue<llvm::APFloat>();
  const bool is_neg = init_val.isNegative();
  const bool is_inf = init_val.isInfinity();
  const bool is_largest = init_val.isLargest();
  if (kind == ReduceKind::kMax) {
    if (!is_neg || (!is_inf && !is_largest)) {
      return std::nullopt;
    }
  } else {
    if (is_neg || (!is_inf && !is_largest)) {
      return std::nullopt;
    }
  }
  return std::make_pair(kind, is_inf);
}

// Returns true if `val` is invariant along all dimensions in `red_dims` (either
// a splat constant or a BroadcastInDimOp whose broadcast_dimensions are
// disjoint from `red_dims`).
[[nodiscard]] bool IsReductionInvariantOperand(
    mlir::Value val, const llvm::DenseSet<int64_t>& red_dims) {
  if (ExtractSplatFloatConstant(val).has_value()) {
    return true;
  }
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    for (int64_t d : bcast.getBroadcastDimensions()) {
      if (red_dims.contains(d)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

// Projects a reduction-invariant operand `val` to `reduced_type` by removing
// `red_dims`.
[[nodiscard]] mlir::Value ProjectInvariantOperandToReducedShape(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val,
    mlir::RankedTensorType reduced_type,
    const llvm::DenseSet<int64_t>& red_dims) {
  if (std::optional<double> c = ExtractSplatFloatConstant(val)) {
    return CreateSplatFloatConstant(builder, loc, reduced_type, *c);
  }
  auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  llvm::SmallVector<int64_t> new_dims;
  for (int64_t d : bcast.getBroadcastDimensions()) {
    int64_t shift = 0;
    for (int64_t r : red_dims) {
      if (r < d) {
        ++shift;
      }
    }
    new_dims.push_back(d - shift);
  }
  mlir::Value bcast_in = bcast.getOperand();
  if (bcast_in.getType() !=
      mlir::RankedTensorType::get(
          llvm::cast<mlir::RankedTensorType>(bcast_in.getType()).getShape(),
          reduced_type.getElementType())) {
    auto conv_ty = mlir::RankedTensorType::get(
        llvm::cast<mlir::RankedTensorType>(bcast_in.getType()).getShape(),
        reduced_type.getElementType());
    bcast_in =
        mlir::stablehlo::ConvertOp::create(builder, loc, conv_ty, bcast_in);
  }
  if (bcast_in.getType() == reduced_type) {
    if (llvm::equal(new_dims, llvm::seq<int64_t>(0, reduced_type.getRank()))) {
      return bcast_in;
    }
  }
  return mlir::stablehlo::BroadcastInDimOp::create(
      builder, loc, reduced_type, bcast_in,
      builder.getDenseI64ArrayAttr(new_dims));
}

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

// Hoists monotonically non-decreasing scalar/invariant-broadcast elementwise
// chains out of `reduce_op` (`MaxOp` or `MinOp`).
bool TryHoistMonotonicChainFromReduce(mlir::stablehlo::ReduceOp reduce_op) {
  auto match = MatchMaxOrMinReduce(reduce_op);
  if (!match.has_value()) {
    return false;
  }
  const auto [red_kind, init_is_inf] = *match;

  mlir::Value input = reduce_op.getInputs()[0];
  auto in_type = GetStaticFloatTensorType(input);
  auto out_type = GetStaticFloatTensorType(reduce_op.getResult(0));
  if (!in_type || !out_type) {
    return false;
  }

  llvm::DenseSet<int64_t> red_dims;
  for (int64_t d : reduce_op.getDimensions()) {
    if (in_type.getDimSize(d) <= 1) {
      return false;
    }
    red_dims.insert(d);
  }
  if (red_dims.empty()) {
    return false;
  }

  llvm::SmallVector<MonotonicStep, 8> rev_steps;
  mlir::Value cur = input;
  bool all_single_use = true;

  while (mlir::Operation* def = cur.getDefiningOp()) {
    auto cur_ty = GetStaticFloatTensorType(cur);
    if (!cur_ty || cur_ty.getShape() != in_type.getShape()) {
      break;
    }
    const bool step_single_use = cur.hasOneUse();

    if (auto conv = llvm::dyn_cast<mlir::stablehlo::ConvertOp>(def)) {
      auto src_ty = GetStaticFloatTensorType(conv.getOperand());
      if (!src_ty || src_ty.getShape() != in_type.getShape()) {
        break;
      }
      auto src_fl = llvm::cast<mlir::FloatType>(src_ty.getElementType());
      auto dst_fl = llvm::cast<mlir::FloatType>(cur_ty.getElementType());
      if (src_fl.getWidth() > dst_fl.getWidth()) {
        break;
      }
      rev_steps.push_back(
          {StepKind::kConvert, def, cur_ty.getElementType(), nullptr, 0.0});
      all_single_use = all_single_use && step_single_use;
      cur = conv.getOperand();
      continue;
    }

    if (llvm::isa<mlir::stablehlo::TanhOp>(def)) {
      rev_steps.push_back(
          {StepKind::kTanh, def, cur_ty.getElementType(), nullptr, 0.0});
      all_single_use = all_single_use && step_single_use;
      cur = def->getOperand(0);
      continue;
    }
    if (llvm::isa<mlir::stablehlo::ExpOp>(def)) {
      rev_steps.push_back(
          {StepKind::kExp, def, cur_ty.getElementType(), nullptr, 0.0});
      all_single_use = all_single_use && step_single_use;
      cur = def->getOperand(0);
      continue;
    }
    if (llvm::isa<mlir::stablehlo::LogisticOp>(def)) {
      rev_steps.push_back(
          {StepKind::kLogistic, def, cur_ty.getElementType(), nullptr, 0.0});
      all_single_use = all_single_use && step_single_use;
      cur = def->getOperand(0);
      continue;
    }
    if (llvm::isa<mlir::chlo::ErfOp>(def)) {
      rev_steps.push_back(
          {StepKind::kErf, def, cur_ty.getElementType(), nullptr, 0.0});
      all_single_use = all_single_use && step_single_use;
      cur = def->getOperand(0);
      continue;
    }

    if (auto mul = llvm::dyn_cast<mlir::stablehlo::MulOp>(def)) {
      if (std::optional<double> c = ExtractSplatFloatConstant(mul.getRhs());
          c.has_value() && *c > 0.0) {
        rev_steps.push_back({StepKind::kMulPosConst, def,
                             cur_ty.getElementType(), nullptr, *c});
        all_single_use = all_single_use && step_single_use;
        cur = mul.getLhs();
        continue;
      }
      if (std::optional<double> c = ExtractSplatFloatConstant(mul.getLhs());
          c.has_value() && *c > 0.0) {
        rev_steps.push_back({StepKind::kMulPosConst, def,
                             cur_ty.getElementType(), nullptr, *c});
        all_single_use = all_single_use && step_single_use;
        cur = mul.getRhs();
        continue;
      }
      break;
    }

    if (auto div = llvm::dyn_cast<mlir::stablehlo::DivOp>(def)) {
      if (std::optional<double> c = ExtractSplatFloatConstant(div.getRhs());
          c.has_value() && *c > 0.0) {
        rev_steps.push_back({StepKind::kDivPosConst, def,
                             cur_ty.getElementType(), nullptr, *c});
        all_single_use = all_single_use && step_single_use;
        cur = div.getLhs();
        continue;
      }
      break;
    }

    if (auto add = llvm::dyn_cast<mlir::stablehlo::AddOp>(def)) {
      if (IsReductionInvariantOperand(add.getRhs(), red_dims)) {
        rev_steps.push_back({StepKind::kAddOnLhs, def, cur_ty.getElementType(),
                             add.getRhs(), 0.0});
        all_single_use = all_single_use && step_single_use;
        cur = add.getLhs();
        continue;
      }
      if (IsReductionInvariantOperand(add.getLhs(), red_dims)) {
        rev_steps.push_back({StepKind::kAddOnRhs, def, cur_ty.getElementType(),
                             add.getLhs(), 0.0});
        all_single_use = all_single_use && step_single_use;
        cur = add.getRhs();
        continue;
      }
      break;
    }

    if (auto sub = llvm::dyn_cast<mlir::stablehlo::SubtractOp>(def)) {
      if (IsReductionInvariantOperand(sub.getRhs(), red_dims)) {
        rev_steps.push_back({StepKind::kSubOnLhs, def, cur_ty.getElementType(),
                             sub.getRhs(), 0.0});
        all_single_use = all_single_use && step_single_use;
        cur = sub.getLhs();
        continue;
      }
      break;
    }

    break;
  }

  // Peel any leading ConvertOp steps if they are not needed or check whether
  // the chain has at least one non-convert operation.
  bool has_non_convert = false;
  bool has_nonlinear = false;
  for (const MonotonicStep& s : rev_steps) {
    if (s.kind != StepKind::kConvert) {
      has_non_convert = true;
    }
    if (IsNonlinearStep(s.kind)) {
      has_nonlinear = true;
    }
  }
  if (!has_non_convert) {
    return false;
  }

  // When intermediate values in the chain have additional users (for example,
  // `y = c * tanh(x / c)` feeding both `reduce_max(y)` and `y - max_y`),
  // hoisting across the shared chain is profitable when the chain either is
  // single-use or widens a narrower float type (`bf16 -> f32`) before
  // evaluating a nonlinear monotonic op (`tanh`, `exp`, etc.).
  const bool starts_with_widening_convert =
      rev_steps.back().kind == StepKind::kConvert &&
      llvm::cast<mlir::FloatType>(
          GetStaticFloatTensorType(cur).getElementType())
              .getWidth() <
          llvm::cast<mlir::FloatType>(rev_steps.back().result_elem_type)
              .getWidth();
  if (!all_single_use && !(has_nonlinear && starts_with_widening_convert)) {
    return false;
  }

  auto base_ty = GetStaticFloatTensorType(cur);
  auto base_fl = llvm::cast<mlir::FloatType>(base_ty.getElementType());
  mlir::OpBuilder builder(reduce_op);
  mlir::Location loc = reduce_op.getLoc();

  auto scalar_base_ty = mlir::RankedTensorType::get({}, base_fl);
  const bool neg_init = (red_kind == ReduceKind::kMax);
  llvm::APFloat init_ap =
      init_is_inf
          ? llvm::APFloat::getInf(base_fl.getFloatSemantics(), neg_init)
          : llvm::APFloat::getLargest(base_fl.getFloatSemantics(), neg_init);
  mlir::Value new_init = mlir::stablehlo::ConstantOp::create(
      builder, loc, scalar_base_ty,
      mlir::DenseElementsAttr::get(scalar_base_ty, init_ap));

  auto red_base_ty = mlir::RankedTensorType::get(out_type.getShape(), base_fl);
  auto new_reduce = mlir::stablehlo::ReduceOp::create(
      builder, loc, mlir::TypeRange{red_base_ty}, mlir::ValueRange{cur},
      mlir::ValueRange{new_init}, reduce_op.getDimensionsAttr());

  mlir::Block* body =
      builder.createBlock(&new_reduce.getBody(), new_reduce.getBody().end(),
                          {scalar_base_ty, scalar_base_ty}, {loc, loc});
  builder.setInsertionPointToStart(body);
  mlir::Value comb_res =
      (red_kind == ReduceKind::kMax)
          ? mlir::stablehlo::MaxOp::create(builder, loc, body->getArgument(0),
                                           body->getArgument(1))
                .getResult()
          : mlir::stablehlo::MinOp::create(builder, loc, body->getArgument(0),
                                           body->getArgument(1))
                .getResult();
  mlir::stablehlo::ReturnOp::create(builder, loc, mlir::ValueRange{comb_res});

  builder.setInsertionPointAfter(new_reduce);
  mlir::Value cur_red = new_reduce.getResult(0);
  for (const MonotonicStep& step : llvm::reverse(rev_steps)) {
    auto step_red_ty =
        mlir::RankedTensorType::get(out_type.getShape(), step.result_elem_type);
    mlir::Location step_loc = step.op->getLoc();
    switch (step.kind) {
      case StepKind::kConvert:
        cur_red = mlir::stablehlo::ConvertOp::create(builder, step_loc,
                                                     step_red_ty, cur_red);
        break;
      case StepKind::kTanh:
        cur_red = mlir::stablehlo::TanhOp::create(builder, step_loc,
                                                  step_red_ty, cur_red);
        break;
      case StepKind::kExp:
        cur_red = mlir::stablehlo::ExpOp::create(builder, step_loc, step_red_ty,
                                                 cur_red);
        break;
      case StepKind::kLogistic:
        cur_red = mlir::stablehlo::LogisticOp::create(builder, step_loc,
                                                      step_red_ty, cur_red);
        break;
      case StepKind::kErf:
        cur_red =
            mlir::chlo::ErfOp::create(builder, step_loc, step_red_ty, cur_red);
        break;
      case StepKind::kMulPosConst: {
        mlir::Value c_red = CreateSplatFloatConstant(
            builder, step_loc, step_red_ty, step.pos_const_val);
        cur_red = mlir::stablehlo::MulOp::create(builder, step_loc, step_red_ty,
                                                 cur_red, c_red);
        break;
      }
      case StepKind::kDivPosConst: {
        mlir::Value c_red = CreateSplatFloatConstant(
            builder, step_loc, step_red_ty, step.pos_const_val);
        cur_red = mlir::stablehlo::DivOp::create(builder, step_loc, step_red_ty,
                                                 cur_red, c_red);
        break;
      }
      case StepKind::kAddOnLhs: {
        mlir::Value other_red = ProjectInvariantOperandToReducedShape(
            builder, step_loc, step.invariant_other, step_red_ty, red_dims);
        cur_red = mlir::stablehlo::AddOp::create(builder, step_loc, step_red_ty,
                                                 cur_red, other_red);
        break;
      }
      case StepKind::kAddOnRhs: {
        mlir::Value other_red = ProjectInvariantOperandToReducedShape(
            builder, step_loc, step.invariant_other, step_red_ty, red_dims);
        cur_red = mlir::stablehlo::AddOp::create(builder, step_loc, step_red_ty,
                                                 other_red, cur_red);
        break;
      }
      case StepKind::kSubOnLhs: {
        mlir::Value other_red = ProjectInvariantOperandToReducedShape(
            builder, step_loc, step.invariant_other, step_red_ty, red_dims);
        cur_red = mlir::stablehlo::SubtractOp::create(
            builder, step_loc, step_red_ty, cur_red, other_red);
        break;
      }
    }
  }

  reduce_op.getResult(0).replaceAllUsesWith(cur_red);
  reduce_op.erase();
  if (all_single_use) {
    for (const MonotonicStep& step : rev_steps) {
      if (step.op && step.op->use_empty()) {
        step.op->erase();
      }
    }
  }
  return true;
}

// Maps a contracting dimension `contract_dim` on `user_val` back to the axis of
// `base_val` through `TransposeOp` and unit-dimension `ReshapeOp` chains.
[[nodiscard]] std::optional<int64_t> MapAxisBackToBase(mlir::Value base_val,
                                                       mlir::Value user_val,
                                                       int64_t axis) {
  if (user_val == base_val) {
    return axis;
  }
  mlir::Operation* def = user_val.getDefiningOp();
  if (!def) {
    return std::nullopt;
  }
  if (auto tr = llvm::dyn_cast<mlir::stablehlo::TransposeOp>(def)) {
    llvm::ArrayRef<int64_t> perm = tr.getPermutation();
    if (axis < 0 || axis >= static_cast<int64_t>(perm.size())) {
      return std::nullopt;
    }
    return MapAxisBackToBase(base_val, tr.getOperand(), perm[axis]);
  }
  if (auto resh = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(def)) {
    auto in_ty = GetStaticFloatTensorType(resh.getOperand());
    auto out_ty = GetStaticFloatTensorType(resh.getResult());
    if (!in_ty || !out_ty || axis < 0 || axis >= out_ty.getRank()) {
      return std::nullopt;
    }
    const int64_t target_len = out_ty.getDimSize(axis);
    int64_t out_prefix = 1;
    for (int64_t i = 0; i < axis; ++i) {
      out_prefix *= out_ty.getDimSize(i);
    }
    int64_t in_prefix = 1;
    for (int64_t j = 0; j < in_ty.getRank(); ++j) {
      if (in_prefix == out_prefix && in_ty.getDimSize(j) == target_len) {
        return MapAxisBackToBase(base_val, resh.getOperand(), j);
      }
      in_prefix *= in_ty.getDimSize(j);
    }
  }
  return std::nullopt;
}

// Collects distinct contracting axes on `base_val` across all direct or
// view-mediated `DotGeneralOp` / `DotOp` consumers.
[[nodiscard]] llvm::SmallVector<int64_t, 4> CollectContractedBaseAxes(
    mlir::Value base_val) {
  llvm::SmallVector<int64_t, 4> axes;
  llvm::SmallVector<mlir::Value, 8> worklist = {base_val};
  llvm::DenseSet<mlir::Value> visited;

  while (!worklist.empty()) {
    mlir::Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second) {
      continue;
    }
    for (mlir::OpOperand& use : cur.getUses()) {
      mlir::Operation* user = use.getOwner();
      if (llvm::isa<mlir::stablehlo::TransposeOp, mlir::stablehlo::ReshapeOp>(
              user)) {
        worklist.push_back(user->getResult(0));
        continue;
      }
      if (auto dot_gen = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(user)) {
        auto dims = dot_gen.getDotDimensionNumbers();
        llvm::ArrayRef<int64_t> c_dims =
            (use.getOperandNumber() == 0) ? dims.getLhsContractingDimensions()
                                          : dims.getRhsContractingDimensions();
        for (int64_t cd : c_dims) {
          if (std::optional<int64_t> mapped =
                  MapAxisBackToBase(base_val, cur, cd)) {
            if (!llvm::is_contained(axes, *mapped)) {
              axes.push_back(*mapped);
            }
          }
        }
      } else if (llvm::isa<mlir::stablehlo::DotOp>(user)) {
        auto cur_ty = GetStaticFloatTensorType(cur);
        if (!cur_ty || cur_ty.getRank() < 1) {
          continue;
        }
        // GENERIC: semantic DotOp LHS contracts last axis, RHS contracts axis 0
        const int64_t cd =
            (use.getOperandNumber() == 0) ? (cur_ty.getRank() - 1) : 0;
        if (std::optional<int64_t> mapped =
                MapAxisBackToBase(base_val, cur, cd)) {
          if (!llvm::is_contained(axes, *mapped)) {
            axes.push_back(*mapped);
          }
        }
      }
    }
  }
  return axes;
}

// Returns true if the wider-float elementwise producer DAG of `wide_val`
// contains at least two transcendental operations (e.g. `TanhOp`, `ExpOp`,
// `LogisticOp`, `chlo::ErfOp`) at `wide_width` bits.
[[nodiscard]] bool HasMultipleWideTranscendentalsInProducerDag(
    mlir::Value wide_val, unsigned wide_width) {
  int transcendental_count = 0;
  llvm::SmallVector<mlir::Value, 16> stack = {wide_val};
  llvm::DenseSet<mlir::Value> visited;

  while (!stack.empty()) {
    mlir::Value cur = stack.pop_back_val();
    if (!visited.insert(cur).second) {
      continue;
    }
    mlir::Operation* def = cur.getDefiningOp();
    if (!def) {
      continue;
    }
    if (llvm::isa<mlir::stablehlo::DotGeneralOp, mlir::stablehlo::DotOp,
                  mlir::stablehlo::ConvolutionOp, mlir::stablehlo::ReduceOp,
                  mlir::stablehlo::GatherOp, mlir::stablehlo::ScatterOp,
                  mlir::stablehlo::OptimizationBarrierOp>(def)) {
      continue;
    }
    if (auto cur_ty = GetStaticFloatTensorType(cur)) {
      auto fl = llvm::cast<mlir::FloatType>(cur_ty.getElementType());
      if (fl.getWidth() == wide_width) {
        if (llvm::isa<mlir::stablehlo::TanhOp, mlir::stablehlo::ExpOp,
                      mlir::stablehlo::LogisticOp, mlir::chlo::ErfOp>(def)) {
          ++transcendental_count;
          // GENERIC: semantic chain of at least two transcendentals
          if (transcendental_count >= 2) {
            return true;
          }
        }
      }
    }
    for (mlir::Value opnd : def->getOperands()) {
      stack.push_back(opnd);
    }
  }
  return false;
}

// Inserts `OptimizationBarrierOp` on narrowing `ConvertOp` results (`f32 ->
// bf16`) whose wider-float producer DAG evaluates two or more transcendental
// operations and whose narrowed output feeds two or more contractions along
// distinct contracting axes.
bool TryBarrierNarrowedMultiDotGradient(mlir::stablehlo::ConvertOp conv_op) {
  mlir::Value res = conv_op.getResult();
  auto dst_ty = GetStaticFloatTensorType(res);
  auto src_ty = GetStaticFloatTensorType(conv_op.getOperand());
  if (!dst_ty || !src_ty) {
    return false;
  }
  auto dst_fl = llvm::cast<mlir::FloatType>(dst_ty.getElementType());
  auto src_fl = llvm::cast<mlir::FloatType>(src_ty.getElementType());
  if (src_fl.getWidth() <= dst_fl.getWidth()) {
    return false;
  }

  for (mlir::Operation* user : res.getUsers()) {
    if (llvm::isa<mlir::stablehlo::OptimizationBarrierOp>(user)) {
      return false;
    }
  }

  // Peel unit-dimension ReshapeOps if the fanout happens on a reshaped view.
  mlir::Value fanout_val = res;
  while (fanout_val.hasOneUse()) {
    if (auto resh = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(
            *fanout_val.getUsers().begin())) {
      fanout_val = resh.getResult();
      continue;
    }
    break;
  }
  for (mlir::Operation* user : fanout_val.getUsers()) {
    if (llvm::isa<mlir::stablehlo::OptimizationBarrierOp>(user)) {
      return false;
    }
  }

  llvm::SmallVector<int64_t, 4> contracted_axes =
      CollectContractedBaseAxes(fanout_val);
  // GENERIC: semantic requirement that >= 2 orthogonal axes of the gradient
  // matrix are contracted by consumer matmuls (e.g. dX = G @ W and dW = G^T @
  // X)
  if (contracted_axes.size() < 2) {
    return false;
  }

  if (!HasMultipleWideTranscendentalsInProducerDag(conv_op.getOperand(),
                                                   src_fl.getWidth())) {
    return false;
  }

  mlir::Operation* def_op = fanout_val.getDefiningOp();
  mlir::OpBuilder builder(def_op);
  builder.setInsertionPointAfter(def_op);
  auto barrier = mlir::stablehlo::OptimizationBarrierOp::create(
      builder, def_op->getLoc(), mlir::ValueRange{fanout_val});
  mlir::Value barrier_out = barrier->getResult(0);
  fanout_val.replaceAllUsesExcept(barrier_out, barrier);
  return true;
}

class HoistMonotonicReduceAndBarrierMultiDotGradPass
    : public mlir::PassWrapper<HoistMonotonicReduceAndBarrierMultiDotGradPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      HoistMonotonicReduceAndBarrierMultiDotGradPass)

  llvm::StringRef getArgument() const final {
    return "hoist-monotonic-reduce-and-barrier-multi-dot-grad";
  }

  llvm::StringRef getDescription() const final {
    return "Hoists monotonically non-decreasing chains out of ReduceMax/Min "
           "and barriers narrowed multi-transcendental dual-dot gradients.";
  }

  void runOnOperation() final {
    mlir::func::FuncOp func = getOperation();
    FoldConsecutiveConstantMulDiv(func);

    llvm::SmallVector<mlir::stablehlo::ReduceOp, 16> reduces;
    func.walk([&](mlir::stablehlo::ReduceOp op) { reduces.push_back(op); });
    for (mlir::stablehlo::ReduceOp op : reduces) {
      if (op && op->getBlock() != nullptr) {
        TryHoistMonotonicChainFromReduce(op);
      }
    }

    FoldConsecutiveConstantMulDiv(func);

    llvm::SmallVector<mlir::stablehlo::ConvertOp, 16> converts;
    func.walk([&](mlir::stablehlo::ConvertOp op) { converts.push_back(op); });
    for (mlir::stablehlo::ConvertOp op : converts) {
      if (op && op->getBlock() != nullptr) {
        TryBarrierNarrowedMultiDotGradient(op);
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createHoistMonotonicReduceAndBarrierMultiDotGradPass() {
  return std::make_unique<HoistMonotonicReduceAndBarrierMultiDotGradPass>();
}

}  // namespace torch_tpu
