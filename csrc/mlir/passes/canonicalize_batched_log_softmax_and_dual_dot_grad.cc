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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
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
#include "mlir/IR/IRMapping.h"
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
  mlir::Value input_val;
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

// Returns true if `op` is a pure elementwise operation whose semantics are
// invariant to bijective shape reshapes.
[[nodiscard]] bool IsShapeAgnosticElementwiseOp(mlir::Operation* op) {
  return llvm::isa<mlir::stablehlo::ConvertOp, mlir::stablehlo::TanhOp,
                   mlir::stablehlo::ExpOp, mlir::stablehlo::LogisticOp,
                   mlir::stablehlo::LogOp, mlir::stablehlo::NegOp,
                   mlir::stablehlo::AbsOp, mlir::stablehlo::SqrtOp,
                   mlir::stablehlo::RsqrtOp, mlir::stablehlo::AddOp,
                   mlir::stablehlo::SubtractOp, mlir::stablehlo::MulOp,
                   mlir::stablehlo::DivOp, mlir::stablehlo::MaxOp,
                   mlir::stablehlo::MinOp, mlir::chlo::ErfOp>(op);
}

// Returns true if `op` is a splat ConstantOp or a scalar BroadcastInDimOp
// (empty broadcast_dimensions) producing `expected_shape`.
[[nodiscard]] bool IsSplatOrScalarBroadcastOfShape(
    mlir::Operation* op, llvm::ArrayRef<int64_t> expected_shape) {
  if (!op || op->getNumResults() != 1) {
    return false;
  }
  auto res_ty = GetStaticFloatTensorType(op->getResult(0));
  if (!res_ty || res_ty.getShape() != expected_shape) {
    return false;
  }
  if (auto const_op = llvm::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
    auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    return dense && dense.isSplat();
  }
  if (auto bcast = llvm::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
    return bcast.getBroadcastDimensions().empty();
  }
  return false;
}

// Rewrites closed connected components of shape-agnostic elementwise operations
// that are entered exclusively via `ReshapeOp(S_outer -> S_inner)` and exited
// exclusively via `ReshapeOp(S_inner -> S_outer)` with `rank(S_outer) <
// rank(S_inner)` to execute directly at the lower rank `S_outer`.
bool FoldRedundantElementwiseReshapeRoundtrips(mlir::func::FuncOp func) {
  bool any_changed = false;
  llvm::DenseSet<mlir::Operation*> processed_entries;

  for (mlir::Block& block : func.getBody()) {
    llvm::SmallVector<mlir::stablehlo::ReshapeOp, 16> candidate_entries;
    for (mlir::Operation& op : block) {
      if (auto resh = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(&op)) {
        auto in_ty = GetStaticFloatTensorType(resh.getOperand());
        auto out_ty = GetStaticFloatTensorType(resh.getResult());
        // GENERIC: semantic canonicalization of rank-expanding elementwise
        // reshape detours back to the lower outer rank.
        if (in_ty && out_ty && in_ty.getRank() < out_ty.getRank()) {
          candidate_entries.push_back(resh);
        }
      }
    }

    for (mlir::stablehlo::ReshapeOp seed_entry : candidate_entries) {
      if (!seed_entry ||
          processed_entries.contains(seed_entry.getOperation())) {
        continue;
      }
      auto outer_sample_ty = GetStaticFloatTensorType(seed_entry.getOperand());
      auto inner_sample_ty = GetStaticFloatTensorType(seed_entry.getResult());
      if (!outer_sample_ty || !inner_sample_ty) {
        continue;
      }
      llvm::ArrayRef<int64_t> outer_shape = outer_sample_ty.getShape();
      llvm::ArrayRef<int64_t> inner_shape = inner_sample_ty.getShape();

      llvm::DenseSet<mlir::Operation*> comp_ops;
      llvm::DenseSet<mlir::Operation*> entry_reshapes;
      llvm::DenseSet<mlir::Operation*> exit_reshapes;
      llvm::SmallVector<mlir::Operation*, 16> worklist = {
          seed_entry.getOperation()};
      bool valid = true;
      bool has_elementwise = false;

      while (!worklist.empty() && valid) {
        mlir::Operation* cur = worklist.pop_back_val();
        if (!comp_ops.insert(cur).second) {
          continue;
        }
        if (cur->getBlock() != &block || cur->getNumResults() != 1) {
          valid = false;
          break;
        }

        if (auto resh = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(cur)) {
          auto r_in = GetStaticFloatTensorType(resh.getOperand());
          auto r_out = GetStaticFloatTensorType(resh.getResult());
          if (r_in && r_out && r_in.getShape() == outer_shape &&
              r_out.getShape() == inner_shape) {
            entry_reshapes.insert(cur);
            for (mlir::Operation* user : resh.getResult().getUsers()) {
              worklist.push_back(user);
            }
            continue;
          }
          if (r_in && r_out && r_in.getShape() == inner_shape &&
              r_out.getShape() == outer_shape) {
            exit_reshapes.insert(cur);
            if (mlir::Operation* in_def = resh.getOperand().getDefiningOp()) {
              worklist.push_back(in_def);
            } else {
              valid = false;
            }
            continue;
          }
          valid = false;
          break;
        }

        if (IsSplatOrScalarBroadcastOfShape(cur, inner_shape)) {
          continue;
        }

        if (IsShapeAgnosticElementwiseOp(cur)) {
          auto res_ty = GetStaticFloatTensorType(cur->getResult(0));
          if (!res_ty || res_ty.getShape() != inner_shape) {
            valid = false;
            break;
          }
          has_elementwise = true;
          for (mlir::Value opnd : cur->getOperands()) {
            auto opnd_ty = GetStaticFloatTensorType(opnd);
            mlir::Operation* opnd_def = opnd.getDefiningOp();
            if (!opnd_ty || opnd_ty.getShape() != inner_shape || !opnd_def) {
              valid = false;
              break;
            }
            worklist.push_back(opnd_def);
          }
          if (!valid) {
            break;
          }
          for (mlir::Operation* user : cur->getResult(0).getUsers()) {
            worklist.push_back(user);
          }
          continue;
        }

        valid = false;
      }

      for (mlir::Operation* e : entry_reshapes) {
        processed_entries.insert(e);
      }
      if (!valid || !has_elementwise || exit_reshapes.empty()) {
        continue;
      }

      llvm::SmallVector<mlir::Operation*, 16> topo_ops;
      for (mlir::Operation& op : block) {
        if (comp_ops.contains(&op)) {
          topo_ops.push_back(&op);
        }
      }

      llvm::DenseMap<mlir::Value, mlir::Value> rewritten;
      for (mlir::Operation* op : topo_ops) {
        if (entry_reshapes.contains(op)) {
          auto resh = llvm::cast<mlir::stablehlo::ReshapeOp>(op);
          rewritten[resh.getResult()] = resh.getOperand();
          continue;
        }
        if (exit_reshapes.contains(op)) {
          continue;
        }
        mlir::OpBuilder builder(op);
        mlir::Location loc = op->getLoc();
        auto old_res_ty = GetStaticFloatTensorType(op->getResult(0));
        auto new_res_ty = mlir::RankedTensorType::get(
            outer_shape, old_res_ty.getElementType());

        if (auto const_op = llvm::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
          auto dense = llvm::cast<mlir::DenseElementsAttr>(const_op.getValue());
          auto new_attr = mlir::DenseElementsAttr::get(
              new_res_ty, dense.getSplatValue<mlir::Attribute>());
          mlir::Value new_c = mlir::stablehlo::ConstantOp::create(
              builder, loc, new_res_ty, new_attr);
          rewritten[const_op.getResult()] = new_c;
          continue;
        }
        if (auto bcast =
                llvm::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
          mlir::Value new_b = mlir::stablehlo::BroadcastInDimOp::create(
              builder, loc, new_res_ty, bcast.getOperand(),
              builder.getDenseI64ArrayAttr({}));
          rewritten[bcast.getResult()] = new_b;
          continue;
        }

        mlir::IRMapping mapping;
        for (mlir::Value opnd : op->getOperands()) {
          mapping.map(opnd, rewritten.lookup(opnd));
        }
        mlir::Operation* cloned = builder.clone(*op, mapping);
        cloned->getResult(0).setType(new_res_ty);
        rewritten[op->getResult(0)] = cloned->getResult(0);
      }

      for (mlir::Operation* op : topo_ops) {
        if (exit_reshapes.contains(op)) {
          auto resh = llvm::cast<mlir::stablehlo::ReshapeOp>(op);
          mlir::Value repl = rewritten.lookup(resh.getOperand());
          resh.getResult().replaceAllUsesWith(repl);
        }
      }

      for (mlir::Operation* op : llvm::reverse(topo_ops)) {
        if (op->use_empty()) {
          op->erase();
        }
      }
      any_changed = true;
    }
  }
  return any_changed;
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

[[nodiscard]] bool SameNonZeroSplatConstant(mlir::Value a, mlir::Value b) {
  std::optional<double> va = ExtractSplatFloatConstant(a);
  std::optional<double> vb = ExtractSplatFloatConstant(b);
  if (!va.has_value() || !vb.has_value() || *va == 0.0 || *vb == 0.0) {
    return false;
  }
  return *va == *vb;
}

[[nodiscard]] mlir::Value PeelConvertAndReshape(mlir::Value v) {
  while (true) {
    if (auto conv = v.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      v = conv.getOperand();
      continue;
    }
    if (auto resh = v.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      v = resh.getOperand();
      continue;
    }
    break;
  }
  return v;
}

[[nodiscard]] bool IsOneMinusTanhSquared(mlir::Value factor) {
  factor = PeelConvertAndReshape(factor);
  auto sub = factor.getDefiningOp<mlir::stablehlo::SubtractOp>();
  if (!sub) {
    return false;
  }
  // GENERIC: semantic identity 1 - tanh(x)^2 derivative factor check
  std::optional<double> lhs_const = ExtractSplatFloatConstant(sub.getLhs());
  if (!lhs_const.has_value() || *lhs_const != 1.0) {
    return false;
  }
  mlir::Value rhs = PeelConvertAndReshape(sub.getRhs());
  auto sq = rhs.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!sq || PeelConvertAndReshape(sq.getLhs()) !=
                 PeelConvertAndReshape(sq.getRhs())) {
    return false;
  }
  mlir::Value t = PeelConvertAndReshape(sq.getLhs());
  return t.getDefiningOp<mlir::stablehlo::TanhOp>() != nullptr;
}

// Cancels softcapping backward scale multiplication and division:
// `((dy * c) * (1 - tanh(x)^2)) / c -> dy * (1 - tanh(x)^2)`.
bool TryCancelSoftcapBackwardScale(mlir::stablehlo::DivOp div_op) {
  if (!GetStaticFloatTensorType(div_op.getResult())) {
    return false;
  }
  mlir::Value numer = div_op.getLhs();
  mlir::Value denom = div_op.getRhs();
  auto dt_mul = numer.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!dt_mul) {
    return false;
  }

  auto try_extract =
      [&](mlir::Value candidate_dm,
          mlir::Value other) -> std::pair<mlir::Value, mlir::Value> {
    if (!IsOneMinusTanhSquared(other)) {
      return {nullptr, nullptr};
    }
    auto dm_mul = candidate_dm.getDefiningOp<mlir::stablehlo::MulOp>();
    if (!dm_mul) {
      return {nullptr, nullptr};
    }
    if (SameNonZeroSplatConstant(dm_mul.getRhs(), denom)) {
      return {dm_mul.getLhs(), other};
    }
    if (SameNonZeroSplatConstant(dm_mul.getLhs(), denom)) {
      return {dm_mul.getRhs(), other};
    }
    return {nullptr, nullptr};
  };

  auto [dy, factor] = try_extract(dt_mul.getLhs(), dt_mul.getRhs());
  if (!dy) {
    std::tie(dy, factor) = try_extract(dt_mul.getRhs(), dt_mul.getLhs());
  }
  if (!dy || !factor) {
    return false;
  }

  mlir::OpBuilder builder(div_op);
  auto new_mul = mlir::stablehlo::MulOp::create(builder, div_op.getLoc(),
                                                div_op.getType(), dy, factor);
  div_op.replaceAllUsesWith(new_mul.getResult());
  div_op.erase();
  if (dt_mul->use_empty()) {
    mlir::Operation* lhs_def = dt_mul.getLhs().getDefiningOp();
    mlir::Operation* rhs_def = dt_mul.getRhs().getDefiningOp();
    dt_mul.erase();
    if (lhs_def && lhs_def->use_empty() &&
        llvm::isa<mlir::stablehlo::MulOp>(lhs_def)) {
      lhs_def->erase();
    }
    if (rhs_def && rhs_def != lhs_def && rhs_def->use_empty() &&
        llvm::isa<mlir::stablehlo::MulOp>(rhs_def)) {
      rhs_def->erase();
    }
  }
  return true;
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
      // GENERIC: capability IEEE-754 f32/f64 precision required for constant
      // reassociation without bf16 rounding drift
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

// Returns true if `val` is invariant along all dimensions in `red_dims`.
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
  auto in_shape =
      llvm::cast<mlir::RankedTensorType>(bcast_in.getType()).getShape();
  auto expected_in_ty =
      mlir::RankedTensorType::get(in_shape, reduced_type.getElementType());
  if (bcast_in.getType() != expected_in_ty) {
    bcast_in = mlir::stablehlo::ConvertOp::create(builder, loc, expected_in_ty,
                                                  bcast_in);
  }
  if (bcast_in.getType() == reduced_type &&
      llvm::equal(new_dims, llvm::seq<int64_t>(0, reduced_type.getRank()))) {
    return bcast_in;
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
  bool seen_non_convert = false;
  bool seen_widening_convert = false;
  size_t best_prefix_len = 0;

  while (mlir::Operation* def = cur.getDefiningOp()) {
    auto cur_ty = GetStaticFloatTensorType(cur);
    if (!cur_ty || cur_ty.getShape() != in_type.getShape()) {
      break;
    }
    // The root input of Reduce(Max/Min) in log_softmax is shared with
    // Subtract(input, Broadcast(max)); intermediate ops in the prefix must be
    // single-use unless a widening ConvertOp reduces the reduction bitwidth.
    const bool step_single_use = (cur == input) || cur.hasOneUse();

    auto record_step = [&](StepKind kind, mlir::Value next_in,
                           mlir::Value inv_other, double pos_c,
                           bool is_widening_conv) {
      rev_steps.push_back(
          {kind, def, next_in, cur_ty.getElementType(), inv_other, pos_c});
      all_single_use = all_single_use && step_single_use;
      if (kind != StepKind::kConvert) {
        seen_non_convert = true;
      }
      if (is_widening_conv) {
        seen_widening_convert = true;
      }
      if (all_single_use && seen_non_convert &&
          (input.hasOneUse() || seen_widening_convert)) {
        best_prefix_len = rev_steps.size();
      }
      cur = next_in;
    };

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
      record_step(StepKind::kConvert, conv.getOperand(), nullptr, 0.0,
                  src_fl.getWidth() < dst_fl.getWidth());
      continue;
    }

    if (llvm::isa<mlir::stablehlo::TanhOp>(def)) {
      record_step(StepKind::kTanh, def->getOperand(0), nullptr, 0.0, false);
      continue;
    }
    if (llvm::isa<mlir::stablehlo::ExpOp>(def)) {
      record_step(StepKind::kExp, def->getOperand(0), nullptr, 0.0, false);
      continue;
    }
    if (llvm::isa<mlir::stablehlo::LogisticOp>(def)) {
      record_step(StepKind::kLogistic, def->getOperand(0), nullptr, 0.0, false);
      continue;
    }
    if (llvm::isa<mlir::chlo::ErfOp>(def)) {
      record_step(StepKind::kErf, def->getOperand(0), nullptr, 0.0, false);
      continue;
    }

    if (auto mul = llvm::dyn_cast<mlir::stablehlo::MulOp>(def)) {
      if (std::optional<double> c = ExtractSplatFloatConstant(mul.getRhs());
          c.has_value() && *c > 0.0) {
        record_step(StepKind::kMulPosConst, mul.getLhs(), nullptr, *c, false);
        continue;
      }
      if (std::optional<double> c = ExtractSplatFloatConstant(mul.getLhs());
          c.has_value() && *c > 0.0) {
        record_step(StepKind::kMulPosConst, mul.getRhs(), nullptr, *c, false);
        continue;
      }
      break;
    }

    if (auto div = llvm::dyn_cast<mlir::stablehlo::DivOp>(def)) {
      if (std::optional<double> c = ExtractSplatFloatConstant(div.getRhs());
          c.has_value() && *c > 0.0) {
        record_step(StepKind::kDivPosConst, div.getLhs(), nullptr, *c, false);
        continue;
      }
      break;
    }

    if (auto add = llvm::dyn_cast<mlir::stablehlo::AddOp>(def)) {
      if (IsReductionInvariantOperand(add.getRhs(), red_dims)) {
        record_step(StepKind::kAddOnLhs, add.getLhs(), add.getRhs(), 0.0,
                    false);
        continue;
      }
      if (IsReductionInvariantOperand(add.getLhs(), red_dims)) {
        record_step(StepKind::kAddOnRhs, add.getRhs(), add.getLhs(), 0.0,
                    false);
        continue;
      }
      break;
    }

    if (auto sub = llvm::dyn_cast<mlir::stablehlo::SubtractOp>(def)) {
      if (IsReductionInvariantOperand(sub.getRhs(), red_dims)) {
        record_step(StepKind::kSubOnLhs, sub.getLhs(), sub.getRhs(), 0.0,
                    false);
        continue;
      }
      break;
    }

    break;
  }

  if (rev_steps.empty()) {
    return false;
  }

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

  const bool starts_with_widening_convert =
      rev_steps.back().kind == StepKind::kConvert &&
      llvm::cast<mlir::FloatType>(
          GetStaticFloatTensorType(cur).getElementType())
              .getWidth() <
          llvm::cast<mlir::FloatType>(rev_steps.back().result_elem_type)
              .getWidth();
  const bool can_use_full_chain =
      (all_single_use && (input.hasOneUse() || seen_widening_convert)) ||
      (has_nonlinear && starts_with_widening_convert);
  if (!can_use_full_chain) {
    if (best_prefix_len == 0) {
      return false;
    }
    rev_steps.truncate(best_prefix_len);
    cur = rev_steps.back().input_val;
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

// Maps a contracting dimension `axis` on `user_val` back to the corresponding
// axis on `base_val` through `TransposeOp` and `ReshapeOp` chains.
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
    if (axis == 0 && in_ty.getRank() >= 1) {
      int64_t prod = 1;
      for (int64_t j = 0; j < in_ty.getRank(); ++j) {
        prod *= in_ty.getDimSize(j);
        if (prod == target_len) {
          return MapAxisBackToBase(base_val, resh.getOperand(), 0);
        }
      }
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

// Returns true if the elementwise producer DAG of `root_val` (bounded at
// contractions, reductions, gathers, scatters, and barriers) contains at least
// two distinct transcendental operations (`TanhOp`, `ExpOp`, `LogisticOp`,
// `chlo::ErfOp`).
[[nodiscard]] bool HasMultipleTranscendentalsInProducerDag(
    mlir::Value root_val) {
  int transcendental_count = 0;
  llvm::SmallVector<mlir::Value, 16> stack = {root_val};
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
    if (GetStaticFloatTensorType(cur)) {
      if (llvm::isa<mlir::stablehlo::TanhOp, mlir::stablehlo::ExpOp,
                    mlir::stablehlo::LogisticOp, mlir::chlo::ErfOp>(def)) {
        ++transcendental_count;
        // GENERIC: semantic chain of at least two transcendentals
        if (transcendental_count >= 2) {
          return true;
        }
      }
    }
    for (mlir::Value opnd : def->getOperands()) {
      stack.push_back(opnd);
    }
  }
  return false;
}

// Advances from a narrowing `ConvertOp` result through any single-use
// narrow-float elementwise/reshape chain (such as `dy * (1 - tanh(x)^2)` and
// batch-flattening `ReshapeOp`s) to find the fanout value feeding downstream
// contractions.
[[nodiscard]] mlir::Value AdvanceToNarrowedFanoutValue(mlir::Value start_val,
                                                       unsigned narrow_width) {
  mlir::Value cur = start_val;
  while (cur.hasOneUse()) {
    mlir::Operation* user = *cur.getUsers().begin();
    if (user->getNumResults() != 1) {
      break;
    }
    if (llvm::isa<mlir::stablehlo::ReshapeOp>(user) ||
        IsShapeAgnosticElementwiseOp(user)) {
      auto user_ty = GetStaticFloatTensorType(user->getResult(0));
      if (!user_ty) {
        break;
      }
      auto user_fl = llvm::cast<mlir::FloatType>(user_ty.getElementType());
      if (user_fl.getWidth() <= narrow_width) {
        cur = user->getResult(0);
        continue;
      }
    }
    break;
  }
  return cur;
}

// Inserts `OptimizationBarrierOp` on narrowed gradient tensors (`f32 -> bf16`
// followed by optional narrow elementwise/reshape ops) whose producer DAG
// evaluates two or more transcendental operations and whose narrowed output
// feeds two or more contractions along distinct contracting axes.
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

  mlir::Value fanout_val = AdvanceToNarrowedFanoutValue(res, dst_fl.getWidth());
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

  if (!HasMultipleTranscendentalsInProducerDag(fanout_val)) {
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

class CanonicalizeBatchedLogSoftmaxAndDualDotGradPass
    : public mlir::PassWrapper<CanonicalizeBatchedLogSoftmaxAndDualDotGradPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CanonicalizeBatchedLogSoftmaxAndDualDotGradPass)

  llvm::StringRef getArgument() const final {
    return "canonicalize-batched-log-softmax-and-dual-dot-grad";
  }

  llvm::StringRef getDescription() const final {
    return "Folds rank-expanding elementwise reshape detours, cancels "
           "softcap backward scaling, hoists monotonic chains out of "
           "ReduceMax/Min, and barriers narrowed multi-dot gradients.";
  }

  void runOnOperation() final {
    mlir::func::FuncOp func = getOperation();

    FoldRedundantElementwiseReshapeRoundtrips(func);

    llvm::SmallVector<mlir::stablehlo::DivOp, 16> divs;
    func.walk([&](mlir::stablehlo::DivOp op) { divs.push_back(op); });
    for (mlir::stablehlo::DivOp op : divs) {
      TryCancelSoftcapBackwardScale(op);
    }

    FoldConsecutiveConstantMulDiv(func);

    llvm::SmallVector<mlir::stablehlo::ReduceOp, 16> reduces;
    func.walk([&](mlir::stablehlo::ReduceOp op) { reduces.push_back(op); });
    for (mlir::stablehlo::ReduceOp op : reduces) {
      TryHoistMonotonicChainFromReduce(op);
    }

    FoldConsecutiveConstantMulDiv(func);

    llvm::SmallVector<mlir::stablehlo::ConvertOp, 16> converts;
    func.walk([&](mlir::stablehlo::ConvertOp op) { converts.push_back(op); });
    for (mlir::stablehlo::ConvertOp op : converts) {
      TryBarrierNarrowedMultiDotGradient(op);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createCanonicalizeBatchedLogSoftmaxAndDualDotGradPass() {
  return std::make_unique<CanonicalizeBatchedLogSoftmaxAndDualDotGradPass>();
}

}  // namespace torch_tpu
