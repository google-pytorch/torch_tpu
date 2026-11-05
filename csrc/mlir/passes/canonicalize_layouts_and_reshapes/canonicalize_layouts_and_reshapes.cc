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

#include "csrc/mlir/passes/canonicalize_layouts_and_reshapes/canonicalize_layouts_and_reshapes.h"  // NOLINT

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // NOLINT
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"  // NOLINT
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"  // NOLINT
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// =============================================================================
// Shared Helpers & Utilities
// =============================================================================

[[nodiscard]] bool IsStaticRankedTensor(
    mlir::Type type, mlir::RankedTensorType* out_type = nullptr) {
  auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!ranked || !ranked.hasStaticShape()) {
    return false;
  }
  if (out_type != nullptr) {
    *out_type = ranked;
  }
  return true;
}

mlir::Value PeelBroadcastsAndReshapes(mlir::Value val) {
  while (val) {
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      val = bcast.getOperand();
      continue;
    }
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastOp>()) {
      val = bcast.getOperand();
      continue;
    }
    if (auto reshape = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      val = reshape.getOperand();
      continue;
    }
    break;
  }
  return val;
}

[[nodiscard]] bool IsScalarSplatOrBroadcast(mlir::Value val) {
  val = PeelBroadcastsAndReshapes(val);
  if (!val) {
    return false;
  }
  if (auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
    auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    return dense && !dense.empty() && dense.isSplat();
  }
  mlir::RankedTensorType type;
  return IsStaticRankedTensor(val.getType(), &type) &&
         type.getNumElements() == 1;
}

std::optional<mlir::Attribute> GetSplatConstantAttr(mlir::Value val) {
  val = PeelBroadcastsAndReshapes(val);
  if (!val) {
    return std::nullopt;
  }
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense || dense.empty() || !dense.isSplat()) {
    return std::nullopt;
  }
  return dense.getSplatValue<mlir::Attribute>();
}

mlir::Value RebuildScalarSplatOrBroadcast(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val,
    llvm::ArrayRef<int64_t> target_shape) {
  auto orig_type = llvm::cast<mlir::RankedTensorType>(val.getType());
  auto target_type =
      mlir::RankedTensorType::get(target_shape, orig_type.getElementType());
  if (auto attr = GetSplatConstantAttr(val)) {
    auto splat_attr = mlir::DenseElementsAttr::get(target_type, *attr);
    return builder.create<mlir::stablehlo::ConstantOp>(loc, target_type,
                                                       splat_attr);
  }
  mlir::Value root = PeelBroadcastsAndReshapes(val);
  auto root_type = llvm::cast<mlir::RankedTensorType>(root.getType());
  if (root_type.getRank() != 0) {
    auto scalar_type =
        mlir::RankedTensorType::get({}, target_type.getElementType());
    root = builder.create<mlir::stablehlo::ReshapeOp>(loc, scalar_type, root);
  }
  return builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, target_type, root, builder.getDenseI64ArrayAttr({}));
}

[[nodiscard]] bool IsSupportedPureElementwiseOp(mlir::Operation* op) {
  if (!op || op->getNumResults() != 1) {
    return false;
  }
  return llvm::isa<
      mlir::stablehlo::ConvertOp, mlir::stablehlo::NegOp,
      mlir::stablehlo::ExpOp, mlir::stablehlo::Expm1Op, mlir::stablehlo::LogOp,
      mlir::stablehlo::Log1pOp, mlir::chlo::ErfOp, mlir::stablehlo::TanhOp,
      mlir::stablehlo::SineOp, mlir::stablehlo::CosineOp,
      mlir::stablehlo::SqrtOp, mlir::stablehlo::RsqrtOp, mlir::stablehlo::AbsOp,
      mlir::stablehlo::LogisticOp, mlir::stablehlo::FloorOp,
      mlir::stablehlo::CeilOp, mlir::stablehlo::SignOp, mlir::stablehlo::AddOp,
      mlir::stablehlo::SubtractOp, mlir::stablehlo::MulOp,
      mlir::stablehlo::DivOp, mlir::stablehlo::MaxOp, mlir::stablehlo::MinOp,
      mlir::stablehlo::PowOp, mlir::stablehlo::CompareOp,
      mlir::stablehlo::SelectOp, mlir::stablehlo::ClampOp>(op);
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    func.walk([&changed](mlir::Block* block) {
      for (auto it = block->rbegin(); it != block->rend();) {
        mlir::Operation* op = &(*it++);
        if (mlir::isOpTriviallyDead(op)) {
          op->erase();
          changed = true;
        }
      }
    });
  }
}

mlir::Value StripOptionalConvert(mlir::Value val, bool& seen_convert) {
  while (val) {
    auto cvt = val.getDefiningOp<mlir::stablehlo::ConvertOp>();
    if (!cvt) break;
    auto src_ty =
        llvm::dyn_cast<mlir::RankedTensorType>(cvt.getOperand().getType());
    auto dst_ty = llvm::dyn_cast<mlir::RankedTensorType>(cvt.getType());
    if (!src_ty || !dst_ty) break;
    if (src_ty.getElementType() == dst_ty.getElementType()) {
      val = cvt.getOperand();
      continue;
    }
    if (seen_convert) break;
    seen_convert = true;
    val = cvt.getOperand();
  }
  return val;
}

mlir::Value ConvertIfNeeded(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value val, mlir::Type target_elem_ty) {
  auto ranked_ty = llvm::cast<mlir::RankedTensorType>(val.getType());
  if (ranked_ty.getElementType() == target_elem_ty) {
    return val;
  }
  auto out_ty =
      mlir::RankedTensorType::get(ranked_ty.getShape(), target_elem_ty);
  return builder.create<mlir::stablehlo::ConvertOp>(loc, out_ty, val);
}

// =============================================================================
// Mechanism 5: Unwrap Grouped-Query Attention KV Broadcast
// =============================================================================

struct ExpandedGqaOperand {
  mlir::Value base_kv;
  int64_t h_kv = 0;
  int64_t group_size = 0;
  bool transposed_last_two = false;
};

bool CheckTransposeLastTwo(mlir::Value& val, bool& seen_convert) {
  val = StripOptionalConvert(val, seen_convert);
  auto tr = val.getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!tr) {
    return false;
  }
  llvm::ArrayRef<int64_t> perm = tr.getPermutation();
  const int64_t r = perm.size();
  if (r < 4 || perm[r - 2] != r - 1 || perm[r - 1] != r - 2) {
    return false;
  }
  for (int64_t i = 0; i < r - 2; ++i) {
    if (perm[i] != i) {
      return false;
    }
  }
  val = StripOptionalConvert(tr.getOperand(), seen_convert);
  return true;
}

bool NonHeadDimsMatch(mlir::RankedTensorType actual_ty,
                      mlir::RankedTensorType expected_ty,
                      int64_t head_dim_idx) {
  const int64_t rank = expected_ty.getRank();
  if (actual_ty.getRank() != rank) {
    return false;
  }
  for (int64_t i = 0; i < rank; ++i) {
    if (i != head_dim_idx &&
        actual_ty.getDimSize(i) != expected_ty.getDimSize(i)) {
      return false;
    }
  }
  return true;
}

bool ValidateGqaBroadcastType(mlir::RankedTensorType bcast_ty,
                              mlir::RankedTensorType reshape_ty,
                              int64_t head_dim_idx, int64_t h_kv,
                              int64_t group_size) {
  const int64_t rank = reshape_ty.getRank();
  if (bcast_ty.getRank() != rank + 1) {
    return false;
  }
  if (bcast_ty.getDimSize(head_dim_idx) != h_kv ||
      bcast_ty.getDimSize(head_dim_idx + 1) != group_size) {
    return false;
  }
  for (int64_t i = 0; i < head_dim_idx; ++i) {
    if (bcast_ty.getDimSize(i) != reshape_ty.getDimSize(i)) {
      return false;
    }
  }
  for (int64_t i = head_dim_idx + 2; i < rank + 1; ++i) {
    if (bcast_ty.getDimSize(i) != reshape_ty.getDimSize(i - 1)) {
      return false;
    }
  }
  return true;
}

bool MatchGqaDirectBroadcast(mlir::stablehlo::BroadcastInDimOp bcast_op,
                             mlir::RankedTensorType reshape_ty,
                             int64_t head_dim_idx, int64_t h_kv) {
  auto src_ty =
      llvm::dyn_cast<mlir::RankedTensorType>(bcast_op.getOperand().getType());
  if (!src_ty || !src_ty.hasStaticShape()) {
    return false;
  }
  const int64_t rank = reshape_ty.getRank();
  if (src_ty.getRank() != rank || src_ty.getDimSize(head_dim_idx) != h_kv) {
    return false;
  }
  if (!NonHeadDimsMatch(src_ty, reshape_ty, head_dim_idx)) {
    return false;
  }
  llvm::ArrayRef<int64_t> bdims = bcast_op.getBroadcastDimensions();
  if (static_cast<int64_t>(bdims.size()) != rank) {
    return false;
  }
  for (int64_t i = 0; i <= head_dim_idx; ++i) {
    if (bdims[i] != i) {
      return false;
    }
  }
  for (int64_t i = head_dim_idx + 1; i < rank; ++i) {
    if (bdims[i] != i + 1) {
      return false;
    }
  }
  return true;
}

std::optional<mlir::Value> MatchGqaReshapedBroadcast(
    mlir::stablehlo::BroadcastInDimOp bcast_op,
    mlir::RankedTensorType reshape_ty, int64_t head_dim_idx, int64_t h_kv,
    bool& seen_convert) {
  auto src_ty =
      llvm::dyn_cast<mlir::RankedTensorType>(bcast_op.getOperand().getType());
  if (!src_ty || !src_ty.hasStaticShape()) {
    return std::nullopt;
  }
  const int64_t rank = reshape_ty.getRank();
  if (src_ty.getRank() != rank + 1 || src_ty.getDimSize(head_dim_idx) != h_kv ||
      src_ty.getDimSize(head_dim_idx + 1) != 1) {
    return std::nullopt;
  }
  for (int64_t i = 0; i < head_dim_idx; ++i) {
    if (src_ty.getDimSize(i) != reshape_ty.getDimSize(i)) {
      return std::nullopt;
    }
  }
  for (int64_t i = head_dim_idx + 2; i < rank + 1; ++i) {
    if (src_ty.getDimSize(i) != reshape_ty.getDimSize(i - 1)) {
      return std::nullopt;
    }
  }
  llvm::ArrayRef<int64_t> bdims = bcast_op.getBroadcastDimensions();
  if (static_cast<int64_t>(bdims.size()) != rank + 1) {
    return std::nullopt;
  }
  for (int64_t i = 0; i < rank + 1; ++i) {
    if (bdims[i] != i) {
      return std::nullopt;
    }
  }
  auto unsq = bcast_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!unsq) {
    return std::nullopt;
  }
  mlir::Value base = StripOptionalConvert(unsq.getOperand(), seen_convert);
  auto base_ty = llvm::dyn_cast<mlir::RankedTensorType>(base.getType());
  if (!base_ty || !base_ty.hasStaticShape() || base_ty.getRank() != rank ||
      base_ty.getDimSize(head_dim_idx) != h_kv) {
    return std::nullopt;
  }
  if (!NonHeadDimsMatch(base_ty, reshape_ty, head_dim_idx)) {
    return std::nullopt;
  }
  return base;
}

std::optional<ExpandedGqaOperand> MatchGqaExpandedOperand(
    mlir::Value val, llvm::ArrayRef<int64_t> expected_leading_batch,
    int64_t expected_hq) {
  bool seen_convert = false;
  const bool transposed = CheckTransposeLastTwo(val, seen_convert);

  auto reshape_op = val.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!reshape_op) return std::nullopt;
  auto reshape_ty =
      llvm::dyn_cast<mlir::RankedTensorType>(reshape_op.getType());
  if (!reshape_ty || !reshape_ty.hasStaticShape()) {
    return std::nullopt;
  }
  const int64_t rank = reshape_ty.getRank();
  const int64_t head_dim_idx = expected_leading_batch.size();
  if (rank <= head_dim_idx + 2 ||
      reshape_ty.getDimSize(head_dim_idx) != expected_hq) {
    return std::nullopt;
  }
  for (size_t i = 0; i < expected_leading_batch.size(); ++i) {
    if (reshape_ty.getDimSize(i) != expected_leading_batch[i]) {
      return std::nullopt;
    }
  }

  mlir::Value bcast_val =
      StripOptionalConvert(reshape_op.getOperand(), seen_convert);
  auto bcast_ty = llvm::dyn_cast<mlir::RankedTensorType>(bcast_val.getType());
  if (!bcast_ty || !bcast_ty.hasStaticShape()) {
    return std::nullopt;
  }
  const int64_t h_kv = bcast_ty.getDimSize(head_dim_idx);
  const int64_t group_size = bcast_ty.getDimSize(head_dim_idx + 1);
  if (group_size < 2 || h_kv * group_size != expected_hq) {
    return std::nullopt;
  }
  if (!ValidateGqaBroadcastType(bcast_ty, reshape_ty, head_dim_idx, h_kv,
                                group_size)) {
    return std::nullopt;
  }

  auto bcast_op = bcast_val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast_op) return std::nullopt;

  if (MatchGqaDirectBroadcast(bcast_op, reshape_ty, head_dim_idx, h_kv)) {
    return ExpandedGqaOperand{bcast_op.getOperand(), h_kv, group_size,
                              transposed};
  }

  if (auto base = MatchGqaReshapedBroadcast(bcast_op, reshape_ty, head_dim_idx,
                                            h_kv, seen_convert)) {
    return ExpandedGqaOperand{*base, h_kv, group_size, transposed};
  }

  return std::nullopt;
}

bool TryUnwrapGqaBroadcast(mlir::stablehlo::DotGeneralOp dot_op) {
  mlir::Value lhs = dot_op.getLhs();
  mlir::Value rhs = dot_op.getRhs();
  if (lhs == rhs) return false;

  auto lhs_ty = llvm::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  auto rhs_ty = llvm::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  auto res_ty =
      llvm::dyn_cast<mlir::RankedTensorType>(dot_op.getResult().getType());
  if (!lhs_ty || !rhs_ty || !res_ty || !lhs_ty.hasStaticShape() ||
      !rhs_ty.hasStaticShape() || lhs_ty.getRank() < 3 ||
      lhs_ty.getRank() != rhs_ty.getRank()) {
    return false;
  }

  auto dnums = dot_op.getDotDimensionNumbers();
  llvm::ArrayRef<int64_t> lhs_batch = dnums.getLhsBatchingDimensions();
  llvm::ArrayRef<int64_t> rhs_batch = dnums.getRhsBatchingDimensions();
  llvm::ArrayRef<int64_t> lhs_contract = dnums.getLhsContractingDimensions();
  llvm::ArrayRef<int64_t> rhs_contract = dnums.getRhsContractingDimensions();

  if (lhs_batch.empty() || lhs_batch != rhs_batch || lhs_contract.size() != 1 ||
      rhs_contract.size() != 1) {
    return false;
  }

  const int64_t rank = lhs_ty.getRank();
  const int64_t head_dim_idx = lhs_batch.back();
  if (head_dim_idx < 0 || head_dim_idx >= rank) {
    return false;
  }
  if (static_cast<int64_t>(lhs_batch.size()) != head_dim_idx + 1) {
    return false;
  }
  for (int64_t i = 0; i <= head_dim_idx; ++i) {
    if (lhs_batch[i] != i) {
      return false;
    }
  }

  const int64_t h_q = lhs_ty.getDimSize(head_dim_idx);
  llvm::SmallVector<int64_t> leading_batch;
  for (const int64_t b : lhs_batch.drop_back()) {
    leading_batch.push_back(lhs_ty.getDimSize(b));
  }

  auto rhs_exp = MatchGqaExpandedOperand(rhs, leading_batch, h_q);
  if (!rhs_exp) return false;

  const int64_t h_kv = rhs_exp->h_kv;
  const int64_t group_size = rhs_exp->group_size;
  const int64_t c_lhs = lhs_contract[0];
  const int64_t c_rhs = rhs_contract[0];

  int64_t free_lhs = -1;
  for (int64_t i = 0; i < rank; ++i) {
    if (!llvm::is_contained(lhs_batch, i) && i != c_lhs) {
      free_lhs = i;
      break;
    }
  }
  if (free_lhs == -1 || free_lhs != head_dim_idx + 1) {
    return false;
  }
  if (rhs_exp->transposed_last_two && c_rhs < rank - 2) {
    return false;
  }

  const int64_t s_q = lhs_ty.getDimSize(free_lhs);
  const int64_t q_seq_size = s_q;
  // COST_MODEL: Unwrapping 5D GQA KV broadcast into 4D head-grouped dots is
  // only profitable in autoregressive decode (q_seq_size <= 16) where small Q
  // tiles underutilize the 128x128 MXU.
  if (q_seq_size <= 0 || q_seq_size > 16) {
    return false;
  }

  const int64_t eff_c_rhs =
      rhs_exp->transposed_last_two
          ? ((c_rhs == rank - 1) ? (rank - 2) : (rank - 1))
          : c_rhs;

  mlir::OpBuilder builder(dot_op);
  const mlir::Location loc = dot_op.getLoc();
  mlir::Value base_rhs =
      ConvertIfNeeded(builder, loc, rhs_exp->base_kv, rhs_ty.getElementType());

  llvm::SmallVector<int64_t> lhs_collapsed_shape(lhs_ty.getShape().begin(),
                                                 lhs_ty.getShape().end());
  lhs_collapsed_shape[head_dim_idx] = h_kv;
  lhs_collapsed_shape[free_lhs] = group_size * s_q;
  auto lhs_collapsed_ty =
      mlir::RankedTensorType::get(lhs_collapsed_shape, lhs_ty.getElementType());
  mlir::Value lhs_collapsed =
      builder.create<mlir::stablehlo::ReshapeOp>(loc, lhs_collapsed_ty, lhs);

  auto new_dnums = mlir::stablehlo::DotDimensionNumbersAttr::get(
      dot_op.getContext(), lhs_batch, rhs_batch, {c_lhs}, {eff_c_rhs});

  llvm::SmallVector<int64_t> dot_collapsed_shape(res_ty.getShape().begin(),
                                                 res_ty.getShape().end());
  dot_collapsed_shape[head_dim_idx] = h_kv;
  dot_collapsed_shape[free_lhs] = group_size * s_q;
  auto dot_collapsed_ty =
      mlir::RankedTensorType::get(dot_collapsed_shape, res_ty.getElementType());

  auto new_dot = builder.create<mlir::stablehlo::DotGeneralOp>(
      loc, dot_collapsed_ty, lhs_collapsed, base_rhs, new_dnums,
      dot_op.getPrecisionConfigAttr(), dot_op.getAlgorithmAttr());

  mlir::Value restored = builder.create<mlir::stablehlo::ReshapeOp>(
      loc, res_ty, new_dot.getResult());
  dot_op.getResult().replaceAllUsesWith(restored);
  dot_op.erase();
  return true;
}

// =============================================================================
// Mechanism 1: Sink Projection Reshapes Across Elementwise Chains
// =============================================================================

bool IsValidTrailingBroadcast(mlir::Value val, mlir::RankedTensorType in_type,
                              mlir::RankedTensorType out_type) {
  auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast) {
    return false;
  }
  auto src_ty =
      llvm::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
  if (!src_ty || !src_ty.hasStaticShape()) {
    return false;
  }
  const int64_t in_rank = in_type.getRank();
  const int64_t out_rank = out_type.getRank();
  const int64_t src_rank = src_ty.getRank();
  if (src_rank > in_rank || src_rank > out_rank) {
    return false;
  }
  llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
  if (static_cast<int64_t>(bdims.size()) != src_rank) {
    return false;
  }
  for (int64_t i = 0; i < src_rank; ++i) {
    if (bdims[i] != in_rank - src_rank + i) {
      return false;
    }
  }
  if (in_type.getShape().take_back(src_rank) != src_ty.getShape()) {
    return false;
  }
  return out_type.getShape().take_back(src_rank) == src_ty.getShape();
}

bool CollectProjectionElementwiseOps(
    mlir::Value cur, mlir::RankedTensorType in_type,
    mlir::RankedTensorType out_type,
    llvm::SmallVectorImpl<mlir::Operation*>& rev_elem_ops,
    mlir::Operation*& dot_op) {
  while (cur && cur.getDefiningOp() && cur.hasOneUse()) {
    mlir::Operation* def = cur.getDefiningOp();
    if (llvm::isa<mlir::stablehlo::DotOp, mlir::stablehlo::DotGeneralOp>(def)) {
      dot_op = def;
      return true;
    }
    if (!IsSupportedPureElementwiseOp(def)) {
      return false;
    }
    mlir::Value next_cur;
    for (mlir::Value opnd : def->getOperands()) {
      if (IsScalarSplatOrBroadcast(opnd) ||
          IsValidTrailingBroadcast(opnd, in_type, out_type)) {
        continue;
      }
      auto opnd_ty = llvm::dyn_cast<mlir::RankedTensorType>(opnd.getType());
      if (opnd_ty && opnd_ty.getShape() == in_type.getShape() &&
          opnd.hasOneUse() && opnd.getParentBlock() == def->getBlock()) {
        if (next_cur) {
          return false;
        }
        next_cur = opnd;
      } else {
        return false;
      }
    }
    if (!next_cur) {
      return false;
    }
    rev_elem_ops.push_back(def);
    cur = next_cur;
  }
  return false;
}

void EmitSunkProjectionReshape(mlir::stablehlo::ReshapeOp reshape_op,
                               mlir::Operation* dot_op,
                               llvm::ArrayRef<mlir::Operation*> rev_elem_ops,
                               mlir::RankedTensorType out_type) {
  mlir::OpBuilder builder(reshape_op);
  const mlir::Location loc = reshape_op.getLoc();

  mlir::Value reshaped_dot = builder.create<mlir::stablehlo::ReshapeOp>(
      loc, out_type, dot_op->getResult(0));

  mlir::IRMapping mapping;
  mapping.map(dot_op->getResult(0), reshaped_dot);

  for (auto it = rev_elem_ops.rbegin(); it != rev_elem_ops.rend(); ++it) {
    mlir::Operation* op = *it;
    for (mlir::Value opnd : op->getOperands()) {
      if (!mapping.contains(opnd)) {
        mlir::Value rebuilt;
        if (IsScalarSplatOrBroadcast(opnd)) {
          rebuilt = RebuildScalarSplatOrBroadcast(builder, loc, opnd,
                                                  out_type.getShape());
        } else if (auto bcast = opnd.getDefiningOp<
                                mlir::stablehlo::BroadcastInDimOp>()) {
          auto src_ty =
              llvm::cast<mlir::RankedTensorType>(bcast.getOperand().getType());
          llvm::SmallVector<int64_t> new_bdims;
          for (int64_t d = out_type.getRank() - src_ty.getRank();
               d < out_type.getRank(); ++d) {
            new_bdims.push_back(d);
          }
          auto target_ty = mlir::RankedTensorType::get(out_type.getShape(),
                                                       src_ty.getElementType());
          rebuilt = builder.create<mlir::stablehlo::BroadcastInDimOp>(
              loc, target_ty, bcast.getOperand(),
              builder.getDenseI64ArrayAttr(new_bdims));
        }
        mapping.map(opnd, rebuilt);
      }
    }
    mlir::Operation* cloned = builder.clone(*op, mapping);
    auto old_ty =
        llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    cloned->getResult(0).setType(mlir::RankedTensorType::get(
        out_type.getShape(), old_ty.getElementType()));
    mapping.map(op->getResult(0), cloned->getResult(0));
  }

  mlir::Value new_exit = mapping.lookup(rev_elem_ops.front()->getResult(0));
  reshape_op.replaceAllUsesWith(new_exit);
  reshape_op.erase();
  for (mlir::Operation* op : rev_elem_ops) {
    op->erase();
  }
}

bool TrySinkProjectionReshape(mlir::stablehlo::ReshapeOp reshape_op) {
  if (!reshape_op || !reshape_op->getBlock() || !reshape_op->hasOneUse()) {
    return false;
  }

  auto out_type = llvm::dyn_cast<mlir::RankedTensorType>(reshape_op.getType());
  auto in_type =
      llvm::dyn_cast<mlir::RankedTensorType>(reshape_op.getOperand().getType());
  if (!out_type || !in_type || !out_type.hasStaticShape() ||
      !in_type.hasStaticShape()) {
    return false;
  }

  mlir::Value cur = reshape_op.getOperand();
  llvm::SmallVector<mlir::Operation*, 6> rev_elem_ops;
  mlir::Operation* dot_op = nullptr;

  if (!CollectProjectionElementwiseOps(cur, in_type, out_type, rev_elem_ops,
                                       dot_op)) {
    return false;
  }

  if (rev_elem_ops.empty() || !dot_op) {
    return false;
  }

  EmitSunkProjectionReshape(reshape_op, dot_op, rev_elem_ops, out_type);
  return true;
}

// =============================================================================
// Pass Definition
// =============================================================================

bool RunUnwrapGqaBroadcast(mlir::func::FuncOp func) {
  bool changed = false;
  llvm::SmallVector<mlir::stablehlo::DotGeneralOp> dots;
  func.walk([&dots](mlir::stablehlo::DotGeneralOp op) { dots.push_back(op); });
  for (auto op : dots) {
    if (op && op->getBlock() && !op->use_empty()) {
      changed |= TryUnwrapGqaBroadcast(op);
    }
  }
  return changed;
}

bool RunSinkProjectionReshape(mlir::func::FuncOp func) {
  bool changed = false;
  llvm::SmallVector<mlir::stablehlo::ReshapeOp> reshapes;
  func.walk(
      [&reshapes](mlir::stablehlo::ReshapeOp op) { reshapes.push_back(op); });
  for (auto op : reshapes) {
    if (op && op->getBlock() && !op->use_empty()) {
      changed |= TrySinkProjectionReshape(op);
    }
  }
  return changed;
}

bool RunAllRewrites(mlir::func::FuncOp func) {
  bool changed = false;
  changed |= RunUnwrapGqaBroadcast(func);
  changed |= RunSinkProjectionReshape(func);
  return changed;
}

struct CanonicalizeLayoutsAndReshapesPass
    : public mlir::PassWrapper<CanonicalizeLayoutsAndReshapesPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CanonicalizeLayoutsAndReshapesPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-canonicalize-layouts-and-reshapes";
  }

  llvm::StringRef getDescription() const override {
    return "Commutes reshapes across projection matmuls and unwraps GQA KV "
           "broadcasts.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    constexpr int kMaxIterations = 10;
    bool changed = false;
    for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
      changed = RunAllRewrites(func);
      if (!changed) {
        break;
      }
    }
    if (changed && RunAllRewrites(func)) {
      signalPassFailure();
      return;
    }
    EraseTriviallyDeadOps(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCanonicalizeLayoutsAndReshapesPass() {
  return std::make_unique<CanonicalizeLayoutsAndReshapesPass>();
}

}  // namespace torch_tpu
