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
#include <iterator>
#include <memory>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
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

std::optional<int64_t> GetTensorElementBitWidth(mlir::Type type) {
  if (auto tensor_ty = llvm::dyn_cast<mlir::ShapedType>(type)) {
    type = tensor_ty.getElementType();
  }
  if (type.isIntOrFloat()) {
    return type.getIntOrFloatBitWidth();
  }
  if (auto complex_ty = llvm::dyn_cast<mlir::ComplexType>(type)) {
    mlir::Type elem = complex_ty.getElementType();
    if (elem.isIntOrFloat()) {
      return 2 * elem.getIntOrFloatBitWidth();
    }
  }
  return std::nullopt;
}

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

// Matched single entry single exit pure elementwise DAG.
struct PureElementwiseDagMatch {
  llvm::SmallVector<mlir::Operation*, 12> ops;
  mlir::Value exit_val;
  bool has_branch = false;
};

bool IsCandidateDagOp(mlir::Operation* op, mlir::Block* block,
                      llvm::ArrayRef<int64_t> expected_shape,
                      const llvm::DenseSet<mlir::Operation*>& dag_ops_set,
                      const llvm::DenseSet<mlir::Operation*>& enqueued_ops,
                      const llvm::DenseSet<mlir::Value>& dag_values) {
  if (dag_ops_set.contains(op) || enqueued_ops.contains(op)) {
    return false;
  }
  if (op->getBlock() != block || !IsSupportedPureElementwiseOp(op)) {
    return false;
  }
  mlir::RankedTensorType res_type;
  if (!IsStaticRankedTensor(op->getResult(0).getType(), &res_type) ||
      res_type.getShape() != expected_shape || op->getResult(0).use_empty()) {
    return false;
  }
  for (mlir::Value opnd : op->getOperands()) {
    if (!dag_values.contains(opnd) && !IsScalarSplatOrBroadcast(opnd)) {
      return false;
    }
  }
  return true;
}

mlir::Operation* PopEarliestOp(
    llvm::SmallVectorImpl<mlir::Operation*>& ready_ops) {
  auto best_it = ready_ops.begin();
  for (auto it = ready_ops.begin() + 1; it != ready_ops.end(); ++it) {
    if ((*it)->isBeforeInBlock(*best_it)) {
      best_it = it;
    }
  }
  mlir::Operation* next_op = *best_it;
  ready_ops.erase(best_it);
  return next_op;
}

void RecordDagOperandUses(
    mlir::Operation* op, const llvm::DenseSet<mlir::Value>& dag_values,
    llvm::DenseMap<mlir::Value, int>& remaining_external_uses,
    llvm::DenseSet<mlir::Value>& external_use_values,
    llvm::DenseMap<mlir::Value, int>& internal_uses, bool& has_branch) {
  int dag_inputs = 0;
  for (mlir::Value opnd : op->getOperands()) {
    if (dag_values.contains(opnd)) {
      ++dag_inputs;
      int& ext = remaining_external_uses[opnd];
      --ext;
      if (ext == 0) {
        external_use_values.erase(opnd);
      }
      int& in_use = internal_uses[opnd];
      ++in_use;
      if (in_use > 1) {
        has_branch = true;
      }
    }
  }
  if (dag_inputs > 1) {
    has_branch = true;
  }
}

// Matches a single entry (`entry_val`), single exit (`exit_val`) pure
// elementwise DAG where every non entry operand is a scalar splat or broadcast,
// and no intermediate value escapes outside the DAG.
std::optional<PureElementwiseDagMatch> MatchSingleEntrySingleExitElementwiseDag(
    mlir::Value entry_val) {
  mlir::RankedTensorType entry_type;
  if (!IsStaticRankedTensor(entry_val.getType(), &entry_type) ||
      entry_val.use_empty()) {
    return std::nullopt;
  }
  mlir::Block* block = entry_val.getParentBlock();
  if (!block) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> expected_shape = entry_type.getShape();

  llvm::DenseSet<mlir::Value> dag_values;
  dag_values.insert(entry_val);

  llvm::DenseSet<mlir::Operation*> dag_ops_set;
  llvm::SmallVector<mlir::Operation*, 12> dag_ops;

  llvm::DenseMap<mlir::Value, int> remaining_external_uses;
  remaining_external_uses[entry_val] =
      std::distance(entry_val.use_begin(), entry_val.use_end());

  llvm::DenseSet<mlir::Value> external_use_values;
  if (remaining_external_uses[entry_val] > 0) {
    external_use_values.insert(entry_val);
  }

  llvm::DenseMap<mlir::Value, int> internal_uses;
  bool has_branch = false;
  std::optional<PureElementwiseDagMatch> best_match;

  llvm::SmallVector<mlir::Operation*, 8> ready_ops;
  llvm::DenseSet<mlir::Operation*> enqueued_ops;

  auto try_enqueue = [&ready_ops, &enqueued_ops, block, expected_shape,
                      &dag_ops_set, &dag_values](mlir::Operation* u) {
    if (IsCandidateDagOp(u, block, expected_shape, dag_ops_set, enqueued_ops,
                         dag_values)) {
      ready_ops.push_back(u);
      enqueued_ops.insert(u);
    }
  };

  for (mlir::Operation* u : entry_val.getUsers()) {
    try_enqueue(u);
  }

  while (!ready_ops.empty()) {
    mlir::Operation* next_op = PopEarliestOp(ready_ops);

    RecordDagOperandUses(next_op, dag_values, remaining_external_uses,
                         external_use_values, internal_uses, has_branch);

    dag_ops_set.insert(next_op);
    dag_ops.push_back(next_op);

    mlir::Value res = next_op->getResult(0);
    dag_values.insert(res);
    int res_total_uses = std::distance(res.use_begin(), res.use_end());
    remaining_external_uses[res] = res_total_uses;
    if (res_total_uses > 0) {
      external_use_values.insert(res);
    }

    if (!external_use_values.contains(entry_val) &&
        external_use_values.size() == 1) {
      mlir::Value exit_cand = *external_use_values.begin();
      best_match = PureElementwiseDagMatch{dag_ops, exit_cand, has_branch};
    }

    for (mlir::Operation* u : res.getUsers()) {
      try_enqueue(u);
    }
  }

  return best_match;
}

mlir::Value CloneElementwiseDagOnShape(mlir::OpBuilder& builder,
                                       mlir::Location loc,
                                       mlir::Value new_entry,
                                       llvm::ArrayRef<int64_t> target_shape,
                                       mlir::Value old_entry,
                                       const PureElementwiseDagMatch& match) {
  mlir::IRMapping mapping;
  mapping.map(old_entry, new_entry);

  for (mlir::Operation* op : match.ops) {
    for (mlir::Value opnd : op->getOperands()) {
      if (!mapping.contains(opnd)) {
        mlir::Value rebuilt =
            RebuildScalarSplatOrBroadcast(builder, loc, opnd, target_shape);
        mapping.map(opnd, rebuilt);
      }
    }
    mlir::Operation* cloned = builder.clone(*op, mapping);
    auto old_res_type =
        llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto new_res_type = mlir::RankedTensorType::get(
        target_shape, old_res_type.getElementType());
    cloned->getResult(0).setType(new_res_type);
    mapping.map(op->getResult(0), cloned->getResult(0));
  }
  return mapping.lookup(match.exit_val);
}

// =============================================================================
// Mechanism 6: Fold Transpose(BroadcastInDim(x, bdims), perm) -> BroadcastInDim
// Algebraic identity:
//   Transpose(BroadcastInDim(x, bdims), perm) = BroadcastInDim(x,
//   inv_perm[bdims]) when inv_perm[bdims] is strictly ascending.
// =============================================================================
bool TryFoldTransposeOfBroadcast(mlir::stablehlo::TransposeOp tr_op) {
  auto bcast_op =
      tr_op.getOperand().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast_op) {
    return false;
  }
  auto perm = tr_op.getPermutation();
  const int64_t rank = perm.size();
  llvm::SmallVector<int64_t, 6> inv_perm(rank, -1);
  for (int64_t i = 0; i < rank; ++i) {
    if (perm[i] < 0 || perm[i] >= rank) {
      return false;
    }
    inv_perm[perm[i]] = i;
  }
  auto old_bdims = bcast_op.getBroadcastDimensions();
  llvm::SmallVector<int64_t, 6> new_bdims;
  new_bdims.reserve(old_bdims.size());
  for (int64_t d : old_bdims) {
    if (d < 0 || d >= rank) {
      return false;
    }
    new_bdims.push_back(inv_perm[d]);
  }
  for (size_t i = 1; i < new_bdims.size(); ++i) {
    if (new_bdims[i] <= new_bdims[i - 1]) {
      return false;
    }
  }
  mlir::OpBuilder builder(tr_op);
  mlir::Value folded = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      tr_op.getLoc(), tr_op.getType(), bcast_op.getOperand(),
      builder.getDenseI64ArrayAttr(new_bdims));
  tr_op.replaceAllUsesWith(folded);
  tr_op.erase();
  if (bcast_op->use_empty()) {
    bcast_op.erase();
  }
  return true;
}

// =============================================================================
// Mechanism 4: Commute Pointwise DAGs Across Channel Transposes
// Algebraic identity:
//   F(Pi(x)) = Pi(F(x)) for any pure elementwise DAG F and permutation Pi.
// =============================================================================
bool TryCommutePointwiseDagAcrossTranspose(mlir::stablehlo::TransposeOp tr_op) {
  if (!tr_op || !tr_op->getBlock() || tr_op->use_empty()) {
    return false;
  }
  mlir::Value y = tr_op.getResult();
  mlir::RankedTensorType y_type;
  if (!IsStaticRankedTensor(y.getType(), &y_type)) {
    return false;
  }

  mlir::Value tr_in = tr_op.getOperand();
  mlir::RankedTensorType x_type;
  if (!IsStaticRankedTensor(tr_in.getType(), &x_type)) {
    return false;
  }

  auto dag_match = MatchSingleEntrySingleExitElementwiseDag(y);
  if (!dag_match || dag_match->ops.empty()) {
    return false;
  }

  mlir::Operation* exit_op = dag_match->exit_val.getDefiningOp();
  if (!exit_op) {
    return false;
  }

  auto entry_type = llvm::cast<mlir::RankedTensorType>(y.getType());
  auto exit_type =
      llvm::cast<mlir::RankedTensorType>(dag_match->exit_val.getType());
  // COST_MODEL: bitwidth(exit) <= bitwidth(entry)
  // Commuting an elementwise DAG across a transpose before the rearrangement
  // increases total data volume if the exit element type is wider than entry
  // (e.g. widening f16 -> f32). Only commute if bandwidth is non-increasing.
  auto exit_bw = GetTensorElementBitWidth(exit_type);
  auto entry_bw = GetTensorElementBitWidth(entry_type);
  if (!exit_bw.has_value() || !entry_bw.has_value() || *exit_bw > *entry_bw) {
    return false;
  }

  // Anchor builder at DAG exit op so both tr_in and all DAG operands dominate.
  mlir::OpBuilder builder(exit_op);
  mlir::Location loc = tr_op.getLoc();

  mlir::Value f_of_x = CloneElementwiseDagOnShape(
      builder, loc, tr_in, x_type.getShape(), y, *dag_match);

  auto trans_out_type = mlir::RankedTensorType::get(y_type.getShape(),
                                                    exit_type.getElementType());
  mlir::Value new_trans = builder.create<mlir::stablehlo::TransposeOp>(
      loc, trans_out_type, f_of_x, tr_op.getPermutationAttr());

  dag_match->exit_val.replaceAllUsesWith(new_trans);

  for (auto it = dag_match->ops.rbegin(); it != dag_match->ops.rend(); ++it) {
    (*it)->erase();
  }
  if (tr_op->use_empty()) {
    tr_op.erase();
  }
  return true;
}

// =============================================================================
// Mechanism 5: Unwrap Grouped Query Attention Broadcasts to Bitcast
// Algebraic identity:
//   DotGeneral(Q, Broadcast(K, [H_q])) -> Reshape(DotGeneral(Reshape(Q,
//   [H_kv]), K))
// =============================================================================
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
  // CAPABILITY: Excludes rank-3 transposed K matrices; requires rank >= 4 with
  // transposed trailing 2 dimensions (e.g. [..., S, D] -> [..., D, S]).
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
  bool transposed = CheckTransposeLastTwo(val, seen_convert);

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
  // CAPABILITY: GQA requires group_size >= 2 (group_size == 1 is MHA, handled
  // by standard dot lowerings).
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
  // CAPABILITY: Canonical GQA layout requires rank >= 3 with identical static
  // shapes and ranks between LHS and RHS.
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
  for (int64_t b : lhs_batch.drop_back()) {
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
  const int64_t eff_c_rhs =
      rhs_exp->transposed_last_two
          ? ((c_rhs == rank - 1) ? (rank - 2) : (rank - 1))
          : c_rhs;

  mlir::OpBuilder builder(dot_op);
  mlir::Location loc = dot_op.getLoc();
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
// Mechanism 3: Commute Token Permutation Across Linear Projections
// Algebraic identity:
//   Pi(XW + b) = Pi(X)W + b
// Cost formula:
//   Moving permutation upstream is volume neutral or volume reducing when
//   size_in_bytes(X) <= size_in_bytes(Y) (i.e. c_in <= c_out for equal dtypes).
// =============================================================================
bool VerifyBiasBroadcast(mlir::Value bias_bcast, int64_t c_out) {
  auto bcast = bias_bcast.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast) {
    return IsScalarSplatOrBroadcast(bias_bcast);
  }
  llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
  if (bdims.empty()) {
    return true;
  }
  if (bdims.size() == 1 && bdims[0] == 1) {
    auto bias_ty =
        llvm::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
    return bias_ty && bias_ty.hasStaticShape() && bias_ty.getRank() == 1 &&
           bias_ty.getDimSize(0) == c_out;
  }
  return false;
}

bool MatchLinearProjection(mlir::Value proj_out, mlir::Operation*& dot_op,
                           mlir::Value& dot_lhs, mlir::Value& dot_rhs,
                           mlir::Value& dot_val,
                           mlir::stablehlo::AddOp& bias_add,
                           mlir::Value& bias_bcast) {
  if (!proj_out.hasOneUse()) {
    return false;
  }
  dot_val = proj_out;
  if (auto add = proj_out.getDefiningOp<mlir::stablehlo::AddOp>()) {
    if (add.getLhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      bias_bcast = add.getLhs();
      dot_val = add.getRhs();
      bias_add = add;
    } else if (add.getRhs()
                   .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      bias_bcast = add.getRhs();
      dot_val = add.getLhs();
      bias_add = add;
    }
  }
  if (bias_add && !dot_val.hasOneUse()) {
    return false;
  }
  dot_op = dot_val.getDefiningOp();
  if (!dot_op) {
    return false;
  }
  if (auto dot = llvm::dyn_cast<mlir::stablehlo::DotOp>(dot_op)) {
    dot_lhs = dot.getLhs();
    dot_rhs = dot.getRhs();
    return true;
  }
  if (auto dg = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(dot_op)) {
    auto dnums = dg.getDotDimensionNumbers();
    if (!dnums.getLhsBatchingDimensions().empty() ||
        !dnums.getRhsBatchingDimensions().empty()) {
      return false;
    }
    if (dnums.getLhsContractingDimensions() != llvm::ArrayRef<int64_t>{1} ||
        dnums.getRhsContractingDimensions() != llvm::ArrayRef<int64_t>{0}) {
      return false;
    }
    dot_lhs = dg.getLhs();
    dot_rhs = dg.getRhs();
    return true;
  }
  return false;
}

void EmitCommutedLinearProjection(
    mlir::stablehlo::TransposeOp tr_op, mlir::stablehlo::ReshapeOp r_in,
    mlir::stablehlo::ReshapeOp r_out, mlir::Operation* dot_op,
    mlir::Value dot_rhs, mlir::Value dot_val, mlir::stablehlo::AddOp bias_add,
    mlir::Value bias_bcast, mlir::Value x_src, mlir::RankedTensorType r_in_type,
    mlir::RankedTensorType tr_type, mlir::RankedTensorType r_out_type,
    mlir::RankedTensorType lhs_ty, mlir::RankedTensorType proj_ty,
    int64_t c_in) {
  const int64_t rank = r_in_type.getRank();
  mlir::OpBuilder builder(r_out);
  mlir::Location loc = tr_op.getLoc();

  llvm::SmallVector<int64_t> pre_tr_shape(r_in_type.getShape().begin(),
                                          r_in_type.getShape().end());
  pre_tr_shape[rank - 1] = c_in;
  llvm::SmallVector<int64_t> post_tr_shape(tr_type.getShape().begin(),
                                           tr_type.getShape().end());
  post_tr_shape[rank - 1] = c_in;

  auto pre_tr_type =
      mlir::RankedTensorType::get(pre_tr_shape, lhs_ty.getElementType());
  auto post_tr_type =
      mlir::RankedTensorType::get(post_tr_shape, lhs_ty.getElementType());

  mlir::Value resh_in =
      builder.create<mlir::stablehlo::ReshapeOp>(loc, pre_tr_type, x_src);
  mlir::Value permuted_x = builder.create<mlir::stablehlo::TransposeOp>(
      loc, post_tr_type, resh_in, tr_op.getPermutationAttr());
  mlir::Value new_lhs_2d =
      builder.create<mlir::stablehlo::ReshapeOp>(loc, lhs_ty, permuted_x);

  mlir::Value new_proj;
  if (auto dg = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(dot_op)) {
    new_proj = builder.create<mlir::stablehlo::DotGeneralOp>(
        loc, proj_ty, new_lhs_2d, dot_rhs, dg.getDotDimensionNumbersAttr(),
        dg.getPrecisionConfigAttr(), dg.getAlgorithmAttr());
  } else {
    auto dot = llvm::cast<mlir::stablehlo::DotOp>(dot_op);
    new_proj = builder.create<mlir::stablehlo::DotOp>(
        loc, proj_ty, new_lhs_2d, dot_rhs, dot.getPrecisionConfigAttr());
  }

  if (bias_add) {
    mlir::Value add_lhs =
        (bias_add.getLhs() == dot_val) ? new_proj : bias_bcast;
    mlir::Value add_rhs =
        (bias_add.getLhs() == dot_val) ? bias_bcast : new_proj;
    new_proj =
        builder.create<mlir::stablehlo::AddOp>(loc, proj_ty, add_lhs, add_rhs);
  }

  mlir::Value new_out = builder.create<mlir::stablehlo::ReshapeOp>(
      r_out.getLoc(), r_out_type, new_proj);
  r_out.replaceAllUsesWith(new_out);
  r_out.erase();
  tr_op.erase();
  r_in.erase();
  if (bias_add) bias_add.erase();
  dot_op->erase();
}

bool TryCommuteTokenPermutationAcrossLinear(
    mlir::stablehlo::TransposeOp tr_op) {
  if (!tr_op || !tr_op->hasOneUse()) {
    return false;
  }
  auto r_in = tr_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
  auto r_out = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(*tr_op->user_begin());
  if (!r_in || !r_in->hasOneUse() || !r_out) {
    return false;
  }

  auto r_in_type = llvm::dyn_cast<mlir::RankedTensorType>(r_in.getType());
  auto tr_type = llvm::dyn_cast<mlir::RankedTensorType>(tr_op.getType());
  auto r_out_type = llvm::dyn_cast<mlir::RankedTensorType>(r_out.getType());
  if (!r_in_type || !r_in_type.hasStaticShape() || !tr_type ||
      !tr_type.hasStaticShape() || !r_out_type ||
      !r_out_type.hasStaticShape()) {
    return false;
  }

  mlir::Value proj_out = r_in.getOperand();
  mlir::Operation* dot_op = nullptr;
  mlir::Value dot_lhs;
  mlir::Value dot_rhs;
  mlir::Value dot_val;
  mlir::stablehlo::AddOp bias_add;
  mlir::Value bias_bcast;

  if (!MatchLinearProjection(proj_out, dot_op, dot_lhs, dot_rhs, dot_val,
                             bias_add, bias_bcast)) {
    return false;
  }

  auto lhs_ty = llvm::dyn_cast<mlir::RankedTensorType>(dot_lhs.getType());
  auto rhs_ty = llvm::dyn_cast<mlir::RankedTensorType>(dot_rhs.getType());
  auto proj_ty = llvm::dyn_cast<mlir::RankedTensorType>(dot_val.getType());
  if (!lhs_ty || !rhs_ty || !proj_ty || !lhs_ty.hasStaticShape() ||
      !rhs_ty.hasStaticShape() || lhs_ty.getRank() != 2 ||
      proj_ty.getRank() != 2) {
    return false;
  }

  const int64_t n_tokens = lhs_ty.getDimSize(0);
  const int64_t c_in = lhs_ty.getDimSize(1);
  const int64_t c_out = proj_ty.getDimSize(1);
  if (n_tokens <= 1 || c_in <= 0 || c_out <= 0) {
    return false;
  }

  if (bias_add && !VerifyBiasBroadcast(bias_bcast, c_out)) {
    return false;
  }

  // Cost formula: element bits moved before dot must not exceed bits moved
  // after dot (c_in * elem_bits <= c_out * elem_bits).
  auto lhs_elem_bits = GetTensorElementBitWidth(lhs_ty);
  auto proj_elem_bits = GetTensorElementBitWidth(proj_ty);
  if (!lhs_elem_bits.has_value() || !proj_elem_bits.has_value()) {
    return false;
  }
  if (c_in * (*lhs_elem_bits) > c_out * (*proj_elem_bits)) {
    return false;
  }

  llvm::ArrayRef<int64_t> perm = tr_op.getPermutation();
  const int64_t rank = r_in_type.getRank();
  if (static_cast<int64_t>(perm.size()) != rank || perm[rank - 1] != rank - 1 ||
      r_in_type.getDimSize(rank - 1) != c_out) {
    return false;
  }

  auto dot_lhs_resh = dot_lhs.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!dot_lhs_resh) return false;
  mlir::Value x_src = dot_lhs_resh.getOperand();
  auto x_src_ty = llvm::dyn_cast<mlir::RankedTensorType>(x_src.getType());
  if (!x_src_ty || !x_src_ty.hasStaticShape()) return false;

  EmitCommutedLinearProjection(tr_op, r_in, r_out, dot_op, dot_rhs, dot_val,
                               bias_add, bias_bcast, x_src, r_in_type, tr_type,
                               r_out_type, lhs_ty, proj_ty, c_in);
  if (dot_lhs_resh->use_empty()) {
    dot_lhs_resh.erase();
  }
  return true;
}

// =============================================================================
// Mechanism 2: Flatten Spatial Reduction (commute leading axis collapse)
// Algebraic identity:
//   Collapse(LeadingReduction(X)) = LeadingReduction(Collapse(X))
// =============================================================================
bool CollectSpatialReductionSubgraph(
    mlir::stablehlo::ReshapeOp flatten_op, mlir::Block* block,
    llvm::ArrayRef<int64_t> in_shape, int64_t rank,
    llvm::SetVector<mlir::Operation*>& subgraph_ops,
    llvm::SmallVectorImpl<mlir::Operation*>& order,
    llvm::SetVector<mlir::Value>& inputs_to_flatten,
    bool& found_trailing_reduce) {
  llvm::SmallVector<mlir::Value, 8> worklist;
  worklist.push_back(flatten_op.getOperand());

  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    if (IsScalarSplatOrBroadcast(v)) {
      continue;
    }
    if (auto bcast = v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      if (bcast.getBroadcastDimensions().size() == 1 &&
          bcast.getBroadcastDimensions()[0] == rank - 1) {
        continue;
      }
    }
    mlir::Operation* def = v.getDefiningOp();
    if (!def || def->getBlock() != block) {
      if (auto ty = llvm::dyn_cast<mlir::RankedTensorType>(v.getType())) {
        if (ty.getShape() == in_shape) {
          inputs_to_flatten.insert(v);
          continue;
        }
      }
      return false;
    }
    if (subgraph_ops.contains(def)) {
      continue;
    }

    if (auto red = llvm::dyn_cast<mlir::stablehlo::ReduceOp>(def)) {
      llvm::ArrayRef<int64_t> red_dims = red.getDimensions();
      if (red_dims.size() != 1 || red_dims[0] != rank - 1) {
        return false;
      }
      found_trailing_reduce = true;
      subgraph_ops.insert(def);
      order.push_back(def);
      for (mlir::Value in : red.getInputs()) {
        worklist.push_back(in);
      }
      continue;
    }

    if (auto bcast = llvm::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(def)) {
      llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
      if (static_cast<int64_t>(bdims.size()) == rank - 1) {
        bool leading_match = true;
        for (int64_t i = 0; i < rank - 1; ++i) {
          if (bdims[i] != i) {
            leading_match = false;
            break;
          }
        }
        if (leading_match) {
          subgraph_ops.insert(def);
          order.push_back(def);
          worklist.push_back(bcast.getOperand());
          continue;
        }
      }
      return false;
    }

    if (IsSupportedPureElementwiseOp(def)) {
      subgraph_ops.insert(def);
      order.push_back(def);
      for (mlir::Value opnd : def->getOperands()) {
        worklist.push_back(opnd);
      }
      continue;
    }

    auto ty = llvm::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (ty && ty.getShape() == in_shape) {
      inputs_to_flatten.insert(v);
      continue;
    }
    return false;
  }
  return true;
}

bool ValidateSpatialReductionSubgraph(
    mlir::stablehlo::ReshapeOp flatten_op,
    const llvm::SetVector<mlir::Operation*>& subgraph_ops,
    const llvm::SetVector<mlir::Value>& inputs_to_flatten,
    llvm::ArrayRef<int64_t> in_shape, int64_t rank, int64_t trailing_channels) {
  llvm::ArrayRef<int64_t> reduced_shape = in_shape.drop_back();
  for (mlir::Operation* op : subgraph_ops) {
    for (mlir::Value res : op->getResults()) {
      for (mlir::Operation* user : res.getUsers()) {
        if (user != flatten_op && !subgraph_ops.contains(user)) {
          return false;
        }
      }
    }

    if (auto red = llvm::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
      for (mlir::Value in : red.getInputs()) {
        auto in_ty = llvm::dyn_cast<mlir::RankedTensorType>(in.getType());
        if (!in_ty || in_ty.getShape() != in_shape) {
          return false;
        }
        if (!inputs_to_flatten.contains(in) &&
            !subgraph_ops.contains(in.getDefiningOp())) {
          return false;
        }
      }
      continue;
    }

    if (auto bcast = llvm::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
      auto opnd_ty =
          llvm::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
      if (!opnd_ty || opnd_ty.getShape() != reduced_shape) {
        return false;
      }
      if (!subgraph_ops.contains(bcast.getOperand().getDefiningOp())) {
        return false;
      }
      continue;
    }

    if (IsSupportedPureElementwiseOp(op)) {
      auto res_ty =
          llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
      llvm::ArrayRef<int64_t> res_shape = res_ty.getShape();
      if (res_shape == reduced_shape) {
        for (mlir::Value opnd : op->getOperands()) {
          if (IsScalarSplatOrBroadcast(opnd)) {
            continue;
          }
          auto opnd_ty = llvm::dyn_cast<mlir::RankedTensorType>(opnd.getType());
          if (!opnd_ty || opnd_ty.getShape() != reduced_shape ||
              !subgraph_ops.contains(opnd.getDefiningOp())) {
            return false;
          }
        }
      } else if (res_shape == in_shape) {
        for (mlir::Value opnd : op->getOperands()) {
          if (IsScalarSplatOrBroadcast(opnd)) {
            continue;
          }
          if (auto bcast =
                  opnd.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
            if (bcast.getBroadcastDimensions().size() == 1 &&
                bcast.getBroadcastDimensions()[0] == rank - 1) {
              auto p_ty = llvm::dyn_cast<mlir::RankedTensorType>(
                  bcast.getOperand().getType());
              if (p_ty && p_ty.getRank() == 1 &&
                  p_ty.getDimSize(0) == trailing_channels) {
                continue;
              }
            }
          }
          auto opnd_ty = llvm::dyn_cast<mlir::RankedTensorType>(opnd.getType());
          if (!opnd_ty || opnd_ty.getShape() != in_shape) {
            return false;
          }
          if (!inputs_to_flatten.contains(opnd) &&
              !subgraph_ops.contains(opnd.getDefiningOp())) {
            return false;
          }
        }
      } else {
        return false;
      }
    }
  }
  return true;
}

void EmitFlattenedSpatialReduction(
    mlir::stablehlo::ReshapeOp flatten_op,
    llvm::SmallVectorImpl<mlir::Operation*>& order,
    const llvm::SetVector<mlir::Value>& inputs_to_flatten,
    llvm::ArrayRef<int64_t> in_shape, int64_t rank, int64_t collapsed_rows,
    int64_t trailing_channels) {
  mlir::OpBuilder builder(flatten_op);
  mlir::Location loc = flatten_op.getLoc();
  mlir::IRMapping mapping;
  llvm::ArrayRef<int64_t> reduced_shape = in_shape.drop_back();

  for (mlir::Value in_v : inputs_to_flatten) {
    auto flat_ty = mlir::RankedTensorType::get(
        {collapsed_rows, trailing_channels},
        llvm::cast<mlir::RankedTensorType>(in_v.getType()).getElementType());
    mlir::Value flat_in =
        builder.create<mlir::stablehlo::ReshapeOp>(loc, flat_ty, in_v);
    mapping.map(in_v, flat_in);
  }

  llvm::sort(order, [](mlir::Operation* a, mlir::Operation* b) {
    return a->isBeforeInBlock(b);
  });

  for (mlir::Operation* op : order) {
    if (auto red = llvm::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
      llvm::SmallVector<mlir::Value> new_inputs;
      for (mlir::Value in : red.getInputs()) {
        new_inputs.push_back(mapping.lookup(in));
      }
      llvm::SmallVector<mlir::Type> new_res_types;
      for (mlir::Type t : red.getResultTypes()) {
        auto ranked = llvm::cast<mlir::RankedTensorType>(t);
        new_res_types.push_back(mlir::RankedTensorType::get(
            {collapsed_rows}, ranked.getElementType()));
      }
      auto new_red = builder.create<mlir::stablehlo::ReduceOp>(
          loc, new_res_types, new_inputs, red.getInitValues(),
          builder.getDenseI64ArrayAttr({1}));
      mlir::IRMapping body_map;
      red.getBody().cloneInto(&new_red.getBody(), body_map);
      for (unsigned i = 0; i < red.getNumResults(); ++i) {
        mapping.map(red.getResult(i), new_red.getResult(i));
      }
      continue;
    }

    if (auto bcast = llvm::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
      mlir::Value mapped_in = mapping.lookup(bcast.getOperand());
      auto out_elem_ty =
          llvm::cast<mlir::RankedTensorType>(bcast.getType()).getElementType();
      auto new_bcast_ty = mlir::RankedTensorType::get(
          {collapsed_rows, trailing_channels}, out_elem_ty);
      auto new_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, new_bcast_ty, mapped_in, builder.getDenseI64ArrayAttr({0}));
      mapping.map(bcast.getResult(), new_bcast.getResult());
      continue;
    }

    auto old_ty =
        llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    bool is_reduced_shape = (old_ty.getShape() == reduced_shape);
    llvm::SmallVector<int64_t, 2> new_target_shape =
        is_reduced_shape
            ? llvm::SmallVector<int64_t, 2>{collapsed_rows}
            : llvm::SmallVector<int64_t, 2>{collapsed_rows, trailing_channels};

    llvm::SmallVector<mlir::Value> new_operands;
    for (mlir::Value opnd : op->getOperands()) {
      if (mapping.contains(opnd)) {
        new_operands.push_back(mapping.lookup(opnd));
      } else if (IsScalarSplatOrBroadcast(opnd)) {
        new_operands.push_back(RebuildScalarSplatOrBroadcast(builder, loc, opnd,
                                                             new_target_shape));
      } else if (auto bcast =
                     opnd.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
        auto param_ty =
            llvm::cast<mlir::RankedTensorType>(bcast.getOperand().getType());
        auto new_bcast_ty = mlir::RankedTensorType::get(
            {collapsed_rows, trailing_channels}, param_ty.getElementType());
        auto new_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, new_bcast_ty, bcast.getOperand(),
            builder.getDenseI64ArrayAttr({1}));
        new_operands.push_back(new_bcast.getResult());
      }
    }
    auto new_res_ty =
        mlir::RankedTensorType::get(new_target_shape, old_ty.getElementType());
    mlir::OperationState state(loc, op->getName().getStringRef(), new_operands,
                               {new_res_ty}, op->getAttrs());
    mlir::Operation* cloned = builder.create(state);
    mapping.map(op->getResult(0), cloned->getResult(0));
  }

  mlir::Value final_res = mapping.lookup(flatten_op.getOperand());
  flatten_op.replaceAllUsesWith(final_res);
  flatten_op.erase();
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    (*it)->erase();
  }
}

bool TryFlattenSpatialReduction(mlir::stablehlo::ReshapeOp flatten_op) {
  if (!flatten_op || !flatten_op->getBlock()) return false;

  mlir::RankedTensorType in_type;
  mlir::RankedTensorType out_type;
  if (!IsStaticRankedTensor(flatten_op.getOperand().getType(), &in_type) ||
      !IsStaticRankedTensor(flatten_op.getType(), &out_type)) {
    return false;
  }
  const int64_t rank = in_type.getRank();
  // CAPABILITY: Limited to full-collapse leading dimensions [D0, D1, ..., C] ->
  // [prod(D0..), C] with out_rank == 2.
  if (rank < 3 || out_type.getRank() != 2) {
    return false;
  }
  const int64_t trailing_channels = in_type.getDimSize(rank - 1);
  const int64_t collapsed_rows = out_type.getDimSize(0);
  if (trailing_channels <= 0 || collapsed_rows <= 0 ||
      out_type.getDimSize(1) != trailing_channels) {
    return false;
  }

  int64_t leading_prod = 1;
  for (int64_t i = 0; i < rank - 1; ++i) {
    leading_prod *= in_type.getDimSize(i);
  }
  if (leading_prod != collapsed_rows) {
    return false;
  }

  mlir::Block* block = flatten_op->getBlock();
  llvm::ArrayRef<int64_t> in_shape = in_type.getShape();

  llvm::SetVector<mlir::Operation*> subgraph_ops;
  llvm::SmallVector<mlir::Operation*, 16> order;
  llvm::SetVector<mlir::Value> inputs_to_flatten;
  bool found_trailing_reduce = false;

  if (!CollectSpatialReductionSubgraph(flatten_op, block, in_shape, rank,
                                       subgraph_ops, order, inputs_to_flatten,
                                       found_trailing_reduce)) {
    return false;
  }

  if (!found_trailing_reduce || inputs_to_flatten.empty()) {
    return false;
  }

  if (!ValidateSpatialReductionSubgraph(flatten_op, subgraph_ops,
                                        inputs_to_flatten, in_shape, rank,
                                        trailing_channels)) {
    return false;
  }

  EmitFlattenedSpatialReduction(flatten_op, order, inputs_to_flatten, in_shape,
                                rank, collapsed_rows, trailing_channels);
  return true;
}

// =============================================================================
// Mechanism 1: Sink Projection Reshapes Across Elementwise Chains
// Algebraic identity:
//   Pi(E(Dot(X, W), b)) = E(Pi(Dot(X, W)), Pi(b))
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
  mlir::Location loc = reshape_op.getLoc();

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
bool RunFoldTransposeOfBroadcast(mlir::func::FuncOp func) {
  bool changed = false;
  llvm::SmallVector<mlir::stablehlo::TransposeOp> transposes;
  func.walk([&transposes](mlir::stablehlo::TransposeOp op) {
    transposes.push_back(op);
  });
  for (auto op : transposes) {
    if (op && op->getBlock() && !op->use_empty()) {
      changed |= TryFoldTransposeOfBroadcast(op);
    }
  }
  return changed;
}

bool RunCommutePointwiseDagAcrossTranspose(mlir::func::FuncOp func) {
  bool changed = false;
  llvm::SmallVector<mlir::stablehlo::TransposeOp> transposes;
  func.walk([&transposes](mlir::stablehlo::TransposeOp op) {
    transposes.push_back(op);
  });
  for (auto op : transposes) {
    if (op && op->getBlock() && !op->use_empty()) {
      changed |= TryCommutePointwiseDagAcrossTranspose(op);
    }
  }
  return changed;
}

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

bool RunCommuteTokenPermutationAcrossLinear(mlir::func::FuncOp func) {
  bool changed = false;
  llvm::SmallVector<mlir::stablehlo::TransposeOp> transposes;
  func.walk([&transposes](mlir::stablehlo::TransposeOp op) {
    transposes.push_back(op);
  });
  for (auto op : transposes) {
    if (op && op->getBlock() && !op->use_empty()) {
      changed |= TryCommuteTokenPermutationAcrossLinear(op);
    }
  }
  return changed;
}

bool RunFlattenSpatialReduction(mlir::func::FuncOp func) {
  bool changed = false;
  llvm::SmallVector<mlir::stablehlo::ReshapeOp> reshapes;
  func.walk(
      [&reshapes](mlir::stablehlo::ReshapeOp op) { reshapes.push_back(op); });
  for (auto op : reshapes) {
    if (op && op->getBlock() && !op->use_empty()) {
      changed |= TryFlattenSpatialReduction(op);
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
  changed |= RunFoldTransposeOfBroadcast(func);
  changed |= RunCommutePointwiseDagAcrossTranspose(func);
  changed |= RunUnwrapGqaBroadcast(func);
  changed |= RunCommuteTokenPermutationAcrossLinear(func);
  changed |= RunFlattenSpatialReduction(func);
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
    return "Commutes reshapes, transposes, and permutations across elementwise "
           "DAGs and linear contractions so they cancel or fold.";
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
    // A fixed-point iteration cap of 10 prevents infinite looping on cyclic
    // rewrite sequences. If the 10th iteration still produced changes, check if
    // an 11th iteration is a no-op (converged) or still changing after the cap.
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
