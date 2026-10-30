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

#include "csrc/mlir/passes/optimize_sdpa_and_softmax_reductions.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/log/absl_check.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// GENERIC: hardware TPU vector register lane count and MXU systolic array
// contracting width is 128; XLA FlashAttention online-softmax requires sequence
// dimensions aligned to 128 and head_dim fitting within a 128-lane vector tile.
constexpr int64_t kTpuLaneWidth = 128;

// GENERIC: hardware TPU vector register sublane tile height is 8.
constexpr int64_t kTpuSublaneCount = 8;

// GENERIC: cost-model + hardware TPU 16 MiB VMEM capacity (4M f32 elements);
// 4D f32 attention matrices reaching 16 MiB spill to HBM unless streamed via
// online-softmax or kept as 1D reductions.
constexpr int64_t kTpuScopedVmemLimitBytes = 16 * 1024 * 1024;

bool IsDropLastDimBroadcast(llvm::ArrayRef<int64_t> dims, int64_t rank) {
  if (static_cast<int64_t>(dims.size()) != rank - 1) {
    return false;
  }
  for (int64_t i = 0; i < rank - 1; ++i) {
    if (dims[i] != i) {
      return false;
    }
  }
  return true;
}

bool IsIdentityBroadcastDims(llvm::ArrayRef<int64_t> dims, int64_t rank) {
  if (static_cast<int64_t>(dims.size()) != rank) {
    return false;
  }
  for (int64_t i = 0; i < rank; ++i) {
    if (dims[i] != i) {
      return false;
    }
  }
  return true;
}

mlir::DenseI64ArrayAttr GetDropLastDimBroadcastAttr(mlir::OpBuilder& builder,
                                                    int64_t rank) {
  ABSL_DCHECK_GE(rank, 1);
  llvm::SmallVector<int64_t, 4> dims;
  dims.reserve(rank - 1);
  for (int64_t i = 0; i < rank - 1; ++i) {
    dims.push_back(i);
  }
  return builder.getDenseI64ArrayAttr(dims);
}

bool MatchesPrefixOrUnitLastDim(mlir::RankedTensorType type, int64_t rank,
                                llvm::ArrayRef<int64_t> prefix_shape) {
  if (!type || !type.hasStaticShape()) {
    return false;
  }
  if (type.getRank() == rank - 1) {
    return type.getShape() == prefix_shape;
  }
  if (type.getRank() == rank && type.getDimSize(rank - 1) == 1) {
    return type.getShape().drop_back(1) == prefix_shape;
  }
  return false;
}

mlir::Value StepPeelBroadcastInDim(mlir::stablehlo::BroadcastInDimOp bcast,
                                   int64_t rank,
                                   llvm::ArrayRef<int64_t> prefix_shape,
                                   int64_t last_dim, bool& saw_bcast) {
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
  if (!in_type || !out_type || !in_type.hasStaticShape() ||
      !out_type.hasStaticShape()) {
    return nullptr;
  }
  llvm::ArrayRef<int64_t> dims = bcast.getBroadcastDimensions();
  if (out_type.getRank() == rank && in_type.getRank() == rank - 1 &&
      IsDropLastDimBroadcast(dims, rank) &&
      in_type.getShape() == prefix_shape) {
    saw_bcast = true;
    return bcast.getOperand();
  }
  if (out_type.getRank() == rank && in_type.getRank() == rank &&
      IsIdentityBroadcastDims(dims, rank) &&
      MatchesPrefixOrUnitLastDim(in_type, rank, prefix_shape)) {
    if (out_type.getDimSize(rank - 1) == last_dim) {
      saw_bcast = true;
    }
    return bcast.getOperand();
  }
  return nullptr;
}

mlir::Value StepPeelReshape(mlir::stablehlo::ReshapeOp resh, int64_t rank,
                            llvm::ArrayRef<int64_t> prefix_shape) {
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
  if (MatchesPrefixOrUnitLastDim(in_type, rank, prefix_shape)) {
    return resh.getOperand();
  }
  return nullptr;
}

mlir::Value PeelReshapeBroadcastChain(mlir::Value val,
                                      llvm::ArrayRef<int64_t> outer_shape) {
  const int64_t rank = outer_shape.size();
  if (rank < 2) {
    return nullptr;
  }
  llvm::ArrayRef<int64_t> prefix_shape = outer_shape.drop_back(1);
  const int64_t last_dim = outer_shape[rank - 1];
  mlir::Value cur = val;
  bool saw_bcast = false;
  while (true) {
    if (auto bcast = cur.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      if (mlir::Value next = StepPeelBroadcastInDim(bcast, rank, prefix_shape,
                                                    last_dim, saw_bcast)) {
        cur = next;
        continue;
      }
      break;
    }
    if (auto resh = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      if (mlir::Value next = StepPeelReshape(resh, rank, prefix_shape)) {
        cur = next;
        continue;
      }
      break;
    }
    break;
  }
  if (!saw_bcast) {
    return nullptr;
  }
  auto cur_type = mlir::dyn_cast<mlir::RankedTensorType>(cur.getType());
  if (!MatchesPrefixOrUnitLastDim(cur_type, rank, prefix_shape)) {
    return nullptr;
  }
  return cur;
}

template <typename BinaryOpTy>
bool IsSingleDimReduceWithOp(mlir::Value val, int64_t expected_dim,
                             mlir::stablehlo::ReduceOp* out_reduce = nullptr) {
  auto reduce_op = val.getDefiningOp<mlir::stablehlo::ReduceOp>();
  if (!reduce_op || reduce_op.getInputs().size() != 1 ||
      reduce_op.getResults().size() != 1 ||
      reduce_op.getInitValues().size() != 1) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = reduce_op.getDimensions();
  if (dims.size() != 1 || dims[0] != expected_dim) {
    return false;
  }
  mlir::Region& body = reduce_op.getBody();
  if (!body.hasOneBlock()) {
    return false;
  }
  mlir::Block& block = body.front();
  if (block.getNumArguments() != 2) {
    return false;
  }
  auto ret_op =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  if (!ret_op || ret_op.getOperands().size() != 1) {
    return false;
  }
  auto bin_op = ret_op.getOperands()[0].getDefiningOp<BinaryOpTy>();
  if (!bin_op) {
    return false;
  }
  mlir::Value arg0 = block.getArgument(0);
  mlir::Value arg1 = block.getArgument(1);
  mlir::Value lhs = bin_op.getOperand(0);
  mlir::Value rhs = bin_op.getOperand(1);
  if (!((lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0))) {
    return false;
  }
  if (out_reduce) {
    *out_reduce = reduce_op;
  }
  return true;
}

bool IsZeroSplatConstant(mlir::Value val) {
  while (true) {
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      val = bcast.getOperand();
      continue;
    }
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastOp>()) {
      val = bcast.getOperand();
      continue;
    }
    if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      val = conv.getOperand();
      continue;
    }
    break;
  }
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr || attr.empty() || !attr.isSplat() ||
      !mlir::isa<mlir::FloatType>(attr.getElementType())) {
    return false;
  }
  return attr.getSplatValue<llvm::APFloat>().isZero();
}

mlir::Value CreateSplatFloatConst(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::RankedTensorType type, float value) {
  auto float_type = mlir::cast<mlir::FloatType>(type.getElementType());
  llvm::APFloat apf(value);
  bool loses_info = false;
  apf.convert(float_type.getFloatSemantics(),
              llvm::APFloat::rmNearestTiesToEven, &loses_info);
  auto attr = mlir::DenseElementsAttr::get(type, apf);
  return builder.create<mlir::stablehlo::ConstantOp>(loc, type, attr);
}

mlir::Value ConvertIfNeeded(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value val, mlir::Type target_elem_type) {
  auto ranked = mlir::cast<mlir::RankedTensorType>(val.getType());
  if (ranked.getElementType() == target_elem_type) {
    return val;
  }
  auto new_type =
      mlir::RankedTensorType::get(ranked.getShape(), target_elem_type);
  return builder.create<mlir::stablehlo::ConvertOp>(loc, new_type, val);
}

bool MatchRowAllMaskedSelect(mlir::Value val, mlir::Value expected_on_false,
                             mlir::Value& out_pred_reduced,
                             mlir::stablehlo::SelectOp& out_select) {
  auto select_op = val.getDefiningOp<mlir::stablehlo::SelectOp>();
  if (!select_op) {
    return false;
  }
  auto sel_type = mlir::dyn_cast<mlir::RankedTensorType>(select_op.getType());
  if (!sel_type || !sel_type.hasStaticShape() || sel_type.getRank() < 3) {
    return false;
  }
  if (!IsZeroSplatConstant(select_op.getOnTrue()) ||
      select_op.getOnFalse() != expected_on_false) {
    return false;
  }
  mlir::Value pred_peeled =
      PeelReshapeBroadcastChain(select_op.getPred(), sel_type.getShape());
  if (!pred_peeled) {
    return false;
  }
  auto pred_type =
      mlir::dyn_cast<mlir::RankedTensorType>(pred_peeled.getType());
  if (!pred_type || pred_type.getRank() != sel_type.getRank() - 1) {
    return false;
  }
  out_pred_reduced = pred_peeled;
  out_select = select_op;
  return true;
}

bool HasBatchPermutingTransposeOperand(mlir::Value val, int depth = 0) {
  if (!val || depth > 6) {
    return false;
  }
  mlir::Operation* def = val.getDefiningOp();
  if (!def) {
    return false;
  }
  if (auto tr = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(def)) {
    llvm::ArrayRef<int64_t> perm = tr.getPermutation();
    if (!perm.empty() && perm[0] != 0) {
      return true;
    }
    return HasBatchPermutingTransposeOperand(tr.getOperand(), depth + 1);
  }
  if (mlir::isa<mlir::stablehlo::ConvertOp, mlir::stablehlo::ReshapeOp,
                mlir::stablehlo::SliceOp, mlir::stablehlo::BroadcastInDimOp,
                mlir::stablehlo::MulOp, mlir::stablehlo::AddOp,
                mlir::stablehlo::SelectOp, mlir::stablehlo::DotGeneralOp>(
          def)) {
    for (mlir::Value opnd : def->getOperands()) {
      if (HasBatchPermutingTransposeOperand(opnd, depth + 1)) {
        return true;
      }
    }
  }
  return false;
}

// Returns true when an unaligned multi-tile self-attention Div(exp, sum)
// fits within a single 8-sublane VMEM tile (and either the whole 4D f32
// tensor fits in the 16 MiB VMEM capacity or Q/K have a batch-permuting
// TransposeOp that blocks XLA conv-output fusion).
bool ShouldHoistUnalignedSelfAttentionReciprocal(mlir::RankedTensorType type,
                                                 mlir::Value masked_logits) {
  if (!type || !type.hasStaticShape() || type.getRank() < 3 ||
      !mlir::isa<mlir::FloatType>(type.getElementType())) {
    return false;
  }
  const int64_t rank = type.getRank();
  const int64_t q_len = type.getDimSize(rank - 2);
  const int64_t kv_len = type.getDimSize(rank - 1);
  if (q_len <= kTpuLaneWidth || q_len != kv_len ||
      (q_len % kTpuLaneWidth) == 0) {
    return false;
  }
  const int64_t f32_elem_bytes = static_cast<int64_t>(sizeof(float));
  const int64_t single_sublane_tile_bytes =
      kTpuSublaneCount * q_len * kv_len * f32_elem_bytes;
  if (single_sublane_tile_bytes > kTpuScopedVmemLimitBytes) {
    return false;
  }
  const int64_t total_f32_bytes = type.getNumElements() * f32_elem_bytes;
  return total_f32_bytes <= kTpuScopedVmemLimitBytes ||
         HasBatchPermutingTransposeOperand(masked_logits);
}

// Returns true when a pre-normalized softmax tensor (already using 3D
// reciprocal multiplication Mul(exp, Broadcast(Div(1, sum_exp)))) has a shape
// where XLA's PrefuseReduceBroadcastReuse creates an unaligned or VMEM-spilling
// reduce-window fusion:
// (a) unaligned multi-tile self-attention where one 8-sublane head tile fits
//     in the 16 MiB VMEM capacity and batch_heads >= 8 sublanes, or
// (b) 128-lane-aligned multi-tile self-attention whose 4D f32 tensor reaches
//     the 16 MiB VMEM capacity with either 8-sublane-aligned batch_heads or a
//     single 8-sublane head tile exceeding 16 MiB VMEM.
bool IsTargetPreNormalizedSoftmaxShape(mlir::RankedTensorType type) {
  if (!type || !type.hasStaticShape() || type.getRank() < 3 ||
      !mlir::isa<mlir::FloatType>(type.getElementType())) {
    return false;
  }
  const int64_t rank = type.getRank();
  const int64_t q_len = type.getDimSize(rank - 2);
  const int64_t kv_len = type.getDimSize(rank - 1);
  if (q_len <= kTpuLaneWidth || q_len != kv_len) {
    return false;
  }
  const int64_t total_elements = type.getNumElements();
  const int64_t batch_heads = total_elements / (q_len * kv_len);
  const int64_t f32_elem_bytes = static_cast<int64_t>(sizeof(float));
  const int64_t total_f32_bytes = total_elements * f32_elem_bytes;
  const int64_t single_sublane_tile_bytes =
      kTpuSublaneCount * q_len * kv_len * f32_elem_bytes;

  if ((kv_len % kTpuLaneWidth) != 0) {
    return batch_heads >= kTpuSublaneCount &&
           single_sublane_tile_bytes <= kTpuScopedVmemLimitBytes;
  }

  return total_f32_bytes >= kTpuScopedVmemLimitBytes &&
         ((batch_heads % kTpuSublaneCount) == 0 ||
          single_sublane_tile_bytes > kTpuScopedVmemLimitBytes);
}

// Match result for exp(logits - max(logits)) or exp(logits + neg(max(logits))).
// When matched via an already-rewritten AddOp, shifted_sub is null.
struct ShiftedLogitsMatch {
  mlir::Value masked_logits;
  mlir::stablehlo::ReduceOp max_reduce;
  mlir::stablehlo::SubtractOp shifted_sub;
};

std::optional<ShiftedLogitsMatch> MatchSubtractMaxLogits(
    mlir::stablehlo::SubtractOp sub_op) {
  auto sub_type = mlir::dyn_cast<mlir::RankedTensorType>(sub_op.getType());
  if (!sub_type || !sub_type.hasStaticShape() || sub_type.getRank() < 3) {
    return std::nullopt;
  }
  const int64_t rank = sub_type.getRank();
  mlir::Value max_peeled =
      PeelReshapeBroadcastChain(sub_op.getRhs(), sub_type.getShape());
  if (!max_peeled) {
    return std::nullopt;
  }
  mlir::stablehlo::ReduceOp max_reduce;
  if (!IsSingleDimReduceWithOp<mlir::stablehlo::MaxOp>(max_peeled, rank - 1,
                                                       &max_reduce)) {
    return std::nullopt;
  }
  if (max_reduce.getInputs()[0] != sub_op.getLhs()) {
    return std::nullopt;
  }
  return ShiftedLogitsMatch{sub_op.getLhs(), max_reduce, sub_op};
}

std::optional<ShiftedLogitsMatch> MatchAddNegMaxLogits(
    mlir::stablehlo::AddOp add_op) {
  auto add_type = mlir::dyn_cast<mlir::RankedTensorType>(add_op.getType());
  if (!add_type || !add_type.hasStaticShape() || add_type.getRank() < 3) {
    return std::nullopt;
  }
  const int64_t rank = add_type.getRank();
  mlir::Value neg_peeled =
      PeelReshapeBroadcastChain(add_op.getRhs(), add_type.getShape());
  if (!neg_peeled) {
    return std::nullopt;
  }
  auto neg_op = neg_peeled.getDefiningOp<mlir::stablehlo::NegOp>();
  if (!neg_op) {
    return std::nullopt;
  }
  mlir::stablehlo::ReduceOp max_reduce;
  if (!IsSingleDimReduceWithOp<mlir::stablehlo::MaxOp>(neg_op.getOperand(),
                                                       rank - 1, &max_reduce)) {
    return std::nullopt;
  }
  if (max_reduce.getInputs()[0] != add_op.getLhs()) {
    return std::nullopt;
  }
  return ShiftedLogitsMatch{add_op.getLhs(), max_reduce, nullptr};
}

std::optional<ShiftedLogitsMatch> MatchExpOfShiftedLogits(mlir::Value exp_val) {
  auto exp_op = exp_val.getDefiningOp<mlir::stablehlo::ExpOp>();
  if (!exp_op) {
    return std::nullopt;
  }
  mlir::Value logits = exp_op.getOperand();
  if (auto conv = logits.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    logits = conv.getOperand();
  }
  if (auto sub_op = logits.getDefiningOp<mlir::stablehlo::SubtractOp>()) {
    return MatchSubtractMaxLogits(sub_op);
  }
  if (auto add_op = logits.getDefiningOp<mlir::stablehlo::AddOp>()) {
    return MatchAddNegMaxLogits(add_op);
  }
  return std::nullopt;
}

void ReplaceSubtractWithAddNeg(mlir::stablehlo::SubtractOp sub_op,
                               mlir::Value reduced_rhs) {
  auto sub_type = mlir::cast<mlir::RankedTensorType>(sub_op.getType());
  const int64_t rank = sub_type.getRank();
  mlir::OpBuilder builder(sub_op);
  mlir::Location loc = sub_op.getLoc();
  mlir::Value neg_rhs =
      builder.create<mlir::stablehlo::NegOp>(loc, reduced_rhs);
  mlir::Value neg_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, sub_type, neg_rhs, GetDropLastDimBroadcastAttr(builder, rank));
  mlir::Value add_val =
      builder.create<mlir::stablehlo::AddOp>(loc, sub_op.getLhs(), neg_bcast);
  sub_op.replaceAllUsesWith(add_val);
}

void RewriteSubtractMaxInPlace(ShiftedLogitsMatch& match) {
  mlir::stablehlo::SubtractOp sub_op = match.shifted_sub;
  if (!sub_op || sub_op->getParentOp() == nullptr || sub_op->use_empty()) {
    return;
  }
  ReplaceSubtractWithAddNeg(sub_op, match.max_reduce.getResult(0));
}

bool MatchesSumExpOfExpVal(
    mlir::Value val, mlir::Value exp_val, int64_t expected_dim,
    mlir::stablehlo::ReduceOp* out_sum_reduce = nullptr) {
  mlir::Value cur = val;
  while (true) {
    if (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      cur = conv.getOperand();
      continue;
    }
    if (auto resh = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      cur = resh.getOperand();
      continue;
    }
    if (auto sel = cur.getDefiningOp<mlir::stablehlo::SelectOp>()) {
      if (IsZeroSplatConstant(sel.getOnTrue())) {
        cur = sel.getOnFalse();
        continue;
      }
    }
    break;
  }
  mlir::stablehlo::ReduceOp sum_reduce;
  if (!IsSingleDimReduceWithOp<mlir::stablehlo::AddOp>(cur, expected_dim,
                                                       &sum_reduce)) {
    return false;
  }
  if (sum_reduce.getInputs()[0] != exp_val) {
    return false;
  }
  if (out_sum_reduce) {
    *out_sum_reduce = sum_reduce;
  }
  return true;
}

bool IsValidUnalignedSdpaDenominator(mlir::Value denom_peeled,
                                     mlir::Value exp_val, int64_t rank) {
  if (auto safe_select =
          denom_peeled.getDefiningOp<mlir::stablehlo::SelectOp>()) {
    auto cmp_op =
        safe_select.getPred().getDefiningOp<mlir::stablehlo::CompareOp>();
    if (cmp_op &&
        cmp_op.getComparisonDirection() ==
            mlir::stablehlo::ComparisonDirection::EQ &&
        MatchesSumExpOfExpVal(safe_select.getOnFalse(), exp_val, rank - 1)) {
      return true;
    }
  } else if (MatchesSumExpOfExpVal(denom_peeled, exp_val, rank - 1)) {
    return true;
  }
  return false;
}

void EmitHoistedReciprocalMul(mlir::stablehlo::DivOp div_op,
                              mlir::Value exp_val, mlir::Value denom_peeled,
                              mlir::RankedTensorType type,
                              mlir::RankedTensorType denom_type) {
  const int64_t rank = type.getRank();
  mlir::OpBuilder builder(div_op);
  mlir::Location loc = div_op.getLoc();
  mlir::DenseI64ArrayAttr bcast_dims_attr =
      GetDropLastDimBroadcastAttr(builder, rank);
  mlir::Value one_3d = CreateSplatFloatConst(builder, loc, denom_type, 1.0f);
  mlir::Value inv_sum_3d =
      builder.create<mlir::stablehlo::DivOp>(loc, one_3d, denom_peeled);

  mlir::stablehlo::SelectOp row_mask_select;
  mlir::Value row_mask_pred_3d;
  bool matched_row_mask = false;
  if (div_op->hasOneUse()) {
    mlir::Operation* user = *div_op->getUsers().begin();
    if (user->getNumResults() > 0 &&
        MatchRowAllMaskedSelect(user->getResult(0), div_op.getResult(),
                                row_mask_pred_3d, row_mask_select)) {
      matched_row_mask = true;
    }
  }
  if (matched_row_mask) {
    mlir::OpBuilder sel_builder(row_mask_select);
    mlir::Value zero_3d =
        CreateSplatFloatConst(sel_builder, loc, denom_type, 0.0f);
    mlir::Value masked_inv_sum_3d =
        sel_builder.create<mlir::stablehlo::SelectOp>(loc, row_mask_pred_3d,
                                                      zero_3d, inv_sum_3d);
    mlir::Value inv_bcast =
        sel_builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, type, masked_inv_sum_3d, bcast_dims_attr);
    mlir::Value mul_val =
        sel_builder.create<mlir::stablehlo::MulOp>(loc, exp_val, inv_bcast);
    row_mask_select.replaceAllUsesWith(mul_val);
  } else {
    mlir::Value inv_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, type, inv_sum_3d, bcast_dims_attr);
    mlir::Value mul_val =
        builder.create<mlir::stablehlo::MulOp>(loc, exp_val, inv_bcast);
    div_op.replaceAllUsesWith(mul_val);
  }
}

void TryRewriteUnalignedSdpaDiv(mlir::stablehlo::DivOp div_op) {
  if (!div_op || div_op->getParentOp() == nullptr || div_op->use_empty()) {
    return;
  }
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(div_op.getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 3 ||
      !mlir::isa<mlir::FloatType>(type.getElementType())) {
    return;
  }
  const int64_t rank = type.getRank();
  mlir::Value exp_val = div_op.getLhs();
  auto shifted_match = MatchExpOfShiftedLogits(exp_val);
  if (!shifted_match || !ShouldHoistUnalignedSelfAttentionReciprocal(
                            type, shifted_match->masked_logits)) {
    return;
  }

  mlir::Value denom_peeled =
      PeelReshapeBroadcastChain(div_op.getRhs(), type.getShape());
  if (!denom_peeled) {
    return;
  }
  auto denom_type =
      mlir::dyn_cast<mlir::RankedTensorType>(denom_peeled.getType());
  if (!denom_type || denom_type.getRank() != rank - 1 ||
      !IsValidUnalignedSdpaDenominator(denom_peeled, exp_val, rank)) {
    return;
  }

  RewriteSubtractMaxInPlace(*shifted_match);
  EmitHoistedReciprocalMul(div_op, exp_val, denom_peeled, type, denom_type);
}

// 1. Rewrites unaligned multi-tile SDPA forward and backward 4D Div(exp,
// Broadcast(sum_exp)) in-place to Add(masked_logits, Broadcast(Neg(max))) and
// 3D reciprocal multiplication Mul(exp, Broadcast(Div(1, sum_exp))).
void RewriteUnalignedSdpaDivs(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::DivOp> div_ops;
  func.walk([&](mlir::stablehlo::DivOp op) { div_ops.push_back(op); });
  for (mlir::stablehlo::DivOp div_op : div_ops) {
    TryRewriteUnalignedSdpaDiv(div_op);
  }
}

bool IsReciprocalMulOfExp(mlir::stablehlo::MulOp mul_op, mlir::Value exp_result,
                          int64_t rank, llvm::ArrayRef<int64_t> exp_shape) {
  mlir::Value other =
      mul_op.getLhs() == exp_result ? mul_op.getRhs() : mul_op.getLhs();
  mlir::Value inv_peeled = PeelReshapeBroadcastChain(other, exp_shape);
  if (!inv_peeled) {
    return false;
  }
  if (auto sel = inv_peeled.getDefiningOp<mlir::stablehlo::SelectOp>()) {
    if (IsZeroSplatConstant(sel.getOnTrue())) {
      inv_peeled = sel.getOnFalse();
    }
  }
  auto inv_div = inv_peeled.getDefiningOp<mlir::stablehlo::DivOp>();
  if (!inv_div) {
    return false;
  }
  mlir::Value denom = inv_div.getRhs();
  if (auto safe_sel = denom.getDefiningOp<mlir::stablehlo::SelectOp>()) {
    auto cmp_op =
        safe_sel.getPred().getDefiningOp<mlir::stablehlo::CompareOp>();
    if (cmp_op && cmp_op.getComparisonDirection() ==
                      mlir::stablehlo::ComparisonDirection::EQ) {
      denom = safe_sel.getOnFalse();
    }
  }
  return MatchesSumExpOfExpVal(denom, exp_result, rank - 1);
}

bool HasPreNormalizedReciprocalMulUser(mlir::stablehlo::ExpOp exp_op) {
  auto exp_type = mlir::dyn_cast<mlir::RankedTensorType>(exp_op.getType());
  if (!exp_type || !exp_type.hasStaticShape() || exp_type.getRank() < 3) {
    return false;
  }
  for (mlir::Operation* u : exp_op.getResult().getUsers()) {
    if (mlir::isa<mlir::stablehlo::DivOp>(u) && !u->use_empty()) {
      return false;
    }
  }
  const int64_t rank = exp_type.getRank();
  for (mlir::Operation* u : exp_op.getResult().getUsers()) {
    if (auto mul_op = mlir::dyn_cast<mlir::stablehlo::MulOp>(u)) {
      if (IsReciprocalMulOfExp(mul_op, exp_op.getResult(), rank,
                               exp_type.getShape())) {
        return true;
      }
    }
  }
  return false;
}

bool IsPreNormalizedMulProbability(mlir::Value val) {
  mlir::Value cur = val;
  if (auto sel = cur.getDefiningOp<mlir::stablehlo::SelectOp>()) {
    if (IsZeroSplatConstant(sel.getOnTrue())) {
      cur = sel.getOnFalse();
    }
  }
  auto mul_op = cur.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul_op) {
    return false;
  }
  if (auto exp_lhs = mul_op.getLhs().getDefiningOp<mlir::stablehlo::ExpOp>()) {
    if (HasPreNormalizedReciprocalMulUser(exp_lhs)) {
      return true;
    }
  }
  if (auto exp_rhs = mul_op.getRhs().getDefiningOp<mlir::stablehlo::ExpOp>()) {
    if (HasPreNormalizedReciprocalMulUser(exp_rhs)) {
      return true;
    }
  }
  return false;
}

bool IsTargetBackwardRowsumSubtract(mlir::Value rhs_peeled, int64_t rank,
                                    mlir::stablehlo::SubtractOp sub_op) {
  mlir::stablehlo::ReduceOp red_op;
  if (!IsSingleDimReduceWithOp<mlir::stablehlo::AddOp>(rhs_peeled, rank - 1,
                                                       &red_op)) {
    return false;
  }
  auto prod_mul = red_op.getInputs()[0].getDefiningOp<mlir::stablehlo::MulOp>();
  if (!prod_mul) {
    return false;
  }
  mlir::Value grad_probs = sub_op.getLhs();
  mlir::Value probs;
  if (prod_mul.getLhs() == grad_probs) {
    probs = prod_mul.getRhs();
  } else if (prod_mul.getRhs() == grad_probs) {
    probs = prod_mul.getLhs();
  } else {
    return false;
  }
  return IsPreNormalizedMulProbability(probs);
}

mlir::stablehlo::ExpOp GetSingleExpUser(mlir::stablehlo::SubtractOp sub_op) {
  if (!sub_op->hasOneUse()) {
    return nullptr;
  }
  mlir::Operation* u = *sub_op->getUsers().begin();
  if (auto exp_op = mlir::dyn_cast<mlir::stablehlo::ExpOp>(u)) {
    return exp_op;
  }
  if (auto conv_op = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(u)) {
    if (conv_op->hasOneUse()) {
      return mlir::dyn_cast<mlir::stablehlo::ExpOp>(
          *conv_op->getUsers().begin());
    }
  }
  return nullptr;
}

void TryRewritePreNormalizedSoftmaxSubtraction(
    mlir::stablehlo::SubtractOp sub_op) {
  if (!sub_op || sub_op->getParentOp() == nullptr || sub_op->use_empty()) {
    return;
  }
  auto sub_type = mlir::dyn_cast<mlir::RankedTensorType>(sub_op.getType());
  if (!IsTargetPreNormalizedSoftmaxShape(sub_type)) {
    return;
  }
  const int64_t rank = sub_type.getRank();
  mlir::Value rhs_peeled =
      PeelReshapeBroadcastChain(sub_op.getRhs(), sub_type.getShape());
  if (!rhs_peeled) {
    return;
  }
  auto rhs_peeled_type =
      mlir::dyn_cast<mlir::RankedTensorType>(rhs_peeled.getType());
  if (!rhs_peeled_type || rhs_peeled_type.getRank() != rank - 1) {
    return;
  }

  if (IsTargetBackwardRowsumSubtract(rhs_peeled, rank, sub_op)) {
    ReplaceSubtractWithAddNeg(sub_op, rhs_peeled);
    return;
  }

  mlir::stablehlo::ReduceOp max_reduce;
  if (!IsSingleDimReduceWithOp<mlir::stablehlo::MaxOp>(rhs_peeled, rank - 1,
                                                       &max_reduce) ||
      max_reduce.getInputs()[0] != sub_op.getLhs()) {
    return;
  }
  mlir::stablehlo::ExpOp exp_op = GetSingleExpUser(sub_op);
  if (!exp_op || !HasPreNormalizedReciprocalMulUser(exp_op)) {
    return;
  }

  ReplaceSubtractWithAddNeg(sub_op, max_reduce.getResult(0));
}

// 2. Rewrites Subtract(x, Broadcast(max_red)) -> Add(x,
// Broadcast(Neg(max_red))) and backward Subtract(dP, Broadcast(rowsum)) ->
// Add(dP, Broadcast(Neg(rowsum))) for pre-normalized Mul(exp, Broadcast(Div(1,
// sum_exp))) softmaxes when IsTargetPreNormalizedSoftmaxShape holds.
void RewritePreNormalizedSoftmaxSubtractions(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::SubtractOp> sub_ops;
  func.walk([&](mlir::stablehlo::SubtractOp op) { sub_ops.push_back(op); });
  for (mlir::stablehlo::SubtractOp sub_op : sub_ops) {
    TryRewritePreNormalizedSoftmaxSubtraction(sub_op);
  }
}

// Match result for downstream attention DotGeneral after softmax.
struct DownstreamOutputDotMatch {
  mlir::stablehlo::DotGeneralOp dot_op;
  mlir::stablehlo::ReshapeOp gqa_probs_reshape;
  mlir::stablehlo::ReshapeOp gqa_out_reshape;
};

bool HasExpectedMhaDotDims(mlir::stablehlo::DotDimensionNumbersAttr dnums) {
  return dnums &&
         dnums.getLhsBatchingDimensions() == llvm::ArrayRef<int64_t>{0, 1} &&
         dnums.getRhsBatchingDimensions() == llvm::ArrayRef<int64_t>{0, 1} &&
         dnums.getLhsContractingDimensions() == llvm::ArrayRef<int64_t>{3} &&
         dnums.getRhsContractingDimensions() == llvm::ArrayRef<int64_t>{2};
}

bool HasExpectedGqaDotDims(mlir::stablehlo::DotDimensionNumbersAttr dnums) {
  return dnums &&
         dnums.getLhsBatchingDimensions() == llvm::ArrayRef<int64_t>{0, 1} &&
         dnums.getRhsBatchingDimensions() == llvm::ArrayRef<int64_t>{0, 1} &&
         dnums.getLhsContractingDimensions() == llvm::ArrayRef<int64_t>{4} &&
         dnums.getRhsContractingDimensions() == llvm::ArrayRef<int64_t>{2};
}

std::optional<DownstreamOutputDotMatch> TryMatchGqaOutputDot(
    mlir::stablehlo::DotGeneralOp dot_op, mlir::stablehlo::ReshapeOp probs_resh,
    int64_t b_size, int64_t h_q, int64_t q_len) {
  if (dot_op.getLhs() != probs_resh.getResult() ||
      !HasExpectedGqaDotDims(dot_op.getDotDimensionNumbers())) {
    return std::nullopt;
  }
  if (!dot_op->hasOneUse()) {
    return std::nullopt;
  }
  auto out_resh = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(
      *dot_op.getResult().getUsers().begin());
  if (!out_resh) {
    return std::nullopt;
  }
  auto out_resh_type =
      mlir::dyn_cast<mlir::RankedTensorType>(out_resh.getType());
  if (!out_resh_type || !out_resh_type.hasStaticShape() ||
      out_resh_type.getRank() != 4 || out_resh_type.getDimSize(0) != b_size ||
      out_resh_type.getDimSize(1) != h_q ||
      out_resh_type.getDimSize(2) != q_len) {
    return std::nullopt;
  }
  return DownstreamOutputDotMatch{dot_op, probs_resh, out_resh};
}

std::optional<DownstreamOutputDotMatch> InspectSoftmaxConsumerOp(
    mlir::Operation* op, mlir::Value val, int64_t b_size, int64_t h_q,
    int64_t q_len, int64_t kv_len) {
  if (auto dot = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
    if (dot.getLhs() == val &&
        HasExpectedMhaDotDims(dot.getDotDimensionNumbers())) {
      return DownstreamOutputDotMatch{dot, nullptr, nullptr};
    }
    return std::nullopt;
  }
  auto resh = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op);
  if (!resh || resh.getOperand() != val) {
    return std::nullopt;
  }
  auto resh_type = mlir::dyn_cast<mlir::RankedTensorType>(resh.getType());
  if (!resh_type || !resh_type.hasStaticShape() || resh_type.getRank() != 5 ||
      resh_type.getDimSize(0) != b_size || resh_type.getDimSize(3) != q_len ||
      resh_type.getDimSize(4) != kv_len) {
    return std::nullopt;
  }
  const int64_t h_kv = resh_type.getDimSize(1);
  const int64_t g = resh_type.getDimSize(2);
  if (h_kv <= 0 || g <= 0 || (h_kv * g) != h_q) {
    return std::nullopt;
  }
  if (!resh->hasOneUse()) {
    return std::nullopt;
  }
  auto dot_user = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(
      *resh.getResult().getUsers().begin());
  if (!dot_user) {
    return std::nullopt;
  }
  return TryMatchGqaOutputDot(dot_user, resh, b_size, h_q, q_len);
}

std::optional<DownstreamOutputDotMatch> FindDownstreamAttentionOutputDot(
    mlir::Value softmax_val) {
  auto softmax_type =
      mlir::dyn_cast<mlir::RankedTensorType>(softmax_val.getType());
  if (!softmax_type || !softmax_type.hasStaticShape() ||
      softmax_type.getRank() != 4) {
    return std::nullopt;
  }
  const int64_t b_size = softmax_type.getDimSize(0);
  const int64_t h_q = softmax_type.getDimSize(1);
  const int64_t q_len = softmax_type.getDimSize(2);
  const int64_t kv_len = softmax_type.getDimSize(3);

  for (mlir::Operation* user : softmax_val.getUsers()) {
    if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(user)) {
      if (conv->hasOneUse()) {
        for (mlir::Operation* u2 : conv.getResult().getUsers()) {
          if (auto match = InspectSoftmaxConsumerOp(
                  u2, conv.getResult(), b_size, h_q, q_len, kv_len)) {
            return match;
          }
        }
      }
    } else {
      if (auto match = InspectSoftmaxConsumerOp(user, softmax_val, b_size, h_q,
                                                q_len, kv_len)) {
        return match;
      }
    }
  }
  return std::nullopt;
}

bool IsForwardOnlineSoftmaxDivCandidate(mlir::stablehlo::DivOp div_op,
                                        mlir::RankedTensorType& out_type,
                                        mlir::Operation*& out_root_softmax,
                                        mlir::Value& out_row_mask_pred_3d) {
  if (!div_op || div_op->getParentOp() == nullptr || !div_op->hasOneUse()) {
    return false;
  }
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(div_op.getType());
  if (!type || !type.hasStaticShape() || type.getRank() != 4 ||
      !mlir::isa<mlir::FloatType>(type.getElementType())) {
    return false;
  }
  const int64_t q_len = type.getDimSize(2);
  const int64_t kv_len = type.getDimSize(3);
  if (q_len <= 0 || kv_len <= 0 || (q_len % kTpuLaneWidth) != 0 ||
      (kv_len % kTpuLaneWidth) != 0) {
    return false;
  }
  const int64_t single_head_matrix_f32_bytes =
      q_len * kv_len * static_cast<int64_t>(sizeof(float));
  if (single_head_matrix_f32_bytes < kTpuScopedVmemLimitBytes) {
    return false;
  }

  mlir::Value exp_val = div_op.getLhs();
  auto bcast_op =
      div_op.getRhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast_op || !bcast_op->hasOneUse() ||
      !IsDropLastDimBroadcast(bcast_op.getBroadcastDimensions(), 4)) {
    return false;
  }
  mlir::stablehlo::ReduceOp sum_reduce;
  if (!IsSingleDimReduceWithOp<mlir::stablehlo::AddOp>(bcast_op.getOperand(), 3,
                                                       &sum_reduce) ||
      sum_reduce.getInputs()[0] != exp_val || !sum_reduce->hasOneUse()) {
    return false;
  }
  for (mlir::Operation* u : exp_val.getUsers()) {
    if (u != div_op.getOperation() && u != sum_reduce.getOperation()) {
      return false;
    }
  }
  if (!MatchExpOfShiftedLogits(exp_val)) {
    return false;
  }

  mlir::Operation* root_softmax_op = div_op.getOperation();
  mlir::Value row_mask_pred;
  if (div_op->hasOneUse()) {
    mlir::Operation* user = *div_op->getUsers().begin();
    if (user->getNumResults() > 0) {
      mlir::stablehlo::SelectOp masked_sel;
      mlir::Value pred_3d;
      if (MatchRowAllMaskedSelect(user->getResult(0), div_op.getResult(),
                                  pred_3d, masked_sel)) {
        root_softmax_op = masked_sel.getOperation();
        row_mask_pred = pred_3d;
      }
    }
  }
  if (!root_softmax_op->hasOneUse()) {
    return false;
  }

  out_type = type;
  out_root_softmax = root_softmax_op;
  out_row_mask_pred_3d = row_mask_pred;
  return true;
}

bool ValidateOnlineSoftmaxValueAndOutputTypes(
    DownstreamOutputDotMatch& dot_match, mlir::RankedTensorType type,
    mlir::RankedTensorType& out_v_type, mlir::RankedTensorType& out_4d_type,
    mlir::Operation*& out_old_out_op) {
  const int64_t b_size = type.getDimSize(0);
  const int64_t h_q = type.getDimSize(1);
  const int64_t q_len = type.getDimSize(2);
  const int64_t kv_len = type.getDimSize(3);

  mlir::Value v_4d = dot_match.dot_op.getRhs();
  auto v_type = mlir::dyn_cast<mlir::RankedTensorType>(v_4d.getType());
  if (!v_type || !v_type.hasStaticShape() || v_type.getRank() != 4 ||
      v_type.getDimSize(0) != b_size || v_type.getDimSize(2) != kv_len) {
    return false;
  }
  if (dot_match.gqa_probs_reshape) {
    const int64_t h_kv = v_type.getDimSize(1);
    if (h_kv <= 0 || (h_q % h_kv) != 0) {
      return false;
    }
    auto v_rep_type = mlir::RankedTensorType::get(
        {b_size, h_q, kv_len, v_type.getDimSize(3)}, v_type.getElementType());
    if (v_rep_type.getNumElements() > type.getNumElements()) {
      return false;
    }
  } else {
    if (v_type.getDimSize(1) != h_q) {
      return false;
    }
  }
  const int64_t head_dim = v_type.getDimSize(3);
  if (head_dim <= 0 || head_dim > kTpuLaneWidth) {
    return false;
  }

  mlir::Operation* old_out_op = dot_match.gqa_out_reshape
                                    ? dot_match.gqa_out_reshape.getOperation()
                                    : dot_match.dot_op.getOperation();
  if (old_out_op->getNumResults() == 0) {
    return false;
  }
  auto out_4d = mlir::dyn_cast<mlir::RankedTensorType>(
      old_out_op->getResult(0).getType());
  if (!out_4d || !out_4d.hasStaticShape() || out_4d.getRank() != 4 ||
      out_4d.getDimSize(0) != b_size || out_4d.getDimSize(1) != h_q ||
      out_4d.getDimSize(2) != q_len || out_4d.getDimSize(3) != head_dim ||
      out_4d.getNumElements() > type.getNumElements()) {
    return false;
  }

  out_v_type = v_type;
  out_4d_type = out_4d;
  out_old_out_op = old_out_op;
  return true;
}

void EmitNormalizedGqaAndSunkMaskSelect(
    mlir::stablehlo::DivOp div_op, DownstreamOutputDotMatch& dot_match,
    mlir::Operation* old_out_op, mlir::Value row_mask_pred_3d,
    mlir::RankedTensorType v_type, mlir::RankedTensorType out_4d_type,
    int64_t b_size, int64_t h_q, int64_t kv_len, int64_t head_dim) {
  mlir::OpBuilder builder(old_out_op);
  mlir::Location loc = old_out_op->getLoc();

  mlir::Value unmasked_probs = div_op.getResult();
  mlir::Value probs_for_dot =
      ConvertIfNeeded(builder, loc, unmasked_probs, v_type.getElementType());

  mlir::Value v_for_dot = dot_match.dot_op.getRhs();
  if (dot_match.gqa_probs_reshape) {
    const int64_t h_kv = v_type.getDimSize(1);
    const int64_t g = h_q / h_kv;
    auto v_rep_type = mlir::RankedTensorType::get(
        {b_size, h_q, kv_len, head_dim}, v_type.getElementType());
    auto v_5d_type = mlir::RankedTensorType::get(
        {b_size, h_kv, g, kv_len, head_dim}, v_type.getElementType());
    mlir::Value v_5d = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, v_5d_type, v_for_dot, builder.getDenseI64ArrayAttr({0, 1, 3, 4}));
    v_for_dot =
        builder.create<mlir::stablehlo::ReshapeOp>(loc, v_rep_type, v_5d);
  }

  auto dnums = mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(), /*lhsBatchingDimensions=*/{0, 1},
      /*rhsBatchingDimensions=*/{0, 1}, /*lhsContractingDimensions=*/{3},
      /*rhsContractingDimensions=*/{2});
  mlir::Value new_out = builder.create<mlir::stablehlo::DotGeneralOp>(
      loc, out_4d_type, probs_for_dot, v_for_dot, dnums,
      dot_match.dot_op.getPrecisionConfigAttr(),
      dot_match.dot_op.getAlgorithmAttr());

  if (row_mask_pred_3d) {
    auto pred_4d_type = mlir::RankedTensorType::get(out_4d_type.getShape(),
                                                    builder.getI1Type());
    mlir::Value pred_4d = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, pred_4d_type, row_mask_pred_3d,
        builder.getDenseI64ArrayAttr({0, 1, 2}));
    mlir::Value zero_out =
        CreateSplatFloatConst(builder, loc, out_4d_type, 0.0f);
    new_out = builder.create<mlir::stablehlo::SelectOp>(loc, pred_4d, zero_out,
                                                        new_out);
  }
  old_out_op->replaceAllUsesWith(mlir::ValueRange{new_out});
}

void TryRewriteForwardGqaAndMaskedOnlineSoftmax(mlir::stablehlo::DivOp div_op) {
  mlir::RankedTensorType type;
  mlir::Operation* root_softmax_op = nullptr;
  mlir::Value row_mask_pred_3d;
  if (!IsForwardOnlineSoftmaxDivCandidate(div_op, type, root_softmax_op,
                                          row_mask_pred_3d)) {
    return;
  }

  auto dot_match =
      FindDownstreamAttentionOutputDot(root_softmax_op->getResult(0));
  if (!dot_match) {
    return;
  }

  mlir::RankedTensorType v_type;
  mlir::RankedTensorType out_4d_type;
  mlir::Operation* old_out_op = nullptr;
  if (!ValidateOnlineSoftmaxValueAndOutputTypes(*dot_match, type, v_type,
                                                out_4d_type, old_out_op)) {
    return;
  }

  const int64_t b_size = type.getDimSize(0);
  const int64_t h_q = type.getDimSize(1);
  const int64_t kv_len = type.getDimSize(3);
  const int64_t head_dim = v_type.getDimSize(3);
  EmitNormalizedGqaAndSunkMaskSelect(div_op, *dot_match, old_out_op,
                                     row_mask_pred_3d, v_type, out_4d_type,
                                     b_size, h_q, kv_len, head_dim);
}

// 3. Normalizes standalone forward 5D GQA `probs @ V` to 4D and sinks
// row-masked `Select` past `DotGeneral` when the single-head [Q, K] f32 matrix
// reaches the 16 MiB VMEM capacity and head_dim fits in a 128-lane MXU tile,
// enabling XLA FlashAttention (`MatchSoftmax`) to fuse into `online-softmax`.
void RewriteForwardGqaAndMaskedOnlineSoftmax(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::DivOp> div_ops;
  func.walk([&](mlir::stablehlo::DivOp op) { div_ops.push_back(op); });
  for (mlir::stablehlo::DivOp div_op : div_ops) {
    TryRewriteForwardGqaAndMaskedOnlineSoftmax(div_op);
  }
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation*> dead_ops;
    func.walk([&](mlir::Operation* op) {
      if (op != func.getOperation() && op->use_empty() &&
          (mlir::isOpTriviallyDead(op) ||
           mlir::isa<mlir::stablehlo::ReduceOp>(op))) {
        dead_ops.push_back(op);
      }
    });
    for (mlir::Operation* op : dead_ops) {
      op->erase();
      changed = true;
    }
  }
}

struct OptimizeSdpaAndSoftmaxReductionsPass
    : public mlir::PassWrapper<OptimizeSdpaAndSoftmaxReductionsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      OptimizeSdpaAndSoftmaxReductionsPass)

  llvm::StringRef getArgument() const override {
    return "optimize-sdpa-and-softmax-reductions";
  }
  llvm::StringRef getDescription() const override {
    return "Eliminates standalone reduce-window fusions in attention softmax "
           "reductions and normalizes standalone forward GQA and row-masked "
           "SDPA for XLA online-softmax fusion.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    RewriteUnalignedSdpaDivs(func);
    EraseTriviallyDeadOps(func);
    RewritePreNormalizedSoftmaxSubtractions(func);
    EraseTriviallyDeadOps(func);
    RewriteForwardGqaAndMaskedOnlineSoftmax(func);
    EraseTriviallyDeadOps(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createOptimizeSdpaAndSoftmaxReductionsPass() {
  return std::make_unique<OptimizeSdpaAndSoftmaxReductionsPass>();
}

}  // namespace torch_tpu
