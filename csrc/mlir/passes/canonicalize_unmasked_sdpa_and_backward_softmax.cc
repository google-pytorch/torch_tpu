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

#include "csrc/mlir/passes/canonicalize_unmasked_sdpa_and_backward_softmax.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/log/absl_check.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
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

// GENERIC: hardware TPU 128-lane vector width required by XLA FlashAttention
// online-softmax pattern matcher (q_len == kv_len && q_len % 128 == 0).
constexpr int64_t kTpuLaneWidth = 128;

// GENERIC: semantic bound on wrapper peeling depth.
constexpr int kMaxPeelDepth = 8;

[[nodiscard]] bool IsDropLastDimBroadcast(llvm::ArrayRef<int64_t> dims,
                                          int64_t rank) {
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

[[nodiscard]] bool IsIdentityBroadcastDims(llvm::ArrayRef<int64_t> dims,
                                           int64_t rank) {
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

[[nodiscard]] mlir::DenseI64ArrayAttr GetDropLastDimBroadcastAttr(
    mlir::OpBuilder& builder, int64_t rank) {
  ABSL_CHECK_GE(rank, 1);
  llvm::SmallVector<int64_t, 4> dims;
  dims.reserve(rank - 1);
  for (int64_t i = 0; i < rank - 1; ++i) {
    dims.push_back(i);
  }
  return builder.getDenseI64ArrayAttr(dims);
}

[[nodiscard]] mlir::Value UnwrapBroadcastAndReshapeToConst(mlir::Value val) {
  for (int depth = 0; depth < kMaxPeelDepth; ++depth) {
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
    if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      val = conv.getOperand();
      continue;
    }
    break;
  }
  return val;
}

[[nodiscard]] std::optional<double> MatchFiniteFloatSplat(mlir::Value val) {
  val = UnwrapBroadcastAndReshapeToConst(val);
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense || dense.empty() || !dense.isSplat() ||
      !mlir::isa<mlir::FloatType>(dense.getElementType())) {
    return std::nullopt;
  }
  llvm::APFloat apf = dense.getSplatValue<llvm::APFloat>();
  if (apf.isInfinity() || apf.isNaN()) {
    return std::nullopt;
  }
  return apf.convertToDouble();
}

[[nodiscard]] bool IsZeroSplatConstant(mlir::Value val) {
  std::optional<double> v = MatchFiniteFloatSplat(val);
  return v.has_value() && *v == 0.0;
}

[[nodiscard]] bool IsOneSplatConstant(mlir::Value val) {
  std::optional<double> v = MatchFiniteFloatSplat(val);
  return v.has_value() && *v == 1.0;
}

[[nodiscard]] mlir::Value CreateSplatFloatConst(mlir::OpBuilder& builder,
                                                mlir::Location loc,
                                                mlir::RankedTensorType type,
                                                double value) {
  auto float_type = mlir::cast<mlir::FloatType>(type.getElementType());
  llvm::APFloat apf(value);
  bool loses_info = false;
  apf.convert(float_type.getFloatSemantics(),
              llvm::APFloat::rmNearestTiesToEven, &loses_info);
  auto attr = mlir::DenseElementsAttr::get(type, apf);
  return mlir::stablehlo::ConstantOp::create(builder, loc, type, attr);
}

[[nodiscard]] mlir::Value ConvertIfNeeded(mlir::OpBuilder& builder,
                                          mlir::Location loc, mlir::Value val,
                                          mlir::Type target_elem_type) {
  auto ranked = mlir::cast<mlir::RankedTensorType>(val.getType());
  if (ranked.getElementType() == target_elem_type) {
    return val;
  }
  auto new_type =
      mlir::RankedTensorType::get(ranked.getShape(), target_elem_type);
  return mlir::stablehlo::ConvertOp::create(builder, loc, new_type, val);
}

[[nodiscard]] mlir::Value PeelReshapeBroadcastChain(
    mlir::Value val, llvm::ArrayRef<int64_t> outer_shape) {
  const int64_t rank = outer_shape.size();
  // GENERIC: semantic rank >= 2 required to drop the reduced trailing axis
  if (rank <= 1) {
    return nullptr;
  }
  llvm::ArrayRef<int64_t> prefix_shape = outer_shape.drop_back(1);
  auto matches_prefix_or_unit_last = [&](mlir::RankedTensorType t) -> bool {
    if (!t || !t.hasStaticShape()) {
      return false;
    }
    if (t.getRank() == rank - 1) {
      return t.getShape() == prefix_shape;
    }
    if (t.getRank() == rank && t.getDimSize(rank - 1) == 1) {
      return t.getShape().drop_back(1) == prefix_shape;
    }
    return false;
  };

  mlir::Value cur = val;
  bool saw_bcast = false;
  for (int depth = 0; depth < kMaxPeelDepth; ++depth) {
    if (auto bcast = cur.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      auto in_type =
          mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
      auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
      if (!in_type || !out_type || !in_type.hasStaticShape() ||
          !out_type.hasStaticShape()) {
        break;
      }
      llvm::ArrayRef<int64_t> dims = bcast.getBroadcastDimensions();
      if (out_type.getRank() == rank && in_type.getRank() == rank - 1 &&
          IsDropLastDimBroadcast(dims, rank) &&
          in_type.getShape() == prefix_shape) {
        cur = bcast.getOperand();
        saw_bcast = true;
        continue;
      }
      if (out_type.getRank() == rank && in_type.getRank() == rank &&
          IsIdentityBroadcastDims(dims, rank) &&
          matches_prefix_or_unit_last(in_type)) {
        cur = bcast.getOperand();
        if (out_type.getDimSize(rank - 1) == outer_shape[rank - 1]) {
          saw_bcast = true;
        }
        continue;
      }
      break;
    }
    if (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      auto in_type =
          mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
      if (in_type && in_type.hasStaticShape() &&
          (in_type.getShape() == outer_shape ||
           matches_prefix_or_unit_last(in_type))) {
        cur = conv.getOperand();
        continue;
      }
      break;
    }
    if (auto resh = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      auto in_type =
          mlir::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
      if (matches_prefix_or_unit_last(in_type)) {
        cur = resh.getOperand();
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
  if (!matches_prefix_or_unit_last(cur_type)) {
    return nullptr;
  }
  return cur;
}

template <typename BinaryOpTy>
[[nodiscard]] bool IsSingleDimReduceWithOp(
    mlir::Value val, int64_t expected_dim,
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
  // GENERIC: semantic 2-arg scalar reduction combiner block
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

struct ShiftedLogitsMatch {
  mlir::Value masked_logits;
  mlir::stablehlo::ReduceOp max_reduce;
  mlir::stablehlo::SubtractOp shifted_sub;
};

[[nodiscard]] std::optional<ShiftedLogitsMatch> MatchExpOfShiftedLogits(
    mlir::Value exp_val) {
  auto exp_op = exp_val.getDefiningOp<mlir::stablehlo::ExpOp>();
  if (!exp_op) {
    return std::nullopt;
  }
  mlir::Value logits = exp_op.getOperand();
  if (auto conv = logits.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    logits = conv.getOperand();
  }
  if (auto sub_op = logits.getDefiningOp<mlir::stablehlo::SubtractOp>()) {
    auto sub_type = mlir::dyn_cast<mlir::RankedTensorType>(sub_op.getType());
    if (!sub_type || !sub_type.hasStaticShape() || sub_type.getRank() <= 1) {
      return std::nullopt;
    }
    const int64_t rank = sub_type.getRank();
    mlir::Value max_peeled =
        PeelReshapeBroadcastChain(sub_op.getRhs(), sub_type.getShape());
    if (!max_peeled) {
      return std::nullopt;
    }
    if (auto conv = max_peeled.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      max_peeled = conv.getOperand();
    }
    mlir::stablehlo::ReduceOp max_reduce;
    if (!IsSingleDimReduceWithOp<mlir::stablehlo::MaxOp>(max_peeled, rank - 1,
                                                         &max_reduce)) {
      return std::nullopt;
    }
    mlir::Value lhs = sub_op.getLhs();
    if (auto conv = lhs.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      lhs = conv.getOperand();
    }
    if (max_reduce.getInputs()[0] != lhs) {
      return std::nullopt;
    }
    return ShiftedLogitsMatch{sub_op.getLhs(), max_reduce, sub_op};
  }
  if (auto add_op = logits.getDefiningOp<mlir::stablehlo::AddOp>()) {
    auto add_type = mlir::dyn_cast<mlir::RankedTensorType>(add_op.getType());
    if (!add_type || !add_type.hasStaticShape() || add_type.getRank() <= 1) {
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
    mlir::Value neg_in = neg_op.getOperand();
    if (auto conv = neg_in.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      neg_in = conv.getOperand();
    }
    mlir::stablehlo::ReduceOp max_reduce;
    if (!IsSingleDimReduceWithOp<mlir::stablehlo::MaxOp>(neg_in, rank - 1,
                                                         &max_reduce)) {
      return std::nullopt;
    }
    mlir::Value lhs = add_op.getLhs();
    if (auto conv = lhs.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      lhs = conv.getOperand();
    }
    if (max_reduce.getInputs()[0] != lhs) {
      return std::nullopt;
    }
    return ShiftedLogitsMatch{add_op.getLhs(), max_reduce, nullptr};
  }
  return std::nullopt;
}

void RewriteSubtractMaxInPlace(ShiftedLogitsMatch& match) {
  mlir::stablehlo::SubtractOp sub_op = match.shifted_sub;
  if (!sub_op || sub_op->getParentOp() == nullptr || sub_op->use_empty()) {
    return;
  }
  auto sub_type = mlir::cast<mlir::RankedTensorType>(sub_op.getType());
  const int64_t rank = sub_type.getRank();
  mlir::OpBuilder builder(sub_op);
  mlir::Location loc = sub_op.getLoc();
  mlir::Value max_val = ConvertIfNeeded(
      builder, loc, match.max_reduce.getResult(0), sub_type.getElementType());
  mlir::Value neg_max = mlir::stablehlo::NegOp::create(builder, loc, max_val);
  mlir::Value neg_bcast = mlir::stablehlo::BroadcastInDimOp::create(
      builder, loc, sub_type, neg_max,
      GetDropLastDimBroadcastAttr(builder, rank));
  mlir::Value add_val = mlir::stablehlo::AddOp::create(
      builder, loc, match.masked_logits, neg_bcast);
  sub_op.replaceAllUsesWith(add_val);
}

// Matches `val` as a direct `ConvertOp`/`ReshapeOp` chain from `ReduceOp(AddOp,
// init=0.0)` over `exp_val` without any intermediate row-zeroing `SelectOp`.
[[nodiscard]] bool MatchesUnmaskedSumExpOfExpVal(
    mlir::Value val, mlir::Value exp_val, int64_t expected_dim,
    mlir::stablehlo::ReduceOp* out_sum_reduce = nullptr) {
  mlir::Value cur = val;
  for (int depth = 0; depth < kMaxPeelDepth; ++depth) {
    if (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      cur = conv.getOperand();
      continue;
    }
    if (auto resh = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      cur = resh.getOperand();
      continue;
    }
    break;
  }
  mlir::stablehlo::ReduceOp sum_reduce;
  if (!IsSingleDimReduceWithOp<mlir::stablehlo::AddOp>(cur, expected_dim,
                                                       &sum_reduce)) {
    return false;
  }
  mlir::Value reduce_input = sum_reduce.getInputs()[0];
  while (auto conv = reduce_input.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    reduce_input = conv.getOperand();
  }
  mlir::Value raw_exp = exp_val;
  while (auto conv = raw_exp.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    raw_exp = conv.getOperand();
  }
  if (reduce_input != raw_exp ||
      !IsZeroSplatConstant(sum_reduce.getInitValues()[0])) {
    return false;
  }
  if (out_sum_reduce) {
    *out_sum_reduce = sum_reduce;
  }
  return true;
}

// Matches `safe_sum_exp = SelectOp(CompareOp(sum_exp, 0.0, EQ), 1.0, sum_exp)`.
[[nodiscard]] mlir::Value MatchSafeSumExpZeroGuard(mlir::Value val) {
  mlir::Value cur = val;
  for (int depth = 0; depth < kMaxPeelDepth; ++depth) {
    if (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      cur = conv.getOperand();
      continue;
    }
    if (auto resh = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      cur = resh.getOperand();
      continue;
    }
    if (auto sel = cur.getDefiningOp<mlir::stablehlo::SelectOp>()) {
      if (!IsOneSplatConstant(sel.getOnTrue())) {
        return nullptr;
      }
      auto cmp = sel.getPred().getDefiningOp<mlir::stablehlo::CompareOp>();
      if (!cmp || cmp.getComparisonDirection() !=
                      mlir::stablehlo::ComparisonDirection::EQ) {
        return nullptr;
      }
      mlir::Value sum_cand = sel.getOnFalse();
      if ((cmp.getLhs() == sum_cand && IsZeroSplatConstant(cmp.getRhs())) ||
          (cmp.getRhs() == sum_cand && IsZeroSplatConstant(cmp.getLhs()))) {
        return sum_cand;
      }
      return nullptr;
    }
    break;
  }
  return nullptr;
}

// Builds a scalar `AddOp` reduction with 0.0 init over `input` along `dim`.
[[nodiscard]] mlir::stablehlo::ReduceOp CreateSumReduceAlongDim(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
    int64_t dim) {
  auto in_type = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto elem_type = mlir::cast<mlir::FloatType>(in_type.getElementType());
  auto scalar_type = mlir::RankedTensorType::get({}, elem_type);
  mlir::Value zero_init = CreateSplatFloatConst(builder, loc, scalar_type, 0.0);

  llvm::SmallVector<int64_t, 4> red_shape;
  for (int64_t i = 0; i < in_type.getRank(); ++i) {
    if (i != dim) {
      red_shape.push_back(in_type.getDimSize(i));
    }
  }
  auto red_type = mlir::RankedTensorType::get(red_shape, elem_type);
  auto reduce_op = mlir::stablehlo::ReduceOp::create(
      builder, loc, mlir::TypeRange{red_type}, mlir::ValueRange{input},
      mlir::ValueRange{zero_init}, builder.getDenseI64ArrayAttr({dim}));

  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Block* body =
      builder.createBlock(&reduce_op.getBody(), reduce_op.getBody().end(),
                          {scalar_type, scalar_type}, {loc, loc});
  mlir::Value add_res = mlir::stablehlo::AddOp::create(
      builder, loc, body->getArgument(0), body->getArgument(1));
  mlir::stablehlo::ReturnOp::create(builder, loc, mlir::ValueRange{add_res});
  return reduce_op;
}

// Returns true if `exp_val` is also used by a forward `DivOp` (not guarded by
// `safe_sum_exp`) and is not wrapped by a row-mask `SelectOp(..., 0.0, ...)`.
[[nodiscard]] bool HasUnmaskedForwardDivUser(mlir::Value exp_val,
                                             mlir::Operation* bwd_op) {
  for (mlir::Operation* user : exp_val.getUsers()) {
    if (user == bwd_op || llvm::isa<mlir::stablehlo::ReduceOp>(user)) {
      continue;
    }
    if (auto div_op = llvm::dyn_cast<mlir::stablehlo::DivOp>(user)) {
      for (mlir::Operation* div_user : div_op.getResult().getUsers()) {
        if (auto sel = llvm::dyn_cast<mlir::stablehlo::SelectOp>(div_user)) {
          if (IsZeroSplatConstant(sel.getOnTrue())) {
            return false;
          }
        }
      }
      return true;
    }
  }
  return false;
}

// Returns true if `div_op` or its converted result is wrapped in a row-mask
// `SelectOp(pred, 0.0, ...)`.
[[nodiscard]] bool IsWrappedInRowMaskSelect(mlir::Operation* op) {
  for (mlir::Operation* u : op->getUsers()) {
    if (auto sel = llvm::dyn_cast<mlir::stablehlo::SelectOp>(u)) {
      if (IsZeroSplatConstant(sel.getOnTrue())) {
        return true;
      }
    }
    if (auto conv = llvm::dyn_cast<mlir::stablehlo::ConvertOp>(u)) {
      for (mlir::Operation* u2 : conv.getResult().getUsers()) {
        if (auto sel = llvm::dyn_cast<mlir::stablehlo::SelectOp>(u2)) {
          if (IsZeroSplatConstant(sel.getOnTrue())) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

// Returns true if `masked_logits` is a direct `DotGeneralOp` (unmasked SDPA)
// or `SelectOp` (causal SDPA), rather than an additive 2D mask `AddOp`.
[[nodiscard]] bool IsUnmaskedOrCausalAttentionLogits(
    mlir::Value masked_logits) {
  mlir::Value cur = masked_logits;
  while (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    cur = conv.getOperand();
  }
  if (auto resh = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    cur = resh.getOperand();
  }
  if (llvm::isa_and_nonnull<mlir::stablehlo::DotGeneralOp>(
          cur.getDefiningOp())) {
    return true;
  }
  if (auto sel = cur.getDefiningOp<mlir::stablehlo::SelectOp>()) {
    auto is_dot = [](mlir::Value v) -> bool {
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
      return llvm::isa_and_nonnull<mlir::stablehlo::DotGeneralOp>(
          v.getDefiningOp());
    };
    return is_dot(sel.getOnTrue()) || is_dot(sel.getOnFalse());
  }
  return false;
}

// 1. Canonicalizes unmasked/causal backward SDPA softmax operations using
// `safe_sum_exp = Select(Compare(sum_exp, 0.0, EQ), 1.0, sum_exp)`:
//   - Folds the provably false `sum_exp == 0.0` guard (`sum_exp >= 1.0 > 0.0`).
//   - When the sequence dimensions satisfy XLA FlashAttention online-softmax
//     alignment (`(q_len % 128) == 0 && (kv_len % 128) == 0`) and `exp_val` is
//     shared with forward SDPA, decouples backward `shifted_logits`, `exp`, and
//     `sum_exp` using `Add(masked_logits, BroadcastInDim(Neg(max_logits)))` so
//     forward `online-softmax` matches and dead-code-eliminates the forward
//     `Subtract`, while backward avoids `reduce-window`.
//   - When sequence dimensions are unaligned (`(q_len % 128) != 0 || (kv_len %
//     128) != 0`), rewrites `Subtract` in place to `Add(masked_logits,
//     BroadcastInDim(Neg(max_logits)))` and unifies forward and backward
//     softmax denominators.
void CanonicalizeUnmaskedBackwardSdpaSoftmax(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::Operation*, 16> candidate_ops;
  func.walk([&](mlir::Operation* op) {
    if (llvm::isa<mlir::stablehlo::DivOp, mlir::stablehlo::MulOp>(op)) {
      candidate_ops.push_back(op);
    }
  });

  for (mlir::Operation* op : candidate_ops) {
    if (!op || op->getParentOp() == nullptr || op->use_empty() ||
        IsWrappedInRowMaskSelect(op)) {
      continue;
    }
    auto type =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!type || !type.hasStaticShape() || type.getRank() <= 1 ||
        !mlir::isa<mlir::FloatType>(type.getElementType())) {
      continue;
    }
    const int64_t rank = type.getRank();
    const int64_t q_len = type.getDimSize(rank - 2);
    const int64_t kv_len = type.getDimSize(rank - 1);

    mlir::Value exp_val;
    mlir::Value bcast_denom_or_inv;
    bool is_div = false;
    if (auto div_op = llvm::dyn_cast<mlir::stablehlo::DivOp>(op)) {
      exp_val = div_op.getLhs();
      bcast_denom_or_inv = div_op.getRhs();
      is_div = true;
    } else if (auto mul_op = llvm::dyn_cast<mlir::stablehlo::MulOp>(op)) {
      if (mul_op.getLhs().getDefiningOp<mlir::stablehlo::ExpOp>()) {
        exp_val = mul_op.getLhs();
        bcast_denom_or_inv = mul_op.getRhs();
      } else if (mul_op.getRhs().getDefiningOp<mlir::stablehlo::ExpOp>()) {
        exp_val = mul_op.getRhs();
        bcast_denom_or_inv = mul_op.getLhs();
      } else {
        continue;
      }
    }

    auto exp_type = mlir::dyn_cast<mlir::RankedTensorType>(exp_val.getType());
    if (!exp_type || !exp_type.hasStaticShape() ||
        exp_type.getShape() != type.getShape()) {
      continue;
    }
    auto shifted_match = MatchExpOfShiftedLogits(exp_val);
    if (!shifted_match) {
      continue;
    }

    mlir::Value peeled_denom =
        PeelReshapeBroadcastChain(bcast_denom_or_inv, type.getShape());
    if (!peeled_denom) {
      continue;
    }
    if (!is_div) {
      auto inv_div = peeled_denom.getDefiningOp<mlir::stablehlo::DivOp>();
      if (!inv_div || !IsOneSplatConstant(inv_div.getLhs())) {
        continue;
      }
      peeled_denom = inv_div.getRhs();
    }

    mlir::Value raw_sum_exp = MatchSafeSumExpZeroGuard(peeled_denom);
    if (!raw_sum_exp) {
      continue;
    }
    mlir::stablehlo::ReduceOp sum_reduce;
    if (!MatchesUnmaskedSumExpOfExpVal(raw_sum_exp, exp_val, rank - 1,
                                       &sum_reduce)) {
      continue;
    }

    // GENERIC: hardware TPU 128-lane online-softmax capability check.
    const bool can_use_online_softmax_fwd =
        (q_len % kTpuLaneWidth == 0) && (kv_len % kTpuLaneWidth == 0);

    mlir::OpBuilder builder(op);
    mlir::Location loc = op->getLoc();
    auto reduced_shape = type.getShape().drop_back(1);
    auto reduced_calc_type =
        mlir::RankedTensorType::get(reduced_shape, type.getElementType());

    if (can_use_online_softmax_fwd && HasUnmaskedForwardDivUser(exp_val, op) &&
        shifted_match->shifted_sub &&
        IsUnmaskedOrCausalAttentionLogits(shifted_match->masked_logits)) {
      auto logits_type = mlir::cast<mlir::RankedTensorType>(
          shifted_match->masked_logits.getType());
      mlir::Value max_val =
          ConvertIfNeeded(builder, loc, shifted_match->max_reduce.getResult(0),
                          logits_type.getElementType());
      mlir::Value neg_max =
          mlir::stablehlo::NegOp::create(builder, loc, max_val);
      mlir::Value neg_bcast = mlir::stablehlo::BroadcastInDimOp::create(
          builder, loc, logits_type, neg_max,
          GetDropLastDimBroadcastAttr(builder, rank));
      mlir::Value shifted_bwd = mlir::stablehlo::AddOp::create(
          builder, loc, shifted_match->masked_logits, neg_bcast);
      mlir::Value shifted_bwd_calc =
          ConvertIfNeeded(builder, loc, shifted_bwd, type.getElementType());
      mlir::Value exp_bwd =
          mlir::stablehlo::ExpOp::create(builder, loc, type, shifted_bwd_calc);
      mlir::stablehlo::ReduceOp sum_bwd =
          CreateSumReduceAlongDim(builder, loc, exp_bwd, rank - 1);
      mlir::Value sum_bwd_val = sum_bwd.getResult(0);
      if (sum_bwd_val.getType() != reduced_calc_type) {
        sum_bwd_val = mlir::stablehlo::ReshapeOp::create(
            builder, loc, reduced_calc_type, sum_bwd_val);
      }
      if (is_div) {
        mlir::Value bcast_sum = mlir::stablehlo::BroadcastInDimOp::create(
            builder, loc, type, sum_bwd_val,
            GetDropLastDimBroadcastAttr(builder, rank));
        mlir::Value new_probs =
            mlir::stablehlo::DivOp::create(builder, loc, exp_bwd, bcast_sum);
        op->replaceAllUsesWith(mlir::ValueRange{new_probs});
      } else {
        mlir::Value one_reduced =
            CreateSplatFloatConst(builder, loc, reduced_calc_type, 1.0);
        mlir::Value inv_sum = mlir::stablehlo::DivOp::create(
            builder, loc, one_reduced, sum_bwd_val);
        mlir::Value bcast_inv = mlir::stablehlo::BroadcastInDimOp::create(
            builder, loc, type, inv_sum,
            GetDropLastDimBroadcastAttr(builder, rank));
        mlir::Value new_probs =
            mlir::stablehlo::MulOp::create(builder, loc, exp_bwd, bcast_inv);
        op->replaceAllUsesWith(mlir::ValueRange{new_probs});
      }
      continue;
    }

    RewriteSubtractMaxInPlace(*shifted_match);

    mlir::Value sum_val = ConvertIfNeeded(builder, loc, sum_reduce.getResult(0),
                                          type.getElementType());
    if (sum_val.getType() != reduced_calc_type) {
      sum_val = mlir::stablehlo::ReshapeOp::create(builder, loc,
                                                   reduced_calc_type, sum_val);
    }
    if (is_div) {
      mlir::Value bcast_sum = mlir::stablehlo::BroadcastInDimOp::create(
          builder, loc, type, sum_val,
          GetDropLastDimBroadcastAttr(builder, rank));
      mlir::Value new_probs =
          mlir::stablehlo::DivOp::create(builder, loc, exp_val, bcast_sum);
      op->replaceAllUsesWith(mlir::ValueRange{new_probs});
    } else {
      mlir::Value one_reduced =
          CreateSplatFloatConst(builder, loc, reduced_calc_type, 1.0);
      mlir::Value inv_sum =
          mlir::stablehlo::DivOp::create(builder, loc, one_reduced, sum_val);
      mlir::Value bcast_inv = mlir::stablehlo::BroadcastInDimOp::create(
          builder, loc, type, inv_sum,
          GetDropLastDimBroadcastAttr(builder, rank));
      mlir::Value new_probs =
          mlir::stablehlo::MulOp::create(builder, loc, exp_val, bcast_inv);
      op->replaceAllUsesWith(mlir::ValueRange{new_probs});
    }
  }
}

// 2. For unmasked or causal SDPA softmax `Div(exp_val,
// BroadcastInDim(sum_exp))` where TPU `online-softmax` cannot apply
// (`(q_len % 128) != 0 || (kv_len % 128) != 0`), rewrites
// `Subtract(masked_logits, BroadcastInDim(max_logits))` to `Add(masked_logits,
// BroadcastInDim(Neg(max_logits)))` so XLA does not duplicate `Reduce(MaxOp)`
// into `(2*K - 1)`-wide padded `reduce-window` fusions.
void CanonicalizeUnalignedForwardSdpaMaxSubtract(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::DivOp, 16> div_ops;
  func.walk([&](mlir::stablehlo::DivOp op) { div_ops.push_back(op); });

  for (mlir::stablehlo::DivOp div_op : div_ops) {
    if (!div_op || div_op->getParentOp() == nullptr || div_op->use_empty() ||
        IsWrappedInRowMaskSelect(div_op.getOperation())) {
      continue;
    }
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(div_op.getType());
    if (!type || !type.hasStaticShape() || type.getRank() <= 1 ||
        !mlir::isa<mlir::FloatType>(type.getElementType())) {
      continue;
    }
    const int64_t rank = type.getRank();
    const int64_t q_len = type.getDimSize(rank - 2);
    const int64_t kv_len = type.getDimSize(rank - 1);
    // GENERIC: hardware TPU 128-lane online-softmax capability check.
    if ((q_len % kTpuLaneWidth == 0) && (kv_len % kTpuLaneWidth == 0)) {
      continue;
    }

    mlir::Value exp_val = div_op.getLhs();
    auto shifted_match = MatchExpOfShiftedLogits(exp_val);
    if (!shifted_match || !shifted_match->shifted_sub ||
        !IsUnmaskedOrCausalAttentionLogits(shifted_match->masked_logits)) {
      continue;
    }
    mlir::Value peeled_denom =
        PeelReshapeBroadcastChain(div_op.getRhs(), type.getShape());
    if (!peeled_denom ||
        !MatchesUnmaskedSumExpOfExpVal(peeled_denom, exp_val, rank - 1)) {
      continue;
    }
    RewriteSubtractMaxInPlace(*shifted_match);
  }
}

[[nodiscard]] bool IsSdpaSoftmaxProbability(mlir::Value val) {
  mlir::Value cur = val;
  while (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    cur = conv.getOperand();
  }
  auto cur_type = mlir::dyn_cast<mlir::RankedTensorType>(cur.getType());
  if (!cur_type || !cur_type.hasStaticShape() || cur_type.getRank() <= 1) {
    return false;
  }
  const int64_t rank = cur_type.getRank();
  if (auto div_op = cur.getDefiningOp<mlir::stablehlo::DivOp>()) {
    if (!div_op.getLhs().getDefiningOp<mlir::stablehlo::ExpOp>()) {
      return false;
    }
    mlir::Value peeled =
        PeelReshapeBroadcastChain(div_op.getRhs(), cur_type.getShape());
    if (!peeled) {
      return false;
    }
    return MatchesUnmaskedSumExpOfExpVal(peeled, div_op.getLhs(), rank - 1);
  }
  if (auto mul_op = cur.getDefiningOp<mlir::stablehlo::MulOp>()) {
    auto check_side = [&](mlir::Value maybe_exp,
                          mlir::Value maybe_bcast) -> bool {
      if (!maybe_exp.getDefiningOp<mlir::stablehlo::ExpOp>()) {
        return false;
      }
      mlir::Value peeled =
          PeelReshapeBroadcastChain(maybe_bcast, cur_type.getShape());
      if (!peeled) {
        return false;
      }
      auto inv_div = peeled.getDefiningOp<mlir::stablehlo::DivOp>();
      if (!inv_div || !IsOneSplatConstant(inv_div.getLhs())) {
        return false;
      }
      return MatchesUnmaskedSumExpOfExpVal(inv_div.getRhs(), maybe_exp,
                                           rank - 1);
    };
    return check_side(mul_op.getLhs(), mul_op.getRhs()) ||
           check_side(mul_op.getRhs(), mul_op.getLhs());
  }
  return false;
}

// 3. Rewrites backward `Subtract(dS, BroadcastInDim(rowsum_reduce))` into
// `Add(dS, BroadcastInDim(Neg(rowsum_reduce)))` for unmasked/causal SDPA
// softmax probabilities so XLA fuses `dP` with `dS` without creating backward
// `reduce-window` fusions.
void CanonicalizeBackwardRowsumSubtract(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::SubtractOp, 16> sub_ops;
  func.walk([&](mlir::stablehlo::SubtractOp op) { sub_ops.push_back(op); });

  for (mlir::stablehlo::SubtractOp sub_op : sub_ops) {
    if (!sub_op || sub_op->getParentOp() == nullptr || sub_op->use_empty()) {
      continue;
    }
    auto sub_type = mlir::dyn_cast<mlir::RankedTensorType>(sub_op.getType());
    if (!sub_type || !sub_type.hasStaticShape() || sub_type.getRank() <= 1 ||
        !mlir::isa<mlir::FloatType>(sub_type.getElementType())) {
      continue;
    }
    const int64_t rank = sub_type.getRank();
    mlir::Value rhs_peeled =
        PeelReshapeBroadcastChain(sub_op.getRhs(), sub_type.getShape());
    if (!rhs_peeled) {
      continue;
    }
    auto rhs_type =
        mlir::dyn_cast<mlir::RankedTensorType>(rhs_peeled.getType());
    if (!rhs_type || rhs_type.getRank() != rank - 1) {
      continue;
    }
    mlir::Value raw_red = rhs_peeled;
    if (auto conv = raw_red.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      raw_red = conv.getOperand();
    }
    mlir::stablehlo::ReduceOp red_op;
    if (!IsSingleDimReduceWithOp<mlir::stablehlo::AddOp>(raw_red, rank - 1,
                                                         &red_op)) {
      continue;
    }
    auto prod_mul =
        red_op.getInputs()[0].getDefiningOp<mlir::stablehlo::MulOp>();
    if (!prod_mul) {
      continue;
    }
    mlir::Value dS = sub_op.getLhs();
    mlir::Value raw_ds = dS;
    while (auto conv = raw_ds.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      raw_ds = conv.getOperand();
    }
    mlir::Value raw_mul_lhs = prod_mul.getLhs();
    while (auto conv =
               raw_mul_lhs.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      raw_mul_lhs = conv.getOperand();
    }
    mlir::Value raw_mul_rhs = prod_mul.getRhs();
    while (auto conv =
               raw_mul_rhs.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      raw_mul_rhs = conv.getOperand();
    }
    mlir::Value probs;
    if (raw_mul_lhs == raw_ds) {
      probs = prod_mul.getRhs();
    } else if (raw_mul_rhs == raw_ds) {
      probs = prod_mul.getLhs();
    } else {
      continue;
    }
    if (!IsSdpaSoftmaxProbability(probs)) {
      continue;
    }

    mlir::OpBuilder builder(sub_op);
    mlir::Location loc = sub_op.getLoc();
    mlir::Value neg_rowsum =
        mlir::stablehlo::NegOp::create(builder, loc, rhs_peeled);
    neg_rowsum =
        ConvertIfNeeded(builder, loc, neg_rowsum, sub_type.getElementType());
    mlir::Value neg_bcast = mlir::stablehlo::BroadcastInDimOp::create(
        builder, loc, sub_type, neg_rowsum,
        GetDropLastDimBroadcastAttr(builder, rank));
    mlir::Value add_val =
        mlir::stablehlo::AddOp::create(builder, loc, dS, neg_bcast);
    sub_op.replaceAllUsesWith(add_val);
  }
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::Block& block : func.getBody()) {
      for (mlir::Operation& op :
           llvm::make_early_inc_range(llvm::reverse(block))) {
        if (op.use_empty() && (mlir::isOpTriviallyDead(&op) ||
                               mlir::isa<mlir::stablehlo::ReduceOp>(&op))) {
          op.erase();
          changed = true;
        }
      }
    }
  }
}

class CanonicalizeUnmaskedSdpaAndBackwardSoftmaxPass
    : public mlir::PassWrapper<CanonicalizeUnmaskedSdpaAndBackwardSoftmaxPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CanonicalizeUnmaskedSdpaAndBackwardSoftmaxPass)

  llvm::StringRef getArgument() const final {
    return "canonicalize-unmasked-sdpa-and-backward-softmax";
  }

  llvm::StringRef getDescription() const final {
    return "Folds provably non-zero unmasked SDPA backward sum_exp guards and "
           "canonicalizes unaligned/cross-attention SDPA softmax and backward "
           "rowsum subtractions.";
  }

  void runOnOperation() final {
    mlir::func::FuncOp func = getOperation();
    EraseTriviallyDeadOps(func);
    CanonicalizeUnmaskedBackwardSdpaSoftmax(func);
    EraseTriviallyDeadOps(func);
    CanonicalizeUnalignedForwardSdpaMaxSubtract(func);
    EraseTriviallyDeadOps(func);
    CanonicalizeBackwardRowsumSubtract(func);
    EraseTriviallyDeadOps(func);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createCanonicalizeUnmaskedSdpaAndBackwardSoftmaxPass() {
  return std::make_unique<CanonicalizeUnmaskedSdpaAndBackwardSoftmaxPass>();
}

}  // namespace torch_tpu
