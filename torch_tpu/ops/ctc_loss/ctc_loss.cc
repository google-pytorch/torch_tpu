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

#include "torch_tpu/ops/ctc_loss/ctc_loss.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

// Computes a numerically stable log-sum-exp over a list of MLIR operations.
// The calculation uses the max value to prevent overflow:
// max_val + log(sum(exp(x_i - max_val))).
mlir::MlirOp LogSumExp(std::vector<mlir::MlirOp> inputs, mlir::MlirOp neg_inf,
                       mlir::MlirOp zero, mlir::MlirOp one) {
  mlir::MlirOp max_val = inputs[0];
  for (size_t i = 1; i < inputs.size(); ++i) {
    max_val = mlir::stablehlo::Max(max_val, inputs[i]);
  }

  auto max_is_neg_inf = mlir::stablehlo::Compare(
      max_val, neg_inf, mlir::stablehlo::ComparisonDirection::EQ);
  auto safe_max_val = mlir::stablehlo::Select(max_is_neg_inf, zero, max_val);

  auto sub_0 = mlir::stablehlo::Subtract(inputs[0], safe_max_val);
  mlir::MlirOp sum_exp = mlir::stablehlo::Exp(sub_0);
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto sub_i = mlir::stablehlo::Subtract(inputs[i], safe_max_val);
    auto exp_i = mlir::stablehlo::Exp(sub_i);
    sum_exp = mlir::stablehlo::Add(sum_exp, exp_i);
  }

  auto sum_exp_is_zero = mlir::stablehlo::Compare(
      sum_exp, zero, mlir::stablehlo::ComparisonDirection::EQ);

  auto safe_sum_exp = mlir::stablehlo::Select(sum_exp_is_zero, one, sum_exp);
  auto log_sum_exp = mlir::stablehlo::Log(safe_sum_exp);
  auto result = mlir::stablehlo::Select(sum_exp_is_zero, neg_inf, log_sum_exp);
  return mlir::stablehlo::Add(result, max_val);
}

// Helper to construct GatherDimensionNumbersAttr, configuring how dimensions
// map during StableHLO gather operations.
mlir::stablehlo::GatherDimensionNumbersAttr GetGatherDimensionNumbers(
    mlir::MLIRContext* ctx, llvm::ArrayRef<int64_t> offset_dims,
    llvm::ArrayRef<int64_t> collapsed_slice_dims,
    llvm::ArrayRef<int64_t> operand_batching_dims,
    llvm::ArrayRef<int64_t> start_indices_batching_dims,
    llvm::ArrayRef<int64_t> start_index_map, int64_t index_vector_dim) {
  return mlir::stablehlo::GatherDimensionNumbersAttr::get(
      ctx, offset_dims, collapsed_slice_dims, operand_batching_dims,
      start_indices_batching_dims, start_index_map, index_vector_dim);
}

// Expands the targets [N, S] to include blanks for CTC alignment.
// A sequence [t_1, t_2, ...] becomes [blank, t_1, blank, t_2, ..., blank].
// The returned sequence has a shape of [N, 2 * S + 1].
absl::StatusOr<mlir::MlirOp> CreateTargetsWithBlanks(mlir::MlirBuilder& builder,
                                                     mlir::MlirOp targets,
                                                     int64_t blank, int64_t N,
                                                     int64_t S,
                                                     mlir::Type element_type) {
  auto blank_const = MakeScalarConstant(builder, blank, element_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp targets_unsqueezed, Unsqueeze(targets, 2));

  auto blanks_ns1 = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({N, S, 1}, element_type), blank_const, {});

  auto stacked = mlir::stablehlo::Concatenate(
      builder, {blanks_ns1, targets_unsqueezed}, 2);

  auto reshaped = mlir::stablehlo::Reshape(
      mlir::RankedTensorType::get({N, 2 * S}, element_type), stacked);

  auto blanks_n1 = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({N, 1}, element_type), blank_const, {});

  return mlir::stablehlo::Concatenate(builder, {reshaped, blanks_n1}, 1);
}

// Constructs the [N, L, 2] tensor of indices used to gather log probabilities.
// For each state index l in the extended sequence L, it pairs the batch index
// with the specific token (or blank) present at that state.
absl::StatusOr<mlir::MlirOp> CreateGatherIndices(
    mlir::MlirBuilder& builder, mlir::MlirOp targets_with_blanks,
    mlir::MlirOp batch_seq, int64_t N, int64_t L, mlir::Type element_type) {
  auto batch_indices = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({N, L}, element_type), batch_seq, {0, 1});

  TT_ASSIGN_OR_RETURN(auto batch_indices_unsqueezed,
                      Unsqueeze(batch_indices, 2));
  TT_ASSIGN_OR_RETURN(auto targets_unsqueezed_3d,
                      Unsqueeze(targets_with_blanks, 2));

  return mlir::stablehlo::Concatenate(
      builder, {batch_indices_unsqueezed, targets_unsqueezed_3d}, 2);
}

struct MaskAndAlphaResult {
  mlir::MlirOp mask_M;   // Transition mask for state u-2.
  mlir::MlirOp alpha_0;  // Initial DP state probabilities at t=0.
};

// Computes the initial forward variables (alpha_0) and the transition mask
// (mask_M).
//
// At t=0, only u=0 (blank) and u=1 (first target token) are valid starting
// states. alpha_0 starts with the initial probabilities for these states and
// -inf for others.
//
// mask_M determines if transitioning from u-2 is allowed. In CTC, skipping a
// blank (moving from u-2 to u) is only valid if u is not a blank and the target
// token at u is different from the target token at u-2.
absl::StatusOr<MaskAndAlphaResult> ComputeInitialMaskAndAlpha(
    mlir::MlirBuilder& builder, mlir::MlirOp targets,
    mlir::MlirOp targets_with_blanks, mlir::MlirOp log_probs,
    mlir::MlirOp batch_seq, int64_t blank, int64_t N, int64_t S, int64_t C,
    int64_t L, mlir::Type targets_element_type,
    mlir::Type log_probs_element_type) {
  auto* ctx = &builder.getContext();
  auto blank_const = MakeScalarConstant(builder, blank, targets_element_type);
  auto blank_n_l = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({N, L}, targets_element_type), blank_const,
      {});
  auto t_not_blank = mlir::stablehlo::Compare(
      targets_with_blanks, blank_n_l, mlir::stablehlo::ComparisonDirection::NE);

  auto log_probs_slice =
      mlir::stablehlo::Slice(log_probs, {0, 0, 0}, {1, N, C}, {1, 1, 1});
  auto log_probs_0 = mlir::stablehlo::Reshape(
      mlir::RankedTensorType::get({N, C}, log_probs_element_type),
      log_probs_slice);

  auto val_0 =
      mlir::stablehlo::Slice(log_probs_0, {0, blank}, {N, blank + 1}, {1, 1});

  mlir::MlirOp mask_M;
  mlir::MlirOp alpha_0;

  if (S == 0) {
    alpha_0 = val_0;
    mask_M = MakeConstantLike(targets_with_blanks, false,
                              builder.getOpBuilder().getI1Type());
  } else {
    // To transition from u-2, the target at u must be different from u-2.
    // E.g., [blank, t1, blank, t2] allows t1 -> t2 without a blank.
    // But [blank, t1, blank, t1] requires a blank between identical tokens.
    auto t_slice_0 =
        mlir::stablehlo::Slice(targets_with_blanks, {0, 0}, {N, L - 2}, {1, 1});

    auto t_slice_2 =
        mlir::stablehlo::Slice(targets_with_blanks, {0, 2}, {N, L}, {1, 1});
    auto t_diff = mlir::stablehlo::Compare(
        t_slice_2, t_slice_0, mlir::stablehlo::ComparisonDirection::NE);
    // Pad the beginning with false because u-2 does not exist for u=0 and u=1.
    auto false_const =
        MakeScalarConstant(builder, false, builder.getOpBuilder().getI1Type());
    auto t_diff_padded =
        mlir::stablehlo::Pad(t_diff, false_const, {0, 2}, {0, 0}, {0, 0});

    // Both conditions: current is not blank, and current != current - 2
    mask_M = mlir::stablehlo::And(t_not_blank, t_diff_padded);

    // Extract the log_prob at t=0 for the first target token (u=1).
    auto target_0 = mlir::stablehlo::Slice(targets, {0, 0}, {N, 1}, {1, 1});
    auto val_1_indices =
        mlir::stablehlo::Concatenate(builder, {batch_seq, target_0}, 1);

    const auto gather_dims_attr_v1 =
        GetGatherDimensionNumbers(ctx, {}, {0, 1}, {}, {}, {0, 1}, 1);

    auto val_1_gathered =
        mlir::stablehlo::Gather(log_probs_0, val_1_indices, gather_dims_attr_v1,
                                {1, 1}, /*indices_are_sorted=*/false);
    TT_ASSIGN_OR_RETURN(auto val_1, Unsqueeze(val_1_gathered, 1));

    // Remaining states are unreachable at t=0, so set them to -infinity.
    auto blanks_remaining =
        MakeConstantLike(t_slice_0, -std::numeric_limits<double>::infinity(),
                         log_probs_element_type);

    alpha_0 = mlir::stablehlo::Concatenate(builder,
                                           {val_0, val_1, blanks_remaining}, 1);
  }

  return MaskAndAlphaResult{mask_M, alpha_0};
}

struct CtcLossInitResult {
  mlir::MlirOp alpha_0;
  mlir::MlirOp mask_M;
  mlir::MlirOp gather_indices;
  mlir::MlirOp neg_inf_const;
  mlir::MlirOp batch_seq;
  mlir::RankedTensorType type_nl_probs;
  int64_t N;
  int64_t S;
  int64_t L;
  int64_t T;
  int64_t C;
};

absl::StatusOr<CtcLossInitResult> InitializeCtcLoss(
    mlir::MlirOp log_probs, mlir::MlirOp targets, int64_t blank,
    mlir::MlirBuilder& builder) {
  const mlir::RankedTensorType targets_type = GetTensorTypeOrDie(targets);
  const mlir::RankedTensorType log_probs_type = GetTensorTypeOrDie(log_probs);
  const int64_t N = targets_type.getDimSize(0);
  const int64_t S = targets_type.getDimSize(1);
  const int64_t L = 2 * S + 1;
  const int64_t T = log_probs_type.getDimSize(0);
  const int64_t C = log_probs_type.getDimSize(2);

  TT_ASSIGN_OR_RETURN(auto targets_with_blanks,
                      CreateTargetsWithBlanks(builder, targets, blank, N, S,
                                              targets_type.getElementType()));

  const auto type_n1_iota =
      mlir::RankedTensorType::get({N, 1}, targets_type.getElementType());
  auto batch_seq = mlir::stablehlo::Iota(builder, type_n1_iota, 0);

  TT_ASSIGN_OR_RETURN(
      auto gather_indices,
      CreateGatherIndices(builder, targets_with_blanks, batch_seq, N, L,
                          targets_type.getElementType()));

  TT_ASSIGN_OR_RETURN(
      auto mask_and_alpha,
      ComputeInitialMaskAndAlpha(builder, targets, targets_with_blanks,
                                 log_probs, batch_seq, blank, N, S, C, L,
                                 targets_type.getElementType(),
                                 log_probs_type.getElementType()));

  auto neg_inf_const =
      MakeScalarConstant(builder, -std::numeric_limits<double>::infinity(),
                         log_probs_type.getElementType());

  const auto type_nl_probs =
      mlir::RankedTensorType::get({N, L}, log_probs_type.getElementType());

  return CtcLossInitResult{mask_and_alpha.alpha_0,
                           mask_and_alpha.mask_M,
                           gather_indices,
                           neg_inf_const,
                           batch_seq,
                           type_nl_probs,
                           N,
                           S,
                           L,
                           T,
                           C};
}

// The core dynamic programming body for the CTC forward algorithm at time t.
// It computes alpha[t, u] given the state alpha[t-1, u].
//
// The new probability is the emission probability at time t added to the
// LogSumExp of:
//   1. alpha_curr: Staying at the same state (u).
//   2. alpha_shift_1: Transitioning from the previous state (u-1).
//   3. alpha_shift_2 (term2): Transitioning from u-2 (if allowed by mask_M).
std::vector<mlir::MlirOp> LoopBodyLogic(
    mlir::RegionBuilder& body, mlir::MlirBuilder& body_builder,
    mlir::MlirOp t_curr, mlir::MlirOp alpha_curr, mlir::MlirOp acc_curr,
    mlir::MlirOp log_probs, mlir::MlirOp gather_indices, mlir::MlirOp mask_M,
    mlir::MlirOp neg_inf_const, mlir::RankedTensorType type_nl_probs, int64_t N,
    int64_t C, int64_t L, mlir::Type log_probs_element_type) {
  const auto ctx = &body_builder.getContext();
  const auto i32_type = body_builder.getOpBuilder().getI32Type();

  // alpha_shift_1 represents transitions from the previous state (s-1).
  auto alpha_padded_1 =
      mlir::MlirOp(body, mlir::stablehlo::Pad(/*operand=*/alpha_curr,
                                              /*padding_value=*/neg_inf_const,
                                              /*edge_padding_low=*/{0, 1},
                                              /*edge_padding_high=*/{0, 0},
                                              /*interior_padding=*/{0, 0})
                             .getValue());
  auto alpha_shift_1 = mlir::MlirOp(
      body, mlir::stablehlo::Slice(/*operand=*/alpha_padded_1,
                                   /*start_indices=*/{0, 0},
                                   /*limit_indices=*/{N, L}, /*strides=*/{1, 1})
                .getValue());

  // alpha_shift_2 represents transitions from the state before previous (s-2).
  auto alpha_padded_2 =
      mlir::MlirOp(body, mlir::stablehlo::Pad(/*operand=*/alpha_curr,
                                              /*padding_value=*/neg_inf_const,
                                              /*edge_padding_low=*/{0, 2},
                                              /*edge_padding_high=*/{0, 0},
                                              /*interior_padding=*/{0, 0})
                             .getValue());
  auto alpha_shift_2 = mlir::MlirOp(
      body, mlir::stablehlo::Slice(/*operand=*/alpha_padded_2,
                                   /*start_indices=*/{0, 0},
                                   /*limit_indices=*/{N, L}, /*strides=*/{1, 1})
                .getValue());

  auto zero_i32 = MakeScalarConstant(body_builder, 0, i32_type);

  auto log_probs_in_body = mlir::MlirOp(body, log_probs.getValue());
  auto log_probs_t = mlir::stablehlo::Reshape(
      mlir::stablehlo::DynamicSlice(
          /*operand=*/log_probs_in_body,
          /*start_indices=*/{t_curr, zero_i32, zero_i32},
          /*slice_sizes=*/{1, N, C}),
      {N, C});

  auto log_probs_mapped = mlir::stablehlo::Gather(
      log_probs_t, gather_indices,
      GetGatherDimensionNumbers(ctx, {}, {0, 1}, {}, {}, {0, 1}, 2), {1, 1},
      /*indices_are_sorted=*/false);

  auto neg_inf_in_body = mlir::MlirOp(body, neg_inf_const.getValue());
  auto neginf_nl =
      mlir::stablehlo::BroadcastInDim(type_nl_probs, neg_inf_in_body, {});

  // mask_M allows transitions from s-2 only when current label is not blank
  // and is different from the label at s-2.
  auto mask_M_in_body = mlir::MlirOp(body, mask_M.getValue());
  auto term2 = mlir::stablehlo::Select(
      /*pred=*/mask_M_in_body, /*on_true=*/alpha_shift_2,
      /*on_false=*/neginf_nl);

  auto zero_scalar =
      MakeScalarConstant(body_builder, 0.0, log_probs_element_type);
  auto zero_nl =
      mlir::stablehlo::BroadcastInDim(type_nl_probs, zero_scalar, {});

  auto one_scalar =
      MakeScalarConstant(body_builder, 1.0, log_probs_element_type);
  auto one_nl = mlir::stablehlo::BroadcastInDim(type_nl_probs, one_scalar, {});

  auto lse_result =
      LogSumExp({alpha_curr, alpha_shift_1, term2}, neginf_nl, zero_nl, one_nl);
  auto alpha_next = mlir::stablehlo::Add(lse_result, log_probs_mapped);

  auto alpha_next_reshaped = mlir::stablehlo::Reshape(alpha_next, {N, 1, L});
  auto acc_next = mlir::stablehlo::DynamicUpdateSlice(
      acc_curr, alpha_next_reshaped, {zero_i32, t_curr, zero_i32});

  auto one_const = MakeScalarConstant(body_builder, 1, i32_type);

  return {mlir::stablehlo::Add(t_curr, one_const), alpha_next, acc_next};
}

// Computes the total CTC loss after the dynamic programming loop has completed.
//
// Extracts the log probabilities at the final valid states for the specific
// length of each sequence in the batch. For a target sequence of length S,
// the valid final states are:
//   s1 = 2 * S      (ending on the trailing blank)
//   s2 = 2 * S - 1  (ending on the last target token)
// The loss is the negative LogSumExp of these final states at t = input_lengths
// - 1.
absl::StatusOr<mlir::MlirOp> ComputeFinalLoss(
    mlir::MlirBuilder& builder, mlir::MlirOp final_log_alpha_acc,
    mlir::MlirOp input_lengths, mlir::MlirOp target_lengths,
    mlir::MlirOp batch_seq, mlir::MlirOp neg_inf_const, int64_t N, int64_t L,
    mlir::Type log_probs_element_type, ZeroInfinity zero_infinity) {
  auto* ctx = &builder.getContext();
  const auto i32_type = builder.getOpBuilder().getI32Type();
  const auto type_n_i32 = mlir::RankedTensorType::get({N}, i32_type);
  auto one_i32_loss = MakeScalarConstant(builder, 1, i32_type);
  auto one_n_i32 =
      mlir::stablehlo::BroadcastInDim(type_n_i32, one_i32_loss, {});

  TT_ASSIGN_OR_RETURN(
      auto last_t_unsqueezed,
      Unsqueeze(mlir::stablehlo::Subtract(input_lengths, one_n_i32), 1));

  auto gather_indices_loss =
      mlir::stablehlo::Concatenate(builder, {batch_seq, last_t_unsqueezed}, 1);
  auto final_alpha = mlir::stablehlo::Gather(
      final_log_alpha_acc, gather_indices_loss,
      GetGatherDimensionNumbers(ctx, {1}, {0, 1}, {}, {}, {0, 1}, 1), {1, 1, L},
      /*indices_are_sorted=*/false);

  auto two_i32_loss = MakeScalarConstant(builder, 2, i32_type);
  auto two_n_i32 =
      mlir::stablehlo::BroadcastInDim(type_n_i32, two_i32_loss, {});
  auto s1 = mlir::stablehlo::Mul(target_lengths, two_n_i32);
  auto s2 = mlir::stablehlo::Subtract(s1, one_n_i32);

  auto zero_s2 = MakeConstantLike(s2, 0);
  auto s2_safe = mlir::stablehlo::Max(s2, zero_s2);

  TT_ASSIGN_OR_RETURN(auto s1_unsqueezed, Unsqueeze(s1, 1));
  TT_ASSIGN_OR_RETURN(auto s2_safe_unsqueezed, Unsqueeze(s2_safe, 1));

  auto gather_dims_attr_scalar =
      GetGatherDimensionNumbers(ctx, {}, {0, 1}, {}, {}, {0, 1}, 1);

  const auto type_n = mlir::RankedTensorType::get({N}, log_probs_element_type);

  auto gather_indices_s1 =
      mlir::stablehlo::Concatenate(builder, {batch_seq, s1_unsqueezed}, 1);
  auto alpha_s1 = mlir::stablehlo::Gather(final_alpha, gather_indices_s1,
                                          gather_dims_attr_scalar, {1, 1},
                                          /*indices_are_sorted=*/false);

  auto gather_indices_s2 =
      mlir::stablehlo::Concatenate(builder, {batch_seq, s2_safe_unsqueezed}, 1);
  auto alpha_s2 = mlir::stablehlo::Gather(final_alpha, gather_indices_s2,
                                          gather_dims_attr_scalar, {1, 1},
                                          /*indices_are_sorted=*/false);

  auto neginf_n = mlir::stablehlo::BroadcastInDim(type_n, neg_inf_const, {});
  auto zero_scalar_s = MakeScalarConstant(builder, 0.0, log_probs_element_type);
  auto zero_n = mlir::stablehlo::BroadcastInDim(type_n, zero_scalar_s, {});
  auto one_scalar_s = MakeScalarConstant(builder, 1.0, log_probs_element_type);
  auto one_n = mlir::stablehlo::BroadcastInDim(type_n, one_scalar_s, {});

  auto zero_i32_const = MakeScalarConstant(builder, 0, i32_type);
  auto zero_n_i32 =
      mlir::stablehlo::BroadcastInDim(type_n_i32, zero_i32_const, {});

  auto lengths_gt_zero = mlir::stablehlo::Compare(
      target_lengths, zero_n_i32, mlir::stablehlo::ComparisonDirection::GT);
  auto lse_alpha = LogSumExp({alpha_s1, alpha_s2}, neginf_n, zero_n, one_n);
  auto final_lse =
      mlir::stablehlo::Select(lengths_gt_zero, lse_alpha, alpha_s1);

  auto loss = mlir::stablehlo::Neg(final_lse);

  if (zero_infinity == ZeroInfinity::kYes) {
    auto is_neg_inf = mlir::stablehlo::Compare(
        final_lse, neginf_n, mlir::stablehlo::ComparisonDirection::EQ);
    loss = mlir::stablehlo::Select(is_neg_inf, zero_n, loss);
  }

  return loss;
}

}  // namespace

// Top-level entry point to construct the StableHLO operations for CTC Loss.
//
// The process consists of:
// 1. Initializing constants, masks, and expanded targets with blanks.
// 2. Creating a while loop that executes the forward dynamic programming
//    algorithm along the time dimension of log_probs.
// 3. Extracting the negative log-likelihood from the accumulated alpha tensor.
absl::StatusOr<std::vector<mlir::MlirOp>> BuildCtcLossShlo(
    mlir::MlirOp log_probs, mlir::MlirOp targets, mlir::MlirOp input_lengths,
    mlir::MlirOp target_lengths, int64_t blank, ZeroInfinity zero_infinity,
    mlir::MlirBuilder& builder) {
  TT_ASSIGN_OR_RETURN(targets,
                      ConvertIfInteger(targets, mlir::ElementType::I32));
  TT_ASSIGN_OR_RETURN(input_lengths,
                      ConvertIfInteger(input_lengths, mlir::ElementType::I32));
  TT_ASSIGN_OR_RETURN(target_lengths,
                      ConvertIfInteger(target_lengths, mlir::ElementType::I32));

  const mlir::RankedTensorType log_probs_type = GetTensorTypeOrDie(log_probs);

  const mlir::RankedTensorType input_lengths_type =
      GetTensorTypeOrDie(input_lengths);
  if (input_lengths_type.getRank() != 1) {
    input_lengths = mlir::stablehlo::Reshape(
        input_lengths, {input_lengths_type.getNumElements()});
  }

  const mlir::RankedTensorType target_lengths_type =
      GetTensorTypeOrDie(target_lengths);
  if (target_lengths_type.getRank() != 1) {
    target_lengths = mlir::stablehlo::Reshape(
        target_lengths, {target_lengths_type.getNumElements()});
  }

  TT_ASSIGN_OR_RETURN(auto init_result,
                      InitializeCtcLoss(log_probs, targets, blank, builder));
  auto alpha_0 = init_result.alpha_0;
  auto neg_inf_const = init_result.neg_inf_const;
  const int64_t N = init_result.N;
  const int64_t L = init_result.L;
  const int64_t T = init_result.T;

  TT_ASSIGN_OR_RETURN(auto alpha_0_unsqueezed, Unsqueeze(alpha_0, 1));

  auto log_alpha_acc_init = mlir::stablehlo::Concatenate(
      builder,
      {alpha_0_unsqueezed,
       mlir::stablehlo::BroadcastInDim(
           mlir::RankedTensorType::get({N, T - 1, L},
                                       log_probs_type.getElementType()),
           neg_inf_const, {})},
      1);

  const auto i32_type = builder.getOpBuilder().getI32Type();
  auto whl = mlir::stablehlo::While(
      builder,
      {MakeScalarConstant(builder, 1, i32_type), alpha_0, log_alpha_acc_init},
      [&](mlir::RegionBuilder& cond) {
        mlir::MlirBuilder cond_builder(cond.getOpBuilder(),
                                       cond.getOpBuilder().getUnknownLoc());
        auto args = mlir::stablehlo::Arguments(
            cond, cond.getOp<mlir::stablehlo::WhileOp>());
        const auto i32_type = cond_builder.getOpBuilder().getI32Type();
        auto t_limit = MakeScalarConstant(cond_builder, T, i32_type);
        mlir::stablehlo::Return(
            cond,
            mlir::stablehlo::Compare(args[0], t_limit,
                                     mlir::stablehlo::ComparisonDirection::LT));
      },
      [&](mlir::RegionBuilder& body) {
        mlir::MlirBuilder body_builder(body.getOpBuilder(),
                                       body.getOpBuilder().getUnknownLoc());
        auto args = mlir::stablehlo::Arguments(
            body, body.getOp<mlir::stablehlo::WhileOp>());

        mlir::stablehlo::Return(
            body, LoopBodyLogic(body, body_builder, args[0], args[1], args[2],
                                log_probs, init_result.gather_indices,
                                init_result.mask_M, neg_inf_const,
                                init_result.type_nl_probs, N, init_result.C, L,
                                log_probs_type.getElementType()));
      });

  TT_ASSIGN_OR_RETURN(
      auto loss,
      ComputeFinalLoss(builder, whl[2], input_lengths, target_lengths,
                       init_result.batch_seq, neg_inf_const, N, L,
                       log_probs_type.getElementType(), zero_infinity));

  return std::vector<mlir::MlirOp>{loss, whl[2]};
}

}  // namespace torch_tpu
