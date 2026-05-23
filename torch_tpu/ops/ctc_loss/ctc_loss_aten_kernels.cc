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

#include "torch_tpu/ops/ctc_loss/ctc_loss_aten_kernels.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/max.h"
#include "ATen/ops/tensor.h"
#include "ATen/ops/zeros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

// Whether to zero infinite losses and gradients. If kYes, infinite
// losses/gradients are zeroed out.
enum class ZeroInfinity { kNo, kYes };

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

// The core dynamic programming body for the CTC backward algorithm at time t.
// It computes beta[t, s] given the state beta[t+1, s].
//
// The new probability is the emission probability at time t added to the
// LogSumExp of:
//   1. beta_curr: Staying at the same state (s).
//   2. beta_shift_1: Transitioning from the next state (s+1).
//   3. beta_shift_2 (term2): Transitioning from s+2 (if allowed by mask_M).
std::vector<mlir::MlirOp> BackwardLoopBodyLogic(
    mlir::RegionBuilder& body, mlir::MlirBuilder& body_builder,
    mlir::MlirOp t_curr, mlir::MlirOp beta_curr, mlir::MlirOp acc_curr,
    mlir::MlirOp log_probs, mlir::MlirOp gather_indices, mlir::MlirOp mask_m,
    mlir::MlirOp input_lengths, mlir::MlirOp target_lengths,
    mlir::MlirOp batch_seq, mlir::MlirOp neg_inf_const,
    mlir::RankedTensorType type_nl_probs, int64_t n, int64_t c, int64_t l,
    mlir::Type log_probs_element_type) {
  const auto ctx = &body_builder.getContext();
  const auto i32_type = body_builder.getOpBuilder().getI32Type();
  const auto i1_type = body_builder.getOpBuilder().getI1Type();

  auto log_probs_in_body = mlir::MlirOp(body, log_probs.getValue());
  auto gather_indices_in_body = mlir::MlirOp(body, gather_indices.getValue());
  auto mask_m_in_body = mlir::MlirOp(body, mask_m.getValue());
  auto input_lengths_in_body = mlir::MlirOp(body, input_lengths.getValue());
  auto target_lengths_in_body = mlir::MlirOp(body, target_lengths.getValue());
  auto neg_inf_const_in_body = mlir::MlirOp(body, neg_inf_const.getValue());

  // beta_shift_1 represents transitions from the next state (s+1).
  auto beta_padded_1 = mlir::stablehlo::Pad(beta_curr, neg_inf_const_in_body,
                                            {0, 0}, {0, 1}, {0, 0});
  auto beta_shift_1 =
      mlir::stablehlo::Slice(beta_padded_1, {0, 1}, {n, l + 1}, {1, 1});

  // beta_shift_2 represents transitions from the state after next (s+2).
  auto beta_padded_2 = mlir::stablehlo::Pad(beta_curr, neg_inf_const_in_body,
                                            {0, 0}, {0, 2}, {0, 0});
  auto beta_shift_2 =
      mlir::stablehlo::Slice(beta_padded_2, {0, 2}, {n, l + 2}, {1, 1});

  auto false_const = MakeScalarConstant(body_builder, false, i1_type);
  auto mask_padded_2 =
      mlir::stablehlo::Pad(mask_m_in_body, false_const, {0, 0}, {0, 2}, {0, 0});
  auto mask_shift_2 =
      mlir::stablehlo::Slice(mask_padded_2, {0, 2}, {n, l + 2}, {1, 1});

  auto neginf_nl =
      mlir::stablehlo::BroadcastInDim(type_nl_probs, neg_inf_const_in_body, {});
  auto zero_scalar =
      MakeScalarConstant(body_builder, 0.0, log_probs_element_type);
  auto zero_nl =
      mlir::stablehlo::BroadcastInDim(type_nl_probs, zero_scalar, {});
  auto one_scalar =
      MakeScalarConstant(body_builder, 1.0, log_probs_element_type);
  auto one_nl = mlir::stablehlo::BroadcastInDim(type_nl_probs, one_scalar, {});

  // term2: Transitioning from s+2 (if allowed by mask_m).
  auto term2 = mlir::stablehlo::Select(mask_shift_2, beta_shift_2, neginf_nl);

  auto lse_result =
      LogSumExp({beta_curr, beta_shift_1, term2}, neginf_nl, zero_nl, one_nl);

  auto zero_i32 = MakeScalarConstant(body_builder, 0, i32_type);
  auto log_probs_t = mlir::stablehlo::Reshape(
      mlir::stablehlo::DynamicSlice(log_probs_in_body,
                                    {t_curr, zero_i32, zero_i32}, {1, n, c}),
      {n, c});

  auto log_probs_mapped = mlir::stablehlo::Gather(
      log_probs_t, gather_indices_in_body,
      GetGatherDimensionNumbers(ctx, {}, {0, 1}, {}, {}, {0, 1}, 2), {1, 1},
      false);

  auto beta_step = mlir::stablehlo::Add(lse_result, log_probs_mapped);

  auto t_curr_nl = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n, l}, i32_type), t_curr, {});

  auto input_lengths_nl = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n, l}, i32_type), input_lengths_in_body,
      {0});
  auto one_const_i32 = MakeScalarConstant(body_builder, 1, i32_type);
  auto one_i32_nl = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n, l}, i32_type), one_const_i32, {});
  auto input_lengths_minus_1 =
      mlir::stablehlo::Subtract(input_lengths_nl, one_i32_nl);

  auto is_final_t =
      mlir::stablehlo::Compare(t_curr_nl, input_lengths_minus_1,
                               mlir::stablehlo::ComparisonDirection::EQ);
  auto is_past_t =
      mlir::stablehlo::Compare(t_curr_nl, input_lengths_minus_1,
                               mlir::stablehlo::ComparisonDirection::GT);

  auto two_const_i32 = MakeScalarConstant(body_builder, 2, i32_type);
  auto two_n_i32 = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n}, i32_type), two_const_i32, {});
  auto s1 = mlir::stablehlo::Mul(target_lengths_in_body, two_n_i32);

  auto one_n_i32 = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n}, i32_type), one_const_i32, {});
  auto s2 = mlir::stablehlo::Subtract(s1, one_n_i32);

  auto s1_nl = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n, l}, i32_type), s1, {0});
  auto s2_nl = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n, l}, i32_type), s2, {0});

  auto state_indices = mlir::stablehlo::Iota(
      body_builder, mlir::RankedTensorType::get({n, l}, i32_type), 1);

  auto is_s1 = mlir::stablehlo::Compare(
      state_indices, s1_nl, mlir::stablehlo::ComparisonDirection::EQ);
  auto is_s2 = mlir::stablehlo::Compare(
      state_indices, s2_nl, mlir::stablehlo::ComparisonDirection::EQ);

  auto zero_n_i32 = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n}, i32_type), zero_i32, {});
  auto target_lens_gt_0 =
      mlir::stablehlo::Compare(target_lengths_in_body, zero_n_i32,
                               mlir::stablehlo::ComparisonDirection::GT);
  auto target_lens_gt_0_nl = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n, l}, i1_type), target_lens_gt_0, {0});

  auto valid_s2 = mlir::stablehlo::And(is_s2, target_lens_gt_0_nl);
  auto is_init_state = mlir::stablehlo::Or(is_s1, valid_s2);

  auto beta_init =
      mlir::stablehlo::Select(is_init_state, log_probs_mapped, neginf_nl);

  auto beta_next_inner =
      mlir::stablehlo::Select(is_final_t, beta_init, beta_step);
  auto beta_next =
      mlir::stablehlo::Select(is_past_t, neginf_nl, beta_next_inner);

  auto beta_next_reshaped = mlir::stablehlo::Reshape(beta_next, {n, 1, l});
  auto acc_next = mlir::stablehlo::DynamicUpdateSlice(
      acc_curr, beta_next_reshaped, {zero_i32, t_curr, zero_i32});

  auto t_next = mlir::stablehlo::Subtract(t_curr, one_const_i32);

  return {t_next, beta_next, acc_next};
}

mlir::MlirOp ReduceAddDim2(mlir::MlirBuilder& builder, mlir::MlirOp input,
                           mlir::Type element_type) {
  auto zero_scalar = MakeScalarConstant(builder, 0.0, element_type);
  auto zero = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({}, element_type), zero_scalar, {});
  return mlir::stablehlo::Reduce(
      builder, {input}, {zero},
      [&](mlir::RegionBuilder& region) {
        auto type = mlir::RankedTensorType::get({}, element_type);
        auto arg0 = mlir::Argument(region, type);
        auto arg1 = mlir::Argument(region, type);
        mlir::stablehlo::Return(region, mlir::stablehlo::Add(arg0, arg1));
      },
      {2})[0];
}

// Top-level entry point to construct the StableHLO operations for
// CTC Loss Backward.
//
// The process consists of:
// 1. Initializing constants, masks, and expanded targets with blanks.
// 2. Creating a while loop that executes the backward dynamic programming
//    algorithm along the time dimension of log_probs from T-1 down to 0.
// 3. Accumulating alpha and beta probabilities to calculate the gradient with
//    respect to log_probs.
absl::StatusOr<mlir::MlirOp> BuildCtcLossBackwardShlo(
    mlir::MlirOp grad_out, mlir::MlirOp log_probs, mlir::MlirOp targets,
    mlir::MlirOp input_lengths, mlir::MlirOp target_lengths,
    mlir::MlirOp neg_log_likelihood, mlir::MlirOp log_alpha, int64_t blank,
    ZeroInfinity zero_infinity, mlir::MlirBuilder& builder) {
  TT_ASSIGN_OR_RETURN(targets,
                      ConvertIfInteger(targets, mlir::ElementType::I32));
  TT_ASSIGN_OR_RETURN(input_lengths,
                      ConvertIfInteger(input_lengths, mlir::ElementType::I32));
  TT_ASSIGN_OR_RETURN(target_lengths,
                      ConvertIfInteger(target_lengths, mlir::ElementType::I32));

  const mlir::RankedTensorType log_probs_type = GetTensorTypeOrDie(log_probs);
  auto element_type = log_probs_type.getElementType();
  const auto i32_type = builder.getOpBuilder().getI32Type();

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
  auto neg_inf_const = init_result.neg_inf_const;
  const int64_t n = init_result.N;
  const int64_t l = init_result.L;
  const int64_t t = init_result.T;
  const int64_t c = init_result.C;

  auto neginf_nl = mlir::stablehlo::BroadcastInDim(init_result.type_nl_probs,
                                                   neg_inf_const, {});
  auto beta_0 = neginf_nl;
  auto log_beta_acc_init = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n, t, l}, element_type), neg_inf_const, {});

  auto t_start = MakeScalarConstant(builder, t - 1, i32_type);

  auto whl = mlir::stablehlo::While(
      builder, {t_start, beta_0, log_beta_acc_init},
      [&](mlir::RegionBuilder& cond) {
        mlir::MlirBuilder cond_builder(cond.getOpBuilder(),
                                       cond.getOpBuilder().getUnknownLoc());
        auto args = mlir::stablehlo::Arguments(
            cond, cond.getOp<mlir::stablehlo::WhileOp>());
        auto zero_i32 = MakeScalarConstant(cond_builder, 0, i32_type);
        mlir::stablehlo::Return(
            cond,
            mlir::stablehlo::Compare(args[0], zero_i32,
                                     mlir::stablehlo::ComparisonDirection::GE));
      },
      [&](mlir::RegionBuilder& body) {
        mlir::MlirBuilder body_builder(body.getOpBuilder(),
                                       body.getOpBuilder().getUnknownLoc());
        auto args = mlir::stablehlo::Arguments(
            body, body.getOp<mlir::stablehlo::WhileOp>());

        mlir::stablehlo::Return(
            body,
            BackwardLoopBodyLogic(
                body, body_builder, args[0], args[1], args[2], log_probs,
                init_result.gather_indices, init_result.mask_M, input_lengths,
                target_lengths, init_result.batch_seq, neg_inf_const,
                init_result.type_nl_probs, n, c, l, element_type));
      });

  auto log_beta = whl[2];

  auto sum_alpha_beta = mlir::stablehlo::Add(log_alpha, log_beta);
  auto nll_ntl = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n, t, l}, element_type), neg_log_likelihood,
      {0});
  auto log_prob_term = mlir::stablehlo::Add(sum_alpha_beta, nll_ntl);

  auto batch_indices = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n, t, l}, i32_type), init_result.batch_seq,
      {0, 2});
  auto t_indices = mlir::stablehlo::Iota(
      builder, mlir::RankedTensorType::get({n, t, l}, i32_type), 1);

  TT_ASSIGN_OR_RETURN(auto targets_with_blanks,
                      CreateTargetsWithBlanks(builder, targets, blank, n,
                                              init_result.S, i32_type));
  auto target_indices = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n, t, l}, i32_type), targets_with_blanks,
      {0, 2});

  TT_ASSIGN_OR_RETURN(auto batch_indices_u, Unsqueeze(batch_indices, 3));
  TT_ASSIGN_OR_RETURN(auto t_indices_u, Unsqueeze(t_indices, 3));
  TT_ASSIGN_OR_RETURN(auto target_indices_u, Unsqueeze(target_indices, 3));
  auto gather_indices = mlir::stablehlo::Concatenate(
      builder, {t_indices_u, batch_indices_u, target_indices_u}, 3);

  auto log_probs_ntl = mlir::stablehlo::Gather(
      log_probs, gather_indices,
      GetGatherDimensionNumbers(&builder.getContext(), {}, {0, 1, 2}, {}, {},
                                {0, 1, 2}, 3),
      {1, 1, 1}, false);

  log_prob_term = mlir::stablehlo::Subtract(log_prob_term, log_probs_ntl);

  auto prob_term = mlir::stablehlo::Exp(log_prob_term);

  auto c_iota = mlir::stablehlo::Iota(
      builder, mlir::RankedTensorType::get({c}, i32_type), 0);
  auto c_iota_ntlc = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n, t, l, c}, i32_type), c_iota, {3});
  auto target_indices_ntlc = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n, t, l, c}, i32_type), target_indices,
      {0, 1, 2});

  auto mask =
      mlir::stablehlo::Compare(target_indices_ntlc, c_iota_ntlc,
                               mlir::stablehlo::ComparisonDirection::EQ);
  auto prob_ntlc = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n, t, l, c}, element_type), prob_term,
      {0, 1, 2});

  auto zero_scalar = MakeScalarConstant(builder, 0.0, element_type);
  auto zero_ntlc = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({n, t, l, c}, element_type), zero_scalar, {});

  auto selected_prob = mlir::stablehlo::Select(mask, prob_ntlc, zero_ntlc);

  auto prob_ntc = ReduceAddDim2(builder, selected_prob, element_type);

  auto prob_tnc = mlir::stablehlo::Transpose(prob_ntc, {1, 0, 2});

  auto log_probs_exp = mlir::stablehlo::Exp(log_probs);
  auto grad_inner = mlir::stablehlo::Subtract(log_probs_exp, prob_tnc);

  const mlir::RankedTensorType grad_out_type = GetTensorTypeOrDie(grad_out);
  Dimensions grad_out_bcast_dims;
  if (grad_out_type.getRank() == 1) {
    grad_out_bcast_dims = {1};
  } else {
    grad_out_bcast_dims = {};
  }
  auto grad_out_tnc = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({t, n, c}, element_type), grad_out,
      grad_out_bcast_dims);
  auto grad_final = mlir::stablehlo::Mul(grad_inner, grad_out_tnc);

  auto zero_tnc = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({t, n, c}, element_type), zero_scalar, {});

  if (zero_infinity == ZeroInfinity::kYes) {
    auto inf_scalar = MakeScalarConstant(
        builder, std::numeric_limits<double>::infinity(), element_type);
    auto inf_n = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({n}, element_type), inf_scalar, {});
    auto nll_is_inf = mlir::stablehlo::Compare(
        neg_log_likelihood, inf_n, mlir::stablehlo::ComparisonDirection::EQ);
    auto nll_is_inf_tnc = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({t, n, c},
                                    builder.getOpBuilder().getI1Type()),
        nll_is_inf, {1});

    grad_final = mlir::stablehlo::Select(nll_is_inf_tnc, zero_tnc, grad_final);
  }

  auto t_iota = mlir::stablehlo::Iota(
      builder, mlir::RankedTensorType::get({t}, i32_type), 0);
  auto t_iota_tnc = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({t, n, c}, i32_type), t_iota, {0});
  auto input_lengths_tnc = mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get({t, n, c}, i32_type), input_lengths, {1});
  auto is_valid_t = mlir::stablehlo::Compare(
      t_iota_tnc, input_lengths_tnc, mlir::stablehlo::ComparisonDirection::LT);

  grad_final = mlir::stablehlo::Select(is_valid_t, grad_final, zero_tnc);

  return grad_final;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> AtenCtcLoss(const at::Tensor& log_probs,
                                               const at::Tensor& targets,
                                               at::IntArrayRef input_lengths,
                                               at::IntArrayRef target_lengths,
                                               int64_t blank,
                                               bool zero_infinity) {
  TT_KERNEL(OpName::kCtcLoss, _,
            (log_probs, targets,
             IgnoreInCacheKey(input_lengths, "delegates to AtenCtcLossTensor"),
             IgnoreInCacheKey(target_lengths, "delegates to AtenCtcLossTensor"),
             IgnoreInCacheKey(blank, "delegates to AtenCtcLossTensor"),
             IgnoreInCacheKey(zero_infinity, "delegates to AtenCtcLossTensor")),
            {
              at::Tensor input_lengths_tensor =
                  at::tensor(input_lengths, at::kLong).to(log_probs.device());
              at::Tensor target_lengths_tensor =
                  at::tensor(target_lengths, at::kLong).to(log_probs.device());
              return AtenCtcLossTensor(log_probs, targets, input_lengths_tensor,
                                       target_lengths_tensor, blank,
                                       zero_infinity);
            });
}

std::tuple<at::Tensor, at::Tensor> AtenCtcLossTensor(
    const at::Tensor& log_probs, const at::Tensor& targets,
    const at::Tensor& input_lengths, const at::Tensor& target_lengths,
    int64_t blank, bool zero_infinity) {
  TT_KERNEL(
      OpName::kCtcLossTensor, param_keys,
      (log_probs, targets, input_lengths, target_lengths, blank, zero_infinity),
      {
        TT_CHECK_THROW(log_probs.dim() == 3, error::kInvalidArgument)
            << "expected log_probs to be 3-D, got " << log_probs.dim() << "-D";
        TT_CHECK_THROW(targets.dim() == 1 || targets.dim() == 2,
                       error::kInvalidArgument)
            << "expected targets to be 1-D or 2-D, got " << targets.dim()
            << "-D";

        const int64_t batch_size = log_probs.size(1);
        TT_CHECK_THROW(input_lengths.numel() == batch_size,
                       error::kInvalidArgument)
            << "expected input_lengths to have batch_size (" << batch_size
            << ") elements, got " << input_lengths.numel();
        TT_CHECK_THROW(target_lengths.numel() == batch_size,
                       error::kInvalidArgument)
            << "expected target_lengths to have batch_size (" << batch_size
            << ") elements, got " << target_lengths.numel();

        const int64_t n = log_probs.size(1);
        const int64_t t = log_probs.size(0);

        at::Tensor padded_targets = targets;
        if (targets.dim() == 1) {
          at::Tensor target_lengths_cpu =
              target_lengths.to(at::kLong).cpu().contiguous();
          auto lengths_accessor = target_lengths_cpu.accessor<int64_t, 1>();
          const int64_t max_target_length =
              target_lengths_cpu.numel() > 0
                  ? at::max(target_lengths_cpu).item<int64_t>()
                  : 0;

          padded_targets = at::zeros({n, max_target_length}, targets.options());
          int64_t offset = 0;
          for (int64_t i = 0; i < n; ++i) {
            int64_t len = lengths_accessor[i];
            if (len > 0) {
              padded_targets.select(0, i).narrow(0, 0, len).copy_(
                  targets.narrow(0, offset, len));
            }
            offset += len;
          }
        }

        const int64_t s = padded_targets.size(1);
        const int64_t l = 2 * s + 1;

        TT_ASSIGN_OR_THROW(
            mlir::ElementType output_dtype,
            ConvertTo<mlir::ElementType>(log_probs.scalar_type()));
        const std::array<mlir::ElementType, 2> out_dtypes = {output_dtype,
                                                             output_dtype};

        const Dimensions loss_dims = {n};
        const Dimensions log_alpha_dims = {n, t, l};

        const std::array<absl::Span<const int64_t>, 2> out_dims_list = {
            absl::MakeConstSpan(loss_dims),
            absl::MakeConstSpan(log_alpha_dims)};

        DispatchOpOptions<2> options = {
            .out_dtypes = out_dtypes,
            .out_dims_list = out_dims_list,
            .op_param_cache_keys = std::move(param_keys),
        };

        auto op_builder = [blank,
                           zero_infinity](FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<MlirOpResults<2>> {
          auto& [log_probs_op, targets_op, input_lengths_op,
                 target_lengths_op] = inputs;
          auto& builder = log_probs_op.getBuilder();
          TT_ASSIGN_OR_RETURN(
              auto results,
              BuildCtcLossShlo(
                  log_probs_op, targets_op, input_lengths_op, target_lengths_op,
                  blank, zero_infinity ? ZeroInfinity::kYes : ZeroInfinity::kNo,
                  builder));
          return MlirOpResults<2>{results[0], results[1]};
        };

        TT_ASSIGN_OR_THROW(auto output_bufs,
                           (DispatchOp<4, 2>(std::move(op_builder),
                                             {log_probs, padded_targets,
                                              input_lengths, target_lengths},
                                             std::move(options))));

        at::Tensor loss = at::empty(loss_dims, log_probs.options());
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(output_bufs[0]), loss));

        at::Tensor log_alpha = at::empty(log_alpha_dims, log_probs.options());
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(output_bufs[1]), log_alpha));

        return std::make_tuple(loss, log_alpha);
      });
}

at::Tensor AtenCtcLossBackward(
    const at::Tensor& grad_out, const at::Tensor& log_probs,
    const at::Tensor& targets, at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths, const at::Tensor& neg_log_likelihood,
    const at::Tensor& log_alpha, int64_t blank, bool zero_infinity) {
  TT_KERNEL(OpName::kCtcLossBackward, _,
            (grad_out, log_probs, targets,
             IgnoreInCacheKey(input_lengths,
                              "delegates to AtenCtcLossBackwardTensor"),
             IgnoreInCacheKey(target_lengths,
                              "delegates to AtenCtcLossBackwardTensor"),
             neg_log_likelihood, log_alpha,
             IgnoreInCacheKey(blank, "delegates to AtenCtcLossBackwardTensor"),
             IgnoreInCacheKey(zero_infinity,
                              "delegates to AtenCtcLossBackwardTensor")),
            {
              at::Tensor input_lengths_tensor =
                  at::tensor(input_lengths, at::kLong).to(log_probs.device());
              at::Tensor target_lengths_tensor =
                  at::tensor(target_lengths, at::kLong).to(log_probs.device());
              return AtenCtcLossBackwardTensor(
                  grad_out, log_probs, targets, input_lengths_tensor,
                  target_lengths_tensor, neg_log_likelihood, log_alpha, blank,
                  zero_infinity);
            });
}

at::Tensor AtenCtcLossBackwardTensor(
    const at::Tensor& grad_out, const at::Tensor& log_probs,
    const at::Tensor& targets, const at::Tensor& input_lengths,
    const at::Tensor& target_lengths, const at::Tensor& neg_log_likelihood,
    const at::Tensor& log_alpha, int64_t blank, bool zero_infinity) {
  TT_KERNEL(
      OpName::kCtcLossBackwardTensor, param_keys,
      (grad_out, log_probs, targets, input_lengths, target_lengths,
       neg_log_likelihood, log_alpha, blank, zero_infinity),
      {
        const int64_t n = log_probs.size(1);

        TT_CHECK_THROW(log_probs.dim() == 3, error::kInvalidArgument)
            << "expected log_probs to be 3-D, got " << log_probs.dim() << "-D";
        TT_CHECK_THROW(targets.dim() == 1 || targets.dim() == 2,
                       error::kInvalidArgument)
            << "expected targets to be 1-D or 2-D, got " << targets.dim()
            << "-D";

        TT_CHECK_THROW(input_lengths.numel() == n, error::kInvalidArgument)
            << "expected input_lengths to have batch_size (" << n
            << ") elements, got " << input_lengths.numel();
        TT_CHECK_THROW(target_lengths.numel() == n, error::kInvalidArgument)
            << "expected target_lengths to have batch_size (" << n
            << ") elements, got " << target_lengths.numel();

        at::Tensor padded_targets = targets;
        if (targets.dim() == 1) {
          at::Tensor target_lengths_cpu =
              target_lengths.to(at::kLong).cpu().contiguous();
          auto lengths_accessor = target_lengths_cpu.accessor<int64_t, 1>();
          int64_t max_target_length = 0;
          for (int64_t i = 0; i < target_lengths_cpu.numel(); ++i) {
            max_target_length =
                std::max(max_target_length, lengths_accessor[i]);
          }

          padded_targets = AtenEfficientZeroTensor(
              {n, max_target_length}, targets.scalar_type(), std::nullopt,
              targets.device(), std::nullopt);
          int64_t offset = 0;
          for (int64_t i = 0; i < n; ++i) {
            int64_t len = lengths_accessor[i];
            if (len > 0) {
              padded_targets.select(0, i).narrow(0, 0, len).copy_(
                  targets.narrow(0, offset, len));
            }
            offset += len;
          }
        }

        TT_ASSIGN_OR_THROW(
            mlir::ElementType output_dtype,
            ConvertTo<mlir::ElementType>(log_probs.scalar_type()));
        const Dimensions grad_in_dims = {log_probs.size(0), log_probs.size(1),
                                         log_probs.size(2)};

        DispatchOpOptions<1> options = {
            .out_dtype = output_dtype,
            .out_dims = absl::MakeConstSpan(grad_in_dims),
            .op_param_cache_keys = std::move(param_keys),
        };

        auto op_builder = [blank,
                           zero_infinity](FixedSizeSpan<mlir::MlirOp, 7> inputs)
            -> absl::StatusOr<MlirOpResults<1>> {
          auto& [grad_out_op, log_probs_op, targets_op, input_lengths_op,
                 target_lengths_op, neg_log_likelihood_op, log_alpha_op] =
              inputs;
          auto& builder = log_probs_op.getBuilder();
          TT_ASSIGN_OR_RETURN(
              auto result,
              BuildCtcLossBackwardShlo(
                  grad_out_op, log_probs_op, targets_op, input_lengths_op,
                  target_lengths_op, neg_log_likelihood_op, log_alpha_op, blank,
                  zero_infinity ? ZeroInfinity::kYes : ZeroInfinity::kNo,
                  builder));
          return MlirOpResults<1>{result};
        };

        TT_ASSIGN_OR_THROW(
            auto output_bufs,
            (DispatchOp<7, 1>(
                std::move(op_builder),
                {grad_out, log_probs, padded_targets, input_lengths,
                 target_lengths, neg_log_likelihood, log_alpha},
                std::move(options))));

        TT_ASSIGN_OR_THROW(
            at::Tensor grad_in,
            MakeEmptyTensor(grad_in_dims, log_probs.scalar_type(),
                            log_probs.device()));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(output_bufs), grad_in));

        return grad_in;
      });
}

}  // namespace torch_tpu
