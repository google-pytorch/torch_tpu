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

#include "csrc/ops/scaled_dot_product_attention/scaled_dot_product_attention_shlo.h"

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/Context.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBase.h"
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/scaled_dot_product_attention/helpers.h"
#include "csrc/ops/view_decomposition/contiguous_to_view.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

struct AttentionPrepResults {
  mlir::MlirOp query_4d;
  mlir::MlirOp key_4d;
  mlir::MlirOp value_4d;
  mlir::MlirOp shifted_attn_logits;
  mlir::MlirOp scale_value;
  int64_t head_count_ratio;
};
mlir::MlirOp GetNegInf(mlir::MlirBuilder& builder,
                       mlir::FloatType element_type) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  llvm::APFloat neg_inf = llvm::APFloat::getInf(
      element_type.getFloatSemantics(), /*Negative=*/true);
  return mlir::MlirOp(builder,
                      mlir::stablehlo::ConstantOp::create(
                          op_builder, builder.getLoc(),
                          op_builder.getFloatAttr(element_type, neg_inf)));
}

mlir::MlirOp GetCausalMask(mlir::MlirBuilder& builder,
                           mlir::ArrayRef<int64_t> shape) {
  auto iota_type =
      mlir::RankedTensorType::get(shape, builder.getOpBuilder().getI32Type());
  mlir::MlirOp iota_i = mlir::stablehlo::Iota(builder, iota_type,
                                              /*dimension=*/shape.size() - 2);
  mlir::MlirOp iota_j = mlir::stablehlo::Iota(builder, iota_type,
                                              /*dimension=*/shape.size() - 1);

  return mlir::stablehlo::Compare(iota_i, iota_j,
                                  mlir::stablehlo::ComparisonDirection::LT);
}

enum class SdpaPromotionType {
  kNone,
  kSoftmaxOnly,
  kWholeModule,
};

// Determine exactly where type promotion should happen inside the attn module.
// Softmax mode emits a pattern replaced by XLA with an optimized kernel; we
// use a simple heuristic based on tensor size and sequence alignment to decide
// when that kernel is beneficial. Otherwise we fall back to full precision.
SdpaPromotionType GetSdpaPromotionType(
    mlir::MlirOp query, mlir::MlirOp key,
    bool allow_half_precision_reduction_math) {
  if (allow_half_precision_reduction_math) {
    return SdpaPromotionType::kNone;
  }

  // Softmax-only promotion requires >500k elements to offset kernel launch
  // overhead, and sequence lengths <=8192 aligned to 128 (or 32 for >=2048).
  constexpr int64_t kMinTotalElements = 500000;
  constexpr int64_t kDefaultSeqLenAlignment = 128;
  constexpr int64_t kLargeSeqLenThreshold = 2048;
  constexpr int64_t kLargeSeqLenAlignment = 32;
  constexpr int64_t kMaxSupportedSeqLen = 8192;

  auto query_shape = GetTensorTypeOrDie(query).getShape();
  auto key_shape = GetTensorTypeOrDie(key).getShape();

  int64_t total_elements = 1;
  for (auto dim : query_shape) {
    total_elements *= dim;
  }

  const int query_rank = query_shape.size();
  const int key_rank = key_shape.size();
  bool sequence_length_supported = true;
  bool sequence_length_too_large = false;
  if (query_rank >= 2 && key_rank >= 2) {
    const int64_t seq_len_q = query_shape[query_rank - 2];
    const int64_t seq_len_kv = key_shape[key_rank - 2];
    const bool is_multiple_of_128 =
        (seq_len_q % kDefaultSeqLenAlignment == 0) &&
        (seq_len_kv % kDefaultSeqLenAlignment == 0);
    const bool is_large_multiple_of_32 =
        (seq_len_q >= kLargeSeqLenThreshold &&
         seq_len_kv >= kLargeSeqLenThreshold) &&
        (seq_len_q % kLargeSeqLenAlignment == 0) &&
        (seq_len_kv % kLargeSeqLenAlignment == 0);
    sequence_length_supported = is_multiple_of_128 || is_large_multiple_of_32;
    sequence_length_too_large =
        (seq_len_q > kMaxSupportedSeqLen) || (seq_len_kv > kMaxSupportedSeqLen);
  }

  const bool should_fallback = (total_elements < kMinTotalElements) ||
                               !sequence_length_supported ||
                               sequence_length_too_large;

  return should_fallback ? SdpaPromotionType::kWholeModule
                         : SdpaPromotionType::kSoftmaxOnly;
}

absl::StatusOr<AttentionPrepResults> PrepareAttentionLogits(
    mlir::MlirBuilder& builder, mlir::MLIRContext* context, mlir::MlirOp query,
    mlir::MlirOp key, mlir::MlirOp value, std::optional<mlir::MlirOp> mask,
    bool is_causal, std::optional<double> scale) {
  auto query_tensor_type = GetTensorTypeOrDie(query);
  auto key_tensor_type = GetTensorTypeOrDie(key);
  int rank = query_tensor_type.getRank();
  int batch_size = get_batch_size(query_tensor_type.getShape());
  int64_t head_dim = query_tensor_type.getDimSize(rank - 1);
  int64_t seq_len_q = query_tensor_type.getDimSize(rank - 2);
  int64_t seq_len_kv = key_tensor_type.getDimSize(rank - 2);
  int64_t num_head_q = query_tensor_type.getDimSize(rank - 3);
  int64_t num_head_kv = key_tensor_type.getDimSize(rank - 3);
  auto element_type =
      mlir::cast<mlir::FloatType>(query_tensor_type.getElementType());

  mlir::MlirOp query_4d = flatten_batch_dims(query, batch_size, rank - 3);
  mlir::MlirOp key_4d = flatten_batch_dims(key, batch_size, rank - 3);
  mlir::MlirOp value_4d = flatten_batch_dims(value, batch_size, rank - 3);

  int64_t head_count_ratio = 1;
  if (num_head_kv < num_head_q) {
    if (num_head_q % num_head_kv != 0) {
      return TT_ERROR(error::kInvalidArgument)
             << "num_head_q must be divisible by num_head_kv";
    }
    head_count_ratio = num_head_q / num_head_kv;
  }

  // Scale Q before Dot product - matches pytorch attention implementation and
  // improves stability https://tinyurl.com/sudb9s96
  mlir::MlirOp scale_value =
      GetScaleDefaulted(builder, scale, head_dim, element_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_scale_value,
                      BroadcastIfNeeded(scale_value, query_4d));
  mlir::MlirOp scaled_query_4d =
      mlir::stablehlo::Mul(query_4d, broadcasted_scale_value);

  mlir::MlirOp scaled_attn_logits;
  if (head_count_ratio > 1) {
    mlir::MlirOp scaled_query_5d = mlir::stablehlo::Reshape(
        scaled_query_4d,
        {batch_size, num_head_kv, head_count_ratio, seq_len_q, head_dim});
    auto attention_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        context, /*lhs_batching_dimensions=*/{0, 1},
        /*rhs_batching_dimensions=*/{0, 1}, /*lhs_contracting_dimensions=*/{4},
        /*rhs_contracting_dimensions=*/{3});
    mlir::MlirOp logits_5d = mlir::stablehlo::DotGeneral(
        scaled_query_5d, key_4d, attention_dot_dims);
    scaled_attn_logits = mlir::stablehlo::Reshape(
        logits_5d, {batch_size, num_head_q, seq_len_q, seq_len_kv});
  } else {
    auto attention_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        context, /*lhs_batching_dimensions=*/{0, 1},
        /*rhs_batching_dimensions=*/{0, 1}, /*lhs_contracting_dimensions=*/{3},
        /*rhs_contracting_dimensions=*/{3});
    scaled_attn_logits = mlir::stablehlo::DotGeneral(scaled_query_4d, key_4d,
                                                     attention_dot_dims);
  }

  mlir::Attribute min_attr =
      GetMinFiniteValueAttr(element_type, builder.getOpBuilder());
  mlir::MlirOp min_val = mlir::MlirOp(
      builder, mlir::stablehlo::ConstantOp::create(builder.getOpBuilder(),
                                                   builder.getLoc(), min_attr));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask_out_value,
                      BroadcastIfNeeded(min_val, scaled_attn_logits));

  mlir::MlirOp masked_attn_logits = scaled_attn_logits;
  if (is_causal) {
    auto shape = GetTensorTypeOrDie(scaled_attn_logits).getShape();
    mlir::MlirOp causal_mask = GetCausalMask(builder, shape);
    masked_attn_logits = mlir::stablehlo::Select(causal_mask, mask_out_value,
                                                 scaled_attn_logits);
  } else if (mask) {
    // PyTorch converts boolean masks to float before passing them to the
    // attention op.
    if (IsBooleanType(GetTensorTypeOrDie(*mask))) {
      return TT_ERROR(error::kInvalidArgument)
             << "Boolean mask is not supported";
    }

    int64_t mask_rank = GetTensorTypeOrDie(*mask).getRank();
    mlir::MlirOp flat_mask =
        mask_rank > 4 ? flatten_batch_dims(*mask, batch_size, rank - 3) : *mask;
    TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_mask,
                        BroadcastIfNeeded(flat_mask, scaled_attn_logits));

    // We clamp the mask so that it is finite. This avoids the issue where a
    // whole row is masked out resulting in a divide by zero in the softmax.
    // The reason this works is because when we shift the logits by the max
    // value the result is that the entire row will then just be zero ->
    // exp(0) = 1.
    mlir::MlirOp clamped_mask =
        mlir::stablehlo::Max(broadcasted_mask, mask_out_value);
    masked_attn_logits = mlir::stablehlo::Add(scaled_attn_logits, clamped_mask);
  }

  auto max_reduce_builder = [element_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::MaxOp>(
        element_type, rb.getRegion(), rb.getOpBuilder());
  };
  mlir::MlirOp minus_inf =
      GetNegInf(builder, mlir::cast<mlir::FloatType>(element_type));
  mlir::MlirOp max_logits =
      mlir::stablehlo::Reduce(builder, {masked_attn_logits},
                              /*init_value=*/{minus_inf}, max_reduce_builder,
                              /*dimensions=*/{3})[0];
  TT_ASSIGN_OR_RETURN(mlir::MlirOp max_logits_broadcasted,
                      BroadcastIfNeeded(max_logits, masked_attn_logits,
                                        /*broadcast_dimensions=*/{0, 1, 2}));
  mlir::MlirOp shifted_attn_logits =
      mlir::stablehlo::Subtract(masked_attn_logits, max_logits_broadcasted);

  return AttentionPrepResults{.query_4d = query_4d,
                              .key_4d = key_4d,
                              .value_4d = value_4d,
                              .shifted_attn_logits = shifted_attn_logits,
                              .scale_value = scale_value,
                              .head_count_ratio = head_count_ratio};
}

mlir::MlirOp SumReduce(mlir::MlirBuilder& builder, mlir::MlirOp input,
                       int dimension) {
  auto element_type = GetTensorTypeOrDie(input).getElementType();
  auto sum_reduce_builder = [element_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        element_type, rb.getRegion(), rb.getOpBuilder());
  };
  mlir::MlirOp zero_val = MakeScalarConstant(builder, 0.0f, element_type);
  return mlir::stablehlo::Reduce(builder, {input}, /*init_value=*/{zero_val},
                                 sum_reduce_builder,
                                 /*dimensions=*/{dimension})[0];
}

mlir::MlirOp PromoteInputIfRequired(mlir::MlirBuilder& builder, mlir::MlirOp op,
                                    SdpaPromotionType promotion_type) {
  const auto type = GetTensorTypeOrDie(op).getElementType();
  if (promotion_type == SdpaPromotionType::kWholeModule &&
      (type.isF16() || type.isBF16())) {
    return mlir::stablehlo::ConvertElementType(
        op, builder.getOpBuilder().getF32Type());
  }
  return op;
}

mlir::MlirOp ConvertElementTypeIfNeeded(mlir::MlirOp op,
                                        mlir::Type target_type) {
  if (GetTensorTypeOrDie(op).getElementType() != target_type) {
    return mlir::stablehlo::ConvertElementType(op, target_type);
  }
  return op;
}

mlir::MlirOp ComputeAttentionOutput(mlir::MLIRContext* context,
                                    mlir::MlirOp softmax, mlir::MlirOp value_4d,
                                    mlir::MlirOp query_4d,
                                    int64_t head_count_ratio) {
  if (head_count_ratio > 1) {
    const auto query_shape = GetTensorTypeOrDie(query_4d).getShape();
    const int64_t batch_size = query_shape[0];
    const int64_t num_head_q = query_shape[1];
    const int64_t seq_len_q = query_shape[2];
    const int64_t head_dim = query_shape[3];
    const auto value_shape = GetTensorTypeOrDie(value_4d).getShape();
    const int64_t num_head_kv = value_shape[1];
    const int64_t seq_len_kv = value_shape[2];
    mlir::MlirOp softmax_5d = mlir::stablehlo::Reshape(
        softmax,
        {batch_size, num_head_kv, head_count_ratio, seq_len_q, seq_len_kv});
    const auto output_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        context, /*lhs_batching_dimensions=*/{0, 1},
        /*rhs_batching_dimensions=*/{0, 1}, /*lhs_contracting_dimensions=*/{4},
        /*rhs_contracting_dimensions=*/{2});
    mlir::MlirOp out_5d =
        mlir::stablehlo::DotGeneral(softmax_5d, value_4d, output_dot_dims);
    return mlir::stablehlo::Reshape(
        out_5d, {batch_size, num_head_q, seq_len_q, head_dim});
  }
  const auto output_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      context, /*lhs_batching_dimensions=*/{0, 1},
      /*rhs_batching_dimensions=*/{0, 1}, /*lhs_contracting_dimensions=*/{3},
      /*rhs_contracting_dimensions=*/{2});
  return mlir::stablehlo::DotGeneral(softmax, value_4d, output_dot_dims);
}

absl::StatusOr<MlirOpResults<2>> BuildScaledDotProductFusedAttentionShloOp(
    absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder,
    const Dimensions& out_dims, const Dimensions& lse_dims, bool is_causal,
    std::optional<double> scale, bool allow_half_precision_reduction_math) {
  mlir::MlirOp query_mlir = inputs[0];
  mlir::MlirOp key_mlir = inputs[1];
  mlir::MlirOp value_mlir = inputs[2];
  std::optional<mlir::MlirOp> mask_mlir;
  if (inputs.size() == 4) {
    mask_mlir = inputs[3];
  }
  mlir::MLIRContext* context = &builder.getContext();

  const SdpaPromotionType promotion_type = GetSdpaPromotionType(
      query_mlir, key_mlir, allow_half_precision_reduction_math);

  mlir::MlirOp query =
      PromoteInputIfRequired(builder, query_mlir, promotion_type);
  mlir::MlirOp key = PromoteInputIfRequired(builder, key_mlir, promotion_type);
  mlir::MlirOp value =
      PromoteInputIfRequired(builder, value_mlir, promotion_type);

  std::optional<mlir::MlirOp> mask;
  if (mask_mlir) {
    const auto query_acc_type = GetTensorTypeOrDie(query).getElementType();
    mask = ConvertElementTypeIfNeeded(*mask_mlir, query_acc_type);
  }

  TT_ASSIGN_OR_RETURN(auto prep_results,
                      PrepareAttentionLogits(builder, context, query, key,
                                             value, mask, is_causal, scale));
  auto [query_4d, key_4d, value_4d, shifted_attn_logits, scale_value,
        head_count_ratio] = prep_results;

  const auto original_element_type =
      GetTensorTypeOrDie(query_mlir).getElementType();
  const bool is_half_precision =
      original_element_type.isF16() || original_element_type.isBF16();

  mlir::MlirOp logits = shifted_attn_logits;
  if (promotion_type == SdpaPromotionType::kSoftmaxOnly && is_half_precision) {
    logits = mlir::stablehlo::ConvertElementType(
        shifted_attn_logits, builder.getOpBuilder().getF32Type());
  }

  // Softmax along the last dimension (Lk)
  mlir::MlirOp exp_val = mlir::stablehlo::Exp(logits);
  mlir::MlirOp sum_exp = SumReduce(builder, exp_val, /*dimension=*/3);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp sum_exp_broadcasted,
                      BroadcastIfNeeded(sum_exp, exp_val, {0, 1, 2}));
  mlir::MlirOp softmax = mlir::stablehlo::Div(exp_val, sum_exp_broadcasted);

  if (promotion_type == SdpaPromotionType::kSoftmaxOnly && is_half_precision) {
    softmax =
        mlir::stablehlo::ConvertElementType(softmax, original_element_type);
  }

  // Compute Attention Output: softmax @ value
  mlir::MlirOp out_4d = ComputeAttentionOutput(context, softmax, value_4d,
                                               query_4d, head_count_ratio);

  // Unflatten batch dimensions of output.
  mlir::MlirOp out_unflattened = unflatten_batch_dims(out_4d, out_dims);
  mlir::MlirOp out =
      ConvertElementTypeIfNeeded(out_unflattened, original_element_type);

  // The aten op requires sum_exp to be f32.
  mlir::MlirOp sum_exp_f32 = mlir::stablehlo::ConvertElementType(
      sum_exp, builder.getOpBuilder().getF32Type());
  mlir::MlirOp unflattened_sum_exp =
      unflatten_batch_dims(sum_exp_f32, lse_dims);

  return MlirOpResults<2>{out, unflattened_sum_exp};
}

absl::StatusOr<mlir::MlirOp> ComputeGradP(mlir::MlirBuilder& builder,
                                          mlir::MlirOp softmax,
                                          mlir::MlirOp grad_softmax,
                                          mlir::Type softmax_calc_type) {
  mlir::MlirOp grad_softmax_calc =
      ConvertElementTypeIfNeeded(grad_softmax, softmax_calc_type);
  mlir::MlirOp s_mul_ds = mlir::stablehlo::Mul(softmax, grad_softmax_calc);
  mlir::MlirOp rowsum_s_mul_ds = SumReduce(builder, s_mul_ds, /*dimension=*/3);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp rowsum_broadcasted,
                      BroadcastIfNeeded(rowsum_s_mul_ds, softmax,
                                        /*broadcast_dimensions=*/{0, 1, 2}));
  mlir::MlirOp shifted_ds =
      mlir::stablehlo::Subtract(grad_softmax_calc, rowsum_broadcasted);
  return mlir::stablehlo::Mul(softmax, shifted_ds);
}

struct BackwardGradients4D {
  mlir::MlirOp grad_query_4d;
  mlir::MlirOp grad_key_4d;
  mlir::MlirOp grad_value_4d;
};

absl::StatusOr<BackwardGradients4D> ComputeGqaBackwardGradients(
    mlir::MLIRContext* context, mlir::MlirBuilder& builder,
    mlir::MlirOp softmax, mlir::MlirOp grad_out_4d, mlir::MlirOp query_4d,
    mlir::MlirOp key_4d, mlir::MlirOp value_4d, mlir::Type element_type,
    mlir::Type softmax_calc_type, int64_t head_count_ratio, int64_t batch_size,
    int64_t num_head_q, int64_t num_head_kv, int64_t seq_len_q,
    int64_t seq_len_kv, int64_t head_dim) {
  mlir::MlirOp softmax_for_dv =
      ConvertElementTypeIfNeeded(softmax, element_type);
  mlir::MlirOp softmax_5d = mlir::stablehlo::Reshape(
      softmax_for_dv,
      {batch_size, num_head_kv, head_count_ratio, seq_len_q, seq_len_kv});
  mlir::MlirOp grad_out_5d = mlir::stablehlo::Reshape(
      grad_out_4d,
      {batch_size, num_head_kv, head_count_ratio, seq_len_q, head_dim});
  mlir::MlirOp query_5d = mlir::stablehlo::Reshape(
      query_4d,
      {batch_size, num_head_kv, head_count_ratio, seq_len_q, head_dim});

  // dV = softmax^T @ grad_out
  const auto grad_value_dot_dims =
      mlir::stablehlo::DotDimensionNumbersAttr::get(
          context, /*lhs_batching_dimensions=*/{0, 1},
          /*rhs_batching_dimensions=*/{0, 1},
          /*lhs_contracting_dimensions=*/{2, 3},
          /*rhs_contracting_dimensions=*/{2, 3});
  mlir::MlirOp grad_value_4d =
      mlir::stablehlo::DotGeneral(softmax_5d, grad_out_5d, grad_value_dot_dims);

  // dS = grad_out @ value^T
  const auto grad_softmax_dot_dims =
      mlir::stablehlo::DotDimensionNumbersAttr::get(
          context, /*lhs_batching_dimensions=*/{0, 1},
          /*rhs_batching_dimensions=*/{0, 1},
          /*lhs_contracting_dimensions=*/{4},
          /*rhs_contracting_dimensions=*/{3});
  mlir::MlirOp grad_softmax_5d =
      mlir::stablehlo::DotGeneral(grad_out_5d, value_4d, grad_softmax_dot_dims);
  mlir::MlirOp grad_softmax = mlir::stablehlo::Reshape(
      grad_softmax_5d, {batch_size, num_head_q, seq_len_q, seq_len_kv});

  // dP = S * (dS - rowsum(S * dS))
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp grad_p,
      ComputeGradP(builder, softmax, grad_softmax, softmax_calc_type));

  mlir::MlirOp grad_p_dot = ConvertElementTypeIfNeeded(grad_p, element_type);
  mlir::MlirOp grad_p_5d = mlir::stablehlo::Reshape(
      grad_p_dot,
      {batch_size, num_head_kv, head_count_ratio, seq_len_q, seq_len_kv});

  // dQ = (dP @ key) * scale
  const auto grad_query_dot_dims =
      mlir::stablehlo::DotDimensionNumbersAttr::get(
          context, /*lhs_batching_dimensions=*/{0, 1},
          /*rhs_batching_dimensions=*/{0, 1},
          /*lhs_contracting_dimensions=*/{4},
          /*rhs_contracting_dimensions=*/{2});
  mlir::MlirOp grad_query_5d =
      mlir::stablehlo::DotGeneral(grad_p_5d, key_4d, grad_query_dot_dims);
  mlir::MlirOp grad_query_4d = mlir::stablehlo::Reshape(
      grad_query_5d, {batch_size, num_head_q, seq_len_q, head_dim});

  // dK = (dP^T @ query) * scale
  const auto grad_key_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      context, /*lhs_batching_dimensions=*/{0, 1},
      /*rhs_batching_dimensions=*/{0, 1},
      /*lhs_contracting_dimensions=*/{2, 3},
      /*rhs_contracting_dimensions=*/{2, 3});
  mlir::MlirOp grad_key_4d =
      mlir::stablehlo::DotGeneral(grad_p_5d, query_5d, grad_key_dot_dims);

  return BackwardGradients4D{
      .grad_query_4d = grad_query_4d,
      .grad_key_4d = grad_key_4d,
      .grad_value_4d = grad_value_4d,
  };
}

absl::StatusOr<BackwardGradients4D> ComputeStandardBackwardGradients(
    mlir::MLIRContext* context, mlir::MlirBuilder& builder,
    mlir::MlirOp softmax, mlir::MlirOp grad_out_4d, mlir::MlirOp query_4d,
    mlir::MlirOp key_4d, mlir::MlirOp value_4d, mlir::Type element_type,
    mlir::Type softmax_calc_type) {
  // dV = softmax^T @ grad_out
  mlir::MlirOp softmax_for_dv =
      ConvertElementTypeIfNeeded(softmax, element_type);
  const auto dv_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      context, /*lhs_batching_dimensions=*/{0, 1},
      /*rhs_batching_dimensions=*/{0, 1},
      /*lhs_contracting_dimensions=*/{2},
      /*rhs_contracting_dimensions=*/{2});
  mlir::MlirOp grad_value_4d =
      mlir::stablehlo::DotGeneral(softmax_for_dv, grad_out_4d, dv_dot_dims);

  // dS = grad_out @ value^T
  const auto ds_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      context, /*lhs_batching_dimensions=*/{0, 1},
      /*rhs_batching_dimensions=*/{0, 1},
      /*lhs_contracting_dimensions=*/{3},
      /*rhs_contracting_dimensions=*/{3});
  mlir::MlirOp grad_softmax =
      mlir::stablehlo::DotGeneral(grad_out_4d, value_4d, ds_dot_dims);

  // dP = S * (dS - rowsum(S * dS))
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp grad_p,
      ComputeGradP(builder, softmax, grad_softmax, softmax_calc_type));

  // dQ = (dP @ key) * scale
  mlir::MlirOp grad_p_dot = ConvertElementTypeIfNeeded(grad_p, element_type);
  const auto dq_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      context, /*lhs_batching_dimensions=*/{0, 1},
      /*rhs_batching_dimensions=*/{0, 1},
      /*lhs_contracting_dimensions=*/{3},
      /*rhs_contracting_dimensions=*/{2});
  mlir::MlirOp grad_query_4d =
      mlir::stablehlo::DotGeneral(grad_p_dot, key_4d, dq_dot_dims);

  // dK = (dP^T @ query) * scale
  const auto dk_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      context, /*lhs_batching_dimensions=*/{0, 1},
      /*rhs_batching_dimensions=*/{0, 1},
      /*lhs_contracting_dimensions=*/{2},
      /*rhs_contracting_dimensions=*/{2});
  mlir::MlirOp grad_key_4d =
      mlir::stablehlo::DotGeneral(grad_p_dot, query_4d, dk_dot_dims);

  return BackwardGradients4D{
      .grad_query_4d = grad_query_4d,
      .grad_key_4d = grad_key_4d,
      .grad_value_4d = grad_value_4d,
  };
}

absl::StatusOr<std::array<mlir::MlirOp, 3>>
BuildScaledDotProductAttentionBackward(
    absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder, int rank,
    int batch_size, bool is_causal, std::optional<double> scale,
    int64_t num_head_q, int64_t num_head_kv, int64_t seq_len_q,
    int64_t seq_len_kv, int64_t head_dim,
    bool allow_half_precision_reduction_math) {
  mlir::MlirOp grad_out_mlir = inputs[0];
  mlir::MlirOp query_mlir = inputs[1];
  mlir::MlirOp key_mlir = inputs[2];
  mlir::MlirOp value_mlir = inputs[3];
  mlir::MlirOp sum_exp_mlir = inputs[4];
  std::optional<mlir::MlirOp> mask_mlir;
  if (inputs.size() == 6) {
    mask_mlir = inputs[5];
  }
  mlir::MLIRContext* context = &builder.getContext();

  const SdpaPromotionType promotion_type = GetSdpaPromotionType(
      query_mlir, key_mlir, allow_half_precision_reduction_math);

  mlir::MlirOp query =
      PromoteInputIfRequired(builder, query_mlir, promotion_type);
  mlir::MlirOp key = PromoteInputIfRequired(builder, key_mlir, promotion_type);
  mlir::MlirOp value =
      PromoteInputIfRequired(builder, value_mlir, promotion_type);

  const auto query_acc_type = GetTensorTypeOrDie(query).getElementType();
  mlir::MlirOp grad_out =
      ConvertElementTypeIfNeeded(grad_out_mlir, query_acc_type);
  mlir::MlirOp sum_exp = sum_exp_mlir;

  std::optional<mlir::MlirOp> mask;
  if (mask_mlir) {
    mask = ConvertElementTypeIfNeeded(*mask_mlir, query_acc_type);
  }

  TT_ASSIGN_OR_RETURN(auto prep_results,
                      PrepareAttentionLogits(builder, context, query, key,
                                             value, mask, is_causal, scale));
  auto [query_4d, key_4d, value_4d, shifted_attn_logits, scale_value,
        head_count_ratio] = prep_results;
  const auto element_type = GetTensorTypeOrDie(query).getElementType();
  const bool is_half_precision = element_type.isF16() || element_type.isBF16();

  const auto num_batch_dims = rank - 3;
  mlir::MlirOp grad_out_4d =
      flatten_batch_dims(grad_out, batch_size, num_batch_dims);
  mlir::MlirOp sum_exp_3d =
      flatten_batch_dims(sum_exp, batch_size, num_batch_dims);

  mlir::MlirOp logits = shifted_attn_logits;
  if (promotion_type == SdpaPromotionType::kSoftmaxOnly && is_half_precision) {
    logits = mlir::stablehlo::ConvertElementType(
        shifted_attn_logits, builder.getOpBuilder().getF32Type());
  }
  const auto softmax_calc_type = GetTensorTypeOrDie(logits).getElementType();

  // Softmax using previously computed sum_exp
  mlir::MlirOp sum_exp_converted =
      ConvertElementTypeIfNeeded(sum_exp_3d, softmax_calc_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp sum_exp_broadcasted,
                      BroadcastIfNeeded(sum_exp_converted, logits,
                                        /*broadcast_dimensions=*/{0, 1, 2}));

  // Softmax along the last dimension (Lk)
  mlir::MlirOp exp_val = mlir::stablehlo::Exp(logits);
  mlir::MlirOp softmax = mlir::stablehlo::Div(exp_val, sum_exp_broadcasted);

  // Compute backward gradients for Q, K, V
  BackwardGradients4D grads;
  if (head_count_ratio > 1) {
    TT_ASSIGN_OR_RETURN(
        grads,
        ComputeGqaBackwardGradients(
            context, builder, softmax, grad_out_4d, query_4d, key_4d, value_4d,
            element_type, softmax_calc_type, head_count_ratio, batch_size,
            num_head_q, num_head_kv, seq_len_q, seq_len_kv, head_dim));
  } else {
    TT_ASSIGN_OR_RETURN(
        grads, ComputeStandardBackwardGradients(
                   context, builder, softmax, grad_out_4d, query_4d, key_4d,
                   value_4d, element_type, softmax_calc_type));
  }

  mlir::MlirOp broadcasted_scale_for_query = mlir::stablehlo::Broadcast(
      scale_value, GetTensorTypeOrDie(grads.grad_query_4d).getShape());
  mlir::MlirOp scaled_grad_query_4d =
      mlir::stablehlo::Mul(grads.grad_query_4d, broadcasted_scale_for_query);

  mlir::MlirOp broadcasted_scale_for_key = mlir::stablehlo::Broadcast(
      scale_value, GetTensorTypeOrDie(grads.grad_key_4d).getShape());
  mlir::MlirOp scaled_grad_key_4d =
      mlir::stablehlo::Mul(grads.grad_key_4d, broadcasted_scale_for_key);

  mlir::MlirOp grad_query_unflattened =
      unflatten_batch_dims(scaled_grad_query_4d, query_mlir);
  mlir::MlirOp grad_key_unflattened =
      unflatten_batch_dims(scaled_grad_key_4d, key_mlir);
  mlir::MlirOp grad_value_unflattened =
      unflatten_batch_dims(grads.grad_value_4d, value_mlir);

  const auto original_element_type =
      GetTensorTypeOrDie(query_mlir).getElementType();
  mlir::MlirOp grad_query =
      ConvertElementTypeIfNeeded(grad_query_unflattened, original_element_type);
  mlir::MlirOp grad_key =
      ConvertElementTypeIfNeeded(grad_key_unflattened, original_element_type);
  mlir::MlirOp grad_value =
      ConvertElementTypeIfNeeded(grad_value_unflattened, original_element_type);

  return std::array<mlir::MlirOp, 3>{grad_query, grad_key, grad_value};
}

}  // namespace

absl::StatusOr<FusedAttentionResults> ScaledDotProductFusedAttentionShlo(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias, bool is_causal,
    std::optional<double> scale, bool allow_half_precision_reduction_math,
    OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(query.scalar_type()));

  Dimensions out_dims(query.sizes().begin(), query.sizes().end() - 1);
  out_dims.push_back(value.sizes().back());
  Dimensions lse_dims(query.sizes().begin(), query.sizes().end() - 1);
  auto op_builder =
      [out_dims, lse_dims, is_causal, scale,
       allow_half_precision_reduction_math](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<MlirOpResults<2>> {
    return BuildScaledDotProductFusedAttentionShloOp(
        inputs, builder, out_dims, lse_dims, is_causal, scale,
        allow_half_precision_reduction_math);
  };

  std::vector<at::Tensor> inputs = {query, key, value};
  if (attn_bias.has_value() && attn_bias->defined()) {
    inputs.push_back(*attn_bias);
  }

  TT_ASSIGN_OR_RETURN(auto results,
                      (DispatchOp<kDynamicSize, 2>(
                          std::move(op_builder), inputs,
                          {.out_dtypes = {out_dtype, mlir::ElementType::F32},
                           .out_dims_list = {out_dims, lse_dims},
                           .op_param_cache_keys = std::move(param_keys)})));

  // We return a view with dense strides that preserves the layout permutation
  // of the query. This ensures subsequent view operations (e.g., merging heads)
  // do not crash due to unexpected non-contiguity, while avoiding the memory
  // bloat and performance cost of matching exact CUDA strides (which may have
  // gaps if the query was sliced).
  // Note: This does not preserve the exact strides/offset for explicit
  // `as_strided` calls, which is considered an acceptable trade-off.
  // See: https://pytorch.org/docs/stable/generated/torch.as_strided.html
  if (query.sizes().back() == value.sizes().back()) {
    // We can only do this simple when the query and value have the same head
    // dimension.
    // TODO(willfroom): Check if we need to handle the general case.
    Strides dense_strides = DenseStrides(query.strides(), out_dims);
    TT_ASSIGN_OR_RETURN(at::Tensor view_out,
                        ContiguousToView(std::move(results[0]), dense_strides,
                                         /*target_storage_offset=*/0));
    return FusedAttentionResults{
        .output = std::move(view_out),
        .logsumexp = MakeTensor(std::move(results[1]))};
  } else {
    return FusedAttentionResults{
        .output = MakeTensor(std::move(results[0])),
        .logsumexp = MakeTensor(std::move(results[1]))};
  }
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
ScaledDotProductFusedAttentionShloBackward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& attn_bias,
    const at::Tensor& sum_exp, std::optional<double> scale, bool is_causal,
    bool allow_half_precision_reduction_math, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(query.scalar_type()));

  const int rank = query.ndimension();
  const int batch_size = get_batch_size(query.sizes());
  const int64_t num_head_q = query.size(rank - 3);
  const int64_t num_head_kv = key.size(rank - 3);
  const int64_t seq_len_q = query.size(rank - 2);
  const int64_t seq_len_kv = key.size(rank - 2);
  const int64_t head_dim = query.size(rank - 1);

  auto op_builder =
      [rank, batch_size, is_causal, scale, num_head_q, num_head_kv, seq_len_q,
       seq_len_kv, head_dim, allow_half_precision_reduction_math](
          absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder) {
        return BuildScaledDotProductAttentionBackward(
            inputs, builder, rank, batch_size, is_causal, scale, num_head_q,
            num_head_kv, seq_len_q, seq_len_kv, head_dim,
            allow_half_precision_reduction_math);
      };

  std::vector<at::Tensor> inputs = {grad_out, query, key, value, sum_exp};
  if (attn_bias.defined()) {
    inputs.push_back(attn_bias);
  }

  TT_ASSIGN_OR_RETURN(
      auto results,
      (DispatchOp<kDynamicSize, 3>(
          std::move(op_builder), inputs,
          {.out_dtypes = {out_dtype, out_dtype, out_dtype},
           .out_dims_list = {query.sizes(), key.sizes(), value.sizes()},
           .op_param_cache_keys = std::move(param_keys)})));

  TT_ASSIGN_OR_RETURN(
      at::Tensor strided_grad_query,
      ContiguousToView(results[0], query.strides(), query.storage_offset()));
  TT_ASSIGN_OR_RETURN(
      at::Tensor strided_grad_key,
      ContiguousToView(results[1], key.strides(), key.storage_offset()));
  TT_ASSIGN_OR_RETURN(
      at::Tensor strided_grad_value,
      ContiguousToView(results[2], value.strides(), value.storage_offset()));
  return std::make_tuple(strided_grad_query, strided_grad_key,
                         strided_grad_value);
}

}  // namespace torch_tpu
