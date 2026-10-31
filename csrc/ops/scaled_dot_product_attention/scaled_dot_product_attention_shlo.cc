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
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/precision_context.h"
#include "csrc/ops/scaled_dot_product_attention/helpers.h"
#include "csrc/ops/view_decomposition/contiguous_to_view.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

namespace torch_tpu {

namespace {

struct AttentionPrepResults {
  mlir::MlirOp query_4d;
  mlir::MlirOp key_4d;
  mlir::MlirOp value_4d;
  mlir::MlirOp shifted_attn_logits;
  mlir::MlirOp scale_value;
  int64_t head_count_ratio;
  std::optional<mlir::MlirOp> is_row_all_masked;
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

// Returns a boolean mask with the shape of `reference_op` that is true where
// query position i must not attend to key position j (i.e. i < j).
absl::StatusOr<mlir::MlirOp> GetCausalMask(mlir::MlirOp reference_op) {
  mlir::Type i32_type = reference_op.getBuilder().getOpBuilder().getI32Type();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp iota_i,
                      Iota(reference_op, /*iota_dimension=*/-2, i32_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp iota_j,
                      Iota(reference_op, /*iota_dimension=*/-1, i32_type));
  return mlir::stablehlo::Compare(iota_i, iota_j,
                                  mlir::stablehlo::ComparisonDirection::LT);
}

enum class SdpaPromotionType {
  kNone,
  kSoftmaxOnly,
  kWholeModule,
};

// Determine exactly where type promotion should happen inside the attn module.
// For half-precision inputs (BF16/FP16), we promote only softmax and dP
// gradient scaling to FP32 for numerical stability (matching CUDA
// FlashAttention contract), keeping all matrix multiplications in native half
// precision on TPU MXUs. If allow_half_precision_reduction_math is true, no
// promotion is performed.
SdpaPromotionType GetSdpaPromotionType(
    mlir::MlirOp query, mlir::MlirOp /*key*/,
    bool allow_half_precision_reduction_math) {
  if (allow_half_precision_reduction_math) {
    return SdpaPromotionType::kNone;
  }

  const auto element_type = GetTensorTypeOrDie(query).getElementType();
  const bool is_half_precision = element_type.isF16() || element_type.isBF16();
  if (is_half_precision) {
    return SdpaPromotionType::kSoftmaxOnly;
  }

  return SdpaPromotionType::kNone;
}

mlir::MlirOp ConvertElementTypeIfNeeded(mlir::MlirOp op,
                                        mlir::Type target_type) {
  if (GetTensorTypeOrDie(op).getElementType() != target_type) {
    return mlir::stablehlo::ConvertElementType(op, target_type);
  }
  return op;
}

// Computes scaled query-key attention logits, scaling before dot for half
// precision and after dot for full precision.
absl::StatusOr<mlir::MlirOp> ComputeScaledAttentionLogits(
    mlir::MLIRContext* context, mlir::MlirOp query_4d, mlir::MlirOp key_4d,
    mlir::MlirOp scale_value, mlir::FloatType element_type, int64_t batch_size,
    int64_t num_head_q, int64_t num_head_kv, int64_t head_count_ratio,
    int64_t seq_len_q, int64_t seq_len_kv, int64_t head_dim,
    mlir::stablehlo::Precision precision) {
  // For half-precision (F16/BF16), scale Q before the dot product to prevent
  // overflow and pre-scaling rounding loss when DotGeneral outputs half
  // precision. For full-precision (F32/F64), scale the dot-product result to
  // match alpha * Q @ K^T.
  const bool scale_before_dot = element_type.isF16() || element_type.isBF16();
  mlir::MlirOp dot_query_4d = query_4d;
  if (scale_before_dot) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_scale,
                        BroadcastIfNeeded(scale_value, query_4d));
    dot_query_4d = mlir::stablehlo::Mul(query_4d, broadcasted_scale);
  }

  const auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
      context, {precision, precision});
  mlir::MlirOp attn_logits;
  if (head_count_ratio > 1) {
    mlir::MlirOp dot_query_5d = mlir::stablehlo::Reshape(
        dot_query_4d,
        {batch_size, num_head_kv, head_count_ratio, seq_len_q, head_dim});
    auto attention_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        context, /*lhs_batching_dimensions=*/{0, 1},
        /*rhs_batching_dimensions=*/{0, 1}, /*lhs_contracting_dimensions=*/{4},
        /*rhs_contracting_dimensions=*/{3});
    mlir::MlirOp logits_5d = mlir::stablehlo::DotGeneral(
        dot_query_5d, key_4d, attention_dot_dims, precision_config);
    attn_logits = mlir::stablehlo::Reshape(
        logits_5d, {batch_size, num_head_q, seq_len_q, seq_len_kv});
  } else {
    auto attention_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        context, /*lhs_batching_dimensions=*/{0, 1},
        /*rhs_batching_dimensions=*/{0, 1}, /*lhs_contracting_dimensions=*/{3},
        /*rhs_contracting_dimensions=*/{3});
    attn_logits = mlir::stablehlo::DotGeneral(
        dot_query_4d, key_4d, attention_dot_dims, precision_config);
  }
  if (!scale_before_dot) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_scale,
                        BroadcastIfNeeded(scale_value, attn_logits));
    return mlir::stablehlo::Mul(attn_logits, broadcasted_scale);
  }
  return attn_logits;
}

// Validates rank, static head dimensions, and dynamic batch constraints on
// attention inputs.
absl::Status ValidateAttentionInputs(mlir::MlirOp query, mlir::MlirOp key,
                                     mlir::MlirOp value) {
  const int64_t query_rank = GetTensorTypeOrDie(query).getRank();
  // The head count and head dim are used as concrete values below (e.g. the
  // GQA ratio and the default scale), so they must be static. Only batch and
  // sequence dimensions may be bounded dynamic.
  for (mlir::MlirOp input : {query, key, value}) {
    const mlir::RankedTensorType type = GetTensorTypeOrDie(input);
    const int64_t input_rank = type.getRank();
    if (input_rank < 3 || input_rank != query_rank) {
      return TT_ERROR(error::kInvalidArgument)
             << "expected query, key and value "
                "to have the same rank >= 3, got "
             << query_rank << ", " << GetTensorTypeOrDie(key).getRank()
             << " and " << GetTensorTypeOrDie(value).getRank();
    }
    if (type.isDynamicDim(input_rank - 1) ||
        type.isDynamicDim(input_rank - 3)) {
      return TT_ERROR(error::kInvalidArgument)
             << "expected a static head count (dim -3) and head dim (dim -1), "
                "got type "
             << mlir::debugString(type);
    }
    // The batch dims are flattened into one dim and restored afterwards. If a
    // batch dim is dynamic, the merged dim is dynamic, and restoring it is
    // ambiguous when more than one batch dim is not 1.
    const Dimensions input_shape = GetStaticShape(input);
    bool has_dynamic_batch_dim = false;
    int num_non_one_batch_dims = 0;
    for (int64_t i = 0; i < input_rank - 3; ++i) {
      has_dynamic_batch_dim |= type.isDynamicDim(i);
      num_non_one_batch_dims += input_shape[i] != 1;
    }
    if (has_dynamic_batch_dim && num_non_one_batch_dims > 1) {
      return TT_ERROR(error::kInvalidArgument)
             << "expected at most one batch dim "
                "not equal to 1 when a batch dim is dynamic, got shape ["
             << absl::StrJoin(input_shape, ", ") << "]";
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<AttentionPrepResults> PrepareAttentionLogits(
    mlir::MlirBuilder& builder, mlir::MLIRContext* context, mlir::MlirOp query,
    mlir::MlirOp key, mlir::MlirOp value, std::optional<mlir::MlirOp> mask,
    bool is_causal, std::optional<double> scale,
    mlir::stablehlo::Precision precision) {
  TT_RETURN_IF_ERROR(ValidateAttentionInputs(query, key, value));
  auto query_tensor_type = GetTensorTypeOrDie(query);
  // Static sizes, or upper bounds for bounded dynamic dimensions.
  const Dimensions query_shape = GetStaticShape(query);
  const Dimensions key_shape = GetStaticShape(key);
  int rank = query_shape.size();
  const int64_t batch_size = get_batch_size(query_shape);
  int64_t head_dim = query_shape[rank - 1];
  int64_t seq_len_q = query_shape[rank - 2];
  int64_t seq_len_kv = key_shape[rank - 2];
  int64_t num_head_q = query_shape[rank - 3];
  int64_t num_head_kv = key_shape[rank - 3];
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

  // The grouped GQA path reshapes the logits across both sequence dimensions,
  // but stablehlo.reshape supports at most one bounded dimension. For dynamic
  // shapes, replicate K/V across query heads and use the standard path.
  if (head_count_ratio > 1 && !(query_tensor_type.hasStaticShape() &&
                                GetTensorTypeOrDie(key).hasStaticShape() &&
                                GetTensorTypeOrDie(value).hasStaticShape())) {
    auto replicate = [&]  // REF_CAPTURE_OK=called locally
        (mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      mlir::stablehlo::Dimensions target_shape = GetDimensions(input);
      target_shape.insert(target_shape.begin() + 2,
                          mlir::stablehlo::DimensionInfo{head_count_ratio});
      TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted,
                          BroadcastIfNeeded(input, target_shape, {0, 1, 3, 4}));
      const Dimensions shape = GetStaticShape(broadcasted);
      return ReshapeToStaticDimensions(
          broadcasted, {shape[0], num_head_q, shape[3], shape[4]});
    };
    TT_ASSIGN_OR_RETURN(key_4d, replicate(key_4d));
    TT_ASSIGN_OR_RETURN(value_4d, replicate(value_4d));
    head_count_ratio = 1;
  }

  mlir::MlirOp scale_value =
      GetScaleDefaulted(builder, scale, head_dim, element_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp scaled_attn_logits,
                      ComputeScaledAttentionLogits(
                          context, query_4d, key_4d, scale_value, element_type,
                          batch_size, num_head_q, num_head_kv, head_count_ratio,
                          seq_len_q, seq_len_kv, head_dim, precision));

  mlir::Attribute min_attr =
      GetMinFiniteValueAttr(element_type, builder.getOpBuilder());
  mlir::MlirOp min_val = mlir::MlirOp(
      builder, mlir::stablehlo::ConstantOp::create(builder.getOpBuilder(),
                                                   builder.getLoc(), min_attr));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask_out_value,
                      BroadcastIfNeeded(min_val, scaled_attn_logits));

  std::optional<mlir::MlirOp> is_row_all_masked;
  mlir::MlirOp masked_attn_logits = scaled_attn_logits;
  if (is_causal) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp causal_mask,
                        GetCausalMask(scaled_attn_logits));
    masked_attn_logits = mlir::stablehlo::Select(causal_mask, mask_out_value,
                                                 scaled_attn_logits);
  } else if (mask) {
    // PyTorch converts boolean masks to float before passing them to the
    // attention op.
    if (IsBooleanType(GetTensorTypeOrDie(*mask))) {
      return TT_ERROR(error::kInvalidArgument)
             << "boolean mask is not supported";
    }

    mlir::MlirOp promoted_mask =
        ConvertElementTypeIfNeeded(*mask, element_type);
    int64_t mask_rank = GetTensorTypeOrDie(promoted_mask).getRank();
    mlir::MlirOp flat_mask =
        mask_rank > 4 ? flatten_batch_dims(promoted_mask, batch_size, rank - 3)
                      : promoted_mask;
    TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_mask,
                        BroadcastIfNeeded(flat_mask, scaled_attn_logits));

    mlir::MlirOp minus_inf =
        GetNegInf(builder, mlir::cast<mlir::FloatType>(element_type));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_minus_inf,
                        BroadcastIfNeeded(minus_inf, broadcasted_mask));
    mlir::MlirOp is_neg_inf =
        mlir::stablehlo::Compare(broadcasted_mask, broadcasted_minus_inf,
                                 mlir::stablehlo::ComparisonDirection::EQ);
    auto and_reduce_builder = [](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::AndOp>(
          rb.getOpBuilder().getI1Type(), rb.getRegion(), rb.getOpBuilder());
    };
    mlir::MlirOp true_val =
        MakeScalarConstant(builder, true, builder.getOpBuilder().getI1Type());
    is_row_all_masked = mlir::stablehlo::Reduce(
        builder, {is_neg_inf}, /*init_value=*/{true_val}, and_reduce_builder,
        /*dimensions=*/{3})[0];

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

  return AttentionPrepResults{
      .query_4d = query_4d,
      .key_4d = key_4d,
      .value_4d = value_4d,
      .shifted_attn_logits = shifted_attn_logits,
      .scale_value = scale_value,
      .head_count_ratio = head_count_ratio,
      .is_row_all_masked = is_row_all_masked,
  };
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

mlir::MlirOp ComputeAttentionOutput(mlir::MLIRContext* context,
                                    mlir::MlirOp softmax, mlir::MlirOp value_4d,
                                    mlir::MlirOp query_4d,
                                    int64_t head_count_ratio,
                                    mlir::stablehlo::Precision precision) {
  const auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
      context, {precision, precision});
  if (head_count_ratio > 1) {
    const auto query_shape = GetTensorTypeOrDie(query_4d).getShape();
    const int64_t batch_size = query_shape[0];
    const int64_t num_head_q = query_shape[1];
    const int64_t seq_len_q = query_shape[2];
    const auto value_shape = GetTensorTypeOrDie(value_4d).getShape();
    const int64_t num_head_kv = value_shape[1];
    const int64_t seq_len_kv = value_shape[2];
    const int64_t v_head_dim = value_shape[3];
    mlir::MlirOp softmax_5d = mlir::stablehlo::Reshape(
        softmax,
        {batch_size, num_head_kv, head_count_ratio, seq_len_q, seq_len_kv});
    const auto output_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        context, /*lhs_batching_dimensions=*/{0, 1},
        /*rhs_batching_dimensions=*/{0, 1}, /*lhs_contracting_dimensions=*/{4},
        /*rhs_contracting_dimensions=*/{2});
    mlir::MlirOp out_5d = mlir::stablehlo::DotGeneral(
        softmax_5d, value_4d, output_dot_dims, precision_config);
    return mlir::stablehlo::Reshape(
        out_5d, {batch_size, num_head_q, seq_len_q, v_head_dim});
  }
  const auto output_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      context, /*lhs_batching_dimensions=*/{0, 1},
      /*rhs_batching_dimensions=*/{0, 1}, /*lhs_contracting_dimensions=*/{3},
      /*rhs_contracting_dimensions=*/{2});
  return mlir::stablehlo::DotGeneral(softmax, value_4d, output_dot_dims,
                                     precision_config);
}

absl::StatusOr<MlirOpResults<2>> BuildScaledDotProductFusedAttentionShloOp(
    absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder, bool is_causal,
    std::optional<double> scale, bool allow_half_precision_reduction_math,
    mlir::stablehlo::Precision precision) {
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

  TT_ASSIGN_OR_RETURN(
      auto prep_results,
      PrepareAttentionLogits(builder, context, query, key, value, mask_mlir,
                             is_causal, scale, precision));
  auto [query_4d, key_4d, value_4d, shifted_attn_logits, scale_value,
        head_count_ratio, is_row_all_masked] = prep_results;

  const auto original_element_type =
      GetTensorTypeOrDie(query_mlir).getElementType();
  const bool is_half_precision =
      original_element_type.isF16() || original_element_type.isBF16();

  mlir::MlirOp logits = shifted_attn_logits;
  if (promotion_type == SdpaPromotionType::kSoftmaxOnly && is_half_precision) {
    logits = mlir::stablehlo::ConvertElementType(
        shifted_attn_logits, builder.getOpBuilder().getF32Type());
  }

  // Softmax along the last dimension (Lk).
  mlir::MlirOp exp_val = mlir::stablehlo::Exp(logits);
  mlir::MlirOp sum_exp = SumReduce(builder, exp_val, /*dimension=*/3);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp sum_exp_broadcasted,
                      BroadcastIfNeeded(sum_exp, exp_val, {0, 1, 2}));
  mlir::MlirOp softmax = mlir::stablehlo::Div(exp_val, sum_exp_broadcasted);

  if (is_row_all_masked.has_value()) {
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp is_masked_broadcasted,
        BroadcastIfNeeded(*is_row_all_masked, softmax, {0, 1, 2}));
    auto zero_softmax = MakeConstantLike(softmax, 0.0);
    softmax =
        mlir::stablehlo::Select(is_masked_broadcasted, zero_softmax, softmax);
    auto zero_sum_exp = MakeConstantLike(sum_exp, 0.0);
    sum_exp =
        mlir::stablehlo::Select(*is_row_all_masked, zero_sum_exp, sum_exp);
  }

  if (promotion_type == SdpaPromotionType::kSoftmaxOnly && is_half_precision) {
    softmax =
        mlir::stablehlo::ConvertElementType(softmax, original_element_type);
  }

  // Compute Attention Output: softmax @ value
  mlir::MlirOp out_4d = ComputeAttentionOutput(
      context, softmax, value_4d, query_4d, head_count_ratio, precision);

  // Restore the batch dims flattened in PrepareAttentionLogits: the query's
  // leading batch dims followed by the result's own non-batch dims. Sizes are
  // static, or upper bounds for bounded dynamic dims.
  const Dimensions query_shape = GetStaticShape(query_mlir);
  const Dimensions batch_dims =
      CopyIntVector(absl::Span<const int64_t>(query_shape)
                        .subspan(0, query_shape.size() - 3));
  auto restore_batch_dims = [&batch_dims](mlir::MlirOp flat_op) {
    Dimensions dims = batch_dims;
    const Dimensions flat_shape = GetStaticShape(flat_op);
    dims.insert(dims.end(), flat_shape.begin() + 1, flat_shape.end());
    return unflatten_batch_dims(flat_op, dims);
  };

  mlir::MlirOp out = ConvertElementTypeIfNeeded(restore_batch_dims(out_4d),
                                                original_element_type);

  // The aten op requires sum_exp to be f32.
  mlir::MlirOp sum_exp_f32 = mlir::stablehlo::ConvertElementType(
      sum_exp, builder.getOpBuilder().getF32Type());
  mlir::MlirOp unflattened_sum_exp = restore_batch_dims(sum_exp_f32);

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
    int64_t seq_len_kv, int64_t head_dim,
    mlir::stablehlo::Precision precision) {
  const auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
      context, {precision, precision});
  mlir::MlirOp softmax_for_dv =
      ConvertElementTypeIfNeeded(softmax, element_type);
  mlir::MlirOp softmax_5d = mlir::stablehlo::Reshape(
      softmax_for_dv,
      {batch_size, num_head_kv, head_count_ratio, seq_len_q, seq_len_kv});
  const int64_t v_head_dim = GetTensorTypeOrDie(grad_out_4d).getDimSize(3);
  mlir::MlirOp grad_out_5d = mlir::stablehlo::Reshape(
      grad_out_4d,
      {batch_size, num_head_kv, head_count_ratio, seq_len_q, v_head_dim});
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
  mlir::MlirOp grad_value_4d = mlir::stablehlo::DotGeneral(
      softmax_5d, grad_out_5d, grad_value_dot_dims, precision_config);

  // dS = grad_out @ value^T
  const auto grad_softmax_dot_dims =
      mlir::stablehlo::DotDimensionNumbersAttr::get(
          context, /*lhs_batching_dimensions=*/{0, 1},
          /*rhs_batching_dimensions=*/{0, 1},
          /*lhs_contracting_dimensions=*/{4},
          /*rhs_contracting_dimensions=*/{3});
  mlir::MlirOp grad_softmax_5d = mlir::stablehlo::DotGeneral(
      grad_out_5d, value_4d, grad_softmax_dot_dims, precision_config);
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
  mlir::MlirOp grad_query_5d = mlir::stablehlo::DotGeneral(
      grad_p_5d, key_4d, grad_query_dot_dims, precision_config);
  mlir::MlirOp grad_query_4d = mlir::stablehlo::Reshape(
      grad_query_5d, {batch_size, num_head_q, seq_len_q, head_dim});

  // dK = (dP^T @ query) * scale
  const auto grad_key_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      context, /*lhs_batching_dimensions=*/{0, 1},
      /*rhs_batching_dimensions=*/{0, 1},
      /*lhs_contracting_dimensions=*/{2, 3},
      /*rhs_contracting_dimensions=*/{2, 3});
  mlir::MlirOp grad_key_4d = mlir::stablehlo::DotGeneral(
      grad_p_5d, query_5d, grad_key_dot_dims, precision_config);

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
    mlir::Type softmax_calc_type, mlir::stablehlo::Precision precision) {
  const auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
      context, {precision, precision});
  // dV = softmax^T @ grad_out
  mlir::MlirOp softmax_for_dv =
      ConvertElementTypeIfNeeded(softmax, element_type);
  const auto dv_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      context, /*lhs_batching_dimensions=*/{0, 1},
      /*rhs_batching_dimensions=*/{0, 1},
      /*lhs_contracting_dimensions=*/{2},
      /*rhs_contracting_dimensions=*/{2});
  mlir::MlirOp grad_value_4d = mlir::stablehlo::DotGeneral(
      softmax_for_dv, grad_out_4d, dv_dot_dims, precision_config);

  // dS = grad_out @ value^T
  const auto ds_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      context, /*lhs_batching_dimensions=*/{0, 1},
      /*rhs_batching_dimensions=*/{0, 1},
      /*lhs_contracting_dimensions=*/{3},
      /*rhs_contracting_dimensions=*/{3});
  mlir::MlirOp grad_softmax = mlir::stablehlo::DotGeneral(
      grad_out_4d, value_4d, ds_dot_dims, precision_config);

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
  mlir::MlirOp grad_query_4d = mlir::stablehlo::DotGeneral(
      grad_p_dot, key_4d, dq_dot_dims, precision_config);

  // dK = (dP^T @ query) * scale
  const auto dk_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      context, /*lhs_batching_dimensions=*/{0, 1},
      /*rhs_batching_dimensions=*/{0, 1},
      /*lhs_contracting_dimensions=*/{2},
      /*rhs_contracting_dimensions=*/{2});
  mlir::MlirOp grad_key_4d = mlir::stablehlo::DotGeneral(
      grad_p_dot, query_4d, dk_dot_dims, precision_config);

  return BackwardGradients4D{
      .grad_query_4d = grad_query_4d,
      .grad_key_4d = grad_key_4d,
      .grad_value_4d = grad_value_4d,
  };
}

absl::StatusOr<std::array<mlir::MlirOp, 3>>
BuildScaledDotProductAttentionBackward(absl::Span<mlir::MlirOp> inputs,
                                       mlir::MlirBuilder& builder, int rank,
                                       int batch_size, bool is_causal,
                                       std::optional<double> scale,
                                       int64_t num_head_q, int64_t num_head_kv,
                                       int64_t seq_len_q, int64_t seq_len_kv,
                                       int64_t head_dim,
                                       bool allow_half_precision_reduction_math,
                                       mlir::stablehlo::Precision precision) {
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

  TT_ASSIGN_OR_RETURN(
      auto prep_results,
      PrepareAttentionLogits(builder, context, query, key, value, mask_mlir,
                             is_causal, scale, precision));
  auto [query_4d, key_4d, value_4d, shifted_attn_logits, scale_value,
        head_count_ratio, is_row_all_masked] = prep_results;
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

  // Softmax using previously computed sum_exp.
  mlir::MlirOp sum_exp_converted =
      ConvertElementTypeIfNeeded(sum_exp_3d, softmax_calc_type);
  auto zero_sum_exp_converted = MakeConstantLike(sum_exp_converted, 0.0);
  auto one_sum_exp_converted = MakeConstantLike(sum_exp_converted, 1.0);
  mlir::MlirOp is_zero_sum_exp =
      mlir::stablehlo::Compare(sum_exp_converted, zero_sum_exp_converted,
                               mlir::stablehlo::ComparisonDirection::EQ);
  mlir::MlirOp safe_sum_exp = mlir::stablehlo::Select(
      is_zero_sum_exp, one_sum_exp_converted, sum_exp_converted);

  mlir::MlirOp exp_val = mlir::stablehlo::Exp(logits);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp sum_exp_broadcasted,
                      BroadcastIfNeeded(safe_sum_exp, logits,
                                        /*broadcast_dimensions=*/{0, 1, 2}));
  mlir::MlirOp softmax = mlir::stablehlo::Div(exp_val, sum_exp_broadcasted);

  if (is_row_all_masked.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp is_masked_broadcasted,
                        BroadcastIfNeeded(*is_row_all_masked, softmax,
                                          /*broadcast_dimensions=*/{0, 1, 2}));
    auto zero_softmax = MakeConstantLike(softmax, 0.0);
    softmax =
        mlir::stablehlo::Select(is_masked_broadcasted, zero_softmax, softmax);
  }

  // Compute backward gradients for Q, K, V
  BackwardGradients4D grads;
  if (head_count_ratio > 1) {
    TT_ASSIGN_OR_RETURN(
        grads, ComputeGqaBackwardGradients(
                   context, builder, softmax, grad_out_4d, query_4d, key_4d,
                   value_4d, element_type, softmax_calc_type, head_count_ratio,
                   batch_size, num_head_q, num_head_kv, seq_len_q, seq_len_kv,
                   head_dim, precision));
  } else {
    TT_ASSIGN_OR_RETURN(
        grads, ComputeStandardBackwardGradients(
                   context, builder, softmax, grad_out_4d, query_4d, key_4d,
                   value_4d, element_type, softmax_calc_type, precision));
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

  const int64_t rank = query.dim();
  Dimensions out_dims = CopyIntVector(query.sizes().slice(0, rank - 1));
  out_dims.push_back(value.sizes().back());
  Dimensions lse_dims = CopyIntVector(query.sizes().slice(0, rank - 1));
  const mlir::stablehlo::Precision precision = GetAndAddPrecisionTo(param_keys);
  auto op_builder =
      [is_causal, scale, allow_half_precision_reduction_math, precision](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<MlirOpResults<2>> {
    return BuildScaledDotProductFusedAttentionShloOp(
        inputs, builder, is_causal, scale, allow_half_precision_reduction_math,
        precision);
  };

  std::vector<at::Tensor> inputs = {query, key, value};
  if (attn_bias.has_value() && attn_bias->defined() && attn_bias->numel() > 0) {
    inputs.push_back(*attn_bias);
  }

  TT_ASSIGN_OR_RETURN(auto results,
                      (DispatchOp<kDynamicSize, 2>(
                          std::move(op_builder), inputs,
                          {.out_dtypes = {out_dtype, mlir::ElementType::F32},
                           .out_dims_list = {out_dims, lse_dims},
                           .op_param_cache_keys = std::move(param_keys)})));

  // Construct the LSE tensor before calling ContiguousToView so that if
  // ContiguousToView materializes the stream, the LSE DeviceBufferRef is not
  // dropped.
  at::Tensor logsumexp = MakeTensor(results[1]);

  // Return the output with the strides of the meta function, so that views
  // traced by torch.compile (e.g. merging heads) are also valid in eager mode.
  TT_ASSIGN_OR_RETURN(at::Tensor view_out,
                      ContiguousToView(std::move(results[0]),
                                       MatchingLayoutStrides(query, out_dims),
                                       /*target_storage_offset=*/0));
  return FusedAttentionResults{.output = std::move(view_out),
                               .logsumexp = std::move(logsumexp)};
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
  const mlir::stablehlo::Precision precision = GetAndAddPrecisionTo(param_keys);

  auto op_builder =
      [rank, batch_size, is_causal, scale, num_head_q, num_head_kv, seq_len_q,
       seq_len_kv, head_dim, allow_half_precision_reduction_math,
       precision](absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder) {
        return BuildScaledDotProductAttentionBackward(
            inputs, builder, rank, batch_size, is_causal, scale, num_head_q,
            num_head_kv, seq_len_q, seq_len_kv, head_dim,
            allow_half_precision_reduction_math, precision);
      };

  std::vector<at::Tensor> inputs = {grad_out, query, key, value, sum_exp};
  if (attn_bias.defined() && attn_bias.numel() > 0) {
    inputs.push_back(attn_bias);
  }

  TT_ASSIGN_OR_RETURN(
      auto results,
      (DispatchOp<kDynamicSize, 3>(
          std::move(op_builder), inputs,
          {.out_dtypes = {out_dtype, out_dtype, out_dtype},
           .out_dims_list = {query.sizes(), key.sizes(), value.sizes()},
           .op_param_cache_keys = std::move(param_keys)})));

  // Artificially create a simulated DataPtr to prevent the result node from
  // being flushed during the first or second ContiguousToView if there is an
  // early materialization.
  results[0].device_buffer_list()->IncrementLiveDataPtrs(results[0].index());
  absl::Cleanup cleanup = [&results] {
    results[0].device_buffer_list()->DecrementLiveDataPtrs(results[0].index());
  };

  // Return the gradients with the strides of the meta function, which
  // allocates them with `torch.empty_like`.
  TT_ASSIGN_OR_RETURN(at::Tensor strided_grad_query,
                      ContiguousToView(results[0], EmptyLikeStrides(query),
                                       /*target_storage_offset=*/0));
  TT_ASSIGN_OR_RETURN(at::Tensor strided_grad_key,
                      ContiguousToView(results[1], EmptyLikeStrides(key),
                                       /*target_storage_offset=*/0));

  // It's okay if the materialization happens during the third
  // ContiguousToView; all other necessary tensors have already been created, so
  // nothing can be invalidated.
  std::move(cleanup).Invoke();

  TT_ASSIGN_OR_RETURN(at::Tensor strided_grad_value,
                      ContiguousToView(results[2], EmptyLikeStrides(value),
                                       /*target_storage_offset=*/0));
  return std::make_tuple(strided_grad_query, strided_grad_key,
                         strided_grad_value);
}

}  // namespace torch_tpu
