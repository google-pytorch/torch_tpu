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

#include "torch_tpu/ops/scaled_dot_product_attention/scaled_dot_product_attention_shlo.h"

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
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/scaled_dot_product_attention/helpers.h"
#include "torch_tpu/ops/view_decomposition/contiguous_to_view.h"

namespace torch_tpu {

namespace {

// Returns strides for `target_sizes` that preserve the physical memory layout
// permutation of `template_strides` on `template_sizes`, but compacted to be
// dense (no gaps). This is useful for returning a tensor that is compatible
// with subsequent view operations that expect a specific permutation, while
// avoiding memory bloat and slice-by-slice writes (dynamic-update-slice).
// Note: `template_sizes` and `target_sizes` must only be permutations of each
// other.
Strides DenseStrides(absl::Span<const int64_t> template_sizes,
                     absl::Span<const int64_t> template_strides,
                     absl::Span<const int64_t> target_sizes) {
  int rank = template_sizes.size();
  Strides target_strides(rank, 0);

  struct Dim {
    int index;
    int64_t size;
    int64_t stride;
  };
  std::vector<Dim> dims(rank);
  for (int i = 0; i < rank; ++i) {
    dims[i] = {i, template_sizes[i], template_strides[i]};
  }
  absl::c_stable_sort(
      dims, [](const Dim& a, const Dim& b) { return a.stride > b.stride; });

  int64_t current_stride = 1;
  for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
    target_strides[it->index] = current_stride;
    current_stride *= target_sizes[it->index];
  }

  return target_strides;
}

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

absl::StatusOr<AttentionPrepResults> PrepareAttentionLogits(
    mlir::MlirBuilder& builder, mlir::MLIRContext* context, mlir::MlirOp query,
    mlir::MlirOp key, mlir::MlirOp value, std::optional<mlir::MlirOp> mask,
    bool is_causal, std::optional<double> scale) {
  auto query_tensor_type = GetTensorTypeOrDie(query);
  auto key_tensor_type = GetTensorTypeOrDie(key);
  int rank = query_tensor_type.getRank();
  int batch_size = get_batch_size(query_tensor_type.getShape());
  int64_t head_dim = query_tensor_type.getDimSize(rank - 1);
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
    auto replicate = [&](mlir::MlirOp input) {
      auto broadcast_type = mlir::RankedTensorType::get(
          {batch_size, num_head_kv, head_count_ratio, seq_len_kv, head_dim},
          GetTensorTypeOrDie(input).getElementType());
      mlir::MlirOp broadcasted_input =
          mlir::stablehlo::BroadcastInDim(broadcast_type, input, {0, 1, 3, 4});
      return mlir::stablehlo::Reshape(
          broadcasted_input, {batch_size, num_head_q, seq_len_kv, head_dim});
    };
    key_4d = replicate(key_4d);
    value_4d = replicate(value_4d);
  }

  // Scale Q before Dot product - matches pytorch attention implementation and
  // improves stability https://tinyurl.com/sudb9s96
  mlir::MlirOp scale_value =
      GetScaleDefaulted(builder, scale, head_dim, element_type);
  mlir::MlirOp broadcasted_scale_value = mlir::stablehlo::Broadcast(
      scale_value, GetTensorTypeOrDie(query_4d).getShape());
  mlir::MlirOp scaled_query_4d =
      mlir::stablehlo::Mul(query_4d, broadcasted_scale_value);

  auto attention_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      context, /*lhs_batching_dimensions=*/{0, 1},
      /*rhs_batching_dimensions=*/{0, 1}, /*lhs_contracting_dimensions=*/{3},
      /*rhs_contracting_dimensions=*/{3});

  mlir::MlirOp scaled_attn_logits =
      mlir::stablehlo::DotGeneral(scaled_query_4d, key_4d, attention_dot_dims);

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
  mlir::MlirOp max_logits_broadcasted = mlir::stablehlo::BroadcastInDim(
      masked_attn_logits.getType(), max_logits, {0, 1, 2});
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

}  // namespace

absl::StatusOr<std::pair<at::Tensor, at::Tensor>>
ScaledDotProductFusedAttentionShlo(const at::Tensor& query,
                                   const at::Tensor& key,
                                   const at::Tensor& value,
                                   const std::optional<at::Tensor>& attn_bias,
                                   bool is_causal, std::optional<double> scale,
                                   OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(query.scalar_type()));

  Dimensions out_dims(query.sizes().begin(), query.sizes().end() - 1);
  out_dims.push_back(value.sizes().back());
  Dimensions lse_dims(query.sizes().begin(), query.sizes().end() - 1);

  auto op_builder =
      [out_dims, lse_dims, is_causal, scale](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<MlirOpResults<2>> {
    mlir::MlirOp query_mlir = inputs[0];
    mlir::MlirOp key_mlir = inputs[1];
    mlir::MlirOp value_mlir = inputs[2];
    std::optional<mlir::MlirOp> mask_mlir;
    if (inputs.size() == 4) {
      mask_mlir = inputs[3];
    }
    mlir::MLIRContext* context = &builder.getContext();

    auto& ctx = at::globalContext();
    bool allow_half_precision_reduction_math =
        ctx.allowFP16BF16ReductionMathSDP();

    auto get_element_type = [](mlir::MlirOp op) {
      return GetTensorTypeOrDie(op).getElementType();
    };

    auto should_promote = [&](mlir::MlirOp op) {
      auto type = get_element_type(op);
      return !allow_half_precision_reduction_math &&
             (type.isF16() || type.isBF16());
    };

    auto promote_if_required = [&](mlir::MlirOp op) {
      return should_promote(op) ? mlir::stablehlo::ConvertElementType(
                                      op, builder.getOpBuilder().getF32Type())
                                : op;
    };

    mlir::MlirOp query = promote_if_required(query_mlir);

    mlir::MlirOp key = promote_if_required(key_mlir);

    mlir::MlirOp value = promote_if_required(value_mlir);

    std::optional<mlir::MlirOp> mask;
    if (mask_mlir) {
      auto query_acc_type = get_element_type(query);
      mask =
          (get_element_type(*mask_mlir) != query_acc_type)
              ? mlir::stablehlo::ConvertElementType(*mask_mlir, query_acc_type)
              : *mask_mlir;
    }

    TT_ASSIGN_OR_RETURN(auto prep_results,
                        PrepareAttentionLogits(builder, context, query, key,
                                               value, mask, is_causal, scale));
    auto [query_4d, key_4d, value_4d, shifted_attn_logits, scale_value,
          head_count_ratio] = prep_results;

    // Softmax along the last dimension (Lk)
    mlir::MlirOp exp_val = mlir::stablehlo::Exp(shifted_attn_logits);
    mlir::MlirOp sum_exp = SumReduce(builder, exp_val, /*dimension=*/3);
    mlir::MlirOp sum_exp_broadcasted =
        mlir::stablehlo::BroadcastInDim(exp_val.getType(), sum_exp, {0, 1, 2});
    mlir::MlirOp softmax = mlir::stablehlo::Div(exp_val, sum_exp_broadcasted);

    // Compute Attention Output: softmax @ value
    // softmax: [B, H, Lq, Lk]
    // value_4d: [B, H, Lk, D]
    // Output: [B, H, Lq, D]
    auto output_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        context, /*lhs_batching_dimensions=*/{0, 1},
        /*rhs_batching_dimensions=*/{0, 1}, /*lhs_contracting_dimensions=*/{3},
        /*rhs_contracting_dimensions=*/{2});
    mlir::MlirOp out_4d =
        mlir::stablehlo::DotGeneral(softmax, value_4d, output_dot_dims);

    // Unflatten batch dimensions of output.
    mlir::MlirOp out_unflattened = unflatten_batch_dims(out_4d, out_dims);

    // Convert back to original dtype
    auto original_element_type =
        GetTensorTypeOrDie(query_mlir).getElementType();
    mlir::MlirOp out = out_unflattened;
    if (GetTensorTypeOrDie(out_unflattened).getElementType() !=
        original_element_type) {
      out = mlir::stablehlo::ConvertElementType(out_unflattened,
                                                original_element_type);
    }

    // The aten op requires sum_exp to be f32.
    mlir::MlirOp sum_exp_f32 = mlir::stablehlo::ConvertElementType(
        sum_exp, builder.getOpBuilder().getF32Type());
    mlir::MlirOp unflattened_sum_exp =
        unflatten_batch_dims(sum_exp_f32, lse_dims);

    return {{out, unflattened_sum_exp}};
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
    Strides dense_strides =
        DenseStrides(query.sizes(), query.strides(), out_dims);
    TT_ASSIGN_OR_RETURN(at::Tensor view_out,
                        ContiguousToView(std::move(results[0]), dense_strides,
                                         /*target_storage_offset=*/0));
    return std::make_pair(std::move(view_out),
                          MakeTensor(std::move(results[1])));
  } else {
    return std::make_pair(MakeTensor(std::move(results[0])),
                          MakeTensor(std::move(results[1])));
  }
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
ScaledDotProductFusedAttentionShloBackward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& attn_bias,
    const at::Tensor& sum_exp, std::optional<double> scale, bool is_causal,
    OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(query.scalar_type()));

  int rank = query.ndimension();
  int batch_size = get_batch_size(query.sizes());
  int64_t num_head_kv = key.size(rank - 3);
  int64_t seq_len_kv = key.size(rank - 2);
  int64_t head_dim = query.size(rank - 1);

  auto op_builder = [rank, batch_size, is_causal, scale, num_head_kv,
                     seq_len_kv, head_dim](absl::Span<mlir::MlirOp> inputs,
                                           mlir::MlirBuilder& builder)
      -> absl::StatusOr<std::array<mlir::MlirOp, 3>> {
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

    auto& ctx = at::globalContext();
    bool allow_half_precision_reduction_math =
        ctx.allowFP16BF16ReductionMathSDP();

    auto get_element_type = [](mlir::MlirOp op) {
      return GetTensorTypeOrDie(op).getElementType();
    };

    auto should_promote = [&](mlir::MlirOp op) {
      auto type = get_element_type(op);
      return !allow_half_precision_reduction_math &&
             (type.isF16() || type.isBF16());
    };

    auto promote_if_required = [&](mlir::MlirOp op) {
      return should_promote(op) ? mlir::stablehlo::ConvertElementType(
                                      op, builder.getOpBuilder().getF32Type())
                                : op;
    };

    mlir::MlirOp query = promote_if_required(query_mlir);

    mlir::MlirOp key = promote_if_required(key_mlir);

    mlir::MlirOp value = promote_if_required(value_mlir);

    auto query_acc_type = get_element_type(query);
    mlir::MlirOp grad_out =
        (get_element_type(grad_out_mlir) != query_acc_type)
            ? mlir::stablehlo::ConvertElementType(grad_out_mlir, query_acc_type)
            : grad_out_mlir;

    mlir::MlirOp sum_exp = sum_exp_mlir;

    std::optional<mlir::MlirOp> mask;
    if (mask_mlir) {
      mask =
          (get_element_type(*mask_mlir) != query_acc_type)
              ? mlir::stablehlo::ConvertElementType(*mask_mlir, query_acc_type)
              : *mask_mlir;
    }

    TT_ASSIGN_OR_RETURN(auto prep_results,
                        PrepareAttentionLogits(builder, context, query, key,
                                               value, mask, is_causal, scale));
    auto [query_4d, key_4d, value_4d, shifted_attn_logits, scale_value,
          hedad_count_ratio] = prep_results;
    auto element_type = GetTensorTypeOrDie(query).getElementType();

    auto num_batch_dims = rank - 3;
    mlir::MlirOp grad_out_4d =
        flatten_batch_dims(grad_out, batch_size, num_batch_dims);
    mlir::MlirOp sum_exp_3d =
        flatten_batch_dims(sum_exp, batch_size, num_batch_dims);

    // Softmax using previously computed sum_exp
    // sum_exp_3d has shape [B, H, Lq].
    // shifted_attn_logits has shape [B, H, Lq, Lk].
    mlir::MlirOp sum_exp_converted =
        mlir::stablehlo::ConvertElementType(sum_exp_3d, element_type);
    mlir::MlirOp sum_exp_broadcasted = mlir::stablehlo::BroadcastInDim(
        shifted_attn_logits.getType(), sum_exp_converted, {0, 1, 2});

    // Softmax along the last dimension (Lk)
    mlir::MlirOp exp_val = mlir::stablehlo::Exp(shifted_attn_logits);
    mlir::MlirOp softmax = mlir::stablehlo::Div(exp_val, sum_exp_broadcasted);

    // BACKWARD PASS

    // dV = softmax^T @ grad_out
    auto dv_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        context, /*lhs_batching_dimensions=*/{0, 1},
        /*rhs_batching_dimensions=*/{0, 1}, /*lhs_contracting_dimensions=*/{2},
        /*rhs_contracting_dimensions=*/{2});
    mlir::MlirOp grad_value_4d =
        mlir::stablehlo::DotGeneral(softmax, grad_out_4d, dv_dot_dims);

    // dS = grad_out @ value^T
    auto ds_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        context, /*lhs_batching_dimensions=*/{0, 1},
        /*rhs_batching_dimensions=*/{0, 1}, /*lhs_contracting_dimensions=*/{3},
        /*rhs_contracting_dimensions=*/{3});
    mlir::MlirOp grad_softmax =
        mlir::stablehlo::DotGeneral(grad_out_4d, value_4d, ds_dot_dims);

    // dP = S * (dS - rowsum(S * dS))
    mlir::MlirOp s_mul_ds = mlir::stablehlo::Mul(softmax, grad_softmax);
    mlir::MlirOp rowsum_s_mul_ds =
        SumReduce(builder, s_mul_ds, /*dimension=*/3);
    mlir::MlirOp rowsum_broadcasted = mlir::stablehlo::BroadcastInDim(
        softmax.getType(), rowsum_s_mul_ds, {0, 1, 2});
    mlir::MlirOp shifted_ds =
        mlir::stablehlo::Subtract(grad_softmax, rowsum_broadcasted);
    mlir::MlirOp grad_p = mlir::stablehlo::Mul(softmax, shifted_ds);

    // dQ = (dP @ key) * scale
    auto dq_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        context, /*lhs_batching_dimensions=*/{0, 1},
        /*rhs_batching_dimensions=*/{0, 1}, /*lhs_contracting_dimensions=*/{3},
        /*rhs_contracting_dimensions=*/{2});
    mlir::MlirOp grad_query_4d =
        mlir::stablehlo::DotGeneral(grad_p, key_4d, dq_dot_dims);
    mlir::MlirOp broadcasted_scale_for_query = mlir::stablehlo::Broadcast(
        scale_value, GetTensorTypeOrDie(grad_query_4d).getShape());
    mlir::MlirOp scaled_grad_query_4d =
        mlir::stablehlo::Mul(grad_query_4d, broadcasted_scale_for_query);

    // dK = (dP^T @ query) * scale
    auto dk_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        context, /*lhs_batching_dimensions=*/{0, 1},
        /*rhs_batching_dimensions=*/{0, 1}, /*lhs_contracting_dimensions=*/{2},
        /*rhs_contracting_dimensions=*/{2});
    mlir::MlirOp grad_key_4d =
        mlir::stablehlo::DotGeneral(grad_p, query_4d, dk_dot_dims);
    mlir::MlirOp broadcasted_scale_for_key = mlir::stablehlo::Broadcast(
        scale_value, GetTensorTypeOrDie(grad_key_4d).getShape());
    mlir::MlirOp scaled_grad_key_4d =
        mlir::stablehlo::Mul(grad_key_4d, broadcasted_scale_for_key);

    if (prep_results.head_count_ratio > 1) {
      auto reduce_gqa = [&](mlir::MlirOp grad_input) {
        auto reshaped_type = mlir::RankedTensorType::get(
            {batch_size, num_head_kv, prep_results.head_count_ratio, seq_len_kv,
             head_dim},
            GetTensorTypeOrDie(grad_input).getElementType());
        mlir::MlirOp reshaped_grad =
            mlir::stablehlo::Reshape(grad_input, reshaped_type.getShape());
        mlir::MlirOp reduced_grad =
            SumReduce(builder, reshaped_grad, /*dimension=*/2);
        auto final_type = mlir::RankedTensorType::get(
            {batch_size, num_head_kv, seq_len_kv, head_dim},
            GetTensorTypeOrDie(reduced_grad).getElementType());
        return mlir::stablehlo::Reshape(reduced_grad, final_type.getShape());
      };
      scaled_grad_key_4d = reduce_gqa(scaled_grad_key_4d);
      grad_value_4d = reduce_gqa(grad_value_4d);
    }

    mlir::MlirOp grad_query_unflattened =
        unflatten_batch_dims(scaled_grad_query_4d, query_mlir);
    mlir::MlirOp grad_key_unflattened =
        unflatten_batch_dims(scaled_grad_key_4d, key_mlir);
    mlir::MlirOp grad_value_unflattened =
        unflatten_batch_dims(grad_value_4d, value_mlir);

    auto original_element_type =
        GetTensorTypeOrDie(query_mlir).getElementType();

    auto convert_if_required = [&](mlir::MlirOp grad_input) {
      if (GetTensorTypeOrDie(grad_input).getElementType() !=
          original_element_type) {
        return mlir::stablehlo::ConvertElementType(grad_input,
                                                   original_element_type);
      }
      return grad_input;
    };

    mlir::MlirOp grad_query = convert_if_required(grad_query_unflattened);
    mlir::MlirOp grad_key = convert_if_required(grad_key_unflattened);
    mlir::MlirOp grad_value = convert_if_required(grad_value_unflattened);

    return {{grad_query, grad_key, grad_value}};
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
