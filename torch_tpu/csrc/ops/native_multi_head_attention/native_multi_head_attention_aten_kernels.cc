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

#include "torch_tpu/csrc/ops/native_multi_head_attention/native_multi_head_attention_aten_kernels.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/aten_utils.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/precision_context.h"
#include "torch_tpu/csrc/ops/reductions/reductions.h"
#include "torch_tpu/csrc/ops/reductions/sum.h"
#include "torch_tpu/csrc/ops/softmax/softmax.h"

namespace torch_tpu {
namespace {
enum class AttnWeightsReduction { kNone, kAverage };

absl::Status CheckNativeMultiHeadAttentionInputs(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    int64_t embed_dim, int64_t num_head, const at::Tensor& qkv_weight,
    const at::Tensor& qkv_bias, const at::Tensor& proj_weight,
    const at::Tensor& proj_bias, const std::optional<at::Tensor>& mask) {
  TT_RET_CHECK(embed_dim > 0, error::kInvalidArgument)
      << "expected embed_dim to be positive, got " << embed_dim;

  TT_RET_CHECK(query.dim() == 3, error::kInvalidArgument)
      << "expected 3-D query, got " << query.dim() << "-D tensor";

  TT_RET_CHECK(query.size(2) == embed_dim, error::kInvalidArgument)
      << "expected embed_dim to match the last dim of query (" << query.size(2)
      << "), got " << embed_dim;

  TT_RET_CHECK(key.dim() == 3, error::kInvalidArgument)
      << "expected 3-D key, got " << key.dim() << "-D tensor";

  TT_RET_CHECK(value.dim() == 3, error::kInvalidArgument)
      << "expected 3-D value, got " << value.dim() << "-D tensor";

  TT_RET_CHECK(query.sizes() == key.sizes() && key.sizes() == value.sizes(),
               error::kInvalidArgument)
      << "expected query, key, and value shapes to match, got "
      << ToString(query.sizes()) << ", " << ToString(key.sizes()) << ", and "
      << ToString(value.sizes());

  TT_RET_CHECK(qkv_weight.dim() == 2, error::kInvalidArgument)
      << "expected 2-D qkv_weight, got " << qkv_weight.dim() << "-D tensor";

  TT_RET_CHECK(qkv_weight.size(0) == 3 * embed_dim, error::kInvalidArgument)
      << "expected qkv_weight first dim to be 3x embed_dim (" << (3 * embed_dim)
      << "), got " << qkv_weight.size(0);

  TT_RET_CHECK(qkv_weight.size(1) == embed_dim, error::kInvalidArgument)
      << "expected qkv_weight second dim to be embed_dim (" << embed_dim
      << "), got " << qkv_weight.size(1);

  TT_RET_CHECK(qkv_bias.dim() == 1, error::kInvalidArgument)
      << "expected 1-D qkv_bias, got " << qkv_bias.dim() << "-D tensor";

  TT_RET_CHECK(qkv_bias.size(0) == 3 * embed_dim, error::kInvalidArgument)
      << "expected qkv_bias first dim to be 3x embed_dim (" << (3 * embed_dim)
      << "), got " << qkv_bias.size(0);

  TT_RET_CHECK(num_head > 0, error::kInvalidArgument)
      << "expected num_head to be positive, got " << num_head;

  TT_RET_CHECK(embed_dim % num_head == 0, error::kInvalidArgument)
      << "expected embed_dim to be divisible by num_head, got " << embed_dim
      << ", which is not divisible by " << num_head;

  TT_RET_CHECK(proj_weight.dim() == 2, error::kInvalidArgument)
      << "expected 2-D proj_weight, got " << proj_weight.dim() << "-D tensor";

  TT_RET_CHECK(proj_weight.size(0) == embed_dim, error::kInvalidArgument)
      << "expected proj_weight first dim to be embed_dim (" << embed_dim
      << "), got " << proj_weight.size(0);

  TT_RET_CHECK(proj_weight.size(1) == embed_dim, error::kInvalidArgument)
      << "expected proj_weight second dim to be embed_dim (" << embed_dim
      << "), got " << proj_weight.size(1);

  TT_RET_CHECK(proj_bias.dim() == 1, error::kInvalidArgument)
      << "expected 1-D proj_bias, got " << proj_bias.dim() << "-D tensor";

  TT_RET_CHECK(proj_bias.size(0) == embed_dim, error::kInvalidArgument)
      << "expected proj_bias first dim to be embed_dim (" << embed_dim
      << "), got " << proj_bias.size(0);

  const auto query_dtype = query.scalar_type();
  TT_RET_CHECK(key.scalar_type() == query_dtype &&
                   value.scalar_type() == query_dtype &&
                   qkv_weight.scalar_type() == query_dtype &&
                   qkv_bias.scalar_type() == query_dtype &&
                   proj_weight.scalar_type() == query_dtype &&
                   proj_bias.scalar_type() == query_dtype,
               error::kInvalidArgument)
      << "expected query, key, value, qkv_weight, qkv_bias, proj_weight, and "
         "proj_bias to have matching dtypes, got "
      << ToString(query_dtype) << ", " << ToString(key.scalar_type()) << ", "
      << ToString(value.scalar_type()) << ", "
      << ToString(qkv_weight.scalar_type()) << ", "
      << ToString(qkv_bias.scalar_type()) << ", "
      << ToString(proj_weight.scalar_type()) << ", "
      << ToString(proj_bias.scalar_type());

  if (mask.has_value() && mask->defined()) {
    const int64_t mask_rank = mask->dim();
    TT_RET_CHECK(mask_rank == 2 || mask_rank == 4, error::kInvalidArgument)
        << "expected 2-D or 4-D mask, got " << mask_rank << "-D tensor";
    if (mask_rank == 4) {
      TT_RET_CHECK(
          mask->size(0) == query.size(0) && mask->size(1) == num_head &&
              mask->size(2) == query.size(1) && mask->size(3) == key.size(1),
          error::kInvalidArgument)
          << "expected 4-D mask shape to be [" << query.size(0) << ", "
          << num_head << ", " << query.size(1) << ", " << key.size(1)
          << "], got " << mask->sizes();
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    absl::Span<const int64_t> lhs_contracting,
    absl::Span<const int64_t> rhs_contracting,
    absl::Span<const int64_t> lhs_batching = {},
    absl::Span<const int64_t> rhs_batching = {},
    mlir::stablehlo::Precision precision =
        mlir::stablehlo::Precision::DEFAULT) {  // EXPLICIT_PRECISION_OK=default
  mlir::MlirBuilder& builder = lhs.getBuilder();
  mlir::MLIRContext& ctx = builder.getContext();

  // Configure contracting and batching dimensions for StableHLO DotGeneral.
  auto dot_dimension_numbers = mlir::stablehlo::DotDimensionNumbersAttr::get(
      &ctx, lhs_batching, rhs_batching, lhs_contracting, rhs_contracting);

  auto precision_config_attr =
      mlir::stablehlo::PrecisionConfigAttr::get(&ctx, {precision, precision});

  const auto lhs_type = GetTensorTypeOrDie(lhs);
  const auto rhs_type = GetTensorTypeOrDie(rhs);
  const auto lhs_shape = lhs_type.getShape();
  const auto rhs_shape = rhs_type.getShape();

  // Compute output dimensions: batch dims followed by non-contracting dims.
  Dimensions result_shape;
  for (int64_t dim : lhs_batching) {
    result_shape.push_back(lhs_shape[dim]);
  }
  for (int64_t i = 0; i < lhs_type.getRank(); ++i) {
    if (absl::c_find(lhs_batching, i) != lhs_batching.end() ||
        absl::c_find(lhs_contracting, i) != lhs_contracting.end()) {
      continue;
    }
    result_shape.push_back(lhs_shape[i]);
  }
  for (int64_t i = 0; i < rhs_type.getRank(); ++i) {
    if (absl::c_find(rhs_batching, i) != rhs_batching.end() ||
        absl::c_find(rhs_contracting, i) != rhs_contracting.end()) {
      continue;
    }
    result_shape.push_back(rhs_shape[i]);
  }

  // Infer computation dtype (e.g. promote f16/bf16 to f32 during GEMM).
  TT_ASSIGN_OR_RETURN(mlir::ElementType lhs_elem_type,
                      ConvertTo<mlir::ElementType>(lhs_type.getElementType()));
  TT_ASSIGN_OR_RETURN(auto comp_dtype, InferComputationDtype(lhs_elem_type));

  auto result_type = mlir::RankedTensorType::get(
      result_shape, mlir::getElementType(ctx, comp_dtype));

  return mlir::stablehlo::DotGeneral(
      result_type, lhs, rhs, dot_dimension_numbers, precision_config_attr);
}

absl::StatusOr<mlir::MlirOp> BuildLinear(mlir::MlirOp input,
                                         mlir::MlirOp weight, mlir::MlirOp bias,
                                         mlir::stablehlo::Precision precision) {
  // Execute GEMM: input [B, T, D_in] @ weight [D_out, D_in]^T -> [B, T, D_out].
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp mm,
      BuildDotGeneral(input, weight, /*lhs_contracting=*/{2},
                      /*rhs_contracting=*/{1}, /*lhs_batching=*/{},
                      /*rhs_batching=*/{}, precision));

  auto input_type = GetTensorTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(
      mlir::ElementType input_elem_type,
      ConvertTo<mlir::ElementType>(input_type.getElementType()));
  TT_ASSIGN_OR_RETURN(mm, CastIfNeeded(mm, input_elem_type));

  // Reshape 1-D bias to [1, 1, D_out] and broadcast add to GEMM output.
  auto bias_type = GetTensorTypeOrDie(bias);
  const int64_t d_out = bias_type.getShape()[0];
  TT_ASSIGN_OR_RETURN(mlir::MlirOp bias_reshaped,
                      ReshapeFromStaticDimensions(bias, Dimensions{d_out},
                                                  Dimensions{1, 1, d_out}));

  TT_ASSIGN_OR_RETURN(bias_reshaped,
                      CastIfNeeded(bias_reshaped, input_elem_type));

  TT_ASSIGN_OR_RETURN(bias_reshaped, BroadcastIfNeeded(bias_reshaped, mm));

  return mlir::stablehlo::Add(mm, bias_reshaped);
}

// Common function to compute core MHA intermediate tensors
// (out, attn_weights_full).
struct MhaCoreTensors {
  mlir::MlirOp out;
  mlir::MlirOp attn_weights_full;
};

absl::StatusOr<MhaCoreTensors> BuildMhaCore(
    mlir::MlirOp query, mlir::MlirOp key, mlir::MlirOp value,
    mlir::MlirOp qkv_weight, mlir::MlirOp qkv_bias, mlir::MlirOp proj_weight,
    mlir::MlirOp proj_bias, std::optional<mlir::MlirOp> mask, int64_t embed_dim,
    int64_t num_head, mlir::stablehlo::Precision precision,
    std::optional<int64_t> mask_type_param = std::nullopt) {
  mlir::MlirBuilder& builder = query.getBuilder();
  auto query_type = GetTensorTypeOrDie(query);
  TT_ASSIGN_OR_RETURN(
      mlir::ElementType query_elem_type,
      ConvertTo<mlir::ElementType>(query_type.getElementType()));
  auto shape = query_type.getShape();
  const int64_t batch_size = shape[0];
  const int64_t seq_len = shape[1];
  const int64_t dim_per_head = embed_dim / num_head;

  // 1. Slice qkv_weight into W_q, W_k, W_v [embed_dim, embed_dim]
  const mlir::MlirOp w_q = mlir::stablehlo::Slice(
      qkv_weight, Dimensions{0, 0}, Dimensions{embed_dim, embed_dim},
      Dimensions{1, 1});
  const mlir::MlirOp w_k = mlir::stablehlo::Slice(
      qkv_weight, Dimensions{embed_dim, 0},
      Dimensions{2 * embed_dim, embed_dim}, Dimensions{1, 1});
  const mlir::MlirOp w_v = mlir::stablehlo::Slice(
      qkv_weight, Dimensions{2 * embed_dim, 0},
      Dimensions{3 * embed_dim, embed_dim}, Dimensions{1, 1});

  // qkv_bias shape: [3 * embed_dim]
  const mlir::MlirOp b_q = mlir::stablehlo::Slice(
      qkv_bias, Dimensions{0}, Dimensions{embed_dim}, Dimensions{1});
  const mlir::MlirOp b_k =
      mlir::stablehlo::Slice(qkv_bias, Dimensions{embed_dim},
                             Dimensions{2 * embed_dim}, Dimensions{1});
  const mlir::MlirOp b_v =
      mlir::stablehlo::Slice(qkv_bias, Dimensions{2 * embed_dim},
                             Dimensions{3 * embed_dim}, Dimensions{1});

  // 2. Linear projections: Q_proj, K_proj, V_proj [B, T, D]
  TT_ASSIGN_OR_RETURN(mlir::MlirOp q_proj,
                      BuildLinear(query, w_q, b_q, precision));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp k_proj,
                      BuildLinear(key, w_k, b_k, precision));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp v_proj,
                      BuildLinear(value, w_v, b_v, precision));

  // 3. Scale Q by 1.0 / sqrt(dim_per_head)
  const double scale_factor =
      1.0 / std::sqrt(static_cast<double>(dim_per_head));
  mlir::MlirOp scale_const = MakeConstantLike(q_proj, scale_factor);
  const mlir::MlirOp q_scaled = mlir::stablehlo::Mul(q_proj, scale_const);

  // 4. Reshape and Transpose to [B, H, T, dim_per_head]
  const Dimensions target_4d = {batch_size, seq_len, num_head, dim_per_head};
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp q_4d,
      ReshapeFromStaticDimensions(
          q_scaled, Dimensions{batch_size, seq_len, embed_dim}, target_4d));
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp k_4d,
      ReshapeFromStaticDimensions(
          k_proj, Dimensions{batch_size, seq_len, embed_dim}, target_4d));
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp v_4d,
      ReshapeFromStaticDimensions(
          v_proj, Dimensions{batch_size, seq_len, embed_dim}, target_4d));

  const mlir::MlirOp q_mha = mlir::stablehlo::Transpose(q_4d, {0, 2, 1, 3});
  const mlir::MlirOp k_mha = mlir::stablehlo::Transpose(k_4d, {0, 2, 1, 3});
  const mlir::MlirOp v_mha = mlir::stablehlo::Transpose(v_4d, {0, 2, 1, 3});

  // 5. QK^T Batch MatMul: Q_mha [B, H, T, d] @ K_mha [B, H, T, d]^T -> qkt
  // [B, H, T, T]
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp qkt,
      BuildDotGeneral(q_mha, k_mha, /*lhs_contracting=*/{3},
                      /*rhs_contracting=*/{3}, /*lhs_batching=*/{0, 1},
                      /*rhs_batching=*/{0, 1}, precision));

  TT_ASSIGN_OR_RETURN(qkt, CastIfNeeded(qkt, query_elem_type));

  // 6. Apply optional mask
  if (mask.has_value()) {
    mlir::MlirOp mask_op = *mask;
    const auto tensor_mask_type = GetTensorTypeOrDie(mask_op);
    const auto mask_shape = tensor_mask_type.getShape();
    const int64_t mask_rank = tensor_mask_type.getRank();

    mlir::MlirOp mask_4d = mask_op;
    if (mask_rank == 2) {
      if (mask_type_param.has_value() && *mask_type_param == 1) {
        // mask_type == 1: Key Padding Mask [B, L] -> [B, 1, 1, L]
        TT_ASSIGN_OR_RETURN(
            mask_4d, ReshapeFromStaticDimensions(
                         mask_op, Dimensions{mask_shape[0], mask_shape[1]},
                         Dimensions{mask_shape[0], 1, 1, mask_shape[1]}));
      } else {
        // mask_type == 0 (or default): Attention Mask [L, L] -> [1, 1, L, L]
        TT_ASSIGN_OR_RETURN(
            mask_4d, ReshapeFromStaticDimensions(
                         mask_op, Dimensions{mask_shape[0], mask_shape[1]},
                         Dimensions{1, 1, mask_shape[0], mask_shape[1]}));
      }
    }

    mlir::MlirOp mask_bool = mask_4d;
    if (!tensor_mask_type.getElementType().isInteger(1)) {
      TT_ASSIGN_OR_RETURN(
          mlir::ElementType mask_elem_type,
          ConvertTo<mlir::ElementType>(tensor_mask_type.getElementType()));
      mlir::MlirOp zero_mask_elem =
          MakeScalarConstant(builder, 0.0, mask_elem_type);
      TT_ASSIGN_OR_RETURN(zero_mask_elem,
                          BroadcastIfNeeded(zero_mask_elem, mask_4d));
      mask_bool = mlir::stablehlo::Compare(
          mask_4d, zero_mask_elem, mlir::stablehlo::ComparisonDirection::NE);
    }

    mlir::MlirOp neg_inf = MakeScalarConstant(
        builder, -std::numeric_limits<float>::infinity(), query_elem_type);
    mlir::MlirOp zero = MakeScalarConstant(builder, 0.0, query_elem_type);
    TT_ASSIGN_OR_RETURN(neg_inf, BroadcastIfNeeded(neg_inf, mask_bool));
    TT_ASSIGN_OR_RETURN(zero, BroadcastIfNeeded(zero, mask_bool));

    mlir::MlirOp mask_additive =
        mlir::stablehlo::Select(mask_bool, neg_inf, zero);
    TT_ASSIGN_OR_RETURN(mask_additive, BroadcastIfNeeded(mask_additive, qkt));
    qkt = mlir::stablehlo::Add(qkt, mask_additive);
  }

  // 7. Softmax along axis 3
  TT_ASSIGN_OR_RETURN(mlir::MlirOp attn_weights_full,
                      BuildSoftmaxShlo(qkt, /*dim=*/3, SoftmaxMode::kSoftmax));

  // 8. Context: attn_weights_full [B, H, T, T] @ V_mha [B, H, T, d] ->
  // attn_ctx [B, H, T, d]
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp attn_ctx,
      BuildDotGeneral(attn_weights_full, v_mha, /*lhs_contracting=*/{3},
                      /*rhs_contracting=*/{2}, /*lhs_batching=*/{0, 1},
                      /*rhs_batching=*/{0, 1}, precision));

  TT_ASSIGN_OR_RETURN(attn_ctx, CastIfNeeded(attn_ctx, query_elem_type));

  // 9. Output projection: transpose to [B, T, H, d], reshape to [B, T, D],
  // linear proj
  const mlir::MlirOp attn_ctx_trans =
      mlir::stablehlo::Transpose(attn_ctx, {0, 2, 1, 3});
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp attn_ctx_reshaped,
      ReshapeFromStaticDimensions(
          attn_ctx_trans,
          Dimensions{batch_size, seq_len, num_head, dim_per_head},
          Dimensions{batch_size, seq_len, embed_dim}));

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp out,
      BuildLinear(attn_ctx_reshaped, proj_weight, proj_bias, precision));

  return MhaCoreTensors{.out = out, .attn_weights_full = attn_weights_full};
}

absl::StatusOr<std::array<mlir::MlirOp, 2>>
BuildNativeMultiHeadAttentionWithWeightsShlo(
    mlir::MlirOp query, mlir::MlirOp key, mlir::MlirOp value,
    mlir::MlirOp qkv_weight, mlir::MlirOp qkv_bias, mlir::MlirOp proj_weight,
    mlir::MlirOp proj_bias, std::optional<mlir::MlirOp> mask, int64_t embed_dim,
    int64_t num_head, AttnWeightsReduction weights_reduction,
    mlir::stablehlo::Precision precision,
    std::optional<int64_t> mask_type_param = std::nullopt) {
  TT_ASSIGN_OR_RETURN(
      auto core, BuildMhaCore(query, key, value, qkv_weight, qkv_bias,
                              proj_weight, proj_bias, mask, embed_dim, num_head,
                              precision, mask_type_param));

  // Average attention weights across heads if requested; otherwise return full
  // 4-D weights.
  mlir::MlirOp attn_weights;
  if (weights_reduction == AttnWeightsReduction::kAverage) {
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp sum_weights,
        BuildSumShlo(core.attn_weights_full, {1}, ReductionMode::kDropDims));
    mlir::MlirOp scale_op =
        MakeConstantLike(sum_weights, 1.0 / static_cast<double>(num_head));
    attn_weights = mlir::stablehlo::Mul(sum_weights, scale_op);
  } else {
    attn_weights = core.attn_weights_full;
  }

  return std::array<mlir::MlirOp, 2>{core.out, attn_weights};
}

absl::StatusOr<mlir::MlirOp> BuildNativeMultiHeadAttentionNoWeightsShlo(
    mlir::MlirOp query, mlir::MlirOp key, mlir::MlirOp value,
    mlir::MlirOp qkv_weight, mlir::MlirOp qkv_bias, mlir::MlirOp proj_weight,
    mlir::MlirOp proj_bias, std::optional<mlir::MlirOp> mask, int64_t embed_dim,
    int64_t num_head, mlir::stablehlo::Precision precision,
    std::optional<int64_t> mask_type_param = std::nullopt) {
  TT_ASSIGN_OR_RETURN(
      auto core, BuildMhaCore(query, key, value, qkv_weight, qkv_bias,
                              proj_weight, proj_bias, mask, embed_dim, num_head,
                              precision, mask_type_param));
  return core.out;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> AtenNativeMultiHeadAttention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    int64_t embed_dim, int64_t num_head, const at::Tensor& qkv_weight,
    const at::Tensor& qkv_bias, const at::Tensor& proj_weight,
    const at::Tensor& proj_bias, const std::optional<at::Tensor>& mask,
    bool need_weights, bool average_attn_weights,
    std::optional<int64_t> mask_type) {
  TT_KERNEL(
      OpName::kNativeMultiHeadAttention, param_keys,
      (query, key, value, embed_dim, num_head, qkv_weight, qkv_bias,
       proj_weight, proj_bias, mask, need_weights, average_attn_weights,
       mask_type),
      {
        // Validate input tensor shapes, ranks, dimensions, and data types.
        TT_THROW_IF_ERROR(CheckNativeMultiHeadAttentionInputs(
            query, key, value, embed_dim, num_head, qkv_weight, qkv_bias,
            proj_weight, proj_bias, mask));

        TT_ASSIGN_OR_THROW(mlir::ElementType query_dtype_mlir,
                           ConvertTo<mlir::ElementType>(query.scalar_type()));

        const Dimensions out_dims = {query.size(0), query.size(1), embed_dim};
        const auto current_precision = GetAndAddPrecisionTo(param_keys);

        if (need_weights) {
          // Build and dispatch multi-head attention graph returning both output
          // and weights.
          Dimensions attn_weights_dims;
          if (average_attn_weights) {
            attn_weights_dims = {query.size(0), query.size(1), query.size(1)};
          } else {
            attn_weights_dims = {query.size(0), num_head, query.size(1),
                                 query.size(1)};
          }

          const std::array<absl::Span<const int64_t>, 2> out_dims_list = {
              out_dims, attn_weights_dims};
          const std::array<mlir::ElementType, 2> out_dtypes = {
              query_dtype_mlir, query_dtype_mlir};

          const AttnWeightsReduction weights_reduction =
              average_attn_weights ? AttnWeightsReduction::kAverage
                                   : AttnWeightsReduction::kNone;

          if (mask.has_value() && mask->defined()) {
            const auto op_builder =
                [embed_dim, num_head, weights_reduction, current_precision,
                 mask_type](FixedSizeSpan<mlir::MlirOp, 8> inputs)
                -> absl::StatusOr<std::array<mlir::MlirOp, 2>> {
              auto& [q_op, k_op, v_op, qkv_w_op, qkv_b_op, proj_w_op, proj_b_op,
                     mask_op] = inputs;
              return BuildNativeMultiHeadAttentionWithWeightsShlo(
                  q_op, k_op, v_op, qkv_w_op, qkv_b_op, proj_w_op, proj_b_op,
                  mask_op, embed_dim, num_head, weights_reduction,
                  current_precision, mask_type);
            };

            TT_ASSIGN_OR_THROW(
                auto results,
                (DispatchOp<8, 2>(
                    std::move(op_builder),
                    {query, key, value, qkv_weight, qkv_bias, proj_weight,
                     proj_bias, *mask},
                    {.out_dtypes = out_dtypes,
                     .out_dims_list = out_dims_list,
                     .op_param_cache_keys = std::move(param_keys)})));

            return std::make_tuple(
                MakeTensor(results[0]),
                query.numel() == 0 ? at::Tensor() : MakeTensor(results[1]));
          } else {
            const auto op_builder =
                [embed_dim, num_head, weights_reduction,
                 current_precision](FixedSizeSpan<mlir::MlirOp, 7> inputs)
                -> absl::StatusOr<std::array<mlir::MlirOp, 2>> {
              auto& [q_op, k_op, v_op, qkv_w_op, qkv_b_op, proj_w_op,
                     proj_b_op] = inputs;
              return BuildNativeMultiHeadAttentionWithWeightsShlo(
                  q_op, k_op, v_op, qkv_w_op, qkv_b_op, proj_w_op, proj_b_op,
                  std::nullopt, embed_dim, num_head, weights_reduction,
                  current_precision);
            };

            TT_ASSIGN_OR_THROW(
                auto results,
                (DispatchOp<7, 2>(
                    std::move(op_builder),
                    {query, key, value, qkv_weight, qkv_bias, proj_weight,
                     proj_bias},
                    {.out_dtypes = out_dtypes,
                     .out_dims_list = out_dims_list,
                     .op_param_cache_keys = std::move(param_keys)})));

            return std::make_tuple(
                MakeTensor(results[0]),
                query.numel() == 0 ? at::Tensor() : MakeTensor(results[1]));
          }
        } else {
          // Build and dispatch multi-head attention graph returning output
          // only.
          if (mask.has_value() && mask->defined()) {
            const auto op_builder =
                [embed_dim, num_head, current_precision,
                 mask_type](FixedSizeSpan<mlir::MlirOp, 8> inputs)
                -> absl::StatusOr<mlir::MlirOp> {
              auto& [q_op, k_op, v_op, qkv_w_op, qkv_b_op, proj_w_op, proj_b_op,
                     mask_op] = inputs;
              return BuildNativeMultiHeadAttentionNoWeightsShlo(
                  q_op, k_op, v_op, qkv_w_op, qkv_b_op, proj_w_op, proj_b_op,
                  mask_op, embed_dim, num_head, current_precision, mask_type);
            };

            TT_ASSIGN_OR_THROW(
                auto result_buffer,
                (DispatchOp<8, 1>(
                    std::move(op_builder),
                    {query, key, value, qkv_weight, qkv_bias, proj_weight,
                     proj_bias, *mask},
                    {.out_dtype = query_dtype_mlir,
                     .out_dims = out_dims,
                     .op_param_cache_keys = std::move(param_keys)})));

            return std::make_tuple(MakeTensor(result_buffer), at::Tensor());
          } else {
            const auto op_builder = [embed_dim, num_head, current_precision](
                                        FixedSizeSpan<mlir::MlirOp, 7> inputs)
                -> absl::StatusOr<mlir::MlirOp> {
              auto& [q_op, k_op, v_op, qkv_w_op, qkv_b_op, proj_w_op,
                     proj_b_op] = inputs;
              return BuildNativeMultiHeadAttentionNoWeightsShlo(
                  q_op, k_op, v_op, qkv_w_op, qkv_b_op, proj_w_op, proj_b_op,
                  std::nullopt, embed_dim, num_head, current_precision);
            };

            TT_ASSIGN_OR_THROW(
                auto result_buffer,
                (DispatchOp<7, 1>(
                    std::move(op_builder),
                    {query, key, value, qkv_weight, qkv_bias, proj_weight,
                     proj_bias},
                    {.out_dtype = query_dtype_mlir,
                     .out_dims = out_dims,
                     .op_param_cache_keys = std::move(param_keys)})));

            return std::make_tuple(MakeTensor(result_buffer), at::Tensor());
          }
        }
      });
}

}  // namespace torch_tpu
