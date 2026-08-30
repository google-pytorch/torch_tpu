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

#include "csrc/ops/transformer_encoder_layer_fwd/_transformer_encoder_layer_fwd_aten_kernels.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/gelu/gelu.h"
#include "csrc/ops/layer_norm/layer_norm.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/softmax/softmax.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

void ValidateTransformerEncoderInputs(
    const at::Tensor& src, const int64_t embed_dim, const int64_t num_heads,
    const at::Tensor& qkv_weight, const at::Tensor& qkv_bias,
    const at::Tensor& proj_weight, const at::Tensor& proj_bias,
    const at::Tensor& norm_weight_1, const at::Tensor& norm_bias_1,
    const at::Tensor& norm_weight_2, const at::Tensor& norm_bias_2,
    const at::Tensor& ffn_weight_1, const at::Tensor& ffn_bias_1,
    const at::Tensor& ffn_weight_2, const at::Tensor& ffn_bias_2,
    const std::optional<at::Tensor>& mask) {
  TT_CHECK_THROW(src.dim() == 3, error::kInvalidArgument)
      << "expected 3-D src, got " << src.dim() << "-D tensor";
  TT_CHECK_THROW(src.size(2) == embed_dim, error::kInvalidArgument)
      << "expected last dimension of src to match embed_dim " << embed_dim
      << ", got " << src.size(2);
  TT_CHECK_THROW(embed_dim % num_heads == 0, error::kInvalidArgument)
      << "expected embed_dim to divide cleanly by num_heads, got "
      << "embed_dim " << embed_dim << " and num_heads " << num_heads;
  TT_CHECK_THROW(qkv_weight.dim() == 2, error::kInvalidArgument)
      << "expected 2-D qkv_weight, got " << qkv_weight.dim() << "-D tensor";
  TT_CHECK_THROW(
      qkv_weight.size(0) == 3 * embed_dim && qkv_weight.size(1) == embed_dim,
      error::kInvalidArgument)
      << "expected qkv_weight shape [" << 3 * embed_dim << ", " << embed_dim
      << "], got [" << qkv_weight.size(0) << ", " << qkv_weight.size(1) << "]";
  TT_CHECK_THROW(qkv_bias.dim() == 1, error::kInvalidArgument)
      << "expected 1-D qkv_bias, got " << qkv_bias.dim() << "-D tensor";
  TT_CHECK_THROW(qkv_bias.size(0) == 3 * embed_dim, error::kInvalidArgument)
      << "expected qkv_bias size " << 3 * embed_dim << ", got "
      << qkv_bias.size(0);
  TT_CHECK_THROW(proj_weight.dim() == 2, error::kInvalidArgument)
      << "expected 2-D proj_weight, got " << proj_weight.dim() << "-D tensor";
  TT_CHECK_THROW(
      proj_weight.size(0) == embed_dim && proj_weight.size(1) == embed_dim,
      error::kInvalidArgument)
      << "expected proj_weight shape [" << embed_dim << ", " << embed_dim
      << "], got [" << proj_weight.size(0) << ", " << proj_weight.size(1)
      << "]";
  TT_CHECK_THROW(proj_bias.dim() == 1, error::kInvalidArgument)
      << "expected 1-D proj_bias, got " << proj_bias.dim() << "-D tensor";
  TT_CHECK_THROW(proj_bias.size(0) == embed_dim, error::kInvalidArgument)
      << "expected proj_bias size " << embed_dim << ", got "
      << proj_bias.size(0);

  TT_CHECK_THROW(norm_weight_1.dim() == 1 && norm_weight_1.size(0) == embed_dim,
                 error::kInvalidArgument)
      << "expected 1-D norm_weight_1 with size " << embed_dim << ", got "
      << norm_weight_1.sizes();
  TT_CHECK_THROW(norm_bias_1.dim() == 1 && norm_bias_1.size(0) == embed_dim,
                 error::kInvalidArgument)
      << "expected 1-D norm_bias_1 with size " << embed_dim << ", got "
      << norm_bias_1.sizes();
  TT_CHECK_THROW(norm_weight_2.dim() == 1 && norm_weight_2.size(0) == embed_dim,
                 error::kInvalidArgument)
      << "expected 1-D norm_weight_2 with size " << embed_dim << ", got "
      << norm_weight_2.sizes();
  TT_CHECK_THROW(norm_bias_2.dim() == 1 && norm_bias_2.size(0) == embed_dim,
                 error::kInvalidArgument)
      << "expected 1-D norm_bias_2 with size " << embed_dim << ", got "
      << norm_bias_2.sizes();

  TT_CHECK_THROW(ffn_weight_1.dim() == 2, error::kInvalidArgument)
      << "expected 2-D ffn_weight_1, got " << ffn_weight_1.dim() << "-D tensor";
  const int64_t d_ff = ffn_weight_1.size(0);
  TT_CHECK_THROW(ffn_weight_1.size(1) == embed_dim, error::kInvalidArgument)
      << "expected ffn_weight_1 second dim to be " << embed_dim << ", got "
      << ffn_weight_1.size(1);
  TT_CHECK_THROW(ffn_bias_1.dim() == 1 && ffn_bias_1.size(0) == d_ff,
                 error::kInvalidArgument)
      << "expected 1-D ffn_bias_1 with size " << d_ff << ", got "
      << ffn_bias_1.sizes();
  TT_CHECK_THROW(ffn_weight_2.dim() == 2 && ffn_weight_2.size(0) == embed_dim &&
                     ffn_weight_2.size(1) == d_ff,
                 error::kInvalidArgument)
      << "expected ffn_weight_2 shape [" << embed_dim << ", " << d_ff
      << "], got " << ffn_weight_2.sizes();
  TT_CHECK_THROW(ffn_bias_2.dim() == 1 && ffn_bias_2.size(0) == embed_dim,
                 error::kInvalidArgument)
      << "expected 1-D ffn_bias_2 with size " << embed_dim << ", got "
      << ffn_bias_2.sizes();

  if (mask.has_value() && mask->defined()) {
    TT_CHECK_THROW(mask->dim() >= 2 && mask->dim() <= 4,
                   error::kInvalidArgument)
        << "expected 2-D, 3-D, or 4-D mask, got " << mask->dim() << "-D tensor";
    TT_CHECK_THROW(mask->scalar_type() == at::kBool ||
                       c10::isFloatingType(mask->scalar_type()),
                   error::kInvalidArgument)
        << "expected mask dtype to be bool or floating-point, got "
        << ToString(mask->scalar_type());
  }
}

absl::StatusOr<mlir::MlirOp> ReshapeMask2DShlo(
    mlir::MlirOp mask_val, const std::optional<int64_t> mask_type) {
  const mlir::RankedTensorType mask_t = GetTensorTypeOrDie(mask_val);
  const auto mask_shape = mask_t.getShape();
  const bool is_key_padding =
      (mask_type.has_value() && *mask_type == 1) ||
      (!mask_type.has_value() && mask_shape[0] != mask_shape[1]);
  if (is_key_padding) {
    // Key-padding mask [B, T] -> [B, 1, 1, T]
    TT_ASSIGN_OR_RETURN(mask_val, Unsqueeze(mask_val, 1));
    return Unsqueeze(mask_val, 2);
  }
  // Attn mask [T, T] -> [1, 1, T, T]
  TT_ASSIGN_OR_RETURN(mask_val, Unsqueeze(mask_val, 0));
  return Unsqueeze(mask_val, 1);
}

absl::StatusOr<mlir::MlirOp> ReshapeMask3DShlo(
    mlir::MlirOp mask_val, const std::optional<int64_t> mask_type,
    const int64_t B_dummy, const int64_t T_dummy, const int64_t nH) {
  const mlir::RankedTensorType mask_t = GetTensorTypeOrDie(mask_val);
  const auto mask_shape = mask_t.getShape();
  if (mask_type.has_value() && *mask_type == 1) {
    // Key-padding mask [B, 1, T] -> [B, 1, 1, T]
    return Unsqueeze(mask_val, 2);
  }
  if (mask_shape[0] == nH) {
    // Per-head mask [nH, T, T] -> [1, nH, T, T]
    return Unsqueeze(mask_val, 0);
  }
  if (B_dummy > 0 && mask_shape[0] > 0 && mask_shape[0] == B_dummy * nH) {
    // 3D mask [B * nH, T, T] -> [B, nH, T, T] (static shapes only)
    return ReshapeFromStaticDimensions(mask_val,
                                       {B_dummy * nH, T_dummy, T_dummy},
                                       {B_dummy, nH, T_dummy, T_dummy});
  }
  // 3D mask [B, T, T] -> [B, 1, T, T] (supports static and dynamic B)
  return Unsqueeze(mask_val, 1);
}

absl::StatusOr<mlir::MlirOp> ReshapeMaskTo4DShlo(
    mlir::MlirOp mask_val, const std::optional<int64_t> mask_type,
    const int64_t B_dummy, const int64_t T_dummy, const int64_t nH) {
  const mlir::RankedTensorType mask_t = GetTensorTypeOrDie(mask_val);
  const int64_t mask_rank = mask_t.getRank();
  if (mask_rank == 2) {
    return ReshapeMask2DShlo(mask_val, mask_type);
  }
  if (mask_rank == 3) {
    return ReshapeMask3DShlo(mask_val, mask_type, B_dummy, T_dummy, nH);
  }
  return mask_val;
}

absl::StatusOr<mlir::MlirOp> ApplyAttentionMaskShlo(
    mlir::MlirOp attn_logits, mlir::MlirOp mask_op,
    const std::optional<int64_t> mask_type, const mlir::Type elem_type,
    const int64_t B_dummy, const int64_t T_dummy, const int64_t nH,
    mlir::MlirBuilder& builder) {
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp mask_val,
      ReshapeMaskTo4DShlo(mask_op, mask_type, B_dummy, T_dummy, nH));

  const mlir::RankedTensorType mask_t = GetTensorTypeOrDie(mask_val);
  const mlir::Type mask_elem_type = mask_t.getElementType();
  if (mask_elem_type.isInteger(1)) {
    const mlir::Attribute min_attr =
        GetMinFiniteValueAttr(elem_type, builder.getOpBuilder());
    mlir::MlirOp min_val = mlir::MlirOp(
        builder, mlir::stablehlo::ConstantOp::create(
                     builder.getOpBuilder(), builder.getLoc(), min_attr));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp neg_inf_bcast,
                        BroadcastIfNeeded(min_val, attn_logits));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp mask_bcast,
                        BroadcastIfNeeded(mask_val, attn_logits));
    return mlir::stablehlo::Select(mask_bcast, neg_inf_bcast, attn_logits);
  }

  TT_ASSIGN_OR_RETURN(const auto target_elem_type,
                      ConvertTo<mlir::ElementType>(elem_type));
  TT_ASSIGN_OR_RETURN(mask_val, CastIfNeeded(mask_val, target_elem_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask_bcast,
                      BroadcastIfNeeded(mask_val, attn_logits));
  return mlir::stablehlo::Add(attn_logits, mask_bcast);
}

absl::StatusOr<mlir::MlirOp> BuildMultiheadAttentionShlo(
    mlir::MlirOp x_op, mlir::MlirOp qkv_weight_op, mlir::MlirOp qkv_bias_op,
    mlir::MlirOp proj_weight_op, mlir::MlirOp proj_bias_op,
    const std::optional<mlir::MlirOp>& mask_op,
    const std::optional<int64_t> mask_type, const int64_t B, const int64_t T,
    const int64_t D, const int64_t nH, const int64_t d_k,
    mlir::MlirBuilder& builder) {
  mlir::MLIRContext& ctx = builder.getContext();

  // 1. QKV Linear Projection: x_op @ qkv_weight_op.T + qkv_bias_op
  const auto qkv_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      &ctx, /*lhs_batching_dimensions=*/{},
      /*rhs_batching_dimensions=*/{},
      /*lhs_contracting_dimensions=*/{2},
      /*rhs_contracting_dimensions=*/{1});
  mlir::MlirOp qkv_mat =
      mlir::stablehlo::DotGeneral(x_op, qkv_weight_op, qkv_dot_dims);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp qkv_bias_bcast,
                      BroadcastIfNeeded(qkv_bias_op, qkv_mat));
  mlir::MlirOp qkv_op = mlir::stablehlo::Add(qkv_mat, qkv_bias_bcast);

  // 2. Slice and reshape Q, K, V
  const int64_t B_dummy = B > 0 ? B : 1;
  const int64_t T_dummy = T > 0 ? T : 1;

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp qkv_4d,
      ReshapeFromStaticDimensions(qkv_op, {B_dummy, T_dummy, 3 * D},
                                  {B_dummy, T_dummy, 3, D}));

  const mlir::RankedTensorType qkv_4d_type = GetTensorTypeOrDie(qkv_4d);
  const auto qkv_4d_shape = qkv_4d_type.getShape();
  const int64_t B_s = qkv_4d_shape[0];
  const int64_t T_s = qkv_4d_shape[1];

  mlir::MlirOp q_raw_4d = mlir::stablehlo::Slice(
      qkv_4d, /*start_indices=*/{0, 0, 0, 0},
      /*limit_indices=*/{B_s, T_s, 1, D}, /*strides=*/{1, 1, 1, 1});
  mlir::MlirOp k_raw_4d = mlir::stablehlo::Slice(
      qkv_4d, /*start_indices=*/{0, 0, 1, 0},
      /*limit_indices=*/{B_s, T_s, 2, D}, /*strides=*/{1, 1, 1, 1});
  mlir::MlirOp v_raw_4d = mlir::stablehlo::Slice(
      qkv_4d, /*start_indices=*/{0, 0, 2, 0},
      /*limit_indices=*/{B_s, T_s, 3, D}, /*strides=*/{1, 1, 1, 1});

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp q_raw,
      ReshapeFromStaticDimensions(q_raw_4d, {B_dummy, T_dummy, 1, D},
                                  {B_dummy, T_dummy, D}));
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp k_raw,
      ReshapeFromStaticDimensions(k_raw_4d, {B_dummy, T_dummy, 1, D},
                                  {B_dummy, T_dummy, D}));
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp v_raw,
      ReshapeFromStaticDimensions(v_raw_4d, {B_dummy, T_dummy, 1, D},
                                  {B_dummy, T_dummy, D}));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp q_4d,
                      ReshapeFromStaticDimensions(q_raw, {B_dummy, T_dummy, D},
                                                  {B_dummy, T_dummy, nH, d_k}));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp k_4d,
                      ReshapeFromStaticDimensions(k_raw, {B_dummy, T_dummy, D},
                                                  {B_dummy, T_dummy, nH, d_k}));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp v_4d,
                      ReshapeFromStaticDimensions(v_raw, {B_dummy, T_dummy, D},
                                                  {B_dummy, T_dummy, nH, d_k}));

  mlir::MlirOp q_trans = mlir::stablehlo::Transpose(q_4d, {0, 2, 1, 3});
  mlir::MlirOp k_trans = mlir::stablehlo::Transpose(k_4d, {0, 2, 1, 3});
  mlir::MlirOp v_trans = mlir::stablehlo::Transpose(v_4d, {0, 2, 1, 3});

  // Scale Q by 1 / sqrt(d_k)
  const mlir::RankedTensorType src_type = GetTensorTypeOrDie(x_op);
  const auto elem_type = mlir::cast<mlir::FloatType>(src_type.getElementType());
  const double scale_val = 1.0 / std::sqrt(static_cast<double>(d_k));
  mlir::MlirOp scale_const = MakeScalarConstant(builder, scale_val, elem_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp scale_bcast,
                      BroadcastIfNeeded(scale_const, q_trans));
  mlir::MlirOp q_scaled = mlir::stablehlo::Mul(q_trans, scale_bcast);

  // 3. Attention Logits: q_scaled @ k_trans.T -> [B, nH, T, T]
  const auto attn_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      &ctx, /*lhs_batching_dimensions=*/{0, 1},
      /*rhs_batching_dimensions=*/{0, 1},
      /*lhs_contracting_dimensions=*/{3},
      /*rhs_contracting_dimensions=*/{3});
  mlir::MlirOp attn_logits =
      mlir::stablehlo::DotGeneral(q_scaled, k_trans, attn_dot_dims);

  // Mask handling
  if (mask_op.has_value()) {
    TT_ASSIGN_OR_RETURN(
        attn_logits,
        ApplyAttentionMaskShlo(attn_logits, *mask_op, mask_type, elem_type,
                               B_dummy, T_dummy, nH, builder));
  }

  // Softmax along last dim (-1)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp attn_weights,
                      BuildSoftmaxShlo(attn_logits, -1, SoftmaxMode::kSoftmax));

  // 4. Attention Output: attn_weights @ v_trans -> [B, nH, T, d_k]
  const auto v_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      &ctx, /*lhs_batching_dimensions=*/{0, 1},
      /*rhs_batching_dimensions=*/{0, 1},
      /*lhs_contracting_dimensions=*/{3},
      /*rhs_contracting_dimensions=*/{2});
  mlir::MlirOp attn_out_4d =
      mlir::stablehlo::DotGeneral(attn_weights, v_trans, v_dot_dims);

  // 5. Reshape & Permute back: [B, nH, T, d_k] -> [B, T, nH, d_k] -> [B, T, D]
  mlir::MlirOp attn_out_perm =
      mlir::stablehlo::Transpose(attn_out_4d, {0, 2, 1, 3});
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp attn_out_3d,
      ReshapeFromStaticDimensions(attn_out_perm, {B_dummy, T_dummy, nH, d_k},
                                  {B_dummy, T_dummy, D}));

  // 6. Output Projection: attn_out_3d @ proj_weight_op.T + proj_bias_op
  const auto proj_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      &ctx, /*lhs_batching_dimensions=*/{},
      /*rhs_batching_dimensions=*/{},
      /*lhs_contracting_dimensions=*/{2},
      /*rhs_contracting_dimensions=*/{1});
  mlir::MlirOp proj_mat =
      mlir::stablehlo::DotGeneral(attn_out_3d, proj_weight_op, proj_dot_dims);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp proj_bias_bcast,
                      BroadcastIfNeeded(proj_bias_op, proj_mat));
  mlir::MlirOp proj_out = mlir::stablehlo::Add(proj_mat, proj_bias_bcast);

  return proj_out;
}

enum class Activation { kRelu, kGelu };

absl::StatusOr<mlir::MlirOp> BuildFeedForwardShlo(
    mlir::MlirOp x_op, mlir::MlirOp ffn_weight_1_op, mlir::MlirOp ffn_bias_1_op,
    mlir::MlirOp ffn_weight_2_op, mlir::MlirOp ffn_bias_2_op,
    const Activation activation, const mlir::ElementType out_dtype,
    mlir::MlirBuilder& builder) {
  mlir::MLIRContext& ctx = builder.getContext();

  // 1. FFN Linear 1: x_op @ ffn_weight_1_op.T + ffn_bias_1_op
  const auto ffn1_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      &ctx, /*lhs_batching_dimensions=*/{},
      /*rhs_batching_dimensions=*/{},
      /*lhs_contracting_dimensions=*/{2},
      /*rhs_contracting_dimensions=*/{1});
  mlir::MlirOp ffn1_mat =
      mlir::stablehlo::DotGeneral(x_op, ffn_weight_1_op, ffn1_dot_dims);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp ffn1_bias_bcast,
                      BroadcastIfNeeded(ffn_bias_1_op, ffn1_mat));
  mlir::MlirOp ffn1_out = mlir::stablehlo::Add(ffn1_mat, ffn1_bias_bcast);

  // 2. FFN Activation: GELU / RELU
  mlir::MlirOp ffn1_act;
  if (activation == Activation::kGelu) {
    TT_ASSIGN_OR_RETURN(ffn1_act, BuildGeluShlo(ffn1_out, "none", out_dtype));
  } else {
    mlir::MlirOp zero_const = MakeConstantLike(ffn1_out, 0.0);
    ffn1_act = mlir::stablehlo::Max(ffn1_out, zero_const);
  }

  // 3. FFN Linear 2: ffn1_act @ ffn_weight_2_op.T + ffn_bias_2_op
  const auto ffn2_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      &ctx, /*lhs_batching_dimensions=*/{},
      /*rhs_batching_dimensions=*/{},
      /*lhs_contracting_dimensions=*/{2},
      /*rhs_contracting_dimensions=*/{1});
  mlir::MlirOp ffn2_mat =
      mlir::stablehlo::DotGeneral(ffn1_act, ffn_weight_2_op, ffn2_dot_dims);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp ffn2_bias_bcast,
                      BroadcastIfNeeded(ffn_bias_2_op, ffn2_mat));
  mlir::MlirOp ffn2_out = mlir::stablehlo::Add(ffn2_mat, ffn2_bias_bcast);

  return ffn2_out;
}

}  // namespace

at::Tensor AtenTransformerEncoderLayerFwd(
    const at::Tensor& src, const int64_t embed_dim, const int64_t num_heads,
    const at::Tensor& qkv_weight, const at::Tensor& qkv_bias,
    const at::Tensor& proj_weight, const at::Tensor& proj_bias,
    const bool use_gelu, const bool norm_first, const double eps,
    const at::Tensor& norm_weight_1, const at::Tensor& norm_bias_1,
    const at::Tensor& norm_weight_2, const at::Tensor& norm_bias_2,
    const at::Tensor& ffn_weight_1, const at::Tensor& ffn_bias_1,
    const at::Tensor& ffn_weight_2, const at::Tensor& ffn_bias_2,
    const std::optional<at::Tensor>& mask,
    const std::optional<int64_t> mask_type) {
  TT_KERNEL(
      OpName::kTransformerEncoderLayerFwd, param_keys,
      (src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias,
       use_gelu, norm_first, eps, norm_weight_1, norm_bias_1, norm_weight_2,
       norm_bias_2, ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, mask,
       mask_type),
      {
        ValidateTransformerEncoderInputs(
            src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight,
            proj_bias, norm_weight_1, norm_bias_1, norm_weight_2, norm_bias_2,
            ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, mask);

        if (src.numel() == 0) {
          TT_ASSIGN_OR_THROW(
              at::Tensor out,
              MakeEmptyTensor(src.sizes(), src.scalar_type(), src.device()));
          return out;
        }

        const bool has_mask = mask.has_value() && mask->defined();
        std::vector<at::Tensor> inputs = {
            src,         qkv_weight,    qkv_bias,    proj_weight,
            proj_bias,   norm_weight_1, norm_bias_1, norm_weight_2,
            norm_bias_2, ffn_weight_1,  ffn_bias_1,  ffn_weight_2,
            ffn_bias_2};
        if (has_mask) {
          inputs.push_back(*mask);
        }

        const c10::ScalarType out_type = InferOutputDtype(src);
        TT_ASSIGN_OR_THROW(const auto out_dtype,
                           ConvertTo<mlir::ElementType>(out_type));
        const Dimensions output_shape = CopyIntVector(src.sizes());

        auto op_builder =
            [embed_dim, num_heads, use_gelu, norm_first, eps, has_mask,
             mask_type, out_dtype](
                absl::Span<mlir::MlirOp> builder_inputs,
                mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
          mlir::MlirOp src_op = builder_inputs[0];
          mlir::MlirOp qkv_weight_op = builder_inputs[1];
          mlir::MlirOp qkv_bias_op = builder_inputs[2];
          mlir::MlirOp proj_weight_op = builder_inputs[3];
          mlir::MlirOp proj_bias_op = builder_inputs[4];
          mlir::MlirOp norm_weight_1_op = builder_inputs[5];
          mlir::MlirOp norm_bias_1_op = builder_inputs[6];
          mlir::MlirOp norm_weight_2_op = builder_inputs[7];
          mlir::MlirOp norm_bias_2_op = builder_inputs[8];
          mlir::MlirOp ffn_weight_1_op = builder_inputs[9];
          mlir::MlirOp ffn_bias_1_op = builder_inputs[10];
          mlir::MlirOp ffn_weight_2_op = builder_inputs[11];
          mlir::MlirOp ffn_bias_2_op = builder_inputs[12];
          std::optional<mlir::MlirOp> mask_op;
          if (has_mask) {
            mask_op = builder_inputs[13];
          }

          const mlir::RankedTensorType src_type = GetTensorTypeOrDie(src_op);
          const auto src_shape = src_type.getShape();
          const int64_t B = src_shape[0];
          const int64_t T = src_shape[1];
          const int64_t D = embed_dim;
          const int64_t nH = num_heads;
          const int64_t d_k = D / nH;

          mlir::MlirOp x_op = src_op;

          // 1. LayerNorm 1 (Pre-LN)
          if (norm_first) {
            TT_ASSIGN_OR_RETURN(
                const LayerNormShloResults norm1_res,
                BuildLayerNormShlo(x_op, norm_weight_1_op, norm_bias_1_op,
                                   /*normalized_num_dims=*/1, eps));
            x_op = norm1_res.normalized_values;
          }

          // 2. Multihead Attention
          TT_ASSIGN_OR_RETURN(
              mlir::MlirOp proj_out,
              BuildMultiheadAttentionShlo(
                  x_op, qkv_weight_op, qkv_bias_op, proj_weight_op,
                  proj_bias_op, mask_op, mask_type, B, T, D, nH, d_k, builder));

          // 3. First residual connection & optional Post-LN 1
          x_op = mlir::stablehlo::Add(src_op, proj_out);
          if (!norm_first) {
            TT_ASSIGN_OR_RETURN(
                const LayerNormShloResults norm1_res,
                BuildLayerNormShlo(x_op, norm_weight_1_op, norm_bias_1_op,
                                   /*normalized_num_dims=*/1, eps));
            x_op = norm1_res.normalized_values;
          }

          // 4. Pre-FFN residual save & optional Pre-LN 2
          mlir::MlirOp pre_ffn_res = x_op;
          if (norm_first) {
            TT_ASSIGN_OR_RETURN(
                const LayerNormShloResults norm2_res,
                BuildLayerNormShlo(x_op, norm_weight_2_op, norm_bias_2_op,
                                   /*normalized_num_dims=*/1, eps));
            x_op = norm2_res.normalized_values;
          }

          // 5. FeedForward Network
          TT_ASSIGN_OR_RETURN(
              mlir::MlirOp ffn2_out,
              BuildFeedForwardShlo(
                  x_op, ffn_weight_1_op, ffn_bias_1_op, ffn_weight_2_op,
                  ffn_bias_2_op,
                  use_gelu ? Activation::kGelu : Activation::kRelu, out_dtype,
                  builder));

          // 6. Second residual connection & optional Post-LN 2
          x_op = mlir::stablehlo::Add(pre_ffn_res, ffn2_out);
          if (!norm_first) {
            TT_ASSIGN_OR_RETURN(
                const LayerNormShloResults norm2_res,
                BuildLayerNormShlo(x_op, norm_weight_2_op, norm_bias_2_op,
                                   /*normalized_num_dims=*/1, eps));
            x_op = norm2_res.normalized_values;
          }

          return x_op;
        };

        TT_ASSIGN_OR_THROW(DeviceBufferRef result,
                           DispatchOp<kDynamicSize>(
                               std::move(op_builder), inputs,
                               {.out_dtype = out_dtype,
                                .out_dims = output_shape,
                                .op_param_cache_keys = std::move(param_keys)}));

        return MakeTensor(result);
      });
}

}  // namespace torch_tpu
