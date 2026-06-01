/*
 * Copyright 2025 Google LLC
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

#include "torch_tpu/ops/scaled_dot_product_attention/scaled_dot_product_attention_aten_kernels.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/Context.h"
#include "ATen/SDPBackend.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBase.h"
#include "ATen/core/grad_mode.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/zeros.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "c10/core/DeviceType.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Exception.h"
#include "c10/util/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/_internal/mosaic/op_builders.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/reductions/sum.h"
#include "torch_tpu/ops/scaled_dot_product_attention/flash_attention_kernel.h"
#include "torch_tpu/ops/scaled_dot_product_attention/helpers.h"
#include "torch_tpu/ops/scaled_dot_product_attention/scaled_dot_product_attention_shlo.h"
#include "torch_tpu/ops/view_decomposition/contiguous_to_view.h"

namespace torch_tpu {

namespace {

absl::Status CheckDtype(at::ScalarType dtype) {
  if (dtype == at::kFloat || dtype == at::kBFloat16) {
    return absl::OkStatus();
  }
  return TT_ERROR(error::kPythonNotImplementedError)
         << "unsupported dtype for sdpa custom kernel";
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor>>
ScaledDotProductFusedAttentionInternal(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask, bool is_causal,
    std::optional<double> scale, bool return_lse) {
  TT_RETURN_IF_ERROR(CheckDtype(query.scalar_type()));

  int rank = query.ndimension();
  int batch_size = 1;
  for (int i = 0; i < rank - 3; i++) batch_size *= query.size(i);
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(query.scalar_type()));

  Dimensions out_dims(query.sizes().begin(), query.sizes().end());
  out_dims[rank - 1] = value.size(rank - 1);

  const auto query_scalar_type = query.scalar_type();
  // Capture static dimension parameters (like 'head_dim') by copy instead of
  // capturing the 'at::Tensor query' handle. This prevents locking 'query' and
  // its entire autograd graph in memory, avoiding high peak HBM consumption.
  const int64_t head_dim = query.size(rank - 1);
  const int64_t num_heads = query.size(rank - 3);
  const int64_t kv_num_heads = key.size(rank - 3);
  const int64_t vo_head_dim = value.size(rank - 1);
  const int64_t q_seq_len = query.size(rank - 2);
  const int64_t kv_seq_len = key.size(rank - 2);

  auto op_builder =
      [rank, batch_size, out_dims, query_scalar_type, is_causal, scale,
       head_dim, num_heads, kv_num_heads, vo_head_dim, q_seq_len, kv_seq_len,
       return_lse](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<DynamicMlirOpResults> {
    mlir::MlirOp query_mlir = inputs[0];
    mlir::MlirOp key_mlir = inputs[1];
    mlir::MlirOp value_mlir = inputs[2];
    mlir::MlirOp mask_mlir;
    if (inputs.size() == 4) {
      mask_mlir = inputs[3];
      mlir::RankedTensorType mask_type = GetTensorTypeOrDie(mask_mlir);
      int mask_rank = mask_type.getRank();
      if (mask_rank > 4) {
        mask_mlir = flatten_batch_dims(mask_mlir, batch_size, rank - 3);
      } else if (mask_rank < 4) {
        int prefix_dims = 4 - mask_rank;
        mlir::SmallVector<int64_t> new_dims(prefix_dims, 1);
        new_dims.append(mask_type.getShape().begin(),
                        mask_type.getShape().end());
        mask_mlir = mlir::stablehlo::Reshape(mask_mlir, new_dims);
      }
    }

    mlir::torch_tpu::FlashAttnConfig config;
    config.return_lse = return_lse;
    TT_ASSIGN_OR_RETURN(config.element_type,
                        ConvertTo<mlir::ElementType>(query_scalar_type));
    config.batch_size = batch_size;
    config.num_heads = num_heads;
    const bool is_mqa = kv_num_heads < num_heads;
    int64_t head_count_ratio = 1;
    if (is_mqa) {
      if (num_heads % kv_num_heads != 0) {
        return TT_ERROR(error::kInvalidArgument)
               << "num_heads must be divisible by kv_num_heads for MQA/GQA";
      }
      config.kv_num_heads = num_heads;
      head_count_ratio = num_heads / kv_num_heads;
    } else {
      config.kv_num_heads = kv_num_heads;
    }
    config.qk_head_dim = head_dim;
    config.vo_head_dim = vo_head_dim;
    config.q_sequence_length = q_seq_len;
    config.kv_sequence_length = kv_seq_len;
    config.is_causal = is_causal;
    config.scale = scale.value_or(1.0 / std::sqrt(head_dim));
    config.contract_precision_qk_dot =
        query_scalar_type == at::kFloat
            ? mlir::torch_tpu::ContractPrecision::kF32
            : mlir::torch_tpu::ContractPrecision::kBF16;
    config.contract_precision_root_dot = config.contract_precision_qk_dot;
    config.has_attn_bias = mask_mlir.isValid();

    if (mask_mlir.isValid()) {
      auto shape = GetTensorTypeOrDie(mask_mlir).getShape();
      for (auto [idx, dim] : llvm::enumerate(shape)) {
        if (dim == 1) {
          config.mask_broadcast_dims.push_back(idx);
        }
      }
    }

    TT_ASSIGN_OR_RETURN(std::string kernel_mlir,
                        mlir::torch_tpu::CreateKernel(config));

    builder.getContext().getOrLoadDialect<mlir::scf::SCFDialect>();
    builder.getContext().getOrLoadDialect<mlir::vector::VectorDialect>();
    builder.getContext().getOrLoadDialect<mlir::math::MathDialect>();

    // Flatten batch dimensions of inputs.
    mlir::MlirOp query_4d =
        flatten_batch_dims(query_mlir, batch_size, rank - 3);
    mlir::MlirOp key_4d = flatten_batch_dims(key_mlir, batch_size, rank - 3);
    mlir::MlirOp value_4d =
        flatten_batch_dims(value_mlir, batch_size, rank - 3);

    if (is_mqa) {
      auto replicate = [&](mlir::MlirOp input, int64_t seq_len, int64_t h_dim) {
        auto broadcast_type = mlir::RankedTensorType::get(
            {batch_size, kv_num_heads, head_count_ratio, seq_len, h_dim},
            GetTensorTypeOrDie(input).getElementType());
        mlir::MlirOp broadcasted_input = mlir::stablehlo::BroadcastInDim(
            broadcast_type, input, {0, 1, 3, 4});
        return mlir::stablehlo::Reshape(
            broadcasted_input, {batch_size, num_heads, seq_len, h_dim});
      };
      key_4d = replicate(key_4d, kv_seq_len, head_dim);
      value_4d = replicate(value_4d, kv_seq_len, vo_head_dim);
    }

    Dimensions out_dims_4d = {batch_size, num_heads, q_seq_len, vo_head_dim};
    auto out_type = mlir::RankedTensorType::get(
        out_dims_4d, GetTensorTypeOrDie(query_mlir).getElementType());

    mlir::SmallVector<mlir::Value, 4> operands{
        query_4d.getValue(), key_4d.getValue(), value_4d.getValue()};
    if (mask_mlir.isValid()) {
      operands.push_back(mask_mlir.getValue());
    }

    mlir::SmallVector<mlir::Type, 2> result_types = {out_type};
    auto query_type = GetTensorTypeOrDie(query_mlir);
    Dimensions lse_dims(query_type.getShape().begin(),
                        query_type.getShape().end() - 1);
    Dimensions lse_dims_4d = {batch_size, num_heads, 1, q_seq_len};

    if (return_lse) {
      // LSE is 4D [B, N, 1, S]
      auto ml_type = mlir::RankedTensorType::get(
          lse_dims_4d, builder.getOpBuilder().getF32Type());
      result_types.push_back(ml_type);
    }

    auto custom_call = mlir::torch_tpu::CreateCustomCallOp(
        builder.getOpBuilder(), builder.getLoc(), kernel_mlir, operands,
        result_types);

    DynamicMlirOpResults results;
    results.push_back(unflatten_batch_dims(
        mlir::MlirOp(builder, custom_call.getResult(0)), out_dims));

    if (return_lse) {
      results.push_back(unflatten_batch_dims(
          mlir::MlirOp(builder, custom_call.getResult(1)), lse_dims));
    }

    return results;
  };

  TT_ASSIGN_OR_RETURN(auto param_keys,
                      *OpParamCacheKeysBuilder()
                           .SetParam("is_causal", is_causal)
                           .SetParam("scale", scale)
                           .SetParam("attn_mask", attn_mask)
                           .SetParam("return_lse", return_lse));

  std::vector<at::Tensor> dispatch_inputs = {query, key, value};

  if (attn_mask.has_value() && attn_mask->defined()) {
    dispatch_inputs.push_back(*attn_mask);
  }

  Dimensions lse_dims(query.sizes().begin(), query.sizes().end() - 1);
  std::vector<mlir::ElementType> out_dtypes = {out_dtype};
  std::vector<absl::Span<const int64_t>> out_dims_list = {out_dims};
  if (return_lse) {
    out_dtypes.push_back(mlir::ElementType::F32);
    out_dims_list.push_back(lse_dims);
  }

  TT_ASSIGN_OR_RETURN(auto results,
                      (DispatchOp<kDynamicSize, kDynamicSize>(
                          std::move(op_builder), dispatch_inputs,
                          {.out_dtypes = out_dtypes,
                           .out_dims_list = out_dims_list,
                           .op_param_cache_keys = std::move(param_keys)})));

  TT_ASSIGN_OR_RETURN(
      at::Tensor strided_result,
      ContiguousToView(results[0], query.strides(), query.storage_offset()));

  if (return_lse) {
    return {{strided_result, MakeTensor(std::move(results[1]))}};
  } else {
    return {{strided_result,
             at::empty(lse_dims, query.options().dtype(at::kFloat))}};
  }
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor>>
ScaledDotProductFusedAttentionWithLseImpl(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask, bool is_causal,
    std::optional<double> scale) {
  const bool return_lse =
      at::GradMode::is_enabled() &&
      (query.requires_grad() || key.requires_grad() || value.requires_grad());
  return ScaledDotProductFusedAttentionInternal(query, key, value, attn_mask,
                                                is_causal, scale, return_lse);
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
ScaledDotProductFusedAttentionBackwardWithLseImpl(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& out, const at::Tensor& logsumexp,
    std::optional<at::Tensor> attn_bias, std::optional<double> scale,
    bool is_causal) {
  TT_RETURN_IF_ERROR(CheckDtype(query.scalar_type()));

  int rank = query.ndimension();
  int batch_size = 1;
  for (int i = 0; i < rank - 3; i++) batch_size *= query.size(i);

  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(query.scalar_type()));

  const auto query_scalar_type = query.scalar_type();
  // Capture static dimension parameters (like 'head_dim') by copy instead of
  // capturing the 'at::Tensor query' handle. This prevents locking 'query' and
  // its entire autograd graph in memory, avoiding high peak HBM consumption.
  const int64_t head_dim = query.size(rank - 1);
  const int64_t num_heads = query.size(rank - 3);
  const int64_t kv_num_heads = key.size(rank - 3);
  const int64_t vo_head_dim = value.size(rank - 1);
  const int64_t q_seq_len = query.size(rank - 2);
  const int64_t kv_seq_len = key.size(rank - 2);

  auto op_builder =
      [rank, batch_size, query_scalar_type, is_causal, scale, head_dim,
       num_heads, kv_num_heads, vo_head_dim, q_seq_len, kv_seq_len](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<MlirOpResults<3>> {
    mlir::MlirOp grad_out_mlir = inputs[0];
    mlir::MlirOp query_mlir = inputs[1];
    mlir::MlirOp key_mlir = inputs[2];
    mlir::MlirOp value_mlir = inputs[3];
    mlir::MlirOp logsumexp_mlir = inputs[4];
    mlir::MlirOp out_mlir = inputs[5];

    // Compute di in SHLO
    mlir::Type f32_type = builder.getOpBuilder().getF32Type();
    mlir::MlirOp out_f32 =
        mlir::stablehlo::ConvertElementType(out_mlir, f32_type);
    mlir::MlirOp grad_out_f32 =
        mlir::stablehlo::ConvertElementType(grad_out_mlir, f32_type);

    TT_ASSIGN_OR_RETURN(auto mul_di, BuildMulShlo(out_f32, grad_out_f32));
    TT_ASSIGN_OR_RETURN(auto di_shlo, BuildSumShlo(mul_di, {rank - 1},
                                                   ReductionMode::kDropDims));

    // Reshape di_shlo to 4D [B_flat, N, 1, S]
    Dimensions aux_dims_4d = {batch_size, num_heads, 1, q_seq_len};
    mlir::MlirOp di_4d = mlir::stablehlo::Reshape(di_shlo, aux_dims_4d);

    mlir::MlirOp mask_mlir;
    if (inputs.size() == 7) {
      mask_mlir = inputs[6];
      mlir::RankedTensorType mask_type = GetTensorTypeOrDie(mask_mlir);
      int mask_rank = mask_type.getRank();
      if (mask_rank > 4) {
        mask_mlir = flatten_batch_dims(mask_mlir, batch_size, rank - 3);
      } else if (mask_rank < 4) {
        int prefix_dims = 4 - mask_rank;
        mlir::SmallVector<int64_t> new_dims(prefix_dims, 1);
        new_dims.append(mask_type.getShape().begin(),
                        mask_type.getShape().end());
        mask_mlir = mlir::stablehlo::Reshape(mask_mlir, new_dims);
      }
    }

    mlir::torch_tpu::FlashAttnConfig config;
    TT_ASSIGN_OR_RETURN(config.element_type,
                        ConvertTo<mlir::ElementType>(query_scalar_type));
    config.batch_size = batch_size;
    config.num_heads = num_heads;
    const bool is_mqa = kv_num_heads < num_heads;
    int64_t head_count_ratio = 1;
    if (is_mqa) {
      if (num_heads % kv_num_heads != 0) {
        return TT_ERROR(error::kInvalidArgument)
               << "num_heads must be divisible by kv_num_heads for MQA/GQA";
      }
      config.kv_num_heads = num_heads;
      head_count_ratio = num_heads / kv_num_heads;
    } else {
      config.kv_num_heads = kv_num_heads;
    }
    config.qk_head_dim = head_dim;
    config.vo_head_dim = vo_head_dim;
    config.q_sequence_length = q_seq_len;
    config.kv_sequence_length = kv_seq_len;
    config.is_causal = is_causal;
    config.scale = scale.value_or(1.0 / std::sqrt(head_dim));
    config.has_attn_bias = mask_mlir.isValid();

    if (mask_mlir.isValid()) {
      auto shape = GetTensorTypeOrDie(mask_mlir).getShape();
      for (auto [idx, dim] : llvm::enumerate(shape)) {
        if (dim == 1) {
          config.mask_broadcast_dims.push_back(idx);
        }
      }
    }

    config.contract_precision_qk_dot =
        query_scalar_type == at::kFloat
            ? mlir::torch_tpu::ContractPrecision::kF32
            : mlir::torch_tpu::ContractPrecision::kBF16;
    config.contract_precision_root_dot = config.contract_precision_qk_dot;

    TT_ASSIGN_OR_RETURN(std::string dkv_kernel_mlir,
                        mlir::torch_tpu::CreateBackwardDkvKernel(config));
    TT_ASSIGN_OR_RETURN(std::string dq_kernel_mlir,
                        mlir::torch_tpu::CreateBackwardDqKernel(config));

    mlir::MlirOp grad_out_batch =
        flatten_batch_dims(grad_out_mlir, batch_size, rank - 3);
    mlir::MlirOp query_batch =
        flatten_batch_dims(query_mlir, batch_size, rank - 3);
    mlir::MlirOp key_batch = flatten_batch_dims(key_mlir, batch_size, rank - 3);
    mlir::MlirOp value_batch =
        flatten_batch_dims(value_mlir, batch_size, rank - 3);

    if (is_mqa) {
      auto replicate = [&](mlir::MlirOp input, int64_t seq_len, int64_t h_dim) {
        auto broadcast_type = mlir::RankedTensorType::get(
            {batch_size, kv_num_heads, head_count_ratio, seq_len, h_dim},
            GetTensorTypeOrDie(input).getElementType());
        mlir::MlirOp broadcasted_input = mlir::stablehlo::BroadcastInDim(
            broadcast_type, input, {0, 1, 3, 4});
        return mlir::stablehlo::Reshape(
            broadcasted_input, {batch_size, num_heads, seq_len, h_dim});
      };
      key_batch = replicate(key_batch, kv_seq_len, head_dim);
      value_batch = replicate(value_batch, kv_seq_len, vo_head_dim);
    }

    // Reshape logsumexp to 4D [B_flat, N, 1, S]
    mlir::MlirOp logsumexp_4d =
        mlir::stablehlo::Reshape(logsumexp_mlir, aux_dims_4d);

    std::vector<mlir::Value> dkv_inputs = {
        query_batch.getValue(),  key_batch.getValue(),
        value_batch.getValue(),  grad_out_batch.getValue(),
        logsumexp_4d.getValue(), di_4d.getValue()};

    if (mask_mlir.isValid()) {
      dkv_inputs.push_back(mask_mlir.getValue());
    }

    auto dkv_custom_call = mlir::torch_tpu::CreateCustomCallOp(
        builder.getOpBuilder(), builder.getLoc(), dkv_kernel_mlir, dkv_inputs,
        {key_batch.getType(), value_batch.getType()});

    mlir::MlirOp out_batch = flatten_batch_dims(out_mlir, batch_size, rank - 3);

    std::vector<mlir::Value> dq_inputs = {
        query_batch.getValue(),  key_batch.getValue(),
        value_batch.getValue(),  grad_out_batch.getValue(),
        logsumexp_4d.getValue(), di_4d.getValue()};

    if (mask_mlir.isValid()) {
      dq_inputs.push_back(mask_mlir.getValue());
    }

    dq_inputs.push_back(out_batch.getValue());

    auto dq_custom_call = mlir::torch_tpu::CreateCustomCallOp(
        builder.getOpBuilder(), builder.getLoc(), dq_kernel_mlir, dq_inputs,
        {query_batch.getType()});

    mlir::MlirOp grad_key_batch(builder, dkv_custom_call.getResult(0));
    mlir::MlirOp grad_value_batch(builder, dkv_custom_call.getResult(1));
    mlir::MlirOp grad_query_batch(builder, dq_custom_call.getResult(0));

    if (is_mqa) {
      auto reduce_gradients =
          [&](mlir::MlirOp grad_batch, int64_t seq_len,
              int64_t h_dim) -> absl::StatusOr<mlir::MlirOp> {
        mlir::MlirOp reshaped_grad = mlir::stablehlo::Reshape(
            grad_batch,
            {batch_size, kv_num_heads, head_count_ratio, seq_len, h_dim});
        TT_ASSIGN_OR_RETURN(
            mlir::MlirOp reduced_grad,
            BuildSumShlo(reshaped_grad, {2}, ReductionMode::kDropDims));
        return reduced_grad;
      };
      TT_ASSIGN_OR_RETURN(
          grad_key_batch,
          reduce_gradients(grad_key_batch, kv_seq_len, head_dim));
      TT_ASSIGN_OR_RETURN(
          grad_value_batch,
          reduce_gradients(grad_value_batch, kv_seq_len, vo_head_dim));
    }

    mlir::MlirOp grad_query =
        unflatten_batch_dims(grad_query_batch, query_mlir);
    mlir::MlirOp grad_key = unflatten_batch_dims(grad_key_batch, key_mlir);
    mlir::MlirOp grad_value =
        unflatten_batch_dims(grad_value_batch, value_mlir);

    return {{grad_query, grad_key, grad_value}};
  };

  TT_ASSIGN_OR_RETURN(auto param_keys, *OpParamCacheKeysBuilder()
                                            .SetParam("is_causal", is_causal)
                                            .SetParam("scale", scale)
                                            .SetParam("attn_mask", attn_bias));

  std::vector<at::Tensor> dispatch_inputs = {grad_out, query,     key,
                                             value,    logsumexp, out};
  if (attn_bias.has_value() && attn_bias->defined()) {
    dispatch_inputs.push_back(*attn_bias);
  }

  TT_ASSIGN_OR_RETURN(
      auto results,
      (DispatchOp<kDynamicSize, 3>(
          std::move(op_builder), dispatch_inputs,
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

bool IsSupportedFlashAttentionShape(const at::Tensor& query,
                                    const at::Tensor& key,
                                    const at::Tensor& value) {
  // TODO(elliotenglish): Add support for attn_mask and attributes.
  constexpr int min_block_size = 128;
  constexpr int config_block_size = 512;
  constexpr int min_batch_size = 1;

  // TODO(b/483131156): Support all shapes.
  // We don't support all shapes because the kernel is specialized to
  // block_size 512 and we don't want to generate a new kernel for every
  // possible sequence length and head dim.
  bool has_batch = query.ndimension() >= 4 &&
                   get_batch_size(query.sizes()) >= min_batch_size;
  bool valid_seq_lens =
      query.size(query.ndimension() - 2) % config_block_size == 0 &&
      key.size(key.ndimension() - 2) % config_block_size == 0 &&
      value.size(value.ndimension() - 2) % config_block_size == 0;
  bool valid_head_dim =
      (query.size(query.ndimension() - 1) < min_block_size ||
       query.size(query.ndimension() - 1) % min_block_size == 0);

  return has_batch && valid_seq_lens && valid_head_dim;
}

}  // namespace

// Torch's extensibility is pretty gross here due to weird SDPA
// implementation.
// *  at::native::scaled_dot_product_attention calls _fused_sdp_choice_stub,
//    which then dispatches to either _fused_sdp_choice_cpp (for CPU) or
//    _fused_sdp_choice_cuda. This can be overridden by
//    REGISTER_PRIVATEUSE1_DISPATCH, which we do in tpu_aten_kernels.cc.
// *  However, these then use either sdp::select_sdp_backend_cpp (CPU) or
//    sdp::select_sdp_backend (CUDA). select_sdp_backend is *not* TORCH_API
//    overrideable, and these hardcode which backends are supported on
//    CPU and CUDA.
//
// So we have to inline the logic from sdp::select_sdp_backend* that accesses
// the at::globalContext() flags, and return either overrideable or math,
// depending on whether the arguments are supported by our current
// implementation.
//
// Also, to support easy migration for users, we only warn on an unsupported
// backend instead of erroring (we only error if *all* backends are disabled).
int64_t AtenFusedSdpChoice(const at::Tensor& query, const at::Tensor& key,
                           const at::Tensor& value,
                           const std::optional<at::Tensor>& attn_mask,
                           double dropout_p, bool is_causal,
                           std::optional<double> scale, bool enable_gqa) {
  TT_KERNEL(
      OpName::kFusedSdpChoice, _,
      (query, key, value, IgnoreInCacheKey(attn_mask, "Doesn't affect SHLO"),
       IgnoreInCacheKey(dropout_p, "Doesn't affect SHLO"),
       IgnoreInCacheKey(is_causal, "Unused"), IgnoreInCacheKey(scale, "Unused"),
       IgnoreInCacheKey(enable_gqa, "Unused")),
      {
        const auto& ctx = at::globalContext();
        bool flash_enabled = ctx.userEnabledFlashSDP();
        bool math_enabled = ctx.userEnabledMathSDP();
        bool overrideable_enabled = ctx.userEnabledOverrideableSDP();

        bool consistent_ranks = query.ndimension() == key.ndimension() &&
                                query.ndimension() == value.ndimension();
        bool has_batch = query.ndimension() >= 4;
        bool has_dropout = dropout_p != 0.0;

        if (flash_enabled && c10::get_privateuse1_backend() != "xla_cpu") {
          bool supported_flash_dtype =
              (query.scalar_type() == at::ScalarType::Float ||
               query.scalar_type() == at::ScalarType::BFloat16);
          if (supported_flash_dtype &&
              IsSupportedFlashAttentionShape(query, key, value) &&
              consistent_ranks && !has_dropout) {
            return static_cast<int64_t>(at::SDPBackend::flash_attention);
          } else {
            // Use c10::str rather than absl::StrCat as it will convert enums
            // into their string representation.
            TORCH_WARN_ONCE(
                "TorchTPU only supports FLASH_ATTENTION SDPBackend "
                "for scaled_dot_product_attention when these conditions are "
                "met:\n"
                "- attn_mask is None (current: ",
                (attn_mask.has_value() ? "present" : "None"),
                ")\n"
                "- dropout_p is 0.0 (current: ",
                dropout_p,
                ")\n"
                "- inputs are float32 or bfloat16 (current: ",
                query.scalar_type(),
                ")\n"
                "- inputs have the same rank (query: ",
                query.ndimension(), ", key: ", key.ndimension(),
                ", value: ", value.ndimension(),
                ")\n"
                "- query rank is at least 4 (current: ",
                query.ndimension(),
                ")\n"
                "- batch size is at least 1 (current: ",
                get_batch_size(query.sizes()),
                ")\n"
                "- Sequence lengths (dim - 2) are divisible by 512 (query: ",
                query.size(query.ndimension() - 2),
                ", key: ", key.size(key.ndimension() - 2),
                ", value: ", value.size(value.ndimension() - 2),
                ")\n"
                "- Head dimension (dim - 1) is less than 128 or divisible by "
                "128 "
                "(query: ",
                query.size(query.ndimension() - 1), ")");
          }
        }

        // We treat the SHLO implementation as an optimized version of MATH so
        // also try it when it is enabled.
        if (math_enabled || overrideable_enabled) {
          bool is_floating_type = c10::isFloatingType(query.scalar_type());
          if (consistent_ranks && has_batch && !has_dropout &&
              is_floating_type) {
            return static_cast<int64_t>(at::SDPBackend::overrideable);
          } else {
            TORCH_WARN_ONCE(
                "TorchTPU only supports SHLO optimized MATH SDPBackend when "
                "these conditions "
                "are met:\n"
                "- dropout_p is 0.0 (current: ",
                dropout_p,
                ")\n"
                "- inputs are not complex (current: ",
                query.scalar_type(),
                ")\n"
                "- inputs have the same rank (query: ",
                query.ndimension(), ", key: ", key.ndimension(),
                ", value: ", value.ndimension(), ")");
          }
        }

        TT_CHECK_THROW(math_enabled, error::kFailedPrecondition)
            << "no viable SDPBackend found: all supported backends are "
               "disabled, including the fallback MATH backend; enable at "
               "least one of FLASH, OVERRIDEABLE, or MATH for "
               "TorchTPU";
        return static_cast<int64_t>(at::SDPBackend::math);
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt,
           c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
GenerateResults(at::Tensor out, at::Tensor logsumexp) {
  int64_t batch = get_batch_size(out.sizes());
  at::Tensor cum_seq_q = at::zeros({batch + 1}, out.options().dtype(at::kInt));
  at::Tensor cum_seq_k = at::zeros({batch + 1}, out.options().dtype(at::kInt));
  at::Tensor philox_seed = at::zeros({1}, out.options().dtype(at::kLong));
  at::Tensor philox_offset = at::zeros({1}, out.options().dtype(at::kLong));
  at::Tensor debug_mask = at::zeros({1}, out.options().dtype(at::kBool));

  return std::make_tuple(std::move(out), std::move(logsumexp),
                         std::move(cum_seq_q), std::move(cum_seq_k),
                         c10::SymInt(0), c10::SymInt(0), std::move(philox_seed),
                         std::move(philox_offset), std::move(debug_mask));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt,
           c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFusedAttentionOverrideable(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias, double dropout_p,
    bool is_causal, bool return_debug_mask, std::optional<double> scale) {
  TT_KERNEL(
      OpName::kScaledDotProductFusedAttentionOverrideable, param_keys,
      (query, key, value,
       IgnoreInCacheKey(attn_bias, "Tensors are implicitly fingerprinted."),
       IgnoreInCacheKey(dropout_p, "Unused"), is_causal,
       IgnoreInCacheKey(return_debug_mask, "Unused"), scale),
      {
        const bool allow_half_precision_reduction_math =
            at::globalContext().allowFP16BF16ReductionMathSDP();
        TT_THROW_IF_ERROR(param_keys.SetParam(
            "allow_fp16_bf16_reduction", allow_half_precision_reduction_math));

        TT_ASSIGN_OR_THROW(
            auto results,
            ScaledDotProductFusedAttentionShlo(
                query, key, value, attn_bias, is_causal, scale,
                allow_half_precision_reduction_math, std::move(param_keys)));
        auto [out, logsumexp] = results;
        return GenerateResults(out, logsumexp);
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFusedAttentionOverrideableBackward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& attn_bias,
    std::array<bool, 4> grad_input_mask, const at::Tensor& out,
    const at::Tensor& logsumexp, const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k, at::SymInt max_q, at::SymInt max_k,
    double dropout_p, bool is_causal, const at::Tensor& philox_seed,
    const at::Tensor& philox_offset, std::optional<double> scale) {
  TT_KERNEL(
      OpName::kScaledDotProductFusedAttentionOverrideableBackward, param_keys,
      (grad_out, query, key, value, attn_bias,
       IgnoreInCacheKey(grad_input_mask, "Doesn't affect SHLO"), out, logsumexp,
       cum_seq_q, cum_seq_k, IgnoreInCacheKey(max_q, "Unused"),
       IgnoreInCacheKey(max_k, "Unused"), IgnoreInCacheKey(dropout_p, "Unused"),
       is_causal, philox_seed, philox_offset, scale),
      {
        const bool allow_half_precision_reduction_math =
            at::globalContext().allowFP16BF16ReductionMathSDP();
        TT_THROW_IF_ERROR(param_keys.SetParam(
            "allow_fp16_bf16_reduction", allow_half_precision_reduction_math));

        TT_ASSIGN_OR_THROW(
            auto out, ScaledDotProductFusedAttentionShloBackward(
                          grad_out, query, key, value, attn_bias, logsumexp,
                          scale, is_causal, allow_half_precision_reduction_math,
                          std::move(param_keys)));

        return std::make_tuple(
            grad_input_mask[0] ? std::get<0>(out)
                               : at::zeros({0}, grad_out.options()),
            grad_input_mask[1] ? std::get<1>(out)
                               : at::zeros({0}, grad_out.options()),
            grad_input_mask[2] ? std::get<2>(out)
                               : at::zeros({0}, grad_out.options()),
            at::zeros({0}, grad_out.options()));
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductEfficientAttention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias, bool compute_log_sumexp,
    double dropout_p, bool is_causal, std::optional<double> scale) {
  TT_KERNEL(OpName::kScaledDotProductEfficientAttention, _,
            (query, key, value, IgnoreInCacheKey(attn_bias, "Unused"),
             IgnoreInCacheKey(compute_log_sumexp, "Unused"),
             IgnoreInCacheKey(dropout_p, "Unused"),
             IgnoreInCacheKey(is_causal, "Delegates to implementation"),
             IgnoreInCacheKey(scale, "Delegates to implementation")),
            {
              // Unused arguments: attn_bias, compute_log_sumexp, dropout_p,
              // scale.
              TT_ASSIGN_OR_THROW(auto results,
                                 ScaledDotProductFusedAttentionInternal(
                                     query, key, value, attn_bias, is_causal,
                                     scale, compute_log_sumexp));
              auto [out, logsumexp] = results;
              at::Tensor philox_seed =
                  at::zeros({1}, query.options().dtype(at::kLong));
              at::Tensor philox_offset =
                  at::zeros({1}, query.options().dtype(at::kLong));

              return std::make_tuple(std::move(out), std::move(logsumexp),
                                     std::move(philox_seed),
                                     std::move(philox_offset));
            });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductEfficientAttentionBackward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& attn_bias, const at::Tensor& out,
    const at::Tensor& logsumexp, const at::Tensor& philox_seed,
    const at::Tensor& philox_offset, double dropout_p,
    std::array<bool, 4> grad_input_mask, bool is_causal,
    std::optional<double> scale) {
  TT_KERNEL(
      OpName::kScaledDotProductEfficientAttentionBackward, _,
      (grad_out, query, key, value, attn_bias, out, logsumexp, philox_seed,
       philox_offset, IgnoreInCacheKey(dropout_p, "Unused"),
       IgnoreInCacheKey(grad_input_mask, "Doesn't affect SHLO"),
       IgnoreInCacheKey(is_causal, "Delegates to implementation"),
       IgnoreInCacheKey(scale, "Delegates to implementation")),
      {
        TT_ASSIGN_OR_THROW(auto result,
                           ScaledDotProductFusedAttentionBackwardWithLseImpl(
                               grad_out, query, key, value, out, logsumexp,
                               attn_bias, scale, is_causal));

        const at::Tensor zero = AtenEfficientZeroTensor(
            {0}, grad_out.scalar_type(), /*layout_opt=*/c10::nullopt,
            grad_out.device(), /*pin_memory_opt=*/c10::nullopt);

        return std::make_tuple(grad_input_mask[0] ? std::get<0>(result) : zero,
                               grad_input_mask[1] ? std::get<1>(result) : zero,
                               grad_input_mask[2] ? std::get<2>(result) : zero,
                               zero);
      });
}

std::tuple<at::Tensor, at::Tensor> AtenScaledDotProductFlashAttention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    double dropout_p, bool is_causal,
    const std::optional<at::Tensor>& attn_mask, std::optional<double> scale) {
  TT_KERNEL(OpName::kScaledDotProductFlashAttention, _,
            (query, key, value, IgnoreInCacheKey(dropout_p, "Unused"),
             IgnoreInCacheKey(is_causal, "Delegates to implementation"),
             IgnoreInCacheKey(attn_mask, "Delegates to implementation"),
             IgnoreInCacheKey(scale, "Delegates to implementation")),
            {
              TT_ASSIGN_OR_THROW(
                  auto out,
                  ScaledDotProductFusedAttentionWithLseImpl(
                      query, key, value, attn_mask, is_causal, scale));
              return out;
            });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFlashAttentionBackward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& out, const at::Tensor& logsumexp,
    double dropout_p, bool is_causal,
    const std::optional<at::Tensor>& attn_mask, std::optional<double> scale) {
  TT_KERNEL(OpName::kScaledDotProductFlashAttentionBackward, _,
            (grad_out, query, key, value, out, logsumexp,
             IgnoreInCacheKey(dropout_p, "Unused"),
             IgnoreInCacheKey(is_causal, "Delegates to implementation"),
             IgnoreInCacheKey(attn_mask, "Unused"),
             IgnoreInCacheKey(scale, "Delegates to implementation")),
            {
              TT_ASSIGN_OR_THROW(
                  auto out, ScaledDotProductFusedAttentionBackwardWithLseImpl(
                                grad_out, query, key, value, out, logsumexp,
                                attn_mask, scale, is_causal));

              return out;
            });
}

}  // namespace torch_tpu
