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

#include "torch_tpu/csrc/ops/scaled_dot_product_attention/flash_attention_kernel_builder.h"

#include <algorithm>
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
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBase.h"
#include "ATen/core/grad_mode.h"
#include "ATen/ops/empty.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/internal/mosaic/op_builders.h"
#include "torch_tpu/csrc/ops/binary.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/reductions/reductions.h"
#include "torch_tpu/csrc/ops/reductions/sum.h"
#include "torch_tpu/csrc/ops/scaled_dot_product_attention/flash_attention_kernel.h"
#include "torch_tpu/csrc/ops/scaled_dot_product_attention/helpers.h"
#include "torch_tpu/csrc/ops/scaled_dot_product_attention/util.h"
#include "torch_tpu/csrc/ops/view_decomposition/contiguous_to_view.h"

namespace torch_tpu {

namespace {

bool IsDefined(const std::optional<at::Tensor>& tensor) {
  return tensor.has_value() && tensor->defined();
}

absl::Status CheckDtype(at::ScalarType dtype) {
  if (dtype == at::kFloat || dtype == at::kBFloat16) {
    return absl::OkStatus();
  }
  return TT_ERROR(error::kPythonNotImplementedError)
         << "unsupported dtype for sdpa custom kernel";
}

mlir::MlirOp PadSequenceDim(mlir::MlirOp input, int64_t seq_dim,
                            int64_t target_len) {
  auto input_type = GetTensorTypeOrDie(input);
  int64_t current_len = input_type.getShape()[seq_dim];
  if (current_len == target_len) {
    return input;
  }
  int64_t rank = input_type.getRank();
  Dimensions low_padding(rank, 0);
  Dimensions high_padding(rank, 0);
  high_padding[seq_dim] = target_len - current_len;
  Dimensions interior_padding(rank, 0);
  auto zero_pad_value =
      MakeScalarConstant(input.getBuilder(), 0.0, input_type.getElementType());
  return mlir::stablehlo::Pad(input, zero_pad_value, low_padding, high_padding,
                              interior_padding);
}

mlir::MlirOp SliceSequenceDim(mlir::MlirOp input, int64_t seq_dim,
                              int64_t target_len) {
  auto input_type = GetTensorTypeOrDie(input);
  int64_t current_len = input_type.getShape()[seq_dim];
  if (current_len == target_len) {
    return input;
  }
  int64_t rank = input_type.getRank();
  Dimensions start_indices(rank, 0);
  Dimensions limit_indices(input_type.getShape().begin(),
                           input_type.getShape().end());
  limit_indices[seq_dim] = target_len;
  Strides strides(rank, 1);
  return mlir::stablehlo::Slice(input, start_indices, limit_indices, strides);
}

// Pad the mask if the sequence lengths are not the same as the padded sequence
// lengths.
// Note: it can be padded with 0 as the bias will be combined with the
// structured bias within the kernel.
mlir::MlirOp PadBias(mlir::MlirOp bias, int64_t padded_q_len,
                     int64_t padded_kv_len) {
  auto bias_type = GetTensorTypeOrDie(bias);
  if (bias_type.getShape()[2] != 1) {
    bias = PadSequenceDim(bias, 2, padded_q_len);
  }
  if (bias_type.getShape()[3] != 1) {
    bias = PadSequenceDim(bias, 3, padded_kv_len);
  }
  return bias;
}

int64_t RoundUpToTileSize(int64_t seq_len, int64_t tile_size) {
  return llvm::divideCeil(seq_len, tile_size) * tile_size;
}

absl::StatusOr<mlir::torch_tpu::FlashAttnConfig> CreateFlashAttnConfig(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias, bool is_causal,
    std::optional<double> scale, bool return_lse,
    mlir::torch_tpu::Tiling tiling) {
  if (is_causal && IsDefined(attn_bias)) {
    return TT_ERROR(error::kInvalidArgument)
           << "Causal mask and attn bias are mutually exclusive.";
  }

  int rank = query.ndimension();
  int batch_size = 1;
  for (int i = 0; i < rank - 3; i++) {
    batch_size *= query.size(i);
  }
  Dimensions out_dims(query.sizes().begin(), query.sizes().end());
  out_dims[rank - 1] = value.size(rank - 1);
  const auto query_scalar_type = query.scalar_type();
  const int64_t head_dim = query.size(rank - 1);
  const int64_t num_heads = query.size(rank - 3);
  const int64_t kv_num_heads = key.size(rank - 3);
  const int64_t vo_head_dim = value.size(rank - 1);
  const int64_t q_seq_len = query.size(rank - 2);
  const int64_t kv_seq_len = key.size(rank - 2);

  if (num_heads % kv_num_heads != 0) {
    return TT_ERROR(error::kInvalidArgument)
           << "num_heads must be divisible by kv_num_heads for MQA/GQA";
  }

  mlir::torch_tpu::FlashAttnConfig config;
  config.return_lse = return_lse;
  TT_ASSIGN_OR_RETURN(config.element_type,
                      ConvertTo<mlir::ElementType>(query_scalar_type));
  config.batch_size = batch_size;
  config.num_heads = num_heads;
  config.kv_num_heads = kv_num_heads;
  config.qk_head_dim = head_dim;
  config.vo_head_dim = vo_head_dim;
  config.q_sequence_length = q_seq_len;
  config.kv_sequence_length = kv_seq_len;
  config.padded_q_sequence_length = RoundUpToTileSize(q_seq_len, tiling.qt);
  config.padded_kv_sequence_length = RoundUpToTileSize(kv_seq_len, tiling.kt);
  config.is_causal = is_causal;
  config.scale = scale.value_or(1.0 / std::sqrt(head_dim));
  config.has_attn_bias = IsDefined(attn_bias);

  if (config.has_attn_bias) {
    llvm::ArrayRef<int64_t> shape = attn_bias->sizes();
    llvm::SmallVector<int64_t, 4> flattened_shape;
    if (shape.size() > 4) {
      flattened_shape.push_back(batch_size);
      llvm::append_range(flattened_shape, shape.take_back(3));
    } else {
      flattened_shape = llvm::SmallVector<int64_t, 4>(4 - shape.size(), 1);
      llvm::append_range(flattened_shape, shape);
    }

    for (auto [idx, dim] : llvm::enumerate(flattened_shape)) {
      if (dim == 1) {
        config.mask_broadcast_dims.push_back(idx);
      }
    }
  }

  return config;
}

mlir::torch_tpu::Tiling CreateTiling(int64_t q_seq_len, int64_t k_seq_len) {
  constexpr int64_t kMinTileSize = 128;
  int64_t qt = std::min(mlir::torch_tpu::kDefaultQTileSize,
                        RoundUpToTileSize(q_seq_len, kMinTileSize));
  int64_t kt = std::min(mlir::torch_tpu::kDefaultKTileSize,
                        RoundUpToTileSize(k_seq_len, kMinTileSize));
  return {
      .qt = std::max(kMinTileSize, qt),
      .kt = std::max(kMinTileSize, kt),
  };
}

mlir::MlirOp ReshapeMask(mlir::MlirOp mask_mlir, int batch_size) {
  mlir::RankedTensorType mask_type = GetTensorTypeOrDie(mask_mlir);
  int rank = mask_type.getRank();
  if (rank > 4) {
    return flatten_batch_dims(mask_mlir, batch_size, rank - 3);
  } else if (rank < 4) {
    int prefix_dims = 4 - rank;
    mlir::SmallVector<int64_t> new_dims(prefix_dims, 1);
    new_dims.append(mask_type.getShape().begin(), mask_type.getShape().end());
    return mlir::stablehlo::Reshape(mask_mlir, new_dims);
  }
  return mask_mlir;
}

std::tuple<mlir::MlirOp, mlir::MlirOp> ReplicateKV(mlir::MlirOp key,
                                                   mlir::MlirOp value,
                                                   int num_heads,
                                                   int kv_num_heads) {
  if (kv_num_heads == num_heads) {
    return {key, value};
  }

  auto replicate = [&](mlir::MlirOp input) {
    mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
    llvm::ArrayRef<int64_t> input_shape = input_type.getShape();
    int64_t batch_size = input_shape[0];
    int64_t seq_len = input_shape[2];
    int64_t head_dim = input_shape[3];
    int64_t head_count_ratio = num_heads / kv_num_heads;
    auto broadcast_type = mlir::RankedTensorType::get(
        {batch_size, kv_num_heads, head_count_ratio, seq_len, head_dim},
        input_type.getElementType());
    mlir::MlirOp broadcasted_input =
        mlir::stablehlo::BroadcastInDim(broadcast_type, input, {0, 1, 3, 4});
    return mlir::stablehlo::Reshape(broadcasted_input,
                                    {batch_size, num_heads, seq_len, head_dim});
  };
  return {replicate(key), replicate(value)};
}

absl::StatusOr<std::tuple<mlir::MlirOp, mlir::MlirOp>> AccumulateKVGrads(
    mlir::MlirOp grad_key, mlir::MlirOp grad_value, int num_heads,
    int kv_num_heads) {
  if (kv_num_heads == num_heads) {
    return std::make_tuple(grad_key, grad_value);
  }

  auto reduce_gradients =
      [&](mlir::MlirOp grad) -> absl::StatusOr<mlir::MlirOp> {
    mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad);
    llvm::ArrayRef<int64_t> grad_shape = grad_type.getShape();
    int batch_size = grad_shape[0];
    int64_t seq_len = grad_shape[2];
    int64_t head_dim = grad_shape[3];
    int64_t head_count_ratio = num_heads / kv_num_heads;
    mlir::MlirOp reshaped_grad = mlir::stablehlo::Reshape(
        grad, {batch_size, kv_num_heads, head_count_ratio, seq_len, head_dim});
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp reduced_grad,
        BuildSumShlo(reshaped_grad, {2}, ReductionMode::kDropDims));
    return reduced_grad;
  };

  TT_ASSIGN_OR_RETURN(grad_key, reduce_gradients(grad_key));
  TT_ASSIGN_OR_RETURN(grad_value, reduce_gradients(grad_value));

  return std::make_tuple(grad_key, grad_value);
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor>>
CreateFlashAttentionKernelImpl(const at::Tensor& query, const at::Tensor& key,
                               const at::Tensor& value,
                               const std::optional<at::Tensor>& attn_bias,
                               bool is_causal, std::optional<double> scale,
                               bool return_lse) {
  TT_RETURN_IF_ERROR(CheckDtype(query.scalar_type()));

  int64_t q_seq_len = query.size(query.ndimension() - 2);
  int64_t k_seq_len = key.size(key.ndimension() - 2);
  const auto tiling = CreateTiling(q_seq_len, k_seq_len);

  TT_ASSIGN_OR_RETURN(
      auto config_init,
      CreateFlashAttnConfig(query, key, value, attn_bias, is_causal, scale,
                            return_lse, tiling));

  int rank = query.ndimension();
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(query.scalar_type()));

  Dimensions out_dims(query.sizes().begin(), query.sizes().end());
  out_dims[rank - 1] = value.size(rank - 1);

  auto op_builder =
      [rank, out_dims, config_init, tiling](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<DynamicMlirOpResults> {
    mlir::torch_tpu::FlashAttnConfig config = config_init;
    mlir::MlirOp query_mlir = inputs[0];
    mlir::MlirOp key_mlir = inputs[1];
    mlir::MlirOp value_mlir = inputs[2];
    mlir::MlirOp mask_mlir = inputs.size() == 4
                                 ? ReshapeMask(inputs[3], config.batch_size)
                                 : mlir::MlirOp();

    // Flatten batch dimensions of inputs.
    mlir::MlirOp query_4d =
        flatten_batch_dims(query_mlir, config.batch_size, rank - 3);
    mlir::MlirOp key_4d =
        flatten_batch_dims(key_mlir, config.batch_size, rank - 3);
    mlir::MlirOp value_4d =
        flatten_batch_dims(value_mlir, config.batch_size, rank - 3);

    std::tie(key_4d, value_4d) =
        ReplicateKV(key_4d, value_4d, config.num_heads, config.kv_num_heads);
    config.kv_num_heads = config.num_heads;

    query_4d = PadSequenceDim(query_4d, 2, config.padded_q_sequence_length);
    key_4d = PadSequenceDim(key_4d, 2, config.padded_kv_sequence_length);
    value_4d = PadSequenceDim(value_4d, 2, config.padded_kv_sequence_length);

    if (mask_mlir.isValid()) {
      mask_mlir = PadBias(mask_mlir, config.padded_q_sequence_length,
                          config.padded_kv_sequence_length);
    }

    Dimensions out_dims_4d = {config.batch_size, config.num_heads,
                              config.padded_q_sequence_length,
                              config.vo_head_dim};
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
    Dimensions lse_dims_4d = {config.batch_size, config.num_heads, 1,
                              config.padded_q_sequence_length};

    if (config.return_lse) {
      // LSE is 4D [B, N, 1, S]
      auto ml_type = mlir::RankedTensorType::get(
          lse_dims_4d, builder.getOpBuilder().getF32Type());
      result_types.push_back(ml_type);
    }

    auto context = mlir::torch_tpu::CreateMlirContextWithDialects();
    TT_ASSIGN_OR_RETURN(
        mlir::OwningOpRef<mlir::ModuleOp> kernel,
        mlir::torch_tpu::CreateKernel(context.get(), config, tiling));

    TT_ASSIGN_OR_RETURN(auto custom_call,
                        mlir::torch_tpu::CreateCustomCallOp(
                            builder.getOpBuilder(), builder.getLoc(),
                            std::move(kernel), operands, result_types));

    mlir::MlirOp out_padded(builder, custom_call.getResult(0));
    mlir::MlirOp out_sliced =
        SliceSequenceDim(out_padded, 2, config.q_sequence_length);

    DynamicMlirOpResults results;
    results.push_back(unflatten_batch_dims(out_sliced, out_dims));

    if (config.return_lse) {
      mlir::MlirOp lse_padded(builder, custom_call.getResult(1));
      mlir::MlirOp lse_sliced =
          SliceSequenceDim(lse_padded, 3, config.q_sequence_length);
      results.push_back(unflatten_batch_dims(lse_sliced, lse_dims));
    }

    return results;
  };

  TT_ASSIGN_OR_RETURN(auto param_keys,
                      *OpParamCacheKeysBuilder()
                           .SetParam("is_causal", is_causal)
                           .SetParam("scale", scale)
                           .SetParam("attn_bias", attn_bias)
                           .SetParam("return_lse", return_lse));

  std::vector<at::Tensor> dispatch_inputs = {query, key, value};

  if (IsDefined(attn_bias)) {
    dispatch_inputs.push_back(*attn_bias);
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

}  // namespace

absl::StatusOr<std::tuple<at::Tensor, at::Tensor>> CreateFlashAttentionKernel(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias, bool is_causal,
    std::optional<double> scale) {
  const bool return_lse =
      at::GradMode::is_enabled() &&
      (query.requires_grad() || key.requires_grad() || value.requires_grad());
  return CreateFlashAttentionKernelImpl(query, key, value, attn_bias, is_causal,
                                        scale, return_lse);
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
CreateFlashAttentionBackwardKernel(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& out, const at::Tensor& logsumexp,
    std::optional<at::Tensor> attn_bias, std::optional<double> scale,
    bool is_causal) {
  TT_RETURN_IF_ERROR(CheckDtype(query.scalar_type()));

  int64_t q_seq_len = query.size(query.ndimension() - 2);
  int64_t k_seq_len = key.size(key.ndimension() - 2);
  const auto tiling = CreateTiling(q_seq_len, k_seq_len);

  TT_ASSIGN_OR_RETURN(auto config_init,
                      CreateFlashAttnConfig(query, key, value, attn_bias,
                                            is_causal, scale, false, tiling));

  int rank = query.ndimension();

  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(query.scalar_type()));

  auto op_builder =
      [rank, config_init, tiling](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<MlirOpResults<3>> {
    mlir::torch_tpu::FlashAttnConfig config = config_init;

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
    Dimensions aux_dims_4d = {config.batch_size, config.num_heads, 1,
                              config.q_sequence_length};
    mlir::MlirOp di_4d = mlir::stablehlo::Reshape(di_shlo, aux_dims_4d);

    mlir::MlirOp mask_mlir = inputs.size() == 7
                                 ? ReshapeMask(inputs[6], config.batch_size)
                                 : mlir::MlirOp();

    mlir::MlirOp grad_out_batch =
        flatten_batch_dims(grad_out_mlir, config.batch_size, rank - 3);
    mlir::MlirOp query_batch =
        flatten_batch_dims(query_mlir, config.batch_size, rank - 3);
    mlir::MlirOp key_batch =
        flatten_batch_dims(key_mlir, config.batch_size, rank - 3);
    mlir::MlirOp value_batch =
        flatten_batch_dims(value_mlir, config.batch_size, rank - 3);

    int64_t original_kv_num_heads = config.kv_num_heads;
    std::tie(key_batch, value_batch) = ReplicateKV(
        key_batch, value_batch, config.num_heads, config.kv_num_heads);
    config.kv_num_heads = config.num_heads;

    // Reshape logsumexp to 4D [B_flat, N, 1, S]
    mlir::MlirOp logsumexp_4d =
        mlir::stablehlo::Reshape(logsumexp_mlir, aux_dims_4d);

    query_batch =
        PadSequenceDim(query_batch, 2, config.padded_q_sequence_length);
    grad_out_batch =
        PadSequenceDim(grad_out_batch, 2, config.padded_q_sequence_length);
    key_batch = PadSequenceDim(key_batch, 2, config.padded_kv_sequence_length);
    value_batch =
        PadSequenceDim(value_batch, 2, config.padded_kv_sequence_length);
    logsumexp_4d =
        PadSequenceDim(logsumexp_4d, 3, config.padded_q_sequence_length);
    di_4d = PadSequenceDim(di_4d, 3, config.padded_q_sequence_length);

    if (mask_mlir.isValid()) {
      mask_mlir = PadBias(mask_mlir, config.padded_q_sequence_length,
                          config.padded_kv_sequence_length);
    }

    std::vector<mlir::Value> dkv_inputs = {
        query_batch.getValue(),  key_batch.getValue(),
        value_batch.getValue(),  grad_out_batch.getValue(),
        logsumexp_4d.getValue(), di_4d.getValue()};

    if (mask_mlir.isValid()) {
      dkv_inputs.push_back(mask_mlir.getValue());
    }

    auto context = mlir::torch_tpu::CreateMlirContextWithDialects();

    TT_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> dkv_kernel,
                        mlir::torch_tpu::CreateBackwardDkvKernel(
                            context.get(), config, tiling));
    TT_ASSIGN_OR_RETURN(
        auto dkv_custom_call,
        mlir::torch_tpu::CreateCustomCallOp(
            builder.getOpBuilder(), builder.getLoc(), std::move(dkv_kernel),
            dkv_inputs, {key_batch.getType(), value_batch.getType()}));

    mlir::MlirOp out_batch =
        flatten_batch_dims(out_mlir, config.batch_size, rank - 3);

    out_batch = PadSequenceDim(out_batch, 2, config.padded_q_sequence_length);

    std::vector<mlir::Value> dq_inputs = {
        query_batch.getValue(),  key_batch.getValue(),
        value_batch.getValue(),  grad_out_batch.getValue(),
        logsumexp_4d.getValue(), di_4d.getValue()};

    if (mask_mlir.isValid()) {
      dq_inputs.push_back(mask_mlir.getValue());
    }

    dq_inputs.push_back(out_batch.getValue());

    TT_ASSIGN_OR_RETURN(
        mlir::OwningOpRef<mlir::ModuleOp> dq_kernel,
        mlir::torch_tpu::CreateBackwardDqKernel(context.get(), config, tiling));
    TT_ASSIGN_OR_RETURN(
        auto dq_custom_call,
        mlir::torch_tpu::CreateCustomCallOp(
            builder.getOpBuilder(), builder.getLoc(), std::move(dq_kernel),
            dq_inputs, {query_batch.getType()}));

    mlir::MlirOp grad_key_batch_padded(builder, dkv_custom_call.getResult(0));
    mlir::MlirOp grad_value_batch_padded(builder, dkv_custom_call.getResult(1));
    mlir::MlirOp grad_query_batch_padded(builder, dq_custom_call.getResult(0));

    mlir::MlirOp grad_key_batch =
        SliceSequenceDim(grad_key_batch_padded, 2, config.kv_sequence_length);
    mlir::MlirOp grad_value_batch =
        SliceSequenceDim(grad_value_batch_padded, 2, config.kv_sequence_length);
    mlir::MlirOp grad_query_batch =
        SliceSequenceDim(grad_query_batch_padded, 2, config.q_sequence_length);

    TT_ASSIGN_OR_RETURN(
        std::tie(grad_key_batch, grad_value_batch),
        AccumulateKVGrads(grad_key_batch, grad_value_batch, config.num_heads,
                          original_kv_num_heads));

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
                                            .SetParam("attn_bias", attn_bias));

  std::vector<at::Tensor> dispatch_inputs = {grad_out, query,     key,
                                             value,    logsumexp, out};
  if (IsDefined(attn_bias)) {
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

}  // namespace torch_tpu
