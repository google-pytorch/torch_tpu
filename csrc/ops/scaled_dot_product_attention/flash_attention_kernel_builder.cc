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

#include "csrc/ops/scaled_dot_product_attention/flash_attention_kernel_builder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/Context.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBase.h"
#include "ATen/core/grad_mode.h"
#include "ATen/ops/empty.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Optional.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/internal/device/vmem_capacity.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "csrc/internal/mosaic/kiln_custom_call.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/precision_context.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/rng_utils.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_config.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_kernel.h"
#include "csrc/ops/scaled_dot_product_attention/helpers.h"
#include "csrc/ops/view_decomposition/contiguous_to_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

int64_t GetFlashDeviceVmemLimitBytes() {
  return GetDeviceVmemCapacityBytes() / 2;
}

namespace {

// Scoped VMEM limit for the kernels: half the capacity, to avoid spilling.
int64_t GetVmemLimitBytes() { return GetFlashDeviceVmemLimitBytes(); }

// TPU MXU/VPU hardware lane width.
constexpr int64_t kMinHardwareTileSize = kFlashMinHardwareTileSize;
// Minimum block size produced by CreateTiling's loop.
constexpr int64_t kMinHeuristicTileSize = kFlashMinSplitTileSize;
// Lane width of the VMEM tiles, to which arrays pad their minor dim.
constexpr int64_t kLanes = 128;
constexpr int64_t kF32Bytes = 4;

int64_t RoundUpToTileSize(int64_t seq_len, int64_t tile_size) {
  return llvm::divideCeil(seq_len, tile_size) * tile_size;
}

int64_t GetInitialTileSize(int64_t seq_len) {
  if (seq_len <= kMinHardwareTileSize) {
    return kMinHardwareTileSize;
  }
  if (seq_len <= kMinHeuristicTileSize) {
    return kMinHeuristicTileSize;
  }
  return RoundUpToTileSize(seq_len, kMinHeuristicTileSize);
}

// Heuristic to determine the KV compute sub-tile size.
int64_t GetKvComputeTileSize(int64_t kt) {
  // Chosen such that the sub-tiles are not too big.
  int64_t target_kc = 1024;
  // If the KV tile size is smaller than the target, don't split it.
  if (kt < target_kc) {
    return kt;
  }
  while (target_kc >= kMinHeuristicTileSize) {
    if (kt % target_kc == 0) {
      return target_kc;
    }

    target_kc -= kMinHeuristicTileSize;
  }

  return kMinHardwareTileSize;
}

// Bytes of a [rows, cols] array in VMEM. Mosaic lays out VMEM arrays in tiles
// of 8 32-bit sublanes by 128 lanes, so the minor dim is padded to a multiple
// of 128 (a head dim of 64 takes as much VMEM as one of 128), and a single row
// (e.g. the LSE) to a whole tile.
int64_t VmemArrayBytes(int64_t rows, int64_t cols, int64_t elem_size) {
  const int64_t sublanes = 8 * kF32Bytes / elem_size;
  return RoundUpToTileSize(rows, sublanes) * RoundUpToTileSize(cols, kLanes) *
         elem_size;
}

// The parameters of a flash attention call, other than the tile sizes, that
// determine the VMEM footprint of its kernels.
struct TileVmemParams {
  int64_t qk_head_dim;
  int64_t vo_head_dim;
  // Bytes per element of q, k, v, the output and the mask.
  int64_t elem_size;
  bool has_mask = false;
  // Whether the mask is broadcast along the query / key sequence (e.g. a
  // [B, 1, 1, S] key padding mask), which makes its blocks a single row /
  // column (see BiasBlockShape).
  bool mask_q_broadcast = false;
  bool mask_kv_broadcast = false;
  bool has_dropout = false;
  bool is_backward = false;
};

int64_t MaskBlockBytes(const TileVmemParams& params, int64_t qt, int64_t kt) {
  if (!params.has_mask) {
    return 0;
  }
  return VmemArrayBytes(params.mask_q_broadcast ? 1 : qt,
                        params.mask_kv_broadcast ? 1 : kt, params.elem_size);
}

// The estimates below are upper bounds of the scoped VMEM that the flash
// attention kernels need for [qt, kt] tiles. They follow the kernel specs:
// Mosaic double buffers the blocks of the pipelined operands, allocates the
// scratch operands once, and spills the vector values that don't fit in vregs
// to VMEM. The allowances for the spilled values are calibrated against the
// scoped VMEM that XLA reports for the kernels on TPU v7.

int64_t EstimateForwardVmemBytes(const TileVmemParams& params, int64_t qt,
                                 int64_t kt) {
  const int64_t elem_size = params.elem_size;
  const int64_t d_qk = params.qk_head_dim;
  const int64_t d_vo = params.vo_head_dim;
  // Blocks of the pipelined q, k, v, mask, output and LSE operands.
  const int64_t block_bytes =
      VmemArrayBytes(qt, d_qk, elem_size) +
      VmemArrayBytes(kt, d_qk, elem_size) +
      VmemArrayBytes(kt, d_vo, elem_size) + MaskBlockBytes(params, qt, kt) +
      VmemArrayBytes(qt, d_vo, elem_size) + VmemArrayBytes(1, qt, kF32Bytes);
  // Scratch: the output accumulator and the lane-replicated row statistics m
  // and l.
  const int64_t scratch_bytes = VmemArrayBytes(qt, d_vo, kF32Bytes) +
                                2 * VmemArrayBytes(qt, kLanes, kF32Bytes);
  // Spilled values:
  //  * The q tile and the softmax state (m, l, acc), which are live across the
  //    KV sub-blocks, plus one more q-sized tile for the smaller per-step
  //    values (K / V tiles, PV, new row statistics).
  //  * One f32 logits-shaped tile, as P is computed in place of S. It is sized
  //    [qt, kt] rather than [qt, kc], as the scheduler overlaps the [qt, kc]
  //    sub-blocks of the unrolled KV loop. Dropout adds ~5 more for the
  //    per-element intermediates of its keep mask.
  //  * The f32 [qt, kc] mask sub-block (zero-padded on a ragged tail), which
  //    is live alongside S.
  int64_t value_bytes =
      2 * VmemArrayBytes(qt, d_qk, elem_size) + scratch_bytes +
      (params.has_dropout ? 6 : 1) * VmemArrayBytes(qt, kt, kF32Bytes);
  if (params.has_mask) {
    value_bytes += VmemArrayBytes(qt, GetKvComputeTileSize(kt), kF32Bytes);
  }
  return 2 * block_bytes + scratch_bytes + value_bytes;
}

// The q, k, v and dO tiles, which the backward kernels take as pipelined
// operands and also spill.
int64_t BackwardQkvdoBytes(const TileVmemParams& params, int64_t qt,
                           int64_t kt) {
  const int64_t elem_size = params.elem_size;
  return VmemArrayBytes(qt, params.qk_head_dim, elem_size) +
         VmemArrayBytes(kt, params.qk_head_dim, elem_size) +
         VmemArrayBytes(kt, params.vo_head_dim, elem_size) +
         VmemArrayBytes(qt, params.vo_head_dim, elem_size);
}

// The f32 [qt, kt] logits-shaped tiles that the backward kernels spill (S / P,
// dP and dS). Dropout adds the intermediates of its keep mask and of applying
// it to P and dP.
int64_t BackwardLogitsBytes(const TileVmemParams& params, int64_t qt,
                            int64_t kt) {
  return (params.has_dropout ? 12 : 3) * VmemArrayBytes(qt, kt, kF32Bytes);
}

// The dK / dV kernel (see CreateBackwardDkvKernel).
int64_t EstimateBackwardDkvVmemBytes(const TileVmemParams& params, int64_t qt,
                                     int64_t kt) {
  const int64_t elem_size = params.elem_size;
  const int64_t d_qk = params.qk_head_dim;
  const int64_t d_vo = params.vo_head_dim;
  const int64_t qkvdo_bytes = BackwardQkvdoBytes(params, qt, kt);
  // Blocks of the pipelined q, k, v, dO, LSE, di and mask operands, and of the
  // dK and dV outputs.
  const int64_t block_bytes =
      qkvdo_bytes + 2 * VmemArrayBytes(1, qt, kF32Bytes) +
      MaskBlockBytes(params, qt, kt) + VmemArrayBytes(kt, d_qk, elem_size) +
      VmemArrayBytes(kt, d_vo, elem_size);
  // Scratch: the f32 dK and dV accumulators.
  const int64_t scratch_bytes =
      VmemArrayBytes(kt, d_qk, kF32Bytes) + VmemArrayBytes(kt, d_vo, kF32Bytes);
  // Spilled values: q, k, v, dO, the loaded accumulators and the logits-shaped
  // tiles.
  const int64_t value_bytes =
      qkvdo_bytes + scratch_bytes + BackwardLogitsBytes(params, qt, kt);
  return 2 * block_bytes + scratch_bytes + value_bytes;
}

// The dQ kernel (see CreateBackwardDqKernel).
int64_t EstimateBackwardDqVmemBytes(const TileVmemParams& params, int64_t qt,
                                    int64_t kt) {
  const int64_t elem_size = params.elem_size;
  const int64_t d_qk = params.qk_head_dim;
  const int64_t d_vo = params.vo_head_dim;
  const int64_t qkvdo_bytes = BackwardQkvdoBytes(params, qt, kt);
  // Blocks of the pipelined q, k, v, dO, LSE, di, mask and (unused) forward
  // output operands, and of the dQ output.
  const int64_t block_bytes =
      qkvdo_bytes + 2 * VmemArrayBytes(1, qt, kF32Bytes) +
      MaskBlockBytes(params, qt, kt) + VmemArrayBytes(qt, d_vo, elem_size) +
      VmemArrayBytes(qt, d_qk, elem_size);
  // Scratch: the f32 dQ accumulator.
  const int64_t scratch_bytes = VmemArrayBytes(qt, d_qk, kF32Bytes);
  // Spilled values: q, k, v, dO, the loaded accumulator and the logits-shaped
  // tiles.
  const int64_t value_bytes =
      qkvdo_bytes + scratch_bytes + BackwardLogitsBytes(params, qt, kt);
  return 2 * block_bytes + scratch_bytes + value_bytes;
}

// The dK / dV and dQ kernels are compiled separately but share their tiles, so
// this is the larger of their estimates.
int64_t EstimateBackwardVmemBytes(const TileVmemParams& params, int64_t qt,
                                  int64_t kt) {
  return std::max(EstimateBackwardDkvVmemBytes(params, qt, kt),
                  EstimateBackwardDqVmemBytes(params, qt, kt));
}

int64_t EstimateTileVmemBytes(const TileVmemParams& params, int64_t qt,
                              int64_t kt) {
  return params.is_backward ? EstimateBackwardVmemBytes(params, qt, kt)
                            : EstimateForwardVmemBytes(params, qt, kt);
}

// Constraints on the shape of the [qt, kt] tiles.
enum class TileShape {
  kAny,
  // qt == kt. The causal forward kernel only skips the fully masked KV
  // sub-blocks of the tiles on the diagonal when they are square (see
  // EmitGridStep), which typically saves more than larger non-square tiles
  // gain.
  kSquare,
};

Tiling CreateTiling(int64_t q_seq_len, int64_t k_seq_len,
                    const TileVmemParams& params, TileShape shape,
                    std::optional<int64_t> vmem_limit_override = std::nullopt) {
  const int64_t vmem_limit = vmem_limit_override.value_or(GetVmemLimitBytes());

  int64_t qt = GetInitialTileSize(q_seq_len);
  int64_t kt = GetInitialTileSize(k_seq_len);
  // Only square the tiles of sequences that start with the same tile size, so
  // that e.g. a short query doesn't shrink the tiles of a long KV sequence.
  const bool square = shape == TileShape::kSquare && qt == kt;

  while (EstimateTileVmemBytes(params, qt, kt) > vmem_limit &&
         (qt > kMinHeuristicTileSize || kt > kMinHeuristicTileSize)) {
    if (qt > kt && qt > kMinHeuristicTileSize) {
      qt = std::max(kMinHeuristicTileSize,
                    RoundUpToTileSize(qt / 2, kMinHeuristicTileSize));
    } else if (kt > kMinHeuristicTileSize) {
      kt = std::max(kMinHeuristicTileSize,
                    RoundUpToTileSize(kt / 2, kMinHeuristicTileSize));
    } else if (qt > kMinHeuristicTileSize) {
      qt = std::max(kMinHeuristicTileSize,
                    RoundUpToTileSize(qt / 2, kMinHeuristicTileSize));
    }
  }
  if (square) {
    qt = kt = std::min(qt, kt);
  }

  return {
      .qt = qt,
      .kt = kt,
      .kc = GetKvComputeTileSize(kt),
  };
}

bool IsDefined(const std::optional<at::Tensor>& tensor) {
  return tensor.has_value() && tensor->defined();
}

absl::Status ValidateDtype(at::ScalarType dtype) {
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
  Dimensions limit_indices = CopyIntVector(input_type.getShape());
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

Tiling CreateTiling(mlir::MlirOp query, mlir::MlirOp key, mlir::MlirOp value,
                    const std::optional<mlir::MlirOp>& attn_bias,
                    bool is_backward, bool has_dropout, TileShape shape) {
  const auto query_type = GetTensorTypeOrDie(query);
  const auto key_type = GetTensorTypeOrDie(key);
  const auto value_type = GetTensorTypeOrDie(value);
  const int64_t q_seq_len = query_type.getShape()[query_type.getRank() - 2];
  const int64_t k_seq_len = key_type.getShape()[key_type.getRank() - 2];
  TileVmemParams params{
      .qk_head_dim = query_type.getShape()[query_type.getRank() - 1],
      .vo_head_dim = value_type.getShape()[value_type.getRank() - 1],
      .elem_size = query_type.getElementType().isF32() ? 4 : 2,
      .has_mask = attn_bias.has_value(),
      .has_dropout = has_dropout,
      .is_backward = is_backward,
  };
  if (attn_bias.has_value()) {
    // As in CreateFlashAttnConfig, size-1 dims of the mask are broadcast.
    const llvm::ArrayRef<int64_t> mask_shape =
        GetTensorTypeOrDie(*attn_bias).getShape();
    const int64_t mask_rank = mask_shape.size();
    params.mask_q_broadcast = mask_rank < 2 || mask_shape[mask_rank - 2] == 1;
    params.mask_kv_broadcast = mask_rank < 1 || mask_shape[mask_rank - 1] == 1;
  }
  return CreateTiling(q_seq_len, k_seq_len, params, shape);
}

// Unpacks a 2-element i64 tensor `[seed, offset]` into a 4-element i32 tensor
// `[seed_lo, offset_lo, seed_hi, offset_hi]`.
mlir::MlirOp PackPhiloxPairTo4xI32(mlir::MlirOp pair_i64) {
  mlir::MlirBuilder& builder = pair_i64.getBuilder();
  mlir::Type i64_type = builder.getOpBuilder().getI64Type();
  mlir::Type i32_type = builder.getOpBuilder().getI32Type();

  mlir::MlirOp shift_32 =
      MakeConstant(builder, static_cast<int64_t>(32), i64_type, {2});
  mlir::MlirOp lo_i32 = mlir::stablehlo::ConvertElementType(pair_i64, i32_type);
  mlir::MlirOp hi_i32 = mlir::stablehlo::ConvertElementType(
      mlir::stablehlo::ShiftRightLogical(pair_i64, shift_32), i32_type);
  return mlir::stablehlo::Concatenate(builder, {lo_i32, hi_i32},
                                      /*dimension=*/0);
}

// Packs `philox_seed` and `philox_offset` into a 4-element i32 tensor. The
// inputs may be 0-d (e.g. from the torch.compile meta function) or have shape
// [1] (from the eager forward kernel).
mlir::MlirOp PackPhiloxSeedAndOffsetTo4xI32(mlir::MlirOp philox_seed,
                                            mlir::MlirOp philox_offset) {
  return PackPhiloxPairTo4xI32(mlir::stablehlo::Concatenate(
      philox_seed.getBuilder(),
      {mlir::stablehlo::Reshape(philox_seed, {1}),
       mlir::stablehlo::Reshape(philox_offset, {1})},
      /*dimension=*/0));
}

kiln::DType ToKilnDType(mlir::ElementType element_type) {
  switch (element_type) {
    case mlir::ElementType::PRED:
      return kiln::bool_;
    case mlir::ElementType::I8:
      return kiln::i8;
    case mlir::ElementType::I16:
      return kiln::i16;
    case mlir::ElementType::I32:
      return kiln::i32;
    case mlir::ElementType::I64:
      return kiln::i64;
    case mlir::ElementType::F16:
      return kiln::f16;
    case mlir::ElementType::BF16:
      return kiln::bf16;
    case mlir::ElementType::F32:
      return kiln::f32;
    default:
      ABSL_CHECK(false) << "Unsupported element type for Kiln";  // CRASH_OK
  }
}

absl::StatusOr<FlashAttnConfig> CreateFlashAttnConfig(
    mlir::MlirOp query, mlir::MlirOp key, mlir::MlirOp value,
    const std::optional<mlir::MlirOp>& attn_bias, bool is_causal,
    std::optional<double> scale, double dropout_p, bool return_lse,
    const Tiling& tiling, mlir::stablehlo::Precision precision) {
  if (is_causal && attn_bias.has_value()) {
    return TT_ERROR(error::kInvalidArgument)
           << "Causal mask and attn bias are mutually exclusive.";
  }

  auto query_type = GetTensorTypeOrDie(query);
  auto key_type = GetTensorTypeOrDie(key);
  auto value_type = GetTensorTypeOrDie(value);

  int rank = query_type.getRank();
  int batch_size = 1;
  auto query_shape = query_type.getShape();
  for (int i = 0; i < rank - 3; ++i) {
    batch_size *= query_shape[i];
  }

  const int64_t head_dim = query_shape[rank - 1];
  const int64_t num_heads = query_shape[rank - 3];
  auto key_shape = key_type.getShape();
  const int64_t kv_num_heads = key_shape[key_type.getRank() - 3];
  auto value_shape = value_type.getShape();
  const int64_t vo_head_dim = value_shape[value_type.getRank() - 1];
  const int64_t q_seq_len = query_shape[rank - 2];
  const int64_t kv_seq_len = key_shape[key_type.getRank() - 2];

  if (num_heads % kv_num_heads != 0) {
    return TT_ERROR(error::kInvalidArgument)
           << "num_heads must be divisible by kv_num_heads for MQA/GQA";
  }

  FlashAttnConfig config;
  config.return_lse = return_lse;
  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      GetElementType(query));
  config.element_type = ToKilnDType(element_type);
  if (config.element_type == kiln::f32) {
    if (precision ==
        mlir::stablehlo::Precision::HIGHEST) {  // EXPLICIT_PRECISION_OK=sdpa
      config.precision = kiln::ContractPrecision::kFP32;
    } else if (precision == mlir::stablehlo::Precision::
                                HIGH) {  // EXPLICIT_PRECISION_OK=sdpa
      config.precision = kiln::ContractPrecision::kBF16x3;
    }
  }
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
  config.dropout_rate = static_cast<float>(dropout_p);
  config.has_attn_bias = attn_bias.has_value();

  if (config.has_attn_bias) {
    llvm::ArrayRef<int64_t> shape = GetTensorTypeOrDie(*attn_bias).getShape();
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

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>
CreateFlashAttentionKernelImpl(const at::Tensor& query, const at::Tensor& key,
                               const at::Tensor& value,
                               const std::optional<at::Tensor>& attn_bias,
                               bool is_causal, std::optional<double> scale,
                               double dropout_p, bool return_lse) {
  TT_RETURN_IF_ERROR(ValidateDtype(query.scalar_type()));

  int rank = query.ndimension();
  const bool has_attn_bias = IsDefined(attn_bias);
  const bool has_dropout = dropout_p > 0.0;

  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(query.scalar_type()));

  Dimensions out_dims = CopyIntVector(query.sizes());
  out_dims[rank - 1] = value.size(rank - 1);
  Dimensions lse_dims = CopyIntVector(query.sizes().slice(0, rank - 1));
  lse_dims.push_back(1);

  TT_ASSIGN_OR_RETURN(auto param_keys,
                      *OpParamCacheKeysBuilder()
                           .SetParam("is_causal", is_causal)
                           .SetParam("scale", scale)
                           .SetParam("dropout_p", dropout_p)
                           .SetParam("attn_bias", attn_bias)
                           .SetParam("return_lse", return_lse));
  const mlir::stablehlo::Precision precision = GetAndAddPrecisionTo(param_keys);

  auto op_builder =
      [rank, out_dims, lse_dims, is_causal, scale, dropout_p, return_lse,
       has_attn_bias, has_dropout, precision](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<DynamicMlirOpResults> {
    size_t idx = 0;
    mlir::MlirOp query_mlir = inputs[idx++];
    mlir::MlirOp key_mlir = inputs[idx++];
    mlir::MlirOp value_mlir = inputs[idx++];
    std::optional<mlir::MlirOp> attn_bias_mlir;
    if (has_attn_bias) {
      attn_bias_mlir = inputs[idx++];
    }
    std::optional<mlir::MlirOp> rng_state_mlir;
    if (has_dropout) {
      rng_state_mlir = inputs[idx++];
    }

    const auto tiling =
        CreateTiling(query_mlir, key_mlir, value_mlir, attn_bias_mlir,
                     /*is_backward=*/false, has_dropout,
                     is_causal ? TileShape::kSquare : TileShape::kAny);
    TT_ASSIGN_OR_RETURN(
        FlashAttnConfig config,
        CreateFlashAttnConfig(query_mlir, key_mlir, value_mlir, attn_bias_mlir,
                              is_causal, scale, dropout_p, return_lse, tiling,
                              precision));

    mlir::MlirOp mask_mlir =
        attn_bias_mlir.has_value()
            ? ReshapeMask(*attn_bias_mlir, config.batch_size)
            : mlir::MlirOp();

    // The sequences are not padded to a multiple of the tile sizes: the kernel
    // handles the out-of-bounds tail of the last tile itself.
    mlir::MlirOp query_4d =
        flatten_batch_dims(query_mlir, config.batch_size, rank - 3);
    mlir::MlirOp key_4d =
        flatten_batch_dims(key_mlir, config.batch_size, rank - 3);
    mlir::MlirOp value_4d =
        flatten_batch_dims(value_mlir, config.batch_size, rank - 3);

    Dimensions out_dims_4d = {config.batch_size, config.num_heads,
                              config.q_sequence_length, config.vo_head_dim};
    auto out_type = mlir::RankedTensorType::get(
        out_dims_4d, GetTensorTypeOrDie(query_mlir).getElementType());

    mlir::SmallVector<mlir::Value, 5> operands;
    std::optional<mlir::MlirOp> philox_seed_mlir;
    std::optional<mlir::MlirOp> philox_offset_mlir;
    if (has_dropout) {
      mlir::Type i64_type = builder.getOpBuilder().getI64Type();
      mlir::MlirOp state_i64 =
          mlir::stablehlo::ConvertElementType(*rng_state_mlir, i64_type);
      philox_seed_mlir = mlir::stablehlo::Slice(state_i64, {0}, {1}, {1});
      philox_offset_mlir = mlir::stablehlo::Slice(state_i64, {1}, {2}, {1});
      operands.push_back(PackPhiloxPairTo4xI32(state_i64).getValue());
    }
    operands.push_back(query_4d.getValue());
    operands.push_back(key_4d.getValue());
    operands.push_back(value_4d.getValue());
    if (mask_mlir.isValid()) {
      operands.push_back(mask_mlir.getValue());
    }

    mlir::SmallVector<mlir::Type, 2> result_types = {out_type};
    Dimensions lse_dims_4d = {config.batch_size, config.num_heads, 1,
                              config.q_sequence_length};

    if (config.return_lse) {
      // LSE is 4D [B, N, 1, S]
      auto ml_type = mlir::RankedTensorType::get(
          lse_dims_4d, builder.getOpBuilder().getF32Type());
      result_types.push_back(ml_type);
    }

    TT_ASSIGN_OR_RETURN(kiln::Kernel kernel, CreateKernel(config, tiling));

    TT_ASSIGN_OR_RETURN(
        auto custom_call,
        CreateKilnCustomCallOp(builder.getOpBuilder(), builder.getLoc(),
                               std::move(kernel), operands, result_types,
                               GetVmemLimitBytes()));

    DynamicMlirOpResults results;
    results.push_back(unflatten_batch_dims(
        mlir::MlirOp(builder, custom_call.getResult(0)), out_dims));

    if (config.return_lse) {
      mlir::MlirOp lse(builder, custom_call.getResult(1));
      // The cuddnn implementation requires that the LSE is [B, N, S, 1]
      // TODO(willfroom): fold this into the kernel.
      mlir::MlirOp lse_transposed =
          mlir::stablehlo::Transpose(lse, {0, 1, 3, 2});
      results.push_back(unflatten_batch_dims(lse_transposed, lse_dims));
    }

    if (has_dropout) {
      results.push_back(*philox_seed_mlir);
      results.push_back(*philox_offset_mlir);
    }

    return results;
  };

  std::vector<at::Tensor> dispatch_inputs = {query, key, value};

  if (has_attn_bias) {
    dispatch_inputs.push_back(*attn_bias);
  }

  const Dimensions seed_dims = {1};
  std::vector<mlir::ElementType> out_dtypes = {out_dtype};
  std::vector<absl::Span<const int64_t>> out_dims_list = {out_dims};
  if (return_lse) {
    out_dtypes.push_back(mlir::ElementType::F32);
    out_dims_list.push_back(lse_dims);
  }
  if (has_dropout) {
    out_dtypes.push_back(mlir::ElementType::I64);
    out_dims_list.push_back(seed_dims);
    out_dtypes.push_back(mlir::ElementType::I64);
    out_dims_list.push_back(seed_dims);
  }

  auto dispatch = [&]() {  // REF_CAPTURE_OK=called locally
    return DispatchOp<kDynamicSize, kDynamicSize>(
        std::move(op_builder), dispatch_inputs,
        {.out_dtypes = out_dtypes,
         .out_dims_list = out_dims_list,
         .op_param_cache_keys = std::move(param_keys)});
  };

  std::vector<DeviceBufferRef> results;
  if (has_dropout) {
    TT_ASSIGN_OR_RETURN(results, DispatchRngOpGeneral(
                                     /*generator=*/c10::nullopt,
                                     [&]  // REF_CAPTURE_OK=called synchronously
                                     (const at::Tensor& state) {
                                       dispatch_inputs.push_back(state);
                                       return dispatch();
                                     }));
  } else {
    TT_ASSIGN_OR_RETURN(results, dispatch());
  }

  // If LSE or dropout return tensors are necessary, construct them **before**
  // calling ContiguousToView. This ensures that if ContiguousToView
  // materializes the stream, the result DeviceBufferRefs are not dropped.
  // TODO(bawilson): this is clunky. A better way is to only do background
  // materializations after returning from this function, immediately before
  // returning to Python, so that the Tensor state has stabilized.
  size_t res_idx = 1;
  at::Tensor lse_tensor =
      return_lse ? MakeTensor(std::move(results[res_idx++]))
                 : at::empty(lse_dims, query.options().dtype(at::kFloat));
  at::Tensor philox_seed =
      has_dropout ? MakeTensor(std::move(results[res_idx++])) : at::Tensor();
  at::Tensor philox_offset =
      has_dropout ? MakeTensor(std::move(results[res_idx++])) : at::Tensor();

  // Return the output with the strides of the meta function, so that views
  // traced by torch.compile (e.g. merging heads) are also valid in eager mode.
  TT_ASSIGN_OR_RETURN(
      at::Tensor strided_result,
      ContiguousToView(results[0], MatchingLayoutStrides(query, out_dims),
                       /*target_storage_offset=*/0));

  return std::make_tuple(std::move(strided_result), std::move(lse_tensor),
                         std::move(philox_seed), std::move(philox_offset));
}

}  // namespace

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>
CreateFlashAttentionKernel(const at::Tensor& query, const at::Tensor& key,
                           const at::Tensor& value,
                           const std::optional<at::Tensor>& attn_bias,
                           bool is_causal, std::optional<double> scale,
                           double dropout_p, std::optional<bool> return_lse) {
  const bool should_return_lse = return_lse.value_or(
      at::GradMode::is_enabled() &&
      (query.requires_grad() || key.requires_grad() || value.requires_grad()));
  return CreateFlashAttentionKernelImpl(query, key, value, attn_bias, is_causal,
                                        scale, dropout_p, should_return_lse);
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
CreateFlashAttentionBackwardKernel(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& out, const at::Tensor& logsumexp,
    const std::optional<at::Tensor>& attn_bias, std::optional<double> scale,
    bool is_causal, double dropout_p,
    const std::optional<at::Tensor>& philox_seed,
    const std::optional<at::Tensor>& philox_offset) {
  TT_RETURN_IF_ERROR(ValidateDtype(query.scalar_type()));

  int rank = query.ndimension();
  const bool has_attn_bias = IsDefined(attn_bias);
  const bool has_dropout = dropout_p > 0.0;

  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(query.scalar_type()));

  TT_ASSIGN_OR_RETURN(auto param_keys, *OpParamCacheKeysBuilder()
                                            .SetParam("is_causal", is_causal)
                                            .SetParam("scale", scale)
                                            .SetParam("dropout_p", dropout_p)
                                            .SetParam("attn_bias", attn_bias));
  const mlir::stablehlo::Precision precision = GetAndAddPrecisionTo(param_keys);

  auto op_builder =
      [rank, is_causal, scale, dropout_p, has_attn_bias, has_dropout,
       precision](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<MlirOpResults<3>> {
    size_t idx = 0;
    mlir::MlirOp grad_out_mlir = inputs[idx++];
    mlir::MlirOp query_mlir = inputs[idx++];
    mlir::MlirOp key_mlir = inputs[idx++];
    mlir::MlirOp value_mlir = inputs[idx++];
    mlir::MlirOp logsumexp_mlir = inputs[idx++];
    mlir::MlirOp out_mlir = inputs[idx++];
    std::optional<mlir::MlirOp> attn_bias_mlir;
    if (has_attn_bias) {
      attn_bias_mlir = inputs[idx++];
    }
    std::optional<mlir::MlirOp> philox_seed_mlir;
    std::optional<mlir::MlirOp> philox_offset_mlir;
    if (has_dropout) {
      philox_seed_mlir = inputs[idx++];
      philox_offset_mlir = inputs[idx++];
    }

    const auto tiling =
        CreateTiling(query_mlir, key_mlir, value_mlir, attn_bias_mlir,
                     /*is_backward=*/true, has_dropout, TileShape::kAny);
    TT_ASSIGN_OR_RETURN(
        FlashAttnConfig config,
        CreateFlashAttnConfig(query_mlir, key_mlir, value_mlir, attn_bias_mlir,
                              is_causal, scale, dropout_p,
                              /*return_lse=*/false, tiling, precision));

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

    mlir::MlirOp mask_mlir =
        attn_bias_mlir.has_value()
            ? ReshapeMask(*attn_bias_mlir, config.batch_size)
            : mlir::MlirOp();

    mlir::MlirOp grad_out_batch =
        flatten_batch_dims(grad_out_mlir, config.batch_size, rank - 3);
    mlir::MlirOp query_batch =
        flatten_batch_dims(query_mlir, config.batch_size, rank - 3);
    mlir::MlirOp key_batch =
        flatten_batch_dims(key_mlir, config.batch_size, rank - 3);
    mlir::MlirOp value_batch =
        flatten_batch_dims(value_mlir, config.batch_size, rank - 3);

    // Reshape logsumexp to 4D [B_flat, N, S, 1]
    mlir::MlirOp logsumexp_4d =
        flatten_batch_dims(logsumexp_mlir, config.batch_size, rank - 3);
    // TODO(willfroom): fold this into the kernel.
    mlir::MlirOp logsumexp_transposed =
        mlir::stablehlo::Transpose(logsumexp_4d, {0, 1, 3, 2});

    query_batch =
        PadSequenceDim(query_batch, 2, config.padded_q_sequence_length);
    grad_out_batch =
        PadSequenceDim(grad_out_batch, 2, config.padded_q_sequence_length);
    key_batch = PadSequenceDim(key_batch, 2, config.padded_kv_sequence_length);
    value_batch =
        PadSequenceDim(value_batch, 2, config.padded_kv_sequence_length);
    logsumexp_transposed = PadSequenceDim(logsumexp_transposed, 3,
                                          config.padded_q_sequence_length);
    di_4d = PadSequenceDim(di_4d, 3, config.padded_q_sequence_length);

    if (mask_mlir.isValid()) {
      mask_mlir = PadBias(mask_mlir, config.padded_q_sequence_length,
                          config.padded_kv_sequence_length);
    }

    std::optional<mlir::Value> seed_4xi32_val;
    if (has_dropout) {
      seed_4xi32_val =
          PackPhiloxSeedAndOffsetTo4xI32(*philox_seed_mlir, *philox_offset_mlir)
              .getValue();
    }

    std::vector<mlir::Value> dkv_inputs;
    if (seed_4xi32_val.has_value()) {
      dkv_inputs.push_back(*seed_4xi32_val);
    }
    dkv_inputs.insert(dkv_inputs.end(),
                      {query_batch.getValue(), key_batch.getValue(),
                       value_batch.getValue(), grad_out_batch.getValue(),
                       logsumexp_transposed.getValue(), di_4d.getValue()});

    if (mask_mlir.isValid()) {
      dkv_inputs.push_back(mask_mlir.getValue());
    }

    TT_ASSIGN_OR_RETURN(kiln::Kernel dkv_kernel,
                        CreateBackwardDkvKernel(config, tiling));
    mlir::RankedTensorType key_type = GetTensorTypeOrDie(key_batch);
    mlir::RankedTensorType value_type = GetTensorTypeOrDie(value_batch);
    TT_ASSIGN_OR_RETURN(
        auto dkv_custom_call,
        CreateKilnCustomCallOp(builder.getOpBuilder(), builder.getLoc(),
                               std::move(dkv_kernel), dkv_inputs,
                               {key_type, value_type}, GetVmemLimitBytes()));

    mlir::MlirOp out_batch =
        flatten_batch_dims(out_mlir, config.batch_size, rank - 3);

    out_batch = PadSequenceDim(out_batch, 2, config.padded_q_sequence_length);

    std::vector<mlir::Value> dq_inputs;
    if (seed_4xi32_val.has_value()) {
      dq_inputs.push_back(*seed_4xi32_val);
    }
    dq_inputs.insert(dq_inputs.end(),
                     {query_batch.getValue(), key_batch.getValue(),
                      value_batch.getValue(), grad_out_batch.getValue(),
                      logsumexp_transposed.getValue(), di_4d.getValue()});

    if (mask_mlir.isValid()) {
      dq_inputs.push_back(mask_mlir.getValue());
    }

    dq_inputs.push_back(out_batch.getValue());

    TT_ASSIGN_OR_RETURN(kiln::Kernel dq_kernel,
                        CreateBackwardDqKernel(config, tiling));
    TT_ASSIGN_OR_RETURN(
        auto dq_custom_call,
        CreateKilnCustomCallOp(builder.getOpBuilder(), builder.getLoc(),
                               std::move(dq_kernel), dq_inputs,
                               {query_batch.getType()}, GetVmemLimitBytes()));

    mlir::MlirOp grad_key_batch_padded(builder, dkv_custom_call.getResult(0));
    mlir::MlirOp grad_value_batch_padded(builder, dkv_custom_call.getResult(1));
    mlir::MlirOp grad_query_batch_padded(builder, dq_custom_call.getResult(0));

    mlir::MlirOp grad_key_batch =
        SliceSequenceDim(grad_key_batch_padded, 2, config.kv_sequence_length);
    mlir::MlirOp grad_value_batch =
        SliceSequenceDim(grad_value_batch_padded, 2, config.kv_sequence_length);
    mlir::MlirOp grad_query_batch =
        SliceSequenceDim(grad_query_batch_padded, 2, config.q_sequence_length);

    mlir::MlirOp grad_query =
        unflatten_batch_dims(grad_query_batch, query_mlir);
    mlir::MlirOp grad_key = unflatten_batch_dims(grad_key_batch, key_mlir);
    mlir::MlirOp grad_value =
        unflatten_batch_dims(grad_value_batch, value_mlir);

    return {{grad_query, grad_key, grad_value}};
  };

  std::vector<at::Tensor> dispatch_inputs = {grad_out, query,     key,
                                             value,    logsumexp, out};
  if (has_attn_bias) {
    dispatch_inputs.push_back(*attn_bias);
  }
  if (has_dropout) {
    dispatch_inputs.push_back(*philox_seed);
    dispatch_inputs.push_back(*philox_offset);
  }

  TT_ASSIGN_OR_RETURN(
      auto results,
      (DispatchOp<kDynamicSize, 3>(
          std::move(op_builder), dispatch_inputs,
          {.out_dtypes = {out_dtype, out_dtype, out_dtype},
           .out_dims_list = {query.sizes(), key.sizes(), value.sizes()},
           .op_param_cache_keys = std::move(param_keys)})));

  // Artificially create a simulated DataPtr to prevent the result node from
  // being flushed during the first or second ContiguousToView if there is an
  // early materialization.
  // TODO(bawilson): this is clunky. A better way is to only do background
  // materializations after returning from this function, immediately before
  // returning to Python, so that the Tensor state has stabilized.
  results[0].device_buffer_list()->IncrementLiveDataPtrs(results[0].index());

  // Return the gradients with the strides of the meta function, which
  // allocates them with `torch.empty_like`.
  TT_ASSIGN_OR_RETURN(at::Tensor strided_grad_query,
                      ContiguousToView(results[0], EmptyLikeStrides(query),
                                       /*target_storage_offset=*/0));
  TT_ASSIGN_OR_RETURN(at::Tensor strided_grad_key,
                      ContiguousToView(results[1], EmptyLikeStrides(key),
                                       /*target_storage_offset=*/0));

  // It's okay if the materialization happens during the third ContiguousToView;
  // all other necessary tensors have already been created, so nothing can be
  // invalidated.
  results[0].device_buffer_list()->DecrementLiveDataPtrs(results[0].index());

  TT_ASSIGN_OR_RETURN(at::Tensor strided_grad_value,
                      ContiguousToView(results[2], EmptyLikeStrides(value),
                                       /*target_storage_offset=*/0));
  return std::make_tuple(strided_grad_query, strided_grad_key,
                         strided_grad_value);
}

}  // namespace torch_tpu
