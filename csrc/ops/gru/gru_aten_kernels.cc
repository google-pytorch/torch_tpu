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

#include "csrc/ops/gru/gru_aten_kernels.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/GradMode.h"
#include "c10/util/ArrayRef.h"  // IWYU pragma: keep for at::IntArrayRef
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/gru/gru_common.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "csrc/ops/rng_utils.h"
#include "csrc/ops/uniform/uniform.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/autograd/custom_function.h"

// =============================================================================
// Fused StableHLO GRU (aten::gru.input) Kernel Implementation for Google TPU
// =============================================================================
//
// 1. MATHEMATICAL FORMULATION:
//    For each timestep t in sequence length T and batch element b in B:
//      r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})   (Reset
//      Gate) z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz})
//      (Update Gate) n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn}
//      h_{t-1} + b_{hn})) (New/Candidate) h_t = (1 - z_t) \odot n_t + z_t \odot
//      h_{t-1}                (Hidden State)
//
// 2. PRE-ACTIVATION TENSOR LAYOUT:
//    Input projection W_ih and recurrent projection W_hh have row dimension
//    3*hidden:
//      [0 : hidden]         -> Reset gate (r)
//      [hidden : 2*hidden]   -> Update gate (z)
//      [2*hidden : 3*hidden] -> Candidate hidden state (n)
//
// 3. KEY HARDWARE ACCELERATION OPTIMIZATIONS FOR TPU:
//    a) Upfront Batched 3D GEMM Input Projection:
//       The full sequence input projection X @ W_ih^T is computed upfront via
//       MixedPrecisionDotGeneral as a single batched systolic array matrix
//       multiplication. This transforms O(T) fragmented multiplications into 1
//       large GEMM, saturating MXU.
//
//    b) Chunked Recurrence Unrolling (kGruUnrollFactor = 8):
//       Static loop unrolling of 8 timesteps per while-loop step keeps active
//       hidden states and gate pre-activations resident in TPU vector registers
//       (VMEM), eliminating while-loop branch evaluation and dynamic dispatch
//       overhead.
//
//    c) Concurrent Bidirectional Loop Fusion:
//       Forward (t = 0..T-1) and reverse (t = T-1..0) recurrent steps are
//       evaluated simultaneously in parallel within the same chunk body,
//       maximizing VPU/MXU arithmetic density and halving loop dispatch
//       latency.
//
//    d) Inter-Layer Pipelined Wavefronts (L >= 2):
//       For stacked multi-layer unidirectional GRUs, chunks of k=8 outputs
//       stream directly between layer l and layer l+1 in vector registers,
//       eliminating intermediate [T, B, H] sequence tensor HBM roundtrips.
//
//    e) Analytical Autograd Backward Graph (Reverse BPTT):
//       Computes analytical gradients directly via closed-form gate adjoint
//       algebra without materializing intermediate autodiff graphs, and streams
//       weight gradient accumulation per chunk to avoid full [T, B, 3*H]
//       adjoint tensor materialization.
// =============================================================================

namespace torch_tpu {
namespace {

// The recurrence unroll factor k = 8 governs the granularity of chunked loop
// execution across both forward inference and reverse BPTT backward passes.
//
// Architectural Rationale for k = 8 on Google TPU Accelerators:
// 1. Vector Register Residency & VMEM Capacity:
//    TPU Vector Processing Units (VPUs) operate on on-chip vector register
//    files (VMEM). Unrolling 8 consecutive timesteps statically within a single
//    dynamic stablehlo::WhileOp iteration allows active intermediate
//    activations (gate pre- activations, reset/update/candidate gates, and next
//    hidden states) to remain in vector registers without spilling to off-chip
//    High Bandwidth Memory (HBM).
// 2. Loop Dispatch & Control Flow Amortization:
//    Dynamic WhileOp executions incur host-device scheduling and control-flow
//    branch evaluation latency. A factor of k = 8 reduces the dynamic loop trip
//    count by 8x (e.g., from T = 128 to 16 iterations), drastically diminishing
//    control-flow overhead and instruction cache churn.
// 3. Systolic Array (MXU) Tile Saturation:
//    When combined with batch GEMM operations, chunking 8 timesteps together
//    provides sufficient row dimension depth (k * batch = 8 * B) to saturate
//    the TPU Matrix Multiplication Unit (MXU) sublanes without fragmentation.
// 4. Optimal Register Pressure Trade-Off:
//    Empirical benchmarking shows that k < 8 incurs excessive while-loop
//    overhead, whereas k > 8 increases register pressure and risks spilling
//    live vector registers to VMEM/HBM for wide hidden state configurations (H
//    >= 2048).
constexpr int64_t kGruUnrollFactor = 8;

// Validates the shape and dimensionality of the initial hidden state tensor
// `hx`.
//
// What it computes / checks:
// - Verifies that `hx` is exactly 3-dimensional [num_layers * num_directions,
// batch, hidden].
// - Confirms that dimension 0 matches num_layers * num_directions (where
// num_directions
//   is 2 for bidirectional GRUs and 1 for unidirectional GRUs).
// - Confirms that dimension 1 matches runtime batch size `batch`.
//
// Parameters:
// - `hx`: Initial hidden state tensor for all layers and directions.
// - `num_layers`: Number of stacked recurrent layers (>= 1).
// - `batch`: Runtime batch size.
// - `bidirectional`: True for bidirectional GRUs, false for unidirectional.
//
// Returns:
// - `absl::OkStatus()` if all dimensions match expected values; otherwise an
// InvalidArgument status.
absl::Status ValidateGruTensorShapes(const at::Tensor& hx,
                                     const int64_t num_layers,
                                     const int64_t batch,
                                     const bool bidirectional = false) {
  const int64_t num_directions = bidirectional ? 2 : 1;
  // Step 1: Verify rank is 3D [num_layers * num_directions, batch, hidden]
  TT_RET_CHECK(hx.dim() == 3, error::kInvalidArgument)
      << "expected hx to be a 3D tensor [num_layers * num_directions, batch, "
         "hidden_size], got "
      << hx.dim() << "D";
  // Step 2: Verify layer/direction dimension matches num_layers *
  // num_directions
  TT_RET_CHECK(hx.size(0) == num_layers * num_directions,
               error::kInvalidArgument)
      << "expected hx size(0) to match num_layers * num_directions ("
      << num_layers * num_directions << "), got " << hx.size(0);
  // Step 3: Verify batch size matches the input batch dimension
  TT_RET_CHECK(hx.size(1) == batch, error::kInvalidArgument)
      << "expected hx size(1) to match batch (" << batch << "), got "
      << hx.size(1);
  return absl::OkStatus();
}

// Validates the count and tensor dimensions of all GRU weight and bias
// parameters.
//
// What it computes / checks:
// - Parameter count: expects num_layers * num_directions * (has_biases ? 4 : 2)
// tensors.
// - For each layer l in [0, num_layers - 1] and direction dir in [0,
// num_directions - 1]:
//   - Input-hidden weight `w_ih`: shape [3 * hidden, in_dim], where in_dim is
//   `input_size`
//     for layer 0 and `hidden * num_directions` for stacked upper layers (l >
//     0).
//   - Hidden-hidden recurrent weight `w_hh`: shape [3 * hidden, hidden].
//   - If biases are enabled:
//     - Input bias `b_ih`: shape [3 * hidden].
//     - Recurrent bias `b_hh`: shape [3 * hidden].
//
// Parameters:
// - `params`: Flattened tensor list containing weight matrices and bias
// vectors.
// - `has_biases`: True if bias vectors (b_ih, b_hh) are present.
// - `num_layers`: Number of stacked recurrent layers.
// - `input_size`: Feature size of layer 0 input sequence.
// - `hidden`: Hidden state feature dimension.
// - `bidirectional`: True if bidirectional GRU.
//
// Returns:
// - `absl::OkStatus()` on success, or an InvalidArgument status describing any
// mismatched shape.
absl::Status ValidateGruWeights(const at::TensorList params,
                                const bool has_biases, const int64_t num_layers,
                                const int64_t input_size, const int64_t hidden,
                                const bool bidirectional = false) {
  const int64_t num_directions = bidirectional ? 2 : 1;
  const size_t params_per_direction = has_biases ? 4 : 2;
  const size_t params_per_layer = params_per_direction * num_directions;
  // Verify total parameter list length
  TT_RET_CHECK(
      params.size() == static_cast<size_t>(num_layers * params_per_layer),
      error::kInvalidArgument)
      << "expected " << num_layers * params_per_layer << " parameters, got "
      << params.size();

  // Validate matrix and bias dimensions layer-by-layer and
  // direction-by-direction
  for (int64_t l = 0; l < num_layers; ++l) {
    const int64_t in_dim = (l == 0) ? input_size : hidden * num_directions;
    for (int64_t d_dir = 0; d_dir < num_directions; ++d_dir) {
      const size_t p_offset =
          (l * num_directions + d_dir) * params_per_direction;
      // w_ih shape check: [3 * hidden, in_dim]
      TT_RET_CHECK(
          params[p_offset].sizes() == at::IntArrayRef({3 * hidden, in_dim}),
          error::kInvalidArgument)
          << "w_ih layer " << l << " dir " << d_dir << " expected shape ["
          << 3 * hidden << ", " << in_dim << "], got "
          << ToString(params[p_offset].sizes());
      // w_hh shape check: [3 * hidden, hidden]
      TT_RET_CHECK(
          params[p_offset + 1].sizes() == at::IntArrayRef({3 * hidden, hidden}),
          error::kInvalidArgument)
          << "w_hh layer " << l << " dir " << d_dir << " expected shape ["
          << 3 * hidden << ", " << hidden << "], got "
          << ToString(params[p_offset + 1].sizes());
      if (has_biases) {
        // b_ih shape check: [3 * hidden]
        TT_RET_CHECK(
            params[p_offset + 2].sizes() == at::IntArrayRef({3 * hidden}),
            error::kInvalidArgument)
            << "b_ih layer " << l << " dir " << d_dir << " expected shape ["
            << 3 * hidden << "], got "
            << ToString(params[p_offset + 2].sizes());
        // b_hh shape check: [3 * hidden]
        TT_RET_CHECK(
            params[p_offset + 3].sizes() == at::IntArrayRef({3 * hidden}),
            error::kInvalidArgument)
            << "b_hh layer " << l << " dir " << d_dir << " expected shape ["
            << 3 * hidden << "], got "
            << ToString(params[p_offset + 3].sizes());
      }
    }
  }
  return absl::OkStatus();
}

// Top-level input validator verifying user-supplied input tensors and
// hyperparameters.
//
// What it computes / checks:
// - Confirms `input` is 3D: [seq_len, batch, input_size] or [batch, seq_len,
// input_size].
// - Verifies sequence length, batch size, and layer count are strictly positive
// (> 0).
// - Verifies dropout probability falls within the unit interval [0.0, 1.0].
// - Validates `hx` shapes via `ValidateGruTensorShapes`.
// - Validates weight matrix and bias dimensions via `ValidateGruWeights`.
//
// Parameters:
// - `input`: Input sequence tensor.
// - `hx`: Initial hidden state tensor.
// - `params`: Weight and bias parameter list.
// - `has_biases`: Whether bias vectors are enabled.
// - `num_layers`: Number of stacked recurrent layers.
// - `dropout`: Dropout probability applied between stacked layers.
// - `train`: True for training mode, false for inference evaluation.
// - `bidirectional`: True for bidirectional GRUs.
// - `batch_first`: Memory layout indicator (true for [batch, seq, feat]).
//
// Returns:
// - `absl::OkStatus()` if valid, or an InvalidArgument status.
absl::Status ValidateGruInputs(const at::Tensor& input, const at::Tensor& hx,
                               const at::TensorList params,
                               const bool has_biases, const int64_t num_layers,
                               const double dropout, const bool train,
                               const bool bidirectional,
                               const bool batch_first) {
  // Check input tensor dimensionality
  TT_RET_CHECK(input.dim() == 3, error::kInvalidArgument)
      << "expected input to be a 3D tensor, got " << input.dim() << "D";

  const int64_t batch = batch_first ? input.size(0) : input.size(1);
  const int64_t seq_len = batch_first ? input.size(1) : input.size(0);
  const int64_t input_size = input.size(2);

  // Validate positive dimensions
  TT_RET_CHECK(seq_len > 0, error::kInvalidArgument)
      << "expected sequence length to be larger than 0 in RNN, got " << seq_len;
  TT_RET_CHECK(batch > 0, error::kInvalidArgument)
      << "expected batch size > 0 in RNN, got " << batch;
  TT_RET_CHECK(num_layers > 0, error::kInvalidArgument)
      << "expected num_layers > 0 in RNN, got " << num_layers;
  TT_RET_CHECK(dropout >= 0.0 && dropout <= 1.0, error::kInvalidArgument)
      << "expected dropout to be in range [0, 1], got " << dropout;

  // Validate initial hidden state tensor shapes
  TT_RETURN_IF_ERROR(
      ValidateGruTensorShapes(hx, num_layers, batch, bidirectional));
  const int64_t hidden = hx.size(2);
  // Validate weight and bias parameter tensor shapes
  TT_RETURN_IF_ERROR(ValidateGruWeights(params, has_biases, num_layers,
                                        input_size, hidden, bidirectional));

  return absl::OkStatus();
}

// Constructs DotDimensionNumbersAttr for StableHLO DotGeneral matrix
// multiplication.
//
// What it computes:
// - Creates a DotDimensionNumbersAttr specifying contracting dimension indices
// between
//   LHS and RHS operands without batch dimensions, configuring tensor
//   contraction on TPU MXU.
//
// Parameters:
// - `ctx`: MLIR context pointer.
// - `lhs_contracting`: Contracting dimension index/indices for the left-hand
// side tensor.
// - `rhs_contracting`: Contracting dimension index/indices for the right-hand
// side tensor.
//
// Returns:
// - DotDimensionNumbersAttr defining the matrix multiplication contraction
// dimensions.
mlir::stablehlo::DotDimensionNumbersAttr MakeDotDims(
    mlir::MLIRContext* ctx, llvm::ArrayRef<int64_t> lhs_contracting,
    llvm::ArrayRef<int64_t> rhs_contracting) {
  return mlir::stablehlo::DotDimensionNumbersAttr::get(
      ctx, /*lhs_batching_dimensions=*/{}, /*rhs_batching_dimensions=*/{},
      lhs_contracting, rhs_contracting);
}

// Concatenates a span of StableHLO tensor operations along the specified
// dimension.
//
// What it computes:
// - If the span has size 1, returns the lone operation directly to avoid
// redundant ConcatenateOp IR nodes.
// - Otherwise, emits a stablehlo::Concatenate operation joining all tensors
// along `dim`.
//
// Parameters:
// - `builder`: MLIR builder for op emission.
// - `ops`: Span of input tensor operations to concatenate.
// - `dim`: Axis along which concatenation is performed.
//
// Returns:
// - Concatenated tensor operation.
mlir::MlirOp ConcatDim(mlir::MlirBuilder& builder,
                       absl::Span<const mlir::MlirOp> ops, int64_t dim) {
  return (ops.size() == 1) ? ops[0]
                           : mlir::stablehlo::Concatenate(builder, ops, dim);
}

// Extracts a single layer's 2D hidden state from a stacked 3D hidden state
// tensor.
//
// What it computes:
// - Slices a 1-layer slice [1, batch, hidden] at index `layer_idx` from
// `tensor_3d` [L, B, H].
// - Reshapes the 3D slice to 2D [batch, hidden] for direct GEMM and vector
// operations.
//
// Parameters:
// - `tensor_3d`: 3D stacked hidden state tensor [num_layers * num_directions,
// batch, hidden].
// - `layer_idx`: 0-based index of the layer/direction to extract.
// - `batch`: Batch dimension size.
// - `hidden`: Hidden state feature dimension.
//
// Returns:
// - 2D hidden state tensor [batch, hidden].
mlir::MlirOp ExtractLayer2D(mlir::MlirOp tensor_3d, int64_t layer_idx,
                            int64_t batch, int64_t hidden) {
  mlir::MlirOp sl = mlir::stablehlo::Slice(
      tensor_3d, {layer_idx, 0, 0}, {layer_idx + 1, batch, hidden}, {1, 1, 1});
  return mlir::stablehlo::Reshape(sl, {batch, hidden});
}

// Extracts a single timestep slice from a 3D sequence tensor as a 2D tensor.
//
// What it computes:
// - When `batch_first` is true: slices [batch, 1, feature_dim] at time `t`
// along dim 1.
// - When `batch_first` is false: slices [1, batch, feature_dim] at time `t`
// along dim 0.
// - Reshapes the extracted 3D slice to 2D [batch, feature_dim].
//
// Parameters:
// - `seq_3d`: 3D sequence tensor.
// - `t`: Timestep index to slice.
// - `batch`: Batch size.
// - `feature_dim`: Feature size (e.g. 3*hidden or hidden).
// - `batch_first`: Memory layout indicator.
//
// Returns:
// - 2D slice [batch, feature_dim] for timestep `t`.
inline mlir::MlirOp SliceStep2D(mlir::MlirOp seq_3d, int64_t t, int64_t batch,
                                int64_t feature_dim, bool batch_first) {
  mlir::MlirOp slice_3d =
      batch_first
          ? mlir::stablehlo::Slice(seq_3d, {0, t, 0},
                                   {batch, t + 1, feature_dim}, {1, 1, 1})
          : mlir::stablehlo::Slice(seq_3d, {t, 0, 0},
                                   {t + 1, batch, feature_dim}, {1, 1, 1});
  return mlir::stablehlo::Reshape(slice_3d, {batch, feature_dim});
}

// Reshapes a 2D timestep tensor [batch, feature_dim] into a 3D sequence tensor
// of length 1.
//
// What it computes:
// - When `batch_first` is true: reshapes to [batch, 1, feature_dim].
// - When `batch_first` is false: reshapes to [1, batch, feature_dim].
//
// Parameters:
// - `step_2d`: 2D tensor of shape [batch, feature_dim].
// - `batch`: Batch size.
// - `feature_dim`: Feature size.
// - `batch_first`: Memory layout indicator.
//
// Returns:
// - 3D single-timestep tensor suitable for concatenation along the sequence
// axis.
inline mlir::MlirOp ExpandStep3D(mlir::MlirOp step_2d, int64_t batch,
                                 int64_t feature_dim, bool batch_first) {
  return batch_first
             ? mlir::stablehlo::Reshape(step_2d, {batch, 1, feature_dim})
             : mlir::stablehlo::Reshape(step_2d, {1, batch, feature_dim});
}

// Converts an MLIR Type into the corresponding mlir::ElementType enum.
//
// What it computes:
// - Maps MLIR primitive types (BF16, F32, F16, F64) to the corresponding
// mlir::ElementType enum.
//
// Parameters:
// - `type`: The MLIR type to inspect.
//
// Returns:
// - std::optional containing the ElementType enum if recognised, or
// std::nullopt.
inline std::optional<mlir::ElementType> GetElementTypeEnum(mlir::Type type) {
  if (type.isBF16()) return mlir::ElementType::BF16;
  if (type.isF32()) return mlir::ElementType::F32;
  if (type.isF16()) return mlir::ElementType::F16;
  if (type.isF64()) return mlir::ElementType::F64;
  return std::nullopt;
}

// Emits a StableHLO DotGeneral matrix multiplication on TPU with
// mixed-precision support.
//
// What it computes:
// - Determines the target execution dtype for matrix multiplication on the TPU
// MXU systolic array.
// - Converts LHS and RHS operands to `target_dtype` if they differ.
// - Executes DotGeneral with accumulator type `acc_dtype` (typically F32 on TPU
// to prevent numerical underflow/overflow).
//
// Parameters:
// - `lhs`: Left-hand side tensor operation.
// - `rhs`: Right-hand side tensor operation.
// - `dot_dims`: Contracting and batching dimension mappings.
// - `acc_dtype`: Accumulation data type (e.g. F32).
// - `gemm_dtype`: Optional compute type override for operands.
// - `precision`: Hardware precision level (DEFAULT or HIGH).
//
// Returns:
// - Resulting matrix multiplication tensor operation.
inline mlir::MlirOp MixedPrecisionDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    mlir::stablehlo::DotDimensionNumbersAttr dot_dims, mlir::Type acc_dtype,
    std::optional<mlir::Type> gemm_dtype,
    mlir::stablehlo::Precision precision) {
  mlir::Type target_dtype;
  if (gemm_dtype.has_value()) {
    target_dtype = *gemm_dtype;
  } else {
    const mlir::Type lhs_dtype = GetTensorTypeOrDie(lhs).getElementType();
    const mlir::Type rhs_dtype = GetTensorTypeOrDie(rhs).getElementType();
    if (lhs_dtype != acc_dtype) {
      target_dtype = lhs_dtype;
    } else if (rhs_dtype != acc_dtype) {
      target_dtype = rhs_dtype;
    } else {
      target_dtype = acc_dtype;
    }
  }

  mlir::MlirOp lhs_op =
      (GetTensorTypeOrDie(lhs).getElementType() == target_dtype)
          ? lhs
          : mlir::stablehlo::ConvertElementType(lhs, target_dtype);
  mlir::MlirOp rhs_op =
      (GetTensorTypeOrDie(rhs).getElementType() == target_dtype)
          ? rhs
          : mlir::stablehlo::ConvertElementType(rhs, target_dtype);

  mlir::MLIRContext& ctx = lhs_op.getContext();
  mlir::stablehlo::Precision precisions[2] = {precision, precision};
  auto precision_config =
      mlir::stablehlo::PrecisionConfigAttr::get(&ctx, precisions);

  return mlir::stablehlo::DotGeneral(lhs_op, rhs_op, dot_dims, precision_config,
                                     GetElementTypeEnum(acc_dtype));
}

// MixedPrecisionDotGeneral overload accepting ElementType enum with optional
// gemm_dtype.
inline mlir::MlirOp MixedPrecisionDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    mlir::stablehlo::DotDimensionNumbersAttr dot_dims,
    mlir::ElementType acc_dtype, std::optional<mlir::Type> gemm_dtype,
    mlir::stablehlo::Precision precision) {
  return MixedPrecisionDotGeneral(
      lhs, rhs, dot_dims, mlir::getElementType(lhs.getContext(), acc_dtype),
      gemm_dtype, precision);
}

// MixedPrecisionDotGeneral overload accepting mlir::Type with default nullopt
// gemm_dtype.
inline mlir::MlirOp MixedPrecisionDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    mlir::stablehlo::DotDimensionNumbersAttr dot_dims, mlir::Type acc_dtype,
    mlir::stablehlo::Precision precision) {
  return MixedPrecisionDotGeneral(lhs, rhs, dot_dims, acc_dtype,
                                  /*gemm_dtype=*/std::nullopt, precision);
}

// MixedPrecisionDotGeneral overload accepting ElementType enum with default
// nullopt gemm_dtype.
inline mlir::MlirOp MixedPrecisionDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    mlir::stablehlo::DotDimensionNumbersAttr dot_dims,
    mlir::ElementType acc_dtype, mlir::stablehlo::Precision precision) {
  return MixedPrecisionDotGeneral(lhs, rhs, dot_dims, acc_dtype,
                                  /*gemm_dtype=*/std::nullopt, precision);
}

// Container holding the intermediate and final outputs of a single GRU forward
// recurrent step.
//
// Fields:
// - `h_next`: Updated hidden state h_t [B, hidden] computed via convex
// combination: (1 - z) * n + z * h_curr.
// - `resetgate`: Reset gate activation r_t [B, hidden] in (0, 1).
// - `updategate`: Update gate activation z_t [B, hidden] in (0, 1).
// - `newgate`: Candidate hidden state n_t [B, hidden] in (-1, 1).
// - `h_n`: Recurrent contribution to candidate pre-activation [B, hidden]:
// (W_hn * h_curr + b_hn).
// - `act_4h`: Packed 4H activations [rz (2*H), newgate (H), h_n (H)] [B,
// 4*hidden] cached for autograd.
struct GruStepResults {
  mlir::MlirOp h_next;
  mlir::MlirOp resetgate;
  mlir::MlirOp updategate;
  mlir::MlirOp newgate;
  mlir::MlirOp h_n;
  mlir::MlirOp act_4h;
};

// Evaluates a single recurrent timestep of the GRU cell forward pass on TPU
// VPU/MXU.
//
// What it computes:
// 1. Recurrent projection GEMM:
//      h_proj = h_curr @ W_hh^T + b_hh  [B, 3*hidden]
// 2. Gate pre-activation slicing:
//      i_rz = x_proj_step[:, 0:2*H],  i_n = x_proj_step[:, 2*H:3*H]
//      h_rz = h_proj[:, 0:2*H],       h_n = h_proj[:, 2*H:3*H]
// 3. Vectorized reset and update gate evaluation:
//      pre_rz = i_rz + h_rz
//      rz = sigmoid(pre_rz)  -> resetgate r_t = rz[:, 0:H], updategate z_t =
//      rz[:, H:2*H]
// 4. Candidate hidden state evaluation:
//      r_hn = r_t * h_n
//      pre_n = i_n + r_hn
//      newgate n_t = tanh(pre_n)
// 5. Convex combination hidden state update:
//      h_next = (h_curr - n_t) * z_t + n_t = (1 - z_t) * n_t + z_t * h_curr
// 6. Packed 4H activations for autograd backward caching:
//      act_4h = concat([rz, newgate, h_n], dim=1)  [B, 4*hidden]
//
// Parameters:
// - `x_proj_step`: Pre-projected input for timestep t: (W_ih * x_t + b_ih) [B,
// 3*hidden].
// - `h_curr`: Previous hidden state h_{t-1} [B, hidden].
// - `w_hh`: Recurrent weight matrix [3*hidden, hidden].
// - `hh_dot_dims`: Contracting dimensions for recurrent GEMM ({1}, {1}).
// - `batch`: Runtime batch size B.
// - `hidden`: Hidden dimension size H.
// - `b_hh`: Optional recurrent bias vector [3*hidden].
// - `precision`: Hardware precision level for matrix multiplication.
//
// Returns:
// - `GruStepResults` struct containing h_next, gates, and packed 4H
// activations.
GruStepResults ComputeGruStep(
    mlir::MlirOp x_proj_step, mlir::MlirOp h_curr, mlir::MlirOp w_hh,
    mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims, int64_t batch,
    int64_t hidden, std::optional<mlir::MlirOp> b_hh,
    mlir::stablehlo::Precision precision) {
  const mlir::Type acc_dtype = GetTensorTypeOrDie(h_curr).getElementType();
  // Step 1: Evaluate recurrent hidden-to-hidden projection GEMM:
  // h_proj: [B, hidden] @ [3*H, hidden]^T -> [B, 3*H]
  mlir::MlirOp h_proj =
      MixedPrecisionDotGeneral(h_curr, w_hh, hh_dot_dims, acc_dtype, precision);
  if (b_hh.has_value()) {
    mlir::MlirOp b_bcast = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(h_proj), *b_hh, {/*broadcast_dimensions=*/1});
    h_proj = mlir::stablehlo::Add(h_proj, b_bcast);
  }

  // Step 2: Slice pre-activations: [0..2*H] for reset/update gates, and
  // [2*H..3*H] for candidate gate
  mlir::MlirOp i_rz =
      mlir::stablehlo::Slice(x_proj_step, {0, 0}, {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp i_n = mlir::stablehlo::Slice(x_proj_step, {0, 2 * hidden},
                                            {batch, 3 * hidden}, {1, 1});
  mlir::MlirOp h_rz =
      mlir::stablehlo::Slice(h_proj, {0, 0}, {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp h_n = mlir::stablehlo::Slice(h_proj, {0, 2 * hidden},
                                            {batch, 3 * hidden}, {1, 1});

  // Step 3: Vectorized sigmoid over [B, 2*H] evaluating reset and update gates
  // simultaneously
  mlir::MlirOp pre_rz = mlir::stablehlo::Add(i_rz, h_rz);
  mlir::MlirOp rz = mlir::stablehlo::Logistic(pre_rz);

  mlir::MlirOp resetgate =
      mlir::stablehlo::Slice(rz, {0, 0}, {batch, hidden}, {1, 1});
  mlir::MlirOp updategate =
      mlir::stablehlo::Slice(rz, {0, hidden}, {batch, 2 * hidden}, {1, 1});

  // Step 4: Evaluate candidate gate: n_t = tanh(i_n + r_t * h_n)
  mlir::MlirOp r_hn = mlir::stablehlo::Mul(resetgate, h_n);
  mlir::MlirOp pre_n = mlir::stablehlo::Add(i_n, r_hn);
  mlir::MlirOp newgate = mlir::stablehlo::Tanh(pre_n);

  // Step 5: Convex combination hidden state update: h_t = (h_{t-1} - n_t) * z_t
  // + n_t
  mlir::MlirOp h_minus_n = mlir::stablehlo::Subtract(h_curr, newgate);
  mlir::MlirOp scaled = mlir::stablehlo::Mul(h_minus_n, updategate);
  mlir::MlirOp h_next = mlir::stablehlo::Add(scaled, newgate);

  // Step 6: Pack cached 4H activations: [resetgate, updategate, newgate, h_n]
  // [B, 4*H] for autograd
  mlir::MlirOp act_4h = mlir::stablehlo::Concatenate(
      h_curr.getBuilder(), {rz, newgate, h_n}, /*dim=*/1);

  return {h_next, resetgate, updategate, newgate, h_n, act_4h};
}

// Container holding the adjoint gradients resulting from a single backward GRU
// step.
//
// Fields:
// - `delta_pre_ig`: Gradient w.r.t. input pre-activations [delta_pre_r,
// delta_pre_z, delta_pre_n] [B, 3*H].
// - `delta_pre_hg`: Gradient w.r.t. recurrent pre-activations [delta_pre_r,
// delta_pre_z, delta_h_n] [B, 3*H].
// - `delta_h_prev`: Backpropagated hidden state gradient delta_h_{t-1} [B,
// hidden].
struct GruBackwardStepResult {
  mlir::MlirOp delta_pre_ig;
  mlir::MlirOp delta_pre_hg;
  mlir::MlirOp delta_h_prev;
};

// Evaluates a single recurrent timestep of the GRU cell backward pass using
// closed-form adjoint algebra.
//
// What it computes:
// 1. Unpacks cached gate activations from `act_4h`:
//      resetgate r_t = act_4h[:, 0:H],   updategate z_t = act_4h[:, H:2*H]
//      newgate n_t   = act_4h[:, 2*H:3*H], h_n        = act_4h[:, 3*H:4*H]
// 2. Invokes ComputeGruGateAdjoints to compute analytical gate adjoints:
//      delta_n = delta_h * (1 - z)
//      delta_pre_n = delta_n * (1 - n^2)
//      delta_z = delta_h * (h_prev - n)
//      delta_pre_z = delta_z * z * (1 - z)
//      delta_r = delta_pre_n * h_n
//      delta_pre_r = delta_r * r * (1 - r)
//      delta_h_n = delta_pre_n * r
//      delta_hx_skip = delta_h * z
// 3. Assembles pre-activation adjoints:
//      delta_pre_ig = concat([delta_pre_r, delta_pre_z, delta_pre_n], dim=1)
//      [B, 3*H] delta_pre_hg = concat([delta_pre_r, delta_pre_z, delta_h_n],
//      dim=1)    [B, 3*H]
// 4. Backpropagates through recurrent projection:
//      delta_h_from_hh = delta_pre_hg @ W_hh  [B, hidden]
//      delta_h_prev = delta_hx_skip + delta_h_from_hh [B, hidden]
//
// Parameters:
// - `builder`: MLIR builder for op emission.
// - `delta_h`: Upstream gradient arriving at h_t: grad_y_t + delta_h_{t+1} [B,
// hidden].
// - `h_prev`: Previous hidden state h_{t-1} [B, hidden].
// - `act_4h`: Cached forward gate activations [B, 4*hidden].
// - `w_hh`: Recurrent weight matrix [3*hidden, hidden].
// - `hh_bwd_dot_dims`: Contracting dimensions ({1}, {0}) for backward GEMM.
// - `batch`: Batch size B.
// - `hidden`: Hidden dimension H.
// - `one`: Constant 1.0 tensor matching dtype and shape for algebraic adjoints.
// - `precision`: Hardware precision level for matrix multiplication.
//
// Returns:
// - `GruBackwardStepResult` struct containing delta_pre_ig, delta_pre_hg, and
// delta_h_prev.
GruBackwardStepResult ComputeGruBackwardStep(
    mlir::MlirBuilder& builder, mlir::MlirOp delta_h, mlir::MlirOp h_prev,
    mlir::MlirOp act_4h, mlir::MlirOp w_hh,
    mlir::stablehlo::DotDimensionNumbersAttr hh_bwd_dot_dims, int64_t batch,
    int64_t hidden, mlir::MlirOp one, mlir::stablehlo::Precision precision) {
  const mlir::Type acc_dtype = GetTensorTypeOrDie(delta_h).getElementType();
  auto to_acc = [acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
    if (GetTensorTypeOrDie(op).getElementType() == acc_dtype) {
      return op;
    }
    return mlir::stablehlo::ConvertElementType(op, acc_dtype);
  };
  // Step 1: Ensure all backward inputs are converted to accumulation dtype
  // (F32)
  delta_h = to_acc(delta_h);
  h_prev = to_acc(h_prev);
  act_4h = to_acc(act_4h);
  one = to_acc(one);

  // Step 2: Unpack cached forward activations [resetgate, updategate, newgate,
  // h_n]
  mlir::MlirOp resetgate =
      mlir::stablehlo::Slice(act_4h, {0, 0}, {batch, hidden}, {1, 1});
  mlir::MlirOp updategate =
      mlir::stablehlo::Slice(act_4h, {0, hidden}, {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp newgate = mlir::stablehlo::Slice(act_4h, {0, 2 * hidden},
                                                {batch, 3 * hidden}, {1, 1});
  mlir::MlirOp h_n = mlir::stablehlo::Slice(act_4h, {0, 3 * hidden},
                                            {batch, 4 * hidden}, {1, 1});

  // Step 3: Compute closed-form gate adjoints via fused algebraic solver
  GruGateAdjoints adj = ComputeGruGateAdjoints(delta_h, h_prev, resetgate,
                                               updategate, newgate, h_n, one);

  // Step 4: Concatenate pre-activation adjoints for input and recurrent GEMMs
  mlir::MlirOp delta_pre_ig = mlir::stablehlo::Concatenate(
      builder, {adj.delta_pre_r, adj.delta_pre_z, adj.delta_pre_n}, /*dim=*/1);
  mlir::MlirOp delta_pre_hg = mlir::stablehlo::Concatenate(
      builder, {adj.delta_pre_r, adj.delta_pre_z, adj.delta_h_n}, /*dim=*/1);

  // Step 5: Backpropagate through recurrent weight matrix W_hh and add skip
  // connection
  mlir::MlirOp delta_h_from_hh = MixedPrecisionDotGeneral(
      delta_pre_hg, w_hh, hh_bwd_dot_dims, acc_dtype, precision);
  mlir::MlirOp delta_h_prev =
      mlir::stablehlo::Add(adj.delta_hx_skip, delta_h_from_hh);

  return {delta_pre_ig, delta_pre_hg, delta_h_prev};
}

// Applies inverted dropout between stacked recurrent layers during the forward
// pass.
//
// What it computes:
// - When dropout <= 0.0: returns `layer_out` unchanged.
// - When dropout >= 1.0: returns a zero tensor and a false mask.
// - Inverted dropout with probability `p = dropout`:
//     mask = rand_tensor >= p
//     dropped = select(mask, layer_out, 0.0) * (1.0 / (1.0 - p))
//
// Parameters:
// - `layer_out`: Output sequence tensor from the preceding recurrent layer.
// - `rand_tensor`: Uniform pseudo-random tensor in [0.0, 1.0) generated from
// hardware RNG.
// - `dropout`: Dropout probability p in [0.0, 1.0].
//
// Returns:
// - Pair of {dropped_tensor, boolean_mask} where the mask is saved for backward
// propagation.
// Result of applying inter-layer dropout.
struct InterLayerDropoutResult {
  mlir::MlirOp dropped_output;
  mlir::MlirOp dropout_mask;
};

inline InterLayerDropoutResult ApplyInterLayerDropout(mlir::MlirOp layer_out,
                                                      mlir::MlirOp rand_tensor,
                                                      double dropout) {
  // Bypass when dropout probability is non-positive
  if (dropout <= 0.0) {
    return {layer_out, layer_out};
  }
  // Clamp full dropout to complete zeroing
  if (dropout >= 1.0) {
    mlir::MlirOp zero_const = MakeConstantLike(layer_out, 0.0);
    mlir::MlirOp false_const = MakeConstantLike(rand_tensor, false);
    return {zero_const, false_const};
  }
  // Generate boolean retention mask: rand >= dropout
  mlir::MlirOp p_const = MakeConstantLike(rand_tensor, dropout);
  mlir::MlirOp mask = mlir::stablehlo::Compare(
      rand_tensor, p_const, mlir::stablehlo::ComparisonDirection::GE);
  // Zero dropped elements and scale retained elements by 1.0 / (1.0 - dropout)
  mlir::MlirOp zero = MakeConstantLike(layer_out, 0.0);
  mlir::MlirOp masked = mlir::stablehlo::Select(mask, layer_out, zero);
  const double scale = 1.0 / (1.0 - dropout);
  mlir::MlirOp scale_const = MakeConstantLike(masked, scale);
  mlir::MlirOp dropped = mlir::stablehlo::Mul(masked, scale_const);
  return {dropped, mask};
}

// Applies the adjoint backward pass of inverted dropout to upstream gradients.
//
// What it computes:
// - When dropout <= 0.0: returns `grad_output` unchanged.
// - Multiplies upstream gradients by the saved forward boolean mask and scales
// by 1 / (1 - p):
//     grad_in = select(mask, grad_output, 0.0) * (1.0 / (1.0 - p))
//
// Parameters:
// - `grad_output`: Upstream gradient tensor from the succeeding layer.
// - `mask`: Boolean mask saved from the forward dropout pass.
// - `dropout`: Dropout probability p in [0.0, 1.0].
//
// Returns:
// - Backpropagated gradient tensor.
inline mlir::MlirOp ApplyDropoutBackward(mlir::MlirOp grad_output,
                                         mlir::MlirOp mask, double dropout) {
  if (dropout <= 0.0) {
    return grad_output;
  }
  // Zero out gradients where activations were dropped and scale retained
  // gradients
  mlir::MlirOp zero = MakeConstantLike(grad_output, 0.0);
  mlir::MlirOp masked = mlir::stablehlo::Select(mask, grad_output, zero);
  const double scale = 1.0 / (1.0 - dropout);
  mlir::MlirOp scale_const = MakeConstantLike(masked, scale);
  return mlir::stablehlo::Mul(masked, scale_const);
}

// Weight and bias containers for a single layer in pipelined wavefront
// execution.
//
// Fields:
// - `w_hh`: Recurrent weight matrix [3*hidden, hidden].
// - `w_ih`: Optional input-hidden weight matrix [3*hidden, in_dim] (only needed
// for l > 0).
// - `b_ih`: Optional input bias vector [3*hidden].
// - `b_hh`: Optional recurrent bias vector [3*hidden].
struct GruWavefrontLayerWeight {
  mlir::MlirOp w_hh;
  std::optional<mlir::MlirOp> w_ih;
  std::optional<mlir::MlirOp> b_ih;
  std::optional<mlir::MlirOp> b_hh;
};

// Container holding the forward outputs of a unidirectional GRU layer.
//
// Fields:
// - `layer_output_seq`: Full output sequence tensor [T, B, H] (or [B, T, H] if
// batch_first).
// - `final_h`: Final hidden state tensor reshaped to 3D [1, B, H].
// - `cached_acts`: Optional sequence of cached 4H activations [T, B, 4*H] for
// training autograd.
struct GruLayerOutputs {
  mlir::MlirOp layer_output_seq;
  mlir::MlirOp final_h;
  std::optional<mlir::MlirOp> cached_acts;
};

// Container holding the forward outputs of a bidirectional GRU layer.
//
// Fields:
// - `layer_output_seq`: Full concatenated output sequence [T, B, 2*H] ([B, T,
// 2*H] if batch_first).
// - `final_h`: Concatenated final hidden states for forward and reverse
// directions [2, B, H].
// - `cached_acts_fwd`: Optional cached 4H activations for forward sweep [T, B,
// 4*H].
// - `cached_acts_rev`: Optional cached 4H activations for reverse sweep [T, B,
// 4*H].
struct GruBidirLayerOutputs {
  mlir::MlirOp layer_output_seq;
  mlir::MlirOp final_h;
  std::optional<mlir::MlirOp> cached_acts_fwd;
  std::optional<mlir::MlirOp> cached_acts_rev;
};

// Helper struct capturing the outputs of a series of unrolled GRU forward
// steps.
//
// Fields:
// - `h_final`: Updated hidden state at the completion of the unrolled steps [B,
// hidden].
// - `step_outputs`: Vector of individual step output tensors [1, B, H] (or [B,
// 1, H]).
// - `step_acts`: Vector of individual step 4H cached activation tensors [1, B,
// 4*H] (or [B, 1, 4*H]).
struct GruUnrolledForwardSteps {
  mlir::MlirOp h_final;
  std::vector<mlir::MlirOp> step_outputs;
  std::vector<mlir::MlirOp> step_acts;
};

// Executes a sequence of unrolled recurrent GRU forward steps starting at
// `start_step` for `num_steps`.
//
// What it computes:
// - Loops over timesteps t = start_step to start_step + num_steps - 1:
//   1. Slices x_proj[t] [B, 3*H] from the pre-projected sequence tensor via
//   SliceStep2D.
//   2. Evaluates ComputeGruStep(x_t, curr_h, w_hh, ...) -> {h_next, act_4h}.
//   3. Updates curr_h = h_next.
//   4. Converts curr_h to output dtype via `to_out` and expands to 3D [1, B, H]
//   (or [B, 1, H]).
//   5. If cache_activations is true, records act_4h expanded to 3D [1, B, 4*H].
//
// Parameters:
// - `x_seq_3d`: Pre-projected input sequence tensor [T, B, 3*H] or [B, T, 3*H].
// - `start_step`: Starting timestep offset in the sequence.
// - `num_steps`: Number of steps to unroll (e.g. kGruUnrollFactor = 8 or
// remainder steps).
// - `h_start`: Initial hidden state at start_step [B, hidden].
// - `w_hh`: Recurrent weight matrix [3*hidden, hidden].
// - `hh_dot_dims`: Contracting dimension attribute for recurrent GEMM.
// - `batch`: Batch size B.
// - `hidden`: Hidden dimension H.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Type-conversion lambda from accumulation dtype to output dtype.
// - `b_hh`: Optional recurrent bias vector [3*hidden].
// - `precision`: Hardware precision level.
// - `cache_activations`: True if caching 4H activations for autograd.
//
// Returns:
// - `GruUnrolledForwardSteps` containing h_final, vector of step outputs, and
// vector of cached activations.
template <typename ToOutFn>
GruUnrolledForwardSteps RunGruForwardSteps(
    mlir::MlirOp x_seq_3d, int64_t start_step, int64_t num_steps,
    mlir::MlirOp h_start, mlir::MlirOp w_hh,
    mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims, int64_t batch,
    int64_t hidden, bool batch_first, ToOutFn to_out,
    std::optional<mlir::MlirOp> b_hh, mlir::stablehlo::Precision precision,
    bool cache_activations) {
  mlir::MlirOp curr_h = h_start;
  std::vector<mlir::MlirOp> outputs;
  outputs.reserve(num_steps);
  std::vector<mlir::MlirOp> acts;
  if (cache_activations) {
    acts.reserve(num_steps);
  }

  // Statically unroll num_steps consecutive recurrent forward evaluations
  for (int64_t i = 0; i < num_steps; ++i) {
    const int64_t t = start_step + i;
    // Step 1: Slice timestep t pre-projection [B, 3*H]
    mlir::MlirOp x_t_2d =
        SliceStep2D(x_seq_3d, t, batch, 3 * hidden, batch_first);
    // Step 2: Compute recurrent hidden state update and gate activations
    GruStepResults step = ComputeGruStep(x_t_2d, curr_h, w_hh, hh_dot_dims,
                                         batch, hidden, b_hh, precision);
    curr_h = step.h_next;

    // Step 3: Format and accumulate step sequence output
    mlir::MlirOp h_out = to_out(curr_h);
    outputs.push_back(ExpandStep3D(h_out, batch, hidden, batch_first));
    // Step 4: Accumulate 4H activations for autograd caching
    if (cache_activations) {
      acts.push_back(ExpandStep3D(step.act_4h, batch, 4 * hidden, batch_first));
    }
  }
  return {curr_h, std::move(outputs), std::move(acts)};
}

// Builds a static unrolled forward sweep for short sequences (seq_len <
// kGruUnrollFactor).
//
// What it computes:
// - Executes all `seq_len` steps in a single unrolled pass via
// RunGruForwardSteps without
//   incurring dynamic stablehlo::WhileOp control-flow overhead.
// - Concatenates step outputs along the sequence dimension into full sequence
// output tensor.
// - Formats final hidden state as 3D tensor [1, batch, hidden].
// - If cache_activations is true, concatenates step activations into [seq_len,
// batch, 4*hidden].
//
// Parameters:
// - `builder`: MLIR builder.
// - `x_proj`: Pre-projected input sequence tensor.
// - `h_init`: Initial hidden state [batch, hidden].
// - `w_hh`: Recurrent weight matrix [3*hidden, hidden].
// - `hh_dot_dims`: Contracting dimensions for recurrent GEMM.
// - `seq_len`: Sequence length T (< 8).
// - `batch`: Batch size B.
// - `hidden`: Hidden dimension H.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Dtype conversion lambda.
// - `b_hh`: Optional recurrent bias.
// - `precision`: Hardware precision level.
// - `cache_activations`: Whether to cache 4H activations.
//
// Returns:
// - `GruLayerOutputs` struct with full output sequence, final hidden state, and
// optional cached activations.
template <typename ToOutFn>
GruLayerOutputs BuildGruStaticForward(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj, mlir::MlirOp h_init,
    mlir::MlirOp w_hh, mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    int64_t seq_len, int64_t batch, int64_t hidden, bool batch_first,
    ToOutFn to_out, std::optional<mlir::MlirOp> b_hh,
    mlir::stablehlo::Precision precision, bool cache_activations) {
  // Statically unroll all sequence steps from t = 0 to seq_len
  GruUnrolledForwardSteps steps = RunGruForwardSteps(
      x_proj, /*start_step=*/0, seq_len, h_init, w_hh, hh_dot_dims, batch,
      hidden, batch_first, to_out, b_hh, precision, cache_activations);

  const int64_t concat_dim = batch_first ? 1 : 0;
  std::optional<mlir::MlirOp> acts_out = std::nullopt;
  if (cache_activations) {
    acts_out = ConcatDim(builder, steps.step_acts, concat_dim);
  }
  return {ConcatDim(builder, steps.step_outputs, concat_dim),
          mlir::stablehlo::Reshape(to_out(steps.h_final), {1, batch, hidden}),
          acts_out};
}

// Evaluates the tail remainder steps (seq_len % kGruUnrollFactor) following
// chunked WhileOp execution.
//
// What it computes:
// - Unrolls the remaining `rem_steps` timesteps starting at `chunked_steps`
// using RunGruForwardSteps.
// - Concatenates the while loop chunked sequence outputs with the remainder
// outputs along the sequence axis.
// - If cache_activations is true, concatenates chunked activations with
// remainder activations.
// - Formats the remainder final hidden state as [1, batch, hidden].
//
// Parameters:
// - `builder`: MLIR builder.
// - `x_proj`: Full pre-projected sequence tensor.
// - `chunked_y`: Output sequence tensor generated by the chunked WhileOp
// [chunked_steps, B, H].
// - `final_loop_h`: Hidden state at the exit of the while loop [B, hidden].
// - `chunked_act`: Optional cached activations generated by the while loop.
// - `chunked_steps`: Number of timesteps completed inside the while loop
// (num_chunks * 8).
// - `rem_steps`: Remaining timesteps (seq_len % 8).
// - `batch`: Batch size B.
// - `hidden`: Hidden dimension H.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Dtype conversion lambda.
// - `w_hh`: Recurrent weight matrix [3*hidden, hidden].
// - `hh_dot_dims`: Contracting dimensions for recurrent GEMM.
// - `b_hh`: Optional recurrent bias vector.
// - `precision`: Hardware precision level.
// - `cache_activations`: Whether to cache activations.
//
// Returns:
// - `GruLayerOutputs` struct containing the complete combined sequence output
// and final hidden state.
template <typename ToOutFn>
GruLayerOutputs BuildGruForwardRemainder(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj, mlir::MlirOp chunked_y,
    mlir::MlirOp final_loop_h, std::optional<mlir::MlirOp> chunked_act,
    int64_t chunked_steps, int64_t rem_steps, int64_t batch, int64_t hidden,
    bool batch_first, ToOutFn to_out, mlir::MlirOp w_hh,
    mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    std::optional<mlir::MlirOp> b_hh, mlir::stablehlo::Precision precision,
    bool cache_activations) {
  // Evaluate the remainder steps starting from chunked_steps
  GruUnrolledForwardSteps rem = RunGruForwardSteps(
      x_proj, chunked_steps, rem_steps, final_loop_h, w_hh, hh_dot_dims, batch,
      hidden, batch_first, to_out, b_hh, precision, cache_activations);

  const int64_t concat_dim = batch_first ? 1 : 0;
  mlir::MlirOp rem_y = ConcatDim(builder, rem.step_outputs, concat_dim);
  mlir::MlirOp full_y =
      mlir::stablehlo::Concatenate(builder, {chunked_y, rem_y}, concat_dim);

  std::optional<mlir::MlirOp> full_act = std::nullopt;
  if (cache_activations) {
    mlir::MlirOp rem_act = ConcatDim(builder, rem.step_acts, concat_dim);
    full_act = mlir::stablehlo::Concatenate(builder, {*chunked_act, rem_act},
                                            concat_dim);
  }

  return {full_y,
          mlir::stablehlo::Reshape(to_out(rem.h_final), {1, batch, hidden}),
          full_act};
}

// Builds a unidirectional GRU forward layer using chunked static unrolling
// inside a stablehlo::WhileOp.
//
// What it computes:
// - If seq_len < kGruUnrollFactor: delegates to BuildGruStaticForward for
// zero-overhead static execution.
// - When seq_len >= kGruUnrollFactor:
//   1. Computes chunk counts: num_chunks = seq_len / 8, chunked_steps =
//   num_chunks * 8, rem_steps = seq_len % 8.
//   2. Initializes while loop carry state:
//      - step_idx (i64 scalar initialized to 0)
//      - curr_h (initialized to h_init [B, H])
//      - y_chunked (initialized to zero tensor [chunked_steps, B, H])
//      - optional act_chunked (initialized to zero tensor [chunked_steps, B,
//      4*H])
//   3. Builds WhileOp condition block: step_idx < chunked_steps.
//   4. Builds WhileOp body block:
//      - Dynamically slices a chunk of k=8 timesteps from x_proj using
//      DynamicSlice at step_idx.
//      - Evaluates 8 unrolled recurrent steps via RunGruForwardSteps.
//      - Commits chunk outputs into y_chunked using DynamicUpdateSlice at
//      step_idx.
//      - If caching, commits chunk activations into act_chunked using
//      DynamicUpdateSlice.
//      - Increments step_idx by kGruUnrollFactor (8).
//   5. Remainder handoff:
//      - If rem_steps == 0: returns while loop results directly.
//      - If rem_steps > 0: invokes BuildGruForwardRemainder to evaluate
//      remaining steps and concatenate.
//
// Parameters:
// - `builder`: MLIR builder.
// - `x_proj`: Full pre-projected input sequence tensor [T, B, 3*H].
// - `h_init`: Initial hidden state for the layer [B, H].
// - `w_hh`: Recurrent weight matrix [3*H, H].
// - `hh_dot_dims`: Contracting dimension attribute for recurrent GEMM.
// - `seq_len`: Sequence length T.
// - `batch`: Batch size B.
// - `hidden`: Hidden dimension H.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Type-conversion lambda to output dtype.
// - `b_hh`: Optional recurrent bias vector [3*H].
// - `precision`: Hardware precision level.
// - `cache_activations`: Whether to cache 4H activations for autograd.
//
// Returns:
// - `GruLayerOutputs` struct containing sequence output [T, B, H], final hidden
// state [1, B, H], and cached acts.
template <typename ToOutFn>
GruLayerOutputs BuildGruLayerForward(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj, mlir::MlirOp h_init,
    mlir::MlirOp w_hh, mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    int64_t seq_len, int64_t batch, int64_t hidden, bool batch_first,
    ToOutFn to_out, std::optional<mlir::MlirOp> b_hh,
    mlir::stablehlo::Precision precision, bool cache_activations) {
  // Fast-path: sequences shorter than unroll factor k=8 are unrolled statically
  // without WhileOp
  if (seq_len < kGruUnrollFactor) {
    return BuildGruStaticForward(builder, x_proj, h_init, w_hh, hh_dot_dims,
                                 seq_len, batch, hidden, batch_first, to_out,
                                 b_hh, precision, cache_activations);
  }

  // Step 1: Compute chunked recurrence parameters
  const int64_t num_chunks = seq_len / kGruUnrollFactor;
  const int64_t chunked_steps = num_chunks * kGruUnrollFactor;
  const int64_t rem_steps = seq_len % kGruUnrollFactor;
  const int64_t concat_dim = batch_first ? 1 : 0;

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = x_proj.getValue().getLoc();
  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType i64_scalar_type =
      mlir::RankedTensorType::get({}, i64);

  // Step 2: Initialize output sequence tensor accumulator [chunked_steps,
  // batch, hidden]
  const mlir::Type out_elem_type =
      GetTensorTypeOrDie(to_out(h_init)).getElementType();
  const llvm::SmallVector<int64_t, 3> y_chunk_shape =
      batch_first ? llvm::SmallVector<int64_t, 3>{batch, chunked_steps, hidden}
                  : llvm::SmallVector<int64_t, 3>{chunked_steps, batch, hidden};
  const mlir::RankedTensorType y_type =
      mlir::RankedTensorType::get(y_chunk_shape, out_elem_type);

  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0f, out_elem_type);
  mlir::MlirOp y_init =
      mlir::stablehlo::BroadcastInDim(y_type, zero_scalar, {});
  mlir::MlirOp step_idx_init = MakeScalarConstant(builder, 0, i64);

  // Step 3: Initialize activation cache buffer [chunked_steps, batch, 4*hidden]
  // if training
  const mlir::Type acc_elem_type = GetTensorTypeOrDie(h_init).getElementType();
  const llvm::SmallVector<int64_t, 3> act_chunk_shape =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, chunked_steps, 4 * hidden}
          : llvm::SmallVector<int64_t, 3>{chunked_steps, batch, 4 * hidden};
  const mlir::RankedTensorType act_type =
      mlir::RankedTensorType::get(act_chunk_shape, acc_elem_type);
  mlir::MlirOp zero_acc_scalar =
      MakeScalarConstant(builder, 0.0f, acc_elem_type);
  mlir::MlirOp act_init =
      mlir::stablehlo::BroadcastInDim(act_type, zero_acc_scalar, {});

  // Step 4: Assemble while loop carried types and initial operand values
  llvm::SmallVector<mlir::Type> loop_types;
  llvm::SmallVector<mlir::Value> loop_inits;
  if (cache_activations) {
    loop_types = {i64_scalar_type, h_init.getType(), y_type, act_type};
    loop_inits = {step_idx_init.getValue(), h_init.getValue(),
                  y_init.getValue(), act_init.getValue()};
  } else {
    loop_types = {i64_scalar_type, h_init.getType(), y_type};
    loop_inits = {step_idx_init.getValue(), h_init.getValue(),
                  y_init.getValue()};
  }

  // Step 5: Construct stablehlo::WhileOp for chunked forward recurrence
  auto while_op =
      mlir::stablehlo::WhileOp::create(op_builder, loc, loop_types, loop_inits);

  // Condition block: continue while step_idx < chunked_steps
  mlir::Block* const cond_block = op_builder.createBlock(&while_op.getCond());
  cond_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(cond_block);

  mlir::MlirOp limit = MakeScalarConstant(builder, chunked_steps, i64);
  const mlir::Value cond =
      mlir::stablehlo::CompareOp::create(
          op_builder, loc, cond_block->getArgument(0), limit.getValue(),
          mlir::stablehlo::ComparisonDirection::LT)
          .getResult();
  mlir::stablehlo::ReturnOp::create(op_builder, loc, cond);

  // Body block: evaluate k=8 steps and commit via DynamicUpdateSlice
  mlir::Block* const body_block = op_builder.createBlock(&while_op.getBody());
  body_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(body_block);

  mlir::MlirOp body_step_idx(builder, body_block->getArgument(0));
  mlir::MlirOp body_h(builder, body_block->getArgument(1));
  mlir::MlirOp body_y(builder, body_block->getArgument(2));
  std::optional<mlir::MlirOp> body_act =
      cache_activations ? std::make_optional(
                              mlir::MlirOp(builder, body_block->getArgument(3)))
                        : std::nullopt;

  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64);
  llvm::SmallVector<mlir::MlirOp, 3> x_start_indices;
  llvm::SmallVector<int64_t, 3> x_slice_sizes;
  if (batch_first) {
    x_start_indices = {zero_i64, body_step_idx, zero_i64};
    x_slice_sizes = {batch, kGruUnrollFactor, 3 * hidden};
  } else {
    x_start_indices = {body_step_idx, zero_i64, zero_i64};
    x_slice_sizes = {kGruUnrollFactor, batch, 3 * hidden};
  }
  // Dynamic slice of k=8 timesteps from pre-projected sequence tensor
  mlir::MlirOp x_chunk =
      mlir::stablehlo::DynamicSlice(x_proj, x_start_indices, x_slice_sizes);

  // Evaluate unrolled 8 steps in vector registers
  GruUnrolledForwardSteps chunk_steps = RunGruForwardSteps(
      x_chunk, /*start_step=*/0, kGruUnrollFactor, body_h, w_hh, hh_dot_dims,
      batch, hidden, batch_first, to_out, b_hh, precision, cache_activations);

  // Commit chunk outputs into sequence buffer
  mlir::MlirOp chunk_out =
      ConcatDim(builder, chunk_steps.step_outputs, concat_dim);
  llvm::SmallVector<mlir::MlirOp, 3> y_start_indices;
  if (batch_first) {
    y_start_indices = {zero_i64, body_step_idx, zero_i64};
  } else {
    y_start_indices = {body_step_idx, zero_i64, zero_i64};
  }
  mlir::MlirOp next_y =
      mlir::stablehlo::DynamicUpdateSlice(body_y, chunk_out, y_start_indices);

  // Advance step index by k=8
  mlir::MlirOp k_factor_op = MakeScalarConstant(builder, kGruUnrollFactor, i64);
  mlir::MlirOp next_step_idx = mlir::stablehlo::Add(body_step_idx, k_factor_op);

  llvm::SmallVector<mlir::Value> next_state;
  if (cache_activations) {
    mlir::MlirOp chunk_act_out =
        ConcatDim(builder, chunk_steps.step_acts, concat_dim);
    mlir::MlirOp next_act = mlir::stablehlo::DynamicUpdateSlice(
        *body_act, chunk_act_out, y_start_indices);
    next_state = {next_step_idx.getValue(), chunk_steps.h_final.getValue(),
                  next_y.getValue(), next_act.getValue()};
  } else {
    next_state = {next_step_idx.getValue(), chunk_steps.h_final.getValue(),
                  next_y.getValue()};
  }
  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_state);

  op_builder.setInsertionPointAfter(while_op);

  // Step 6: Extract results from while loop
  mlir::MlirOp final_loop_h(builder, while_op.getResult(1));
  mlir::MlirOp chunked_y(builder, while_op.getResult(2));
  std::optional<mlir::MlirOp> chunked_act =
      cache_activations
          ? std::make_optional(mlir::MlirOp(builder, while_op.getResult(3)))
          : std::nullopt;

  // Step 7: Handle tail remainder timesteps if seq_len % 8 != 0
  if (rem_steps == 0) {
    return {chunked_y,
            mlir::stablehlo::Reshape(to_out(final_loop_h), {1, batch, hidden}),
            chunked_act};
  }

  return BuildGruForwardRemainder(
      builder, x_proj, chunked_y, final_loop_h, chunked_act, chunked_steps,
      rem_steps, batch, hidden, batch_first, to_out, w_hh, hh_dot_dims, b_hh,
      precision, cache_activations);
}

// Evaluates wavefront remainder steps across all layers when seq_len %
// kGruUnrollFactor > 0.
//
// What it computes:
// - Evaluates remainder timesteps t = chunked_steps to seq_len - 1 across all
// stacked layers l = 0 .. num_layers - 1.
// - Layer 0 processes remainder slice of x_proj_0.
// - Layers l > 0 apply inter-layer dropout (if enabled), project input
// activations via W_ih_l, and unroll remainder steps.
// - Concatenates chunked sequence output with remainder sequence output.
// - Stacks all layer final hidden states into a single [num_layers, batch,
// hidden] tensor.
//
// Parameters:
// - `builder`: MLIR builder.
// - `x_proj_0`: Layer 0 pre-projected input sequence tensor.
// - `chunked_y`: Output sequence tensor generated by the chunked wavefront
// while loop.
// - `final_h`: Vector of final hidden states per layer at exit of while loop.
// - `layer_weights`: Weights and biases for all stacked layers.
// - `hh_dot_dims`: Contracting dimensions for recurrent GEMM.
// - `ih_dot_dims`: Contracting dimensions for input projection GEMM.
// - `chunked_steps`: Steps processed inside while loop (num_chunks * 8).
// - `rem_steps`: Remainder timesteps (seq_len % 8).
// - `seq_len`: Total sequence length.
// - `batch`: Batch size.
// - `hidden`: Hidden dimension size.
// - `num_layers`: Number of stacked layers.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Dtype conversion lambda.
// - `dropout`: Dropout probability.
// - `has_dropout`: True if dropout is enabled.
// - `rand_op`: Optional random uniform tensor for dropout masking.
// - `precision`: Hardware precision level.
//
// Returns:
// - `GruLayerOutputs` struct containing complete multi-layer sequence output
// and stacked final hidden states.
// Evaluates remainder recurrence for a single subsequent layer l (l >= 1)
// during wavefront forward.
//
// What it computes:
// - Optionally extracts and applies inter-layer dropout on the previous layer's
// remainder sequence.
// - Projects input activations through W_ih_l and adds optional bias b_ih_l.
// - Evaluates rem_steps forward recurrence steps using RunGruForwardSteps.
// - Concatenates the step outputs along the sequence dimension.
//
// Parameters:
// - `builder`: MLIR builder.
// - `curr_rem_seq`: Output sequence from previous layer l - 1.
// - `prev_h`: Initial hidden state for layer l.
// - `layer_weight`: Weight and bias bundle for layer l.
// - `hh_dot_dims`: Contracting dimensions for recurrent GEMM.
// - `ih_dot_dims`: Contracting dimensions for input projection GEMM.
// - `l`: Layer index.
// - `chunked_steps`: Steps processed inside while loop (num_chunks * 8).
// - `rem_steps`: Remainder timesteps (seq_len % 8).
// - `seq_len`: Total sequence length.
// - `batch`: Batch size.
// - `hidden`: Hidden dimension size.
// - `batch_first`: Memory layout indicator.
// - `concat_dim`: Concatenation dimension along sequence axis.
// - `to_out`: Dtype conversion lambda.
// - `dropout`: Dropout probability.
// - `has_dropout`: True if dropout is enabled.
// - `rand_op`: Optional random uniform tensor for dropout masking.
// - `precision`: Hardware precision level.
//
// Returns:
// - Pair of (updated hidden state for layer l, remainder sequence output for
// layer l). Result of evaluating wavefront forward remainder layer.
struct WavefrontRemainderLayerResult {
  mlir::MlirOp final_h_l;
  mlir::MlirOp next_rem_seq;
};

template <typename ToOutFn>
WavefrontRemainderLayerResult EvaluateWavefrontRemainderLayer(
    mlir::MlirBuilder& builder, mlir::MlirOp curr_rem_seq, mlir::MlirOp prev_h,
    const GruWavefrontLayerWeight& layer_weight,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr ih_dot_dims, int64_t l,
    int64_t chunked_steps, int64_t rem_steps, int64_t seq_len, int64_t batch,
    int64_t hidden, bool batch_first, int64_t concat_dim, ToOutFn to_out,
    double dropout, bool has_dropout, std::optional<mlir::MlirOp> rand_op,
    mlir::stablehlo::Precision precision) {
  if (has_dropout) {
    const Dimensions layer_rand_shape =
        batch_first ? Dimensions{batch, rem_steps, hidden}
                    : Dimensions{rem_steps, batch, hidden};
    const Dimensions slice_starts =
        batch_first ? Dimensions{l - 1, 0, chunked_steps, 0}
                    : Dimensions{l - 1, chunked_steps, 0, 0};
    const Dimensions slice_limits = batch_first
                                        ? Dimensions{l, batch, seq_len, hidden}
                                        : Dimensions{l, seq_len, batch, hidden};
    const Dimensions slice_strides = {1, 1, 1, 1};
    mlir::MlirOp rem_rand_4d = mlir::stablehlo::Slice(
        *rand_op, slice_starts, slice_limits, slice_strides);
    mlir::MlirOp rem_rand =
        mlir::stablehlo::Reshape(rem_rand_4d, layer_rand_shape);
    auto [dropped_rem, mask_rem] =
        ApplyInterLayerDropout(curr_rem_seq, rem_rand, dropout);
    curr_rem_seq = dropped_rem;
  }

  const mlir::Type acc_dtype = GetTensorTypeOrDie(prev_h).getElementType();
  mlir::MlirOp x_proj_rem = MixedPrecisionDotGeneral(
      curr_rem_seq, *layer_weight.w_ih, ih_dot_dims, acc_dtype, precision);
  if (layer_weight.b_ih.has_value()) {
    mlir::MlirOp b_ih = *layer_weight.b_ih;
    mlir::MlirOp b_bcast = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(x_proj_rem), b_ih, {/*broadcast_dimensions=*/2});
    x_proj_rem = mlir::stablehlo::Add(x_proj_rem, b_bcast);
  }

  GruUnrolledForwardSteps rem_l = RunGruForwardSteps(
      x_proj_rem, /*start_step=*/0, rem_steps, prev_h, layer_weight.w_hh,
      hh_dot_dims, batch, hidden, batch_first, to_out, layer_weight.b_hh,
      precision, /*cache_activations=*/false);
  mlir::MlirOp next_rem_seq =
      ConcatDim(builder, rem_l.step_outputs, concat_dim);
  return {rem_l.h_final, next_rem_seq};
}

// Stacks per-layer final hidden states [num_layers, batch, hidden].
//
// Parameters:
// - `builder`: MLIR builder.
// - `final_h`: Vector of final hidden state tensors for each layer.
// - `batch`: Batch size.
// - `hidden`: Hidden dimension size.
// - `to_out`: Dtype conversion lambda.
//
// Returns:
// - Stacked hidden state tensor of shape [num_layers, batch, hidden].
template <typename ToOutFn>
mlir::MlirOp StackPerLayerFinalHiddenStates(
    mlir::MlirBuilder& builder, const absl::Span<const mlir::MlirOp> final_h,
    int64_t batch, int64_t hidden, ToOutFn to_out) {
  std::vector<mlir::MlirOp> final_h_3d;
  final_h_3d.reserve(final_h.size());
  for (const mlir::MlirOp& h : final_h) {
    final_h_3d.push_back(
        mlir::stablehlo::Reshape(to_out(h), {1, batch, hidden}));
  }
  return ConcatDim(builder, final_h_3d, /*dim=*/0);
}

// Evaluates wavefront remainder steps across all layers when seq_len %
// kGruUnrollFactor > 0.
//
// What it computes:
// - Evaluates remainder timesteps t = chunked_steps to seq_len - 1 across all
// stacked layers l = 0 .. num_layers - 1.
// - Layer 0 processes remainder slice of x_proj_0.
// - Layers l > 0 apply inter-layer dropout (if enabled), project input
// activations via W_ih_l, and unroll remainder steps via
// EvaluateWavefrontRemainderLayer.
// - Concatenates chunked sequence output with remainder sequence output.
// - Stacks all layer final hidden states into a single [num_layers, batch,
// hidden] tensor.
//
// Parameters:
// - `builder`: MLIR builder.
// - `x_proj_0`: Layer 0 pre-projected input sequence tensor.
// - `chunked_y`: Output sequence tensor generated by the chunked wavefront
// while loop.
// - `final_h`: Vector of final hidden states per layer at exit of while loop.
// - `layer_weights`: Weights and biases for all stacked layers.
// - `hh_dot_dims`: Contracting dimensions for recurrent GEMM.
// - `ih_dot_dims`: Contracting dimensions for input projection GEMM.
// - `chunked_steps`: Steps processed inside while loop (num_chunks * 8).
// - `rem_steps`: Remainder timesteps (seq_len % 8).
// - `seq_len`: Total sequence length.
// - `batch`: Batch size.
// - `hidden`: Hidden dimension size.
// - `num_layers`: Number of stacked layers.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Dtype conversion lambda.
// - `dropout`: Dropout probability.
// - `has_dropout`: True if dropout is enabled.
// - `rand_op`: Optional random uniform tensor for dropout masking.
// - `precision`: Hardware precision level.
//
// Returns:
// - `GruLayerOutputs` struct containing complete multi-layer sequence output
// and stacked final hidden states.
template <typename ToOutFn>
GruLayerOutputs BuildWavefrontForwardRemainder(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj_0, mlir::MlirOp chunked_y,
    const std::vector<mlir::MlirOp>& final_h,
    const absl::Span<const GruWavefrontLayerWeight> layer_weights,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr ih_dot_dims,
    int64_t chunked_steps, int64_t rem_steps, int64_t seq_len, int64_t batch,
    int64_t hidden, int64_t num_layers, bool batch_first, ToOutFn to_out,
    double dropout, bool has_dropout, std::optional<mlir::MlirOp> rand_op,
    mlir::stablehlo::Precision precision) {
  const int64_t concat_dim = batch_first ? 1 : 0;
  std::vector<mlir::MlirOp> rem_curr_h = final_h;

  // Step 1: Layer 0 remainder evaluation
  GruUnrolledForwardSteps rem_0 = RunGruForwardSteps(
      x_proj_0, chunked_steps, rem_steps, rem_curr_h[0], layer_weights[0].w_hh,
      hh_dot_dims, batch, hidden, batch_first, to_out, layer_weights[0].b_hh,
      precision, /*cache_activations=*/false);
  rem_curr_h[0] = rem_0.h_final;
  mlir::MlirOp curr_rem_seq =
      ConcatDim(builder, rem_0.step_outputs, concat_dim);

  // Step 2: Stream remainder through subsequent layers l = 1 .. num_layers - 1
  for (int64_t l = 1; l < num_layers; ++l) {
    WavefrontRemainderLayerResult rem_l = EvaluateWavefrontRemainderLayer(
        builder, curr_rem_seq, rem_curr_h[l], layer_weights[l], hh_dot_dims,
        ih_dot_dims, l, chunked_steps, rem_steps, seq_len, batch, hidden,
        batch_first, concat_dim, to_out, dropout, has_dropout, rand_op,
        precision);
    rem_curr_h[l] = rem_l.final_h_l;
    curr_rem_seq = rem_l.next_rem_seq;
  }

  // Step 3: Concatenate while loop output sequence with remainder output
  // sequence
  mlir::MlirOp full_y = mlir::stablehlo::Concatenate(
      builder, {chunked_y, curr_rem_seq}, concat_dim);

  // Step 4: Stack per-layer final hidden states [num_layers, batch, hidden]
  mlir::MlirOp stacked_h = StackPerLayerFinalHiddenStates(
      builder, rem_curr_h, batch, hidden, to_out);
  return {full_y, stacked_h, std::nullopt};
}

// Evaluates a chunk of k=8 timesteps for a single subsequent layer l (l >= 1)
// during wavefront forward while loop.
//
// What it computes:
// - Slices and applies inter-layer dropout on previous layer chunk output if
// dropout is enabled.
// - Computes input projection GEMM through W_ih_l and adds optional bias.
// - Unrolls 8 forward steps using RunGruForwardSteps.
// - Concatenates step outputs into a single chunk tensor.
//
// Parameters:
// - `builder`: MLIR builder.
// - `curr_chunk`: Chunk output from previous layer l - 1.
// - `body_h_l`: Hidden state for layer l at start of this chunk.
// - `body_step_idx`: Dynamic scalar timestep index.
// - `zero_i64`: Pre-computed i64 zero scalar.
// - `i64`: MLIR i64 type.
// - `layer_weight`: Weight and bias bundle for layer l.
// - `hh_dot_dims`: Contracting dimensions for recurrent GEMM.
// - `ih_dot_dims`: Contracting dimensions for input projection GEMM.
// - `l`: Layer index.
// - `batch`: Batch size.
// - `hidden`: Hidden dimension size.
// - `batch_first`: Memory layout indicator.
// - `concat_dim`: Concatenation dimension along sequence axis.
// - `to_out`: Dtype conversion lambda.
// - `dropout`: Dropout probability.
// - `has_dropout`: True if dropout is enabled.
// - `rand_op`: Optional random uniform tensor for dropout masking.
// - `precision`: Hardware precision level.
//
// Returns:
// - Pair of (updated hidden state for layer l, chunk output for layer l).
// Result of evaluating wavefront forward chunk layer.
struct WavefrontForwardChunkLayerResult {
  mlir::MlirOp next_h_l;
  mlir::MlirOp next_chunk;
};

template <typename ToOutFn>
WavefrontForwardChunkLayerResult EvaluateWavefrontForwardChunkLayer(
    mlir::MlirBuilder& builder, mlir::MlirOp curr_chunk, mlir::MlirOp body_h_l,
    mlir::MlirOp body_step_idx, mlir::MlirOp zero_i64, mlir::IntegerType i64,
    const GruWavefrontLayerWeight& layer_weight,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr ih_dot_dims, int64_t l,
    int64_t batch, int64_t hidden, bool batch_first, int64_t concat_dim,
    ToOutFn to_out, double dropout, bool has_dropout,
    std::optional<mlir::MlirOp> rand_op, mlir::stablehlo::Precision precision) {
  if (has_dropout) {
    llvm::SmallVector<mlir::MlirOp, 4> rand_start_indices;
    llvm::SmallVector<int64_t, 4> rand_slice_sizes;
    if (batch_first) {
      rand_start_indices = {MakeScalarConstant(builder, l - 1, i64), zero_i64,
                            body_step_idx, zero_i64};
      rand_slice_sizes = {1, batch, kGruUnrollFactor, hidden};
    } else {
      rand_start_indices = {MakeScalarConstant(builder, l - 1, i64),
                            body_step_idx, zero_i64, zero_i64};
      rand_slice_sizes = {1, kGruUnrollFactor, batch, hidden};
    }
    mlir::MlirOp rand_chunk = mlir::stablehlo::DynamicSlice(
        *rand_op, rand_start_indices, rand_slice_sizes);
    rand_chunk = mlir::stablehlo::Reshape(
        rand_chunk, GetTensorTypeOrDie(curr_chunk).getShape());
    auto [dropped_chunk, mask_chunk] =
        ApplyInterLayerDropout(curr_chunk, rand_chunk, dropout);
    curr_chunk = dropped_chunk;
  }

  const mlir::Type acc_dtype = GetTensorTypeOrDie(body_h_l).getElementType();
  mlir::MlirOp x_proj_chunk = MixedPrecisionDotGeneral(
      curr_chunk, *layer_weight.w_ih, ih_dot_dims, acc_dtype, precision);
  if (layer_weight.b_ih.has_value()) {
    mlir::MlirOp b_ih = *layer_weight.b_ih;
    mlir::MlirOp b_bcast = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(x_proj_chunk), b_ih, {/*broadcast_dimensions=*/2});
    x_proj_chunk = mlir::stablehlo::Add(x_proj_chunk, b_bcast);
  }

  GruUnrolledForwardSteps steps_l = RunGruForwardSteps(
      x_proj_chunk, /*start_step=*/0, kGruUnrollFactor, body_h_l,
      layer_weight.w_hh, hh_dot_dims, batch, hidden, batch_first, to_out,
      layer_weight.b_hh, precision, /*cache_activations=*/false);
  mlir::MlirOp next_chunk =
      ConcatDim(builder, steps_l.step_outputs, concat_dim);
  return {steps_l.h_final, next_chunk};
}

// Builds the body block of the pipelined wavefront forward while loop.
//
// What it computes:
// - Unpacks carried states: step index, per-layer hidden states, and output
// sequence buffer.
// - Computes layer 0 unrolled chunk steps from input sequence slice.
// - Cascades chunk through layers 1 .. num_layers - 1 using
// EvaluateWavefrontForwardChunkLayer.
// - Updates the top-most layer output sequence buffer using DynamicUpdateSlice.
// - Advances step index by kGruUnrollFactor and returns the next loop carry
// state.
template <typename ToOutFn>
void BuildWavefrontForwardWhileBody(
    mlir::MlirBuilder& builder, mlir::Block* body_block, mlir::Location loc,
    mlir::MlirOp x_proj_0,
    const absl::Span<const GruWavefrontLayerWeight> layer_weights,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr ih_dot_dims, int64_t batch,
    int64_t hidden, int64_t num_layers, bool batch_first, int64_t concat_dim,
    ToOutFn to_out, double dropout, bool has_dropout,
    std::optional<mlir::MlirOp> rand_op, mlir::stablehlo::Precision precision) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::IntegerType i64 = op_builder.getI64Type();
  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64);

  mlir::MlirOp body_step_idx(builder, body_block->getArgument(0));
  std::vector<mlir::MlirOp> body_h(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    body_h[l] = mlir::MlirOp(builder, body_block->getArgument(1 + l));
  }
  mlir::MlirOp body_y(builder, body_block->getArgument(1 + num_layers));

  // Layer 0: slice chunk from x_proj_0
  llvm::SmallVector<mlir::MlirOp, 3> x_start_indices;
  llvm::SmallVector<int64_t, 3> x_slice_sizes;
  if (batch_first) {
    x_start_indices = {zero_i64, body_step_idx, zero_i64};
    x_slice_sizes = {batch, kGruUnrollFactor, 3 * hidden};
  } else {
    x_start_indices = {body_step_idx, zero_i64, zero_i64};
    x_slice_sizes = {kGruUnrollFactor, batch, 3 * hidden};
  }
  mlir::MlirOp x_chunk =
      mlir::stablehlo::DynamicSlice(x_proj_0, x_start_indices, x_slice_sizes);

  GruUnrolledForwardSteps steps_0 = RunGruForwardSteps(
      x_chunk, /*start_step=*/0, kGruUnrollFactor, body_h[0],
      layer_weights[0].w_hh, hh_dot_dims, batch, hidden, batch_first, to_out,
      layer_weights[0].b_hh, precision, /*cache_activations=*/false);
  mlir::MlirOp curr_chunk =
      ConcatDim(builder, steps_0.step_outputs, concat_dim);

  std::vector<mlir::MlirOp> next_h(num_layers);
  next_h[0] = steps_0.h_final;

  // Stream chunk through subsequent layers l = 1 .. num_layers - 1
  for (int64_t l = 1; l < num_layers; ++l) {
    WavefrontForwardChunkLayerResult chunk_l =
        EvaluateWavefrontForwardChunkLayer(
            builder, curr_chunk, body_h[l], body_step_idx, zero_i64, i64,
            layer_weights[l], hh_dot_dims, ih_dot_dims, l, batch, hidden,
            batch_first, concat_dim, to_out, dropout, has_dropout, rand_op,
            precision);
    next_h[l] = chunk_l.next_h_l;
    curr_chunk = chunk_l.next_chunk;
  }

  // Commit final layer chunk into body_y
  llvm::SmallVector<mlir::MlirOp, 3> y_start_indices;
  if (batch_first) {
    y_start_indices = {zero_i64, body_step_idx, zero_i64};
  } else {
    y_start_indices = {body_step_idx, zero_i64, zero_i64};
  }
  mlir::MlirOp next_y =
      mlir::stablehlo::DynamicUpdateSlice(body_y, curr_chunk, y_start_indices);

  mlir::MlirOp k_factor_op = MakeScalarConstant(builder, kGruUnrollFactor, i64);
  mlir::MlirOp next_step_idx = mlir::stablehlo::Add(body_step_idx, k_factor_op);

  llvm::SmallVector<mlir::Value> next_state;
  next_state.reserve(num_layers + 2);
  next_state.push_back(next_step_idx.getValue());
  for (int64_t l = 0; l < num_layers; ++l) {
    next_state.push_back(next_h[l].getValue());
  }
  next_state.push_back(next_y.getValue());
  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_state);
}

// Builds a pipelined wavefront forward pass across stacked recurrent layers (L
// >= 2).
//
// What it computes:
// - Pipelined wavefront execution streams chunks of k=8 timesteps directly from
// layer l to layer l+1
//   within on-chip TPU vector register files (VMEM), eliminating intermediate
//   full sequence tensor roundtrips to off-chip High Bandwidth Memory (HBM).
// - Iterates chunk-by-chunk in a single stablehlo::WhileOp carrying step_idx,
// all layer hidden states body_h[0..L-1], and the final layer sequence output
// buffer body_y.
// - Delegates while body execution to BuildWavefrontForwardWhileBody.
// - Following the while loop, invokes BuildWavefrontForwardRemainder for tail
// steps if seq_len % 8 != 0.
//
// Parameters:
// - `builder`: MLIR builder.
// - `x_proj_0`: Layer 0 pre-projected input sequence tensor [T, B, 3*H].
// - `h_inits`: Initial hidden state tensors for each layer [B, H].
// - `layer_weights`: Weights and biases for each layer.
// - `hh_dot_dims`: Contracting dimensions for recurrent GEMM.
// - `ih_dot_dims`: Contracting dimensions for input projection GEMM.
// - `seq_len`: Total sequence length.
// - `batch`: Batch size.
// - `hidden`: Hidden dimension size.
// - `num_layers`: Number of stacked recurrent layers (>= 2).
// - `batch_first`: Memory layout indicator.
// - `to_out`: Dtype conversion lambda.
// - `dropout`: Dropout probability.
// - `train`: True if training mode.
// - `rand_op`: Optional random uniform tensor for dropout masking.
// - `precision`: Hardware precision level.
//
// Returns:
// - `GruLayerOutputs` struct containing complete multi-layer sequence output
// and stacked final hidden states.
template <typename ToOutFn>
GruLayerOutputs BuildGruPipelinedWavefrontForward(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj_0,
    const absl::Span<const mlir::MlirOp> h_inits,
    const absl::Span<const GruWavefrontLayerWeight> layer_weights,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr ih_dot_dims,
    const int64_t seq_len, const int64_t batch, const int64_t hidden,
    const int64_t num_layers, const bool batch_first, ToOutFn to_out,
    const double dropout, const bool train, std::optional<mlir::MlirOp> rand_op,
    const mlir::stablehlo::Precision precision) {
  const int64_t num_chunks = seq_len / kGruUnrollFactor;
  const int64_t chunked_steps = num_chunks * kGruUnrollFactor;
  const int64_t rem_steps = seq_len % kGruUnrollFactor;
  const int64_t concat_dim = batch_first ? 1 : 0;
  const bool has_dropout =
      (dropout > 0.0 && train && num_layers > 1 && rand_op.has_value());

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = x_proj_0.getValue().getLoc();
  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType i64_scalar_type =
      mlir::RankedTensorType::get({}, i64);

  // Initialize output buffer for the top-most layer [chunked_steps, batch,
  // hidden]
  const mlir::Type out_elem_type =
      GetTensorTypeOrDie(to_out(h_inits[0])).getElementType();
  const llvm::SmallVector<int64_t, 3> y_chunk_shape =
      batch_first ? llvm::SmallVector<int64_t, 3>{batch, chunked_steps, hidden}
                  : llvm::SmallVector<int64_t, 3>{chunked_steps, batch, hidden};
  const mlir::RankedTensorType y_type =
      mlir::RankedTensorType::get(y_chunk_shape, out_elem_type);

  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0f, out_elem_type);
  mlir::MlirOp y_init =
      mlir::stablehlo::BroadcastInDim(y_type, zero_scalar, {});
  mlir::MlirOp step_idx_init = MakeScalarConstant(builder, 0, i64);

  // Assemble loop carry types: step_idx, h[0]..h[L-1], y_chunked
  llvm::SmallVector<mlir::Type> loop_types;
  loop_types.reserve(num_layers + 2);
  loop_types.push_back(i64_scalar_type);
  for (int64_t l = 0; l < num_layers; ++l) {
    loop_types.push_back(h_inits[l].getType());
  }
  loop_types.push_back(y_type);

  llvm::SmallVector<mlir::Value> loop_inits;
  loop_inits.reserve(num_layers + 2);
  loop_inits.push_back(step_idx_init.getValue());
  for (int64_t l = 0; l < num_layers; ++l) {
    loop_inits.push_back(h_inits[l].getValue());
  }
  loop_inits.push_back(y_init.getValue());

  auto while_op =
      mlir::stablehlo::WhileOp::create(op_builder, loc, loop_types, loop_inits);

  // Condition block: continue while step_idx < chunked_steps
  mlir::Block* const cond_block = op_builder.createBlock(&while_op.getCond());
  cond_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(cond_block);

  mlir::MlirOp limit = MakeScalarConstant(builder, chunked_steps, i64);
  const mlir::Value cond =
      mlir::stablehlo::CompareOp::create(
          op_builder, loc, cond_block->getArgument(0), limit.getValue(),
          mlir::stablehlo::ComparisonDirection::LT)
          .getResult();
  mlir::stablehlo::ReturnOp::create(op_builder, loc, cond);

  // Body block: pipeline chunk through all layers concurrently
  mlir::Block* const body_block = op_builder.createBlock(&while_op.getBody());
  body_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(body_block);

  BuildWavefrontForwardWhileBody(
      builder, body_block, loc, x_proj_0, layer_weights, hh_dot_dims,
      ih_dot_dims, batch, hidden, num_layers, batch_first, concat_dim, to_out,
      dropout, has_dropout, rand_op, precision);

  op_builder.setInsertionPointAfter(while_op);

  // Extract final hidden states and chunked sequence output
  std::vector<mlir::MlirOp> final_h(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    final_h[l] = mlir::MlirOp(builder, while_op.getResult(1 + l));
  }
  mlir::MlirOp chunked_y(builder, while_op.getResult(1 + num_layers));

  if (rem_steps == 0) {
    mlir::MlirOp stacked_h =
        StackPerLayerFinalHiddenStates(builder, final_h, batch, hidden, to_out);
    return {chunked_y, stacked_h, std::nullopt};
  }

  // Remainder steps evaluated across stacked wavefront layers
  return BuildWavefrontForwardRemainder(
      builder, x_proj_0, chunked_y, final_h, layer_weights, hh_dot_dims,
      ih_dot_dims, chunked_steps, rem_steps, seq_len, batch, hidden, num_layers,
      batch_first, to_out, dropout, has_dropout, rand_op, precision);
}

// Builds a static forward sweep for bidirectional GRU for short sequences
// (seq_len < kGruUnrollFactor).
//
// What it computes:
// - Evaluates both forward (t = 0 .. T-1) and reverse (t_rev = T-1 .. 0)
// directions step-by-step
//   in a single unrolled pass without WhileOp overhead.
// - Concatenates forward and reverse sequences along feature dimension: [T, B,
// 2*hidden] ([B, T, 2*hidden] if batch_first).
// - Concatenates forward and reverse final hidden states along layer dimension:
// [2, batch, hidden].
// - Optionally returns forward and reverse cached 4H activations for autograd.
//
// Parameters:
// - `builder`: MLIR builder.
// - `x_proj_fwd`: Forward input projection sequence [T, B, 3*H].
// - `x_proj_rev`: Reverse input projection sequence [T, B, 3*H].
// - `h_init_fwd`: Initial hidden state for forward sweep [B, H].
// - `h_init_rev`: Initial hidden state for reverse sweep [B, H].
// - `w_hh_fwd`: Recurrent weight matrix for forward sweep [3*H, H].
// - `w_hh_rev`: Recurrent weight matrix for reverse sweep [3*H, H].
// - `hh_dot_dims`: Contracting dimensions for recurrent GEMM.
// - `seq_len`: Sequence length T (< 8).
// - `batch`: Batch size B.
// - `hidden`: Hidden dimension H.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Dtype conversion lambda.
// - `b_hh_fwd`: Optional recurrent bias for forward direction.
// - `b_hh_rev`: Optional recurrent bias for reverse direction.
// - `precision`: Hardware precision level.
// - `cache_activations`: Whether to cache activations for autograd.
//
// Returns:
// - `GruBidirLayerOutputs` struct containing bidirectional sequence output and
// concatenated final hidden states.
template <typename ToOutFn>
GruBidirLayerOutputs BuildGruBidirStaticForward(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj_fwd,
    mlir::MlirOp x_proj_rev, mlir::MlirOp h_init_fwd, mlir::MlirOp h_init_rev,
    mlir::MlirOp w_hh_fwd, mlir::MlirOp w_hh_rev,
    mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims, int64_t seq_len,
    int64_t batch, int64_t hidden, bool batch_first, ToOutFn to_out,
    std::optional<mlir::MlirOp> b_hh_fwd, std::optional<mlir::MlirOp> b_hh_rev,
    mlir::stablehlo::Precision precision, bool cache_activations) {
  mlir::MlirOp curr_h_fwd = h_init_fwd;
  mlir::MlirOp curr_h_rev = h_init_rev;

  std::vector<mlir::MlirOp> fwd_step_outputs(seq_len);
  std::vector<mlir::MlirOp> rev_step_outputs(seq_len);
  std::vector<mlir::MlirOp> fwd_acts(cache_activations ? seq_len : 0);
  std::vector<mlir::MlirOp> rev_acts(cache_activations ? seq_len : 0);

  // Statically unroll all sequence steps for both forward and reverse sweeps
  for (int64_t t = 0; t < seq_len; ++t) {
    // Step 1: Forward recurrent step at timestep t
    mlir::MlirOp x_fwd_2d =
        SliceStep2D(x_proj_fwd, t, batch, 3 * hidden, batch_first);
    GruStepResults step_fwd =
        ComputeGruStep(x_fwd_2d, curr_h_fwd, w_hh_fwd, hh_dot_dims, batch,
                       hidden, b_hh_fwd, precision);
    curr_h_fwd = step_fwd.h_next;
    fwd_step_outputs[t] =
        ExpandStep3D(to_out(curr_h_fwd), batch, hidden, batch_first);
    if (cache_activations) {
      fwd_acts[t] =
          ExpandStep3D(step_fwd.act_4h, batch, 4 * hidden, batch_first);
    }

    // Step 2: Reverse recurrent step at timestep t_rev = seq_len - 1 - t
    const int64_t t_rev = seq_len - 1 - t;
    mlir::MlirOp x_rev_2d =
        SliceStep2D(x_proj_rev, t_rev, batch, 3 * hidden, batch_first);
    GruStepResults step_rev =
        ComputeGruStep(x_rev_2d, curr_h_rev, w_hh_rev, hh_dot_dims, batch,
                       hidden, b_hh_rev, precision);
    curr_h_rev = step_rev.h_next;
    rev_step_outputs[t_rev] =
        ExpandStep3D(to_out(curr_h_rev), batch, hidden, batch_first);
    if (cache_activations) {
      rev_acts[t_rev] =
          ExpandStep3D(step_rev.act_4h, batch, 4 * hidden, batch_first);
    }
  }

  // Step 3: Concatenate forward and reverse sequences along feature dimension
  const int64_t concat_dim = batch_first ? 1 : 0;
  mlir::MlirOp fwd_seq = ConcatDim(builder, fwd_step_outputs, concat_dim);
  mlir::MlirOp rev_seq = ConcatDim(builder, rev_step_outputs, concat_dim);
  mlir::MlirOp bidir_seq =
      mlir::stablehlo::Concatenate(builder, {fwd_seq, rev_seq}, /*dim=*/2);

  // Step 4: Format and stack forward and reverse final hidden states: [2,
  // batch, hidden]
  mlir::MlirOp final_h_fwd_3d =
      mlir::stablehlo::Reshape(to_out(curr_h_fwd), {1, batch, hidden});
  mlir::MlirOp final_h_rev_3d =
      mlir::stablehlo::Reshape(to_out(curr_h_rev), {1, batch, hidden});
  mlir::MlirOp final_h = mlir::stablehlo::Concatenate(
      builder, {final_h_fwd_3d, final_h_rev_3d}, /*dim=*/0);

  std::optional<mlir::MlirOp> acts_fwd_out = std::nullopt;
  std::optional<mlir::MlirOp> acts_rev_out = std::nullopt;
  if (cache_activations) {
    acts_fwd_out = ConcatDim(builder, fwd_acts, concat_dim);
    acts_rev_out = ConcatDim(builder, rev_acts, concat_dim);
  }

  return {bidir_seq, final_h, acts_fwd_out, acts_rev_out};
}

// Result container for dual-directional chunk step unrolling.
//
// Fields:
// - `h_next_fwd`: Updated forward hidden state at the end of the chunk [B,
// hidden].
// - `h_next_rev`: Updated reverse hidden state at the start of the chunk [B,
// hidden].
// - `fwd_chunk_outputs`: Vector of 8 step output tensors for forward direction.
// - `rev_chunk_outputs`: Vector of 8 step output tensors for reverse direction.
// - `fwd_chunk_acts`: Vector of 8 cached 4H activation tensors for forward
// direction.
// - `rev_chunk_acts`: Vector of 8 cached 4H activation tensors for reverse
// direction.
struct GruBidirStepsResult {
  mlir::MlirOp h_next_fwd;
  mlir::MlirOp h_next_rev;
  std::vector<mlir::MlirOp> fwd_chunk_outputs;
  std::vector<mlir::MlirOp> rev_chunk_outputs;
  std::vector<mlir::MlirOp> fwd_chunk_acts;
  std::vector<mlir::MlirOp> rev_chunk_acts;
};

// Concurrently evaluates kGruUnrollFactor forward and reverse steps for a
// chunk.
//
// What it computes:
// - Simultaneously evaluates forward steps (k = 0 .. 7) and reverse steps
// (rev_k = 7 .. 0)
//   within the same unrolled iteration, maximizing TPU MXU/VPU arithmetic
//   density and instruction cache locality.
// - For each k in [0, 7]:
//   - Slices x_k_fwd from x_chunk_fwd at step k.
//   - Slices x_k_rev from x_chunk_rev at step rev_k = 7 - k.
//   - Evaluates ComputeGruStep for forward direction -> h_next_fwd, act_fwd.
//   - Evaluates ComputeGruStep for reverse direction -> h_next_rev, act_rev.
//   - Formats outputs and optional cached activations for both directions.
//
// Parameters:
// - `x_chunk_fwd`: Forward chunk pre-projection tensor [k, B, 3*H] (or [B, k,
// 3*H]).
// - `x_chunk_rev`: Reverse chunk pre-projection tensor [k, B, 3*H] (or [B, k,
// 3*H]).
// - `curr_h_fwd`: Active forward hidden state [B, hidden].
// - `curr_h_rev`: Active reverse hidden state [B, hidden].
// - `w_hh_fwd`: Forward recurrent weight [3*H, H].
// - `w_hh_rev`: Reverse recurrent weight [3*H, H].
// - `hh_dot_dims`: Contracting dimensions for recurrent GEMM.
// - `batch`: Batch size B.
// - `hidden`: Hidden dimension H.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Dtype conversion lambda.
// - `b_hh_fwd`: Optional forward recurrent bias.
// - `b_hh_rev`: Optional reverse recurrent bias.
// - `precision`: Hardware precision level.
// - `cache_activations`: Whether to cache activations.
//
// Returns:
// - `GruBidirStepsResult` struct containing updated hidden states and chunk
// outputs for both directions.
template <typename ToOutFn>
GruBidirStepsResult RunGruBidirForwardSteps(
    mlir::MlirOp x_chunk_fwd, mlir::MlirOp x_chunk_rev, mlir::MlirOp curr_h_fwd,
    mlir::MlirOp curr_h_rev, mlir::MlirOp w_hh_fwd, mlir::MlirOp w_hh_rev,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    const int64_t batch, const int64_t hidden, const bool batch_first,
    ToOutFn to_out, std::optional<mlir::MlirOp> b_hh_fwd,
    std::optional<mlir::MlirOp> b_hh_rev,
    const mlir::stablehlo::Precision precision, const bool cache_activations) {
  std::vector<mlir::MlirOp> fwd_chunk_outputs(kGruUnrollFactor);
  std::vector<mlir::MlirOp> rev_chunk_outputs(kGruUnrollFactor);
  std::vector<mlir::MlirOp> fwd_chunk_acts;
  std::vector<mlir::MlirOp> rev_chunk_acts;
  if (cache_activations) {
    fwd_chunk_acts.resize(kGruUnrollFactor);
    rev_chunk_acts.resize(kGruUnrollFactor);
  }

  // Concurrent unrolled lockstep evaluation across forward and reverse
  // timesteps
  for (int64_t k = 0; k < kGruUnrollFactor; ++k) {
    // Slices for forward timestep k and reverse timestep rev_k
    mlir::MlirOp x_k_fwd_2d =
        SliceStep2D(x_chunk_fwd, k, batch, 3 * hidden, batch_first);
    const int64_t rev_k = kGruUnrollFactor - 1 - k;
    mlir::MlirOp x_k_rev_2d =
        SliceStep2D(x_chunk_rev, rev_k, batch, 3 * hidden, batch_first);

    // Evaluate forward and reverse cell steps simultaneously
    const GruStepResults fwd_step =
        ComputeGruStep(x_k_fwd_2d, curr_h_fwd, w_hh_fwd, hh_dot_dims, batch,
                       hidden, b_hh_fwd, precision);
    const GruStepResults rev_step =
        ComputeGruStep(x_k_rev_2d, curr_h_rev, w_hh_rev, hh_dot_dims, batch,
                       hidden, b_hh_rev, precision);

    curr_h_fwd = fwd_step.h_next;
    curr_h_rev = rev_step.h_next;

    // Collect step outputs
    fwd_chunk_outputs[k] =
        ExpandStep3D(to_out(curr_h_fwd), batch, hidden, batch_first);
    rev_chunk_outputs[rev_k] =
        ExpandStep3D(to_out(curr_h_rev), batch, hidden, batch_first);

    if (cache_activations) {
      fwd_chunk_acts[k] =
          ExpandStep3D(fwd_step.act_4h, batch, 4 * hidden, batch_first);
      rev_chunk_acts[rev_k] =
          ExpandStep3D(rev_step.act_4h, batch, 4 * hidden, batch_first);
    }
  }

  return {curr_h_fwd,
          curr_h_rev,
          std::move(fwd_chunk_outputs),
          std::move(rev_chunk_outputs),
          std::move(fwd_chunk_acts),
          std::move(rev_chunk_acts)};
}

// Result container for bidirectional GRU remainder step evaluation.
//
// Fields:
// - `final_seq_fwd`: Combined forward sequence output including while loop and
// remainder steps.
// - `final_seq_rev`: Combined reverse sequence output including while loop and
// remainder steps.
// - `final_h_fwd`: Final forward hidden state [B, hidden].
// - `final_h_rev`: Final reverse hidden state [B, hidden].
// - `final_act_fwd`: Optional combined forward cached activations.
// - `final_act_rev`: Optional combined reverse cached activations.
struct GruBidirForwardRemainderResult {
  mlir::MlirOp final_seq_fwd;
  mlir::MlirOp final_seq_rev;
  mlir::MlirOp final_h_fwd;
  mlir::MlirOp final_h_rev;
  std::optional<mlir::MlirOp> final_act_fwd;
  std::optional<mlir::MlirOp> final_act_rev;
};

// Computes tail remainder steps (seq_len % kGruUnrollFactor) for bidirectional
// GRU.
//
// What it computes:
// - Unrolls rem_steps timesteps for both forward (chunked_steps .. seq_len - 1)
// and
//   reverse (rem_steps - 1 down to 0) directions.
// - Concatenates chunked while loop sequence outputs with remainder outputs for
// each direction.
// - Concatenates chunked activations with remainder activations if caching is
// enabled.
//
// Parameters:
// - `builder`: MLIR builder.
// - `x_proj_fwd`: Forward input projection sequence.
// - `x_proj_rev`: Reverse input projection sequence.
// - `h_fwd_chunked`: Forward hidden state at the exit of the while loop.
// - `h_rev_chunked`: Reverse hidden state at the exit of the while loop.
// - `y_fwd_chunked`: Forward sequence output from the while loop.
// - `y_rev_chunked`: Reverse sequence output from the while loop.
// - `act_fwd_chunked`: Optional forward cached activations from while loop.
// - `act_rev_chunked`: Optional reverse cached activations from while loop.
// - `w_hh_fwd`: Forward recurrent weight.
// - `w_hh_rev`: Reverse recurrent weight.
// - `hh_dot_dims`: Contracting dimensions attribute.
// - `chunked_steps`: Completed chunked steps (num_chunks * 8).
// - `rem_steps`: Remainder steps (seq_len % 8).
// - `batch`: Batch size.
// - `hidden`: Hidden dimension.
// - `batch_first`: Memory layout indicator.
// - `concat_dim`: Sequence concatenation dimension (1 if batch_first, else 0).
// - `to_out`: Dtype conversion lambda.
// - `b_hh_fwd`: Optional forward recurrent bias.
// - `b_hh_rev`: Optional reverse recurrent bias.
// - `precision`: Hardware precision level.
// - `cache_activations`: Whether to cache activations.
//
// Returns:
// - `GruBidirForwardRemainderResult` struct containing combined sequences and
// final hidden states.
template <typename ToOutFn>
GruBidirForwardRemainderResult BuildGruBidirForwardRemainder(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj_fwd,
    mlir::MlirOp x_proj_rev, mlir::MlirOp h_fwd_chunked,
    mlir::MlirOp h_rev_chunked, mlir::MlirOp y_fwd_chunked,
    mlir::MlirOp y_rev_chunked, std::optional<mlir::MlirOp> act_fwd_chunked,
    std::optional<mlir::MlirOp> act_rev_chunked, mlir::MlirOp w_hh_fwd,
    mlir::MlirOp w_hh_rev,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    const int64_t chunked_steps, const int64_t rem_steps, const int64_t batch,
    const int64_t hidden, const bool batch_first, const int64_t concat_dim,
    ToOutFn to_out, std::optional<mlir::MlirOp> b_hh_fwd,
    std::optional<mlir::MlirOp> b_hh_rev,
    const mlir::stablehlo::Precision precision, const bool cache_activations) {
  std::vector<mlir::MlirOp> rem_outputs_fwd(rem_steps);
  std::vector<mlir::MlirOp> rem_outputs_rev(rem_steps);
  std::vector<mlir::MlirOp> rem_acts_fwd;
  std::vector<mlir::MlirOp> rem_acts_rev;
  if (cache_activations) {
    rem_acts_fwd.resize(rem_steps);
    rem_acts_rev.resize(rem_steps);
  }

  mlir::MlirOp final_h_fwd = h_fwd_chunked;
  mlir::MlirOp final_h_rev = h_rev_chunked;

  // Unroll remaining tail timesteps for both directions
  for (int64_t r = 0; r < rem_steps; ++r) {
    const int64_t fwd_t = chunked_steps + r;
    mlir::MlirOp x_fwd_2d =
        SliceStep2D(x_proj_fwd, fwd_t, batch, 3 * hidden, batch_first);

    const int64_t rev_t = rem_steps - 1 - r;
    mlir::MlirOp x_rev_2d =
        SliceStep2D(x_proj_rev, rev_t, batch, 3 * hidden, batch_first);

    const GruStepResults fwd_step =
        ComputeGruStep(x_fwd_2d, final_h_fwd, w_hh_fwd, hh_dot_dims, batch,
                       hidden, b_hh_fwd, precision);
    const GruStepResults rev_step =
        ComputeGruStep(x_rev_2d, final_h_rev, w_hh_rev, hh_dot_dims, batch,
                       hidden, b_hh_rev, precision);

    final_h_fwd = fwd_step.h_next;
    final_h_rev = rev_step.h_next;

    rem_outputs_fwd[r] =
        ExpandStep3D(to_out(final_h_fwd), batch, hidden, batch_first);
    rem_outputs_rev[rev_t] =
        ExpandStep3D(to_out(final_h_rev), batch, hidden, batch_first);

    if (cache_activations) {
      rem_acts_fwd[r] =
          ExpandStep3D(fwd_step.act_4h, batch, 4 * hidden, batch_first);
      rem_acts_rev[rev_t] =
          ExpandStep3D(rev_step.act_4h, batch, 4 * hidden, batch_first);
    }
  }

  // Concatenate forward while loop outputs with forward remainder outputs
  mlir::MlirOp rem_out_fwd = ConcatDim(builder, rem_outputs_fwd, concat_dim);
  mlir::MlirOp final_seq_fwd =
      ConcatDim(builder, {y_fwd_chunked, rem_out_fwd}, concat_dim);

  // Concatenate reverse while loop outputs with reverse remainder outputs
  mlir::MlirOp rem_out_rev = ConcatDim(builder, rem_outputs_rev, concat_dim);
  mlir::MlirOp final_seq_rev =
      ConcatDim(builder, {rem_out_rev, y_rev_chunked}, concat_dim);

  std::optional<mlir::MlirOp> final_act_fwd = std::nullopt;
  std::optional<mlir::MlirOp> final_act_rev = std::nullopt;
  if (cache_activations) {
    mlir::MlirOp rem_act_out_fwd = ConcatDim(builder, rem_acts_fwd, concat_dim);
    final_act_fwd =
        ConcatDim(builder, {*act_fwd_chunked, rem_act_out_fwd}, concat_dim);

    mlir::MlirOp rem_act_out_rev = ConcatDim(builder, rem_acts_rev, concat_dim);
    final_act_rev =
        ConcatDim(builder, {rem_act_out_rev, *act_rev_chunked}, concat_dim);
  }

  return {final_seq_fwd, final_seq_rev, final_h_fwd,
          final_h_rev,   final_act_fwd, final_act_rev};
}

// Builds a bidirectional GRU forward layer using chunked recurrence inside a
// stablehlo::WhileOp.
//
// What it computes:
// - Unrolls bidirectional recurrence in chunks of k=8 timesteps inside a single
// while loop.
// - At each loop iteration (step_idx = 0, 8, 16, ...):
//   - Slices forward chunk at step_idx (t = step_idx .. step_idx + 7).
//   - Slices reverse chunk at rev_x_start_idx = (seq_len - 8) - step_idx.
//   - Concurrently executes RunGruBidirForwardSteps for both directions in
//   lockstep.
//   - Commits forward chunk outputs into body_y_fwd at body_step_idx.
//   - Commits reverse chunk outputs into body_y_rev at rev_y_write_idx =
//   (chunked_steps - 8) - body_step_idx.
//   - If caching, commits forward and reverse activations into body_act_fwd and
//   body_act_rev.
//   - Advances body_step_idx by 8.
// - Remainder handoff:
//   - If rem_steps == 0: returns while loop results.
//   - If rem_steps > 0: invokes BuildGruBidirForwardRemainder to evaluate
//   remainder and concatenate.
// - Joins forward and reverse sequence outputs along feature dimension (dim 2)
// into [T, B, 2*H].
// - Stacks forward and reverse final hidden states along layer dimension (dim
// 0) into [2, B, H].
//
// Parameters:
// - `builder`: MLIR builder.
// - `x_proj_fwd`: Forward input projection sequence.
// - `x_proj_rev`: Reverse input projection sequence.
// - `h_init_fwd`: Initial forward hidden state [B, H].
// - `h_init_rev`: Initial reverse hidden state [B, H].
// - `w_hh_fwd`: Forward recurrent weight [3*H, H].
// - `w_hh_rev`: Reverse recurrent weight [3*H, H].
// - `hh_dot_dims`: Recurrent GEMM contraction dimensions.
// - `seq_len`: Sequence length T.
// - `batch`: Batch size B.
// - `hidden`: Hidden dimension H.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Dtype conversion lambda.
// - `b_hh_fwd`: Optional forward recurrent bias.
// - `b_hh_rev`: Optional reverse recurrent bias.
// - `precision`: Hardware precision level.
// - `cache_activations`: Whether to cache activations.
//
// Returns:
// - `GruBidirLayerOutputs` struct containing concatenated sequence output and
// stacked final hidden states.
// Builds the body block for the chunked bidirectional forward while loop.
//
// What it computes:
// - Dynamically slices forward input projection chunk at body_step_idx.
// - Dynamically slices reverse input projection chunk at (seq_len - 8) -
// body_step_idx.
// - Executes forward and reverse recurrent steps concurrently via
// RunGruBidirForwardSteps.
// - Commits forward chunk outputs into body_y_fwd and reverse chunk outputs
// into body_y_rev.
// - Optionally commits forward and reverse cached activations if
// cache_activations is true.
// - Increments body_step_idx by kGruUnrollFactor (8) and returns the updated
// carry values.
template <typename ToOutFn>
void BuildGruBidirChunkedForwardWhileBody(
    mlir::MlirBuilder& builder, mlir::Block* body_block, mlir::Location loc,
    mlir::MlirOp x_proj_fwd, mlir::MlirOp x_proj_rev, mlir::MlirOp w_hh_fwd,
    mlir::MlirOp w_hh_rev,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims, int64_t seq_len,
    int64_t chunked_steps, int64_t batch, int64_t hidden, bool batch_first,
    int64_t concat_dim, ToOutFn to_out, std::optional<mlir::MlirOp> b_hh_fwd,
    std::optional<mlir::MlirOp> b_hh_rev, mlir::stablehlo::Precision precision,
    bool cache_activations) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::IntegerType i64 = op_builder.getI64Type();

  mlir::MlirOp body_step_idx(builder, body_block->getArgument(0));
  mlir::MlirOp body_h_fwd(builder, body_block->getArgument(1));
  mlir::MlirOp body_h_rev(builder, body_block->getArgument(2));
  mlir::MlirOp body_y_fwd(builder, body_block->getArgument(3));
  mlir::MlirOp body_y_rev(builder, body_block->getArgument(4));
  std::optional<mlir::MlirOp> body_act_fwd;
  std::optional<mlir::MlirOp> body_act_rev;
  if (cache_activations) {
    body_act_fwd = mlir::MlirOp(builder, body_block->getArgument(5));
    body_act_rev = mlir::MlirOp(builder, body_block->getArgument(6));
  }

  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64);
  const llvm::SmallVector<int64_t, 3> x_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kGruUnrollFactor, 3 * hidden}
          : llvm::SmallVector<int64_t, 3>{kGruUnrollFactor, batch, 3 * hidden};

  // Dynamic slice for forward chunk
  const llvm::SmallVector<mlir::MlirOp, 3> fwd_x_start_indices =
      batch_first ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, body_step_idx,
                                                       zero_i64}
                  : llvm::SmallVector<mlir::MlirOp, 3>{body_step_idx, zero_i64,
                                                       zero_i64};
  mlir::MlirOp x_chunk_fwd = mlir::stablehlo::DynamicSlice(
      x_proj_fwd, fwd_x_start_indices, x_slice_sizes);

  // Dynamic slice for reverse chunk: start index is (seq_len - k) -
  // body_step_idx
  mlir::MlirOp const_T_minus_K =
      MakeScalarConstant(builder, seq_len - kGruUnrollFactor, i64);
  mlir::MlirOp rev_x_start_idx =
      mlir::stablehlo::Subtract(const_T_minus_K, body_step_idx);
  const llvm::SmallVector<mlir::MlirOp, 3> rev_x_start_indices =
      batch_first
          ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, rev_x_start_idx,
                                               zero_i64}
          : llvm::SmallVector<mlir::MlirOp, 3>{rev_x_start_idx, zero_i64,
                                               zero_i64};
  mlir::MlirOp x_chunk_rev = mlir::stablehlo::DynamicSlice(
      x_proj_rev, rev_x_start_indices, x_slice_sizes);

  // Unroll forward and reverse steps concurrently in vector registers
  GruBidirStepsResult step_res = RunGruBidirForwardSteps(
      x_chunk_fwd, x_chunk_rev, body_h_fwd, body_h_rev, w_hh_fwd, w_hh_rev,
      hh_dot_dims, batch, hidden, batch_first, to_out, b_hh_fwd, b_hh_rev,
      precision, cache_activations);

  mlir::MlirOp curr_h_fwd = step_res.h_next_fwd;
  mlir::MlirOp curr_h_rev = step_res.h_next_rev;

  // Commit forward chunk outputs into body_y_fwd
  mlir::MlirOp fwd_chunk_out =
      ConcatDim(builder, step_res.fwd_chunk_outputs, concat_dim);
  const llvm::SmallVector<mlir::MlirOp, 3> fwd_y_start_indices =
      batch_first ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, body_step_idx,
                                                       zero_i64}
                  : llvm::SmallVector<mlir::MlirOp, 3>{body_step_idx, zero_i64,
                                                       zero_i64};
  mlir::MlirOp next_y_fwd = mlir::stablehlo::DynamicUpdateSlice(
      body_y_fwd, fwd_chunk_out, fwd_y_start_indices);

  // Commit reverse chunk outputs into body_y_rev
  mlir::MlirOp rev_chunk_out =
      ConcatDim(builder, step_res.rev_chunk_outputs, concat_dim);
  mlir::MlirOp const_chunked_minus_K =
      MakeScalarConstant(builder, chunked_steps - kGruUnrollFactor, i64);
  mlir::MlirOp rev_y_write_idx =
      mlir::stablehlo::Subtract(const_chunked_minus_K, body_step_idx);
  const llvm::SmallVector<mlir::MlirOp, 3> rev_y_start_indices =
      batch_first
          ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, rev_y_write_idx,
                                               zero_i64}
          : llvm::SmallVector<mlir::MlirOp, 3>{rev_y_write_idx, zero_i64,
                                               zero_i64};
  mlir::MlirOp next_y_rev = mlir::stablehlo::DynamicUpdateSlice(
      body_y_rev, rev_chunk_out, rev_y_start_indices);

  mlir::MlirOp k_factor_op = MakeScalarConstant(builder, kGruUnrollFactor, i64);
  mlir::MlirOp next_step_idx = mlir::stablehlo::Add(body_step_idx, k_factor_op);

  llvm::SmallVector<mlir::Value> next_state;
  if (cache_activations) {
    mlir::MlirOp fwd_act_chunk_out =
        ConcatDim(builder, step_res.fwd_chunk_acts, concat_dim);
    mlir::MlirOp rev_act_chunk_out =
        ConcatDim(builder, step_res.rev_chunk_acts, concat_dim);
    mlir::MlirOp next_act_fwd = mlir::stablehlo::DynamicUpdateSlice(
        *body_act_fwd, fwd_act_chunk_out, fwd_y_start_indices);
    mlir::MlirOp next_act_rev = mlir::stablehlo::DynamicUpdateSlice(
        *body_act_rev, rev_act_chunk_out, rev_y_start_indices);
    next_state = {next_step_idx.getValue(), curr_h_fwd.getValue(),
                  curr_h_rev.getValue(),    next_y_fwd.getValue(),
                  next_y_rev.getValue(),    next_act_fwd.getValue(),
                  next_act_rev.getValue()};
  } else {
    next_state = {next_step_idx.getValue(), curr_h_fwd.getValue(),
                  curr_h_rev.getValue(), next_y_fwd.getValue(),
                  next_y_rev.getValue()};
  }
  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_state);
}

// Finalizes bidirectional GRU forward outputs after chunked while loop
// execution.
//
// What it computes:
// - If rem_steps == 0: adopts while loop outputs directly.
// - If rem_steps > 0: executes BuildGruBidirForwardRemainder to evaluate tail
// timesteps.
// - Concatenates forward and reverse sequence outputs along feature dimension
// (dim 2).
// - Stacks forward and reverse final hidden states along layer dimension (dim
// 0).
template <typename ToOutFn>
GruBidirLayerOutputs FinalizeGruBidirForwardOutputs(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj_fwd,
    mlir::MlirOp x_proj_rev, mlir::MlirOp h_fwd_after, mlir::MlirOp h_rev_after,
    mlir::MlirOp y_fwd_chunked, mlir::MlirOp y_rev_chunked,
    std::optional<mlir::MlirOp> act_fwd_chunked,
    std::optional<mlir::MlirOp> act_rev_chunked, mlir::MlirOp w_hh_fwd,
    mlir::MlirOp w_hh_rev,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    int64_t chunked_steps, int64_t rem_steps, int64_t batch, int64_t hidden,
    bool batch_first, int64_t concat_dim, int64_t feature_dim, ToOutFn to_out,
    std::optional<mlir::MlirOp> b_hh_fwd, std::optional<mlir::MlirOp> b_hh_rev,
    mlir::stablehlo::Precision precision, bool cache_activations) {
  mlir::MlirOp final_seq_fwd;
  mlir::MlirOp final_seq_rev;
  mlir::MlirOp final_h_fwd;
  mlir::MlirOp final_h_rev;
  std::optional<mlir::MlirOp> final_act_fwd;
  std::optional<mlir::MlirOp> final_act_rev;

  if (rem_steps == 0) {
    final_seq_fwd = y_fwd_chunked;
    final_seq_rev = y_rev_chunked;
    final_h_fwd = h_fwd_after;
    final_h_rev = h_rev_after;
    final_act_fwd = act_fwd_chunked;
    final_act_rev = act_rev_chunked;
  } else {
    GruBidirForwardRemainderResult rem = BuildGruBidirForwardRemainder(
        builder, x_proj_fwd, x_proj_rev, h_fwd_after, h_rev_after,
        y_fwd_chunked, y_rev_chunked, act_fwd_chunked, act_rev_chunked,
        w_hh_fwd, w_hh_rev, hh_dot_dims, chunked_steps, rem_steps, batch,
        hidden, batch_first, concat_dim, to_out, b_hh_fwd, b_hh_rev, precision,
        cache_activations);
    final_seq_fwd = rem.final_seq_fwd;
    final_seq_rev = rem.final_seq_rev;
    final_h_fwd = rem.final_h_fwd;
    final_h_rev = rem.final_h_rev;
    final_act_fwd = rem.final_act_fwd;
    final_act_rev = rem.final_act_rev;
  }

  // Concatenate forward and reverse sequence outputs along feature dimension
  // [T, B, 2*H]
  mlir::MlirOp y_layer =
      ConcatDim(builder, {final_seq_fwd, final_seq_rev}, feature_dim);
  // Stack forward and reverse final hidden states [2, batch, hidden]
  mlir::MlirOp final_h_fwd_3d =
      mlir::stablehlo::Reshape(to_out(final_h_fwd), {1, batch, hidden});
  mlir::MlirOp final_h_rev_3d =
      mlir::stablehlo::Reshape(to_out(final_h_rev), {1, batch, hidden});
  mlir::MlirOp final_h = mlir::stablehlo::Concatenate(
      builder, {final_h_fwd_3d, final_h_rev_3d}, /*dim=*/0);

  return {y_layer, final_h, final_act_fwd, final_act_rev};
}

// Builds a bidirectional GRU forward layer using chunked recurrence inside a
// stablehlo::WhileOp.
//
// What it computes:
// - Unrolls bidirectional recurrence in chunks of k=8 timesteps inside a single
// while loop.
// - Delegates while body execution to BuildGruBidirChunkedForwardWhileBody.
// - Delegates remainder evaluation and output packaging to
// FinalizeGruBidirForwardOutputs.
//
// Parameters:
// - `builder`: MLIR builder.
// - `x_proj_fwd`: Forward input projection sequence.
// - `x_proj_rev`: Reverse input projection sequence.
// - `h_init_fwd`: Initial forward hidden state [B, H].
// - `h_init_rev`: Initial reverse hidden state [B, H].
// - `w_hh_fwd`: Forward recurrent weight [3*H, H].
// - `w_hh_rev`: Reverse recurrent weight [3*H, H].
// - `hh_dot_dims`: Recurrent GEMM contraction dimensions.
// - `seq_len`: Sequence length T.
// - `batch`: Batch size B.
// - `hidden`: Hidden dimension H.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Dtype conversion lambda.
// - `b_hh_fwd`: Optional forward recurrent bias.
// - `b_hh_rev`: Optional reverse recurrent bias.
// - `precision`: Hardware precision level.
// - `cache_activations`: Whether to cache activations.
//
// Returns:
// - `GruBidirLayerOutputs` struct containing concatenated sequence output and
// stacked final hidden states.
template <typename ToOutFn>
GruBidirLayerOutputs BuildGruBidirChunkedForward(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj_fwd,
    mlir::MlirOp x_proj_rev, mlir::MlirOp h_init_fwd, mlir::MlirOp h_init_rev,
    mlir::MlirOp w_hh_fwd, mlir::MlirOp w_hh_rev,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    const int64_t seq_len, const int64_t batch, const int64_t hidden,
    const bool batch_first, ToOutFn to_out,
    std::optional<mlir::MlirOp> b_hh_fwd, std::optional<mlir::MlirOp> b_hh_rev,
    const mlir::stablehlo::Precision precision, const bool cache_activations) {
  const int64_t num_chunks = seq_len / kGruUnrollFactor;
  const int64_t chunked_steps = num_chunks * kGruUnrollFactor;
  const int64_t rem_steps = seq_len % kGruUnrollFactor;
  const int64_t concat_dim = batch_first ? 1 : 0;
  const int64_t feature_dim = 2;

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = x_proj_fwd.getValue().getLoc();
  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType i64_scalar_type =
      mlir::RankedTensorType::get({}, i64);

  // Initialize output buffers for forward and reverse sweeps [chunked_steps,
  // batch, hidden]
  const mlir::Type out_elem_type =
      GetTensorTypeOrDie(to_out(h_init_fwd)).getElementType();
  const llvm::SmallVector<int64_t, 3> y_chunk_shape =
      batch_first ? llvm::SmallVector<int64_t, 3>{batch, chunked_steps, hidden}
                  : llvm::SmallVector<int64_t, 3>{chunked_steps, batch, hidden};
  const mlir::RankedTensorType y_type =
      mlir::RankedTensorType::get(y_chunk_shape, out_elem_type);

  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0f, out_elem_type);
  mlir::MlirOp y_fwd_init =
      mlir::stablehlo::BroadcastInDim(y_type, zero_scalar, {});
  mlir::MlirOp y_rev_init =
      mlir::stablehlo::BroadcastInDim(y_type, zero_scalar, {});

  mlir::MlirOp step_idx_init = MakeScalarConstant(builder, 0, i64);

  // Initialize activation buffers [chunked_steps, batch, 4*hidden] if training
  const mlir::Type acc_elem_type =
      GetTensorTypeOrDie(h_init_fwd).getElementType();
  const llvm::SmallVector<int64_t, 3> act_chunk_shape =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, chunked_steps, 4 * hidden}
          : llvm::SmallVector<int64_t, 3>{chunked_steps, batch, 4 * hidden};
  const mlir::RankedTensorType act_type =
      mlir::RankedTensorType::get(act_chunk_shape, acc_elem_type);
  mlir::MlirOp zero_acc_scalar =
      MakeScalarConstant(builder, 0.0f, acc_elem_type);
  mlir::MlirOp act_fwd_init =
      mlir::stablehlo::BroadcastInDim(act_type, zero_acc_scalar, {});
  mlir::MlirOp act_rev_init =
      mlir::stablehlo::BroadcastInDim(act_type, zero_acc_scalar, {});

  llvm::SmallVector<mlir::Type> loop_types;
  llvm::SmallVector<mlir::Value> loop_inits;
  if (cache_activations) {
    loop_types = {i64_scalar_type,
                  h_init_fwd.getType(),
                  h_init_rev.getType(),
                  y_type,
                  y_type,
                  act_type,
                  act_type};
    loop_inits = {step_idx_init.getValue(), h_init_fwd.getValue(),
                  h_init_rev.getValue(),    y_fwd_init.getValue(),
                  y_rev_init.getValue(),    act_fwd_init.getValue(),
                  act_rev_init.getValue()};
  } else {
    loop_types = {i64_scalar_type, h_init_fwd.getType(), h_init_rev.getType(),
                  y_type, y_type};
    loop_inits = {step_idx_init.getValue(), h_init_fwd.getValue(),
                  h_init_rev.getValue(), y_fwd_init.getValue(),
                  y_rev_init.getValue()};
  }

  // Construct stablehlo::WhileOp for chunked bidirectional forward
  auto while_op =
      mlir::stablehlo::WhileOp::create(op_builder, loc, loop_types, loop_inits);

  // Condition block: continue while step_idx < chunked_steps
  mlir::Block* const cond_block = op_builder.createBlock(&while_op.getCond());
  cond_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(cond_block);

  mlir::MlirOp limit = MakeScalarConstant(builder, chunked_steps, i64);
  const mlir::Value cond =
      mlir::stablehlo::CompareOp::create(
          op_builder, loc, cond_block->getArgument(0), limit.getValue(),
          mlir::stablehlo::ComparisonDirection::LT)
          .getResult();
  mlir::stablehlo::ReturnOp::create(op_builder, loc, cond);

  // Body block: evaluate forward chunk and reverse chunk concurrently
  mlir::Block* const body_block = op_builder.createBlock(&while_op.getBody());
  body_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(body_block);

  BuildGruBidirChunkedForwardWhileBody(
      builder, body_block, loc, x_proj_fwd, x_proj_rev, w_hh_fwd, w_hh_rev,
      hh_dot_dims, seq_len, chunked_steps, batch, hidden, batch_first,
      concat_dim, to_out, b_hh_fwd, b_hh_rev, precision, cache_activations);

  op_builder.setInsertionPointAfter(while_op);

  // Extract while loop carried states
  mlir::MlirOp h_fwd_after(builder, while_op.getResult(1));
  mlir::MlirOp h_rev_after(builder, while_op.getResult(2));
  mlir::MlirOp y_fwd_chunked(builder, while_op.getResult(3));
  mlir::MlirOp y_rev_chunked(builder, while_op.getResult(4));
  std::optional<mlir::MlirOp> act_fwd_chunked;
  std::optional<mlir::MlirOp> act_rev_chunked;
  if (cache_activations) {
    act_fwd_chunked = mlir::MlirOp(builder, while_op.getResult(5));
    act_rev_chunked = mlir::MlirOp(builder, while_op.getResult(6));
  }

  return FinalizeGruBidirForwardOutputs(
      builder, x_proj_fwd, x_proj_rev, h_fwd_after, h_rev_after, y_fwd_chunked,
      y_rev_chunked, act_fwd_chunked, act_rev_chunked, w_hh_fwd, w_hh_rev,
      hh_dot_dims, chunked_steps, rem_steps, batch, hidden, batch_first,
      concat_dim, feature_dim, to_out, b_hh_fwd, b_hh_rev, precision,
      cache_activations);
}

// Dispatcher for bidirectional GRU layer forward evaluation.
//
// What it computes:
// - If seq_len < kGruUnrollFactor: delegates to BuildGruBidirStaticForward for
// static unrolled execution.
// - Otherwise: delegates to BuildGruBidirChunkedForward for chunked execution
// in stablehlo::WhileOp.
//
// Parameters:
// - `builder`: MLIR builder.
// - `x_proj_fwd`: Forward input projection sequence.
// - `x_proj_rev`: Reverse input projection sequence.
// - `h_init_fwd`: Forward initial hidden state [B, H].
// - `h_init_rev`: Reverse initial hidden state [B, H].
// - `w_hh_fwd`: Forward recurrent weight [3*H, H].
// - `w_hh_rev`: Reverse recurrent weight [3*H, H].
// - `hh_dot_dims`: Contracting dimensions for recurrent GEMM.
// - `seq_len`: Sequence length.
// - `batch`: Batch size.
// - `hidden`: Hidden dimension size.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Dtype conversion lambda.
// - `b_hh_fwd`: Optional forward recurrent bias.
// - `b_hh_rev`: Optional reverse recurrent bias.
// - `precision`: Hardware precision level.
// - `cache_activations`: Whether to cache activations for autograd.
//
// Returns:
// - `GruBidirLayerOutputs` struct containing bidirectional sequence outputs and
// final hidden states.
template <typename ToOutFn>
GruBidirLayerOutputs BuildGruBidirLayerForward(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj_fwd,
    mlir::MlirOp x_proj_rev, mlir::MlirOp h_init_fwd, mlir::MlirOp h_init_rev,
    mlir::MlirOp w_hh_fwd, mlir::MlirOp w_hh_rev,
    mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims, int64_t seq_len,
    int64_t batch, int64_t hidden, bool batch_first, ToOutFn to_out,
    std::optional<mlir::MlirOp> b_hh_fwd, std::optional<mlir::MlirOp> b_hh_rev,
    mlir::stablehlo::Precision precision, bool cache_activations) {
  if (seq_len < kGruUnrollFactor) {
    return BuildGruBidirStaticForward(
        builder, x_proj_fwd, x_proj_rev, h_init_fwd, h_init_rev, w_hh_fwd,
        w_hh_rev, hh_dot_dims, seq_len, batch, hidden, batch_first, to_out,
        b_hh_fwd, b_hh_rev, precision, cache_activations);
  }
  return BuildGruBidirChunkedForward(
      builder, x_proj_fwd, x_proj_rev, h_init_fwd, h_init_rev, w_hh_fwd,
      w_hh_rev, hh_dot_dims, seq_len, batch, hidden, batch_first, to_out,
      b_hh_fwd, b_hh_rev, precision, cache_activations);
}

// Struct encapsulating backward outputs for a single unidirectional layer.
//
// Fields:
// - `grad_x`: Sequence gradient w.r.t. layer input [T, B, in_dim] ([B, T,
// in_dim] if batch_first).
// - `grad_h0`: Gradient w.r.t. initial hidden state [1, B, hidden].
// - `grad_w_ih`: Accumulated gradient w.r.t. input-hidden weights [3*hidden,
// in_dim].
// - `grad_w_hh`: Accumulated gradient w.r.t. recurrent weights [3*hidden,
// hidden].
// - `grad_b_ih`: Optional accumulated gradient w.r.t. input bias [3*hidden].
// - `grad_b_hh`: Optional accumulated gradient w.r.t. recurrent bias
// [3*hidden].
struct GruLayerBackwardOutputs {
  mlir::MlirOp grad_x;
  mlir::MlirOp grad_h0;
  mlir::MlirOp grad_w_ih;
  mlir::MlirOp grad_w_hh;
  std::optional<mlir::MlirOp> grad_b_ih;
  std::optional<mlir::MlirOp> grad_b_hh;
};

// Struct encapsulating backward outputs for a bidirectional layer.
//
// Fields:
// - `grad_x`: Combined sequence gradient w.r.t. layer input (grad_x_fwd +
// grad_x_rev) [T, B, in_dim].
// - `grad_h0`: Concatenated gradients w.r.t. forward and reverse initial hidden
// states [2, B, hidden].
// - `grad_w_ih_fwd`: Gradient w.r.t. forward input-hidden weight [3*hidden,
// in_dim].
// - `grad_w_hh_fwd`: Gradient w.r.t. forward recurrent weight [3*hidden,
// hidden].
// - `grad_b_ih_fwd`: Optional gradient w.r.t. forward input bias [3*hidden].
// - `grad_b_hh_fwd`: Optional gradient w.r.t. forward recurrent bias
// [3*hidden].
// - `grad_w_ih_rev`: Gradient w.r.t. reverse input-hidden weight [3*hidden,
// in_dim].
// - `grad_w_hh_rev`: Gradient w.r.t. reverse recurrent weight [3*hidden,
// hidden].
// - `grad_b_ih_rev`: Optional gradient w.r.t. reverse input bias [3*hidden].
// - `grad_b_hh_rev`: Optional gradient w.r.t. reverse recurrent bias
// [3*hidden].
struct GruBidirLayerBackwardOutputs {
  mlir::MlirOp grad_x;
  mlir::MlirOp grad_h0;
  mlir::MlirOp grad_w_ih_fwd;
  mlir::MlirOp grad_w_hh_fwd;
  std::optional<mlir::MlirOp> grad_b_ih_fwd;
  std::optional<mlir::MlirOp> grad_b_hh_fwd;
  mlir::MlirOp grad_w_ih_rev;
  mlir::MlirOp grad_w_hh_rev;
  std::optional<mlir::MlirOp> grad_b_ih_rev;
  std::optional<mlir::MlirOp> grad_b_hh_rev;
};

// Computes static backward pass (reverse BPTT) for a single unidirectional GRU
// layer for short sequences (seq_len < kGruUnrollFactor).
//
// What it computes:
// - Sweeps backwards through time t = seq_len - 1 down to 0:
//   1. Sums upstream output sequence gradient grad_y[t] and incoming recurrent
//   hidden gradient delta_h_next: delta_h = grad_y_t + delta_h_next.
//   2. Extracts previous hidden state h_prev = (t == 0 ? h_init : y_out[t-1]).
//   3. Evaluates ComputeGruBackwardStep -> {delta_pre_ig, delta_pre_hg,
//   delta_h_prev}.
//   4. Updates delta_h_next = delta_h_prev.
//   5. Projects input gradient dx_t = delta_pre_ig @ W_ih [B, in_dim].
// - Computes full parameter gradients via 2D matrix contractions:
//   - grad_w_ih = delta_pre_ig_2d^T @ layer_in_2d  [3*H, in_dim]
//   - grad_w_hh = delta_pre_hg_2d^T @ h_prev_2d    [3*H, hidden]
// - If has_biases is true:
//   - Reduces delta_pre_ig_2d and delta_pre_hg_2d along dimension 0 (batch
//   dimension) to form grad_b_ih and grad_b_hh [3*H].
//
// Parameters:
// - `builder`: MLIR builder.
// - `grad_y`: Upstream output sequence gradient [T, B, H].
// - `grad_hy`: Upstream final hidden state gradient [1, B, H].
// - `layer_in`: Input sequence tensor to this layer [T, B, in_dim].
// - `h_init`: Initial hidden state for this layer [B, H].
// - `y_out`: Output sequence produced during forward pass [T, B, H].
// - `act_4h_seq`: Cached forward gate activations [T, B, 4*H].
// - `w_ih`: Input-hidden weight matrix [3*H, in_dim].
// - `w_hh`: Recurrent weight matrix [3*H, H].
// - `has_biases`: Whether bias vectors are enabled.
// - `seq_len`: Sequence length T (< 8).
// - `batch`: Batch size B.
// - `hidden`: Hidden dimension H.
// - `in_dim`: Feature dimension of layer_in.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Dtype conversion lambda.
// - `zero_const`: Zero scalar constant for summation reductions.
// - `sum_reduce_builder`: Region builder lambda for reduction operations.
// - `precision`: Hardware precision level.
//
// Returns:
// - `GruLayerBackwardOutputs` struct containing grad_x, grad_h0, grad_w_ih,
// grad_w_hh, and optional bias grads.
template <typename ToOutFn, typename ReduceBuilderFn>
GruLayerBackwardOutputs ComputeLayerBackwardStatic(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_y, mlir::MlirOp grad_hy,
    mlir::MlirOp layer_in, mlir::MlirOp h_init, mlir::MlirOp y_out,
    mlir::MlirOp act_4h_seq, mlir::MlirOp w_ih, mlir::MlirOp w_hh,
    bool has_biases, int64_t seq_len, int64_t batch, int64_t hidden,
    int64_t in_dim, bool batch_first, ToOutFn to_out, mlir::MlirOp zero_const,
    ReduceBuilderFn sum_reduce_builder, mlir::stablehlo::Precision precision) {
  mlir::MLIRContext& ctx = builder.getContext();
  const auto hh_bwd_dot_dims = MakeDotDims(&ctx, {1}, {0});
  const auto x_bwd_dot_dims = MakeDotDims(&ctx, {1}, {0});
  const auto w_dot_dims = MakeDotDims(&ctx, {0}, {0});

  const mlir::Type acc_dtype = GetTensorTypeOrDie(grad_hy).getElementType();
  auto to_acc = [acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
    if (GetTensorTypeOrDie(op).getElementType() == acc_dtype) {
      return op;
    }
    return mlir::stablehlo::ConvertElementType(op, acc_dtype);
  };
  grad_y = to_acc(grad_y);
  grad_hy = to_acc(grad_hy);
  h_init = to_acc(h_init);
  y_out = to_acc(y_out);
  act_4h_seq = to_acc(act_4h_seq);

  mlir::MlirOp one = MakeConstantLike(grad_hy, 1.0);

  mlir::MlirOp delta_h_next = grad_hy;
  std::vector<mlir::MlirOp> grad_x_steps(seq_len);
  std::vector<mlir::MlirOp> delta_pre_ig_steps(seq_len);
  std::vector<mlir::MlirOp> delta_pre_hg_steps(seq_len);

  for (int64_t t = seq_len - 1; t >= 0; --t) {
    mlir::MlirOp grad_y_t = SliceStep2D(grad_y, t, batch, hidden, batch_first);
    mlir::MlirOp delta_h = mlir::stablehlo::Add(grad_y_t, delta_h_next);

    mlir::MlirOp h_prev =
        (t == 0) ? h_init
                 : SliceStep2D(y_out, t - 1, batch, hidden, batch_first);
    mlir::MlirOp act_t =
        SliceStep2D(act_4h_seq, t, batch, 4 * hidden, batch_first);

    GruBackwardStepResult step =
        ComputeGruBackwardStep(builder, delta_h, h_prev, act_t, w_hh,
                               hh_bwd_dot_dims, batch, hidden, one, precision);

    delta_h_next = step.delta_h_prev;
    delta_pre_ig_steps[t] =
        ExpandStep3D(step.delta_pre_ig, batch, 3 * hidden, batch_first);
    delta_pre_hg_steps[t] =
        ExpandStep3D(step.delta_pre_hg, batch, 3 * hidden, batch_first);

    // delta_x_t = delta_pre_ig @ w_ih: [B, 3*H] @ [3*H, in_dim] -> [B, in_dim]
    mlir::MlirOp dx_t = MixedPrecisionDotGeneral(
        step.delta_pre_ig, w_ih, x_bwd_dot_dims, acc_dtype, precision);
    grad_x_steps[t] = ExpandStep3D(dx_t, batch, in_dim, batch_first);
  }

  const int64_t concat_dim = batch_first ? 1 : 0;
  mlir::MlirOp grad_x = ConcatDim(builder, grad_x_steps, concat_dim);
  mlir::MlirOp grad_h0 =
      mlir::stablehlo::Reshape(to_out(delta_h_next), {1, batch, hidden});

  // Parameter gradients: full sequence dot products
  mlir::MlirOp delta_pre_ig_seq =
      ConcatDim(builder, delta_pre_ig_steps, concat_dim);
  mlir::MlirOp delta_pre_hg_seq =
      ConcatDim(builder, delta_pre_hg_steps, concat_dim);

  mlir::MlirOp delta_pre_ig_2d =
      mlir::stablehlo::Reshape(delta_pre_ig_seq, {seq_len * batch, 3 * hidden});
  mlir::MlirOp delta_pre_hg_2d =
      mlir::stablehlo::Reshape(delta_pre_hg_seq, {seq_len * batch, 3 * hidden});

  mlir::MlirOp layer_in_2d =
      mlir::stablehlo::Reshape(layer_in, {seq_len * batch, in_dim});

  // h_prev_seq: concat(h_init, y_out[0..seq_len-2])
  std::vector<mlir::MlirOp> h_prev_steps;
  h_prev_steps.reserve(seq_len);
  h_prev_steps.push_back(ExpandStep3D(h_init, batch, hidden, batch_first));
  for (int64_t t = 0; t < seq_len - 1; ++t) {
    mlir::MlirOp y_t = SliceStep2D(y_out, t, batch, hidden, batch_first);
    h_prev_steps.push_back(ExpandStep3D(y_t, batch, hidden, batch_first));
  }
  mlir::MlirOp h_prev_seq = ConcatDim(builder, h_prev_steps, concat_dim);
  mlir::MlirOp h_prev_2d =
      mlir::stablehlo::Reshape(h_prev_seq, {seq_len * batch, hidden});

  // grad_w_ih = delta_pre_ig_2d^T @ layer_in_2d: [3*H, T*B] @ [T*B, in_dim] ->
  // [3*H, in_dim]
  mlir::MlirOp grad_w_ih = MixedPrecisionDotGeneral(
      delta_pre_ig_2d, layer_in_2d, w_dot_dims, acc_dtype, precision);
  // grad_w_hh = delta_pre_hg_2d^T @ h_prev_2d: [3*H, T*B] @ [T*B, hidden] ->
  // [3*H, hidden]
  mlir::MlirOp grad_w_hh = MixedPrecisionDotGeneral(
      delta_pre_hg_2d, h_prev_2d, w_dot_dims, acc_dtype, precision);

  std::optional<mlir::MlirOp> grad_b_ih = std::nullopt;
  std::optional<mlir::MlirOp> grad_b_hh = std::nullopt;
  if (has_biases) {
    grad_b_ih = to_out(mlir::stablehlo::Reduce(
        builder, delta_pre_ig_2d, zero_const, sum_reduce_builder, {0})[0]);
    grad_b_hh = to_out(mlir::stablehlo::Reduce(
        builder, delta_pre_hg_2d, zero_const, sum_reduce_builder, {0})[0]);
  }

  return {grad_x,    grad_h0,  to_out(grad_w_ih), to_out(grad_w_hh),
          grad_b_ih, grad_b_hh};
}

// Container holding initialized backward carry state and remainder gradients
// for chunked backward execution.
//
// Fields:
// - `delta_h_next`: Backpropagated hidden state gradient arriving at the exit
// of remainder steps [B, hidden].
// - `grad_w_ih_init`: Initial accumulated input-hidden weight gradient
// [3*hidden, in_dim].
// - `grad_w_hh_init`: Initial accumulated recurrent weight gradient [3*hidden,
// hidden].
// - `grad_b_ih_init`: Initial accumulated input bias gradient [3*hidden].
// - `grad_b_hh_init`: Initial accumulated recurrent bias gradient [3*hidden].
// - `rem_grad_x`: Optional sequence gradient for remainder timesteps
// [rem_steps, B, in_dim].
struct GruBackwardRemainderInit {
  mlir::MlirOp delta_h_next;
  mlir::MlirOp grad_w_ih_init;
  mlir::MlirOp grad_w_hh_init;
  mlir::MlirOp grad_b_ih_init;
  mlir::MlirOp grad_b_hh_init;
  std::optional<mlir::MlirOp> rem_grad_x;
};

// Evaluates tail remainder backward steps (from seq_len - 1 down to
// chunked_steps) to seed parameter gradient accumulators and initial
// delta_h_next.
//
// What it computes:
// - If rem_steps == 0: initializes weight and bias gradient accumulators to
// zero tensors and returns delta_h_start directly.
// - If rem_steps > 0:
//   - Sweeps backward t = seq_len - 1 down to chunked_steps:
//     1. Computes delta_h = grad_y[t] + delta_h_next.
//     2. Evaluates ComputeGruBackwardStep -> delta_pre_ig, delta_pre_hg,
//     delta_h_prev.
//     3. Projects input gradient dx_t = delta_pre_ig @ W_ih.
//   - Computes initial weight gradients from remainder steps via DotGeneral
//   contractions.
//   - Computes initial bias gradients via summation reduction over remainder
//   timesteps and batches.
//
// Parameters:
// - `builder`: MLIR builder.
// - `grad_y`: Upstream sequence gradient [T, B, H].
// - `delta_h_start`: Incoming gradient at t = seq_len - 1 (grad_hy) [B, H].
// - `layer_in`: Layer input sequence [T, B, in_dim].
// - `h_prev_seq`: Sequence of previous hidden states [T, B, H].
// - `act_4h_seq`: Cached forward activations [T, B, 4*H].
// - `w_ih`: Input-hidden weight [3*H, in_dim].
// - `w_hh`: Recurrent weight [3*H, H].
// - `hh_bwd_dot_dims`: Contracting dimensions for recurrent GEMM backward.
// - `x_bwd_dot_dims`: Contracting dimensions for input projection backward.
// - `w_dot_dims`: Contracting dimensions for weight gradient contraction.
// - `has_biases`: Whether bias gradients should be computed.
// - `seq_len`: Total sequence length.
// - `chunked_steps`: Steps processed inside while loop (num_chunks * 8).
// - `rem_steps`: Remainder steps (seq_len % 8).
// - `batch`: Batch size.
// - `hidden`: Hidden dimension.
// - `in_dim`: Input feature dimension.
// - `batch_first`: Memory layout indicator.
// - `concat_dim`: Sequence concatenation dimension.
// - `acc_dtype`: Accumulation data type.
// - `one`: Constant 1.0 tensor.
// - `zero_scalar`: Constant 0.0 scalar.
// - `zero_const`: Zero scalar for reductions.
// - `sum_reduce_builder`: Region builder lambda for reductions.
// - `precision`: Hardware precision level.
//
// Returns:
// - `GruBackwardRemainderInit` struct containing seeded accumulators and
// remainder grad_x.
template <typename ReduceBuilderFn>
GruBackwardRemainderInit ComputeGruBackwardRemainder(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_y, mlir::MlirOp delta_h_start,
    mlir::MlirOp layer_in, mlir::MlirOp h_prev_seq, mlir::MlirOp act_4h_seq,
    mlir::MlirOp w_ih, mlir::MlirOp w_hh,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr x_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr w_dot_dims,
    const bool has_biases, const int64_t seq_len, const int64_t chunked_steps,
    const int64_t rem_steps, const int64_t batch, const int64_t hidden,
    const int64_t in_dim, const bool batch_first, const int64_t concat_dim,
    const mlir::Type acc_dtype, mlir::MlirOp one, mlir::MlirOp zero_scalar,
    mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    const mlir::stablehlo::Precision precision) {
  if (rem_steps == 0) {
    mlir::MlirOp grad_w_ih_init = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({3 * hidden, in_dim}, acc_dtype),
        zero_scalar, {});
    mlir::MlirOp grad_w_hh_init = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({3 * hidden, hidden}, acc_dtype),
        zero_scalar, {});
    mlir::MlirOp grad_b_ih_init;
    mlir::MlirOp grad_b_hh_init;
    if (has_biases) {
      grad_b_ih_init = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({3 * hidden}, acc_dtype), zero_scalar,
          {});
      grad_b_hh_init = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({3 * hidden}, acc_dtype), zero_scalar,
          {});
    }
    return {delta_h_start,  grad_w_ih_init, grad_w_hh_init,
            grad_b_ih_init, grad_b_hh_init, std::nullopt};
  }

  std::vector<mlir::MlirOp> delta_pre_ig_rem(rem_steps);
  std::vector<mlir::MlirOp> delta_pre_hg_rem(rem_steps);
  std::vector<mlir::MlirOp> grad_x_rem(rem_steps);
  mlir::MlirOp delta_h_next = delta_h_start;

  for (int64_t t = seq_len - 1; t >= chunked_steps; --t) {
    const int64_t rem_idx = t - chunked_steps;
    mlir::MlirOp grad_y_t = SliceStep2D(grad_y, t, batch, hidden, batch_first);
    mlir::MlirOp delta_h = mlir::stablehlo::Add(grad_y_t, delta_h_next);
    mlir::MlirOp h_prev_t =
        SliceStep2D(h_prev_seq, t, batch, hidden, batch_first);
    mlir::MlirOp act_t =
        SliceStep2D(act_4h_seq, t, batch, 4 * hidden, batch_first);

    const GruBackwardStepResult step =
        ComputeGruBackwardStep(builder, delta_h, h_prev_t, act_t, w_hh,
                               hh_bwd_dot_dims, batch, hidden, one, precision);
    delta_h_next = step.delta_h_prev;
    delta_pre_ig_rem[rem_idx] =
        ExpandStep3D(step.delta_pre_ig, batch, 3 * hidden, batch_first);
    delta_pre_hg_rem[rem_idx] =
        ExpandStep3D(step.delta_pre_hg, batch, 3 * hidden, batch_first);

    mlir::MlirOp dx_t = MixedPrecisionDotGeneral(
        step.delta_pre_ig, w_ih, x_bwd_dot_dims, acc_dtype, precision);
    grad_x_rem[rem_idx] = ExpandStep3D(dx_t, batch, in_dim, batch_first);
  }

  mlir::MlirOp delta_pre_ig_rem_seq =
      ConcatDim(builder, delta_pre_ig_rem, concat_dim);
  mlir::MlirOp delta_pre_hg_rem_seq =
      ConcatDim(builder, delta_pre_hg_rem, concat_dim);
  mlir::MlirOp delta_pre_ig_rem_2d = mlir::stablehlo::Reshape(
      delta_pre_ig_rem_seq, {rem_steps * batch, 3 * hidden});
  mlir::MlirOp delta_pre_hg_rem_2d = mlir::stablehlo::Reshape(
      delta_pre_hg_rem_seq, {rem_steps * batch, 3 * hidden});

  mlir::MlirOp rem_layer_in =
      batch_first ? mlir::stablehlo::Slice(layer_in, {0, chunked_steps, 0},
                                           {batch, seq_len, in_dim}, {1, 1, 1})
                  : mlir::stablehlo::Slice(layer_in, {chunked_steps, 0, 0},
                                           {seq_len, batch, in_dim}, {1, 1, 1});
  mlir::MlirOp rem_layer_in_2d =
      mlir::stablehlo::Reshape(rem_layer_in, {rem_steps * batch, in_dim});

  mlir::MlirOp rem_h_prev =
      batch_first ? mlir::stablehlo::Slice(h_prev_seq, {0, chunked_steps, 0},
                                           {batch, seq_len, hidden}, {1, 1, 1})
                  : mlir::stablehlo::Slice(h_prev_seq, {chunked_steps, 0, 0},
                                           {seq_len, batch, hidden}, {1, 1, 1});
  mlir::MlirOp rem_h_prev_2d =
      mlir::stablehlo::Reshape(rem_h_prev, {rem_steps * batch, hidden});

  mlir::MlirOp grad_w_ih_init = MixedPrecisionDotGeneral(
      delta_pre_ig_rem_2d, rem_layer_in_2d, w_dot_dims, acc_dtype, precision);
  mlir::MlirOp grad_w_hh_init = MixedPrecisionDotGeneral(
      delta_pre_hg_rem_2d, rem_h_prev_2d, w_dot_dims, acc_dtype, precision);

  mlir::MlirOp grad_b_ih_init;
  mlir::MlirOp grad_b_hh_init;
  if (has_biases) {
    grad_b_ih_init = mlir::stablehlo::Reduce(
        builder, delta_pre_ig_rem_2d, zero_const, sum_reduce_builder, {0})[0];
    grad_b_hh_init = mlir::stablehlo::Reduce(
        builder, delta_pre_hg_rem_2d, zero_const, sum_reduce_builder, {0})[0];
  }

  mlir::MlirOp rem_grad_x = ConcatDim(builder, grad_x_rem, concat_dim);

  return {delta_h_next,   grad_w_ih_init, grad_w_hh_init,
          grad_b_ih_init, grad_b_hh_init, rem_grad_x};
}

// Container holding the adjoint pre-activations and input gradients for an
// unrolled chunk of k=8 backward steps.
//
// Fields:
// - `cur_delta_h_next`: Backpropagated hidden state gradient at the start of
// the chunk [B, hidden].
// - `delta_pre_ig_chunk_2d`: Reshaped 2D input pre-activation adjoints [k *
// batch, 3*hidden].
// - `delta_pre_hg_chunk_2d`: Reshaped 2D recurrent pre-activation adjoints [k *
// batch, 3*hidden].
// - `chunk_grad_x`: Concatenated sequence input gradients for this chunk [k,
// batch, in_dim].
struct GruBackwardChunkStepsResult {
  mlir::MlirOp cur_delta_h_next;
  mlir::MlirOp delta_pre_ig_chunk_2d;
  mlir::MlirOp delta_pre_hg_chunk_2d;
  mlir::MlirOp chunk_grad_x;
};

// Computes kGruUnrollFactor backward steps for a single chunk.
//
// What it computes:
// - Sweeps backwards through chunk timesteps k = kGruUnrollFactor - 1 down to
// 0:
//   1. Slices grad_y_step, h_prev_step, act_step at chunk offset k.
//   2. Evaluates ComputeGruBackwardStep -> step adjoints and backpropagated
//   delta_h.
//   3. Projects step input gradient dx_k = delta_pre_ig @ W_ih.
// - Formats and reshapes chunk pre-activation adjoints into 2D matrices [k *
// batch, 3*hidden] for streaming GEMMs.
// - Concatenates chunk input gradients into [k, batch, in_dim].
//
// Parameters:
// - `builder`: MLIR builder.
// - `grad_y_chunk`: Chunk of upstream sequence gradients [k, batch, hidden].
// - `act_chunk`: Chunk of cached 4H activations [k, batch, 4*hidden].
// - `h_prev_chunk`: Chunk of previous hidden states [k, batch, hidden].
// - `cur_delta_h_next`: Incoming hidden gradient arriving at chunk end [B,
// hidden].
// - `w_hh`: Recurrent weight matrix [3*H, H].
// - `w_ih`: Input-hidden weight matrix [3*H, in_dim].
// - `hh_bwd_dot_dims`: Contracting dimensions for recurrent backward.
// - `x_bwd_dot_dims`: Contracting dimensions for input projection backward.
// - `batch`: Batch size.
// - `hidden`: Hidden dimension.
// - `in_dim`: Input feature dimension.
// - `batch_first`: Memory layout indicator.
// - `concat_dim`: Sequence concatenation dimension.
// - `acc_dtype`: Accumulation data type.
// - `one`: Constant 1.0 tensor.
// - `precision`: Hardware precision level.
//
// Returns:
// - `GruBackwardChunkStepsResult` struct containing backpropagated delta_h, 2D
// pre-activation adjoints, and chunk grad_x.
inline GruBackwardChunkStepsResult ComputeGruBackwardChunkSteps(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_y_chunk,
    mlir::MlirOp act_chunk, mlir::MlirOp h_prev_chunk,
    mlir::MlirOp cur_delta_h_next, mlir::MlirOp w_hh, mlir::MlirOp w_ih,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr x_bwd_dot_dims,
    const int64_t batch, const int64_t hidden, const int64_t in_dim,
    const bool batch_first, const int64_t concat_dim,
    const mlir::Type acc_dtype, mlir::MlirOp one,
    const mlir::stablehlo::Precision precision) {
  std::vector<mlir::MlirOp> delta_pre_ig_k(kGruUnrollFactor);
  std::vector<mlir::MlirOp> delta_pre_hg_k(kGruUnrollFactor);

  for (int64_t k = kGruUnrollFactor - 1; k >= 0; --k) {
    mlir::MlirOp grad_y_step =
        SliceStep2D(grad_y_chunk, k, batch, hidden, batch_first);
    mlir::MlirOp delta_h = mlir::stablehlo::Add(grad_y_step, cur_delta_h_next);
    mlir::MlirOp h_prev_step =
        SliceStep2D(h_prev_chunk, k, batch, hidden, batch_first);
    mlir::MlirOp act_step =
        SliceStep2D(act_chunk, k, batch, 4 * hidden, batch_first);

    const GruBackwardStepResult step =
        ComputeGruBackwardStep(builder, delta_h, h_prev_step, act_step, w_hh,
                               hh_bwd_dot_dims, batch, hidden, one, precision);
    cur_delta_h_next = step.delta_h_prev;
    delta_pre_ig_k[k] =
        ExpandStep3D(step.delta_pre_ig, batch, 3 * hidden, batch_first);
    delta_pre_hg_k[k] =
        ExpandStep3D(step.delta_pre_hg, batch, 3 * hidden, batch_first);
  }

  const mlir::MlirOp delta_pre_ig_chunk =
      ConcatDim(builder, delta_pre_ig_k, concat_dim);
  const mlir::MlirOp delta_pre_hg_chunk =
      ConcatDim(builder, delta_pre_hg_k, concat_dim);
  const mlir::MlirOp delta_pre_ig_chunk_2d = mlir::stablehlo::Reshape(
      delta_pre_ig_chunk, {kGruUnrollFactor * batch, 3 * hidden});
  const mlir::MlirOp delta_pre_hg_chunk_2d = mlir::stablehlo::Reshape(
      delta_pre_hg_chunk, {kGruUnrollFactor * batch, 3 * hidden});

  const mlir::MlirOp chunk_grad_x_2d = MixedPrecisionDotGeneral(
      delta_pre_ig_chunk_2d, w_ih, x_bwd_dot_dims, acc_dtype, precision);
  const llvm::SmallVector<int64_t, 3> chunk_grad_x_dims =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kGruUnrollFactor, in_dim}
          : llvm::SmallVector<int64_t, 3>{kGruUnrollFactor, batch, in_dim};
  const mlir::MlirOp chunk_grad_x =
      mlir::stablehlo::Reshape(chunk_grad_x_2d, chunk_grad_x_dims);

  return {cur_delta_h_next, delta_pre_ig_chunk_2d, delta_pre_hg_chunk_2d,
          chunk_grad_x};
}

// Computes chunked reverse BPTT backward pass for a single unidirectional GRU
// layer using stablehlo::WhileOp.
//
// What it computes:
// - Evaluates reverse BPTT in chunks of k=8 timesteps in reverse chronological
// order:
//   1. Invokes ComputeGruBackwardRemainder to evaluate tail steps (seq_len - 1
//   down to chunked_steps)
//      and initialize parameter gradient accumulators (grad_w_ih, grad_w_hh,
//      grad_b_ih, grad_b_hh).
//   2. Constructs WhileOp carrying step_idx, delta_h_next, grad_x_chunked
//   buffer, and weight/bias gradient accumulators.
//   3. In each chunk iteration:
//      - Computes chunk_start_t = (chunked_steps - 8) - step_idx.
//      - Slices chunk inputs, activations, and previous hidden states at
//      chunk_start_t.
//      - Unrolls 8 backward steps via ComputeGruBackwardChunkSteps.
//      - Accumulates parameter gradients directly via streaming GEMMs:
//          next_grad_w_ih = cur_grad_w_ih + delta_pre_ig_2d^T @ layer_in_2d
//          next_grad_w_hh = cur_grad_w_hh + delta_pre_hg_2d^T @ h_prev_2d
//      - If biases enabled, accumulates bias reductions:
//          next_grad_b_ih = cur_grad_b_ih + reduce_sum(delta_pre_ig_2d, axis=0)
//          next_grad_b_hh = cur_grad_b_hh + reduce_sum(delta_pre_hg_2d, axis=0)
//      - Commits chunk input gradients into grad_x_chunked via
//      DynamicUpdateSlice.
//      - Increments step_idx by 8.
//   4. Joins chunked grad_x with remainder grad_x (if present).
//   5. Returns complete gradients for input sequence, initial hidden state,
//   weights, and biases.
//
// Parameters:
// - `builder`: MLIR builder.
// - `grad_y`: Upstream sequence gradient [T, B, H].
// - `grad_hy`: Upstream final hidden state gradient [1, B, H].
// - `layer_in`: Input sequence tensor to this layer [T, B, in_dim].
// - `h_init`: Initial hidden state [B, H].
// - `y_out`: Output sequence tensor from forward pass [T, B, H].
// - `act_4h_seq`: Cached forward gate activations [T, B, 4*H].
// - `w_ih`: Input-hidden weight matrix [3*H, in_dim].
// - `w_hh`: Recurrent weight matrix [3*H, H].
// - `has_biases`: Whether biases are enabled.
// - `seq_len`: Total sequence length T.
// - `batch`: Batch size B.
// - `hidden`: Hidden dimension H.
// - `in_dim`: Input feature dimension.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Type-conversion lambda to output dtype.
// - `zero_const`: Zero scalar for reductions.
// - `sum_reduce_builder`: Region builder lambda for reductions.
// - `precision`: Hardware precision level.
//
// Returns:
// - `GruLayerBackwardOutputs` struct containing complete layer backward
// gradients.
// Builds the body block for the chunked unidirectional backward while loop.
//
// What it computes:
// - Calculates reverse-chronological chunk start index chunk_start_t =
// (chunked_steps - 8) - body_step_idx.
// - Slices chunk inputs, activations, and previous hidden states at
// chunk_start_t.
// - Unrolls 8 backward steps via ComputeGruBackwardChunkSteps.
// - Accumulates parameter gradients directly via streaming GEMMs:
//     next_grad_w_ih = cur_grad_w_ih + delta_pre_ig_2d^T @ layer_in_2d
//     next_grad_w_hh = cur_grad_w_hh + delta_pre_hg_2d^T @ h_prev_2d
// - If biases enabled, accumulates bias reductions:
//     next_grad_b_ih = cur_grad_b_ih + reduce_sum(delta_pre_ig_2d, axis=0)
//     next_grad_b_hh = cur_grad_b_hh + reduce_sum(delta_pre_hg_2d, axis=0)
// - Commits chunk input gradients into grad_x_chunked via DynamicUpdateSlice.
// - Increments step_idx by 8.
template <typename ReduceBuilderFn>
void BuildLayerBackwardChunkedWhileBody(
    mlir::MlirBuilder& builder, mlir::Block* body_block, mlir::Location loc,
    mlir::MlirOp grad_y, mlir::MlirOp act_4h_seq, mlir::MlirOp h_prev_seq,
    mlir::MlirOp layer_in, mlir::MlirOp w_hh, mlir::MlirOp w_ih,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr x_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr w_dot_dims,
    int64_t chunked_steps, int64_t batch, int64_t hidden, int64_t in_dim,
    bool batch_first, int64_t concat_dim, mlir::Type acc_dtype,
    mlir::MlirOp one, bool has_biases, mlir::MlirOp zero_const,
    ReduceBuilderFn sum_reduce_builder, mlir::stablehlo::Precision precision) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::IntegerType i64 = op_builder.getI64Type();

  mlir::MlirOp body_step_idx(builder, body_block->getArgument(0));
  mlir::MlirOp body_delta_h_next(builder, body_block->getArgument(1));
  mlir::MlirOp body_grad_x_chunked(builder, body_block->getArgument(2));
  mlir::MlirOp body_grad_w_ih(builder, body_block->getArgument(3));
  mlir::MlirOp body_grad_w_hh(builder, body_block->getArgument(4));
  mlir::MlirOp body_grad_b_ih;
  mlir::MlirOp body_grad_b_hh;
  if (has_biases) {
    body_grad_b_ih = mlir::MlirOp(builder, body_block->getArgument(5));
    body_grad_b_hh = mlir::MlirOp(builder, body_block->getArgument(6));
  }

  mlir::MlirOp const_chunk_limit =
      MakeScalarConstant(builder, chunked_steps - kGruUnrollFactor, i64);
  mlir::MlirOp chunk_start_t =
      mlir::stablehlo::Subtract(const_chunk_limit, body_step_idx);

  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64);
  const llvm::SmallVector<mlir::MlirOp, 3> chunk_start_indices =
      batch_first ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, chunk_start_t,
                                                       zero_i64}
                  : llvm::SmallVector<mlir::MlirOp, 3>{chunk_start_t, zero_i64,
                                                       zero_i64};

  const llvm::SmallVector<int64_t, 3> grad_y_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kGruUnrollFactor, hidden}
          : llvm::SmallVector<int64_t, 3>{kGruUnrollFactor, batch, hidden};
  const llvm::SmallVector<int64_t, 3> act_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kGruUnrollFactor, 4 * hidden}
          : llvm::SmallVector<int64_t, 3>{kGruUnrollFactor, batch, 4 * hidden};
  const llvm::SmallVector<int64_t, 3> in_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kGruUnrollFactor, in_dim}
          : llvm::SmallVector<int64_t, 3>{kGruUnrollFactor, batch, in_dim};

  mlir::MlirOp grad_y_chunk = mlir::stablehlo::DynamicSlice(
      grad_y, chunk_start_indices, grad_y_slice_sizes);
  mlir::MlirOp act_chunk = mlir::stablehlo::DynamicSlice(
      act_4h_seq, chunk_start_indices, act_slice_sizes);
  mlir::MlirOp h_prev_chunk = mlir::stablehlo::DynamicSlice(
      h_prev_seq, chunk_start_indices, grad_y_slice_sizes);
  mlir::MlirOp layer_in_chunk = mlir::stablehlo::DynamicSlice(
      layer_in, chunk_start_indices, in_slice_sizes);

  GruBackwardChunkStepsResult chunk_steps = ComputeGruBackwardChunkSteps(
      builder, grad_y_chunk, act_chunk, h_prev_chunk, body_delta_h_next, w_hh,
      w_ih, hh_bwd_dot_dims, x_bwd_dot_dims, batch, hidden, in_dim, batch_first,
      concat_dim, acc_dtype, one, precision);

  mlir::MlirOp layer_in_chunk_2d = mlir::stablehlo::Reshape(
      layer_in_chunk, {kGruUnrollFactor * batch, in_dim});
  mlir::MlirOp h_prev_chunk_2d = mlir::stablehlo::Reshape(
      h_prev_chunk, {kGruUnrollFactor * batch, hidden});

  mlir::MlirOp chunk_grad_w_ih = MixedPrecisionDotGeneral(
      chunk_steps.delta_pre_ig_chunk_2d, layer_in_chunk_2d, w_dot_dims,
      acc_dtype, precision);
  mlir::MlirOp next_grad_w_ih =
      mlir::stablehlo::Add(body_grad_w_ih, chunk_grad_w_ih);

  mlir::MlirOp chunk_grad_w_hh = MixedPrecisionDotGeneral(
      chunk_steps.delta_pre_hg_chunk_2d, h_prev_chunk_2d, w_dot_dims, acc_dtype,
      precision);
  mlir::MlirOp next_grad_w_hh =
      mlir::stablehlo::Add(body_grad_w_hh, chunk_grad_w_hh);

  mlir::MlirOp next_grad_b_ih;
  mlir::MlirOp next_grad_b_hh;
  if (has_biases) {
    mlir::MlirOp chunk_grad_b_ih =
        mlir::stablehlo::Reduce(builder, chunk_steps.delta_pre_ig_chunk_2d,
                                zero_const, sum_reduce_builder, {0})[0];
    next_grad_b_ih = mlir::stablehlo::Add(body_grad_b_ih, chunk_grad_b_ih);

    mlir::MlirOp chunk_grad_b_hh =
        mlir::stablehlo::Reduce(builder, chunk_steps.delta_pre_hg_chunk_2d,
                                zero_const, sum_reduce_builder, {0})[0];
    next_grad_b_hh = mlir::stablehlo::Add(body_grad_b_hh, chunk_grad_b_hh);
  }

  mlir::MlirOp next_grad_x_chunked = mlir::stablehlo::DynamicUpdateSlice(
      body_grad_x_chunked, chunk_steps.chunk_grad_x, chunk_start_indices);

  mlir::MlirOp step_k = MakeScalarConstant(builder, kGruUnrollFactor, i64);
  mlir::MlirOp next_step_idx = mlir::stablehlo::Add(body_step_idx, step_k);

  llvm::SmallVector<mlir::Value> next_loop_values = {
      next_step_idx.getValue(), chunk_steps.cur_delta_h_next.getValue(),
      next_grad_x_chunked.getValue(), next_grad_w_ih.getValue(),
      next_grad_w_hh.getValue()};
  if (has_biases) {
    next_loop_values.push_back(next_grad_b_ih.getValue());
    next_loop_values.push_back(next_grad_b_hh.getValue());
  }
  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_loop_values);
}

// Computes chunked reverse BPTT backward pass for a single unidirectional GRU
// layer using stablehlo::WhileOp.
//
// What it computes:
// - Evaluates reverse BPTT in chunks of k=8 timesteps in reverse chronological
// order:
//   1. Invokes ComputeGruBackwardRemainder to evaluate tail steps (seq_len - 1
//   down to chunked_steps)
//      and initialize parameter gradient accumulators (grad_w_ih, grad_w_hh,
//      grad_b_ih, grad_b_hh).
//   2. Constructs WhileOp carrying step_idx, delta_h_next, grad_x_chunked
//   buffer, and weight/bias gradient accumulators.
//   3. Delegates while body execution to BuildLayerBackwardChunkedWhileBody.
//   4. Joins chunked grad_x with remainder grad_x (if present).
//   5. Returns complete gradients for input sequence, initial hidden state,
//   weights, and biases.
//
// Parameters:
// - `builder`: MLIR builder.
// - `grad_y`: Upstream sequence gradient [T, B, H].
// - `grad_hy`: Upstream final hidden state gradient [1, B, H].
// - `layer_in`: Input sequence tensor to this layer [T, B, in_dim].
// - `h_init`: Initial hidden state [B, H].
// - `y_out`: Output sequence tensor from forward pass [T, B, H].
// - `act_4h_seq`: Cached forward gate activations [T, B, 4*H].
// - `w_ih`: Input-hidden weight matrix [3*H, in_dim].
// - `w_hh`: Recurrent weight matrix [3*H, H].
// - `has_biases`: Whether biases are enabled.
// - `seq_len`: Total sequence length T.
// - `batch`: Batch size B.
// - `hidden`: Hidden dimension H.
// - `in_dim`: Input feature dimension.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Type-conversion lambda to output dtype.
// - `zero_const`: Zero scalar for reductions.
// - `sum_reduce_builder`: Region builder lambda for reductions.
// - `precision`: Hardware precision level.
//
// Returns:
// - `GruLayerBackwardOutputs` struct containing complete layer backward
// gradients.
template <typename ToOutFn, typename ReduceBuilderFn>
GruLayerBackwardOutputs ComputeLayerBackwardChunked(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_y, mlir::MlirOp grad_hy,
    mlir::MlirOp layer_in, mlir::MlirOp h_init, mlir::MlirOp y_out,
    mlir::MlirOp act_4h_seq, mlir::MlirOp w_ih, mlir::MlirOp w_hh,
    const bool has_biases, const int64_t seq_len, const int64_t batch,
    const int64_t hidden, const int64_t in_dim, const bool batch_first,
    ToOutFn to_out, mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    const mlir::stablehlo::Precision precision) {
  const int64_t num_chunks = seq_len / kGruUnrollFactor;
  const int64_t chunked_steps = num_chunks * kGruUnrollFactor;
  const int64_t rem_steps = seq_len % kGruUnrollFactor;
  const int64_t concat_dim = batch_first ? 1 : 0;

  mlir::MLIRContext& ctx = builder.getContext();
  const auto hh_bwd_dot_dims = MakeDotDims(&ctx, {1}, {0});
  const auto x_bwd_dot_dims = MakeDotDims(&ctx, {1}, {0});
  const auto w_dot_dims = MakeDotDims(&ctx, {0}, {0});

  const mlir::Type acc_dtype = GetTensorTypeOrDie(grad_hy).getElementType();
  auto to_acc = [acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
    if (GetTensorTypeOrDie(op).getElementType() == acc_dtype) {
      return op;
    }
    return mlir::stablehlo::ConvertElementType(op, acc_dtype);
  };
  grad_y = to_acc(grad_y);
  grad_hy = to_acc(grad_hy);
  h_init = to_acc(h_init);
  y_out = to_acc(y_out);
  act_4h_seq = to_acc(act_4h_seq);

  mlir::MlirOp one = MakeConstantLike(grad_hy, 1.0);

  // Construct h_prev_seq: [h_init, y_out[0..seq_len-2]]
  mlir::MlirOp h_init_3d = ExpandStep3D(h_init, batch, hidden, batch_first);
  mlir::MlirOp h_prev_seq;
  if (seq_len == 1) {
    h_prev_seq = h_init_3d;
  } else {
    mlir::MlirOp y_prev_slice =
        batch_first
            ? mlir::stablehlo::Slice(y_out, {0, 0, 0},
                                     {batch, seq_len - 1, hidden}, {1, 1, 1})
            : mlir::stablehlo::Slice(y_out, {0, 0, 0},
                                     {seq_len - 1, batch, hidden}, {1, 1, 1});
    h_prev_seq = ConcatDim(builder, {h_init_3d, y_prev_slice}, concat_dim);
  }

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = grad_y.getValue().getLoc();
  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType i64_scalar_type =
      mlir::RankedTensorType::get({}, i64);

  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0f, acc_dtype);

  GruBackwardRemainderInit rem = ComputeGruBackwardRemainder(
      builder, grad_y, /*delta_h_start=*/grad_hy, layer_in, h_prev_seq,
      act_4h_seq, w_ih, w_hh, hh_bwd_dot_dims, x_bwd_dot_dims, w_dot_dims,
      has_biases, seq_len, chunked_steps, rem_steps, batch, hidden, in_dim,
      batch_first, concat_dim, acc_dtype, one, zero_scalar, zero_const,
      sum_reduce_builder, precision);

  const mlir::RankedTensorType grad_x_chunked_type =
      batch_first ? mlir::RankedTensorType::get({batch, chunked_steps, in_dim},
                                                acc_dtype)
                  : mlir::RankedTensorType::get({chunked_steps, batch, in_dim},
                                                acc_dtype);
  mlir::MlirOp grad_x_chunked_init =
      mlir::stablehlo::BroadcastInDim(grad_x_chunked_type, zero_scalar, {});

  mlir::MlirOp step_idx_init = MakeScalarConstant(builder, 0, i64);

  llvm::SmallVector<mlir::Type> loop_types = {
      i64_scalar_type, rem.delta_h_next.getType(), grad_x_chunked_type,
      rem.grad_w_ih_init.getType(), rem.grad_w_hh_init.getType()};
  llvm::SmallVector<mlir::Value> loop_inits = {
      step_idx_init.getValue(), rem.delta_h_next.getValue(),
      grad_x_chunked_init.getValue(), rem.grad_w_ih_init.getValue(),
      rem.grad_w_hh_init.getValue()};

  if (has_biases) {
    loop_types.push_back(rem.grad_b_ih_init.getType());
    loop_types.push_back(rem.grad_b_hh_init.getType());
    loop_inits.push_back(rem.grad_b_ih_init.getValue());
    loop_inits.push_back(rem.grad_b_hh_init.getValue());
  }

  auto while_op =
      mlir::stablehlo::WhileOp::create(op_builder, loc, loop_types, loop_inits);

  mlir::Block* const cond_block = op_builder.createBlock(&while_op.getCond());
  cond_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(cond_block);

  mlir::MlirOp limit = MakeScalarConstant(builder, chunked_steps, i64);
  const mlir::Value cond =
      mlir::stablehlo::CompareOp::create(
          op_builder, loc, cond_block->getArgument(0), limit.getValue(),
          mlir::stablehlo::ComparisonDirection::LT)
          .getResult();
  mlir::stablehlo::ReturnOp::create(op_builder, loc, cond);

  mlir::Block* const body_block = op_builder.createBlock(&while_op.getBody());
  body_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(body_block);

  BuildLayerBackwardChunkedWhileBody(
      builder, body_block, loc, grad_y, act_4h_seq, h_prev_seq, layer_in, w_hh,
      w_ih, hh_bwd_dot_dims, x_bwd_dot_dims, w_dot_dims, chunked_steps, batch,
      hidden, in_dim, batch_first, concat_dim, acc_dtype, one, has_biases,
      zero_const, sum_reduce_builder, precision);

  op_builder.setInsertionPointAfter(while_op);

  mlir::MlirOp final_delta_h_next(builder, while_op.getResult(1));
  mlir::MlirOp final_grad_x_chunked(builder, while_op.getResult(2));
  mlir::MlirOp final_grad_w_ih(builder, while_op.getResult(3));
  mlir::MlirOp final_grad_w_hh(builder, while_op.getResult(4));

  mlir::MlirOp grad_x_layer;
  if (rem.rem_grad_x.has_value()) {
    grad_x_layer =
        ConcatDim(builder, {final_grad_x_chunked, *rem.rem_grad_x}, concat_dim);
  } else {
    grad_x_layer = final_grad_x_chunked;
  }

  mlir::MlirOp grad_h0_layer =
      mlir::stablehlo::Reshape(to_out(final_delta_h_next), {1, batch, hidden});

  std::optional<mlir::MlirOp> grad_b_ih;
  std::optional<mlir::MlirOp> grad_b_hh;
  if (has_biases) {
    grad_b_ih = to_out(mlir::MlirOp(builder, while_op.getResult(5)));
    grad_b_hh = to_out(mlir::MlirOp(builder, while_op.getResult(6)));
  }

  return {grad_x_layer,
          grad_h0_layer,
          to_out(final_grad_w_ih),
          to_out(final_grad_w_hh),
          grad_b_ih,
          grad_b_hh};
}

// Dispatcher for unidirectional GRU layer backward evaluation.
//
// What it computes:
// - If seq_len < kGruUnrollFactor: delegates to ComputeLayerBackwardStatic for
// static unrolled execution.
// - Otherwise: delegates to ComputeLayerBackwardChunked for chunked streaming
// execution inside stablehlo::WhileOp.
//
// Parameters:
// - `builder`: MLIR builder.
// - `grad_y`: Upstream sequence gradient.
// - `grad_hy`: Upstream final hidden state gradient.
// - `layer_in`: Layer input sequence.
// - `h_init`: Initial hidden state.
// - `y_out`: Layer output sequence.
// - `act_4h_seq`: Cached forward activations.
// - `w_ih`: Input-hidden weight.
// - `w_hh`: Recurrent weight.
// - `has_biases`: Whether biases are present.
// - `seq_len`: Sequence length.
// - `batch`: Batch size.
// - `hidden`: Hidden dimension.
// - `in_dim`: Input feature dimension.
// - `batch_first`: Memory layout indicator.
// - `to_out`: Dtype conversion lambda.
// - `zero_const`: Zero constant for reductions.
// - `sum_reduce_builder`: Region builder for reductions.
// - `precision`: Hardware precision level.
//
// Returns:
// - `GruLayerBackwardOutputs` struct with all computed gradients for this
// layer.
template <typename ToOutFn, typename ReduceBuilderFn>
GruLayerBackwardOutputs ComputeLayerBackward(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_y, mlir::MlirOp grad_hy,
    mlir::MlirOp layer_in, mlir::MlirOp h_init, mlir::MlirOp y_out,
    mlir::MlirOp act_4h_seq, mlir::MlirOp w_ih, mlir::MlirOp w_hh,
    const bool has_biases, const int64_t seq_len, const int64_t batch,
    const int64_t hidden, const int64_t in_dim, const bool batch_first,
    ToOutFn to_out, mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    const mlir::stablehlo::Precision precision) {
  if (seq_len < kGruUnrollFactor) {
    return ComputeLayerBackwardStatic(
        builder, grad_y, grad_hy, layer_in, h_init, y_out, act_4h_seq, w_ih,
        w_hh, has_biases, seq_len, batch, hidden, in_dim, batch_first, to_out,
        zero_const, sum_reduce_builder, precision);
  }
  return ComputeLayerBackwardChunked(
      builder, grad_y, grad_hy, layer_in, h_init, y_out, act_4h_seq, w_ih, w_hh,
      has_biases, seq_len, batch, hidden, in_dim, batch_first, to_out,
      zero_const, sum_reduce_builder, precision);
}

// =============================================================================
// Bidirectional GRU Backward Pass (Static Unrolling)
// =============================================================================

// Computes the backward pass for a single bidirectional GRU layer using static
// unrolling when seq_len < kGruUnrollFactor.
//
// Mathematical and Algorithmic Flow:
// 1. Slices incoming sequence gradients grad_y [T, B, 2*H] into forward
//    [0..H] and reverse [H..2*H] halves along the hidden dimension.
// 2. Forward BPTT: unrolls time steps backwards from t = seq_len - 1 down to 0,
//    evaluating ComputeGruBackwardStep at each step to propagate delta_h and
//    accumulate gate pre-activation deltas (delta_pre_ig, delta_pre_hg).
// 3. Reverse BPTT: unrolls time steps forwards from t = 0 up to seq_len - 1,
//    propagating reverse delta_h and accumulating reverse gate deltas.
// 4. Parameter Gradients: aggregates pre-activation deltas across all time
// steps
//    and executes 2D matrix contractions against input activations (layer_in)
//    and previous hidden states (h_prev) for both forward and reverse
//    directions.
//
// Arguments:
//   builder: MLIR StableHLO graph builder.
//   grad_y: Incoming output sequence gradient [T, B, 2*H] (or [B, T, 2*H]).
//   grad_hy: Incoming final hidden state gradient [2, B, H].
//   layer_in: Input sequence tensor for this layer [T, B, in_dim].
//   h_init_fwd: Initial forward hidden state [B, H].
//   h_init_rev: Initial reverse hidden state [B, H].
//   y_out_fwd: Forward output sequence activations [T, B, H].
//   y_out_rev: Reverse output sequence activations [T, B, H].
//   act_fwd_seq: Forward pre-activation cache [T, B, 4*H].
//   act_rev_seq: Reverse pre-activation cache [T, B, 4*H].
//   w_ih_fwd, w_hh_fwd: Forward input-hidden and hidden-hidden weight matrices.
//   w_ih_rev, w_hh_rev: Reverse input-hidden and hidden-hidden weight matrices.
//   has_biases: Whether layer includes additive bias parameters.
//   seq_len, batch, hidden, in_dim: Layer dimensions.
//   batch_first: Whether batch is dimension 0.
//   to_out: Functor converting accumulator tensors to output datatype.
//   zero_const: Scalar zero constant for bias reduction.
//   sum_reduce_builder: Reduction region builder for StableHLO ReduceOp.
//   precision: Precision configuration for matrix multiplication operations.
//
// Returns:
//   GruBidirLayerBackwardOutputs containing input gradients (grad_x), initial
//   hidden state gradients (grad_h0), and parameter gradients for both
//   directions.
template <typename ToOutFn, typename ReduceBuilderFn>
GruBidirLayerBackwardOutputs ComputeBidirLayerBackwardStatic(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_y, mlir::MlirOp grad_hy,
    mlir::MlirOp layer_in, mlir::MlirOp h_init_fwd, mlir::MlirOp h_init_rev,
    mlir::MlirOp y_out_fwd, mlir::MlirOp y_out_rev, mlir::MlirOp act_fwd_seq,
    mlir::MlirOp act_rev_seq, mlir::MlirOp w_ih_fwd, mlir::MlirOp w_hh_fwd,
    mlir::MlirOp w_ih_rev, mlir::MlirOp w_hh_rev, bool has_biases,
    int64_t seq_len, int64_t batch, int64_t hidden, int64_t in_dim,
    bool batch_first, ToOutFn to_out, mlir::MlirOp zero_const,
    ReduceBuilderFn sum_reduce_builder, mlir::stablehlo::Precision precision) {
  mlir::MLIRContext& ctx = builder.getContext();
  // Contracting dimensions: [B, 3*H] x [3*H, H] -> [B, H]
  const auto hh_bwd_dot_dims = MakeDotDims(&ctx, {1}, {0});
  // Contracting dimensions: [B, 3*H] x [3*H, in_dim] -> [B, in_dim]
  const auto x_bwd_dot_dims = MakeDotDims(&ctx, {1}, {0});
  // Parameter gradient contraction: [T*B, 3*H]^T x [T*B, in_dim] -> [3*H,
  // in_dim]
  const auto w_dot_dims = MakeDotDims(&ctx, {0}, {0});

  // Promote all inputs and activations to accumulator precision (F32) for
  // numerical stability during gradient backpropagation.
  const mlir::Type acc_dtype = GetTensorTypeOrDie(grad_hy).getElementType();
  auto to_acc = [acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
    if (GetTensorTypeOrDie(op).getElementType() == acc_dtype) {
      return op;
    }
    return mlir::stablehlo::ConvertElementType(op, acc_dtype);
  };
  grad_y = to_acc(grad_y);
  grad_hy = to_acc(grad_hy);
  h_init_fwd = to_acc(h_init_fwd);
  h_init_rev = to_acc(h_init_rev);
  y_out_fwd = to_acc(y_out_fwd);
  y_out_rev = to_acc(y_out_rev);
  act_fwd_seq = to_acc(act_fwd_seq);
  act_rev_seq = to_acc(act_rev_seq);

  mlir::MlirOp one = MakeConstantLike(h_init_fwd, 1.0);

  // Step 1: Slice incoming sequence gradient grad_y along feature dimension
  // into forward [0..hidden] and reverse [hidden..2*hidden] sections.
  mlir::MlirOp grad_y_fwd =
      batch_first ? mlir::stablehlo::Slice(grad_y, {0, 0, 0},
                                           {batch, seq_len, hidden}, {1, 1, 1})
                  : mlir::stablehlo::Slice(grad_y, {0, 0, 0},
                                           {seq_len, batch, hidden}, {1, 1, 1});
  mlir::MlirOp grad_y_rev =
      batch_first
          ? mlir::stablehlo::Slice(grad_y, {0, 0, hidden},
                                   {batch, seq_len, 2 * hidden}, {1, 1, 1})
          : mlir::stablehlo::Slice(grad_y, {0, 0, hidden},
                                   {seq_len, batch, 2 * hidden}, {1, 1, 1});

  // Extract initial layer gradients for forward (index 0) and reverse (index
  // 1).
  mlir::MlirOp grad_hy_fwd = ExtractLayer2D(grad_hy, 0, batch, hidden);
  mlir::MlirOp grad_hy_rev = ExtractLayer2D(grad_hy, 1, batch, hidden);

  // Step 2: Forward direction BPTT - unrolls from t = seq_len - 1 down to 0.
  mlir::MlirOp delta_h_next_fwd = grad_hy_fwd;
  std::vector<mlir::MlirOp> grad_x_fwd_steps(seq_len);
  std::vector<mlir::MlirOp> delta_pre_ig_fwd_steps(seq_len);
  std::vector<mlir::MlirOp> delta_pre_hg_fwd_steps(seq_len);

  for (int64_t t = seq_len - 1; t >= 0; --t) {
    mlir::MlirOp gy_t = SliceStep2D(grad_y_fwd, t, batch, hidden, batch_first);
    mlir::MlirOp dh = mlir::stablehlo::Add(gy_t, delta_h_next_fwd);
    mlir::MlirOp hp =
        (t == 0) ? h_init_fwd
                 : SliceStep2D(y_out_fwd, t - 1, batch, hidden, batch_first);
    mlir::MlirOp act_t =
        SliceStep2D(act_fwd_seq, t, batch, 4 * hidden, batch_first);

    GruBackwardStepResult step =
        ComputeGruBackwardStep(builder, dh, hp, act_t, w_hh_fwd,
                               hh_bwd_dot_dims, batch, hidden, one, precision);
    delta_h_next_fwd = step.delta_h_prev;
    delta_pre_ig_fwd_steps[t] =
        ExpandStep3D(step.delta_pre_ig, batch, 3 * hidden, batch_first);
    delta_pre_hg_fwd_steps[t] =
        ExpandStep3D(step.delta_pre_hg, batch, 3 * hidden, batch_first);

    mlir::MlirOp dx_t = MixedPrecisionDotGeneral(
        step.delta_pre_ig, w_ih_fwd, x_bwd_dot_dims, acc_dtype, precision);
    grad_x_fwd_steps[t] = ExpandStep3D(dx_t, batch, in_dim, batch_first);
  }

  // Step 3: Reverse direction BPTT - unrolls from t = 0 up to seq_len - 1.
  mlir::MlirOp delta_h_next_rev = grad_hy_rev;
  std::vector<mlir::MlirOp> grad_x_rev_steps(seq_len);
  std::vector<mlir::MlirOp> delta_pre_ig_rev_steps(seq_len);
  std::vector<mlir::MlirOp> delta_pre_hg_rev_steps(seq_len);

  for (int64_t t = 0; t < seq_len; ++t) {
    mlir::MlirOp gy_t = SliceStep2D(grad_y_rev, t, batch, hidden, batch_first);
    mlir::MlirOp dh = mlir::stablehlo::Add(gy_t, delta_h_next_rev);
    mlir::MlirOp hp = (t == seq_len - 1) ? h_init_rev
                                         : SliceStep2D(y_out_rev, t + 1, batch,
                                                       hidden, batch_first);
    mlir::MlirOp act_t =
        SliceStep2D(act_rev_seq, t, batch, 4 * hidden, batch_first);

    GruBackwardStepResult step =
        ComputeGruBackwardStep(builder, dh, hp, act_t, w_hh_rev,
                               hh_bwd_dot_dims, batch, hidden, one, precision);
    delta_h_next_rev = step.delta_h_prev;
    delta_pre_ig_rev_steps[t] =
        ExpandStep3D(step.delta_pre_ig, batch, 3 * hidden, batch_first);
    delta_pre_hg_rev_steps[t] =
        ExpandStep3D(step.delta_pre_hg, batch, 3 * hidden, batch_first);

    mlir::MlirOp dx_t = MixedPrecisionDotGeneral(
        step.delta_pre_ig, w_ih_rev, x_bwd_dot_dims, acc_dtype, precision);
    grad_x_rev_steps[t] = ExpandStep3D(dx_t, batch, in_dim, batch_first);
  }

  // Step 4: Combine input gradients grad_x from forward and reverse passes.
  const int64_t concat_dim = batch_first ? 1 : 0;
  mlir::MlirOp grad_x_fwd_seq =
      ConcatDim(builder, grad_x_fwd_steps, concat_dim);
  mlir::MlirOp grad_x_rev_seq =
      ConcatDim(builder, grad_x_rev_steps, concat_dim);
  mlir::MlirOp grad_x = mlir::stablehlo::Add(grad_x_fwd_seq, grad_x_rev_seq);

  // Combine initial hidden state gradients [2, B, H].
  mlir::MlirOp grad_h0_fwd_3d =
      mlir::stablehlo::Reshape(to_out(delta_h_next_fwd), {1, batch, hidden});
  mlir::MlirOp grad_h0_rev_3d =
      mlir::stablehlo::Reshape(to_out(delta_h_next_rev), {1, batch, hidden});
  mlir::MlirOp grad_h0 = mlir::stablehlo::Concatenate(
      builder, {grad_h0_fwd_3d, grad_h0_rev_3d}, /*dim=*/0);

  // Step 5: Full 2D batch contractions for weight and bias parameter gradients.
  mlir::MlirOp layer_in_2d =
      mlir::stablehlo::Reshape(layer_in, {seq_len * batch, in_dim});

  // Forward weight gradients.
  mlir::MlirOp delta_pre_ig_fwd_2d = mlir::stablehlo::Reshape(
      ConcatDim(builder, delta_pre_ig_fwd_steps, concat_dim),
      {seq_len * batch, 3 * hidden});
  mlir::MlirOp delta_pre_hg_fwd_2d = mlir::stablehlo::Reshape(
      ConcatDim(builder, delta_pre_hg_fwd_steps, concat_dim),
      {seq_len * batch, 3 * hidden});

  std::vector<mlir::MlirOp> h_prev_fwd_steps;
  h_prev_fwd_steps.reserve(seq_len);
  h_prev_fwd_steps.push_back(
      ExpandStep3D(h_init_fwd, batch, hidden, batch_first));
  for (int64_t t = 0; t < seq_len - 1; ++t) {
    mlir::MlirOp y_t = SliceStep2D(y_out_fwd, t, batch, hidden, batch_first);
    h_prev_fwd_steps.push_back(ExpandStep3D(y_t, batch, hidden, batch_first));
  }
  mlir::MlirOp h_prev_fwd_2d =
      mlir::stablehlo::Reshape(ConcatDim(builder, h_prev_fwd_steps, concat_dim),
                               {seq_len * batch, hidden});

  mlir::MlirOp grad_w_ih_fwd = MixedPrecisionDotGeneral(
      delta_pre_ig_fwd_2d, layer_in_2d, w_dot_dims, acc_dtype, precision);
  mlir::MlirOp grad_w_hh_fwd = MixedPrecisionDotGeneral(
      delta_pre_hg_fwd_2d, h_prev_fwd_2d, w_dot_dims, acc_dtype, precision);

  // Reverse weight gradients.
  mlir::MlirOp delta_pre_ig_rev_2d = mlir::stablehlo::Reshape(
      ConcatDim(builder, delta_pre_ig_rev_steps, concat_dim),
      {seq_len * batch, 3 * hidden});
  mlir::MlirOp delta_pre_hg_rev_2d = mlir::stablehlo::Reshape(
      ConcatDim(builder, delta_pre_hg_rev_steps, concat_dim),
      {seq_len * batch, 3 * hidden});

  std::vector<mlir::MlirOp> h_prev_rev_steps(seq_len);
  for (int64_t t = 0; t < seq_len - 1; ++t) {
    mlir::MlirOp y_t =
        SliceStep2D(y_out_rev, t + 1, batch, hidden, batch_first);
    h_prev_rev_steps[t] = ExpandStep3D(y_t, batch, hidden, batch_first);
  }
  h_prev_rev_steps[seq_len - 1] =
      ExpandStep3D(h_init_rev, batch, hidden, batch_first);
  mlir::MlirOp h_prev_rev_2d =
      mlir::stablehlo::Reshape(ConcatDim(builder, h_prev_rev_steps, concat_dim),
                               {seq_len * batch, hidden});

  mlir::MlirOp grad_w_ih_rev = MixedPrecisionDotGeneral(
      delta_pre_ig_rev_2d, layer_in_2d, w_dot_dims, acc_dtype, precision);
  mlir::MlirOp grad_w_hh_rev = MixedPrecisionDotGeneral(
      delta_pre_hg_rev_2d, h_prev_rev_2d, w_dot_dims, acc_dtype, precision);

  // Forward and reverse bias gradients (sum across batch and time dimensions).
  std::optional<mlir::MlirOp> grad_b_ih_fwd = std::nullopt;
  std::optional<mlir::MlirOp> grad_b_hh_fwd = std::nullopt;
  std::optional<mlir::MlirOp> grad_b_ih_rev = std::nullopt;
  std::optional<mlir::MlirOp> grad_b_hh_rev = std::nullopt;
  if (has_biases) {
    grad_b_ih_fwd = to_out(mlir::stablehlo::Reduce(
        builder, delta_pre_ig_fwd_2d, zero_const, sum_reduce_builder, {0})[0]);
    grad_b_hh_fwd = to_out(mlir::stablehlo::Reduce(
        builder, delta_pre_hg_fwd_2d, zero_const, sum_reduce_builder, {0})[0]);
    grad_b_ih_rev = to_out(mlir::stablehlo::Reduce(
        builder, delta_pre_ig_rev_2d, zero_const, sum_reduce_builder, {0})[0]);
    grad_b_hh_rev = to_out(mlir::stablehlo::Reduce(
        builder, delta_pre_hg_rev_2d, zero_const, sum_reduce_builder, {0})[0]);
  }

  return {grad_x,
          grad_h0,
          to_out(grad_w_ih_fwd),
          to_out(grad_w_hh_fwd),
          grad_b_ih_fwd,
          grad_b_hh_fwd,
          to_out(grad_w_ih_rev),
          to_out(grad_w_hh_rev),
          grad_b_ih_rev,
          grad_b_hh_rev};
}

// Result of evaluating tail remainder steps for bidirectional GRU backward
// pass. Serves as the initial state for the subsequent chunked while loop.
struct GruBidirBackwardRemainderInit {
  mlir::MlirOp delta_h_next_fwd;
  mlir::MlirOp delta_h_next_rev;
  mlir::MlirOp grad_w_ih_init_fwd;
  mlir::MlirOp grad_w_hh_init_fwd;
  mlir::MlirOp grad_b_ih_init_fwd;
  mlir::MlirOp grad_b_hh_init_fwd;
  mlir::MlirOp grad_w_ih_init_rev;
  mlir::MlirOp grad_w_hh_init_rev;
  mlir::MlirOp grad_b_ih_init_rev;
  mlir::MlirOp grad_b_hh_init_rev;
  std::optional<mlir::MlirOp> rem_grad_x_fwd;
  std::optional<mlir::MlirOp> rem_grad_x_rev;
};

// Evaluates tail remainder backward steps for bidirectional GRU.
//
// When seq_len is not a multiple of kGruUnrollFactor, rem_steps = seq_len %
// kGruUnrollFactor steps cannot form a full chunk. This helper executes the
// remainder steps concurrently for forward and reverse directions, producing:
// 1. Updated hidden state gradients (delta_h_next_fwd, delta_h_next_rev) to
// seed
//    the chunked backward loop.
// 2. Initialized parameter gradient accumulators containing contributions from
//    the remainder steps.
// 3. Sliced input gradients (rem_grad_x_fwd, rem_grad_x_rev) for concatenation.
//
// Arguments:
//   builder: MLIR StableHLO graph builder.
//   grad_y_fwd: Forward output gradient slice [T, B, H].
//   grad_y_rev_time_reversed: Time-reversed reverse output gradient slice [T,
//   B, H]. delta_h_start_fwd, delta_h_start_rev: Initial incoming hidden state
//   gradients. h_prev_fwd_seq, h_prev_rev_seq: Previous hidden state sequences.
//   act_fwd_seq, act_rev_time_reversed: Pre-activation caches.
//   layer_in, layer_in_time_reversed: Layer input sequences.
//   w_ih_fwd, w_hh_fwd, w_ih_rev, w_hh_rev: Layer weight matrices.
//   hh_bwd_dot_dims, x_bwd_dot_dims, w_dot_dims: Dot dimension configurations.
//   has_biases: Whether bias gradients should be computed.
//   seq_len, chunked_steps, rem_steps, batch, hidden, in_dim: Dimensions.
//   batch_first, concat_dim: Layout flags.
//   acc_dtype, one, zero_scalar, zero_const: Constant helper values.
//   sum_reduce_builder: Reduction region builder.
//   precision: StableHLO dot precision mode.
//
// Returns:
//   GruBidirBackwardRemainderInit containing initialized accumulators and
//   deltas.
template <typename ReduceBuilderFn>
GruBidirBackwardRemainderInit ComputeBidirBackwardRemainder(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_y_fwd,
    mlir::MlirOp grad_y_rev_time_reversed, mlir::MlirOp delta_h_start_fwd,
    mlir::MlirOp delta_h_start_rev, mlir::MlirOp h_prev_fwd_seq,
    mlir::MlirOp h_prev_rev_seq, mlir::MlirOp act_fwd_seq,
    mlir::MlirOp act_rev_time_reversed, mlir::MlirOp layer_in,
    mlir::MlirOp layer_in_time_reversed, mlir::MlirOp w_ih_fwd,
    mlir::MlirOp w_hh_fwd, mlir::MlirOp w_ih_rev, mlir::MlirOp w_hh_rev,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr x_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr w_dot_dims,
    const bool has_biases, const int64_t seq_len, const int64_t chunked_steps,
    const int64_t rem_steps, const int64_t batch, const int64_t hidden,
    const int64_t in_dim, const bool batch_first, const int64_t concat_dim,
    const mlir::Type acc_dtype, mlir::MlirOp one, mlir::MlirOp zero_scalar,
    mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    const mlir::stablehlo::Precision precision) {
  // If sequence length divides evenly into chunks, initialize accumulators to
  // zero.
  if (rem_steps == 0) {
    mlir::MlirOp grad_w_ih_init_fwd = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({3 * hidden, in_dim}, acc_dtype),
        zero_scalar, {});
    mlir::MlirOp grad_w_hh_init_fwd = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({3 * hidden, hidden}, acc_dtype),
        zero_scalar, {});
    mlir::MlirOp grad_w_ih_init_rev = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({3 * hidden, in_dim}, acc_dtype),
        zero_scalar, {});
    mlir::MlirOp grad_w_hh_init_rev = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({3 * hidden, hidden}, acc_dtype),
        zero_scalar, {});
    mlir::MlirOp grad_b_ih_init_fwd;
    mlir::MlirOp grad_b_hh_init_fwd;
    mlir::MlirOp grad_b_ih_init_rev;
    mlir::MlirOp grad_b_hh_init_rev;
    if (has_biases) {
      grad_b_ih_init_fwd = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({3 * hidden}, acc_dtype), zero_scalar,
          {});
      grad_b_hh_init_fwd = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({3 * hidden}, acc_dtype), zero_scalar,
          {});
      grad_b_ih_init_rev = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({3 * hidden}, acc_dtype), zero_scalar,
          {});
      grad_b_hh_init_rev = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({3 * hidden}, acc_dtype), zero_scalar,
          {});
    }
    return {delta_h_start_fwd,  delta_h_start_rev,  grad_w_ih_init_fwd,
            grad_w_hh_init_fwd, grad_b_ih_init_fwd, grad_b_hh_init_fwd,
            grad_w_ih_init_rev, grad_w_hh_init_rev, grad_b_ih_init_rev,
            grad_b_hh_init_rev, std::nullopt,       std::nullopt};
  }

  // Pre-allocate step deltas for remainder time steps.
  std::vector<mlir::MlirOp> delta_pre_ig_fwd_rem(rem_steps);
  std::vector<mlir::MlirOp> delta_pre_hg_fwd_rem(rem_steps);
  std::vector<mlir::MlirOp> grad_x_fwd_rem_steps(rem_steps);
  std::vector<mlir::MlirOp> delta_pre_ig_rev_rem(rem_steps);
  std::vector<mlir::MlirOp> delta_pre_hg_rev_rem(rem_steps);
  std::vector<mlir::MlirOp> grad_x_rev_rem_steps(rem_steps);

  mlir::MlirOp delta_h_next_fwd = delta_h_start_fwd;
  mlir::MlirOp delta_h_next_rev = delta_h_start_rev;

  // Unroll tail remainder steps from t = seq_len - 1 down to chunked_steps.
  for (int64_t t = seq_len - 1; t >= chunked_steps; --t) {
    const int64_t rem_idx = t - chunked_steps;

    // Step 1: Forward remainder step at time index t.
    mlir::MlirOp gy_fwd_t =
        SliceStep2D(grad_y_fwd, t, batch, hidden, batch_first);
    mlir::MlirOp dh_fwd = mlir::stablehlo::Add(gy_fwd_t, delta_h_next_fwd);
    mlir::MlirOp hp_fwd_t =
        SliceStep2D(h_prev_fwd_seq, t, batch, hidden, batch_first);
    mlir::MlirOp act_fwd_t =
        SliceStep2D(act_fwd_seq, t, batch, 4 * hidden, batch_first);

    const GruBackwardStepResult step_fwd =
        ComputeGruBackwardStep(builder, dh_fwd, hp_fwd_t, act_fwd_t, w_hh_fwd,
                               hh_bwd_dot_dims, batch, hidden, one, precision);
    delta_h_next_fwd = step_fwd.delta_h_prev;
    delta_pre_ig_fwd_rem[rem_idx] =
        ExpandStep3D(step_fwd.delta_pre_ig, batch, 3 * hidden, batch_first);
    delta_pre_hg_fwd_rem[rem_idx] =
        ExpandStep3D(step_fwd.delta_pre_hg, batch, 3 * hidden, batch_first);
    mlir::MlirOp dx_fwd_t = MixedPrecisionDotGeneral(
        step_fwd.delta_pre_ig, w_ih_fwd, x_bwd_dot_dims, acc_dtype, precision);
    grad_x_fwd_rem_steps[rem_idx] =
        ExpandStep3D(dx_fwd_t, batch, in_dim, batch_first);

    // Step 2: Reverse remainder step at time index t (operating on
    // time-reversed data).
    mlir::MlirOp gy_rev_t =
        SliceStep2D(grad_y_rev_time_reversed, t, batch, hidden, batch_first);
    mlir::MlirOp dh_rev = mlir::stablehlo::Add(gy_rev_t, delta_h_next_rev);
    mlir::MlirOp hp_rev_t =
        SliceStep2D(h_prev_rev_seq, t, batch, hidden, batch_first);
    mlir::MlirOp act_rev_t =
        SliceStep2D(act_rev_time_reversed, t, batch, 4 * hidden, batch_first);

    const GruBackwardStepResult step_rev =
        ComputeGruBackwardStep(builder, dh_rev, hp_rev_t, act_rev_t, w_hh_rev,
                               hh_bwd_dot_dims, batch, hidden, one, precision);
    delta_h_next_rev = step_rev.delta_h_prev;
    delta_pre_ig_rev_rem[rem_idx] =
        ExpandStep3D(step_rev.delta_pre_ig, batch, 3 * hidden, batch_first);
    delta_pre_hg_rev_rem[rem_idx] =
        ExpandStep3D(step_rev.delta_pre_hg, batch, 3 * hidden, batch_first);
    mlir::MlirOp dx_rev_t = MixedPrecisionDotGeneral(
        step_rev.delta_pre_ig, w_ih_rev, x_bwd_dot_dims, acc_dtype, precision);
    grad_x_rev_rem_steps[rem_idx] =
        ExpandStep3D(dx_rev_t, batch, in_dim, batch_first);
  }

  // Concatenate remainder step deltas into 2D matrices for GEMM parameter
  // gradients.
  mlir::MlirOp delta_pre_ig_fwd_rem_2d = mlir::stablehlo::Reshape(
      ConcatDim(builder, delta_pre_ig_fwd_rem, concat_dim),
      {rem_steps * batch, 3 * hidden});
  mlir::MlirOp delta_pre_hg_fwd_rem_2d = mlir::stablehlo::Reshape(
      ConcatDim(builder, delta_pre_hg_fwd_rem, concat_dim),
      {rem_steps * batch, 3 * hidden});

  mlir::MlirOp delta_pre_ig_rev_rem_2d = mlir::stablehlo::Reshape(
      ConcatDim(builder, delta_pre_ig_rev_rem, concat_dim),
      {rem_steps * batch, 3 * hidden});
  mlir::MlirOp delta_pre_hg_rev_rem_2d = mlir::stablehlo::Reshape(
      ConcatDim(builder, delta_pre_hg_rev_rem, concat_dim),
      {rem_steps * batch, 3 * hidden});

  // Slice layer inputs and previous states corresponding to the remainder
  // steps.
  mlir::MlirOp rem_layer_in_fwd =
      batch_first ? mlir::stablehlo::Slice(layer_in, {0, chunked_steps, 0},
                                           {batch, seq_len, in_dim}, {1, 1, 1})
                  : mlir::stablehlo::Slice(layer_in, {chunked_steps, 0, 0},
                                           {seq_len, batch, in_dim}, {1, 1, 1});
  mlir::MlirOp rem_layer_in_fwd_2d =
      mlir::stablehlo::Reshape(rem_layer_in_fwd, {rem_steps * batch, in_dim});

  mlir::MlirOp rem_layer_in_rev =
      batch_first ? mlir::stablehlo::Slice(layer_in_time_reversed,
                                           {0, chunked_steps, 0},
                                           {batch, seq_len, in_dim}, {1, 1, 1})
                  : mlir::stablehlo::Slice(layer_in_time_reversed,
                                           {chunked_steps, 0, 0},
                                           {seq_len, batch, in_dim}, {1, 1, 1});
  mlir::MlirOp rem_layer_in_rev_2d =
      mlir::stablehlo::Reshape(rem_layer_in_rev, {rem_steps * batch, in_dim});

  mlir::MlirOp rem_h_prev_fwd =
      batch_first
          ? mlir::stablehlo::Slice(h_prev_fwd_seq, {0, chunked_steps, 0},
                                   {batch, seq_len, hidden}, {1, 1, 1})
          : mlir::stablehlo::Slice(h_prev_fwd_seq, {chunked_steps, 0, 0},
                                   {seq_len, batch, hidden}, {1, 1, 1});
  mlir::MlirOp rem_h_prev_fwd_2d =
      mlir::stablehlo::Reshape(rem_h_prev_fwd, {rem_steps * batch, hidden});

  mlir::MlirOp rem_h_prev_rev =
      batch_first
          ? mlir::stablehlo::Slice(h_prev_rev_seq, {0, chunked_steps, 0},
                                   {batch, seq_len, hidden}, {1, 1, 1})
          : mlir::stablehlo::Slice(h_prev_rev_seq, {chunked_steps, 0, 0},
                                   {seq_len, batch, hidden}, {1, 1, 1});
  mlir::MlirOp rem_h_prev_rev_2d =
      mlir::stablehlo::Reshape(rem_h_prev_rev, {rem_steps * batch, hidden});

  // Parameter gradients for remainder steps.
  mlir::MlirOp grad_w_ih_init_fwd =
      MixedPrecisionDotGeneral(delta_pre_ig_fwd_rem_2d, rem_layer_in_fwd_2d,
                               w_dot_dims, acc_dtype, precision);
  mlir::MlirOp grad_w_hh_init_fwd =
      MixedPrecisionDotGeneral(delta_pre_hg_fwd_rem_2d, rem_h_prev_fwd_2d,
                               w_dot_dims, acc_dtype, precision);

  mlir::MlirOp grad_w_ih_init_rev =
      MixedPrecisionDotGeneral(delta_pre_ig_rev_rem_2d, rem_layer_in_rev_2d,
                               w_dot_dims, acc_dtype, precision);
  mlir::MlirOp grad_w_hh_init_rev =
      MixedPrecisionDotGeneral(delta_pre_hg_rev_rem_2d, rem_h_prev_rev_2d,
                               w_dot_dims, acc_dtype, precision);

  mlir::MlirOp grad_b_ih_init_fwd;
  mlir::MlirOp grad_b_hh_init_fwd;
  mlir::MlirOp grad_b_ih_init_rev;
  mlir::MlirOp grad_b_hh_init_rev;
  if (has_biases) {
    grad_b_ih_init_fwd =
        mlir::stablehlo::Reduce(builder, delta_pre_ig_fwd_rem_2d, zero_const,
                                sum_reduce_builder, {0})[0];
    grad_b_hh_init_fwd =
        mlir::stablehlo::Reduce(builder, delta_pre_hg_fwd_rem_2d, zero_const,
                                sum_reduce_builder, {0})[0];
    grad_b_ih_init_rev =
        mlir::stablehlo::Reduce(builder, delta_pre_ig_rev_rem_2d, zero_const,
                                sum_reduce_builder, {0})[0];
    grad_b_hh_init_rev =
        mlir::stablehlo::Reduce(builder, delta_pre_hg_rev_rem_2d, zero_const,
                                sum_reduce_builder, {0})[0];
  }

  mlir::MlirOp rem_grad_x_fwd =
      ConcatDim(builder, grad_x_fwd_rem_steps, concat_dim);
  mlir::MlirOp rem_grad_x_rev =
      ConcatDim(builder, grad_x_rev_rem_steps, concat_dim);

  return {delta_h_next_fwd,   delta_h_next_rev,   grad_w_ih_init_fwd,
          grad_w_hh_init_fwd, grad_b_ih_init_fwd, grad_b_hh_init_fwd,
          grad_w_ih_init_rev, grad_w_hh_init_rev, grad_b_ih_init_rev,
          grad_b_hh_init_rev, rem_grad_x_fwd,     rem_grad_x_rev};
}

// Intermediate result of executing kGruUnrollFactor backward steps concurrently
// for both forward and reverse directions.
struct GruBidirBackwardChunkStepsResult {
  mlir::MlirOp cur_delta_h_fwd;
  mlir::MlirOp cur_delta_h_rev;
  mlir::MlirOp delta_pre_ig_fwd_chunk_2d;
  mlir::MlirOp delta_pre_hg_fwd_chunk_2d;
  mlir::MlirOp delta_pre_ig_rev_chunk_2d;
  mlir::MlirOp delta_pre_hg_rev_chunk_2d;
  mlir::MlirOp chunk_grad_x_fwd;
  mlir::MlirOp chunk_grad_x_rev;
};

// Computes kGruUnrollFactor backward steps concurrently for forward and reverse
// directions.
//
// Unrolls kGruUnrollFactor (8) backward steps in parallel across forward and
// reverse directions:
// - Slices the chunk tensors at sub-step index k (from 7 down to 0).
// - Computes ComputeGruBackwardStep for forward and reverse streams.
// - Performs mixed precision GEMM to compute chunk input gradients
// (chunk_grad_x).
// - Reshapes pre-activation gate deltas into 2D chunk matrices for batched
//   parameter gradient accumulation.
//
// Arguments:
//   builder: MLIR StableHLO graph builder.
//   grad_y_chunk_fwd, grad_y_chunk_rev: Slices of output gradients for chunk.
//   act_chunk_fwd, act_chunk_rev: Slices of forward/reverse activation cache.
//   h_prev_chunk_fwd, h_prev_chunk_rev: Slices of previous hidden states.
//   cur_delta_h_fwd_in, cur_delta_h_rev_in: Incoming hidden state gradients.
//   w_hh_fwd, w_ih_fwd, w_hh_rev, w_ih_rev: Layer weight matrices.
//   hh_bwd_dot_dims, x_bwd_dot_dims: Dot dimension configurations.
//   batch, hidden, in_dim: Layer dimensions.
//   batch_first, concat_dim: Layout flags.
//   acc_dtype, one: Precision and constant helpers.
//   precision: StableHLO dot precision mode.
//
// Returns:
//   GruBidirBackwardChunkStepsResult with updated deltas, 2D chunk matrices,
//   and chunk input gradients.
inline GruBidirBackwardChunkStepsResult ComputeBidirBackwardChunkSteps(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_y_chunk_fwd,
    mlir::MlirOp grad_y_chunk_rev, mlir::MlirOp act_chunk_fwd,
    mlir::MlirOp act_chunk_rev, mlir::MlirOp h_prev_chunk_fwd,
    mlir::MlirOp h_prev_chunk_rev, mlir::MlirOp cur_delta_h_fwd_in,
    mlir::MlirOp cur_delta_h_rev_in, mlir::MlirOp w_hh_fwd,
    mlir::MlirOp w_ih_fwd, mlir::MlirOp w_hh_rev, mlir::MlirOp w_ih_rev,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr x_bwd_dot_dims,
    const int64_t batch, const int64_t hidden, const int64_t in_dim,
    const bool batch_first, const int64_t concat_dim,
    const mlir::Type acc_dtype, mlir::MlirOp one,
    const mlir::stablehlo::Precision precision) {
  std::vector<mlir::MlirOp> delta_pre_ig_fwd_k(kGruUnrollFactor);
  std::vector<mlir::MlirOp> delta_pre_hg_fwd_k(kGruUnrollFactor);

  std::vector<mlir::MlirOp> delta_pre_ig_rev_k(kGruUnrollFactor);
  std::vector<mlir::MlirOp> delta_pre_hg_rev_k(kGruUnrollFactor);

  mlir::MlirOp cur_delta_h_fwd = cur_delta_h_fwd_in;
  mlir::MlirOp cur_delta_h_rev = cur_delta_h_rev_in;

  for (int64_t k = kGruUnrollFactor - 1; k >= 0; --k) {
    // Step 1: Forward direction step k.
    mlir::MlirOp gy_fwd_step =
        SliceStep2D(grad_y_chunk_fwd, k, batch, hidden, batch_first);
    mlir::MlirOp dh_fwd = mlir::stablehlo::Add(gy_fwd_step, cur_delta_h_fwd);
    mlir::MlirOp hp_fwd_step =
        SliceStep2D(h_prev_chunk_fwd, k, batch, hidden, batch_first);
    mlir::MlirOp act_fwd_step =
        SliceStep2D(act_chunk_fwd, k, batch, 4 * hidden, batch_first);

    const GruBackwardStepResult step_fwd = ComputeGruBackwardStep(
        builder, dh_fwd, hp_fwd_step, act_fwd_step, w_hh_fwd, hh_bwd_dot_dims,
        batch, hidden, one, precision);
    cur_delta_h_fwd = step_fwd.delta_h_prev;
    delta_pre_ig_fwd_k[k] =
        ExpandStep3D(step_fwd.delta_pre_ig, batch, 3 * hidden, batch_first);
    delta_pre_hg_fwd_k[k] =
        ExpandStep3D(step_fwd.delta_pre_hg, batch, 3 * hidden, batch_first);

    // Step 2: Reverse direction step k (operating on time-reversed chunk).
    mlir::MlirOp gy_rev_step =
        SliceStep2D(grad_y_chunk_rev, k, batch, hidden, batch_first);
    mlir::MlirOp dh_rev = mlir::stablehlo::Add(gy_rev_step, cur_delta_h_rev);
    mlir::MlirOp hp_rev_step =
        SliceStep2D(h_prev_chunk_rev, k, batch, hidden, batch_first);
    mlir::MlirOp act_rev_step =
        SliceStep2D(act_chunk_rev, k, batch, 4 * hidden, batch_first);

    const GruBackwardStepResult step_rev = ComputeGruBackwardStep(
        builder, dh_rev, hp_rev_step, act_rev_step, w_hh_rev, hh_bwd_dot_dims,
        batch, hidden, one, precision);
    cur_delta_h_rev = step_rev.delta_h_prev;
    delta_pre_ig_rev_k[k] =
        ExpandStep3D(step_rev.delta_pre_ig, batch, 3 * hidden, batch_first);
    delta_pre_hg_rev_k[k] =
        ExpandStep3D(step_rev.delta_pre_hg, batch, 3 * hidden, batch_first);
  }

  // Concatenate step deltas into 2D matrices [k*B, 3*H].
  const mlir::MlirOp delta_pre_ig_fwd_chunk_2d = mlir::stablehlo::Reshape(
      ConcatDim(builder, delta_pre_ig_fwd_k, concat_dim),
      {kGruUnrollFactor * batch, 3 * hidden});
  const mlir::MlirOp delta_pre_hg_fwd_chunk_2d = mlir::stablehlo::Reshape(
      ConcatDim(builder, delta_pre_hg_fwd_k, concat_dim),
      {kGruUnrollFactor * batch, 3 * hidden});
  const mlir::MlirOp delta_pre_ig_rev_chunk_2d = mlir::stablehlo::Reshape(
      ConcatDim(builder, delta_pre_ig_rev_k, concat_dim),
      {kGruUnrollFactor * batch, 3 * hidden});
  const mlir::MlirOp delta_pre_hg_rev_chunk_2d = mlir::stablehlo::Reshape(
      ConcatDim(builder, delta_pre_hg_rev_k, concat_dim),
      {kGruUnrollFactor * batch, 3 * hidden});

  const llvm::SmallVector<int64_t, 3> chunk_grad_x_dims =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kGruUnrollFactor, in_dim}
          : llvm::SmallVector<int64_t, 3>{kGruUnrollFactor, batch, in_dim};

  // Single 2D Batch GEMMs for forward and reverse directions:
  // [k * B, 3*H] x [3*H, in_dim] -> [k * B, in_dim]
  const mlir::MlirOp chunk_grad_x_fwd_2d =
      MixedPrecisionDotGeneral(delta_pre_ig_fwd_chunk_2d, w_ih_fwd,
                               x_bwd_dot_dims, acc_dtype, precision);
  const mlir::MlirOp chunk_grad_x_fwd =
      mlir::stablehlo::Reshape(chunk_grad_x_fwd_2d, chunk_grad_x_dims);

  const mlir::MlirOp chunk_grad_x_rev_2d =
      MixedPrecisionDotGeneral(delta_pre_ig_rev_chunk_2d, w_ih_rev,
                               x_bwd_dot_dims, acc_dtype, precision);
  const mlir::MlirOp chunk_grad_x_rev =
      mlir::stablehlo::Reshape(chunk_grad_x_rev_2d, chunk_grad_x_dims);

  return {cur_delta_h_fwd,           cur_delta_h_rev,
          delta_pre_ig_fwd_chunk_2d, delta_pre_hg_fwd_chunk_2d,
          delta_pre_ig_rev_chunk_2d, delta_pre_hg_rev_chunk_2d,
          chunk_grad_x_fwd,          chunk_grad_x_rev};
}

// Computes the chunked recurrence backward pass for a bidirectional GRU layer
// when seq_len >= kGruUnrollFactor.
//
// Key Optimizations:
// 1. Time-Reversal Alignment: Uses mlir::stablehlo::Reverse along the sequence
//    dimension for reverse gradients and activations so both forward and
//    reverse backward sweeps progress in the same loop direction concurrently.
// 2. Tail Remainder Execution: Runs unaligned remainder steps first, seeding
//    the while loop with accurate hidden deltas and parameter accumulators.
// 3. Chunked Recurrence Loop (WhileOp): Unrolls kGruUnrollFactor steps per loop
//    iteration, accumulating parameter gradients streaming in registers to
//    maximize TPU VPU/XLU compute intensity.
// 4. In-Place Gradient Buffering: Updates grad_x dynamically using
//    mlir::stablehlo::DynamicUpdateSlice.
//
// Arguments:
//   builder: MLIR StableHLO graph builder.
//   grad_y: Incoming output gradient tensor [T, B, 2*H].
//   grad_hy: Incoming hidden state gradient [2, B, H].
//   layer_in: Input sequence tensor [T, B, in_dim].
//   h_init_fwd, h_init_rev: Initial forward and reverse hidden states [B, H].
//   y_out_fwd, y_out_rev: Forward and reverse output sequences [T, B, H].
//   act_fwd_seq, act_rev_seq: Forward and reverse pre-activation caches.
//   w_ih_fwd, w_hh_fwd, w_ih_rev, w_hh_rev: Layer weight matrices.
//   has_biases: Whether biases are present.
//   seq_len, batch, hidden, in_dim: Layer dimensions.
//   batch_first: Whether batch is dimension 0.
//   to_out: Functor converting tensors to output datatype.
//   zero_const: Constant scalar zero for bias reduction.
//   sum_reduce_builder: Reduction region builder.
//   precision: StableHLO dot precision mode.
//
// Returns:
//   GruBidirLayerBackwardOutputs containing all input and parameter gradients.
// Builds the body block for the chunked bidirectional backward while loop.
//
// What it computes:
// - Slices chunk inputs for forward and time-reversed reverse directions.
// - Unrolls 8 backward steps concurrently for forward and reverse sweeps via
// ComputeBidirBackwardChunkSteps.
// - Accumulates streaming parameter gradients in registers for both directions:
//     next_gw_ih_fwd = cur_gw_ih_fwd + delta_pre_ig_fwd_2d^T @
//     layer_in_chunk_fwd_2d next_gw_hh_fwd = cur_gw_hh_fwd +
//     delta_pre_hg_fwd_2d^T @ h_prev_chunk_fwd_2d next_gw_ih_rev =
//     cur_gw_ih_rev + delta_pre_ig_rev_2d^T @ layer_in_chunk_rev_2d
//     next_gw_hh_rev = cur_gw_hh_rev + delta_pre_hg_rev_2d^T @
//     h_prev_chunk_rev_2d
// - If biases enabled, accumulates bias reductions for forward and reverse.
// - Updates forward and reverse chunked input gradients dynamically.
// - Increments body_step_idx by 8.
template <typename ReduceBuilderFn>
void BuildBidirLayerBackwardChunkedWhileBody(
    mlir::MlirBuilder& builder, mlir::Block* body_block, mlir::Location loc,
    mlir::MlirOp grad_y_fwd, mlir::MlirOp grad_y_rev_time_reversed,
    mlir::MlirOp act_fwd_seq, mlir::MlirOp act_rev_time_reversed,
    mlir::MlirOp h_prev_fwd_seq, mlir::MlirOp h_prev_rev_seq,
    mlir::MlirOp layer_in, mlir::MlirOp layer_in_time_reversed,
    mlir::MlirOp w_hh_fwd, mlir::MlirOp w_ih_fwd, mlir::MlirOp w_hh_rev,
    mlir::MlirOp w_ih_rev,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr x_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr w_dot_dims,
    int64_t chunked_steps, int64_t batch, int64_t hidden, int64_t in_dim,
    bool batch_first, int64_t concat_dim, mlir::Type acc_dtype,
    mlir::MlirOp one, bool has_biases, mlir::MlirOp zero_const,
    ReduceBuilderFn sum_reduce_builder, mlir::stablehlo::Precision precision) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::IntegerType i64 = op_builder.getI64Type();

  mlir::MlirOp body_step_idx(builder, body_block->getArgument(0));
  mlir::MlirOp body_delta_h_fwd(builder, body_block->getArgument(1));
  mlir::MlirOp body_delta_h_rev(builder, body_block->getArgument(2));
  mlir::MlirOp body_grad_x_fwd_chunked(builder, body_block->getArgument(3));
  mlir::MlirOp body_grad_x_rev_chunked(builder, body_block->getArgument(4));
  mlir::MlirOp body_grad_w_ih_fwd(builder, body_block->getArgument(5));
  mlir::MlirOp body_grad_w_hh_fwd(builder, body_block->getArgument(6));
  mlir::MlirOp body_grad_w_ih_rev(builder, body_block->getArgument(7));
  mlir::MlirOp body_grad_w_hh_rev(builder, body_block->getArgument(8));

  mlir::MlirOp body_grad_b_ih_fwd;
  mlir::MlirOp body_grad_b_hh_fwd;
  mlir::MlirOp body_grad_b_ih_rev;
  mlir::MlirOp body_grad_b_hh_rev;
  if (has_biases) {
    body_grad_b_ih_fwd = mlir::MlirOp(builder, body_block->getArgument(9));
    body_grad_b_hh_fwd = mlir::MlirOp(builder, body_block->getArgument(10));
    body_grad_b_ih_rev = mlir::MlirOp(builder, body_block->getArgument(11));
    body_grad_b_hh_rev = mlir::MlirOp(builder, body_block->getArgument(12));
  }

  // Calculate chunk start index for forward sweep (reverse chronological: T -
  // 8, T - 16, ...)
  mlir::MlirOp const_chunk_limit =
      MakeScalarConstant(builder, chunked_steps - kGruUnrollFactor, i64);
  mlir::MlirOp chunk_start_t =
      mlir::stablehlo::Subtract(const_chunk_limit, body_step_idx);

  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64);
  const llvm::SmallVector<mlir::MlirOp, 3> fwd_chunk_start_indices =
      batch_first ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, chunk_start_t,
                                                       zero_i64}
                  : llvm::SmallVector<mlir::MlirOp, 3>{chunk_start_t, zero_i64,
                                                       zero_i64};

  // Reverse chunk start index progresses chronologically along the
  // time-reversed buffer
  const llvm::SmallVector<mlir::MlirOp, 3> rev_chunk_start_indices =
      batch_first ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, body_step_idx,
                                                       zero_i64}
                  : llvm::SmallVector<mlir::MlirOp, 3>{body_step_idx, zero_i64,
                                                       zero_i64};

  const llvm::SmallVector<int64_t, 3> grad_y_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kGruUnrollFactor, hidden}
          : llvm::SmallVector<int64_t, 3>{kGruUnrollFactor, batch, hidden};
  const llvm::SmallVector<int64_t, 3> act_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kGruUnrollFactor, 4 * hidden}
          : llvm::SmallVector<int64_t, 3>{kGruUnrollFactor, batch, 4 * hidden};
  const llvm::SmallVector<int64_t, 3> in_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kGruUnrollFactor, in_dim}
          : llvm::SmallVector<int64_t, 3>{kGruUnrollFactor, batch, in_dim};

  // Slice chunk inputs for forward and reverse directions
  mlir::MlirOp grad_y_chunk_fwd = mlir::stablehlo::DynamicSlice(
      grad_y_fwd, fwd_chunk_start_indices, grad_y_slice_sizes);
  mlir::MlirOp act_chunk_fwd = mlir::stablehlo::DynamicSlice(
      act_fwd_seq, fwd_chunk_start_indices, act_slice_sizes);
  mlir::MlirOp h_prev_chunk_fwd = mlir::stablehlo::DynamicSlice(
      h_prev_fwd_seq, fwd_chunk_start_indices, grad_y_slice_sizes);
  mlir::MlirOp layer_in_chunk_fwd = mlir::stablehlo::DynamicSlice(
      layer_in, fwd_chunk_start_indices, in_slice_sizes);

  mlir::MlirOp grad_y_chunk_rev = mlir::stablehlo::DynamicSlice(
      grad_y_rev_time_reversed, rev_chunk_start_indices, grad_y_slice_sizes);
  mlir::MlirOp act_chunk_rev = mlir::stablehlo::DynamicSlice(
      act_rev_time_reversed, rev_chunk_start_indices, act_slice_sizes);
  mlir::MlirOp h_prev_chunk_rev = mlir::stablehlo::DynamicSlice(
      h_prev_rev_seq, rev_chunk_start_indices, grad_y_slice_sizes);
  mlir::MlirOp layer_in_chunk_rev = mlir::stablehlo::DynamicSlice(
      layer_in_time_reversed, rev_chunk_start_indices, in_slice_sizes);

  // Compute unrolled backward steps for this chunk
  GruBidirBackwardChunkStepsResult chunk_steps = ComputeBidirBackwardChunkSteps(
      builder, grad_y_chunk_fwd, grad_y_chunk_rev, act_chunk_fwd, act_chunk_rev,
      h_prev_chunk_fwd, h_prev_chunk_rev, body_delta_h_fwd, body_delta_h_rev,
      w_hh_fwd, w_ih_fwd, w_hh_rev, w_ih_rev, hh_bwd_dot_dims, x_bwd_dot_dims,
      batch, hidden, in_dim, batch_first, concat_dim, acc_dtype, one,
      precision);

  // Accumulate forward parameter gradients streaming in registers
  mlir::MlirOp layer_in_chunk_fwd_2d = mlir::stablehlo::Reshape(
      layer_in_chunk_fwd, {kGruUnrollFactor * batch, in_dim});
  mlir::MlirOp h_prev_chunk_fwd_2d = mlir::stablehlo::Reshape(
      h_prev_chunk_fwd, {kGruUnrollFactor * batch, hidden});

  mlir::MlirOp chunk_gw_ih_fwd = MixedPrecisionDotGeneral(
      chunk_steps.delta_pre_ig_fwd_chunk_2d, layer_in_chunk_fwd_2d, w_dot_dims,
      acc_dtype, precision);
  mlir::MlirOp next_gw_ih_fwd =
      mlir::stablehlo::Add(body_grad_w_ih_fwd, chunk_gw_ih_fwd);

  mlir::MlirOp chunk_gw_hh_fwd = MixedPrecisionDotGeneral(
      chunk_steps.delta_pre_hg_fwd_chunk_2d, h_prev_chunk_fwd_2d, w_dot_dims,
      acc_dtype, precision);
  mlir::MlirOp next_gw_hh_fwd =
      mlir::stablehlo::Add(body_grad_w_hh_fwd, chunk_gw_hh_fwd);

  // Accumulate reverse parameter gradients streaming in registers
  mlir::MlirOp layer_in_chunk_rev_2d = mlir::stablehlo::Reshape(
      layer_in_chunk_rev, {kGruUnrollFactor * batch, in_dim});
  mlir::MlirOp h_prev_chunk_rev_2d = mlir::stablehlo::Reshape(
      h_prev_chunk_rev, {kGruUnrollFactor * batch, hidden});

  mlir::MlirOp chunk_gw_ih_rev = MixedPrecisionDotGeneral(
      chunk_steps.delta_pre_ig_rev_chunk_2d, layer_in_chunk_rev_2d, w_dot_dims,
      acc_dtype, precision);
  mlir::MlirOp next_gw_ih_rev =
      mlir::stablehlo::Add(body_grad_w_ih_rev, chunk_gw_ih_rev);

  mlir::MlirOp chunk_gw_hh_rev = MixedPrecisionDotGeneral(
      chunk_steps.delta_pre_hg_rev_chunk_2d, h_prev_chunk_rev_2d, w_dot_dims,
      acc_dtype, precision);
  mlir::MlirOp next_gw_hh_rev =
      mlir::stablehlo::Add(body_grad_w_hh_rev, chunk_gw_hh_rev);

  mlir::MlirOp next_gb_ih_fwd;
  mlir::MlirOp next_gb_hh_fwd;
  mlir::MlirOp next_gb_ih_rev;
  mlir::MlirOp next_gb_hh_rev;
  if (has_biases) {
    mlir::MlirOp chunk_gb_ih_fwd =
        mlir::stablehlo::Reduce(builder, chunk_steps.delta_pre_ig_fwd_chunk_2d,
                                zero_const, sum_reduce_builder, {0})[0];
    next_gb_ih_fwd = mlir::stablehlo::Add(body_grad_b_ih_fwd, chunk_gb_ih_fwd);

    mlir::MlirOp chunk_gb_hh_fwd =
        mlir::stablehlo::Reduce(builder, chunk_steps.delta_pre_hg_fwd_chunk_2d,
                                zero_const, sum_reduce_builder, {0})[0];
    next_gb_hh_fwd = mlir::stablehlo::Add(body_grad_b_hh_fwd, chunk_gb_hh_fwd);

    mlir::MlirOp chunk_gb_ih_rev =
        mlir::stablehlo::Reduce(builder, chunk_steps.delta_pre_ig_rev_chunk_2d,
                                zero_const, sum_reduce_builder, {0})[0];
    next_gb_ih_rev = mlir::stablehlo::Add(body_grad_b_ih_rev, chunk_gb_ih_rev);

    mlir::MlirOp chunk_gb_hh_rev =
        mlir::stablehlo::Reduce(builder, chunk_steps.delta_pre_hg_rev_chunk_2d,
                                zero_const, sum_reduce_builder, {0})[0];
    next_gb_hh_rev = mlir::stablehlo::Add(body_grad_b_hh_rev, chunk_gb_hh_rev);
  }

  // Update chunked input gradients in place
  mlir::MlirOp next_grad_x_fwd_chunked = mlir::stablehlo::DynamicUpdateSlice(
      body_grad_x_fwd_chunked, chunk_steps.chunk_grad_x_fwd,
      fwd_chunk_start_indices);

  mlir::MlirOp next_grad_x_rev_chunked = mlir::stablehlo::DynamicUpdateSlice(
      body_grad_x_rev_chunked, chunk_steps.chunk_grad_x_rev,
      rev_chunk_start_indices);

  mlir::MlirOp step_k = MakeScalarConstant(builder, kGruUnrollFactor, i64);
  mlir::MlirOp next_step_idx = mlir::stablehlo::Add(body_step_idx, step_k);

  llvm::SmallVector<mlir::Value> next_loop_values = {
      next_step_idx.getValue(),
      chunk_steps.cur_delta_h_fwd.getValue(),
      chunk_steps.cur_delta_h_rev.getValue(),
      next_grad_x_fwd_chunked.getValue(),
      next_grad_x_rev_chunked.getValue(),
      next_gw_ih_fwd.getValue(),
      next_gw_hh_fwd.getValue(),
      next_gw_ih_rev.getValue(),
      next_gw_hh_rev.getValue()};

  if (has_biases) {
    next_loop_values.push_back(next_gb_ih_fwd.getValue());
    next_loop_values.push_back(next_gb_hh_fwd.getValue());
    next_loop_values.push_back(next_gb_ih_rev.getValue());
    next_loop_values.push_back(next_gb_hh_rev.getValue());
  }
  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_loop_values);
}

// Computes the chunked recurrence backward pass for a bidirectional GRU layer
// when seq_len >= kGruUnrollFactor.
//
// Key Optimizations:
// 1. Time-Reversal Alignment: Uses mlir::stablehlo::Reverse along the sequence
//    dimension for reverse gradients and activations so both forward and
//    reverse backward sweeps progress in the same loop direction concurrently.
// 2. Tail Remainder Execution: Runs unaligned remainder steps first, seeding
//    the while loop with accurate hidden deltas and parameter accumulators.
// 3. Chunked Recurrence Loop (WhileOp): Delegates while body to
// BuildBidirLayerBackwardChunkedWhileBody.
// 4. In-Place Gradient Buffering: Updates grad_x dynamically using
//    mlir::stablehlo::DynamicUpdateSlice.
//
// Arguments:
//   builder: MLIR StableHLO graph builder.
//   grad_y: Incoming output gradient tensor [T, B, 2*H].
//   grad_hy: Incoming hidden state gradient [2, B, H].
//   layer_in: Input sequence tensor [T, B, in_dim].
//   h_init_fwd, h_init_rev: Initial forward and reverse hidden states [B, H].
//   y_out_fwd, y_out_rev: Forward and reverse output sequences [T, B, H].
//   act_fwd_seq, act_rev_seq: Forward and reverse pre-activation caches.
//   w_ih_fwd, w_hh_fwd, w_ih_rev, w_hh_rev: Layer weight matrices.
//   has_biases: Whether biases are present.
//   seq_len, batch, hidden, in_dim: Layer dimensions.
//   batch_first: Whether batch is dimension 0.
//   to_out: Functor converting tensors to output datatype.
//   zero_const: Constant scalar zero for bias reduction.
//   sum_reduce_builder: Reduction region builder.
//   precision: StableHLO dot precision mode.
//
// Returns:
//   GruBidirLayerBackwardOutputs containing all input and parameter gradients.
template <typename ToOutFn, typename ReduceBuilderFn>
GruBidirLayerBackwardOutputs ComputeBidirLayerBackwardChunked(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_y, mlir::MlirOp grad_hy,
    mlir::MlirOp layer_in, mlir::MlirOp h_init_fwd, mlir::MlirOp h_init_rev,
    mlir::MlirOp y_out_fwd, mlir::MlirOp y_out_rev, mlir::MlirOp act_fwd_seq,
    mlir::MlirOp act_rev_seq, mlir::MlirOp w_ih_fwd, mlir::MlirOp w_hh_fwd,
    mlir::MlirOp w_ih_rev, mlir::MlirOp w_hh_rev, const bool has_biases,
    const int64_t seq_len, const int64_t batch, const int64_t hidden,
    const int64_t in_dim, const bool batch_first, ToOutFn to_out,
    mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    const mlir::stablehlo::Precision precision) {
  const int64_t num_chunks = seq_len / kGruUnrollFactor;
  const int64_t chunked_steps = num_chunks * kGruUnrollFactor;
  const int64_t rem_steps = seq_len % kGruUnrollFactor;
  const int64_t concat_dim = batch_first ? 1 : 0;

  mlir::MLIRContext& ctx = builder.getContext();
  const auto hh_bwd_dot_dims = MakeDotDims(&ctx, {1}, {0});
  const auto x_bwd_dot_dims = MakeDotDims(&ctx, {1}, {0});
  const auto w_dot_dims = MakeDotDims(&ctx, {0}, {0});

  // Promote all inputs to accumulator precision (F32).
  const mlir::Type acc_dtype = GetTensorTypeOrDie(grad_hy).getElementType();
  auto to_acc = [acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
    if (GetTensorTypeOrDie(op).getElementType() == acc_dtype) {
      return op;
    }
    return mlir::stablehlo::ConvertElementType(op, acc_dtype);
  };
  grad_y = to_acc(grad_y);
  grad_hy = to_acc(grad_hy);
  h_init_fwd = to_acc(h_init_fwd);
  h_init_rev = to_acc(h_init_rev);
  y_out_fwd = to_acc(y_out_fwd);
  y_out_rev = to_acc(y_out_rev);
  act_fwd_seq = to_acc(act_fwd_seq);
  act_rev_seq = to_acc(act_rev_seq);

  mlir::MlirOp one = MakeConstantLike(h_init_fwd, 1.0);

  // Slice grad_y into forward and reverse halves.
  mlir::MlirOp grad_y_fwd =
      batch_first ? mlir::stablehlo::Slice(grad_y, {0, 0, 0},
                                           {batch, seq_len, hidden}, {1, 1, 1})
                  : mlir::stablehlo::Slice(grad_y, {0, 0, 0},
                                           {seq_len, batch, hidden}, {1, 1, 1});
  mlir::MlirOp grad_y_rev_raw =
      batch_first
          ? mlir::stablehlo::Slice(grad_y, {0, 0, hidden},
                                   {batch, seq_len, 2 * hidden}, {1, 1, 1})
          : mlir::stablehlo::Slice(grad_y, {0, 0, hidden},
                                   {seq_len, batch, 2 * hidden}, {1, 1, 1});

  // Time-reverse reverse sequence tensors so forward and reverse chunk loops
  // run in lockstep.
  mlir::MlirOp grad_y_rev_time_reversed =
      mlir::stablehlo::Reverse(grad_y_rev_raw, {concat_dim});
  mlir::MlirOp act_rev_time_reversed =
      mlir::stablehlo::Reverse(act_rev_seq, {concat_dim});
  mlir::MlirOp y_out_rev_time_reversed =
      mlir::stablehlo::Reverse(y_out_rev, {concat_dim});
  mlir::MlirOp layer_in_time_reversed =
      mlir::stablehlo::Reverse(layer_in, {concat_dim});

  mlir::MlirOp h_fwd_init_3d =
      ExpandStep3D(h_init_fwd, batch, hidden, batch_first);
  mlir::MlirOp h_rev_init_3d =
      ExpandStep3D(h_init_rev, batch, hidden, batch_first);

  // Assemble full previous hidden state sequences [h_0, y_1, ..., y_{T-1}].
  mlir::MlirOp h_prev_fwd_seq;
  mlir::MlirOp h_prev_rev_seq;
  if (seq_len == 1) {
    h_prev_fwd_seq = h_fwd_init_3d;
    h_prev_rev_seq = h_rev_init_3d;
  } else {
    mlir::MlirOp y_fwd_slice =
        batch_first
            ? mlir::stablehlo::Slice(y_out_fwd, {0, 0, 0},
                                     {batch, seq_len - 1, hidden}, {1, 1, 1})
            : mlir::stablehlo::Slice(y_out_fwd, {0, 0, 0},
                                     {seq_len - 1, batch, hidden}, {1, 1, 1});
    h_prev_fwd_seq =
        ConcatDim(builder, {h_fwd_init_3d, y_fwd_slice}, concat_dim);

    mlir::MlirOp y_rev_slice =
        batch_first
            ? mlir::stablehlo::Slice(y_out_rev_time_reversed, {0, 0, 0},
                                     {batch, seq_len - 1, hidden}, {1, 1, 1})
            : mlir::stablehlo::Slice(y_out_rev_time_reversed, {0, 0, 0},
                                     {seq_len - 1, batch, hidden}, {1, 1, 1});
    h_prev_rev_seq =
        ConcatDim(builder, {h_rev_init_3d, y_rev_slice}, concat_dim);
  }

  mlir::MlirOp delta_h_start_fwd = ExtractLayer2D(grad_hy, 0, batch, hidden);
  mlir::MlirOp delta_h_start_rev = ExtractLayer2D(grad_hy, 1, batch, hidden);

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = grad_y.getValue().getLoc();
  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType i64_scalar_type =
      mlir::RankedTensorType::get({}, i64);
  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0f, acc_dtype);

  // Evaluate remainder steps before entering the chunked while loop.
  GruBidirBackwardRemainderInit rem = ComputeBidirBackwardRemainder(
      builder, grad_y_fwd, grad_y_rev_time_reversed, delta_h_start_fwd,
      delta_h_start_rev, h_prev_fwd_seq, h_prev_rev_seq, act_fwd_seq,
      act_rev_time_reversed, layer_in, layer_in_time_reversed, w_ih_fwd,
      w_hh_fwd, w_ih_rev, w_hh_rev, hh_bwd_dot_dims, x_bwd_dot_dims, w_dot_dims,
      has_biases, seq_len, chunked_steps, rem_steps, batch, hidden, in_dim,
      batch_first, concat_dim, acc_dtype, one, zero_scalar, zero_const,
      sum_reduce_builder, precision);

  const mlir::RankedTensorType grad_x_chunked_type =
      batch_first ? mlir::RankedTensorType::get({batch, chunked_steps, in_dim},
                                                acc_dtype)
                  : mlir::RankedTensorType::get({chunked_steps, batch, in_dim},
                                                acc_dtype);
  mlir::MlirOp grad_x_fwd_chunked_init =
      mlir::stablehlo::BroadcastInDim(grad_x_chunked_type, zero_scalar, {});
  mlir::MlirOp grad_x_rev_chunked_init =
      mlir::stablehlo::BroadcastInDim(grad_x_chunked_type, zero_scalar, {});

  mlir::MlirOp step_idx_init = MakeScalarConstant(builder, 0, i64);

  // Set up loop carry types and initial values for while loop.
  llvm::SmallVector<mlir::Type> loop_types = {i64_scalar_type,
                                              rem.delta_h_next_fwd.getType(),
                                              rem.delta_h_next_rev.getType(),
                                              grad_x_chunked_type,
                                              grad_x_chunked_type,
                                              rem.grad_w_ih_init_fwd.getType(),
                                              rem.grad_w_hh_init_fwd.getType(),
                                              rem.grad_w_ih_init_rev.getType(),
                                              rem.grad_w_hh_init_rev.getType()};
  llvm::SmallVector<mlir::Value> loop_inits = {
      step_idx_init.getValue(),           rem.delta_h_next_fwd.getValue(),
      rem.delta_h_next_rev.getValue(),    grad_x_fwd_chunked_init.getValue(),
      grad_x_rev_chunked_init.getValue(), rem.grad_w_ih_init_fwd.getValue(),
      rem.grad_w_hh_init_fwd.getValue(),  rem.grad_w_ih_init_rev.getValue(),
      rem.grad_w_hh_init_rev.getValue()};

  if (has_biases) {
    loop_types.push_back(rem.grad_b_ih_init_fwd.getType());
    loop_types.push_back(rem.grad_b_hh_init_fwd.getType());
    loop_types.push_back(rem.grad_b_ih_init_rev.getType());
    loop_types.push_back(rem.grad_b_hh_init_rev.getType());

    loop_inits.push_back(rem.grad_b_ih_init_fwd.getValue());
    loop_inits.push_back(rem.grad_b_hh_init_fwd.getValue());
    loop_inits.push_back(rem.grad_b_ih_init_rev.getValue());
    loop_inits.push_back(rem.grad_b_hh_init_rev.getValue());
  }

  auto while_op =
      mlir::stablehlo::WhileOp::create(op_builder, loc, loop_types, loop_inits);

  // While loop condition: step_idx < chunked_steps.
  mlir::Block* const cond_block = op_builder.createBlock(&while_op.getCond());
  cond_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(cond_block);

  mlir::MlirOp limit = MakeScalarConstant(builder, chunked_steps, i64);
  const mlir::Value cond =
      mlir::stablehlo::CompareOp::create(
          op_builder, loc, cond_block->getArgument(0), limit.getValue(),
          mlir::stablehlo::ComparisonDirection::LT)
          .getResult();
  mlir::stablehlo::ReturnOp::create(op_builder, loc, cond);

  // While loop body: processes one chunk of kGruUnrollFactor steps per
  // iteration.
  mlir::Block* const body_block = op_builder.createBlock(&while_op.getBody());
  body_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(body_block);

  BuildBidirLayerBackwardChunkedWhileBody(
      builder, body_block, loc, grad_y_fwd, grad_y_rev_time_reversed,
      act_fwd_seq, act_rev_time_reversed, h_prev_fwd_seq, h_prev_rev_seq,
      layer_in, layer_in_time_reversed, w_hh_fwd, w_ih_fwd, w_hh_rev, w_ih_rev,
      hh_bwd_dot_dims, x_bwd_dot_dims, w_dot_dims, chunked_steps, batch, hidden,
      in_dim, batch_first, concat_dim, acc_dtype, one, has_biases, zero_const,
      sum_reduce_builder, precision);

  // Unpack results after while loop.
  op_builder.setInsertionPointAfter(while_op);

  mlir::MlirOp final_delta_h_fwd(builder, while_op.getResult(1));
  mlir::MlirOp final_delta_h_rev(builder, while_op.getResult(2));
  mlir::MlirOp final_grad_x_fwd_chunked(builder, while_op.getResult(3));
  mlir::MlirOp final_grad_x_rev_chunked(builder, while_op.getResult(4));
  mlir::MlirOp final_grad_w_ih_fwd(builder, while_op.getResult(5));
  mlir::MlirOp final_grad_w_hh_fwd(builder, while_op.getResult(6));
  mlir::MlirOp final_grad_w_ih_rev(builder, while_op.getResult(7));
  mlir::MlirOp final_grad_w_hh_rev(builder, while_op.getResult(8));

  // Assemble full input gradient sequences by concatenating remainder steps.
  mlir::MlirOp grad_x_fwd_full;
  mlir::MlirOp grad_x_rev_time_reversed_full;
  if (rem.rem_grad_x_fwd.has_value()) {
    grad_x_fwd_full = ConcatDim(
        builder, {final_grad_x_fwd_chunked, *rem.rem_grad_x_fwd}, concat_dim);
    grad_x_rev_time_reversed_full = ConcatDim(
        builder, {final_grad_x_rev_chunked, *rem.rem_grad_x_rev}, concat_dim);
  } else {
    grad_x_fwd_full = final_grad_x_fwd_chunked;
    grad_x_rev_time_reversed_full = final_grad_x_rev_chunked;
  }

  // Reverse reverse input gradients back to chronological order
  mlir::MlirOp grad_x_rev_full =
      mlir::stablehlo::Reverse(grad_x_rev_time_reversed_full, {concat_dim});
  mlir::MlirOp grad_x_layer =
      mlir::stablehlo::Add(grad_x_fwd_full, grad_x_rev_full);

  mlir::MlirOp final_delta_h_fwd_3d =
      mlir::stablehlo::Reshape(to_out(final_delta_h_fwd), {1, batch, hidden});
  mlir::MlirOp final_delta_h_rev_3d =
      mlir::stablehlo::Reshape(to_out(final_delta_h_rev), {1, batch, hidden});
  mlir::MlirOp grad_h0_layer = mlir::stablehlo::Concatenate(
      builder, {final_delta_h_fwd_3d, final_delta_h_rev_3d}, /*dim=*/0);

  std::optional<mlir::MlirOp> grad_b_ih_fwd;
  std::optional<mlir::MlirOp> grad_b_hh_fwd;
  std::optional<mlir::MlirOp> grad_b_ih_rev;
  std::optional<mlir::MlirOp> grad_b_hh_rev;
  if (has_biases) {
    grad_b_ih_fwd = to_out(mlir::MlirOp(builder, while_op.getResult(9)));
    grad_b_hh_fwd = to_out(mlir::MlirOp(builder, while_op.getResult(10)));
    grad_b_ih_rev = to_out(mlir::MlirOp(builder, while_op.getResult(11)));
    grad_b_hh_rev = to_out(mlir::MlirOp(builder, while_op.getResult(12)));
  }

  return {grad_x_layer,
          grad_h0_layer,
          to_out(final_grad_w_ih_fwd),
          to_out(final_grad_w_hh_fwd),
          grad_b_ih_fwd,
          grad_b_hh_fwd,
          to_out(final_grad_w_ih_rev),
          to_out(final_grad_w_hh_rev),
          grad_b_ih_rev,
          grad_b_hh_rev};
}

// Dispatches backward pass for a bidirectional GRU layer to static unrolling
// (seq_len < kGruUnrollFactor) or chunked recurrence (seq_len >=
// kGruUnrollFactor).
//
// Arguments:
//   builder: MLIR StableHLO graph builder.
//   grad_y: Incoming output sequence gradients [T, B, 2*H].
//   grad_hy: Incoming final hidden state gradients [2, B, H].
//   layer_in: Input sequence tensor [T, B, in_dim].
//   h_init_fwd, h_init_rev: Initial forward and reverse hidden states [B, H].
//   y_out_fwd, y_out_rev: Forward and reverse output sequences [T, B, H].
//   act_fwd_seq, act_rev_seq: Forward and reverse pre-activation caches.
//   w_ih_fwd, w_hh_fwd, w_ih_rev, w_hh_rev: Layer weight matrices.
//   has_biases: Whether bias parameters are enabled.
//   seq_len, batch, hidden, in_dim: Layer dimensions.
//   batch_first: Layout configuration.
//   to_out: Datatype conversion functor.
//   zero_const: Scalar zero constant for bias reduction.
//   sum_reduce_builder: Reduction region builder.
//   precision: StableHLO dot precision mode.
//
// Returns:
//   GruBidirLayerBackwardOutputs containing all computed gradients.
template <typename ToOutFn, typename ReduceBuilderFn>
GruBidirLayerBackwardOutputs ComputeBidirLayerBackward(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_y, mlir::MlirOp grad_hy,
    mlir::MlirOp layer_in, mlir::MlirOp h_init_fwd, mlir::MlirOp h_init_rev,
    mlir::MlirOp y_out_fwd, mlir::MlirOp y_out_rev, mlir::MlirOp act_fwd_seq,
    mlir::MlirOp act_rev_seq, mlir::MlirOp w_ih_fwd, mlir::MlirOp w_hh_fwd,
    mlir::MlirOp w_ih_rev, mlir::MlirOp w_hh_rev, bool has_biases,
    int64_t seq_len, int64_t batch, int64_t hidden, int64_t in_dim,
    bool batch_first, ToOutFn to_out, mlir::MlirOp zero_const,
    ReduceBuilderFn sum_reduce_builder, mlir::stablehlo::Precision precision) {
  if (seq_len < kGruUnrollFactor) {
    return ComputeBidirLayerBackwardStatic(
        builder, grad_y, grad_hy, layer_in, h_init_fwd, h_init_rev, y_out_fwd,
        y_out_rev, act_fwd_seq, act_rev_seq, w_ih_fwd, w_hh_fwd, w_ih_rev,
        w_hh_rev, has_biases, seq_len, batch, hidden, in_dim, batch_first,
        to_out, zero_const, sum_reduce_builder, precision);
  }
  return ComputeBidirLayerBackwardChunked(
      builder, grad_y, grad_hy, layer_in, h_init_fwd, h_init_rev, y_out_fwd,
      y_out_rev, act_fwd_seq, act_rev_seq, w_ih_fwd, w_hh_fwd, w_ih_rev,
      w_hh_rev, has_biases, seq_len, batch, hidden, in_dim, batch_first, to_out,
      zero_const, sum_reduce_builder, precision);
}

// Container holding all gradient outputs for a multi-layer GRU backward pass.
struct GruMultiLayerBackwardOutputs {
  mlir::MlirOp grad_x;  // Input gradient tensor [T, B, input_size].
  mlir::MlirOp
      grad_h0;  // Initial hidden states gradient [num_layers, B, hidden].
  std::vector<mlir::MlirOp> all_grad_params;  // Parameter gradients [W_ih,
                                              // W_hh, (b_ih, b_hh)] per layer.
};

// Container holding the initialized accumulators and carry states computed
// during the tail remainder backward pass of a multi-layer wavefront GRU.
struct WavefrontBackwardRemainderResult {
  std::vector<mlir::MlirOp> rem_grad_w_ih_init;
  std::vector<mlir::MlirOp> rem_grad_w_hh_init;
  std::vector<std::optional<mlir::MlirOp>> rem_grad_b_ih_init;
  std::vector<std::optional<mlir::MlirOp>> rem_grad_b_hh_init;
  std::vector<mlir::MlirOp> rem_delta_h;
  std::optional<mlir::MlirOp> rem_grad_x_0;
};

// Computes remainder backward pass across layers when rem_steps > 0.
//
// For multi-layer wavefront GRUs where seq_len is not a multiple of
// kGruUnrollFactor, this function computes the backward pass for the tail
// remainder steps [chunked_steps, seq_len) across all layers:
// 1. Iterates layers backwards from l = num_layers - 1 down to 0.
// 2. Propagates sequence gradients through inter-layer dropout backward.
// 3. Unrolls BPTT steps for each layer on the remainder window, computing
//    layer input gradients (curr_rem_grad_y) and hidden state deltas
//    (rem_delta_h).
// 4. Computes parameter gradient contributions for the remainder steps to seed
//    the subsequent chunked while loop accumulators.
//
// Arguments:
//   builder: MLIR StableHLO graph builder.
//   layer_inputs: Forward layer input sequence tensors [num_layers].
//   fwd_cached_acts: Forward gate pre-activation caches [num_layers].
//   h_prev_seq: Forward previous hidden state sequences [num_layers].
//   incoming_seq_grad: Incoming output sequence gradient [T, B, hidden].
//   grad_hy_list: Incoming final hidden state gradients [num_layers].
//   w_ih_list, w_hh_list: Weight matrices for all layers.
//   has_biases: Whether bias parameters are enabled.
//   zero_const, sum_reduce_builder: Constants and builder for bias reduction.
//   hh_bwd_dot_dims, x_bwd_dot_dims, w_dot_dims: Dot dimension configurations.
//   seq_len, chunked_steps, rem_steps, batch, hidden, input_size, num_layers:
//   Dimensions. to_acc: Functor converting tensors to accumulator datatype.
//   has_dropout, dropout, dropout_masks: Inter-layer dropout settings.
//   batch_first, concat_dim: Layout flags.
//   acc_elem_type, zero_scalar: Precision and constant helpers.
//   precision: StableHLO dot precision mode.
//
// Returns:
//   WavefrontBackwardRemainderResult containing initialized parameter
//   accumulators and hidden state deltas for each layer.
// Evaluates backward pass for a single layer l on the remainder window in
// multi-layer wavefront GRU.
//
// What it computes:
// - Slices incoming remainder output gradient and applies dropout backward if
// enabled.
// - Unrolls remainder timesteps k = rem_steps - 1 down to 0 for layer l via
// ComputeGruBackwardStep.
// - Accumulates parameter gradient contributions into rem_grad_w_ih_init,
// rem_grad_w_hh_init, and biases.
// - Returns the updated hidden state delta and layer input gradient.
// Result of evaluating wavefront backward remainder layer.
struct WavefrontBackwardRemainderLayerResult {
  mlir::MlirOp cur_delta_h;
  mlir::MlirOp next_curr_rem_grad_y;
};

template <typename ToAccFn, typename ReduceBuilderFn>
WavefrontBackwardRemainderLayerResult EvaluateWavefrontBackwardRemainderLayer(
    mlir::MlirBuilder& builder, int64_t l, int64_t num_layers,
    mlir::MlirOp curr_rem_grad_y, mlir::MlirOp incoming_seq_grad,
    absl::Span<const mlir::MlirOp> dropout_masks, bool has_dropout,
    double dropout, absl::Span<const mlir::MlirOp> fwd_cached_acts,
    absl::Span<const mlir::MlirOp> h_prev_seq,
    absl::Span<const mlir::MlirOp> layer_inputs,
    absl::Span<const mlir::MlirOp> grad_hy_list,
    absl::Span<const mlir::MlirOp> w_ih_list,
    absl::Span<const mlir::MlirOp> w_hh_list,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr x_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr w_dot_dims, int64_t seq_len,
    int64_t chunked_steps, int64_t rem_steps, int64_t batch, int64_t hidden,
    int64_t input_size, bool batch_first, int64_t concat_dim,
    mlir::Type acc_elem_type, ToAccFn to_acc, bool has_biases,
    mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    mlir::stablehlo::Precision precision,
    std::vector<mlir::MlirOp>& rem_grad_w_ih_init,
    std::vector<mlir::MlirOp>& rem_grad_w_hh_init,
    std::vector<std::optional<mlir::MlirOp>>& rem_grad_b_ih_init,
    std::vector<std::optional<mlir::MlirOp>>& rem_grad_b_hh_init,
    std::vector<mlir::MlirOp>& rem_delta_h) {
  const int64_t in_dim_l = (l == 0) ? input_size : hidden;
  if (l == num_layers - 1) {
    curr_rem_grad_y =
        batch_first
            ? mlir::stablehlo::Slice(incoming_seq_grad, {0, chunked_steps, 0},
                                     {batch, seq_len, hidden}, {1, 1, 1})
            : mlir::stablehlo::Slice(incoming_seq_grad, {chunked_steps, 0, 0},
                                     {seq_len, batch, hidden}, {1, 1, 1});
  } else if (has_dropout) {
    mlir::MlirOp mask_l = dropout_masks[l];
    mlir::MlirOp rem_mask =
        batch_first
            ? mlir::stablehlo::Slice(mask_l, {0, chunked_steps, 0},
                                     {batch, seq_len, hidden}, {1, 1, 1})
            : mlir::stablehlo::Slice(mask_l, {chunked_steps, 0, 0},
                                     {seq_len, batch, hidden}, {1, 1, 1});
    curr_rem_grad_y = ApplyDropoutBackward(curr_rem_grad_y, rem_mask, dropout);
  }

  mlir::MlirOp act_l = to_acc(fwd_cached_acts[l]);
  mlir::MlirOp rem_act =
      batch_first
          ? mlir::stablehlo::Slice(act_l, {0, chunked_steps, 0},
                                   {batch, seq_len, 4 * hidden}, {1, 1, 1})
          : mlir::stablehlo::Slice(act_l, {chunked_steps, 0, 0},
                                   {seq_len, batch, 4 * hidden}, {1, 1, 1});
  mlir::MlirOp h_prev_l = h_prev_seq[l];
  mlir::MlirOp rem_h_prev =
      batch_first ? mlir::stablehlo::Slice(h_prev_l, {0, chunked_steps, 0},
                                           {batch, seq_len, hidden}, {1, 1, 1})
                  : mlir::stablehlo::Slice(h_prev_l, {chunked_steps, 0, 0},
                                           {seq_len, batch, hidden}, {1, 1, 1});
  mlir::MlirOp in_l = layer_inputs[l];
  mlir::MlirOp rem_layer_in =
      batch_first
          ? mlir::stablehlo::Slice(in_l, {0, chunked_steps, 0},
                                   {batch, seq_len, in_dim_l}, {1, 1, 1})
          : mlir::stablehlo::Slice(in_l, {chunked_steps, 0, 0},
                                   {seq_len, batch, in_dim_l}, {1, 1, 1});

  mlir::MlirOp cur_delta_h = grad_hy_list[l];
  mlir::MlirOp one = MakeConstantLike(cur_delta_h, 1.0);

  std::vector<mlir::MlirOp> delta_pre_ig_rem(rem_steps);
  std::vector<mlir::MlirOp> delta_pre_hg_rem(rem_steps);
  std::vector<mlir::MlirOp> grad_x_rem(rem_steps);

  for (int64_t k = rem_steps - 1; k >= 0; --k) {
    mlir::MlirOp gy_k =
        SliceStep2D(curr_rem_grad_y, k, batch, hidden, batch_first);
    mlir::MlirOp dh = mlir::stablehlo::Add(gy_k, cur_delta_h);
    mlir::MlirOp hp_k = SliceStep2D(rem_h_prev, k, batch, hidden, batch_first);
    mlir::MlirOp act_k =
        SliceStep2D(rem_act, k, batch, 4 * hidden, batch_first);

    mlir::MlirOp w_hh_l = w_hh_list[l];
    const GruBackwardStepResult step =
        ComputeGruBackwardStep(builder, dh, hp_k, act_k, w_hh_l,
                               hh_bwd_dot_dims, batch, hidden, one, precision);
    cur_delta_h = step.delta_h_prev;
    delta_pre_ig_rem[k] =
        ExpandStep3D(step.delta_pre_ig, batch, 3 * hidden, batch_first);
    delta_pre_hg_rem[k] =
        ExpandStep3D(step.delta_pre_hg, batch, 3 * hidden, batch_first);
    mlir::MlirOp w_ih_l = w_ih_list[l];
    mlir::MlirOp dx_k = MixedPrecisionDotGeneral(
        step.delta_pre_ig, w_ih_l, x_bwd_dot_dims, acc_elem_type, precision);
    grad_x_rem[k] = ExpandStep3D(dx_k, batch, in_dim_l, batch_first);
  }

  rem_delta_h[l] = cur_delta_h;
  mlir::MlirOp next_curr_rem_grad_y =
      ConcatDim(builder, grad_x_rem, concat_dim);

  mlir::MlirOp rem_delta_pre_ig_2d =
      mlir::stablehlo::Reshape(ConcatDim(builder, delta_pre_ig_rem, concat_dim),
                               {rem_steps * batch, 3 * hidden});
  mlir::MlirOp rem_delta_pre_hg_2d =
      mlir::stablehlo::Reshape(ConcatDim(builder, delta_pre_hg_rem, concat_dim),
                               {rem_steps * batch, 3 * hidden});
  mlir::MlirOp rem_layer_in_2d =
      mlir::stablehlo::Reshape(rem_layer_in, {rem_steps * batch, in_dim_l});
  mlir::MlirOp rem_h_prev_2d =
      mlir::stablehlo::Reshape(rem_h_prev, {rem_steps * batch, hidden});

  rem_grad_w_ih_init[l] =
      MixedPrecisionDotGeneral(rem_delta_pre_ig_2d, rem_layer_in_2d, w_dot_dims,
                               acc_elem_type, precision);
  rem_grad_w_hh_init[l] = MixedPrecisionDotGeneral(
      rem_delta_pre_hg_2d, rem_h_prev_2d, w_dot_dims, acc_elem_type, precision);

  if (has_biases) {
    rem_grad_b_ih_init[l] = mlir::stablehlo::Reduce(
        builder, rem_delta_pre_ig_2d, zero_const, sum_reduce_builder, {0})[0];
    rem_grad_b_hh_init[l] = mlir::stablehlo::Reduce(
        builder, rem_delta_pre_hg_2d, zero_const, sum_reduce_builder, {0})[0];
  }
  return {cur_delta_h, next_curr_rem_grad_y};
}

// Initializes zero parameter gradient accumulators when rem_steps == 0.
inline WavefrontBackwardRemainderResult InitWavefrontBackwardRemainderZero(
    absl::Span<const mlir::MlirOp> grad_hy_list, int64_t num_layers,
    int64_t input_size, int64_t hidden, bool has_biases,
    mlir::Type acc_elem_type, mlir::MlirOp zero_scalar) {
  std::vector<mlir::MlirOp> rem_grad_w_ih_init(num_layers);
  std::vector<mlir::MlirOp> rem_grad_w_hh_init(num_layers);
  std::vector<std::optional<mlir::MlirOp>> rem_grad_b_ih_init(num_layers);
  std::vector<std::optional<mlir::MlirOp>> rem_grad_b_hh_init(num_layers);
  std::vector<mlir::MlirOp> rem_delta_h(num_layers);

  for (int64_t l = 0; l < num_layers; ++l) {
    const int64_t in_dim_l = (l == 0) ? input_size : hidden;
    rem_delta_h[l] = grad_hy_list[l];
    rem_grad_w_ih_init[l] = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({3 * hidden, in_dim_l}, acc_elem_type),
        zero_scalar, {});
    rem_grad_w_hh_init[l] = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({3 * hidden, hidden}, acc_elem_type),
        zero_scalar, {});
    if (has_biases) {
      rem_grad_b_ih_init[l] = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({3 * hidden}, acc_elem_type), zero_scalar,
          {});
      rem_grad_b_hh_init[l] = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({3 * hidden}, acc_elem_type), zero_scalar,
          {});
    }
  }
  return {std::move(rem_grad_w_ih_init), std::move(rem_grad_w_hh_init),
          std::move(rem_grad_b_ih_init), std::move(rem_grad_b_hh_init),
          std::move(rem_delta_h),        std::nullopt};
}

// Computes remainder backward pass across layers when rem_steps > 0.
template <typename ToAccFn, typename ReduceBuilderFn>
WavefrontBackwardRemainderResult BuildWavefrontBackwardRemainder(
    mlir::MlirBuilder& builder, absl::Span<const mlir::MlirOp> layer_inputs,
    absl::Span<const mlir::MlirOp> fwd_cached_acts,
    absl::Span<const mlir::MlirOp> h_prev_seq, mlir::MlirOp incoming_seq_grad,
    absl::Span<const mlir::MlirOp> grad_hy_list,
    absl::Span<const mlir::MlirOp> w_ih_list,
    absl::Span<const mlir::MlirOp> w_hh_list, const bool has_biases,
    mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr x_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr w_dot_dims,
    const int64_t seq_len, const int64_t chunked_steps, const int64_t rem_steps,
    const int64_t batch, const int64_t hidden, const int64_t input_size,
    const int64_t num_layers, ToAccFn to_acc, const bool has_dropout,
    const double dropout, absl::Span<const mlir::MlirOp> dropout_masks,
    const bool batch_first, const int64_t concat_dim,
    const mlir::Type acc_elem_type, mlir::MlirOp zero_scalar,
    const mlir::stablehlo::Precision precision) {
  if (rem_steps == 0) {
    return InitWavefrontBackwardRemainderZero(grad_hy_list, num_layers,
                                              input_size, hidden, has_biases,
                                              acc_elem_type, zero_scalar);
  }

  std::vector<mlir::MlirOp> rem_grad_w_ih_init(num_layers);
  std::vector<mlir::MlirOp> rem_grad_w_hh_init(num_layers);
  std::vector<std::optional<mlir::MlirOp>> rem_grad_b_ih_init(num_layers);
  std::vector<std::optional<mlir::MlirOp>> rem_grad_b_hh_init(num_layers);
  std::vector<mlir::MlirOp> rem_delta_h(num_layers);
  std::optional<mlir::MlirOp> rem_grad_x_0;

  mlir::MlirOp curr_rem_grad_y;
  for (int64_t l = num_layers - 1; l >= 0; --l) {
    auto [delta_h_l, next_rem_grad_y] = EvaluateWavefrontBackwardRemainderLayer(
        builder, l, num_layers, curr_rem_grad_y, incoming_seq_grad,
        dropout_masks, has_dropout, dropout, fwd_cached_acts, h_prev_seq,
        layer_inputs, grad_hy_list, w_ih_list, w_hh_list, hh_bwd_dot_dims,
        x_bwd_dot_dims, w_dot_dims, seq_len, chunked_steps, rem_steps, batch,
        hidden, input_size, batch_first, concat_dim, acc_elem_type, to_acc,
        has_biases, zero_const, sum_reduce_builder, precision,
        rem_grad_w_ih_init, rem_grad_w_hh_init, rem_grad_b_ih_init,
        rem_grad_b_hh_init, rem_delta_h);
    curr_rem_grad_y = next_rem_grad_y;
    if (l == 0) {
      rem_grad_x_0 = curr_rem_grad_y;
    }
  }

  return {std::move(rem_grad_w_ih_init), std::move(rem_grad_w_hh_init),
          std::move(rem_grad_b_ih_init), std::move(rem_grad_b_hh_init),
          std::move(rem_delta_h),        rem_grad_x_0};
}

// Result of evaluating a single layer's chunk backward execution in the
// wavefront.
struct WavefrontLayerChunkBackwardResult {
  mlir::MlirOp next_delta_h_l;
  mlir::MlirOp layer_grad_x;
  mlir::MlirOp chunk_gw_ih;
  mlir::MlirOp chunk_gw_hh;
  std::optional<mlir::MlirOp> chunk_gb_ih;
  std::optional<mlir::MlirOp> chunk_gb_hh;
};

// Computes backward chunk for a single layer in the wavefront backward loop.
template <typename ReduceBuilderFn>
inline WavefrontLayerChunkBackwardResult ProcessWavefrontBackwardLayerChunk(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_y_chunk, mlir::MlirOp act_l,
    mlir::MlirOp h_prev_l, mlir::MlirOp in_l, mlir::MlirOp cur_delta_h_in,
    mlir::MlirOp w_hh_l, mlir::MlirOp w_ih_l,
    const llvm::SmallVector<mlir::MlirOp, 3>& chunk_start_indices,
    const llvm::SmallVector<int64_t, 3>& hidden_slice_sizes,
    const llvm::SmallVector<int64_t, 3>& act_slice_sizes,
    const llvm::SmallVector<int64_t, 3>& in_slice_sizes,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr x_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr w_dot_dims,
    const bool has_biases, mlir::MlirOp zero_const,
    ReduceBuilderFn sum_reduce_builder, const int64_t batch,
    const int64_t hidden, const int64_t in_dim_l, const bool batch_first,
    const int64_t concat_dim, const mlir::Type acc_elem_type,
    const mlir::stablehlo::Precision precision) {
  mlir::MlirOp act_chunk = mlir::stablehlo::DynamicSlice(
      act_l, chunk_start_indices, act_slice_sizes);
  mlir::MlirOp h_prev_chunk = mlir::stablehlo::DynamicSlice(
      h_prev_l, chunk_start_indices, hidden_slice_sizes);
  mlir::MlirOp layer_in_chunk =
      mlir::stablehlo::DynamicSlice(in_l, chunk_start_indices, in_slice_sizes);

  mlir::MlirOp cur_delta_h = cur_delta_h_in;
  mlir::MlirOp one = MakeConstantLike(cur_delta_h, 1.0);

  std::vector<mlir::MlirOp> delta_pre_ig_k(kGruUnrollFactor);
  std::vector<mlir::MlirOp> delta_pre_hg_k(kGruUnrollFactor);
  std::vector<mlir::MlirOp> grad_x_k(kGruUnrollFactor);

  for (int64_t k = kGruUnrollFactor - 1; k >= 0; --k) {
    mlir::MlirOp gy_k =
        SliceStep2D(grad_y_chunk, k, batch, hidden, batch_first);
    mlir::MlirOp dh = mlir::stablehlo::Add(gy_k, cur_delta_h);
    mlir::MlirOp hp_k =
        SliceStep2D(h_prev_chunk, k, batch, hidden, batch_first);
    mlir::MlirOp act_k =
        SliceStep2D(act_chunk, k, batch, 4 * hidden, batch_first);

    const GruBackwardStepResult step =
        ComputeGruBackwardStep(builder, dh, hp_k, act_k, w_hh_l,
                               hh_bwd_dot_dims, batch, hidden, one, precision);
    cur_delta_h = step.delta_h_prev;
    delta_pre_ig_k[k] =
        ExpandStep3D(step.delta_pre_ig, batch, 3 * hidden, batch_first);
    delta_pre_hg_k[k] =
        ExpandStep3D(step.delta_pre_hg, batch, 3 * hidden, batch_first);

    mlir::MlirOp dx_k = MixedPrecisionDotGeneral(
        step.delta_pre_ig, w_ih_l, x_bwd_dot_dims, acc_elem_type, precision);
    grad_x_k[k] = ExpandStep3D(dx_k, batch, in_dim_l, batch_first);
  }

  mlir::MlirOp delta_pre_ig_chunk_2d =
      mlir::stablehlo::Reshape(ConcatDim(builder, delta_pre_ig_k, concat_dim),
                               {kGruUnrollFactor * batch, 3 * hidden});
  mlir::MlirOp delta_pre_hg_chunk_2d =
      mlir::stablehlo::Reshape(ConcatDim(builder, delta_pre_hg_k, concat_dim),
                               {kGruUnrollFactor * batch, 3 * hidden});
  mlir::MlirOp layer_in_chunk_2d = mlir::stablehlo::Reshape(
      layer_in_chunk, {kGruUnrollFactor * batch, in_dim_l});
  mlir::MlirOp h_prev_chunk_2d = mlir::stablehlo::Reshape(
      h_prev_chunk, {kGruUnrollFactor * batch, hidden});

  mlir::MlirOp chunk_gw_ih =
      MixedPrecisionDotGeneral(delta_pre_ig_chunk_2d, layer_in_chunk_2d,
                               w_dot_dims, acc_elem_type, precision);
  mlir::MlirOp chunk_gw_hh =
      MixedPrecisionDotGeneral(delta_pre_hg_chunk_2d, h_prev_chunk_2d,
                               w_dot_dims, acc_elem_type, precision);

  std::optional<mlir::MlirOp> chunk_gb_ih = std::nullopt;
  std::optional<mlir::MlirOp> chunk_gb_hh = std::nullopt;
  if (has_biases) {
    chunk_gb_ih = mlir::stablehlo::Reduce(
        builder, delta_pre_ig_chunk_2d, zero_const, sum_reduce_builder, {0})[0];
    chunk_gb_hh = mlir::stablehlo::Reduce(
        builder, delta_pre_hg_chunk_2d, zero_const, sum_reduce_builder, {0})[0];
  }

  mlir::MlirOp layer_grad_x = ConcatDim(builder, grad_x_k, concat_dim);

  return {cur_delta_h, layer_grad_x, chunk_gw_ih,
          chunk_gw_hh, chunk_gb_ih,  chunk_gb_hh};
}

// Reconstructs previous hidden state sequences [h_0, y_1, ..., y_{T-1}] for
// each layer.
template <typename ToAccFn>
std::vector<mlir::MlirOp> BuildWavefrontHPrevSeq(
    mlir::MlirBuilder& builder, absl::Span<const mlir::MlirOp> h_inits,
    absl::Span<const mlir::MlirOp> layer_outputs, int64_t num_layers,
    int64_t seq_len, int64_t batch, int64_t hidden, bool batch_first,
    int64_t concat_dim, ToAccFn to_acc) {
  std::vector<mlir::MlirOp> h_prev_seq(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    mlir::MlirOp h_init_l = to_acc(h_inits[l]);
    mlir::MlirOp h_init_3d = ExpandStep3D(h_init_l, batch, hidden, batch_first);
    if (seq_len == 1) {
      h_prev_seq[l] = h_init_3d;
    } else {
      mlir::MlirOp layer_out_l = to_acc(layer_outputs[l]);
      mlir::MlirOp y_prev_slice =
          batch_first
              ? mlir::stablehlo::Slice(layer_out_l, {0, 0, 0},
                                       {batch, seq_len - 1, hidden}, {1, 1, 1})
              : mlir::stablehlo::Slice(layer_out_l, {0, 0, 0},
                                       {seq_len - 1, batch, hidden}, {1, 1, 1});
      h_prev_seq[l] = ConcatDim(builder, {h_init_3d, y_prev_slice}, concat_dim);
    }
  }
  return h_prev_seq;
}

// Builds while loop body for pipelined wavefront backward pass.
template <typename ToAccFn, typename ReduceBuilderFn>
void BuildWavefrontBackwardWhileBody(
    mlir::MlirBuilder& builder, mlir::Block* body_block, mlir::Location loc,
    mlir::MlirOp incoming_seq_grad,
    absl::Span<const mlir::MlirOp> fwd_cached_acts,
    absl::Span<const mlir::MlirOp> h_prev_seq,
    absl::Span<const mlir::MlirOp> layer_inputs,
    absl::Span<const mlir::MlirOp> w_hh_list,
    absl::Span<const mlir::MlirOp> w_ih_list,
    absl::Span<const mlir::MlirOp> dropout_masks,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr x_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr w_dot_dims,
    int64_t chunked_steps, int64_t batch, int64_t hidden, int64_t input_size,
    int64_t num_layers, bool batch_first, int64_t concat_dim,
    mlir::Type acc_elem_type, ToAccFn to_acc, bool has_biases, bool has_dropout,
    double dropout, mlir::MlirOp zero_const, mlir::MlirOp zero_i64,
    mlir::IntegerType i64, ReduceBuilderFn sum_reduce_builder,
    mlir::stablehlo::Precision precision) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  mlir::MlirOp body_step_idx(builder, body_block->getArgument(0));
  int arg_idx = 1;
  std::vector<mlir::MlirOp> body_delta_h(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    body_delta_h[l] = mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
  }
  mlir::MlirOp body_grad_x_chunked(builder, body_block->getArgument(arg_idx++));
  std::vector<mlir::MlirOp> body_grad_w_ih(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    body_grad_w_ih[l] =
        mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
  }
  std::vector<mlir::MlirOp> body_grad_w_hh(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    body_grad_w_hh[l] =
        mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
  }
  std::vector<mlir::MlirOp> body_grad_b_ih(has_biases ? num_layers : 0);
  std::vector<mlir::MlirOp> body_grad_b_hh(has_biases ? num_layers : 0);
  if (has_biases) {
    for (int64_t l = 0; l < num_layers; ++l) {
      body_grad_b_ih[l] =
          mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
    }
    for (int64_t l = 0; l < num_layers; ++l) {
      body_grad_b_hh[l] =
          mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
    }
  }

  // Calculate reverse chronological chunk index: T - 8, T - 16, etc.
  mlir::MlirOp const_chunk_limit =
      MakeScalarConstant(builder, chunked_steps - kGruUnrollFactor, i64);
  mlir::MlirOp chunk_start_t =
      mlir::stablehlo::Subtract(const_chunk_limit, body_step_idx);

  const llvm::SmallVector<mlir::MlirOp, 3> chunk_start_indices =
      batch_first ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, chunk_start_t,
                                                       zero_i64}
                  : llvm::SmallVector<mlir::MlirOp, 3>{chunk_start_t, zero_i64,
                                                       zero_i64};

  const llvm::SmallVector<int64_t, 3> hidden_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kGruUnrollFactor, hidden}
          : llvm::SmallVector<int64_t, 3>{kGruUnrollFactor, batch, hidden};
  const llvm::SmallVector<int64_t, 3> act_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kGruUnrollFactor, 4 * hidden}
          : llvm::SmallVector<int64_t, 3>{kGruUnrollFactor, batch, 4 * hidden};

  std::vector<mlir::MlirOp> next_delta_h = body_delta_h;
  std::vector<mlir::MlirOp> next_grad_w_ih = body_grad_w_ih;
  std::vector<mlir::MlirOp> next_grad_w_hh = body_grad_w_hh;
  std::vector<mlir::MlirOp> next_grad_b_ih = body_grad_b_ih;
  std::vector<mlir::MlirOp> next_grad_b_hh = body_grad_b_hh;

  mlir::MlirOp curr_layer_grad_x;

  for (int64_t l = num_layers - 1; l >= 0; --l) {
    const int64_t in_dim_l = (l == 0) ? input_size : hidden;
    mlir::MlirOp grad_y_chunk;
    if (l == num_layers - 1) {
      grad_y_chunk = mlir::stablehlo::DynamicSlice(
          incoming_seq_grad, chunk_start_indices, hidden_slice_sizes);
    } else {
      grad_y_chunk = curr_layer_grad_x;
      if (has_dropout) {
        mlir::MlirOp mask_l = dropout_masks[l];
        mlir::MlirOp mask_chunk = mlir::stablehlo::DynamicSlice(
            mask_l, chunk_start_indices, hidden_slice_sizes);
        grad_y_chunk = ApplyDropoutBackward(grad_y_chunk, mask_chunk, dropout);
      }
    }

    mlir::MlirOp act_l = to_acc(fwd_cached_acts[l]);
    const llvm::SmallVector<int64_t, 3> in_slice_sizes =
        batch_first
            ? llvm::SmallVector<int64_t, 3>{batch, kGruUnrollFactor, in_dim_l}
            : llvm::SmallVector<int64_t, 3>{kGruUnrollFactor, batch, in_dim_l};

    WavefrontLayerChunkBackwardResult layer_res =
        ProcessWavefrontBackwardLayerChunk(
            builder, grad_y_chunk, act_l, h_prev_seq[l], layer_inputs[l],
            body_delta_h[l], w_hh_list[l], w_ih_list[l], chunk_start_indices,
            hidden_slice_sizes, act_slice_sizes, in_slice_sizes,
            hh_bwd_dot_dims, x_bwd_dot_dims, w_dot_dims, has_biases, zero_const,
            sum_reduce_builder, batch, hidden, in_dim_l, batch_first,
            concat_dim, acc_elem_type, precision);

    next_delta_h[l] = layer_res.next_delta_h_l;
    curr_layer_grad_x = layer_res.layer_grad_x;

    next_grad_w_ih[l] =
        mlir::stablehlo::Add(body_grad_w_ih[l], layer_res.chunk_gw_ih);
    next_grad_w_hh[l] =
        mlir::stablehlo::Add(body_grad_w_hh[l], layer_res.chunk_gw_hh);

    if (has_biases) {
      next_grad_b_ih[l] =
          mlir::stablehlo::Add(body_grad_b_ih[l], *layer_res.chunk_gb_ih);
      next_grad_b_hh[l] =
          mlir::stablehlo::Add(body_grad_b_hh[l], *layer_res.chunk_gb_hh);
    }
  }

  mlir::MlirOp next_grad_x_chunked = mlir::stablehlo::DynamicUpdateSlice(
      body_grad_x_chunked, curr_layer_grad_x, chunk_start_indices);

  mlir::MlirOp step_k = MakeScalarConstant(builder, kGruUnrollFactor, i64);
  mlir::MlirOp next_step_idx = mlir::stablehlo::Add(body_step_idx, step_k);

  llvm::SmallVector<mlir::Value> next_loop_values;
  next_loop_values.push_back(next_step_idx.getValue());
  for (int64_t l = 0; l < num_layers; ++l) {
    next_loop_values.push_back(next_delta_h[l].getValue());
  }
  next_loop_values.push_back(next_grad_x_chunked.getValue());
  for (int64_t l = 0; l < num_layers; ++l) {
    next_loop_values.push_back(next_grad_w_ih[l].getValue());
  }
  for (int64_t l = 0; l < num_layers; ++l) {
    next_loop_values.push_back(next_grad_w_hh[l].getValue());
  }
  if (has_biases) {
    for (int64_t l = 0; l < num_layers; ++l) {
      next_loop_values.push_back(next_grad_b_ih[l].getValue());
    }
    for (int64_t l = 0; l < num_layers; ++l) {
      next_loop_values.push_back(next_grad_b_hh[l].getValue());
    }
  }
  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_loop_values);
}

// Finalizes output gradients for pipelined wavefront backward pass.
template <typename ToOutFn>
GruMultiLayerBackwardOutputs FinalizeWavefrontBackwardOutputs(
    mlir::MlirBuilder& builder, mlir::stablehlo::WhileOp while_op,
    const std::optional<mlir::MlirOp>& rem_grad_x_0, int64_t num_layers,
    int64_t batch, int64_t hidden, bool has_biases, int64_t concat_dim,
    ToOutFn to_out) {
  int res_idx = 1;
  std::vector<mlir::MlirOp> final_delta_h(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    final_delta_h[l] = mlir::MlirOp(builder, while_op.getResult(res_idx++));
  }
  mlir::MlirOp final_grad_x_chunked(builder, while_op.getResult(res_idx++));

  std::vector<mlir::MlirOp> final_grad_w_ih(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    final_grad_w_ih[l] = mlir::MlirOp(builder, while_op.getResult(res_idx++));
  }
  std::vector<mlir::MlirOp> final_grad_w_hh(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    final_grad_w_hh[l] = mlir::MlirOp(builder, while_op.getResult(res_idx++));
  }
  std::vector<std::optional<mlir::MlirOp>> final_grad_b_ih(
      has_biases ? num_layers : 0);
  std::vector<std::optional<mlir::MlirOp>> final_grad_b_hh(
      has_biases ? num_layers : 0);
  if (has_biases) {
    for (int64_t l = 0; l < num_layers; ++l) {
      final_grad_b_ih[l] =
          to_out(mlir::MlirOp(builder, while_op.getResult(res_idx++)));
    }
    for (int64_t l = 0; l < num_layers; ++l) {
      final_grad_b_hh[l] =
          to_out(mlir::MlirOp(builder, while_op.getResult(res_idx++)));
    }
  }

  // Combine remainder input gradients with chunked input gradients.
  mlir::MlirOp final_grad_x;
  if (rem_grad_x_0.has_value()) {
    final_grad_x =
        ConcatDim(builder, {final_grad_x_chunked, *rem_grad_x_0}, concat_dim);
  } else {
    final_grad_x = final_grad_x_chunked;
  }

  // Stack initial hidden state gradients [num_layers, batch, hidden].
  std::vector<mlir::MlirOp> all_h0_3d;
  all_h0_3d.reserve(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    all_h0_3d.push_back(
        mlir::stablehlo::Reshape(to_out(final_delta_h[l]), {1, batch, hidden}));
  }
  mlir::MlirOp final_grad_h0 = ConcatDim(builder, all_h0_3d, /*dim=*/0);

  // Flatten parameter gradients in PyTorch order: (W_ih, W_hh, [b_ih, b_hh])
  // per layer.
  const size_t params_per_layer = has_biases ? 4 : 2;
  std::vector<mlir::MlirOp> all_grad_params;
  all_grad_params.reserve(num_layers * params_per_layer);
  for (int64_t l = 0; l < num_layers; ++l) {
    all_grad_params.push_back(to_out(final_grad_w_ih[l]));
    all_grad_params.push_back(to_out(final_grad_w_hh[l]));
    if (has_biases) {
      all_grad_params.push_back(*final_grad_b_ih[l]);
      all_grad_params.push_back(*final_grad_b_hh[l]);
    }
  }

  return {final_grad_x, final_grad_h0, std::move(all_grad_params)};
}

// Builds pipelined wavefront backward pass for multi-layer unidirectional GRU.
template <typename ToOutFn, typename ReduceBuilderFn>
GruMultiLayerBackwardOutputs BuildGruPipelinedWavefrontBackward(
    mlir::MlirBuilder& builder, absl::Span<const mlir::MlirOp> layer_inputs,
    absl::Span<const mlir::MlirOp> layer_outputs,
    absl::Span<const mlir::MlirOp> fwd_cached_acts,
    absl::Span<const mlir::MlirOp> h_inits, mlir::MlirOp incoming_seq_grad,
    absl::Span<const mlir::MlirOp> grad_hy_list,
    absl::Span<const mlir::MlirOp> w_ih_list,
    absl::Span<const mlir::MlirOp> w_hh_list, const bool has_biases,
    mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr x_bwd_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr w_dot_dims,
    const int64_t seq_len, const int64_t batch, const int64_t hidden,
    const int64_t input_size, const int64_t num_layers, ToOutFn to_out,
    const bool has_dropout, const double dropout,
    absl::Span<const mlir::MlirOp> dropout_masks, const bool batch_first,
    const mlir::stablehlo::Precision precision) {
  const int64_t num_chunks = seq_len / kGruUnrollFactor;
  const int64_t chunked_steps = num_chunks * kGruUnrollFactor;
  const int64_t rem_steps = seq_len % kGruUnrollFactor;
  const int64_t concat_dim = batch_first ? 1 : 0;

  const mlir::Type acc_elem_type =
      GetTensorTypeOrDie(grad_hy_list[0]).getElementType();

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = incoming_seq_grad.getValue().getLoc();
  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType i64_scalar_type =
      mlir::RankedTensorType::get({}, i64);

  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0f, acc_elem_type);
  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64);

  auto to_acc = [acc_elem_type](mlir::MlirOp op) -> mlir::MlirOp {
    if (GetTensorTypeOrDie(op).getElementType() == acc_elem_type) {
      return op;
    }
    return mlir::stablehlo::ConvertElementType(op, acc_elem_type);
  };

  std::vector<mlir::MlirOp> h_prev_seq = BuildWavefrontHPrevSeq(
      builder, h_inits, layer_outputs, num_layers, seq_len, batch, hidden,
      batch_first, concat_dim, to_acc);

  WavefrontBackwardRemainderResult rem = BuildWavefrontBackwardRemainder(
      builder, layer_inputs, fwd_cached_acts, h_prev_seq, incoming_seq_grad,
      grad_hy_list, w_ih_list, w_hh_list, has_biases, zero_const,
      sum_reduce_builder, hh_bwd_dot_dims, x_bwd_dot_dims, w_dot_dims, seq_len,
      chunked_steps, rem_steps, batch, hidden, input_size, num_layers, to_acc,
      has_dropout, dropout, dropout_masks, batch_first, concat_dim,
      acc_elem_type, zero_scalar, precision);

  const mlir::RankedTensorType grad_x_chunked_type =
      batch_first ? mlir::RankedTensorType::get(
                        {batch, chunked_steps, input_size}, acc_elem_type)
                  : mlir::RankedTensorType::get(
                        {chunked_steps, batch, input_size}, acc_elem_type);
  mlir::MlirOp grad_x_chunked_init =
      mlir::stablehlo::BroadcastInDim(grad_x_chunked_type, zero_scalar, {});

  mlir::MlirOp step_idx_init = MakeScalarConstant(builder, 0, i64);

  llvm::SmallVector<mlir::Type> loop_types;
  llvm::SmallVector<mlir::Value> loop_inits;
  loop_types.push_back(i64_scalar_type);
  loop_inits.push_back(step_idx_init.getValue());

  for (int64_t l = 0; l < num_layers; ++l) {
    loop_types.push_back(rem.rem_delta_h[l].getType());
    loop_inits.push_back(rem.rem_delta_h[l].getValue());
  }
  loop_types.push_back(grad_x_chunked_type);
  loop_inits.push_back(grad_x_chunked_init.getValue());

  for (int64_t l = 0; l < num_layers; ++l) {
    loop_types.push_back(rem.rem_grad_w_ih_init[l].getType());
    loop_inits.push_back(rem.rem_grad_w_ih_init[l].getValue());
  }
  for (int64_t l = 0; l < num_layers; ++l) {
    loop_types.push_back(rem.rem_grad_w_hh_init[l].getType());
    loop_inits.push_back(rem.rem_grad_w_hh_init[l].getValue());
  }
  if (has_biases) {
    for (int64_t l = 0; l < num_layers; ++l) {
      loop_types.push_back((*rem.rem_grad_b_ih_init[l]).getType());
      loop_inits.push_back((*rem.rem_grad_b_ih_init[l]).getValue());
    }
    for (int64_t l = 0; l < num_layers; ++l) {
      loop_types.push_back((*rem.rem_grad_b_hh_init[l]).getType());
      loop_inits.push_back((*rem.rem_grad_b_hh_init[l]).getValue());
    }
  }

  auto while_op =
      mlir::stablehlo::WhileOp::create(op_builder, loc, loop_types, loop_inits);

  mlir::Block* const cond_block = op_builder.createBlock(&while_op.getCond());
  cond_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(cond_block);

  mlir::MlirOp limit = MakeScalarConstant(builder, chunked_steps, i64);
  const mlir::Value cond =
      mlir::stablehlo::CompareOp::create(
          op_builder, loc, cond_block->getArgument(0), limit.getValue(),
          mlir::stablehlo::ComparisonDirection::LT)
          .getResult();
  mlir::stablehlo::ReturnOp::create(op_builder, loc, cond);

  mlir::Block* const body_block = op_builder.createBlock(&while_op.getBody());
  body_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(body_block);

  BuildWavefrontBackwardWhileBody(
      builder, body_block, loc, incoming_seq_grad, fwd_cached_acts, h_prev_seq,
      layer_inputs, w_hh_list, w_ih_list, dropout_masks, hh_bwd_dot_dims,
      x_bwd_dot_dims, w_dot_dims, chunked_steps, batch, hidden, input_size,
      num_layers, batch_first, concat_dim, acc_elem_type, to_acc, has_biases,
      has_dropout, dropout, zero_const, zero_i64, i64, sum_reduce_builder,
      precision);

  op_builder.setInsertionPointAfter(while_op);

  return FinalizeWavefrontBackwardOutputs(builder, while_op, rem.rem_grad_x_0,
                                          num_layers, batch, hidden, has_biases,
                                          concat_dim, to_out);
}

// Builds the complete StableHLO computational graph for multi-layer GRU forward
// pass.
//
// Graph Structure:
// 1. Accumulator Type Promotion: Upconverts inputs and initial states to
// accumulator
//    precision (F32) if needed for numerical precision.
// 2. Inter-Layer Dropout Initialization: Generates pseudo-random uniform
// dropout
//    masks across stacked layers using BuildUniformShlo if dropout is enabled.
// 3. Pipelined Wavefront Optimization: For unidirectional multi-layer networks
// (L >= 2,
//    seq_len >= kGruUnrollFactor) without activation caching, dispatches to
//    BuildGruPipelinedWavefrontForward to stream chunk activations between
//    layers in on-chip TPU vector memory.
// 4. Layer-by-Layer Execution:
//    - Performs upfront 3D batched GEMM projection (X * W_ih^T) across all time
//    steps.
//    - Executes recurrence unrolling (static or chunked) for layer l via
//      BuildGruLayerForward or BuildGruBidirLayerForward.
//    - Applies inter-layer dropout between stacked layers (l < num_layers - 1).
//    - Collects final hidden states across all layers and directions.
// 5. Activation Caching: Caches pre-activation tensors if cache_activations is
// true
//    for consumption by the backward autograd pass.
//
// Arguments:
//   builder_inputs: Array containing input tensor, initial hidden state,
//   weight/bias
//     parameters for all layers, and optional RNG state.
//   builder: MLIR StableHLO graph builder.
//   has_biases: Whether additive bias parameters are present.
//   num_layers: Number of stacked recurrent layers.
//   batch_first: Whether sequence layout is [batch, seq, feature].
//   batch, seq_len, hidden: Tensor dimension sizes.
//   out_dtype: Target output tensor element type.
//   acc_dtype: Accumulator element type (typically F32 for bfloat16/half
//   inputs). params_per_layer: Count of parameter tensors per layer.
//   bidirectional: Whether bidirectional recurrence is enabled.
//   num_directions: 2 for bidirectional, 1 for unidirectional.
//   params_per_direction: Count of parameter tensors per direction (4 with
//   bias, 2 without). has_dropout: Whether inter-layer dropout is active.
//   dropout: Dropout probability.
//   train: Whether model is in training mode.
//   cache_activations: Whether gate pre-activations should be retained for
//   backward pass. current_precision: Precision mode for StableHLO matrix
//   multiplication operations.
//
// Returns:
//   Vector of MLIR output operations: [output_sequence, final_hidden_state,
//   (cached_acts...)].
// Evaluates forward pass for a single unidirectional layer l.
//
// What it computes:
// - Computes input projection X * W_ih^T via mixed-precision dot general.
// - Broadcasts and adds b_ih if has_biases is true.
// - Extracts initial hidden state h_0 for layer l.
// - Calls BuildGruLayerForward to execute recurrent cell loop.
// - Returns layer output sequence and final hidden state (plus optional cached
// acts).
template <typename ToOutFn, typename ToAccFn>
inline GruLayerOutputs EvaluateGruUnidirLayerForward(
    mlir::MlirBuilder& builder, mlir::MlirOp current_input,
    absl::Span<mlir::MlirOp> builder_inputs, const size_t params_offset,
    const size_t params_per_layer, const int64_t l,
    const mlir::stablehlo::DotDimensionNumbersAttr ih_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    const mlir::ElementType acc_dtype,
    const mlir::stablehlo::Precision current_precision, const bool has_biases,
    mlir::MlirOp h_0_acc, const int64_t batch, const int64_t hidden,
    const int64_t seq_len, const bool batch_first, ToOutFn to_out,
    ToAccFn to_acc, const bool cache_activations) {
  mlir::MlirOp w_ih = builder_inputs[params_offset + l * params_per_layer + 0];
  mlir::MlirOp w_hh = builder_inputs[params_offset + l * params_per_layer + 1];

  mlir::MlirOp x_proj = MixedPrecisionDotGeneral(
      current_input, w_ih, ih_dot_dims, acc_dtype, current_precision);

  std::optional<mlir::MlirOp> b_hh = std::nullopt;
  if (has_biases) {
    mlir::MlirOp b_ih =
        to_acc(builder_inputs[params_offset + l * params_per_layer + 2]);
    mlir::MlirOp b_ih_bcast = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(x_proj), b_ih, {/*broadcast_dimensions=*/2});
    x_proj = mlir::stablehlo::Add(x_proj, b_ih_bcast);
    b_hh = to_acc(builder_inputs[params_offset + l * params_per_layer + 3]);
  }

  mlir::MlirOp h_init_l = ExtractLayer2D(h_0_acc, l, batch, hidden);
  return BuildGruLayerForward(builder, x_proj, h_init_l, w_hh, hh_dot_dims,
                              seq_len, batch, hidden, batch_first, to_out, b_hh,
                              current_precision, cache_activations);
}

// Evaluates forward pass for a single bidirectional layer l.
//
// What it computes:
// - Performs forward and reverse input projections: X * W_ih_fwd^T, X *
// W_ih_rev^T.
// - Broadcasts and adds biases b_ih_fwd, b_ih_rev if has_biases is true.
// - Extracts h_0 for both directions (2*l and 2*l + 1).
// - Calls BuildGruBidirLayerForward to execute bidirectional recurrence.
template <typename ToOutFn, typename ToAccFn>
inline GruBidirLayerOutputs EvaluateGruBidirLayerForward(
    mlir::MlirBuilder& builder, mlir::MlirOp current_input,
    absl::Span<mlir::MlirOp> builder_inputs, const size_t params_offset,
    const size_t params_per_layer, const size_t params_per_direction,
    const int64_t l, const mlir::stablehlo::DotDimensionNumbersAttr ih_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    const mlir::ElementType acc_dtype,
    const mlir::stablehlo::Precision current_precision, const bool has_biases,
    mlir::MlirOp h_0_acc, const int64_t batch, const int64_t hidden,
    const int64_t seq_len, const bool batch_first, ToOutFn to_out,
    ToAccFn to_acc, const bool cache_activations) {
  const size_t p_fwd = params_offset + l * params_per_layer;
  const size_t p_rev = p_fwd + params_per_direction;

  mlir::MlirOp w_ih_fwd = builder_inputs[p_fwd + 0];
  mlir::MlirOp w_hh_fwd = builder_inputs[p_fwd + 1];
  mlir::MlirOp w_ih_rev = builder_inputs[p_rev + 0];
  mlir::MlirOp w_hh_rev = builder_inputs[p_rev + 1];

  mlir::MlirOp x_proj_fwd = MixedPrecisionDotGeneral(
      current_input, w_ih_fwd, ih_dot_dims, acc_dtype, current_precision);
  mlir::MlirOp x_proj_rev = MixedPrecisionDotGeneral(
      current_input, w_ih_rev, ih_dot_dims, acc_dtype, current_precision);

  std::optional<mlir::MlirOp> b_hh_fwd = std::nullopt;
  std::optional<mlir::MlirOp> b_hh_rev = std::nullopt;
  if (has_biases) {
    mlir::MlirOp b_ih_fwd = to_acc(builder_inputs[p_fwd + 2]);
    mlir::MlirOp b_ih_fwd_bcast = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(x_proj_fwd), b_ih_fwd, {2});
    x_proj_fwd = mlir::stablehlo::Add(x_proj_fwd, b_ih_fwd_bcast);
    b_hh_fwd = to_acc(builder_inputs[p_fwd + 3]);

    mlir::MlirOp b_ih_rev = to_acc(builder_inputs[p_rev + 2]);
    mlir::MlirOp b_ih_rev_bcast = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(x_proj_rev), b_ih_rev, {2});
    x_proj_rev = mlir::stablehlo::Add(x_proj_rev, b_ih_rev_bcast);
    b_hh_rev = to_acc(builder_inputs[p_rev + 3]);
  }

  mlir::MlirOp h_init_fwd = ExtractLayer2D(h_0_acc, 2 * l, batch, hidden);
  mlir::MlirOp h_init_rev = ExtractLayer2D(h_0_acc, 2 * l + 1, batch, hidden);

  return BuildGruBidirLayerForward(
      builder, x_proj_fwd, x_proj_rev, h_init_fwd, h_init_rev, w_hh_fwd,
      w_hh_rev, hh_dot_dims, seq_len, batch, hidden, batch_first, to_out,
      b_hh_fwd, b_hh_rev, current_precision, cache_activations);
}

// Applies inter-layer dropout between stacked recurrent layers.
inline InterLayerDropoutResult ApplyLayerDropoutStep(
    mlir::MlirOp current_layer_out, mlir::MlirOp rand_op, const int64_t l,
    const int64_t batch, const int64_t seq_len, const int64_t out_dim,
    const bool batch_first, const double dropout) {
  Dimensions layer_rand_shape = batch_first
                                    ? Dimensions{batch, seq_len, out_dim}
                                    : Dimensions{seq_len, batch, out_dim};
  Dimensions slice_starts = {l, 0, 0, 0};
  Dimensions slice_limits = batch_first
                                ? Dimensions{l + 1, batch, seq_len, out_dim}
                                : Dimensions{l + 1, seq_len, batch, out_dim};
  Dimensions slice_strides = {1, 1, 1, 1};
  mlir::MlirOp layer_rand_4d = mlir::stablehlo::Slice(
      rand_op, slice_starts, slice_limits, slice_strides);
  mlir::MlirOp layer_rand =
      mlir::stablehlo::Reshape(layer_rand_4d, layer_rand_shape);
  return ApplyInterLayerDropout(current_layer_out, layer_rand, dropout);
}

// Builds pipelined wavefront forward path for multi-layer unidirectional GRU.
template <typename ToOutFn, typename ToAccFn>
mlir::SmallVector<mlir::MlirOp> BuildGruForwardWavefrontPath(
    mlir::MlirBuilder& builder, mlir::MlirOp current_input,
    absl::Span<mlir::MlirOp> builder_inputs, mlir::MlirOp h_0_acc,
    const size_t params_per_layer, const bool has_biases,
    const int64_t num_layers, const int64_t seq_len, const int64_t batch,
    const int64_t hidden, const bool batch_first,
    const mlir::stablehlo::DotDimensionNumbersAttr ih_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    const mlir::ElementType acc_dtype, ToOutFn to_out, ToAccFn to_acc,
    const double dropout, const bool train,
    const std::optional<mlir::MlirOp>& rand_op,
    const mlir::stablehlo::Precision current_precision) {
  mlir::MlirOp w_ih_0 = builder_inputs[2 + 0 * params_per_layer + 0];
  mlir::MlirOp x_proj_0 = MixedPrecisionDotGeneral(
      current_input, w_ih_0, ih_dot_dims, acc_dtype, current_precision);
  if (has_biases) {
    mlir::MlirOp b_ih = to_acc(builder_inputs[2 + 0 * params_per_layer + 2]);
    mlir::MlirOp b_ih_bcast = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(x_proj_0), b_ih, {/*broadcast_dimensions=*/2});
    x_proj_0 = mlir::stablehlo::Add(x_proj_0, b_ih_bcast);
  }

  std::vector<GruWavefrontLayerWeight> layer_weights(num_layers);
  std::vector<mlir::MlirOp> h_inits(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    h_inits[l] = ExtractLayer2D(h_0_acc, l, batch, hidden);
    layer_weights[l].w_hh = builder_inputs[2 + l * params_per_layer + 1];
    if (has_biases) {
      layer_weights[l].b_hh =
          to_acc(builder_inputs[2 + l * params_per_layer + 3]);
    }
    if (l > 0) {
      layer_weights[l].w_ih = builder_inputs[2 + l * params_per_layer + 0];
      if (has_biases) {
        layer_weights[l].b_ih =
            to_acc(builder_inputs[2 + l * params_per_layer + 2]);
      }
    }
  }

  GruLayerOutputs wavefront_out = BuildGruPipelinedWavefrontForward(
      builder, x_proj_0, h_inits, layer_weights, hh_dot_dims, ih_dot_dims,
      seq_len, batch, hidden, num_layers, batch_first, to_out, dropout, train,
      rand_op, current_precision);

  mlir::SmallVector<mlir::MlirOp> results;
  results.push_back(wavefront_out.layer_output_seq);
  results.push_back(wavefront_out.final_h);
  return results;
}

// Result of evaluating a single layer forward step.
struct GruForwardLayerStepResult {
  mlir::MlirOp layer_output;
  mlir::MlirOp final_h;
  std::vector<mlir::MlirOp> cached_acts;
};

// Executes forward computation for a single layer l and collects cached
// activations.
template <typename ToOutFn, typename ToAccFn>
inline GruForwardLayerStepResult ExecuteGruForwardLayerStep(
    mlir::MlirBuilder& builder, mlir::MlirOp current_input,
    absl::Span<mlir::MlirOp> builder_inputs, const size_t params_offset,
    const size_t params_per_layer, const size_t params_per_direction,
    const int64_t l, const bool bidirectional,
    const mlir::stablehlo::DotDimensionNumbersAttr ih_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    const mlir::ElementType acc_dtype,
    const mlir::stablehlo::Precision current_precision, const bool has_biases,
    mlir::MlirOp h_0_acc, const int64_t batch, const int64_t hidden,
    const int64_t seq_len, const bool batch_first, ToOutFn to_out,
    ToAccFn to_acc, const bool cache_activations) {
  GruForwardLayerStepResult res;
  if (!bidirectional) {
    GruLayerOutputs layer_out = EvaluateGruUnidirLayerForward(
        builder, current_input, builder_inputs, params_offset, params_per_layer,
        l, ih_dot_dims, hh_dot_dims, acc_dtype, current_precision, has_biases,
        h_0_acc, batch, hidden, seq_len, batch_first, to_out, to_acc,
        cache_activations);
    res.layer_output = layer_out.layer_output_seq;
    res.final_h = layer_out.final_h;
    if (cache_activations && layer_out.cached_acts.has_value()) {
      res.cached_acts.push_back(*layer_out.cached_acts);
    }
  } else {
    GruBidirLayerOutputs bidir_out = EvaluateGruBidirLayerForward(
        builder, current_input, builder_inputs, params_offset, params_per_layer,
        params_per_direction, l, ih_dot_dims, hh_dot_dims, acc_dtype,
        current_precision, has_biases, h_0_acc, batch, hidden, seq_len,
        batch_first, to_out, to_acc, cache_activations);
    res.layer_output = bidir_out.layer_output_seq;
    res.final_h = bidir_out.final_h;
    if (cache_activations) {
      if (bidir_out.cached_acts_fwd.has_value()) {
        res.cached_acts.push_back(*bidir_out.cached_acts_fwd);
      }
      if (bidir_out.cached_acts_rev.has_value()) {
        res.cached_acts.push_back(*bidir_out.cached_acts_rev);
      }
    }
  }
  return res;
}

// Builds the complete StableHLO computational graph for multi-layer GRU forward
// pass.
//
// Graph Structure:
// 1. Accumulator Type Promotion: Upconverts inputs and initial states to
// accumulator
//    precision (typically f32) when mixed-precision execution is enabled.
// 2. Inter-Layer Dropout Tensor Generation: Evaluates BuildUniformShlo to
// generate
//    pseudo-random dropout masks for intermediate layer outputs during
//    training.
// 3. Pipelined Wavefront Optimization: For multi-layer unidirectional GRUs
// (num_layers >= 2,
//    seq_len >= kGruUnrollFactor, inference or no activation caching),
//    dispatches to BuildGruPipelinedWavefrontForward to stream chunk outputs
//    across layers directly in TPU vector registers.
// 4. Layer-by-Layer Forward Recurrence: Iteratively evaluates
// BuildGruLayerForward or
//    BuildGruBidirLayerForward for layers l = 0 to num_layers - 1.
// 5. Result Packing: Stacks final hidden states [num_layers * num_directions,
// batch, hidden]
//    and bundles cached gate pre-activations for backward execution.
//
// Arguments:
//   builder_inputs: Array containing input sequence, initial states (hx),
//   weights,
//     biases, and optional RNG seed tensors.
//   builder: MLIR StableHLO graph builder.
//   has_biases: Whether bias parameters are included.
//   num_layers: Number of recurrent layers.
//   batch_first: Whether sequence layout is [batch, seq, feature].
//   batch, seq_len, hidden: Tensor dimension sizes.
//   out_dtype, acc_dtype: Element datatypes for storage and accumulation.
//   params_per_layer: Count of parameter tensors per layer.
//   bidirectional: Whether bidirectional recurrence is enabled.
//   num_directions: 2 for bidirectional, 1 for unidirectional.
//   params_per_direction: Count of parameter tensors per direction (4 with
//   bias, 2 without). has_dropout: Whether inter-layer dropout is active.
//   dropout: Dropout probability.
//   train: Whether model is in training mode.
//   cache_activations: Whether gate pre-activations should be retained for
//   backward pass. current_precision: Precision mode for StableHLO matrix
//   multiplication operations.
//
// Returns:
//   Vector of MLIR output operations: [output_sequence, final_hidden_state,
//   (cached_acts...)].
absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildGruForwardShloGraph(
    absl::Span<mlir::MlirOp> builder_inputs, mlir::MlirBuilder& builder,
    const bool has_biases, const int64_t num_layers, const bool batch_first,
    const int64_t batch, const int64_t seq_len, const int64_t hidden,
    const mlir::ElementType out_dtype, const mlir::ElementType acc_dtype,
    const size_t params_per_layer, const bool bidirectional,
    const int64_t num_directions, const size_t params_per_direction,
    const bool has_dropout, const double dropout, const bool train,
    const bool cache_activations,
    const mlir::stablehlo::Precision current_precision) {
  auto to_acc = [acc_dtype, out_dtype](mlir::MlirOp op) -> mlir::MlirOp {
    return (acc_dtype == out_dtype)
               ? op
               : mlir::stablehlo::ConvertElementType(op, acc_dtype);
  };
  auto to_out = [out_dtype, acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
    return (acc_dtype == out_dtype)
               ? op
               : mlir::stablehlo::ConvertElementType(op, out_dtype);
  };

  mlir::MlirOp current_input = builder_inputs[0];
  mlir::MLIRContext& ctx = builder.getContext();

  const auto ih_dot_dims = MakeDotDims(&ctx, {2}, {1});
  const auto hh_dot_dims = MakeDotDims(&ctx, {1}, {1});

  mlir::MlirOp h_0_acc = to_acc(builder_inputs[1]);

  std::optional<mlir::MlirOp> rand_op;
  if (has_dropout) {
    mlir::MlirOp rng_input_state = builder_inputs.back();
    const int64_t inter_layers = num_layers - 1;
    const int64_t layer_out_dim = num_directions * hidden;
    Dimensions rand_shape =
        batch_first ? Dimensions{inter_layers, batch, seq_len, layer_out_dim}
                    : Dimensions{inter_layers, seq_len, batch, layer_out_dim};
    TT_ASSIGN_OR_RETURN(rand_op, BuildUniformShlo(rng_input_state, 0.0, 1.0,
                                                  rand_shape, out_dtype));
  }

  if (!bidirectional && num_layers >= 2 && seq_len >= kGruUnrollFactor &&
      !cache_activations) {
    return BuildGruForwardWavefrontPath(
        builder, current_input, builder_inputs, h_0_acc, params_per_layer,
        has_biases, num_layers, seq_len, batch, hidden, batch_first,
        ih_dot_dims, hh_dot_dims, acc_dtype, to_out, to_acc, dropout, train,
        rand_op, current_precision);
  }

  std::vector<mlir::MlirOp> all_final_h;
  all_final_h.reserve(num_layers * num_directions);
  std::vector<mlir::MlirOp> cached_acts;
  if (cache_activations) {
    cached_acts.reserve(num_layers * num_directions);
  }

  mlir::MlirOp current_layer_out;
  const size_t params_offset = 2;

  for (int64_t l = 0; l < num_layers; ++l) {
    GruForwardLayerStepResult step_res = ExecuteGruForwardLayerStep(
        builder, current_input, builder_inputs, params_offset, params_per_layer,
        params_per_direction, l, bidirectional, ih_dot_dims, hh_dot_dims,
        acc_dtype, current_precision, has_biases, h_0_acc, batch, hidden,
        seq_len, batch_first, to_out, to_acc, cache_activations);

    current_layer_out = step_res.layer_output;
    all_final_h.push_back(step_res.final_h);
    for (const auto& act : step_res.cached_acts) {
      cached_acts.push_back(act);
    }

    if (has_dropout && l < num_layers - 1) {
      auto [dropped, mask] =
          ApplyLayerDropoutStep(current_layer_out, *rand_op, l, batch, seq_len,
                                num_directions * hidden, batch_first, dropout);
      current_layer_out = dropped;
    }

    current_input = current_layer_out;
  }

  mlir::MlirOp stacked_h = ConcatDim(builder, all_final_h, /*dim=*/0);

  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(2 + cached_acts.size());
  results.push_back(current_layer_out);
  results.push_back(stacked_h);
  for (const auto& act : cached_acts) {
    results.push_back(act);
  }
  return results;
}

// Collects and packs all Torch input tensors for GRU execution.
inline std::vector<at::Tensor> CollectGruInputTensors(
    const at::Tensor& input, const at::Tensor& hx, const at::TensorList params,
    bool has_dropout, const std::optional<at::Tensor>& rng_state) {
  std::vector<at::Tensor> inputs;
  inputs.reserve(2 + params.size() + (has_dropout ? 1 : 0));
  inputs.push_back(input);
  inputs.push_back(hx);
  for (const at::Tensor& p : params) {
    inputs.push_back(p);
  }
  if (has_dropout) {
    inputs.push_back(*rng_state);
  }
  return inputs;
}

// Layout configuration of output tensors.
struct GruOutputLayouts {
  std::vector<mlir::ElementType> dtypes;
  std::vector<Dimensions> dims_storage;
};

// Computes output dtypes and tensor dimension shapes for GRU forward dispatch.
inline GruOutputLayouts ComputeGruForwardOutputLayouts(
    mlir::ElementType out_dtype, mlir::ElementType acc_dtype, int64_t batch,
    int64_t seq_len, int64_t hidden, int64_t num_layers, int64_t num_directions,
    bool batch_first, bool cache_activations) {
  GruOutputLayouts layouts;
  layouts.dtypes.push_back(out_dtype);
  if (batch_first) {
    layouts.dims_storage.push_back(
        Dimensions{batch, seq_len, num_directions * hidden});
  } else {
    layouts.dims_storage.push_back(
        Dimensions{seq_len, batch, num_directions * hidden});
  }

  layouts.dtypes.push_back(out_dtype);
  layouts.dims_storage.push_back(
      Dimensions{num_layers * num_directions, batch, hidden});

  if (cache_activations) {
    const Dimensions act_dims = batch_first
                                    ? Dimensions{batch, seq_len, 4 * hidden}
                                    : Dimensions{seq_len, batch, 4 * hidden};
    for (int64_t l = 0; l < num_layers; ++l) {
      for (int64_t d = 0; d < num_directions; ++d) {
        layouts.dtypes.push_back(acc_dtype);
        layouts.dims_storage.push_back(act_dims);
      }
    }
  }
  return layouts;
}

// Internal implementation helper that validates GRU inputs and dispatches the
// forward StableHLO graph construction to the TPU execution runtime.
//
// Arguments:
//   input: Input sequence tensor.
//   hx: Initial hidden states tensor.
//   params: List of weight and bias parameter tensors.
//   has_biases: Whether biases are included.
//   num_layers: Number of recurrent layers.
//   dropout: Dropout probability.
//   train: Whether execution is in training mode.
//   bidirectional: Whether layer is bidirectional.
//   batch_first: Whether sequence layout is [batch, seq, feature].
//   param_keys: Kernel compilation cache keys.
//   cache_activations: Whether to cache forward activations for autograd.
//   rng_state: Optional RNG seed tensor for dropout generation.
//
// Returns:
//   Vector of device buffer references corresponding to the kernel outputs.
absl::StatusOr<std::vector<DeviceBufferRef>> GruInputImpl(
    const at::Tensor& input, const at::Tensor& hx, const at::TensorList params,
    const bool has_biases, const int64_t num_layers, const double dropout,
    const bool train, const bool bidirectional, const bool batch_first,
    OpParamCacheKeys param_keys, const bool cache_activations,
    std::optional<at::Tensor> rng_state = std::nullopt) {
  TT_RETURN_IF_ERROR(ValidateGruInputs(input, hx, params, has_biases,
                                       num_layers, dropout, train,
                                       bidirectional, batch_first));

  const auto current_precision = GetAndAddPrecisionTo(param_keys);
  const bool has_dropout = (dropout > 0.0 && train && num_layers > 1);
  if (has_dropout) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal autograd RNG state
        rng_state.has_value(), error::kInvalidArgument)
        << "gru with dropout requires rng_state";
  }

  const int64_t batch = batch_first ? input.size(0) : input.size(1);
  const int64_t seq_len = batch_first ? input.size(1) : input.size(0);
  const int64_t hidden = hx.size(2);
  const int64_t num_directions = bidirectional ? 2 : 1;
  const size_t params_per_direction = has_biases ? 4 : 2;
  const size_t params_per_layer = params_per_direction * num_directions;

  std::vector<at::Tensor> inputs =
      CollectGruInputTensors(input, hx, params, has_dropout, rng_state);

  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));
  TT_ASSIGN_OR_RETURN(
      const auto acc_dtype,
      ConvertTo<mlir::ElementType>(ToAccumulateType(input.scalar_type())));

  GruOutputLayouts layouts = ComputeGruForwardOutputLayouts(
      out_dtype, acc_dtype, batch, seq_len, hidden, num_layers, num_directions,
      batch_first, cache_activations);

  std::vector<absl::Span<const int64_t>> out_dims_list;
  out_dims_list.reserve(layouts.dims_storage.size());
  for (const auto& dims : layouts.dims_storage) {
    out_dims_list.push_back(dims);
  }

  auto op_builder = [has_biases, num_layers, batch_first, batch, seq_len,
                     hidden, out_dtype, acc_dtype, params_per_layer,
                     bidirectional, num_directions, params_per_direction,
                     has_dropout, dropout, train, cache_activations,
                     current_precision](absl::Span<mlir::MlirOp> builder_inputs,
                                        mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    return BuildGruForwardShloGraph(
        builder_inputs, builder, has_biases, num_layers, batch_first, batch,
        seq_len, hidden, out_dtype, acc_dtype, params_per_layer, bidirectional,
        num_directions, params_per_direction, has_dropout, dropout, train,
        cache_activations, current_precision);
  };

  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = layouts.dtypes,
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = std::move(param_keys),
  };

  return DispatchOp<kDynamicSize, kDynamicSize>(std::move(op_builder), inputs,
                                                std::move(options));
}

// Reconstructed forward data required by backward pass.
struct GruReconstructedForwardData {
  std::vector<mlir::MlirOp> layer_inputs;
  std::vector<mlir::MlirOp> layer_outputs;
  std::vector<mlir::MlirOp> fwd_cached_acts;
  std::vector<mlir::MlirOp> dropout_masks;
};

// Reconstructs forward activations and layer outputs needed for backward
// gradient computation.
template <typename ToOutFn, typename ToAccFn>
GruReconstructedForwardData ReconstructGruForwardPass(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> builder_inputs,
    mlir::MlirOp input_seq, mlir::MlirOp hx_val,
    const std::optional<mlir::MlirOp>& rand_op, const size_t params_offset,
    const size_t cached_acts_offset, const size_t params_per_layer,
    const size_t params_per_direction, const int64_t num_layers,
    const int64_t num_directions, const int64_t batch, const int64_t seq_len,
    const int64_t hidden, const bool batch_first, const bool bidirectional,
    const bool has_biases, const bool has_dropout, const double dropout,
    const mlir::stablehlo::DotDimensionNumbersAttr ih_dot_dims,
    const mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    const mlir::ElementType acc_dtype, ToOutFn to_out, ToAccFn to_acc,
    const mlir::stablehlo::Precision current_precision) {
  GruReconstructedForwardData fwd_data;
  fwd_data.layer_inputs.resize(num_layers);
  fwd_data.layer_outputs.resize(num_layers);
  fwd_data.fwd_cached_acts.resize(num_layers * num_directions);
  fwd_data.dropout_masks.resize(num_layers > 1 ? num_layers - 1 : 0);

  mlir::MlirOp curr_fwd_in = input_seq;

  for (int64_t l = 0; l < num_layers; ++l) {
    fwd_data.layer_inputs[l] = curr_fwd_in;
    if (!bidirectional) {
      GruLayerOutputs l_out = EvaluateGruUnidirLayerForward(
          builder, curr_fwd_in, builder_inputs, params_offset, params_per_layer,
          l, ih_dot_dims, hh_dot_dims, acc_dtype, current_precision, has_biases,
          hx_val, batch, hidden, seq_len, batch_first, to_out, to_acc,
          /*cache_activations=*/true);

      fwd_data.layer_outputs[l] = l_out.layer_output_seq;
      if (cached_acts_offset + l < builder_inputs.size()) {
        fwd_data.fwd_cached_acts[l] = builder_inputs[cached_acts_offset + l];
      } else if (l_out.cached_acts.has_value()) {
        fwd_data.fwd_cached_acts[l] = *l_out.cached_acts;
      }
      curr_fwd_in = l_out.layer_output_seq;
    } else {
      GruBidirLayerOutputs bidir_out = EvaluateGruBidirLayerForward(
          builder, curr_fwd_in, builder_inputs, params_offset, params_per_layer,
          params_per_direction, l, ih_dot_dims, hh_dot_dims, acc_dtype,
          current_precision, has_biases, hx_val, batch, hidden, seq_len,
          batch_first, to_out, to_acc, /*cache_activations=*/true);

      fwd_data.layer_outputs[l] = bidir_out.layer_output_seq;
      const size_t act_idx = 2 * l;
      if (cached_acts_offset + act_idx + 1 < builder_inputs.size()) {
        fwd_data.fwd_cached_acts[act_idx] =
            builder_inputs[cached_acts_offset + act_idx];
        fwd_data.fwd_cached_acts[act_idx + 1] =
            builder_inputs[cached_acts_offset + act_idx + 1];
      } else {
        if (bidir_out.cached_acts_fwd.has_value()) {
          fwd_data.fwd_cached_acts[act_idx] = *bidir_out.cached_acts_fwd;
        }
        if (bidir_out.cached_acts_rev.has_value()) {
          fwd_data.fwd_cached_acts[act_idx + 1] = *bidir_out.cached_acts_rev;
        }
      }
      curr_fwd_in = bidir_out.layer_output_seq;
    }

    if (has_dropout && l < num_layers - 1) {
      auto [dropped, mask] =
          ApplyLayerDropoutStep(curr_fwd_in, *rand_op, l, batch, seq_len,
                                num_directions * hidden, batch_first, dropout);
      curr_fwd_in = dropped;
      fwd_data.dropout_masks[l] = mask;
    }
  }

  return fwd_data;
}

// Dispatches pipelined wavefront backward pass for multi-layer unidirectional
// GRU.
template <typename ToOutFn, typename ReduceBuilderFn>
mlir::SmallVector<mlir::MlirOp> DispatchGruWavefrontBackwardPath(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> builder_inputs,
    mlir::MlirOp hx_val, mlir::MlirOp incoming_grad_y,
    mlir::MlirOp incoming_grad_hy, const GruReconstructedForwardData& fwd_data,
    const size_t params_offset, const size_t params_per_layer,
    const int64_t num_layers, const int64_t batch, const int64_t seq_len,
    const int64_t hidden, const int64_t input_size, const bool batch_first,
    const bool has_biases, const bool has_dropout, const double dropout,
    mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder, ToOutFn to_out,
    const mlir::stablehlo::Precision current_precision) {
  mlir::MLIRContext& ctx = builder.getContext();
  const auto hh_bwd_dot_dims = MakeDotDims(&ctx, {1}, {0});
  const auto x_bwd_dot_dims = MakeDotDims(&ctx, {1}, {0});
  const auto w_dot_dims = MakeDotDims(&ctx, {0}, {0});

  std::vector<mlir::MlirOp> h_inits(num_layers);
  std::vector<mlir::MlirOp> grad_hy_list(num_layers);
  std::vector<mlir::MlirOp> w_ih_list(num_layers);
  std::vector<mlir::MlirOp> w_hh_list(num_layers);

  for (int64_t l = 0; l < num_layers; ++l) {
    h_inits[l] = ExtractLayer2D(hx_val, l, batch, hidden);
    grad_hy_list[l] = ExtractLayer2D(incoming_grad_hy, l, batch, hidden);
    w_ih_list[l] = builder_inputs[params_offset + l * params_per_layer + 0];
    w_hh_list[l] = builder_inputs[params_offset + l * params_per_layer + 1];
  }

  GruMultiLayerBackwardOutputs wavefront_bwd =
      BuildGruPipelinedWavefrontBackward(
          builder, fwd_data.layer_inputs, fwd_data.layer_outputs,
          fwd_data.fwd_cached_acts, h_inits, incoming_grad_y, grad_hy_list,
          w_ih_list, w_hh_list, has_biases, zero_const, sum_reduce_builder,
          hh_bwd_dot_dims, x_bwd_dot_dims, w_dot_dims, seq_len, batch, hidden,
          input_size, num_layers, to_out, has_dropout, dropout,
          fwd_data.dropout_masks, batch_first, current_precision);

  mlir::SmallVector<mlir::MlirOp> outputs;
  outputs.reserve(2 + wavefront_bwd.all_grad_params.size());
  outputs.push_back(to_out(wavefront_bwd.grad_x));
  outputs.push_back(wavefront_bwd.grad_h0);
  for (const auto& gp : wavefront_bwd.all_grad_params) {
    outputs.push_back(gp);
  }
  return outputs;
}

// Evaluates backward pass for a single layer in the reverse topological
// cascade.
template <typename ToOutFn, typename ReduceBuilderFn>
void ComputeLayerByLayerBackwardStep(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> builder_inputs,
    int64_t l, int64_t num_layers, int64_t num_directions, int64_t batch,
    int64_t seq_len, int64_t hidden, int64_t input_size, bool batch_first,
    bool bidirectional, bool has_biases, size_t params_offset,
    size_t params_per_layer, size_t params_per_direction, mlir::MlirOp hx_val,
    mlir::MlirOp incoming_grad_hy, const GruReconstructedForwardData& fwd_data,
    mlir::MlirOp& curr_grad_y, std::vector<mlir::MlirOp>& all_grad_h0,
    std::vector<mlir::MlirOp>& all_grad_params, mlir::MlirOp zero_const,
    ReduceBuilderFn sum_reduce_builder, ToOutFn to_out,
    mlir::stablehlo::Precision current_precision) {
  const int64_t in_dim = (l == 0) ? input_size : num_directions * hidden;
  mlir::MlirOp layer_out_l = fwd_data.layer_outputs[l];

  if (!bidirectional) {
    mlir::MlirOp w_ih =
        builder_inputs[params_offset + l * params_per_layer + 0];
    mlir::MlirOp w_hh =
        builder_inputs[params_offset + l * params_per_layer + 1];
    mlir::MlirOp h_init_l = ExtractLayer2D(hx_val, l, batch, hidden);
    mlir::MlirOp grad_hy_l = ExtractLayer2D(incoming_grad_hy, l, batch, hidden);

    GruLayerBackwardOutputs bwd = ComputeLayerBackward(
        builder, curr_grad_y, grad_hy_l, fwd_data.layer_inputs[l], h_init_l,
        layer_out_l, fwd_data.fwd_cached_acts[l], w_ih, w_hh, has_biases,
        seq_len, batch, hidden, in_dim, batch_first, to_out, zero_const,
        sum_reduce_builder, current_precision);

    all_grad_h0[l] = bwd.grad_h0;

    const size_t p_out_idx = l * params_per_layer;
    all_grad_params[p_out_idx + 0] = bwd.grad_w_ih;
    all_grad_params[p_out_idx + 1] = bwd.grad_w_hh;
    if (has_biases) {
      all_grad_params[p_out_idx + 2] = *bwd.grad_b_ih;
      all_grad_params[p_out_idx + 3] = *bwd.grad_b_hh;
    }

    curr_grad_y = bwd.grad_x;
  } else {
    const size_t p_fwd = params_offset + l * params_per_layer;
    const size_t p_rev = p_fwd + params_per_direction;

    mlir::MlirOp w_ih_fwd = builder_inputs[p_fwd + 0];
    mlir::MlirOp w_hh_fwd = builder_inputs[p_fwd + 1];
    mlir::MlirOp w_ih_rev = builder_inputs[p_rev + 0];
    mlir::MlirOp w_hh_rev = builder_inputs[p_rev + 1];

    mlir::MlirOp h_init_fwd = ExtractLayer2D(hx_val, 2 * l, batch, hidden);
    mlir::MlirOp h_init_rev = ExtractLayer2D(hx_val, 2 * l + 1, batch, hidden);

    mlir::MlirOp y_out_fwd =
        batch_first
            ? mlir::stablehlo::Slice(layer_out_l, {0, 0, 0},
                                     {batch, seq_len, hidden}, {1, 1, 1})
            : mlir::stablehlo::Slice(layer_out_l, {0, 0, 0},
                                     {seq_len, batch, hidden}, {1, 1, 1});
    mlir::MlirOp y_out_rev =
        batch_first
            ? mlir::stablehlo::Slice(layer_out_l, {0, 0, hidden},
                                     {batch, seq_len, 2 * hidden}, {1, 1, 1})
            : mlir::stablehlo::Slice(layer_out_l, {0, 0, hidden},
                                     {seq_len, batch, 2 * hidden}, {1, 1, 1});

    mlir::MlirOp grad_hy_fwd =
        ExtractLayer2D(incoming_grad_hy, 2 * l, batch, hidden);
    mlir::MlirOp grad_hy_rev =
        ExtractLayer2D(incoming_grad_hy, 2 * l + 1, batch, hidden);
    mlir::MlirOp grad_hy_layer = mlir::stablehlo::Concatenate(
        builder,
        {mlir::stablehlo::Reshape(grad_hy_fwd, {1, batch, hidden}),
         mlir::stablehlo::Reshape(grad_hy_rev, {1, batch, hidden})},
        /*dim=*/0);

    GruBidirLayerBackwardOutputs bwd = ComputeBidirLayerBackward(
        builder, curr_grad_y, grad_hy_layer, fwd_data.layer_inputs[l],
        h_init_fwd, h_init_rev, y_out_fwd, y_out_rev,
        fwd_data.fwd_cached_acts[2 * l], fwd_data.fwd_cached_acts[2 * l + 1],
        w_ih_fwd, w_hh_fwd, w_ih_rev, w_hh_rev, has_biases, seq_len, batch,
        hidden, in_dim, batch_first, to_out, zero_const, sum_reduce_builder,
        current_precision);

    all_grad_h0[2 * l] = ExtractLayer2D(bwd.grad_h0, 0, batch, hidden);
    all_grad_h0[2 * l + 1] = ExtractLayer2D(bwd.grad_h0, 1, batch, hidden);

    const size_t p_out_fwd = l * params_per_layer;
    const size_t p_out_rev = p_out_fwd + params_per_direction;

    all_grad_params[p_out_fwd + 0] = bwd.grad_w_ih_fwd;
    all_grad_params[p_out_fwd + 1] = bwd.grad_w_hh_fwd;
    if (has_biases) {
      all_grad_params[p_out_fwd + 2] = *bwd.grad_b_ih_fwd;
      all_grad_params[p_out_fwd + 3] = *bwd.grad_b_hh_fwd;
    }

    all_grad_params[p_out_rev + 0] = bwd.grad_w_ih_rev;
    all_grad_params[p_out_rev + 1] = bwd.grad_w_hh_rev;
    if (has_biases) {
      all_grad_params[p_out_rev + 2] = *bwd.grad_b_ih_rev;
      all_grad_params[p_out_rev + 3] = *bwd.grad_b_hh_rev;
    }

    curr_grad_y = bwd.grad_x;
  }
}

// Formats final input gradients, initial hidden gradients, and parameter
// gradients.
template <typename ToOutFn>
inline mlir::SmallVector<mlir::MlirOp> FormatGruBackwardOutputs(
    mlir::MlirBuilder& builder, mlir::MlirOp curr_grad_y,
    absl::Span<const mlir::MlirOp> all_grad_h0,
    absl::Span<const mlir::MlirOp> all_grad_params, int64_t batch,
    int64_t hidden, ToOutFn to_out) {
  std::vector<mlir::MlirOp> grad_h0_3d;
  grad_h0_3d.reserve(all_grad_h0.size());
  for (const auto& gh : all_grad_h0) {
    grad_h0_3d.push_back(
        mlir::stablehlo::Reshape(to_out(gh), {1, batch, hidden}));
  }
  mlir::MlirOp final_grad_h0 = ConcatDim(builder, grad_h0_3d, /*dim=*/0);

  mlir::SmallVector<mlir::MlirOp> outputs;
  outputs.reserve(2 + all_grad_params.size());
  outputs.push_back(to_out(curr_grad_y));
  outputs.push_back(final_grad_h0);
  for (const auto& gp : all_grad_params) {
    outputs.push_back(gp);
  }
  return outputs;
}

// Builds the complete StableHLO computational graph for multi-layer GRU
// backward pass.
//
// Algorithmic Flow:
// 1. Forward Reconstruction: Reconstructs intermediate layer inputs and outputs
// (or
//    unpacks pre-computed activation caches) layer-by-layer.
// 2. Inter-Layer Dropout Reproduction: Regenerates the deterministic
// pseudo-random
//    dropout masks used during forward evaluation to accurately mask
//    backpropagated gradients.
// 3. Pipelined Wavefront Optimization: For multi-layer unidirectional GRUs
// (num_layers > 1,
//    seq_len >= kGruUnrollFactor), dispatches to
//    BuildGruPipelinedWavefrontBackward to stream chunk gradients across layers
//    in TPU vector registers.
// 4. Layer-by-Layer Backward BPTT: Cascades backwards from layer l = num_layers
// - 1
//    down to 0:
//    - Evaluates ComputeLayerBackward (unidirectional) or
//    ComputeBidirLayerBackward
//      (bidirectional).
//    - Computes gradients with respect to weights (W_ih, W_hh) and biases
//    (b_ih, b_hh).
//    - Propagates gradients through inter-layer dropout backward.
// 5. Output Packing: Stacks initial hidden state gradients [num_layers *
// num_directions,
//    batch, hidden] and flattens parameter gradients into PyTorch canonical
//    order.
//
// Arguments:
//   builder_inputs: Array containing incoming sequence gradient (grad_y),
//   incoming
//     final hidden state gradient (grad_hy), input sequence, initial states,
//     parameters, and cached activations.
//   builder: MLIR StableHLO graph builder.
//   batch, seq_len, hidden, input_size: Tensor dimension sizes.
//   has_biases: Whether bias parameters are enabled.
//   num_layers: Number of recurrent layers.
//   dropout: Dropout probability.
//   bidirectional: Whether bidirectional recurrence is enabled.
//   num_directions: 2 for bidirectional, 1 for unidirectional.
//   batch_first: Whether sequence layout is [batch, seq, feature].
//   out_dtype, acc_dtype: Element datatypes.
//   params_per_layer, params_per_direction: Parameter counts.
//   has_dropout: Whether dropout was applied.
//   current_precision: Precision configuration for StableHLO matrix
//   multiplications.
//
// Returns:
//   Vector of MLIR output operations: [grad_input, grad_hx, grad_params...].
absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildGruBackwardShloGraph(
    absl::Span<mlir::MlirOp> builder_inputs, mlir::MlirBuilder& builder,
    const int64_t batch, const int64_t seq_len, const int64_t hidden,
    const int64_t input_size, const bool has_biases, const int64_t num_layers,
    const double dropout, const bool bidirectional,
    const int64_t num_directions, const bool batch_first,
    const mlir::ElementType out_dtype, const mlir::ElementType acc_dtype,
    const size_t params_per_layer, const size_t params_per_direction,
    const bool has_dropout,
    const mlir::stablehlo::Precision current_precision) {
  auto to_acc = [acc_dtype, out_dtype](mlir::MlirOp op) -> mlir::MlirOp {
    return (acc_dtype == out_dtype)
               ? op
               : mlir::stablehlo::ConvertElementType(op, acc_dtype);
  };
  auto to_out = [out_dtype, acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
    return (acc_dtype == out_dtype)
               ? op
               : mlir::stablehlo::ConvertElementType(op, out_dtype);
  };

  mlir::MlirOp incoming_grad_y = to_acc(builder_inputs[0]);
  mlir::MlirOp incoming_grad_hy = to_acc(builder_inputs[1]);
  mlir::MlirOp input_seq = builder_inputs[2];
  mlir::MlirOp hx_val = to_acc(builder_inputs[3]);

  const size_t params_offset = 4;
  const size_t cached_acts_offset =
      params_offset + (num_layers * params_per_layer);

  mlir::MLIRContext& ctx = builder.getContext();
  mlir::MlirOp zero_const = MakeScalarConstant(builder, 0.0f, acc_dtype);

  const mlir::Type acc_type = mlir::getElementType(ctx, acc_dtype);
  auto sum_reduce_builder = [acc_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        acc_type, rb.getRegion(), rb.getOpBuilder());
  };

  std::optional<mlir::MlirOp> rand_op = std::nullopt;
  if (has_dropout) {
    mlir::MlirOp rng_input_state = builder_inputs.back();
    const int64_t inter_layers = num_layers - 1;
    const int64_t layer_out_dim = num_directions * hidden;
    Dimensions rand_shape =
        batch_first ? Dimensions{inter_layers, batch, seq_len, layer_out_dim}
                    : Dimensions{inter_layers, seq_len, batch, layer_out_dim};
    TT_ASSIGN_OR_RETURN(rand_op, BuildUniformShlo(rng_input_state, 0.0, 1.0,
                                                  rand_shape, out_dtype));
  }

  const auto ih_dot_dims = MakeDotDims(&ctx, {2}, {1});
  const auto hh_dot_dims = MakeDotDims(&ctx, {1}, {1});

  GruReconstructedForwardData fwd_data = ReconstructGruForwardPass(
      builder, builder_inputs, input_seq, hx_val, rand_op, params_offset,
      cached_acts_offset, params_per_layer, params_per_direction, num_layers,
      num_directions, batch, seq_len, hidden, batch_first, bidirectional,
      has_biases, has_dropout, dropout, ih_dot_dims, hh_dot_dims, acc_dtype,
      to_out, to_acc, current_precision);

  if (num_layers > 1 && !bidirectional && seq_len >= kGruUnrollFactor) {
    return DispatchGruWavefrontBackwardPath(
        builder, builder_inputs, hx_val, incoming_grad_y, incoming_grad_hy,
        fwd_data, params_offset, params_per_layer, num_layers, batch, seq_len,
        hidden, input_size, batch_first, has_biases, has_dropout, dropout,
        zero_const, sum_reduce_builder, to_out, current_precision);
  }

  mlir::MlirOp curr_grad_y = incoming_grad_y;
  std::vector<mlir::MlirOp> all_grad_h0(num_layers * num_directions);
  std::vector<mlir::MlirOp> all_grad_params(num_layers * params_per_layer);

  for (int64_t l = num_layers - 1; l >= 0; --l) {
    ComputeLayerByLayerBackwardStep(
        builder, builder_inputs, l, num_layers, num_directions, batch, seq_len,
        hidden, input_size, batch_first, bidirectional, has_biases,
        params_offset, params_per_layer, params_per_direction, hx_val,
        incoming_grad_hy, fwd_data, curr_grad_y, all_grad_h0, all_grad_params,
        zero_const, sum_reduce_builder, to_out, current_precision);

    if (has_dropout && l > 0) {
      curr_grad_y = ApplyDropoutBackward(
          curr_grad_y, fwd_data.dropout_masks[l - 1], dropout);
    }
  }

  return FormatGruBackwardOutputs(builder, curr_grad_y, all_grad_h0,
                                  all_grad_params, batch, hidden, to_out);
}

// Internal implementation helper that validates inputs and dispatches the
// backward StableHLO graph construction to the TPU execution runtime.
//
// Arguments:
//   grad_output: Incoming sequence output gradient.
//   grad_hy: Incoming final hidden state gradient.
//   input: Original forward input sequence tensor.
//   hx: Initial hidden states tensor.
//   params: List of layer weight and bias parameter tensors.
//   cached_activations: Forward gate pre-activations saved for autograd.
//   has_biases: Whether biases are included.
//   num_layers: Number of recurrent layers.
//   dropout: Dropout probability.
//   train: Whether execution was in training mode.
//   bidirectional: Whether layer is bidirectional.
//   batch_first: Whether sequence layout is [batch, seq, feature].
//   param_keys: Kernel compilation cache keys.
//
// Returns:
//   Vector of device buffer references containing input, hidden, and parameter
//   gradients.
absl::StatusOr<std::vector<DeviceBufferRef>> GruInputBackwardImpl(
    const at::Tensor& grad_output, const at::Tensor& grad_hy,
    const at::Tensor& input, const at::Tensor& hx, const at::TensorList params,
    const at::TensorList cached_activations, const bool has_biases,
    const int64_t num_layers, const double dropout, const bool train,
    const bool bidirectional, const bool batch_first,
    OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateGruInputs(input, hx, params, has_biases,
                                       num_layers, dropout, train,
                                       bidirectional, batch_first));

  const auto current_precision = GetAndAddPrecisionTo(param_keys);

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=enforced by PyTorch autograd engine
      grad_hy.sizes() == hx.sizes(), error::kInvalidArgument)
      << "grad_hy sizes (" << ToString(grad_hy.sizes())
      << ") must match hx sizes (" << ToString(hx.sizes()) << ")";

  const bool has_dropout = (dropout > 0.0 && train && num_layers > 1);
  if (has_dropout) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=enforced by autograd forward state
        !cached_activations.empty(), error::kInvalidArgument)
        << "gru backward with dropout requires cached forward activations";
  }

  const int64_t batch = batch_first ? input.size(0) : input.size(1);
  const int64_t seq_len = batch_first ? input.size(1) : input.size(0);
  const int64_t input_size = input.size(2);
  const int64_t hidden = hx.size(2);
  const int64_t num_directions = bidirectional ? 2 : 1;
  const size_t params_per_direction = has_biases ? 4 : 2;
  const size_t params_per_layer = params_per_direction * num_directions;

  std::vector<at::Tensor> input_tensors;
  input_tensors.reserve(4 + params.size() + cached_activations.size());
  input_tensors.push_back(grad_output);
  input_tensors.push_back(grad_hy);
  input_tensors.push_back(input);
  input_tensors.push_back(hx);
  for (const at::Tensor& param : params) {
    input_tensors.push_back(param);
  }
  for (const at::Tensor& act : cached_activations) {
    input_tensors.push_back(act);
  }

  const size_t total_outputs = 2 + params.size();
  std::vector<mlir::ElementType> out_dtypes;
  out_dtypes.reserve(total_outputs);
  std::vector<absl::Span<const int64_t>> out_dims_spans;
  out_dims_spans.reserve(total_outputs);

  TT_ASSIGN_OR_RETURN(auto in_dtype,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));
  const auto out_dtype = in_dtype;
  TT_ASSIGN_OR_RETURN(
      const auto acc_dtype,
      ConvertTo<mlir::ElementType>(ToAccumulateType(input.scalar_type())));

  out_dtypes.push_back(out_dtype);
  out_dims_spans.push_back(input.sizes());
  out_dtypes.push_back(out_dtype);
  out_dims_spans.push_back(hx.sizes());

  for (const at::Tensor& param : params) {
    out_dtypes.push_back(out_dtype);
    out_dims_spans.push_back(param.sizes());
  }

  auto op_builder =
      [batch, seq_len, hidden, input_size, has_biases, num_layers, dropout,
       bidirectional, num_directions, batch_first, out_dtype, acc_dtype,
       params_per_layer, params_per_direction, has_dropout, current_precision](
          absl::Span<mlir::MlirOp> builder_inputs, mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    return BuildGruBackwardShloGraph(
        builder_inputs, builder, batch, seq_len, hidden, input_size, has_biases,
        num_layers, dropout, bidirectional, num_directions, batch_first,
        out_dtype, acc_dtype, params_per_layer, params_per_direction,
        has_dropout, current_precision);
  };

  absl::Span<const mlir::ElementType> out_dtypes_span = out_dtypes;
  absl::Span<const absl::Span<const int64_t>> out_dims_list = out_dims_spans;

  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = out_dtypes_span,
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = std::move(param_keys),
  };

  return DispatchOp<kDynamicSize, kDynamicSize>(
      std::move(op_builder), input_tensors, std::move(options));
}

// ATen entry point for GRU forward pass without autograd activation caching.
//
// Arguments:
//   input: Input sequence tensor [T, B, in_dim] or [B, T, in_dim].
//   hx: Initial hidden states [num_layers * num_directions, B, hidden].
//   params: Flat list of weight and bias tensors.
//   has_biases: Whether biases are enabled.
//   num_layers: Number of stacked layers.
//   dropout: Inter-layer dropout probability.
//   train: Whether running in training mode.
//   bidirectional: Whether bidirectional recurrence is enabled.
//   batch_first: Whether batch is the first dimension.
//
// Returns:
//   Tuple of (output_sequence, final_hidden_state).
std::tuple<at::Tensor, at::Tensor> AtenGruInputForward(
    const at::Tensor& input, const at::Tensor& hx, const at::TensorList params,
    const bool has_biases, const int64_t num_layers, const double dropout,
    const bool train, const bool bidirectional, const bool batch_first) {
  const bool has_dropout = (dropout > 0.0 && train && num_layers > 1);
  if (has_dropout) {
    const int64_t batch = batch_first ? input.size(0) : input.size(1);
    const int64_t seq_len = batch_first ? input.size(1) : input.size(0);
    const int64_t hidden = hx.size(2);
    const int64_t num_directions = bidirectional ? 2 : 1;
    const int64_t num_elements =
        (num_layers - 1) * seq_len * batch * (num_directions * hidden);
    TT_ASSIGN_OR_THROW(const mlir::ElementType in_dtype,
                       ConvertTo<mlir::ElementType>(input.scalar_type()));
    const int64_t bit_width = TorchEquivalentBitwidth(in_dtype);
    TT_KERNEL(
        OpName::kGruInput, param_keys,
        (input, hx, params, has_biases, num_layers, dropout, train,
         bidirectional, batch_first),
        {
          TT_ASSIGN_OR_THROW(
              const std::vector<DeviceBufferRef> result_buffers,
              DispatchRngOpGeneral(
                  /*generator=*/std::nullopt,
                  [&](at::Tensor rng_state)
                      -> absl::StatusOr<std::vector<DeviceBufferRef>> {
                    return GruInputImpl(input, hx, params, has_biases,
                                        num_layers, dropout, train,
                                        bidirectional, batch_first,
                                        std::move(param_keys),
                                        /*cache_activations=*/false, rng_state);
                  },
                  RngUsage{num_elements, bit_width}));
          return {MakeTensor(result_buffers[0]), MakeTensor(result_buffers[1])};
        });
  } else {
    TT_KERNEL(
        OpName::kGruInput, param_keys,
        (input, hx, params, has_biases, num_layers, dropout, train,
         bidirectional, batch_first),
        {
          TT_ASSIGN_OR_THROW(
              const std::vector<DeviceBufferRef> result_buffers,
              GruInputImpl(input, hx, params, has_biases, num_layers, dropout,
                           train, bidirectional, batch_first,
                           std::move(param_keys), /*cache_activations=*/false,
                           /*rng_state=*/std::nullopt));
          return {MakeTensor(result_buffers[0]), MakeTensor(result_buffers[1])};
        });
  }
}

// ATen entry point for GRU forward pass with activation caching for autograd
// backward.
//
// Arguments:
//   input, hx, params: Model inputs and parameters.
//   has_biases, num_layers, dropout, train, bidirectional, batch_first: Config
//   flags.
//
// Returns:
//   Tuple of (output_sequence, final_hidden_state, cached_activation_tensors).
std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>>
AtenGruInputForwardCached(const at::Tensor& input, const at::Tensor& hx,
                          const at::TensorList params, const bool has_biases,
                          const int64_t num_layers, const double dropout,
                          const bool train, const bool bidirectional,
                          const bool batch_first) {
  const bool has_dropout = (dropout > 0.0 && train && num_layers > 1);
  if (has_dropout) {
    const int64_t batch = batch_first ? input.size(0) : input.size(1);
    const int64_t seq_len = batch_first ? input.size(1) : input.size(0);
    const int64_t hidden = hx.size(2);
    const int64_t num_directions = bidirectional ? 2 : 1;
    const int64_t num_elements =
        (num_layers - 1) * seq_len * batch * (num_directions * hidden);
    TT_ASSIGN_OR_THROW(const mlir::ElementType in_dtype,
                       ConvertTo<mlir::ElementType>(input.scalar_type()));
    const int64_t bit_width = TorchEquivalentBitwidth(in_dtype);
    TT_KERNEL(
        OpName::kGruInputForwardCached, param_keys,
        (input, hx, params, has_biases, num_layers, dropout, train,
         bidirectional, batch_first),
        {
          std::optional<at::Tensor> saved_rng_state;
          TT_ASSIGN_OR_THROW(const std::vector<DeviceBufferRef> result_buffers,
                             DispatchRngOpGeneral(
                                 /*generator=*/std::nullopt,
                                 [&](at::Tensor rng_state) {
                                   saved_rng_state = rng_state;
                                   return GruInputImpl(
                                       input, hx, params, has_biases,
                                       num_layers, dropout, train,
                                       bidirectional, batch_first,
                                       std::move(param_keys),
                                       /*cache_activations=*/true, rng_state);
                                 },
                                 RngUsage{num_elements, bit_width}));
          at::Tensor output = MakeTensor(result_buffers[0]);
          at::Tensor hy = MakeTensor(result_buffers[1]);
          std::vector<at::Tensor> cached_acts;
          cached_acts.reserve(result_buffers.size() - 2 + 1);
          for (size_t i = 2; i < result_buffers.size(); ++i) {
            cached_acts.push_back(MakeTensor(result_buffers[i]));
          }
          if (saved_rng_state.has_value()) {
            cached_acts.push_back(std::move(*saved_rng_state));
          }
          return {std::move(output), std::move(hy), std::move(cached_acts)};
        });
  } else {
    TT_KERNEL(OpName::kGruInputForwardCached, param_keys,
              (input, hx, params, has_biases, num_layers, dropout, train,
               bidirectional, batch_first),
              {
                TT_ASSIGN_OR_THROW(
                    const std::vector<DeviceBufferRef> result_buffers,
                    GruInputImpl(input, hx, params, has_biases, num_layers,
                                 dropout, train, bidirectional, batch_first,
                                 std::move(param_keys),
                                 /*cache_activations=*/true,
                                 /*rng_state=*/std::nullopt));
                at::Tensor output = MakeTensor(result_buffers[0]);
                at::Tensor hy = MakeTensor(result_buffers[1]);
                std::vector<at::Tensor> cached_acts;
                cached_acts.reserve(result_buffers.size() - 2);
                for (size_t i = 2; i < result_buffers.size(); ++i) {
                  cached_acts.push_back(MakeTensor(result_buffers[i]));
                }
                return {std::move(output), std::move(hy),
                        std::move(cached_acts)};
              });
  }
}

// ATen entry point for GRU backward pass.
//
// Arguments:
//   grad_output: Sequence output gradients.
//   grad_hy: Final hidden state gradients.
//   input, hx, params: Original forward input and parameters.
//   has_biases, num_layers, dropout, train, bidirectional, batch_first: Config.
//   cached_activations: Saved activation tensors from forward execution.
//
// Returns:
//   Tuple of (grad_input, grad_hx, grad_params).
std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>>
AtenGruInputBackward(const at::Tensor& grad_output, const at::Tensor& grad_hy,
                     const at::Tensor& input, const at::Tensor& hx,
                     const at::TensorList params, const bool has_biases,
                     const int64_t num_layers, const double dropout,
                     const bool train, const bool bidirectional,
                     const bool batch_first,
                     const at::TensorList cached_activations) {
  TT_KERNEL(
      OpName::kGruInputBackward, param_keys,
      (grad_output, grad_hy, input, hx, params, has_biases, num_layers, dropout,
       train, bidirectional, batch_first, cached_activations),
      {
        TT_ASSIGN_OR_THROW(
            const std::vector<DeviceBufferRef> result_buffers,
            GruInputBackwardImpl(grad_output, grad_hy, input, hx, params,
                                 cached_activations, has_biases, num_layers,
                                 dropout, train, bidirectional, batch_first,
                                 std::move(param_keys)));
        at::Tensor grad_input = MakeTensor(result_buffers[0]);
        at::Tensor grad_hx = MakeTensor(result_buffers[1]);
        std::vector<at::Tensor> grad_params;
        grad_params.reserve(result_buffers.size() - 2);
        for (size_t i = 2; i < result_buffers.size(); ++i) {
          grad_params.push_back(MakeTensor(result_buffers[i]));
        }
        return {std::move(grad_input), std::move(grad_hx),
                std::move(grad_params)};
      });
}

// PyTorch Autograd Function integration for GRU.
// Connects the forward cached kernel and backward kernel into PyTorch's
// automatic differentiation engine.
struct AtenGruInputAutograd
    : public torch::autograd::Function<AtenGruInputAutograd> {
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx, const at::Tensor& input,
      const at::Tensor& hx, at::TensorList params, bool has_biases,
      int64_t num_layers, double dropout, bool train, bool bidirectional,
      bool batch_first);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs);
};

// Autograd forward implementation: executes cached forward pass and saves
// variables and attributes for the backward pass.
torch::autograd::variable_list AtenGruInputAutograd::forward(
    torch::autograd::AutogradContext* ctx, const at::Tensor& input,
    const at::Tensor& hx, const at::TensorList params, const bool has_biases,
    const int64_t num_layers, const double dropout, const bool train,
    const bool bidirectional, const bool batch_first) {
  auto [output, hy, cached_acts] =
      AtenGruInputForwardCached(input, hx, params, has_biases, num_layers,
                                dropout, train, bidirectional, batch_first);

  std::vector<at::Tensor> to_save;
  to_save.reserve(2 + params.size() + cached_acts.size());
  to_save.push_back(input);
  to_save.push_back(hx);
  for (const at::Tensor& p : params) {
    to_save.push_back(p);
  }
  for (const at::Tensor& a : cached_acts) {
    to_save.push_back(a);
  }
  ctx->save_for_backward(to_save);

  ctx->saved_data["has_biases"] = has_biases;
  ctx->saved_data["num_layers"] = num_layers;
  ctx->saved_data["dropout"] = dropout;
  ctx->saved_data["train"] = train;
  ctx->saved_data["bidirectional"] = bidirectional;
  ctx->saved_data["batch_first"] = batch_first;
  ctx->saved_data["params_count"] = static_cast<int64_t>(params.size());

  return {output, hy};
}

// Autograd backward implementation: unpacks saved variables, calls
// AtenGruInputBackward, and maps returned gradients back to autograd variable
// list.
torch::autograd::variable_list AtenGruInputAutograd::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
  const auto saved = ctx->get_saved_variables();
  const at::Tensor& input = saved[0];
  const at::Tensor& hx = saved[1];
  const int64_t params_count = ctx->saved_data["params_count"].toInt();
  std::vector<at::Tensor> params(saved.begin() + 2,
                                 saved.begin() + 2 + params_count);
  std::vector<at::Tensor> cached_acts(saved.begin() + 2 + params_count,
                                      saved.end());

  const bool has_biases = ctx->saved_data["has_biases"].toBool();
  const int64_t num_layers = ctx->saved_data["num_layers"].toInt();
  const double dropout = ctx->saved_data["dropout"].toDouble();
  const bool train = ctx->saved_data["train"].toBool();
  const bool bidirectional = ctx->saved_data["bidirectional"].toBool();
  const bool batch_first = ctx->saved_data["batch_first"].toBool();

  const int64_t batch = batch_first ? input.size(0) : input.size(1);
  const int64_t seq_len = batch_first ? input.size(1) : input.size(0);
  const int64_t hidden = hx.size(2);
  const int64_t num_directions = bidirectional ? 2 : 1;

  at::Tensor grad_output = grad_outputs[0];
  if (!grad_output.defined()) {
    Dimensions out_shape =
        batch_first ? Dimensions{batch, seq_len, hidden * num_directions}
                    : Dimensions{seq_len, batch, hidden * num_directions};
    grad_output = at::zeros(out_shape, input.options());
  }

  at::Tensor grad_hy = grad_outputs[1];
  if (!grad_hy.defined()) {
    grad_hy = at::zeros({num_layers * num_directions, batch, hidden},
                        input.options());
  }

  auto [grad_input, grad_hx, grad_params] = AtenGruInputBackward(
      grad_output, grad_hy, input, hx, params, has_biases, num_layers, dropout,
      train, bidirectional, batch_first, cached_acts);

  torch::autograd::variable_list result;
  result.reserve(2 + params_count + 6);
  result.push_back(std::move(grad_input));
  result.push_back(std::move(grad_hx));
  for (at::Tensor& p_grad : grad_params) {
    result.push_back(std::move(p_grad));
  }
  result.push_back(at::Tensor());  // has_biases
  result.push_back(at::Tensor());  // num_layers
  result.push_back(at::Tensor());  // dropout
  result.push_back(at::Tensor());  // train
  result.push_back(at::Tensor());  // bidirectional
  result.push_back(at::Tensor());  // batch_first

  return result;
}

}  // namespace

// Top-level public ATen operator for GRU execution on TPU.
// Automatically routes between autograd-aware differentiation (when gradients
// are required) and optimized direct inference.
//
// Arguments:
//   input: Input sequence tensor.
//   hx: Initial hidden states.
//   params: List of layer weight and bias parameters.
//   has_biases: Whether bias parameters are used.
//   num_layers: Number of stacked layers.
//   dropout: Inter-layer dropout probability.
//   train: Whether running in training mode.
//   bidirectional: Whether layer is bidirectional.
//   batch_first: Whether sequence layout is [batch, seq, feature].
//
// Returns:
//   Tuple of (output_sequence, final_hidden_state).
std::tuple<at::Tensor, at::Tensor> AtenGruInput(
    const at::Tensor& input, const at::Tensor& hx, const at::TensorList params,
    const bool has_biases, const int64_t num_layers, const double dropout,
    const bool train, const bool bidirectional, const bool batch_first) {
  bool requires_grad = input.requires_grad() || hx.requires_grad();
  if (!requires_grad) {
    for (const at::Tensor& p : params) {
      if (p.requires_grad()) {
        requires_grad = true;
        break;
      }
    }
  }

  if (c10::GradMode::is_enabled() && requires_grad) {
    auto results =
        AtenGruInputAutograd::apply(input, hx, params, has_biases, num_layers,
                                    dropout, train, bidirectional, batch_first);
    return {results[0], results[1]};
  }

  return AtenGruInputForward(input, hx, params, has_biases, num_layers, dropout,
                             train, bidirectional, batch_first);
}

}  // namespace torch_tpu
