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

#include "csrc/ops/lstm/lstm_aten_kernels.h"

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
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_gen_impl.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/dropout/dropout.h"
#include "csrc/ops/lstm/lstm_common.h"
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
#include "mlir/Support/LLVM.h"  // IWYU pragma: keep for mlir::SmallVector
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/autograd/custom_function.h"  // IWYU pragma: keep for torch::autograd::variable_list

// =============================================================================
// StableHLO LSTM Forward & Backward Kernels for TorchTPU
// =============================================================================
//
// 1. MATHEMATICAL FORMULATION:
// -----------------------------
// For each layer l in [0, num_layers) and each timestep t in [0, seq_len):
//
//   Gates projection:
//     gates_t = x_t @ W_ih^T + b_ih + h_{t-1} @ W_hh^T + b_hh
//
//   Unpacking the 4 concatenated gates [4*H]:
//     i_t = sigmoid(gates_t[0 : H])           (Input gate: how much new info to
//     store) f_t = sigmoid(gates_t[H : 2*H])         (Forget gate: how much old
//     memory to keep) g_t = tanh(gates_t[2*H : 3*H])          (Cell gate:
//     candidate memory content) o_t = sigmoid(gates_t[3*H : 4*H])       (Output
//     gate: how much cell memory to expose)
//
//   Cell state and hidden state updates:
//     c_t = f_t * c_{t-1} + i_t * g_t         (Elementwise cell recurrence)
//     h_t = o_t * tanh(c_t)                   (Hidden activation output)
//
// 2. FORWARD PASS ARCHITECTURAL OPTIMIZATIONS:
// --------------------------------------------
// - Batched Input Projection Upfront:
//   Instead of executing seq_len separate matrix multiplications (x_t @ W_ih^T)
//   at every recurrent step, we multiply the entire sequence tensor by W_ih^T
//   upfront in a single 3D batched GEMM:
//     [T, B, in_dim] x [4*H, in_dim]^T -> [T, B, 4*H] (or [B, T, in_dim] -> [B,
//     T, 4*H])
//   This converts seq_len small memory-bound GEMMs into one large compute-dense
//   GEMM.
//
// - 1D Bias Pre-Combining:
//   PyTorch provides separate biases b_ih and b_hh of shape [4*H]. Adding them
//   first in 1D (b_total = b_ih + b_hh) before broadcasting across [T, B, 4*H]
//   cuts broadcast and 3D elementwise addition memory traffic in half.
//
// - Zero-Copy Direct Batch-First Support:
//   For batch_first=True ([B, T, D]), we directly project across dimension 2,
//   slice along dimension 1 during recurrence, and concatenate step outputs
//   along dimension 1. This completely eliminates expensive 3D memory
//   transpositions on input and output.
//
// - Precision Upcasting via ToAccumulateType:
//   For low-precision inputs (e.g. bfloat16), internal recurrent accumulations
//   are evaluated in float32 (ToAccumulateType) and converted back to out_dtype
//   at layer boundaries, preventing numerical drift and underflow over long
//   sequences.
//
// - Chunked Recurrent Loop with Partial Unrolling (kLstmUnrollFactor):
//   Rather than unrolling all T recurrent steps into a massive MLIR graph (O(T)
//   IR size), an outer stablehlo.while loop steps through sequence chunks.
//   Inside each chunk, exactly kLstmUnrollFactor steps are statically unrolled
//   in vector registers:
//     * Slicing inputs is performed via DynamicSlice in a single burst of
//       kLstmUnrollFactor timesteps.
//     * kLstmUnrollFactor steps execute in vector registers with zero loop
//     overhead.
//     * kLstmUnrollFactor step outputs are written to the output buffer via
//     DynamicUpdateSlice.
//   Sequences shorter than kLstmUnrollFactor bypass the while loop entirely and
//   execute via direct static unrolling. Remainder steps (T % kLstmUnrollFactor
//   != 0) are handled by static unrolling and concatenated.
//
// 3. BACKWARD PASS (BPTT) ARCHITECTURAL DESIGN:
// ---------------------------------------------
// - Layer-by-Layer Recomputation:
//   During backpropagation through time, forward gate activations (i, f, g, o)
//   and cell states are recomputed layer-by-layer rather than cached during
//   forward, achieving O(T) compute with minimal accelerator HBM activation
//   memory footprint.
//
// - Reverse Recurrent Gradient Propagation:
//   Time steps are evaluated in reverse order (t = T-1 down to 0). At each
//   step:
//     delta_h_t = grad_output_t + delta_h_{rec, t}
//     delta_c_t = delta_c_{next} + delta_h_t * o_t * (1 - tanh^2(c_t))
//     delta_c_{next} = delta_c_t * f_t
//     Gate pre-activation adjoints delta_pre_{i,f,g,o} are computed using
//     analytic derivatives of sigmoid and tanh.
//
// - Batched Parameter Gradient Contractions:
//   After gathering pre-activation adjoints delta_pre across time into [T*B,
//   4*H], parameter gradients (grad_w_ih, grad_w_hh) are computed via single
//   batched outer product GEMMs against the flattened activations [T*B, in_dim]
//   and [T*B, H].
//
// 4. PYTORCH AUTOGRAD INTEGRATION:
// --------------------------------
// In PyTorch core, aten::lstm.input does not have an entry in derivatives.yaml
// and has no autograd backward kernel schema (such as
// aten::lstm_input_backward). Therefore, when requires_grad is enabled,
// AtenLstmInput intercepts execution and dispatches via
// torch::autograd::Function (AtenLstmInputAutograd), which directly invokes
// AtenLstmInputBackward during the autograd backward phase.
// =============================================================================

namespace torch_tpu {
namespace {
// NOLINTBEGIN(readability-function-cognitive-complexity)

// Static unrolling factor kLstmUnrollFactor for chunked recurrent loop
// execution.
//
// Rationale & Architecture:
// - Full static unrolling of all T recurrent steps produces O(T) IR operations
//   in the StableHLO graph. For long sequences (e.g. T >= 512), this leads to
//   excessive compilation latency, huge MLIR bytecode, and high compiler memory
//   pressure.
// - Conversely, a fully dynamic loop stepping 1 timestep at a time introduces
//   per-step loop boundary overhead, scalar condition evaluations, and frequent
//   DynamicUpdateSlice writebacks.
// - A partial unroll factor of kLstmUnrollFactor strikes the optimal trade-off:
//   1. kLstmUnrollFactor steps easily fit within TPU VPU vector register
//   residency without
//      register spilling.
//   2. Loop iterations are reduced by a factor of kLstmUnrollFactor.
//   3. Slicing inputs and updating outputs happen in kLstmUnrollFactor-step
//   bursts,
//      significantly amortizing DynamicSlice / DynamicUpdateSlice memory
//      overhead.
//   4. Short sequences (seq_len < kLstmUnrollFactor) bypass the while loop
//   entirely
//      and execute via direct static unrolling.
//   5. Any remainder steps (seq_len % kLstmUnrollFactor != 0) are handled by
//   static
//      unrolling and concatenated.
constexpr int64_t kLstmUnrollFactor = 8;

// Validates that recurrent initial hidden state (h_0) and cell state (c_0)
// match expected 3D rank and shapes:
// - c_0: [num_layers * num_directions, batch, hidden_size]
// - h_0: [num_layers * num_directions, batch, out_size] where out_size is
//   proj_size if proj_size > 0, else hidden_size.
absl::Status ValidateLstmTensorShapes(const at::TensorList hx,
                                      const int64_t num_layers,
                                      const int64_t batch,
                                      const bool bidirectional = false) {
  const int64_t num_directions = bidirectional ? 2 : 1;
  const at::Tensor& h_0 = hx[0];
  const at::Tensor& c_0 = hx[1];
  TT_RET_CHECK(h_0.dim() == 3, error::kInvalidArgument)
      << "expected h_0 to be a 3D tensor [num_layers * num_directions, batch, "
         "out_size], got "
      << h_0.dim() << "D";
  TT_RET_CHECK(c_0.dim() == 3, error::kInvalidArgument)
      << "expected c_0 to be a 3D tensor [num_layers * num_directions, batch, "
         "hidden_size], got "
      << c_0.dim() << "D";
  TT_RET_CHECK(h_0.size(0) == num_layers * num_directions,
               error::kInvalidArgument)
      << "expected h_0 size(0) to match num_layers * num_directions ("
      << num_layers * num_directions << "), got " << h_0.size(0);
  TT_RET_CHECK(h_0.size(1) == batch, error::kInvalidArgument)
      << "expected h_0 size(1) to match batch (" << batch << "), got "
      << h_0.size(1);
  TT_RET_CHECK(c_0.size(0) == num_layers * num_directions,
               error::kInvalidArgument)
      << "expected c_0 size(0) to match num_layers * num_directions ("
      << num_layers * num_directions << "), got " << c_0.size(0);
  TT_RET_CHECK(c_0.size(1) == batch, error::kInvalidArgument)
      << "expected c_0 size(1) to match batch (" << batch << "), got "
      << c_0.size(1);
  TT_RET_CHECK(h_0.size(2) <= c_0.size(2), error::kInvalidArgument)
      << "h_0 size(2) (" << h_0.size(2) << ") cannot exceed c_0 size(2) ("
      << c_0.size(2) << ")";
  return absl::OkStatus();
}

// Validates that weights and biases conform to PyTorch's parameter layout:
// - Without proj_size:
//   - Without bias (2 tensors/layer/direction): {W_ih_l, W_hh_l}
//   - With bias (4 tensors/layer/direction):    {W_ih_l, W_hh_l, b_ih_l,
//   b_hh_l}
// - With proj_size (proj_size > 0):
//   - Without bias (3 tensors/layer/direction): {W_ih_l, W_hh_l, W_hr_l}
//   - With bias (5 tensors/layer/direction):    {W_ih_l, W_hh_l, b_ih_l,
//   b_hh_l, W_hr_l}
// Dimensionality expectations:
// - Layer 0: W_ih has shape [4*H, input_size]
// - Layer >0: W_ih has shape [4*H, out_h * num_directions]
// - All layers: W_hh has shape [4*H, out_h], biases have shape [4*H]
// - When proj_size > 0: W_hr has shape [proj_size, H]
absl::Status ValidateLstmWeights(const at::TensorList params,
                                 const bool has_biases,
                                 const int64_t num_layers,
                                 const int64_t input_size, const int64_t hidden,
                                 const int64_t proj_size = 0,
                                 const bool bidirectional = false) {
  const int64_t num_directions = bidirectional ? 2 : 1;
  const size_t params_per_direction =
      has_biases ? (proj_size > 0 ? 5 : 4) : (proj_size > 0 ? 3 : 2);
  const size_t params_per_layer = params_per_direction * num_directions;
  const int64_t out_h = (proj_size > 0) ? proj_size : hidden;
  TT_RET_CHECK(
      params.size() == static_cast<size_t>(num_layers * params_per_layer),
      error::kInvalidArgument)
      << "expected " << num_layers * params_per_layer << " parameters, got "
      << params.size();

  for (int64_t l = 0; l < num_layers; ++l) {
    const int64_t in_dim = (l == 0) ? input_size : out_h * num_directions;
    for (int64_t d_dir = 0; d_dir < num_directions; ++d_dir) {
      const size_t p_offset =
          (l * num_directions + d_dir) * params_per_direction;
      TT_RET_CHECK(
          params[p_offset].sizes() == at::IntArrayRef({4 * hidden, in_dim}),
          error::kInvalidArgument)
          << "w_ih layer " << l << " dir " << d_dir << " expected shape ["
          << 4 * hidden << ", " << in_dim << "], got "
          << ToString(params[p_offset].sizes());
      TT_RET_CHECK(
          params[p_offset + 1].sizes() == at::IntArrayRef({4 * hidden, out_h}),
          error::kInvalidArgument)
          << "w_hh layer " << l << " dir " << d_dir << " expected shape ["
          << 4 * hidden << ", " << out_h << "], got "
          << ToString(params[p_offset + 1].sizes());
      if (has_biases) {
        TT_RET_CHECK(
            params[p_offset + 2].sizes() == at::IntArrayRef({4 * hidden}),
            error::kInvalidArgument)
            << "b_ih layer " << l << " dir " << d_dir << " expected shape ["
            << 4 * hidden << "], got "
            << ToString(params[p_offset + 2].sizes());
        TT_RET_CHECK(
            params[p_offset + 3].sizes() == at::IntArrayRef({4 * hidden}),
            error::kInvalidArgument)
            << "b_hh layer " << l << " dir " << d_dir << " expected shape ["
            << 4 * hidden << "], got "
            << ToString(params[p_offset + 3].sizes());
        if (proj_size > 0) {
          TT_RET_CHECK(params[p_offset + 4].sizes() ==
                           at::IntArrayRef({proj_size, hidden}),
                       error::kInvalidArgument)
              << "w_hr layer " << l << " dir " << d_dir << " expected shape ["
              << proj_size << ", " << hidden << "], got "
              << ToString(params[p_offset + 4].sizes());
        }
      } else {
        if (proj_size > 0) {
          TT_RET_CHECK(params[p_offset + 2].sizes() ==
                           at::IntArrayRef({proj_size, hidden}),
                       error::kInvalidArgument)
              << "w_hr layer " << l << " dir " << d_dir << " expected shape ["
              << proj_size << ", " << hidden << "], got "
              << ToString(params[p_offset + 2].sizes());
        }
      }
    }
  }
  return absl::OkStatus();
}

// Master validation routine verifying sequence dimensions, batch sizes,
// supported feature flags (e.g. rejecting unsupported dropout modes),
// and dispatching detailed shape checks for states and weights.
absl::Status ValidateLstmInputs(const at::Tensor& input,
                                const at::TensorList hx,
                                const at::TensorList params,
                                const bool has_biases, const int64_t num_layers,
                                const double dropout, const bool train,
                                const bool bidirectional,
                                const bool batch_first) {
  TT_RET_CHECK(input.dim() == 3, error::kInvalidArgument)
      << "expected input to be a 3D tensor, got " << input.dim() << "D";
  TT_RET_CHECK(hx.size() == 2, error::kInvalidArgument)
      << "expected hx to contain exactly 2 tensors (h_0, c_0), got "
      << hx.size();

  const int64_t batch = batch_first ? input.size(0) : input.size(1);
  const int64_t seq_len = batch_first ? input.size(1) : input.size(0);
  const int64_t input_size = input.size(2);

  TT_RET_CHECK(seq_len > 0, error::kInvalidArgument)
      << "expected sequence length to be larger than 0 in RNN, got " << seq_len;
  TT_RET_CHECK(batch > 0, error::kInvalidArgument)
      << "expected batch size > 0 in RNN, got " << batch;
  TT_RET_CHECK(num_layers > 0, error::kInvalidArgument)
      << "expected num_layers > 0 in RNN, got " << num_layers;
  TT_RET_CHECK(dropout >= 0.0 && dropout <= 1.0, error::kInvalidArgument)
      << "expected dropout to be in range [0, 1], got " << dropout;

  const int64_t hidden = hx[1].dim() == 3 ? hx[1].size(2) : 0;
  const int64_t proj_size =
      (hx[0].dim() == 3 && hx[1].dim() == 3 && hx[0].size(2) < hx[1].size(2))
          ? hx[0].size(2)
          : 0;
  TT_RETURN_IF_ERROR(
      ValidateLstmTensorShapes(hx, num_layers, batch, bidirectional));
  return ValidateLstmWeights(params, has_biases, num_layers, input_size, hidden,
                             proj_size, bidirectional);
}

// Constructs StableHLO DotDimensionNumbersAttr for matrix contractions.
// For standard 2D matrix multiplication A @ B^T (contracting dim 1 with dim 1):
//   lhs_contracting = {1}, rhs_contracting = {1}
// For 3D batched GEMM [T, B, in_dim] @ [4*H, in_dim]^T:
//   lhs_contracting = {2}, rhs_contracting = {1}
mlir::stablehlo::DotDimensionNumbersAttr MakeDotDims(
    mlir::MLIRContext* ctx, llvm::ArrayRef<int64_t> lhs_contracting,
    llvm::ArrayRef<int64_t> rhs_contracting) {
  return mlir::stablehlo::DotDimensionNumbersAttr::get(
      ctx, /*lhs_batching_dimensions=*/{}, /*rhs_batching_dimensions=*/{},
      lhs_contracting, rhs_contracting);
}

// Concatenates a vector of tensors along the specified dimension.
// Short-circuits directly to ops[0] when size == 1 to avoid emitting redundant
// no-op Concatenate operations into the StableHLO graph.
mlir::MlirOp ConcatDim(mlir::MlirBuilder& builder,
                       absl::Span<const mlir::MlirOp> ops, int64_t dim) {
  return (ops.size() == 1) ? ops[0]
                           : mlir::stablehlo::Concatenate(builder, ops, dim);
}

// Extracts a 2D recurrent state slice [B, H] for layer `layer_idx` from the
// full stacked 3D state tensor [num_layers, B, H].
// Performs Slice([layer_idx, 0, 0] to [layer_idx + 1, B, H]) -> Reshape([B,
// H]).
mlir::MlirOp ExtractLayer2D(mlir::MlirOp tensor_3d, int64_t layer_idx,
                            int64_t batch, int64_t hidden) {
  mlir::MlirOp sl = mlir::stablehlo::Slice(
      tensor_3d, {layer_idx, 0, 0}, {layer_idx + 1, batch, hidden}, {1, 1, 1});
  return mlir::stablehlo::Reshape(sl, {batch, hidden});
}

// Slices a 2D timestep [B, feature_dim] at step t from a 3D sequence tensor
// ([T, B, feature_dim] if !batch_first, or [B, T, feature_dim] if batch_first).
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

// Expands a 2D timestep tensor [B, feature_dim] to a 3D singleton sequence
// slice ([B, 1, feature_dim] if batch_first, or [1, B, feature_dim] if
// !batch_first).
inline mlir::MlirOp ExpandStep3D(mlir::MlirOp step_2d, int64_t batch,
                                 int64_t feature_dim, bool batch_first) {
  return batch_first
             ? mlir::stablehlo::Reshape(step_2d, {batch, 1, feature_dim})
             : mlir::stablehlo::Reshape(step_2d, {1, batch, feature_dim});
}

// Encapsulates intermediate and final activation tensors produced during a
// single LSTM recurrent timestep evaluation.
// Saved for state chaining across time in forward and for BPTT in backward.
struct LstmStepResults {
  mlir::MlirOp h_next;  // Next hidden state: h_t = o_t * tanh(c_t) [B, H]
  mlir::MlirOp
      c_next;  // Next cell state: c_t = f_t * c_{t-1} + i_t * g_t [B, H]
  mlir::MlirOp ingate;  // Input gate activation: i_t = sigmoid(i_pre) [B, H]
  mlir::MlirOp
      forgetgate;         // Forget gate activation: f_t = sigmoid(f_pre) [B, H]
  mlir::MlirOp cellgate;  // Cell candidate activation: g_t = tanh(g_pre) [B, H]
  mlir::MlirOp outgate;   // Output gate activation: o_t = sigmoid(o_pre) [B, H]
  mlir::MlirOp tanh_c;    // Non-linear cell projection: tanh(c_t) [B, H]
  mlir::MlirOp ifo_act;   // 3H Gate Clumped activations: [i, f, o] [B, 3*H]
};

// Permutes a 2D weight matrix [4*H, in_dim] from PyTorch [i, f, g, o] order to
// [i, f, o, g] (3H gate clumped) order.
mlir::MlirOp ClumpWeight3H(mlir::MlirBuilder& builder, mlir::MlirOp w,
                           int64_t hidden, int64_t in_dim) {
  // [0..2*H]: i and f
  mlir::MlirOp if_part =
      mlir::stablehlo::Slice(w, {0, 0}, {2 * hidden, in_dim}, {1, 1});
  // [3*H..4*H]: o
  mlir::MlirOp o_part =
      mlir::stablehlo::Slice(w, {3 * hidden, 0}, {4 * hidden, in_dim}, {1, 1});
  // [2*H..3*H]: g
  mlir::MlirOp g_part =
      mlir::stablehlo::Slice(w, {2 * hidden, 0}, {3 * hidden, in_dim}, {1, 1});
  return ConcatDim(builder, {if_part, o_part, g_part}, /*dim=*/0);
}

// Permutes a 1D bias vector [4*H] from PyTorch [i, f, g, o] order to
// [i, f, o, g] (3H gate clumped) order.
mlir::MlirOp ClumpBias3H(mlir::MlirBuilder& builder, mlir::MlirOp b,
                         int64_t hidden) {
  // [0..2*H]: i and f
  mlir::MlirOp if_part = mlir::stablehlo::Slice(b, {0}, {2 * hidden}, {1});
  // [3*H..4*H]: o
  mlir::MlirOp o_part =
      mlir::stablehlo::Slice(b, {3 * hidden}, {4 * hidden}, {1});
  // [2*H..3*H]: g
  mlir::MlirOp g_part =
      mlir::stablehlo::Slice(b, {2 * hidden}, {3 * hidden}, {1});
  return ConcatDim(builder, {if_part, o_part, g_part}, /*dim=*/0);
}

// Permutes a 2D weight gradient matrix [4*H, in_dim] from [i, f, o, g] clumped
// order back to standard PyTorch [i, f, g, o] order.
mlir::MlirOp UnclumpWeight3H(mlir::MlirBuilder& builder, mlir::MlirOp w,
                             int64_t hidden, int64_t in_dim) {
  // [0..2*H]: i and f
  mlir::MlirOp if_part =
      mlir::stablehlo::Slice(w, {0, 0}, {2 * hidden, in_dim}, {1, 1});
  // [2*H..3*H]: o
  mlir::MlirOp o_part =
      mlir::stablehlo::Slice(w, {2 * hidden, 0}, {3 * hidden, in_dim}, {1, 1});
  // [3*H..4*H]: g
  mlir::MlirOp g_part =
      mlir::stablehlo::Slice(w, {3 * hidden, 0}, {4 * hidden, in_dim}, {1, 1});
  return ConcatDim(builder, {if_part, g_part, o_part}, /*dim=*/0);
}

// Permutes a 1D bias gradient vector [4*H] from [i, f, o, g] clumped order
// back to standard PyTorch [i, f, g, o] order.
mlir::MlirOp UnclumpBias3H(mlir::MlirBuilder& builder, mlir::MlirOp b,
                           int64_t hidden) {
  // [0..2*H]: i and f
  mlir::MlirOp if_part = mlir::stablehlo::Slice(b, {0}, {2 * hidden}, {1});
  // [2*H..3*H]: o
  mlir::MlirOp o_part =
      mlir::stablehlo::Slice(b, {2 * hidden}, {3 * hidden}, {1});
  // [3*H..4*H]: g
  mlir::MlirOp g_part =
      mlir::stablehlo::Slice(b, {3 * hidden}, {4 * hidden}, {1});
  return ConcatDim(builder, {if_part, g_part, o_part}, /*dim=*/0);
}

inline std::optional<mlir::ElementType> GetElementTypeEnum(mlir::Type type) {
  if (type.isBF16()) return mlir::ElementType::BF16;
  if (type.isF32()) return mlir::ElementType::F32;
  if (type.isF16()) return mlir::ElementType::F16;
  if (type.isF64()) return mlir::ElementType::F64;
  return std::nullopt;
}

// Evaluates DotGeneral with native mixed precision:
// If operands are in lower precision (e.g. BF16), contracts directly into
// acc_dtype (e.g. FP32) on the TPU MXU systolic array without upcasting inputs.
// If an operand is in acc_dtype, converts it to target_dtype (e.g. BF16) before
// contraction.
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

inline mlir::MlirOp MixedPrecisionDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    mlir::stablehlo::DotDimensionNumbersAttr dot_dims,
    mlir::ElementType acc_dtype, std::optional<mlir::Type> gemm_dtype,
    mlir::stablehlo::Precision precision) {
  return MixedPrecisionDotGeneral(
      lhs, rhs, dot_dims, mlir::getElementType(lhs.getContext(), acc_dtype),
      gemm_dtype, precision);
}

inline mlir::MlirOp MixedPrecisionDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    mlir::stablehlo::DotDimensionNumbersAttr dot_dims, mlir::Type acc_dtype,
    mlir::stablehlo::Precision precision) {
  return MixedPrecisionDotGeneral(lhs, rhs, dot_dims, acc_dtype,
                                  /*gemm_dtype=*/std::nullopt, precision);
}

inline mlir::MlirOp MixedPrecisionDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    mlir::stablehlo::DotDimensionNumbersAttr dot_dims,
    mlir::ElementType acc_dtype, mlir::stablehlo::Precision precision) {
  return MixedPrecisionDotGeneral(lhs, rhs, dot_dims, acc_dtype,
                                  /*gemm_dtype=*/std::nullopt, precision);
}

LstmStepResults ComputeLstmStep(
    mlir::MlirOp x_proj_step, mlir::MlirOp h_curr, mlir::MlirOp c_curr,
    mlir::MlirOp w_hh, mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    int64_t batch, int64_t hidden, std::optional<mlir::MlirOp> w_hr,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> hr_dot_dims,
    mlir::stablehlo::Precision precision) {
  const mlir::Type acc_dtype = GetTensorTypeOrDie(c_curr).getElementType();
  // Step hidden-to-hidden projection: [B, out_h] @ [4*H, out_h]^T -> [B, 4*H]
  mlir::MlirOp h_proj =
      MixedPrecisionDotGeneral(h_curr, w_hh, hh_dot_dims, acc_dtype, precision);
  mlir::MlirOp gates = mlir::stablehlo::Add(x_proj_step, h_proj);

  // 3H Gate Clumping: unpack [i, f, o] as contiguous [B, 3*H] and [g] as [B, H]
  mlir::MlirOp ifo_pre =
      mlir::stablehlo::Slice(gates, {0, 0}, {batch, 3 * hidden}, {1, 1});
  mlir::MlirOp g_pre = mlir::stablehlo::Slice(gates, {0, 3 * hidden},
                                              {batch, 4 * hidden}, {1, 1});

  // Vectorized non-linearities: single Logistic over [B, 3*H] and single Tanh
  // over [B, H]
  mlir::MlirOp ifo_act = mlir::stablehlo::Logistic(ifo_pre);
  mlir::MlirOp cellgate = mlir::stablehlo::Tanh(g_pre);

  // Unpack individual gate activations from [B, 3*H]
  mlir::MlirOp ingate =
      mlir::stablehlo::Slice(ifo_act, {0, 0}, {batch, hidden}, {1, 1});
  mlir::MlirOp forgetgate =
      mlir::stablehlo::Slice(ifo_act, {0, hidden}, {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp outgate = mlir::stablehlo::Slice(ifo_act, {0, 2 * hidden},
                                                {batch, 3 * hidden}, {1, 1});

  // Update cell state: c_next = f * c + i * g
  mlir::MlirOp f_c = mlir::stablehlo::Mul(forgetgate, c_curr);
  mlir::MlirOp i_g = mlir::stablehlo::Mul(ingate, cellgate);
  mlir::MlirOp c_next = mlir::stablehlo::Add(f_c, i_g);

  // Update unprojected hidden state: r = o * tanh(c_next) [B, H]
  mlir::MlirOp tanh_c = mlir::stablehlo::Tanh(c_next);
  mlir::MlirOp r = mlir::stablehlo::Mul(outgate, tanh_c);

  // If projected (w_hr provided), project r through W_hr: [B, H] @ [P, H]^T ->
  // [B, P]
  mlir::MlirOp h_next = w_hr.has_value()
                            ? MixedPrecisionDotGeneral(r, *w_hr, *hr_dot_dims,
                                                       acc_dtype, precision)
                            : r;

  return {h_next,   c_next,  ingate, forgetgate,
          cellgate, outgate, tanh_c, ifo_act};
}

// Encapsulates intermediate results produced during a single LSTM recurrent
// backward BPTT step:
// - delta_pre: pre-activation gate gradients in 3H clumped [i, f, o, g] layout
// [B, 4*H]
// - delta_c_next: cell gradient propagating to previous timestep t-1 [B, H]
// - delta_h_rec: recurrent hidden gradient propagating to previous timestep t-1
// [B, H]
struct BackwardStepResult {
  mlir::MlirOp delta_pre;
  mlir::MlirOp delta_c_next;
  mlir::MlirOp delta_h_rec;
};

// Computes a single recurrent LSTM backward BPTT step using Fused 4H Gate
// Backward:
// 1. Computes gate pre-activation and cell adjoints via shared
// ComputeLstmGateAdjoints.
// 2. Micro-Concat Elimination (Fused4HGateBackward): Directly concatenates all
//    4 gate adjoints into [B, 4*H] ([i, f, o, g] layout) in a single stage,
//    eliminating intermediate [B, 3*H] buffer allocations and layout
//    reshuffles: delta_pre = Concat([delta_pre_i, delta_pre_f, delta_pre_o,
//    delta_pre_g])
// 3. Projects recurrent hidden:
//    next_delta_h_rec = delta_pre @ W_hh_clumped
BackwardStepResult ComputeBackwardStep(
    mlir::MlirBuilder& builder, mlir::MlirOp delta_h,
    mlir::MlirOp cur_delta_c_next, mlir::MlirOp tanh_c, mlir::MlirOp c_prev,
    mlir::MlirOp g, mlir::MlirOp ifo, mlir::MlirOp w_hh_clumped,
    mlir::stablehlo::DotDimensionNumbersAttr bwd_weight_dot_dims, int64_t batch,
    int64_t hidden, std::optional<mlir::MlirOp> w_hr,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> bwd_hr_dot_dims,
    mlir::stablehlo::Precision precision) {
  const mlir::Type acc_dtype =
      GetTensorTypeOrDie(cur_delta_c_next).getElementType();
  // If projected LSTM (proj_size > 0), backprop delta_h [B, out_h] through
  // W_hr [out_h, hidden] to obtain unprojected cell output adjoint delta_r [B,
  // hidden].
  mlir::MlirOp delta_r =
      w_hr.has_value()
          ? MixedPrecisionDotGeneral(delta_h, *w_hr, *bwd_hr_dot_dims,
                                     acc_dtype, precision)
          : delta_h;

  mlir::MlirOp i = mlir::stablehlo::Slice(ifo, {0, 0}, {batch, hidden}, {1, 1});
  mlir::MlirOp f =
      mlir::stablehlo::Slice(ifo, {0, hidden}, {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp o =
      mlir::stablehlo::Slice(ifo, {0, 2 * hidden}, {batch, 3 * hidden}, {1, 1});

  mlir::MlirOp one_like_h = MakeConstantLike(tanh_c, 1.0);

  LstmGateAdjoints adjoints = ComputeLstmGateAdjoints(
      delta_r, cur_delta_c_next, tanh_c, c_prev, i, f, g, o, one_like_h);

  mlir::MlirOp delta_pre =
      ConcatDim(builder,
                {adjoints.delta_pre_i, adjoints.delta_pre_f,
                 adjoints.delta_pre_o, adjoints.delta_pre_g},
                /*dim=*/1);

  mlir::MlirOp next_delta_h_rec = MixedPrecisionDotGeneral(
      delta_pre, w_hh_clumped, bwd_weight_dot_dims, acc_dtype, precision);

  return {delta_pre, adjoints.delta_c_prev, next_delta_h_rec};
}

// Encapsulates output sequence and final hidden/cell states for a single layer.
struct LstmLayerOutputs {
  mlir::MlirOp layer_output_seq;  // [T, B, H] or [B, T, H]
  mlir::MlirOp final_h;           // [1, B, H]
  mlir::MlirOp final_c;           // [1, B, H]
};

// Computes forward recurrence for a single LSTM layer across sequence length T.
//
// Key Optimizations:
// Static unrolling helper for short sequences (seq_len < kLstmUnrollFactor).
//
// Key Optimizations:
// - Direct Zero-Copy Slicing: If batch_first=True, slices along dim 1 [B, 1,
// 4*H]
//   and concatenates along dim 1 [B, T, H], completely avoiding 3D
//   transposition copies.
// - Static Recurrence Unrolling: For short sequence lengths (T <
// kLstmUnrollFactor),
//   recurrence is statically unrolled, exposing cross-step fusion and
//   instruction scheduling to XLA without loop boundary overhead.
// - Precision Casting: to_out converts accumulator activations back to target
// out_dtype.
template <typename ToOutFn>
LstmLayerOutputs BuildLstmStaticForward(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj, mlir::MlirOp h_init,
    mlir::MlirOp c_init, mlir::MlirOp w_hh,
    mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims, int64_t seq_len,
    int64_t batch, int64_t hidden, int64_t out_h, bool batch_first,
    ToOutFn to_out, std::optional<mlir::MlirOp> w_hr,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> hr_dot_dims,
    mlir::stablehlo::Precision precision) {
  mlir::MlirOp h_curr = h_init;
  mlir::MlirOp c_curr = c_init;
  std::vector<mlir::MlirOp> step_outputs;
  step_outputs.reserve(seq_len);

  for (int64_t t = 0; t < seq_len; ++t) {
    // Slice current timestep input projection without memory layout
    // transpositions:
    mlir::MlirOp x_t_2d =
        SliceStep2D(x_proj, t, batch, 4 * hidden, batch_first);

    // Execute single step recurrence in accumulator precision
    LstmStepResults step =
        ComputeLstmStep(x_t_2d, h_curr, c_curr, w_hh, hh_dot_dims, batch,
                        hidden, w_hr, hr_dot_dims, precision);
    h_curr = step.h_next;
    c_curr = step.c_next;

    // Convert step output to target out_dtype and record for sequence stacking
    mlir::MlirOp h_out = to_out(h_curr);
    step_outputs.push_back(ExpandStep3D(h_out, batch, out_h, batch_first));
  }

  // Concatenate all step outputs along the sequence dimension:
  // - batch_first=True:  concatenate along dim 1 -> [B, T, out_h]
  // - batch_first=False: concatenate along dim 0 -> [T, B, out_h]
  const int64_t concat_dim = batch_first ? 1 : 0;
  return {ConcatDim(builder, step_outputs, concat_dim),
          mlir::stablehlo::Reshape(to_out(h_curr), {1, batch, out_h}),
          mlir::stablehlo::Reshape(to_out(c_curr), {1, batch, hidden})};
}

// Evaluates forward recurrence using chunked execution with partial unrolling
// (kLstmUnrollFactor). When seq_len < kLstmUnrollFactor, delegates directly
// to BuildLstmStaticForward. For seq_len >= kLstmUnrollFactor, executes an
// outer stablehlo.while loop over sequence chunks, statically unrolling
// kLstmUnrollFactor steps per chunk, and handles any remainder steps via
// static unrolling.
template <typename ToOutFn>
LstmLayerOutputs BuildLstmLayerForward(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj, mlir::MlirOp h_init,
    mlir::MlirOp c_init, mlir::MlirOp w_hh,
    mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims, int64_t seq_len,
    int64_t batch, int64_t hidden, int64_t out_h, bool batch_first,
    ToOutFn to_out, std::optional<mlir::MlirOp> w_hr,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> hr_dot_dims,
    mlir::stablehlo::Precision precision) {
  // --- Phase 1: Short-Sequence Fast Path ---
  // If the total sequence length is strictly less than kLstmUnrollFactor,
  // executing through a while-loop introduces unnecessary control flow,
  // loop initialization, scalar branch evaluations, and DynamicUpdateSlice
  // writebacks. We bypass the while-loop entirely and execute all steps
  // via direct static unrolling with straight-line MLIR instructions.
  if (seq_len < kLstmUnrollFactor) {
    return BuildLstmStaticForward(
        builder, x_proj, h_init, c_init, w_hh, hh_dot_dims, seq_len, batch,
        hidden, out_h, batch_first, to_out, w_hr, hr_dot_dims, precision);
  }

  // --- Phase 2: Sequence Chunk Decomposition & Dimension Setup ---
  // Partition sequence length into full chunks of size kLstmUnrollFactor and
  // remainder steps:
  // - num_chunks: Total number of while-loop iterations to execute.
  // - chunked_steps: Number of timesteps processed inside the while loop.
  // - rem_steps: Trailing timesteps (seq_len % kLstmUnrollFactor) to handle
  //   after the loop.
  // - concat_dim: Dimension index along which timesteps are aligned.
  //   For batch_first=True, sequence shape is [B, T, out_h] -> concat_dim = 1.
  //   For batch_first=False, sequence shape is [T, B, out_h] -> concat_dim = 0.
  const int64_t num_chunks = seq_len / kLstmUnrollFactor;
  const int64_t chunked_steps = num_chunks * kLstmUnrollFactor;
  const int64_t rem_steps = seq_len % kLstmUnrollFactor;
  const int64_t concat_dim = batch_first ? 1 : 0;

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = x_proj.getValue().getLoc();
  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType i64_scalar_type =
      mlir::RankedTensorType::get({}, i64);

  // --- Phase 3: Loop State Inits & Destination-Passing Output Buffer ---
  // To avoid allocating new memory buffers on every while-loop iteration,
  // we pre-allocate the output buffer y_chunked initialized to zeros.
  // StableHLO's DynamicUpdateSlice updates slices of y_chunked in-place,
  // enabling XLA buffer assignment to reuse the same device HBM buffer.
  const mlir::Type out_elem_type =
      GetTensorTypeOrDie(to_out(h_init)).getElementType();
  const llvm::SmallVector<int64_t, 3> y_chunk_shape =
      batch_first ? llvm::SmallVector<int64_t, 3>{batch, chunked_steps, out_h}
                  : llvm::SmallVector<int64_t, 3>{chunked_steps, batch, out_h};
  const mlir::RankedTensorType y_type =
      mlir::RankedTensorType::get(y_chunk_shape, out_elem_type);

  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0f, out_elem_type);
  mlir::MlirOp y_init =
      mlir::stablehlo::BroadcastInDim(y_type, zero_scalar, {});

  mlir::MlirOp step_idx_init = MakeScalarConstant(builder, 0, i64);

  // Loop carry types and initial values:
  // [0] step_idx: Scalar loop counter tracking starting timestep of current
  // chunk. [1] curr_h:   Running recurrent hidden state tensor [B, out_h]. [2]
  // curr_c:   Running recurrent cell state tensor [B, H]. [3] y_buffer:
  // Destination-passing sequence output tensor.
  const llvm::SmallVector<mlir::Type> loop_types = {
      i64_scalar_type, h_init.getType(), c_init.getType(), y_type};
  const llvm::SmallVector<mlir::Value> loop_inits = {
      step_idx_init.getValue(), h_init.getValue(), c_init.getValue(),
      y_init.getValue()};

  auto while_op =
      mlir::stablehlo::WhileOp::create(op_builder, loc, loop_types, loop_inits);

  // --- Phase 4: While Loop Condition Region ---
  // Evaluates loop condition: step_idx < chunked_steps.
  // The loop runs exactly num_chunks iterations, incrementing by
  // kLstmUnrollFactor each time.
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

  // --- Phase 5: While Loop Body Region ---
  // Executes kLstmUnrollFactor unrolled recurrent steps per chunk.
  mlir::Block* const body_block = op_builder.createBlock(&while_op.getBody());
  body_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(body_block);

  // Extract loop carry values for the current iteration
  mlir::MlirOp body_step_idx(builder, body_block->getArgument(0));
  mlir::MlirOp body_h(builder, body_block->getArgument(1));
  mlir::MlirOp body_c(builder, body_block->getArgument(2));
  mlir::MlirOp body_y(builder, body_block->getArgument(3));

  // Step 5.1: Burst-fetch kLstmUnrollFactor timesteps of projected inputs
  // using a single DynamicSlice operation. Slicing kLstmUnrollFactor timesteps
  // at once amortizes slice indexing overhead and avoids individual per-step
  // slice dispatches:
  // - batch_first=True:  dynamic slice at [0, step_idx, 0] with shape [B,
  // kLstmUnrollFactor, 4*H]
  // - batch_first=False: dynamic slice at [step_idx, 0, 0] with shape
  // [kLstmUnrollFactor, B, 4*H]
  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64);
  llvm::SmallVector<mlir::MlirOp, 3> x_start_indices;
  llvm::SmallVector<int64_t, 3> x_slice_sizes;
  if (batch_first) {
    x_start_indices = {zero_i64, body_step_idx, zero_i64};
    x_slice_sizes = {batch, kLstmUnrollFactor, 4 * hidden};
  } else {
    x_start_indices = {body_step_idx, zero_i64, zero_i64};
    x_slice_sizes = {kLstmUnrollFactor, batch, 4 * hidden};
  }
  mlir::MlirOp x_chunk =
      mlir::stablehlo::DynamicSlice(x_proj, x_start_indices, x_slice_sizes);

  mlir::MlirOp curr_h = body_h;
  mlir::MlirOp curr_c = body_c;
  std::vector<mlir::MlirOp> chunk_outputs;
  chunk_outputs.reserve(kLstmUnrollFactor);

  // Step 5.2: Statically unroll the kLstmUnrollFactor recurrent steps inside
  // the chunk. All intermediate activations remain in TPU VPU vector registers
  // and VMEM without spilling to accelerator HBM.
  for (int64_t k = 0; k < kLstmUnrollFactor; ++k) {
    // Statically extract the 2D projected input slice [B, 4*H] for step k
    mlir::MlirOp x_k_2d =
        SliceStep2D(x_chunk, k, batch, 4 * hidden, batch_first);

    // Compute recurrent step:
    // - Computes recurrent GEMM: curr_h @ W_hh -> [B, 4*H]
    // - Evaluates gates with 3H clumping: Logistic over [B, 3*H], Tanh over [B,
    // H]
    // - Updates cell state: c_next = f * curr_c + i * g
    // - Computes hidden state: h_next = o * tanh(c_next)
    // - Applies optional projection layer if w_hr is present: h_next = h_next @
    // W_hr
    LstmStepResults step =
        ComputeLstmStep(x_k_2d, curr_h, curr_c, w_hh, hh_dot_dims, batch,
                        hidden, w_hr, hr_dot_dims, precision);
    curr_h = step.h_next;
    curr_c = step.c_next;

    // Convert step output to target output dtype and expand to 3D for sequence
    // concatenation
    mlir::MlirOp h_out = to_out(curr_h);
    chunk_outputs.push_back(ExpandStep3D(h_out, batch, out_h, batch_first));
  }

  // Step 5.3: Concatenate the kLstmUnrollFactor step outputs along the sequence
  // dimension:
  // - batch_first=True:  concatenate along dim 1 -> [B, kLstmUnrollFactor,
  // out_h]
  // - batch_first=False: concatenate along dim 0 -> [kLstmUnrollFactor, B,
  // out_h]
  mlir::MlirOp chunk_out = ConcatDim(builder, chunk_outputs, concat_dim);

  // Step 5.4: In-place update of the sequence output buffer via
  // DynamicUpdateSlice. Writes the concatenated chunk into body_y at offset
  // body_step_idx.
  llvm::SmallVector<mlir::MlirOp, 3> y_start_indices;
  if (batch_first) {
    y_start_indices = {zero_i64, body_step_idx, zero_i64};
  } else {
    y_start_indices = {body_step_idx, zero_i64, zero_i64};
  }
  mlir::MlirOp next_y =
      mlir::stablehlo::DynamicUpdateSlice(body_y, chunk_out, y_start_indices);

  // Advance loop counter by kLstmUnrollFactor
  mlir::MlirOp k_factor_op =
      MakeScalarConstant(builder, kLstmUnrollFactor, i64);
  mlir::MlirOp next_step_idx = mlir::stablehlo::Add(body_step_idx, k_factor_op);

  // Return updated loop state for next iteration
  const llvm::SmallVector<mlir::Value> next_state = {
      next_step_idx.getValue(), curr_h.getValue(), curr_c.getValue(),
      next_y.getValue()};
  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_state);

  // --- Phase 6: Post-Loop Results Extraction ---
  op_builder.setInsertionPointAfter(while_op);

  mlir::MlirOp h_after_chunks(builder, while_op.getResult(1));
  mlir::MlirOp c_after_chunks(builder, while_op.getResult(2));
  mlir::MlirOp y_chunked(builder, while_op.getResult(3));

  mlir::MlirOp final_h = h_after_chunks;
  mlir::MlirOp final_c = c_after_chunks;
  mlir::MlirOp final_seq_out;

  // --- Phase 7: Remainder Step Handling ---
  if (rem_steps == 0) {
    // Exact multiple of kLstmUnrollFactor: the chunked loop output is the
    // complete sequence
    final_seq_out = y_chunked;
  } else {
    // If sequence length is not an exact multiple of kLstmUnrollFactor,
    // statically unroll remainder steps (t = chunked_steps to seq_len - 1)
    // starting from the state produced by the while loop, and concatenate the
    // remainder output with y_chunked.
    std::vector<mlir::MlirOp> rem_outputs;
    rem_outputs.reserve(rem_steps);
    for (int64_t t = chunked_steps; t < seq_len; ++t) {
      mlir::MlirOp x_t_2d =
          SliceStep2D(x_proj, t, batch, 4 * hidden, batch_first);

      LstmStepResults step =
          ComputeLstmStep(x_t_2d, final_h, final_c, w_hh, hh_dot_dims, batch,
                          hidden, w_hr, hr_dot_dims, precision);
      final_h = step.h_next;
      final_c = step.c_next;

      mlir::MlirOp h_out = to_out(final_h);
      rem_outputs.push_back(ExpandStep3D(h_out, batch, out_h, batch_first));
    }
    mlir::MlirOp rem_out = ConcatDim(builder, rem_outputs, concat_dim);
    final_seq_out = ConcatDim(builder, {y_chunked, rem_out}, concat_dim);
  }

  // --- Phase 8: Format Final Hidden and Cell States ---
  // PyTorch returns hidden and cell states formatted with a leading layer
  // dimension [1, B, H].
  return {final_seq_out,
          mlir::stablehlo::Reshape(to_out(final_h), {1, batch, out_h}),
          mlir::stablehlo::Reshape(to_out(final_c), {1, batch, hidden})};
}

// Applies dropout between LSTM layers on the sequence tensor.
// When train=True and dropout > 0.0, elements are dropped with probability
// `dropout` and non-dropped elements are scaled by `1.0 / (1.0 - dropout)`.
std::pair<mlir::MlirOp, mlir::MlirOp>  // STD_PAIR_OK=output and mask tensors
ApplyInterLayerDropout(mlir::MlirOp layer_out, mlir::MlirOp rand_tensor,
                       double dropout) {
  if (dropout <= 0.0) {
    return {layer_out, layer_out};
  }
  if (dropout >= 1.0) {
    mlir::MlirOp zero_const = MakeConstantLike(layer_out, 0.0);
    mlir::MlirOp false_const = MakeConstantLike(rand_tensor, false);
    return {zero_const, false_const};
  }
  mlir::MlirOp p_const = MakeConstantLike(rand_tensor, dropout);
  mlir::MlirOp mask = mlir::stablehlo::Compare(
      rand_tensor, p_const, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp zero = MakeConstantLike(layer_out, 0.0);
  mlir::MlirOp masked = mlir::stablehlo::Select(mask, layer_out, zero);
  const double scale = 1.0 / (1.0 - dropout);
  mlir::MlirOp scale_const = MakeConstantLike(masked, scale);
  mlir::MlirOp dropped = mlir::stablehlo::Mul(masked, scale_const);
  return {dropped, mask};
}

// Applies backward dropout between LSTM layers on the sequence gradient tensor.
mlir::MlirOp ApplyDropoutBackward(mlir::MlirOp grad_output, mlir::MlirOp mask,
                                  double dropout) {
  if (dropout <= 0.0) {
    return grad_output;
  }
  mlir::MlirOp zero = MakeConstantLike(grad_output, 0.0);
  mlir::MlirOp masked = mlir::stablehlo::Select(mask, grad_output, zero);
  const double scale = 1.0 / (1.0 - dropout);
  mlir::MlirOp scale_const = MakeConstantLike(masked, scale);
  return mlir::stablehlo::Mul(masked, scale_const);
}

// Weights and optional projection for a single layer in pipelined wavefront.
struct WavefrontLayerWeight {
  mlir::MlirOp w_hh;                    // clumped [4*H, out_h]
  std::optional<mlir::MlirOp> w_ih;     // clumped [4*H, out_h] (for l >= 1)
  std::optional<mlir::MlirOp> b_total;  // clumped [4*H] (for l >= 1)
  std::optional<mlir::MlirOp> w_hr;     // optional [out_h, hidden]
};

// Evaluates forward recurrence for multi-layer unidirectional LSTM (L >= 2)
// using inter-layer chunk streaming (wavefront pipelining).
// Instead of writing/reading full sequence tensors [T, B, out_h] to/from HBM
// for intermediate layers, small chunk buffers [kLstmUnrollFactor, B, out_h]
// are streamed directly between layer l and layer l+1 in TPU vector memory.
template <typename ToOutFn, typename ToAccFn>
LstmLayerOutputs BuildLstmPipelinedWavefrontForward(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj_0,
    absl::Span<const mlir::MlirOp> h_inits,
    absl::Span<const mlir::MlirOp> c_inits,
    absl::Span<const WavefrontLayerWeight> layer_weights,
    mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    mlir::stablehlo::DotDimensionNumbersAttr ih_dot_dims, int64_t seq_len,
    int64_t batch, int64_t hidden, int64_t out_h, int64_t num_layers,
    bool batch_first, ToOutFn to_out, ToAccFn to_acc,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> hr_dot_dims,
    double dropout, bool train, std::optional<mlir::MlirOp> rand_op,
    mlir::stablehlo::Precision precision) {
  const int64_t num_chunks = seq_len / kLstmUnrollFactor;
  const int64_t chunked_steps = num_chunks * kLstmUnrollFactor;
  const int64_t rem_steps = seq_len % kLstmUnrollFactor;
  const int64_t concat_dim = batch_first ? 1 : 0;
  const bool has_dropout =
      (dropout > 0.0 && train && num_layers > 1 && rand_op.has_value());

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = x_proj_0.getValue().getLoc();
  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType i64_scalar_type =
      mlir::RankedTensorType::get({}, i64);

  const mlir::Type out_elem_type =
      GetTensorTypeOrDie(to_out(h_inits[0])).getElementType();
  const llvm::SmallVector<int64_t, 3> y_chunk_shape =
      batch_first ? llvm::SmallVector<int64_t, 3>{batch, chunked_steps, out_h}
                  : llvm::SmallVector<int64_t, 3>{chunked_steps, batch, out_h};
  const mlir::RankedTensorType y_type =
      mlir::RankedTensorType::get(y_chunk_shape, out_elem_type);

  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0f, out_elem_type);
  mlir::MlirOp y_init =
      mlir::stablehlo::BroadcastInDim(y_type, zero_scalar, {});

  mlir::MlirOp step_idx_init = MakeScalarConstant(builder, 0, i64);

  // Loop types: [step_idx, h_inits[0..L-1], c_inits[0..L-1], y_init]
  llvm::SmallVector<mlir::Type> loop_types;
  loop_types.reserve(2 * num_layers + 2);
  loop_types.push_back(i64_scalar_type);
  for (int64_t l = 0; l < num_layers; ++l) {
    loop_types.push_back(h_inits[l].getType());
  }
  for (int64_t l = 0; l < num_layers; ++l) {
    loop_types.push_back(c_inits[l].getType());
  }
  loop_types.push_back(y_type);

  llvm::SmallVector<mlir::Value> loop_inits;
  loop_inits.reserve(2 * num_layers + 2);
  loop_inits.push_back(step_idx_init.getValue());
  for (int64_t l = 0; l < num_layers; ++l) {
    loop_inits.push_back(h_inits[l].getValue());
  }
  for (int64_t l = 0; l < num_layers; ++l) {
    loop_inits.push_back(c_inits[l].getValue());
  }
  loop_inits.push_back(y_init.getValue());

  auto while_op =
      mlir::stablehlo::WhileOp::create(op_builder, loc, loop_types, loop_inits);

  // Condition region: step_idx < chunked_steps
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

  // Body region
  mlir::Block* const body_block = op_builder.createBlock(&while_op.getBody());
  body_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(body_block);

  mlir::MlirOp body_step_idx(builder, body_block->getArgument(0));
  std::vector<mlir::MlirOp> body_h(num_layers);
  std::vector<mlir::MlirOp> body_c(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    body_h[l] = mlir::MlirOp(builder, body_block->getArgument(1 + l));
    body_c[l] =
        mlir::MlirOp(builder, body_block->getArgument(1 + num_layers + l));
  }
  mlir::MlirOp body_y(builder, body_block->getArgument(1 + 2 * num_layers));

  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64);

  // 1. Layer 0: slice chunk from x_proj_0
  llvm::SmallVector<mlir::MlirOp, 3> x_start_indices;
  llvm::SmallVector<int64_t, 3> x_slice_sizes;
  if (batch_first) {
    x_start_indices = {zero_i64, body_step_idx, zero_i64};
    x_slice_sizes = {batch, kLstmUnrollFactor, 4 * hidden};
  } else {
    x_start_indices = {body_step_idx, zero_i64, zero_i64};
    x_slice_sizes = {kLstmUnrollFactor, batch, 4 * hidden};
  }
  mlir::MlirOp x_chunk =
      mlir::stablehlo::DynamicSlice(x_proj_0, x_start_indices, x_slice_sizes);

  mlir::MlirOp curr_h_0 = body_h[0];
  mlir::MlirOp curr_c_0 = body_c[0];
  std::vector<mlir::MlirOp> chunk_outputs_0;
  chunk_outputs_0.reserve(kLstmUnrollFactor);
  for (int64_t k = 0; k < kLstmUnrollFactor; ++k) {
    mlir::MlirOp x_k_2d =
        SliceStep2D(x_chunk, k, batch, 4 * hidden, batch_first);
    LstmStepResults step = ComputeLstmStep(
        x_k_2d, curr_h_0, curr_c_0, layer_weights[0].w_hh, hh_dot_dims, batch,
        hidden, layer_weights[0].w_hr, hr_dot_dims, precision);
    curr_h_0 = step.h_next;
    curr_c_0 = step.c_next;
    chunk_outputs_0.push_back(
        ExpandStep3D(to_out(curr_h_0), batch, out_h, batch_first));
  }
  mlir::MlirOp curr_chunk = ConcatDim(builder, chunk_outputs_0, concat_dim);

  std::vector<mlir::MlirOp> next_h(num_layers);
  std::vector<mlir::MlirOp> next_c(num_layers);
  next_h[0] = curr_h_0;
  next_c[0] = curr_c_0;

  // 2. Stream chunk through subsequent layers l = 1 .. num_layers - 1
  for (int64_t l = 1; l < num_layers; ++l) {
    if (has_dropout) {
      llvm::SmallVector<mlir::MlirOp, 4> rand_start_indices;
      llvm::SmallVector<int64_t, 4> rand_slice_sizes;
      if (batch_first) {
        rand_start_indices = {MakeScalarConstant(builder, l - 1, i64), zero_i64,
                              body_step_idx, zero_i64};
        rand_slice_sizes = {1, batch, kLstmUnrollFactor, out_h};
      } else {
        rand_start_indices = {MakeScalarConstant(builder, l - 1, i64),
                              body_step_idx, zero_i64, zero_i64};
        rand_slice_sizes = {1, kLstmUnrollFactor, batch, out_h};
      }
      mlir::MlirOp rand_chunk = mlir::stablehlo::DynamicSlice(
          *rand_op, rand_start_indices, rand_slice_sizes);
      rand_chunk = mlir::stablehlo::Reshape(
          rand_chunk, GetTensorTypeOrDie(curr_chunk).getShape());
      auto [dropped_chunk, mask_chunk] =
          ApplyInterLayerDropout(curr_chunk, rand_chunk, dropout);
      curr_chunk = dropped_chunk;
    }

    const mlir::Type acc_dtype = GetTensorTypeOrDie(body_c[l]).getElementType();
    mlir::MlirOp x_proj_chunk = MixedPrecisionDotGeneral(
        curr_chunk, *layer_weights[l].w_ih, ih_dot_dims, acc_dtype, precision);
    if (layer_weights[l].b_total.has_value()) {
      mlir::MlirOp b_tot = *layer_weights[l].b_total;
      mlir::MlirOp b_bcast =
          mlir::stablehlo::BroadcastInDim(GetTensorTypeOrDie(x_proj_chunk),
                                          b_tot, {/*broadcast_dimensions=*/2});
      x_proj_chunk = mlir::stablehlo::Add(x_proj_chunk, b_bcast);
    }

    mlir::MlirOp curr_h_l = body_h[l];
    mlir::MlirOp curr_c_l = body_c[l];
    std::vector<mlir::MlirOp> chunk_outputs_l;
    chunk_outputs_l.reserve(kLstmUnrollFactor);
    for (int64_t k = 0; k < kLstmUnrollFactor; ++k) {
      mlir::MlirOp x_k_2d =
          SliceStep2D(x_proj_chunk, k, batch, 4 * hidden, batch_first);
      LstmStepResults step = ComputeLstmStep(
          x_k_2d, curr_h_l, curr_c_l, layer_weights[l].w_hh, hh_dot_dims, batch,
          hidden, layer_weights[l].w_hr, hr_dot_dims, precision);
      curr_h_l = step.h_next;
      curr_c_l = step.c_next;
      chunk_outputs_l.push_back(
          ExpandStep3D(to_out(curr_h_l), batch, out_h, batch_first));
    }
    curr_chunk = ConcatDim(builder, chunk_outputs_l, concat_dim);
    next_h[l] = curr_h_l;
    next_c[l] = curr_c_l;
  }

  // 3. Commit final layer chunk into body_y
  llvm::SmallVector<mlir::MlirOp, 3> y_start_indices;
  if (batch_first) {
    y_start_indices = {zero_i64, body_step_idx, zero_i64};
  } else {
    y_start_indices = {body_step_idx, zero_i64, zero_i64};
  }
  mlir::MlirOp next_y =
      mlir::stablehlo::DynamicUpdateSlice(body_y, curr_chunk, y_start_indices);

  mlir::MlirOp k_factor_op =
      MakeScalarConstant(builder, kLstmUnrollFactor, i64);
  mlir::MlirOp next_step_idx = mlir::stablehlo::Add(body_step_idx, k_factor_op);

  llvm::SmallVector<mlir::Value> next_state;
  next_state.reserve(2 * num_layers + 2);
  next_state.push_back(next_step_idx.getValue());
  for (int64_t l = 0; l < num_layers; ++l) {
    next_state.push_back(next_h[l].getValue());
  }
  for (int64_t l = 0; l < num_layers; ++l) {
    next_state.push_back(next_c[l].getValue());
  }
  next_state.push_back(next_y.getValue());

  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_state);
  op_builder.setInsertionPointAfter(while_op);

  // Extract loop results
  std::vector<mlir::MlirOp> while_h(num_layers);
  std::vector<mlir::MlirOp> while_c(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    while_h[l] = mlir::MlirOp(builder, while_op.getResult(1 + l));
    while_c[l] = mlir::MlirOp(builder, while_op.getResult(1 + num_layers + l));
  }
  mlir::MlirOp while_y(builder, while_op.getResult(1 + 2 * num_layers));

  std::vector<mlir::MlirOp> all_final_h(num_layers);
  std::vector<mlir::MlirOp> all_final_c(num_layers);
  mlir::MlirOp final_output;

  if (rem_steps > 0) {
    mlir::MlirOp x_rem_0 =
        batch_first
            ? mlir::stablehlo::Slice(x_proj_0, {0, chunked_steps, 0},
                                     {batch, seq_len, 4 * hidden}, {1, 1, 1})
            : mlir::stablehlo::Slice(x_proj_0, {chunked_steps, 0, 0},
                                     {seq_len, batch, 4 * hidden}, {1, 1, 1});
    mlir::MlirOp rem_h_0 = while_h[0];
    mlir::MlirOp rem_c_0 = while_c[0];
    std::vector<mlir::MlirOp> rem_outputs_0;
    rem_outputs_0.reserve(rem_steps);
    for (int64_t k = 0; k < rem_steps; ++k) {
      mlir::MlirOp x_k_2d =
          SliceStep2D(x_rem_0, k, batch, 4 * hidden, batch_first);
      LstmStepResults step = ComputeLstmStep(
          x_k_2d, rem_h_0, rem_c_0, layer_weights[0].w_hh, hh_dot_dims, batch,
          hidden, layer_weights[0].w_hr, hr_dot_dims, precision);
      rem_h_0 = step.h_next;
      rem_c_0 = step.c_next;
      rem_outputs_0.push_back(
          ExpandStep3D(to_out(rem_h_0), batch, out_h, batch_first));
    }
    mlir::MlirOp curr_rem = ConcatDim(builder, rem_outputs_0, concat_dim);
    all_final_h[0] =
        mlir::stablehlo::Reshape(to_out(rem_h_0), {1, batch, out_h});
    all_final_c[0] =
        mlir::stablehlo::Reshape(to_out(rem_c_0), {1, batch, hidden});

    for (int64_t l = 1; l < num_layers; ++l) {
      if (has_dropout) {
        mlir::MlirOp rand_rem =
            batch_first
                ? mlir::stablehlo::Slice(*rand_op, {l - 1, 0, chunked_steps, 0},
                                         {l, batch, seq_len, out_h},
                                         {1, 1, 1, 1})
                : mlir::stablehlo::Slice(*rand_op, {l - 1, chunked_steps, 0, 0},
                                         {l, seq_len, batch, out_h},
                                         {1, 1, 1, 1});
        rand_rem = mlir::stablehlo::Reshape(
            rand_rem, GetTensorTypeOrDie(curr_rem).getShape());
        auto [dropped_rem, mask_rem] =
            ApplyInterLayerDropout(curr_rem, rand_rem, dropout);
        curr_rem = dropped_rem;
      }
      const mlir::Type acc_dtype =
          GetTensorTypeOrDie(while_c[l]).getElementType();
      mlir::MlirOp x_proj_rem = MixedPrecisionDotGeneral(
          curr_rem, *layer_weights[l].w_ih, ih_dot_dims, acc_dtype, precision);
      if (layer_weights[l].b_total.has_value()) {
        mlir::MlirOp b_tot = *layer_weights[l].b_total;
        mlir::MlirOp b_bcast = mlir::stablehlo::BroadcastInDim(
            GetTensorTypeOrDie(x_proj_rem), b_tot,
            {/*broadcast_dimensions=*/2});
        x_proj_rem = mlir::stablehlo::Add(x_proj_rem, b_bcast);
      }
      mlir::MlirOp rem_h_l = while_h[l];
      mlir::MlirOp rem_c_l = while_c[l];
      std::vector<mlir::MlirOp> rem_outputs_l;
      rem_outputs_l.reserve(rem_steps);
      for (int64_t k = 0; k < rem_steps; ++k) {
        mlir::MlirOp x_k_2d =
            SliceStep2D(x_proj_rem, k, batch, 4 * hidden, batch_first);
        LstmStepResults step = ComputeLstmStep(
            x_k_2d, rem_h_l, rem_c_l, layer_weights[l].w_hh, hh_dot_dims, batch,
            hidden, layer_weights[l].w_hr, hr_dot_dims, precision);
        rem_h_l = step.h_next;
        rem_c_l = step.c_next;
        rem_outputs_l.push_back(
            ExpandStep3D(to_out(rem_h_l), batch, out_h, batch_first));
      }
      curr_rem = ConcatDim(builder, rem_outputs_l, concat_dim);
      all_final_h[l] =
          mlir::stablehlo::Reshape(to_out(rem_h_l), {1, batch, out_h});
      all_final_c[l] =
          mlir::stablehlo::Reshape(to_out(rem_c_l), {1, batch, hidden});
    }
    final_output = ConcatDim(builder, {while_y, curr_rem}, concat_dim);
  } else {
    for (int64_t l = 0; l < num_layers; ++l) {
      all_final_h[l] =
          mlir::stablehlo::Reshape(to_out(while_h[l]), {1, batch, out_h});
      all_final_c[l] =
          mlir::stablehlo::Reshape(to_out(while_c[l]), {1, batch, hidden});
    }
    final_output = while_y;
  }

  mlir::MlirOp h_n = ConcatDim(builder, all_final_h, /*dim=*/0);
  mlir::MlirOp c_n = ConcatDim(builder, all_final_c, /*dim=*/0);
  return {final_output, h_n, c_n};
}

// Encapsulates output tensors produced by a bidirectional LSTM layer:
// - layer_output_seq: concatenated forward and reverse sequence outputs
//   [B, T, 2*H] (if batch_first) or [T, B, 2*H] (if !batch_first)
// - final_h_fwd, final_c_fwd: forward final states [1, B, H] (at timestep T-1)
// - final_h_rev, final_c_rev: reverse final states [1, B, H] (at timestep 0)
struct LstmBidirLayerOutputs {
  mlir::MlirOp layer_output_seq;
  mlir::MlirOp final_h_fwd;
  mlir::MlirOp final_c_fwd;
  mlir::MlirOp final_h_rev;
  mlir::MlirOp final_c_rev;
};

// Evaluates forward recurrence for a bidirectional layer using Concurrent
// Bidirectional Fusion with partial unrolling (kLstmUnrollFactor = 8).
// Forward advances t = 0 to T - 1, while reverse concurrently advances
// t = T - 1 down to 0 inside the SAME loop body.
// This executes forward and reverse steps simultaneously, enabling TPU
// instruction scheduling to overlap matrix multiplications and vector ops
// across execution units and cutting loop control overhead in half.
// Static unrolling helper for bidirectional LSTM forward recurrence for short
// sequences (seq_len < kLstmUnrollFactor).
//
// Statically unrolls all timesteps s in [0, seq_len - 1].
// In each timestep, executes forward recurrence at index s and reverse
// recurrence at index rev_t = seq_len - 1 - s concurrently in lockstep. This
// interleaves vector and matrix operations, maximizing TPU execution unit
// concurrency while eliminating while-loop overhead.
template <typename ToOutFn>
LstmBidirLayerOutputs BuildLstmBidirStaticForward(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj_fwd,
    mlir::MlirOp x_proj_rev, mlir::MlirOp h_init_fwd, mlir::MlirOp c_init_fwd,
    mlir::MlirOp h_init_rev, mlir::MlirOp c_init_rev, mlir::MlirOp w_hh_fwd,
    mlir::MlirOp w_hh_rev, mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    int64_t seq_len, int64_t batch, int64_t hidden, int64_t out_h,
    bool batch_first, ToOutFn to_out, std::optional<mlir::MlirOp> w_hr_fwd,
    std::optional<mlir::MlirOp> w_hr_rev,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> hr_dot_dims,
    mlir::stablehlo::Precision precision) {
  const int64_t concat_dim = batch_first ? 1 : 0;
  const int64_t feature_dim = 2;

  mlir::MlirOp curr_h_fwd = h_init_fwd;
  mlir::MlirOp curr_c_fwd = c_init_fwd;
  mlir::MlirOp curr_h_rev = h_init_rev;
  mlir::MlirOp curr_c_rev = c_init_rev;

  std::vector<mlir::MlirOp> fwd_step_outputs(seq_len);
  std::vector<mlir::MlirOp> rev_step_outputs(seq_len);

  for (int64_t s = 0; s < seq_len; ++s) {
    // Forward timestep: s
    mlir::MlirOp x_fwd_2d =
        SliceStep2D(x_proj_fwd, s, batch, 4 * hidden, batch_first);

    // Reverse timestep: seq_len - 1 - s
    const int64_t rev_t = seq_len - 1 - s;
    mlir::MlirOp x_rev_2d =
        SliceStep2D(x_proj_rev, rev_t, batch, 4 * hidden, batch_first);

    // Concurrent recurrence: both steps are independent in the StableHLO DAG
    LstmStepResults fwd_step =
        ComputeLstmStep(x_fwd_2d, curr_h_fwd, curr_c_fwd, w_hh_fwd, hh_dot_dims,
                        batch, hidden, w_hr_fwd, hr_dot_dims, precision);
    LstmStepResults rev_step =
        ComputeLstmStep(x_rev_2d, curr_h_rev, curr_c_rev, w_hh_rev, hh_dot_dims,
                        batch, hidden, w_hr_rev, hr_dot_dims, precision);

    curr_h_fwd = fwd_step.h_next;
    curr_c_fwd = fwd_step.c_next;
    curr_h_rev = rev_step.h_next;
    curr_c_rev = rev_step.c_next;

    mlir::MlirOp h_fwd_out = to_out(curr_h_fwd);
    fwd_step_outputs[s] = ExpandStep3D(h_fwd_out, batch, out_h, batch_first);

    mlir::MlirOp h_rev_out = to_out(curr_h_rev);
    rev_step_outputs[rev_t] =
        ExpandStep3D(h_rev_out, batch, out_h, batch_first);
  }

  mlir::MlirOp y_fwd = ConcatDim(builder, fwd_step_outputs, concat_dim);
  mlir::MlirOp y_rev = ConcatDim(builder, rev_step_outputs, concat_dim);
  mlir::MlirOp y_layer = ConcatDim(builder, {y_fwd, y_rev}, feature_dim);

  return {y_layer,
          mlir::stablehlo::Reshape(to_out(curr_h_fwd), {1, batch, out_h}),
          mlir::stablehlo::Reshape(to_out(curr_c_fwd), {1, batch, hidden}),
          mlir::stablehlo::Reshape(to_out(curr_h_rev), {1, batch, out_h}),
          mlir::stablehlo::Reshape(to_out(curr_c_rev), {1, batch, hidden})};
}

// Chunked streaming helper for bidirectional LSTM forward recurrence (seq_len
// >= kLstmUnrollFactor).
//
// Key Optimizations:
// 1. Partial unrolling: Unrolls kLstmUnrollFactor timesteps inside a single
// StableHLO while-loop.
// 2. Concurrent execution: Evaluates forward and reverse recurrence chunks
// concurrently in lockstep.
// 3. Dynamic slicing: Slices only kLstmUnrollFactor steps of projected inputs
// per chunk.
// 4. In-place sequence update: Writes chunk outputs directly into sequence
// buffers via DynamicUpdateSlice.
// 5. Remainder handling: Statically unrolls leftover steps (seq_len %
// kLstmUnrollFactor).
template <typename ToOutFn>
LstmBidirLayerOutputs BuildLstmBidirChunkedForward(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj_fwd,
    mlir::MlirOp x_proj_rev, mlir::MlirOp h_init_fwd, mlir::MlirOp c_init_fwd,
    mlir::MlirOp h_init_rev, mlir::MlirOp c_init_rev, mlir::MlirOp w_hh_fwd,
    mlir::MlirOp w_hh_rev, mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    int64_t seq_len, int64_t batch, int64_t hidden, int64_t out_h,
    bool batch_first, ToOutFn to_out, std::optional<mlir::MlirOp> w_hr_fwd,
    std::optional<mlir::MlirOp> w_hr_rev,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> hr_dot_dims,
    mlir::stablehlo::Precision precision) {
  const int64_t concat_dim = batch_first ? 1 : 0;
  const int64_t feature_dim = 2;
  // Chunked execution: seq_len >= kLstmUnrollFactor
  const int64_t num_chunks = seq_len / kLstmUnrollFactor;
  const int64_t chunked_steps = num_chunks * kLstmUnrollFactor;
  const int64_t rem_steps = seq_len % kLstmUnrollFactor;

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = x_proj_fwd.getValue().getLoc();
  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType i64_scalar_type =
      mlir::RankedTensorType::get({}, i64);

  const mlir::Type out_elem_type =
      GetTensorTypeOrDie(to_out(h_init_fwd)).getElementType();
  const llvm::SmallVector<int64_t, 3> y_chunk_shape =
      batch_first ? llvm::SmallVector<int64_t, 3>{batch, chunked_steps, out_h}
                  : llvm::SmallVector<int64_t, 3>{chunked_steps, batch, out_h};
  const mlir::RankedTensorType y_type =
      mlir::RankedTensorType::get(y_chunk_shape, out_elem_type);

  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0f, out_elem_type);
  mlir::MlirOp y_fwd_init =
      mlir::stablehlo::BroadcastInDim(y_type, zero_scalar, {});
  mlir::MlirOp y_rev_init =
      mlir::stablehlo::BroadcastInDim(y_type, zero_scalar, {});

  mlir::MlirOp step_idx_init = MakeScalarConstant(builder, 0, i64);

  const llvm::SmallVector<mlir::Type> loop_types = {i64_scalar_type,
                                                    h_init_fwd.getType(),
                                                    c_init_fwd.getType(),
                                                    h_init_rev.getType(),
                                                    c_init_rev.getType(),
                                                    y_type,
                                                    y_type};
  const llvm::SmallVector<mlir::Value> loop_inits = {
      step_idx_init.getValue(), h_init_fwd.getValue(), c_init_fwd.getValue(),
      h_init_rev.getValue(),    c_init_rev.getValue(), y_fwd_init.getValue(),
      y_rev_init.getValue()};

  auto while_op =
      mlir::stablehlo::WhileOp::create(op_builder, loc, loop_types, loop_inits);

  // Cond region: step_idx < chunked_steps
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

  // Body region: execute 8 forward & 8 reverse recurrent steps concurrently per
  // chunk
  mlir::Block* const body_block = op_builder.createBlock(&while_op.getBody());
  body_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(body_block);

  mlir::MlirOp body_step_idx(builder, body_block->getArgument(0));
  mlir::MlirOp body_h_fwd(builder, body_block->getArgument(1));
  mlir::MlirOp body_c_fwd(builder, body_block->getArgument(2));
  mlir::MlirOp body_h_rev(builder, body_block->getArgument(3));
  mlir::MlirOp body_c_rev(builder, body_block->getArgument(4));
  mlir::MlirOp body_y_fwd(builder, body_block->getArgument(5));
  mlir::MlirOp body_y_rev(builder, body_block->getArgument(6));

  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64);
  const llvm::SmallVector<int64_t, 3> x_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kLstmUnrollFactor, 4 * hidden}
          : llvm::SmallVector<int64_t, 3>{kLstmUnrollFactor, batch, 4 * hidden};

  // 1. Forward burst fetch via DynamicSlice at [0, step_idx, 0] or [step_idx,
  // 0, 0]
  llvm::SmallVector<mlir::MlirOp, 3> fwd_x_start_indices =
      batch_first ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, body_step_idx,
                                                       zero_i64}
                  : llvm::SmallVector<mlir::MlirOp, 3>{body_step_idx, zero_i64,
                                                       zero_i64};
  mlir::MlirOp x_chunk_fwd = mlir::stablehlo::DynamicSlice(
      x_proj_fwd, fwd_x_start_indices, x_slice_sizes);

  // 2. Reverse burst fetch via DynamicSlice at (seq_len - kLstmUnrollFactor) -
  // step_idx
  mlir::MlirOp const_T_minus_K =
      MakeScalarConstant(builder, seq_len - kLstmUnrollFactor, i64);
  mlir::MlirOp rev_x_start_idx =
      mlir::stablehlo::Subtract(const_T_minus_K, body_step_idx);
  llvm::SmallVector<mlir::MlirOp, 3> rev_x_start_indices =
      batch_first
          ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, rev_x_start_idx,
                                               zero_i64}
          : llvm::SmallVector<mlir::MlirOp, 3>{rev_x_start_idx, zero_i64,
                                               zero_i64};
  mlir::MlirOp x_chunk_rev = mlir::stablehlo::DynamicSlice(
      x_proj_rev, rev_x_start_indices, x_slice_sizes);

  mlir::MlirOp curr_h_fwd = body_h_fwd;
  mlir::MlirOp curr_c_fwd = body_c_fwd;
  mlir::MlirOp curr_h_rev = body_h_rev;
  mlir::MlirOp curr_c_rev = body_c_rev;

  std::vector<mlir::MlirOp> fwd_chunk_outputs(kLstmUnrollFactor);
  std::vector<mlir::MlirOp> rev_chunk_outputs(kLstmUnrollFactor);

  // 3. Statically unroll 8 forward & 8 reverse steps concurrently in vector
  // registers
  for (int64_t k = 0; k < kLstmUnrollFactor; ++k) {
    // Forward slice at k
    mlir::MlirOp x_k_fwd_2d =
        SliceStep2D(x_chunk_fwd, k, batch, 4 * hidden, batch_first);

    // Reverse slice at kLstmUnrollFactor - 1 - k (reverse chronological within
    // chunk)
    const int64_t rev_k = kLstmUnrollFactor - 1 - k;
    mlir::MlirOp x_k_rev_2d =
        SliceStep2D(x_chunk_rev, rev_k, batch, 4 * hidden, batch_first);

    // Compute both steps concurrently
    LstmStepResults fwd_step = ComputeLstmStep(
        x_k_fwd_2d, curr_h_fwd, curr_c_fwd, w_hh_fwd, hh_dot_dims, batch,
        hidden, w_hr_fwd, hr_dot_dims, precision);
    LstmStepResults rev_step = ComputeLstmStep(
        x_k_rev_2d, curr_h_rev, curr_c_rev, w_hh_rev, hh_dot_dims, batch,
        hidden, w_hr_rev, hr_dot_dims, precision);

    curr_h_fwd = fwd_step.h_next;
    curr_c_fwd = fwd_step.c_next;
    curr_h_rev = rev_step.h_next;
    curr_c_rev = rev_step.c_next;

    mlir::MlirOp h_fwd_out = to_out(curr_h_fwd);
    fwd_chunk_outputs[k] = ExpandStep3D(h_fwd_out, batch, out_h, batch_first);

    mlir::MlirOp h_rev_out = to_out(curr_h_rev);
    // Reverse outputs stored in chronological order (rev_k = kLstmUnrollFactor
    // - 1 - k)
    rev_chunk_outputs[rev_k] =
        ExpandStep3D(h_rev_out, batch, out_h, batch_first);
  }

  // 4. Commit chunk outputs into sequence buffers via DynamicUpdateSlice
  mlir::MlirOp fwd_chunk_out =
      ConcatDim(builder, fwd_chunk_outputs, concat_dim);
  llvm::SmallVector<mlir::MlirOp, 3> fwd_y_start_indices =
      batch_first ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, body_step_idx,
                                                       zero_i64}
                  : llvm::SmallVector<mlir::MlirOp, 3>{body_step_idx, zero_i64,
                                                       zero_i64};
  mlir::MlirOp next_y_fwd = mlir::stablehlo::DynamicUpdateSlice(
      body_y_fwd, fwd_chunk_out, fwd_y_start_indices);

  mlir::MlirOp rev_chunk_out =
      ConcatDim(builder, rev_chunk_outputs, concat_dim);
  mlir::MlirOp const_chunked_minus_K =
      MakeScalarConstant(builder, chunked_steps - kLstmUnrollFactor, i64);
  mlir::MlirOp rev_y_write_idx =
      mlir::stablehlo::Subtract(const_chunked_minus_K, body_step_idx);
  llvm::SmallVector<mlir::MlirOp, 3> rev_y_start_indices =
      batch_first
          ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, rev_y_write_idx,
                                               zero_i64}
          : llvm::SmallVector<mlir::MlirOp, 3>{rev_y_write_idx, zero_i64,
                                               zero_i64};
  mlir::MlirOp next_y_rev = mlir::stablehlo::DynamicUpdateSlice(
      body_y_rev, rev_chunk_out, rev_y_start_indices);

  mlir::MlirOp k_factor_op =
      MakeScalarConstant(builder, kLstmUnrollFactor, i64);
  mlir::MlirOp next_step_idx = mlir::stablehlo::Add(body_step_idx, k_factor_op);

  const llvm::SmallVector<mlir::Value> next_state = {
      next_step_idx.getValue(), curr_h_fwd.getValue(), curr_c_fwd.getValue(),
      curr_h_rev.getValue(),    curr_c_rev.getValue(), next_y_fwd.getValue(),
      next_y_rev.getValue()};
  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_state);

  op_builder.setInsertionPointAfter(while_op);

  mlir::MlirOp h_fwd_after(builder, while_op.getResult(1));
  mlir::MlirOp c_fwd_after(builder, while_op.getResult(2));
  mlir::MlirOp h_rev_after(builder, while_op.getResult(3));
  mlir::MlirOp c_rev_after(builder, while_op.getResult(4));
  mlir::MlirOp y_fwd_chunked(builder, while_op.getResult(5));
  mlir::MlirOp y_rev_chunked(builder, while_op.getResult(6));

  mlir::MlirOp final_h_fwd = h_fwd_after;
  mlir::MlirOp final_c_fwd = c_fwd_after;
  mlir::MlirOp final_h_rev = h_rev_after;
  mlir::MlirOp final_c_rev = c_rev_after;

  mlir::MlirOp final_seq_fwd;
  mlir::MlirOp final_seq_rev;

  if (rem_steps == 0) {
    final_seq_fwd = y_fwd_chunked;
    final_seq_rev = y_rev_chunked;
  } else {
    // Statically unroll remaining steps concurrently
    std::vector<mlir::MlirOp> rem_outputs_fwd(rem_steps);
    std::vector<mlir::MlirOp> rem_outputs_rev(rem_steps);

    for (int64_t r = 0; r < rem_steps; ++r) {
      // Forward remainder: chunked_steps + r
      const int64_t fwd_t = chunked_steps + r;
      mlir::MlirOp x_fwd_2d =
          SliceStep2D(x_proj_fwd, fwd_t, batch, 4 * hidden, batch_first);

      // Reverse remainder: rem_steps - 1 - r (timesteps rem_steps - 1 down to
      // 0)
      const int64_t rev_t = rem_steps - 1 - r;
      mlir::MlirOp x_rev_2d =
          SliceStep2D(x_proj_rev, rev_t, batch, 4 * hidden, batch_first);

      LstmStepResults fwd_step = ComputeLstmStep(
          x_fwd_2d, final_h_fwd, final_c_fwd, w_hh_fwd, hh_dot_dims, batch,
          hidden, w_hr_fwd, hr_dot_dims, precision);
      LstmStepResults rev_step = ComputeLstmStep(
          x_rev_2d, final_h_rev, final_c_rev, w_hh_rev, hh_dot_dims, batch,
          hidden, w_hr_rev, hr_dot_dims, precision);

      final_h_fwd = fwd_step.h_next;
      final_c_fwd = fwd_step.c_next;
      final_h_rev = rev_step.h_next;
      final_c_rev = rev_step.c_next;

      mlir::MlirOp h_fwd_out = to_out(final_h_fwd);
      rem_outputs_fwd[r] = ExpandStep3D(h_fwd_out, batch, out_h, batch_first);

      mlir::MlirOp h_rev_out = to_out(final_h_rev);
      rem_outputs_rev[rev_t] =
          ExpandStep3D(h_rev_out, batch, out_h, batch_first);
    }

    mlir::MlirOp rem_out_fwd = ConcatDim(builder, rem_outputs_fwd, concat_dim);
    final_seq_fwd =
        ConcatDim(builder, {y_fwd_chunked, rem_out_fwd}, concat_dim);

    mlir::MlirOp rem_out_rev = ConcatDim(builder, rem_outputs_rev, concat_dim);
    // For reverse: remainder covers 0..rem_steps-1, chunked covers
    // rem_steps..T-1
    final_seq_rev =
        ConcatDim(builder, {rem_out_rev, y_rev_chunked}, concat_dim);
  }

  // Concatenate forward and reverse along feature dimension (dim 2)
  mlir::MlirOp y_layer =
      ConcatDim(builder, {final_seq_fwd, final_seq_rev}, feature_dim);

  return {y_layer,
          mlir::stablehlo::Reshape(to_out(final_h_fwd), {1, batch, out_h}),
          mlir::stablehlo::Reshape(to_out(final_c_fwd), {1, batch, hidden}),
          mlir::stablehlo::Reshape(to_out(final_h_rev), {1, batch, out_h}),
          mlir::stablehlo::Reshape(to_out(final_c_rev), {1, batch, hidden})};
}

// Evaluates forward recurrence for a single bidirectional LSTM layer.
//
// Key Optimizations:
// 1. Dispatch:
//    - For short sequences (seq_len < kLstmUnrollFactor), calls
//    BuildLstmBidirStaticForward
//      to eliminate while-loop overhead.
//    - For standard sequences (seq_len >= kLstmUnrollFactor), calls
//    BuildLstmBidirChunkedForward
//      for chunked execution with partial unrolling.
template <typename ToOutFn>
LstmBidirLayerOutputs BuildLstmBidirLayerForward(
    mlir::MlirBuilder& builder, mlir::MlirOp x_proj_fwd,
    mlir::MlirOp x_proj_rev, mlir::MlirOp h_init_fwd, mlir::MlirOp c_init_fwd,
    mlir::MlirOp h_init_rev, mlir::MlirOp c_init_rev, mlir::MlirOp w_hh_fwd,
    mlir::MlirOp w_hh_rev, mlir::stablehlo::DotDimensionNumbersAttr hh_dot_dims,
    int64_t seq_len, int64_t batch, int64_t hidden, int64_t out_h,
    bool batch_first, ToOutFn to_out, std::optional<mlir::MlirOp> w_hr_fwd,
    std::optional<mlir::MlirOp> w_hr_rev,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> hr_dot_dims,
    mlir::stablehlo::Precision precision) {
  if (seq_len < kLstmUnrollFactor) {
    return BuildLstmBidirStaticForward(
        builder, x_proj_fwd, x_proj_rev, h_init_fwd, c_init_fwd, h_init_rev,
        c_init_rev, w_hh_fwd, w_hh_rev, hh_dot_dims, seq_len, batch, hidden,
        out_h, batch_first, to_out, w_hr_fwd, w_hr_rev, hr_dot_dims, precision);
  }

  return BuildLstmBidirChunkedForward(
      builder, x_proj_fwd, x_proj_rev, h_init_fwd, c_init_fwd, h_init_rev,
      c_init_rev, w_hh_fwd, w_hh_rev, hh_dot_dims, seq_len, batch, hidden,
      out_h, batch_first, to_out, w_hr_fwd, w_hr_rev, hr_dot_dims, precision);
}

// Forward activations recorded per layer during backpropagation recomputation.
// These activations are required to compute the exact gate and parameter
// gradients:
// - h_prev_list[t]: previous hidden state h_{t-1} for recurrent weight grad
// (grad_w_hh)
// - c_prev_list[t]: previous cell state c_{t-1} for forget gate derivative
// - ingate_list[t]: input gate activation i_t for cell derivative
// - forgetgate_list[t]: forget gate activation f_t for cell gradient
// propagation
// - cellgate_list[t]: cell candidate activation g_t for input gate derivative
// - outgate_list[t]: output gate activation o_t for cell gradient contribution
// - tanh_c_list[t]: tanh(c_t) for output gate derivative and cell derivative
struct LayerForwardState {
  mlir::MlirOp layer_input;
  std::vector<mlir::MlirOp> h_prev_list;
  std::vector<mlir::MlirOp> c_prev_list;
  std::vector<mlir::MlirOp> cellgate_list;
  std::vector<mlir::MlirOp> c_curr_list;
  std::vector<mlir::MlirOp> tanh_c_list;
  std::vector<mlir::MlirOp>
      ifo_list;  // 3H clumped [i, f, o] activations [B, 3*H]

  // Sequence tensors for chunked BPTT
  mlir::MlirOp h_prev_seq;
  mlir::MlirOp c_prev_seq;
  mlir::MlirOp cellgate_seq;
  mlir::MlirOp tanh_c_seq;
  mlir::MlirOp ifo_seq;  // 3H clumped [i, f, o] sequence [seq_len, batch, 3*H]
};

// Encapsulates recorded forward activation state and sequence output for layer
// l.
struct LayerRecomputeResult {
  LayerForwardState state;
  mlir::MlirOp layer_output_seq;
  mlir::MlirOp final_h;
  mlir::MlirOp final_c;
};

// Encapsulates 4 unprojected activation sequences unpacked from a cached packed
// sequence tensor: [cellgate (H), tanh_c (H), c_prev (H), h_prev (out_h)]
struct UnpackedActivations {
  mlir::MlirOp cellgate_seq;
  mlir::MlirOp tanh_c_seq;
  mlir::MlirOp c_prev_seq;
  mlir::MlirOp h_prev_seq;
};

// Unpacks cached activations packed across feature dimension (dim 2):
// [cellgate (H), tanh_c (H), c_prev (H), h_prev (out_h)]
inline UnpackedActivations UnpackCachedActivations(
    mlir::MlirOp pack_3d, int64_t seq_len, int64_t batch, int64_t hidden,
    int64_t out_h, bool batch_first) {
  const int64_t d0 = batch_first ? batch : seq_len;
  const int64_t d1 = batch_first ? seq_len : batch;
  mlir::MlirOp cellgate =
      mlir::stablehlo::Slice(pack_3d, {0, 0, 0}, {d0, d1, hidden}, {1, 1, 1});
  mlir::MlirOp tanh_c = mlir::stablehlo::Slice(pack_3d, {0, 0, hidden},
                                               {d0, d1, 2 * hidden}, {1, 1, 1});
  mlir::MlirOp c_prev = mlir::stablehlo::Slice(pack_3d, {0, 0, 2 * hidden},
                                               {d0, d1, 3 * hidden}, {1, 1, 1});
  mlir::MlirOp h_prev = mlir::stablehlo::Slice(
      pack_3d, {0, 0, 3 * hidden}, {d0, d1, 3 * hidden + out_h}, {1, 1, 1});
  return {cellgate, tanh_c, c_prev, h_prev};
}

// Slices an individual timestep activation from a sequence tensor if not
// already present in the unrolled list (e.g. during chunked execution or
// activation caching).
inline mlir::MlirOp GetStepActivation(mlir::MlirBuilder& builder,
                                      const std::vector<mlir::MlirOp>& list,
                                      mlir::MlirOp seq, const int64_t t,
                                      const int64_t batch,
                                      const int64_t feature_dim,
                                      const bool batch_first = false) {
  if (t < static_cast<int64_t>(list.size()) && list[t].getValue()) {
    return list[t];
  }
  return SliceStep2D(seq, t, batch, feature_dim, batch_first);
}

// Result gradients computed for a single LSTM layer during BPTT:
// - grad_x_layer: gradient w.r.t. layer input sequence [T, B, in_dim]
// - grad_h0_layer: gradient w.r.t. initial hidden state h_0 for this layer [1,
// B, H]
// - grad_c0_layer: gradient w.r.t. initial cell state c_0 for this layer [1, B,
// H]
// - grad_w_ih: gradient w.r.t. input weights [4*H, in_dim]
// - grad_w_hh: gradient w.r.t. recurrent weights [4*H, H]
// - grad_bias: gradient w.r.t. combined bias [4*H] (if bias=True)
struct LayerBackwardOutputs {
  mlir::MlirOp grad_x_layer;
  mlir::MlirOp grad_h0_layer;
  mlir::MlirOp grad_c0_layer;
  mlir::MlirOp grad_w_ih;
  mlir::MlirOp grad_w_hh;
  std::optional<mlir::MlirOp> grad_bias;
  std::optional<mlir::MlirOp> grad_w_hr;
};

// Result gradients computed for multi-layer LSTM during Pipelined Wavefront
// BPTT:
// - grad_x: gradient w.r.t. network input sequence [T, B, input_size]
// - grad_h0: initial hidden state gradients for all layers
// - grad_c0: initial cell state gradients for all layers
// - grad_w_ih: input weight gradients for all layers
// - grad_w_hh: recurrent weight gradients for all layers
// - grad_bias: combined bias gradients for all layers (if has_biases)
// - grad_w_hr: projection weight gradients for all layers (if is_projected)
struct MultiLayerBackwardOutputs {
  mlir::MlirOp grad_x;
  std::vector<mlir::MlirOp> grad_h0;
  std::vector<mlir::MlirOp> grad_c0;
  std::vector<mlir::MlirOp> grad_w_ih;
  std::vector<mlir::MlirOp> grad_w_hh;
  std::vector<std::optional<mlir::MlirOp>> grad_bias;
  std::vector<std::optional<mlir::MlirOp>> grad_w_hr;
};

// Result gradients computed for a bidirectional LSTM layer during BPTT:
// - grad_x_layer: summed gradient w.r.t. layer input sequence [T, B, in_dim]
// - grad_h0_fwd, grad_c0_fwd: initial state gradients for forward branch [1, B,
// out_h] / [1, B, H]
// - grad_h0_rev, grad_c0_rev: initial state gradients for reverse branch [1, B,
// out_h] / [1, B, H]
// - grad_w_ih_fwd, grad_w_hh_fwd: forward branch weight gradients
// - grad_w_ih_rev, grad_w_hh_rev: reverse branch weight gradients
// - grad_bias_fwd, grad_bias_rev: combined bias gradients (if bias=True)
// - grad_w_hr_fwd, grad_w_hr_rev: projection weight gradients (if proj_size >
// 0)
struct LstmBidirLayerBackwardOutputs {
  mlir::MlirOp grad_x_layer;
  mlir::MlirOp grad_h0_fwd;
  mlir::MlirOp grad_c0_fwd;
  mlir::MlirOp grad_h0_rev;
  mlir::MlirOp grad_c0_rev;
  mlir::MlirOp grad_w_ih_fwd;
  mlir::MlirOp grad_w_hh_fwd;
  mlir::MlirOp grad_w_ih_rev;
  mlir::MlirOp grad_w_hh_rev;
  std::optional<mlir::MlirOp> grad_bias_fwd;
  std::optional<mlir::MlirOp> grad_bias_rev;
  std::optional<mlir::MlirOp> grad_w_hr_fwd;
  std::optional<mlir::MlirOp> grad_w_hr_rev;
};

// Recomputes forward activations for layer l during backpropagation.
// Uses chunked stablehlo.while loop execution with kLstmUnrollFactor for
// seq_len >= kLstmUnrollFactor to prevent O(T) graph bloat and reduce
// compilation overhead.
LayerRecomputeResult RecomputeLayerForward(
    mlir::MlirBuilder& builder, mlir::MlirOp layer_input_comp,
    mlir::MlirOp h_0_l, mlir::MlirOp c_0_l, mlir::MlirOp w_ih,
    mlir::MlirOp w_hh, std::optional<mlir::MlirOp> b_total,
    mlir::stablehlo::DotDimensionNumbersAttr ih_fwd_dot_dims,
    mlir::stablehlo::DotDimensionNumbersAttr hh_fwd_dot_dims,
    const int64_t seq_len, const int64_t batch, const int64_t hidden,
    const int64_t out_h, std::optional<mlir::MlirOp> w_hr,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> hr_dot_dims,
    const bool batch_first, mlir::stablehlo::Precision precision) {
  const int64_t in_dim = GetTensorTypeOrDie(w_ih).getDimSize(1);
  mlir::MlirOp w_ih_clumped = ClumpWeight3H(builder, w_ih, hidden, in_dim);
  mlir::MlirOp w_hh_clumped = ClumpWeight3H(builder, w_hh, hidden, out_h);

  const mlir::Type acc_dtype = GetTensorTypeOrDie(c_0_l).getElementType();

  // Project sequence input: [T, B, in_dim] or [B, T, in_dim] x [4H, in_dim]^T
  mlir::MlirOp x_proj_seq = MixedPrecisionDotGeneral(
      layer_input_comp, w_ih_clumped, ih_fwd_dot_dims, acc_dtype, precision);

  if (b_total.has_value()) {
    mlir::MlirOp b_clumped = ClumpBias3H(builder, *b_total, hidden);
    mlir::MlirOp b_bcast = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(x_proj_seq), b_clumped, {2});
    x_proj_seq = mlir::stablehlo::Add(x_proj_seq, b_bcast);
  }

  LayerForwardState state;
  state.layer_input = layer_input_comp;

  const int64_t concat_dim = batch_first ? 1 : 0;

  if (seq_len < kLstmUnrollFactor) {
    mlir::MlirOp h_curr = h_0_l;
    mlir::MlirOp c_curr = c_0_l;

    state.h_prev_list.reserve(seq_len);
    state.c_prev_list.reserve(seq_len);
    state.cellgate_list.reserve(seq_len);
    state.c_curr_list.reserve(seq_len);
    state.tanh_c_list.reserve(seq_len);
    state.ifo_list.reserve(seq_len);

    std::vector<mlir::MlirOp> step_outputs;
    step_outputs.reserve(seq_len);

    for (int64_t t = 0; t < seq_len; ++t) {
      state.h_prev_list.push_back(h_curr);
      state.c_prev_list.push_back(c_curr);

      mlir::MlirOp x_proj_step =
          SliceStep2D(x_proj_seq, t, batch, 4 * hidden, batch_first);

      LstmStepResults step = ComputeLstmStep(
          x_proj_step, h_curr, c_curr, w_hh_clumped, hh_fwd_dot_dims, batch,
          hidden, w_hr, hr_dot_dims, precision);
      h_curr = step.h_next;
      c_curr = step.c_next;

      state.ifo_list.push_back(step.ifo_act);
      state.cellgate_list.push_back(step.cellgate);
      state.c_curr_list.push_back(step.c_next);
      state.tanh_c_list.push_back(step.tanh_c);

      step_outputs.push_back(ExpandStep3D(h_curr, batch, out_h, batch_first));
    }

    std::vector<mlir::MlirOp> h_prev_reshaped(seq_len);
    std::vector<mlir::MlirOp> c_prev_reshaped(seq_len);
    std::vector<mlir::MlirOp> ifo_reshaped(seq_len);
    std::vector<mlir::MlirOp> cellgate_reshaped(seq_len);
    std::vector<mlir::MlirOp> tanh_c_reshaped(seq_len);

    for (int64_t t = 0; t < seq_len; ++t) {
      h_prev_reshaped[t] =
          ExpandStep3D(state.h_prev_list[t], batch, out_h, batch_first);
      c_prev_reshaped[t] =
          ExpandStep3D(state.c_prev_list[t], batch, hidden, batch_first);
      ifo_reshaped[t] =
          ExpandStep3D(state.ifo_list[t], batch, 3 * hidden, batch_first);
      cellgate_reshaped[t] =
          ExpandStep3D(state.cellgate_list[t], batch, hidden, batch_first);
      tanh_c_reshaped[t] =
          ExpandStep3D(state.tanh_c_list[t], batch, hidden, batch_first);
    }

    state.h_prev_seq = ConcatDim(builder, h_prev_reshaped, concat_dim);
    state.c_prev_seq = ConcatDim(builder, c_prev_reshaped, concat_dim);
    state.ifo_seq = ConcatDim(builder, ifo_reshaped, concat_dim);
    state.cellgate_seq = ConcatDim(builder, cellgate_reshaped, concat_dim);
    state.tanh_c_seq = ConcatDim(builder, tanh_c_reshaped, concat_dim);

    return {std::move(state), ConcatDim(builder, step_outputs, concat_dim),
            h_curr, c_curr};
  }

  // Chunked execution with partial unrolling factor kLstmUnrollFactor using
  // stablehlo.while
  const int64_t num_chunks = seq_len / kLstmUnrollFactor;
  const int64_t chunked_steps = num_chunks * kLstmUnrollFactor;
  const int64_t rem_steps = seq_len % kLstmUnrollFactor;

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = x_proj_seq.getValue().getLoc();
  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType i64_scalar_type =
      mlir::RankedTensorType::get({}, i64);

  const mlir::Type acc_elem_type = GetTensorTypeOrDie(h_0_l).getElementType();

  const mlir::RankedTensorType y_type =
      batch_first ? mlir::RankedTensorType::get({batch, chunked_steps, out_h},
                                                acc_elem_type)
                  : mlir::RankedTensorType::get({chunked_steps, batch, out_h},
                                                acc_elem_type);
  const mlir::RankedTensorType ifo_type =
      batch_first ? mlir::RankedTensorType::get(
                        {batch, chunked_steps, 3 * hidden}, acc_elem_type)
                  : mlir::RankedTensorType::get(
                        {chunked_steps, batch, 3 * hidden}, acc_elem_type);
  const mlir::RankedTensorType h_buf_type =
      batch_first ? mlir::RankedTensorType::get({batch, chunked_steps, hidden},
                                                acc_elem_type)
                  : mlir::RankedTensorType::get({chunked_steps, batch, hidden},
                                                acc_elem_type);
  const mlir::RankedTensorType h_prev_buf_type =
      batch_first ? mlir::RankedTensorType::get({batch, chunked_steps, out_h},
                                                acc_elem_type)
                  : mlir::RankedTensorType::get({chunked_steps, batch, out_h},
                                                acc_elem_type);

  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0f, acc_elem_type);
  mlir::MlirOp y_init =
      mlir::stablehlo::BroadcastInDim(y_type, zero_scalar, {});
  mlir::MlirOp ifo_init =
      mlir::stablehlo::BroadcastInDim(ifo_type, zero_scalar, {});
  mlir::MlirOp cellgate_init =
      mlir::stablehlo::BroadcastInDim(h_buf_type, zero_scalar, {});
  mlir::MlirOp tanh_c_init =
      mlir::stablehlo::BroadcastInDim(h_buf_type, zero_scalar, {});
  mlir::MlirOp c_prev_init =
      mlir::stablehlo::BroadcastInDim(h_buf_type, zero_scalar, {});
  mlir::MlirOp h_prev_init =
      mlir::stablehlo::BroadcastInDim(h_prev_buf_type, zero_scalar, {});

  mlir::MlirOp step_idx_init = MakeScalarConstant(builder, 0, i64);

  const llvm::SmallVector<mlir::Type> loop_types = {
      i64_scalar_type, h_0_l.getType(), c_0_l.getType(), ifo_type, h_buf_type,
      h_buf_type,      h_buf_type,      h_prev_buf_type, y_type};
  const llvm::SmallVector<mlir::Value> loop_inits = {
      step_idx_init.getValue(), h_0_l.getValue(),
      c_0_l.getValue(),         ifo_init.getValue(),
      cellgate_init.getValue(), tanh_c_init.getValue(),
      c_prev_init.getValue(),   h_prev_init.getValue(),
      y_init.getValue()};

  auto while_op =
      mlir::stablehlo::WhileOp::create(op_builder, loc, loop_types, loop_inits);

  // Condition region: step_idx < chunked_steps
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

  // Body region: execute kLstmUnrollFactor unrolled recurrent steps per chunk
  mlir::Block* const body_block = op_builder.createBlock(&while_op.getBody());
  body_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(body_block);

  mlir::MlirOp body_step_idx(builder, body_block->getArgument(0));
  mlir::MlirOp body_h(builder, body_block->getArgument(1));
  mlir::MlirOp body_c(builder, body_block->getArgument(2));
  mlir::MlirOp body_ifo(builder, body_block->getArgument(3));
  mlir::MlirOp body_cellgate(builder, body_block->getArgument(4));
  mlir::MlirOp body_tanh_c(builder, body_block->getArgument(5));
  mlir::MlirOp body_c_prev(builder, body_block->getArgument(6));
  mlir::MlirOp body_h_prev(builder, body_block->getArgument(7));
  mlir::MlirOp body_y(builder, body_block->getArgument(8));

  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64);
  const llvm::SmallVector<mlir::MlirOp, 3> x_start_indices =
      batch_first ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, body_step_idx,
                                                       zero_i64}
                  : llvm::SmallVector<mlir::MlirOp, 3>{body_step_idx, zero_i64,
                                                       zero_i64};
  const llvm::SmallVector<int64_t, 3> x_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kLstmUnrollFactor, 4 * hidden}
          : llvm::SmallVector<int64_t, 3>{kLstmUnrollFactor, batch, 4 * hidden};
  mlir::MlirOp x_chunk =
      mlir::stablehlo::DynamicSlice(x_proj_seq, x_start_indices, x_slice_sizes);

  mlir::MlirOp curr_h = body_h;
  mlir::MlirOp curr_c = body_c;

  std::vector<mlir::MlirOp> chunk_h_prev(kLstmUnrollFactor);
  std::vector<mlir::MlirOp> chunk_c_prev(kLstmUnrollFactor);
  std::vector<mlir::MlirOp> chunk_cellgate(kLstmUnrollFactor);
  std::vector<mlir::MlirOp> chunk_tanh_c(kLstmUnrollFactor);
  std::vector<mlir::MlirOp> chunk_ifo(kLstmUnrollFactor);
  std::vector<mlir::MlirOp> chunk_y(kLstmUnrollFactor);

  for (int64_t k = 0; k < kLstmUnrollFactor; ++k) {
    chunk_h_prev[k] = ExpandStep3D(curr_h, batch, out_h, batch_first);
    chunk_c_prev[k] = ExpandStep3D(curr_c, batch, hidden, batch_first);

    mlir::MlirOp x_k_2d =
        SliceStep2D(x_chunk, k, batch, 4 * hidden, batch_first);

    LstmStepResults step =
        ComputeLstmStep(x_k_2d, curr_h, curr_c, w_hh_clumped, hh_fwd_dot_dims,
                        batch, hidden, w_hr, hr_dot_dims, precision);
    curr_h = step.h_next;
    curr_c = step.c_next;

    chunk_ifo[k] = ExpandStep3D(step.ifo_act, batch, 3 * hidden, batch_first);
    chunk_cellgate[k] = ExpandStep3D(step.cellgate, batch, hidden, batch_first);
    chunk_tanh_c[k] = ExpandStep3D(step.tanh_c, batch, hidden, batch_first);
    chunk_y[k] = ExpandStep3D(curr_h, batch, out_h, batch_first);
  }

  mlir::MlirOp chunk_ifo_out = ConcatDim(builder, chunk_ifo, concat_dim);
  mlir::MlirOp chunk_cellgate_out =
      ConcatDim(builder, chunk_cellgate, concat_dim);
  mlir::MlirOp chunk_tanh_c_out = ConcatDim(builder, chunk_tanh_c, concat_dim);
  mlir::MlirOp chunk_c_prev_out = ConcatDim(builder, chunk_c_prev, concat_dim);
  mlir::MlirOp chunk_h_prev_out = ConcatDim(builder, chunk_h_prev, concat_dim);
  mlir::MlirOp chunk_y_out = ConcatDim(builder, chunk_y, concat_dim);

  const llvm::SmallVector<mlir::MlirOp, 3> update_start_indices =
      batch_first ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, body_step_idx,
                                                       zero_i64}
                  : llvm::SmallVector<mlir::MlirOp, 3>{body_step_idx, zero_i64,
                                                       zero_i64};
  mlir::MlirOp next_ifo = mlir::stablehlo::DynamicUpdateSlice(
      body_ifo, chunk_ifo_out, update_start_indices);
  mlir::MlirOp next_cellgate = mlir::stablehlo::DynamicUpdateSlice(
      body_cellgate, chunk_cellgate_out, update_start_indices);
  mlir::MlirOp next_tanh_c = mlir::stablehlo::DynamicUpdateSlice(
      body_tanh_c, chunk_tanh_c_out, update_start_indices);
  mlir::MlirOp next_c_prev = mlir::stablehlo::DynamicUpdateSlice(
      body_c_prev, chunk_c_prev_out, update_start_indices);
  mlir::MlirOp next_h_prev = mlir::stablehlo::DynamicUpdateSlice(
      body_h_prev, chunk_h_prev_out, update_start_indices);
  mlir::MlirOp next_y = mlir::stablehlo::DynamicUpdateSlice(
      body_y, chunk_y_out, update_start_indices);

  mlir::MlirOp k_factor_op =
      MakeScalarConstant(builder, kLstmUnrollFactor, i64);
  mlir::MlirOp next_step_idx = mlir::stablehlo::Add(body_step_idx, k_factor_op);

  const llvm::SmallVector<mlir::Value> next_state = {
      next_step_idx.getValue(), curr_h.getValue(),
      curr_c.getValue(),        next_ifo.getValue(),
      next_cellgate.getValue(), next_tanh_c.getValue(),
      next_c_prev.getValue(),   next_h_prev.getValue(),
      next_y.getValue()};
  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_state);

  op_builder.setInsertionPointAfter(while_op);

  mlir::MlirOp final_chunk_h(builder, while_op.getResult(1));
  mlir::MlirOp final_chunk_c(builder, while_op.getResult(2));
  mlir::MlirOp ifo_chunked(builder, while_op.getResult(3));
  mlir::MlirOp cellgate_chunked(builder, while_op.getResult(4));
  mlir::MlirOp tanh_c_chunked(builder, while_op.getResult(5));
  mlir::MlirOp c_prev_chunked(builder, while_op.getResult(6));
  mlir::MlirOp h_prev_chunked(builder, while_op.getResult(7));
  mlir::MlirOp y_chunked(builder, while_op.getResult(8));

  if (rem_steps == 0) {
    state.ifo_seq = ifo_chunked;
    state.cellgate_seq = cellgate_chunked;
    state.tanh_c_seq = tanh_c_chunked;
    state.c_prev_seq = c_prev_chunked;
    state.h_prev_seq = h_prev_chunked;
    return {std::move(state), y_chunked, final_chunk_h, final_chunk_c};
  }

  // Handle remainder steps (t = chunked_steps ... seq_len - 1)
  mlir::MlirOp rem_h = final_chunk_h;
  mlir::MlirOp rem_c = final_chunk_c;
  std::vector<mlir::MlirOp> rem_ifo(rem_steps);
  std::vector<mlir::MlirOp> rem_cellgate(rem_steps);
  std::vector<mlir::MlirOp> rem_tanh_c(rem_steps);
  std::vector<mlir::MlirOp> rem_c_prev(rem_steps);
  std::vector<mlir::MlirOp> rem_h_prev(rem_steps);
  std::vector<mlir::MlirOp> rem_y(rem_steps);

  state.h_prev_list.resize(seq_len);
  state.c_prev_list.resize(seq_len);
  state.cellgate_list.resize(seq_len);
  state.c_curr_list.resize(seq_len);
  state.tanh_c_list.resize(seq_len);
  state.ifo_list.resize(seq_len);

  for (int64_t t = chunked_steps; t < seq_len; ++t) {
    const int64_t rem_idx = t - chunked_steps;
    state.h_prev_list[t] = rem_h;
    state.c_prev_list[t] = rem_c;
    rem_h_prev[rem_idx] = ExpandStep3D(rem_h, batch, out_h, batch_first);
    rem_c_prev[rem_idx] = ExpandStep3D(rem_c, batch, hidden, batch_first);

    mlir::MlirOp x_t_2d =
        SliceStep2D(x_proj_seq, t, batch, 4 * hidden, batch_first);

    LstmStepResults step =
        ComputeLstmStep(x_t_2d, rem_h, rem_c, w_hh_clumped, hh_fwd_dot_dims,
                        batch, hidden, w_hr, hr_dot_dims, precision);
    rem_h = step.h_next;
    rem_c = step.c_next;

    state.ifo_list[t] = step.ifo_act;
    state.cellgate_list[t] = step.cellgate;
    state.c_curr_list[t] = step.c_next;
    state.tanh_c_list[t] = step.tanh_c;

    rem_ifo[rem_idx] =
        ExpandStep3D(step.ifo_act, batch, 3 * hidden, batch_first);
    rem_cellgate[rem_idx] =
        ExpandStep3D(step.cellgate, batch, hidden, batch_first);
    rem_tanh_c[rem_idx] = ExpandStep3D(step.tanh_c, batch, hidden, batch_first);
    rem_y[rem_idx] = ExpandStep3D(rem_h, batch, out_h, batch_first);
  }

  mlir::MlirOp rem_ifo_seq = ConcatDim(builder, rem_ifo, concat_dim);
  mlir::MlirOp rem_cellgate_seq = ConcatDim(builder, rem_cellgate, concat_dim);
  mlir::MlirOp rem_tanh_c_seq = ConcatDim(builder, rem_tanh_c, concat_dim);
  mlir::MlirOp rem_c_prev_seq = ConcatDim(builder, rem_c_prev, concat_dim);
  mlir::MlirOp rem_h_prev_seq = ConcatDim(builder, rem_h_prev, concat_dim);
  mlir::MlirOp rem_y_seq = ConcatDim(builder, rem_y, concat_dim);

  state.ifo_seq = ConcatDim(builder, {ifo_chunked, rem_ifo_seq}, concat_dim);
  state.cellgate_seq =
      ConcatDim(builder, {cellgate_chunked, rem_cellgate_seq}, concat_dim);
  state.tanh_c_seq =
      ConcatDim(builder, {tanh_c_chunked, rem_tanh_c_seq}, concat_dim);
  state.c_prev_seq =
      ConcatDim(builder, {c_prev_chunked, rem_c_prev_seq}, concat_dim);
  state.h_prev_seq =
      ConcatDim(builder, {h_prev_chunked, rem_h_prev_seq}, concat_dim);
  mlir::MlirOp final_y = ConcatDim(builder, {y_chunked, rem_y_seq}, concat_dim);

  return {std::move(state), final_y, rem_h, rem_c};
}

// Computes backward gradients through time (BPTT) for a single LSTM layer:
//
// 1. REVERSE TIME RECURRENCE (t = seq_len - 1 down to 0):
//    At each step t:
//    - Accumulate hidden gradient from above and from recurrent step t+1:
//        delta_h = grad_output_t + delta_h_{rec, t+1} (or grad_hy_l for final
//        step)
//    - Compute cell gradient:
//        delta_o = delta_h * tanh(c_t)
//        delta_c = delta_c_{next} + delta_h * o_t * (1 - tanh(c_t)^2)
//    - Propagate cell gradient to previous step t-1:
//        delta_c_{next} = delta_c * f_t
//    - 3H Gate Clumping:
//        Upstream adjoints for [i, f, o] share the common sigmoid derivative
//        form delta * act * (1 - act). We pack upstream adjoints into a
//        contiguous [B, 3*H] tensor:
//          delta_in_ifo = [delta_c * g_t,  delta_c * c_{t-1},  delta_o]
//        and evaluate a single vectorized SigmoidBackward over [B, 3*H]:
//          delta_pre_ifo = SigmoidBackward(delta_in_ifo, ifo_t, 1.0)
//        Cell candidate gradient evaluates single TanhBackward over [B, H]:
//          delta_pre_g = TanhBackward(delta_c * i_t, g_t, 1.0)
//        Concatenate into delta_pre_t [B, 4*H] via a fast 2-way concat ([ifo,
//        g]) instead of 4-way concat.
//    - Compute recurrent gradient for step t-1 using 3H clumped W_hh:
//        delta_h_{rec, t} = delta_pre_t @ W_hh_clumped
//
// 2. BATCHED WEIGHT & BIAS CONTRACTIONS & UNCLUMPING:
//    Flatten delta_pre_seq into [T*B, 4*H]:
//    - Input sequence gradient:
//        grad_x = delta_pre_2d @ W_ih_clumped     [T*B, in_dim] -> [T, B,
//        in_dim]
//    - Input weight gradient (unclumped back to [i, f, g, o]):
//        grad_w_ih = UnclumpWeight3H(delta_pre_2d^T @ layer_input_2d)  [4*H,
//        in_dim]
//    - Recurrent weight gradient (unclumped back to [i, f, g, o]):
//        grad_w_hh = UnclumpWeight3H(delta_pre_2d^T @ h_prev_2d)       [4*H, H]
//    - Bias gradient (unclumped back to [i, f, g, o]):
//        grad_bias = UnclumpBias3H(sum_{t, b}(delta_pre_seq))         [4*H]
// Static unrolling helper for unidirectional LSTM backward pass for short
// sequences (seq_len < kLstmUnrollFactor).
//
// In short sequences, while-loop carry tuple state and dynamic slicing
// introduce runtime dispatch overhead that exceeds the benefits of chunking. We
// statically unroll BPTT across all timesteps t = seq_len - 1 down to 0:
// - Computes adjoints with respect to activations in layer_state.
// - Directly accumulates gradients into full-sequence concatenated adjoint
// buffers.
// - Performs full-sequence 2D dot generals to accumulate W_ih and W_hh
// gradients across all steps.
template <typename ToOutFn, typename ReduceBuilderFn>
LayerBackwardOutputs ComputeLayerBackwardStatic(
    mlir::MlirBuilder& builder, const LayerForwardState& layer_state,
    mlir::MlirOp grad_x_curr, mlir::MlirOp grad_hy_l, mlir::MlirOp grad_cy_l,
    mlir::MlirOp w_ih_clumped, mlir::MlirOp w_hh_clumped, const bool has_biases,
    mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    mlir::stablehlo::DotDimensionNumbersAttr bwd_weight_dot_dims,
    mlir::stablehlo::DotDimensionNumbersAttr grad_w_dot_dims,
    const int64_t seq_len, const int64_t batch, const int64_t hidden,
    const int64_t in_dim, const int64_t out_h, ToOutFn to_out,
    std::optional<mlir::MlirOp> w_hr,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> bwd_hr_dot_dims,
    const bool batch_first, mlir::stablehlo::Precision precision) {
  const int64_t concat_dim = batch_first ? 1 : 0;
  mlir::MlirOp layer_in = layer_state.layer_input;
  mlir::MlirOp h_prev = layer_state.h_prev_seq;

  mlir::MlirOp delta_c_next = grad_cy_l;
  mlir::MlirOp delta_h_rec = grad_hy_l;
  std::vector<mlir::MlirOp> delta_pre_steps(seq_len);
  std::vector<mlir::MlirOp> delta_h_steps(w_hr.has_value() ? seq_len : 0);
  std::vector<mlir::MlirOp> r_steps(w_hr.has_value() ? seq_len : 0);

  for (int64_t t = seq_len - 1; t >= 0; --t) {
    mlir::MlirOp grad_from_above_t =
        SliceStep2D(grad_x_curr, t, batch, out_h, batch_first);

    mlir::MlirOp delta_h =
        (t == seq_len - 1)
            ? mlir::stablehlo::Add(grad_from_above_t, grad_hy_l)
            : mlir::stablehlo::Add(grad_from_above_t, delta_h_rec);

    mlir::MlirOp tanh_c_val = GetStepActivation(
        builder, layer_state.tanh_c_list, layer_state.tanh_c_seq, t, batch,
        hidden, batch_first);
    mlir::MlirOp ifo_val =
        GetStepActivation(builder, layer_state.ifo_list, layer_state.ifo_seq, t,
                          batch, 3 * hidden, batch_first);

    BackwardStepResult step = ComputeBackwardStep(
        builder, delta_h, delta_c_next, tanh_c_val,
        GetStepActivation(builder, layer_state.c_prev_list,
                          layer_state.c_prev_seq, t, batch, hidden,
                          batch_first),
        GetStepActivation(builder, layer_state.cellgate_list,
                          layer_state.cellgate_seq, t, batch, hidden,
                          batch_first),
        ifo_val, w_hh_clumped, bwd_weight_dot_dims, batch, hidden, w_hr,
        bwd_hr_dot_dims, precision);

    delta_c_next = step.delta_c_next;
    delta_h_rec = step.delta_h_rec;
    delta_pre_steps[t] =
        ExpandStep3D(step.delta_pre, batch, 4 * hidden, batch_first);

    if (w_hr.has_value()) {
      delta_h_steps[t] = ExpandStep3D(delta_h, batch, out_h, batch_first);
      mlir::MlirOp o_val = mlir::stablehlo::Slice(ifo_val, {0, 2 * hidden},
                                                  {batch, 3 * hidden}, {1, 1});
      mlir::MlirOp r_val = mlir::stablehlo::Mul(o_val, tanh_c_val);
      r_steps[t] = ExpandStep3D(r_val, batch, hidden, batch_first);
    }
  }

  mlir::MlirOp grad_h0_layer =
      mlir::stablehlo::Reshape(to_out(delta_h_rec), {1, batch, out_h});
  mlir::MlirOp grad_c0_layer =
      mlir::stablehlo::Reshape(to_out(delta_c_next), {1, batch, hidden});

  mlir::MlirOp delta_pre_seq = ConcatDim(builder, delta_pre_steps, concat_dim);
  mlir::MlirOp delta_pre_2d =
      mlir::stablehlo::Reshape(delta_pre_seq, {seq_len * batch, 4 * hidden});

  const mlir::Type acc_dtype =
      GetTensorTypeOrDie(delta_pre_2d).getElementType();
  const mlir::Type out_dtype =
      GetTensorTypeOrDie(w_ih_clumped).getElementType();

  mlir::MlirOp grad_x_2d = MixedPrecisionDotGeneral(
      delta_pre_2d, w_ih_clumped, bwd_weight_dot_dims, acc_dtype, precision);
  mlir::MlirOp grad_x_layer =
      batch_first
          ? mlir::stablehlo::Reshape(grad_x_2d, {batch, seq_len, in_dim})
          : mlir::stablehlo::Reshape(grad_x_2d, {seq_len, batch, in_dim});

  mlir::MlirOp layer_in_2d =
      mlir::stablehlo::Reshape(layer_in, {seq_len * batch, in_dim});
  mlir::MlirOp grad_w_ih_clumped =
      MixedPrecisionDotGeneral(delta_pre_2d, layer_in_2d, grad_w_dot_dims,
                               acc_dtype, out_dtype, precision);
  mlir::MlirOp grad_w_ih =
      UnclumpWeight3H(builder, grad_w_ih_clumped, hidden, in_dim);

  mlir::MlirOp h_prev_2d =
      mlir::stablehlo::Reshape(h_prev, {seq_len * batch, out_h});
  mlir::MlirOp grad_w_hh_clumped =
      MixedPrecisionDotGeneral(delta_pre_2d, h_prev_2d, grad_w_dot_dims,
                               acc_dtype, out_dtype, precision);
  mlir::MlirOp grad_w_hh =
      UnclumpWeight3H(builder, grad_w_hh_clumped, hidden, out_h);

  std::optional<mlir::MlirOp> grad_bias;
  if (has_biases) {
    mlir::MlirOp grad_b_clumped = mlir::stablehlo::Reduce(
        builder, delta_pre_seq, zero_const, sum_reduce_builder, {0, 1})[0];
    mlir::MlirOp grad_b = UnclumpBias3H(builder, grad_b_clumped, hidden);
    grad_bias = to_out(grad_b);
  }

  std::optional<mlir::MlirOp> grad_w_hr;
  if (w_hr.has_value()) {
    mlir::MlirOp delta_h_seq = ConcatDim(builder, delta_h_steps, concat_dim);
    mlir::MlirOp delta_h_2d =
        mlir::stablehlo::Reshape(delta_h_seq, {seq_len * batch, out_h});
    mlir::MlirOp r_seq = ConcatDim(builder, r_steps, concat_dim);
    mlir::MlirOp r_2d =
        mlir::stablehlo::Reshape(r_seq, {seq_len * batch, hidden});
    mlir::MlirOp grad_w_hr_val = MixedPrecisionDotGeneral(
        delta_h_2d, r_2d, grad_w_dot_dims, acc_dtype, out_dtype, precision);
    grad_w_hr = to_out(grad_w_hr_val);
  }

  return {grad_x_layer,      grad_h0_layer, grad_c0_layer, to_out(grad_w_ih),
          to_out(grad_w_hh), grad_bias,     grad_w_hr};
}

// Chunked streaming helper for unidirectional LSTM backward pass (seq_len >=
// kLstmUnrollFactor).
//
// Key Optimizations:
// 1. Partial unrolling of kLstmUnrollFactor timesteps inside a single StableHLO
// while-loop.
// 2. Dynamic slicing: Slices only kLstmUnrollFactor steps of forward
// activations per iteration,
//    minimizing register pressure and keeping the active working set in TPU
//    on-chip memory.
// 3. Streaming parameter gradient accumulation: Directly accumulates W_ih,
// W_hh, bias, and
//    W_hr gradients inside the loop carry tuple. Eliminates full-sequence
//    intermediate adjoint buffers in HBM, reducing peak memory from O(T) to
//    O(kLstmUnrollFactor).
// 4. Tail remainder handling: Statically unrolls any leftover steps (seq_len %
// kLstmUnrollFactor)
//    upfront at sequence boundary t = seq_len - 1 down to chunked_steps,
//    initializing the loop carry states.
template <typename ToOutFn, typename ReduceBuilderFn>
LayerBackwardOutputs ComputeLayerBackwardChunked(
    mlir::MlirBuilder& builder, const LayerForwardState& layer_state,
    mlir::MlirOp grad_x_curr, mlir::MlirOp grad_hy_l, mlir::MlirOp grad_cy_l,
    mlir::MlirOp w_ih_clumped, mlir::MlirOp w_hh_clumped, const bool has_biases,
    mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    mlir::stablehlo::DotDimensionNumbersAttr bwd_weight_dot_dims,
    mlir::stablehlo::DotDimensionNumbersAttr grad_w_dot_dims,
    const int64_t seq_len, const int64_t batch, const int64_t hidden,
    const int64_t in_dim, const int64_t out_h, ToOutFn to_out,
    std::optional<mlir::MlirOp> w_hr,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> bwd_hr_dot_dims,
    const bool batch_first, mlir::stablehlo::Precision precision) {
  const int64_t concat_dim = batch_first ? 1 : 0;
  // --- STREAMING / CHUNKED BPTT EXECUTION (kLstmUnrollFactor) ---
  const int64_t num_chunks = seq_len / kLstmUnrollFactor;
  const int64_t chunked_steps = num_chunks * kLstmUnrollFactor;
  const int64_t rem_steps = seq_len % kLstmUnrollFactor;

  mlir::MlirOp delta_c_next = grad_cy_l;
  mlir::MlirOp delta_h_rec = grad_hy_l;

  mlir::MlirOp layer_in = layer_state.layer_input;
  mlir::MlirOp h_prev = layer_state.h_prev_seq;

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = grad_x_curr.getValue().getLoc();
  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType i64_scalar_type =
      mlir::RankedTensorType::get({}, i64);

  const mlir::Type acc_elem_type =
      GetTensorTypeOrDie(delta_h_rec).getElementType();
  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0f, acc_elem_type);

  mlir::MlirOp grad_w_ih_init;
  mlir::MlirOp grad_w_hh_init;
  mlir::MlirOp grad_b_init;
  mlir::MlirOp grad_w_hr_init;
  mlir::MlirOp rem_grad_x;

  if (rem_steps > 0) {
    std::vector<mlir::MlirOp> delta_pre_rem_steps(rem_steps);
    std::vector<mlir::MlirOp> delta_h_rem_steps(w_hr.has_value() ? rem_steps
                                                                 : 0);
    std::vector<mlir::MlirOp> r_rem_steps(w_hr.has_value() ? rem_steps : 0);

    for (int64_t t = seq_len - 1; t >= chunked_steps; --t) {
      const int64_t rem_idx = t - chunked_steps;
      mlir::MlirOp grad_from_above_t =
          SliceStep2D(grad_x_curr, t, batch, out_h, batch_first);

      mlir::MlirOp delta_h =
          mlir::stablehlo::Add(grad_from_above_t, delta_h_rec);

      mlir::MlirOp tanh_c_val = GetStepActivation(
          builder, layer_state.tanh_c_list, layer_state.tanh_c_seq, t, batch,
          hidden, batch_first);
      mlir::MlirOp ifo_val =
          GetStepActivation(builder, layer_state.ifo_list, layer_state.ifo_seq,
                            t, batch, 3 * hidden, batch_first);

      BackwardStepResult step = ComputeBackwardStep(
          builder, delta_h, delta_c_next, tanh_c_val,
          GetStepActivation(builder, layer_state.c_prev_list,
                            layer_state.c_prev_seq, t, batch, hidden,
                            batch_first),
          GetStepActivation(builder, layer_state.cellgate_list,
                            layer_state.cellgate_seq, t, batch, hidden,
                            batch_first),
          ifo_val, w_hh_clumped, bwd_weight_dot_dims, batch, hidden, w_hr,
          bwd_hr_dot_dims, precision);

      delta_c_next = step.delta_c_next;
      delta_h_rec = step.delta_h_rec;
      delta_pre_rem_steps[rem_idx] =
          ExpandStep3D(step.delta_pre, batch, 4 * hidden, batch_first);

      if (w_hr.has_value()) {
        delta_h_rem_steps[rem_idx] =
            ExpandStep3D(delta_h, batch, out_h, batch_first);
        mlir::MlirOp o_val = mlir::stablehlo::Slice(
            ifo_val, {0, 2 * hidden}, {batch, 3 * hidden}, {1, 1});
        mlir::MlirOp r_val = mlir::stablehlo::Mul(o_val, tanh_c_val);
        r_rem_steps[rem_idx] = ExpandStep3D(r_val, batch, hidden, batch_first);
      }
    }

    mlir::MlirOp delta_pre_rem =
        ConcatDim(builder, delta_pre_rem_steps, concat_dim);
    mlir::MlirOp delta_pre_rem_2d = mlir::stablehlo::Reshape(
        delta_pre_rem, {rem_steps * batch, 4 * hidden});

    mlir::MlirOp rem_layer_in =
        batch_first
            ? mlir::stablehlo::Slice(layer_in, {0, chunked_steps, 0},
                                     {batch, seq_len, in_dim}, {1, 1, 1})
            : mlir::stablehlo::Slice(layer_in, {chunked_steps, 0, 0},
                                     {seq_len, batch, in_dim}, {1, 1, 1});
    mlir::MlirOp rem_layer_in_2d =
        mlir::stablehlo::Reshape(rem_layer_in, {rem_steps * batch, in_dim});
    const mlir::Type acc_dtype =
        GetTensorTypeOrDie(delta_pre_rem_2d).getElementType();
    const mlir::Type out_dtype =
        GetTensorTypeOrDie(w_ih_clumped).getElementType();

    grad_w_ih_init = MixedPrecisionDotGeneral(delta_pre_rem_2d, rem_layer_in_2d,
                                              grad_w_dot_dims, acc_dtype,
                                              out_dtype, precision);

    mlir::MlirOp rem_h_prev =
        batch_first
            ? mlir::stablehlo::Slice(h_prev, {0, chunked_steps, 0},
                                     {batch, seq_len, out_h}, {1, 1, 1})
            : mlir::stablehlo::Slice(h_prev, {chunked_steps, 0, 0},
                                     {seq_len, batch, out_h}, {1, 1, 1});
    mlir::MlirOp rem_h_prev_2d =
        mlir::stablehlo::Reshape(rem_h_prev, {rem_steps * batch, out_h});
    grad_w_hh_init = MixedPrecisionDotGeneral(delta_pre_rem_2d, rem_h_prev_2d,
                                              grad_w_dot_dims, acc_dtype,
                                              out_dtype, precision);

    if (has_biases) {
      grad_b_init = mlir::stablehlo::Reduce(builder, delta_pre_rem, zero_const,
                                            sum_reduce_builder, {0, 1})[0];
    } else {
      grad_b_init = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({4 * hidden}, acc_elem_type), zero_scalar,
          {});
    }

    if (w_hr.has_value()) {
      mlir::MlirOp delta_h_rem =
          ConcatDim(builder, delta_h_rem_steps, concat_dim);
      mlir::MlirOp delta_h_rem_2d =
          mlir::stablehlo::Reshape(delta_h_rem, {rem_steps * batch, out_h});
      mlir::MlirOp r_rem = ConcatDim(builder, r_rem_steps, concat_dim);
      mlir::MlirOp r_rem_2d =
          mlir::stablehlo::Reshape(r_rem, {rem_steps * batch, hidden});
      grad_w_hr_init =
          MixedPrecisionDotGeneral(delta_h_rem_2d, r_rem_2d, grad_w_dot_dims,
                                   acc_dtype, out_dtype, precision);
    }

    mlir::MlirOp rem_grad_x_2d =
        MixedPrecisionDotGeneral(delta_pre_rem_2d, w_ih_clumped,
                                 bwd_weight_dot_dims, acc_dtype, precision);
    rem_grad_x = batch_first ? mlir::stablehlo::Reshape(
                                   rem_grad_x_2d, {batch, rem_steps, in_dim})
                             : mlir::stablehlo::Reshape(
                                   rem_grad_x_2d, {rem_steps, batch, in_dim});
  } else {
    grad_w_ih_init = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({4 * hidden, in_dim}, acc_elem_type),
        zero_scalar, {});
    grad_w_hh_init = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({4 * hidden, out_h}, acc_elem_type),
        zero_scalar, {});
    grad_b_init = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({4 * hidden}, acc_elem_type), zero_scalar,
        {});
    if (w_hr.has_value()) {
      grad_w_hr_init = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({out_h, hidden}, acc_elem_type),
          zero_scalar, {});
    }
  }

  const mlir::RankedTensorType grad_x_chunked_type =
      batch_first ? mlir::RankedTensorType::get({batch, chunked_steps, in_dim},
                                                acc_elem_type)
                  : mlir::RankedTensorType::get({chunked_steps, batch, in_dim},
                                                acc_elem_type);
  mlir::MlirOp grad_x_chunked_init =
      mlir::stablehlo::BroadcastInDim(grad_x_chunked_type, zero_scalar, {});

  mlir::MlirOp step_idx_init = MakeScalarConstant(builder, 0, i64);

  llvm::SmallVector<mlir::Type> loop_types = {
      i64_scalar_type,     delta_h_rec.getType(),    delta_c_next.getType(),
      grad_x_chunked_type, grad_w_ih_init.getType(), grad_w_hh_init.getType()};
  llvm::SmallVector<mlir::Value> loop_inits = {
      step_idx_init.getValue(),  delta_h_rec.getValue(),
      delta_c_next.getValue(),   grad_x_chunked_init.getValue(),
      grad_w_ih_init.getValue(), grad_w_hh_init.getValue()};

  if (has_biases) {
    loop_types.push_back(grad_b_init.getType());
    loop_inits.push_back(grad_b_init.getValue());
  }
  if (w_hr.has_value()) {
    loop_types.push_back(grad_w_hr_init.getType());
    loop_inits.push_back(grad_w_hr_init.getValue());
  }

  auto while_op =
      mlir::stablehlo::WhileOp::create(op_builder, loc, loop_types, loop_inits);

  // Cond region: step_idx < chunked_steps
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

  // Body region: processes kLstmUnrollFactor steps in reverse order with
  // streaming accumulation
  mlir::Block* const body_block = op_builder.createBlock(&while_op.getBody());
  body_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(body_block);

  mlir::MlirOp body_step_idx(builder, body_block->getArgument(0));
  mlir::MlirOp body_delta_h_rec(builder, body_block->getArgument(1));
  mlir::MlirOp body_delta_c_next(builder, body_block->getArgument(2));
  mlir::MlirOp body_grad_x_chunked(builder, body_block->getArgument(3));
  mlir::MlirOp body_grad_w_ih(builder, body_block->getArgument(4));
  mlir::MlirOp body_grad_w_hh(builder, body_block->getArgument(5));
  int arg_idx = 6;
  mlir::MlirOp body_grad_b;
  if (has_biases) {
    body_grad_b = mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
  }
  mlir::MlirOp body_grad_w_hr;
  if (w_hr.has_value()) {
    body_grad_w_hr = mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
  }

  // Chronological start timestep: chunk_start_t = (chunked_steps -
  // kLstmUnrollFactor) - step_idx
  mlir::MlirOp const_chunk_limit =
      MakeScalarConstant(builder, chunked_steps - kLstmUnrollFactor, i64);
  mlir::MlirOp chunk_start_t =
      mlir::stablehlo::Subtract(const_chunk_limit, body_step_idx);

  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64);
  const llvm::SmallVector<mlir::MlirOp, 3> chunk_start_indices =
      batch_first ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, chunk_start_t,
                                                       zero_i64}
                  : llvm::SmallVector<mlir::MlirOp, 3>{chunk_start_t, zero_i64,
                                                       zero_i64};
  const llvm::SmallVector<int64_t, 3> out_h_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kLstmUnrollFactor, out_h}
          : llvm::SmallVector<int64_t, 3>{kLstmUnrollFactor, batch, out_h};
  const llvm::SmallVector<int64_t, 3> hidden_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kLstmUnrollFactor, hidden}
          : llvm::SmallVector<int64_t, 3>{kLstmUnrollFactor, batch, hidden};
  const llvm::SmallVector<int64_t, 3> ifo_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kLstmUnrollFactor, 3 * hidden}
          : llvm::SmallVector<int64_t, 3>{kLstmUnrollFactor, batch, 3 * hidden};
  const llvm::SmallVector<int64_t, 3> in_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kLstmUnrollFactor, in_dim}
          : llvm::SmallVector<int64_t, 3>{kLstmUnrollFactor, batch, in_dim};

  mlir::MlirOp ifo_seq = layer_state.ifo_seq;
  mlir::MlirOp cellgate_seq = layer_state.cellgate_seq;
  mlir::MlirOp tanh_c_seq = layer_state.tanh_c_seq;
  mlir::MlirOp c_prev_seq = layer_state.c_prev_seq;

  mlir::MlirOp grad_x_chunk = mlir::stablehlo::DynamicSlice(
      grad_x_curr, chunk_start_indices, out_h_slice_sizes);
  mlir::MlirOp ifo_chunk = mlir::stablehlo::DynamicSlice(
      ifo_seq, chunk_start_indices, ifo_slice_sizes);
  mlir::MlirOp cellgate_chunk = mlir::stablehlo::DynamicSlice(
      cellgate_seq, chunk_start_indices, hidden_slice_sizes);
  mlir::MlirOp tanh_c_chunk = mlir::stablehlo::DynamicSlice(
      tanh_c_seq, chunk_start_indices, hidden_slice_sizes);
  mlir::MlirOp c_prev_chunk = mlir::stablehlo::DynamicSlice(
      c_prev_seq, chunk_start_indices, hidden_slice_sizes);

  mlir::MlirOp cur_delta_h_rec = body_delta_h_rec;
  mlir::MlirOp cur_delta_c_next = body_delta_c_next;

  std::vector<mlir::MlirOp> delta_pre_chunk_steps(kLstmUnrollFactor);
  std::vector<mlir::MlirOp> delta_h_chunk_steps(
      w_hr.has_value() ? kLstmUnrollFactor : 0);
  std::vector<mlir::MlirOp> r_chunk_steps(w_hr.has_value() ? kLstmUnrollFactor
                                                           : 0);

  for (int64_t k = kLstmUnrollFactor - 1; k >= 0; --k) {
    mlir::MlirOp grad_from_above_k =
        SliceStep2D(grad_x_chunk, k, batch, out_h, batch_first);

    mlir::MlirOp delta_h =
        mlir::stablehlo::Add(grad_from_above_k, cur_delta_h_rec);

    mlir::MlirOp tanh_c_k =
        SliceStep2D(tanh_c_chunk, k, batch, hidden, batch_first);
    mlir::MlirOp c_prev_k =
        SliceStep2D(c_prev_chunk, k, batch, hidden, batch_first);
    mlir::MlirOp g_k =
        SliceStep2D(cellgate_chunk, k, batch, hidden, batch_first);
    mlir::MlirOp ifo_k =
        SliceStep2D(ifo_chunk, k, batch, 3 * hidden, batch_first);

    BackwardStepResult step = ComputeBackwardStep(
        builder, delta_h, cur_delta_c_next, tanh_c_k, c_prev_k, g_k, ifo_k,
        w_hh_clumped, bwd_weight_dot_dims, batch, hidden, w_hr, bwd_hr_dot_dims,
        precision);

    cur_delta_c_next = step.delta_c_next;
    cur_delta_h_rec = step.delta_h_rec;
    delta_pre_chunk_steps[k] =
        ExpandStep3D(step.delta_pre, batch, 4 * hidden, batch_first);

    if (w_hr.has_value()) {
      delta_h_chunk_steps[k] = ExpandStep3D(delta_h, batch, out_h, batch_first);
      mlir::MlirOp o_k = mlir::stablehlo::Slice(ifo_k, {0, 2 * hidden},
                                                {batch, 3 * hidden}, {1, 1});
      mlir::MlirOp r_k = mlir::stablehlo::Mul(o_k, tanh_c_k);
      r_chunk_steps[k] = ExpandStep3D(r_k, batch, hidden, batch_first);
    }
  }

  mlir::MlirOp delta_pre_chunk =
      ConcatDim(builder, delta_pre_chunk_steps, concat_dim);
  mlir::MlirOp delta_pre_chunk_2d = mlir::stablehlo::Reshape(
      delta_pre_chunk, {kLstmUnrollFactor * batch, 4 * hidden});

  const mlir::Type acc_dtype =
      GetTensorTypeOrDie(delta_pre_chunk_2d).getElementType();
  const mlir::Type out_dtype =
      GetTensorTypeOrDie(w_ih_clumped).getElementType();

  // Streaming accumulation: Input weight gradient
  mlir::MlirOp layer_in_chunk = mlir::stablehlo::DynamicSlice(
      layer_in, chunk_start_indices, in_slice_sizes);
  mlir::MlirOp layer_in_chunk_2d = mlir::stablehlo::Reshape(
      layer_in_chunk, {kLstmUnrollFactor * batch, in_dim});
  mlir::MlirOp chunk_grad_w_ih = MixedPrecisionDotGeneral(
      delta_pre_chunk_2d, layer_in_chunk_2d, grad_w_dot_dims, acc_dtype,
      out_dtype, precision);
  mlir::MlirOp next_grad_w_ih =
      mlir::stablehlo::Add(body_grad_w_ih, chunk_grad_w_ih);

  // Streaming accumulation: Recurrent weight gradient
  mlir::MlirOp h_prev_chunk = mlir::stablehlo::DynamicSlice(
      h_prev, chunk_start_indices, out_h_slice_sizes);
  mlir::MlirOp h_prev_chunk_2d = mlir::stablehlo::Reshape(
      h_prev_chunk, {kLstmUnrollFactor * batch, out_h});
  mlir::MlirOp chunk_grad_w_hh = MixedPrecisionDotGeneral(
      delta_pre_chunk_2d, h_prev_chunk_2d, grad_w_dot_dims, acc_dtype,
      out_dtype, precision);
  mlir::MlirOp next_grad_w_hh =
      mlir::stablehlo::Add(body_grad_w_hh, chunk_grad_w_hh);

  // Streaming accumulation: Bias gradient
  mlir::MlirOp next_grad_b;
  if (has_biases) {
    mlir::MlirOp chunk_grad_b = mlir::stablehlo::Reduce(
        builder, delta_pre_chunk, zero_const, sum_reduce_builder, {0, 1})[0];
    next_grad_b = mlir::stablehlo::Add(body_grad_b, chunk_grad_b);
  }

  // Streaming accumulation: Projection weight gradient
  mlir::MlirOp next_grad_w_hr;
  if (w_hr.has_value()) {
    mlir::MlirOp delta_h_chunk =
        ConcatDim(builder, delta_h_chunk_steps, concat_dim);
    mlir::MlirOp delta_h_chunk_2d = mlir::stablehlo::Reshape(
        delta_h_chunk, {kLstmUnrollFactor * batch, out_h});
    mlir::MlirOp r_chunk = ConcatDim(builder, r_chunk_steps, concat_dim);
    mlir::MlirOp r_chunk_2d =
        mlir::stablehlo::Reshape(r_chunk, {kLstmUnrollFactor * batch, hidden});
    mlir::MlirOp chunk_grad_w_hr =
        MixedPrecisionDotGeneral(delta_h_chunk_2d, r_chunk_2d, grad_w_dot_dims,
                                 acc_dtype, out_dtype, precision);
    next_grad_w_hr = mlir::stablehlo::Add(body_grad_w_hr, chunk_grad_w_hr);
  }

  // Layer input gradient chunk: delta_pre_chunk_2d @ w_ih_clumped
  mlir::MlirOp chunk_grad_x_2d =
      MixedPrecisionDotGeneral(delta_pre_chunk_2d, w_ih_clumped,
                               bwd_weight_dot_dims, acc_dtype, precision);
  mlir::MlirOp chunk_grad_x =
      batch_first ? mlir::stablehlo::Reshape(chunk_grad_x_2d,
                                             {batch, kLstmUnrollFactor, in_dim})
                  : mlir::stablehlo::Reshape(
                        chunk_grad_x_2d, {kLstmUnrollFactor, batch, in_dim});
  mlir::MlirOp next_grad_x_chunked = mlir::stablehlo::DynamicUpdateSlice(
      body_grad_x_chunked, chunk_grad_x, chunk_start_indices);

  mlir::MlirOp step_k = MakeScalarConstant(builder, kLstmUnrollFactor, i64);
  mlir::MlirOp next_step_idx = mlir::stablehlo::Add(body_step_idx, step_k);

  llvm::SmallVector<mlir::Value> next_loop_values = {
      next_step_idx.getValue(),    cur_delta_h_rec.getValue(),
      cur_delta_c_next.getValue(), next_grad_x_chunked.getValue(),
      next_grad_w_ih.getValue(),   next_grad_w_hh.getValue()};
  if (has_biases) {
    next_loop_values.push_back(next_grad_b.getValue());
  }
  if (w_hr.has_value()) {
    next_loop_values.push_back(next_grad_w_hr.getValue());
  }
  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_loop_values);

  op_builder.setInsertionPointAfter(while_op);

  mlir::MlirOp final_delta_h_rec(builder, while_op.getResult(1));
  mlir::MlirOp final_delta_c_next(builder, while_op.getResult(2));
  mlir::MlirOp grad_x_chunked_final(builder, while_op.getResult(3));
  mlir::MlirOp final_grad_w_ih_clumped(builder, while_op.getResult(4));
  mlir::MlirOp final_grad_w_hh_clumped(builder, while_op.getResult(5));

  mlir::MlirOp grad_x_layer;
  if (rem_steps > 0) {
    grad_x_layer =
        ConcatDim(builder, {grad_x_chunked_final, rem_grad_x}, concat_dim);
  } else {
    grad_x_layer = grad_x_chunked_final;
  }

  mlir::MlirOp grad_h0_layer =
      mlir::stablehlo::Reshape(to_out(final_delta_h_rec), {1, batch, out_h});
  mlir::MlirOp grad_c0_layer =
      mlir::stablehlo::Reshape(to_out(final_delta_c_next), {1, batch, hidden});

  mlir::MlirOp grad_w_ih =
      UnclumpWeight3H(builder, final_grad_w_ih_clumped, hidden, in_dim);
  mlir::MlirOp grad_w_hh =
      UnclumpWeight3H(builder, final_grad_w_hh_clumped, hidden, out_h);

  int res_idx = 6;
  std::optional<mlir::MlirOp> grad_bias;
  if (has_biases) {
    mlir::MlirOp final_grad_b_clumped(builder, while_op.getResult(res_idx++));
    mlir::MlirOp grad_b = UnclumpBias3H(builder, final_grad_b_clumped, hidden);
    grad_bias = to_out(grad_b);
  }

  std::optional<mlir::MlirOp> grad_w_hr;
  if (w_hr.has_value()) {
    mlir::MlirOp final_grad_w_hr(builder, while_op.getResult(res_idx++));
    grad_w_hr = to_out(final_grad_w_hr);
  }

  return {grad_x_layer,      grad_h0_layer, grad_c0_layer, to_out(grad_w_ih),
          to_out(grad_w_hh), grad_bias,     grad_w_hr};
}

// Evaluates backward recurrence through time (BPTT) for a single unidirectional
// LSTM layer.
//
// Key Optimizations:
// 1. Weight 3H clumping: Clumps W_ih and W_hh into contiguous 3H layout (i, f,
// o) upfront,
//    ensuring all downstream backward matrix multiplications are packed and
//    efficient.
// 2. Dispatch:
//    - For short sequences (seq_len < kLstmUnrollFactor), calls
//    ComputeLayerBackwardStatic
//      to eliminate while-loop overhead.
//    - For standard sequences (seq_len >= kLstmUnrollFactor), calls
//    ComputeLayerBackwardChunked
//      for streaming memory reduction.
template <typename ToOutFn, typename ReduceBuilderFn>
LayerBackwardOutputs ComputeLayerBackward(
    mlir::MlirBuilder& builder, const LayerForwardState& layer_state,
    mlir::MlirOp grad_x_curr, mlir::MlirOp grad_hy_l, mlir::MlirOp grad_cy_l,
    mlir::MlirOp w_ih, mlir::MlirOp w_hh, const bool has_biases,
    mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    mlir::stablehlo::DotDimensionNumbersAttr bwd_weight_dot_dims,
    mlir::stablehlo::DotDimensionNumbersAttr grad_w_dot_dims,
    const int64_t seq_len, const int64_t batch, const int64_t hidden,
    const int64_t in_dim, const int64_t out_h, ToOutFn to_out,
    std::optional<mlir::MlirOp> w_hr,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> bwd_hr_dot_dims,
    const bool batch_first, mlir::stablehlo::Precision precision) {
  mlir::MlirOp w_ih_clumped = ClumpWeight3H(builder, w_ih, hidden, in_dim);
  mlir::MlirOp w_hh_clumped = ClumpWeight3H(builder, w_hh, hidden, out_h);

  if (seq_len < kLstmUnrollFactor) {
    return ComputeLayerBackwardStatic(
        builder, layer_state, grad_x_curr, grad_hy_l, grad_cy_l, w_ih_clumped,
        w_hh_clumped, has_biases, zero_const, sum_reduce_builder,
        bwd_weight_dot_dims, grad_w_dot_dims, seq_len, batch, hidden, in_dim,
        out_h, to_out, w_hr, bwd_hr_dot_dims, batch_first, precision);
  }

  return ComputeLayerBackwardChunked(
      builder, layer_state, grad_x_curr, grad_hy_l, grad_cy_l, w_ih_clumped,
      w_hh_clumped, has_biases, zero_const, sum_reduce_builder,
      bwd_weight_dot_dims, grad_w_dot_dims, seq_len, batch, hidden, in_dim,
      out_h, to_out, w_hr, bwd_hr_dot_dims, batch_first, precision);
}

// Evaluates backward recurrence through time (BPTT) for a multi-layer
// unidirectional LSTM using Pipelined Wavefront scheduling across layers.
//
// In multi-layer BPTT, gradients flow from top layer (num_layers - 1) down to
// bottom layer 0. For each chunk of kLstmUnrollFactor timesteps (in reverse
// chronological order from chunk N-1 down to chunk 0):
// - Layer l computes its kLstmUnrollFactor backward recurrence steps on chunk
// C.
// - The resulting input gradient chunk grad_x_chunk [kLstmUnrollFactor, B,
// out_h] (scaled by
//   inter-layer dropout backward if applicable) is forwarded directly in
//   vector memory / registers to layer l - 1 within the SAME while-loop body.
// - Layer l - 1 immediately backpropagates chunk C, and so forth down to layer
// 0.
// - All intermediate [T, B, out_h] sequence gradient buffers between layers are
//   completely eliminated, reducing peak HBM usage and enabling fine-grained
//   inter-layer execution overlap on the TPU.
template <typename ToOutFn, typename ReduceBuilderFn>
MultiLayerBackwardOutputs BuildLstmPipelinedWavefrontBackward(
    mlir::MlirBuilder& builder,
    absl::Span<const LayerForwardState> layer_states,
    mlir::MlirOp incoming_seq_grad, absl::Span<const mlir::MlirOp> grad_hy_list,
    absl::Span<const mlir::MlirOp> grad_cy_list,
    absl::Span<const mlir::MlirOp> w_ih_clumped_list,
    absl::Span<const mlir::MlirOp> w_hh_clumped_list, const bool has_biases,
    mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    mlir::stablehlo::DotDimensionNumbersAttr bwd_weight_dot_dims,
    mlir::stablehlo::DotDimensionNumbersAttr grad_w_dot_dims,
    const int64_t seq_len, const int64_t batch, const int64_t hidden,
    const int64_t input_size, const int64_t out_h, const int64_t num_layers,
    ToOutFn to_out, absl::Span<const std::optional<mlir::MlirOp>> w_hr_list,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> bwd_hr_dot_dims,
    const bool has_dropout, const double dropout,
    absl::Span<const mlir::MlirOp> dropout_masks, const bool batch_first,
    mlir::stablehlo::Precision precision) {
  const int64_t num_chunks = seq_len / kLstmUnrollFactor;
  const int64_t chunked_steps = num_chunks * kLstmUnrollFactor;
  const int64_t rem_steps = seq_len % kLstmUnrollFactor;
  const bool is_projected = !w_hr_list.empty() && w_hr_list[0].has_value();

  const mlir::Type acc_elem_type =
      GetTensorTypeOrDie(grad_hy_list[0]).getElementType();
  const mlir::Type out_dtype =
      GetTensorTypeOrDie(w_ih_clumped_list[0]).getElementType();

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = incoming_seq_grad.getValue().getLoc();
  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType i64_scalar_type =
      mlir::RankedTensorType::get({}, i64);

  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0f, acc_elem_type);
  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64);

  const int64_t concat_dim = batch_first ? 1 : 0;

  std::vector<mlir::MlirOp> rem_grad_w_ih_init(num_layers);
  std::vector<mlir::MlirOp> rem_grad_w_hh_init(num_layers);
  std::vector<std::optional<mlir::MlirOp>> rem_grad_b_init(num_layers);
  std::vector<std::optional<mlir::MlirOp>> rem_grad_w_hr_init(num_layers);
  std::vector<mlir::MlirOp> rem_delta_h_rec(num_layers);
  std::vector<mlir::MlirOp> rem_delta_c_next(num_layers);
  std::optional<mlir::MlirOp> rem_grad_x_0;

  // ---------------------------------------------------------------------------
  // Phase 1: Remainder steps (t = seq_len - 1 down to chunked_steps)
  // ---------------------------------------------------------------------------
  if (rem_steps > 0) {
    mlir::MlirOp curr_rem_grad_y;
    for (int64_t l = num_layers - 1; l >= 0; --l) {
      const int64_t in_dim_l = (l == 0) ? input_size : out_h;
      if (l == num_layers - 1) {
        curr_rem_grad_y = batch_first
                              ? mlir::stablehlo::Slice(
                                    incoming_seq_grad, {0, chunked_steps, 0},
                                    {batch, seq_len, out_h}, {1, 1, 1})
                              : mlir::stablehlo::Slice(
                                    incoming_seq_grad, {chunked_steps, 0, 0},
                                    {seq_len, batch, out_h}, {1, 1, 1});
      } else {
        if (has_dropout) {
          mlir::MlirOp mask_l = dropout_masks[l];
          mlir::MlirOp rem_mask =
              batch_first
                  ? mlir::stablehlo::Slice(mask_l, {0, chunked_steps, 0},
                                           {batch, seq_len, out_h}, {1, 1, 1})
                  : mlir::stablehlo::Slice(mask_l, {chunked_steps, 0, 0},
                                           {seq_len, batch, out_h}, {1, 1, 1});
          curr_rem_grad_y =
              ApplyDropoutBackward(curr_rem_grad_y, rem_mask, dropout);
        }
      }

      mlir::MlirOp ifo_seq_l = layer_states[l].ifo_seq;
      mlir::MlirOp rem_ifo =
          batch_first
              ? mlir::stablehlo::Slice(ifo_seq_l, {0, chunked_steps, 0},
                                       {batch, seq_len, 3 * hidden}, {1, 1, 1})
              : mlir::stablehlo::Slice(ifo_seq_l, {chunked_steps, 0, 0},
                                       {seq_len, batch, 3 * hidden}, {1, 1, 1});
      mlir::MlirOp cellgate_seq_l = layer_states[l].cellgate_seq;
      mlir::MlirOp rem_cellgate =
          batch_first
              ? mlir::stablehlo::Slice(cellgate_seq_l, {0, chunked_steps, 0},
                                       {batch, seq_len, hidden}, {1, 1, 1})
              : mlir::stablehlo::Slice(cellgate_seq_l, {chunked_steps, 0, 0},
                                       {seq_len, batch, hidden}, {1, 1, 1});
      mlir::MlirOp tanh_c_seq_l = layer_states[l].tanh_c_seq;
      mlir::MlirOp rem_tanh_c =
          batch_first
              ? mlir::stablehlo::Slice(tanh_c_seq_l, {0, chunked_steps, 0},
                                       {batch, seq_len, hidden}, {1, 1, 1})
              : mlir::stablehlo::Slice(tanh_c_seq_l, {chunked_steps, 0, 0},
                                       {seq_len, batch, hidden}, {1, 1, 1});
      mlir::MlirOp c_prev_seq_l = layer_states[l].c_prev_seq;
      mlir::MlirOp rem_c_prev =
          batch_first
              ? mlir::stablehlo::Slice(c_prev_seq_l, {0, chunked_steps, 0},
                                       {batch, seq_len, hidden}, {1, 1, 1})
              : mlir::stablehlo::Slice(c_prev_seq_l, {chunked_steps, 0, 0},
                                       {seq_len, batch, hidden}, {1, 1, 1});
      mlir::MlirOp h_prev_seq_l = layer_states[l].h_prev_seq;
      mlir::MlirOp rem_h_prev =
          batch_first
              ? mlir::stablehlo::Slice(h_prev_seq_l, {0, chunked_steps, 0},
                                       {batch, seq_len, out_h}, {1, 1, 1})
              : mlir::stablehlo::Slice(h_prev_seq_l, {chunked_steps, 0, 0},
                                       {seq_len, batch, out_h}, {1, 1, 1});
      mlir::MlirOp layer_in_l = layer_states[l].layer_input;
      mlir::MlirOp rem_layer_in =
          batch_first
              ? mlir::stablehlo::Slice(layer_in_l, {0, chunked_steps, 0},
                                       {batch, seq_len, in_dim_l}, {1, 1, 1})
              : mlir::stablehlo::Slice(layer_in_l, {chunked_steps, 0, 0},
                                       {seq_len, batch, in_dim_l}, {1, 1, 1});

      mlir::MlirOp cur_delta_h_rec = grad_hy_list[l];
      mlir::MlirOp cur_delta_c_next = grad_cy_list[l];

      std::vector<mlir::MlirOp> delta_pre_rem_steps(rem_steps);
      std::vector<mlir::MlirOp> delta_h_rem_steps(is_projected ? rem_steps : 0);
      std::vector<mlir::MlirOp> r_rem_steps(is_projected ? rem_steps : 0);

      for (int64_t k = rem_steps - 1; k >= 0; --k) {
        mlir::MlirOp grad_from_above_k =
            SliceStep2D(curr_rem_grad_y, k, batch, out_h, batch_first);

        mlir::MlirOp delta_h =
            mlir::stablehlo::Add(grad_from_above_k, cur_delta_h_rec);

        mlir::MlirOp tanh_c_k =
            SliceStep2D(rem_tanh_c, k, batch, hidden, batch_first);
        mlir::MlirOp c_prev_k =
            SliceStep2D(rem_c_prev, k, batch, hidden, batch_first);
        mlir::MlirOp g_k =
            SliceStep2D(rem_cellgate, k, batch, hidden, batch_first);
        mlir::MlirOp ifo_k =
            SliceStep2D(rem_ifo, k, batch, 3 * hidden, batch_first);

        BackwardStepResult step = ComputeBackwardStep(
            builder, delta_h, cur_delta_c_next, tanh_c_k, c_prev_k, g_k, ifo_k,
            w_hh_clumped_list[l], bwd_weight_dot_dims, batch, hidden,
            w_hr_list[l], bwd_hr_dot_dims, precision);

        cur_delta_c_next = step.delta_c_next;
        cur_delta_h_rec = step.delta_h_rec;
        delta_pre_rem_steps[k] =
            ExpandStep3D(step.delta_pre, batch, 4 * hidden, batch_first);

        if (is_projected) {
          delta_h_rem_steps[k] =
              ExpandStep3D(delta_h, batch, out_h, batch_first);
          mlir::MlirOp o_k = mlir::stablehlo::Slice(
              ifo_k, {0, 2 * hidden}, {batch, 3 * hidden}, {1, 1});
          mlir::MlirOp r_k = mlir::stablehlo::Mul(o_k, tanh_c_k);
          r_rem_steps[k] = ExpandStep3D(r_k, batch, hidden, batch_first);
        }
      }

      mlir::MlirOp delta_pre_rem =
          ConcatDim(builder, delta_pre_rem_steps, concat_dim);
      mlir::MlirOp delta_pre_rem_2d = mlir::stablehlo::Reshape(
          delta_pre_rem, {rem_steps * batch, 4 * hidden});

      mlir::MlirOp rem_layer_in_2d =
          mlir::stablehlo::Reshape(rem_layer_in, {rem_steps * batch, in_dim_l});
      rem_grad_w_ih_init[l] = MixedPrecisionDotGeneral(
          delta_pre_rem_2d, rem_layer_in_2d, grad_w_dot_dims, acc_elem_type,
          out_dtype, precision);

      mlir::MlirOp rem_h_prev_2d =
          mlir::stablehlo::Reshape(rem_h_prev, {rem_steps * batch, out_h});
      rem_grad_w_hh_init[l] = MixedPrecisionDotGeneral(
          delta_pre_rem_2d, rem_h_prev_2d, grad_w_dot_dims, acc_elem_type,
          out_dtype, precision);

      if (has_biases) {
        rem_grad_b_init[l] = mlir::stablehlo::Reduce(
            builder, delta_pre_rem, zero_const, sum_reduce_builder, {0, 1})[0];
      }
      if (is_projected) {
        mlir::MlirOp delta_h_rem_seq =
            ConcatDim(builder, delta_h_rem_steps, concat_dim);
        mlir::MlirOp delta_h_rem_seq_2d = mlir::stablehlo::Reshape(
            delta_h_rem_seq, {rem_steps * batch, out_h});
        mlir::MlirOp r_rem_seq = ConcatDim(builder, r_rem_steps, concat_dim);
        mlir::MlirOp r_rem_seq_2d =
            mlir::stablehlo::Reshape(r_rem_seq, {rem_steps * batch, hidden});
        rem_grad_w_hr_init[l] = MixedPrecisionDotGeneral(
            delta_h_rem_seq_2d, r_rem_seq_2d, grad_w_dot_dims, acc_elem_type,
            out_dtype, precision);
      }

      rem_delta_h_rec[l] = cur_delta_h_rec;
      rem_delta_c_next[l] = cur_delta_c_next;

      mlir::MlirOp rem_grad_x_2d = MixedPrecisionDotGeneral(
          delta_pre_rem_2d, w_ih_clumped_list[l], bwd_weight_dot_dims,
          acc_elem_type, precision);
      mlir::MlirOp rem_grad_x =
          batch_first ? mlir::stablehlo::Reshape(rem_grad_x_2d,
                                                 {batch, rem_steps, in_dim_l})
                      : mlir::stablehlo::Reshape(rem_grad_x_2d,
                                                 {rem_steps, batch, in_dim_l});

      if (l == 0) {
        rem_grad_x_0 = rem_grad_x;
      } else {
        curr_rem_grad_y = rem_grad_x;
      }
    }
  } else {
    for (int64_t l = 0; l < num_layers; ++l) {
      const int64_t in_dim_l = (l == 0) ? input_size : out_h;
      rem_delta_h_rec[l] = grad_hy_list[l];
      rem_delta_c_next[l] = grad_cy_list[l];
      rem_grad_w_ih_init[l] = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({4 * hidden, in_dim_l}, acc_elem_type),
          zero_scalar, {});
      rem_grad_w_hh_init[l] = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({4 * hidden, out_h}, acc_elem_type),
          zero_scalar, {});
      if (has_biases) {
        rem_grad_b_init[l] = mlir::stablehlo::BroadcastInDim(
            mlir::RankedTensorType::get({4 * hidden}, acc_elem_type),
            zero_scalar, {});
      }
      if (is_projected) {
        rem_grad_w_hr_init[l] = mlir::stablehlo::BroadcastInDim(
            mlir::RankedTensorType::get({out_h, hidden}, acc_elem_type),
            zero_scalar, {});
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Phase 2: Unified Wavefront While Loop over Chunks (Reverse Order)
  // ---------------------------------------------------------------------------
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

  loop_types.push_back(grad_x_chunked_type);
  loop_inits.push_back(grad_x_chunked_init.getValue());

  for (int64_t l = 0; l < num_layers; ++l) {
    loop_types.push_back(rem_delta_h_rec[l].getType());
    loop_inits.push_back(rem_delta_h_rec[l].getValue());

    loop_types.push_back(rem_delta_c_next[l].getType());
    loop_inits.push_back(rem_delta_c_next[l].getValue());

    loop_types.push_back(rem_grad_w_ih_init[l].getType());
    loop_inits.push_back(rem_grad_w_ih_init[l].getValue());

    loop_types.push_back(rem_grad_w_hh_init[l].getType());
    loop_inits.push_back(rem_grad_w_hh_init[l].getValue());

    if (has_biases) {
      loop_types.push_back((*rem_grad_b_init[l]).getType());
      loop_inits.push_back((*rem_grad_b_init[l]).getValue());
    }
    if (is_projected) {
      loop_types.push_back((*rem_grad_w_hr_init[l]).getType());
      loop_inits.push_back((*rem_grad_w_hr_init[l]).getValue());
    }
  }

  auto while_op =
      mlir::stablehlo::WhileOp::create(op_builder, loc, loop_types, loop_inits);

  // Condition region: step_idx < chunked_steps
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

  // Body region
  mlir::Block* const body_block = op_builder.createBlock(&while_op.getBody());
  body_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(body_block);

  mlir::MlirOp body_step_idx(builder, body_block->getArgument(0));
  mlir::MlirOp body_grad_x_chunked(builder, body_block->getArgument(1));

  struct LayerCarriedState {
    mlir::MlirOp delta_h_rec;
    mlir::MlirOp delta_c_next;
    mlir::MlirOp grad_w_ih;
    mlir::MlirOp grad_w_hh;
    std::optional<mlir::MlirOp> grad_b;
    std::optional<mlir::MlirOp> grad_w_hr;
  };

  int arg_idx = 2;
  std::vector<LayerCarriedState> body_layers(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    body_layers[l].delta_h_rec =
        mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
    body_layers[l].delta_c_next =
        mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
    body_layers[l].grad_w_ih =
        mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
    body_layers[l].grad_w_hh =
        mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
    if (has_biases) {
      body_layers[l].grad_b =
          mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
    }
    if (is_projected) {
      body_layers[l].grad_w_hr =
          mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
    }
  }

  // Chronological chunk start timestep:
  // chunk_start_t = (chunked_steps - kLstmUnrollFactor) - body_step_idx
  mlir::MlirOp const_chunk_limit =
      MakeScalarConstant(builder, chunked_steps - kLstmUnrollFactor, i64);
  mlir::MlirOp chunk_start_t =
      mlir::stablehlo::Subtract(const_chunk_limit, body_step_idx);

  const llvm::SmallVector<mlir::MlirOp, 3> chunk_start_indices =
      batch_first ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, chunk_start_t,
                                                       zero_i64}
                  : llvm::SmallVector<mlir::MlirOp, 3>{chunk_start_t, zero_i64,
                                                       zero_i64};
  const llvm::SmallVector<int64_t, 3> out_h_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kLstmUnrollFactor, out_h}
          : llvm::SmallVector<int64_t, 3>{kLstmUnrollFactor, batch, out_h};
  const llvm::SmallVector<int64_t, 3> hidden_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kLstmUnrollFactor, hidden}
          : llvm::SmallVector<int64_t, 3>{kLstmUnrollFactor, batch, hidden};
  const llvm::SmallVector<int64_t, 3> ifo_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kLstmUnrollFactor, 3 * hidden}
          : llvm::SmallVector<int64_t, 3>{kLstmUnrollFactor, batch, 3 * hidden};

  std::vector<LayerCarriedState> next_layers(num_layers);
  mlir::MlirOp next_grad_x_chunked = body_grad_x_chunked;

  mlir::MlirOp curr_chunk_grad_y;

  for (int64_t l = num_layers - 1; l >= 0; --l) {
    const int64_t in_dim_l = (l == 0) ? input_size : out_h;
    const llvm::SmallVector<int64_t, 3> in_slice_sizes =
        batch_first
            ? llvm::SmallVector<int64_t, 3>{batch, kLstmUnrollFactor, in_dim_l}
            : llvm::SmallVector<int64_t, 3>{kLstmUnrollFactor, batch, in_dim_l};

    if (l == num_layers - 1) {
      curr_chunk_grad_y = mlir::stablehlo::DynamicSlice(
          incoming_seq_grad, chunk_start_indices, out_h_slice_sizes);
    } else {
      if (has_dropout) {
        mlir::MlirOp mask_l = dropout_masks[l];
        mlir::MlirOp mask_chunk = mlir::stablehlo::DynamicSlice(
            mask_l, chunk_start_indices, out_h_slice_sizes);
        curr_chunk_grad_y =
            ApplyDropoutBackward(curr_chunk_grad_y, mask_chunk, dropout);
      }
    }

    mlir::MlirOp ifo_seq_l = layer_states[l].ifo_seq;
    mlir::MlirOp ifo_chunk = mlir::stablehlo::DynamicSlice(
        ifo_seq_l, chunk_start_indices, ifo_slice_sizes);
    mlir::MlirOp cellgate_seq_l = layer_states[l].cellgate_seq;
    mlir::MlirOp cellgate_chunk = mlir::stablehlo::DynamicSlice(
        cellgate_seq_l, chunk_start_indices, hidden_slice_sizes);
    mlir::MlirOp tanh_c_seq_l = layer_states[l].tanh_c_seq;
    mlir::MlirOp tanh_c_chunk = mlir::stablehlo::DynamicSlice(
        tanh_c_seq_l, chunk_start_indices, hidden_slice_sizes);
    mlir::MlirOp c_prev_seq_l = layer_states[l].c_prev_seq;
    mlir::MlirOp c_prev_chunk = mlir::stablehlo::DynamicSlice(
        c_prev_seq_l, chunk_start_indices, hidden_slice_sizes);
    mlir::MlirOp h_prev_seq_l = layer_states[l].h_prev_seq;
    mlir::MlirOp h_prev_chunk = mlir::stablehlo::DynamicSlice(
        h_prev_seq_l, chunk_start_indices, out_h_slice_sizes);
    mlir::MlirOp layer_in_l = layer_states[l].layer_input;
    mlir::MlirOp layer_in_chunk = mlir::stablehlo::DynamicSlice(
        layer_in_l, chunk_start_indices, in_slice_sizes);

    mlir::MlirOp cur_delta_h_rec = body_layers[l].delta_h_rec;
    mlir::MlirOp cur_delta_c_next = body_layers[l].delta_c_next;

    std::vector<mlir::MlirOp> delta_pre_chunk_steps(kLstmUnrollFactor);
    std::vector<mlir::MlirOp> delta_h_chunk_steps(
        is_projected ? kLstmUnrollFactor : 0);
    std::vector<mlir::MlirOp> r_chunk_steps(is_projected ? kLstmUnrollFactor
                                                         : 0);

    for (int64_t k = kLstmUnrollFactor - 1; k >= 0; --k) {
      mlir::MlirOp grad_from_above_k =
          SliceStep2D(curr_chunk_grad_y, k, batch, out_h, batch_first);

      mlir::MlirOp delta_h =
          mlir::stablehlo::Add(grad_from_above_k, cur_delta_h_rec);

      mlir::MlirOp tanh_c_k =
          SliceStep2D(tanh_c_chunk, k, batch, hidden, batch_first);
      mlir::MlirOp c_prev_k =
          SliceStep2D(c_prev_chunk, k, batch, hidden, batch_first);
      mlir::MlirOp g_k =
          SliceStep2D(cellgate_chunk, k, batch, hidden, batch_first);
      mlir::MlirOp ifo_k =
          SliceStep2D(ifo_chunk, k, batch, 3 * hidden, batch_first);

      BackwardStepResult step = ComputeBackwardStep(
          builder, delta_h, cur_delta_c_next, tanh_c_k, c_prev_k, g_k, ifo_k,
          w_hh_clumped_list[l], bwd_weight_dot_dims, batch, hidden,
          w_hr_list[l], bwd_hr_dot_dims, precision);

      cur_delta_c_next = step.delta_c_next;
      cur_delta_h_rec = step.delta_h_rec;
      delta_pre_chunk_steps[k] =
          ExpandStep3D(step.delta_pre, batch, 4 * hidden, batch_first);

      if (is_projected) {
        delta_h_chunk_steps[k] =
            ExpandStep3D(delta_h, batch, out_h, batch_first);
        mlir::MlirOp o_k = mlir::stablehlo::Slice(ifo_k, {0, 2 * hidden},
                                                  {batch, 3 * hidden}, {1, 1});
        mlir::MlirOp r_k = mlir::stablehlo::Mul(o_k, tanh_c_k);
        r_chunk_steps[k] = ExpandStep3D(r_k, batch, hidden, batch_first);
      }
    }

    mlir::MlirOp delta_pre_chunk =
        ConcatDim(builder, delta_pre_chunk_steps, concat_dim);
    mlir::MlirOp delta_pre_chunk_2d = mlir::stablehlo::Reshape(
        delta_pre_chunk, {kLstmUnrollFactor * batch, 4 * hidden});

    // Streaming accumulation: Input weight gradient
    mlir::MlirOp layer_in_chunk_2d = mlir::stablehlo::Reshape(
        layer_in_chunk, {kLstmUnrollFactor * batch, in_dim_l});
    mlir::MlirOp chunk_grad_w_ih = MixedPrecisionDotGeneral(
        delta_pre_chunk_2d, layer_in_chunk_2d, grad_w_dot_dims, acc_elem_type,
        out_dtype, precision);
    next_layers[l].grad_w_ih =
        mlir::stablehlo::Add(body_layers[l].grad_w_ih, chunk_grad_w_ih);

    // Streaming accumulation: Recurrent weight gradient
    mlir::MlirOp h_prev_chunk_2d = mlir::stablehlo::Reshape(
        h_prev_chunk, {kLstmUnrollFactor * batch, out_h});
    mlir::MlirOp chunk_grad_w_hh = MixedPrecisionDotGeneral(
        delta_pre_chunk_2d, h_prev_chunk_2d, grad_w_dot_dims, acc_elem_type,
        out_dtype, precision);
    next_layers[l].grad_w_hh =
        mlir::stablehlo::Add(body_layers[l].grad_w_hh, chunk_grad_w_hh);

    // Streaming accumulation: Bias gradient
    if (has_biases) {
      mlir::MlirOp chunk_grad_b = mlir::stablehlo::Reduce(
          builder, delta_pre_chunk, zero_const, sum_reduce_builder, {0, 1})[0];
      next_layers[l].grad_b =
          mlir::stablehlo::Add(*body_layers[l].grad_b, chunk_grad_b);
    }

    // Streaming accumulation: Projection weight gradient
    if (is_projected) {
      mlir::MlirOp delta_h_chunk =
          ConcatDim(builder, delta_h_chunk_steps, concat_dim);
      mlir::MlirOp delta_h_chunk_2d = mlir::stablehlo::Reshape(
          delta_h_chunk, {kLstmUnrollFactor * batch, out_h});
      mlir::MlirOp r_chunk = ConcatDim(builder, r_chunk_steps, concat_dim);
      mlir::MlirOp r_chunk_2d = mlir::stablehlo::Reshape(
          r_chunk, {kLstmUnrollFactor * batch, hidden});
      mlir::MlirOp chunk_grad_w_hr = MixedPrecisionDotGeneral(
          delta_h_chunk_2d, r_chunk_2d, grad_w_dot_dims, acc_elem_type,
          out_dtype, precision);
      next_layers[l].grad_w_hr =
          mlir::stablehlo::Add(*body_layers[l].grad_w_hr, chunk_grad_w_hr);
    }

    next_layers[l].delta_h_rec = cur_delta_h_rec;
    next_layers[l].delta_c_next = cur_delta_c_next;

    // Layer input gradient chunk: delta_pre_chunk_2d @ w_ih_clumped
    mlir::MlirOp chunk_grad_x_2d =
        MixedPrecisionDotGeneral(delta_pre_chunk_2d, w_ih_clumped_list[l],
                                 bwd_weight_dot_dims, acc_elem_type, precision);
    mlir::MlirOp chunk_grad_x =
        batch_first
            ? mlir::stablehlo::Reshape(chunk_grad_x_2d,
                                       {batch, kLstmUnrollFactor, in_dim_l})
            : mlir::stablehlo::Reshape(chunk_grad_x_2d,
                                       {kLstmUnrollFactor, batch, in_dim_l});

    if (l == 0) {
      next_grad_x_chunked = mlir::stablehlo::DynamicUpdateSlice(
          body_grad_x_chunked, chunk_grad_x, chunk_start_indices);
    } else {
      curr_chunk_grad_y = chunk_grad_x;
    }
  }

  mlir::MlirOp step_k = MakeScalarConstant(builder, kLstmUnrollFactor, i64);
  mlir::MlirOp next_step_idx = mlir::stablehlo::Add(body_step_idx, step_k);

  llvm::SmallVector<mlir::Value> next_loop_values;
  next_loop_values.reserve(loop_types.size());
  next_loop_values.push_back(next_step_idx.getValue());
  next_loop_values.push_back(next_grad_x_chunked.getValue());

  for (int64_t l = 0; l < num_layers; ++l) {
    next_loop_values.push_back(next_layers[l].delta_h_rec.getValue());
    next_loop_values.push_back(next_layers[l].delta_c_next.getValue());
    next_loop_values.push_back(next_layers[l].grad_w_ih.getValue());
    next_loop_values.push_back(next_layers[l].grad_w_hh.getValue());
    if (has_biases) {
      next_loop_values.push_back((*next_layers[l].grad_b).getValue());
    }
    if (is_projected) {
      next_loop_values.push_back((*next_layers[l].grad_w_hr).getValue());
    }
  }

  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_loop_values);

  op_builder.setInsertionPointAfter(while_op);

  mlir::MlirOp final_grad_x_chunked(builder, while_op.getResult(1));

  int res_idx = 2;
  MultiLayerBackwardOutputs result;
  result.grad_h0.resize(num_layers);
  result.grad_c0.resize(num_layers);
  result.grad_w_ih.resize(num_layers);
  result.grad_w_hh.resize(num_layers);
  result.grad_bias.resize(num_layers);
  result.grad_w_hr.resize(num_layers);

  for (int64_t l = 0; l < num_layers; ++l) {
    const int64_t in_dim_l = (l == 0) ? input_size : out_h;
    mlir::MlirOp final_delta_h_rec(builder, while_op.getResult(res_idx++));
    mlir::MlirOp final_delta_c_next(builder, while_op.getResult(res_idx++));
    mlir::MlirOp final_grad_w_ih_clumped(builder,
                                         while_op.getResult(res_idx++));
    mlir::MlirOp final_grad_w_hh_clumped(builder,
                                         while_op.getResult(res_idx++));

    result.grad_h0[l] =
        mlir::stablehlo::Reshape(to_out(final_delta_h_rec), {1, batch, out_h});
    result.grad_c0[l] = mlir::stablehlo::Reshape(to_out(final_delta_c_next),
                                                 {1, batch, hidden});
    result.grad_w_ih[l] = to_out(
        UnclumpWeight3H(builder, final_grad_w_ih_clumped, hidden, in_dim_l));
    result.grad_w_hh[l] = to_out(
        UnclumpWeight3H(builder, final_grad_w_hh_clumped, hidden, out_h));

    if (has_biases) {
      mlir::MlirOp final_grad_b_clumped(builder, while_op.getResult(res_idx++));
      result.grad_bias[l] =
          to_out(UnclumpBias3H(builder, final_grad_b_clumped, hidden));
    }
    if (is_projected) {
      mlir::MlirOp final_grad_w_hr(builder, while_op.getResult(res_idx++));
      result.grad_w_hr[l] = to_out(final_grad_w_hr);
    }
  }

  if (rem_steps > 0) {
    result.grad_x =
        ConcatDim(builder, {final_grad_x_chunked, *rem_grad_x_0}, concat_dim);
  } else {
    result.grad_x = final_grad_x_chunked;
  }

  return result;
}

// Evaluates backward recurrence (BPTT) for a bidirectional layer using
// Concurrent Bidirectional Fusion with partial unrolling (kLstmUnrollFactor).
// Forward BPTT (t = T-1 down to 0) and reverse BPTT (s = T-1 down to 0,
// matching chronological t = 0 to T-1) execute concurrently inside the SAME
// loop body. This allows the TPU VPU and MXU to overlap forward and reverse
// adjoint updates, halving loop overhead and doubling arithmetic intensity.
// Static unrolling helper for bidirectional LSTM backward pass for short
// sequences (seq_len < kLstmUnrollFactor).
//
// In short sequences, while-loop carry tuple state and dynamic slicing
// introduce runtime dispatch overhead that exceeds the benefits of chunking. We
// statically unroll BPTT across all timesteps t = seq_len - 1 down to 0,
// running forward and reverse recurrent steps concurrently within the unrolled
// loop:
// - Step t forward: computes adjoints with respect to activations in state_fwd.
// - Step t reverse: computes adjoints with respect to activations in state_rev
// (in its time-reversed frame).
// - Interleaves vector and matrix ops between directions to maximize TPU
// execution concurrency.
// - Performs full-sequence 2D dot generals to accumulate W_ih and W_hh
// gradients across all steps.
template <typename ToOutFn, typename ReduceBuilderFn>
LstmBidirLayerBackwardOutputs ComputeBidirLayerBackwardStatic(
    mlir::MlirBuilder& builder, const LayerForwardState& state_fwd,
    const LayerForwardState& state_rev, mlir::MlirOp grad_y_fwd,
    mlir::MlirOp grad_y_rev_time_reversed, mlir::MlirOp grad_hy_fwd,
    mlir::MlirOp grad_cy_fwd, mlir::MlirOp grad_hy_rev,
    mlir::MlirOp grad_cy_rev, mlir::MlirOp w_ih_fwd_clumped,
    mlir::MlirOp w_hh_fwd_clumped, mlir::MlirOp w_ih_rev_clumped,
    mlir::MlirOp w_hh_rev_clumped, const bool has_biases,
    mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    mlir::stablehlo::DotDimensionNumbersAttr bwd_weight_dot_dims,
    mlir::stablehlo::DotDimensionNumbersAttr grad_w_dot_dims,
    const int64_t seq_len, const int64_t batch, const int64_t hidden,
    const int64_t in_dim, const int64_t out_h, ToOutFn to_out,
    std::optional<mlir::MlirOp> w_hr_fwd, std::optional<mlir::MlirOp> w_hr_rev,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> bwd_hr_dot_dims,
    const bool batch_first, mlir::stablehlo::Precision precision) {
  const int64_t concat_dim = batch_first ? 1 : 0;
  mlir::MlirOp delta_c_next_fwd = grad_cy_fwd;
  mlir::MlirOp delta_h_rec_fwd = grad_hy_fwd;
  mlir::MlirOp delta_c_next_rev = grad_cy_rev;
  mlir::MlirOp delta_h_rec_rev = grad_hy_rev;

  std::vector<mlir::MlirOp> delta_pre_steps_fwd(seq_len);
  std::vector<mlir::MlirOp> delta_pre_steps_rev(seq_len);
  std::vector<mlir::MlirOp> delta_h_steps_fwd(w_hr_fwd.has_value() ? seq_len
                                                                   : 0);
  std::vector<mlir::MlirOp> delta_h_steps_rev(w_hr_rev.has_value() ? seq_len
                                                                   : 0);
  std::vector<mlir::MlirOp> r_steps_fwd(w_hr_fwd.has_value() ? seq_len : 0);
  std::vector<mlir::MlirOp> r_steps_rev(w_hr_rev.has_value() ? seq_len : 0);

  // Statically unroll BPTT steps from t = seq_len - 1 down to 0.
  for (int64_t t = seq_len - 1; t >= 0; --t) {
    // --- Forward Step t ---
    // Extract incoming sequence gradient slice and add recurrent adjoint
    // delta_h.
    mlir::MlirOp grad_from_above_fwd =
        SliceStep2D(grad_y_fwd, t, batch, out_h, batch_first);
    mlir::MlirOp delta_h_fwd =
        (t == seq_len - 1)
            ? mlir::stablehlo::Add(grad_from_above_fwd, grad_hy_fwd)
            : mlir::stablehlo::Add(grad_from_above_fwd, delta_h_rec_fwd);

    mlir::MlirOp tanh_c_fwd =
        GetStepActivation(builder, state_fwd.tanh_c_list, state_fwd.tanh_c_seq,
                          t, batch, hidden, batch_first);
    mlir::MlirOp ifo_fwd =
        GetStepActivation(builder, state_fwd.ifo_list, state_fwd.ifo_seq, t,
                          batch, 3 * hidden, batch_first);

    BackwardStepResult step_fwd = ComputeBackwardStep(
        builder, delta_h_fwd, delta_c_next_fwd, tanh_c_fwd,
        GetStepActivation(builder, state_fwd.c_prev_list, state_fwd.c_prev_seq,
                          t, batch, hidden, batch_first),
        GetStepActivation(builder, state_fwd.cellgate_list,
                          state_fwd.cellgate_seq, t, batch, hidden,
                          batch_first),
        ifo_fwd, w_hh_fwd_clumped, bwd_weight_dot_dims, batch, hidden, w_hr_fwd,
        bwd_hr_dot_dims, precision);
    delta_c_next_fwd = step_fwd.delta_c_next;
    delta_h_rec_fwd = step_fwd.delta_h_rec;
    delta_pre_steps_fwd[t] =
        ExpandStep3D(step_fwd.delta_pre, batch, 4 * hidden, batch_first);

    if (w_hr_fwd.has_value()) {
      delta_h_steps_fwd[t] =
          ExpandStep3D(delta_h_fwd, batch, out_h, batch_first);
      mlir::MlirOp o_val_fwd = mlir::stablehlo::Slice(
          ifo_fwd, {0, 2 * hidden}, {batch, 3 * hidden}, {1, 1});
      mlir::MlirOp r_val_fwd = mlir::stablehlo::Mul(o_val_fwd, tanh_c_fwd);
      r_steps_fwd[t] = ExpandStep3D(r_val_fwd, batch, hidden, batch_first);
    }

    // --- Reverse Step t (executed concurrently with forward step t) ---
    // The reverse direction's incoming gradients and forward activations are
    // indexed in time-reversed alignment so step t corresponds to reverse
    // chronological order.
    mlir::MlirOp grad_from_above_rev =
        SliceStep2D(grad_y_rev_time_reversed, t, batch, out_h, batch_first);
    mlir::MlirOp delta_h_rev =
        (t == seq_len - 1)
            ? mlir::stablehlo::Add(grad_from_above_rev, grad_hy_rev)
            : mlir::stablehlo::Add(grad_from_above_rev, delta_h_rec_rev);

    mlir::MlirOp tanh_c_rev =
        GetStepActivation(builder, state_rev.tanh_c_list, state_rev.tanh_c_seq,
                          t, batch, hidden, batch_first);
    mlir::MlirOp ifo_rev =
        GetStepActivation(builder, state_rev.ifo_list, state_rev.ifo_seq, t,
                          batch, 3 * hidden, batch_first);

    BackwardStepResult step_rev = ComputeBackwardStep(
        builder, delta_h_rev, delta_c_next_rev, tanh_c_rev,
        GetStepActivation(builder, state_rev.c_prev_list, state_rev.c_prev_seq,
                          t, batch, hidden, batch_first),
        GetStepActivation(builder, state_rev.cellgate_list,
                          state_rev.cellgate_seq, t, batch, hidden,
                          batch_first),
        ifo_rev, w_hh_rev_clumped, bwd_weight_dot_dims, batch, hidden, w_hr_rev,
        bwd_hr_dot_dims, precision);
    delta_c_next_rev = step_rev.delta_c_next;
    delta_h_rec_rev = step_rev.delta_h_rec;
    delta_pre_steps_rev[t] =
        ExpandStep3D(step_rev.delta_pre, batch, 4 * hidden, batch_first);

    if (w_hr_rev.has_value()) {
      delta_h_steps_rev[t] =
          ExpandStep3D(delta_h_rev, batch, out_h, batch_first);
      mlir::MlirOp o_val_rev = mlir::stablehlo::Slice(
          ifo_rev, {0, 2 * hidden}, {batch, 3 * hidden}, {1, 1});
      mlir::MlirOp r_val_rev = mlir::stablehlo::Mul(o_val_rev, tanh_c_rev);
      r_steps_rev[t] = ExpandStep3D(r_val_rev, batch, hidden, batch_first);
    }
  }

  // Final initial state gradients: delta_h_rec (at t=0) -> grad_h0,
  // delta_c_next (at t=0) -> grad_c0.
  mlir::MlirOp grad_h0_fwd =
      mlir::stablehlo::Reshape(to_out(delta_h_rec_fwd), {1, batch, out_h});
  mlir::MlirOp grad_c0_fwd =
      mlir::stablehlo::Reshape(to_out(delta_c_next_fwd), {1, batch, hidden});
  mlir::MlirOp grad_h0_rev =
      mlir::stablehlo::Reshape(to_out(delta_h_rec_rev), {1, batch, out_h});
  mlir::MlirOp grad_c0_rev =
      mlir::stablehlo::Reshape(to_out(delta_c_next_rev), {1, batch, hidden});

  // Concatenate pre-activation adjoints across time and flatten to 2D for
  // batched GEMMs.
  mlir::MlirOp delta_pre_seq_fwd =
      ConcatDim(builder, delta_pre_steps_fwd, concat_dim);
  mlir::MlirOp delta_pre_seq_rev =
      ConcatDim(builder, delta_pre_steps_rev, concat_dim);

  mlir::MlirOp delta_pre_2d_fwd = mlir::stablehlo::Reshape(
      delta_pre_seq_fwd, {seq_len * batch, 4 * hidden});
  mlir::MlirOp delta_pre_2d_rev = mlir::stablehlo::Reshape(
      delta_pre_seq_rev, {seq_len * batch, 4 * hidden});

  const mlir::Type acc_dtype =
      GetTensorTypeOrDie(delta_pre_2d_fwd).getElementType();
  const mlir::Type out_dtype =
      GetTensorTypeOrDie(w_ih_fwd_clumped).getElementType();

  // Input sequence gradients:
  // grad_x_fwd = delta_pre_fwd @ W_ih_fwd
  mlir::MlirOp grad_x_2d_fwd =
      MixedPrecisionDotGeneral(delta_pre_2d_fwd, w_ih_fwd_clumped,
                               bwd_weight_dot_dims, acc_dtype, precision);
  mlir::MlirOp grad_x_layer_fwd =
      batch_first
          ? mlir::stablehlo::Reshape(grad_x_2d_fwd, {batch, seq_len, in_dim})
          : mlir::stablehlo::Reshape(grad_x_2d_fwd, {seq_len, batch, in_dim});

  // grad_x_rev = delta_pre_rev @ W_ih_rev
  mlir::MlirOp grad_x_2d_rev =
      MixedPrecisionDotGeneral(delta_pre_2d_rev, w_ih_rev_clumped,
                               bwd_weight_dot_dims, acc_dtype, precision);
  mlir::MlirOp grad_x_layer_rev =
      batch_first
          ? mlir::stablehlo::Reshape(grad_x_2d_rev, {batch, seq_len, in_dim})
          : mlir::stablehlo::Reshape(grad_x_2d_rev, {seq_len, batch, in_dim});

  // Reverse reverse direction gradients back to chronological time and sum with
  // forward gradients.
  mlir::MlirOp grad_x_from_rev =
      mlir::stablehlo::Reverse(grad_x_layer_rev, {/*dimensions=*/concat_dim});
  mlir::MlirOp grad_x_layer =
      mlir::stablehlo::Add(grad_x_layer_fwd, grad_x_from_rev);

  // Input weight gradients: grad_W_ih = delta_pre^T @ X
  mlir::MlirOp layer_in_2d_fwd = mlir::stablehlo::Reshape(
      state_fwd.layer_input, {seq_len * batch, in_dim});
  mlir::MlirOp grad_w_ih_clumped_fwd = MixedPrecisionDotGeneral(
      delta_pre_2d_fwd, layer_in_2d_fwd, grad_w_dot_dims, acc_dtype, out_dtype,
      precision);
  mlir::MlirOp grad_w_ih_fwd =
      UnclumpWeight3H(builder, grad_w_ih_clumped_fwd, hidden, in_dim);

  mlir::MlirOp layer_in_2d_rev = mlir::stablehlo::Reshape(
      state_rev.layer_input, {seq_len * batch, in_dim});
  mlir::MlirOp grad_w_ih_clumped_rev = MixedPrecisionDotGeneral(
      delta_pre_2d_rev, layer_in_2d_rev, grad_w_dot_dims, acc_dtype, out_dtype,
      precision);
  mlir::MlirOp grad_w_ih_rev =
      UnclumpWeight3H(builder, grad_w_ih_clumped_rev, hidden, in_dim);

  // Recurrent weight gradients: grad_W_hh = delta_pre^T @ H_prev
  mlir::MlirOp h_prev_2d_fwd =
      mlir::stablehlo::Reshape(state_fwd.h_prev_seq, {seq_len * batch, out_h});
  mlir::MlirOp grad_w_hh_clumped_fwd =
      MixedPrecisionDotGeneral(delta_pre_2d_fwd, h_prev_2d_fwd, grad_w_dot_dims,
                               acc_dtype, out_dtype, precision);
  mlir::MlirOp grad_w_hh_fwd =
      UnclumpWeight3H(builder, grad_w_hh_clumped_fwd, hidden, out_h);

  mlir::MlirOp h_prev_2d_rev =
      mlir::stablehlo::Reshape(state_rev.h_prev_seq, {seq_len * batch, out_h});
  mlir::MlirOp grad_w_hh_clumped_rev =
      MixedPrecisionDotGeneral(delta_pre_2d_rev, h_prev_2d_rev, grad_w_dot_dims,
                               acc_dtype, out_dtype, precision);
  mlir::MlirOp grad_w_hh_rev =
      UnclumpWeight3H(builder, grad_w_hh_clumped_rev, hidden, out_h);

  // Bias gradients: Sum pre-activation adjoints across time and batch
  // dimensions.
  std::optional<mlir::MlirOp> grad_bias_fwd;
  std::optional<mlir::MlirOp> grad_bias_rev;
  if (has_biases) {
    mlir::MlirOp grad_b_fwd_clumped = mlir::stablehlo::Reduce(
        builder, delta_pre_seq_fwd, zero_const, sum_reduce_builder, {0, 1})[0];
    grad_bias_fwd = to_out(UnclumpBias3H(builder, grad_b_fwd_clumped, hidden));

    mlir::MlirOp grad_b_rev_clumped = mlir::stablehlo::Reduce(
        builder, delta_pre_seq_rev, zero_const, sum_reduce_builder, {0, 1})[0];
    grad_bias_rev = to_out(UnclumpBias3H(builder, grad_b_rev_clumped, hidden));
  }

  // Projection weight gradients (if enabled).
  std::optional<mlir::MlirOp> grad_w_hr_fwd;
  std::optional<mlir::MlirOp> grad_w_hr_rev;
  if (w_hr_fwd.has_value()) {
    mlir::MlirOp delta_h_seq_fwd =
        ConcatDim(builder, delta_h_steps_fwd, concat_dim);
    mlir::MlirOp delta_h_2d_fwd =
        mlir::stablehlo::Reshape(delta_h_seq_fwd, {seq_len * batch, out_h});
    mlir::MlirOp r_seq_fwd = ConcatDim(builder, r_steps_fwd, concat_dim);
    mlir::MlirOp r_2d_fwd =
        mlir::stablehlo::Reshape(r_seq_fwd, {seq_len * batch, hidden});
    grad_w_hr_fwd = to_out(MixedPrecisionDotGeneral(delta_h_2d_fwd, r_2d_fwd,
                                                    grad_w_dot_dims, acc_dtype,
                                                    out_dtype, precision));

    mlir::MlirOp delta_h_seq_rev =
        ConcatDim(builder, delta_h_steps_rev, concat_dim);
    mlir::MlirOp delta_h_2d_rev =
        mlir::stablehlo::Reshape(delta_h_seq_rev, {seq_len * batch, out_h});
    mlir::MlirOp r_seq_rev = ConcatDim(builder, r_steps_rev, concat_dim);
    mlir::MlirOp r_2d_rev =
        mlir::stablehlo::Reshape(r_seq_rev, {seq_len * batch, hidden});
    grad_w_hr_rev = to_out(MixedPrecisionDotGeneral(delta_h_2d_rev, r_2d_rev,
                                                    grad_w_dot_dims, acc_dtype,
                                                    out_dtype, precision));
  }

  return {grad_x_layer,          grad_h0_fwd,           grad_c0_fwd,
          grad_h0_rev,           grad_c0_rev,           to_out(grad_w_ih_fwd),
          to_out(grad_w_hh_fwd), to_out(grad_w_ih_rev), to_out(grad_w_hh_rev),
          grad_bias_fwd,         grad_bias_rev,         grad_w_hr_fwd,
          grad_w_hr_rev};
}

// Chunked streaming helper for bidirectional LSTM backward pass (seq_len >=
// kLstmUnrollFactor).
//
// Key Optimizations:
// 1. Partial unrolling of kLstmUnrollFactor timesteps inside a single StableHLO
// while-loop.
// 2. Lockstep concurrent execution: forward and reverse branches execute
// concurrently within
//    each iteration, doubling arithmetic intensity and instruction scheduling
//    opportunities.
// 3. Dynamic slicing: Slices only kLstmUnrollFactor steps of forward
// activations per iteration,
//    minimizing register pressure and keeping the active working set in TPU
//    on-chip memory.
// 4. Streaming parameter gradient accumulation: Directly accumulates W_ih,
// W_hh, bias, and
//    W_hr gradients inside the loop carry tuple. Eliminates full-sequence
//    intermediate adjoint buffers in HBM, reducing peak memory from O(T) to
//    O(kLstmUnrollFactor).
// 5. Tail remainder handling: Statically unrolls any leftover steps (seq_len %
// kLstmUnrollFactor)
//    upfront at sequence boundary t = seq_len - 1 down to chunked_steps,
//    initializing the loop carry states.
template <typename ToOutFn, typename ReduceBuilderFn>
LstmBidirLayerBackwardOutputs ComputeBidirLayerBackwardChunked(
    mlir::MlirBuilder& builder, const LayerForwardState& state_fwd,
    const LayerForwardState& state_rev, mlir::MlirOp grad_y_fwd,
    mlir::MlirOp grad_y_rev_time_reversed, mlir::MlirOp grad_hy_fwd,
    mlir::MlirOp grad_cy_fwd, mlir::MlirOp grad_hy_rev,
    mlir::MlirOp grad_cy_rev, mlir::MlirOp w_ih_fwd_clumped,
    mlir::MlirOp w_hh_fwd_clumped, mlir::MlirOp w_ih_rev_clumped,
    mlir::MlirOp w_hh_rev_clumped, const bool has_biases,
    mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    mlir::stablehlo::DotDimensionNumbersAttr bwd_weight_dot_dims,
    mlir::stablehlo::DotDimensionNumbersAttr grad_w_dot_dims,
    const int64_t seq_len, const int64_t batch, const int64_t hidden,
    const int64_t in_dim, const int64_t out_h, ToOutFn to_out,
    std::optional<mlir::MlirOp> w_hr_fwd, std::optional<mlir::MlirOp> w_hr_rev,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> bwd_hr_dot_dims,
    const bool batch_first, mlir::stablehlo::Precision precision) {
  const int64_t concat_dim = batch_first ? 1 : 0;
  // --- STREAMING / CHUNKED CONCURRENT BIDIRECTIONAL BPTT EXECUTION
  // (kLstmUnrollFactor) ---
  const int64_t num_chunks = seq_len / kLstmUnrollFactor;
  const int64_t chunked_steps = num_chunks * kLstmUnrollFactor;
  const int64_t rem_steps = seq_len % kLstmUnrollFactor;

  mlir::MlirOp delta_c_next_fwd = grad_cy_fwd;
  mlir::MlirOp delta_h_rec_fwd = grad_hy_fwd;
  mlir::MlirOp delta_c_next_rev = grad_cy_rev;
  mlir::MlirOp delta_h_rec_rev = grad_hy_rev;

  mlir::MlirOp layer_in_fwd = state_fwd.layer_input;
  mlir::MlirOp layer_in_rev = state_rev.layer_input;
  mlir::MlirOp h_prev_fwd = state_fwd.h_prev_seq;
  mlir::MlirOp h_prev_rev = state_rev.h_prev_seq;

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = grad_y_fwd.getValue().getLoc();
  const mlir::IntegerType i64 = op_builder.getI64Type();
  const mlir::RankedTensorType i64_scalar_type =
      mlir::RankedTensorType::get({}, i64);

  const mlir::Type acc_elem_type =
      GetTensorTypeOrDie(delta_h_rec_fwd).getElementType();
  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0f, acc_elem_type);

  mlir::MlirOp grad_w_ih_init_fwd;
  mlir::MlirOp grad_w_hh_init_fwd;
  mlir::MlirOp grad_w_ih_init_rev;
  mlir::MlirOp grad_w_hh_init_rev;
  mlir::MlirOp grad_b_init_fwd;
  mlir::MlirOp grad_b_init_rev;
  mlir::MlirOp grad_w_hr_init_fwd;
  mlir::MlirOp grad_w_hr_init_rev;
  mlir::MlirOp rem_grad_x_fwd;
  mlir::MlirOp rem_grad_x_rev;

  if (rem_steps > 0) {
    std::vector<mlir::MlirOp> delta_pre_rem_steps_fwd(rem_steps);
    std::vector<mlir::MlirOp> delta_pre_rem_steps_rev(rem_steps);
    std::vector<mlir::MlirOp> delta_h_rem_steps_fwd(
        w_hr_fwd.has_value() ? rem_steps : 0);
    std::vector<mlir::MlirOp> delta_h_rem_steps_rev(
        w_hr_rev.has_value() ? rem_steps : 0);
    std::vector<mlir::MlirOp> r_rem_steps_fwd(w_hr_fwd.has_value() ? rem_steps
                                                                   : 0);
    std::vector<mlir::MlirOp> r_rem_steps_rev(w_hr_rev.has_value() ? rem_steps
                                                                   : 0);

    for (int64_t t = seq_len - 1; t >= chunked_steps; --t) {
      const int64_t rem_idx = t - chunked_steps;

      // Forward remainder step t
      mlir::MlirOp grad_from_above_fwd =
          SliceStep2D(grad_y_fwd, t, batch, out_h, batch_first);
      mlir::MlirOp delta_h_fwd =
          mlir::stablehlo::Add(grad_from_above_fwd, delta_h_rec_fwd);

      mlir::MlirOp tanh_c_fwd = GetStepActivation(
          builder, state_fwd.tanh_c_list, state_fwd.tanh_c_seq, t, batch,
          hidden, batch_first);
      mlir::MlirOp ifo_fwd =
          GetStepActivation(builder, state_fwd.ifo_list, state_fwd.ifo_seq, t,
                            batch, 3 * hidden, batch_first);

      BackwardStepResult step_fwd = ComputeBackwardStep(
          builder, delta_h_fwd, delta_c_next_fwd, tanh_c_fwd,
          GetStepActivation(builder, state_fwd.c_prev_list,
                            state_fwd.c_prev_seq, t, batch, hidden,
                            batch_first),
          GetStepActivation(builder, state_fwd.cellgate_list,
                            state_fwd.cellgate_seq, t, batch, hidden,
                            batch_first),
          ifo_fwd, w_hh_fwd_clumped, bwd_weight_dot_dims, batch, hidden,
          w_hr_fwd, bwd_hr_dot_dims, precision);
      delta_c_next_fwd = step_fwd.delta_c_next;
      delta_h_rec_fwd = step_fwd.delta_h_rec;
      delta_pre_rem_steps_fwd[rem_idx] =
          ExpandStep3D(step_fwd.delta_pre, batch, 4 * hidden, batch_first);

      if (w_hr_fwd.has_value()) {
        delta_h_rem_steps_fwd[rem_idx] =
            ExpandStep3D(delta_h_fwd, batch, out_h, batch_first);
        mlir::MlirOp o_val_fwd = mlir::stablehlo::Slice(
            ifo_fwd, {0, 2 * hidden}, {batch, 3 * hidden}, {1, 1});
        mlir::MlirOp r_val_fwd = mlir::stablehlo::Mul(o_val_fwd, tanh_c_fwd);
        r_rem_steps_fwd[rem_idx] =
            ExpandStep3D(r_val_fwd, batch, hidden, batch_first);
      }

      // Reverse remainder step t (running concurrently)
      mlir::MlirOp grad_from_above_rev =
          SliceStep2D(grad_y_rev_time_reversed, t, batch, out_h, batch_first);
      mlir::MlirOp delta_h_rev =
          mlir::stablehlo::Add(grad_from_above_rev, delta_h_rec_rev);

      mlir::MlirOp tanh_c_rev = GetStepActivation(
          builder, state_rev.tanh_c_list, state_rev.tanh_c_seq, t, batch,
          hidden, batch_first);
      mlir::MlirOp ifo_rev =
          GetStepActivation(builder, state_rev.ifo_list, state_rev.ifo_seq, t,
                            batch, 3 * hidden, batch_first);

      BackwardStepResult step_rev = ComputeBackwardStep(
          builder, delta_h_rev, delta_c_next_rev, tanh_c_rev,
          GetStepActivation(builder, state_rev.c_prev_list,
                            state_rev.c_prev_seq, t, batch, hidden,
                            batch_first),
          GetStepActivation(builder, state_rev.cellgate_list,
                            state_rev.cellgate_seq, t, batch, hidden,
                            batch_first),
          ifo_rev, w_hh_rev_clumped, bwd_weight_dot_dims, batch, hidden,
          w_hr_rev, bwd_hr_dot_dims, precision);
      delta_c_next_rev = step_rev.delta_c_next;
      delta_h_rec_rev = step_rev.delta_h_rec;
      delta_pre_rem_steps_rev[rem_idx] =
          ExpandStep3D(step_rev.delta_pre, batch, 4 * hidden, batch_first);

      if (w_hr_rev.has_value()) {
        delta_h_rem_steps_rev[rem_idx] =
            ExpandStep3D(delta_h_rev, batch, out_h, batch_first);
        mlir::MlirOp o_val_rev = mlir::stablehlo::Slice(
            ifo_rev, {0, 2 * hidden}, {batch, 3 * hidden}, {1, 1});
        mlir::MlirOp r_val_rev = mlir::stablehlo::Mul(o_val_rev, tanh_c_rev);
        r_rem_steps_rev[rem_idx] =
            ExpandStep3D(r_val_rev, batch, hidden, batch_first);
      }
    }

    mlir::MlirOp delta_pre_rem_fwd =
        ConcatDim(builder, delta_pre_rem_steps_fwd, concat_dim);
    mlir::MlirOp delta_pre_rem_rev =
        ConcatDim(builder, delta_pre_rem_steps_rev, concat_dim);
    mlir::MlirOp delta_pre_rem_2d_fwd = mlir::stablehlo::Reshape(
        delta_pre_rem_fwd, {rem_steps * batch, 4 * hidden});
    mlir::MlirOp delta_pre_rem_2d_rev = mlir::stablehlo::Reshape(
        delta_pre_rem_rev, {rem_steps * batch, 4 * hidden});

    mlir::MlirOp rem_layer_in_fwd =
        batch_first
            ? mlir::stablehlo::Slice(layer_in_fwd, {0, chunked_steps, 0},
                                     {batch, seq_len, in_dim}, {1, 1, 1})
            : mlir::stablehlo::Slice(layer_in_fwd, {chunked_steps, 0, 0},
                                     {seq_len, batch, in_dim}, {1, 1, 1});
    mlir::MlirOp rem_layer_in_2d_fwd =
        mlir::stablehlo::Reshape(rem_layer_in_fwd, {rem_steps * batch, in_dim});
    const mlir::Type acc_dtype =
        GetTensorTypeOrDie(delta_pre_rem_2d_fwd).getElementType();
    const mlir::Type out_dtype =
        GetTensorTypeOrDie(w_ih_fwd_clumped).getElementType();

    grad_w_ih_init_fwd = MixedPrecisionDotGeneral(
        delta_pre_rem_2d_fwd, rem_layer_in_2d_fwd, grad_w_dot_dims, acc_dtype,
        out_dtype, precision);

    mlir::MlirOp rem_layer_in_rev =
        batch_first
            ? mlir::stablehlo::Slice(layer_in_rev, {0, chunked_steps, 0},
                                     {batch, seq_len, in_dim}, {1, 1, 1})
            : mlir::stablehlo::Slice(layer_in_rev, {chunked_steps, 0, 0},
                                     {seq_len, batch, in_dim}, {1, 1, 1});
    mlir::MlirOp rem_layer_in_2d_rev =
        mlir::stablehlo::Reshape(rem_layer_in_rev, {rem_steps * batch, in_dim});
    grad_w_ih_init_rev = MixedPrecisionDotGeneral(
        delta_pre_rem_2d_rev, rem_layer_in_2d_rev, grad_w_dot_dims, acc_dtype,
        out_dtype, precision);

    mlir::MlirOp rem_h_prev_fwd =
        batch_first
            ? mlir::stablehlo::Slice(h_prev_fwd, {0, chunked_steps, 0},
                                     {batch, seq_len, out_h}, {1, 1, 1})
            : mlir::stablehlo::Slice(h_prev_fwd, {chunked_steps, 0, 0},
                                     {seq_len, batch, out_h}, {1, 1, 1});
    mlir::MlirOp rem_h_prev_2d_fwd =
        mlir::stablehlo::Reshape(rem_h_prev_fwd, {rem_steps * batch, out_h});
    grad_w_hh_init_fwd = MixedPrecisionDotGeneral(
        delta_pre_rem_2d_fwd, rem_h_prev_2d_fwd, grad_w_dot_dims, acc_dtype,
        out_dtype, precision);

    mlir::MlirOp rem_h_prev_rev =
        batch_first
            ? mlir::stablehlo::Slice(h_prev_rev, {0, chunked_steps, 0},
                                     {batch, seq_len, out_h}, {1, 1, 1})
            : mlir::stablehlo::Slice(h_prev_rev, {chunked_steps, 0, 0},
                                     {seq_len, batch, out_h}, {1, 1, 1});
    mlir::MlirOp rem_h_prev_2d_rev =
        mlir::stablehlo::Reshape(rem_h_prev_rev, {rem_steps * batch, out_h});
    grad_w_hh_init_rev = MixedPrecisionDotGeneral(
        delta_pre_rem_2d_rev, rem_h_prev_2d_rev, grad_w_dot_dims, acc_dtype,
        out_dtype, precision);

    if (has_biases) {
      grad_b_init_fwd =
          mlir::stablehlo::Reduce(builder, delta_pre_rem_fwd, zero_const,
                                  sum_reduce_builder, {0, 1})[0];
      grad_b_init_rev =
          mlir::stablehlo::Reduce(builder, delta_pre_rem_rev, zero_const,
                                  sum_reduce_builder, {0, 1})[0];
    } else {
      grad_b_init_fwd = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({4 * hidden}, acc_elem_type), zero_scalar,
          {});
      grad_b_init_rev = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({4 * hidden}, acc_elem_type), zero_scalar,
          {});
    }

    if (w_hr_fwd.has_value()) {
      mlir::MlirOp delta_h_rem_fwd =
          ConcatDim(builder, delta_h_rem_steps_fwd, concat_dim);
      mlir::MlirOp delta_h_rem_2d_fwd =
          mlir::stablehlo::Reshape(delta_h_rem_fwd, {rem_steps * batch, out_h});
      mlir::MlirOp r_rem_fwd = ConcatDim(builder, r_rem_steps_fwd, concat_dim);
      mlir::MlirOp r_rem_2d_fwd =
          mlir::stablehlo::Reshape(r_rem_fwd, {rem_steps * batch, hidden});
      grad_w_hr_init_fwd = MixedPrecisionDotGeneral(
          delta_h_rem_2d_fwd, r_rem_2d_fwd, grad_w_dot_dims, acc_dtype,
          out_dtype, precision);

      mlir::MlirOp delta_h_rem_rev =
          ConcatDim(builder, delta_h_rem_steps_rev, concat_dim);
      mlir::MlirOp delta_h_rem_2d_rev =
          mlir::stablehlo::Reshape(delta_h_rem_rev, {rem_steps * batch, out_h});
      mlir::MlirOp r_rem_rev = ConcatDim(builder, r_rem_steps_rev, concat_dim);
      mlir::MlirOp r_rem_2d_rev =
          mlir::stablehlo::Reshape(r_rem_rev, {rem_steps * batch, hidden});
      grad_w_hr_init_rev = MixedPrecisionDotGeneral(
          delta_h_rem_2d_rev, r_rem_2d_rev, grad_w_dot_dims, acc_dtype,
          out_dtype, precision);
    }

    mlir::MlirOp rem_grad_x_2d_fwd =
        MixedPrecisionDotGeneral(delta_pre_rem_2d_fwd, w_ih_fwd_clumped,
                                 bwd_weight_dot_dims, acc_dtype, precision);
    rem_grad_x_fwd = batch_first
                         ? mlir::stablehlo::Reshape(rem_grad_x_2d_fwd,
                                                    {batch, rem_steps, in_dim})
                         : mlir::stablehlo::Reshape(rem_grad_x_2d_fwd,
                                                    {rem_steps, batch, in_dim});

    mlir::MlirOp rem_grad_x_2d_rev =
        MixedPrecisionDotGeneral(delta_pre_rem_2d_rev, w_ih_rev_clumped,
                                 bwd_weight_dot_dims, acc_dtype, precision);
    rem_grad_x_rev = batch_first
                         ? mlir::stablehlo::Reshape(rem_grad_x_2d_rev,
                                                    {batch, rem_steps, in_dim})
                         : mlir::stablehlo::Reshape(rem_grad_x_2d_rev,
                                                    {rem_steps, batch, in_dim});
  } else {
    grad_w_ih_init_fwd = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({4 * hidden, in_dim}, acc_elem_type),
        zero_scalar, {});
    grad_w_ih_init_rev = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({4 * hidden, in_dim}, acc_elem_type),
        zero_scalar, {});
    grad_w_hh_init_fwd = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({4 * hidden, out_h}, acc_elem_type),
        zero_scalar, {});
    grad_w_hh_init_rev = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({4 * hidden, out_h}, acc_elem_type),
        zero_scalar, {});
    grad_b_init_fwd = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({4 * hidden}, acc_elem_type), zero_scalar,
        {});
    grad_b_init_rev = mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get({4 * hidden}, acc_elem_type), zero_scalar,
        {});
    if (w_hr_fwd.has_value()) {
      grad_w_hr_init_fwd = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({out_h, hidden}, acc_elem_type),
          zero_scalar, {});
      grad_w_hr_init_rev = mlir::stablehlo::BroadcastInDim(
          mlir::RankedTensorType::get({out_h, hidden}, acc_elem_type),
          zero_scalar, {});
    }
  }

  const mlir::RankedTensorType grad_x_chunked_type =
      mlir::RankedTensorType::get(
          batch_first ? llvm::ArrayRef<int64_t>{batch, chunked_steps, in_dim}
                      : llvm::ArrayRef<int64_t>{chunked_steps, batch, in_dim},
          acc_elem_type);
  mlir::MlirOp grad_x_chunked_init_fwd =
      mlir::stablehlo::BroadcastInDim(grad_x_chunked_type, zero_scalar, {});
  mlir::MlirOp grad_x_chunked_init_rev =
      mlir::stablehlo::BroadcastInDim(grad_x_chunked_type, zero_scalar, {});

  mlir::MlirOp step_idx_init = MakeScalarConstant(builder, 0, i64);

  llvm::SmallVector<mlir::Type> loop_types = {i64_scalar_type,
                                              delta_h_rec_fwd.getType(),
                                              delta_c_next_fwd.getType(),
                                              delta_h_rec_rev.getType(),
                                              delta_c_next_rev.getType(),
                                              grad_x_chunked_type,
                                              grad_x_chunked_type,
                                              grad_w_ih_init_fwd.getType(),
                                              grad_w_hh_init_fwd.getType(),
                                              grad_w_ih_init_rev.getType(),
                                              grad_w_hh_init_rev.getType()};
  llvm::SmallVector<mlir::Value> loop_inits = {
      step_idx_init.getValue(),           delta_h_rec_fwd.getValue(),
      delta_c_next_fwd.getValue(),        delta_h_rec_rev.getValue(),
      delta_c_next_rev.getValue(),        grad_x_chunked_init_fwd.getValue(),
      grad_x_chunked_init_rev.getValue(), grad_w_ih_init_fwd.getValue(),
      grad_w_hh_init_fwd.getValue(),      grad_w_ih_init_rev.getValue(),
      grad_w_hh_init_rev.getValue()};

  if (has_biases) {
    loop_types.push_back(grad_b_init_fwd.getType());
    loop_types.push_back(grad_b_init_rev.getType());
    loop_inits.push_back(grad_b_init_fwd.getValue());
    loop_inits.push_back(grad_b_init_rev.getValue());
  }
  if (w_hr_fwd.has_value()) {
    loop_types.push_back(grad_w_hr_init_fwd.getType());
    loop_types.push_back(grad_w_hr_init_rev.getType());
    loop_inits.push_back(grad_w_hr_init_fwd.getValue());
    loop_inits.push_back(grad_w_hr_init_rev.getValue());
  }

  auto while_op =
      mlir::stablehlo::WhileOp::create(op_builder, loc, loop_types, loop_inits);

  // Cond region: step_idx < chunked_steps
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

  // Body region: processes kLstmUnrollFactor forward & reverse steps
  // concurrently with streaming accumulation
  mlir::Block* const body_block = op_builder.createBlock(&while_op.getBody());
  body_block->addArguments(
      loop_types, llvm::SmallVector<mlir::Location>(loop_types.size(), loc));
  op_builder.setInsertionPointToStart(body_block);

  mlir::MlirOp body_step_idx(builder, body_block->getArgument(0));
  mlir::MlirOp body_delta_h_fwd(builder, body_block->getArgument(1));
  mlir::MlirOp body_delta_c_fwd(builder, body_block->getArgument(2));
  mlir::MlirOp body_delta_h_rev(builder, body_block->getArgument(3));
  mlir::MlirOp body_delta_c_rev(builder, body_block->getArgument(4));
  mlir::MlirOp body_grad_x_fwd(builder, body_block->getArgument(5));
  mlir::MlirOp body_grad_x_rev(builder, body_block->getArgument(6));
  mlir::MlirOp body_grad_w_ih_fwd(builder, body_block->getArgument(7));
  mlir::MlirOp body_grad_w_hh_fwd(builder, body_block->getArgument(8));
  mlir::MlirOp body_grad_w_ih_rev(builder, body_block->getArgument(9));
  mlir::MlirOp body_grad_w_hh_rev(builder, body_block->getArgument(10));
  int arg_idx = 11;
  mlir::MlirOp body_grad_b_fwd;
  mlir::MlirOp body_grad_b_rev;
  if (has_biases) {
    body_grad_b_fwd = mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
    body_grad_b_rev = mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
  }
  mlir::MlirOp body_grad_w_hr_fwd;
  mlir::MlirOp body_grad_w_hr_rev;
  if (w_hr_fwd.has_value()) {
    body_grad_w_hr_fwd =
        mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
    body_grad_w_hr_rev =
        mlir::MlirOp(builder, body_block->getArgument(arg_idx++));
  }

  mlir::MlirOp const_chunk_limit =
      MakeScalarConstant(builder, chunked_steps - kLstmUnrollFactor, i64);
  mlir::MlirOp chunk_start_t =
      mlir::stablehlo::Subtract(const_chunk_limit, body_step_idx);

  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64);
  const llvm::SmallVector<mlir::MlirOp, 3> chunk_start_indices =
      batch_first ? llvm::SmallVector<mlir::MlirOp, 3>{zero_i64, chunk_start_t,
                                                       zero_i64}
                  : llvm::SmallVector<mlir::MlirOp, 3>{chunk_start_t, zero_i64,
                                                       zero_i64};
  const llvm::SmallVector<int64_t, 3> out_h_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kLstmUnrollFactor, out_h}
          : llvm::SmallVector<int64_t, 3>{kLstmUnrollFactor, batch, out_h};
  const llvm::SmallVector<int64_t, 3> hidden_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kLstmUnrollFactor, hidden}
          : llvm::SmallVector<int64_t, 3>{kLstmUnrollFactor, batch, hidden};
  const llvm::SmallVector<int64_t, 3> ifo_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kLstmUnrollFactor, 3 * hidden}
          : llvm::SmallVector<int64_t, 3>{kLstmUnrollFactor, batch, 3 * hidden};
  const llvm::SmallVector<int64_t, 3> in_slice_sizes =
      batch_first
          ? llvm::SmallVector<int64_t, 3>{batch, kLstmUnrollFactor, in_dim}
          : llvm::SmallVector<int64_t, 3>{kLstmUnrollFactor, batch, in_dim};

  mlir::MlirOp ifo_seq_fwd = state_fwd.ifo_seq;
  mlir::MlirOp cellgate_seq_fwd = state_fwd.cellgate_seq;
  mlir::MlirOp tanh_c_seq_fwd = state_fwd.tanh_c_seq;
  mlir::MlirOp c_prev_seq_fwd = state_fwd.c_prev_seq;

  mlir::MlirOp ifo_seq_rev = state_rev.ifo_seq;
  mlir::MlirOp cellgate_seq_rev = state_rev.cellgate_seq;
  mlir::MlirOp tanh_c_seq_rev = state_rev.tanh_c_seq;
  mlir::MlirOp c_prev_seq_rev = state_rev.c_prev_seq;

  // Dynamic slices for forward branch
  mlir::MlirOp grad_x_chunk_fwd = mlir::stablehlo::DynamicSlice(
      grad_y_fwd, chunk_start_indices, out_h_slice_sizes);
  mlir::MlirOp ifo_chunk_fwd = mlir::stablehlo::DynamicSlice(
      ifo_seq_fwd, chunk_start_indices, ifo_slice_sizes);
  mlir::MlirOp cellgate_chunk_fwd = mlir::stablehlo::DynamicSlice(
      cellgate_seq_fwd, chunk_start_indices, hidden_slice_sizes);
  mlir::MlirOp tanh_c_chunk_fwd = mlir::stablehlo::DynamicSlice(
      tanh_c_seq_fwd, chunk_start_indices, hidden_slice_sizes);
  mlir::MlirOp c_prev_chunk_fwd = mlir::stablehlo::DynamicSlice(
      c_prev_seq_fwd, chunk_start_indices, hidden_slice_sizes);

  // Dynamic slices for reverse branch
  mlir::MlirOp grad_x_chunk_rev = mlir::stablehlo::DynamicSlice(
      grad_y_rev_time_reversed, chunk_start_indices, out_h_slice_sizes);
  mlir::MlirOp ifo_chunk_rev = mlir::stablehlo::DynamicSlice(
      ifo_seq_rev, chunk_start_indices, ifo_slice_sizes);
  mlir::MlirOp cellgate_chunk_rev = mlir::stablehlo::DynamicSlice(
      cellgate_seq_rev, chunk_start_indices, hidden_slice_sizes);
  mlir::MlirOp tanh_c_chunk_rev = mlir::stablehlo::DynamicSlice(
      tanh_c_seq_rev, chunk_start_indices, hidden_slice_sizes);
  mlir::MlirOp c_prev_chunk_rev = mlir::stablehlo::DynamicSlice(
      c_prev_seq_rev, chunk_start_indices, hidden_slice_sizes);

  mlir::MlirOp cur_delta_h_fwd = body_delta_h_fwd;
  mlir::MlirOp cur_delta_c_fwd = body_delta_c_fwd;
  mlir::MlirOp cur_delta_h_rev = body_delta_h_rev;
  mlir::MlirOp cur_delta_c_rev = body_delta_c_rev;

  std::vector<mlir::MlirOp> delta_pre_chunk_steps_fwd(kLstmUnrollFactor);
  std::vector<mlir::MlirOp> delta_pre_chunk_steps_rev(kLstmUnrollFactor);
  std::vector<mlir::MlirOp> delta_h_chunk_steps_fwd(
      w_hr_fwd.has_value() ? kLstmUnrollFactor : 0);
  std::vector<mlir::MlirOp> delta_h_chunk_steps_rev(
      w_hr_rev.has_value() ? kLstmUnrollFactor : 0);
  std::vector<mlir::MlirOp> r_chunk_steps_fwd(
      w_hr_fwd.has_value() ? kLstmUnrollFactor : 0);
  std::vector<mlir::MlirOp> r_chunk_steps_rev(
      w_hr_rev.has_value() ? kLstmUnrollFactor : 0);

  for (int64_t k = kLstmUnrollFactor - 1; k >= 0; --k) {
    // Forward step k
    mlir::MlirOp grad_from_above_k_fwd =
        SliceStep2D(grad_x_chunk_fwd, k, batch, out_h, batch_first);
    mlir::MlirOp delta_h_fwd =
        mlir::stablehlo::Add(grad_from_above_k_fwd, cur_delta_h_fwd);

    mlir::MlirOp tanh_c_k_fwd =
        SliceStep2D(tanh_c_chunk_fwd, k, batch, hidden, batch_first);
    mlir::MlirOp c_prev_k_fwd =
        SliceStep2D(c_prev_chunk_fwd, k, batch, hidden, batch_first);
    mlir::MlirOp g_k_fwd =
        SliceStep2D(cellgate_chunk_fwd, k, batch, hidden, batch_first);
    mlir::MlirOp ifo_k_fwd =
        SliceStep2D(ifo_chunk_fwd, k, batch, 3 * hidden, batch_first);

    BackwardStepResult step_fwd = ComputeBackwardStep(
        builder, delta_h_fwd, cur_delta_c_fwd, tanh_c_k_fwd, c_prev_k_fwd,
        g_k_fwd, ifo_k_fwd, w_hh_fwd_clumped, bwd_weight_dot_dims, batch,
        hidden, w_hr_fwd, bwd_hr_dot_dims, precision);
    cur_delta_c_fwd = step_fwd.delta_c_next;
    cur_delta_h_fwd = step_fwd.delta_h_rec;
    delta_pre_chunk_steps_fwd[k] =
        ExpandStep3D(step_fwd.delta_pre, batch, 4 * hidden, batch_first);

    if (w_hr_fwd.has_value()) {
      delta_h_chunk_steps_fwd[k] =
          ExpandStep3D(delta_h_fwd, batch, out_h, batch_first);
      mlir::MlirOp o_k_fwd = mlir::stablehlo::Slice(
          ifo_k_fwd, {0, 2 * hidden}, {batch, 3 * hidden}, {1, 1});
      mlir::MlirOp r_k_fwd = mlir::stablehlo::Mul(o_k_fwd, tanh_c_k_fwd);
      r_chunk_steps_fwd[k] = ExpandStep3D(r_k_fwd, batch, hidden, batch_first);
    }

    // Reverse step k (evaluated concurrently)
    mlir::MlirOp grad_from_above_k_rev =
        SliceStep2D(grad_x_chunk_rev, k, batch, out_h, batch_first);
    mlir::MlirOp delta_h_rev =
        mlir::stablehlo::Add(grad_from_above_k_rev, cur_delta_h_rev);

    mlir::MlirOp tanh_c_k_rev =
        SliceStep2D(tanh_c_chunk_rev, k, batch, hidden, batch_first);
    mlir::MlirOp c_prev_k_rev =
        SliceStep2D(c_prev_chunk_rev, k, batch, hidden, batch_first);
    mlir::MlirOp g_k_rev =
        SliceStep2D(cellgate_chunk_rev, k, batch, hidden, batch_first);
    mlir::MlirOp ifo_k_rev =
        SliceStep2D(ifo_chunk_rev, k, batch, 3 * hidden, batch_first);

    BackwardStepResult step_rev = ComputeBackwardStep(
        builder, delta_h_rev, cur_delta_c_rev, tanh_c_k_rev, c_prev_k_rev,
        g_k_rev, ifo_k_rev, w_hh_rev_clumped, bwd_weight_dot_dims, batch,
        hidden, w_hr_rev, bwd_hr_dot_dims, precision);
    cur_delta_c_rev = step_rev.delta_c_next;
    cur_delta_h_rev = step_rev.delta_h_rec;
    delta_pre_chunk_steps_rev[k] =
        ExpandStep3D(step_rev.delta_pre, batch, 4 * hidden, batch_first);

    if (w_hr_rev.has_value()) {
      delta_h_chunk_steps_rev[k] =
          ExpandStep3D(delta_h_rev, batch, out_h, batch_first);
      mlir::MlirOp o_k_rev = mlir::stablehlo::Slice(
          ifo_k_rev, {0, 2 * hidden}, {batch, 3 * hidden}, {1, 1});
      mlir::MlirOp r_k_rev = mlir::stablehlo::Mul(o_k_rev, tanh_c_k_rev);
      r_chunk_steps_rev[k] = ExpandStep3D(r_k_rev, batch, hidden, batch_first);
    }
  }

  mlir::MlirOp delta_pre_chunk_fwd =
      ConcatDim(builder, delta_pre_chunk_steps_fwd, concat_dim);
  mlir::MlirOp delta_pre_chunk_rev =
      ConcatDim(builder, delta_pre_chunk_steps_rev, concat_dim);

  mlir::MlirOp delta_pre_chunk_2d_fwd = mlir::stablehlo::Reshape(
      delta_pre_chunk_fwd, {kLstmUnrollFactor * batch, 4 * hidden});
  mlir::MlirOp delta_pre_chunk_2d_rev = mlir::stablehlo::Reshape(
      delta_pre_chunk_rev, {kLstmUnrollFactor * batch, 4 * hidden});

  const mlir::Type acc_dtype =
      GetTensorTypeOrDie(delta_pre_chunk_2d_fwd).getElementType();
  const mlir::Type out_dtype =
      GetTensorTypeOrDie(w_ih_fwd_clumped).getElementType();

  // Streaming accumulation: Input weight gradients
  mlir::MlirOp layer_in_chunk_fwd = mlir::stablehlo::DynamicSlice(
      layer_in_fwd, chunk_start_indices, in_slice_sizes);
  mlir::MlirOp layer_in_chunk_2d_fwd = mlir::stablehlo::Reshape(
      layer_in_chunk_fwd, {kLstmUnrollFactor * batch, in_dim});
  mlir::MlirOp chunk_grad_w_ih_fwd = MixedPrecisionDotGeneral(
      delta_pre_chunk_2d_fwd, layer_in_chunk_2d_fwd, grad_w_dot_dims, acc_dtype,
      out_dtype, precision);
  mlir::MlirOp next_grad_w_ih_fwd =
      mlir::stablehlo::Add(body_grad_w_ih_fwd, chunk_grad_w_ih_fwd);

  mlir::MlirOp layer_in_chunk_rev = mlir::stablehlo::DynamicSlice(
      layer_in_rev, chunk_start_indices, in_slice_sizes);
  mlir::MlirOp layer_in_chunk_2d_rev = mlir::stablehlo::Reshape(
      layer_in_chunk_rev, {kLstmUnrollFactor * batch, in_dim});
  mlir::MlirOp chunk_grad_w_ih_rev = MixedPrecisionDotGeneral(
      delta_pre_chunk_2d_rev, layer_in_chunk_2d_rev, grad_w_dot_dims, acc_dtype,
      out_dtype, precision);
  mlir::MlirOp next_grad_w_ih_rev =
      mlir::stablehlo::Add(body_grad_w_ih_rev, chunk_grad_w_ih_rev);

  // Streaming accumulation: Recurrent weight gradients
  mlir::MlirOp h_prev_chunk_fwd = mlir::stablehlo::DynamicSlice(
      h_prev_fwd, chunk_start_indices, out_h_slice_sizes);
  mlir::MlirOp h_prev_chunk_2d_fwd = mlir::stablehlo::Reshape(
      h_prev_chunk_fwd, {kLstmUnrollFactor * batch, out_h});
  mlir::MlirOp chunk_grad_w_hh_fwd = MixedPrecisionDotGeneral(
      delta_pre_chunk_2d_fwd, h_prev_chunk_2d_fwd, grad_w_dot_dims, acc_dtype,
      out_dtype, precision);
  mlir::MlirOp next_grad_w_hh_fwd =
      mlir::stablehlo::Add(body_grad_w_hh_fwd, chunk_grad_w_hh_fwd);

  mlir::MlirOp h_prev_chunk_rev = mlir::stablehlo::DynamicSlice(
      h_prev_rev, chunk_start_indices, out_h_slice_sizes);
  mlir::MlirOp h_prev_chunk_2d_rev = mlir::stablehlo::Reshape(
      h_prev_chunk_rev, {kLstmUnrollFactor * batch, out_h});
  mlir::MlirOp chunk_grad_w_hh_rev = MixedPrecisionDotGeneral(
      delta_pre_chunk_2d_rev, h_prev_chunk_2d_rev, grad_w_dot_dims, acc_dtype,
      out_dtype, precision);
  mlir::MlirOp next_grad_w_hh_rev =
      mlir::stablehlo::Add(body_grad_w_hh_rev, chunk_grad_w_hh_rev);

  // Streaming accumulation: Bias gradients
  mlir::MlirOp next_grad_b_fwd;
  mlir::MlirOp next_grad_b_rev;
  if (has_biases) {
    mlir::MlirOp chunk_grad_b_fwd =
        mlir::stablehlo::Reduce(builder, delta_pre_chunk_fwd, zero_const,
                                sum_reduce_builder, {0, 1})[0];
    next_grad_b_fwd = mlir::stablehlo::Add(body_grad_b_fwd, chunk_grad_b_fwd);

    mlir::MlirOp chunk_grad_b_rev =
        mlir::stablehlo::Reduce(builder, delta_pre_chunk_rev, zero_const,
                                sum_reduce_builder, {0, 1})[0];
    next_grad_b_rev = mlir::stablehlo::Add(body_grad_b_rev, chunk_grad_b_rev);
  }

  // Streaming accumulation: Projection weight gradients
  mlir::MlirOp next_grad_w_hr_fwd;
  mlir::MlirOp next_grad_w_hr_rev;
  if (w_hr_fwd.has_value()) {
    mlir::MlirOp delta_h_chunk_fwd =
        ConcatDim(builder, delta_h_chunk_steps_fwd, concat_dim);
    mlir::MlirOp delta_h_chunk_2d_fwd = mlir::stablehlo::Reshape(
        delta_h_chunk_fwd, {kLstmUnrollFactor * batch, out_h});
    mlir::MlirOp r_chunk_fwd =
        ConcatDim(builder, r_chunk_steps_fwd, concat_dim);
    mlir::MlirOp r_chunk_2d_fwd = mlir::stablehlo::Reshape(
        r_chunk_fwd, {kLstmUnrollFactor * batch, hidden});
    mlir::MlirOp chunk_grad_w_hr_fwd = MixedPrecisionDotGeneral(
        delta_h_chunk_2d_fwd, r_chunk_2d_fwd, grad_w_dot_dims, acc_dtype,
        out_dtype, precision);
    next_grad_w_hr_fwd =
        mlir::stablehlo::Add(body_grad_w_hr_fwd, chunk_grad_w_hr_fwd);

    mlir::MlirOp delta_h_chunk_rev =
        ConcatDim(builder, delta_h_chunk_steps_rev, concat_dim);
    mlir::MlirOp delta_h_chunk_2d_rev = mlir::stablehlo::Reshape(
        delta_h_chunk_rev, {kLstmUnrollFactor * batch, out_h});
    mlir::MlirOp r_chunk_rev =
        ConcatDim(builder, r_chunk_steps_rev, concat_dim);
    mlir::MlirOp r_chunk_2d_rev = mlir::stablehlo::Reshape(
        r_chunk_rev, {kLstmUnrollFactor * batch, hidden});
    mlir::MlirOp chunk_grad_w_hr_rev = MixedPrecisionDotGeneral(
        delta_h_chunk_2d_rev, r_chunk_2d_rev, grad_w_dot_dims, acc_dtype,
        out_dtype, precision);
    next_grad_w_hr_rev =
        mlir::stablehlo::Add(body_grad_w_hr_rev, chunk_grad_w_hr_rev);
  }

  // Layer input gradient chunks
  mlir::MlirOp chunk_grad_x_2d_fwd =
      MixedPrecisionDotGeneral(delta_pre_chunk_2d_fwd, w_ih_fwd_clumped,
                               bwd_weight_dot_dims, acc_dtype, precision);
  mlir::MlirOp chunk_grad_x_fwd =
      batch_first
          ? mlir::stablehlo::Reshape(chunk_grad_x_2d_fwd,
                                     {batch, kLstmUnrollFactor, in_dim})
          : mlir::stablehlo::Reshape(chunk_grad_x_2d_fwd,
                                     {kLstmUnrollFactor, batch, in_dim});
  mlir::MlirOp next_grad_x_fwd = mlir::stablehlo::DynamicUpdateSlice(
      body_grad_x_fwd, chunk_grad_x_fwd, chunk_start_indices);

  mlir::MlirOp chunk_grad_x_2d_rev =
      MixedPrecisionDotGeneral(delta_pre_chunk_2d_rev, w_ih_rev_clumped,
                               bwd_weight_dot_dims, acc_dtype, precision);
  mlir::MlirOp chunk_grad_x_rev =
      batch_first
          ? mlir::stablehlo::Reshape(chunk_grad_x_2d_rev,
                                     {batch, kLstmUnrollFactor, in_dim})
          : mlir::stablehlo::Reshape(chunk_grad_x_2d_rev,
                                     {kLstmUnrollFactor, batch, in_dim});
  mlir::MlirOp next_grad_x_rev = mlir::stablehlo::DynamicUpdateSlice(
      body_grad_x_rev, chunk_grad_x_rev, chunk_start_indices);

  mlir::MlirOp step_k = MakeScalarConstant(builder, kLstmUnrollFactor, i64);
  mlir::MlirOp next_step_idx = mlir::stablehlo::Add(body_step_idx, step_k);

  llvm::SmallVector<mlir::Value> next_loop_values = {
      next_step_idx.getValue(),      cur_delta_h_fwd.getValue(),
      cur_delta_c_fwd.getValue(),    cur_delta_h_rev.getValue(),
      cur_delta_c_rev.getValue(),    next_grad_x_fwd.getValue(),
      next_grad_x_rev.getValue(),    next_grad_w_ih_fwd.getValue(),
      next_grad_w_hh_fwd.getValue(), next_grad_w_ih_rev.getValue(),
      next_grad_w_hh_rev.getValue()};
  if (has_biases) {
    next_loop_values.push_back(next_grad_b_fwd.getValue());
    next_loop_values.push_back(next_grad_b_rev.getValue());
  }
  if (w_hr_fwd.has_value()) {
    next_loop_values.push_back(next_grad_w_hr_fwd.getValue());
    next_loop_values.push_back(next_grad_w_hr_rev.getValue());
  }
  mlir::stablehlo::ReturnOp::create(op_builder, loc, next_loop_values);

  op_builder.setInsertionPointAfter(while_op);

  mlir::MlirOp final_delta_h_rec_fwd(builder, while_op.getResult(1));
  mlir::MlirOp final_delta_c_next_fwd(builder, while_op.getResult(2));
  mlir::MlirOp final_delta_h_rec_rev(builder, while_op.getResult(3));
  mlir::MlirOp final_delta_c_next_rev(builder, while_op.getResult(4));
  mlir::MlirOp grad_x_fwd_final(builder, while_op.getResult(5));
  mlir::MlirOp grad_x_rev_final(builder, while_op.getResult(6));
  mlir::MlirOp final_grad_w_ih_fwd(builder, while_op.getResult(7));
  mlir::MlirOp final_grad_w_hh_fwd(builder, while_op.getResult(8));
  mlir::MlirOp final_grad_w_ih_rev(builder, while_op.getResult(9));
  mlir::MlirOp final_grad_w_hh_rev(builder, while_op.getResult(10));

  mlir::MlirOp grad_x_layer_fwd;
  mlir::MlirOp grad_x_layer_rev;
  if (rem_steps > 0) {
    grad_x_layer_fwd =
        ConcatDim(builder, {grad_x_fwd_final, rem_grad_x_fwd}, concat_dim);
    grad_x_layer_rev =
        ConcatDim(builder, {grad_x_rev_final, rem_grad_x_rev}, concat_dim);
  } else {
    grad_x_layer_fwd = grad_x_fwd_final;
    grad_x_layer_rev = grad_x_rev_final;
  }

  mlir::MlirOp grad_x_from_rev =
      mlir::stablehlo::Reverse(grad_x_layer_rev, {/*dimensions=*/concat_dim});
  mlir::MlirOp grad_x_layer =
      mlir::stablehlo::Add(grad_x_layer_fwd, grad_x_from_rev);

  mlir::MlirOp grad_h0_fwd = mlir::stablehlo::Reshape(
      to_out(final_delta_h_rec_fwd), {1, batch, out_h});
  mlir::MlirOp grad_c0_fwd = mlir::stablehlo::Reshape(
      to_out(final_delta_c_next_fwd), {1, batch, hidden});
  mlir::MlirOp grad_h0_rev = mlir::stablehlo::Reshape(
      to_out(final_delta_h_rec_rev), {1, batch, out_h});
  mlir::MlirOp grad_c0_rev = mlir::stablehlo::Reshape(
      to_out(final_delta_c_next_rev), {1, batch, hidden});

  mlir::MlirOp grad_w_ih_fwd =
      UnclumpWeight3H(builder, final_grad_w_ih_fwd, hidden, in_dim);
  mlir::MlirOp grad_w_hh_fwd =
      UnclumpWeight3H(builder, final_grad_w_hh_fwd, hidden, out_h);
  mlir::MlirOp grad_w_ih_rev =
      UnclumpWeight3H(builder, final_grad_w_ih_rev, hidden, in_dim);
  mlir::MlirOp grad_w_hh_rev =
      UnclumpWeight3H(builder, final_grad_w_hh_rev, hidden, out_h);

  int res_idx = 11;
  std::optional<mlir::MlirOp> grad_bias_fwd;
  std::optional<mlir::MlirOp> grad_bias_rev;
  if (has_biases) {
    mlir::MlirOp final_grad_b_fwd(builder, while_op.getResult(res_idx++));
    mlir::MlirOp final_grad_b_rev(builder, while_op.getResult(res_idx++));
    grad_bias_fwd = to_out(UnclumpBias3H(builder, final_grad_b_fwd, hidden));
    grad_bias_rev = to_out(UnclumpBias3H(builder, final_grad_b_rev, hidden));
  }

  std::optional<mlir::MlirOp> grad_w_hr_fwd;
  std::optional<mlir::MlirOp> grad_w_hr_rev;
  if (w_hr_fwd.has_value()) {
    mlir::MlirOp final_grad_w_hr_fwd(builder, while_op.getResult(res_idx++));
    mlir::MlirOp final_grad_w_hr_rev(builder, while_op.getResult(res_idx++));
    grad_w_hr_fwd = to_out(final_grad_w_hr_fwd);
    grad_w_hr_rev = to_out(final_grad_w_hr_rev);
  }

  return {grad_x_layer,          grad_h0_fwd,           grad_c0_fwd,
          grad_h0_rev,           grad_c0_rev,           to_out(grad_w_ih_fwd),
          to_out(grad_w_hh_fwd), to_out(grad_w_ih_rev), to_out(grad_w_hh_rev),
          grad_bias_fwd,         grad_bias_rev,         grad_w_hr_fwd,
          grad_w_hr_rev};
}

// Evaluates backward recurrence through time (BPTT) for a single bidirectional
// LSTM layer.
//
// Key Optimizations:
// 1. Weight 3H clumping: Clumps W_ih and W_hh into contiguous 3H layout (i, f,
// o) upfront,
//    ensuring all downstream backward matrix multiplications are packed and
//    efficient.
// 2. Dispatch:
//    - For short sequences (seq_len < kLstmUnrollFactor), calls
//    ComputeBidirLayerBackwardStatic
//      to eliminate while-loop overhead.
//    - For standard sequences (seq_len >= kLstmUnrollFactor), calls
//    ComputeBidirLayerBackwardChunked
//      for streaming memory reduction and concurrent execution.
template <typename ToOutFn, typename ReduceBuilderFn>
LstmBidirLayerBackwardOutputs ComputeBidirLayerBackward(
    mlir::MlirBuilder& builder, const LayerForwardState& state_fwd,
    const LayerForwardState& state_rev, mlir::MlirOp grad_y_fwd,
    mlir::MlirOp grad_y_rev_time_reversed, mlir::MlirOp grad_hy_fwd,
    mlir::MlirOp grad_cy_fwd, mlir::MlirOp grad_hy_rev,
    mlir::MlirOp grad_cy_rev, mlir::MlirOp w_ih_fwd, mlir::MlirOp w_hh_fwd,
    mlir::MlirOp w_ih_rev, mlir::MlirOp w_hh_rev, const bool has_biases,
    mlir::MlirOp zero_const, ReduceBuilderFn sum_reduce_builder,
    mlir::stablehlo::DotDimensionNumbersAttr bwd_weight_dot_dims,
    mlir::stablehlo::DotDimensionNumbersAttr grad_w_dot_dims,
    const int64_t seq_len, const int64_t batch, const int64_t hidden,
    const int64_t in_dim, const int64_t out_h, ToOutFn to_out,
    std::optional<mlir::MlirOp> w_hr_fwd, std::optional<mlir::MlirOp> w_hr_rev,
    std::optional<mlir::stablehlo::DotDimensionNumbersAttr> bwd_hr_dot_dims,
    const bool batch_first, mlir::stablehlo::Precision precision) {
  // Pre-clump the forward and reverse weight matrices into [4H, input_size] and
  // [4H, hidden_size] with contiguous 3H gate layout (i, f, o) followed by cell
  // gate (g).
  mlir::MlirOp w_ih_fwd_clumped =
      ClumpWeight3H(builder, w_ih_fwd, hidden, in_dim);
  mlir::MlirOp w_hh_fwd_clumped =
      ClumpWeight3H(builder, w_hh_fwd, hidden, out_h);
  mlir::MlirOp w_ih_rev_clumped =
      ClumpWeight3H(builder, w_ih_rev, hidden, in_dim);
  mlir::MlirOp w_hh_rev_clumped =
      ClumpWeight3H(builder, w_hh_rev, hidden, out_h);

  if (seq_len < kLstmUnrollFactor) {
    return ComputeBidirLayerBackwardStatic(
        builder, state_fwd, state_rev, grad_y_fwd, grad_y_rev_time_reversed,
        grad_hy_fwd, grad_cy_fwd, grad_hy_rev, grad_cy_rev, w_ih_fwd_clumped,
        w_hh_fwd_clumped, w_ih_rev_clumped, w_hh_rev_clumped, has_biases,
        zero_const, sum_reduce_builder, bwd_weight_dot_dims, grad_w_dot_dims,
        seq_len, batch, hidden, in_dim, out_h, to_out, w_hr_fwd, w_hr_rev,
        bwd_hr_dot_dims, batch_first, precision);
  }

  return ComputeBidirLayerBackwardChunked(
      builder, state_fwd, state_rev, grad_y_fwd, grad_y_rev_time_reversed,
      grad_hy_fwd, grad_cy_fwd, grad_hy_rev, grad_cy_rev, w_ih_fwd_clumped,
      w_hh_fwd_clumped, w_ih_rev_clumped, w_hh_rev_clumped, has_biases,
      zero_const, sum_reduce_builder, bwd_weight_dot_dims, grad_w_dot_dims,
      seq_len, batch, hidden, in_dim, out_h, to_out, w_hr_fwd, w_hr_rev,
      bwd_hr_dot_dims, batch_first, precision);
}

// =============================================================================
// StableHLO Kernel Implementation: Forward Pass (LstmInputImpl)
// =============================================================================
// Constructs the complete MLIR computation graph for multi-layer LSTM forward:
// 1. Gathers all input tensors: input, h_0, c_0, and layer weights/biases.
// 2. Maps input scalar types to ElementType and determines accumulation dtype
//    (ToAccumulateType, e.g. bfloat16 -> float32).
// 3. For each layer l in [0, num_layers):
//    - Executes batched input projection upfront: [T, B, in_dim] @ W_ih^T ->
//    [T, B, 4*H]
//      (or [B, T, in_dim] -> [B, T, 4*H] if batch_first=True).
//    - Pre-combines biases in 1D (b_ih + b_hh) and broadcasts across sequence.
//    - Unrolls recurrent loop across T steps with direct zero-copy slicing.
//    - Passes layer output sequence as input sequence to layer l+1.
// 4. Returns 3 outputs: {output, h_n, c_n}.
absl::StatusOr<DeviceBufferRefArray<3>> LstmInputImpl(
    const at::Tensor& input, const at::TensorList hx,
    const at::TensorList params, const bool has_biases,
    const int64_t num_layers, const double dropout, const bool train,
    const bool bidirectional, const bool batch_first,
    OpParamCacheKeys param_keys,
    std::optional<at::Tensor> rng_state = std::nullopt) {
  TT_RETURN_IF_ERROR(ValidateLstmInputs(input, hx, params, has_biases,
                                        num_layers, dropout, train,
                                        bidirectional, batch_first));

  const auto current_precision = GetAndAddPrecisionTo(param_keys);

  const bool has_dropout = (dropout > 0.0 && train && num_layers > 1);

  const int64_t batch = batch_first ? input.size(0) : input.size(1);
  const int64_t seq_len = batch_first ? input.size(1) : input.size(0);
  const at::Tensor& h_0 = hx[0];
  const at::Tensor& c_0 = hx[1];
  const int64_t out_h = h_0.size(2);
  const int64_t hidden = c_0.size(2);
  const bool is_projected = (out_h != hidden);
  const int64_t num_directions = bidirectional ? 2 : 1;
  const size_t params_per_direction =
      has_biases ? (is_projected ? 5 : 4) : (is_projected ? 3 : 2);
  const size_t params_per_layer = params_per_direction * num_directions;

  std::vector<at::Tensor> inputs;
  inputs.reserve(3 + params.size() + (has_dropout ? 1 : 0));
  inputs.push_back(input);
  inputs.push_back(h_0);
  inputs.push_back(c_0);
  for (const at::Tensor& p : params) {
    inputs.push_back(p);
  }
  if (has_dropout && rng_state.has_value()) {
    inputs.push_back(*rng_state);
  }

  // Determine output and accumulator dtypes (e.g. bfloat16 accumulates in
  // float32)
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));
  TT_ASSIGN_OR_RETURN(
      const auto acc_dtype,
      ConvertTo<mlir::ElementType>(ToAccumulateType(input.scalar_type())));

  const std::array<int64_t, 3> out_dims_arr =
      batch_first
          ? std::array<int64_t, 3>{batch, seq_len, num_directions * out_h}
          : std::array<int64_t, 3>{seq_len, batch, num_directions * out_h};
  const std::array<int64_t, 3> h_dims_arr = {num_layers * num_directions, batch,
                                             out_h};
  const std::array<int64_t, 3> c_dims_arr = {num_layers * num_directions, batch,
                                             hidden};
  const std::array<mlir::ElementType, 3> out_dtypes = {out_dtype, out_dtype,
                                                       out_dtype};
  const std::array<absl::Span<const int64_t>, 3> out_dims_list = {
      out_dims_arr, h_dims_arr, c_dims_arr};

  auto op_builder = [has_biases, num_layers, batch_first, batch, seq_len,
                     hidden, out_h, is_projected, out_dtype, acc_dtype,
                     params_per_layer, bidirectional, num_directions,
                     params_per_direction, has_dropout, dropout, train,
                     current_precision](absl::Span<mlir::MlirOp> builder_inputs,
                                        mlir::MlirBuilder& builder)
      -> absl::StatusOr<std::array<mlir::MlirOp, 3>> {
    // Short-circuit conversions when acc_dtype == out_dtype to avoid redundant
    // IR casts
    auto to_acc = [acc_dtype, out_dtype](mlir::MlirOp op) -> mlir::MlirOp {
      if (acc_dtype == out_dtype) {
        return op;
      }
      return mlir::stablehlo::ConvertElementType(op, acc_dtype);
    };
    auto to_out = [out_dtype, acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
      if (acc_dtype == out_dtype) {
        return op;
      }
      return mlir::stablehlo::ConvertElementType(op, out_dtype);
    };

    mlir::MlirOp current_input = builder_inputs[0];
    mlir::MLIRContext& ctx = builder.getContext();

    // DotDimensionNumbers:
    // ih_dot_dims contracts dimension 2 (in_dim) of input with dimension 1
    // (in_dim) of W_ih
    // hh_dot_dims contracts dimension 1 (out_h) of h_curr with dimension 1
    // (out_h) of W_hh
    const auto ih_dot_dims = MakeDotDims(&ctx, {2}, {1});
    const auto hh_dot_dims = MakeDotDims(&ctx, {1}, {1});
    const std::optional<mlir::stablehlo::DotDimensionNumbersAttr> hr_dot_dims =
        is_projected ? std::make_optional(MakeDotDims(&ctx, {1}, {1}))
                     : std::nullopt;

    mlir::MlirOp h_0_acc = to_acc(builder_inputs[1]);
    mlir::MlirOp c_0_acc = to_acc(builder_inputs[2]);

    // =========================================================================
    // OPTIMIZATION: Inter-Layer Pipelined Wavefront for Multi-Layer LSTMs
    // (L>=2)
    // =========================================================================
    // For unidirectional multi-layer LSTMs, stream chunk outputs
    // (kLstmUnrollFactor) between layer l and layer l+1 directly in TPU vector
    // memory without writing or reading intermediate sequence tensors [T, B,
    // out_h] to/from HBM.
    if (!bidirectional && num_layers >= 2 && seq_len >= kLstmUnrollFactor) {
      mlir::MlirOp w_ih_0 = builder_inputs[3 + 0 * params_per_layer + 0];
      const int64_t in_dim_0 = GetTensorTypeOrDie(w_ih_0).getDimSize(1);
      mlir::MlirOp w_ih_0_clumped =
          ClumpWeight3H(builder, w_ih_0, hidden, in_dim_0);
      mlir::MlirOp x_proj_0 =
          MixedPrecisionDotGeneral(current_input, w_ih_0_clumped, ih_dot_dims,
                                   acc_dtype, current_precision);
      if (has_biases) {
        mlir::MlirOp b_ih =
            to_acc(builder_inputs[3 + 0 * params_per_layer + 2]);
        mlir::MlirOp b_hh =
            to_acc(builder_inputs[3 + 0 * params_per_layer + 3]);
        mlir::MlirOp b_total = mlir::stablehlo::Add(b_ih, b_hh);
        mlir::MlirOp b_clumped = ClumpBias3H(builder, b_total, hidden);
        mlir::MlirOp b_bcast = mlir::stablehlo::BroadcastInDim(
            GetTensorTypeOrDie(x_proj_0), b_clumped,
            {/*broadcast_dimensions=*/2});
        x_proj_0 = mlir::stablehlo::Add(x_proj_0, b_bcast);
      }

      std::vector<WavefrontLayerWeight> layer_weights(num_layers);
      std::vector<mlir::MlirOp> h_inits(num_layers);
      std::vector<mlir::MlirOp> c_inits(num_layers);
      for (int64_t l = 0; l < num_layers; ++l) {
        h_inits[l] = ExtractLayer2D(h_0_acc, l, batch, out_h);
        c_inits[l] = ExtractLayer2D(c_0_acc, l, batch, hidden);

        mlir::MlirOp w_hh = builder_inputs[3 + l * params_per_layer + 1];
        layer_weights[l].w_hh = ClumpWeight3H(builder, w_hh, hidden, out_h);
        if (is_projected) {
          const size_t w_hr_idx =
              3 + l * params_per_layer + (has_biases ? 4 : 2);
          layer_weights[l].w_hr = builder_inputs[w_hr_idx];
        }
        if (l > 0) {
          mlir::MlirOp w_ih = builder_inputs[3 + l * params_per_layer + 0];
          layer_weights[l].w_ih = ClumpWeight3H(builder, w_ih, hidden, out_h);
          if (has_biases) {
            mlir::MlirOp b_ih =
                to_acc(builder_inputs[3 + l * params_per_layer + 2]);
            mlir::MlirOp b_hh =
                to_acc(builder_inputs[3 + l * params_per_layer + 3]);
            mlir::MlirOp b_tot = mlir::stablehlo::Add(b_ih, b_hh);
            layer_weights[l].b_total = ClumpBias3H(builder, b_tot, hidden);
          }
        }
      }

      std::optional<mlir::MlirOp> rand_op;
      if (has_dropout) {
        mlir::MlirOp rng_input_state = builder_inputs.back();
        const int64_t inter_layers = num_layers - 1;
        Dimensions rand_shape =
            batch_first ? Dimensions{inter_layers, batch, seq_len, out_h}
                        : Dimensions{inter_layers, seq_len, batch, out_h};
        TT_ASSIGN_OR_RETURN(rand_op, BuildUniformShlo(rng_input_state, 0.0, 1.0,
                                                      rand_shape, out_dtype));
      }

      LstmLayerOutputs wavefront_out = BuildLstmPipelinedWavefrontForward(
          builder, x_proj_0, h_inits, c_inits, layer_weights, hh_dot_dims,
          ih_dot_dims, seq_len, batch, hidden, out_h, num_layers, batch_first,
          to_out, to_acc, hr_dot_dims, dropout, train, rand_op,
          current_precision);
      return std::array<mlir::MlirOp, 3>{wavefront_out.layer_output_seq,
                                         wavefront_out.final_h,
                                         wavefront_out.final_c};
    }

    std::optional<mlir::MlirOp> rand_op;
    if (has_dropout) {
      mlir::MlirOp rng_input_state = builder_inputs.back();
      const int64_t inter_layers = num_layers - 1;
      const int64_t layer_out_dim = num_directions * out_h;
      Dimensions rand_shape =
          batch_first ? Dimensions{inter_layers, batch, seq_len, layer_out_dim}
                      : Dimensions{inter_layers, seq_len, batch, layer_out_dim};
      TT_ASSIGN_OR_RETURN(rand_op, BuildUniformShlo(rng_input_state, 0.0, 1.0,
                                                    rand_shape, out_dtype));
    }

    std::vector<mlir::MlirOp> all_final_h;
    std::vector<mlir::MlirOp> all_final_c;
    all_final_h.reserve(num_layers * num_directions);
    all_final_c.reserve(num_layers * num_directions);

    mlir::MlirOp current_layer_out;

    // Stacked layer recurrence
    for (int64_t l = 0; l < num_layers; ++l) {
      if (!bidirectional) {
        // --- UNIDIRECTIONAL LAYER RECURRENCE ---
        mlir::MlirOp w_ih = builder_inputs[3 + l * params_per_layer + 0];
        mlir::MlirOp w_hh = builder_inputs[3 + l * params_per_layer + 1];

        const int64_t in_dim = GetTensorTypeOrDie(w_ih).getDimSize(1);
        mlir::MlirOp w_ih_clumped =
            ClumpWeight3H(builder, w_ih, hidden, in_dim);
        mlir::MlirOp w_hh_clumped = ClumpWeight3H(builder, w_hh, hidden, out_h);

        std::optional<mlir::MlirOp> w_hr;
        if (is_projected) {
          const size_t w_hr_idx =
              3 + l * params_per_layer + (has_biases ? 4 : 2);
          w_hr = builder_inputs[w_hr_idx];
        }

        // 1. Batched input projection across the entire sequence upfront:
        //    [T, B, in_dim] or [B, T, in_dim] x [4*H, in_dim]^T -> [T, B, 4*H]
        //    or [B, T, 4*H]
        mlir::MlirOp x_proj =
            MixedPrecisionDotGeneral(current_input, w_ih_clumped, ih_dot_dims,
                                     acc_dtype, current_precision);

        if (has_biases) {
          mlir::MlirOp b_ih = builder_inputs[3 + l * params_per_layer + 2];
          mlir::MlirOp b_hh = builder_inputs[3 + l * params_per_layer + 3];

          // 2. Pre-combine 1D bias vectors [4*H] before broadcasting across 3D
          // sequence
          mlir::MlirOp b_ih_acc = to_acc(b_ih);
          mlir::MlirOp b_hh_acc = to_acc(b_hh);
          mlir::MlirOp b_total = mlir::stablehlo::Add(b_ih_acc, b_hh_acc);
          mlir::MlirOp b_clumped = ClumpBias3H(builder, b_total, hidden);
          mlir::MlirOp b_bcast = mlir::stablehlo::BroadcastInDim(
              GetTensorTypeOrDie(x_proj), b_clumped,
              {/*broadcast_dimensions=*/2});
          x_proj = mlir::stablehlo::Add(x_proj, b_bcast);
        }

        // 3. Extract layer initial states: [num_layers, B, out_h/H] -> [B,
        // out_h/H]
        mlir::MlirOp h_curr = ExtractLayer2D(h_0_acc, l, batch, out_h);
        mlir::MlirOp c_curr = ExtractLayer2D(c_0_acc, l, batch, hidden);

        // 4. Evaluate recurrence across sequence length T for this layer
        LstmLayerOutputs layer_out = BuildLstmLayerForward(
            builder, x_proj, h_curr, c_curr, w_hh_clumped, hh_dot_dims, seq_len,
            batch, hidden, out_h, batch_first, to_out, w_hr, hr_dot_dims,
            current_precision);
        current_layer_out = layer_out.layer_output_seq;

        if (has_dropout && l < num_layers - 1) {
          mlir::MlirOp rand_l =
              batch_first
                  ? mlir::stablehlo::Slice(*rand_op, {l, 0, 0, 0},
                                           {l + 1, batch, seq_len, out_h},
                                           {1, 1, 1, 1})
                  : mlir::stablehlo::Slice(*rand_op, {l, 0, 0, 0},
                                           {l + 1, seq_len, batch, out_h},
                                           {1, 1, 1, 1});
          rand_l = mlir::stablehlo::Reshape(
              rand_l, GetTensorTypeOrDie(current_layer_out).getShape());
          auto [dropped, mask] =
              ApplyInterLayerDropout(current_layer_out, rand_l, dropout);
          current_layer_out = dropped;
        }

        current_input = current_layer_out;

        all_final_h.push_back(layer_out.final_h);
        all_final_c.push_back(layer_out.final_c);
      } else {
        // --- BIDIRECTIONAL LAYER RECURRENCE ---
        const size_t p_fwd_base = (l * 2 + 0) * params_per_direction;
        const size_t p_rev_base = (l * 2 + 1) * params_per_direction;

        mlir::MlirOp w_ih_fwd = builder_inputs[3 + p_fwd_base + 0];
        mlir::MlirOp w_hh_fwd = builder_inputs[3 + p_fwd_base + 1];
        mlir::MlirOp w_ih_rev = builder_inputs[3 + p_rev_base + 0];
        mlir::MlirOp w_hh_rev = builder_inputs[3 + p_rev_base + 1];

        const int64_t in_dim = GetTensorTypeOrDie(w_ih_fwd).getDimSize(1);
        mlir::MlirOp w_ih_fwd_clumped =
            ClumpWeight3H(builder, w_ih_fwd, hidden, in_dim);
        mlir::MlirOp w_hh_fwd_clumped =
            ClumpWeight3H(builder, w_hh_fwd, hidden, out_h);
        mlir::MlirOp w_ih_rev_clumped =
            ClumpWeight3H(builder, w_ih_rev, hidden, in_dim);
        mlir::MlirOp w_hh_rev_clumped =
            ClumpWeight3H(builder, w_hh_rev, hidden, out_h);

        std::optional<mlir::MlirOp> w_hr_fwd;
        std::optional<mlir::MlirOp> w_hr_rev;
        if (is_projected) {
          const size_t w_hr_fwd_idx = 3 + p_fwd_base + (has_biases ? 4 : 2);
          const size_t w_hr_rev_idx = 3 + p_rev_base + (has_biases ? 4 : 2);
          w_hr_fwd = builder_inputs[w_hr_fwd_idx];
          w_hr_rev = builder_inputs[w_hr_rev_idx];
        }

        mlir::MlirOp h_curr_fwd = ExtractLayer2D(h_0_acc, 2 * l, batch, out_h);
        mlir::MlirOp c_curr_fwd = ExtractLayer2D(c_0_acc, 2 * l, batch, hidden);
        mlir::MlirOp h_curr_rev =
            ExtractLayer2D(h_0_acc, 2 * l + 1, batch, out_h);
        mlir::MlirOp c_curr_rev =
            ExtractLayer2D(c_0_acc, 2 * l + 1, batch, hidden);

        // 1. Joint Input Projection GEMM across whole sequence:
        // Combines W_ih_fwd and W_ih_rev along dim 0 -> [8*H, in_dim].
        // Executes a single large GEMM to double TPU MXU utilization.
        mlir::MlirOp w_ih_both =
            ConcatDim(builder, {w_ih_fwd_clumped, w_ih_rev_clumped}, /*dim=*/0);
        mlir::MlirOp x_proj_both =
            MixedPrecisionDotGeneral(current_input, w_ih_both, ih_dot_dims,
                                     acc_dtype, current_precision);

        if (has_biases) {
          mlir::MlirOp b_ih_fwd = to_acc(builder_inputs[3 + p_fwd_base + 2]);
          mlir::MlirOp b_hh_fwd = to_acc(builder_inputs[3 + p_fwd_base + 3]);
          mlir::MlirOp b_total_fwd = mlir::stablehlo::Add(b_ih_fwd, b_hh_fwd);

          mlir::MlirOp b_ih_rev = to_acc(builder_inputs[3 + p_rev_base + 2]);
          mlir::MlirOp b_hh_rev = to_acc(builder_inputs[3 + p_rev_base + 3]);
          mlir::MlirOp b_total_rev = mlir::stablehlo::Add(b_ih_rev, b_hh_rev);

          mlir::MlirOp b_clumped_fwd =
              ClumpBias3H(builder, b_total_fwd, hidden);
          mlir::MlirOp b_clumped_rev =
              ClumpBias3H(builder, b_total_rev, hidden);

          mlir::MlirOp b_total_both =
              ConcatDim(builder, {b_clumped_fwd, b_clumped_rev}, /*dim=*/0);
          mlir::MlirOp b_bcast = mlir::stablehlo::BroadcastInDim(
              GetTensorTypeOrDie(x_proj_both), b_total_both,
              {/*broadcast_dimensions=*/2});
          x_proj_both = mlir::stablehlo::Add(x_proj_both, b_bcast);
        }

        // Slice projections into forward [0..4*H] and reverse [4*H..8*H]
        mlir::MlirOp x_proj_fwd =
            batch_first ? mlir::stablehlo::Slice(x_proj_both, {0, 0, 0},
                                                 {batch, seq_len, 4 * hidden},
                                                 {1, 1, 1})
                        : mlir::stablehlo::Slice(x_proj_both, {0, 0, 0},
                                                 {seq_len, batch, 4 * hidden},
                                                 {1, 1, 1});
        mlir::MlirOp x_proj_rev =
            batch_first
                ? mlir::stablehlo::Slice(x_proj_both, {0, 0, 4 * hidden},
                                         {batch, seq_len, 8 * hidden},
                                         {1, 1, 1})
                : mlir::stablehlo::Slice(x_proj_both, {0, 0, 4 * hidden},
                                         {seq_len, batch, 8 * hidden},
                                         {1, 1, 1});

        // 2. Execute Concurrent Bidirectional Fusion in a single fused loop
        LstmBidirLayerOutputs layer_out = BuildLstmBidirLayerForward(
            builder, x_proj_fwd, x_proj_rev, h_curr_fwd, c_curr_fwd, h_curr_rev,
            c_curr_rev, w_hh_fwd_clumped, w_hh_rev_clumped, hh_dot_dims,
            seq_len, batch, hidden, out_h, batch_first, to_out, w_hr_fwd,
            w_hr_rev, hr_dot_dims, current_precision);

        current_layer_out = layer_out.layer_output_seq;
        if (has_dropout && l < num_layers - 1) {
          const int64_t bidir_out_dim = 2 * out_h;
          mlir::MlirOp rand_l =
              batch_first
                  ? mlir::stablehlo::Slice(
                        *rand_op, {l, 0, 0, 0},
                        {l + 1, batch, seq_len, bidir_out_dim}, {1, 1, 1, 1})
                  : mlir::stablehlo::Slice(
                        *rand_op, {l, 0, 0, 0},
                        {l + 1, seq_len, batch, bidir_out_dim}, {1, 1, 1, 1});
          rand_l = mlir::stablehlo::Reshape(
              rand_l, GetTensorTypeOrDie(current_layer_out).getShape());
          auto [dropped, mask] =
              ApplyInterLayerDropout(current_layer_out, rand_l, dropout);
          current_layer_out = dropped;
        }

        current_input =
            current_layer_out;  // next layer consumes [B, T, 2*out_h]

        all_final_h.push_back(layer_out.final_h_fwd);
        all_final_h.push_back(layer_out.final_h_rev);
        all_final_c.push_back(layer_out.final_c_fwd);
        all_final_c.push_back(layer_out.final_c_rev);
      }
    }

    mlir::MlirOp output = current_layer_out;
    // Stack final layer hidden and cell states into [num_layers *
    // num_directions, B, out_h/hidden]
    mlir::MlirOp h_n = ConcatDim(builder, all_final_h, /*dim=*/0);
    mlir::MlirOp c_n = ConcatDim(builder, all_final_c, /*dim=*/0);

    return std::array<mlir::MlirOp, 3>{output, h_n, c_n};
  };

  return DispatchOp<kDynamicSize, 3>(
      std::move(op_builder), inputs,
      {.out_dtypes = out_dtypes,
       .out_dims_list = out_dims_list,
       .op_param_cache_keys = std::move(param_keys)});
}

// =============================================================================
// StableHLO Kernel Implementation: Backward Pass (LstmInputBackwardImpl)
// =============================================================================
// Computes analytical gradients for all inputs and parameters:
//   Inputs:     grad_output [T, B, H] or [B, T, H], grad_hy [L, B, H], grad_cy
//   [L, B, H],
//               input sequence, h_0, c_0, and all layer weights & biases.
//   Outputs:    grad_input, grad_h0, grad_c0, and {grad_w_ih, grad_w_hh,
//   [grad_b_ih, grad_b_hh]} per layer.
//
// Execution Flow:
// 1. Step 1 (Forward Recomputation): Re-evaluates forward pass layer-by-layer
// to
//    recover intermediate gate and cell activations without keeping them in
//    memory during the forward pass.
// 2. Step 2 (Reverse BPTT): Loops backwards through layers (L-1 down to 0) and
//    timesteps (T-1 down to 0), accumulating adjoints and executing batched
//    GEMMs for weight gradients.
absl::StatusOr<std::vector<DeviceBufferRef>> LstmInputBackwardImpl(
    const at::Tensor& grad_output, const at::Tensor& grad_hy,
    const at::Tensor& grad_cy, const at::Tensor& input, const at::TensorList hx,
    const at::TensorList params, const at::TensorList cached_activations,
    const bool has_biases, const int64_t num_layers, const double dropout,
    const bool train, const bool bidirectional, const bool batch_first,
    OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateLstmInputs(input, hx, params, has_biases,
                                        num_layers, dropout, train,
                                        bidirectional, batch_first));

  const auto current_precision = GetAndAddPrecisionTo(param_keys);

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=enforced by PyTorch autograd engine
      grad_hy.sizes() == hx[0].sizes(), error::kInvalidArgument)
      << "grad_hy sizes (" << ToString(grad_hy.sizes())
      << ") must match h_0 sizes (" << ToString(hx[0].sizes()) << ")";
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=enforced by PyTorch autograd engine
      grad_cy.sizes() == hx[1].sizes(), error::kInvalidArgument)
      << "grad_cy sizes (" << ToString(grad_cy.sizes())
      << ") must match c_0 sizes (" << ToString(hx[1].sizes()) << ")";

  const bool has_dropout = (dropout > 0.0 && train && num_layers > 1);
  if (has_dropout) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=enforced by autograd forward state
        !cached_activations.empty(), error::kInvalidArgument)
        << "lstm backward with dropout requires "
           "cached forward activations";
  }

  const int64_t batch = batch_first ? input.size(0) : input.size(1);
  const int64_t seq_len = batch_first ? input.size(1) : input.size(0);
  const int64_t input_size = input.size(2);
  const at::Tensor& h_0 = hx[0];
  const at::Tensor& c_0 = hx[1];
  const int64_t out_h = h_0.size(2);
  const int64_t hidden = c_0.size(2);
  const bool is_projected = (out_h != hidden);
  const int64_t num_directions = bidirectional ? 2 : 1;
  const size_t params_per_direction =
      has_biases ? (is_projected ? 5 : 4) : (is_projected ? 3 : 2);

  // Collect input tensors: grad_output, grad_hy, grad_cy, input, hx[0], hx[1],
  // params..., cached_activations...
  std::vector<at::Tensor> input_tensors;
  input_tensors.reserve(6 + params.size() + cached_activations.size());
  input_tensors.push_back(grad_output);
  input_tensors.push_back(grad_hy);
  input_tensors.push_back(grad_cy);
  input_tensors.push_back(input);
  input_tensors.push_back(hx[0]);
  input_tensors.push_back(hx[1]);
  for (const at::Tensor& param : params) {
    input_tensors.push_back(param);
  }
  for (const at::Tensor& act : cached_activations) {
    input_tensors.push_back(act);
  }

  const size_t total_outputs = 3 + params.size();
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
  out_dtypes.push_back(in_dtype);
  out_dims_spans.push_back(input.sizes());

  TT_ASSIGN_OR_RETURN(auto h_dtype,
                      ConvertTo<mlir::ElementType>(h_0.scalar_type()));
  out_dtypes.push_back(h_dtype);
  out_dims_spans.push_back(h_0.sizes());

  TT_ASSIGN_OR_RETURN(auto c_dtype,
                      ConvertTo<mlir::ElementType>(c_0.scalar_type()));
  out_dtypes.push_back(c_dtype);
  out_dims_spans.push_back(c_0.sizes());

  for (const at::Tensor& param : params) {
    TT_ASSIGN_OR_RETURN(auto p_dtype,
                        ConvertTo<mlir::ElementType>(param.scalar_type()));
    out_dtypes.push_back(p_dtype);
    out_dims_spans.push_back(param.sizes());
  }

  auto op_builder =
      [=](absl::Span<mlir::MlirOp> mlir_inputs,  // DEFAULT_CAPTURE_OK
          mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    mlir::MLIRContext& ctx = builder.getContext();

    const auto bwd_weight_dot_dims = MakeDotDims(&ctx, {1}, {0});
    const auto grad_w_dot_dims = MakeDotDims(&ctx, {0}, {0});
    const std::optional<mlir::stablehlo::DotDimensionNumbersAttr>
        bwd_hr_dot_dims =
            is_projected ? std::make_optional(MakeDotDims(&ctx, {1}, {0}))
                         : std::nullopt;

    mlir::MlirOp grad_output_op = mlir_inputs[0];
    mlir::MlirOp grad_hy_op = mlir_inputs[1];
    mlir::MlirOp grad_cy_op = mlir_inputs[2];
    mlir::MlirOp input_op = mlir_inputs[3];
    const size_t num_params = params.size();
    std::vector<mlir::MlirOp> params_op(mlir_inputs.begin() + 6,
                                        mlir_inputs.begin() + 6 + num_params);
    absl::Span<mlir::MlirOp> cached_acts_op;
    if (mlir_inputs.size() > 6 + num_params) {
      cached_acts_op = mlir_inputs.subspan(6 + num_params);
    }

    auto to_acc = [acc_dtype, out_dtype](mlir::MlirOp op) -> mlir::MlirOp {
      if (acc_dtype == out_dtype) {
        return op;
      }
      return mlir::stablehlo::ConvertElementType(op, acc_dtype);
    };
    auto to_out = [out_dtype, acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
      if (acc_dtype == out_dtype) {
        return op;
      }
      return mlir::stablehlo::ConvertElementType(op, out_dtype);
    };

    mlir::MlirOp zero_const = MakeScalarConstant(builder, 0.0, acc_dtype);

    std::vector<LayerForwardState> layer_states_fwd(num_layers);
    std::vector<LayerForwardState> layer_states_rev(bidirectional ? num_layers
                                                                  : 0);
    std::vector<mlir::MlirOp> dropout_masks(num_layers > 1 ? num_layers - 1
                                                           : 0);

    mlir::MlirOp current_input = input_op;

    if (!cached_acts_op.empty()) {
      // -----------------------------------------------------------------------
      // Zero-Recomputation Activation Caching: Restore LayerForwardState
      // directly from cached forward activations, completely skipping forward
      // re-evaluation!
      // -----------------------------------------------------------------------
      size_t act_idx = 0;
      for (int64_t l = 0; l < num_layers; ++l) {
        mlir::MlirOp layer_in;
        if (l == 0) {
          layer_in = current_input;
        } else {
          layer_in = to_acc(cached_acts_op[act_idx++]);
        }
        mlir::MlirOp ifo_fwd = to_acc(cached_acts_op[act_idx++]);
        mlir::MlirOp pack_fwd = to_acc(cached_acts_op[act_idx++]);

        UnpackedActivations unpacked_fwd = UnpackCachedActivations(
            pack_fwd, seq_len, batch, hidden, out_h, batch_first);

        layer_states_fwd[l].layer_input = layer_in;
        layer_states_fwd[l].ifo_seq = ifo_fwd;
        layer_states_fwd[l].cellgate_seq = unpacked_fwd.cellgate_seq;
        layer_states_fwd[l].tanh_c_seq = unpacked_fwd.tanh_c_seq;
        layer_states_fwd[l].c_prev_seq = unpacked_fwd.c_prev_seq;
        layer_states_fwd[l].h_prev_seq = unpacked_fwd.h_prev_seq;

        if (bidirectional) {
          mlir::MlirOp ifo_rev = to_acc(cached_acts_op[act_idx++]);
          mlir::MlirOp pack_rev = to_acc(cached_acts_op[act_idx++]);

          UnpackedActivations unpacked_rev = UnpackCachedActivations(
              pack_rev, seq_len, batch, hidden, out_h, batch_first);

          mlir::MlirOp layer_in_rev = mlir::stablehlo::Reverse(
              layer_in, {/*dimensions=*/batch_first ? 1 : 0});

          layer_states_rev[l].layer_input = layer_in_rev;
          layer_states_rev[l].ifo_seq = ifo_rev;
          layer_states_rev[l].cellgate_seq = unpacked_rev.cellgate_seq;
          layer_states_rev[l].tanh_c_seq = unpacked_rev.tanh_c_seq;
          layer_states_rev[l].c_prev_seq = unpacked_rev.c_prev_seq;
          layer_states_rev[l].h_prev_seq = unpacked_rev.h_prev_seq;
        }
      }

      if (has_dropout) {
        mlir::MlirOp rng_input_state = cached_acts_op.back();
        const int64_t inter_layers = num_layers - 1;
        const int64_t layer_out_dim = num_directions * out_h;
        Dimensions rand_shape =
            batch_first
                ? Dimensions{inter_layers, batch, seq_len, layer_out_dim}
                : Dimensions{inter_layers, seq_len, batch, layer_out_dim};
        TT_ASSIGN_OR_RETURN(
            auto rand_op,
            BuildUniformShlo(rng_input_state, 0.0, 1.0, rand_shape, out_dtype));
        for (int64_t l = 0; l < num_layers - 1; ++l) {
          mlir::MlirOp rand_l =
              batch_first
                  ? mlir::stablehlo::Slice(
                        rand_op, {l, 0, 0, 0},
                        {l + 1, batch, seq_len, layer_out_dim}, {1, 1, 1, 1})
                  : mlir::stablehlo::Slice(
                        rand_op, {l, 0, 0, 0},
                        {l + 1, seq_len, batch, layer_out_dim}, {1, 1, 1, 1});
          rand_l = mlir::stablehlo::Reshape(
              rand_l, batch_first ? Dimensions{batch, seq_len, layer_out_dim}
                                  : Dimensions{seq_len, batch, layer_out_dim});
          mlir::MlirOp p_const = MakeConstantLike(rand_l, dropout);
          dropout_masks[l] = mlir::stablehlo::Compare(
              rand_l, p_const, mlir::stablehlo::ComparisonDirection::GE);
        }
      }
    } else {
      // -----------------------------------------------------------------------
      // Rematerialization Fallback: Recompute forward activations on-the-fly
      // when cached activations are omitted (e.g. direct C++ calls or memory
      // rematerialization).
      // -----------------------------------------------------------------------
      const auto ih_fwd_dot_dims = MakeDotDims(&ctx, {2}, {1});
      const auto hh_fwd_dot_dims = MakeDotDims(&ctx, {1}, {1});
      const std::optional<mlir::stablehlo::DotDimensionNumbersAttr>
          hr_dot_dims =
              is_projected ? std::make_optional(MakeDotDims(&ctx, {1}, {1}))
                           : std::nullopt;
      mlir::MlirOp h_0_op = mlir_inputs[4];
      mlir::MlirOp c_0_op = mlir_inputs[5];

      for (int64_t l = 0; l < num_layers; ++l) {
        const size_t p_fwd_base = (l * num_directions) * params_per_direction;
        mlir::MlirOp w_ih_fwd = params_op[p_fwd_base];
        mlir::MlirOp w_hh_fwd = params_op[p_fwd_base + 1];

        std::optional<mlir::MlirOp> w_hr_fwd;
        if (is_projected) {
          w_hr_fwd = params_op[p_fwd_base + (has_biases ? 4 : 2)];
        }

        mlir::MlirOp h_0_fwd =
            ExtractLayer2D(h_0_op, l * num_directions, batch, out_h);
        mlir::MlirOp c_0_fwd =
            ExtractLayer2D(c_0_op, l * num_directions, batch, hidden);

        std::optional<mlir::MlirOp> b_total_fwd;
        if (has_biases) {
          mlir::MlirOp b_ih = to_acc(params_op[p_fwd_base + 2]);
          mlir::MlirOp b_hh = to_acc(params_op[p_fwd_base + 3]);
          b_total_fwd = mlir::stablehlo::Add(b_ih, b_hh);
        }

        LayerRecomputeResult recompute_fwd = RecomputeLayerForward(
            builder, current_input, to_acc(h_0_fwd), to_acc(c_0_fwd), w_ih_fwd,
            w_hh_fwd, b_total_fwd, ih_fwd_dot_dims, hh_fwd_dot_dims, seq_len,
            batch, hidden, out_h, w_hr_fwd, hr_dot_dims, batch_first,
            current_precision);
        layer_states_fwd[l] = std::move(recompute_fwd.state);
        mlir::MlirOp layer_out_fwd = recompute_fwd.layer_output_seq;

        if (bidirectional) {
          const size_t p_rev_base =
              (l * num_directions + 1) * params_per_direction;
          mlir::MlirOp w_ih_rev = params_op[p_rev_base];
          mlir::MlirOp w_hh_rev = params_op[p_rev_base + 1];

          std::optional<mlir::MlirOp> w_hr_rev;
          if (is_projected) {
            w_hr_rev = params_op[p_rev_base + (has_biases ? 4 : 2)];
          }

          mlir::MlirOp h_0_rev =
              ExtractLayer2D(h_0_op, l * num_directions + 1, batch, out_h);
          mlir::MlirOp c_0_rev =
              ExtractLayer2D(c_0_op, l * num_directions + 1, batch, hidden);

          std::optional<mlir::MlirOp> b_total_rev;
          if (has_biases) {
            mlir::MlirOp b_ih = to_acc(params_op[p_rev_base + 2]);
            mlir::MlirOp b_hh = to_acc(params_op[p_rev_base + 3]);
            b_total_rev = mlir::stablehlo::Add(b_ih, b_hh);
          }

          mlir::MlirOp current_input_rev = mlir::stablehlo::Reverse(
              current_input, {/*dimensions=*/batch_first ? 1 : 0});

          LayerRecomputeResult recompute_rev = RecomputeLayerForward(
              builder, current_input_rev, to_acc(h_0_rev), to_acc(c_0_rev),
              w_ih_rev, w_hh_rev, b_total_rev, ih_fwd_dot_dims, hh_fwd_dot_dims,
              seq_len, batch, hidden, out_h, w_hr_rev, hr_dot_dims, batch_first,
              current_precision);
          layer_states_rev[l] = std::move(recompute_rev.state);

          mlir::MlirOp layer_out_rev =
              mlir::stablehlo::Reverse(recompute_rev.layer_output_seq,
                                       {/*dimensions=*/batch_first ? 1 : 0});

          current_input =
              ConcatDim(builder, {layer_out_fwd, layer_out_rev}, /*dim=*/2);
        } else {
          current_input = layer_out_fwd;
        }
      }
    }

    // Prepare incoming sequence output gradient
    mlir::MlirOp incoming_seq_grad = to_acc(grad_output_op);

    // Builder for sum reduction across time and batch dimensions for bias
    // gradients
    const mlir::Type acc_type = mlir::getElementType(ctx, acc_dtype);
    auto sum_reduce_builder = [acc_type](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
          acc_type, rb.getRegion(), rb.getOpBuilder());
    };

    std::vector<mlir::MlirOp> all_grad_h0(num_layers * num_directions);
    std::vector<mlir::MlirOp> all_grad_c0(num_layers * num_directions);
    std::vector<mlir::MlirOp> grad_params_out(params.size());

    mlir::MlirOp grad_x_curr = incoming_seq_grad;
    mlir::MlirOp grad_hy_comp = to_acc(grad_hy_op);
    mlir::MlirOp grad_cy_comp = to_acc(grad_cy_op);

    // -------------------------------------------------------------------------
    // Step 2: Backpropagation Through Time (BPTT) in Reverse Layer Order
    // -------------------------------------------------------------------------
    // Iterates from top layer (num_layers - 1) down to bottom layer 0.
    // At each layer:
    // - Computes layer input gradient grad_x_layer, which becomes incoming
    // grad_x_curr for the layer below (l - 1). For bidirectional layers,
    // input gradients from forward and reverse branches are summed.
    // - Computes parameter gradients (grad_w_ih, grad_w_hh, grad_bias,
    // grad_w_hr).
    // - Computes initial state gradients (grad_h0_layer, grad_c0_layer).
    if (num_layers > 1 && !bidirectional && seq_len >= kLstmUnrollFactor) {
      std::vector<mlir::MlirOp> w_ih_clumped_list(num_layers);
      std::vector<mlir::MlirOp> w_hh_clumped_list(num_layers);
      std::vector<std::optional<mlir::MlirOp>> w_hr_list(num_layers);
      std::vector<mlir::MlirOp> grad_hy_list(num_layers);
      std::vector<mlir::MlirOp> grad_cy_list(num_layers);

      for (int64_t l = 0; l < num_layers; ++l) {
        const int64_t in_dim_l = (l == 0) ? input_size : out_h;
        const size_t p_fwd_base = l * params_per_direction;
        mlir::MlirOp w_ih = params_op[p_fwd_base];
        mlir::MlirOp w_hh = params_op[p_fwd_base + 1];
        w_ih_clumped_list[l] = ClumpWeight3H(builder, w_ih, hidden, in_dim_l);
        w_hh_clumped_list[l] = ClumpWeight3H(builder, w_hh, hidden, out_h);
        if (is_projected) {
          w_hr_list[l] = params_op[p_fwd_base + (has_biases ? 4 : 2)];
        }
        grad_hy_list[l] = ExtractLayer2D(grad_hy_comp, l, batch, out_h);
        grad_cy_list[l] = ExtractLayer2D(grad_cy_comp, l, batch, hidden);
      }

      MultiLayerBackwardOutputs wavefront_bwd =
          BuildLstmPipelinedWavefrontBackward(
              builder, layer_states_fwd, incoming_seq_grad, grad_hy_list,
              grad_cy_list, w_ih_clumped_list, w_hh_clumped_list, has_biases,
              zero_const, sum_reduce_builder, bwd_weight_dot_dims,
              grad_w_dot_dims, seq_len, batch, hidden, input_size, out_h,
              num_layers, to_out, w_hr_list, bwd_hr_dot_dims, has_dropout,
              dropout, dropout_masks, batch_first, current_precision);

      for (int64_t l = 0; l < num_layers; ++l) {
        const size_t p_fwd_base = l * params_per_direction;
        all_grad_h0[l] = wavefront_bwd.grad_h0[l];
        all_grad_c0[l] = wavefront_bwd.grad_c0[l];
        grad_params_out[p_fwd_base] = wavefront_bwd.grad_w_ih[l];
        grad_params_out[p_fwd_base + 1] = wavefront_bwd.grad_w_hh[l];
        if (has_biases) {
          grad_params_out[p_fwd_base + 2] = *wavefront_bwd.grad_bias[l];
          grad_params_out[p_fwd_base + 3] = *wavefront_bwd.grad_bias[l];
          if (is_projected) {
            grad_params_out[p_fwd_base + 4] = *wavefront_bwd.grad_w_hr[l];
          }
        } else {
          if (is_projected) {
            grad_params_out[p_fwd_base + 2] = *wavefront_bwd.grad_w_hr[l];
          }
        }
      }
      grad_x_curr = wavefront_bwd.grad_x;
    } else {
      for (int64_t l = num_layers - 1; l >= 0; --l) {
        const int64_t in_dim = (l == 0) ? input_size : (out_h * num_directions);
        const size_t p_fwd_base = (l * num_directions) * params_per_direction;
        mlir::MlirOp w_ih_fwd = params_op[p_fwd_base];
        mlir::MlirOp w_hh_fwd = params_op[p_fwd_base + 1];

        std::optional<mlir::MlirOp> w_hr_fwd;
        if (is_projected) {
          w_hr_fwd = params_op[p_fwd_base + (has_biases ? 4 : 2)];
        }

        mlir::MlirOp grad_hy_fwd =
            ExtractLayer2D(grad_hy_comp, l * num_directions, batch, out_h);
        mlir::MlirOp grad_cy_fwd =
            ExtractLayer2D(grad_cy_comp, l * num_directions, batch, hidden);

        if (!bidirectional) {
          LayerBackwardOutputs bwd_fwd = ComputeLayerBackward(
              builder, layer_states_fwd[l], grad_x_curr, grad_hy_fwd,
              grad_cy_fwd, w_ih_fwd, w_hh_fwd, has_biases, zero_const,
              sum_reduce_builder, bwd_weight_dot_dims, grad_w_dot_dims, seq_len,
              batch, hidden, in_dim, out_h, to_out, w_hr_fwd, bwd_hr_dot_dims,
              batch_first, current_precision);

          all_grad_h0[l * num_directions] = bwd_fwd.grad_h0_layer;
          all_grad_c0[l * num_directions] = bwd_fwd.grad_c0_layer;

          grad_params_out[p_fwd_base] = bwd_fwd.grad_w_ih;
          grad_params_out[p_fwd_base + 1] = bwd_fwd.grad_w_hh;
          if (has_biases) {
            grad_params_out[p_fwd_base + 2] = *bwd_fwd.grad_bias;
            grad_params_out[p_fwd_base + 3] = *bwd_fwd.grad_bias;
            if (is_projected) {
              grad_params_out[p_fwd_base + 4] = *bwd_fwd.grad_w_hr;
            }
          } else {
            if (is_projected) {
              grad_params_out[p_fwd_base + 2] = *bwd_fwd.grad_w_hr;
            }
          }

          grad_x_curr = bwd_fwd.grad_x_layer;
        } else {
          const size_t p_rev_base =
              (l * num_directions + 1) * params_per_direction;
          mlir::MlirOp w_ih_rev = params_op[p_rev_base];
          mlir::MlirOp w_hh_rev = params_op[p_rev_base + 1];

          std::optional<mlir::MlirOp> w_hr_rev;
          if (is_projected) {
            w_hr_rev = params_op[p_rev_base + (has_biases ? 4 : 2)];
          }

          mlir::MlirOp grad_hy_rev = ExtractLayer2D(
              grad_hy_comp, l * num_directions + 1, batch, out_h);
          mlir::MlirOp grad_cy_rev = ExtractLayer2D(
              grad_cy_comp, l * num_directions + 1, batch, hidden);

          mlir::MlirOp grad_y_fwd =
              batch_first
                  ? mlir::stablehlo::Slice(grad_x_curr, {0, 0, 0},
                                           {batch, seq_len, out_h}, {1, 1, 1})
                  : mlir::stablehlo::Slice(grad_x_curr, {0, 0, 0},
                                           {seq_len, batch, out_h}, {1, 1, 1});
          mlir::MlirOp grad_y_rev =
              batch_first ? mlir::stablehlo::Slice(grad_x_curr, {0, 0, out_h},
                                                   {batch, seq_len, 2 * out_h},
                                                   {1, 1, 1})
                          : mlir::stablehlo::Slice(grad_x_curr, {0, 0, out_h},
                                                   {seq_len, batch, 2 * out_h},
                                                   {1, 1, 1});
          // Reverse along time dimension to align with reverse forward state
          mlir::MlirOp grad_y_rev_time_reversed = mlir::stablehlo::Reverse(
              grad_y_rev, {/*dimensions=*/batch_first ? 1 : 0});

          // Concurrent Bidirectional Backward
          LstmBidirLayerBackwardOutputs bwd_bidir = ComputeBidirLayerBackward(
              builder, layer_states_fwd[l], layer_states_rev[l], grad_y_fwd,
              grad_y_rev_time_reversed, grad_hy_fwd, grad_cy_fwd, grad_hy_rev,
              grad_cy_rev, w_ih_fwd, w_hh_fwd, w_ih_rev, w_hh_rev, has_biases,
              zero_const, sum_reduce_builder, bwd_weight_dot_dims,
              grad_w_dot_dims, seq_len, batch, hidden, in_dim, out_h, to_out,
              w_hr_fwd, w_hr_rev, bwd_hr_dot_dims, batch_first,
              current_precision);

          all_grad_h0[l * num_directions] = bwd_bidir.grad_h0_fwd;
          all_grad_c0[l * num_directions] = bwd_bidir.grad_c0_fwd;
          all_grad_h0[l * num_directions + 1] = bwd_bidir.grad_h0_rev;
          all_grad_c0[l * num_directions + 1] = bwd_bidir.grad_c0_rev;

          grad_params_out[p_fwd_base] = bwd_bidir.grad_w_ih_fwd;
          grad_params_out[p_fwd_base + 1] = bwd_bidir.grad_w_hh_fwd;
          grad_params_out[p_rev_base] = bwd_bidir.grad_w_ih_rev;
          grad_params_out[p_rev_base + 1] = bwd_bidir.grad_w_hh_rev;

          if (has_biases) {
            grad_params_out[p_fwd_base + 2] = *bwd_bidir.grad_bias_fwd;
            grad_params_out[p_fwd_base + 3] = *bwd_bidir.grad_bias_fwd;
            grad_params_out[p_rev_base + 2] = *bwd_bidir.grad_bias_rev;
            grad_params_out[p_rev_base + 3] = *bwd_bidir.grad_bias_rev;
            if (is_projected) {
              grad_params_out[p_fwd_base + 4] = *bwd_bidir.grad_w_hr_fwd;
              grad_params_out[p_rev_base + 4] = *bwd_bidir.grad_w_hr_rev;
            }
          } else {
            if (is_projected) {
              grad_params_out[p_fwd_base + 2] = *bwd_bidir.grad_w_hr_fwd;
              grad_params_out[p_rev_base + 2] = *bwd_bidir.grad_w_hr_rev;
            }
          }

          grad_x_curr = bwd_bidir.grad_x_layer;
        }

        if (has_dropout && l > 0) {
          const double scale = 1.0 / (1.0 - dropout);
          TT_ASSIGN_OR_RETURN(grad_x_curr,
                              BuildDropoutBackwardShlo(
                                  grad_x_curr, dropout_masks[l - 1], scale));
        }
      }
    }

    mlir::MlirOp grad_input = to_out(grad_x_curr);

    mlir::MlirOp grad_h0 = ConcatDim(builder, all_grad_h0, /*dim=*/0);
    mlir::MlirOp grad_c0 = ConcatDim(builder, all_grad_c0, /*dim=*/0);

    mlir::SmallVector<mlir::MlirOp> results;
    results.reserve(3 + grad_params_out.size());
    results.push_back(grad_input);
    results.push_back(grad_h0);
    results.push_back(grad_c0);
    for (mlir::MlirOp p_grad : grad_params_out) {
      results.push_back(p_grad);
    }
    return results;
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

// Evaluates forward multi-layer recurrence while collecting and packing
// intermediate activations for zero-recomputation backward passes.
//
// Memory & Caching Architecture:
// In training mode (requires_grad=True), standard autograd either caches all
// fine-grained elemental activations (causing severe HBM bloat) or recomputes
// the forward pass entirely during BPTT (wasting compute). This kernel adopts
// a zero-recomputation strategy by caching only the minimal essential sequence
// tensors required for analytical BPTT:
// - Intermediate layer inputs: [T, B, prev_out_dim] for l > 0.
// - Clumped gate pre-activations: ifo_seq [T, B, 3*H] per layer/direction.
// - Packed recurrence state: [T, B, 3*H + out_h] per layer/direction, packing
//   [cellgate, tanh(c), c_prev, h_prev] along dimension 2.
//
// Returning these packed activations enables AtenLstmInputBackward to bypass
// forward recomputation entirely and directly execute analytical BPTT.
absl::StatusOr<std::vector<DeviceBufferRef>> LstmInputForwardCachedImpl(
    const at::Tensor& input, const at::TensorList hx,
    const at::TensorList params, const bool has_biases,
    const int64_t num_layers, const double dropout, const bool train,
    const bool bidirectional, const bool batch_first,
    OpParamCacheKeys param_keys,
    std::optional<at::Tensor> rng_state = std::nullopt) {
  // --- Step 1: Input Validation & Dimension Extraction ---
  TT_RETURN_IF_ERROR(ValidateLstmInputs(input, hx, params, has_biases,
                                        num_layers, dropout, train,
                                        bidirectional, batch_first));

  const auto current_precision = GetAndAddPrecisionTo(param_keys);

  const int64_t batch = batch_first ? input.size(0) : input.size(1);
  const int64_t seq_len = batch_first ? input.size(1) : input.size(0);
  const at::Tensor& h_0 = hx[0];
  const at::Tensor& c_0 = hx[1];
  const int64_t out_h = h_0.size(2);
  const int64_t hidden = c_0.size(2);
  const bool is_projected = (out_h != hidden);
  const int64_t num_directions = bidirectional ? 2 : 1;
  const size_t params_per_direction =
      has_biases ? (is_projected ? 5 : 4) : (is_projected ? 3 : 2);
  const bool has_dropout = (dropout > 0.0 && train && num_layers > 1);

  // --- Step 2: Flatten Input Tensors for DispatchOp ---
  // Pack PyTorch tensors into a contiguous vector:
  // [0]: input, [1]: h_0, [2]: c_0, [3..]: params, [optional last]: rng_state
  std::vector<at::Tensor> inputs;
  inputs.reserve(3 + params.size() + (has_dropout ? 1 : 0));
  inputs.push_back(input);
  inputs.push_back(h_0);
  inputs.push_back(c_0);
  for (const at::Tensor& param : params) {
    inputs.push_back(param);
  }
  if (has_dropout && rng_state.has_value()) {
    inputs.push_back(*rng_state);
  }

  // --- Step 3: Type Resolution & Accumulator Precision ---
  TT_ASSIGN_OR_RETURN(const mlir::ElementType in_dtype,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));
  const mlir::ElementType out_dtype = in_dtype;
  TT_ASSIGN_OR_RETURN(
      const mlir::ElementType acc_dtype,
      ConvertTo<mlir::ElementType>(ToAccumulateType(input.scalar_type())));

  // --- Step 4: Output Shapes Planning ---
  // Primary PyTorch outputs:
  // 0: sequence output [B, T, D*out_h] (or [T, B, D*out_h])
  // 1: final hidden states hy [num_layers * num_directions, B, out_h]
  // 2: final cell states cy [num_layers * num_directions, B, hidden]
  const Dimensions output_shape =
      batch_first ? Dimensions{batch, seq_len, out_h * num_directions}
                  : Dimensions{seq_len, batch, out_h * num_directions};
  const Dimensions hy_shape = {num_layers * num_directions, batch, out_h};
  const Dimensions cy_shape = {num_layers * num_directions, batch, hidden};

  std::vector<mlir::ElementType> out_dtypes;
  std::vector<Dimensions> out_dims_list;

  out_dtypes.push_back(out_dtype);
  out_dims_list.push_back(output_shape);

  out_dtypes.push_back(out_dtype);
  out_dims_list.push_back(hy_shape);

  out_dtypes.push_back(out_dtype);
  out_dims_list.push_back(cy_shape);

  // Cached activation shapes:
  // - ifo_shape: [T, B, 3*H] containing clumped [i, f, o] gate pre-activations
  // - pack_shape: [T, B, 3*H + out_h] packing [cellgate (H), tanh_c (H), c_prev
  // (H), h_prev (out_h)]
  const Dimensions ifo_shape = batch_first
                                   ? Dimensions{batch, seq_len, 3 * hidden}
                                   : Dimensions{seq_len, batch, 3 * hidden};
  const Dimensions pack_shape =
      batch_first ? Dimensions{batch, seq_len, 3 * hidden + out_h}
                  : Dimensions{seq_len, batch, 3 * hidden + out_h};

  // Plan layouts for all cached activations across layers and directions
  for (int64_t l = 0; l < num_layers; ++l) {
    if (l > 0) {
      // Intermediate sequence inputs between layers (dim = out_h *
      // num_directions)
      const int64_t prev_out_dim = out_h * num_directions;
      out_dtypes.push_back(out_dtype);
      out_dims_list.push_back(batch_first
                                  ? Dimensions{batch, seq_len, prev_out_dim}
                                  : Dimensions{seq_len, batch, prev_out_dim});
    }
    // Forward direction cached activations: [ifo_seq, pack_seq]
    out_dtypes.push_back(out_dtype);
    out_dims_list.push_back(ifo_shape);
    out_dtypes.push_back(out_dtype);
    out_dims_list.push_back(pack_shape);

    if (bidirectional) {
      // Reverse direction cached activations: [ifo_seq_rev, pack_seq_rev]
      out_dtypes.push_back(out_dtype);
      out_dims_list.push_back(ifo_shape);
      out_dtypes.push_back(out_dtype);
      out_dims_list.push_back(pack_shape);
    }
  }

  // --- Step 5: Construct StableHLO IR Builder Lambda ---
  auto op_builder =
      [=](absl::Span<mlir::MlirOp> mlir_inputs,  // DEFAULT_CAPTURE_OK
          mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    mlir::MLIRContext& ctx = builder.getContext();

    // GEMM contraction dimensions:
    // ih_fwd_dot_dims contracts dim 2 (in_dim) with dim 1 (in_dim) of W_ih
    // hh_fwd_dot_dims contracts dim 1 (out_h) with dim 1 (out_h) of W_hh
    const auto ih_fwd_dot_dims = MakeDotDims(&ctx, {2}, {1});
    const auto hh_fwd_dot_dims = MakeDotDims(&ctx, {1}, {1});
    const std::optional<mlir::stablehlo::DotDimensionNumbersAttr> hr_dot_dims =
        is_projected ? std::make_optional(MakeDotDims(&ctx, {1}, {1}))
                     : std::nullopt;

    mlir::MlirOp input_op = mlir_inputs[0];
    mlir::MlirOp h_0_op = mlir_inputs[1];
    mlir::MlirOp c_0_op = mlir_inputs[2];
    std::vector<mlir::MlirOp> params_op(
        mlir_inputs.begin() + 3, mlir_inputs.begin() + 3 + params.size());

    // Type conversion helpers between storage dtype and accumulator dtype
    auto to_acc = [acc_dtype, out_dtype](mlir::MlirOp op) -> mlir::MlirOp {
      if (acc_dtype == out_dtype) {
        return op;
      }
      return mlir::stablehlo::ConvertElementType(op, acc_dtype);
    };

    auto to_out = [acc_dtype, out_dtype](mlir::MlirOp op) -> mlir::MlirOp {
      if (acc_dtype == out_dtype) {
        return op;
      }
      return mlir::stablehlo::ConvertElementType(op, out_dtype);
    };

    mlir::MlirOp current_input = input_op;

    std::vector<mlir::MlirOp> all_final_h;
    std::vector<mlir::MlirOp> all_final_c;
    all_final_h.reserve(num_layers * num_directions);
    all_final_c.reserve(num_layers * num_directions);

    mlir::SmallVector<mlir::MlirOp> cached_acts;

    // --- Phase 5.1: Philox PRNG Evaluation for Inter-Layer Dropout ---
    // Generates a single 4D uniform random tensor [num_layers - 1, ...] for all
    // intermediate layers from the 16-byte Philox seed state.
    std::optional<mlir::MlirOp> rand_op;
    if (has_dropout) {
      mlir::MlirOp rng_input_state = mlir_inputs.back();
      const int64_t inter_layers = num_layers - 1;
      const int64_t layer_out_dim = num_directions * out_h;
      Dimensions rand_shape =
          batch_first ? Dimensions{inter_layers, batch, seq_len, layer_out_dim}
                      : Dimensions{inter_layers, seq_len, batch, layer_out_dim};
      TT_ASSIGN_OR_RETURN(rand_op, BuildUniformShlo(rng_input_state, 0.0, 1.0,
                                                    rand_shape, out_dtype));
    }

    // --- Phase 5.2: Layer-by-Layer Forward Recurrence & Activation Caching ---
    for (int64_t l = 0; l < num_layers; ++l) {
      // For stacked layers (l > 0), save the layer input sequence tensor
      // to avoid re-evaluating dropout or activation recomputation in backward.
      if (l > 0) {
        cached_acts.push_back(to_out(current_input));
      }

      // Extract forward weights and optional projection parameter
      const size_t p_fwd_base = (l * num_directions) * params_per_direction;
      mlir::MlirOp w_ih_fwd = params_op[p_fwd_base];
      mlir::MlirOp w_hh_fwd = params_op[p_fwd_base + 1];

      std::optional<mlir::MlirOp> w_hr_fwd;
      if (is_projected) {
        w_hr_fwd = params_op[p_fwd_base + (has_biases ? 4 : 2)];
      }

      // Slice 2D initial states for layer l
      mlir::MlirOp h_0_fwd =
          ExtractLayer2D(to_acc(h_0_op), l * num_directions, batch, out_h);
      mlir::MlirOp c_0_fwd =
          ExtractLayer2D(to_acc(c_0_op), l * num_directions, batch, hidden);

      // Pre-combine 1D biases: b_total = b_ih + b_hh
      std::optional<mlir::MlirOp> b_total_fwd;
      if (has_biases) {
        mlir::MlirOp b_ih = to_acc(params_op[p_fwd_base + 2]);
        mlir::MlirOp b_hh = to_acc(params_op[p_fwd_base + 3]);
        b_total_fwd = mlir::stablehlo::Add(b_ih, b_hh);
      }

      // Execute forward recurrence for forward direction
      LayerRecomputeResult recompute_fwd = RecomputeLayerForward(
          builder, current_input, h_0_fwd, c_0_fwd, w_ih_fwd, w_hh_fwd,
          b_total_fwd, ih_fwd_dot_dims, hh_fwd_dot_dims, seq_len, batch, hidden,
          out_h, w_hr_fwd, hr_dot_dims, batch_first, current_precision);

      all_final_h.push_back(to_out(recompute_fwd.final_h));
      all_final_c.push_back(to_out(recompute_fwd.final_c));

      // Pack forward cached activations:
      // 1. ifo_seq: [T, B, 3*H]
      // 2. pack_fwd: [T, B, 3*H + out_h] = [cellgate, tanh_c, c_prev, h_prev]
      // along dim 2
      cached_acts.push_back(to_out(recompute_fwd.state.ifo_seq));
      mlir::MlirOp pack_fwd = ConcatDim(
          builder,
          {recompute_fwd.state.cellgate_seq, recompute_fwd.state.tanh_c_seq,
           recompute_fwd.state.c_prev_seq, recompute_fwd.state.h_prev_seq},
          /*dim=*/2);
      cached_acts.push_back(to_out(pack_fwd));

      mlir::MlirOp layer_out_fwd = recompute_fwd.layer_output_seq;

      // --- Phase 5.3: Reverse Direction Forward Recurrence (Bidirectional
      // Mode) ---
      if (bidirectional) {
        const size_t p_rev_base =
            (l * num_directions + 1) * params_per_direction;
        mlir::MlirOp w_ih_rev = params_op[p_rev_base];
        mlir::MlirOp w_hh_rev = params_op[p_rev_base + 1];

        std::optional<mlir::MlirOp> w_hr_rev;
        if (is_projected) {
          w_hr_rev = params_op[p_rev_base + (has_biases ? 4 : 2)];
        }

        mlir::MlirOp h_0_rev = ExtractLayer2D(
            to_acc(h_0_op), l * num_directions + 1, batch, out_h);
        mlir::MlirOp c_0_rev = ExtractLayer2D(
            to_acc(c_0_op), l * num_directions + 1, batch, hidden);

        std::optional<mlir::MlirOp> b_total_rev;
        if (has_biases) {
          mlir::MlirOp b_ih = to_acc(params_op[p_rev_base + 2]);
          mlir::MlirOp b_hh = to_acc(params_op[p_rev_base + 3]);
          b_total_rev = mlir::stablehlo::Add(b_ih, b_hh);
        }

        // Time-reverse input sequence for the reverse recurrence sweep
        mlir::MlirOp current_input_rev = mlir::stablehlo::Reverse(
            current_input, {/*dimensions=*/batch_first ? 1 : 0});

        LayerRecomputeResult recompute_rev = RecomputeLayerForward(
            builder, current_input_rev, h_0_rev, c_0_rev, w_ih_rev, w_hh_rev,
            b_total_rev, ih_fwd_dot_dims, hh_fwd_dot_dims, seq_len, batch,
            hidden, out_h, w_hr_rev, hr_dot_dims, batch_first,
            current_precision);

        all_final_h.push_back(to_out(recompute_rev.final_h));
        all_final_c.push_back(to_out(recompute_rev.final_c));

        // Pack reverse cached activations
        cached_acts.push_back(to_out(recompute_rev.state.ifo_seq));
        mlir::MlirOp pack_rev = ConcatDim(
            builder,
            {recompute_rev.state.cellgate_seq, recompute_rev.state.tanh_c_seq,
             recompute_rev.state.c_prev_seq, recompute_rev.state.h_prev_seq},
            /*dim=*/2);
        cached_acts.push_back(to_out(pack_rev));

        // Time-reverse reverse layer output back to chronological order
        mlir::MlirOp layer_out_rev =
            mlir::stablehlo::Reverse(recompute_rev.layer_output_seq,
                                     {/*dimensions=*/batch_first ? 1 : 0});

        // Concatenate forward and reverse outputs along feature dimension (dim
        // 2)
        current_input =
            ConcatDim(builder, {layer_out_fwd, layer_out_rev}, /*dim=*/2);
      } else {
        current_input = layer_out_fwd;
      }

      // --- Phase 5.4: Inter-Layer Dropout Application ---
      // Apply dropout on sequence tensor between stacked layers (l < num_layers
      // - 1)
      if (has_dropout && l < num_layers - 1) {
        const int64_t layer_out_dim = num_directions * out_h;
        mlir::MlirOp rand_l =
            batch_first
                ? mlir::stablehlo::Slice(*rand_op, {l, 0, 0, 0},
                                         {l + 1, batch, seq_len, layer_out_dim},
                                         {1, 1, 1, 1})
                : mlir::stablehlo::Slice(*rand_op, {l, 0, 0, 0},
                                         {l + 1, seq_len, batch, layer_out_dim},
                                         {1, 1, 1, 1});
        rand_l = mlir::stablehlo::Reshape(
            rand_l, GetTensorTypeOrDie(current_input).getShape());
        auto [dropped, mask] =
            ApplyInterLayerDropout(current_input, rand_l, dropout);
        current_input = dropped;
      }
    }

    mlir::MlirOp final_output = current_input;

    // --- Phase 5.5: Format Final Hidden and Cell State Tensors ---
    // Reshape all [batch, out_h] states to [1, batch, out_h] and concatenate
    // along dim 0
    std::vector<mlir::MlirOp> reshaped_final_h(all_final_h.size());
    std::vector<mlir::MlirOp> reshaped_final_c(all_final_c.size());
    for (size_t i = 0; i < all_final_h.size(); ++i) {
      reshaped_final_h[i] =
          mlir::stablehlo::Reshape(all_final_h[i], {1, batch, out_h});
      reshaped_final_c[i] =
          mlir::stablehlo::Reshape(all_final_c[i], {1, batch, hidden});
    }

    mlir::MlirOp final_hy = ConcatDim(builder, reshaped_final_h, /*dim=*/0);
    mlir::MlirOp final_cy = ConcatDim(builder, reshaped_final_c, /*dim=*/0);

    // Assemble complete output list: [output, hy, cy, cached_acts...]
    mlir::SmallVector<mlir::MlirOp> results;
    results.reserve(3 + cached_acts.size());
    results.push_back(to_out(final_output));
    results.push_back(final_hy);
    results.push_back(final_cy);
    for (auto act : cached_acts) {
      results.push_back(act);
    }
    return results;
  };

  // --- Step 6: Dispatch Operator via TT_KERNEL Execution Engine ---
  std::vector<absl::Span<const int64_t>> out_dims_spans;
  out_dims_spans.reserve(out_dims_list.size());
  for (const auto& d : out_dims_list) {
    out_dims_spans.push_back(d);
  }

  absl::Span<const mlir::ElementType> out_dtypes_span = out_dtypes;
  absl::Span<const absl::Span<const int64_t>> out_dims_spans_ref =
      out_dims_spans;

  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = out_dtypes_span,
      .out_dims_list = out_dims_spans_ref,
      .op_param_cache_keys = std::move(param_keys),
  };

  return DispatchOp<kDynamicSize, kDynamicSize>(std::move(op_builder), inputs,
                                                std::move(options));
}

// Forward execution entry point: wraps LstmInputImpl with TT_KERNEL for
// automatic compilation caching, device buffer tracking, and TPU execution.
std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenLstmInputForward(
    const at::Tensor& input, const at::TensorList hx,
    const at::TensorList params, const bool has_biases,
    const int64_t num_layers, const double dropout, const bool train,
    const bool bidirectional, const bool batch_first) {
  const bool has_dropout = (dropout > 0.0 && train && num_layers > 1);
  if (has_dropout) {
    const int64_t batch = batch_first ? input.size(0) : input.size(1);
    const int64_t seq_len = batch_first ? input.size(1) : input.size(0);
    const int64_t out_h = hx[0].size(2);
    const int64_t num_directions = bidirectional ? 2 : 1;
    const int64_t num_elements =
        (num_layers - 1) * seq_len * batch * (num_directions * out_h);
    TT_ASSIGN_OR_THROW(const mlir::ElementType in_dtype,
                       ConvertTo<mlir::ElementType>(input.scalar_type()));
    const int64_t bit_width = TorchEquivalentBitwidth(in_dtype);
    TT_KERNEL(OpName::kLstmInput, param_keys,
              (input, hx, params, has_biases, num_layers, dropout, train,
               bidirectional, batch_first),
              {
                TT_ASSIGN_OR_THROW(
                    const std::vector<DeviceBufferRef> result_buffers,
                    DispatchRngOpGeneral(
                        /*generator=*/std::nullopt,
                        [&](at::Tensor rng_state)
                            -> absl::StatusOr<std::vector<DeviceBufferRef>> {
                          TT_ASSIGN_OR_RETURN(
                              DeviceBufferRefArray<3> bufs,
                              LstmInputImpl(input, hx, params, has_biases,
                                            num_layers, dropout, train,
                                            bidirectional, batch_first,
                                            std::move(param_keys), rng_state));
                          return std::vector<DeviceBufferRef>(bufs.begin(),
                                                              bufs.end());
                        },
                        RngUsage{num_elements, bit_width}));
                return {MakeTensor(result_buffers[0]),
                        MakeTensor(result_buffers[1]),
                        MakeTensor(result_buffers[2])};
              });
  } else {
    TT_KERNEL(
        OpName::kLstmInput, param_keys,
        (input, hx, params, has_biases, num_layers, dropout, train,
         bidirectional, batch_first),
        {
          TT_ASSIGN_OR_THROW(
              const DeviceBufferRefArray<3> result_buffers,
              LstmInputImpl(input, hx, params, has_biases, num_layers, dropout,
                            train, bidirectional, batch_first,
                            std::move(param_keys), /*rng_state=*/std::nullopt));
          return {MakeTensor(result_buffers[0]), MakeTensor(result_buffers[1]),
                  MakeTensor(result_buffers[2])};
        });
  }
}

// Forward execution with intermediate activation caching for zero-recomputation
// backward passes.
std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<at::Tensor>>
AtenLstmInputForwardCached(const at::Tensor& input, const at::TensorList hx,
                           const at::TensorList params, const bool has_biases,
                           const int64_t num_layers, const double dropout,
                           const bool train, const bool bidirectional,
                           const bool batch_first) {
  const bool has_dropout = (dropout > 0.0 && train && num_layers > 1);
  if (has_dropout) {
    const int64_t batch = batch_first ? input.size(0) : input.size(1);
    const int64_t seq_len = batch_first ? input.size(1) : input.size(0);
    const int64_t out_h = hx[0].size(2);
    const int64_t num_directions = bidirectional ? 2 : 1;
    const int64_t num_elements =
        (num_layers - 1) * seq_len * batch * (num_directions * out_h);
    TT_ASSIGN_OR_THROW(const mlir::ElementType in_dtype,
                       ConvertTo<mlir::ElementType>(input.scalar_type()));
    const int64_t bit_width = TorchEquivalentBitwidth(in_dtype);
    TT_KERNEL(OpName::kLstmInputForwardCached, param_keys,
              (input, hx, params, has_biases, num_layers, dropout, train,
               bidirectional, batch_first),
              {
                std::optional<at::Tensor> saved_rng_state;
                TT_ASSIGN_OR_THROW(
                    const std::vector<DeviceBufferRef> result_buffers,
                    DispatchRngOpGeneral(
                        /*generator=*/std::nullopt,
                        [&](at::Tensor rng_state) {
                          saved_rng_state = rng_state;
                          return LstmInputForwardCachedImpl(
                              input, hx, params, has_biases, num_layers,
                              dropout, train, bidirectional, batch_first,
                              std::move(param_keys), rng_state);
                        },
                        RngUsage{num_elements, bit_width}));
                at::Tensor output = MakeTensor(result_buffers[0]);
                at::Tensor hy = MakeTensor(result_buffers[1]);
                at::Tensor cy = MakeTensor(result_buffers[2]);
                std::vector<at::Tensor> cached_acts;
                cached_acts.reserve(result_buffers.size() - 3 + 1);
                for (size_t i = 3; i < result_buffers.size(); ++i) {
                  cached_acts.push_back(MakeTensor(result_buffers[i]));
                }
                if (saved_rng_state.has_value()) {
                  cached_acts.push_back(std::move(*saved_rng_state));
                }
                return {std::move(output), std::move(hy), std::move(cy),
                        std::move(cached_acts)};
              });
  } else {
    TT_KERNEL(OpName::kLstmInputForwardCached, param_keys,
              (input, hx, params, has_biases, num_layers, dropout, train,
               bidirectional, batch_first),
              {
                TT_ASSIGN_OR_THROW(
                    const std::vector<DeviceBufferRef> result_buffers,
                    LstmInputForwardCachedImpl(input, hx, params, has_biases,
                                               num_layers, dropout, train,
                                               bidirectional, batch_first,
                                               std::move(param_keys),
                                               /*rng_state=*/std::nullopt));
                at::Tensor output = MakeTensor(result_buffers[0]);
                at::Tensor hy = MakeTensor(result_buffers[1]);
                at::Tensor cy = MakeTensor(result_buffers[2]);
                std::vector<at::Tensor> cached_acts;
                cached_acts.reserve(result_buffers.size() - 3);
                for (size_t i = 3; i < result_buffers.size(); ++i) {
                  cached_acts.push_back(MakeTensor(result_buffers[i]));
                }
                return {std::move(output), std::move(hy), std::move(cy),
                        std::move(cached_acts)};
              });
  }
}

// Backward execution entry point: wraps LstmInputBackwardImpl with TT_KERNEL
// under OpName::kLstmInputBackward for compilation caching and device buffer
// dispatch.
std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<at::Tensor>>
AtenLstmInputBackward(const at::Tensor& grad_output, const at::Tensor& grad_hy,
                      const at::Tensor& grad_cy, const at::Tensor& input,
                      const at::TensorList hx, const at::TensorList params,
                      const bool has_biases, const int64_t num_layers,
                      const double dropout, const bool train,
                      const bool bidirectional, const bool batch_first,
                      const at::TensorList cached_activations) {
  TT_KERNEL(
      OpName::kLstmInputBackward, param_keys,
      (grad_output, grad_hy, grad_cy, input, hx, params, has_biases, num_layers,
       dropout, train, bidirectional, batch_first, cached_activations),
      {
        TT_ASSIGN_OR_THROW(
            const std::vector<DeviceBufferRef> result_buffers,
            LstmInputBackwardImpl(grad_output, grad_hy, grad_cy, input, hx,
                                  params, cached_activations, has_biases,
                                  num_layers, dropout, train, bidirectional,
                                  batch_first, std::move(param_keys)));
        at::Tensor grad_input = MakeTensor(result_buffers[0]);
        at::Tensor grad_h0 = MakeTensor(result_buffers[1]);
        at::Tensor grad_c0 = MakeTensor(result_buffers[2]);
        std::vector<at::Tensor> grad_params;
        grad_params.reserve(result_buffers.size() - 3);
        for (size_t i = 3; i < result_buffers.size(); ++i) {
          grad_params.push_back(MakeTensor(result_buffers[i]));
        }
        return {std::move(grad_input), std::move(grad_h0), std::move(grad_c0),
                std::move(grad_params)};
      });
}

// =============================================================================
// PyTorch Autograd Integration (AtenLstmInputAutograd)
// =============================================================================
// In PyTorch core:
// - `aten::lstm.input` has NO derivative formula in `derivatives.yaml`.
// - `native_functions.yaml` does not declare `aten::lstm_input_backward`.
// As a consequence, PyTorch's native autograd engine cannot differentiate
// through `aten::lstm.input` automatically on non-CPU devices.
//
// To seamlessly bridge this gap in TorchTPU:
// - `AtenLstmInputAutograd` inherits from
// `torch::autograd::Function<AtenLstmInputAutograd>`.
// - In `forward()`, input tensors, initial states, and weights are saved via
// `ctx->save_for_backward()`.
// - When activation caching is enabled, forward activations are cached on TPU
// and reused directly in backward, completely skipping forward recomputation.
// - In `backward()`, incoming adjoints (grad_output, grad_hy, grad_cy) are
// passed
//   directly to `AtenLstmInputBackward` to compute gradients on TPU in
//   StableHLO.
// - Gradients are returned in the precise sequence matching `forward()`
// parameters,
//   with undefined `at::Tensor()` placeholders for non-differentiable primitive
//   arguments.
// NOLINTBEGIN(misc-include-cleaner)
struct AtenLstmInputAutograd
    : public torch::autograd::Function<AtenLstmInputAutograd> {
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx, const at::Tensor& input,
      at::TensorList hx, at::TensorList params, bool has_biases,
      int64_t num_layers, double dropout, bool train, bool bidirectional,
      bool batch_first);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs);
};

torch::autograd::variable_list AtenLstmInputAutograd::forward(
    torch::autograd::AutogradContext* ctx, const at::Tensor& input,
    const at::TensorList hx, const at::TensorList params, const bool has_biases,
    const int64_t num_layers, const double dropout, const bool train,
    const bool bidirectional, const bool batch_first) {
  auto [output, hy, cy, cached_acts] =
      AtenLstmInputForwardCached(input, hx, params, has_biases, num_layers,
                                 dropout, train, bidirectional, batch_first);

  std::vector<at::Tensor> to_save;
  to_save.reserve(3 + params.size() + cached_acts.size());
  to_save.push_back(input);
  to_save.push_back(hx[0]);
  to_save.push_back(hx[1]);
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

  return {output, hy, cy};
}

torch::autograd::variable_list AtenLstmInputAutograd::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
  const auto saved = ctx->get_saved_variables();
  const at::Tensor& input = saved[0];
  const at::Tensor& h_0 = saved[1];
  const at::Tensor& c_0 = saved[2];
  const int64_t params_count = ctx->saved_data["params_count"].toInt();
  std::vector<at::Tensor> params(saved.begin() + 3,
                                 saved.begin() + 3 + params_count);
  std::vector<at::Tensor> cached_acts(saved.begin() + 3 + params_count,
                                      saved.end());

  const bool has_biases = ctx->saved_data["has_biases"].toBool();
  const int64_t num_layers = ctx->saved_data["num_layers"].toInt();
  const double dropout = ctx->saved_data["dropout"].toDouble();
  const bool train = ctx->saved_data["train"].toBool();
  const bool bidirectional = ctx->saved_data["bidirectional"].toBool();
  const bool batch_first = ctx->saved_data["batch_first"].toBool();

  const int64_t batch = batch_first ? input.size(0) : input.size(1);
  const int64_t seq_len = batch_first ? input.size(1) : input.size(0);
  const int64_t out_h = h_0.size(2);
  const int64_t hidden = c_0.size(2);

  const int64_t num_directions = bidirectional ? 2 : 1;

  // Default undefined gradients to zero if not provided by downstream loss
  at::Tensor grad_output = grad_outputs[0];
  if (!grad_output.defined()) {
    Dimensions out_shape =
        batch_first ? Dimensions{batch, seq_len, out_h * num_directions}
                    : Dimensions{seq_len, batch, out_h * num_directions};
    grad_output = at::zeros(out_shape, input.options());
  }

  at::Tensor grad_hy = grad_outputs[1];
  if (!grad_hy.defined()) {
    grad_hy =
        at::zeros({num_layers * num_directions, batch, out_h}, input.options());
  }

  at::Tensor grad_cy = grad_outputs[2];
  if (!grad_cy.defined()) {
    grad_cy = at::zeros({num_layers * num_directions, batch, hidden},
                        input.options());
  }

  auto [grad_input, grad_h0, grad_c0, grad_params] = AtenLstmInputBackward(
      grad_output, grad_hy, grad_cy, input, {h_0, c_0}, params, has_biases,
      num_layers, dropout, train, bidirectional, batch_first, cached_acts);

  // Return gradients matching forward arguments:
  // 1 (input) + 2 (hx) + params_count (params) + 6 (non-tensor options)
  torch::autograd::variable_list result;
  result.reserve(9 + params_count);
  result.push_back(std::move(grad_input));
  result.push_back(std::move(grad_h0));
  result.push_back(std::move(grad_c0));
  for (at::Tensor& p_grad : grad_params) {
    result.push_back(std::move(p_grad));
  }
  // Undefined gradients for non-tensor forward arguments:
  result.push_back(at::Tensor());  // has_biases
  result.push_back(at::Tensor());  // num_layers
  result.push_back(at::Tensor());  // dropout
  result.push_back(at::Tensor());  // train
  result.push_back(at::Tensor());  // bidirectional
  result.push_back(at::Tensor());  // batch_first

  return result;
}
// NOLINTEND(misc-include-cleaner)
// NOLINTEND(readability-function-cognitive-complexity)

}  // namespace

// =============================================================================
// Public Dispatch Entry Point (AtenLstmInput)
// =============================================================================
// Dispatched by PyTorch's eager operator dispatcher when aten::lstm.input is
// called. Checks whether autograd gradient tracking is active on any input
// tensor or parameter.
// - If requires_grad is true and GradMode is enabled: dispatches via
// AtenLstmInputAutograd::apply().
// - Otherwise: directly executes AtenLstmInputForward().
std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenLstmInput(
    const at::Tensor& input, const at::TensorList hx,
    const at::TensorList params, const bool has_biases,
    const int64_t num_layers, const double dropout, const bool train,
    const bool bidirectional, const bool batch_first) {
  bool requires_grad = input.requires_grad() ||
                       (!hx.empty() && hx[0].requires_grad()) ||
                       (hx.size() > 1 && hx[1].requires_grad());
  if (!requires_grad) {
    for (const at::Tensor& p : params) {
      if (p.requires_grad()) {
        requires_grad = true;
        break;
      }
    }
  }

  if (c10::GradMode::is_enabled() && requires_grad) {
    auto results = AtenLstmInputAutograd::apply(input, hx, params, has_biases,
                                                num_layers, dropout, train,
                                                bidirectional, batch_first);
    return {results[0], results[1], results[2]};
  }

  return AtenLstmInputForward(input, hx, params, has_biases, num_layers,
                              dropout, train, bidirectional, batch_first);
}

}  // namespace torch_tpu
