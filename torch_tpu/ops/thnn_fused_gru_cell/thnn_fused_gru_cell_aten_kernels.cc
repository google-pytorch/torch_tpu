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

#include "torch_tpu/ops/thnn_fused_gru_cell/thnn_fused_gru_cell_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/sum.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

// From the (bias-added) input/hidden gate pre-activations [B, 3H] and previous
// hidden state hx, applies the GRU nonlinearities and returns
// (hy, workspace=concat(r, z, n, hx, h_n)). All StableHLO builder ops take
// non-const MlirOp& lvalues, so every intermediate is a named variable.
absl::StatusOr<std::array<mlir::MlirOp, 2>> BuildGruCellShlo(
    mlir::MlirOp ig, mlir::MlirOp hg, mlir::MlirOp hx, int64_t batch,
    int64_t hidden) {
  namespace shlo = mlir::stablehlo;
  mlir::MlirOp i_r = shlo::Slice(ig, {0, 0}, {batch, hidden}, {1, 1});
  mlir::MlirOp i_z = shlo::Slice(ig, {0, hidden}, {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp i_n =
      shlo::Slice(ig, {0, 2 * hidden}, {batch, 3 * hidden}, {1, 1});
  mlir::MlirOp h_r = shlo::Slice(hg, {0, 0}, {batch, hidden}, {1, 1});
  mlir::MlirOp h_z = shlo::Slice(hg, {0, hidden}, {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp h_n =
      shlo::Slice(hg, {0, 2 * hidden}, {batch, 3 * hidden}, {1, 1});

  mlir::MlirOp pre_r = shlo::Add(i_r, h_r);
  mlir::MlirOp resetgate = shlo::Logistic(pre_r);
  mlir::MlirOp pre_z = shlo::Add(i_z, h_z);
  mlir::MlirOp updategate = shlo::Logistic(pre_z);
  mlir::MlirOp r_hn = shlo::Mul(resetgate, h_n);
  mlir::MlirOp pre_n = shlo::Add(i_n, r_hn);
  mlir::MlirOp newgate = shlo::Tanh(pre_n);

  // hy = (hx - n) * z + n
  mlir::MlirOp hx_minus_n = shlo::Subtract(hx, newgate);
  mlir::MlirOp scaled = shlo::Mul(hx_minus_n, updategate);
  mlir::MlirOp hy = shlo::Add(scaled, newgate);

  mlir::MlirOp workspace = shlo::Concatenate(
      ig.getBuilder(), {resetgate, updategate, newgate, hx, h_n}, /*dim=*/1);
  return std::array<mlir::MlirOp, 2>{hy, workspace};
}

// Backward. workspace = concat(r, z, n, hx, h_n). Returns
// (grad_input_gates [B,3H], grad_hidden_gates [B,3H], grad_hx [B,H]); the
// input/hidden gate grads differ only in the new-gate slice (h_n carries an
// extra factor r). Bias grads are batch-sums computed by the caller.
absl::StatusOr<std::array<mlir::MlirOp, 3>> BuildGruCellBackwardShlo(
    mlir::MlirOp grad_hy, mlir::MlirOp workspace, int64_t batch,
    int64_t hidden) {
  namespace shlo = mlir::stablehlo;
  mlir::MlirOp resetgate =
      shlo::Slice(workspace, {0, 0}, {batch, hidden}, {1, 1});
  mlir::MlirOp updategate =
      shlo::Slice(workspace, {0, hidden}, {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp newgate =
      shlo::Slice(workspace, {0, 2 * hidden}, {batch, 3 * hidden}, {1, 1});
  mlir::MlirOp hx =
      shlo::Slice(workspace, {0, 3 * hidden}, {batch, 4 * hidden}, {1, 1});
  mlir::MlirOp h_n =
      shlo::Slice(workspace, {0, 4 * hidden}, {batch, 5 * hidden}, {1, 1});

  mlir::MlirOp one = MakeConstantLike(updategate, 1.0);

  // gz = grad_hy * (hx - n);  gn = grad_hy * (1 - z);  ghx = grad_hy * z
  mlir::MlirOp hx_minus_n = shlo::Subtract(hx, newgate);
  mlir::MlirOp d_updategate = shlo::Mul(grad_hy, hx_minus_n);
  mlir::MlirOp one_minus_z = shlo::Subtract(one, updategate);
  mlir::MlirOp d_newgate = shlo::Mul(grad_hy, one_minus_z);
  mlir::MlirOp grad_hx = shlo::Mul(grad_hy, updategate);

  // through the new-gate tanh: gpre_n = gn * (1 - n^2)
  mlir::MlirOp newgate_sq = shlo::Mul(newgate, newgate);
  mlir::MlirOp one_minus_n_sq = shlo::Subtract(one, newgate_sq);
  mlir::MlirOp d_pre_n = shlo::Mul(d_newgate, one_minus_n_sq);

  // reset gate: grad flows through r * h_n
  mlir::MlirOp d_reset = shlo::Mul(d_pre_n, h_n);
  mlir::MlirOp one_minus_r = shlo::Subtract(one, resetgate);
  mlir::MlirOp r_deriv = shlo::Mul(resetgate, one_minus_r);
  mlir::MlirOp d_pre_r = shlo::Mul(d_reset, r_deriv);

  // update gate: gpre_z = gz * z * (1 - z)
  mlir::MlirOp z_deriv = shlo::Mul(updategate, one_minus_z);
  mlir::MlirOp d_pre_z = shlo::Mul(d_updategate, z_deriv);

  // hidden new-gate grad = gpre_n * r (the asymmetry vs. the input new-gate)
  mlir::MlirOp d_h_n = shlo::Mul(d_pre_n, resetgate);

  mlir::MlirOp grad_input_gates =
      shlo::Concatenate(updategate.getBuilder(), {d_pre_r, d_pre_z, d_pre_n},
                        /*dim=*/1);
  mlir::MlirOp grad_hidden_gates =
      shlo::Concatenate(updategate.getBuilder(), {d_pre_r, d_pre_z, d_h_n},
                        /*dim=*/1);
  return std::array<mlir::MlirOp, 3>{grad_input_gates, grad_hidden_gates,
                                     grad_hx};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> AtenThnnFusedGruCell(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& hx, const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias) {
  const int64_t batch = hx.size(0);
  const int64_t hidden = hx.size(1);
  // nn.GRU(bias=False) passes the biases as optionals holding *undefined*
  // tensors (has_value() is true but .defined() is false), so check both.
  const bool has_bias = input_bias.has_value() && input_bias->defined() &&
                        hidden_bias.has_value() && hidden_bias->defined();

  TT_KERNEL(OpName::kThnnFusedGruCell, param_keys,
            (input_gates, hidden_gates, hx, input_bias, hidden_bias), {
    TT_ASSIGN_OR_THROW(const auto out_dtype,
                       ConvertTo<mlir::ElementType>(hx.scalar_type()));
    const std::array<int64_t, 2> hy_dims = {batch, hidden};
    const std::array<int64_t, 2> ws_dims = {batch, 5 * hidden};
    const std::array<mlir::ElementType, 2> out_dtypes = {out_dtype, out_dtype};
    const std::array<absl::Span<const int64_t>, 2> out_dims_list = {hy_dims,
                                                                    ws_dims};

    if (has_bias) {
      auto op_builder =
          [batch, hidden](FixedSizeSpan<mlir::MlirOp, 5> inputs)
          -> absl::StatusOr<std::array<mlir::MlirOp, 2>> {
        auto& [ig, hg, hx_op, ib, hb] = inputs;
        mlir::MlirOp ig_b = mlir::chlo::BroadcastAdd(ig, ib);
        mlir::MlirOp hg_b = mlir::chlo::BroadcastAdd(hg, hb);
        return BuildGruCellShlo(ig_b, hg_b, hx_op, batch, hidden);
      };
      TT_ASSIGN_OR_THROW(
          (auto [hy, workspace]),
          (DispatchOp<5, 2>(
              std::move(op_builder),
              {input_gates, hidden_gates, hx, *input_bias, *hidden_bias},
              {.out_dtypes = out_dtypes,
               .out_dims_list = out_dims_list,
               .op_param_cache_keys = std::move(param_keys)})));
      return {MakeTensor(std::move(hy)), MakeTensor(std::move(workspace))};
    } else {
      auto op_builder =
          [batch, hidden](FixedSizeSpan<mlir::MlirOp, 3> inputs)
          -> absl::StatusOr<std::array<mlir::MlirOp, 2>> {
        auto& [ig, hg, hx_op] = inputs;
        return BuildGruCellShlo(ig, hg, hx_op, batch, hidden);
      };
      TT_ASSIGN_OR_THROW(
          (auto [hy, workspace]),
          (DispatchOp<3, 2>(std::move(op_builder),
                            {input_gates, hidden_gates, hx},
                            {.out_dtypes = out_dtypes,
                             .out_dims_list = out_dims_list,
                             .op_param_cache_keys = std::move(param_keys)})));
      return {MakeTensor(std::move(hy)), MakeTensor(std::move(workspace))};
    }
  });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenThnnFusedGruCellBackward(const at::Tensor& grad_hy,
                             const at::Tensor& workspace, bool has_bias) {
  const int64_t batch = grad_hy.size(0);
  const int64_t hidden = grad_hy.size(1);

  at::Tensor grad_input_gates;
  at::Tensor grad_hidden_gates;
  at::Tensor grad_hx;
  TT_KERNEL(OpName::kThnnFusedGruCellBackward, param_keys,
            (grad_hy, workspace, has_bias), {
    TT_ASSIGN_OR_THROW(const auto out_dtype,
                       ConvertTo<mlir::ElementType>(grad_hy.scalar_type()));
    const std::array<int64_t, 2> g_dims = {batch, 3 * hidden};
    const std::array<int64_t, 2> hx_dims = {batch, hidden};
    const std::array<mlir::ElementType, 3> out_dtypes = {out_dtype, out_dtype,
                                                         out_dtype};
    const std::array<absl::Span<const int64_t>, 3> out_dims_list = {
        g_dims, g_dims, hx_dims};

    auto op_builder = [batch, hidden](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
      auto& [ghy, ws] = inputs;
      return BuildGruCellBackwardShlo(ghy, ws, batch, hidden);
    };
    TT_ASSIGN_OR_THROW(
        (auto [gig, ghg, ghx]),
        (DispatchOp<2, 3>(std::move(op_builder), {grad_hy, workspace},
                          {.out_dtypes = out_dtypes,
                           .out_dims_list = out_dims_list,
                           .op_param_cache_keys = std::move(param_keys)})));
    grad_input_gates = MakeTensor(std::move(gig));
    grad_hidden_gates = MakeTensor(std::move(ghg));
    grad_hx = MakeTensor(std::move(ghx));
  });

  // Bias grads are the batch-sums of the gate grads, or undefined when the
  // forward had no biases (matches the CUDA op's undefined-tensor return).
  at::Tensor grad_input_bias;
  at::Tensor grad_hidden_bias;
  if (has_bias) {
    grad_input_bias = at::sum(grad_input_gates, /*dim=*/{0});
    grad_hidden_bias = at::sum(grad_hidden_gates, /*dim=*/{0});
  }
  return {grad_input_gates, grad_hidden_gates, grad_hx, grad_input_bias,
          grad_hidden_bias};
}

}  // namespace torch_tpu
