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

#include "torch_tpu/ops/thnn_fused_lstm_cell/thnn_fused_lstm_cell_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/sum.h"
#include "ATen/ops/zeros_like.h"
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

// Given the summed pre-activation gates [B, 4H] and previous cell state cx,
// applies the LSTM nonlinearities and returns (hy, cy, workspace) where
// workspace = concat(i, f, g, o) of the activated gates (for the fused
// backward). All StableHLO builder ops take non-const MlirOp& lvalues, so
// every intermediate is a named variable.
absl::StatusOr<std::array<mlir::MlirOp, 3>> LstmCellFromGates(
    mlir::MlirOp gates, mlir::MlirOp cx, int64_t batch, int64_t hidden) {
  namespace shlo = mlir::stablehlo;
  mlir::MlirOp i_pre = shlo::Slice(gates, {0, 0}, {batch, hidden}, {1, 1});
  mlir::MlirOp f_pre =
      shlo::Slice(gates, {0, hidden}, {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp g_pre =
      shlo::Slice(gates, {0, 2 * hidden}, {batch, 3 * hidden}, {1, 1});
  mlir::MlirOp o_pre =
      shlo::Slice(gates, {0, 3 * hidden}, {batch, 4 * hidden}, {1, 1});

  mlir::MlirOp ingate = shlo::Logistic(i_pre);
  mlir::MlirOp forgetgate = shlo::Logistic(f_pre);
  mlir::MlirOp cellgate = shlo::Tanh(g_pre);
  mlir::MlirOp outgate = shlo::Logistic(o_pre);

  mlir::MlirOp f_cx = shlo::Mul(forgetgate, cx);
  mlir::MlirOp i_g = shlo::Mul(ingate, cellgate);
  mlir::MlirOp cy = shlo::Add(f_cx, i_g);
  mlir::MlirOp tanh_cy = shlo::Tanh(cy);
  mlir::MlirOp hy = shlo::Mul(outgate, tanh_cy);
  mlir::MlirOp workspace = shlo::Concatenate(
      gates.getBuilder(), {ingate, forgetgate, cellgate, outgate}, /*dim=*/1);
  return std::array<mlir::MlirOp, 3>{hy, cy, workspace};
}

// Backward of the fused LSTM cell. `workspace` holds the activated gates
// concat(i, f, g, o) produced by the forward. Returns (grad_gates, grad_cx)
// where grad_gates [B, 4H] are the gradients w.r.t. the pre-activation gates
// (== grad w.r.t. both input_gates and hidden_gates). grad_bias is the batch
// sum of grad_gates, computed by the caller with a plain reduction.
absl::StatusOr<std::array<mlir::MlirOp, 2>> BuildLstmCellBackwardShlo(
    mlir::MlirOp grad_hy, mlir::MlirOp grad_cy, mlir::MlirOp cx, mlir::MlirOp cy,
    mlir::MlirOp workspace, int64_t batch, int64_t hidden) {
  namespace shlo = mlir::stablehlo;
  mlir::MlirOp ingate = shlo::Slice(workspace, {0, 0}, {batch, hidden}, {1, 1});
  mlir::MlirOp forgetgate =
      shlo::Slice(workspace, {0, hidden}, {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp cellgate =
      shlo::Slice(workspace, {0, 2 * hidden}, {batch, 3 * hidden}, {1, 1});
  mlir::MlirOp outgate =
      shlo::Slice(workspace, {0, 3 * hidden}, {batch, 4 * hidden}, {1, 1});

  mlir::MlirOp one = MakeConstantLike(cy, 1.0);
  mlir::MlirOp tanh_cy = shlo::Tanh(cy);

  // d(cy) from the hy path: grad_hy * outgate * (1 - tanh(cy)^2), plus grad_cy.
  mlir::MlirOp tanh_cy_sq = shlo::Mul(tanh_cy, tanh_cy);
  mlir::MlirOp one_minus_tanh_cy_sq = shlo::Subtract(one, tanh_cy_sq);
  mlir::MlirOp grad_hy_out = shlo::Mul(grad_hy, outgate);
  mlir::MlirOp d_cy_from_hy = shlo::Mul(grad_hy_out, one_minus_tanh_cy_sq);
  mlir::MlirOp d_cy = shlo::Add(grad_cy, d_cy_from_hy);
  mlir::MlirOp d_outgate = shlo::Mul(grad_hy, tanh_cy);

  // Gradients into the activated gates.
  mlir::MlirOp d_forgetgate = shlo::Mul(d_cy, cx);
  mlir::MlirOp grad_cx = shlo::Mul(d_cy, forgetgate);
  mlir::MlirOp d_ingate = shlo::Mul(d_cy, cellgate);
  mlir::MlirOp d_cellgate = shlo::Mul(d_cy, ingate);

  // Back through the gate activations to the pre-activation gates.
  mlir::MlirOp one_minus_i = shlo::Subtract(one, ingate);
  mlir::MlirOp i_deriv = shlo::Mul(ingate, one_minus_i);
  mlir::MlirOp gi = shlo::Mul(d_ingate, i_deriv);

  mlir::MlirOp one_minus_f = shlo::Subtract(one, forgetgate);
  mlir::MlirOp f_deriv = shlo::Mul(forgetgate, one_minus_f);
  mlir::MlirOp gf = shlo::Mul(d_forgetgate, f_deriv);

  mlir::MlirOp cellgate_sq = shlo::Mul(cellgate, cellgate);
  mlir::MlirOp one_minus_g_sq = shlo::Subtract(one, cellgate_sq);
  mlir::MlirOp gg = shlo::Mul(d_cellgate, one_minus_g_sq);

  mlir::MlirOp one_minus_o = shlo::Subtract(one, outgate);
  mlir::MlirOp o_deriv = shlo::Mul(outgate, one_minus_o);
  mlir::MlirOp go = shlo::Mul(d_outgate, o_deriv);

  mlir::MlirOp grad_gates =
      shlo::Concatenate(cy.getBuilder(), {gi, gf, gg, go}, /*dim=*/1);
  return std::array<mlir::MlirOp, 2>{grad_gates, grad_cx};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenThnnFusedLstmCell(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& cx, const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias) {
  const int64_t batch = cx.size(0);
  const int64_t hidden = cx.size(1);
  // nn.LSTM(bias=False) passes the biases as optionals holding *undefined*
  // tensors (has_value() is true but .defined() is false), so check both.
  const bool has_bias = input_bias.has_value() && input_bias->defined() &&
                        hidden_bias.has_value() && hidden_bias->defined();

  TT_KERNEL(OpName::kThnnFusedLstmCell, param_keys,
            (input_gates, hidden_gates, cx, input_bias, hidden_bias), {
    TT_ASSIGN_OR_THROW(const auto out_dtype,
                       ConvertTo<mlir::ElementType>(cx.scalar_type()));
    const std::array<int64_t, 2> hc_dims = {batch, hidden};
    const std::array<int64_t, 2> ws_dims = {batch, 4 * hidden};
    const std::array<mlir::ElementType, 3> out_dtypes = {out_dtype, out_dtype,
                                                          out_dtype};
    const std::array<absl::Span<const int64_t>, 3> out_dims_list = {
        hc_dims, hc_dims, ws_dims};

    if (has_bias) {
      auto op_builder =
          [batch, hidden](FixedSizeSpan<mlir::MlirOp, 5> inputs)
          -> absl::StatusOr<std::array<mlir::MlirOp, 3>> {
        auto& [ig, hg, c, ib, hb] = inputs;
        mlir::MlirOp g1 = mlir::stablehlo::Add(ig, hg);
        mlir::MlirOp g2 = mlir::chlo::BroadcastAdd(g1, ib);
        mlir::MlirOp g3 = mlir::chlo::BroadcastAdd(g2, hb);
        return LstmCellFromGates(g3, c, batch, hidden);
      };
      TT_ASSIGN_OR_THROW(
          (auto [hy, cy, workspace]),
          (DispatchOp<5, 3>(
              std::move(op_builder),
              {input_gates, hidden_gates, cx, *input_bias, *hidden_bias},
              {.out_dtypes = out_dtypes,
               .out_dims_list = out_dims_list,
               .op_param_cache_keys = std::move(param_keys)})));
      return {MakeTensor(std::move(hy)), MakeTensor(std::move(cy)),
              MakeTensor(std::move(workspace))};
    } else {
      auto op_builder =
          [batch, hidden](FixedSizeSpan<mlir::MlirOp, 3> inputs)
          -> absl::StatusOr<std::array<mlir::MlirOp, 3>> {
        auto& [ig, hg, c] = inputs;
        mlir::MlirOp g1 = mlir::stablehlo::Add(ig, hg);
        return LstmCellFromGates(g1, c, batch, hidden);
      };
      TT_ASSIGN_OR_THROW(
          (auto [hy, cy, workspace]),
          (DispatchOp<3, 3>(std::move(op_builder),
                            {input_gates, hidden_gates, cx},
                            {.out_dtypes = out_dtypes,
                             .out_dims_list = out_dims_list,
                             .op_param_cache_keys = std::move(param_keys)})));
      return {MakeTensor(std::move(hy)), MakeTensor(std::move(cy)),
              MakeTensor(std::move(workspace))};
    }
  });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenThnnFusedLstmCellBackwardImpl(const std::optional<at::Tensor>& grad_hy,
                                  const std::optional<at::Tensor>& grad_cy,
                                  const at::Tensor& cx, const at::Tensor& cy,
                                  const at::Tensor& workspace, bool has_bias) {
  const int64_t batch = cy.size(0);
  const int64_t hidden = cy.size(1);
  // Undefined output grads are treated as zeros (autograd omits grad_cy on the
  // final timestep). Note an undefined grad arrives as an optional that HAS a
  // value but whose tensor is not .defined(), so check both. Ordinary TPU ops.
  const at::Tensor grad_hy_t = (grad_hy.has_value() && grad_hy->defined())
                                   ? *grad_hy
                                   : at::zeros_like(cy);
  const at::Tensor grad_cy_t = (grad_cy.has_value() && grad_cy->defined())
                                   ? *grad_cy
                                   : at::zeros_like(cy);

  at::Tensor grad_gates;
  at::Tensor grad_cx;
  TT_KERNEL(OpName::kThnnFusedLstmCellBackwardImpl, param_keys,
            (grad_hy_t, grad_cy_t, cx, cy, workspace, has_bias), {
    TT_ASSIGN_OR_THROW(const auto out_dtype,
                       ConvertTo<mlir::ElementType>(cy.scalar_type()));
    const std::array<int64_t, 2> gg_dims = {batch, 4 * hidden};
    const std::array<int64_t, 2> gcx_dims = {batch, hidden};
    const std::array<mlir::ElementType, 2> out_dtypes = {out_dtype, out_dtype};
    const std::array<absl::Span<const int64_t>, 2> out_dims_list = {gg_dims,
                                                                    gcx_dims};

    auto op_builder = [batch, hidden](FixedSizeSpan<mlir::MlirOp, 5> inputs) {
      auto& [ghy, gcy, c, y, ws] = inputs;
      return BuildLstmCellBackwardShlo(ghy, gcy, c, y, ws, batch, hidden);
    };
    TT_ASSIGN_OR_THROW(
        (auto [gg, gcx]),
        (DispatchOp<5, 2>(std::move(op_builder),
                          {grad_hy_t, grad_cy_t, cx, cy, workspace},
                          {.out_dtypes = out_dtypes,
                           .out_dims_list = out_dims_list,
                           .op_param_cache_keys = std::move(param_keys)})));
    grad_gates = MakeTensor(std::move(gg));
    grad_cx = MakeTensor(std::move(gcx));
  });

  // grad_bias = batch sum of grad_gates (the fused-cell wrapper duplicates it
  // to both bias grads and drops it when has_bias is false). Plain reduction.
  const at::Tensor grad_bias = at::sum(grad_gates, /*dim=*/{0});
  return {grad_gates, grad_cx, grad_bias};
}

}  // namespace torch_tpu
