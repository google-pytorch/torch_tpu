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

#include "torch_tpu/csrc/ops/thnn_fused_lstm_cell/thnn_fused_lstm_cell_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/aten_utils.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/reductions/reduction_utils.h"
#include "torch_tpu/csrc/ops/reductions/reductions.h"

namespace torch_tpu {
namespace {

// Builds forward LSTM cell recurrence in StableHLO from pre-activation gates
// (`gates` [B, 4H]) and previous cell state (`cx` [B, H]).
//
// Gate slicing order along dim 1 of `gates` [0..4H):
//   [0, H)   -> i_pre (input gate pre-activation)
//   [H, 2H)  -> f_pre (forget gate pre-activation)
//   [2H, 3H) -> g_pre (cell / candidate gate pre-activation)
//   [3H, 4H) -> o_pre (output gate pre-activation)
//
// Recurrence equations implemented:
//   i  = sigmoid(i_pre)
//   f  = sigmoid(f_pre)
//   g  = tanh(g_pre)
//   o  = sigmoid(o_pre)
//   cy = f * cx + i * g
//   hy = o * tanh(cy)
//
// Returns (hy, cy, workspace), where `workspace` = concat(i, f, g, o) along
// dim 1 [B, 4H] holding activated gates for use in the fused backward pass.
// Note: All StableHLO builder ops take non-const MlirOp& lvalues, so every
// intermediate is a named variable.
absl::StatusOr<std::array<mlir::MlirOp, 3>> LstmCellFromGates(
    mlir::MlirOp gates, mlir::MlirOp cx, int64_t batch, int64_t hidden,
    mlir::ElementType out_dtype) {
  const bool needs_upcast = (out_dtype == mlir::ElementType::F16 ||
                             out_dtype == mlir::ElementType::BF16);
  const mlir::ElementType acc_dtype =
      needs_upcast ? mlir::ElementType::F32 : out_dtype;
  auto to_acc = [acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
    return mlir::stablehlo::ConvertElementType(op, acc_dtype);
  };
  auto to_out = [out_dtype](mlir::MlirOp op) -> mlir::MlirOp {
    return mlir::stablehlo::ConvertElementType(op, out_dtype);
  };

  mlir::MlirOp gates_acc = to_acc(gates);
  mlir::MlirOp cx_acc = to_acc(cx);

  mlir::MlirOp i_pre =
      mlir::stablehlo::Slice(gates_acc, {0, 0}, {batch, hidden}, {1, 1});
  mlir::MlirOp f_pre = mlir::stablehlo::Slice(gates_acc, {0, hidden},
                                              {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp g_pre = mlir::stablehlo::Slice(gates_acc, {0, 2 * hidden},
                                              {batch, 3 * hidden}, {1, 1});
  mlir::MlirOp o_pre = mlir::stablehlo::Slice(gates_acc, {0, 3 * hidden},
                                              {batch, 4 * hidden}, {1, 1});

  mlir::MlirOp ingate = mlir::stablehlo::Logistic(i_pre);
  mlir::MlirOp forgetgate = mlir::stablehlo::Logistic(f_pre);
  mlir::MlirOp cellgate = mlir::stablehlo::Tanh(g_pre);
  mlir::MlirOp outgate = mlir::stablehlo::Logistic(o_pre);

  mlir::MlirOp f_cx = mlir::stablehlo::Mul(forgetgate, cx_acc);
  mlir::MlirOp i_g = mlir::stablehlo::Mul(ingate, cellgate);
  mlir::MlirOp cy_acc = mlir::stablehlo::Add(f_cx, i_g);
  mlir::MlirOp tanh_cy = mlir::stablehlo::Tanh(cy_acc);
  mlir::MlirOp hy_acc = mlir::stablehlo::Mul(outgate, tanh_cy);

  mlir::MlirOp hy = to_out(hy_acc);
  mlir::MlirOp cy = to_out(cy_acc);
  mlir::MlirOp workspace = mlir::stablehlo::Concatenate(
      gates.getBuilder(),
      {to_out(ingate), to_out(forgetgate), to_out(cellgate), to_out(outgate)},
      /*dim=*/1);
  return std::array<mlir::MlirOp, 3>{hy, cy, workspace};
}

// Backward of the fused LSTM cell. `workspace` holds the activated gates
// concat(i, f, g, o) produced by the forward. Returns (grad_gates, grad_cx)
// where grad_gates [B, 4H] are the gradients w.r.t. the pre-activation gates
// (== grad w.r.t. both input_gates and hidden_gates). grad_bias is the batch
// sum of grad_gates, computed by the caller with a plain reduction.
absl::StatusOr<std::array<mlir::MlirOp, 2>> BuildLstmCellBackwardShlo(
    mlir::MlirOp grad_hy, mlir::MlirOp grad_cy, mlir::MlirOp cx,
    mlir::MlirOp cy, mlir::MlirOp workspace, int64_t batch, int64_t hidden,
    mlir::ElementType out_dtype) {
  const bool needs_upcast = (out_dtype == mlir::ElementType::F16 ||
                             out_dtype == mlir::ElementType::BF16);
  const mlir::ElementType acc_dtype =
      needs_upcast ? mlir::ElementType::F32 : out_dtype;
  auto to_acc = [acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
    return mlir::stablehlo::ConvertElementType(op, acc_dtype);
  };
  auto to_out = [out_dtype](mlir::MlirOp op) -> mlir::MlirOp {
    return mlir::stablehlo::ConvertElementType(op, out_dtype);
  };

  mlir::MlirOp grad_hy_acc = to_acc(grad_hy);
  mlir::MlirOp grad_cy_acc = to_acc(grad_cy);
  mlir::MlirOp cx_acc = to_acc(cx);
  mlir::MlirOp cy_acc = to_acc(cy);
  mlir::MlirOp workspace_acc = to_acc(workspace);

  mlir::MlirOp ingate =
      mlir::stablehlo::Slice(workspace_acc, {0, 0}, {batch, hidden}, {1, 1});
  mlir::MlirOp forgetgate = mlir::stablehlo::Slice(workspace_acc, {0, hidden},
                                                   {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp cellgate = mlir::stablehlo::Slice(workspace_acc, {0, 2 * hidden},
                                                 {batch, 3 * hidden}, {1, 1});
  mlir::MlirOp outgate = mlir::stablehlo::Slice(workspace_acc, {0, 3 * hidden},
                                                {batch, 4 * hidden}, {1, 1});

  mlir::MlirOp one = MakeConstantLike(cy_acc, 1.0);
  mlir::MlirOp tanh_cy = mlir::stablehlo::Tanh(cy_acc);

  // d(cy) from the hy path: grad_hy * outgate * (1 - tanh(cy)^2), plus grad_cy.
  mlir::MlirOp tanh_cy_sq = mlir::stablehlo::Mul(tanh_cy, tanh_cy);
  mlir::MlirOp one_minus_tanh_cy_sq =
      mlir::stablehlo::Subtract(one, tanh_cy_sq);
  mlir::MlirOp grad_hy_out = mlir::stablehlo::Mul(grad_hy_acc, outgate);
  mlir::MlirOp d_cy_from_hy =
      mlir::stablehlo::Mul(grad_hy_out, one_minus_tanh_cy_sq);
  mlir::MlirOp d_cy = mlir::stablehlo::Add(grad_cy_acc, d_cy_from_hy);
  mlir::MlirOp d_outgate = mlir::stablehlo::Mul(grad_hy_acc, tanh_cy);

  // Gradients into the activated gates.
  mlir::MlirOp d_forgetgate = mlir::stablehlo::Mul(d_cy, cx_acc);
  mlir::MlirOp grad_cx_acc = mlir::stablehlo::Mul(d_cy, forgetgate);
  mlir::MlirOp d_ingate = mlir::stablehlo::Mul(d_cy, cellgate);
  mlir::MlirOp d_cellgate = mlir::stablehlo::Mul(d_cy, ingate);

  // Back through the gate activations to the pre-activation gates.
  mlir::MlirOp one_minus_i = mlir::stablehlo::Subtract(one, ingate);
  mlir::MlirOp i_deriv = mlir::stablehlo::Mul(ingate, one_minus_i);
  mlir::MlirOp gi = mlir::stablehlo::Mul(d_ingate, i_deriv);

  mlir::MlirOp one_minus_f = mlir::stablehlo::Subtract(one, forgetgate);
  mlir::MlirOp f_deriv = mlir::stablehlo::Mul(forgetgate, one_minus_f);
  mlir::MlirOp gf = mlir::stablehlo::Mul(d_forgetgate, f_deriv);

  mlir::MlirOp cellgate_sq = mlir::stablehlo::Mul(cellgate, cellgate);
  mlir::MlirOp one_minus_g_sq = mlir::stablehlo::Subtract(one, cellgate_sq);
  mlir::MlirOp gg = mlir::stablehlo::Mul(d_cellgate, one_minus_g_sq);

  mlir::MlirOp one_minus_o = mlir::stablehlo::Subtract(one, outgate);
  mlir::MlirOp o_deriv = mlir::stablehlo::Mul(outgate, one_minus_o);
  mlir::MlirOp go = mlir::stablehlo::Mul(d_outgate, o_deriv);

  mlir::MlirOp grad_gates = mlir::stablehlo::Concatenate(
      cy.getBuilder(), {to_out(gi), to_out(gf), to_out(gg), to_out(go)},
      /*dim=*/1);
  mlir::MlirOp grad_cx = to_out(grad_cx_acc);
  return std::array<mlir::MlirOp, 2>{grad_gates, grad_cx};
}

absl::StatusOr<DeviceBufferRefArray<3>> ThnnFusedLstmCellImpl(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& cx, const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias, OpParamCacheKeys param_keys) {
  TT_RET_CHECK(input_gates.sizes() == hidden_gates.sizes(),
               error::kInvalidArgument)
      << "expected size of argument #1 'input_gates' to match size of "
         "argument #2 'hidden_gates' ("
      << ToString(hidden_gates.sizes()) << "), got "
      << ToString(input_gates.sizes());
  TT_RET_CHECK(input_gates.size(0) == cx.size(0), error::kInvalidArgument)
      << "expected batch size of argument #1 'input_gates' to match batch size "
         "of argument #3 'cx' ("
      << cx.size(0) << "), got " << input_gates.size(0);
  TT_RET_CHECK(input_gates.size(1) == 4 * cx.size(1), error::kInvalidArgument)
      << "expected feature size of argument #1 'input_gates' to match 4 * "
         "feature size of argument #3 'cx' ("
      << 4 * cx.size(1) << "), got " << input_gates.size(1);

  const int64_t batch = cx.size(0);
  const int64_t hidden = cx.size(1);
  auto s_input_bias = SanitizeOptionalTensor(input_bias);
  auto s_hidden_bias = SanitizeOptionalTensor(hidden_bias);
  const bool has_bias = s_input_bias.has_value() && s_hidden_bias.has_value();

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=unsupported dtypes caught at
                        // tensor creation
      const auto out_dtype, ConvertTo<mlir::ElementType>(cx.scalar_type()));
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=unsupported dtypes caught at
                        // tensor creation
      const auto acc_dtype,
      ConvertTo<mlir::ElementType>(ToAccumulateType(cx.scalar_type())));
  const std::array<int64_t, 2> hc_dims = {batch, hidden};
  const std::array<int64_t, 2> ws_dims = {batch, 4 * hidden};
  const std::array<mlir::ElementType, 3> out_dtypes = {out_dtype, out_dtype,
                                                       out_dtype};
  const std::array<absl::Span<const int64_t>, 3> out_dims_list = {
      hc_dims, hc_dims, ws_dims};

  if (has_bias) {
    auto op_builder = [batch, hidden, acc_dtype,
                       out_dtype](FixedSizeSpan<mlir::MlirOp, 5> inputs)
        -> absl::StatusOr<std::array<mlir::MlirOp, 3>> {
      auto to_acc = [acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
        return mlir::stablehlo::ConvertElementType(op, acc_dtype);
      };
      auto& [ig, hg, c, ib, hb] = inputs;
      mlir::MlirOp ig_acc = to_acc(ig);
      mlir::MlirOp hg_acc = to_acc(hg);
      mlir::MlirOp ib_acc = to_acc(ib);
      mlir::MlirOp hb_acc = to_acc(hb);
      mlir::MlirOp ib_b = mlir::stablehlo::BroadcastInDim(
          GetTensorTypeOrDie(ig_acc), ib_acc, {/*broadcast_dimensions=*/1});
      mlir::MlirOp hb_b = mlir::stablehlo::BroadcastInDim(
          GetTensorTypeOrDie(hg_acc), hb_acc, {/*broadcast_dimensions=*/1});
      mlir::MlirOp ig_b = mlir::stablehlo::Add(ig_acc, ib_b);
      mlir::MlirOp hg_b = mlir::stablehlo::Add(hg_acc, hb_b);
      mlir::MlirOp g3 = mlir::stablehlo::Add(ig_b, hg_b);
      return LstmCellFromGates(g3, c, batch, hidden, out_dtype);
    };
    return DispatchOp<5, 3>(
        std::move(op_builder),
        {input_gates, hidden_gates, cx, *s_input_bias, *s_hidden_bias},
        {.out_dtypes = out_dtypes,
         .out_dims_list = out_dims_list,
         .op_param_cache_keys = std::move(param_keys)});
  } else {
    auto op_builder = [batch, hidden, acc_dtype,
                       out_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<std::array<mlir::MlirOp, 3>> {
      auto to_acc = [acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
        return mlir::stablehlo::ConvertElementType(op, acc_dtype);
      };
      auto& [ig, hg, c] = inputs;
      mlir::MlirOp ig_acc = to_acc(ig);
      mlir::MlirOp hg_acc = to_acc(hg);
      mlir::MlirOp g1 = mlir::stablehlo::Add(ig_acc, hg_acc);
      return LstmCellFromGates(g1, c, batch, hidden, out_dtype);
    };
    return DispatchOp<3, 3>(std::move(op_builder),
                            {input_gates, hidden_gates, cx},
                            {.out_dtypes = out_dtypes,
                             .out_dims_list = out_dims_list,
                             .op_param_cache_keys = std::move(param_keys)});
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenThnnFusedLstmCell(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& cx, const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias) {
  TT_KERNEL(
      OpName::kThnnFusedLstmCell, param_keys,
      (input_gates, hidden_gates, cx, input_bias, hidden_bias), {
        TT_ASSIGN_OR_THROW(
            const DeviceBufferRefArray<3> result_buffers,
            ThnnFusedLstmCellImpl(input_gates, hidden_gates, cx, input_bias,
                                  hidden_bias, std::move(param_keys)));
        return {MakeTensor(result_buffers[0]), MakeTensor(result_buffers[1]),
                MakeTensor(result_buffers[2])};
      });
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> AtenThnnFusedLstmCellOut(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& cx, const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias, at::Tensor& out0,
    at::Tensor& out1, at::Tensor& out2) {
  TT_KERNEL(
      OpName::kThnnFusedLstmCellOut, param_keys,
      (input_gates, hidden_gates, cx, input_bias, hidden_bias, out0, out1,
       out2),
      {
        TT_ASSIGN_OR_THROW(
            const DeviceBufferRefArray<3> result_buffers,
            ThnnFusedLstmCellImpl(input_gates, hidden_gates, cx, input_bias,
                                  hidden_bias, std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(result_buffers[0], out0));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(result_buffers[1], out1));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(result_buffers[2], out2));
        return {out0, out1, out2};
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenThnnFusedLstmCellBackwardImpl(const std::optional<at::Tensor>& grad_hy,
                                  const std::optional<at::Tensor>& grad_cy,
                                  const at::Tensor& cx, const at::Tensor& cy,
                                  const at::Tensor& workspace, bool has_bias) {
  TT_KERNEL(
      OpName::kThnnFusedLstmCellBackwardImpl, param_keys,
      (grad_hy, grad_cy, cx, cy, workspace, has_bias), {
        const bool grad_hy_defined = grad_hy.has_value() && grad_hy->defined();
        const bool grad_cy_defined = grad_cy.has_value() && grad_cy->defined();
        const at::Tensor grad_hy_t = grad_hy_defined ? *grad_hy : cy;
        const at::Tensor grad_cy_t = grad_cy_defined ? *grad_cy : cy;
        const int64_t batch = cy.size(0);
        const int64_t hidden = cy.size(1);
        TT_ASSIGN_OR_THROW(const auto out_dtype,
                           ConvertTo<mlir::ElementType>(cy.scalar_type()));
        const std::array<int64_t, 2> gg_dims = {batch, 4 * hidden};
        const std::array<int64_t, 2> gcx_dims = {batch, hidden};
        const std::array<mlir::ElementType, 2> out_dtypes = {out_dtype,
                                                             out_dtype};
        const std::array<absl::Span<const int64_t>, 2> out_dims_list = {
            gg_dims, gcx_dims};

        auto op_builder = [batch, hidden, out_dtype, grad_hy_defined,
                           grad_cy_defined](
                              FixedSizeSpan<mlir::MlirOp, 5> inputs) {
          auto& [ghy_in, gcy_in, c, y, ws] = inputs;
          const mlir::MlirOp ghy =
              grad_hy_defined ? ghy_in : MakeConstantLike(y, 0.0, out_dtype);
          const mlir::MlirOp gcy =
              grad_cy_defined ? gcy_in : MakeConstantLike(y, 0.0, out_dtype);
          return BuildLstmCellBackwardShlo(ghy, gcy, c, y, ws, batch, hidden,
                                           out_dtype);
        };
        TT_ASSIGN_OR_THROW(
            (auto [gg, gcx]),
            (DispatchOp<5, 2>(std::move(op_builder),
                              {grad_hy_t, grad_cy_t, cx, cy, workspace},
                              {.out_dtypes = out_dtypes,
                               .out_dims_list = out_dims_list,
                               .op_param_cache_keys = std::move(param_keys)})));
        at::Tensor grad_gates = MakeTensor(std::move(gg));
        at::Tensor grad_cx = MakeTensor(std::move(gcx));

        at::Tensor grad_bias;  // UNINITIALIZED_TENSOR_OK
        if (has_bias) {
          TT_ASSIGN_OR_THROW(
              grad_bias,
              ApplySumReduction(grad_gates, {0}, ReductionMode::kDropDims,
                                grad_gates.scalar_type()));
        }
        return {grad_gates, grad_cx, grad_bias};
      });
}

}  // namespace torch_tpu
