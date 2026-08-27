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

#include "torch_tpu/csrc/ops/thnn_fused_gru_cell/thnn_fused_gru_cell_aten_kernels.h"

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

// Builds forward GRU cell recurrence in StableHLO from input/hidden gate
// pre-activations (`ig`, `hg` [B, 3H]) and previous hidden state (`hx` [B, H]).
//
// Gate slicing order along dim 1 of `ig` and `hg` [0..3H):
//   [0, H)   -> i_r, h_r (reset gate pre-activations)
//   [H, 2H)  -> i_z, h_z (update gate pre-activations)
//   [2H, 3H) -> i_n, h_n (new / candidate gate pre-activations)
//
// Recurrence equations implemented:
//   r  = sigmoid(i_r + h_r)
//   z  = sigmoid(i_z + h_z)
//   n  = tanh(i_n + r * h_n)
//   hy = (hx - n) * z + n
//
// Returns (hy, workspace), where `workspace` = concat(r, z, n, hx, h_n) along
// dim 1 [B, 5H] holding intermediate tensors for the fused backward pass.
// Note: All StableHLO builder ops take non-const MlirOp& lvalues, so every
// intermediate is a named variable.
absl::StatusOr<std::array<mlir::MlirOp, 2>> BuildGruCellShlo(
    mlir::MlirOp ig, mlir::MlirOp hg, mlir::MlirOp hx, int64_t batch,
    int64_t hidden, mlir::ElementType out_dtype) {
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

  mlir::MlirOp ig_acc = to_acc(ig);
  mlir::MlirOp hg_acc = to_acc(hg);
  mlir::MlirOp hx_acc = to_acc(hx);

  mlir::MlirOp i_r =
      mlir::stablehlo::Slice(ig_acc, {0, 0}, {batch, hidden}, {1, 1});
  mlir::MlirOp i_z =
      mlir::stablehlo::Slice(ig_acc, {0, hidden}, {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp i_n = mlir::stablehlo::Slice(ig_acc, {0, 2 * hidden},
                                            {batch, 3 * hidden}, {1, 1});
  mlir::MlirOp h_r =
      mlir::stablehlo::Slice(hg_acc, {0, 0}, {batch, hidden}, {1, 1});
  mlir::MlirOp h_z =
      mlir::stablehlo::Slice(hg_acc, {0, hidden}, {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp h_n = mlir::stablehlo::Slice(hg_acc, {0, 2 * hidden},
                                            {batch, 3 * hidden}, {1, 1});

  mlir::MlirOp pre_r = mlir::stablehlo::Add(i_r, h_r);
  mlir::MlirOp resetgate = mlir::stablehlo::Logistic(pre_r);
  mlir::MlirOp pre_z = mlir::stablehlo::Add(i_z, h_z);
  mlir::MlirOp updategate = mlir::stablehlo::Logistic(pre_z);
  mlir::MlirOp r_hn = mlir::stablehlo::Mul(resetgate, h_n);
  mlir::MlirOp pre_n = mlir::stablehlo::Add(i_n, r_hn);
  mlir::MlirOp newgate = mlir::stablehlo::Tanh(pre_n);

  // hy = (hx - n) * z + n
  mlir::MlirOp hx_minus_n = mlir::stablehlo::Subtract(hx_acc, newgate);
  mlir::MlirOp scaled = mlir::stablehlo::Mul(hx_minus_n, updategate);
  mlir::MlirOp hy_acc = mlir::stablehlo::Add(scaled, newgate);

  mlir::MlirOp hy = to_out(hy_acc);
  mlir::MlirOp workspace = mlir::stablehlo::Concatenate(
      ig.getBuilder(),
      {to_out(resetgate), to_out(updategate), to_out(newgate), hx, to_out(h_n)},
      /*dim=*/1);
  return std::array<mlir::MlirOp, 2>{hy, workspace};
}

// Backward. workspace = concat(r, z, n, hx, h_n). Returns
// (grad_input_gates [B,3H], grad_hidden_gates [B,3H], grad_hx [B,H]); the
// input/hidden gate grads differ only in the new-gate slice (h_n carries an
// extra factor r). Bias grads are batch-sums computed by the caller.
absl::StatusOr<std::array<mlir::MlirOp, 3>> BuildGruCellBackwardShlo(
    mlir::MlirOp grad_hy, mlir::MlirOp workspace, int64_t batch, int64_t hidden,
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
  mlir::MlirOp workspace_acc = to_acc(workspace);

  mlir::MlirOp resetgate =
      mlir::stablehlo::Slice(workspace_acc, {0, 0}, {batch, hidden}, {1, 1});
  mlir::MlirOp updategate = mlir::stablehlo::Slice(workspace_acc, {0, hidden},
                                                   {batch, 2 * hidden}, {1, 1});
  mlir::MlirOp newgate = mlir::stablehlo::Slice(workspace_acc, {0, 2 * hidden},
                                                {batch, 3 * hidden}, {1, 1});
  mlir::MlirOp h_n = mlir::stablehlo::Slice(workspace_acc, {0, 4 * hidden},
                                            {batch, 5 * hidden}, {1, 1});
  mlir::MlirOp hx = mlir::stablehlo::Slice(workspace_acc, {0, 3 * hidden},
                                           {batch, 4 * hidden}, {1, 1});

  mlir::MlirOp one = MakeConstantLike(updategate, 1.0);

  // gz = grad_hy * (hx - n);  gn = grad_hy * (1 - z);  ghx = grad_hy * z
  mlir::MlirOp hx_minus_n = mlir::stablehlo::Subtract(hx, newgate);
  mlir::MlirOp d_updategate = mlir::stablehlo::Mul(grad_hy_acc, hx_minus_n);
  mlir::MlirOp one_minus_z = mlir::stablehlo::Subtract(one, updategate);
  mlir::MlirOp d_newgate = mlir::stablehlo::Mul(grad_hy_acc, one_minus_z);
  mlir::MlirOp grad_hx_acc = mlir::stablehlo::Mul(grad_hy_acc, updategate);

  // through the new-gate tanh: gpre_n = gn * (1 - n^2)
  mlir::MlirOp newgate_sq = mlir::stablehlo::Mul(newgate, newgate);
  mlir::MlirOp one_minus_n_sq = mlir::stablehlo::Subtract(one, newgate_sq);
  mlir::MlirOp d_pre_n = mlir::stablehlo::Mul(d_newgate, one_minus_n_sq);

  // reset gate: grad flows through r * h_n
  mlir::MlirOp d_reset = mlir::stablehlo::Mul(d_pre_n, h_n);
  mlir::MlirOp one_minus_r = mlir::stablehlo::Subtract(one, resetgate);
  mlir::MlirOp r_deriv = mlir::stablehlo::Mul(resetgate, one_minus_r);
  mlir::MlirOp d_pre_r = mlir::stablehlo::Mul(d_reset, r_deriv);

  // update gate: gpre_z = gz * z * (1 - z)
  mlir::MlirOp z_deriv = mlir::stablehlo::Mul(updategate, one_minus_z);
  mlir::MlirOp d_pre_z = mlir::stablehlo::Mul(d_updategate, z_deriv);

  // hidden new-gate grad = gpre_n * r (the asymmetry vs. the input new-gate)
  mlir::MlirOp d_h_n = mlir::stablehlo::Mul(d_pre_n, resetgate);

  mlir::MlirOp grad_input_gates = mlir::stablehlo::Concatenate(
      updategate.getBuilder(),
      {to_out(d_pre_r), to_out(d_pre_z), to_out(d_pre_n)}, /*dim=*/1);
  mlir::MlirOp grad_hidden_gates = mlir::stablehlo::Concatenate(
      updategate.getBuilder(),
      {to_out(d_pre_r), to_out(d_pre_z), to_out(d_h_n)}, /*dim=*/1);
  mlir::MlirOp grad_hx = to_out(grad_hx_acc);
  return std::array<mlir::MlirOp, 3>{grad_input_gates, grad_hidden_gates,
                                     grad_hx};
}

absl::StatusOr<DeviceBufferRefArray<2>> ThnnFusedGruCellImpl(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& hx, const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias, OpParamCacheKeys param_keys) {
  TT_RET_CHECK(input_gates.sizes() == hidden_gates.sizes(),
               error::kInvalidArgument)
      << "expected size of argument #1 'input_gates' to match size of "
         "argument #2 'hidden_gates' ("
      << ToString(hidden_gates.sizes()) << "), got "
      << ToString(input_gates.sizes());
  TT_RET_CHECK(input_gates.size(0) == hx.size(0), error::kInvalidArgument)
      << "expected batch size of argument #1 'input_gates' to match batch size "
         "of argument #3 'hx' ("
      << hx.size(0) << "), got " << input_gates.size(0);
  TT_RET_CHECK(input_gates.size(1) == 3 * hx.size(1), error::kInvalidArgument)
      << "expected feature size of argument #1 'input_gates' to match 3 * "
         "feature size of argument #3 'hx' ("
      << 3 * hx.size(1) << "), got " << input_gates.size(1);

  const int64_t batch = hx.size(0);
  const int64_t hidden = hx.size(1);
  auto s_input_bias = SanitizeOptionalTensor(input_bias);
  auto s_hidden_bias = SanitizeOptionalTensor(hidden_bias);
  const bool has_bias = s_input_bias.has_value() && s_hidden_bias.has_value();

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=unsupported dtypes caught at
                        // tensor creation
      const auto out_dtype, ConvertTo<mlir::ElementType>(hx.scalar_type()));
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=unsupported dtypes caught at
                        // tensor creation
      const auto acc_dtype,
      ConvertTo<mlir::ElementType>(ToAccumulateType(hx.scalar_type())));
  const std::array<int64_t, 2> hy_dims = {batch, hidden};
  const std::array<int64_t, 2> ws_dims = {batch, 5 * hidden};
  const std::array<mlir::ElementType, 2> out_dtypes = {out_dtype, out_dtype};
  const std::array<absl::Span<const int64_t>, 2> out_dims_list = {hy_dims,
                                                                  ws_dims};

  if (has_bias) {
    auto op_builder = [batch, hidden, acc_dtype,
                       out_dtype](FixedSizeSpan<mlir::MlirOp, 5> inputs)
        -> absl::StatusOr<std::array<mlir::MlirOp, 2>> {
      auto to_acc = [acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
        return mlir::stablehlo::ConvertElementType(op, acc_dtype);
      };
      auto& [ig, hg, hx_op, ib, hb] = inputs;
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
      return BuildGruCellShlo(ig_b, hg_b, hx_op, batch, hidden, out_dtype);
    };
    return DispatchOp<5, 2>(
        std::move(op_builder),
        {input_gates, hidden_gates, hx, *s_input_bias, *s_hidden_bias},
        {.out_dtypes = out_dtypes,
         .out_dims_list = out_dims_list,
         .op_param_cache_keys = std::move(param_keys)});
  } else {
    auto op_builder = [batch, hidden, acc_dtype,
                       out_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<std::array<mlir::MlirOp, 2>> {
      auto to_acc = [acc_dtype](mlir::MlirOp op) -> mlir::MlirOp {
        return mlir::stablehlo::ConvertElementType(op, acc_dtype);
      };
      auto& [ig, hg, hx_op] = inputs;
      mlir::MlirOp ig_acc = to_acc(ig);
      mlir::MlirOp hg_acc = to_acc(hg);
      return BuildGruCellShlo(ig_acc, hg_acc, hx_op, batch, hidden, out_dtype);
    };
    return DispatchOp<3, 2>(std::move(op_builder),
                            {input_gates, hidden_gates, hx},
                            {.out_dtypes = out_dtypes,
                             .out_dims_list = out_dims_list,
                             .op_param_cache_keys = std::move(param_keys)});
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> AtenThnnFusedGruCell(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& hx, const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias) {
  TT_KERNEL(
      OpName::kThnnFusedGruCell, param_keys,
      (input_gates, hidden_gates, hx, input_bias, hidden_bias), {
        TT_ASSIGN_OR_THROW(
            const DeviceBufferRefArray<2> result_buffers,
            ThnnFusedGruCellImpl(input_gates, hidden_gates, hx, input_bias,
                                 hidden_bias, std::move(param_keys)));
        return {MakeTensor(result_buffers[0]), MakeTensor(result_buffers[1])};
      });
}

std::tuple<at::Tensor&, at::Tensor&> AtenThnnFusedGruCellOut(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& hx, const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias, at::Tensor& out0,
    at::Tensor& out1) {
  TT_KERNEL(
      OpName::kThnnFusedGruCellOut, param_keys,
      (input_gates, hidden_gates, hx, input_bias, hidden_bias, out0, out1), {
        TT_ASSIGN_OR_THROW(
            const DeviceBufferRefArray<2> result_buffers,
            ThnnFusedGruCellImpl(input_gates, hidden_gates, hx, input_bias,
                                 hidden_bias, std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(result_buffers[0], out0));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(result_buffers[1], out1));
        return {out0, out1};
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenThnnFusedGruCellBackward(const at::Tensor& grad_hy,
                             const at::Tensor& workspace, bool has_bias) {
  TT_KERNEL(
      OpName::kThnnFusedGruCellBackward, param_keys,
      (grad_hy, workspace, has_bias), {
        const int64_t batch = grad_hy.size(0);
        const int64_t hidden = grad_hy.size(1);
        TT_ASSIGN_OR_THROW(const auto out_dtype,
                           ConvertTo<mlir::ElementType>(grad_hy.scalar_type()));
        const std::array<int64_t, 2> g_dims = {batch, 3 * hidden};
        const std::array<int64_t, 2> hx_dims = {batch, hidden};
        const std::array<mlir::ElementType, 3> out_dtypes = {
            out_dtype, out_dtype, out_dtype};
        const std::array<absl::Span<const int64_t>, 3> out_dims_list = {
            g_dims, g_dims, hx_dims};

        auto op_builder = [batch, hidden,
                           out_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
          auto& [ghy, ws] = inputs;
          return BuildGruCellBackwardShlo(ghy, ws, batch, hidden, out_dtype);
        };
        TT_ASSIGN_OR_THROW(
            (auto [gig, ghg, ghx]),
            (DispatchOp<2, 3>(std::move(op_builder), {grad_hy, workspace},
                              {.out_dtypes = out_dtypes,
                               .out_dims_list = out_dims_list,
                               .op_param_cache_keys = std::move(param_keys)})));
        at::Tensor grad_input_gates = MakeTensor(std::move(gig));
        at::Tensor grad_hidden_gates = MakeTensor(std::move(ghg));
        at::Tensor grad_hx = MakeTensor(std::move(ghx));

        at::Tensor grad_input_bias;   // UNINITIALIZED_TENSOR_OK
        at::Tensor grad_hidden_bias;  // UNINITIALIZED_TENSOR_OK
        if (has_bias) {
          TT_ASSIGN_OR_THROW(
              grad_input_bias,
              ApplySumReduction(grad_input_gates, {0}, ReductionMode::kDropDims,
                                grad_input_gates.scalar_type()));
          TT_ASSIGN_OR_THROW(
              grad_hidden_bias,
              ApplySumReduction(grad_hidden_gates, {0},
                                ReductionMode::kDropDims,
                                grad_hidden_gates.scalar_type()));
        }
        return {grad_input_gates, grad_hidden_gates, grad_hx, grad_input_bias,
                grad_hidden_bias};
      });
}

}  // namespace torch_tpu
