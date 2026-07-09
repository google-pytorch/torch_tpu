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

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenThnnFusedLstmCell(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& cx, const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias) {
  const int64_t batch = cx.size(0);
  const int64_t hidden = cx.size(1);
  const bool has_bias = input_bias.has_value() && hidden_bias.has_value();

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

}  // namespace torch_tpu
