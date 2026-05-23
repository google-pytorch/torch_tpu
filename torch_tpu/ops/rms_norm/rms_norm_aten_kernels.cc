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

#include "torch_tpu/ops/rms_norm/rms_norm_aten_kernels.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBase.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/layer_norm/layer_norm.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/rms_norm/rms_norm.h"

namespace torch_tpu {

namespace {

absl::Status CheckFusedRmsNormInputs(const at::Tensor& input,
                                     at::IntArrayRef normalized_shape) {
  TT_RET_CHECK(IsFloatingPoint(input), error::kInvalidArgument)
      << "expected the input dtype to be floating point, got "
      << ToString(input.scalar_type());

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches this error in
                 // the caller op `rms_norm()`.
      !normalized_shape.empty(), error::kInvalidArgument)
      << "the normalized shape must have >= 1 dimensions";

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches this error in
                 // the caller op `rms_norm()`.
      normalized_shape.size() <= input.dim(), error::kInvalidArgument)
      << "expected the normalized shape to have <= " << input.dim()
      << " dimensions, got " << normalized_shape.size() << ".";

  return absl::OkStatus();
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> AtenFusedRmsNorm(
    const at::Tensor& input, const at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight, const std::optional<double> eps) {
  // _fused_rms_norm(Tensor input, int[] normalized_shape, Tensor? weight,
  // float? eps) -> (Tensor, Tensor)
  double epsilon = eps.has_value() ? eps.value() : 1e-5;
  TT_KERNEL(
      OpName::kFusedRmsNorm, param_keys,
      (input, normalized_shape, weight, epsilon), {
        TT_THROW_IF_ERROR(CheckFusedRmsNormInputs(input, normalized_shape));

        const size_t normalized_shape_dims = normalized_shape.size();

        auto op_builder = [normalized_shape_dims, epsilon](
                              absl::Span<const mlir::MlirOp> inputs,
                              mlir::MlirBuilder& builder)
            -> absl::StatusOr<MlirOpResults<2>> {
          TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=The inputs vector is always
                         // constructed with this op input, possibly followed by
                         // the weights.
              !inputs.empty() && inputs.size() <= 2, error::kInternal)
              << "expected 1 or 2 inputs, got " << inputs.size();

          mlir::MlirOp input_op = inputs[0];
          std::optional<mlir::MlirOp> weight_op;
          if (inputs.size() > 1) {
            weight_op = inputs[1];
          }

          TT_ASSIGN_OR_RETURN(LayerNormShloResults results,
                              BuildRmsNormShlo(input_op, weight_op,
                                               normalized_shape_dims, epsilon));
          return MlirOpResults<2>{results.normalized_values,
                                  results.reciprocal_std};
        };

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));
        const auto input_sizes = input.sizes();
        Dimensions output_dims = CopyIntVector(input_sizes);

        std::vector<at::Tensor> inputs = {input};
        if (weight.has_value()) {
          inputs.push_back(*weight);
        }

        mlir::ElementType rstd_dtype = output_dtype;
        if (output_dtype != mlir::ElementType::F32 &&
            output_dtype != mlir::ElementType::F64) {
          rstd_dtype = mlir::ElementType::F32;
        }

        const auto elem_dtype = output_dtype;
        const auto rstd_elem_dtype = rstd_dtype;

        const int64_t input_dims = input.dim();

        // The StableHLO reduction op will remove all normalized dimensions, but
        // PyTorch expects the dimensions to be retained as size 1. Return rstd
        // with these singleton dimensions.
        Dimensions rstd_buffer_shape;
        for (int i = 0; i < input_dims - normalized_shape_dims; ++i) {
          rstd_buffer_shape.push_back(input_sizes[i]);
        }
        for (int i = input_dims - normalized_shape_dims; i < input_dims; ++i) {
          rstd_buffer_shape.push_back(1);
        }

        TT_ASSIGN_OR_THROW(
            auto output_bufs,
            (DispatchOp<kDynamicSize, 2>(
                std::move(op_builder), inputs,
                {.out_dtypes = {elem_dtype, rstd_elem_dtype},
                 .out_dims_list = {output_dims, rstd_buffer_shape},
                 .op_param_cache_keys = std::move(param_keys)})));

        return std::make_tuple(MakeTensor(std::move(output_bufs[0])),
                               MakeTensor(std::move(output_bufs[1])));
      });
}

std::tuple<at::Tensor, at::Tensor> AtenFusedRmsNormBackward(
    const at::Tensor& grad_out, const at::Tensor& input,
    at::IntArrayRef normalized_shape, const at::Tensor& rstd,
    const std::optional<at::Tensor>& weight, std::array<bool, 2> output_mask) {
  // _fused_rms_norm_backward(Tensor grad_out, Tensor input, int[]
  // normalized_shape, Tensor rstd, Tensor? weight, bool[2] output_mask) ->
  // (Tensor, Tensor)
  TT_KERNEL(
      OpName::kFusedRmsNormBackward, param_keys,
      (grad_out, input, normalized_shape, rstd, weight, output_mask), {
        // When an optional Tensor (like weight) is None (or nullopt) in the
        // forward pass, PyTorch's autograd engine saves an undefined Tensor for
        // the backward pass. It does not strip the argument from the backward
        // function call; it passes a Tensor object for which .defined() returns
        // false. Thus, it's insufficient to only rely on weight.has_value() to
        // determine presence of the weight Tensor, we must also call
        // weight->defined().
        bool has_weight = false;
        if (weight.has_value() && weight->defined()) {
          has_weight = true;
        }

        Dimensions normalized_shape_vec = CopyIntVector(normalized_shape);
        auto op_builder = [normalized_shape_vec](
                              absl::Span<const mlir::MlirOp> inputs,
                              mlir::MlirBuilder& builder)
            -> absl::StatusOr<MlirOpResults<2>> {
          TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=The inputs sequence is always
                         // constructed with the output grad, this op's input,
                         // the rstd tensor, and possibly followed by the
                         // weights.
              inputs.size() >= 3 && inputs.size() <= 4, error::kInternal)
              << "expected 3 or 4 inputs, got " << inputs.size();

          mlir::MlirOp grad_out_op = inputs[0];
          mlir::MlirOp input_op = inputs[1];
          mlir::MlirOp rstd_op = inputs[2];
          std::optional<mlir::MlirOp> weight_op;
          if (inputs.size() > 3) {
            weight_op = inputs[3];
          }

          TT_ASSIGN_OR_RETURN(
              RmsNormBackwardShloResults results,
              BuildRmsNormBackwardShlo(grad_out_op, input_op, rstd_op,
                                       weight_op, normalized_shape_vec));
          return MlirOpResults<2>{results.grad_input, results.grad_weight};
        };

        const auto input_sizes = input.sizes();
        Dimensions input_dims = CopyIntVector(input_sizes);

        Dimensions weight_dims;
        if (has_weight) {
          weight_dims = CopyIntVector(weight.value().sizes());
        } else {
          weight_dims = CopyIntVector(normalized_shape);
        }

        std::vector<at::Tensor> inputs = {grad_out, input, rstd};
        if (has_weight) {
          inputs.push_back(weight.value());
        }

        TT_ASSIGN_OR_THROW(const auto elem_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto output_bufs,
            (DispatchOp<kDynamicSize, 2>(
                std::move(op_builder), inputs,
                {.out_dtypes = {elem_dtype, elem_dtype},
                 .out_dims_list = {input_dims, weight_dims},
                 .op_param_cache_keys = std::move(param_keys)})));

        return {output_mask[0] ? MakeTensor(std::move(output_bufs[0]))
                               : at::Tensor(),
                output_mask[1] ? MakeTensor(std::move(output_bufs[1]))
                               : at::Tensor()};
      });
}

}  // namespace torch_tpu
