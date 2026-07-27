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

#include "torch_tpu/ops/layer_norm/layer_norm_aten_kernels.h"

#include <array>
#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBase.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/Optional.h"
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
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

absl::Status CheckInputs(const at::Tensor& input,
                         const at::IntArrayRef normalized_shape) {
  TT_RET_CHECK(IsFloatingPoint(input), error::kInvalidArgument)
      << "expected the input dtype to be floating point, got "
      << ToString(input.scalar_type());

  TT_RET_CHECK(!normalized_shape.empty(), error::kInvalidArgument)
      << "the normalized shape must have >= 1 dimensions";

  TT_RET_CHECK(normalized_shape.size() <= input.dim(), error::kInvalidArgument)
      << "expected the normalized shape to have <= " << input.dim()
      << " dimensions, got " << normalized_shape.size()
      << " dimensions of shape " << ToString(normalized_shape);

  return absl::OkStatus();
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenNativeLayerNorm(
    const at::Tensor& input, const at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt, const double eps) {
  TT_KERNEL(
      OpName::kLayerNorm, param_keys,
      (input, normalized_shape, weight_opt, bias_opt, eps), {
        TT_THROW_IF_ERROR(CheckInputs(input, normalized_shape));

        const bool has_weight = weight_opt.has_value() && weight_opt->defined();
        const bool has_bias = bias_opt.has_value() && bias_opt->defined();
        const size_t normalized_shape_dims = normalized_shape.size();

        auto op_builder = [has_weight, has_bias, normalized_shape_dims, eps](
                              absl::Span<const mlir::MlirOp> inputs,
                              mlir::MlirBuilder& builder)
            -> absl::StatusOr<MlirOpResults<3>> {
          LayerNormShloResults results;
          std::optional<mlir::MlirOp> weight_op;
          std::optional<mlir::MlirOp> bias_op;
          if (has_weight && has_bias) {
            ABSL_CHECK_EQ(  // CRASH_OK=Input not set correctly by the backend.
                inputs.size(), 3);
            weight_op = inputs[1];
            bias_op = inputs[2];
          } else if (has_weight) {
            ABSL_CHECK_EQ(  // CRASH_OK=Input not set correctly by the backend.
                inputs.size(), 2);
            weight_op = inputs[1];
          } else if (has_bias) {
            ABSL_CHECK_EQ(  // CRASH_OK=Input not set correctly by the backend.
                inputs.size(), 2);
            bias_op = inputs[1];
          }
          auto input_op = inputs[0];
          TT_ASSIGN_OR_RETURN(results,
                              BuildLayerNormShlo(input_op, weight_op, bias_op,
                                                 normalized_shape_dims, eps));

          return {{results.normalized_values, results.mean,
                   results.reciprocal_std}};
        };

        // Get output dtype and output dims.
        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));
        mlir::ElementType stats_dtype = output_dtype;
        if (output_dtype != mlir::ElementType::F32 &&
            output_dtype != mlir::ElementType::F64) {
          stats_dtype = mlir::ElementType::F32;
        }

        const at::IntArrayRef input_sizes = input.sizes();
        const size_t input_dims = input_sizes.size();
        Dimensions output_dims = CopyIntVector(input_sizes);

        // The StableHLO reduction op will remove all normalized dimensions, but
        // PyTorch expects the dimensions to be retained as size 1.
        // We return a DeviceBufferRef without these size-1 dimensions, wrapped
        // in an at::Tensor with them. Future uses will prepend view logic
        // as necessary to get the desired shape.
        Dimensions mean_rstd_buffer_shape;
        Dimensions mean_rstd_tensor_shape;
        mean_rstd_buffer_shape.reserve(input_dims - normalized_shape_dims);
        mean_rstd_tensor_shape.reserve(input_dims);
        for (int i = 0; i < input_dims - normalized_shape_dims; ++i) {
          mean_rstd_buffer_shape.push_back(input_sizes[i]);
          mean_rstd_tensor_shape.push_back(input_sizes[i]);
        }
        for (int i = input_dims - normalized_shape_dims; i < input_dims; ++i) {
          mean_rstd_tensor_shape.push_back(1);
        }

        // Build inputs for the op.
        std::vector<at::Tensor> inputs = {input};
        if (has_weight) {
          inputs.push_back(weight_opt.value());
        }
        if (has_bias) {
          inputs.push_back(bias_opt.value());
        }
        const auto elem_dtype = output_dtype;
        const auto stats_elem_dtype = stats_dtype;
        TT_ASSIGN_OR_THROW(
            (auto [output_buf, mean_buf, rstd_buf]),
            (DispatchOp<kDynamicSize, 3>(
                std::move(op_builder), inputs,
                {.out_dtypes = {elem_dtype, stats_elem_dtype, stats_elem_dtype},
                 .out_dims_list = {output_dims, mean_rstd_buffer_shape,
                                   mean_rstd_buffer_shape},
                 .op_param_cache_keys = std::move(param_keys)})));
        return {
            MakeTensor(std::move(output_buf)),
            MakeTensor(std::move(mean_buf)).reshape(mean_rstd_tensor_shape),
            MakeTensor(std::move(rstd_buf)).reshape(mean_rstd_tensor_shape)};
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenLayerNormBackward(
    const at::Tensor& dY, const at::Tensor& input,
    at::IntArrayRef normalized_shape, const at::Tensor& mean,
    const at::Tensor& rstd, const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& bias_opt,
    std::array<bool, 3> grad_input_mask) {
  TT_KERNEL(
      OpName::kNativeLayerNormBackward, param_keys,
      (dY, input, normalized_shape, mean, rstd, weight_opt, bias_opt,
       grad_input_mask),
      {
        TT_THROW_IF_ERROR(CheckInputs(input, normalized_shape));

        // Alias input to x, for readability.
        const at::Tensor& x = input;

        // Note that bias is not an input to the backward compute.
        // We just need to know if it was present, in order to compute dbeta
        // from dy.
        const bool compute_dbeta = bias_opt.has_value() && bias_opt->defined();
        const bool has_weight = weight_opt.has_value() && weight_opt->defined();

        Dimensions normalized_shape_vec = CopyIntVector(normalized_shape);
        auto op_builder = [has_weight, compute_dbeta, normalized_shape_vec](
                              absl::Span<const mlir::MlirOp> inputs,
                              mlir::MlirBuilder& builder)
            -> absl::StatusOr<MlirOpResults<3>> {
          std::optional<mlir::MlirOp> weight_op;
          auto dy = inputs[0];
          auto x = inputs[1];
          auto mean = inputs[2];
          auto rstd = inputs[3];
          if (has_weight) {
            ABSL_CHECK_EQ(  // CRASH_OK=Input not set correctly by the backend.
                inputs.size(), 5);
            weight_op = inputs[4];
          } else {
            ABSL_CHECK_EQ(  // CRASH_OK=Input not set correctly by the backend.
                inputs.size(), 4);
          }

          TT_ASSIGN_OR_RETURN(
              LayerNormBackwardShloResults results,
              BuildLayerNormBackwardShlo(dy, x, mean, rstd, weight_op,
                                         normalized_shape_vec, compute_dbeta));
          return {{results.grad_input, results.grad_weight, results.grad_bias}};
        };

        // Get output dtype and output dims.
        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(x.scalar_type()));
        const auto elem_dtype = output_dtype;

        Dimensions input_dims = CopyIntVector(x.sizes());
        Dimensions normalized_dims = CopyIntVector(normalized_shape);
        Dimensions weight_grad_dims =
            has_weight ? normalized_dims : Dimensions();
        Dimensions bias_grad_dims =
            compute_dbeta ? normalized_dims : Dimensions();

        // Build inputs for the op.
        std::vector<at::Tensor> inputs = {dY, x, mean, rstd};
        if (has_weight) {
          inputs.push_back(weight_opt.value());
        }

        TT_ASSIGN_OR_THROW(
            (auto [grad_input_buf, grad_weight_buf, grad_bias_buf]),
            (DispatchOp<kDynamicSize, 3>(
                std::move(op_builder), inputs,
                {.out_dtypes = {elem_dtype, elem_dtype, elem_dtype},
                 .out_dims_list = {input_dims, weight_grad_dims,
                                   bias_grad_dims},
                 .op_param_cache_keys = std::move(param_keys)})));

        return {grad_input_mask[0] ? MakeTensor(std::move(grad_input_buf))
                                   : at::Tensor(),
                grad_input_mask[1] && has_weight
                    ? MakeTensor(std::move(grad_weight_buf))
                    : at::Tensor(),
                grad_input_mask[2] && compute_dbeta
                    ? MakeTensor(std::move(grad_bias_buf))
                    : at::Tensor()};
      });
}

}  // namespace torch_tpu
