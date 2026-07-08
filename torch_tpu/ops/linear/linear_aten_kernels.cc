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

#include "torch_tpu/ops/linear/linear_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/matmul.h"
#include "ATen/ops/result_type.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/precision_context.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildLinearShlo(
    mlir::MlirOp input, mlir::MlirOp weight, std::optional<mlir::MlirOp> bias,
    mlir::stablehlo::Precision precision) {
  namespace stablehlo = mlir::stablehlo;

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const mlir::RankedTensorType weight_type = GetTensorTypeOrDie(weight);

  const int64_t input_rank = input_type.getRank();
  const int64_t weight_rank = weight_type.getRank();

  TT_RET_CHECK(input_rank >= 1, error::kInvalidArgument)
      << "BuildLinearShlo: input rank must be >= 1";
  TT_RET_CHECK(weight_rank == 2, error::kInvalidArgument)
      << "BuildLinearShlo: weight rank must be 2";

  mlir::MLIRContext& ctx = input.getContext();

  // Convert i64 inputs to f64 since XLA/StableHLO doesn't support i64 dot
  // generals.
  const bool is_any_i64 = input_type.getElementType().isInteger(64) ||
                          weight_type.getElementType().isInteger(64);
  if (is_any_i64) {
    input = stablehlo::ConvertElementType(input, mlir::ElementType::F64);
    weight = stablehlo::ConvertElementType(weight, mlir::ElementType::F64);
  }

  // Contracting dims: last dim of input and second dim of weight.
  int64_t lhs_contracting_dimensions[1] = {input_rank - 1};
  int64_t rhs_contracting_dimensions[1] = {1};

  stablehlo::DotDimensionNumbersAttr dot_dimension_numbers =
      stablehlo::DotDimensionNumbersAttr::get(
          &ctx, /*lhs_batching_dimensions=*/{}, /*rhs_batching_dimensions=*/{},
          lhs_contracting_dimensions, rhs_contracting_dimensions);

  stablehlo::Precision precisions[2] = {precision, precision};
  auto precision_config_attr =
      stablehlo::PrecisionConfigAttr::get(&ctx, precisions);

  mlir::MlirOp dot_result = stablehlo::DotGeneral(
      input, weight, dot_dimension_numbers, precision_config_attr);

  if (is_any_i64) {
    dot_result =
        stablehlo::ConvertElementType(dot_result, input_type.getElementType());
  }

  if (bias.has_value()) {
    mlir::MlirOp bias_op = *bias;
    if (is_any_i64) {
      bias_op =
          stablehlo::ConvertElementType(bias_op, input_type.getElementType());
    }
    TT_ASSIGN_OR_RETURN((auto [bcast_bias, bcast_dot_result]),
                        ApplyBroadcastIfNeeded(bias_op, dot_result));
    return stablehlo::Add(bcast_bias, bcast_dot_result);
  }

  return dot_result;
}

absl::Status CheckLinearInputs(const at::Tensor& input,
                               const at::Tensor& weight,
                               const std::optional<at::Tensor>& bias) {
  TT_RET_CHECK(input.dim() >= 1, error::kInvalidArgument)
      << "expected input rank to be >= 1, got " << input.dim();
  TT_RET_CHECK(weight.dim() == 2, error::kInvalidArgument)
      << "expected weight to be a 2D tensor, got " << weight.dim() << "D";
  TT_RET_CHECK(input.size(input.dim() - 1) == weight.size(1),
               error::kInvalidArgument)
      << "expected last dimension of input to match weight dim 1, got "
      << input.size(input.dim() - 1) << " and " << weight.size(1);
  if (bias.has_value() && bias->defined()) {
    TT_RET_CHECK(bias->dim() == 1, error::kInvalidArgument)
        << "expected bias to be a 1D tensor, got " << bias->dim() << "D";
    TT_RET_CHECK(bias->size(0) == weight.size(0), error::kInvalidArgument)
        << "expected bias size 0 to match weight size 0, got " << bias->size(0)
        << " and " << weight.size(0);
  }
  return absl::OkStatus();
}

absl::StatusOr<DeviceBufferRef> Linear(const at::Tensor& input,
                                       const at::Tensor& weight,
                                       const std::optional<at::Tensor>& bias,
                                       OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(CheckLinearInputs(input, weight, bias));

  at::ScalarType target_scalar_type = at::result_type(input, weight);
  if (bias.has_value() && bias->defined()) {
    target_scalar_type =
        c10::promoteTypes(target_scalar_type, bias->scalar_type());
  }
  TT_ASSIGN_OR_RETURN(auto target_elem_dtype,
                      ConvertTo<mlir::ElementType>(target_scalar_type));

  const auto current_precision = GetAndAddPrecisionTo(param_keys);

  const int64_t input_rank = input.dim();
  Dimensions output_dims(input_rank);
  for (int64_t i = 0; i < input_rank - 1; ++i) {
    output_dims[i] = input.size(i);
  }
  output_dims[input_rank - 1] = weight.size(0);

  if (bias.has_value() && bias->defined()) {
    auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      return BuildLinearShlo(inputs[0], inputs[1], inputs[2],
                             current_precision);
    };
    TT_ASSIGN_OR_RETURN(
        auto result,
        DispatchOp<3>(std::move(op_builder), {input, weight, *bias},
                      {.out_dtype = target_elem_dtype,
                       .out_dims = output_dims,
                       .op_param_cache_keys = std::move(param_keys)}));
    return result;
  } else {
    auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      return BuildLinearShlo(inputs[0], inputs[1], std::nullopt,
                             current_precision);
    };
    TT_ASSIGN_OR_RETURN(
        auto result,
        DispatchOp<2>(std::move(op_builder), {input, weight},
                      {.out_dtype = target_elem_dtype,
                       .out_dims = output_dims,
                       .op_param_cache_keys = std::move(param_keys)}));
    return result;
  }
}

}  // namespace

at::Tensor AtenLinear(const at::Tensor& input, const at::Tensor& weight,
                      const std::optional<at::Tensor>& bias) {
  TT_KERNEL(OpName::kLinear, param_keys, (input, weight, bias), {
    const int64_t input_rank = input.dim();
    Dimensions output_dims(input_rank);
    for (int64_t i = 0; i < input_rank - 1; ++i) {
      output_dims[i] = input.size(i);
    }
    output_dims[input_rank - 1] = weight.size(0);

    at::ScalarType target_scalar_type = at::result_type(input, weight);
    if (bias.has_value() && bias->defined()) {
      target_scalar_type =
          c10::promoteTypes(target_scalar_type, bias->scalar_type());
    }

    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        MakeEmptyTensor(output_dims, target_scalar_type, input.device()));

    TT_ASSIGN_OR_THROW(auto result_buf,
                       Linear(input, weight, bias, std::move(param_keys)));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenLinearBackward(
    const at::Tensor& self, const at::Tensor& grad_output,
    const at::Tensor& weight, std::array<bool, 3> output_mask) {
  at::Tensor grad_input;   // UNINITIALIZED_TENSOR_OK
  at::Tensor grad_weight;  // UNINITIALIZED_TENSOR_OK
  at::Tensor grad_bias;    // UNINITIALIZED_TENSOR_OK

  if (output_mask[0]) {
    grad_input = at::matmul(grad_output, weight);
  }
  if (output_mask[1] || output_mask[2]) {
    auto grad_output_flat = grad_output.reshape({-1, grad_output.size(-1)});
    if (output_mask[1]) {
      auto self_flat = self.reshape({-1, self.size(-1)});
      grad_weight = at::matmul(grad_output_flat.t(), self_flat);
    }
    if (output_mask[2]) {
      grad_bias = grad_output_flat.sum(0);
    }
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

}  // namespace torch_tpu
