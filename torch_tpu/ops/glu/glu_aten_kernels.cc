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
#include "torch_tpu/ops/glu/glu_aten_kernels.h"

#include <cstdint>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/native/Resize.h"
#include "absl/functional/bind_front.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildGluShlo(int64_t dim, mlir::MlirOp input) {
  mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  auto shape = input_type.getShape();
  const int64_t rank = shape.size();

  Indices start_indices(rank, 0);
  Indices limit_indices(shape.begin(), shape.end());
  Strides strides(rank, 1);

  const int64_t split_index = shape[dim] / 2;

  // First half
  limit_indices[dim] = split_index;
  mlir::MlirOp first_half =
      mlir::stablehlo::Slice(input, start_indices, limit_indices, strides);

  // Second half
  start_indices[dim] = split_index;
  limit_indices[dim] = shape[dim];
  mlir::MlirOp second_half =
      mlir::stablehlo::Slice(input, start_indices, limit_indices, strides);

  mlir::MlirOp neg = mlir::stablehlo::Neg(second_half);
  mlir::MlirOp exp = mlir::stablehlo::Exp(neg);

  mlir::MlirOp one = MakeConstantLike(second_half, 1.0);
  mlir::MlirOp add = mlir::stablehlo::Add(one, exp);
  mlir::MlirOp div = mlir::stablehlo::Div(one, add);

  mlir::MlirOp result = mlir::stablehlo::Mul(first_half, div);
  return result;
}

absl::Status CheckIsFloatingPoint(const at::Tensor& tensor,
                                  const std::string_view name) {
  TT_RET_CHECK(IsFloatingPoint(tensor), error::kInvalidArgument)
      << "expected the " << name << " dtype to be floating point, got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

// Builds the backward pass for Gated Linear Unit
//
// Mathematical derivation:
// Let input X (self) be split along dim into X1 and X2.
// The forward pass is: Y = GLU(X) = X1 * sigmoid(X2)
// Given gradient dY (grad_output), we compute gradients dX1 and dX2:
//   dX1 = dY * sigmoid(X2)
//   dX2 = dY * X1 * d(sigmoid(X2))/dX2
//       = dY * X1 * sigmoid(X2) * (1.0 - sigmoid(X2))
// Finally, dX is assembled by padding and adding dX1 & dX2 to the full shape.
absl::StatusOr<mlir::MlirOp> BuildGluBackwardShlo(int64_t dim,
                                                  mlir::MlirOp grad_output,
                                                  mlir::MlirOp self) {
  mlir::MlirBuilder& builder = self.getBuilder();
  mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  auto shape = self_type.getShape();
  const int64_t rank = shape.size();

  Indices start_indices(rank, 0);
  Indices limit_indices(shape.begin(), shape.end());
  Strides strides(rank, 1);

  const int64_t split_index = shape[dim] / 2;

  limit_indices[dim] = split_index;
  mlir::MlirOp first_half =
      mlir::stablehlo::Slice(self, start_indices, limit_indices, strides);

  start_indices[dim] = split_index;
  limit_indices[dim] = shape[dim];
  mlir::MlirOp second_half =
      mlir::stablehlo::Slice(self, start_indices, limit_indices, strides);

  mlir::MlirOp sigmoid_second_half = mlir::stablehlo::Logistic(second_half);

  // dX1 = dY * sigmoid(X2)
  mlir::MlirOp dx1 = mlir::stablehlo::Mul(grad_output, sigmoid_second_half);

  // 1.0 - sigmoid(X2)
  mlir::MlirOp one = MakeConstantLike(second_half, 1.0);
  mlir::MlirOp one_minus_sigmoid =
      mlir::stablehlo::Subtract(one, sigmoid_second_half);

  // sigmoid_deriv = sigmoid(X2) * (1.0 - sigmoid(X2))
  mlir::MlirOp sigmoid_deriv =
      mlir::stablehlo::Mul(sigmoid_second_half, one_minus_sigmoid);

  // dX2 = dY * X1 * sigmoid_deriv
  mlir::MlirOp grad_out_first_half =
      mlir::stablehlo::Mul(grad_output, first_half);
  mlir::MlirOp dx2 = mlir::stablehlo::Mul(grad_out_first_half, sigmoid_deriv);

  // Assemble dX by padding dX1 and dX2 to the full shape and adding them.
  Indices interior_padding(rank, 0);
  Indices low_padding_dx1(rank, 0);
  Indices high_padding_dx1(rank, 0);
  high_padding_dx1[dim] = split_index;

  Indices low_padding_dx2(rank, 0);
  Indices high_padding_dx2(rank, 0);
  low_padding_dx2[dim] = split_index;

  mlir::MlirOp zero =
      MakeScalarConstant(builder, 0.0, self_type.getElementType());
  mlir::MlirOp padded_dx1 = mlir::stablehlo::Pad(
      dx1, zero, low_padding_dx1, high_padding_dx1, interior_padding);
  mlir::MlirOp padded_dx2 = mlir::stablehlo::Pad(
      dx2, zero, low_padding_dx2, high_padding_dx2, interior_padding);

  mlir::MlirOp result = mlir::stablehlo::Add(padded_dx1, padded_dx2);
  return result;
}

absl::StatusOr<DeviceBufferRef> BuildGluBackwardBuffer(
    const at::Tensor& grad_output, const at::Tensor& self, int64_t dim,
    OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self.sizes().size()));

  TT_RETURN_IF_ERROR(CheckIsFloatingPoint(self, /* name= */ "self"));
  TT_RETURN_IF_ERROR(
      CheckIsFloatingPoint(grad_output, /* name= */ "grad_output"));
  TT_RET_CHECK(self.scalar_type() == grad_output.scalar_type(),
               error::kInvalidArgument)
      << "expected self and grad_output to have the same dtype, got "
      << ToString(self.scalar_type()) << " and "
      << ToString(grad_output.scalar_type());

  const auto& self_shape = self.sizes();
  TT_RET_CHECK(!self_shape.empty(), error::kInvalidArgument)
      << "expected self to have at least 1 dimension, got 0";
  TT_RET_CHECK(self_shape[dim] % 2 == 0, error::kInvalidArgument)
      << "expected the size of dimension " << dim << " of self to be even, got "
      << self_shape[dim];

  auto expected_grad_output_shape = CopyIntVector(self_shape);
  expected_grad_output_shape[dim] /= 2;
  TT_RET_CHECK(grad_output.sizes() == expected_grad_output_shape,
               error::kInvalidArgument)
      << "expected grad_output shape to be "
      << ToString(expected_grad_output_shape) << ", got "
      << ToString(grad_output.sizes());

  TT_ASSIGN_OR_RETURN(const auto self_mlir_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  auto op_builder = [dim](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [grad_output_op, self_op] = inputs;
    return BuildGluBackwardShlo(dim, grad_output_op, self_op);
  };

  return DispatchOp<2>(std::move(op_builder), {grad_output, self},
                       {.out_dtype = self_mlir_type,
                        .out_dims = self_shape,
                        .op_param_cache_keys = std::move(param_keys)});
}
}  // namespace

at::Tensor& AtenGluOut(const at::Tensor& self, int64_t dim, at::Tensor& out) {
  TT_KERNEL(OpName::kGluOut, params_key, (self, dim, out), {
    TT_ASSIGN_OR_THROW(dim, SafeWrapDim(dim, self.sizes().size()));
    TT_THROW_IF_ERROR(CheckIsFloatingPoint(self, /* name= */ "self"));
    TT_THROW_IF_ERROR(CheckIsFloatingPoint(out, /* name= */ "out"));

    const auto& shape = self.sizes();
    TT_CHECK_THROW(!shape.empty(), error::kInvalidArgument)
        << "expected input tensor to have at least 1 dimension, got 0 "
           "dimensions";
    TT_CHECK_THROW(self.sizes()[dim] % 2 == 0, error::kInvalidArgument)
        << "expected the size of dimension " << dim << " to be even, got "
        << shape[dim];

    auto out_dims = CopyIntVector(shape);
    out_dims[dim] = out_dims[dim] / 2;

    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out, absl::bind_front(&BuildGluShlo, dim),
        {.op_param_cache_keys = std::move(params_key), .out_dims = out_dims}));
    return out;
  });
}

at::Tensor AtenGluBackward(const at::Tensor& grad_output,
                           const at::Tensor& self, int64_t dim) {
  TT_KERNEL(OpName::kGluBackward, params_key, (grad_output, self, dim), {
    TT_ASSIGN_OR_THROW(
        auto result_buf,
        BuildGluBackwardBuffer(grad_output, self, dim, std::move(params_key)));
    return MakeTensor(std::move(result_buf));
  });
}

at::Tensor& AtenGluBackwardGradInput(const at::Tensor& grad_output,
                                     const at::Tensor& self, int64_t dim,
                                     at::Tensor& grad_input) {
  TT_KERNEL(OpName::kGluBackwardGradInput, params_key,
            (grad_output, self, dim, grad_input), {
              TT_THROW_IF_ERROR(
                  CheckIsFloatingPoint(grad_input, /* name= */ "grad_input"));
              TT_CHECK_THROW(self.scalar_type() == grad_input.scalar_type(),
                             error::kInvalidArgument)
                  << "expected self and grad_input to have the same dtype, got "
                  << ToString(self.scalar_type()) << " and "
                  << ToString(grad_input.scalar_type());

              const auto& self_shape = self.sizes();
              at::native::resize_output(grad_input, self_shape);

              TT_ASSIGN_OR_THROW(auto result_buf,
                                 BuildGluBackwardBuffer(grad_output, self, dim,
                                                        std::move(params_key)));
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buf), grad_input));
              return grad_input;
            });
}

}  // namespace torch_tpu
