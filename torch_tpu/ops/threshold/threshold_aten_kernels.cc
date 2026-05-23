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

#include "torch_tpu/ops/threshold/threshold_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/scalar_tensor.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/where/where.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildThresholdShlo(mlir::MlirOp input,
                                                mlir::MlirOp threshold,
                                                mlir::MlirOp value) {
  TT_ASSIGN_OR_RETURN(auto condition, BuildLeShlo(input, threshold));
  TT_ASSIGN_OR_RETURN(auto out_dtype, GetElementType(input));
  return BuildWhereShlo(condition, value, input, out_dtype);
}

absl::StatusOr<mlir::MlirOp> BuildThresholdBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self, mlir::MlirOp threshold) {
  TT_ASSIGN_OR_RETURN(auto condition, BuildGtShlo(self, threshold));
  TT_ASSIGN_OR_RETURN(auto out_dtype, GetElementType(grad_output));
  auto zero = MakeConstantLike(grad_output, 0.0, out_dtype);
  return BuildWhereShlo(condition, grad_output, zero, out_dtype);
}

}  // namespace

at::Tensor& AtenThresholdOut(const at::Tensor& self,
                             const at::Scalar& threshold,
                             const at::Scalar& value, at::Tensor& out) {
  TT_KERNEL(OpName::kThresholdOut, param_keys, (self, threshold, value, out), {
    TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Bool,
                   error::kUnimplemented)
        << "threshold is not implemented for bool type";
    TT_CHECK_THROW(!self.is_complex(), error::kUnimplemented)
        << "threshold is not implemented for complex types";
    TT_ASSIGN_OR_THROW(
        auto result_buf,
        (DispatchOp<3>(
            [](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
              return BuildThresholdShlo(inputs[0], inputs[1], inputs[2]);
            },
            {self, at::scalar_tensor(threshold, self.options()),
             at::scalar_tensor(value, self.options())},
            {.out_dtype =
                 ConvertTo<mlir::ElementType>(out.scalar_type()).value(),
             .out_dims = self.sizes(),
             .op_param_cache_keys = std::move(param_keys)})));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

at::Tensor& AtenThresholdBackwardGradInput(const at::Tensor& grad_output,
                                           const at::Tensor& self,
                                           const at::Scalar& threshold,
                                           at::Tensor& grad_input) {
  TT_KERNEL(OpName::kThresholdBackwardGradInput, param_keys,
            (grad_output, self, threshold, grad_input), {
              TT_CHECK_THROW(!IsBool(self), error::kUnimplemented)
                  << "bool input dtype is not yet supported";
              TT_CHECK_THROW(!IsComplex(self), error::kInvalidArgument)
                  << "expected the input dtype not to be complex, got "
                  << ToString(self.scalar_type());
              TT_ASSIGN_OR_THROW(
                  const auto out_dtype,
                  ConvertTo<mlir::ElementType>(grad_input.scalar_type()));
              TT_ASSIGN_OR_THROW(
                  auto result_buf,
                  (DispatchOp<3>(
                      [](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
                        return BuildThresholdBackwardShlo(inputs[0], inputs[1],
                                                          inputs[2]);
                      },
                      {grad_output, self,
                       at::scalar_tensor(threshold, self.options())},
                      {.out_dtype = out_dtype,
                       .out_dims = self.sizes(),
                       .op_param_cache_keys = std::move(param_keys)})));
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buf), grad_input));
              return grad_input;
            });
}

}  // namespace torch_tpu
