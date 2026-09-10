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

#include "csrc/ops/threshold/threshold_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/where/where.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

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

absl::Status ValidateThresholdInputs(const at::Tensor& self) {
  TT_RET_CHECK(!IsBool(self) && !IsComplex(self),
               error::kPythonNotImplementedError)
      << "expected the input dtype to be non-bool and non-complex, got "
      << ToString(self.scalar_type());
  return absl::OkStatus();
}

}  // namespace

at::Tensor& AtenThresholdOut(const at::Tensor& self,
                             const at::Scalar& threshold,
                             const at::Scalar& value, at::Tensor& out) {
  PromotedScalar promoted_threshold = PromoteScalar(threshold);
  PromotedScalar promoted_value = PromoteScalar(value);
  TT_KERNEL(
      OpName::kThresholdOut, param_keys,
      (self, promoted_threshold, promoted_value, out), {
        TT_THROW_IF_ERROR(ValidateThresholdInputs(self));

        TT_ASSIGN_OR_THROW(auto threshold_tensor,
                           promoted_threshold.GetTensor(self.scalar_type()));
        TT_ASSIGN_OR_THROW(auto value_tensor,
                           promoted_value.GetTensor(self.scalar_type()));

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
          return BuildThresholdShlo(inputs[0], inputs[1], inputs[2]);
        };

        TT_THROW_IF_ERROR(DispatchOpOut<3>(
            std::move(op_builder), {self, threshold_tensor, value_tensor}, out,
            {.out_dtype = output_dtype,
             .out_dims = self.sizes(),
             .op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

at::Tensor& AtenThresholdBackwardGradInput(const at::Tensor& grad_output,
                                           const at::Tensor& self,
                                           const at::Scalar& threshold,
                                           at::Tensor& grad_input) {
  PromotedScalar promoted_threshold = PromoteScalar(threshold);
  TT_KERNEL(
      OpName::kThresholdBackwardGradInput, param_keys,
      (grad_output, self, promoted_threshold, grad_input), {
        TT_THROW_IF_ERROR(ValidateThresholdInputs(self));

        TT_ASSIGN_OR_THROW(auto threshold_tensor,
                           promoted_threshold.GetTensor(self.scalar_type()));

        TT_ASSIGN_OR_THROW(
            const auto output_dtype,
            ConvertTo<mlir::ElementType>(grad_input.scalar_type()));

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
          return BuildThresholdBackwardShlo(inputs[0], inputs[1], inputs[2]);
        };

        TT_THROW_IF_ERROR(
            DispatchOpOut<3>(std::move(op_builder),
                             {grad_output, self, threshold_tensor}, grad_input,
                             {.out_dtype = output_dtype,
                              .out_dims = self.sizes(),
                              .op_param_cache_keys = std::move(param_keys)}));
        return grad_input;
      });
}

}  // namespace torch_tpu
