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
namespace internal {
template <>
inline void CheckScalarInput<OpName::kThresholdOut, UsesScalarInput::kYes>() {}
template <>
inline void
CheckScalarInput<OpName::kThresholdBackwardGradInput, UsesScalarInput::kYes>() {
}
}  // namespace internal

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
  auto zero = MakeScalarConstant(grad_output.getBuilder(), 0.0, out_dtype);
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
  TT_KERNEL(OpName::kThresholdOut, param_keys, (self, threshold, value, out), {
    TT_THROW_IF_ERROR(ValidateThresholdInputs(self));

    TT_ASSIGN_OR_THROW(const auto output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));

    auto op_builder = [threshold, value, output_dtype](
                          mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      auto& builder = input.getBuilder();
      TT_ASSIGN_OR_RETURN(auto threshold_op,
                          MakeConstant(builder, threshold, output_dtype));
      TT_ASSIGN_OR_RETURN(auto value_op,
                          MakeConstant(builder, value, output_dtype));
      return BuildThresholdShlo(input, threshold_op, value_op);
    };

    TT_THROW_IF_ERROR(
        DispatchOpOut<1>(std::move(op_builder), {self}, out,
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
  TT_KERNEL(
      OpName::kThresholdBackwardGradInput, param_keys,
      (grad_output, self, threshold, grad_input), {
        TT_THROW_IF_ERROR(ValidateThresholdInputs(self));

        TT_ASSIGN_OR_THROW(
            const auto output_dtype,
            ConvertTo<mlir::ElementType>(grad_input.scalar_type()));

        auto op_builder = [threshold](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          mlir::MlirOp grad_output = inputs[0];
          mlir::MlirOp self = inputs[1];
          auto& builder = grad_output.getBuilder();
          TT_ASSIGN_OR_RETURN(auto self_dtype, GetElementType(self));
          TT_ASSIGN_OR_RETURN(auto threshold_op,
                              MakeConstant(builder, threshold, self_dtype));
          return BuildThresholdBackwardShlo(grad_output, self, threshold_op);
        };

        TT_THROW_IF_ERROR(DispatchOpOut<2>(
            std::move(op_builder), {grad_output, self}, grad_input,
            {.out_dtype = output_dtype,
             .out_dims = self.sizes(),
             .op_param_cache_keys = std::move(param_keys)}));
        return grad_input;
      });
}

}  // namespace torch_tpu
