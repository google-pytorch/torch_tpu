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

#include "csrc/ops/binary_cross_entropy/binary_cross_entropy_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Reduction.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/unary.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

absl::Status ValidateIsFloatingPoint(const at::Tensor& tensor,
                                     std::string_view arg_name) {
  TT_RET_CHECK(tensor.is_floating_point(), error::kInvalidArgument)
      << "expected floating point " << arg_name << ", got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildBinaryCrossEntropyShlo(
    mlir::MlirOp input, mlir::MlirOp target, std::optional<mlir::MlirOp> weight,
    int64_t reduction, mlir::MlirBuilder& builder) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const mlir::Type mlir_type = input_type.getElementType();

  // raw_log_input = log(input)
  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(mlir_type));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp raw_log_input,
                      BuildLogShlo(input, element_type));

  // neg_input = -input
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp neg_input, BuildNegShlo(input));
  // raw_log_1_minus_input = log(1 - input)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp raw_log_1_minus_input,
                      BuildLog1pShlo(neg_input, element_type));

  // min_log = -100.0
  const mlir::MlirOp min_log = MakeScalarConstant(builder, -100.0, mlir_type);

  // log_input = max(log(input), -100.0)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp log_input,
                      BuildMaximumShlo(raw_log_input, min_log));
  // log_1_minus_input = max(log(1 - input), -100.0)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp log_1_minus_input,
                      BuildMaximumShlo(raw_log_1_minus_input, min_log));

  // loss = term1 - term2
  //      = (target - 1) * log_1_minus_input - target * log_input

  const mlir::MlirOp one = MakeScalarConstant(builder, 1.0, mlir_type);
  // target_minus_one = target - 1
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp target_minus_one,
                      BuildSubShlo(target, one));

  // term1 = (target - 1) * log_1_minus_input
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp term1,
                      BuildMulShlo(target_minus_one, log_1_minus_input));
  // term2 = target * log_input
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp term2,
                      BuildMulShlo(target, log_input));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp loss, BuildSubShlo(term1, term2));

  // loss = loss * weight
  if (weight.has_value()) {
    TT_ASSIGN_OR_RETURN(loss, BuildMulShlo(loss, *weight));
  }

  if (reduction == at::Reduction::None) {
    return loss;
  }

  // reduced_loss = sum(loss)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp reduced_loss,
                      BuildSumShlo(loss, GetAllDimensions(loss)));

  if (reduction == at::Reduction::Sum) {
    return reduced_loss;
  }

  if (reduction == at::Reduction::Mean) {
    // numel = count(input)
    const mlir::MlirOp numel = GetNumElements(input, mlir_type);
    // return reduced_loss / numel
    return BuildDivShlo(reduced_loss, numel);
  }

  return TT_ERROR(error::kInvalidArgument)
         << "expected valid reduction mode (Sum, Mean, or None), got "
         << reduction;
}

absl::Status DispatchBinaryCrossEntropyOut(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction, at::Tensor& out,
    OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateIsFloatingPoint(self, "input"));
  TT_RETURN_IF_ERROR(ValidateIsFloatingPoint(target, "target"));
  TT_RET_CHECK(self.sizes() == target.sizes(), error::kInvalidArgument)
      << "expected input and target shapes to match, got " << self.sizes()
      << " vs " << target.sizes();

  const bool has_weight = weight.has_value() && weight->defined();

  auto op_builder =
      [reduction, has_weight](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    const mlir::MlirOp self_op = inputs[0];
    const mlir::MlirOp target_op = inputs[1];
    const std::optional<mlir::MlirOp> weight_op =
        has_weight ? std::make_optional(inputs[2]) : std::nullopt;

    return BuildBinaryCrossEntropyShlo(self_op, target_op, weight_op, reduction,
                                       builder);
  };

  std::vector<at::Tensor> inputs = {self, target};
  if (has_weight) {
    inputs.push_back(*weight);
  }

  TT_ASSIGN_OR_RETURN(const mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  const auto comp_type = ToAccumulateType(self.scalar_type());
  TT_ASSIGN_OR_RETURN(const mlir::ElementType computation_dtype,
                      ConvertTo<mlir::ElementType>(comp_type));

  Dimensions out_dims = (reduction == at::Reduction::None)
                            ? CopyIntVector(self.sizes())
                            : Dimensions();

  DispatchOpOptions<1> options = {
      .out_dtype = output_dtype,
      .out_dims = out_dims,
      .computation_dtype = computation_dtype,
      .op_param_cache_keys = std::move(param_keys),
  };

  return DispatchOpOut<kDynamicSize>(std::move(op_builder), inputs, out,
                                     std::move(options));
}

absl::StatusOr<mlir::MlirOp> BuildBinaryCrossEntropyBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self, mlir::MlirOp target,
    std::optional<mlir::MlirOp> weight, int64_t reduction,
    mlir::MlirBuilder& builder) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::Type mlir_type = self_type.getElementType();

  // denom = (1 - x) * x
  const mlir::MlirOp one = MakeScalarConstant(builder, 1.0, mlir_type);
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp one_minus_self,
                      BuildSubShlo(one, self));

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp denom_raw,
                      BuildMulShlo(one_minus_self, self));

  // denom = max((1 - x) * x, epsilon)
  const mlir::MlirOp epsilon = MakeScalarConstant(builder, 1e-12, mlir_type);
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp denom,
                      BuildMaximumShlo(denom_raw, epsilon));

  // num = grad_output * (x - y)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp self_minus_target,
                      BuildSubShlo(self, target));

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp num,
                      BuildMulShlo(grad_output, self_minus_target));

  // grad_input = num / denom
  TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_input, BuildDivShlo(num, denom));

  // grad_input = grad_input * weight
  if (weight.has_value()) {
    TT_ASSIGN_OR_RETURN(grad_input, BuildMulShlo(grad_input, *weight));
  }

  // grad_input = grad_input / numel
  if (reduction == at::Reduction::Mean) {
    const mlir::MlirOp numel = GetNumElements(self, mlir_type);
    TT_ASSIGN_OR_RETURN(grad_input, BuildDivShlo(grad_input, numel));
  }

  return grad_input;
}

absl::Status DispatchBinaryCrossEntropyBackwardOut(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const std::optional<at::Tensor>& weight,
    int64_t reduction, at::Tensor& grad_input, OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateIsFloatingPoint(grad_output, "grad_output"));
  TT_RETURN_IF_ERROR(ValidateIsFloatingPoint(self, "input"));
  TT_RETURN_IF_ERROR(ValidateIsFloatingPoint(target, "target"));
  TT_RET_CHECK(self.sizes() == target.sizes(), error::kInvalidArgument)
      << "expected input and target shapes to match, got " << self.sizes()
      << " vs " << target.sizes();

  TT_RET_CHECK(at::is_expandable_to(grad_output.sizes(), self.sizes()),
               error::kInvalidArgument)
      << "expected grad_output to be broadcastable to input shape, got "
      << grad_output.sizes() << " vs " << self.sizes();

  const bool has_weight = weight.has_value() && weight->defined();

  auto op_builder =
      [reduction, has_weight](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    const mlir::MlirOp grad_output_op = inputs[0];
    const mlir::MlirOp self_op = inputs[1];
    const mlir::MlirOp target_op = inputs[2];
    const std::optional<mlir::MlirOp> weight_op =
        has_weight ? std::make_optional(inputs[3]) : std::nullopt;

    return BuildBinaryCrossEntropyBackwardShlo(
        grad_output_op, self_op, target_op, weight_op, reduction, builder);
  };

  std::vector<at::Tensor> inputs = {grad_output, self, target};
  if (has_weight) {
    inputs.push_back(*weight);
  }

  TT_ASSIGN_OR_RETURN(const mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  const auto comp_type = ToAccumulateType(self.scalar_type());
  TT_ASSIGN_OR_RETURN(const mlir::ElementType computation_dtype,
                      ConvertTo<mlir::ElementType>(comp_type));

  Dimensions out_dims = CopyIntVector(self.sizes());

  DispatchOpOptions<1> options = {
      .out_dtype = output_dtype,
      .out_dims = out_dims,
      .computation_dtype = computation_dtype,
      .op_param_cache_keys = std::move(param_keys),
  };

  return DispatchOpOut<kDynamicSize>(std::move(op_builder), inputs, grad_input,
                                     std::move(options));
}

}  // namespace

at::Tensor AtenBinaryCrossEntropy(const at::Tensor& self,
                                  const at::Tensor& target,
                                  const std::optional<at::Tensor>& weight,
                                  int64_t reduction) {
  TT_KERNEL(
      OpName::kBinaryCrossEntropy, param_keys,
      (self, target, weight, reduction), {
        TT_ASSIGN_OR_THROW(
            at::Tensor out,
            MakeEmptyTensor(/*size=*/{0}, self.scalar_type(), self.device()));
        TT_THROW_IF_ERROR(DispatchBinaryCrossEntropyOut(
            self, target, weight, reduction, out, std::move(param_keys)));
        return out;
      });
}

at::Tensor& AtenBinaryCrossEntropyOut(const at::Tensor& self,
                                      const at::Tensor& target,
                                      const std::optional<at::Tensor>& weight,
                                      int64_t reduction, at::Tensor& out) {
  TT_KERNEL(OpName::kBinaryCrossEntropyOut, param_keys,
            (self, target, weight, reduction, out), {
              TT_THROW_IF_ERROR(DispatchBinaryCrossEntropyOut(
                  self, target, weight, reduction, out, std::move(param_keys)));
              return out;
            });
}

at::Tensor AtenBinaryCrossEntropyBackward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const std::optional<at::Tensor>& weight,
    int64_t reduction) {
  TT_KERNEL(
      OpName::kBinaryCrossEntropyBackward, param_keys,
      (grad_output, self, target, weight, reduction), {
        TT_ASSIGN_OR_THROW(
            at::Tensor grad_input,
            MakeEmptyTensor(/*size=*/{0}, self.scalar_type(), self.device()));
        TT_THROW_IF_ERROR(DispatchBinaryCrossEntropyBackwardOut(
            grad_output, self, target, weight, reduction, grad_input,
            std::move(param_keys)));
        return grad_input;
      });
}

at::Tensor& AtenBinaryCrossEntropyBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const std::optional<at::Tensor>& weight,
    int64_t reduction, at::Tensor& grad_input) {
  TT_KERNEL(OpName::kBinaryCrossEntropyBackwardGradInput, param_keys,
            (grad_output, self, target, weight, reduction, grad_input), {
              TT_CHECK_THROW(grad_input.scalar_type() == self.scalar_type(),
                             error::kInvalidArgument)
                  << "expected grad_input dtype "
                  << ToString(self.scalar_type()) << ", got "
                  << ToString(grad_input.scalar_type());

              TT_THROW_IF_ERROR(DispatchBinaryCrossEntropyBackwardOut(
                  grad_output, self, target, weight, reduction, grad_input,
                  std::move(param_keys)));
              return grad_input;
            });
}

}  // namespace torch_tpu
