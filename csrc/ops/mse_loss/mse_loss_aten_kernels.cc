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

#include "csrc/ops/mse_loss/mse_loss_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Reduction.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/sum.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildMseLossShlo(mlir::MlirOp self_op,
                                              mlir::MlirOp target_op,
                                              int64_t reduction) {
  TT_ASSIGN_OR_RETURN((auto [self_b, target_b]),
                      ApplyBroadcastIfNeeded(self_op, target_op));
  mlir::MlirOp diff = mlir::stablehlo::Subtract(self_b, target_b);
  mlir::MlirOp sq_diff = mlir::stablehlo::Mul(diff, diff);

  if (reduction == at::Reduction::None) {
    return sq_diff;
  }

  TT_ASSIGN_OR_RETURN(mlir::MlirOp reduced_loss,
                      BuildSumShlo(sq_diff, GetAllDimensions(sq_diff)));

  if (reduction == at::Reduction::Sum) {
    return reduced_loss;
  }

  const mlir::RankedTensorType sq_type = GetTensorTypeOrDie(sq_diff);
  mlir::MlirOp numel = GetNumElements(sq_diff, sq_type.getElementType());
  TT_ASSIGN_OR_RETURN((auto [reduced_b, numel_b]),
                      ApplyBroadcastIfNeeded(reduced_loss, numel));
  return mlir::stablehlo::Div(reduced_b, numel_b);
}

absl::StatusOr<mlir::MlirOp> BuildMseLossBackwardShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_op, mlir::MlirOp target_op,
    int64_t reduction) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::Type mlir_type = self_type.getElementType();
  mlir::MlirBuilder& builder = self_op.getBuilder();

  mlir::MlirOp alpha = MakeScalarConstant(builder, 2.0, mlir_type);
  if (reduction == at::Reduction::Mean) {
    // Compute the scalar norm (2.0 / numel) in at least F32 before casting to
    // `mlir_type` (e.g., bf16/f16) to avoid integer precision loss for large
    // element counts while supporting dynamic shapes via `GetNumElements`.
    const mlir::Type norm_type =
        mlir_type.isF64() ? mlir_type : builder.getOpBuilder().getF32Type();
    mlir::MlirOp two = MakeScalarConstant(builder, 2.0, norm_type);
    mlir::MlirOp numel = GetNumElements(self_op, norm_type);
    mlir::MlirOp norm = mlir::stablehlo::Div(two, numel);
    alpha = (norm_type == mlir_type)
                ? norm
                : mlir::stablehlo::ConvertElementType(norm, mlir_type);
  }

  TT_ASSIGN_OR_RETURN((auto [self_b, target_b]),
                      ApplyBroadcastIfNeeded(self_op, target_op));
  mlir::MlirOp diff = mlir::stablehlo::Subtract(self_b, target_b);
  TT_ASSIGN_OR_RETURN((auto [alpha_b, diff_b]),
                      ApplyBroadcastIfNeeded(alpha, diff));
  mlir::MlirOp grad_raw = mlir::stablehlo::Mul(alpha_b, diff_b);
  TT_ASSIGN_OR_RETURN((auto [grad_raw_b, grad_out_b]),
                      ApplyBroadcastIfNeeded(grad_raw, grad_output_op));
  return mlir::stablehlo::Mul(grad_raw_b, grad_out_b);
}

}  // namespace

at::Tensor& AtenMseLossOut(const at::Tensor& self, const at::Tensor& target,
                           int64_t reduction, at::Tensor& out) {
  TT_KERNEL(OpName::kMseLossOut, param_keys, (self, target, reduction, out), {
    TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Byte &&
                       self.scalar_type() != at::ScalarType::Char &&
                       self.scalar_type() != at::ScalarType::Short &&
                       self.scalar_type() != at::ScalarType::Int &&
                       self.scalar_type() != at::ScalarType::Long &&
                       self.scalar_type() != at::ScalarType::ComplexFloat,
                   error::kInvalidArgument)
        << "expected the input dtype to be none of (uint8, int8, int16, "
           "int32, int64, complex64), got "
        << ToString(self.scalar_type());
    TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
        << "expected the input dtype to be a floating-point type, got "
        << ToString(self.scalar_type());
    TT_CHECK_THROW(self.scalar_type() == target.scalar_type(),
                   error::kInvalidArgument)
        << "expected input and target to have the same dtype, got "
        << ToString(self.scalar_type()) << " and "
        << ToString(target.scalar_type());
    TT_CHECK_THROW(reduction == at::Reduction::None ||
                       reduction == at::Reduction::Mean ||
                       reduction == at::Reduction::Sum,
                   error::kInvalidArgument)
        << "unrecognized reduction mode " << reduction;

    TT_ASSIGN_OR_THROW(Dimensions bcast_shape, InferSize(self, target));
    Dimensions out_dims = (reduction == at::Reduction::None)
                              ? std::move(bcast_shape)
                              : Dimensions();

    TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    const auto comp_type = ToAccumulateType(self.scalar_type());
    TT_ASSIGN_OR_THROW(const mlir::ElementType computation_dtype,
                       ConvertTo<mlir::ElementType>(comp_type));

    auto op_builder = [reduction](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [self_op, target_op] = inputs;
      return BuildMseLossShlo(self_op, target_op, reduction);
    };

    DispatchOpOptions<1> options = {
        .out_dtype = out_dtype,
        .out_dims = out_dims,
        .computation_dtype = computation_dtype,
        .op_param_cache_keys = std::move(param_keys),
    };
    TT_THROW_IF_ERROR(DispatchOpOut<2>(std::move(op_builder), {self, target},
                                       out, std::move(options)));
    return out;
  });
}

at::Tensor AtenMseLossBackward(const at::Tensor& grad_output,
                               const at::Tensor& self, const at::Tensor& target,
                               int64_t reduction) {
  TT_KERNEL(
      OpName::kMseLossBackward, param_keys,
      (grad_output, self, target, reduction), {
        TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
            << "expected the input dtype to be a floating-point type, got "
            << ToString(self.scalar_type());

        TT_ASSIGN_OR_THROW(Dimensions out_dims,
                           InferSize(self, target, grad_output));

        TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto op_builder = [reduction](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output_op, self_op, target_op] = inputs;
          return BuildMseLossBackwardShlo(grad_output_op, self_op, target_op,
                                          reduction);
        };

        // `.computation_dtype` is intentionally omitted here because backward
        // `mse_loss` is a pointwise operation (`alpha * (self - target) *
        // grad_output`) rather than a reduction, so it executes in the input
        // dtype.
        DispatchOpOptions<1> options = {
            .out_dtype = out_dtype,
            .out_dims = out_dims,
            .op_param_cache_keys = std::move(param_keys),
        };
        TT_ASSIGN_OR_THROW(
            auto result_buf,
            (DispatchOp<3>(std::move(op_builder), {grad_output, self, target},
                           std::move(options))));
        return MakeTensor(result_buf);
      });
}

}  // namespace torch_tpu
