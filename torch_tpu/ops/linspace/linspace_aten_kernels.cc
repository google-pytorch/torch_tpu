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

#include "torch_tpu/ops/linspace/linspace_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "torch/headeronly/core/ScalarType.h"
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
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

mlir::MlirOp PrepareScalar(mlir::MlirOp scalar,
                           const mlir::RankedTensorType output_type) {
  mlir::MlirOp result_scalar = scalar;
  if (GetTensorTypeOrDie(scalar).getElementType() !=
      output_type.getElementType()) {
    result_scalar = mlir::stablehlo::ConvertElementType(
        result_scalar, output_type.getElementType());
  }
  return mlir::stablehlo::BroadcastInDim(output_type, result_scalar, {});
}

mlir::Type GetComputeType(mlir::MlirBuilder& builder, mlir::Type mlir_type) {
  mlir::Type compute_type = builder.getOpBuilder().getF32Type();
  if (mlir::isa<mlir::Float64Type>(mlir_type) ||
      mlir::isa<mlir::IntegerType>(mlir_type)) {
    compute_type = builder.getOpBuilder().getF64Type();
  } else if (auto complex_type = mlir::dyn_cast<mlir::ComplexType>(mlir_type)) {
    if (mlir::isa<mlir::Float64Type>(complex_type.getElementType())) {
      compute_type = complex_type;
    } else {
      compute_type =
          mlir::ComplexType::get(builder.getOpBuilder().getF32Type());
    }
  }
  return compute_type;
}

absl::StatusOr<mlir::MlirOp> BuildLinspaceInterpolation(
    mlir::MlirBuilder& builder, int64_t steps, mlir::MlirOp start_scalar,
    mlir::MlirOp end_scalar, mlir::RankedTensorType compute_tensor_type,
    mlir::Type mlir_type) {
  auto compute_type = compute_tensor_type.getElementType();
  mlir::Type iota_type = compute_type;
  if (auto complex_type = mlir::dyn_cast<mlir::ComplexType>(compute_type)) {
    iota_type = complex_type.getElementType();
  }
  mlir::RankedTensorType iota_tensor_type =
      mlir::RankedTensorType::get({steps}, iota_type);
  auto iota = mlir::stablehlo::Iota(builder, iota_tensor_type, 0);
  TT_ASSIGN_OR_RETURN(auto iota_element_type,
                      ConvertTo<mlir::ElementType>(iota_type));

  mlir::MlirOp iota_real = iota;
  if (compute_type != iota_type) {
    auto zero_tensor = MakeConstantLike(iota_real, 0.0, iota_element_type);
    iota = mlir::stablehlo::Complex(iota_real, zero_tensor);
  }

  // Compute (end - start) / (steps - 1)
  auto diff = mlir::stablehlo::Subtract(end_scalar, start_scalar);
  TT_ASSIGN_OR_RETURN(
      auto steps_minus_one_scalar_real,
      (MakeConstant(builder, at::Scalar(steps - 1), iota_element_type)));

  mlir::MlirOp steps_minus_one_scalar = steps_minus_one_scalar_real;
  if (compute_type != iota_type) {
    TT_ASSIGN_OR_RETURN(
        auto zero_scalar,
        (MakeConstant(builder, at::Scalar(0.0), iota_element_type)));
    steps_minus_one_scalar =
        mlir::stablehlo::Complex(steps_minus_one_scalar_real, zero_scalar);
  }

  auto steps_minus_one =
      PrepareScalar(steps_minus_one_scalar, compute_tensor_type);
  auto step_size = mlir::stablehlo::Div(diff, steps_minus_one);

  // PyTorch casts step to the target dtype immediately for floats.
  // Truncate step_size to output precision (mlir_type) to match PyTorch.
  bool should_truncate = mlir::isa<mlir::FloatType>(mlir_type) ||
                         mlir::isa<mlir::ComplexType>(mlir_type);

  if (should_truncate && compute_type != mlir_type) {
    step_size = mlir::stablehlo::ConvertElementType(step_size, mlir_type);
    step_size = mlir::stablehlo::ConvertElementType(step_size, compute_type);
  }

  // PyTorch uses a halfway point strategy to avoid precision issues
  // Halfway is steps / 2.
  int64_t halfway = steps / 2;

  auto offset = mlir::stablehlo::Mul(iota, step_size);
  if (should_truncate && compute_type != mlir_type) {
    offset = mlir::stablehlo::ConvertElementType(offset, mlir_type);
    offset = mlir::stablehlo::ConvertElementType(offset, compute_type);
  }
  auto interpolated_from_start = mlir::stablehlo::Add(start_scalar, offset);

  // Backward from end: end - step_size * (steps - 1 - iota)
  auto steps_minus_one_tensor =
      PrepareScalar(steps_minus_one_scalar, compute_tensor_type);
  auto reverse_iota = mlir::stablehlo::Subtract(steps_minus_one_tensor, iota);
  auto reverse_offset = mlir::stablehlo::Mul(reverse_iota, step_size);
  if (should_truncate && compute_type != mlir_type) {
    reverse_offset =
        mlir::stablehlo::ConvertElementType(reverse_offset, mlir_type);
    reverse_offset =
        mlir::stablehlo::ConvertElementType(reverse_offset, compute_type);
  }
  auto interpolated_from_end =
      mlir::stablehlo::Subtract(end_scalar, reverse_offset);

  // Select between them based on whether iota < halfway
  TT_ASSIGN_OR_RETURN(
      auto halfway_scalar_real,
      MakeConstant(builder, at::Scalar(halfway), iota_element_type));
  auto halfway_tensor_real =
      PrepareScalar(halfway_scalar_real, iota_tensor_type);
  auto is_first_half = mlir::stablehlo::Compare(
      iota_real, halfway_tensor_real, mlir::stablehlo::ComparisonDirection::LT);

  auto interpolated = mlir::stablehlo::Select(
      is_first_half, interpolated_from_start, interpolated_from_end);

  // For numerical precision, guarantee that the last element is exactly
  // end_scalar
  auto steps_minus_one_tensor_real =
      PrepareScalar(steps_minus_one_scalar_real, iota_tensor_type);
  auto is_last =
      mlir::stablehlo::Compare(iota_real, steps_minus_one_tensor_real,
                               mlir::stablehlo::ComparisonDirection::EQ);
  return mlir::stablehlo::Select(is_last, end_scalar, interpolated);
}

absl::StatusOr<mlir::MlirOp> BuildLinspaceOp(
    mlir::MlirBuilder& builder, int64_t steps, mlir::MlirOp start_op,
    mlir::MlirOp end_op, mlir::ElementType output_mlir_type) {
  if (steps == 0) {
    return MakeZeroSizedTensor(builder, output_mlir_type);
  }

  TT_ASSIGN_OR_RETURN(const mlir::Type mlir_type,
                      GetMlirType(builder.getContext(), output_mlir_type));

  mlir::Type compute_type = GetComputeType(builder, mlir_type);

  mlir::RankedTensorType compute_tensor_type =
      mlir::RankedTensorType::get({steps}, compute_type);

  // PyTorch CPU first casts start and end to output dtype, then converts to
  // compute type.
  mlir::MlirOp start_casted = start_op;
  if (GetTensorTypeOrDie(start_op).getElementType() != mlir_type) {
    start_casted = mlir::stablehlo::ConvertElementType(start_op, mlir_type);
  }
  auto start_scalar =
      mlir::stablehlo::ConvertElementType(start_casted, compute_type);
  start_scalar = PrepareScalar(start_scalar, compute_tensor_type);

  if (steps == 1) {
    if (compute_type != mlir_type) {
      start_scalar =
          mlir::stablehlo::ConvertElementType(start_scalar, mlir_type);
    }
    return start_scalar;
  }

  mlir::MlirOp end_casted = end_op;
  if (GetTensorTypeOrDie(end_op).getElementType() != mlir_type) {
    end_casted = mlir::stablehlo::ConvertElementType(end_op, mlir_type);
  }
  auto end_scalar =
      mlir::stablehlo::ConvertElementType(end_casted, compute_type);
  end_scalar = PrepareScalar(end_scalar, compute_tensor_type);

  TT_ASSIGN_OR_RETURN(
      auto compute_result,
      BuildLinspaceInterpolation(builder, steps, start_scalar, end_scalar,
                                 compute_tensor_type, mlir_type));

  if (compute_type != mlir_type) {
    return mlir::stablehlo::ConvertElementType(compute_result, mlir_type);
  }
  return compute_result;
}

absl::StatusOr<NAryMlirOpBuilder<2>> GetLinspaceOpBuilder(
    int64_t steps, const at::ScalarType output_dtype) {
  TT_ASSIGN_OR_RETURN(const auto output_mlir_type,
                      ConvertTo<mlir::ElementType>(output_dtype));

  return [steps, output_mlir_type](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<mlir::MlirOp> {
    auto& [start_op, end_op] = inputs;
    return BuildLinspaceOp(start_op.getBuilder(), steps, start_op, end_op,
                           output_mlir_type);
  };
}

absl::StatusOr<DeviceBufferRef> LinspaceHelper(
    const at::Tensor& start_tensor, const at::Tensor& end_tensor, int64_t steps,
    at::ScalarType output_scalar_type, OpParamCacheKeys param_keys) {
  TT_RET_CHECK(steps >= 0, error::kInvalidArgument)
      << "expected non-negative steps, got " << steps;

  TT_RET_CHECK(output_scalar_type != at::ScalarType::Bool || steps <= 1,
               error::kInvalidArgument)
      << "expected output dtype to be other than bool, got "
      << output_scalar_type;

  TT_ASSIGN_OR_RETURN(const auto output_mlir_type,
                      ConvertTo<mlir::ElementType>(output_scalar_type));

  TT_RETURN_IF_ERROR(ValidateTensorByteSize({steps}, output_mlir_type));

  TT_ASSIGN_OR_RETURN(auto op_builder,
                      GetLinspaceOpBuilder(steps, output_scalar_type));

  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<2>(std::move(op_builder), {start_tensor, end_tensor},
                    {.out_dtype = output_mlir_type,
                     .out_dims = {steps},
                     .op_param_cache_keys = std::move(param_keys)}));
  return result_buf;
}

}  // namespace

at::Tensor& AtenLinspaceOut(const at::Scalar& start, const at::Scalar& end,
                            int64_t steps, at::Tensor& out) {
  auto promoted_start = PromoteScalar(start);
  auto promoted_end = PromoteScalar(end);
  TT_KERNEL(
      OpName::kLinspaceOut, param_keys,
      (promoted_start, promoted_end, steps, out), {
        TT_ASSIGN_OR_THROW(const at::Tensor start_tensor,
                           promoted_start.GetTensor(out.scalar_type()));
        TT_ASSIGN_OR_THROW(const at::Tensor end_tensor,
                           promoted_end.GetTensor(out.scalar_type()));
        TT_ASSIGN_OR_THROW(
            auto result_buf,
            LinspaceHelper(start_tensor, end_tensor, steps, out.scalar_type(),
                           std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

at::Tensor& AtenLinspaceTensorTensorOut(const at::Tensor& start,
                                        const at::Tensor& end, int64_t steps,
                                        at::Tensor& out) {
  TT_CHECK_THROW(start.dim() == 0 && end.dim() == 0, error::kInvalidArgument)
      << "linspace only supports 0-dimensional start and end tensors, "
         "but got start with "
      << start.dim() << " dimension(s) and end with " << end.dim()
      << " dimension(s).";
  TT_KERNEL(
      OpName::kLinspaceTensorTensorOut, param_keys, (start, end, steps, out), {
        TT_ASSIGN_OR_THROW(auto result_buf,
                           LinspaceHelper(start, end, steps, out.scalar_type(),
                                          std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

at::Tensor& AtenLinspaceTensorScalarOut(const at::Tensor& start,
                                        const at::Scalar& end, int64_t steps,
                                        at::Tensor& out) {
  TT_CHECK_THROW(start.dim() == 0, error::kInvalidArgument)
      << "linspace only supports 0-dimensional start and end tensors, "
         "but got start with "
      << start.dim() << " dimension(s).";
  auto promoted_end = PromoteScalar(end);
  TT_KERNEL(
      OpName::kLinspaceTensorScalarOut, param_keys,
      (start, promoted_end, steps, out), {
        TT_ASSIGN_OR_THROW(const at::Tensor end_tensor,
                           promoted_end.GetTensor(out.scalar_type()));
        TT_ASSIGN_OR_THROW(
            auto result_buf,
            LinspaceHelper(start, end_tensor, steps, out.scalar_type(),
                           std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

at::Tensor& AtenLinspaceScalarTensorOut(const at::Scalar& start,
                                        const at::Tensor& end, int64_t steps,
                                        at::Tensor& out) {
  TT_CHECK_THROW(end.dim() == 0, error::kInvalidArgument)
      << "linspace only supports 0-dimensional start and end tensors, "
         "but got end with "
      << end.dim() << " dimension(s).";
  auto promoted_start = PromoteScalar(start);
  TT_KERNEL(
      OpName::kLinspaceScalarTensorOut, param_keys,
      (promoted_start, end, steps, out), {
        TT_ASSIGN_OR_THROW(const at::Tensor start_tensor,
                           promoted_start.GetTensor(out.scalar_type()));
        TT_ASSIGN_OR_THROW(
            auto result_buf,
            LinspaceHelper(start_tensor, end, steps, out.scalar_type(),
                           std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
