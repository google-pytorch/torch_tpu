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

#include "torch_tpu/csrc/ops/hardtanh/hardtanh_aten_kernels.h"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/common/aten_utils.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"

namespace torch_tpu {
namespace {
absl::StatusOr<mlir::MlirOp> BuildHardtanhShlo(mlir::MlirOp input,
                                               mlir::MlirOp min_val,
                                               mlir::MlirOp max_val) {
  return mlir::stablehlo::Clamp(min_val, input, max_val);
}

absl::StatusOr<mlir::MlirOp> BuildHardtanhBackwardShlo(mlir::MlirOp grad_output,
                                                       mlir::MlirOp self,
                                                       mlir::MlirOp min_val,
                                                       mlir::MlirOp max_val) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp min_val_bcast,
                      BroadcastIfNeeded(min_val, self));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp max_val_bcast,
                      BroadcastIfNeeded(max_val, self));

  mlir::MlirOp gt_min = mlir::stablehlo::Compare(
      self, min_val_bcast, mlir::stablehlo::ComparisonDirection::GT);
  mlir::MlirOp lt_max = mlir::stablehlo::Compare(
      self, max_val_bcast, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp in_range = mlir::stablehlo::And(gt_min, lt_max);
  mlir::MlirOp zero = MakeConstantLike(grad_output, 0.0);
  return mlir::stablehlo::Select(in_range, grad_output, zero);
}

absl::Status CheckHardtanhInputs(const at::Tensor& self) {
  const auto scalar_type = self.scalar_type();

  TT_RET_CHECK(!IsComplex(self), error::kInvalidArgument)
      << "expected the input dtype to be non-complex, got "
      << ToString(scalar_type);
  TT_RET_CHECK(!IsBool(self), error::kInvalidArgument)
      << "expected the input dtype to be non-boolean, got "
      << ToString(scalar_type);
  return absl::OkStatus();
}

// Holds the min and max scalar limits for hardtanh after bound clamping.
struct HardtanhBounds {
  at::Scalar min_val;
  at::Scalar max_val;
};

// Clamps integral scalar limits to the representable range of the target
// integral scalar type T.
//
// If a limit lies strictly outside the representable range as a no-op bound
// (e.g. min_val < lowest for an unsigned type), it is clamped to the type's
// boundary to prevent underflow/overflow during tensor materialization while
// preserving equivalent clamp semantics. Limits exceeding the opposite boundary
// (e.g. min_val > highest) are rejected as invalid arguments.
template <typename T>
  requires std::is_integral_v<T>
absl::StatusOr<HardtanhBounds> ClampIntegralBounds(at::Scalar min_val,
                                                   at::Scalar max_val,
                                                   at::ScalarType scalar_type) {
  const double min_d = min_val.toDouble();
  const double max_d = max_val.toDouble();
  constexpr double lowest =
      static_cast<double>(std::numeric_limits<T>::lowest());
  constexpr double highest = static_cast<double>(std::numeric_limits<T>::max());

  TT_RET_CHECK(min_d <= highest, error::kInvalidArgument)
      << "expected clamp min value to be representable as "
      << ToString(scalar_type) << ", got " << min_d;
  TT_RET_CHECK(max_d >= lowest, error::kInvalidArgument)
      << "expected clamp max value to be representable as "
      << ToString(scalar_type) << ", got " << max_d;

  if (min_d < lowest) {
    min_val =
        at::Scalar(static_cast<int64_t>(std::numeric_limits<T>::lowest()));
  }
  if (max_d > highest) {
    max_val = at::Scalar(static_cast<int64_t>(std::numeric_limits<T>::max()));
  }
  return HardtanhBounds{.min_val = min_val, .max_val = max_val};
}

// Clamps out-of-range bounds with representable range of integral types.
// Floating point types do not require bound clamping or range checks.
absl::StatusOr<HardtanhBounds> ClampHardtanhBounds(at::ScalarType scalar_type,
                                                   at::Scalar min_val,
                                                   at::Scalar max_val) {
  switch (scalar_type) {
    case at::ScalarType::Byte:
      return ClampIntegralBounds<uint8_t>(min_val, max_val, scalar_type);
    case at::ScalarType::Char:
      return ClampIntegralBounds<int8_t>(min_val, max_val, scalar_type);
    case at::ScalarType::Short:
      return ClampIntegralBounds<int16_t>(min_val, max_val, scalar_type);
    case at::ScalarType::Int:
      return ClampIntegralBounds<int32_t>(min_val, max_val, scalar_type);
    case at::ScalarType::Long:
      return ClampIntegralBounds<int64_t>(min_val, max_val, scalar_type);
    default:
      return HardtanhBounds{
          .min_val = min_val,
          .max_val = max_val,
      };
  }
}

// Helper to dispatch the hardtanh computation on the device.
absl::StatusOr<DeviceBufferRef> AtenHardtanhImpl(const at::Tensor& self,
                                                 PromotedScalar& promoted_min,
                                                 PromotedScalar& promoted_max,
                                                 OpParamCacheKeys param_keys) {
  const auto scalar_type = self.scalar_type();
  TT_RETURN_IF_ERROR(CheckHardtanhInputs(self));

  const at::Scalar min_val = promoted_min.scalar();
  const at::Scalar max_val = promoted_max.scalar();
  TT_ASSIGN_OR_RETURN(const HardtanhBounds bounds,
                      ClampHardtanhBounds(scalar_type, min_val, max_val));

  TT_ASSIGN_OR_RETURN(at::Tensor min_tensor,
                      promoted_min.GetTensor(scalar_type));
  TT_ASSIGN_OR_RETURN(at::Tensor max_tensor,
                      promoted_max.GetTensor(scalar_type));

  if (bounds.min_val.toDouble() != min_val.toDouble()) {
    TT_ASSIGN_OR_RETURN(min_tensor, MakeTensor(bounds.min_val, scalar_type));
  }
  if (bounds.max_val.toDouble() != max_val.toDouble()) {
    TT_ASSIGN_OR_RETURN(max_tensor, MakeTensor(bounds.max_val, scalar_type));
  }

  auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, min_op, max_op] = inputs;
    return BuildHardtanhShlo(self_op, min_op, max_op);
  };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(scalar_type));

  return DispatchOp<3>(op_builder, {self, min_tensor, max_tensor},
                       {.out_dtype = output_dtype,
                        .out_dims = CopyIntVector(self.sizes()),
                        .op_param_cache_keys = std::move(param_keys)});
}

// Helper to dispatch the hardtanh_backward computation on the device.
absl::StatusOr<DeviceBufferRef> AtenHardtanhBackwardImpl(
    const at::Tensor& grad_output, const at::Tensor& self,
    PromotedScalar& promoted_min, PromotedScalar& promoted_max,
    OpParamCacheKeys param_keys) {
  auto scalar_type = self.scalar_type();
  TT_RET_CHECK(c10::isFloatingType(scalar_type), error::kInvalidArgument)
      << "expected the input dtype to be floating point, got "
      << ToString(scalar_type);

  TT_ASSIGN_OR_RETURN(at::Tensor min_tensor,
                      promoted_min.GetTensor(scalar_type));
  TT_ASSIGN_OR_RETURN(at::Tensor max_tensor,
                      promoted_max.GetTensor(scalar_type));

  auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 4> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [grad_output_op, self_op, min_op, max_op] = inputs;
    return BuildHardtanhBackwardShlo(grad_output_op, self_op, min_op, max_op);
  };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(scalar_type));

  return DispatchOp<4>(op_builder, {grad_output, self, min_tensor, max_tensor},
                       {.out_dtype = output_dtype,
                        .out_dims = CopyIntVector(self.sizes()),
                        .op_param_cache_keys = std::move(param_keys)});
}
}  // namespace

at::Tensor AtenHardtanh(const at::Tensor& self, const at::Scalar& min_val,
                        const at::Scalar& max_val) {
  PromotedScalar promoted_min = PromoteScalar(min_val);
  PromotedScalar promoted_max = PromoteScalar(max_val);
  TT_KERNEL(OpName::kHardtanh, param_keys, (self, promoted_min, promoted_max), {
    TT_ASSIGN_OR_THROW(auto result_buf,
                       AtenHardtanhImpl(self, promoted_min, promoted_max,
                                        std::move(param_keys)));
    return MakeTensor(std::move(result_buf));
  });
}

at::Tensor& AtenHardtanh_(at::Tensor& self, const at::Scalar& min_val,
                          const at::Scalar& max_val) {
  PromotedScalar promoted_min = PromoteScalar(min_val);
  PromotedScalar promoted_max = PromoteScalar(max_val);
  TT_KERNEL(
      OpName::kHardtanh_, param_keys, (self, promoted_min, promoted_max), {
        TT_ASSIGN_OR_THROW(auto result_buf,
                           AtenHardtanhImpl(self, promoted_min, promoted_max,
                                            std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), self));
        return self;
      });
}

at::Tensor& AtenHardtanhOut(const at::Tensor& self, const at::Scalar& min_val,
                            const at::Scalar& max_val, at::Tensor& out) {
  PromotedScalar promoted_min = PromoteScalar(min_val);
  PromotedScalar promoted_max = PromoteScalar(max_val);
  TT_KERNEL(
      OpName::kHardtanhOut, param_keys, (self, promoted_min, promoted_max, out),
      {
        TT_ASSIGN_OR_THROW(auto result_buf,
                           AtenHardtanhImpl(self, promoted_min, promoted_max,
                                            std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

at::Tensor AtenHardtanhBackward(const at::Tensor& grad_output,
                                const at::Tensor& self,
                                const at::Scalar& min_val,
                                const at::Scalar& max_val) {
  PromotedScalar promoted_min = PromoteScalar(min_val);
  PromotedScalar promoted_max = PromoteScalar(max_val);
  TT_KERNEL(OpName::kHardtanhBackward, param_keys,
            (grad_output, self, promoted_min, promoted_max), {
              TT_ASSIGN_OR_THROW(auto result_buf,
                                 AtenHardtanhBackwardImpl(
                                     grad_output, self, promoted_min,
                                     promoted_max, std::move(param_keys)));
              return MakeTensor(std::move(result_buf));
            });
}

at::Tensor& AtenHardtanhBackwardGradInput(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          const at::Scalar& min_val,
                                          const at::Scalar& max_val,
                                          at::Tensor& grad_input) {
  PromotedScalar promoted_min = PromoteScalar(min_val);
  PromotedScalar promoted_max = PromoteScalar(max_val);
  TT_KERNEL(OpName::kHardtanhBackwardGradInput, param_keys,
            (grad_output, self, promoted_min, promoted_max, grad_input), {
              TT_ASSIGN_OR_THROW(auto result_buf,
                                 AtenHardtanhBackwardImpl(
                                     grad_output, self, promoted_min,
                                     promoted_max, std::move(param_keys)));
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buf), grad_input));
              return grad_input;
            });
}

}  // namespace torch_tpu
