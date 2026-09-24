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

#include "csrc/ops/hardtanh/hardtanh_aten_kernels.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace internal {
template <>
inline void CheckScalarInput<OpName::kHardtanh, UsesScalarInput::kYes>() {}
template <>
inline void CheckScalarInput<OpName::kHardtanhOut, UsesScalarInput::kYes>() {}
template <>
inline void
CheckScalarInput<OpName::kHardtanhBackward, UsesScalarInput::kYes>() {}
template <>
inline void
CheckScalarInput<OpName::kHardtanhBackwardGradInput, UsesScalarInput::kYes>() {}
}  // namespace internal

namespace {
absl::StatusOr<mlir::MlirOp> BuildHardtanhShlo(mlir::MlirOp input,
                                               mlir::MlirOp min_val,
                                               mlir::MlirOp max_val) {
  TT_ASSIGN_OR_RETURN((auto [input_bcast, min_bcast, max_bcast]),
                      ApplyBroadcastIfNeeded(input, min_val, max_val));
  return mlir::stablehlo::Clamp(min_bcast, input_bcast, max_bcast);
}

absl::StatusOr<mlir::MlirOp> BuildHardtanhBackwardShlo(mlir::MlirOp grad_output,
                                                       mlir::MlirOp self,
                                                       mlir::MlirOp min_val,
                                                       mlir::MlirOp max_val) {
  auto& builder = grad_output.getBuilder();
  auto type = GetTensorTypeOrDie(grad_output);
  TT_ASSIGN_OR_RETURN(auto element_type,
                      ConvertTo<mlir::ElementType>(type.getElementType()));
  mlir::MlirOp zero = MakeScalarConstant(builder, 0.0, element_type);

  TT_ASSIGN_OR_RETURN(
      (auto [grad_bcast, self_bcast, min_bcast, max_bcast, zero_bcast]),
      ApplyBroadcastIfNeeded(grad_output, self, min_val, max_val, zero));

  mlir::MlirOp gt_min = mlir::stablehlo::Compare(
      self_bcast, min_bcast, mlir::stablehlo::ComparisonDirection::GT);
  mlir::MlirOp lt_max = mlir::stablehlo::Compare(
      self_bcast, max_bcast, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp in_range = mlir::stablehlo::And(gt_min, lt_max);
  return mlir::stablehlo::Select(in_range, grad_bcast, zero_bcast);
}

absl::Status ValidateHardtanhInputs(const at::Tensor& self) {
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
template <typename ReturnType, typename DispatchFn>
ReturnType DispatchHardtanh(const at::Tensor& self, const at::Scalar& min_val,
                            const at::Scalar& max_val,
                            OpParamCacheKeys param_keys,
                            DispatchFn&& dispatch_fn) {
  const auto scalar_type = self.scalar_type();
  TT_RETURN_IF_ERROR(ValidateHardtanhInputs(self));

  TT_ASSIGN_OR_RETURN(const HardtanhBounds bounds,
                      ClampHardtanhBounds(scalar_type, min_val, max_val));

  auto op_builder = [bounds, scalar_type](
                        mlir::MlirOp self_op) -> absl::StatusOr<mlir::MlirOp> {
    auto& builder = self_op.getBuilder();
    auto type = GetTensorTypeOrDie(self_op);
    TT_ASSIGN_OR_RETURN(auto element_type,
                        ConvertTo<mlir::ElementType>(type.getElementType()));
    mlir::MlirOp min_op;
    mlir::MlirOp max_op;
    if (c10::isIntegralType(scalar_type, /*includeBool=*/false)) {
      min_op =
          MakeScalarConstant(builder, bounds.min_val.toLong(), element_type);
      max_op =
          MakeScalarConstant(builder, bounds.max_val.toLong(), element_type);
    } else {
      min_op =
          MakeScalarConstant(builder, bounds.min_val.toDouble(), element_type);
      max_op =
          MakeScalarConstant(builder, bounds.max_val.toDouble(), element_type);
    }
    return BuildHardtanhShlo(self_op, min_op, max_op);
  };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(scalar_type));

  return dispatch_fn(
      op_builder, OpInputs<1>{self},
      DispatchOpOptions<1>{.out_dtype = output_dtype,
                           .out_dims = CopyIntVector(self.sizes()),
                           .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRef> AtenHardtanhImpl(const at::Tensor& self,
                                                 const at::Scalar& min_val,
                                                 const at::Scalar& max_val,
                                                 OpParamCacheKeys param_keys) {
  return DispatchHardtanh<absl::StatusOr<DeviceBufferRef>>(
      self, min_val, max_val, std::move(param_keys),
      [](auto&& op_builder, const OpInputs<1>& inputs,
         DispatchOpOptions<1> options) {
        return DispatchOp<1>(op_builder, inputs, std::move(options));
      });
}

absl::Status AtenHardtanhOutImpl(const at::Tensor& self,
                                 const at::Scalar& min_val,
                                 const at::Scalar& max_val,
                                 OpParamCacheKeys param_keys, at::Tensor& out) {
  return DispatchHardtanh<absl::Status>(
      self, min_val, max_val, std::move(param_keys),
      [&out](auto&& op_builder, const OpInputs<1>& inputs,
             DispatchOpOptions<1> options) {
        return DispatchOpOut<1>(op_builder, inputs, out, std::move(options));
      });
}

absl::Status ValidateHardtanhBackwardInputs(const at::Tensor& self) {
  auto scalar_type = self.scalar_type();
  TT_RET_CHECK(c10::isFloatingType(scalar_type), error::kInvalidArgument)
      << "expected the input dtype to be floating point, got "
      << ToString(scalar_type);
  return absl::OkStatus();
}

// Helper to dispatch the hardtanh_backward computation on the device.
template <typename ReturnType, typename DispatchFn>
ReturnType DispatchHardtanhBackward(const at::Tensor& grad_output,
                                    const at::Tensor& self,
                                    const at::Scalar& min_val,
                                    const at::Scalar& max_val,
                                    OpParamCacheKeys param_keys,
                                    DispatchFn&& dispatch_fn) {
  TT_RETURN_IF_ERROR(ValidateHardtanhBackwardInputs(self));
  auto scalar_type = self.scalar_type();

  TT_ASSIGN_OR_RETURN(const HardtanhBounds bounds,
                      ClampHardtanhBounds(scalar_type, min_val, max_val));

  auto op_builder = [bounds](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto [grad_output_op, self_op] = inputs;
    auto& builder = grad_output_op.getBuilder();
    auto type = GetTensorTypeOrDie(self_op);
    TT_ASSIGN_OR_RETURN(auto element_type,
                        ConvertTo<mlir::ElementType>(type.getElementType()));
    mlir::MlirOp min_op =
        MakeScalarConstant(builder, bounds.min_val.toDouble(), element_type);
    mlir::MlirOp max_op =
        MakeScalarConstant(builder, bounds.max_val.toDouble(), element_type);
    return BuildHardtanhBackwardShlo(grad_output_op, self_op, min_op, max_op);
  };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(scalar_type));
  Dimensions out_dims = CopyIntVector(self.sizes());

  return dispatch_fn(
      op_builder, OpInputs<2>{grad_output, self},
      DispatchOpOptions<1>{.out_dtype = output_dtype,
                           .out_dims = std::move(out_dims),
                           .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRef> AtenHardtanhBackwardImpl(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Scalar& min_val, const at::Scalar& max_val,
    OpParamCacheKeys param_keys) {
  return DispatchHardtanhBackward<absl::StatusOr<DeviceBufferRef>>(
      grad_output, self, min_val, max_val, std::move(param_keys),
      [](auto&& op_builder, const OpInputs<2>& inputs,
         DispatchOpOptions<1> options) {
        return DispatchOp<2>(op_builder, inputs, std::move(options));
      });
}

absl::Status AtenHardtanhBackwardOutImpl(const at::Tensor& grad_output,
                                         const at::Tensor& self,
                                         const at::Scalar& min_val,
                                         const at::Scalar& max_val,
                                         OpParamCacheKeys param_keys,
                                         at::Tensor& grad_input) {
  return DispatchHardtanhBackward<absl::Status>(
      grad_output, self, min_val, max_val, std::move(param_keys),
      [&grad_input](auto&& op_builder, const OpInputs<2>& inputs,
                    DispatchOpOptions<1> options) {
        return DispatchOpOut<2>(op_builder, inputs, grad_input,
                                std::move(options));
      });
}
}  // namespace

at::Tensor AtenHardtanh(const at::Tensor& self, const at::Scalar& min_val,
                        const at::Scalar& max_val) {
  TT_KERNEL(OpName::kHardtanh, param_keys, (self, min_val, max_val), {
    TT_ASSIGN_OR_THROW(
        auto result_buf,
        AtenHardtanhImpl(self, min_val, max_val, std::move(param_keys)));
    return MakeTensor(std::move(result_buf));
  });
}

at::Tensor& AtenHardtanh_(at::Tensor& self, const at::Scalar& min_val,
                          const at::Scalar& max_val) {
  return AtenHardtanhOut(self, min_val, max_val, self);
}

at::Tensor& AtenHardtanhOut(const at::Tensor& self, const at::Scalar& min_val,
                            const at::Scalar& max_val, at::Tensor& out) {
  TT_KERNEL(OpName::kHardtanhOut, param_keys, (self, min_val, max_val, out), {
    TT_THROW_IF_ERROR(AtenHardtanhOutImpl(self, min_val, max_val,
                                          std::move(param_keys), out));
    return out;
  });
}

at::Tensor AtenHardtanhBackward(const at::Tensor& grad_output,
                                const at::Tensor& self,
                                const at::Scalar& min_val,
                                const at::Scalar& max_val) {
  TT_KERNEL(OpName::kHardtanhBackward, param_keys,
            (grad_output, self, min_val, max_val), {
              TT_ASSIGN_OR_THROW(
                  auto result_buf,
                  AtenHardtanhBackwardImpl(grad_output, self, min_val, max_val,
                                           std::move(param_keys)));
              return MakeTensor(std::move(result_buf));
            });
}

at::Tensor& AtenHardtanhBackwardGradInput(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          const at::Scalar& min_val,
                                          const at::Scalar& max_val,
                                          at::Tensor& grad_input) {
  TT_KERNEL(OpName::kHardtanhBackwardGradInput, param_keys,
            (grad_output, self, min_val, max_val, grad_input), {
              TT_THROW_IF_ERROR(AtenHardtanhBackwardOutImpl(
                  grad_output, self, min_val, max_val, std::move(param_keys),
                  grad_input));
              return grad_input;
            });
}

}  // namespace torch_tpu
