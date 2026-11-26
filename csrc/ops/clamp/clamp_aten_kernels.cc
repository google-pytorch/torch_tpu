// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/ops/clamp/clamp_aten_kernels.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/promote_types.h"
#include "ATen/ops/result_type.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Optional.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/clamp/clamp.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

// Returns the computation `at::ScalarType` for scalar bounds, following PyTorch
// ATen rules: if `self` is floating-point, scalars do not promote it. If `self`
// is non-floating (integral), scalars promote it via `at::result_type`.
at::ScalarType GetComputationType(const at::Tensor& self,
                                  const c10::optional<at::Scalar>& min,
                                  const c10::optional<at::Scalar>& max) {
  at::ScalarType computation_scalar_type = self.scalar_type();
  if (!c10::isFloatingType(computation_scalar_type)) {
    if (min) {
      computation_scalar_type = at::result_type(self, *min);
    }
    if (max) {
      computation_scalar_type =
          min ? at::promote_types(computation_scalar_type,
                                  at::result_type(self, *max))
              : at::result_type(self, *max);
    }
  }
  return computation_scalar_type;
}

// Returns the computation `at::ScalarType` for tensor bounds.
at::ScalarType GetComputationType(const at::Tensor& self,
                                  const c10::optional<at::Tensor>& min,
                                  const c10::optional<at::Tensor>& max) {
  at::ScalarType computation_scalar_type = self.scalar_type();
  if (min) {
    computation_scalar_type = at::result_type(self, *min);
  }
  if (max) {
    computation_scalar_type =
        min ? at::promote_types(computation_scalar_type,
                                at::result_type(self, *max))
            : at::result_type(self, *max);
  }
  return computation_scalar_type;
}

absl::StatusOr<Dimensions> ComputeOutputShape(
    const at::Tensor& self, const std::optional<at::Tensor>& min,
    const std::optional<at::Tensor>& max) {
  if (min && max) {
    return InferSize(self.sizes(), min->sizes(), max->sizes());
  }
  if (min) {
    return InferSize(self.sizes(), min->sizes());
  }
  if (max) {
    return InferSize(self.sizes(), max->sizes());
  }
  return CopyIntVector(self.sizes());
}

template <typename T>
absl::Status ValidateNotComplex(const T& value,
                                const std::string_view arg_name) {
  TT_RET_CHECK(!IsComplex(value), error::kInvalidArgument)
      << "expected " << arg_name << " not to be complex, got "
      << ToString(GetScalarType(value));
  return absl::OkStatus();
}

absl::Status ValidateNotBool(const at::ScalarType computation_scalar_type) {
  TT_RET_CHECK(!IsBool(computation_scalar_type), error::kInvalidArgument)
      << "self must not be bool";
  return absl::OkStatus();
}

absl::Status ValidateOutDevice(const at::Tensor& out) {
  TT_RET_CHECK(IsPrivateUse1Device(out), error::kInvalidArgument)
      << "expected out tensor to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << out.device();
  return absl::OkStatus();
}

// Where a scalar clamp bound falls relative to the representable range of an
// integral dtype, and the limits of that range.
struct IntegralBoundInfo {
  bool below_lowest;
  bool above_max;
  at::Scalar lowest;
  at::Scalar max;
};

// `bound` must hold an integer or bool: a floating-point or complex bound
// promotes the computation dtype away from the integral type `T`.
template <typename T>
IntegralBoundInfo GetIntegralBoundInfoForType(const at::Scalar& bound) {
  const auto info_for = [](const auto val) -> IntegralBoundInfo {
    return {
        .below_lowest = std::cmp_less(val, std::numeric_limits<T>::lowest()),
        .above_max = std::cmp_greater(val, std::numeric_limits<T>::max()),
        .lowest = at::Scalar(std::numeric_limits<T>::lowest()),
        .max = at::Scalar(std::numeric_limits<T>::max()),
    };
  };
  return bound.type() == at::ScalarType::UInt64 ? info_for(bound.toUInt64())
                                                : info_for(bound.toLong());
}

// Returns nullopt if `dtype` is not an integral dtype (bool excluded).
std::optional<IntegralBoundInfo> GetIntegralBoundInfo(
    const at::Scalar& bound, const at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Byte:
      return GetIntegralBoundInfoForType<uint8_t>(bound);
    case at::ScalarType::Char:
      return GetIntegralBoundInfoForType<int8_t>(bound);
    case at::ScalarType::Short:
      return GetIntegralBoundInfoForType<int16_t>(bound);
    case at::ScalarType::UInt16:
      return GetIntegralBoundInfoForType<uint16_t>(bound);
    case at::ScalarType::Int:
      return GetIntegralBoundInfoForType<int32_t>(bound);
    case at::ScalarType::UInt32:
      return GetIntegralBoundInfoForType<uint32_t>(bound);
    case at::ScalarType::Long:
      return GetIntegralBoundInfoForType<int64_t>(bound);
    case at::ScalarType::UInt64:
      return GetIntegralBoundInfoForType<uint64_t>(bound);
    default:
      return std::nullopt;
  }
}

// Saturates a no-op out-of-range integral scalar bound (min < lowest or
// max > max) to the dtype's representable limit before PromoteScalar, so
// materializing the scalar tensor never overflows or wraps. Invalid
// out-of-range bounds (min > max or max < lowest) are left unchanged so
// ValidateIntegralClampScalarBound reports the error inside TT_KERNEL.
at::Scalar SaturateNoOpIntegralBound(const at::Scalar& bound,
                                     const at::ScalarType dtype,
                                     const bool is_lower_bound) {
  const std::optional<IntegralBoundInfo> info =
      GetIntegralBoundInfo(bound, dtype);
  if (!info.has_value()) {
    return bound;
  }
  if (is_lower_bound && info->below_lowest) {
    return info->lowest;
  }
  if (!is_lower_bound && info->above_max) {
    return info->max;
  }
  return bound;
}

std::optional<at::Scalar> SaturateNoOpIntegralBound(
    const c10::optional<at::Scalar>& bound, const at::ScalarType dtype,
    const bool is_lower_bound) {
  if (!bound.has_value()) {
    return std::nullopt;
  }
  return SaturateNoOpIntegralBound(*bound, dtype, is_lower_bound);
}

absl::Status ValidateIntegralClampScalarBound(const at::Scalar& bound,
                                              const at::ScalarType dtype,
                                              const bool is_lower_bound) {
  const std::optional<IntegralBoundInfo> info =
      GetIntegralBoundInfo(bound, dtype);
  if (!info.has_value()) {
    return absl::OkStatus();
  }
  const bool is_invalid = is_lower_bound ? info->above_max : info->below_lowest;
  TT_RET_CHECK(!is_invalid, error::kInvalidArgument)
      << "expected clamp " << (is_lower_bound ? "min" : "max")
      << " value to be representable as " << ToString(dtype) << ", got "
      << ToString(bound);
  return absl::OkStatus();
}

absl::Status AtenClampTensorHelper(const at::Tensor& self,
                                   const c10::optional<at::Tensor>& min_,
                                   const c10::optional<at::Tensor>& max_,
                                   at::ScalarType computation_scalar_type,
                                   bool allow_out_dtype_cast, at::Tensor& out,
                                   OpParamCacheKeys param_keys) {
  auto min = SanitizeOptionalTensor(min_);
  auto max = SanitizeOptionalTensor(max_);

  TT_RETURN_IF_ERROR(ValidateNotComplex(self, /*arg_name=*/"self"));
  if (min) {
    TT_RETURN_IF_ERROR(ValidateNotComplex(*min, /*arg_name=*/"min"));
  }
  if (max) {
    TT_RETURN_IF_ERROR(ValidateNotComplex(*max, /*arg_name=*/"max"));
  }
  // Two-sided tensor clamp is not implemented for bool (matching CUDA
  // clamp_cuda), whereas single-sided min/max tensor clamp dispatches to
  // maximum/minimum which support bool.
  if (min && max) {
    TT_RETURN_IF_ERROR(ValidateNotBool(computation_scalar_type));
  }
  TT_RETURN_IF_ERROR(ValidateOutDevice(out));
  TT_RETURN_IF_ERROR(
      ValidateOutDtype(out, computation_scalar_type, allow_out_dtype_cast));
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(computation_scalar_type));
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(out));

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));
  TT_ASSIGN_OR_RETURN(Dimensions output_dims,
                      ComputeOutputShape(self, min, max));

  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(computation_scalar_type));

  DispatchOpOptions<1> options = {
      .out_dtype = output_dtype,
      .out_dims = output_dims,
      .computation_dtype = computation_dtype,
      .op_param_cache_keys = std::move(param_keys),
  };

  if (min && max) {
    auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
      auto& [self, min, max] = inputs;
      return BuildClampShlo(self, min, max);
    };
    return DispatchOpOut<3>(std::move(op_builder), {self, *min, *max}, out,
                            std::move(options));
  } else if (min) {
    auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
      auto& [self, min] = inputs;
      return BuildClampShlo(self, min, std::nullopt);
    };
    return DispatchOpOut<2>(std::move(op_builder), {self, *min}, out,
                            std::move(options));
  } else if (max) {
    auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
      auto& [input, max] = inputs;
      return BuildClampShlo(input, std::nullopt, max);
    };
    return DispatchOpOut<2>(std::move(op_builder), {self, *max}, out,
                            std::move(options));
  } else {
    auto op_builder = [](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildClampShlo(input, std::nullopt, std::nullopt);
    };
    return DispatchOpOut<1>(std::move(op_builder), self, out,
                            std::move(options));
  }
}

// Helper for clamping with promoted scalar bounds.
absl::Status ClampScalarHelper(const at::Tensor& self,
                               std::optional<PromotedScalar> min,
                               std::optional<PromotedScalar> max,
                               at::ScalarType computation_scalar_type,
                               bool allow_out_dtype_cast, at::Tensor& out,
                               OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateNotComplex(self, /*arg_name=*/"self"));
  if (min.has_value()) {
    TT_RETURN_IF_ERROR(ValidateNotComplex(min->scalar(), /*arg_name=*/"min"));
  }
  if (max.has_value()) {
    TT_RETURN_IF_ERROR(ValidateNotComplex(max->scalar(), /*arg_name=*/"max"));
  }
  TT_RETURN_IF_ERROR(ValidateNotBool(computation_scalar_type));
  TT_RETURN_IF_ERROR(ValidateOutDevice(out));
  TT_RETURN_IF_ERROR(
      ValidateOutDtype(out, computation_scalar_type, allow_out_dtype_cast));
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(computation_scalar_type));
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(out));
  if (min.has_value()) {
    TT_RETURN_IF_ERROR(ValidateIntegralClampScalarBound(
        min->scalar(), computation_scalar_type, /*is_lower_bound=*/true));
  }
  if (max.has_value()) {
    TT_RETURN_IF_ERROR(ValidateIntegralClampScalarBound(
        max->scalar(), computation_scalar_type, /*is_lower_bound=*/false));
  }

  std::optional<at::Tensor> min_tensor;
  if (min.has_value()) {
    TT_ASSIGN_OR_RETURN(auto t, min->GetTensor(computation_scalar_type));
    min_tensor = std::move(t);
  }
  std::optional<at::Tensor> max_tensor;
  if (max.has_value()) {
    TT_ASSIGN_OR_RETURN(auto t, max->GetTensor(computation_scalar_type));
    max_tensor = std::move(t);
  }
  return AtenClampTensorHelper(self, min_tensor, max_tensor,
                               computation_scalar_type, allow_out_dtype_cast,
                               out, std::move(param_keys));
}

}  // namespace

at::Tensor& AtenClampOut(const at::Tensor& self,
                         const c10::optional<at::Scalar>& min,
                         const c10::optional<at::Scalar>& max,
                         at::Tensor& out) {
  const at::ScalarType computation_scalar_type =
      GetComputationType(self, min, max);
  auto promoted_min = PromoteScalar(SaturateNoOpIntegralBound(
      min, computation_scalar_type, /*is_lower_bound=*/true));
  auto promoted_max = PromoteScalar(SaturateNoOpIntegralBound(
      max, computation_scalar_type, /*is_lower_bound=*/false));
  TT_KERNEL(OpName::kClampOut, param_keys,
            (self, promoted_min, promoted_max, out), {
              TT_THROW_IF_ERROR(ClampScalarHelper(
                  self, std::move(promoted_min), std::move(promoted_max),
                  computation_scalar_type, /*allow_out_dtype_cast=*/false, out,
                  std::move(param_keys)));
              return out;
            });
}

at::Tensor& AtenClampMinOut(const at::Tensor& self, const at::Scalar& min,
                            at::Tensor& out) {
  const at::ScalarType computation_scalar_type =
      GetComputationType(self, min, std::nullopt);
  auto promoted_min = PromoteScalar(SaturateNoOpIntegralBound(
      min, computation_scalar_type, /*is_lower_bound=*/true));
  TT_KERNEL(OpName::kClampMinOut, param_keys, (self, promoted_min, out), {
    TT_THROW_IF_ERROR(ClampScalarHelper(
        self, std::move(promoted_min), std::nullopt, computation_scalar_type,
        /*allow_out_dtype_cast=*/false, out, std::move(param_keys)));
    return out;
  });
}

at::Tensor& AtenClampMaxOut(const at::Tensor& self, const at::Scalar& max,
                            at::Tensor& out) {
  const at::ScalarType computation_scalar_type =
      GetComputationType(self, std::nullopt, max);
  auto promoted_max = PromoteScalar(SaturateNoOpIntegralBound(
      max, computation_scalar_type, /*is_lower_bound=*/false));
  TT_KERNEL(OpName::kClampMaxOut, param_keys, (self, promoted_max, out), {
    TT_THROW_IF_ERROR(ClampScalarHelper(
        self, std::nullopt, std::move(promoted_max), computation_scalar_type,
        /*allow_out_dtype_cast=*/false, out, std::move(param_keys)));
    return out;
  });
}

at::Tensor& AtenClampTensorOut(const at::Tensor& self,
                               const c10::optional<at::Tensor>& min,
                               const c10::optional<at::Tensor>& max,
                               at::Tensor& out) {
  TT_KERNEL(OpName::kClampTensorOut, param_keys, (self, min, max, out), {
    const at::ScalarType computation_scalar_type =
        GetComputationType(self, min, max);
    TT_THROW_IF_ERROR(AtenClampTensorHelper(
        self, min, max, computation_scalar_type,
        /*allow_out_dtype_cast=*/true, out, std::move(param_keys)));
    return out;
  });
}

at::Tensor& AtenClampMinTensorOut(const at::Tensor& self, const at::Tensor& min,
                                  at::Tensor& out) {
  TT_KERNEL(OpName::kClampMinTensorOut, param_keys, (self, min, out), {
    const at::ScalarType computation_scalar_type =
        GetComputationType(self, min, std::nullopt);
    TT_THROW_IF_ERROR(AtenClampTensorHelper(
        self, min, std::nullopt, computation_scalar_type,
        /*allow_out_dtype_cast=*/true, out, std::move(param_keys)));
    return out;
  });
}

at::Tensor& AtenClampMaxTensorOut(const at::Tensor& self, const at::Tensor& max,
                                  at::Tensor& out) {
  TT_KERNEL(OpName::kClampMaxTensorOut, param_keys, (self, max, out), {
    const at::ScalarType computation_scalar_type =
        GetComputationType(self, std::nullopt, max);
    TT_THROW_IF_ERROR(AtenClampTensorHelper(
        self, std::nullopt, max, computation_scalar_type,
        /*allow_out_dtype_cast=*/true, out, std::move(param_keys)));
    return out;
  });
}

}  // namespace torch_tpu
