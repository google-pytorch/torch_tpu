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

#ifndef TORCH_TPU_CSRC_COMMON_ATEN_UTILS_H_
#define TORCH_TPU_CSRC_COMMON_ATEN_UTILS_H_

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/device_type.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

// Use these functions only for debugging purposes.
[[nodiscard]] std::string ToString(const at::ScalarType& scalar_type,
                                   const std::string& name);

// Returns the scalar type of the given tensor.
inline at::ScalarType GetScalarType(const at::Tensor& tensor) {
  return tensor.scalar_type();
}

// Returns the scalar type of the given scalar.
inline at::ScalarType GetScalarType(const at::Scalar& scalar) {
  return scalar.type();
}

inline at::ScalarType GetScalarType(at::ScalarType scalar_type) {
  return scalar_type;
}

// Returns the input dtype if it is a floating-point type or complex type.
// Otherwise, returns the current default scalar dtype.
[[nodiscard]] c10::ScalarType InferOutputDtype(const at::Tensor& self);

// Return PyTorch's computation dtype for a given input dtype.
absl::StatusOr<mlir::ElementType> InferComputationDtype(
    mlir::ElementType input_dtype);

inline absl::Status ValidateIsMatrix(const at::Tensor& tensor,
                                     std::string_view arg_name) {
  TT_RET_CHECK(tensor.dim() == 2, error::kInvalidArgument)
      << "expected the " << arg_name
      << " argument to be a 2D tensor (matrix), got " << tensor.dim()
      << "D of shape " << ToString(tensor.sizes());
  return absl::OkStatus();
}

// Verifies that result_dtype matches or can be safely cast to the out tensor's
// dtype according to PyTorch type casting rules.
//
// If allow_out_dtype_cast is false, requires out.scalar_type() == result_dtype
// (matching PyTorch TensorIterator's check_all_same_dtype(true)).
// Otherwise, verifies that result_dtype can be safely cast to out.scalar_type()
// via c10::canCast.
inline absl::Status ValidateOutDtype(const at::Tensor& out,
                                     const at::ScalarType result_dtype,
                                     const bool allow_out_dtype_cast = true) {
  if (!allow_out_dtype_cast) {
    TT_RET_CHECK(out.scalar_type() == result_dtype, error::kInvalidArgument)
        << "expected the output dtype to be " << ToString(result_dtype)
        << ", got " << ToString(out.scalar_type());
    return absl::OkStatus();
  }
  TT_RET_CHECK(c10::canCast(result_dtype, out.scalar_type()),
               error::kInvalidArgument)
      << "result type " << ToString(result_dtype)
      << " can't be cast to the desired output type "
      << ToString(out.scalar_type());
  return absl::OkStatus();
}

// Returns whether `thing` holds boolean data.
//
// The template parameter `T` should be one of:
//
//   - const at::Tensor&
//   - const at::Scalar&
//   - at::ScalarType
template <typename T>
bool IsBool(T thing) {
  return GetScalarType(thing) == at::ScalarType::Bool;
}

// Returns whether `thing` holds complex data.
//
// The template parameter `T` should be one of:
//
//   - const at::Tensor&
//   - const at::Scalar&
//   - at::ScalarType
template <typename T>
bool IsComplex(T thing) {
  return c10::isComplexType(GetScalarType(thing));
}

// Returns whether `tensor` holds floating-point data.
inline bool IsFloatingPoint(const at::Tensor& tensor) {
  return tensor.is_floating_point();
}

// Returns whether `tensor` holds `float32` (float) or `float64` (double) data.
inline bool IsFloatOrDouble(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::ScalarType::Float ||
         tensor.scalar_type() == at::ScalarType::Double;
}

// Returns whether `tensor` holds `float16` (half) data.
inline bool IsHalf(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::ScalarType::Half;
}

// Returns whether `tensor` holds integer data.
inline bool IsInteger(const at::Tensor& tensor) {
  return c10::isIntegralType(tensor.scalar_type(), /* includeBool= */ false);
}

// Returns whether `thing` holds integral data.
//
// The template parameter `T` should be one of:
//
//   - const at::Tensor&
//   - const at::Scalar&
//   - at::ScalarType
template <typename T>
bool IsIntegral(T thing) {
  return c10::isIntegralType(GetScalarType(thing), /* includeBool= */ true);
}

// Returns whether `tensor` holds Long typed data.
inline bool IsLong(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::ScalarType::Long;
}

// Returns whether `tensor` holds complex data that is supported by OpenXLA:
// either complex64 or complex128.
inline bool IsXlaSupportedComplex(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::ScalarType::ComplexFloat ||
         tensor.scalar_type() == at::ScalarType::ComplexDouble;
}

// Returns whether `tensor` device is PrivateUse1.
inline bool IsPrivateUse1Device(const at::Tensor& tensor) {
  return tensor.device().type() == GetPrivateUse1DeviceType();
}

// Returns whether `tensor` device is CPU.
inline bool IsCpuDevice(const at::Tensor& tensor) {
  return tensor.device().type() == c10::DeviceType::CPU;
}

// Unfortunately, the PyTorch dispatcher can call us by passing an optional
// tensor containing a tensor that is not defined. This helper function can be
// used to properly handle those scenarios.
inline std::optional<at::Tensor> SanitizeOptionalTensor(
    std::optional<at::Tensor> t) {
  return t && t->defined() ? t : std::nullopt;
}

// A wrapper struct to make an at::Scalar hashable by absl::Hash.
struct HashableScalar {
  // The actual value of the scalar on host. Note that scalar.type() always
  // returns a 64-bit type (or complex double).
  at::Scalar scalar;
  // The desired type of the scalar.
  at::ScalarType scalar_type;
};

// at::Scalars also lack an == operator by default, so we need to define one
// ourselves as well to support absl containers.
bool operator==(const HashableScalar& lhs, const HashableScalar& rhs);

// Returns true if `scalar` can be cast to `scalar_type` without overflow,
// or false if overflow would occur.
//
// This function crashes if the given `scalar_type` is a type not supported by
// `Scalar` conversion, i.e. template function `scalar.to<TYPE>()` is deleted.
//
// Returns an error if casting `scalar` to the given `scalar_type` is not
// supported, i.e. there's no implementation for `scalar.to<type>()`.
absl::StatusOr<bool> CanCastScalarWithoutOverflow(const at::Scalar& scalar,
                                                  at::ScalarType scalar_type);

// Make c10::Scalar hashable for absl containers. This encodes both the target
// type and host value of the scalar.
template <typename H>
H AbslHashValue(H h, const HashableScalar& hashable_scalar) {
  const c10::Scalar& scalar = hashable_scalar.scalar;

  // Use the actual scalar_type to know which ".toX()" method to call to get
  // the value of the scalar on host.
  const c10::ScalarType host_scalar_type = scalar.type();
  // But use the desired scalar_type in the hash so that scalar with the same
  // host value but different desired types are hashed differently,
  // e.g. 1.0f64 != 1.0bf16.
  const c10::ScalarType desired_scalar_type = hashable_scalar.scalar_type;

  switch (host_scalar_type) {
    case c10::ScalarType::ComplexDouble: {
      double real = scalar.toComplexDouble().real();
      double imag = scalar.toComplexDouble().imag();
      if (std::isnan(real)) {
        real = std::numeric_limits<double>::quiet_NaN();
      }
      if (std::isnan(imag)) {
        imag = std::numeric_limits<double>::quiet_NaN();
      }
      return H::combine(std::move(h), desired_scalar_type, real, imag);
    }
    case c10::ScalarType::Double: {
      double val = scalar.toDouble();
      if (std::isnan(val)) {
        val = std::numeric_limits<double>::quiet_NaN();
      }
      return H::combine(std::move(h), desired_scalar_type, val);
    }
    case c10::ScalarType::UInt64:
      return H::combine(std::move(h), desired_scalar_type, scalar.toUInt64());
    case c10::ScalarType::Long:
      return H::combine(std::move(h), desired_scalar_type, scalar.toLong());
    case c10::ScalarType::Bool:
      return H::combine(std::move(h), desired_scalar_type, scalar.toBool());
    default:
      // This should never happen;  scalar.type() should already have thrown an
      // error in this case. See definition in torch/c10/core/Scalar.h
      // at::Scalars are only ever actually stored as uint64_t, int64_t, double,
      // c10::complex<double>, bool, or symbolic versions of these that get
      // converted to concrete values by the .toX() methods.
      ABSL_CHECK(false) << "at::Scalar has an unexpected type: "  // CRASH_OK
                        << scalar.type();
  }
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_ATEN_UTILS_H_
