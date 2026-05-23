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

#ifndef TORCH_TPU_COMMON_DTYPE_H_
#define TORCH_TPU_COMMON_DTYPE_H_

#include <cstdint>
#include <string_view>
#include <type_traits>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "xla/xla_data.pb.h"

// This library holds utilities for working with various representations of
// tensor element types. These include:
// - at::ScalarType
// - mlir::ElementType
// - mlir::Type
// - xla::PrimitiveType
//
// In most of the op logic, we should use mlir::ElementType. at::ScalarType is
// only used when interacting with PyTorch. xla::PrimitiveType is only used
// when interacting with XLA and PjRt.

namespace torch_tpu {

// Returns a string representation of the given type as a PyTorch
// dtype name (e.g. "float32", "int8"). This is suitable for use in user
// messages.
[[nodiscard]] std::string_view ToString(mlir::ElementType element_type);

// Returns a string representation of the given MLIR element type.
// E.g. "f32", "ui8". This is suitable for use in cache keys but not for user
// messages.
[[nodiscard]] std::string_view ToShortString(mlir::ElementType element_type);

// Validates that `size` is a valid tensor size, and returns that size in bytes.
// Sizes must have no negative dimensions, and the total size of the tensor
// in bytes must fit in an int64_t. Returns an error if the given type is not
// supported or if the size is invalid.
absl::StatusOr<int64_t> ValidateTensorByteSize(at::IntArrayRef size,
                                               mlir::ElementType element_type);

namespace internal {

// Trait for checking whether a type is one of the supported types for
// ConvertTo().
template <typename ToType>
struct IsValidElemType {
  static constexpr bool value = std::is_same_v<ToType, at::ScalarType> ||
                                std::is_same_v<ToType, mlir::ElementType> ||
                                std::is_same_v<ToType, xla::PrimitiveType>;
};

// Converts to mlir::ElementType.
absl::StatusOr<mlir::ElementType> ToElementType(at::ScalarType scalar_type);
absl::StatusOr<mlir::ElementType> ToElementType(xla::PrimitiveType xla_type);
absl::StatusOr<mlir::ElementType> ToElementType(mlir::Type type);

// Converts from mlir::ElementType and mlir::Type. These should never
// fail as the mlir types are only created by us (torch_tpu) and should be
// valid.
[[nodiscard]] at::ScalarType ToScalarType(mlir::ElementType element_type);
[[nodiscard]] at::ScalarType ToScalarType(mlir::Type type);
[[nodiscard]] xla::PrimitiveType ToXlaType(mlir::ElementType element_type);

}  // namespace internal

// Converts an arbitrary element type to another arbitrary element type.
// ToType must be one of the following types:
// - at::ScalarType
// - mlir::ElementType
// - xla::PrimitiveType
template <typename ToType,
          typename = std::enable_if_t<internal::IsValidElemType<ToType>::value>>
ToType ConvertTo(mlir::ElementType element_type) {
  if constexpr (std::is_same_v<ToType, at::ScalarType>) {
    return internal::ToScalarType(element_type);
  } else if constexpr (std::is_same_v<ToType, xla::PrimitiveType>) {
    return internal::ToXlaType(element_type);
  } else {
    return element_type;
  }
}
template <typename ToType,
          typename = std::enable_if_t<internal::IsValidElemType<ToType>::value>>
absl::StatusOr<ToType> ConvertTo(at::ScalarType scalar_type) {
  TT_ASSIGN_OR_RETURN(auto element_type, internal::ToElementType(scalar_type));
  return ConvertTo<ToType>(element_type);
}
template <typename ToType,
          typename = std::enable_if_t<internal::IsValidElemType<ToType>::value>>
absl::StatusOr<ToType> ConvertTo(xla::PrimitiveType xla_type) {
  TT_ASSIGN_OR_RETURN(auto element_type, internal::ToElementType(xla_type));
  return ConvertTo<ToType>(element_type);
}
template <typename ToType,
          typename = std::enable_if_t<internal::IsValidElemType<ToType>::value>>
absl::StatusOr<ToType> ConvertTo(mlir::Type mlir_type) {
  TT_ASSIGN_OR_RETURN(auto element_type, internal::ToElementType(mlir_type));
  return ConvertTo<ToType>(element_type);
}

// Returns the number of bits used to represent the given element type in torch.
// If the type does not have a torch equivalent (such as F6E2M3FN), returns
// the MLIR type size instead to avoid requiring a StatusOr.
//
// torch bitwidths are typically the same as the MLIR type numeric size, with
// the exception of booleans, which are PRED/I1 in StableHLO but represented by
// uint8s in torch.
//
// Note that mlir::ElementType::COMPLEXF32 and mlir::ElementType::COMPLEXF64
// are the equivalent of torch.complex64 and torch.complex128, respectively, so
// the bitwidth of COMPLEXF32 is 64 and the bitwidth of COMPLEXF64 is 128.
int64_t TorchEquivalentBitwidth(mlir::ElementType element_type);

// Returns the real component type of the input type.
//
// If the input is a complex type, returns the type of its real component;
// otherwise returns the input type itself.
mlir::ElementType RealComponentOf(mlir::ElementType element_type);

namespace internal {

// Fingerprint specialization for `mlir::ElementType`.
template <>
struct Fingerprint64Impl<mlir::ElementType, /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(
      const mlir::ElementType element_type) {
    // The short names for ElementType are unique and stable as they are used
    // for parameter cache keys.
    return Fingerprint(ToShortString(element_type));
  }
};

}  // namespace internal

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_DTYPE_H_
