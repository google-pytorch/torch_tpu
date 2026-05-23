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

#ifndef TORCH_TPU_OPS_VIEW_DECOMPOSITION_BITCAST_PRIMITIVE_H_
#define TORCH_TPU_OPS_VIEW_DECOMPOSITION_BITCAST_PRIMITIVE_H_

#include <cstdint>
#include <ostream>
#include <string_view>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

// Bitcasting type conversions are a type of view primitive, which reinterprets
// an existing bit pattern as a tensor of a different type and possibly
// different shape. This is most commonly an application of Tensor.view(dtype)
// in torch, or BitcastConvert in StableHLO.
//
// However, some special handling is required for complex types and boolean
// datatypes, to ensure that the StableHLO bitcast behavior is consistent with
// torch's assumptions about data layout.
//
// For booleans, torch will create boolean values in uint8 layout, but only
// containing the values of 0x00 and 0x01. Bitcasts into boolean type will
// accept any bit pattern of uint8s, and will interpret any non-0x00 byte as
// "true". StableHLO does not guarantee the layout of the PRED/I1 type, but
// applying the "Convert" operation between bools and uint8s matches torch's
// semantics.
//
// For complex types, first note that there is a notation difference.
//   * torch uses "complex64" or "cdouble" to mean "a 64-bit complex number
//     stored as two 32-bit floats".
//   * StableHLO uses "complex64" to mean "a complex number composed of two
//     float64s"
// This means that torch.complex64 is equivalent to StableHLO's complex128.
//
// Second, StableHLO does not support PyTorch's torch.complex32/torch.chalf
// precision.
//
// Finally, torch assumes that the real and imaginary components of a complex
// number are stored as a final dimension of size 2, with element 0 being the
// real component and element 1 being the imaginary component. StableHLO does
// not guarantee this layout, but offers Real() and Imag() ops to extract the
// real and imaginary components, and Complex() to stack the components back
// from two disjoint tensors.
namespace torch_tpu {

// A real-to-real bitcast is a reinterpret cast between two real-valued dtypes,
// for example uint64 -> uint32 or f32 -> bf16. This includes conversions to
// and from booleans.
//
// The shape of the output depends on the shape of the input and the relative
// bitwidths of the two types.
//   * If from_type and to_type have the same bitwidth, the shape is the same.
//     For example, f32[1, 2, 3] -> uint32[1, 2, 3].
//   * If bitwidth(from_type) > bitwidth(to_type), the output will have a
//     dimension appended with size bitwidth(from_type) / bitwidth(to_type).
//     For example, uint64[1, 2, 3] -> f32[1, 2, 3, 2].
//   * If bitwidth(from_type) < bitwidth(to_type), the input must have a last
//     dimension of size bitwidth(to_type) / bitwidth(from_type), and this
//     dimension will be removed.
//     For example, f32[1, 2, 3, 2] -> uint64[1, 2, 3].
// Note that this follows StableHLO's bitcast behavior; torch.Tensor.view(dtype)
// is similar but will divide the trailing dimension by the bitwidth ratio,
// rather than removing the dimension, for example:
// f32[1, 2, 3, 2] -> uint64[1, 2, 3, 1].
// Applying an additional reshape either before or after the view will align the
// two approaches.
struct RealToRealBitcast {
  mlir::ElementType from_type;
  mlir::ElementType to_type;
};

// Formats the bitcast like "bitcast(from_type=f32, to_type=bf16)".
std::ostream& operator<<(std::ostream& os, const RealToRealBitcast& bitcast);

// When converting from a cfloat or cdouble value to a real-valued
// number, there are three distinct approaches:
//   * kReal() will extract the real component and discard the imaginary
//     component. This is the equivalent of torch.real() and StableHLO's Real().
//   * kImag() will extract the imaginary component and discard the real
//     component. This is the equivalent of torch.imag() and StableHLO's Imag().
//   * kViewAsReal() will stack the real and imaginary components into a
//     single tensor along the last dimension. This is the equivalent of
//     torch.view_as_real(). There is no single StableHLO op for this, but
//     the same logical result is achieved as concatenating the real and
//     imaginary components along the last dimension and then reshaping to
//     add a trailing dimension of size 2.
enum class ComplexToRealBitcastType {
  // Creates a float tensor with an appended trailing dimension of size 2, where
  // the real component is in element 0 and the imaginary component is in
  // element 1 of the last dimension.
  kViewAsReal,
  // Creates a float tensor with the same shape as the input, which contains
  // only the real component.
  kReal,
  // Creates a float tensor with the same shape as the input, which contains
  // only the imaginary component.
  kImag,
};

[[nodiscard]] std::string_view ToString(ComplexToRealBitcastType bitcast);
std::ostream& operator<<(std::ostream& os, ComplexToRealBitcastType bitcast);

// The number type of a complex number. This is restricts the cases of
// mlir::ElementType to only the complex options.
// It also disambiguates the inconsistent bitwidth notation of torch and
// StableHLO (torch's complex64 is 32+32=64 bits, StableHLO's is 64+64=128
// bits).
enum class ComplexElementType {
  // Represents real and imaginary components using two float32 values.
  // This is equivalent to StableHLO's complex32, or torch.complex64/cfloat.
  kComplexFloat,
  // Represents real and imaginary components using two float64 values.
  // This is equivalent to StableHLO's complex64, or torch.complex128/cdouble.
  kComplexDouble,
};

[[nodiscard]] std::string_view ToString(ComplexElementType bitcast);
std::ostream& operator<<(std::ostream& os,
                         ComplexElementType complex_element_type);

// A complex-to-real bitcast is a reinterpret cast from a complex-valued
// dtype to a real-valued dtype, possibly dropping either the real or imaginary
// component.
// This must be either cfloat -> float32 or cdouble -> float64.
struct ComplexToRealBitcast {
  // The complex dtype to bitcast from.
  ComplexElementType complex_element_type;
  // The type of complex-to-real bitcast to perform.
  ComplexToRealBitcastType bitcast_type;
};

// Formats the bitcast as one of:
//   * view_as_real(cfloat)
//   * real(cfloat)
//   * imag(cfloat)
std::ostream& operator<<(std::ostream& os, const ComplexToRealBitcast& bitcast);

// A reinterpret cast from a real-valued float tensor to a complex-valued
// tensor. This mostly follows the semantics of torch.view_as_complex(), by
// assuming that the input will be stacked using a trailing dimension of even
// size with the real component in the even-indexed positions and the imaginary
// component in the odd-indexed positions.
// Note: torch.view_as_complex requires that the last dimension is *exactly*
// size 2, but we allow any even size).
struct ViewAsComplex {
  ComplexElementType complex_element_type;
};

// Formats the bitcast like "view_as_complex(cfloat)".
std::ostream& operator<<(std::ostream& os, const ViewAsComplex& bitcast);

// Updates the layout to reflect the effect of applying the given bitcast
// primitive.
//
// Crashes if the bitcast is not valid for the layout, which will be
// due to a missing or incompatible last dimension.
//
// Returns true if the layout was modified, or false if the bitcast is a no-op.
bool UpdateLayout(StridedLayout& layout, const RealToRealBitcast& bitcast);
bool UpdateLayout(StridedLayout& layout, const ComplexToRealBitcast& bitcast);
bool UpdateLayout(StridedLayout& layout, const ViewAsComplex& bitcast);

// An overload for the ViewPrimitiveShlo function that handles bitcast
// primitives. This may involve calls to BitcastConvert, Complex, Real, Imag,
// Convert, Concatenate, and/or Reshape depending on the bitcast type.
absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(
    mlir::MlirOp input, const RealToRealBitcast& bitcast);
absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(
    mlir::MlirOp input, const ComplexToRealBitcast& bitcast);
absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(mlir::MlirOp input,
                                               const ViewAsComplex& bitcast);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_VIEW_DECOMPOSITION_BITCAST_PRIMITIVE_H_
