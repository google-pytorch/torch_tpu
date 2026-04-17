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

#include "torch_tpu/ops/view_decomposition/bitcast_primitive.h"

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

// Computes the shape of the tensor after a real-to-real bitcast.
// This will add a dimension if moving from a larger size to a smaller one,
// or remove a dimension if moving from a smaller size to a larger one.
absl::StatusOr<Dimensions> GetShapeAfterRealToRealBitcast(
    const int64_t from_bitwidth, const int64_t to_bitwidth,
    absl::Span<const int64_t> shape) {
  if (from_bitwidth == to_bitwidth) {
    return CopyIntVector(shape);
  }
  if (from_bitwidth > to_bitwidth) {
    TT_RET_CHECK(from_bitwidth % to_bitwidth == 0, error::kInvalidArgument)
        << "the bitwidths are not divisible";
    const int64_t size_ratio = from_bitwidth / to_bitwidth;
    // Element size is smaller, so there are more elements in the new type,
    // which we add as a new dimension.
    Dimensions result;
    result.reserve(shape.size() + 1);
    for (int64_t dim : shape) {
      result.push_back(dim);
    }
    result.push_back(size_ratio);
    return result;
  }
  // from_bitwidth < to_bitwidth
  TT_RET_CHECK(to_bitwidth % from_bitwidth == 0, error::kInvalidArgument)
      << "the bitwidths are not divisible";
  const int64_t size_ratio = to_bitwidth / from_bitwidth;
  TT_RET_CHECK(!shape.empty() && shape.back() == size_ratio,
               error::kInvalidArgument)
      << "the last dimension does not match the size ratio " << size_ratio;
  // Element size is larger, so the last dimension is removed.
  Dimensions result;
  result.reserve(shape.size() - 1);
  for (int64_t i = 0; i < shape.size() - 1; ++i) {
    result.push_back(shape[i]);
  }
  return result;
}

// Computes the strides of the tensor after a real-to-real bitcast.
// This will add a dimension if moving from a larger size to a smaller one,
// or remove a dimension if moving from a smaller size to a larger one.
absl::StatusOr<Strides> GetStridesAfterRealToRealBitcast(
    const int64_t from_bitwidth, const int64_t to_bitwidth,
    absl::Span<const int64_t> strides) {
  if (from_bitwidth == to_bitwidth) {
    // No change in the element size, so the strides are the same.
    return CopyIntVector(strides);
  }
  if (from_bitwidth > to_bitwidth) {
    TT_RET_CHECK(from_bitwidth % to_bitwidth == 0, error::kInvalidArgument)
        << "because the bitwidths are not divisible";
    const int64_t size_ratio = from_bitwidth / to_bitwidth;
    // New element size is smaller, so we append a new dimension with a stride
    // of 1, and scale up the existing strides by the size ratio.
    Strides result;
    result.reserve(strides.size() + 1);
    for (int64_t stride : strides) {
      result.push_back(stride * size_ratio);
    }
    result.push_back(1);
    return result;
  }
  // from_bitwidth < to_bitwidth
  TT_RET_CHECK(to_bitwidth % from_bitwidth == 0, error::kInvalidArgument)
      << "the bitwidths are not divisible";
  const int64_t size_ratio = to_bitwidth / from_bitwidth;
  // The last stride must be 1 so that it is dense over the new, larger
  // element size.
  TT_RET_CHECK(!strides.empty() && strides.back() == 1, error::kInvalidArgument)
      << "the last dimension does not match the size ratio " << size_ratio;
  // Element size is larger, so the last dimension is removed, and all
  // existing strides are divided by the size ratio.
  Strides result;
  result.reserve(strides.size() - 1);
  for (int64_t i = 0; i < strides.size() - 1; ++i) {
    TT_RET_CHECK(strides[i] % size_ratio == 0, error::kInvalidArgument)
        << "the stride of dimension " << i << " is " << strides[i]
        << " which is not divisible by the size ratio " << size_ratio;
    result.push_back(strides[i] / size_ratio);
  }
  return result;
}

// Computes the storage offset of the tensor after a real-to-real bitcast.
// This may multiply or divide the storage offset by the size ratio, if the
// element size is decreased or increased respectively.
absl::StatusOr<int64_t> GetStorageOffsetAfterRealToRealBitcast(
    const int64_t from_bitwidth, const int64_t to_bitwidth,
    int64_t storage_offset) {
  if (from_bitwidth == to_bitwidth) {
    return storage_offset;
  }
  if (from_bitwidth > to_bitwidth) {
    TT_RET_CHECK(from_bitwidth % to_bitwidth == 0, error::kInvalidArgument)
        << "the bitwidths are not divisible";
    const int64_t size_ratio = (from_bitwidth / to_bitwidth);
    // The new element size is smaller, so the existing storage offset covers
    // more new elements.
    return storage_offset * size_ratio;
  }
  // from_bitwidth < to_bitwidth
  TT_RET_CHECK(to_bitwidth % from_bitwidth == 0, error::kInvalidArgument)
      << "the bitwidths are not divisible";
  const int64_t size_ratio = to_bitwidth / from_bitwidth;
  TT_RET_CHECK(storage_offset % size_ratio == 0, error::kInvalidArgument)
      << "the storage offset " << storage_offset
      << " is not divisible by the size ratio " << size_ratio;
  // The existing storage offset covers fewer new elements.
  return storage_offset / size_ratio;
}

// Returns a common error message prefix for invalid real-to-real bitcasts.
std::string RealToRealBitcastErrorPrefix(const RealToRealBitcast& bitcast) {
  return absl::StrCat("cannot bitcast from ", ToString(bitcast.from_type),
                      " to ", ToString(bitcast.to_type), ": ");
}

// Returns the logical equivalent of torch.view_as_real().
// The PyTorch equivalent code would be:
//   torch.stack([torch.real(tensor), torch.imag(tensor)], -1)
absl::StatusOr<mlir::MlirOp> ViewAsRealShlo(mlir::MlirOp input) {
  mlir::MlirOp real_component = mlir::stablehlo::Real(input);
  mlir::MlirOp imag_component = mlir::stablehlo::Imag(input);
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  // Stacking shape (d0, d1, ..., dn-1) tensors returns a shape
  // (d0, d1, ..., dn-1, 2) tensor
  // Stacking scalars returns a shape {2} tensor
  Dimensions component_shape = CopyIntVector(input_type.getShape());
  component_shape.push_back(1);
  real_component = mlir::stablehlo::Reshape(real_component, component_shape);
  imag_component = mlir::stablehlo::Reshape(imag_component, component_shape);
  return mlir::stablehlo::Concatenate(input.getBuilder(),
                                      {real_component, imag_component},
                                      input_type.getRank());
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const RealToRealBitcast& bitcast) {
  os << "bitcast(from_type=" << ToString(bitcast.from_type)
     << ", to_type=" << ToString(bitcast.to_type) << ")";
  return os;
}

std::string_view ToString(ComplexToRealBitcastType bitcast) {
  switch (bitcast) {
    case ComplexToRealBitcastType::kViewAsReal:
      return "view_as_real";
    case ComplexToRealBitcastType::kReal:
      return "real";
    case ComplexToRealBitcastType::kImag:
      return "imag";
  }
}

std::ostream& operator<<(std::ostream& os,
                         const ComplexToRealBitcastType bitcast) {
  return os << ToString(bitcast);
}

std::ostream& operator<<(std::ostream& os,
                         const ComplexToRealBitcast& bitcast) {
  os << bitcast.bitcast_type << "(" << bitcast.complex_element_type << ")";
  return os;
}

std::string_view ToString(ComplexElementType complex_element_type) {
  switch (complex_element_type) {
    case ComplexElementType::kComplexFloat:
      return "cfloat";
    case ComplexElementType::kComplexDouble:
      return "cdouble";
  }
}

std::ostream& operator<<(std::ostream& os,
                         const ComplexElementType complex_element_type) {
  return os << ToString(complex_element_type);
}

std::ostream& operator<<(std::ostream& os, const ViewAsComplex& bitcast) {
  os << "view_as_complex(" << bitcast.complex_element_type << ")";
  return os;
}

// Updates the layout to reflect the effect of applying the given real-to-real
// bitcast.
// Returns true if the layout was modified, or false if the bitcast is a no-op.
absl::StatusOr<bool> UpdateLayoutImpl(StridedLayout& layout,
                                      const RealToRealBitcast& bitcast) {
  TT_RET_CHECK(bitcast.from_type != mlir::ElementType::COMPLEXF32 &&
                   bitcast.from_type != mlir::ElementType::COMPLEXF64,
               error::kInvalidArgument)
      << RealToRealBitcastErrorPrefix(bitcast)
      << "real-to-real bitcasts must not have complex dtypes";
  TT_RET_CHECK(bitcast.to_type != mlir::ElementType::COMPLEXF32 &&
                   bitcast.to_type != mlir::ElementType::COMPLEXF64,
               error::kInvalidArgument)
      << RealToRealBitcastErrorPrefix(bitcast)
      << "real-to-real bitcasts must not have complex dtypes";
  if (bitcast.from_type == bitcast.to_type) {
    return false;
  }
  const int64_t from_bitwidth = TorchEquivalentBitwidth(bitcast.from_type);
  const int64_t to_bitwidth = TorchEquivalentBitwidth(bitcast.to_type);
  if (from_bitwidth == to_bitwidth) {
    // There's no change in the shape, but this is not a no-op because the
    // dtype is changing.
    return true;
  }
  // We're either adding or removing a dimension; need to compute the new
  // shape, strides, and storage offset.
  Dimensions old_shape;
  old_shape.reserve(layout.strided_dims.size());
  Strides old_strides;
  old_strides.reserve(layout.strided_dims.size());
  for (auto& dim : layout.strided_dims) {
    old_shape.push_back(dim.size);
    old_strides.push_back(dim.stride);
  }
  TT_ASSIGN_OR_RETURN(
      Dimensions new_shape,
      GetShapeAfterRealToRealBitcast(from_bitwidth, to_bitwidth, old_shape),
      _.SetPrepend() << RealToRealBitcastErrorPrefix(bitcast));
  TT_ASSIGN_OR_RETURN(
      Strides new_strides,
      GetStridesAfterRealToRealBitcast(from_bitwidth, to_bitwidth, old_strides),
      _.SetPrepend() << RealToRealBitcastErrorPrefix(bitcast));
  TT_ASSIGN_OR_RETURN(int64_t new_storage_offset,
                      GetStorageOffsetAfterRealToRealBitcast(
                          from_bitwidth, to_bitwidth, layout.storage_offset),
                      _.SetPrepend() << RealToRealBitcastErrorPrefix(bitcast));
  // Update the layout with the new values.
  layout.storage_offset = new_storage_offset;
  layout.strided_dims.clear();
  for (int64_t i = 0; i < new_shape.size(); ++i) {
    layout.strided_dims.push_back(
        StridedDimension{.size = new_shape[i], .stride = new_strides[i]});
  }
  return true;
}

// Updates the layout to reflect the effect of applying the given
// complex-to-real bitcast.
// Returns true if the layout was modified, or false if the bitcast is a no-op.
absl::StatusOr<bool> UpdateLayoutImpl(StridedLayout& layout,
                                      const ComplexToRealBitcast& bitcast) {
  // Double offset and all strides
  layout.storage_offset *= 2;
  for (auto& dim : layout.strided_dims) {
    dim.stride *= 2;
  }
  // Then possibly modify the offset/last stride depending on the bitcast type.
  switch (bitcast.bitcast_type) {
    case ComplexToRealBitcastType::kViewAsReal:
      // Add a new dimension of size 2 and keep the last dimension dense.
      layout.strided_dims.push_back(StridedDimension{.size = 2, .stride = 1});
      break;
    case ComplexToRealBitcastType::kReal:
      // Don't modify the offset, and don't add a new dimension; the last
      // stride will be a multiple of 2 with a +0 relative offset.
      break;
    case ComplexToRealBitcastType::kImag:
      // Don't add a new dimension, but increase the offset so that
      // the last stride is a multiple of 2 with a +1 relative offset.
      layout.storage_offset++;
      break;
  }
  return true;
}

// Updates the layout to reflect the effect of applying the given
// view_as_complex bitcast operation.
// Returns true if the layout was modified, or false if the bitcast is a no-op.
absl::StatusOr<bool> UpdateLayoutImpl(StridedLayout& layout,
                                      const ViewAsComplex& bitcast) {
  TT_RET_CHECK(!layout.strided_dims.empty(), error::kInvalidArgument)
      << "cannot apply view_as_complex to a scalar";
  // Inverse of view_as_real; converts a tensor like f32[2, 3, 8] ->
  // cfloat[2, 3, 4].
  TT_RET_CHECK(layout.strided_dims.back().size % 2 == 0,
               error::kInvalidArgument)
      << "cannot view_as_complex because the last dimension of size "
      << layout.strided_dims.back().size << " is not divisible by 2";
  TT_RET_CHECK(layout.strided_dims.back().stride == 1, error::kInvalidArgument)
      << "cannot view_as_complex because the last dimension is not dense "
         "(stride "
      << layout.strided_dims.back().stride << " != 1)";

  layout.strided_dims.back().size /= 2;

  TT_RET_CHECK(layout.storage_offset % 2 == 0, error::kInvalidArgument)
      << "cannot view_as_complex because the storage offset of "
      << layout.storage_offset << " is not divisible by 2";
  layout.storage_offset /= 2;

  // Reduce the stride of all dimensions except the last one.
  // So strides like (16, 4, 1) become (8, 2, 1).
  for (auto i = 0; i < layout.strided_dims.size() - 1; ++i) {
    auto& dim = layout.strided_dims[i];
    TT_RET_CHECK(dim.stride % 2 == 0, error::kInvalidArgument)
        << "cannot view_as_complex because stride " << dim.stride
        << " is not divisible by 2";
    dim.stride /= 2;
  }
  return true;
}

// Applies a real-to-real bitcast to the given tensor, following PyTorch's
// expectations for bitcast behavior on boolean types.
absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(
    mlir::MlirOp input, const RealToRealBitcast& bitcast) {
  if (bitcast.from_type == bitcast.to_type) {
    // No-op.
    return input;
  }

  // Boolean handling: conversions from PRED to other need to be
  // PRED -convert-> UI8 -bitcast-> {other type} so that PRED values are
  // represented as 0x00 or 0x01 bytes, and not individual 0 or 1 bits.
  if (bitcast.from_type == mlir::ElementType::PRED) {
    input = mlir::stablehlo::ConvertElementType(input, mlir::ElementType::UI8);
    return ViewPrimitiveShlo(input, RealToRealBitcast{
                                        .from_type = mlir::ElementType::UI8,
                                        .to_type = bitcast.to_type,
                                    });
  }
  // Conversions to PRED need to be {other type} -bitcast-> UI8 -convert-> PRED
  // so that all bytes other than 0x00 are converted to "true" bits.
  if (bitcast.to_type == mlir::ElementType::PRED) {
    if (bitcast.from_type != mlir::ElementType::UI8) {
      TT_ASSIGN_OR_RETURN(
          input, ViewPrimitiveShlo(input, RealToRealBitcast{
                                              .from_type = bitcast.from_type,
                                              .to_type = mlir::ElementType::UI8,
                                          }));
    }
    return mlir::stablehlo::ConvertElementType(input, mlir::ElementType::PRED);
  }

  // Otherwise, this is a regular real-to-real bitcast.
  // stablehlo::BitcastConvert requires specifying the full output type.
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const int64_t from_bitwidth = TorchEquivalentBitwidth(bitcast.from_type);
  const int64_t to_bitwidth = TorchEquivalentBitwidth(bitcast.to_type);
  // Need to propagate dynamic bound to output shape for bitcast.
  TT_RET_CHECK(input_type.hasStaticShape(), error::kInvalidArgument)
      << "lowering for bitcast with dynamic shapes is currently not supported";
  TT_ASSIGN_OR_RETURN(Dimensions result_shape,
                      GetShapeAfterRealToRealBitcast(from_bitwidth, to_bitwidth,
                                                     input_type.getShape()));
  mlir::Type result_element_type =
      mlir::getElementType(input.getContext(), bitcast.to_type);
  mlir::RankedTensorType result_type =
      mlir::RankedTensorType::get(result_shape, result_element_type);

  return mlir::stablehlo::BitcastConvert(result_type, input);
}

// Returns the logical equivalent of torch.view_as_real(), torch.real(), or
// torch.imag() as applicable for the complex-to-real bitcast.
absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(
    mlir::MlirOp input, const ComplexToRealBitcast& bitcast) {
  switch (bitcast.bitcast_type) {
    case ComplexToRealBitcastType::kViewAsReal:
      return ViewAsRealShlo(input);
    case ComplexToRealBitcastType::kReal:
      return mlir::stablehlo::Real(input);
    case ComplexToRealBitcastType::kImag:
      return mlir::stablehlo::Imag(input);
  }
}

// Returns the logical equivalent of torch.view_as_complex().
// The PyTorch equivalent code would be:
//   torch.complex(torch.select(tensor, -1, 0), torch.select(tensor, -1, 1))
absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(mlir::MlirOp input,
                                               const ViewAsComplex& bitcast) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  TT_RET_CHECK(input_type.getRank() > 0, error::kInvalidArgument)
      << "cannot view_as_complex because the rank of the input tensor is 0";
  TT_RET_CHECK(input_type.getShape().back() % 2 == 0, error::kInvalidArgument)
      << "cannot view_as_complex because the last dimension of size "
      << input_type.getShape().back() << " is not divisible by 2";

  // Get the real values first
  Indices start_indices(input_type.getRank(), 0);
  Indices limit_indices(input_type.getShape().begin(),
                        input_type.getShape().end());
  Indices strides(input_type.getRank(), 1);
  strides.back() = 2;

  mlir::MlirOp real_values =
      mlir::stablehlo::Slice(input, start_indices, limit_indices, strides);

  // Then get the imaginary values
  start_indices.back() = 1;
  mlir::MlirOp imag_values =
      mlir::stablehlo::Slice(input, start_indices, limit_indices, strides);

  return mlir::stablehlo::Complex(real_values, imag_values);
}

}  // namespace torch_tpu
