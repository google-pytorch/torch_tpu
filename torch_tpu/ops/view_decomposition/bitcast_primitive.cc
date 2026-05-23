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
#include <string_view>

#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/static_shape_check.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/view_primitive_error_utils.h"

namespace torch_tpu {

namespace {

// Bitwidth of bitcast types.
struct BitcastBitwidth {
  const int64_t from;
  const int64_t to;
};

// Convenient constructor of `BitcastBitwidth`, taking a `bitcast` as parameter.
//
// Constructs a `BitcastBitwidth` by calling `TorchEquivalentBitwidth()`
// function on both `from_type` and `to_type`.
BitcastBitwidth GetBitcastBitwidth(const RealToRealBitcast& bitcast) {
  return BitcastBitwidth{
      .from = TorchEquivalentBitwidth(bitcast.from_type),
      .to = TorchEquivalentBitwidth(bitcast.to_type),
  };
}

void CheckConversionBitwidthsAreDivisible(
    const RealToRealBitcast& bitcast, const BitcastBitwidth& bitwidth,
    const std::string_view error_message_suffix) {
  if (bitwidth.from > bitwidth.to) {
    ABSL_CHECK_EQ(  // CRASH_OK=Supported PyTorch type bitwidths are
                    // always multiple of each other.
        bitwidth.from % bitwidth.to, 0)
        << "expected the RealToRealBitcast conversion to be from a type whose "
           "bitwidth is divisible by the target type's bitwidth, got "
        << ToString(bitcast.from_type) << " bitwidth (" << bitwidth.from
        << ") is not divisible by " << ToString(bitcast.to_type)
        << " bitwidth (" << bitwidth.to << ")" << error_message_suffix;
  } else {
    ABSL_CHECK_EQ(  // CRASH_OK=Supported PyTorch type bitwidths are
                    // always multiple of each other.
        bitwidth.to % bitwidth.from, 0)
        << "expected the RealToRealBitcast conversion to be from a type whose "
           "bitwidth divides the target type's bitwidth, got "
        << ToString(bitcast.from_type) << " bitwidth (" << bitwidth.from
        << ") does not divide " << ToString(bitcast.to_type) << " bitwidth ("
        << bitwidth.to << ")" << error_message_suffix;
  }
}

void CheckStridesAreDivisible(const StridedLayout& layout,
                              const RealToRealBitcast& bitcast,
                              const int64_t size_ratio) {
  const int64_t rank = layout.strided_dims.size();
  ABSL_CHECK_GT(rank, 0);  // CRASH_OK=Enforced by caller.

  const Indices not_divisible_stride_indices =
      FilterIndices(rank - 1,
                    /* predicate= */
                    [&layout, size_ratio](const int64_t i) {
                      return layout.strided_dims[i].stride % size_ratio != 0;
                    });

  ABSL_CHECK(  // CRASH_OK=Always true for contiguous strides.
               // Non-contiguous strides never get here.
      not_divisible_stride_indices.empty())
      << "expected the RealToRealBitcast conversion input with strides "
      << ToString(GetStrides(layout))
      << " to have all but the last dimension stride divisible by "
      << size_ratio << ", which is the dtypes' bitwidth ratio, got "
      << FormatCount(not_divisible_stride_indices.size(),
                     /* singular= */ "stride that are not divisible",
                     /* plural= */ "strides that are not divisible")
      << ": "
      << GetValueAtIndexErrorStr(not_divisible_stride_indices,
                                 /* to_value =*/
                                 [&layout](const int64_t i) {
                                   return layout.strided_dims[i].stride;
                                 })
      << GetUpdateLayoutBugSuffix(bitcast, layout);
}

// Computes the shape of the tensor after a real-to-real bitcast.
// This will add a dimension if moving from a larger size to a smaller one,
// or remove a dimension if moving from a smaller size to a larger one.
Dimensions GetShapeAfterRealToRealBitcast(
    const RealToRealBitcast& bitcast, const BitcastBitwidth& bitwidth,
    absl::Span<const int64_t> shape,
    const std::string_view error_message_suffix) {
  CheckConversionBitwidthsAreDivisible(bitcast, bitwidth, error_message_suffix);

  if (bitwidth.from == bitwidth.to) {
    return CopyIntVector(shape);
  }

  if (bitwidth.from > bitwidth.to) {
    const int64_t size_ratio = bitwidth.from / bitwidth.to;

    // Element size is smaller, so there are more elements in the new type,
    // which we add as a new dimension.
    Dimensions result = CopyIntVector(shape);
    result.push_back(size_ratio);

    return result;
  }

  // bitwidth.from < bitwidth.to
  const int64_t size_ratio = bitwidth.to / bitwidth.from;

  ABSL_CHECK(  // CRASH_OK=Internal error on view decomposition.
      !shape.empty())
      << "the RealToRealBitcast conversion input cannot be a scalar"
      << error_message_suffix;

  ABSL_CHECK_EQ(  // CRASH_OK=Decomposition always create the
                  // trailing dimension needed. The inverse never gets here.
      shape.back(), size_ratio)
      << "expected the RealToRealBitcast conversion input to have its last "
         "dimension size equal "
      << size_ratio << ", which is the dtypes' bitwidth ratio ("
      << ToString(bitcast.to_type) << " / " << ToString(bitcast.from_type)
      << "), got " << shape.back() << error_message_suffix;

  // Element size is larger, so the last dimension is removed.
  return Dimensions(shape.begin(), shape.end() - 1);
}

// Computes the strides of the tensor after a real-to-real bitcast.
// This will add a dimension if moving from a larger size to a smaller one,
// or remove a dimension if moving from a smaller size to a larger one.
Strides GetStridesAfterRealToRealBitcast(const RealToRealBitcast& bitcast,
                                         const BitcastBitwidth& bitwidth,
                                         const StridedLayout& layout) {
  CheckConversionBitwidthsAreDivisible(
      bitcast, bitwidth,
      /* error_message_suffix= */ GetUpdateLayoutBugSuffix(bitcast, layout));

  Strides strides = GetStrides(layout);

  if (bitwidth.from == bitwidth.to) {
    // No change in the element size, so the strides are the same.
    return strides;
  }

  if (bitwidth.from > bitwidth.to) {
    const int64_t size_ratio = bitwidth.from / bitwidth.to;

    // New element size is smaller, so we append a new dimension with a stride
    // of 1, and scale up the existing strides by the size ratio.
    absl::c_transform(
        strides, strides.begin(),
        [size_ratio](const int64_t stride) { return stride * size_ratio; });
    strides.push_back(1);

    return strides;
  }

  // bitwidth.from < bitwidth.to
  const int64_t size_ratio = bitwidth.to / bitwidth.from;

  // The last stride must be 1 so that it is dense over the new, larger
  // element size.
  ABSL_CHECK(  // CRASH_OK=Internal error on view decomposition.
      !strides.empty())
      << "the RealToRealBitcast conversion input cannot be a scalar"
      << GetUpdateLayoutBugSuffix(bitcast, layout);

  ABSL_CHECK_EQ(  // CRASH_OK=Decomposition always create the
                  // trailing dimension needed. The inverse never gets here.
      strides.back(), 1)
      << "expected the RealToRealBitcast conversion input to have its last "
         "dimension stride equal 1, got "
      << strides.back() << GetUpdateLayoutBugSuffix(bitcast, layout);

  strides.pop_back();

  // Element size is larger, so the last dimension is removed, and all
  // existing strides are divided by the size ratio.
  CheckStridesAreDivisible(layout, bitcast, size_ratio);

  absl::c_transform(
      strides, strides.begin(),
      [size_ratio](const int64_t stride) { return stride / size_ratio; });

  return strides;
}

// Computes the storage offset of the tensor after a real-to-real bitcast.
// This may multiply or divide the storage offset by the size ratio, if the
// element size is decreased or increased respectively.
int64_t GetStorageOffsetAfterRealToRealBitcast(const RealToRealBitcast& bitcast,
                                               const BitcastBitwidth& bitwidth,
                                               const StridedLayout& layout) {
  CheckConversionBitwidthsAreDivisible(
      bitcast, bitwidth,
      /* error_message_suffix= */ GetUpdateLayoutBugSuffix(bitcast, layout));

  if (bitwidth.from == bitwidth.to) {
    return layout.storage_offset;
  }

  if (bitwidth.from > bitwidth.to) {
    const int64_t size_ratio = (bitwidth.from / bitwidth.to);
    // The new element size is smaller, so the existing storage offset covers
    // more new elements.
    return layout.storage_offset * size_ratio;
  }

  // bitwidth.from < bitwidth.to
  const int64_t size_ratio = bitwidth.to / bitwidth.from;

  ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
      layout.storage_offset % size_ratio, 0)
      << "expected the RealToRealBitcast conversion input to have its storage "
         "offset divisible by "
      << size_ratio << ", which is the dtypes' bitwidth ratio ("
      << ToString(bitcast.to_type) << " / " << ToString(bitcast.from_type)
      << "), got " << layout.storage_offset
      << GetUpdateLayoutBugSuffix(bitcast, layout);

  // The existing storage offset covers fewer new elements.
  return layout.storage_offset / size_ratio;
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

void CheckBitcastTypesAreNotComplex(const RealToRealBitcast& bitcast) {
  ABSL_CHECK(  // CRASH_OK=Internal error on view decomposition.
      bitcast.from_type != mlir::ElementType::COMPLEXF32 &&
      bitcast.from_type != mlir::ElementType::COMPLEXF64)
      << "expected the RealToRealBitcast conversion to be from a non-complex "
         "type, got "
      << ToString(bitcast.from_type) << GetViewPrimitiveErrorSuffix(bitcast);

  ABSL_CHECK(  // CRASH_OK=Internal error on view decomposition.
      bitcast.to_type != mlir::ElementType::COMPLEXF32 &&
      bitcast.to_type != mlir::ElementType::COMPLEXF64)
      << "expected the RealToRealBitcast conversion to be to a non-complex "
         "type, got "
      << ToString(bitcast.to_type) << GetViewPrimitiveErrorSuffix(bitcast);
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
bool UpdateLayout(StridedLayout& layout, const RealToRealBitcast& bitcast) {
  CheckBitcastTypesAreNotComplex(bitcast);

  if (bitcast.from_type == bitcast.to_type) {
    return false;
  }

  BitcastBitwidth bitwidth = GetBitcastBitwidth(bitcast);

  if (bitwidth.from == bitwidth.to) {
    // There's no change in the shape, but this is not a no-op because the
    // dtype is changing.
    return true;
  }

  // We're either adding or removing a dimension; need to compute the new
  // shape, strides, and storage offset.
  Dimensions new_shape = GetShapeAfterRealToRealBitcast(
      bitcast, bitwidth, GetSizes(layout),
      /* error_message_suffix= */ GetUpdateLayoutBugSuffix(bitcast, layout));
  Strides new_strides =
      GetStridesAfterRealToRealBitcast(bitcast, bitwidth, layout);
  int64_t new_storage_offset =
      GetStorageOffsetAfterRealToRealBitcast(bitcast, bitwidth, layout);

  ABSL_CHECK_EQ(  // CRASH_OK
      new_shape.size(), new_strides.size());

  // Update the layout with the new values.
  layout.storage_offset = new_storage_offset;

  layout.strided_dims.clear();
  layout.strided_dims.assign(new_shape.size(), StridedDimension{});

  for (int64_t i = 0; i < new_shape.size(); ++i) {
    layout.strided_dims[i] =
        StridedDimension{.size = new_shape[i], .stride = new_strides[i]};
  }

  return true;
}

// Updates the layout to reflect the effect of applying the given
// complex-to-real bitcast.
// Returns true if the layout was modified, or false if the bitcast is a no-op.
bool UpdateLayout(StridedLayout& layout, const ComplexToRealBitcast& bitcast) {
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
bool UpdateLayout(StridedLayout& layout, const ViewAsComplex& bitcast) {
  ABSL_CHECK(  // CRASH_OK=Internal error on view decomposition.
      !layout.strided_dims.empty())
      << "the ViewAsComplex bitcast conversion input cannot be a scalar"
      << GetUpdateLayoutBugSuffix(bitcast, layout);

  // Inverse of view_as_real; converts a tensor like f32[2, 3, 8] ->
  // cfloat[2, 3, 4].
  ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
      layout.strided_dims.back().size % 2, 0)
      << "expected the ViewAsComplex bitcast conversion input to have its last "
         "dimension size divisible by 2, got "
      << layout.strided_dims.back().size
      << GetUpdateLayoutBugSuffix(bitcast, layout);

  ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
      layout.strided_dims.back().stride, 1)
      << "expected the ViewAsComplex bitcast conversion input to have its last "
         "dimension stride equal 1, got "
      << layout.strided_dims.back().stride
      << GetUpdateLayoutBugSuffix(bitcast, layout);

  ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
      layout.storage_offset % 2, 0)
      << "expected the ViewAsComplex bitcast conversion input to have its "
         "storage offset divisible by 2, got "
      << layout.storage_offset << GetUpdateLayoutBugSuffix(bitcast, layout);

  layout.strided_dims.back().size /= 2;
  layout.storage_offset /= 2;

  // Reduce the stride of all dimensions except the last one.
  // So strides like (16, 4, 1) become (8, 2, 1).
  for (auto i = 0; i < layout.strided_dims.size() - 1; ++i) {
    auto& dim = layout.strided_dims[i];

    ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
        dim.stride % 2, 0)
        << "expected the ViewAsComplex bitcast conversion input to have all "
           "but its last dimension strides divisible by 2, got a stride of "
        << dim.stride << " at index " << i
        << GetUpdateLayoutBugSuffix(bitcast, layout);

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

  const BitcastBitwidth bitwidth = GetBitcastBitwidth(bitcast);

  // Otherwise, this is a regular real-to-real bitcast.
  // stablehlo::BitcastConvert requires specifying the full output type.
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const absl::Span<const int64_t> shape = input_type.getShape();
  // Need to propagate dynamic bound to output shape for bitcast.
  TT_RETURN_IF_ERROR(CheckStaticShape(input_type, "bitcast input"))
      << GetViewPrimitiveShloErrorSuffix(bitcast, shape,
                                         ViewPrimitiveBugSuffix::kHide);
  Dimensions result_shape = GetShapeAfterRealToRealBitcast(
      bitcast, bitwidth, input_type.getShape(),
      /* error_message_suffix= */
      GetViewPrimitiveShloErrorSuffix(bitcast, shape));
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
  const absl::Span<const int64_t> shape = input_type.getShape();

  // TODO: b/501473306 revisit after scalar tensor bitcast is supported.
  ABSL_CHECK_GT(  // CRASH_OK=Internal error on view decomposition.
      input_type.getRank(), 0)
      << "the ViewAsComplex bitcast input cannot be a scalar"
      << GetViewPrimitiveShloErrorSuffix(bitcast, shape);
  ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
      input_type.getShape().back() % 2, 0)
      << "expected the ViewAsComplex bitcast conversion input to "
         "have its last dimension size divisible by 2, got "
      << input_type.getShape().back()
      << GetViewPrimitiveShloErrorSuffix(bitcast, shape);

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
