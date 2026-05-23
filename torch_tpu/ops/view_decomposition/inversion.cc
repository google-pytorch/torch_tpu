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

#include "torch_tpu/ops/view_decomposition/inversion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/view_decomposition/bitcast_primitive.h"
#include "torch_tpu/ops/view_decomposition/broadcast_primitive.h"
#include "torch_tpu/ops/view_decomposition/conj_primitive.h"
#include "torch_tpu/ops/view_decomposition/decomposition.h"
#include "torch_tpu/ops/view_decomposition/pad_primitive.h"
#include "torch_tpu/ops/view_decomposition/reshape_primitive.h"
#include "torch_tpu/ops/view_decomposition/slice_primitive.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/transpose_primitive.h"
#include "torch_tpu/ops/view_decomposition/unfold_primitive.h"
#include "torch_tpu/ops/view_decomposition/view_primitive_error_utils.h"
#include "torch_tpu/ops/view_decomposition/view_sequence.h"

namespace torch_tpu {

namespace {

// Computes the inverse of a reshape primitive.
InverseViewPrimitive InvertViewPrimitive(const ReshapePrimitive& reshape,
                                         const StridedLayout& layout_before) {
  // The inverse of a reshape is another reshape to the original dimensions.
  return ReshapePrimitive{
      .base_sizes = reshape.new_sizes,
      .new_sizes = reshape.base_sizes,
  };
}

// Computes the inverse of a permute primitive.
InverseViewPrimitive InvertViewPrimitive(const TransposePrimitive& transpose,
                                         const StridedLayout& layout_before) {
  ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
      transpose.permutation.size(), layout_before.strided_dims.size())
      << "expected TransposePrimitive permutation size to be "
      << layout_before.strided_dims.size()
      << ", which is the rank of the input, got "
      << transpose.permutation.size()
      << "; calling InvertViewPrimitive() with layout=" << layout_before
      << " and "
      << GetViewPrimitiveErrorSuffix(transpose, {.leading_semicolon = false});

  // The inverse of a transpose is a transpose to the original order.
  // Example:
  //   Shape before: [2, 3, 4]
  //   Forward permutation: [2, 0, 1], permutes [2, 3, 4] -> [4, 2, 3]
  //   Inverse permutation: [1, 2, 0], permutes [4, 2, 3] -> [2, 3, 4]
  TransposePrimitive result;
  result.permutation.resize(transpose.permutation.size(), -1);
  for (int64_t i = 0; i < transpose.permutation.size(); ++i) {
    result.permutation[transpose.permutation[i]] = i;
  }
  return result;
}

// Error message only, for use with std::visit.
InverseViewPrimitive InvertViewPrimitive(const BroadcastPrimitive& broadcast,
                                         const StridedLayout& layout_before) {
  ABSL_LOG(FATAL)  // CRASH_OK
      << "in-place operations on overlapping views are undefined behavior "
         "and are not supported.\nIf you need to apply an in-place "
         "operation to an overlapping view, you must call .clone() or "
         ".contiguous() first.";
}

// Error message only, for use with std::visit.
InverseViewPrimitive InvertViewPrimitive(const UnfoldPrimitive& unfold,
                                         const StridedLayout& layout_before) {
  ABSL_LOG(FATAL)  // CRASH_OK
      << "in-place operations on overlapping views are undefined behavior "
         "and are not supported.\nIf you need to apply an in-place "
         "operation to an overlapping view, you must call .clone() or "
         ".contiguous() first.";
}

// Error message only, for use with std::visit.
InverseViewPrimitive InvertViewPrimitive(const SlicePrimitive& slice,
                                         const StridedLayout& layout_before) {
  ABSL_LOG(FATAL)  // CRASH_OK
      << "Cannot invert SlicesPrimitive. Slices should be handled separately. "
         "Use ComputeInverseViewOperation instead to invert a slice in context "
         "of a full view sequence inversion";
}

InverseViewPrimitive InvertViewPrimitive(const ConjPrimitive& conj,
                                         const StridedLayout& layout_before) {
  // Inversion of conj is conj.
  return ConjPrimitive{};
}

// Computes the inverse of a pad primitive.
InverseViewPrimitive InvertViewPrimitive(const PadPrimitive& pad,
                                         const StridedLayout& layout_before) {
  ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
      pad.pad_dims.size(), layout_before.strided_dims.size())
      << "expected PadPrimitive padding dimensions size to be "
      << layout_before.strided_dims.size()
      << ", which is the rank of the input, got " << pad.pad_dims.size()
      << "; calling InvertViewPrimitive() with layout=" << layout_before
      << " and "
      << GetViewPrimitiveErrorSuffix(pad, {.leading_semicolon = false});

  // The inverse of a pad is a slice.
  SlicePrimitive result;
  result.slice_dims.reserve(layout_before.strided_dims.size());
  for (size_t i = 0; i < pad.pad_dims.size(); ++i) {
    const auto& pad_dim = pad.pad_dims[i];
    const auto& dim_before = layout_before.strided_dims[i];

    const int64_t start_index = pad_dim.low_padding;
    const int64_t stride = pad_dim.interior_padding + 1;
    const int64_t limit_index =
        pad_dim.low_padding + dim_before.size +
        pad_dim.interior_padding * (dim_before.size - 1);
    result.slice_dims.push_back(SliceDimension{.start_index = start_index,
                                               .limit_index = limit_index,
                                               .stride = stride});
  }
  return result;
}

InverseViewPrimitive InvertViewPrimitive(const RealToRealBitcast& real_to_real,
                                         const StridedLayout& layout_before) {
  ABSL_LOG(FATAL)  // CRASH_OK
      << "Cannot invert RealToRealBitcast primitive. Bitcasting "
         "should be taken care of before this point.";
}
InverseViewPrimitive InvertViewPrimitive(
    const ComplexToRealBitcast& complex_to_real,
    const StridedLayout& layout_before) {
  ABSL_LOG(FATAL)  // CRASH_OK
      << "Cannot invert ComplexToRealBitcast primitive. Bitcasting "
         "should be taken care of before this point.";
}
InverseViewPrimitive InvertViewPrimitive(const ViewAsComplex& view_as_complex,
                                         const StridedLayout& layout_before) {
  ABSL_LOG(FATAL)  // CRASH_OK
      << "Cannot invert ViewAsComplex primitive. Bitcasting "
         "should be taken care of before this point.";
}

// Computes the inverse of a view primitive.
InverseViewPrimitive InvertViewPrimitive(const ViewPrimitive& primitive,
                                         const StridedLayout& layout_before) {
  return std::visit(
      [&layout_before](const auto& primitive) {
        return InvertViewPrimitive(primitive, layout_before);
      },
      primitive);
}

Dimensions GetCurrentDimensions(const StridedLayout& layout) {
  Dimensions sizes;
  sizes.reserve(layout.strided_dims.size());
  for (const auto& dim : layout.strided_dims) {
    sizes.push_back(dim.size);
  }
  return sizes;
}

// Replaces a strided slice with an equivalent sequence of pad, reshape,
// slice, reshape.
absl::Status ReplaceStridedSlice(const SlicePrimitive& strided_slice,
                                 StridedLayout& current_layout,
                                 ViewSequence& result) {
  PadPrimitive pad;
  ReshapePrimitive reshape_before;
  SlicePrimitive dense_slice;
  ReshapePrimitive reshape_after;
  for (auto i = 0; i < strided_slice.slice_dims.size(); ++i) {
    const auto& dim = strided_slice.slice_dims[i];
    if (dim.stride == 1) {
      // Not a strided dimension. No-op for pads and reshapes, slice dim is
      // unchanged.
      pad.pad_dims.push_back(PadDimension());
      reshape_before.new_sizes.push_back(current_layout.strided_dims[i].size);
      dense_slice.slice_dims.push_back(SliceDimension{
          .start_index = dim.start_index,
          .limit_index = dim.limit_index,
          .stride = 1,
      });
      reshape_after.new_sizes.push_back(dim.limit_index - dim.start_index);
    } else {
      // Pad below so the first included element is at the next highest multiple
      // of the stride.
      const auto new_start_index =
          (dim.start_index + dim.stride - 1) / dim.stride;
      const auto low_padding = new_start_index * dim.stride - dim.start_index;

      // Pad above so that the total number of elements (including low padding)
      // is a multiple of the stride, and is at least
      // (new_start_index + post_slice_size) * dim.stride.
      const auto pre_slice_size = current_layout.strided_dims[i].size;
      const auto post_slice_size =
          (dim.limit_index - dim.start_index) / dim.stride + 1;
      const auto total_padded_size =
          std::max((pre_slice_size + low_padding + dim.stride - 1) / dim.stride,
                   new_start_index + post_slice_size) *
          dim.stride;
      const auto high_padding =
          total_padded_size - pre_slice_size - low_padding;
      pad.pad_dims.push_back(PadDimension{
          .low_padding = low_padding,
          .high_padding = high_padding,
          .interior_padding = 0,
      });

      // Reshape from total_padded_size to
      //  (total_padded_size / dim.stride, dim.stride) to set up the correct
      // stride.
      reshape_before.new_sizes.push_back(total_padded_size / dim.stride);
      reshape_before.new_sizes.push_back(dim.stride);

      // Slice the first reshaped dimension as
      // [new_start_index::new_start_index + post_slice_size]
      // and the second as [0::1] to remove the padding and original
      // interior elements.
      dense_slice.slice_dims.push_back(SliceDimension{
          .start_index = new_start_index,
          .limit_index = new_start_index + post_slice_size,
          .stride = 1,
      });
      dense_slice.slice_dims.push_back(SliceDimension{
          .start_index = 0,
          .limit_index = 1,
          .stride = 1,
      });

      // Flatten from (post_slice_size, 1) to just post_slice_size on this
      // dimension.
      reshape_after.new_sizes.push_back(post_slice_size);
    }
  }
  // Apply and append the replacement primitives.
  UpdateLayout(current_layout, pad);
  reshape_before.base_sizes = GetCurrentDimensions(current_layout);
  UpdateLayout(current_layout, reshape_before);
  UpdateLayout(current_layout, dense_slice);
  reshape_after.base_sizes = GetCurrentDimensions(current_layout);
  UpdateLayout(current_layout, reshape_after);
  result.push_back(std::move(pad));
  result.push_back(std::move(reshape_before));
  result.push_back(std::move(dense_slice));
  result.push_back(std::move(reshape_after));
  return absl::OkStatus();
}

// If a view sequence contains a strided slice (stride > 1 on at least one
// dimension), inverting it is somewhat complicated and slow to compile, as it
// would require a stablehlo.select operation with a boolean mask to select
// which elements of the contiguous base to overwrite.
//
// However, a strided slice can also be avoided by using padding, reshapes,
// and dense/non-strided slices. This is not necessarily advantageous in the
// forward direction, as pads and reshapes are costly, but can be much more
// efficiently inverted using dense slices, reshapes, and dynamic_update_slice.
//
// Strided slices are rare in practice so this is frequently a no-op.
absl::Status ReplaceStridedSlices(
    absl::Span<const int64_t> contiguous_base_shape,
    ViewSequence& view_sequence) {
  ViewSequence result;  // Intentionally not reserving memory, likely unused
  bool has_strided_slice = false;
  // Keep track of the current layout so we know the shape before each slice.
  StridedLayout current_layout =
      MakeContiguousBaseLayout(contiguous_base_shape);

  for (auto i = 0; i < view_sequence.size(); ++i) {
    auto* slice = std::get_if<SlicePrimitive>(&view_sequence[i]);
    if (slice == nullptr ||
        std::all_of(
            slice->slice_dims.begin(), slice->slice_dims.end(),
            [](const SliceDimension& dim) { return dim.stride == 1; })) {
      // Either this is not a slice, or it's a non-strided slice.
      // No need to replace, just update the layout and retain the original
      // primitive.
      UpdateLayout(current_layout, view_sequence[i]);
      if (has_strided_slice) {
        result.push_back(std::move(view_sequence[i]));
      }
      continue;
    }

    // This is a strided slice.
    // If it's the first strided slice, we need to move the previous
    // non-strided-pad primitives into the result.
    if (!has_strided_slice) {
      for (auto j = 0; j < i; ++j) {
        result.push_back(std::move(view_sequence[j]));
      }
      has_strided_slice = true;
    }

    // Construct the replacement primitives for the strided slice.
    TT_RETURN_IF_ERROR(ReplaceStridedSlice(*slice, current_layout, result));
  }

  // If we never saw a strided slice, then this function is a no-op.
  if (has_strided_slice) {
    std::swap(view_sequence, result);
  }
  return absl::OkStatus();
}

}  // namespace

bool operator==(const InverseViewPrimitive& lhs,
                const InverseViewPrimitive& rhs) {
  if (lhs.index() != rhs.index()) {
    return false;
  }
  if (std::holds_alternative<ReshapePrimitive>(lhs)) {
    return std::get<ReshapePrimitive>(lhs).new_sizes ==
           std::get<ReshapePrimitive>(rhs).new_sizes;
  }
  if (std::holds_alternative<TransposePrimitive>(lhs)) {
    return std::get<TransposePrimitive>(lhs).permutation ==
           std::get<TransposePrimitive>(rhs).permutation;
  }
  if (std::holds_alternative<SlicePrimitive>(lhs)) {
    return std::get<SlicePrimitive>(lhs).slice_dims ==
           std::get<SlicePrimitive>(rhs).slice_dims;
  }
  if (std::holds_alternative<ConjPrimitive>(lhs)) {
    return true;
  }
  ABSL_CHECK(false) << "Unknown view primitive";  // CRASH_OK
}

std::ostream& operator<<(std::ostream& os,
                         const InverseViewPrimitive& primitive) {
  // For some reason, std::visit doesn't work here.
  if (std::holds_alternative<ReshapePrimitive>(primitive)) {
    return os << std::get<ReshapePrimitive>(primitive);
  }
  if (std::holds_alternative<TransposePrimitive>(primitive)) {
    return os << std::get<TransposePrimitive>(primitive);
  }
  if (std::holds_alternative<SlicePrimitive>(primitive)) {
    return os << std::get<SlicePrimitive>(primitive);
  }
  if (std::holds_alternative<ConjPrimitive>(primitive)) {
    return os << "conj()";
  }
  ABSL_CHECK(false) << "Unknown view primitive";  // CRASH_OK
}

bool operator==(const InverseViewStage& lhs, const InverseViewStage& rhs) {
  if (lhs.forward != rhs.forward) {
    return false;
  }
  if (lhs.slice.has_value() != rhs.slice.has_value()) {
    return false;
  }
  if (lhs.slice.has_value() &&
      lhs.slice.value().slice_dims != rhs.slice.value().slice_dims) {
    return false;
  }
  if (lhs.inverse != rhs.inverse) {
    return false;
  }
  return true;
}

std::ostream& operator<<(std::ostream& os, const InverseViewStage& stage) {
  os << "Stage(forward: " << ToString(stage.forward);
  if (stage.slice.has_value()) {
    os << ", slice: " << stage.slice.value();
  } else {
    os << ", slice: none";
  }
  return os << ", inverse: " << ToString(stage.inverse) << ")";
}

bool UpdateLayout(StridedLayout& layout,
                  const InverseViewPrimitive& primitive) {
  return std::visit(
      [&layout](const auto& primitive) -> bool {
        if constexpr (std::is_same_v<decltype(primitive),
                                     const ConjPrimitive&>) {
          return true;
        } else {
          return UpdateLayout(layout, primitive);
        }
      },
      primitive);
}

absl::StatusOr<mlir::MlirOp> InverseViewPrimitiveShlo(
    mlir::MlirOp input, const InverseViewPrimitive& primitive) {
  return std::visit(
      [&](const auto& primitive) {
        return ViewPrimitiveShlo(input, ViewPrimitive(primitive));
      },
      primitive);
}

absl::StatusOr<InverseViewSequence> InvertViewSequence(
    StridedLayout& layout, absl::Span<const ViewPrimitive> view_sequence) {
  std::vector<InverseViewPrimitive> result_sequence(view_sequence.size());
  size_t write_index = result_sequence.size() - 1;
  for (const auto& primitive : view_sequence) {
    result_sequence[write_index] = InvertViewPrimitive(primitive, layout);
    UpdateLayout(layout, primitive);
    --write_index;
  }
  ABSL_VLOG(3) << "[InvertViewSequence] Inverted view sequence:"
               << "\nView sequence: " << ToString(view_sequence)
               << "\nInverted view sequence: " << ToString(result_sequence);
  return result_sequence;
}

absl::StatusOr<mlir::MlirOp> InverseViewSequenceShlo(
    mlir::MlirOp mutated_view, InverseViewSequenceSpan inverted_view_sequence) {
  mlir::MlirOp result = mutated_view;
  for (const auto& primitive : inverted_view_sequence) {
    TT_ASSIGN_OR_RETURN(result, InverseViewPrimitiveShlo(result, primitive));
  }
  return result;
}

namespace {

// Gets a common dtype that the base and view can both be bitcasted into which
// will preserve all data in a view -> base update.
mlir::ElementType GetWritableDtype(const mlir::ElementType base_dtype,
                                   const mlir::ElementType view_dtype) {
  if (base_dtype == view_dtype) {
    // Neither base nor view needs to be bitcasted.
    // This is fine even when base and view are both booleans; no data is lost
    // from base on assignment, since the values were already 0x00 or 0x01.
    return base_dtype;
  }

  const int64_t base_bitwidth = TorchEquivalentBitwidth(base_dtype);
  const int64_t view_bitwidth = TorchEquivalentBitwidth(view_dtype);

  if (view_bitwidth < base_bitwidth) {
    ABSL_CHECK_EQ(  // CRASH_OK=Supported PyTorch type bitwidths are
                    // always multiple of each other.
        base_bitwidth % view_bitwidth, 0)
        << "expected the bitwidth of the base tensor dtype ("
        << ToString(base_dtype)
        << ") to be divisible by the bitwidth of the view tensor dtype ("
        << ToString(view_dtype) << "), got " << base_bitwidth
        << " is not divisible by " << view_bitwidth
        << "; calling GetWritableDtype(); this is a TorchTPU bug";

    if (view_dtype == mlir::ElementType::PRED) {
      // Any bytes in the base that are not 0x00 or 0x01 should be preserved,
      // so we can't cast base to boolean.
      // Instead we cast both base and view to uint8.
      return mlir::ElementType::UI8;
    }

    // base will be bitcasted into view's dtype losslessly.
    return view_dtype;
  }

  ABSL_CHECK_EQ(  // CRASH_OK=Supported PyTorch type bitwidths are
                  // always multiple of each other.
      view_bitwidth % base_bitwidth, 0)
      << "expected the bitwidth of the view tensor dtype ("
      << ToString(view_dtype)
      << ") to be divisible by the bitwidth of the base tensor dtype ("
      << ToString(base_dtype) << "), got " << view_bitwidth
      << " is not divisible by " << base_bitwidth
      << "; calling GetWritableDtype(); this is a TorchTPU bug";

  if (base_dtype == mlir::ElementType::PRED) {
    // view will be writing bytes other than 0x00 or 0x01 to base, so we need
    // to promote the base to UI8 to preserve the full value of written bytes.
    return mlir::ElementType::UI8;
  }
  // view will be bitcasted into base's dtype losslessly.
  return base_dtype;
}

// Gets a view sequence to bitcast from the given dtype to a writable dtype.
// The bitwidth of from_dtype must be greater than or equal to the bitwidth of
// to_dtype, and the bitwidths must be divisible.
// The layout is updated to reflect the effect of the bitcast.
ViewSequence GetBitcastSequenceToWritable(mlir::ElementType from_dtype,
                                          mlir::ElementType to_dtype,
                                          StridedLayout& layout) {
  if (from_dtype == to_dtype) {
    return {};
  }
  ViewSequence result;

  // The full bitcast may be complex_to_real + real_to_real + real_to_complex.
  // Handle the complex_to_real component first (if needed).
  if (from_dtype == mlir::ElementType::COMPLEXF32 ||
      from_dtype == mlir::ElementType::COMPLEXF64) {
    auto complex_element_type = from_dtype == mlir::ElementType::COMPLEXF32
                                    ? ComplexElementType::kComplexFloat
                                    : ComplexElementType::kComplexDouble;
    auto real_component_type = from_dtype == mlir::ElementType::COMPLEXF32
                                   ? mlir::ElementType::F32
                                   : mlir::ElementType::F64;
    auto complex_to_real = ComplexToRealBitcast{
        .complex_element_type = complex_element_type,
        .bitcast_type = ComplexToRealBitcastType::kViewAsReal};
    UpdateLayout(layout, complex_to_real);
    result.push_back(std::move(complex_to_real));
    from_dtype = real_component_type;
  }

  // Then any real-to-real bitcast (if needed).
  mlir::ElementType to_dtype_real = RealComponentOf(to_dtype);
  if (from_dtype != to_dtype_real) {
    auto real_to_real =
        RealToRealBitcast{.from_type = from_dtype, .to_type = to_dtype_real};
    UpdateLayout(layout, real_to_real);
    result.push_back(std::move(real_to_real));
    from_dtype = to_dtype_real;
  }

  // Finally, any real-to-complex bitcast (if needed).
  if (to_dtype_real == to_dtype) {
    return result;
  }

  // to_dtype should never be COMPLEX64 here.
  // COMPLEX64 is the largest dtype, so to_dtype = COMPLEX64 &&
  // bitwdith(from_dtype) >= bitwidth(to_dtype) is only satisfied if
  // from_dtype == to_dtype, which is covered by the == check at the start of
  // this function.
  ABSL_CHECK_NE(to_dtype, mlir::ElementType::COMPLEXF64);  // CRASH_OK

  // If to_dtype is COMPLEXF32 which has a bitwidth of 64, then from_dtype
  // must be either COMPLEX64 or a 64-bit element type (like UI64).
  // If from_dtype was COMPLEX64, then we should have added a ViewAsReal
  // followed by a RealToReal F64->F32 to get here, which means layout should
  // be [*shape, 2, 2].
  // If from_dypt was a non-complex 64-bit element type, then we should have
  // added a RealToReal {64-bit type}->F32 to get here, which means layout
  // should be [*shape, 2].
  // Either way, no reshape should be necessary to create a trailing size-2
  // dense dimension.
  ABSL_CHECK(!layout.strided_dims.empty());             // CRASH_OK
  ABSL_CHECK_EQ(layout.strided_dims.back().size, 2);    // CRASH_OK
  ABSL_CHECK_EQ(layout.strided_dims.back().stride, 1);  // CRASH_OK
  auto view_as_complex =
      ViewAsComplex{.complex_element_type = ComplexElementType::kComplexFloat};
  UpdateLayout(layout, view_as_complex);
  result.push_back(std::move(view_as_complex));

  return result;
}

mlir::MlirOp InvertNonStridedSliceShlo(mlir::MlirOp pre_slice_value,
                                       mlir::MlirOp post_slice_value,
                                       const SlicePrimitive& slice) {
  // Use dynamic_update_slice to invert non-strided slices.
  std::vector<mlir::MlirOp> start_indices;
  start_indices.reserve(slice.slice_dims.size());
  for (int64_t i = 0; i < slice.slice_dims.size(); ++i) {
    const auto& slice_dim = slice.slice_dims[i];

    ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
        slice_dim.stride, 1)
        << "expected SlicePrimitive slice dimensions to have stride = 1, got "
        << slice_dim.stride << " at index " << i
        << "; calling InvertNonStridedSliceShlo() with "
        << GetViewPrimitiveErrorSuffix(slice, {.leading_semicolon = false});

    start_indices.push_back(MakeScalarConstant(pre_slice_value.getBuilder(),
                                               slice_dim.start_index,
                                               mlir::ElementType::I64));
  }
  return mlir::stablehlo::DynamicUpdateSlice(
      /*operand=*/pre_slice_value, /*update=*/post_slice_value, start_indices);
}

}  // namespace

absl::StatusOr<InverseViewOperation> ComputeInverseViewOperation(
    absl::Span<const int64_t> contiguous_base_shape,
    mlir::ElementType contiguous_base_dtype, const StridedLayout& view_layout,
    mlir::ElementType view_dtype, bool is_conj) {
  // Get the common dtype to bitcast base and view into.
  mlir::ElementType writable_dtype =
      GetWritableDtype(contiguous_base_dtype, view_dtype);

  // Cast the contiguous base to the writable dtype and update the shape
  // accordingly.
  ViewSequence base_transform;
  StridedLayout layout = MakeContiguousBaseLayout(contiguous_base_shape);
  base_transform = GetBitcastSequenceToWritable(contiguous_base_dtype,
                                                writable_dtype, layout);

  // Initially set the final shape to the shape after the bitcast.
  Shape final_shape({}, writable_dtype);
  final_shape.dimensions().reserve(layout.strided_dims.size());
  for (const auto& dim : layout.strided_dims) {
    final_shape.dimensions().push_back(dim.size);
  }

  // Cast the view to the writable dtype and update the layout accordingly.
  StridedLayout view_layout_after_bitcast = view_layout;
  ViewSequence bitcast_view = GetBitcastSequenceToWritable(
      view_dtype, writable_dtype, view_layout_after_bitcast);

  // Compute the view sequence that would go from the base to the view...
  ViewSequence base_to_view;
  {
    Dimensions contiguous_base_shape_after_bitcast;
    contiguous_base_shape_after_bitcast.reserve(layout.strided_dims.size());
    for (const auto& dim : layout.strided_dims) {
      contiguous_base_shape_after_bitcast.push_back(dim.size);
    }
    TT_ASSIGN_OR_RETURN(
        base_to_view, DecomposeIntoViewSequence(
                          contiguous_base_shape_after_bitcast, writable_dtype,
                          view_layout_after_bitcast, writable_dtype, is_conj));
    // Don't use strided slices here so the inversion is more efficient.
    TT_RETURN_IF_ERROR(ReplaceStridedSlices(contiguous_base_shape_after_bitcast,
                                            base_to_view));
    Simplify(base_to_view, contiguous_base_shape_after_bitcast);
  }

  // ...and partition it into stages.
  std::vector<InverseViewStage> stages;
  stages.push_back(InverseViewStage());
  for (auto i = 0; i < base_to_view.size(); ++i) {
    if (i == 0 && std::holds_alternative<ReshapePrimitive>(base_to_view[i])) {
      // An initial reshape can be folded into the base transform, since it
      // retains contiguity. This also changes the expected final shape.
      UpdateLayout(layout, base_to_view[i]);
      final_shape.dimensions().clear();
      for (const auto& dim : layout.strided_dims) {
        final_shape.dimensions().push_back(dim.size);
      }
      base_transform.push_back(std::move(base_to_view[i]));
    } else if (SlicePrimitive* slice =
                   std::get_if<SlicePrimitive>(&base_to_view[i])) {
      // A slice ends the current stage and starts the next one.

      // Compute the inverse of this stage's forward sequence (excluding the
      // slice), updating the layout to the pre-slice layout.
      TT_ASSIGN_OR_RETURN(InverseViewSequence inverse,
                          InvertViewSequence(layout, stages.back().forward));
      stages.back().inverse = std::move(inverse);

      // Update the layout to the post-slice layout and assign the slice.
      UpdateLayout(layout, *slice);
      stages.back().slice = std::move(*slice);

      // Start a new empty stage.
      stages.push_back(InverseViewStage());
    } else {
      // Otherwise, we extend the current stage.
      stages.back().forward.push_back(std::move(base_to_view[i]));
    }
  }

  // Invert the final stage.
  TT_ASSIGN_OR_RETURN(InverseViewSequence final_stage_inverse,
                      InvertViewSequence(layout, stages.back().forward));
  stages.back().inverse = std::move(final_stage_inverse);

  return InverseViewOperation{
      .base_transform = std::move(base_transform),
      .bitcast_view = std::move(bitcast_view),
      .stages = std::move(stages),
      .final_shape = std::move(final_shape),
  };
}

absl::StatusOr<mlir::MlirOp> InverseViewOperationShlo(
    mlir::MlirOp contiguous_base, mlir::MlirOp mutated_view,
    const InverseViewOperation& inverse_view_operation) {
  // Apply the base transform to the contiguous base.
  // After this transform, the value will still be a contiguous base, possibly
  // with a different dtype and shape.
  TT_ASSIGN_OR_RETURN(
      contiguous_base,
      ViewSequenceShlo(contiguous_base, inverse_view_operation.base_transform));

  // Run the view operation forward to get the original values before each
  // slice in the original view sequence.
  // The last stage doesn't have a slice.
  std::vector<mlir::MlirOp> pre_slice_values;
  pre_slice_values.reserve(inverse_view_operation.stages.size() - 1);
  for (auto i = 0; i < inverse_view_operation.stages.size() - 1; ++i) {
    mlir::MlirOp before_stage;
    if (i == 0) {
      before_stage = contiguous_base;
    } else {
      ABSL_CHECK(  // CRASH_OK=Internal error on view decomposition.
          inverse_view_operation.stages[i - 1].slice.has_value())
          << "the InverseViewOperation non-final stage (" << i - 1
          << ") must have a slice primitive; calling "
             "InverseViewOperationShlo(); this is a TorchTPU bug";
      TT_ASSIGN_OR_RETURN(
          before_stage,
          ViewPrimitiveShlo(pre_slice_values.back(),
                            *inverse_view_operation.stages[i - 1].slice));
    }
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp pre_slice,
        ViewSequenceShlo(before_stage,
                         inverse_view_operation.stages[i].forward));
    pre_slice_values.push_back(std::move(pre_slice));
  }

  // Run each stage backwards.
  mlir::MlirOp curr_op = mutated_view;
  for (auto i = inverse_view_operation.stages.size() - 1; i >= 0; --i) {
    // Apply the inverse view sequence to the post-slice value.
    TT_ASSIGN_OR_RETURN(curr_op,
                        InverseViewSequenceShlo(
                            curr_op, inverse_view_operation.stages[i].inverse));

    // Get the base value before the slice, if any.
    if (i == 0) {
      break;
    }
    mlir::MlirOp pre_slice_value = pre_slice_values.back();
    pre_slice_values.pop_back();

    // Merge the current op's values with the pre-slice value.
    const auto& slice = *inverse_view_operation.stages[i - 1].slice;

    curr_op = InvertNonStridedSliceShlo(pre_slice_value, curr_op, slice);
  }
  return curr_op;
}

}  // namespace torch_tpu
