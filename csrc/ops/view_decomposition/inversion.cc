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

#include "csrc/ops/view_decomposition/inversion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
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
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/view_decomposition/bitcast_primitive.h"
#include "csrc/ops/view_decomposition/broadcast_primitive.h"
#include "csrc/ops/view_decomposition/conj_primitive.h"
#include "csrc/ops/view_decomposition/decomposition.h"
#include "csrc/ops/view_decomposition/pad_primitive.h"
#include "csrc/ops/view_decomposition/reshape_primitive.h"
#include "csrc/ops/view_decomposition/slice_primitive.h"
#include "csrc/ops/view_decomposition/strided_layout.h"
#include "csrc/ops/view_decomposition/transpose_primitive.h"
#include "csrc/ops/view_decomposition/unfold_primitive.h"
#include "csrc/ops/view_decomposition/view_primitive_error_utils.h"
#include "csrc/ops/view_decomposition/view_sequence.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

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
  ABSL_CHECK_NE(  // CRASH_OK=Internal error on view decomposition.
      real_to_real.to_type, mlir::ElementType::PRED)
      << "cannot invert a bitcast into boolean (PRED) type as this is a lossy "
         "operation";
  return RealToRealBitcast{
      .from_type = real_to_real.to_type,
      .to_type = real_to_real.from_type,
  };
}
InverseViewPrimitive InvertViewPrimitive(
    const ComplexToRealBitcast& complex_to_real,
    const StridedLayout& layout_before) {
  ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
      complex_to_real.bitcast_type, ComplexToRealBitcastType::kViewAsReal)
      << "cannot invert the real() or imag() functions as they are lossy "
         "operations";
  return ViewAsComplex{
      .complex_element_type = complex_to_real.complex_element_type,
  };
}
InverseViewPrimitive InvertViewPrimitive(const ViewAsComplex& view_as_complex,
                                         const StridedLayout& layout_before) {
  return ComplexToRealBitcast{
      .complex_element_type = view_as_complex.complex_element_type,
      .bitcast_type = ComplexToRealBitcastType::kViewAsReal,
  };
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
  if (std::holds_alternative<ComplexToRealBitcast>(lhs)) {
    return std::get<ComplexToRealBitcast>(lhs) ==
           std::get<ComplexToRealBitcast>(rhs);
  }
  if (std::holds_alternative<RealToRealBitcast>(lhs)) {
    return std::get<RealToRealBitcast>(lhs) == std::get<RealToRealBitcast>(rhs);
  }
  if (std::holds_alternative<ViewAsComplex>(lhs)) {
    return std::get<ViewAsComplex>(lhs) == std::get<ViewAsComplex>(rhs);
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
  if (std::holds_alternative<ComplexToRealBitcast>(primitive)) {
    return os << std::get<ComplexToRealBitcast>(primitive);
  }
  if (std::holds_alternative<RealToRealBitcast>(primitive)) {
    return os << std::get<RealToRealBitcast>(primitive);
  }
  if (std::holds_alternative<ViewAsComplex>(primitive)) {
    return os << std::get<ViewAsComplex>(primitive);
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

InverseViewSequence InvertViewSequence(
    StridedLayout& layout, absl::Span<const ViewPrimitive> view_sequence) {
  size_t num_complex_bitcasts = 0;
  for (const auto& primitive : view_sequence) {
    if (std::holds_alternative<ViewAsComplex>(primitive) ||
        std::holds_alternative<ComplexToRealBitcast>(primitive)) {
      ++num_complex_bitcasts;
    }
  }

  std::vector<InverseViewPrimitive> result_sequence(view_sequence.size() +
                                                    num_complex_bitcasts);
  size_t write_index = result_sequence.size() - 1;
  for (const auto& primitive : view_sequence) {
    if (std::holds_alternative<ViewAsComplex>(primitive) ||
        std::holds_alternative<ComplexToRealBitcast>(primitive)) {
      // ViewAsReal and ViewAsComplex are not 1:1 inverses.
      // ViewAsReal will always append a new dimension of size 2, but
      // ViewAsComplex just divides the last dimension by 2.
      // This means that a round-trip needs an extra reshape, like:
      //   cfloat[10] -> ViewAsReal -> f32[10, 2] -> ViewAsComplex
      //     -> cfloat[10, 1] -> Reshape -> cfloat[10]
      // or
      //   f32[10, 2] -> ViewAsComplex -> cfloat[10, 1] -> ViewAsReal
      //     -> f32[10, 1, 2] -> Reshape -> f32[10, 2]
      Dimensions restored_shape = GetSizes(layout);
      auto bitcast = InvertViewPrimitive(primitive, layout);
      UpdateLayout(layout, primitive);
      StridedLayout layout_copy = layout;
      UpdateLayout(layout_copy, bitcast);
      Dimensions actual_shape = GetSizes(layout_copy);
      ReshapePrimitive restore_shape = ReshapePrimitive{
          .base_sizes = std::move(actual_shape),
          .new_sizes = std::move(restored_shape),
      };
      result_sequence[write_index] = std::move(restore_shape);
      --write_index;
      result_sequence[write_index] = std::move(bitcast);
      --write_index;
    } else {
      result_sequence[write_index] = InvertViewPrimitive(primitive, layout);
      UpdateLayout(layout, primitive);
      --write_index;
    }
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

// Inverts a non-strided slice update along a single dimension by concatenating
// the unmodified left prefix (`0..start_index`), the updated slice value
// (`post_slice_value`), and the unmodified right suffix
// (`limit_index..total_size`).
//
// Used specifically when the sliced dimension size exceeds `INT32_MAX`, where
// XLA's `DynamicUpdateSlice` index representation or hardware dynamic indexing
// limits can overflow 32-bit offsets.
mlir::MlirOp InvertSingleDimSliceViaConcat(
    mlir::MlirOp pre_slice_value, mlir::MlirOp post_slice_value,
    int64_t active_dim, int64_t start_index, int64_t limit_index,
    const mlir::stablehlo::Dimensions& pre_slice_dims) {
  const int64_t rank = pre_slice_dims.size();
  const int64_t total_size = pre_slice_dims[active_dim].size;
  std::vector<mlir::MlirOp> parts;
  Dimensions strides(rank, 1);

  if (start_index > 0) {
    Dimensions starts(rank, 0);
    Dimensions limits;
    limits.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
      limits.push_back(i == active_dim ? start_index : pre_slice_dims[i].size);
    }
    parts.push_back(
        mlir::stablehlo::Slice(pre_slice_value, starts, limits, strides));
  }
  parts.push_back(post_slice_value);
  if (limit_index < total_size) {
    Dimensions starts(rank, 0);
    starts[active_dim] = limit_index;
    Dimensions limits;
    limits.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
      limits.push_back(pre_slice_dims[i].size);
    }
    parts.push_back(
        mlir::stablehlo::Slice(pre_slice_value, starts, limits, strides));
  }
  if (parts.size() == 1) {
    return parts[0];
  }
  return mlir::stablehlo::Concatenate(pre_slice_value.getBuilder(), parts,
                                      /*dimension=*/active_dim);
}

mlir::MlirOp InvertNonStridedSliceShlo(mlir::MlirOp pre_slice_value,
                                       mlir::MlirOp post_slice_value,
                                       const SlicePrimitive& slice) {
  const mlir::stablehlo::Dimensions pre_slice_dims =
      GetDimensions(pre_slice_value);

  bool has_dynamic_dim = false;
  int64_t active_slice_dim = -1;
  int64_t num_sliced_dims = 0;
  for (int64_t i = 0; i < slice.slice_dims.size(); ++i) {
    const auto& slice_dim = slice.slice_dims[i];
    ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
        slice_dim.stride, 1)
        << "expected SlicePrimitive slice dimensions to have stride = 1, got "
        << slice_dim.stride << " at index " << i
        << "; calling InvertNonStridedSliceShlo() with "
        << GetViewPrimitiveErrorSuffix(slice, {.leading_semicolon = false});
    if (pre_slice_dims[i].boundOp.has_value()) {
      has_dynamic_dim = true;
    }
    if (slice_dim.start_index > 0 ||
        slice_dim.limit_index < pre_slice_dims[i].size) {
      active_slice_dim = i;
      ++num_sliced_dims;
    }
  }

  // Note on XLA buffer aliasing (`DynamicUpdateSlice` vs `Slice +
  // Concatenate`): For standard dimensions (`size <= INT32_MAX`), we must emit
  // `stablehlo.dynamic_update_slice` rather than 3-way `Slice + Concatenate`.
  // In sequential loops that mutate slices of an output buffer step-by-step
  // (e.g. `output[:, t, :] = h_t` across timesteps in RNNs/LSTMs),
  // `DynamicUpdateSlice` has a single read operand (`pre_slice_value`) that XLA
  // buffer assignment aliases in-place with the output buffer. A 3-way
  // `Concatenate(Slice(output, 0..t), h_t, Slice(output, t+1..L))` reads
  // `output` via two distinct `Slice` operations, preventing XLA from aliasing
  // `output` in-place and doubling memory consumption.
  const int64_t target_dim = active_slice_dim >= 0 ? active_slice_dim : 0;
  if (!has_dynamic_dim && num_sliced_dims <= 1 && !pre_slice_dims.empty() &&
      pre_slice_dims[target_dim].size > std::numeric_limits<int32_t>::max()) {
    const auto& slice_dim = slice.slice_dims[target_dim];
    return InvertSingleDimSliceViaConcat(pre_slice_value, post_slice_value,
                                         target_dim, slice_dim.start_index,
                                         slice_dim.limit_index, pre_slice_dims);
  }

  // Use dynamic_update_slice to invert non-strided slices.
  std::vector<mlir::MlirOp> start_indices;
  start_indices.reserve(slice.slice_dims.size());
  for (int64_t i = 0; i < slice.slice_dims.size(); ++i) {
    const auto& slice_dim = slice.slice_dims[i];
    start_indices.push_back(MakeScalarConstant(pre_slice_value.getBuilder(),
                                               slice_dim.start_index,
                                               mlir::ElementType::I64));
  }
  return mlir::stablehlo::DynamicUpdateSlice(
      /*operand=*/pre_slice_value, /*update=*/post_slice_value, start_indices);
}

struct BitcastTransforms {
  ViewSequence base_transform;
  mlir::ElementType base_dtype_after_transform;
  ViewSequence bitcast_view;
  mlir::ElementType writable_dtype;
  ViewSequence invertible_bitcast;
};
BitcastTransforms GetBitcastTransforms(
    absl::Span<const int64_t> contiguous_base_shape,
    mlir::ElementType contiguous_base_dtype, const StridedLayout& view_layout,
    mlir::ElementType view_dtype) {
  if (contiguous_base_dtype == view_dtype) {
    // No bitcasts are necessary.
    return BitcastTransforms{
        .base_transform = {},
        .base_dtype_after_transform = contiguous_base_dtype,
        .bitcast_view = {},
        .writable_dtype = contiguous_base_dtype,
        .invertible_bitcast = {},
    };
  }

  StridedLayout base_layout_after_transform =
      MakeContiguousBaseLayout(contiguous_base_shape);
  mlir::ElementType base_dtype_after_transform = contiguous_base_dtype;
  ViewSequence base_transform;

  if (contiguous_base_dtype == mlir::ElementType::PRED) {
    // We must transform the base to UI8 to avoid losing data, as XLA stores
    // only 1 bit per boolean while PyTorch expects 1 byte.
    base_transform.push_back(
        RealToRealBitcast{.from_type = mlir::ElementType::PRED,
                          .to_type = mlir::ElementType::UI8});
    UpdateLayout(base_layout_after_transform, base_transform.back());
    base_dtype_after_transform = mlir::ElementType::UI8;
  }

  if (base_dtype_after_transform == view_dtype) {
    // Writing bytes (UI8) to bits, only the base transform is needed.
    return BitcastTransforms{
        .base_transform = std::move(base_transform),
        .base_dtype_after_transform = base_dtype_after_transform,
        .bitcast_view = {},
        .writable_dtype = base_dtype_after_transform,
        .invertible_bitcast = {},
    };
  }

  const int64_t base_bitwidth =
      TorchEquivalentBitwidth(base_dtype_after_transform);
  const int64_t view_bitwidth = TorchEquivalentBitwidth(view_dtype);

  if (view_bitwidth >= base_bitwidth) {
    // We can losslessly transform the view to the base dtype.
    StridedLayout view_layout_after_transform = view_layout;
    auto bitcast_view = GetBitcastSequenceToWritable(
        view_dtype, base_dtype_after_transform, view_layout_after_transform);
    return BitcastTransforms{
        .base_transform = std::move(base_transform),
        .base_dtype_after_transform = base_dtype_after_transform,
        .bitcast_view = std::move(bitcast_view),
        .writable_dtype = base_dtype_after_transform,
        .invertible_bitcast = {},
    };
  }

  // base_bitwidth > view_bitwidth. If the view dtype is not PRED, we can do an
  // invertible bitcast from base -> view dtypes. But if view dtype is PRED,
  // this would not be invertible, so we need to align on UI8 as the writable
  // dtype.
  mlir::ElementType writable_dtype = view_dtype;
  ViewSequence bitcast_view;
  if (view_dtype == mlir::ElementType::PRED) {
    writable_dtype = mlir::ElementType::UI8;
    StridedLayout view_layout_after_transform = view_layout;
    bitcast_view = GetBitcastSequenceToWritable(view_dtype, writable_dtype,
                                                view_layout_after_transform);
  }

  // Cast from the base dtype to the writable dtype, do the write in that dtype,
  // and then transform it back to the base dtype.
  StridedLayout layout_copy = base_layout_after_transform;
  auto invertible_bitcast = GetBitcastSequenceToWritable(
      base_dtype_after_transform, writable_dtype, layout_copy);
  return BitcastTransforms{
      .base_transform = std::move(base_transform),
      .base_dtype_after_transform = base_dtype_after_transform,
      .bitcast_view = std::move(bitcast_view),
      .writable_dtype = writable_dtype,
      .invertible_bitcast = std::move(invertible_bitcast),
  };
}

// Computes a bitcast-free view sequence from the post-bitcast base to the
// post-bitcast view that is suitable for view inversion.
absl::StatusOr<ViewSequence> CalculatePostBitcastBaseToView(
    const StridedLayout& base_layout, const StridedLayout& view_layout,
    ViewSequenceSpan bitcast_view, ViewSequence&& invertible_bitcast,
    mlir::ElementType writable_dtype, bool is_conj) {
  ViewSequence base_to_view;
  // Get the base and view layouts, after they have both been bitcast
  // to the writable dtype.
  StridedLayout base_layout_after_bitcast = base_layout;
  for (const auto& primitive : invertible_bitcast) {
    UpdateLayout(base_layout_after_bitcast, primitive);
  }
  Dimensions contiguous_base_shape_after_bitcast =
      GetSizes(base_layout_after_bitcast);

  StridedLayout view_layout_after_bitcast = view_layout;
  for (const auto& primitive : bitcast_view) {
    UpdateLayout(view_layout_after_bitcast, primitive);
  }
  TT_ASSIGN_OR_RETURN(auto post_bitcast_view_sequence,
                      DecomposeIntoViewSequence(
                          contiguous_base_shape_after_bitcast, writable_dtype,
                          view_layout_after_bitcast, writable_dtype, is_conj));
  // Don't use strided slices here so the inversion is more efficient.
  TT_RETURN_IF_ERROR(ReplaceStridedSlices(contiguous_base_shape_after_bitcast,
                                          post_bitcast_view_sequence));
  Simplify(post_bitcast_view_sequence, contiguous_base_shape_after_bitcast);

  if (invertible_bitcast.empty()) {
    base_to_view = std::move(post_bitcast_view_sequence);
  } else {
    base_to_view.reserve(invertible_bitcast.size() +
                         post_bitcast_view_sequence.size());
    base_to_view.insert(base_to_view.end(),
                        std::make_move_iterator(invertible_bitcast.begin()),
                        std::make_move_iterator(invertible_bitcast.end()));
    base_to_view.insert(
        base_to_view.end(),
        std::make_move_iterator(post_bitcast_view_sequence.begin()),
        std::make_move_iterator(post_bitcast_view_sequence.end()));
  }
  return base_to_view;
}

// Given a base layout, a base transform, and a view sequence to some view
// layout, computes one or more inverse view stages that can go from the view
// back to the post-transform base.
// If preserve_base is false, then the base transform and final shape may be
// modified.
std::vector<InverseViewStage> InvertAndPartition(StridedLayout& base_layout,
                                                 ViewSequence& base_transform,
                                                 ViewSequence&& base_to_view,
                                                 Shape& final_shape,
                                                 bool preserve_base) {
  std::vector<InverseViewStage> stages;
  stages.push_back(InverseViewStage());
  for (auto i = 0; i < base_to_view.size(); ++i) {
    if (i == 0 && !preserve_base &&
        std::holds_alternative<ReshapePrimitive>(base_to_view[i])) {
      // An initial reshape can be folded into the base transform, since it
      // retains contiguity. This also changes the expected final shape, so
      // skip this if we're preserving the base shape.
      UpdateLayout(base_layout, base_to_view[i]);
      final_shape.dimensions().clear();
      for (const auto& dim : base_layout.strided_dims) {
        final_shape.dimensions().push_back(dim.size);
      }
      base_transform.push_back(std::move(base_to_view[i]));
    } else if (SlicePrimitive* slice =
                   std::get_if<SlicePrimitive>(&base_to_view[i])) {
      // A slice ends the current stage and starts the next one.

      // Compute the inverse of this stage's forward sequence (excluding the
      // slice), updating the layout to the pre-slice layout.
      InverseViewSequence inverse =
          InvertViewSequence(base_layout, stages.back().forward);
      stages.back().inverse = std::move(inverse);

      // Update the layout to the post-slice layout and assign the slice.
      UpdateLayout(base_layout, *slice);
      stages.back().slice = std::move(*slice);

      // Start a new empty stage.
      stages.push_back(InverseViewStage());
    } else {
      // Otherwise, we extend the current stage.
      stages.back().forward.push_back(std::move(base_to_view[i]));
    }
  }

  // Invert the final stage.
  InverseViewSequence final_stage_inverse =
      InvertViewSequence(base_layout, stages.back().forward);
  stages.back().inverse = std::move(final_stage_inverse);
  return stages;
}

}  // namespace

absl::StatusOr<InverseViewOperation> ComputeInverseViewOperation(
    absl::Span<const int64_t> contiguous_base_shape,
    mlir::ElementType contiguous_base_dtype, const StridedLayout& view_layout,
    mlir::ElementType view_dtype, bool is_conj, bool preserve_base) {
  // Compute any bitcast transformations necessary to align the dtype.
  auto [base_transform, base_dtype_after_transform, bitcast_view,
        writable_dtype, invertible_bitcast] =
      GetBitcastTransforms(contiguous_base_shape, contiguous_base_dtype,
                           view_layout, view_dtype);

  if (!preserve_base && !invertible_bitcast.empty()) {
    // If we're not preserving the base shape, then just merge the invertible
    // bitcast into the base transform to save the inversion cost.
    base_transform.insert(base_transform.end(),
                          std::make_move_iterator(invertible_bitcast.begin()),
                          std::make_move_iterator(invertible_bitcast.end()));
    invertible_bitcast.clear();
    base_dtype_after_transform = writable_dtype;
  }

  // Compute the final layout and shape after the bitcast transform, but before
  // the invertible part of the view sequence.
  StridedLayout base_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  for (const auto& primitive : base_transform) {
    UpdateLayout(base_layout, primitive);
  }
  Shape final_shape(GetSizes(base_layout), base_dtype_after_transform);

  // Compute the view sequence that would go from the post-bitcast base to the
  // post-bitcast view...
  TT_ASSIGN_OR_RETURN(
      ViewSequence base_to_view,
      CalculatePostBitcastBaseToView(base_layout, view_layout, bitcast_view,
                                     std::move(invertible_bitcast),
                                     writable_dtype, is_conj));

  // ...then invert it and partition it into stages.
  // This may modify the base transform and final shape when !preserve_base.
  auto stages =
      InvertAndPartition(base_layout, base_transform, std::move(base_to_view),
                         final_shape, preserve_base);

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

  // Run each stage backwards. Bitcast the mutated view to the base's writable
  // dtype first so the backward inverse and slice merges run in a single dtype.
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp curr_op,
      ViewSequenceShlo(mutated_view, inverse_view_operation.bitcast_view));
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
