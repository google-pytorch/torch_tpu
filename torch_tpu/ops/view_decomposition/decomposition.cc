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

#include "torch_tpu/ops/view_decomposition/decomposition.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/view_decomposition/bitcast_primitive.h"
#include "torch_tpu/ops/view_decomposition/broadcast_primitive.h"
#include "torch_tpu/ops/view_decomposition/conj_primitive.h"
#include "torch_tpu/ops/view_decomposition/pad_primitive.h"
#include "torch_tpu/ops/view_decomposition/reshape_primitive.h"
#include "torch_tpu/ops/view_decomposition/slice_primitive.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/transpose_primitive.h"
#include "torch_tpu/ops/view_decomposition/unfold_primitive.h"
#include "torch_tpu/ops/view_decomposition/view_sequence.h"

namespace torch_tpu {

namespace {

// Checks necessary preconditions that can be easily validated:
//   * The layout has a positive size.
//   * The layout has a non-negative offset.
//   * The layout doesn't extend beyond the end of the contiguous base tensor
//     in terms of the number of bytes.
void CheckViewPreconditions(absl::Span<const int64_t> contiguous_base_shape,
                            const mlir::ElementType contiguous_base_dtype,
                            const StridedLayout& view_layout,
                            const mlir::ElementType view_dtype) {
  size_t i = 0;

  for (const auto& dim : view_layout.strided_dims) {
    ABSL_CHECK_GE(dim.size, 0)  // CRASH_OK=Dimension sizes cannot be < 0.
        << "expected the dimension sizes of the view layout to be >= 0, "
           "got "
        << dim.size << " at index " << i << "; this is a TorchTPU bug";

    ABSL_CHECK_NE(  // CRASH_OK=0-sized views are handled directly by
                    // both callers: `GetBuffer()` and
                    // `AssignBufferToAtTensor()` functions.
        dim.size, 0)
        << "creating a 0-sized view of shape "
        << ToString(GetSizes(view_layout))
        << " is not yet supported; use empty_strided() instead";

    i++;
  }

  // Make sure that the view layout has a non-negative offset.
  ABSL_CHECK_GE(  // CRASH_OK=Storage offset is never negative.
      view_layout.storage_offset, 0)
      << "expected the storage offset of the view to be >= 0, got "
      << view_layout.storage_offset << "; this is a TorchTPU bug";

  // Make sure that the view layout doesn't extend beyond the end of the
  // contiguous base tensor.
  // The starting tensor is contiguous and has 0 offset, so the bytes are just
  // a multiple of the number of elements.
  // Can't report in terms of bytes due to some sub-byte-sized types (F4, etc).
  int64_t base_bits = TorchEquivalentBitwidth(contiguous_base_dtype);
  for (int64_t dim_size : contiguous_base_shape) {
    base_bits *= dim_size;
  }

  // Since the view might be overlapping, the number of required elements is
  // 1 + the largest actual element offset, and then the bits requirement is
  // the number of required elements times the element size.
  int64_t view_max_index = view_layout.storage_offset;
  for (const auto& dim : view_layout.strided_dims) {
    view_max_index += (dim.size - 1) * dim.stride;
  }
  const int64_t view_required_bits =
      (view_max_index + 1) * TorchEquivalentBitwidth(view_dtype);

  ABSL_CHECK_LE(  // CRASH_OK=Checked at time of view construction.
      view_required_bits, base_bits)
      << "expected the view to fit within the base tensor that has "
      << base_bits / 8 << " bytes of storage ("
      << ToString(contiguous_base_dtype) << " tensor of shape "
      << ToString(contiguous_base_shape) << "), got a " << ToString(view_dtype)
      << " view of shape " << ToString(GetSizes(view_layout))
      << " that requires " << view_required_bits / 8 << " bytes of storage";
}

// A strided dimension, with the index it originated from. Used for constructing
// a row-major transpose or broadcast.
struct IndexedStridedDimension {
  int64_t index = 0;
  int64_t size = 1;
  int64_t stride = 0;
};

// Sorts a list of indexed strided dimensions in the following order:
//   * Size > 1 dimensions before size == 1 dimensions
//   * Descending order of stride, ignored for size == 1 dimensions
//   * Ascending order of index (stable sort)
bool IndexedStridedDimensionSortCriterion(const IndexedStridedDimension& a,
                                          const IndexedStridedDimension& b) {
  // Size > 1 dimensions before size == 1 dimensions
  if (a.size > 1 && b.size == 1) {
    return true;
  }
  if (a.size == 1 && b.size > 1) {
    return false;
  }
  // Stride is ignored for size == 1 dimensions
  if (a.size == 1 && b.size == 1) {
    return a.index < b.index;
  }
  // Descending order of stride for size > 1 dimensions
  if (a.stride > b.stride) {
    return true;
  }
  if (a.stride < b.stride) {
    return false;
  }
  // If size > 1 for both a and b, and a.stride == b.stride, then sort by index
  return a.index < b.index;
}

// Builds the last op in a decomposition with any stride == 0, size > 1
// dimensions. This will be a BroadcastInDim, which handles the
// reshape, permute, and expand from "well-behaved" layout to the final layout.
BroadcastPrimitive BuildTailBroadcast(
    StridedLayout& tail_layout,
    absl::Span<const IndexedStridedDimension> sorted_dimensions) {
  // The broadcast will create the sizes in the correct locations.
  Dimensions new_sizes(sorted_dimensions.size(), -1);

  // Extract the non-broadcasted prefix and set up the broadcast map
  Dimensions broadcast_dimensions;

  // Update tail_layout to be the shape before the broadcast.
  tail_layout.strided_dims.clear();  // existing capacity is sufficient

  for (const auto& dim : sorted_dimensions) {
    new_sizes[dim.index] = dim.size;
    if (dim.size <= 1 || dim.stride == 0) {
      // Size before the broadcast will not include this dimension
      continue;
    }
    tail_layout.strided_dims.push_back(
        StridedDimension{.size = dim.size, .stride = dim.stride});
    broadcast_dimensions.push_back(dim.index);
  }

  Dimensions base_shape;
  base_shape.reserve(tail_layout.strided_dims.size());
  for (const auto& dim : tail_layout.strided_dims) {
    base_shape.push_back(dim.size);
  }

  return BroadcastPrimitive{
      .base_shape = std::move(base_shape),
      .new_sizes = std::move(new_sizes),
      .broadcast_dimensions = std::move(broadcast_dimensions)};
}

bool IsRowMajorIgnoringTrivialDimensions(const StridedLayout& layout) {
  int64_t prev_non_trivial_dim = layout.strided_dims.size();
  // Find the first non-trivial dimension (size > 1).
  for (int64_t i = 0; i < layout.strided_dims.size(); ++i) {
    if (layout.strided_dims[i].size > 1) {
      prev_non_trivial_dim = i;
      break;
    }
  }
  for (int64_t i = prev_non_trivial_dim + 1; i < layout.strided_dims.size();
       ++i) {
    if (layout.strided_dims[i].size == 1) {
      // Ignore trivial dimensions.
      continue;
    }
    if (layout.strided_dims[i].stride >
        layout.strided_dims[prev_non_trivial_dim].stride) {
      // The nontrivial dimensions are not in row-major order.
      return false;
    }
    prev_non_trivial_dim = i;
  }
  return true;
}

// If the view has no expanded dimensions, but the nontrivial dimensions are not
// in row-major order, then the last two ops in the decomposition will be a
// reshape followed by a transpose.
struct ReshapeThenTranspose {
  ReshapePrimitive reshape;
  TransposePrimitive transpose;
};

ReshapeThenTranspose BuildTailNoBroadcast(
    StridedLayout& tail_layout,
    absl::Span<const IndexedStridedDimension> sorted_dimensions) {
  // We will reshape from row_major_shape to (*row_major_shape, 1, 1, 1, ...)
  // for as many size-1 dimensions as necessary, then transpose to the final
  // output order.
  ReshapeThenTranspose result;
  result.reshape.new_sizes.reserve(sorted_dimensions.size());
  result.transpose.permutation.resize(sorted_dimensions.size(), -1);

  // Update tail_layout to be the shape before the reshape as well.
  tail_layout.strided_dims.clear();  // existing capacity is sufficient

  for (int64_t i = 0; i < sorted_dimensions.size(); ++i) {
    const auto& indexed_dim = sorted_dimensions[i];
    result.reshape.new_sizes.push_back(indexed_dim.size);
    result.transpose.permutation[indexed_dim.index] = i;
    if (indexed_dim.size > 1 && indexed_dim.stride > 0) {
      tail_layout.strided_dims.push_back(StridedDimension{
          .size = indexed_dim.size, .stride = indexed_dim.stride});
    }
  }
  result.reshape.base_sizes.reserve(tail_layout.strided_dims.size());
  for (const auto& dim : tail_layout.strided_dims) {
    result.reshape.base_sizes.push_back(dim.size);
  }
  return result;
}

absl::InlinedVector<ViewPrimitive, 2> BuildTail(StridedLayout& tail_layout) {
  bool needs_broadcast = false;
  // We want the layout before the tail to be row-major, with no
  // size-1 or stride=0 dimensions; this requires a sort to order by stride
  std::vector<IndexedStridedDimension> sorted_dimensions;
  sorted_dimensions.reserve(tail_layout.strided_dims.size());
  for (int64_t i = 0; i < tail_layout.strided_dims.size(); ++i) {
    const auto& dim = tail_layout.strided_dims[i];
    needs_broadcast |= dim.size > 1 && dim.stride == 0;
    sorted_dimensions.push_back(IndexedStridedDimension{
        .index = i, .size = dim.size, .stride = dim.stride});
  }
  // Sort so that the sorted dimensions begin with row-major, non-broadcasted
  // dimensions, with some number of output dimensions after that.
  absl::c_sort(sorted_dimensions, &IndexedStridedDimensionSortCriterion);

  // If we need a broadcast, then we can use the BroadcastInDim primitive to
  // handle the reshape, transpose, and expand at the same time.
  absl::InlinedVector<ViewPrimitive, 2> tail;
  if (needs_broadcast) {
    tail.push_back(BuildTailBroadcast(tail_layout, sorted_dimensions));
    return tail;
  }

  // If the tail layout is already row-major (ignoring size-1 dimensions) then
  // we don't need a transpose, we can just unsqueeze (reshape) size-1
  // dimensions into their final positions.
  if (IsRowMajorIgnoringTrivialDimensions(tail_layout)) {
    ReshapePrimitive reshape;
    reshape.new_sizes.reserve(tail_layout.strided_dims.size());
    absl::InlinedVector<StridedDimension, 3> pre_tail_strided_dims;
    for (auto& dim : tail_layout.strided_dims) {
      reshape.new_sizes.push_back(dim.size);
      if (dim.size > 1 && dim.stride > 0) {
        pre_tail_strided_dims.push_back(std::move(dim));
      }
    }
    tail_layout.strided_dims = std::move(pre_tail_strided_dims);

    reshape.base_sizes.reserve(tail_layout.strided_dims.size());
    for (auto& dim : tail_layout.strided_dims) {
      reshape.base_sizes.push_back(dim.size);
    }

    tail.push_back(std::move(reshape));
    return tail;
  }

  // If we don't need a broadcast, but we do need a transpose, then we use a
  // reshape and transpose, both of which are invertible by in-place mutations.
  auto [reshape, transpose] =
      BuildTailNoBroadcast(tail_layout, sorted_dimensions);
  tail = {std::move(reshape), std::move(transpose)};
  return tail;
}

// Builds the first op in the decomposition, which may flatten the head layout
// to a single dimension.
absl::StatusOr<ReshapePrimitive> BuildHeadFlatten(
    const StridedLayout& head_layout) {
  // Flatten the head layout from shape [d0, d1, ..., dn-1] to [d0*d1*...*dn-1],
  // or to shape [] if the input is a single element.
  SmallInt64Vector base_sizes;
  base_sizes.reserve(head_layout.strided_dims.size());
  for (const auto& dim : head_layout.strided_dims) {
    base_sizes.push_back(dim.size);
  }
  TT_ASSIGN_OR_RETURN(const int64_t num_elements, NumElements(base_sizes));
  if (num_elements == 1) {
    return ReshapePrimitive{.base_sizes = std::move(base_sizes),
                            .new_sizes = {}};
  }

  return ReshapePrimitive{.base_sizes = std::move(base_sizes),
                          .new_sizes = {num_elements}};
}

// Builds the second op in the decomposition, which may pad the head layout
// to the required storage capacity.
// head_layout is updated in-place to be the shape after the pad.
PadPrimitive BuildHeadPad(const StridedLayout& head_layout,
                          const int64_t required_storage_capacity) {
  if (head_layout.strided_dims.empty()) {
    return PadPrimitive{.pad_dims = {}};
  }
  ABSL_CHECK(head_layout.strided_dims.size() == 1)  // CRASH_OK
      << "BuildHeadPad called before BuildHeadFlatten";

  PadPrimitive pad;
  // If the head layout already has enough storage, then we don't pad
  // (create a no-op pad).
  int64_t high_padding = std::max(
      0L, required_storage_capacity - head_layout.strided_dims[0].size);

  pad.pad_dims.push_back(PadDimension{
      .low_padding = 0, .high_padding = high_padding, .interior_padding = 0});
  return pad;
}

// Builds the third op in the decomposition, which may slice the head layout
// to the view window for the view.
SlicePrimitive BuildHeadSliceToViewWindow(const StridedLayout& head_layout,
                                          const int64_t storage_offset,
                                          const int64_t view_width) {
  if (head_layout.strided_dims.empty()) {
    return SlicePrimitive{.slice_dims = {}};
  }
  ABSL_CHECK(head_layout.strided_dims.size() == 1)  // CRASH_OK
      << "BuildHeadSliceToViewWindow called before BuildHeadFlatten";
  SlicePrimitive slice;
  const int64_t limit_index = storage_offset + view_width;
  slice.slice_dims.push_back(SliceDimension{
      .start_index = storage_offset, .limit_index = limit_index, .stride = 1});
  return slice;
}

// If the base type is complex, adds a complex-to-real bitcast to the
// result decomposition.
// head_layout is updated in-place to be the shape after the bitcast; if
// the bitcast is real() or imag(), then tail_layout is also updated in-place
// to reflect the relative view on the real or imaginary components.
// Returns the (real) element type after the complex-to-real bitcast.
mlir::ElementType MaybeAddComplexToRealBitcast(
    ViewSequence& result, StridedLayout& head_layout,
    const mlir::ElementType contiguous_base_dtype, StridedLayout& tail_layout) {
  if (contiguous_base_dtype != mlir::ElementType::COMPLEXF32 &&
      contiguous_base_dtype != mlir::ElementType::COMPLEXF64) {
    // Base type is not complex, no complex-to-real bitcast is needed.
    return contiguous_base_dtype;
  }
  // Base is complex, we need to convert it to real-valued floats.
  ComplexElementType complex_element_type =
      contiguous_base_dtype == mlir::ElementType::COMPLEXF32
          ? ComplexElementType::kComplexFloat
          : ComplexElementType::kComplexDouble;
  mlir::ElementType real_element_type =
      complex_element_type == ComplexElementType::kComplexFloat
          ? mlir::ElementType::F32
          : mlir::ElementType::F64;

  // Check if we can apply this as "real()" or "imag()".
  bool is_half_of_complex = true;
  for (const auto& dim : tail_layout.strided_dims) {
    if (dim.stride % 2 != 0) {
      is_half_of_complex = false;
      break;
    }
  }

  if (!is_half_of_complex) {
    // Output view includes both real and imaginary components of the base.
    auto complex_to_real = ComplexToRealBitcast{
        .complex_element_type = complex_element_type,
        .bitcast_type = ComplexToRealBitcastType::kViewAsReal,
    };
    UpdateLayout(head_layout, complex_to_real);
    result.push_back(std::move(complex_to_real));
    return real_element_type;
  }

  if (tail_layout.storage_offset % 2 == 0) {
    // Output view only includes the real part of the base.
    auto real_op = ComplexToRealBitcast{
        .complex_element_type = complex_element_type,
        .bitcast_type = ComplexToRealBitcastType::kReal,
    };
    UpdateLayout(head_layout, real_op);
    result.push_back(std::move(real_op));
  } else {
    // Output view only includes the imaginary part of the base.
    auto imag_op = ComplexToRealBitcast{
        .complex_element_type = complex_element_type,
        .bitcast_type = ComplexToRealBitcastType::kImag,
    };
    UpdateLayout(head_layout, imag_op);
    result.push_back(std::move(imag_op));
  }
  // The rest of the decomposition assumes that the head layout is contiguous,
  // but real() and imag() create non-contiguous views.
  // But we can update head and tail to both be "relatively contiguous" by
  // making only consider strides on the real or imaginary halves.
  if (tail_layout.storage_offset % 2 != 0) {
    head_layout.storage_offset -= 1;
    tail_layout.storage_offset -= 1;
  }
  for (auto& dim : head_layout.strided_dims) {
    dim.stride /= 2;
  }
  for (auto& dim : tail_layout.strided_dims) {
    dim.stride /= 2;
  }
  // Head layout storage offset is always 0 after handling imag() offset, no
  // need to divide it.
  tail_layout.storage_offset /= 2;
  return real_element_type;
}

// Adds reshapes and slices to the result decomposition necessary to create
// a trailing dimension of the given size.
// head_layout is updated in-place to be the layout after the new op(s).
void CreateTrailingDimension(ViewSequence& result, StridedLayout& head_layout,
                             const int64_t trailing_size) {
  if (!head_layout.strided_dims.empty() &&
      head_layout.strided_dims.back().size == trailing_size) {
    // Already has the correct trailing dimension, no extra ops are needed.
    return;
  }
  // Need at least one reshape, and we might also need a slice.
  int64_t head_numel = 1;
  for (const auto& dim : head_layout.strided_dims) {
    head_numel *= dim.size;
  }

  // Do we need to truncate?
  if (head_numel % trailing_size != 0) {
    SmallInt64Vector base_sizes_flatten;
    for (const auto& dim : head_layout.strided_dims) {
      base_sizes_flatten.push_back(dim.size);
    }
    auto flatten_to_1d = ReshapePrimitive{
        .base_sizes = std::move(base_sizes_flatten), .new_sizes = {head_numel}};
    UpdateLayout(head_layout, flatten_to_1d);
    result.push_back(std::move(flatten_to_1d));

    const int64_t truncated_numel =
        (head_numel / trailing_size) * trailing_size;
    auto truncate_slice = SlicePrimitive{
        .slice_dims = {SliceDimension{
            .start_index = 0, .limit_index = truncated_numel, .stride = 1}}};
    UpdateLayout(head_layout, truncate_slice);
    result.push_back(std::move(truncate_slice));
    head_numel = truncated_numel;
  }

  if (head_numel == trailing_size) {
    SmallInt64Vector base_sizes_1d;
    for (const auto& dim : head_layout.strided_dims) {
      base_sizes_1d.push_back(dim.size);
    }
    // Reshape to 1D, so that the only dimension is the trailing dimension.
    auto reshape = ReshapePrimitive{.base_sizes = std::move(base_sizes_1d),
                                    .new_sizes = {head_numel}};
    UpdateLayout(head_layout, reshape);
    result.push_back(std::move(reshape));
    return;
  }

  SmallInt64Vector base_sizes_2d;
  for (const auto& dim : head_layout.strided_dims) {
    base_sizes_2d.push_back(dim.size);
  }
  // Reshape to a 2D value with the trailing dimension of the desired size.
  auto reshape = ReshapePrimitive{
      .base_sizes = std::move(base_sizes_2d),
      .new_sizes = {head_numel / trailing_size, trailing_size}};
  UpdateLayout(head_layout, reshape);
  result.push_back(std::move(reshape));
}

// If necessary, adds a real-to-real bitcast to the decomposition.
// head_layout is updated in-place to be the layout after the bitcast.
// Returns the (real) element type after the bitcast.
mlir::ElementType MaybeAddRealToRealBitcast(
    ViewSequence& result, StridedLayout& head_layout,
    const mlir::ElementType real_head_element_type,
    const StridedLayout& tail_layout, const mlir::ElementType view_dtype) {
  mlir::ElementType real_tail_element_type = RealComponentOf(view_dtype);
  if (real_head_element_type == real_tail_element_type) {
    // No real-to-real bitcast is required.
    return real_tail_element_type;
  }

  // This is the bitcast that we'll add, but we may need to adjust shapes first.
  auto real_to_real = RealToRealBitcast{
      .from_type = real_head_element_type,
      .to_type = real_tail_element_type,
  };
  const int64_t head_bitwidth = TorchEquivalentBitwidth(real_head_element_type);
  const int64_t tail_bitwidth = TorchEquivalentBitwidth(real_tail_element_type);

  if (head_bitwidth >= tail_bitwidth) {
    // Bitcast is either to a smaller size, which will add a dimension, or to
    // the same size, which doesn't affect the shape. This is infallible and
    // requires no reshaping.
    UpdateLayout(head_layout, real_to_real);
    result.push_back(std::move(real_to_real));
    return real_tail_element_type;
  }

  // If we're bitcasting to a larger size, then we need to create a trailing
  // dimension with a size that is divisible by the bitwidth ratio.
  const int64_t size_ratio = tail_bitwidth / head_bitwidth;
  CreateTrailingDimension(result, head_layout, size_ratio);

  // The bitcast can now be added.
  UpdateLayout(head_layout, real_to_real);
  result.push_back(std::move(real_to_real));
  return real_tail_element_type;
}

// If the view type is complex, adds a view_as_complex to the decomposition.
// head_layout is updated in-place to be the layout after the bitcast.
void MaybeAddViewAsComplex(ViewSequence& result, StridedLayout& head_layout,
                           const mlir::ElementType real_tail_element_type,
                           const StridedLayout& tail_layout,
                           const mlir::ElementType view_dtype) {
  if (view_dtype != mlir::ElementType::COMPLEXF32 &&
      view_dtype != mlir::ElementType::COMPLEXF64) {
    // View type is not complex, no view as complex bitcast is needed.
    return;
  }

  if (head_layout.strided_dims.empty() ||
      head_layout.strided_dims.back().size % 2 != 0) {
    // We need the last dimension to have an even number of elements in the
    // last dimension.
    CreateTrailingDimension(result, head_layout, /*trailing_size=*/2);
  }
  auto view_as_complex = ViewAsComplex{
      .complex_element_type = view_dtype == mlir::ElementType::COMPLEXF32
                                  ? ComplexElementType::kComplexFloat
                                  : ComplexElementType::kComplexDouble,
  };
  UpdateLayout(head_layout, view_as_complex);
  result.push_back(std::move(view_as_complex));
}

ViewSequence BuildHeadBitcast(StridedLayout& head_layout,
                              const mlir::ElementType contiguous_base_dtype,
                              StridedLayout& tail_layout,
                              const mlir::ElementType view_dtype) {
  if (view_dtype == contiguous_base_dtype) {
    // No bitcast is required.
    return {};
  }

  // Need some combination of bitcasts, which may also require some reshaping.
  // The worst case is complex-to-real, real-to-real, and view_as_complex,
  // which can happen with a ComplexFloat <-> ComplexDouble cast.
  ViewSequence result;
  mlir::ElementType head_real_element_type = MaybeAddComplexToRealBitcast(
      result, head_layout, contiguous_base_dtype, tail_layout);
  mlir::ElementType view_real_element_type = MaybeAddRealToRealBitcast(
      result, head_layout, head_real_element_type, tail_layout, view_dtype);
  MaybeAddViewAsComplex(result, head_layout, view_real_element_type,
                        tail_layout, view_dtype);
  return result;
}

// The result of building the head of the decomposition.
struct BuildHeadResult {
  ViewSequence head_sequence;
  Dimensions view_widths;
};

// Builds the first ops in the decomposition, which restructure an
// N-dimensional contiguous tensor into a 1-dimensional "storage" tensor with
// the same offset and total width as the target view.
// head_layout is updated in-place to be the shape after this restructuring.
// Returns both the initial sequence of ops that reshape the head into 1D,
// and the width sequence of the view for use in the body of the decomposition.
absl::StatusOr<BuildHeadResult> BuildHead(
    StridedLayout& head_layout, const mlir::ElementType continuous_base_dtype,
    StridedLayout& tail_layout, const mlir::ElementType view_dtype) {
  // Working forwards:
  // The first step is to cast the head layout to the dtype of the view.
  // After aligning dtypes, we no longer need to worry about bytes, only
  // elements.
  // This might also alter the shape if the element types have different
  // bitwidths.
  ViewSequence head_sequence = BuildHeadBitcast(
      head_layout, continuous_base_dtype, tail_layout, view_dtype);

  // The second step is to flatten the head_layout above to a single dimension.
  // This allows us to treat an N-dimension input as a 1-dimension "storage".
  // This might be a no-op if the head_layout is already 1-dimensional or we
  // did some flattening in the bitcast step.
  TT_ASSIGN_OR_RETURN(const auto second_step, BuildHeadFlatten(head_layout));
  UpdateLayout(head_layout, second_step);
  head_sequence.push_back(std::move(second_step));

  // The view width sequence is the number of dense elements required to satisfy
  // each suffix subarray of the tail layout.
  // This is defined to be width[n-1] = (size[n-1] - 1) * stride[n-1] + 1 for
  // the last dimension; this is the minimum number of elements required to
  // satisfy the last dimension with a strided slice.
  // For all prior dimensions, we set the width to be the higher of:
  //   size[i] * stride[i]                     (non-overlapping case)
  //   (size[i] - 1) * stride[i] + width[i+1]  (overlapping case)
  // Note that this implies that width[i] >= size[i] * width[i+1] for all
  // i < n-1.
  Dimensions view_widths;
  if (!tail_layout.strided_dims.empty()) {
    view_widths.resize(tail_layout.strided_dims.size());
    view_widths.back() = (tail_layout.strided_dims.back().size - 1) *
                             tail_layout.strided_dims.back().stride +
                         1;
    // TODO(b/479163731): evaluate strategy for 1D unevenly strided views
    for (int i = tail_layout.strided_dims.size() - 2; i >= 0; --i) {
      const int64_t non_overlapping_width =
          tail_layout.strided_dims[i].size * tail_layout.strided_dims[i].stride;
      const int64_t overlapping_width = (tail_layout.strided_dims[i].size - 1) *
                                            tail_layout.strided_dims[i].stride +
                                        view_widths[i + 1];
      view_widths[i] = std::max(non_overlapping_width, overlapping_width);
    }
  }

  // Determine the width we need to initially slice the head to.
  int64_t initial_view_width = -1;
  if (view_widths.empty()) {
    // The tail is a scalar, so we just slice the head to be shape (1,) at the
    // appropriate offset.
    initial_view_width = 1;
  } else {
    // Set the initial view width to be the largest possible multiple of the
    // stride that doesn't require padding, or just the minimum required view
    // width if padding is unavoidable.
    const int64_t no_padding_max_multiple =
        head_layout.strided_dims[0].size /
        tail_layout.strided_dims.front().stride;
    initial_view_width = std::max(
        view_widths.front(),
        no_padding_max_multiple * tail_layout.strided_dims.front().stride);
  }

  // The storage capacity is enough to cover the initial view width, plus any
  // elements skipped by a nonzero offset. Because view_width may have trailing
  // unused elements, there might be some unused elements at the end of the
  // storage array.
  const int64_t required_storage_capacity =
      tail_layout.storage_offset + initial_view_width;

  // The third step is to pad the head_layout above to ensure we have the
  // required storage capacity. These padding elements are unused; this is
  // guaranteed because we checked the maximum view index against the contiguous
  // base shape above.
  // This will be minimized by ReducePad during BuildBodyStep, ideally
  // eliminating it entirely.
  auto third_step = BuildHeadPad(head_layout, required_storage_capacity);
  UpdateLayout(head_layout, third_step);
  head_sequence.push_back(std::move(third_step));

  // The fourth step is to slice the head_layout at the low and/or high ends
  // to achieve the target view offset and width.
  auto fourth_step = BuildHeadSliceToViewWindow(
      head_layout, tail_layout.storage_offset, initial_view_width);
  UpdateLayout(head_layout, fourth_step);
  head_sequence.push_back(std::move(fourth_step));

  return {{.head_sequence = std::move(head_sequence),
           .view_widths = std::move(view_widths)}};
}

// Reduces or eliminates the initial pad and slice operations to the largest
// possible amount using a given slice operation.
//
// For some as_strided operations, a pad is unavoidable. An example here would
// be torch.arange(5).as_strided(size=(2, 2), stride=(3, 1), offset=0). The
// only valid decomposition is pad({0, 1, 0}) -> reshape(2, 3) ->
//  slice({0, 2, 1}, {0, 2, 1}). The final slice removes the padding, but the
// pad is still necessary for the reshape to be correctly divisible.
//
//
// We also sometimes cannot avoid an initial slice, if the storage offset
// indicates an offset that is not a linear combination of the strides on
// sliced dimensions.
// For example, torch.arange(7).as_strided(size=(3, 2), stride=(2, 1), offset=1)
// is best decomposed as slice({1, 7, 1}) -> reshape(3, 2). In this case, no
// padding is necessary, but we still require the initial slice to achieve the
// correct final offset, as there is no slice to use to shift the window.
//
// However, torch.arange(9).as_strided(size=(3, 2), stride=(3, 1), offset=1)
// can be decomposed to reshape(3, 3) -> slice({0, 3, 1}, {1, 3, 1}) with
// no padding required, using only compact slices.
//
// If there is a positive start_index on the head slice, then when these slices
// are solved for during BuildBodyStep, we can move some amount of the storage
// offset increase into the body slice, reducing the amount of padding and
// slicing in the head. In many cases, this will reduce the pad and slice to
// no-ops, if there is a linear combination of non-overlapping strides that
// can achieve the desired offset.
void ShiftInitialViewWindow(PadDimension& head_pad, SliceDimension& head_slice,
                            const StridedLayout& layout_before_slice,
                            SlicePrimitive& body_slice) {
  ABSL_CHECK_EQ(layout_before_slice.strided_dims.size(),  // CRASH_OK
                body_slice.slice_dims.size());

  // If we increase the start_index of a slice in the body of the decomposition,
  // then the entire initial view window can be shifted by
  // (start_index * stride before the slice), which removes that many elements
  // from the head pad and slice. We can't shift down below a starting offset
  // of 0.
  int64_t max_removable_elements = head_slice.start_index;
  for (int i = 0; i < body_slice.slice_dims.size(); ++i) {
    if (max_removable_elements == 0) return;
    SliceDimension& slice_dim = body_slice.slice_dims[i];

    // If the slice has a relative stride > 1, then the limit index is not
    // unique. For example, torch.arange(4)[::2] could be expressed as either
    // slice({0, 3, 2}) or slice({0, 4, 2}); both produce [0, 2] correctly.
    // To infer the maximum possible amount of padding to remove, we need to
    // choose the lowest possible limit_index of the valid ranges.
    const int64_t slice_numel =
        (slice_dim.limit_index - slice_dim.start_index + slice_dim.stride - 1) /
        slice_dim.stride;
    const int64_t last_included_index =
        (slice_numel - 1) * slice_dim.stride + slice_dim.start_index;
    const int64_t min_limit_index = last_included_index + 1;

    // We need to preserve the size of the slice dimension, so if we increase
    // the start_index, the limit_index must also increase by the same amount.
    // We can't have the limit_index be > the pre-slice dimension size.
    int64_t max_start_index =
        layout_before_slice.strided_dims[i].size - min_limit_index;

    // Each increment of start_index removes stride elements from the head pad,
    // so we round down to the nearest multiple of stride.
    max_start_index = std::min(
        max_start_index,
        max_removable_elements / layout_before_slice.strided_dims[i].stride);
    if (max_start_index <= 0) continue;

    // We take a greedy approach and always remove the maximum amount of
    // padding in every dimension.
    // For a non-overlapping view, this is optimal; because
    // stride[i] > maximum offset of dimensions [i+1, n), for all i, the only
    // way to achieve a pad reduction >= stride[i] elements is to increase the
    // start index of the slice to dimension <= i.
    const int64_t shift_amount =
        max_start_index * layout_before_slice.strided_dims[i].stride;
    ABSL_VLOG(3)
        << "[ShiftInitialViewWindow] Removing " << shift_amount
        << " elements from head pad and slice by increasing sliced dimension "
        << i << " start_index to " << max_start_index << " with a stride of "
        << layout_before_slice.strided_dims[i].stride;
    head_pad.high_padding -= std::min(head_pad.high_padding, shift_amount);
    head_slice.start_index -= shift_amount;
    head_slice.limit_index -= shift_amount;
    max_removable_elements -= shift_amount;
    slice_dim.start_index = max_start_index;
    slice_dim.limit_index = min_limit_index + max_start_index;
  }
}

void ShiftInitialViewWindow(PadDimension& head_pad, SliceDimension& head_slice,
                            const StridedLayout& layout_before_slice,
                            UnfoldPrimitive& unfold) {
  // If we increase the start_index of an unfold in the body of the
  // decomposition, then the entire initial view window can be shifted by
  // start_index, which removes that many elements from the head pad and slice.
  //  We can't shift down below a starting offset of 0.
  const int64_t max_removable_elements = head_slice.start_index;
  const int64_t max_start_index = std::min(
      max_removable_elements,
      layout_before_slice.strided_dims.back().size - unfold.limit_index);
  ABSL_VLOG(3)
      << "[ShiftInitialViewWindow] Removing " << max_start_index
      << " elements from head pad and slice by increasing unfolded dimension "
      << layout_before_slice.strided_dims.size() - 1 << " start_index to "
      << max_start_index << " with a stride of 1";
  head_pad.high_padding -= std::min(head_pad.high_padding, max_start_index);
  head_slice.start_index -= max_start_index;
  head_slice.limit_index -= max_start_index;
  unfold.start_index = max_start_index;
  unfold.limit_index += max_start_index;
}

// Satisfies one or more non-overlapping tail dimensions in the tail layout
// using exactly one reshape and one slice.
// **PRECONDITIONS**:
// head_layout has:
//   rank:    k+1 < n
//   sizes:   [... Z(k-1), c * T(k)], where c is an integer >= Z(k)
//   strides: [... T(k-1), 1]
// tail layout has:
//   rank:    n
//   sizes:   [... Z(k-1), Z(k), Z(k+1), ..., Z(n-1)]
//   strides: [... T(k-1), T(k), T(k+1), ..., T(n-1)]
// Both head and tail have T(i) >= T(i+1) and T(i) > 0 for all i.
// view_widths have values [..., V(k), V(k+1), ..., V(n-1)]
//   where V(i) >= Z(i) * V(i+1) for all i < n-1
//   and V(k) == Z(k) * T(k) at index k
// Note that this implies that T(k) >= V(k+1) by construction of view_widths.
//
// Additionally, we define Z(i) = T(i) = V(i) = 1 for all i >= n. This allows
// the body to append size=1, stride=1 dimensions in excess of the tail layout.
// This sometimes allows for us to include the last real dimension as a
// contiguous slice, which improves view inversion performance.
//
// **ALGORITHM**:
// Find 0 <= m < (n-k) "extra" dimensions [k, k+1, ..., k+m) such that:
//   T(i) >= V(i+1)
//   T(i) % T(i+1) == 0
//
// Construct and apply:
// reshape([... Z(k-1), c, T(k) / T(k+1), ..., T(k+m-1) / T(k+m), T(k+m)]
// head (after reshape):
//   rank:    k+m+2
//   sizes:  [... c,    T(k) / T(k+1), ..., T(k+m-1) / T(k+m), T(k+m)]
//   strides:[... T(k), T(k+1),        ..., T(k+m),            1     ]
//
// Then construct:
// slice([... (0, Z(k+m), 1), (0, X, 1)])
//   where X = max(floor(T(k+m) / T(k+m+1)) * T(k+m+1), V(k+m+1))
// By construction, the slice is valid because:
//   At i == k:
//     T(k) >= V(k+1)                        (precondition)
//     T(k) >= floor(T(k) / T(k+1)) * T(k+1) (definition of floor)
//     => T(k) >= X
//   At k < i <= k+m:
//     T(i) >= V(i+1)                        (checked in algorithm)
//     T(i) % T(i+1) == 0                    (checked in algorithm)
//     T(i) >= floor(T(i) / T(i+1)) * T(i+1) (definition of floor)
//     => T(i) >= X
//
// head (after slice):
//   rank:    k+m+2
//   sizes:   [... Z(k+m), X],
//   strides: [... T(k+m), 1]
// which restablishes the precondition but now with k+m+1 completed dimensions.
absl::Status BuildBodyStepNonOverlapping(StridedLayout& head_layout,
                                         const StridedLayout& tail_layout,
                                         const Dimensions& view_widths,
                                         ViewSequence& result,
                                         const int64_t head_pad_index) {
  const int64_t start_dim = head_layout.strided_dims.size() - 1;
  // Preserve the shape of dimensions before the new dimensions we're adding,
  // and preserve the total element count.
  ReshapePrimitive reshape;
  reshape.base_sizes.reserve(head_layout.strided_dims.size());
  for (const auto& dim : head_layout.strided_dims) {
    reshape.base_sizes.push_back(dim.size);
  }
  for (auto i = 0; i < start_dim; ++i) {
    reshape.new_sizes.push_back(head_layout.strided_dims[i].size);
  }
  int64_t remaining_numel = head_layout.strided_dims[start_dim].size;

  // Split out at least one dimension.
  const auto first_dimension =
      remaining_numel / tail_layout.strided_dims[start_dim].stride;
  reshape.new_sizes.push_back(first_dimension);
  remaining_numel = tail_layout.strided_dims[start_dim].stride;

  // Potentially split out more dimensions, as long as we have divisible strides
  // and non-overlapping dimensions.
  for (int64_t curr_dim = start_dim + 1;
       curr_dim < tail_layout.strided_dims.size(); ++curr_dim) {
    const int64_t curr_stride = tail_layout.strided_dims[curr_dim].stride;
    // Allow reshaping to include an additional dimension to satisfy the last
    // dimension if necessary.
    if (curr_dim + 1 < tail_layout.strided_dims.size() &&
        curr_stride < view_widths[curr_dim + 1]) {
      // This is an overlapping dimension, so we need to stop splitting here.
      // The next iteration will call BuildBodyStepOverlapping to resolve it.
      break;
    }

    if (remaining_numel % curr_stride != 0) {
      // If the remaining_numel is not divisible by the curr_stride, then we
      // can't reshape the last dimension in this step; we need to slice it
      // to a divisible number of elements first.
      break;
    }
    const auto stride_ratio = remaining_numel / curr_stride;
    reshape.new_sizes.push_back(stride_ratio);
    remaining_numel /= stride_ratio;
  }

  if (remaining_numel > 1) {
    // Finish the reshape with the remaining elements to ensure no numel change.
    reshape.new_sizes.push_back(remaining_numel);
  }
  UpdateLayout(head_layout, reshape);
  ABSL_VLOG(3) << "[BuildBodyStepNonOverlapping] Appended " << reshape
               << ", new head_layout: " << head_layout;
  result.push_back(std::move(reshape));

  // Slice all inserted dimensions to their final size.
  const int64_t last_dim = head_layout.strided_dims.size() - 1;
  SlicePrimitive slice;
  slice.slice_dims.reserve(head_layout.strided_dims.size());
  for (auto i = 0; i < last_dim; ++i) {
    slice.slice_dims.push_back(
        SliceDimension{.start_index = 0,
                       .limit_index = tail_layout.strided_dims[i].size,
                       .stride = 1});
  }
  // Slice the last dimension.
  int64_t last_limit_index = -1;  // dummy value to avoid uninitialized memory
  if (last_dim >= tail_layout.strided_dims.size()) {
    // If we appended an extra dimension, slice it to size 1.
    last_limit_index = 1;
  } else {
    // Slice the last dimension to the largest possible multiple of the stride
    // but no smaller than the view width.
    last_limit_index = (head_layout.strided_dims[last_dim].size /
                        tail_layout.strided_dims[last_dim].stride) *
                       tail_layout.strided_dims[last_dim].stride;
    last_limit_index = std::max(last_limit_index, view_widths[last_dim]);
  }
  slice.slice_dims.push_back(SliceDimension{
      .start_index = 0, .limit_index = last_limit_index, .stride = 1});

  // Use the slice to reduce the amount of padding we need to do, if possible.
  // The exact position of the pad may vary if there were bitcasts or not,
  // but it will always be followed immediately by a slice.
  PadDimension& head_pad =
      std::get<PadPrimitive>(result[head_pad_index]).pad_dims[0];
  SliceDimension& head_slice =
      std::get<SlicePrimitive>(result[head_pad_index + 1]).slice_dims[0];
  ShiftInitialViewWindow(head_pad, head_slice, head_layout, slice);

  UpdateLayout(head_layout, slice);
  ABSL_VLOG(3) << "[BuildBodyStepNonOverlapping] Appended " << slice
               << ", new head_layout: " << head_layout;
  result.push_back(std::move(slice));
  return absl::OkStatus();
}

// Satisfies one overlapping tail dimension in the tail layout using exactly one
// reshape and unfold.
// **PRECONDITIONS**:
// head_layout has:
//   rank:    k+1 < n
//   sizes:   [... Z(k-1), V(k)]
//   strides: [... T(k-1), 1]
// tail layout has:
//   rank:    n
//   sizes:   [... Z(k-1), Z(k), Z(k+1), ..., Z(n-1)]
//   strides: [... T(k-1), T(k), T(k+1), ..., T(n-1)]
// Both head and tail have T(i) >= T(i+1) and T(i) > 0 for all i.
// view_widths have values [..., V(k), V(k+1), ..., V(n-1)]
//   where V(i) >= Z(i) * T(i) for all i
//   and in particular V(k) > Z(k) * T(k) at index k
//
// **ALGORITHM**:
// Construct and apply:
// reshape([... Z(k-1), 1, V(k)]
// head (after reshape):
//   rank:    k+2
//   sizes:  [... Z(k-1), 1, V(k)]
//   strides:[... T(k-1), {arbitrary}, 1]
//
// Then unfold with window stride = T(k), window size = V(k+1),
// start_index = 0, and limit_index = (Z(k)-1) * T(k) + V(k+1).
// This is valid because:
//   V(k) = Z(k) * max(T(k), V(k+1)) by definition of view_widths
//   V(k) > Z(k) * T(k) => V(k+1) > T(k) by the precondition
//   V(k) >= Z(k) * V(k+1) by definition of view_widths
//   => V(k) >= (Z(k) - 1) * T(k) + V(k+1) == limit_index
// head (after unfold):
//   rank:    k+2
//   sizes:  [... Z(k-1), Z(k), V(k+1)]
//   strides:[... T(k-1), T(k), 1]
// which restablishes the precondition but now with k+1 completed dimensions.
absl::Status BuildBodyStepOverlapping(StridedLayout& head_layout,
                                      const StridedLayout& tail_layout,
                                      const Dimensions& view_widths,
                                      ViewSequence& result,
                                      const int64_t head_pad_index) {
  const int64_t start_dim = head_layout.strided_dims.size() - 1;
  // Unsqueeze a size-1 dimension before the last dimension.
  ReshapePrimitive reshape;
  reshape.base_sizes.reserve(head_layout.strided_dims.size());
  for (const auto& dim : head_layout.strided_dims) {
    reshape.base_sizes.push_back(dim.size);
  }
  for (auto i = 0; i < start_dim; ++i) {
    reshape.new_sizes.push_back(head_layout.strided_dims[i].size);
  }
  reshape.new_sizes.push_back(1);
  reshape.new_sizes.push_back(head_layout.strided_dims[start_dim].size);
  UpdateLayout(head_layout, reshape);
  ABSL_VLOG(3) << "[BuildBodyStepOverlapping] Appended " << reshape
               << ", new head_layout: " << head_layout;
  result.push_back(std::move(reshape));

  // Unfold the overlapping dimension.
  UnfoldPrimitive unfold;
  unfold.start_index = 0;
  unfold.window_stride = tail_layout.strided_dims[start_dim].stride;
  unfold.window_size = view_widths[start_dim + 1];
  unfold.limit_index =
      (tail_layout.strided_dims[start_dim].size - 1) * unfold.window_stride +
      unfold.window_size;

  // Use the unfold to reduce the amount of padding we need to do, if possible.
  // The exact position of the pad may vary if there were bitcasts or not,
  // but it will always be followed immediately by a slice.
  PadDimension& head_pad =
      std::get<PadPrimitive>(result[head_pad_index]).pad_dims[0];
  SliceDimension& head_slice =
      std::get<SlicePrimitive>(result[head_pad_index + 1]).slice_dims[0];
  ShiftInitialViewWindow(head_pad, head_slice, head_layout, unfold);

  UpdateLayout(head_layout, unfold);
  ABSL_VLOG(3) << "[BuildBodyStepOverlapping] Appended " << unfold
               << ", new head_layout: " << head_layout;
  result.push_back(std::move(unfold));
  return absl::OkStatus();
}

// Adds one reshape and either a slice or an unfold to satisfy one or more
// overlapping tail dimensions.
absl::Status BuildBodyStep(StridedLayout& head_layout,
                           const StridedLayout& tail_layout,
                           const Dimensions& view_widths, ViewSequence& result,
                           const int64_t head_pad_index) {
  const int64_t start_dim = head_layout.strided_dims.size() - 1;

  const int64_t non_overlapping_view_width =
      tail_layout.strided_dims[start_dim].size *
      tail_layout.strided_dims[start_dim].stride;
  const int64_t actual_view_width = view_widths[start_dim];
  // View width should be floored at the non-overlapping width.
  ABSL_CHECK_GE(actual_view_width, non_overlapping_view_width);  // CRASH_OK
  if (actual_view_width == non_overlapping_view_width) {
    return BuildBodyStepNonOverlapping(head_layout, tail_layout, view_widths,
                                       result, head_pad_index);
  }
  return BuildBodyStepOverlapping(head_layout, tail_layout, view_widths, result,
                                  head_pad_index);
}

// If the last (most-minor) dimension has stride > 1, then we need to do either
// a strided slice or a reshape and slice to satisfy it.
//
// ---PRECONDITIONS---
// head layout has:
//   rank:    n
//   sizes:   [... Z(n-2), X]
//     where X is either c * T(n-1) with c >= Z(n-1)
///    or X = (Z(n-1) - 1) * T(n-1) + 1
//   strides: [... T(n-2), 1]
// tail layout has:
//   rank:    n
//   sizes:   [... Z(n-2), Z(n-1)]
//   strides: [... T(n-2), T(n-1)]
// Both head and tail have T(i) > T(i+1) and T(i) > 0 for all i.
//
// ---EVENLY-STRIDED CASE---
// Applies when X == c * T(n-1) with c >= Z(n-1).
// Construct and apply:
// reshape([... Z(n-2), c, T(n-1)])
// slice([... (0, Z(n-1), 1), (0, Z(n), 1), (0, 1, 1)])
// head (after reshape and slice):
//   rank:    n
//   sizes:   [... Z(n-2), Z(n-1), 1]
//   strides: [... T(n-2), T(n-1), 1]
// which equals tail with one extra dimension.
//
//  ---UNEVENLY-STRIDED CASE---
// Applies when X = (Z(n-1) - 1) * T(n-1) + 1
// Construct and apply:
// slice([... (0, Z(n-2), 1), (0, X, T(n-1))])
// head (after slice):
//   rank:    n
//   sizes:   [... Z(n-2), Z(n-1)]
//   strides: [... T(n-2), T(n-1)]
//
// Simplification: If T(n-1) == 1, then the cases are equivalent (X == Z(n-1))
// and we can skip the reshape and just do the slice.
absl::Status BuildFinalDimension(StridedLayout& head_layout,
                                 const StridedLayout& tail_layout,
                                 ViewSequence& result,
                                 const int64_t head_pad_index) {
  int64_t last_dim_limit_index = -1;
  int64_t last_dim_stride = -1;

  const int64_t remaining_numel = head_layout.strided_dims.back().size;
  const int64_t last_size = tail_layout.strided_dims.back().size;
  const int64_t last_stride = tail_layout.strided_dims.back().stride;

  if (last_stride > 1 && remaining_numel % last_stride == 0 &&
      remaining_numel / last_stride >= last_size) {
    // Evenly-strided case: reshape to a final dimension of size=last_stride,
    // stride=1, and then slice it to size=1 with a dense slice (stride = 1).
    ReshapePrimitive reshape;
    reshape.base_sizes.reserve(head_layout.strided_dims.size());
    for (const auto& dim : head_layout.strided_dims) {
      reshape.base_sizes.push_back(dim.size);
    }
    reshape.new_sizes.reserve(head_layout.strided_dims.size() + 1);
    for (auto i = 0; i < head_layout.strided_dims.size() - 1; ++i) {
      reshape.new_sizes.push_back(head_layout.strided_dims[i].size);
    }
    reshape.new_sizes.push_back(remaining_numel / last_stride);
    reshape.new_sizes.push_back(last_stride);
    UpdateLayout(head_layout, reshape);
    ABSL_VLOG(3) << "[BuildFinalDimension] Appended " << reshape
                 << ", new head_layout: " << head_layout;
    result.push_back(std::move(reshape));
    last_dim_limit_index = 1;
    last_dim_stride = 1;
  } else {
    // Either the last_stride == 1 or the last dimension is not evenly-strided.
    // In either case, we just need to apply a slice with no reshape.
    last_dim_limit_index = (last_size - 1) * last_stride + 1;
    last_dim_stride = last_stride;
  }
  SlicePrimitive slice;
  for (auto i = 0; i < head_layout.strided_dims.size() - 1; ++i) {
    slice.slice_dims.push_back(
        SliceDimension{.start_index = 0,
                       .limit_index = tail_layout.strided_dims[i].size,
                       .stride = 1});
  }
  slice.slice_dims.push_back(SliceDimension{.start_index = 0,
                                            .limit_index = last_dim_limit_index,
                                            .stride = last_dim_stride});

  PadDimension& head_pad =
      std::get<PadPrimitive>(result[head_pad_index]).pad_dims[0];
  SliceDimension& head_slice =
      std::get<SlicePrimitive>(result[head_pad_index + 1]).slice_dims[0];
  ShiftInitialViewWindow(head_pad, head_slice, head_layout, slice);

  UpdateLayout(head_layout, slice);
  ABSL_VLOG(3) << "[BuildFinalDimension] Appended " << slice
               << ", new head_layout: " << head_layout;
  result.push_back(std::move(slice));
  return absl::OkStatus();
}

absl::Status RemoveExtraDim(StridedLayout& head_layout,
                            const StridedLayout& tail_layout,
                            ViewSequence& result,
                            absl::Span<ViewPrimitive> tail_sequence) {
  if (std::holds_alternative<ReshapePrimitive>(tail_sequence.front())) {
    // The head_layout may have an extra dimension, update the reshape
    // base_sizes to match.
    ReshapePrimitive& reshape =
        std::get<ReshapePrimitive>(tail_sequence.front());
    reshape.base_sizes.clear();
    reshape.base_sizes.reserve(head_layout.strided_dims.size());
    for (const auto& dim : head_layout.strided_dims) {
      reshape.base_sizes.push_back(dim.size);
    }
    return absl::OkStatus();
  }

  // Otherwise the tail sequence should be a single BroadcastPrimitive.
  ABSL_CHECK(tail_sequence.size() == 1);  // CRASH_OK
  BroadcastPrimitive& broadcast =
      std::get<BroadcastPrimitive>(tail_sequence.front());

  // Determine which dimensions in new_sizes are being inserted by the
  // broadcast.
  absl::InlinedVector<bool, 6> is_inserted_dim(broadcast.new_sizes.size(),
                                               true);
  for (auto dim : broadcast.broadcast_dimensions) {
    is_inserted_dim[dim] = false;
  }
  // Choose the lowest-index one arbitrarily.
  int64_t first_inserted_dim = -1;
  for (int64_t i = 0; i < is_inserted_dim.size(); ++i) {
    if (is_inserted_dim[i]) {
      first_inserted_dim = i;
      break;
    }
  }
  ABSL_CHECK_GE(first_inserted_dim, 0);  // CRASH_OK
  // The extra dimension will be broadcasted to this.
  // new_sizes is unchanged.
  broadcast.broadcast_dimensions.push_back(first_inserted_dim);
  broadcast.base_shape.push_back(head_layout.strided_dims.back().size);

  return absl::OkStatus();
}

}  // namespace

// Implementation notes:
//
// Solving for inner slice operations is tricky. It's easiest to work both
// forward and backwards and meet in the middle.
//
// We are starting from a contiguous base tensor, which can be flattened and
// bitcasted to interpret it as a 1D "storage" array of the target element type.
// But, we need to know how to reshape and slice it to only the elements in the
// target view.
//
// This is a lot easier if the view tensor is "well-behaved": row-major,
// with no size-1 dimensions or stride-0 dimensions. Not all views are
// well-behaved, but by working backwards, we can find a well-behaved view
// that can be transposed, reshaped, and expanded into the target view.
//
// Once we have our "storage" array and our "well-behaved" view, then we can
// use reshaping and slicing to satisfy the remaining operations.
absl::StatusOr<ViewSequence> DecomposeIntoViewSequence(
    absl::Span<const int64_t> contiguous_base_shape,
    mlir::ElementType contiguous_base_dtype, const StridedLayout& view_layout,
    mlir::ElementType view_dtype, bool is_conj) {
  CheckViewPreconditions(contiguous_base_shape, contiguous_base_dtype,
                         view_layout, view_dtype);

  // Work backwards, find the layout shape we need to reshape and slice to,
  // and note the last three operations necessary to achieve the view layout.
  // Make a copy of tail_layout and update it as we solve for the tail ops.
  ABSL_VLOG(3) << "[DecomposeIntoViewSequence] Base shape: "
               << ToString(contiguous_base_shape)
               << ", view layout: " << view_layout;

  ViewSequence result_sequence;

  StridedLayout tail_layout = view_layout;
  auto tail_sequence = BuildTail(tail_layout);
  ABSL_VLOG(3) << "[DecomposeIntoViewSequence] Built tail sequence: "
               << ToString(tail_sequence);

  StridedLayout head_layout = MakeContiguousBaseLayout(contiguous_base_shape);

  // Shortcut: if the output is purely some combination of unsqueeze,
  // transpose, and expand, then we're done.
  if (head_layout == tail_layout && contiguous_base_dtype == view_dtype) {
    ViewSequence result;
    for (auto& primitive : tail_sequence) {
      result.push_back(std::move(primitive));
    }
    if (is_conj && IsComplex(view_dtype)) {
      result.push_back(ConjPrimitive{});
    }
    return result;
  }

  // Work forwards, adjust the contiguous base to be a 1D storage buffer with
  // the right size and data type to cover the view window.
  TT_ASSIGN_OR_RETURN(
      (auto [result, view_widths]),
      BuildHead(head_layout, contiguous_base_dtype, tail_layout, view_dtype));
  ABSL_VLOG(3) << "[DecomposeIntoViewSequence] Built head sequence: "
               << ToString(result)
               << " with view widths: " << ToString(view_widths);

  // Shortcut: if the output has at most 1 non-trivial dimension
  // (size > 1 and stride > 0), then we're done without any additional
  // reshaping or slicing.
  if (head_layout == tail_layout) {
    for (auto& primitive : tail_sequence) {
      result.push_back(std::move(primitive));
    }
    if (is_conj && IsComplex(view_dtype)) {
      result.push_back(ConjPrimitive{});
    }
    return result;
  }

  // Otherwise, we need to do some amount of reshaping and slicing to finish
  // the decomposition.
  // BuildBodyStep will append to result, so we can't pass a direct reference
  // to the head pad operation (as it might get invalidated by a dynamic vector
  // resize), but the index of the pad operation will be stable.
  int64_t head_pad_index = -1;
  for (int i = 0; i < result.size(); ++i) {
    if (std::holds_alternative<PadPrimitive>(result[i])) {
      head_pad_index = i;
      break;
    }
  }
  ABSL_CHECK_GE(head_pad_index, 0);  // CRASH_OK
  while (head_layout.strided_dims.size() < tail_layout.strided_dims.size()) {
    TT_RETURN_IF_ERROR(BuildBodyStep(head_layout, tail_layout, view_widths,
                                     result, head_pad_index));
  }
  if (!tail_layout.strided_dims.empty() &&
      tail_layout.strided_dims.size() == head_layout.strided_dims.size() &&
      (tail_layout.strided_dims.back().size !=
           head_layout.strided_dims.back().size ||
       tail_layout.strided_dims.back().stride !=
           head_layout.strided_dims.back().stride)) {
    TT_RETURN_IF_ERROR(
        BuildFinalDimension(head_layout, tail_layout, result, head_pad_index));
  }

  if (head_layout.strided_dims.size() > tail_layout.strided_dims.size()) {
    // BuildBodyStep or BuildFinalDimension may add one
    // additional size=1, stride=1 dimension that needs to be squeezed out.
    // This may also require adjusting the tail sequence.
    TT_RETURN_IF_ERROR(RemoveExtraDim(head_layout, tail_layout, result,
                                      absl::MakeSpan(tail_sequence)));
  }

  for (auto& primitive : tail_sequence) {
    result.push_back(std::move(primitive));
  }
  if (is_conj && IsComplex(view_dtype)) {
    result.push_back(ConjPrimitive{});
  }
  return result;
}

absl::StatusOr<Dimensions> GetContiguousBaseShape(
    const StridedLayout& view_layout) {
  // If the view is valid, then we know the base must have at least this many
  // elements, but we don't know if it has more.
  int64_t min_numel = view_layout.storage_offset + 1;
  for (const auto& [size, stride] : view_layout.strided_dims) {
    if (size < 1) {
      return Dimensions{0};
    }
    min_numel += (size - 1) * stride;
  }

  // Do the decomposition and simplify it.
  TT_ASSIGN_OR_RETURN(
      auto one_d_view_sequence,
      DecomposeIntoViewSequence({min_numel}, mlir::ElementType::BF16,
                                view_layout, mlir::ElementType::BF16, false));
  Simplify(one_d_view_sequence, {min_numel});

  if (one_d_view_sequence.empty()) {
    // The view was exactly the 1D contiguous base we started with.
    return Dimensions{min_numel};
  }

  if (auto* reshape =
          std::get_if<ReshapePrimitive>(&one_d_view_sequence.front())) {
    // The view starts with a reshape from the 1D minimum element shape to
    // another, better shape. Use that one instead.
    return std::move(reshape->new_sizes);
  }

  // The view starts with a non-reshape operation, so we can't improve the
  // base shape.
  return Dimensions{min_numel};
}

}  // namespace torch_tpu
