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

#include "torch_tpu/ops/view_decomposition/view_sequence.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "c10/core/TensorImpl.h"
#include "llvm/ADT/STLExtras.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/unary.h"
#include "torch_tpu/ops/view_decomposition/bitcast_primitive.h"
#include "torch_tpu/ops/view_decomposition/broadcast_primitive.h"
#include "torch_tpu/ops/view_decomposition/conj_primitive.h"
#include "torch_tpu/ops/view_decomposition/pad_primitive.h"
#include "torch_tpu/ops/view_decomposition/reshape_primitive.h"
#include "torch_tpu/ops/view_decomposition/slice_primitive.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/transpose_primitive.h"
#include "torch_tpu/ops/view_decomposition/unfold_primitive.h"

namespace torch_tpu {

namespace {

bool RemoveNoOps(ViewSequence& sequence,
                 absl::Span<const int64_t> contiguous_base_shape) {
  if (sequence.empty()) {
    return false;
  }
  auto layout = MakeContiguousBaseLayout(contiguous_base_shape);
  size_t write_index = 0;
  bool any_no_ops = false;
  for (size_t read_index = 0; read_index < sequence.size(); ++read_index) {
    if (UpdateLayout(layout, sequence[read_index])) {
      if (write_index != read_index) {
        // Don't overwrite an operation onto itself;
        // if there is a prefix of meaningful operations, leave them as-is.
        sequence[write_index] = std::move(sequence[read_index]);
      }
      ++write_index;
    } else {
      ABSL_VLOG(3) << "[RemoveNoOps] Removing no-op: " << sequence[read_index];
      any_no_ops = true;
    }
  }
  if (any_no_ops) {
    sequence.resize(write_index);
  }
  return any_no_ops;
}

// Returns true if the second primitive was merged into the first.
// If the merge occurred, the second primitive is invalidated.
template <typename T>
bool MaybeMergeSequential(T& first, ViewPrimitive& second) {
  if (std::holds_alternative<T>(second)) {
    ABSL_VLOG(3) << "[MergeAllSequential] Merging " << first << " and "
                 << second;
    first = Merge(std::move(first), std::move(std::get<T>(second)));
    ABSL_VLOG(3) << "[MergeAllSequential] Merged into " << first;
    return true;
  }
  return false;
}
// The decomposition sequence will only produce at most one broadcast, one
// transpose, one pad, and one bitcast each of complex-to-real, real-to-real,
// and real-to-complex, so we don't need to worry about merging them.
// Multiple unfolds may be necessary, but they cannot be merged (as they
// can only do one concatenation per primitive).
// Only reshapes and slices need to be merged.
template <>
bool MaybeMergeSequential(BroadcastPrimitive& first, ViewPrimitive& second) {
  return false;
}
template <>
bool MaybeMergeSequential(TransposePrimitive& first, ViewPrimitive& second) {
  return false;
}
template <>
bool MaybeMergeSequential(PadPrimitive& first, ViewPrimitive& second) {
  return false;
}
template <>
bool MaybeMergeSequential(RealToRealBitcast& first, ViewPrimitive& second) {
  return false;
}
template <>
bool MaybeMergeSequential(ComplexToRealBitcast& first, ViewPrimitive& second) {
  return false;
}
template <>
bool MaybeMergeSequential(ViewAsComplex& first, ViewPrimitive& second) {
  return false;
}
template <>
bool MaybeMergeSequential(UnfoldPrimitive& first, ViewPrimitive& second) {
  return false;
}
template <>
bool MaybeMergeSequential(ConjPrimitive& first, ViewPrimitive& second) {
  if (auto* second_conj = std::get_if<ConjPrimitive>(&second)) {
    // If they are exactly the same, they cancel out.
    // (active + active) -> inactive
    // (inactive + inactive) -> inactive
    // Note: RemoveNoOps runs before this, so we generally only see active ones.
    if (first == *second_conj) {
      first.is_set = false;
      return true;
    }
  }
  return false;
}

template <>
bool MaybeMergeSequential(ViewPrimitive& first, ViewPrimitive& second) {
  return std::visit(
      [&second](auto& first) { return MaybeMergeSequential(first, second); },
      first);
}

bool MergeAllSequential(ViewSequence& sequence) {
  if (sequence.size() < 2) {
    return false;
  }
  bool any_merges = false;
  size_t read_index = 0;  // Index of the last non-merged primitive so far.
  for (size_t to_merge_index = 1; to_merge_index < sequence.size();
       ++to_merge_index) {
    const bool merged =
        MaybeMergeSequential(sequence[read_index], sequence[to_merge_index]);
    if (merged) {
      any_merges = true;
      continue;
    }
    ++read_index;
    if (read_index == to_merge_index) {
      // Do nothing for the non-mergeable prefix of the list
      continue;
    }
    sequence[read_index] = std::move(sequence[to_merge_index]);
  }
  if (any_merges) {
    sequence.resize(read_index + 1);
  }
  return any_merges;
}

}  // namespace

bool operator==(const ViewPrimitive& lhs, const ViewPrimitive& rhs) {
  if (lhs.index() != rhs.index()) {
    return false;
  }
  if (auto* lhs_reshape = std::get_if<ReshapePrimitive>(&lhs)) {
    const auto& rhs_reshape = std::get<ReshapePrimitive>(rhs);
    return lhs_reshape->base_sizes == rhs_reshape.base_sizes &&
           lhs_reshape->new_sizes == rhs_reshape.new_sizes;
  }
  if (auto* lhs_transpose = std::get_if<TransposePrimitive>(&lhs)) {
    return lhs_transpose->permutation ==
           std::get<TransposePrimitive>(rhs).permutation;
  }
  if (auto* lhs_broadcast = std::get_if<BroadcastPrimitive>(&lhs)) {
    const auto& rhs_broadcast = std::get<BroadcastPrimitive>(rhs);
    return lhs_broadcast->base_shape == rhs_broadcast.base_shape &&
           lhs_broadcast->new_sizes == rhs_broadcast.new_sizes &&
           lhs_broadcast->broadcast_dimensions ==
               rhs_broadcast.broadcast_dimensions;
  }
  if (auto* lhs_slice = std::get_if<SlicePrimitive>(&lhs)) {
    return lhs_slice->slice_dims == std::get<SlicePrimitive>(rhs).slice_dims;
  }
  if (auto* lhs_pad = std::get_if<PadPrimitive>(&lhs)) {
    return lhs_pad->pad_dims == std::get<PadPrimitive>(rhs).pad_dims;
  }
  if (auto* lhs_real_to_real = std::get_if<RealToRealBitcast>(&lhs)) {
    const auto& rhs_real_to_real = std::get<RealToRealBitcast>(rhs);
    return lhs_real_to_real->from_type == rhs_real_to_real.from_type &&
           lhs_real_to_real->to_type == rhs_real_to_real.to_type;
  }
  if (auto* lhs_complex_to_real = std::get_if<ComplexToRealBitcast>(&lhs)) {
    const auto& rhs_complex_to_real = std::get<ComplexToRealBitcast>(rhs);
    return lhs_complex_to_real->complex_element_type ==
               rhs_complex_to_real.complex_element_type &&
           lhs_complex_to_real->bitcast_type ==
               rhs_complex_to_real.bitcast_type;
  }
  if (auto* lhs_view_as_complex = std::get_if<ViewAsComplex>(&lhs)) {
    return lhs_view_as_complex->complex_element_type ==
           std::get<ViewAsComplex>(rhs).complex_element_type;
  }
  if (auto* lhs_unfold = std::get_if<UnfoldPrimitive>(&lhs)) {
    const auto& rhs_unfold = std::get<UnfoldPrimitive>(rhs);
    return lhs_unfold->start_index == rhs_unfold.start_index &&
           lhs_unfold->limit_index == rhs_unfold.limit_index &&
           lhs_unfold->window_stride == rhs_unfold.window_stride &&
           lhs_unfold->window_size == rhs_unfold.window_size;
  }
  if (std::holds_alternative<ConjPrimitive>(lhs)) {
    const auto& lhs_conj = std::get<ConjPrimitive>(lhs);
    const auto& rhs_conj = std::get<ConjPrimitive>(rhs);
    return lhs_conj.is_set == rhs_conj.is_set;
  }
  ABSL_CHECK(false) << "Unknown view primitive";  // CRASH_OK
}

bool operator!=(const ViewPrimitive& lhs, const ViewPrimitive& rhs) {
  return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const ViewPrimitive& primitive) {
  // For some reason, std::visit doesn't work here.
  if (auto* reshape = std::get_if<ReshapePrimitive>(&primitive)) {
    return os << *reshape;
  }
  if (auto* transpose = std::get_if<TransposePrimitive>(&primitive)) {
    return os << *transpose;
  }
  if (auto* broadcast = std::get_if<BroadcastPrimitive>(&primitive)) {
    return os << *broadcast;
  }
  if (auto* slice = std::get_if<SlicePrimitive>(&primitive)) {
    return os << *slice;
  }
  if (auto* pad = std::get_if<PadPrimitive>(&primitive)) {
    return os << *pad;
  }
  if (auto* real_to_real = std::get_if<RealToRealBitcast>(&primitive)) {
    return os << *real_to_real;
  }
  if (auto* complex_to_real = std::get_if<ComplexToRealBitcast>(&primitive)) {
    return os << *complex_to_real;
  }
  if (auto* view_as_complex = std::get_if<ViewAsComplex>(&primitive)) {
    return os << *view_as_complex;
  }
  if (auto* unfold = std::get_if<UnfoldPrimitive>(&primitive)) {
    return os << *unfold;
  }
  if (std::holds_alternative<ConjPrimitive>(primitive)) {
    const auto& conj = std::get<ConjPrimitive>(primitive);
    if (conj.is_set) {
      return os << "conj()";
    } else {
      return os << "conj(is_set=false)";
    }
  }
  ABSL_CHECK(false) << "Unknown view primitive";  // CRASH_OK
}

bool UpdateLayout(StridedLayout& layout, const ViewPrimitive& primitive) {
  return std::visit(
      [&layout](const auto& primitive) {
        return UpdateLayout(layout, primitive);
      },
      primitive);
}

absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(mlir::MlirOp input,
                                               const ViewPrimitive& primitive) {
  return std::visit(
      [&input](const auto& primitive) {
        if constexpr (std::is_same_v<decltype(primitive),
                                     const ConjPrimitive&>) {
          if (primitive.is_set) {
            return BuildConjPhysicalShlo(input);
          }
          return absl::StatusOr<mlir::MlirOp>(input);
        } else {
          return ViewPrimitiveShlo(input, primitive);
        }
      },
      primitive);
}

bool UpdateLayout(StridedLayout& layout, ViewSequenceSpan view_sequence) {
  bool updated = false;
  for (const auto& primitive : view_sequence) {
    updated |= UpdateLayout(layout, primitive);
  }
  return updated;
}

void Simplify(ViewSequence& sequence,
              absl::Span<const int64_t> contiguous_base_shape) {
  ABSL_VLOG(3) << "[Simplify] Pre-simplified view sequence: "
               << ToString(sequence);
  bool any_changes = false;
  do {
    const bool any_no_ops = RemoveNoOps(sequence, contiguous_base_shape);
    const bool any_merges = MergeAllSequential(sequence);
    any_changes = any_no_ops || any_merges;
  } while (any_changes);
  ABSL_VLOG(3) << "[Simplify] Simplified view sequence: " << ToString(sequence);
}

absl::StatusOr<mlir::MlirOp> ViewSequenceShlo(mlir::MlirOp input,
                                              ViewSequenceSpan view_sequence) {
  mlir::MlirOp current = input;
  for (const auto& primitive : view_sequence) {
    TT_ASSIGN_OR_RETURN(current, ViewPrimitiveShlo(current, primitive));
  }
  return current;
}

namespace {

struct ViewCacheKeyVisitor {
  // Special handling for reshape primitive since we only support affine
  // reshapes currently.
  // TODO: Enhance this to support squeeze/unsqueeze/expand.
  absl::StatusOr<std::string> operator()(
      const ReshapePrimitive& primitive) const {
    TT_ASSIGN_OR_RETURN(
        const ReshapeReassociation reassociation,
        GetReshapeReassociation(primitive.base_sizes, primitive.new_sizes));

    std::string suffix;

    if (reassociation.type == ReshapeType::kExpand) {
      // Expand reassociation maps inputs to outputs:
      //  [6,4] -> [3,2,4,1] : {{0,1}, {2,3}}
      // Meaning in[0] is distributed between output {out[0],out[1]} and in[1]
      // over {out[2],out[3]}.
      //
      // Encode a suffix for expands to disambiguate between different
      // factorizations and unsqueezes:
      //  [6] -> [2,3] or [3,2], use suffix `d0=3,d1=2`
      //  [6] -> [1,6] or [6,1], use suffix `d0=1`

      // Special case scalar unsqueeze {} -> {1,...,1}
      if (reassociation.reassociation.empty()) {
        absl::StrAppend(&suffix, ":scalar_unsqueeze(",
                        primitive.new_sizes.size(), ")");
      }
      for (int64_t i = 0; i < reassociation.reassociation.size(); ++i) {
        const auto& dim_vec = reassociation.reassociation[i];
        if (dim_vec.size() == 1) {
          continue;
        }
        // Input dim is split, encode static output dim sizes where in[i] !=
        // out[dim].
        for (int64_t dim : dim_vec) {
          if (i < primitive.base_sizes.size() &&
              primitive.base_sizes[i] == primitive.new_sizes[dim]) {
            continue;
          }
          std::string comma = suffix.empty() ? "" : ",";
          absl::StrAppend(&suffix, comma, "d", dim, "=",
                          primitive.new_sizes[dim]);
        }
      }
    } else if (reassociation.type == ReshapeType::kCollapse) {
      // Special case scalar squeeze {1,...,1} -> {}
      if (reassociation.reassociation.empty()) {
        absl::StrAppend(&suffix, ":scalar_squeeze");
      }
    } else if (reassociation.type == ReshapeType::kTransposeLike) {
      // Encode a suffix for where the 1s are in the output.
      for (int64_t dim = 0; dim < primitive.new_sizes.size(); ++dim) {
        if (primitive.new_sizes[dim] != 1) {
          continue;
        }
        std::string comma = suffix.empty() ? ":" : ",";
        absl::StrAppend(&suffix, comma, "d", dim, "=",
                        primitive.new_sizes[dim]);
      }
    } else if (reassociation.type != ReshapeType::kFlatten) {
      ABSL_CHECK(  // CRASH_OK=this should be updated when adding support for
                   // other reshape types.
          false)
          << "unimplemented symbolic cache key for reassociation: "
          << ReassociationToString(reassociation);
    }

    return absl::StrCat("reshape:", ReassociationToString(reassociation),
                        suffix);
  }

  absl::StatusOr<std::string> operator()(
      const TransposePrimitive& primitive) const {
    return absl::StrCat("transpose:", "[",
                        absl::StrJoin(primitive.permutation, ","), "]");
  }

  absl::StatusOr<std::string> operator()(const ConjPrimitive& primitive) const {
    return absl::StrCat("conj:", primitive.is_set);
  }

  absl::StatusOr<std::string> operator()(
      const RealToRealBitcast& primitive) const {
    return absl::StrCat("real_to_real_bitcast:", ToString(primitive.from_type),
                        "->", ToString(primitive.to_type));
  }
  absl::StatusOr<std::string> operator()(
      const ComplexToRealBitcast& primitive) const {
    return absl::StrCat(
        "complex_to_real_bitcast:", ToString(primitive.complex_element_type),
        ":", ToString(primitive.bitcast_type));
  }

  absl::StatusOr<std::string> operator()(const ViewAsComplex& primitive) const {
    return absl::StrCat("view_as_complex:",
                        ToString(primitive.complex_element_type));
  }

  absl::StatusOr<std::string> operator()(const PadPrimitive& primitive) const {
    return absl::StrCat(
        "pad:[",
        absl::StrJoin(primitive.pad_dims, ",",
                      [](std::string* out, const PadDimension& pad_dim) {
                        absl::StrAppend(out, "{l", pad_dim.low_padding, ",h",
                                        pad_dim.high_padding, ",i",
                                        pad_dim.interior_padding, "}");
                      }),
        "]");
  }

  absl::StatusOr<std::string> operator()(
      const BroadcastPrimitive& primitive) const {
    std::string dims;
    for (int64_t i = 0; i < primitive.new_sizes.size(); ++i) {
      auto& bcast_dims = primitive.broadcast_dimensions;
      auto it = absl::c_find(bcast_dims, i);
      auto index = std::distance(bcast_dims.begin(), it);
      std::string comma = i > 0 ? "," : "";
      if (it != bcast_dims.end() && primitive.base_shape[index] > 1) {
        absl::StrAppend(&dims, comma, "in", index);
        continue;
      }
      absl::StrAppend(&dims, comma, primitive.new_sizes[i]);
    }
    return absl::StrCat("broadcast:[", dims, "]");
  }

  // Fallback for all other view primitives.
  // Currently unsupported:
  //  - UnfoldPrimitive relies on static shape information for output.
  //  - SlicePrimitive is often used for masking, and slicing along a dynamic
  //    dimension is a cache miss, not worth a partially symbolic key.
  template <typename T>
  absl::StatusOr<std::string> operator()(const T& primitive) const {
    return TT_ERROR(error::kUnimplemented) << "View primitive does not support "
                                              "dynamic cache keys for type: "
                                           << typeid(T).name();
  }
};

absl::StatusOr<std::string> ViewToCacheKey(const ViewPrimitive& primitive) {
  ViewCacheKeyVisitor visitor{};
  return std::visit(visitor, primitive);
}

absl::StatusOr<std::string> SymbolicViewCacheKey(
    ViewSequenceSpan view_sequence) {
  ABSL_VLOG(3) << "[SymbolicViewCacheKey] View sequence: "
               << ToString(view_sequence);
  std::string view_sequence_cache_key;
  for (auto [index, primitive] : llvm::enumerate(view_sequence)) {
    TT_ASSIGN_OR_RETURN(std::string view_cache_key, ViewToCacheKey(primitive));
    ABSL_VLOG(3) << "[SymbolicViewCacheKey] Symbolic cache key: "
                 << view_cache_key;
    if (index > 0) {
      absl::StrAppend(&view_sequence_cache_key, ";");
    }
    absl::StrAppend(&view_sequence_cache_key, view_cache_key);
  }
  return view_sequence_cache_key;
}

}  // namespace

absl::StatusOr<OpParamCacheKeys> ViewSequenceCacheKey(
    ViewSequenceSpan view_sequence, const c10::TensorImpl& tensor) {
  // First try to create a symbolic cache key that represents view ops as
  // mappings from input to output tensor shapes, i.e. Reshape[A,B]->[A*B]
  // or transpose[A,B]{0,1}, instead of embedding static input and output shapes
  // into the cache key.
  absl::StatusOr<std::string> symbolic_key =
      SymbolicViewCacheKey(view_sequence);
  if (symbolic_key.ok()) {
    ABSL_VLOG(3) << "[ViewSequenceCacheKey] Built symbolic cache key";
    return *OpParamCacheKeysBuilder().SetParam("view", *symbolic_key);
  }

  // Fall back to using static tensor shapes for the cache key.
  ABSL_VLOG(3) << "[ViewSequenceCacheKey] Failed to create symbolic cache key: "
               << symbolic_key.status().message();
  return *OpParamCacheKeysBuilder()
              .SetParam("strides", tensor.strides())
              .SetParam("storage_offset", tensor.storage_offset());
}

}  // namespace torch_tpu
