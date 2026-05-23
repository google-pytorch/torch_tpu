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

#ifndef TORCH_TPU_OPS_VIEW_DECOMPOSITION_VIEW_SEQUENCE_H_
#define TORCH_TPU_OPS_VIEW_DECOMPOSITION_VIEW_SEQUENCE_H_

#include <cstdint>
#include <ostream>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/TensorImpl.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/ops/view_decomposition/bitcast_primitive.h"
#include "torch_tpu/ops/view_decomposition/broadcast_primitive.h"
#include "torch_tpu/ops/view_decomposition/conj_primitive.h"
#include "torch_tpu/ops/view_decomposition/pad_primitive.h"
#include "torch_tpu/ops/view_decomposition/reshape_primitive.h"
#include "torch_tpu/ops/view_decomposition/slice_primitive.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/transpose_primitive.h"
#include "torch_tpu/ops/view_decomposition/unfold_primitive.h"

// A ViewSequence is a sequence of view primitives that can be applied to a
// contiguous base tensor to produce a strided tensor.

namespace torch_tpu {

// A view primitive can be a reshape, transpose, broadcast, slice, pad, bitcast,
// unfold, or conjugate.
using ViewPrimitive =
    std::variant<ReshapePrimitive, TransposePrimitive, BroadcastPrimitive,
                 SlicePrimitive, PadPrimitive, RealToRealBitcast,
                 ComplexToRealBitcast, ViewAsComplex, UnfoldPrimitive,
                 ConjPrimitive>;
static_assert(alignof(ViewPrimitive) == 8, "");
static_assert(sizeof(ViewPrimitive) == 176, "");

bool operator==(const ViewPrimitive& lhs, const ViewPrimitive& rhs);
bool operator!=(const ViewPrimitive& lhs, const ViewPrimitive& rhs);

// Formats the view primitive as one of:
//   * "reshape[1, 2, 3]"
//   * "transpose[1, 2, 0]"
//   * "broadcast(new_sizes=[1, 2, 3], broadcast_dimensions=[1, 2])"
//   * "slice[(start=1, limit=2, stride=3), (start=4, limit=5, stride=6)]"
//   * "pad[(low=1, high=2, interior=3), (low=4, high=5, interior=6)]"
//   * "bitcast(from_type=UI16, to_type=UI32)"
//   * "view_as_real(cfloat)"
//   * "real(cfloat)"
//   * "imag(cfloat)"
//   * "view_as_complex(cfloat)"
//   * "unfold(start_index=1, limit_index=2, window_stride=3, window_size=4)"
std::ostream& operator<<(std::ostream& os, const ViewPrimitive& primitive);

// Updates the layout to reflect the effect of applying the given view
// primitive.
// Returns true if the layout was modified.
bool UpdateLayout(StridedLayout& layout, const ViewPrimitive& primitive);

// Applies the given view primitive to the input, using the corresponding
// StableHLO op (reshape, transpose, broadcast_in_dim, slice, pad, bitcast,
// or unfold).
absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(mlir::MlirOp input,
                                               const ViewPrimitive& primitive);

using ViewSequence = std::vector<ViewPrimitive>;
using ViewSequenceSpan = absl::Span<const ViewPrimitive>;

// Updates the layout to reflect the effect of applying the given view
// sequence.
// Returns true if the layout was modified.
bool UpdateLayout(StridedLayout& layout, ViewSequenceSpan view_sequence);

// Simplifies a sequence of view primitives by removing no-ops and merging
// sequentially-applied primitives.
// The sequence is modified in-place.
void Simplify(ViewSequence& view_sequence,
              absl::Span<const int64_t> contiguous_base_shape);

// An MlirOpBuilder to apply the given view sequence to the input.
absl::StatusOr<mlir::MlirOp> ViewSequenceShlo(mlir::MlirOp input,
                                              ViewSequenceSpan view_sequence);

// Returns the cache key for the given view sequence and tensor.
// ViewSequence cache keys aim to use generic cache keys whenever possible,
// instead of embedding static tensor shapes into the cache key.
// Ex: Basic reshape of [5] -> [5,1] will be stored as [A] -> [A, 1],
// the cache uses input parameter shapes to avoid false cache hits.
absl::StatusOr<OpParamCacheKeys> ViewSequenceCacheKey(
    ViewSequenceSpan view_sequence, const c10::TensorImpl& tensor);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_VIEW_DECOMPOSITION_VIEW_SEQUENCE_H_
