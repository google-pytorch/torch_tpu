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

#ifndef TORCH_TPU_OPS_VIEW_DECOMPOSITION_INVERSION_H_
#define TORCH_TPU_OPS_VIEW_DECOMPOSITION_INVERSION_H_

#include <cstdint>
#include <optional>
#include <ostream>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/ops/view_decomposition/conj_primitive.h"
#include "torch_tpu/ops/view_decomposition/reshape_primitive.h"
#include "torch_tpu/ops/view_decomposition/slice_primitive.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/transpose_primitive.h"
#include "torch_tpu/ops/view_decomposition/view_sequence.h"

// PyTorch supports mutation tensor data through shallows views. That is, you
// can do operations like
// ```python
// x = torch.zeros(3)
// y = x[1]
// y += 1
// print(x)
// ```
// and the expected result is `[0, 1, 0]`.
//
// However, PjRtBuffers are generally immutable; we can change `x` to be
// a pointer to a new buffer, but we cannot directly edit the value of `x[1]`
// like CPU or CUDA devices can. Instead, we need to combine the old, unmodified
// buffer with the updated view to create a new buffer with the mutated value.
//
// To construct this new buffer, we need to be able to *invert* the view being
// mutated. This requires having:
//   * The contiguous base tensor with the pre-update values; in the above
//     example, this is (`[0, 0, 0]`)
//   * The stride pattern of the view that has been updated; in the above
//     example, this is (size=(), stride=(), offset=1)
//   * The updated values of the view; in the above example, this is scalar `1`
//   * The element type of the base tensor and the view.
//
// We first need to cast both tensors to a common dtype. To ensure no loss of
// data, we need to carefully handle two cases:
//   * If the dtypes are different sizes, we need to cast both tensors to the
//     smaller bitwidth of the two dtypes to avoid truncation, and ensure that
//     partial updates to larger base elements preserve unmodified bits.
///  * If the contiguous base tensor is boolean type, then it needs to be
//     converted to a uint8 tensor to preserve all assigned bytes (and not
//     collapse them to only 0x00 or 0x01 values).
//
// Once the types are aligned, we can compute the shape-only decomposition from
// the base tensor to the view using the logic in decomposition.h, and then
// invert the operations:
//   * Reshape is inverted by another reshape
//   * Transpose is inverted by another transpose
//   * Slice is inverted by a pad
//   * Pad is inverted by a slice
//
// Once we have the inverse operations, we can construct two update tensors:
//   * A tensor with the same values as the updated view (possibly bitcasted),
//     but in their original positions relative to the base tensor. Any values
//     not present in the view are replaced with arbitrary padding (zeroes).
//   * A mask tensor with `true` in all positions that were updated and `false`
//     in all other positions.
// These can then be combined with StableHLO select (equivalent to torch.where)
// to compute the updated contiguous base tensor.
//
// If there are no pads in the inverse (no slices in the view) then the mask
// would be all `true` and we can avoid the final select.
// If there is a single pad at the end with no interior padding then the more
// performant dynamic_update_slice can be used instead of pad+select.
//
// Note that overlapping views (including broadcasted dimensions) are not
// invertible. This is because if multiple index values refer to the same
// physical memory location, then writing to those indexes creates multiple
// competing writes to that location, which has undefined behavior (on CPU,
// CUDA, and TPU alike). We raise an exception in this case.

namespace torch_tpu {

// Only some view primitives can be trivially inverted with no ambiguity or loss
// of information.
//
// The invertible operations are:
// * Reshapes and transposes are invertible, as they preserve all data.
//   Reshapes are inverted by reshape, transpose is inverted by transpose.
// * Pads are invertible because dummy pad values do not need to be preserved.
//   Pads are inverted by slice.
// * Conj is invertible because it only flips the sign bit of the imag part.
//   Conj is inverted by conj to undo the flip.
//
// For the other operations:
// * Some bitcasts are invertible, but others cause a loss of information (for
//   example, uint8 -> bool); these are handled separately.
// * Slices are not trivially invertible, as they remove data. These require
//   handling using dynamic update slice or stablehlo.select to merge with the
//   pre-slice value.
// * BroadcastInDim and Unfold are not invertible as they are one-to-many, not
//   one-to-one, which is undefined behavior when inverted.
using InverseViewPrimitive = std::variant<ReshapePrimitive, TransposePrimitive,
                                          SlicePrimitive, ConjPrimitive>;
static_assert(sizeof(InverseViewPrimitive) == 120, "");

bool operator==(const InverseViewPrimitive& lhs,
                const InverseViewPrimitive& rhs);

// Formats the inverse view primitive as one of:
//   * "reshape[1, 2, 3]"
//   * "permute[1, 2, 0]"
//   * "slice[(start=1, limit=2, stride=3), (start=4, limit=5, stride=6)]"
//   * "conj()"
std::ostream& operator<<(std::ostream& os,
                         const InverseViewPrimitive& primitive);

// Updates the layout to reflect the effect of applying the given inverse view
// primitive.
// Returns true if the layout was modified.
bool UpdateLayout(StridedLayout& layout, const InverseViewPrimitive& primitive);

// A inverse view sequence is a (possibly empty) sequence of inverse view
// primitives.
using InverseViewSequence = std::vector<InverseViewPrimitive>;
using InverseViewSequenceSpan = absl::Span<const InverseViewPrimitive>;

// One stage of a view inversion.
struct InverseViewStage {
  // The forward view sequence to be inverted.
  ViewSequence forward;
  // The slice that acts as a boundary between the forward sequence and the
  // next stage. Will be nullopt for the last stage (and only the last stage).
  std::optional<SlicePrimitive> slice;
  // The inverse of the forward sequence, excluding the slice.
  InverseViewSequence inverse;
};

bool operator==(const InverseViewStage& lhs, const InverseViewStage& rhs);
std::ostream& operator<<(std::ostream& os, const InverseViewStage& stage);

struct InverseViewOperation {
  // A (possibly empty) sequence of view operations to be applied to the
  // contiguous base tensor. This may include bitcasts and at most one reshape;
  // after these operations the layout will still be a contiguous base layout,
  // and the shape will be final_shape.
  ViewSequence base_transform;
  // A (possibly empty) sequence of bitcast operations to convert the view
  // to a common dtype with the base.
  ViewSequence bitcast_view;
  // One or more stages of view inversion.
  // There will be one more stage than there are slices in the view; all but
  // the last stage will have a non-null slice field.
  std::vector<InverseViewStage> stages;
  // The final shape of the return tensor.
  Shape final_shape;
};

// Computes the inverse view operation necessary to produce an updated
// contiguous base tensor from a mutated view.
absl::StatusOr<InverseViewOperation> ComputeInverseViewOperation(
    absl::Span<const int64_t> contiguous_base_shape,
    mlir::ElementType contiguous_base_dtype, const StridedLayout& view_layout,
    mlir::ElementType view_dtype, bool is_conj = false);

// Applies the inverse view operation to the mutated view, resulting in a
// tensor with the "final_shape" layout. This may retain some of the data from
// the contiguous base tensor, through stablehlo.dynamic_update_slice or
// stablehlo.select operations.
absl::StatusOr<mlir::MlirOp> InverseViewOperationShlo(
    mlir::MlirOp contiguous_base, mlir::MlirOp mutated_view,
    const InverseViewOperation& inverse_view_operation);

// Constructs the inverse of a view sequence, which will reconstruct the
// pre-view layout from a strided view.
//
// This will fail if the view sequence contains a slice operation, as special
// handling is required to merge an updated slice with the pre-slice value.
absl::StatusOr<InverseViewSequence> InvertViewSequence(
    StridedLayout& layout, ViewSequenceSpan view_sequence);

// Applies the inverse view sequence to the mutated view, resulting in a
// tensor with the same shape as the contiguous base.
// This is only possible when there are no slices in the forward decomposed view
// sequence, so that the inverse is a full overwrite of the contiguous base.
absl::StatusOr<mlir::MlirOp> InverseViewSequenceShlo(
    mlir::MlirOp mutated_view, InverseViewSequenceSpan inverted_view_sequence);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_VIEW_DECOMPOSITION_INVERSION_H_
