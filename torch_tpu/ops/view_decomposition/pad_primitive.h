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

#ifndef TORCH_TPU_OPS_VIEW_DECOMPOSITION_PAD_PRIMITIVE_H_
#define TORCH_TPU_OPS_VIEW_DECOMPOSITION_PAD_PRIMITIVE_H_

#include <cstdint>
#include <ostream>
#include <vector>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

// Pad is technically not a view op. Increasing the size of a tensor typically
// requires a copy.
//
// However, when decomposing a strided layout into StableHLO ops, it is
// sometimes necessary to pad with a value that is later sliced away to achieve
// certain stride patterns.
//
// For example, `torch.arange(5).as_strided(size=(2, 2), stride=(3,1))` is
// a valid restriding; it produce the tensor [[0, 1], [3, 4]]. But, this
// restriding cannot be expressed purely through StableHLO's reshape, transpose,
// slice, and broadcast_in_dim ops.
//
// That restriding can, however, be expressed as:
// ```
// torch.nn.functional.pad(
//   torch.arange(5),
//   pad=(0, 1),
//   value=0
// ).view(2, 3)[:, :2]
// ```
// or its StableHLO equivalents. The padded element is only include to make the
// reshape valid; the padde element is sliced away before the result is
// returned.
//
// As such, we include pad in the set of view primitives, so that view sequences
// can use it in decomposition of view ops.

namespace torch_tpu {

// One dimension of a pad primitive.
// For some input_size, the output dimension will have:
//   result_size[i] = low_padding[i] + (input_size[i]-1) * interior_padding[i] +
//                    high_padding[i]
struct PadDimension {
  // The number of rows to add to below the first row in the input dimension.
  int64_t low_padding = 0;
  // The number of rows to add to above the last row in the input dimension.
  int64_t high_padding = 0;
  // The number of rows to add between each element in the input dimension.
  int64_t interior_padding = 0;
};
static_assert(sizeof(PadDimension) == 24, "");

bool operator==(const PadDimension& lhs, const PadDimension& rhs);

// Formats the pad dimension like "(low=1, high=2, interior=3)".
std::ostream& operator<<(std::ostream& os, const PadDimension& dim);

// A pad primitive converts an N-dimensional tensor into another
// N-dimensional tensor, where:
//   result_size[i] = low_padding[i] + (input_size[i]-1) * interior_padding[i] +
//                    high_padding[i]
//   result_strides[i] = input_strides[i]
//
// Simplifications:
//   * If all padding values are zero, then this is a no-op and can be skipped.
//   * Back-to-back pads can be merged into a single pad.
struct PadPrimitive {
  std::vector<PadDimension> pad_dims;
};
// Using a std::vector instead of absl::InlinedVector to keep the size of the
// PadPrimitive struct below 64 bytes (one cache line on most x86 CPUs).
static_assert(sizeof(PadPrimitive) == 24, "");

bool operator==(const PadPrimitive& lhs, const PadPrimitive& rhs);

// Formats the pad like
// "pad([(low=1, high=2, interior=3), (low=4, high=5, interior=6)])".
std::ostream& operator<<(std::ostream& os, const PadPrimitive& pad);

// Updates the layout to reflect the effect of applying the given pad
// primitive.
//
// Crashes if the number of dimensions does not match, or if the
// pad values are negative.
//
// Returns true if the layout was modified.
bool UpdateLayout(StridedLayout& layout, const PadPrimitive& pad);

// An overload for the ViewPrimitiveShlo function that handles pad
// primitives. This is a direct call into the stablehlo::Pad function; the
// type of the input argument is used to determine the type of the padding value
// (0 for integer types, 0.0 for floating point types).
absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(mlir::MlirOp input,
                                               const PadPrimitive& pad);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_VIEW_DECOMPOSITION_PAD_PRIMITIVE_H_
