/*
 * Copyright 2026 Google LLC
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

#ifndef TORCH_TPU_CSRC_OPS_INDEX_UTILS_H_
#define TORCH_TPU_CSRC_OPS_INDEX_UTILS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/List.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "csrc/common/dimension_types.h"

namespace torch_tpu {

struct UnexpandedZeroStrideIndex {
  at::Tensor tensor;  // UNINITIALIZED_TENSOR_OK
  Dimensions squeezed_dims;
  Dimensions bcast_dims;
  Dimensions logical_dims;
};

// When `index` is an expanded view with zero-stride dimensions (such as
// multi-head `c2p_dynamic_expand` passed across a graph break or in Eager
// mode), un-expand the zero-stride axes to a contiguous zero-copy view of its
// backing storage buffer so `DispatchOp` does not launch an ephemeral
// `tt_jit_execute_as_strided` kernel and `BuildGatherShlo`/`BuildScatterShlo`
// sees the broadcast.
std::optional<UnexpandedZeroStrideIndex> TryUnexpandZeroStrideIndex(
    const at::Tensor& index);

// One defined index tensor of an advanced-indexing indices list, as used by
// `index` and `index_put_`.
struct IndexEntry {
  at::Tensor tensor;  // UNINITIALIZED_TENSOR_OK
  // First dimension of `self` that `tensor` indexes.
  int64_t first_dim = 0;
  // Whether `tensor` is a bool mask, which covers `tensor.dim()` dimensions of
  // `self` starting at `first_dim`. Otherwise, it is an integer index that
  // covers dimension `first_dim`.
  bool is_mask = false;
};

// Returns the first dimension of `self` indexed by each of `entries`.
Indices GetFirstDims(absl::Span<const IndexEntry> entries);

// Resolves negative integer indices in `entries` by adding the size of the
// dimension of `self` (of shape `sizes`) they index. Bool masks are skipped.
//
// All `entries` must already be on TPU; callers move CPU index tensors to TPU
// with `MoveIndexToDevice` first.
absl::Status ResolveNegativeIndices(absl::Span<IndexEntry> entries,
                                    at::IntArrayRef sizes);

// Parses the advanced-indexing `indices` of `self` (as passed to `index` and
// `index_put_`, where an undefined entry is a full `:` slice) into one
// `IndexEntry` per defined index tensor. Checks, in this order, that:
//
// *   every defined index tensor has dtype bool, int32 or int64 (uint8 is
//     rejected), and is either on CPU or on the same device as `self`,
//     matching PyTorch's advanced indexing device rules
//     (`dev_tensor[{cpu,dev}_tensor]`);
// *   the indices fit within `self.dim()`, where a `:` slice or an integer
//     index consumes one dimension and a bool mask consumes `mask.dim()`
//     dimensions (`IndexError`);
// *   the shape of each bool mask matches the dimensions of `self` it covers
//     (`IndexError`);
// *   at least one index tensor is defined.
//
// The entries hold the index tensors as given; CPU tensors are not moved to the
// device (see `MoveIndexToDevice`), so that callers can process CPU masks on
// the host.
absl::StatusOr<std::vector<IndexEntry>> ParseIndexEntries(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices);

// Returns `index` on the TPU `device`, copying it there if it is on CPU (i.e. a
// CPU index tensor used to index a TPU tensor). `index` must be on CPU or
// already on `device`, as checked by `ParseIndexEntries`.
absl::StatusOr<at::Tensor> MoveIndexToDevice(const at::Tensor& index,
                                             const c10::Device& device);

absl::StatusOr<int64_t> ValidateIndexInputsAndGetDim(const at::Tensor& self,
                                                     int64_t dim,
                                                     const at::Tensor& index,
                                                     const at::Tensor& source);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_INDEX_UTILS_H_
