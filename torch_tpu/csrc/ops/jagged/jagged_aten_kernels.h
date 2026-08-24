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

#ifndef TORCH_TPU_OPS_JAGGED_JAGGED_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_JAGGED_JAGGED_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"

namespace torch_tpu {

// Converts a jagged nested tensor's flat values buffer into a padded dense
// tensor on TPU.
//
// Arguments:
//   - values: 1D or ND tensor containing concatenated jagged sequence tokens
//     with shape (total_L, *trailing_feature_dims).
//   - offsets: 1D int64 tensor list (single jagged dimension) with shape
//     (batch_size + 1,) indicating sequence boundary offsets.
//   - max_lengths: SymIntArrayRef containing the target sequence length
//     [max_length] for the padded dense dimension. Sequences longer than
//     max_length are truncated, while shorter sequences are padded with
//     padding_value.
//   - padding_value: Constant scalar value used to fill unoccupied positions
//     in the padded dense output.
//
// Returns:
//   A dense tensor of shape (batch_size, max_length, *trailing_feature_dims).
at::Tensor AtenJaggedToPaddedDenseForward(const at::Tensor& values,
                                          at::TensorList offsets,
                                          c10::SymIntArrayRef max_lengths,
                                          double padding_value);

// Converts a padded dense tensor back into a flat jagged values buffer on TPU.
//
// Arguments:
//   - dense: Tensor of shape (batch_size, max_length, *trailing_feature_dims).
//   - offsets: 1D int64 tensor list with shape (batch_size + 1,) indicating
//     the destination sequence lengths and boundaries in the jagged output.
//   - total_L: Optional total number of valid elements expected in the
//     output values buffer. Defaults to offsets[-1] if not specified.
//
// Returns:
//   A flat jagged values tensor of shape (total_L, *trailing_feature_dims).
at::Tensor AtenPaddedDenseToJaggedForward(const at::Tensor& dense,
                                          at::TensorList offsets,
                                          std::optional<c10::SymInt> total_L);

// Converts a flat jagged values tensor into a nested view tensor.
//
// Arguments:
//   - self: Flat 1D or ND jagged values tensor.
//   - offsets: 1D int64 offsets tensor.
//   - dummy: Dummy jagged tensor used for dispatch.
//   - lengths: Optional 1D int64 lengths tensor.
//   - ragged_idx: Index of the ragged dimension (default 1).
//   - min_seqlen: Optional minimum sequence length tensor.
//   - max_seqlen: Optional maximum sequence length tensor.
//
// Returns:
//   An aliased view of self.
at::Tensor AtenNestedViewFromJagged(
    const at::Tensor& self, const at::Tensor& offsets, const at::Tensor& dummy,
    const std::optional<at::Tensor>& lengths, int64_t ragged_idx,
    const std::optional<at::Tensor>& min_seqlen,
    const std::optional<at::Tensor>& max_seqlen);

// Converts a padded dense tensor into a nested jagged view tensor.
at::Tensor AtenNestedFromPaddedTensor(
    const at::Tensor& padded, const at::Tensor& offsets,
    const at::Tensor& dummy, int64_t ragged_idx = 1,
    const std::optional<at::Tensor>& min_seqlen = std::nullopt,
    const std::optional<at::Tensor>& max_seqlen = std::nullopt,
    std::optional<c10::SymInt> sum_S = std::nullopt);

// Extracts the underlying values buffer from a nested jagged tensor.
at::Tensor AtenNestedGetValues(const at::Tensor& self);

// Extracts the offsets tensor from a nested jagged tensor.
at::Tensor AtenNestedGetOffsets(const at::Tensor& self);

// Extracts the lengths tensor from a nested jagged tensor.
at::Tensor AtenNestedGetLengths(const at::Tensor& self);

// Extracts the ragged index from a nested jagged tensor.
int64_t AtenNestedGetRaggedIdx(const at::Tensor& self);

// Extracts the min sequence length tensor from a nested jagged tensor.
at::Tensor AtenNestedGetMinSeqlen(const at::Tensor& self);

// Extracts the max sequence length tensor from a nested jagged tensor.
at::Tensor AtenNestedGetMaxSeqlen(const at::Tensor& self);

// Returns a jagged dummy tensor for dispatching nested operations.
at::Tensor AtenNestedGetJaggedDummy(const at::Tensor& any);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_JAGGED_JAGGED_ATEN_KERNELS_H_
