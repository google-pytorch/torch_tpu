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

#ifndef TORCH_TPU_OPS_INDEX_PUT_INDEX_PUT_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_INDEX_PUT_INDEX_PUT_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

//
// These function implements the behavior of torch.index_put for the TPU backend
// See: https://pytorch.org/docs/stable/generated/torch.Tensor.index_put_.html
//
//    torch.tensor.index_put_(indices, values, accumulate=False)
//
// The function updates the 'self' tensor by assigning, or accumulating values
// from the 'values' tensor into locations specified by the 'indices' tuple.
//
// Parameters:
//
//   at::Tensor& self:
//     The tensor to be modified. Let its shape be D = (D_0, D_1, ..., D_{N-1}),
//     where N is the number of dimensions (self.ndim).
//
//   const c10::List<std::optional<at::Tensor>>& indices:
//     A list of indexers, with length M, where M <= N. The element indices[i]
//     corresponds to dimension i of the 'self' tensor.
//
//     Each element in 'indices' can be one of the following:
//
//     1.  Long or Int Tensor (dtype torch.int64, torch.int32):
//         Provides the indices to select along dimension `i`. This tensor can
//         have any number of dimensions, including being a scalar (0-D). These
//         tensors participate in index broadcasting.
//
//     2.  Boolean Tensor (dtype torch.bool):
//         -   The boolean tensor must have the same shape as the size of the
//             corresponding dimension in the 'self' tensor (i.e., D_i).
//         -   The boolean tensor can be multi-dimensional, and in this case
//             the shape of the boolean tensor must match the dimensions it
//             represents in the 'self' tensor. For example, if first boolean
//             index tensor is 1-dimensional, its shape should be (D_0), and if
//             second boolean index tensor is 2-dimensional, its shape should
//             match the shape (D_1, D_2).
//
//   const at::Tensor& values:
//     The tensor containing the values to be placed into 'self'. The shape of
//     'values' must be broadcastable to the shape of the view selected by
//     'indices' (explained below).
//
//   bool accumulate:
//     -   false (default): The values from the 'values' tensor REPLACE the
//         existing values in 'self' at the selected locations.
//     -   true: The values from the 'values' tensor are ADDED to the
//         existing values in 'self' at the selected locations.
//
//   bool unsafe:
//     -   unsafe mode is used by CUDA kernel to not assert on OOB indexing
//         assuming that the indexing is done correctly.
//         see: ATen/native/cuda/Indexing.cu,
//         method wrapIndexOnce. StableHLO/XLA clamps in case of OOB indexing,
//         hence unsafe mode is ignored.
//
// The function returns the updated 'self' tensor.
//
// Indexing and Broadcasting Mechanics:
//
// I.   Index Broadcast Shape (B):
//      All effective index tensors are broadcast against each other using
//      NumPy-style broadcasting rules. This results in a common shape 'B'.
//      -  If the shapes are not mutually broadcastable, an error is raised
//      -  Index tensors can index non-contiguously in 'self'. In that case,
//         they are made contiguous by permuting 'self'. With permuting, index
//         tensors become contiguous in 'self' and move to the beginning of the
//         'self' tensor.
//
// II.  Indexed Dimensions:
//      These are the dimensions in the 'self' tensor that the indices tensor
//      map to. For example, self tensor has shape (D0, D1, D2, D3), in indices
//      list, if first index tensor map to dimension 1 of self tensor,
//      second index tensor map to dimension 2 of self tensor, then
//      Indexed dims = {1, 2}.
//
// III. Unindexed Dimensions:
//      These are the dimensions in the 'self' tensor that are not indexed by
//      the indices. In above example, Unindexed dims = {0, 3}.
//
// IV.  Shape of the Indexed View (S_view):
//      The shape of the view into 'self' that is being written to is
//      constructed by concatenating:
//.     a) The sizes of unindexed dimensions of 'self' before indexing taken
//         in their original ascending order. This forms the slice shape before
//         indexing (S_before). This can be empty if indexing starts from
//         dimension 0 of 'self'.
//      a) The index broadcast shape 'B'.
//      b) The sizes of the unindexed dimensions of 'self' after indexing taken
//         in their original ascending order. This forms the slice shape after
//         indexing (S_after). This can be empty if indexing ends at the last
//         dimension of 'self'.
//
//      The shape S_view = (S_before, B, S_after)
//
//      Example1: self.shape=(D0, D1, D2, D3), Indexed dims={1, 2}. B=(B0, B1).
//               Unindexed dims={0, 3}. S_view = (D0, B0, B1, D3).
//      Example2: self.shape=(D0, D1, D2, D3), Indexed dims={1, 2}. B=(B0,).
//               Unindexed dims={0, 3}. S_view = (D0, B0, D3).
//
// V.   'values' Tensor Broadcasting:
//      The 'values' tensor must be broadcastable to the shape S_view, otherwise
//      an error is raised. So for S_view = (D0, B0, B1, D3),
//      'values' tensor can be of shape (D0, B0, B1, D3) or (D0, B0, B1, 1) or
//      (D0, B0, 1, 1) etc.
//
// VI.  Update Operation:
//      The update (overwrite or accumulate) happens element-wise between the
//      broadcasted 'values' tensor and the indexed view of 'self'.
//      -   If accumulate is true, the values are added to the existing values
//         in 'self' at the selected locations.
//      -   If accumulate is false, the values overwrite the existing values in
//         'self' at the selected locations.
//      - stablehlo.scatter is used to implement the update operation.
//
// The function returns the updated 'self' tensor.
//

at::Tensor& TpuAtenIndexPutImpl_(
    at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate, bool unsafe);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_INDEX_PUT_INDEX_PUT_ATEN_KERNELS_H_
