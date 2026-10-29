// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/ops/index_put/index_put_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/nonzero.h"
#include "ATen/ops/permute.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/index_put/index_put.h"
#include "csrc/ops/index_utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/nonzero/nonzero_aten_kernels.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

// Returns the positions of the non-zero elements of `mask` on `device`, with
// shape (L, mask.dim()). A CPU `mask` is evaluated on the host, which avoids a
// TPU nonzero and its device-to-host sync for the data-dependent output size.
// Calling `at::nonzero` on a CPU `mask` is fine here since it only runs the CPU
// kernel and never dispatches a TPU op.
absl::StatusOr<at::Tensor> GetNonzeroPositionsOnDevice(
    const at::Tensor& mask, const c10::Device& device) {
  if (mask.is_cpu()) {
    return MoveIndexToDevice(at::nonzero(mask), device);
  }
  return AtenNonzero(mask);
}

// Converts the parsed `entries` (see `ParseIndexEntries`) into purely integer
// index entries on the device of `self`.
//
// A boolean mask of rank `k` that covers dimensions {d, ..., d+k-1} of `self`
// is converted with `AtenNonzero()` (or `at::nonzero()` on the host for a CPU
// mask) into a tensor of shape (L, k), where L is the number of True elements.
// Its j-th column (0 <= j < k) becomes a 1D integer index entry of shape (L,)
// that indexes dimension d+j of `self`.
absl::StatusOr<std::vector<IndexEntry>> ConvertBooleanIndicesToPositional(
    const at::Tensor& self, absl::Span<const IndexEntry> entries) {
  std::vector<IndexEntry> positional_entries;
  for (const IndexEntry& entry : entries) {
    if (entry.is_mask) {
      TT_ASSIGN_OR_RETURN(
          at::Tensor nonzero_result,  // Shape (L, mask_ndim)
          GetNonzeroPositionsOnDevice(entry.tensor, self.device()));
      // Decompose the columns of the non-zero result, with each column becoming
      // a separate 1D integer tensor. Each dimension of the mask indexes a
      // dimension of `self` in the same order.
      for (int64_t i = 0; i < entry.tensor.dim(); ++i) {
        positional_entries.push_back(
            {.tensor = nonzero_result.select(/*dim=*/1, /*index=*/i),
             .first_dim = entry.first_dim + i,
             .is_mask = false});
      }
    } else {
      TT_ASSIGN_OR_RETURN(at::Tensor tpu_index,
                          MoveIndexToDevice(entry.tensor, self.device()));
      positional_entries.push_back({.tensor = std::move(tpu_index),
                                    .first_dim = entry.first_dim,
                                    .is_mask = false});
    }
  }
  return positional_entries;
}

// Returns whether `entries` index consecutive dimensions of `self`, in order.
bool AreIndicesContiguous(absl::Span<const IndexEntry> entries) {
  for (int64_t i = 1; i < entries.size(); ++i) {
    if (entries[i].first_dim != entries[i - 1].first_dim + 1) {
      return false;
    }
  }
  return true;
}

// Moves the dimensions of `self` indexed by the integer index `entries` to the
// front of the tensor, in the order they are indexed, and updates the
// `first_dim` of `entries` to match.
//
// For example, if 'self' is (4, 5, 6, 7) and the indexed dimensions are {1, 3},
// the result will be (5, 7, 4, 6), and the indexed dimensions become {0, 1}.
//
at::Tensor MoveIndicesToFront(const at::Tensor& self,
                              std::vector<IndexEntry>& entries) {
  absl::InlinedVector<bool, 6> is_dim_indexed(self.dim(), false);
  Indices permutation;
  permutation.reserve(self.dim());
  for (int64_t i = 0; i < entries.size(); ++i) {
    is_dim_indexed[entries[i].first_dim] = true;
    permutation.push_back(entries[i].first_dim);
    entries[i].first_dim = i;
  }
  for (int64_t dim = 0; dim < self.dim(); ++dim) {
    if (!is_dim_indexed[dim]) {
      permutation.push_back(dim);
    }
  }
  return at::permute(self, permutation);
}

// Computes the common broadcast shape of the tensors in the 'indices' list.
//
// All tensors within the input 'indices' list are broadcast against each other
// using NumPy-style rules. These index tensors must be mutually broadcastable;
// otherwise, an error is returned, mirroring PyTorch/NumPy behavior.
//
// The resulting shape defines the shape of the multi-dimensional grid of
// index combinations. Each element in this grid corresponds to a
// set of indices (one from each broadcasted index tensor)
// used to update the 'self' tensor.
//
// As an example:
//  indices is (idx0, idx1) (e.g., for dimensions 0 and 1 of self)
//    idx0.shape = (2, 1)
//    idx1.shape = (1, 5)
//
//  The result of GetIndexBroadcastShape is (2, 5).
//  This means there are 2 * 5 = 10 pairs of (dim0_index, dim1_index)
//  generated by broadcasting idx0 and idx1. For instance, the element at (i, j)
//  in this grid corresponds to the index pair (idx0[i, 0], idx1[0, j]).
//
absl::StatusOr<Dimensions> GetIndexBroadcastShape(
    absl::Span<const IndexEntry> entries) {
  Dimensions broadcast_shape;
  for (const IndexEntry& entry : entries) {
    const at::Tensor& index_tensor = entry.tensor;
    if (broadcast_shape.empty()) {
      broadcast_shape = CopyIntVector(index_tensor.sizes());
    } else {
      TT_ASSIGN_OR_RETURN(
          broadcast_shape, InferSize(broadcast_shape, index_tensor.sizes()),
          _.SetPrepend() << "expected index tensors to be broadcastable, got "
                         << "index tensor shape " << index_tensor.sizes()
                         << " and broadcast shape ["
                         << absl::StrJoin(broadcast_shape, ", ") << "]: ");
    }
  }
  return broadcast_shape;
}

// Computes the shape that the values tensor is broadcastable to.
//
// This shape is formed by concatenating:
// 1. The common broadcast shape of all index tensors (let this be 'B').
// 2. The sizes of the dimensions of the 'self' tensor that were NOT indexed
//    by the indices, taken in their original ascending order.
//
// The indexed dimensions of 'self' must all be contiguous.
//
// As an example:
//  self tensor has shape (10, 20, 30, 40) rank=4
//  indices is (idx0, idx1) for indexing dims 1 and 2 respectively
//    idx0 is shape (2, 1)
//    idx1 is shape (1, 5)
//  1. The 'indices' broadcast shape 'B' is (2, 5).
//  2. Unindexed dimensions of 'self' are {0, 3}.
//
//  The shape of the view being indexed is (10, 2, 5, 40).
//  The 'values' tensor must be broadcastable to this shape (10, 2, 5, 40).
//
//
absl::StatusOr<Dimensions> GetValuesBroadcastShape(
    const at::Tensor& self, const at::Tensor& values,
    const Dimensions& index_broadcast_shape, int64_t index_start_dim,
    int64_t index_end_dim) {
  ABSL_CHECK(  // CRASH_OK=start should be less than or equal to end
      index_start_dim >= 0 && index_start_dim <= index_end_dim)
      << "[IndexPut] index_start_dim: " << index_start_dim
      << " must be less than or equal to index_end_dim: " << index_end_dim;
  ABSL_CHECK(  // CRASH_OK=end should be less than rank of self
      index_end_dim >= 0 && index_end_dim < self.dim())
      << "[IndexPut] index_end_dim: " << index_end_dim
      << " must be less than the rank of self: " << self.dim();

  Dimensions values_broadcast_shape;
  // Sliced dimensions before indexed dimensions.
  for (int64_t dim = 0; dim < index_start_dim; ++dim) {
    values_broadcast_shape.push_back(self.sizes()[dim]);
  }
  // Index tensors broadcast shape.
  for (int64_t i = 0; i < index_broadcast_shape.size(); ++i) {
    values_broadcast_shape.push_back(index_broadcast_shape[i]);
  }
  // Sliced dimensions after indexed dimensions.
  for (int64_t dim = index_end_dim + 1; dim < self.dim(); ++dim) {
    values_broadcast_shape.push_back(self.sizes()[dim]);
  }

  ABSL_VLOG(1) << "[IndexPut] values broadcast shape: ("
               << absl::StrJoin(values_broadcast_shape, ", ") << ")";

  if (!at::is_expandable_to(values.sizes(), values_broadcast_shape)) {
    return TT_ERROR(error::kInvalidArgument) << absl::StrCat(
               "value tensor of shape [", absl::StrJoin(values.sizes(), ", "),
               "] cannot be broadcast to indexing result of shape [",
               absl::StrJoin(values_broadcast_shape, ", "), "]");
  }

  return values_broadcast_shape;
}

// Returns true if the values tensor can be broadcast into self for a boolean
// mask index_put lowering using stablehlo.select.
//
// In PyTorch semantics, self[..., mask, ...] = values replaces the indexed
// slice of shape (leading_dims, K, trailing_dims) with values, where K is the
// number of True elements in mask.
// For stablehlo.select to be mathematically equivalent, values must not vary
// across the masked dimensions (i.e. any dimension in values corresponding to
// a masked dimension in self must have size 1 or be omitted).
bool CanUseSelectWithValues(const at::Tensor& self, const at::Tensor& values,
                            const IndexEntry& mask) {
  Dimensions target_shape;
  for (int64_t i = 0; i < mask.first_dim; ++i) {
    target_shape.push_back(self.size(i));
  }
  target_shape.push_back(1);
  for (int64_t i = mask.first_dim + mask.tensor.dim(); i < self.dim(); ++i) {
    target_shape.push_back(self.size(i));
  }
  return at::is_expandable_to(values.sizes(), target_shape);
}

// When the indices contain a boolean mask and the values tensor does not vary
// across the masked dimensions, we use the more efficient stablehlo.select to
// implement the index_put operation without host CPU synchronization.
//
// self[mask] = broadcastable_value
//
// This function dispatches to the appropriate StableHLO implementation based
// on the presence of a boolean mask in the indices.
absl::Status IndexPutWithBooleanMask(const at::Tensor& self,
                                     const IndexEntry& mask_entry,
                                     const at::Tensor& values,
                                     const bool accumulate, at::Tensor& out) {
  const int64_t index_start_dim = mask_entry.first_dim;
  TT_ASSIGN_OR_RETURN(at::Tensor mask,
                      MoveIndexToDevice(mask_entry.tensor, self.device()));
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=the op creates cache keys
                        // successfully.
      auto param_keys,
      TT_MAKE_OP_PARAM_CACHE_KEYS(accumulate, index_start_dim));
  auto index_op_builder =
      [accumulate, index_start_dim](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
        auto& [self, mask, values] = inputs;
        return BuildIndexPutSelectShlo(self, mask, index_start_dim, values,
                                       accumulate);
      };

  TT_ASSIGN_OR_RETURN(const auto elem_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  return DispatchOpOut<3>(std::move(index_op_builder), {self, mask, values},
                          out,
                          {.out_dtype = elem_type,
                           .out_dims = at::IntArrayRef(self.sizes()),
                           .op_param_cache_keys = std::move(param_keys)});
}

absl::Status IndexPut(at::Tensor& self, absl::Span<const IndexEntry> entries,
                      const at::Tensor& values, const bool accumulate,
                      at::Tensor& out) {
  TT_ASSIGN_OR_RETURN(std::vector<IndexEntry> positional_entries,
                      ConvertBooleanIndicesToPositional(self, entries));
  ABSL_VLOG(1) << "[IndexPut] indexed_dims: "
               << absl::StrJoin(GetFirstDims(positional_entries), ", ");

  TT_RETURN_IF_ERROR(
      ResolveNegativeIndices(absl::MakeSpan(positional_entries), self.sizes()));

  // `ParseIndexEntries` guarantees at least one defined index, but 0-d masks
  // expand to no index tensors. This is reached when every defined index is a
  // 0-d mask and the `IndexPutWithBooleanMask` path was not taken.
  TT_RET_CHECK(!positional_entries.empty(), error::kInvalidArgument)
      << "indices must be specified";

  if (!AreIndicesContiguous(positional_entries)) {
    self = MoveIndicesToFront(self, positional_entries);
    out = self;
    ABSL_VLOG(1) << "[IndexPut] self after moving indices to front: "
                 << self.sizes();
  }
  const int64_t index_start_dim = positional_entries.front().first_dim;
  const int64_t index_end_dim = positional_entries.back().first_dim;

  TT_ASSIGN_OR_RETURN(Dimensions index_broadcast_shape,
                      GetIndexBroadcastShape(positional_entries));
  ABSL_VLOG(1) << "[IndexPut] indices broadcast shape: ("
               << absl::StrJoin(index_broadcast_shape, ", ")
               << "), index_start_dim: " << index_start_dim
               << ", index_end_dim: " << index_end_dim;

  TT_ASSIGN_OR_RETURN(
      Dimensions values_broadcast_shape,
      GetValuesBroadcastShape(self, values, index_broadcast_shape,
                              index_start_dim, index_end_dim));
  ABSL_VLOG(1) << "[IndexPut] values broadcast shape: ("
               << absl::StrJoin(values_broadcast_shape, ", ") << ")";

  std::vector<at::Tensor> all_tensors;
  all_tensors.reserve(positional_entries.size() + 2);
  all_tensors.push_back(self);
  for (IndexEntry& entry : positional_entries) {
    all_tensors.push_back(std::move(entry.tensor));
  }
  all_tensors.push_back(values);

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=the op creates cache keys
                        // successfully.
      auto param_keys,
      TT_MAKE_OP_PARAM_CACHE_KEYS(index_start_dim, index_end_dim, accumulate));

  auto index_op_builder =
      [index_start_dim = index_start_dim, index_end_dim = index_end_dim,
       index_broadcast_shape = std::move(index_broadcast_shape),
       values_broadcast_shape = std::move(values_broadcast_shape), accumulate](
          absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder) {
        ABSL_CHECK(inputs.size() >= 3)  // CRASH_OK
            << "[IndexPut] expects at least 3 inputs, got " << inputs.size();
        // First input is the self tensor, followed by the index tensors,
        // and then the values tensor.
        auto self_op = inputs[0];
        auto values_op = inputs[inputs.size() - 1];
        auto indices_op = inputs.subspan(1, inputs.size() - 2);
        return BuildIndexPutShlo(self_op, indices_op, index_start_dim,
                                 index_end_dim, index_broadcast_shape,
                                 values_op, values_broadcast_shape, accumulate);
      };

  TT_ASSIGN_OR_RETURN(const auto elem_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  return DispatchOpOut<kDynamicSize>(
      std::move(index_op_builder), all_tensors, out,
      {.out_dtype = elem_type,
       .out_dims = at::IntArrayRef(self.sizes()),
       .op_param_cache_keys = std::move(param_keys)});
}

absl::Status IndexPutHelper(
    at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices_list_opt,
    const at::Tensor& values, const bool accumulate, at::Tensor& out) {
  ABSL_VLOG(1) << "[IndexPut] self: " << self.sizes();
  ABSL_VLOG(1) << "[IndexPut] values: " << values.sizes();
  ABSL_VLOG(1) << "[IndexPut] indices_list_opt: " << indices_list_opt.size();

  // Like PyTorch's `index_put_`, this counts list entries (a 0-d mask counts as
  // one) in addition to the dimensions checked by `ParseIndexEntries` (a 0-d
  // mask consumes none). `index` intentionally doesn't have this check.
  TT_RET_CHECK(indices_list_opt.size() <= self.dim(), error::kIndexError)
      << "too many indices for tensor of dimension " << self.dim() << " (got "
      << indices_list_opt.size() << ")";

  TT_RET_CHECK(values.scalar_type() == self.scalar_type(),
               error::kInvalidArgument)
      << "expected dtypes of values and destination to be the same, got "
      << ToString(values.scalar_type()) << " and "
      << ToString(self.scalar_type());

  TT_ASSIGN_OR_RETURN(std::vector<IndexEntry> entries,
                      ParseIndexEntries(self, indices_list_opt));

  if (entries.size() == 1 && entries[0].is_mask &&
      CanUseSelectWithValues(self, values, entries[0])) {
    // For scenarios where the indices contain a boolean mask and values can be
    // broadcast to self without varying across the masked dimensions, we can
    // use the more efficient stablehlo.select to implement the index_put
    // operation without host synchronization.
    //
    // self[mask] = broadcastable_value
    //
    return IndexPutWithBooleanMask(self, entries[0], values, accumulate, out);
  }
  return IndexPut(self, entries, values, accumulate, out);
}

}  // namespace

at::Tensor& TpuAtenIndexPutImpl_(
    at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate, bool unsafe) {
  TT_KERNEL(
      OpName::kIndexPutImpl_, _,
      (self, IgnoreInCacheKey(indices, "Delegates to IndexPutHelper()"), values,
       IgnoreInCacheKey(accumulate, "Delegates to IndexPutHelper()"),
       IgnoreInCacheKey(unsafe, "Unused")),
      {
        TT_THROW_IF_ERROR(
            IndexPutHelper(self, indices, values, accumulate, self));
        return self;
      });
}

}  // namespace torch_tpu
