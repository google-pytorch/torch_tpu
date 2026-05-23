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

#include "torch_tpu/ops/index_put/index_put_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/permute.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/index_put/index_put.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nonzero/nonzero_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

absl::Status ValidateIndicesType(
    const c10::List<std::optional<at::Tensor>>& indices_list_opt) {
  for (int64_t i = 0; i < indices_list_opt.size(); ++i) {
    auto maybe_tensor = indices_list_opt[i];
    if (!maybe_tensor.has_value() || !maybe_tensor.value().defined()) {
      // tensor is not defined when it is None. This happens in case of
      // slicing for example: self[indices, :, indices] = bar. In this case
      // the second index tensor will be undefined.
      continue;
    }
    const auto& index_tensor = maybe_tensor.value();

    TT_RET_CHECK(index_tensor.scalar_type() == c10::ScalarType::Int ||
                     index_tensor.scalar_type() == c10::ScalarType::Long ||
                     index_tensor.scalar_type() == c10::ScalarType::Byte ||
                     index_tensor.scalar_type() == c10::ScalarType::Bool,
                 error::kInvalidArgument)
        << "tensors used as indices must be "
        << "long, int, byte or bool tensors, got "
        << ToString(index_tensor.scalar_type()) << " at index " << i;
  }
  return absl::OkStatus();
}

// Preprocesses indices tensor list by converting boolean tensor masks into
// 1D integer positional index tensors.
//
// This function takes a list of indices tensors. It transforms this list into
// a vector of purely integer tensors. Boolean tensors are handled by:
// 1. Validating their shape against the corresponding dimensions of 'self'.
// 2. Computing the non-zero indices.
// 3. Decomposing the columns of the non-zero result, with each column
//    becoming a separate 1D integer tensor in the output list.
//
// The function also populates an 'indexed_dims' vector, which maps each
// tensor in the returned list to the specific dimension of the 'self' tensor
// it indexes.
//
// Boolean tensors:
//   - A boolean tensor at `indices_list_opt[i]` with rank `k`
//     will index dimensions {i, i+1, ..., i+k-1} of 'self' if the last index
//     tensor indexed up to dimension i-1 of 'self'.
//   - Shape Validation: The shape of the boolean tensor must exactly match the
//     sizes of the corresponding dimensions {i, ..., i+k-1} in 'self'.
//     E.g., index_tensor.shape == (self.size(i), ..., self.size(i+k-1)).
//   - NonZero Conversion: `AtenNonzero()` is called, yielding a
//     tensor of shape (L, k), where L is the number of True elements.
//   - Decomposition: This (L, k) tensor is split into k column vectors. The
//   j-th
//     column (0 <= j < k) becomes a 1D integer tensor of shape (L,) and is
//     added to the result list. This j-th column tensor provides the indices
//     for dimension i+j of 'self'.
//
absl::StatusOr<std::vector<at::Tensor>> ConvertBooleanIndicesToPositional(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices_list_opt,
    Indices& indexed_dims) {
  std::vector<at::Tensor> index_tensors;
  int64_t dim_indexed = 0;
  for (int64_t i = 0; i < indices_list_opt.size(); ++i) {
    auto maybe_tensor = indices_list_opt[i];
    if (!maybe_tensor.has_value() || !maybe_tensor.value().defined()) {
      // this dimension of self tensor sliced out, so we
      // move to the next dimension.
      dim_indexed++;
      continue;
    }
    const auto& index_tensor = maybe_tensor.value();

    if (index_tensor.scalar_type() == c10::ScalarType::Bool) {
      int64_t self_dim = dim_indexed;
      for (int64_t mask_dim = 0; mask_dim < index_tensor.dim(); ++mask_dim) {
        TT_RET_CHECK(self_dim < self.dim() && index_tensor.sizes()[mask_dim] ==
                                                  self.sizes()[self_dim],
                     error::kInvalidArgument)
            << "the shape of the mask at index " << mask_dim
            << " must match the shape of the indexed tensor at index "
            << self_dim << ", got mask shape " << index_tensor.sizes()
            << " and indexed tensor shape " << self.sizes();
        self_dim++;
      }

      at::Tensor nonzero_result =
          AtenNonzero(index_tensor);  // Shape (L, mask_ndim)
      // Decompose the columns of the non-zero result, with each column becoming
      // a separate 1D integer tensor.
      for (int64_t i = 0; i < index_tensor.dim(); ++i) {
        index_tensors.push_back(nonzero_result.select(/*dim=*/1, /*index=*/i));

        // each dimension of mask tensor indexes a dimension of the
        // self tensor in the same order.
        indexed_dims.push_back(dim_indexed++);
      }
    } else {
      index_tensors.push_back(index_tensor);
      indexed_dims.push_back(dim_indexed++);
    }
  }
  ABSL_VLOG(1) << "[IndexPut] indexed_dims: "
               << absl::StrJoin(indexed_dims, ", ");
  return index_tensors;
}

bool AreIndicesContiguous(const Indices& indexed_dims) {
  if (indexed_dims.empty()) {
    return true;
  }
  for (int64_t i = 1; i < indexed_dims.size(); ++i) {
    if (indexed_dims[i] != indexed_dims[i - 1] + 1) {
      return false;
    }
  }
  return true;
}

// Moves the indexed dimensions of 'self' to the front of the tensor.
//
// This function takes a list of index tensors and a list of dimensions that
// were indexed. It returns a new tensor where the indexed dimensions have been
// moved to the front of the tensor, in the order they were indexed.
//
// For example, if 'self' is (4, 5, 6, 7) and the indexed dimensions are {1, 3},
// the result will be (5, 7, 4, 6).
//
at::Tensor MoveIndicesToFront(const at::Tensor& self,
                              std::vector<at::Tensor>& index_tensors,
                              const Indices& indexed_dims) {
  absl::InlinedVector<bool, 6> is_dim_indexed(self.dim(), false);
  for (int64_t d : indexed_dims) {
    is_dim_indexed[d] = true;
  }

  Indices permutation;
  permutation.reserve(self.dim());
  for (int64_t i = 0; i < indexed_dims.size(); ++i) {
    permutation.push_back(indexed_dims[i]);
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
    const std::vector<at::Tensor>& index_tensors) {
  Dimensions broadcast_shape;
  for (int64_t i = 0; i < index_tensors.size(); ++i) {
    const auto& index_tensor = index_tensors[i];
    if (broadcast_shape.empty()) {
      broadcast_shape = CopyIntVector(index_tensor.sizes());
    } else {
      TT_ASSIGN_OR_RETURN(
          broadcast_shape, InferSize(broadcast_shape, index_tensor.sizes()),
          _.SetPrepend() << "index tensors not broadcastable, got "
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

//
// Returns true if the indices list contains a single boolean index tensor.
//
bool CanUseIndicesAsMask(
    const c10::List<std::optional<at::Tensor>>& indices_list_opt) {
  bool found_mask = false;
  for (int64_t i = 0; i < indices_list_opt.size(); ++i) {
    auto maybe_tensor = indices_list_opt[i];
    if (!maybe_tensor.has_value() || !maybe_tensor.value().defined()) {
      continue;
    }
    const auto& index_tensor = maybe_tensor.value();
    if (index_tensor.scalar_type() == c10::ScalarType::Bool) {
      if (found_mask) {
        // Only one boolean mask is allowed.
        return false;
      }
      found_mask = true;
    } else {
      return false;
    }
  }
  return true;
}

inline bool IsValuesScalar(const at::Tensor& values) {
  return values.dim() == 0 && values.numel() == 1;
}

// When the values tensor is a scalar and the indices contain a boolean mask, we
// can use the more efficient stablehlo.select to implement the index_put
// operation.
//
// self[mask] = scalar_value
//
// This function dispatches to the appropriate StableHLO implementation based
// on the presence of a boolean mask in the indices.
absl::StatusOr<DeviceBufferRef> IndexPutWithBooleanMask(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices_list_opt,
    const at::Tensor& values, const bool accumulate) {
  int64_t mask_index = 0;
  for (; mask_index < indices_list_opt.size(); ++mask_index) {
    auto maybe_tensor = indices_list_opt[mask_index];
    if (!maybe_tensor.has_value() || !maybe_tensor.value().defined()) {
      continue;
    }
    break;
  }
  auto maybe_tensor = indices_list_opt[mask_index];
  int64_t self_dim = mask_index;
  const auto& mask = maybe_tensor.value();
  for (int64_t mask_dim = 0; mask_dim < mask.dim(); ++mask_dim) {
    TT_RET_CHECK(self_dim < self.dim() &&
                     mask.sizes()[mask_dim] == self.sizes()[self_dim],
                 error::kInvalidArgument)
        << "the shape of the mask at index " << mask_dim
        << " must match the shape of the indexed tensor at index " << self_dim
        << ", got mask shape " << mask.sizes() << " and indexed tensor shape "
        << self.sizes();
    self_dim++;
  }
  // mask tensor indexes the self tensor at mask_index
  const int64_t index_start_dim = mask_index;
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
  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<3>(std::move(index_op_builder), {self, mask, values},
                    {.out_dtype = elem_type,
                     .out_dims = at::IntArrayRef(self.sizes()),
                     .op_param_cache_keys = std::move(param_keys)}));

  return std::move(result_buf);
}

absl::StatusOr<DeviceBufferRef> IndexPut(
    at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices_list_opt,
    const at::Tensor& values, const bool accumulate) {
  Indices indexed_dims;
  TT_ASSIGN_OR_RETURN(
      std::vector<at::Tensor> index_tensors,
      ConvertBooleanIndicesToPositional(self, indices_list_opt, indexed_dims));

  TT_RET_CHECK(!indexed_dims.empty(), error::kInvalidArgument)
      << "indices must be specified";

  TT_RET_CHECK(index_tensors.size() <= self.dim(), error::kInvalidArgument)
      << "too many indices for tensor of dimension " << self.dim() << ", got "
      << index_tensors.size()
      << " index tensors after expanding boolean indices";

  int64_t index_start_dim = indexed_dims.front();
  int64_t index_end_dim = indexed_dims.back();
  if (!AreIndicesContiguous(indexed_dims)) {
    self = MoveIndicesToFront(self, index_tensors, indexed_dims);
    ABSL_VLOG(1) << "[IndexPut] self after moving indices to front: "
                 << self.sizes();
    index_start_dim = 0;
    index_end_dim = index_tensors.size() - 1;
  }

  TT_ASSIGN_OR_RETURN(Dimensions index_broadcast_shape,
                      GetIndexBroadcastShape(index_tensors));
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
  all_tensors.push_back(self);
  for (int64_t i = 0; i < index_tensors.size(); ++i) {
    const auto& tensor = index_tensors[i];
    all_tensors.push_back(tensor);
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
  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<kDynamicSize>(std::move(index_op_builder), all_tensors,
                               {.out_dtype = elem_type,
                                .out_dims = at::IntArrayRef(self.sizes()),
                                .op_param_cache_keys = std::move(param_keys)}));
  return std::move(result_buf);
}

absl::StatusOr<DeviceBufferRef> IndexPutHelper(
    at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices_list_opt,
    const at::Tensor& values, const bool accumulate, const bool unsafe) {
  ABSL_VLOG(1) << "[IndexPut] self: " << self.sizes();
  ABSL_VLOG(1) << "[IndexPut] values: " << values.sizes();
  ABSL_VLOG(1) << "[IndexPut] indices_list_opt: " << indices_list_opt.size();

  TT_RET_CHECK(indices_list_opt.size() <= self.dim(), error::kInvalidArgument)
      << "too many indices for tensor of dimension " << self.dim() << ", got "
      << indices_list_opt.size();

  TT_RET_CHECK(values.scalar_type() == self.scalar_type(),
               error::kInvalidArgument)
      << "dtypes of values and destination must be the same,"
      << " got " << ToString(values.scalar_type()) << " and "
      << ToString(self.scalar_type());

  TT_RETURN_IF_ERROR(ValidateIndicesType(indices_list_opt));

  if (IsValuesScalar(values) && CanUseIndicesAsMask(indices_list_opt)) {
    // For scenarios where the values tensor is a scalar and the indices
    // contain a boolean mask, we can use the more efficient stablehlo.select
    // to implement the index_put operation.
    //
    // self[mask] = scalar_value
    //
    return IndexPutWithBooleanMask(self, indices_list_opt, values, accumulate);
  }
  return IndexPut(self, indices_list_opt, values, accumulate);
}

}  // namespace

at::Tensor& TpuAtenIndexPutImpl_(
    at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate, bool unsafe) {
  TT_KERNEL(
      OpName::kIndexPutImpl_, _,
      (self, IgnoreInCacheKey(indices, "Legacy usage"), values,
       IgnoreInCacheKey(accumulate, "Legacy usage"),
       IgnoreInCacheKey(unsafe, "Legacy usage")),
      {
        TT_ASSIGN_OR_THROW(
            DeviceBufferRef result_buf,
            IndexPutHelper(self, indices, values, accumulate, unsafe));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), self));
        return self;
      });
}

}  // namespace torch_tpu
