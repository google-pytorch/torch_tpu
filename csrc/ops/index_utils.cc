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

#include "csrc/ops/index_utils.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/List.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/Storage.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/to_copy/to_copy_aten_kernels.h"
#include "csrc/ops/where/where.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

bool HasContiguousUnsqueezedStrides(
    const at::Tensor& index, absl::Span<const int64_t> unsqueezed_sizes) {
  const int64_t rank = unsqueezed_sizes.size();
  Dimensions unsqueezed_strides(rank, 1);
  for (int64_t d = rank - 2; d >= 0; --d) {
    unsqueezed_strides[d] = unsqueezed_strides[d + 1] * unsqueezed_sizes[d + 1];
  }
  for (int64_t d = 0; d < rank; ++d) {
    if (unsqueezed_sizes[d] > 1 && index.stride(d) != unsqueezed_strides[d]) {
      return false;
    }
  }
  return true;
}

// Checks that every defined tensor in `indices` has dtype bool, int32 or int64,
// and is either on CPU or on the same device as `self`.
absl::Status ValidateIndexDtypesAndDevices(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices) {
  for (size_t i = 0; i < indices.size(); ++i) {
    const std::optional<at::Tensor> index = indices[i];
    if (!index.has_value() || !index->defined()) {
      continue;
    }
    const c10::ScalarType dtype = index->scalar_type();
    // Unlike PyTorch, uint8 index tensors are rejected rather than treated as
    // (deprecated) boolean masks, matching masked_fill, masked_select and
    // masked_scatter.
    TT_RET_CHECK(dtype == c10::ScalarType::Bool ||
                     dtype == c10::ScalarType::Int ||
                     dtype == c10::ScalarType::Long,
                 error::kInvalidArgument)
        << "expected index tensors to have dtype bool, int32, or int64, got "
        << ToString(dtype) << " at index " << i
        << (dtype == c10::ScalarType::Byte
                ? "; indexing with uint8 tensors is deprecated in PyTorch and "
                  "not supported in TorchTPU, use bool tensors instead"
                : "");
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Requires indices on a second TPU
                   // device, but the error test environment has a single TPU.
        index->is_cpu() || index->device() == self.device(),
        error::kPythonRuntimeError)
        << "expected indices to be either on cpu or on the same device as the "
           "indexed tensor ("
        << self.device().str() << "), got " << index->device().str()
        << " at index " << i;
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::vector<IndexEntry>> ParseIndexEntries(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices) {
  TT_RETURN_IF_ERROR(ValidateIndexDtypesAndDevices(self, indices));

  std::vector<IndexEntry> parsed;
  // The first dimension of `self` indexed by the current entry.
  int64_t dim = 0;
  for (std::optional<at::Tensor> index : indices) {
    const bool is_slice = !index.has_value() || !index->defined();
    if (is_slice || index->scalar_type() != c10::ScalarType::Bool) {
      // A `:` slice or an integer index consumes one dimension of `self`.
      TT_RET_CHECK(dim < self.dim(), error::kIndexError)
          << "too many indices for tensor of dimension " << self.dim();
      if (!is_slice) {
        parsed.push_back(
            {.tensor = *std::move(index), .first_dim = dim, .is_mask = false});
      }
      ++dim;
      continue;
    }

    // A bool mask consumes `index->dim()` dimensions of `self` (a 0-d mask
    // consumes none and inserts a new dimension at `dim`), whose sizes must
    // match the shape of the mask.
    for (int64_t mask_dim = 0; mask_dim < index->dim(); ++mask_dim) {
      const int64_t self_dim = dim + mask_dim;
      TT_RET_CHECK(
          self_dim < self.dim() && index->size(mask_dim) == self.size(self_dim),
          error::kIndexError)
          << "expected the shape of the mask at index " << mask_dim
          << " to match the shape of the indexed tensor at index " << self_dim
          << ", got mask shape " << index->sizes()
          << " and indexed tensor shape " << self.sizes();
    }
    const int64_t mask_dims = index->dim();
    parsed.push_back(
        {.tensor = *std::move(index), .first_dim = dim, .is_mask = true});
    dim += mask_dims;
  }

  TT_RET_CHECK(!parsed.empty(), error::kInvalidArgument)
      << "expected at least one index tensor to be defined, got 0";
  return parsed;
}

// PyTorch's advanced indexing rules allow index tensors to be on CPU while the
// indexed tensor is on a device (`dev_tensor[{cpu,dev}_tensor]`), so CPU index
// tensors are copied to the device of the indexed tensor here. We use
// CopyToTpuTensor instead of ATen's .to() to transfer CPU tensors to TPU
// without triggering dispatcher re-entry or CompositeOpCheck failures during
// tracing.
absl::StatusOr<at::Tensor> MoveIndexToDevice(const at::Tensor& index,
                                             const c10::Device& device) {
  if (!index.defined() || index.device() == device) {
    return index;
  }
  return CopyToTpuTensor(index, /*dtype=*/std::nullopt,
                         /*non_blocking=*/false,
                         /*memory_format=*/c10::MemoryFormat::Contiguous,
                         /*tpu_device_index=*/device.index());
}

Indices GetFirstDims(absl::Span<const IndexEntry> entries) {
  Indices dims;
  dims.reserve(entries.size());
  for (const IndexEntry& entry : entries) {
    dims.push_back(entry.first_dim);
  }
  return dims;
}

absl::Status ResolveNegativeIndices(absl::Span<IndexEntry> entries,
                                    const at::IntArrayRef sizes) {
  std::vector<at::Tensor> indices_to_resolve;
  std::vector<at::Tensor> dim_size_tensors;
  Indices original_indices;

  for (size_t i = 0; i < entries.size(); ++i) {
    const at::Tensor& index = entries[i].tensor;
    if (index.scalar_type() == c10::ScalarType::Long ||
        index.scalar_type() == c10::ScalarType::Int) {
      indices_to_resolve.push_back(index);

      const int64_t dim_size = sizes[entries[i].first_dim];
      auto promoted_dim_size = PromoteScalar(at::Scalar(dim_size));
      TT_ASSIGN_OR_RETURN(at::Tensor dim_size_tensor,
                          promoted_dim_size.GetTensor(index.scalar_type()));
      dim_size_tensors.push_back(dim_size_tensor);

      original_indices.push_back(i);
    }
  }

  if (indices_to_resolve.empty()) {
    return absl::OkStatus();
  }

  size_t num_resolutions = indices_to_resolve.size();
  std::vector<at::Tensor> inputs;
  inputs.reserve(num_resolutions * 2);
  inputs.insert(inputs.end(), indices_to_resolve.begin(),
                indices_to_resolve.end());
  inputs.insert(inputs.end(), dim_size_tensors.begin(), dim_size_tensors.end());

  std::vector<mlir::ElementType> out_dtypes;
  std::vector<Dimensions> out_dims_storage;
  std::vector<absl::Span<const int64_t>> out_dims_list;

  out_dtypes.reserve(num_resolutions);
  out_dims_storage.reserve(num_resolutions);
  out_dims_list.reserve(num_resolutions);

  for (const auto& index : indices_to_resolve) {
    TT_ASSIGN_OR_RETURN(auto dtype,
                        ConvertTo<mlir::ElementType>(index.scalar_type()));
    out_dtypes.push_back(dtype);

    out_dims_storage.push_back(
        Dimensions(index.sizes().begin(), index.sizes().end()));
  }

  for (const auto& dims : out_dims_storage) {
    out_dims_list.push_back(dims);
  }

  auto op_builder =
      [num_resolutions](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<DynamicMlirOpResults> {
    DynamicMlirOpResults results;
    results.reserve(num_resolutions);

    for (size_t i = 0; i < num_resolutions; ++i) {
      mlir::MlirOp index_op = inputs[i];
      mlir::MlirOp dim_size_op = inputs[i + num_resolutions];

      auto index_type = GetTensorTypeOrDie(index_op);
      auto element_type = index_type.getElementType();
      auto zero = MakeScalarConstant(builder, 0, element_type);

      TT_ASSIGN_OR_RETURN(auto lt_cond, BuildLtShlo(index_op, zero));
      TT_ASSIGN_OR_RETURN(auto added_index,
                          BuildAddShlo(index_op, dim_size_op));
      TT_ASSIGN_OR_RETURN(auto output_element_type, GetElementType(index_op));
      // NOTE: This logic resolves negative indices but doesn't check for
      // out-of-bounds (OOB) results. For example, if `index` is `-6` and
      // `dim_size` is `5`, the result is `-1`. PyTorch typically throws an
      // `IndexError` in this case.
      TT_ASSIGN_OR_RETURN(
          auto resolved_index,
          BuildWhereShlo(lt_cond, added_index, index_op, output_element_type));
      results.push_back(resolved_index);
    }
    return results;
  };

  DispatchOpOptions<kDynamicSize> options = {
      .op_name = OpName::kResolveNegativeIndices,
      .out_dtypes = out_dtypes,
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
  };

  TT_ASSIGN_OR_RETURN(std::vector<DeviceBufferRef> result_bufs,
                      (DispatchOp<kDynamicSize, kDynamicSize>(
                          std::move(op_builder), inputs, std::move(options))));

  for (size_t i = 0; i < num_resolutions; ++i) {
    size_t original_idx = original_indices[i];
    entries[original_idx].tensor = MakeTensor(std::move(result_bufs[i]));
  }

  return absl::OkStatus();
}

absl::StatusOr<int64_t> ValidateIndexInputsAndGetDim(const at::Tensor& self,
                                                     int64_t dim,
                                                     const at::Tensor& index,
                                                     const at::Tensor& source) {
  TT_RET_CHECK(index.dim() == 1, error::kInvalidArgument)
      << "expected index to be 1D, got shape " << index.sizes();
  if (self.dim() == 0) {
    TT_RET_CHECK(dim == 0, error::kInvalidArgument)
        << "expected dim to be 0 for scalar input, got " << dim;
    TT_RET_CHECK(source.dim() == 0, error::kInvalidArgument)
        << "expected source shape to match self shape, excluding the specified "
           "dimension, got source shape "
        << source.sizes() << " and self shape " << self.sizes();
    TT_RET_CHECK(index.size(0) == 1, error::kInvalidArgument)
        << "expected index to be 1D of size 1 for scalar input, got shape "
        << index.sizes();
    return dim;
  }

  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self.dim()));

  TT_RET_CHECK(source.dim() == self.dim(), error::kPythonIndexError)
      << "expected self and source to have the same number of dimensions, got "
      << self.dim() << " and " << source.dim();
  TT_RET_CHECK(source.size(dim) == index.size(0), error::kInvalidArgument)
      << "expected source to have the same number of elements as the index "
         "along "
         "dimension "
      << dim << ", got " << source.size(dim) << " and " << index.size(0);
  for (int i = 0; i < self.dim(); ++i) {
    if (i != dim) {
      TT_RET_CHECK(self.size(i) == source.size(i), error::kInvalidArgument)
          << "expected self and source to have the same size along dimension "
          << i << ", got " << self.size(i) << " and " << source.size(i);
    }
  }
  return dim;
}

std::optional<UnexpandedZeroStrideIndex> TryUnexpandZeroStrideIndex(
    const at::Tensor& index) {
  const int64_t rank = index.dim();
  if (rank <= 0 || index.numel() == 0 || index.is_contiguous() ||
      index.storage_offset() != 0 || index.is_conj()) {
    return std::nullopt;
  }
  Dimensions unsqueezed_sizes(rank);
  Dimensions squeezed_sizes;
  Dimensions bcast_dims;
  squeezed_sizes.reserve(rank);
  bcast_dims.reserve(rank);
  bool has_expanded_dim = false;
  for (int64_t d = 0; d < rank; ++d) {
    const int64_t sz = index.size(d);
    if (sz > 1 && index.stride(d) == 0) {
      unsqueezed_sizes[d] = 1;
      has_expanded_dim = true;
    } else {
      unsqueezed_sizes[d] = sz;
      squeezed_sizes.push_back(sz);
      bcast_dims.push_back(d);
    }
  }
  if (!has_expanded_dim || squeezed_sizes.empty() ||
      !HasContiguousUnsqueezedStrides(index, unsqueezed_sizes)) {
    return std::nullopt;
  }

  const auto base_buf = GetBaseBuffer(index);
  if (!base_buf.ok()) {
    return std::nullopt;
  }
  const Dimensions* target_sizes = nullptr;
  if (base_buf->dimensions() == squeezed_sizes) {
    target_sizes = &squeezed_sizes;
  } else if (base_buf->dimensions() == unsqueezed_sizes) {
    target_sizes = &unsqueezed_sizes;
  } else {
    return std::nullopt;
  }
  at::Tensor view(c10::make_intrusive<c10::TensorImpl>(
      c10::Storage(index.storage()),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), index.dtype()));
  view.unsafeGetTensorImpl()->set_sizes_contiguous(*target_sizes);
  view.unsafeGetTensorImpl()->set_storage_offset(0);
  return UnexpandedZeroStrideIndex{
      .tensor = std::move(view),
      .squeezed_dims = std::move(squeezed_sizes),
      .bcast_dims = std::move(bcast_dims),
      .logical_dims = CopyIntVector(index.sizes()),
  };
}

}  // namespace torch_tpu
