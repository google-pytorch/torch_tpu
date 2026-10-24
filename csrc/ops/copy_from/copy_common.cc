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

#include "csrc/ops/copy_from/copy_common.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/ops/as_strided/as_strided_aten_kernels.h"

namespace torch_tpu {
namespace {

absl::Status RestrideTensor(at::Tensor& tensor, at::IntArrayRef sizes,
                            at::IntArrayRef strides, int64_t storage_offset) {
  if (tensor.device().is_cpu()) {
    tensor = tensor.as_strided(sizes, strides, storage_offset);
    return absl::OkStatus();
  }
  TT_ASSIGN_OR_RETURN(tensor,
                      AsStrided(tensor, sizes, strides, storage_offset));
  return absl::OkStatus();
}

// Using metadata-only views (as_strided), broadcasts src to the shape of dest.
// Returns an error if this is not possible.
absl::Status BroadcastToDest(at::Tensor& src, const at::Tensor& dest) {
  if (src.sizes() == dest.sizes()) {
    // No broadcasting is needed.
    return absl::OkStatus();
  }
  // Broadcast src to match the sizes of dest.
  auto [new_src_sizes, new_src_strides] =
      at::inferExpandGeometry(src.sizes(), src.strides(), dest.sizes());
  return RestrideTensor(src, new_src_sizes, new_src_strides,
                        src.storage_offset());
}

// Using metadata-only views (as_strided), transposes dest so that it is
// row-major, and applies the same transpose to src.
// If any transformation is required, this also strips size-1 dimensions from
// dest.
absl::Status TransposeToRowMajorDest(at::Tensor& src, at::Tensor& dest) {
  ABSL_CHECK_EQ(  // CRASH_OK=precondition established by BroadcastToDest
      src.dim(), dest.dim())
      << "cannot create a matching transpose for tensors of different "
         "dimensions";
  if (dest.is_contiguous()) {
    return absl::OkStatus();
  }

  // Determine a transpose that can be applied to both src and dest to make
  // dest contiguous (even if it makes src non-contiguous).
  struct TransposeDim {
    int64_t size;
    int64_t src_stride;
    int64_t dest_stride;
  };

  std::vector<TransposeDim> transpose_dims;
  transpose_dims.reserve(dest.dim());
  for (int64_t i = 0; i < dest.dim(); ++i) {
    if (dest.size(i) == 1) {
      continue;
    }
    transpose_dims.push_back({dest.size(i), src.stride(i), dest.stride(i)});
  }
  // Sort transpose_dims by dest.stride descending.
  std::sort(transpose_dims.begin(), transpose_dims.end(),
            [](const TransposeDim& a, const TransposeDim& b) {
              return a.dest_stride > b.dest_stride;
            });
  Dimensions new_sizes;
  Strides new_src_strides;
  Strides new_dest_strides;
  new_sizes.reserve(dest.dim());
  new_src_strides.reserve(dest.dim());
  new_dest_strides.reserve(dest.dim());
  for (const auto& dim : transpose_dims) {
    new_sizes.push_back(dim.size);
    new_src_strides.push_back(dim.src_stride);
    new_dest_strides.push_back(dim.dest_stride);
  }
  TT_RETURN_IF_ERROR(
      RestrideTensor(src, new_sizes, new_src_strides, src.storage_offset()));
  TT_RETURN_IF_ERROR(
      RestrideTensor(dest, new_sizes, new_dest_strides, dest.storage_offset()));
  return absl::OkStatus();
}
}  // namespace

absl::Status BroadcastAndTranspose(at::Tensor& src, at::Tensor& dest) {
  TT_RETURN_IF_ERROR(BroadcastToDest(src, dest));
  TT_RETURN_IF_ERROR(TransposeToRowMajorDest(src, dest));
  return absl::OkStatus();
}

TransferSyncMode GetTransferSyncMode(bool non_blocking) {
  const EagerMode eager_mode = GetEagerMode();
  if (eager_mode == EagerMode::kInternalCompileFxGraph) {
    return TransferSyncMode::kIsolated;
  }
  if (eager_mode == EagerMode::kDeferNeverAndLaunchBlocking) {
    return TransferSyncMode::kEagerBlocking;
  }
  return non_blocking ? TransferSyncMode::kEagerNonBlocking
                      : TransferSyncMode::kEagerBlocking;
}

}  // namespace torch_tpu
