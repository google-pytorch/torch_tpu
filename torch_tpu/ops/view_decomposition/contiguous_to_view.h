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

#ifndef TORCH_TPU_OPS_VIEW_DECOMPOSITION_CONTIGUOUS_TO_VIEW_H_
#define TORCH_TPU_OPS_VIEW_DECOMPOSITION_CONTIGUOUS_TO_VIEW_H_

#include <cstdint>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/eager/device_buffer.h"

namespace torch_tpu {

// Converts a contiguous tensor to a view tensor with equivalent logical values
// and specified target strides and storage offset.
//
// This is effectively the opposite of `torch.Tensor.contiguous()`.
// That function operates by taking a view tensor, which may be non-dense,
// broadcasted, or strided in a complex overlapping pattern, and copies it into
// a contiguous tensor.
//
// To invert this, we instead take a contiguous buffer, allocate a new storage,
// and copy non-overlapping windows of data from the contiguous tensor into
// the new storage, such that the desired view will read the correct values.
//
// Note that because `torch.Tensor.contiguous` may result in data duplication
// when copying out broadcasted or overlapping views, this inverse operation may
// deduplicate data. If the contiguous buffer does not actually have duplicate
// data in the expected locations, then this results in a "last write wins"
// behavior.
//
// Additionally note that because `torch.Tensor.contiguous` may remove data from
// the original storage not accessible to the view, this function may insert
// new dummy data of NaNs or max-value integers (following the policy of
// `torch.utils.deterministic.fill_uninitialized_memory`) in those gaps. This
// will not affect the logical values of the returned view tensor, but later
// uses of `torch.as_strided` may recover this "zombie data" if they create an
// out-of-bounds striding on the output view.
//
// In the case where the target strides are already contiguous, this is a no-op
// and just returns the contiguous buffer (wrapping it in an at::Tensor).
// In the case where the target strides are non-overlapping (permitting simple
// broadcasts), this will perform a single write operation (see inversion.cc).
absl::StatusOr<at::Tensor> ContiguousToView(
    DeviceBufferRef contiguous_buffer, absl::Span<const int64_t> target_strides,
    int64_t target_storage_offset);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_VIEW_DECOMPOSITION_CONTIGUOUS_TO_VIEW_H_
