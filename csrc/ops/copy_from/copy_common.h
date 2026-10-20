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

#ifndef TORCH_TPU_CSRC_OPS_COPY_FROM_COPY_COMMON_H_
#define TORCH_TPU_CSRC_OPS_COPY_FROM_COPY_COMMON_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"

namespace torch_tpu {

// Optimizes a view operation by applying broadcasting and transposing.
// src and dest must be broadcastable, otherwise inferExpandGeometry will throw.
// After this function, src and dest will have the same shape, and
// dest will be row-major. Element types and storage offsets are unchanged.
// This may strip size-1 dimensions from src and dest.
absl::Status BroadcastAndTranspose(at::Tensor& src, at::Tensor& dest);

// The synchronization mode of a host-to-device or device-to-host copy.
enum class TransferSyncMode {
  // The copy operation is recorded on the current stream, and then the call
  // returns immediately. Errors will be reported in the future when the stream
  // is synchronized or queried.
  kEagerNonBlocking,
  // The copy operation is not recorded on the current stream. It is initiated
  // immediately (ignoring any cross-stream waits) and then the call blocks
  // until the copy completes without forcing any other device execution.
  kIsolated,
};

// Returns the synchronization mode of a host-to-device or device-to-host copy
// based on the user-requested non-blocking mode and the current eager mode.
//
// If the eager mode is kInternalCompileFxGraph or kDeferNeverAndLaunchBlocking,
// the transfer will be kIsolated, regardless of the value of `non_blocking`.
//
// Otherwise, non_blocking will determine whether kEagerNonBlocking or
// kIsolated is used.
TransferSyncMode GetTransferSyncMode(bool non_blocking);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_COPY_FROM_COPY_COMMON_H_
