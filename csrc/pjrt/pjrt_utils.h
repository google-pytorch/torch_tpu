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

#ifndef TORCH_TPU_CSRC_PJRT_PJRT_UTILS_H_
#define TORCH_TPU_CSRC_PJRT_PJRT_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// clang-format off
#include "ATen/core/TensorBody.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "csrc/common/compilation.h"
#include "csrc/eager/device_buffer.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/xla_data.pb.h"
// clang-format on

namespace torch_tpu {

// If backing tensor is present we keep it alive until the transfer completes,
// otherwise we block on the transfer to complete.
absl::StatusOr<DeviceBufferRef> TpuMallocAndMemcpyHtoD(
    const void* host_data, mlir::ElementType element_type,
    absl::Span<const int64_t> dimensions,
    std::optional<at::Tensor> backing_tensor = std::nullopt);

absl::StatusOr<at::Tensor> TpuMemcpyDtoH(const DeviceBufferRef& buffer_ref,
                                         bool non_blocking = false);

// Copies the data from the device directly into the provided host buffer.
absl::Status TpuMemcpyDtoHDirect(const DeviceBufferRef& buffer_ref,
                                 void* dst_ptr, bool non_blocking = false);

// The result of executing a PjRtExecutable, which may produce multiple non-null
// PjRtBuffers.
using PjRtBufferPointers =
    std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>>;

// Launches the executable. This is a synchronous call, and will block
// until the execution is fully enqueued (but not completed). Calls for D2H
// on the returned buffers will block until the execution and transfer from
// TPU to CPU are complete.
absl::StatusOr<PjRtBufferPointers> Execute(
    const SharedLoadedExecutableWithMetadata& executable,
    std::vector<xla::PjRtBuffer* absl_nullable> argument_buffers);

// Holds the completion future and RAII buffer hold for an asynchronous DMA
// copy.
struct AsyncDmaResult {
  xla::Future<> future;
  std::shared_ptr<void> buffer_hold;
};

// Performs asynchronous non-blocking Device-to-Host (D2H) DMA transfer
// from a TPU device buffer to a host memory pointer using the PjRt RawBuffer C
// API. Returns an AsyncDmaResult indicating copy completion along with an
// opaque hold object preserving device buffer lifetime until completion.
absl::StatusOr<AsyncDmaResult> TpuAsyncDmaCopyDtoH(
    const DeviceBufferRef& src_buffer, void* dst_host_ptr, int64_t copy_bytes);

// Performs asynchronous non-blocking Host-to-Device (H2D) DMA transfer
// from a host memory pointer into a TPU device buffer at a byte offset using
// the PjRt RawBuffer C API. Returns an AsyncDmaResult indicating copy
// completion along with an opaque hold object preserving device buffer lifetime
// until completion.
absl::StatusOr<AsyncDmaResult> TpuAsyncDmaCopyHtoD(
    const void* src_host_ptr, const DeviceBufferRef& dst_buffer,
    int64_t dst_byte_offset, int64_t copy_bytes);

std::string ToString(const xla::PjRtBuffer& buffer);

// Returns true if the active TPU device has SparseCore hardware on chip and
// supports SparseCore offloading. Returns false on architectures without
// SparseCore (e.g., TPU v2, v3, v4, v5e).
bool TpuDeviceSupportsSparseCore();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_PJRT_PJRT_UTILS_H_
