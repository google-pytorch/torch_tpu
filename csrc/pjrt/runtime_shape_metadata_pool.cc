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

#include "csrc/pjrt/runtime_shape_metadata_pool.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

absl::StatusOr<std::unique_ptr<RuntimeShapeMetadataPool::Entry>>
RuntimeShapeMetadataPool::CreateEntry(const xla::Shape& shape,
                                      const xla::PjRtDevice& device) {
  TT_ASSIGN_OR_RETURN(
      xla::PjRtMemorySpace* const memory_space,
      WrapXlaError(device.memory_space_by_kind("pinned_host"),
                   /*context=*/"failed to get the pinned host memory space"));
  auto entry = std::make_unique<Entry>(Entry{.shape = shape});
  TT_ASSIGN_OR_RETURN(
      entry->buffer,
      WrapXlaError(
          device.client()->CreateUninitializedBuffer(shape, memory_space),
          /*context=*/"failed to create a runtime shape metadata buffer"));
  TT_ASSIGN_OR_RETURN(
      entry->host_memory,
      WrapXlaError(entry->buffer->AcquireExternalReference(),
                   /*context=*/"failed to get the host memory of a runtime "
                               "shape metadata buffer"));
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PJRT exposes pinned host memory.
      entry->host_memory->OpaqueDeviceMemoryDataPointer() != nullptr,
      error::kInternal)
      << "PjRt client did not return host memory for a pinned host buffer";
  return entry;
}

absl::StatusOr<absl_nonnull std::shared_ptr<xla::PjRtBuffer>>
RuntimeShapeMetadataPool::Acquire(const absl::Span<const int32_t> dimensions,
                                  const xla::Shape& shape,
                                  const xla::PjRtDevice& device) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by
                 // runtime_shape_metadata_pool_test.
      shape.IsArray() && shape.element_type() == xla::S32 &&
          shape.dimensions().size() == 1 &&
          shape.dimensions(0) == static_cast<int64_t>(dimensions.size()),
      error::kInvalidArgument)
      << "expected S32[" << dimensions.size() << "] metadata shape, got "
      << shape.ToString();

  std::unique_ptr<Entry> entry;
  {
    absl::MutexLock lock(mu_);
    if (const auto it = idle_.find(shape);
        it != idle_.end() && !it->second.empty()) {
      entry = std::move(it->second.back());
      it->second.pop_back();
    }
  }
  if (entry == nullptr) {
    TT_ASSIGN_OR_RETURN(entry, CreateEntry(shape, device));
  }

  // Only this thread owns `entry`, and 1D tiling pads only after the `rank`
  // contiguous elements, so a flat copy into the pinned host buffer is safe.
  absl::c_copy(dimensions,
               static_cast<int32_t*>(
                   entry->host_memory->OpaqueDeviceMemoryDataPointer()));

  xla::PjRtBuffer* const buffer = entry->buffer.get();
  // `std::shared_ptr` deleters must be copyable, so transfer `entry` as a raw
  // pointer and ignore the `PjRtBuffer*` argument (owned inside `entry`).
  return std::shared_ptr<xla::PjRtBuffer>(
      buffer,
      [weak = weak_from_this(), entry = entry.release()](xla::PjRtBuffer*) {
        std::unique_ptr<Entry> owned(entry);
        if (std::shared_ptr<RuntimeShapeMetadataPool> pool = weak.lock()) {
          absl::MutexLock lock(pool->mu_);
          pool->idle_[owned->shape].push_back(std::move(owned));
        }
      });
}

}  // namespace torch_tpu
