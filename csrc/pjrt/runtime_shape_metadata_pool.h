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

#ifndef TORCH_TPU_CSRC_PJRT_RUNTIME_SHAPE_METADATA_POOL_H_
#define TORCH_TPU_CSRC_PJRT_RUNTIME_SHAPE_METADATA_POOL_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"

namespace torch_tpu {

// A pool of S32[rank] buffers in pinned host memory, for the runtime shape
// metadata parameters of unbounded dynamic executables: each holds the runtime
// dimensions of one dynamic operand.
//
// Unlike a value cache keyed by payload dimensions, this pool keys idle buffers
// by the metadata buffer's own `xla::Shape` (rank and layout). Because pinned
// host memory is CPU-writable, each execution overwrites an idle buffer in
// place rather than creating a new device buffer. The executable must take the
// parameter in host memory (`S(5)`) and copy it to the device itself.
//
// PJRT's limit on in-flight executions bounds the pool's growth; idle buffers
// live until the pool is destroyed. Destroy the pool and all acquired buffers
// before destroying the PjRtClient. Thread-safe.
class RuntimeShapeMetadataPool
    : public std::enable_shared_from_this<RuntimeShapeMetadataPool> {
 public:
  [[nodiscard]] static absl_nonnull std::shared_ptr<RuntimeShapeMetadataPool>
  Create() {
    return std::shared_ptr<RuntimeShapeMetadataPool>(
        new RuntimeShapeMetadataPool());
  }

  RuntimeShapeMetadataPool(const RuntimeShapeMetadataPool&) = delete;
  RuntimeShapeMetadataPool& operator=(const RuntimeShapeMetadataPool&) = delete;

  // Returns a pinned host buffer of `shape` on `device` that holds
  // `dimensions`, reusing an idle one if available.
  //
  // The buffer returns to the pool when the last copy of the returned pointer
  // is destroyed; if the pool is already gone, the buffer is destroyed instead.
  // Keep a copy until every execution that reads the buffer is complete.
  //
  // Idle buffers are keyed by `shape` only, so callers must pass the same
  // `device`.
  absl::StatusOr<absl_nonnull std::shared_ptr<xla::PjRtBuffer>> Acquire(
      absl::Span<const int32_t> dimensions, const xla::Shape& shape,
      const xla::PjRtDevice& device) ABSL_LOCKS_EXCLUDED(mu_);

 private:
  // Caches the `idle_` key and pinned host pointer so Acquire() and release
  // make no PJRT queries on the hot path. Holding `host_memory` across calls is
  // safe because checked-out entries have exclusive host access during writes
  // and metadata buffers are never donated.
  struct Entry {
    xla::Shape shape;
    std::unique_ptr<xla::PjRtBuffer> buffer;
    // Declared after `buffer` so the external reference is released first.
    std::unique_ptr<xla::PjRtBuffer::ExternalReference> host_memory;
  };

  static absl::StatusOr<std::unique_ptr<Entry>> CreateEntry(
      const xla::Shape& shape, const xla::PjRtDevice& device);

  RuntimeShapeMetadataPool() = default;

  absl::Mutex mu_;
  // Keyed by the metadata buffer's own `xla::Shape` (e.g.
  // S32[rank]{0:T(128)S(5)}), not by payload dimensions. Entries are popped
  // while in use and pushed back on release, so concurrent executions never
  // share a buffer.
  absl::flat_hash_map<xla::Shape, std::vector<std::unique_ptr<Entry>>> idle_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_PJRT_RUNTIME_SHAPE_METADATA_POOL_H_
