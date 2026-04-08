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

#ifndef TORCH_TPU_PJRT_PJRT_STATE_H_
#define TORCH_TPU_PJRT_PJRT_STATE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "c10/core/Device.h"
#include "torch_tpu/eager/device_types.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

// PjRt initialization options. Defaults to TPU, no pre-mapped buffer.
struct PjRtInitializationOptions {
  std::string device_type = "tpu";
  int64_t premapped_buffer_size_bytes = 0;
};

// PjrtBackend manages the lifecycle and state of the PjRt runtime.
//
// It provides a singleton interface to initialize and access the PjRt client
// and devices. Initialization is lazy and thread-safe.
class PjrtBackend {
 public:
  // PjrtBackend is neither copyable nor movable.
  PjrtBackend(const PjrtBackend&) = delete;
  PjrtBackend& operator=(const PjrtBackend&) = delete;
  PjrtBackend(PjrtBackend&&) = delete;
  PjrtBackend& operator=(PjrtBackend&&) = delete;

  // Returns the singleton PjrtBackend instance.
  [[nodiscard]] static PjrtBackend& GetInstance();

  // Sets the global PjRt initialization options. These options will be used
  // for lazy initialization when the first device or client is requested.
  void SetPjRtInitializationOptions(const PjRtInitializationOptions& options)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Ensures that the PjRt singleton state is initialized. This method is
  // idempotent and is triggered by any operation that requires the PjRt
  // infrastructure (e.g., GetClient, GetDevice).
  absl::Status EnsureInitialized() ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns whether the PjRt runtime is initialized.
  [[nodiscard]] bool IsInitialized() const ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the PjRtClient singleton. Triggers lazy initialization if needed.
  [[nodiscard]] xla::PjRtClient* absl_nullable GetClient()
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the PjRtDevice singleton. Triggers lazy initialization if needed.
  [[nodiscard]] xla::PjRtDevice* absl_nullable GetDevice()
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the global device count. Triggers lazy initialization if needed.
  absl::StatusOr<int> GetGlobalDeviceCount() ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the global ID of the current device. Triggers lazy initialization
  // if needed.
  absl::StatusOr<int> GetGlobalDeviceId() ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the PjRt device type.
  [[nodiscard]] PjRtDeviceType GetDeviceType() ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns allocator stats for the current PjRt device.
  absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats()
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the PjRt host allocator from the PjRt client.
  absl::StatusOr<xla::PjRtClient::HostAllocator*> GetHostAllocator()
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Shuts down the PjRt runtime and resets the global state.
  void Shutdown() ABSL_LOCKS_EXCLUDED(mutex_);

  // Updates the tracked future for the given device.
  void MarkStreamActive(c10::DeviceIndex device_index,
                        xla::Future<void> future);

  // Blocks until all pending operations on the given device have completed.
  void SynchronizeStream(c10::DeviceIndex device_index);

 private:
  friend class absl::NoDestructor<PjrtBackend>;

  PjrtBackend() = default;
  ~PjrtBackend() = default;

  // Internal initialization logic that returns an absl::Status.
  absl::Status InitializeInternal() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutable absl::Mutex mutex_;
  std::unique_ptr<xla::PjRtClient> absl_nullable client_
      ABSL_GUARDED_BY(mutex_);
  xla::PjRtDevice* absl_nullable device_ ABSL_GUARDED_BY(mutex_) = nullptr;
  PjRtDeviceType device_type_ ABSL_GUARDED_BY(mutex_) =
      PjRtDeviceType::kUnknown;
  int global_device_count_ ABSL_GUARDED_BY(mutex_) = 0;
  int global_device_id_ ABSL_GUARDED_BY(mutex_) = -1;
  PjRtInitializationOptions options_ ABSL_GUARDED_BY(mutex_) = {};
  bool init_attempted_ ABSL_GUARDED_BY(mutex_) = false;
  absl::Status init_status_ ABSL_GUARDED_BY(mutex_) = absl::OkStatus();
};

// Records a fence over the async d2h copies already enqueued on the device.
int64_t RecordEventSnapshot(c10::DeviceIndex device_index);

// Blocks until the event snapshot has completed.
absl::Status WaitEventSnapshot(int64_t event_id);

// Returns whether the event snapshot has completed.
absl::StatusOr<bool> QueryEventSnapshot(int64_t event_id);

// Releases the event snapshot. Safe to call multiple times.
void ReleaseEventSnapshot(int64_t event_id);

}  // namespace torch_tpu

#endif  // TORCH_TPU_PJRT_PJRT_STATE_H_
