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

#ifndef TORCH_TPU_IFRT_IFRT_STATE_H_
#define TORCH_TPU_IFRT_IFRT_STATE_H_

#include <memory>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"

namespace torch_tpu {

// IfrtBackend manages the lifecycle and state of the IFRT runtime wrapper.
//
// It provides a singleton interface to initialize and access the IFRT client
// and devices. Initialization is lazy and thread-safe.
class IfrtBackend {
 public:
  // IfrtBackend is neither copyable nor movable.
  IfrtBackend(const IfrtBackend&) = delete;
  IfrtBackend& operator=(const IfrtBackend&) = delete;
  IfrtBackend(IfrtBackend&&) = delete;
  IfrtBackend& operator=(IfrtBackend&&) = delete;

  // Returns the singleton IfrtBackend instance.
  [[nodiscard]] static IfrtBackend& GetInstance();

  // Ensures that the IFRT singleton state is initialized.
  absl::Status EnsureInitialized() ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns whether the IFRT runtime is initialized.
  [[nodiscard]] bool IsInitialized() const ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the IFRT Client singleton. Triggers lazy initialization if needed.
  [[nodiscard]] xla::ifrt::Client* absl_nullable GetClient()
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns a shared pointer to the IFRT Client. Triggers lazy initialization.
  absl::StatusOr<absl_nonnull std::shared_ptr<xla::ifrt::Client>>
  GetSharedClient() ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the default IFRT Device. Triggers lazy initialization if needed.
  [[nodiscard]] xla::ifrt::Device* absl_nullable GetDevice()
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the global device count. Triggers lazy initialization if needed.
  absl::StatusOr<int> GetGlobalDeviceCount() ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the global ID of the current device. Triggers lazy initialization.
  absl::StatusOr<int> GetGlobalDeviceId() ABSL_LOCKS_EXCLUDED(mutex_);

  // Shuts down the IFRT runtime wrapper and resets global state.
  void Shutdown() ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  friend class absl::NoDestructor<IfrtBackend>;

  IfrtBackend() = default;
  ~IfrtBackend() = default;

  absl::Status InitializeInternal() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutable absl::Mutex mutex_;
  std::shared_ptr<xla::ifrt::Client> absl_nullable client_
      ABSL_GUARDED_BY(mutex_);
  xla::ifrt::Device* absl_nullable device_ ABSL_GUARDED_BY(mutex_) = nullptr;
  int global_device_count_ ABSL_GUARDED_BY(mutex_) = 0;
  int global_device_id_ ABSL_GUARDED_BY(mutex_) = -1;
  bool init_attempted_ ABSL_GUARDED_BY(mutex_) = false;
  absl::Status init_status_ ABSL_GUARDED_BY(mutex_) = absl::OkStatus();
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_IFRT_IFRT_STATE_H_
