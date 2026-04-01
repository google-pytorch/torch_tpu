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

#include "torch_tpu/pjrt/pjrt_state.h"

#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/Device.h"
#include "c10/core/TensorImpl.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_types.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {
namespace {

struct GlobalPjRtState {
  absl::Mutex mutex;
  xla::PjRtClient* client = nullptr;
  xla::PjRtDevice* device = nullptr;
  PjRtDeviceType device_type = PjRtDeviceType::kUnknown;
  int global_device_count = 0;
  std::string backend_name = "tpu";
};

GlobalPjRtState& GetGlobalState() {
  static absl::NoDestructor<GlobalPjRtState> state;
  return *state;
}

}  // namespace

xla::PjRtClient* GetPjRtClient() {
  auto& state = GetGlobalState();
  TT_READER_MUTEX_LOCK(lock, state.mutex);
  return state.client;
}

xla::PjRtDevice* GetPjRtDevice() {
  auto& state = GetGlobalState();
  TT_READER_MUTEX_LOCK(lock, state.mutex);
  return state.device;
}

void SetPjRtDevice(xla::PjRtDevice* device) {
  auto& state = GetGlobalState();
  TT_MUTEX_LOCK(lock, state.mutex);
  state.device = device;
}

bool IsPjRtInitialized() {
  auto& state = GetGlobalState();
  TT_READER_MUTEX_LOCK(lock, state.mutex);
  return state.client != nullptr;
}

absl::StatusOr<int> GetGlobalDeviceCount() {
  auto& state = GetGlobalState();
  TT_READER_MUTEX_LOCK(lock, state.mutex);
  if (state.client == nullptr) {
    return TT_ERROR(error::kInternal) << "PjRt is not initialized";
  }
  return state.global_device_count;
}

PjRtDeviceType GetPjRtDeviceType() {
  auto& state = GetGlobalState();
  TT_READER_MUTEX_LOCK(lock, state.mutex);
  return state.device_type;
}

void SetPjRtBackendName(const std::string& device_type) {
  auto& state = GetGlobalState();
  TT_MUTEX_LOCK(lock, state.mutex);
  state.backend_name = device_type;
}

std::string GetPjRtBackendName() {
  auto& state = GetGlobalState();
  TT_READER_MUTEX_LOCK(lock, state.mutex);
  return state.backend_name;
}

void SetPjRtState(xla::PjRtClient* client, xla::PjRtDevice* device,
                  PjRtDeviceType device_type, int global_device_count) {
  auto& state = GetGlobalState();
  TT_MUTEX_LOCK(lock, state.mutex);
  state.client = client;
  state.device = device;
  state.device_type = device_type;
  state.global_device_count = global_device_count;
}

absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() {
  xla::PjRtDevice* device = GetPjRtDevice();
  if (device == nullptr) {
    return TT_ERROR(error::kInternal) << "PjRt device is not initialized";
  }
  return device->GetAllocatorStats();
}

absl::StatusOr<xla::PjRtClient::HostAllocator*> GetHostAllocator() {
  xla::PjRtClient* client = GetPjRtClient();
  if (client == nullptr) {
    return TT_ERROR(error::kInternal) << "PjRt client is not initialized";
  }
  return client->GetHostAllocator();
}

namespace {

struct TensorHash {
  size_t operator()(const at::Tensor& t) const {
    return std::hash<at::TensorImpl*>()(t.unsafeGetTensorImpl());
  }
};

struct TensorEqual {
  bool operator()(const at::Tensor& lhs, const at::Tensor& rhs) const {
    return lhs.unsafeGetTensorImpl() == rhs.unsafeGetTensorImpl();
  }
};

}  // namespace

using TensorToFutureMap =
    absl::flat_hash_map<at::Tensor, xla::Future<void>, TensorHash, TensorEqual>;

struct StreamState {
  absl::Mutex mutex;
  TensorToFutureMap pending_futures ABSL_GUARDED_BY(mutex);
};

StreamState& GetStreamState() {
  static absl::NoDestructor<StreamState> state;
  return *state;
}

absl::Status MarkStreamActive(const at::Tensor& tensor,
                              xla::Future<void> future) {
  StreamState& state = GetStreamState();
  TT_MUTEX_LOCK(lock, state.mutex);
  auto [it, inserted] =
      state.pending_futures.emplace(tensor, std::move(future));
  if (!inserted) {
    return TT_ERROR(error::kInternal)
           << "Duplicate future registered for the same tensor";
  }
  return absl::OkStatus();
}

// Blocks the calling thread until all previously enqueued operations on the
// specified device's stream have completed.
// If multiple threads call SynchronizeStream for the same device concurrently,
// all threads will block until the pending operations at the time of their
// call are finished.
void SynchronizeStream(c10::DeviceIndex device_index) {
  StreamState& state = GetStreamState();
  TT_MUTEX_LOCK(lock, state.mutex);
  std::vector<at::Tensor> to_erase;
  for (const auto& [tensor, future] : state.pending_futures) {
    if (tensor.device().index() == device_index && future.IsValid()) {
      absl::Status s = future.Await();
      if (!s.ok()) {
        ABSL_LOG(ERROR) << "Stream synchronization failed for device "
                        << static_cast<int>(device_index) << ": " << s;
      }
      to_erase.push_back(tensor);
    }
  }
  for (const auto& tensor : to_erase) {
    state.pending_futures.erase(tensor);
  }
}

absl::Status SynchronizeHostPinnedTensors(
    const std::vector<at::Tensor>& tensors) {
  StreamState& state = GetStreamState();
  TT_MUTEX_LOCK(lock, state.mutex);
  for (const at::Tensor& tensor : tensors) {
    auto it = state.pending_futures.find(tensor);
    if (it != state.pending_futures.end()) {
      if (it->second.IsValid()) {
        absl::Status s = it->second.Await();
        state.pending_futures.erase(it);
        if (!s.ok()) {
          return s;
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace torch_tpu
