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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "c10/core/Device.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/distributed/slicebuilder/discovery.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/common/environment.h"
#include "torch_tpu/pjrt/pjrt_client.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/plugin_names.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

PjrtBackend& PjrtBackend::GetInstance() {
  static absl::NoDestructor<PjrtBackend> instance;
  return *instance;
}

void PjrtBackend::SetPjRtInitializationOptions(
    const PjRtInitializationOptions& options) {
  absl::MutexLock lock(mutex_);
  options_ = options;
}

absl::Status PjrtBackend::EnsureInitialized() {
  {
    absl::ReaderMutexLock lock(mutex_);
    if (init_attempted_) {
      return init_status_;
    }
  }

  absl::MutexLock lock(mutex_);
  if (init_attempted_) {
    return init_status_;
  }

  init_status_ = InitializeInternal();
  init_attempted_ = true;
  return init_status_;
}

absl::Status PjrtBackend::InitializeInternal() {
  PjRtDeviceType device_type = PjRtDeviceType::kUnknown;
  std::string plugin_name;
  if (options_.device_type == "tpu") {
    device_type = PjRtDeviceType::kTpu;
    plugin_name = kTpuPjrtName;

    if (auto world_size_or = GetWorldSizeFromEnvOnce(); world_size_or.ok()) {
      TT_ASSIGN_OR_RETURN(auto config, GetDistributedWorkerConfiguration());
      TT_RETURN_IF_ERROR(InitializeDistributedEnvironment(config)).SetPrepend()
          << "InitializeDistributedEnvironment failed: ";
    }
  } else if (options_.device_type == "xla_cuda") {
    plugin_name = kGpuPjrtName;
    device_type = PjRtDeviceType::kCuda;
  } else if (options_.device_type == "xla_cpu") {
    plugin_name = kCpuPjrtName;
    device_type = PjRtDeviceType::kCpu;
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "Unsupported device type: " << options_.device_type;
  }

  TT_ASSIGN_OR_RETURN(bool is_initialized,
                      pjrt::IsPjrtPluginInitialized(plugin_name));

  if (!is_initialized) {
    TT_RETURN_IF_ERROR(pjrt::InitializePjrtPlugin(plugin_name)).SetPrepend()
        << "InitializePjrtPlugin failed: ";
  }

  TT_ASSIGN_OR_RETURN(
      client_,
      torch_tpu::GetPjRtClient(options_.device_type,
                               options_.premapped_buffer_size_bytes),
      _.SetPrepend() << "GetPjrtClient failed: ");

  TT_RET_CHECK(client_ != nullptr, error::kInternal)
      << "PjRtClient is null after initialization";
  TT_RET_CHECK(!client_->devices().empty(), error::kNotFound)
      << "no PjRt devices found after client initialization";

  const auto& addressable_devices = client_->addressable_devices();
  TT_RET_CHECK(!addressable_devices.empty(), error::kInternal)
      << "no addressable PjRt devices found";

  int world_size = 1;
  if (auto world_size_or = GetWorldSizeFromEnvOnce(); world_size_or.ok()) {
    world_size = *world_size_or;
  }

  const int device_count = client_->device_count();

  // Initialize the global device count. This is used by the CompilationCache
  // to determine the number of replicas of the XLA computation.
  TT_RET_CHECK(world_size == 1 || world_size == device_count,
               error::kInvalidArgument)
      << "Invalid world_size configuration: expected 1 or " << device_count
      << ", but got " << world_size
      << ". Please check your environment variables.";

  if (addressable_devices.size() > 1 && world_size == 1) {
    ABSL_LOG(WARNING) << "Only using 1 device out of all the "
                      << addressable_devices.size() << " addressable devices.";
  }

  device_ = addressable_devices[0];
  device_type_ = device_type;
  global_device_count_ = world_size;
  global_device_id_ = device_->global_device_id().value();
  return absl::OkStatus();
}

bool PjrtBackend::IsInitialized() const {
  absl::ReaderMutexLock lock(mutex_);
  return client_ != nullptr;
}

xla::PjRtClient* absl_nullable PjrtBackend::GetClient() {
  if (!EnsureInitialized().ok()) {
    return nullptr;
  }
  absl::ReaderMutexLock lock(mutex_);
  return client_.get();
}

xla::PjRtDevice* absl_nullable PjrtBackend::GetDevice() {
  if (!EnsureInitialized().ok()) {
    return nullptr;
  }
  absl::ReaderMutexLock lock(mutex_);
  return device_;
}

absl::StatusOr<int> PjrtBackend::GetGlobalDeviceCount() {
  TT_RETURN_IF_ERROR(EnsureInitialized());
  absl::ReaderMutexLock lock(mutex_);
  if (client_ == nullptr) {
    return TT_ERROR(error::kInternal) << "PjRt is not initialized";
  }
  return global_device_count_;
}

absl::StatusOr<int> PjrtBackend::GetGlobalDeviceId() {
  TT_RETURN_IF_ERROR(EnsureInitialized());
  absl::ReaderMutexLock lock(mutex_);
  if (client_ == nullptr) {
    return TT_ERROR(error::kInternal) << "PjRt is not initialized";
  }
  return global_device_id_;
}

PjRtDeviceType PjrtBackend::GetDeviceType() {
  absl::ReaderMutexLock lock(mutex_);
  return device_type_;
}

absl::StatusOr<tsl::AllocatorStats> PjrtBackend::GetAllocatorStats() {
  xla::PjRtDevice* device = GetDevice();
  if (device == nullptr) {
    return TT_ERROR(error::kInternal) << "PjRt device is not initialized";
  }
  return device->GetAllocatorStats();
}

absl::StatusOr<xla::PjRtClient::HostAllocator*>
PjrtBackend::GetHostAllocator() {
  xla::PjRtClient* client = GetClient();
  if (client == nullptr) {
    return TT_ERROR(error::kInternal) << "PjRt client is not initialized";
  }
  return client->GetHostAllocator();
}

void PjrtBackend::Shutdown() {
  absl::MutexLock lock(mutex_);
  if (!init_attempted_) {
    return;
  }
  ABSL_VLOG(1) << "ShutdownPjRt";

  client_.reset();
  device_ = nullptr;
  device_type_ = PjRtDeviceType::kUnknown;
  global_device_count_ = 0;
  init_attempted_ = false;
  init_status_ = absl::OkStatus();
  ABSL_VLOG(1) << "PjRt shut down.";
}

namespace {
struct StreamState {
  using SharedFuture = std::shared_ptr<xla::Future<void>>;

  absl::Mutex mutex;
  absl::flat_hash_map<c10::DeviceIndex, std::vector<SharedFuture>>
      pending_futures ABSL_GUARDED_BY(mutex);
  absl::flat_hash_map<int64_t, std::vector<SharedFuture>> event_snapshots
      ABSL_GUARDED_BY(mutex);
  int64_t next_snapshot_id ABSL_GUARDED_BY(mutex) = 1;
};

StreamState& GetStreamState() {
  static absl::NoDestructor<StreamState> state;
  return *state;
}

void PruneCompletedFutures(std::vector<StreamState::SharedFuture>& futures) {
  futures.erase(std::remove_if(futures.begin(), futures.end(),
                               [](const StreamState::SharedFuture& future) {
                                 return future == nullptr ||
                                        !future->IsValid() || future->IsReady();
                               }),
                futures.end());
}
}  // namespace

void PjrtBackend::MarkStreamActive(c10::DeviceIndex device_index,
                                   xla::Future<void> future) {
  StreamState& state = GetStreamState();
  absl::MutexLock lock(state.mutex);
  auto& futures = state.pending_futures[device_index];
  PruneCompletedFutures(futures);
  futures.push_back(std::make_shared<xla::Future<void>>(std::move(future)));
}

// Blocks the calling thread until all previously enqueued operations on the
// specified device's stream have completed.
// If multiple threads call SynchronizeStream for the same device concurrently,
// all threads will block until the pending operations at the time of their
// call are finished.
void PjrtBackend::SynchronizeStream(c10::DeviceIndex device_index) {
  StreamState& state = GetStreamState();
  absl::MutexLock lock(state.mutex);
  auto it = state.pending_futures.find(device_index);
  if (it == state.pending_futures.end()) {
    // No pending futures for this device.
    return;
  }
  for (auto& future : it->second) {
    if (future != nullptr && future->IsValid()) {
      absl::Status s = future->Await();
      if (!s.ok()) {
        ABSL_LOG(ERROR) << "Stream synchronization failed for device "
                        << static_cast<int>(device_index) << ": " << s;
      }
    }
  }
  it->second.clear();
}

int64_t RecordEventSnapshot(c10::DeviceIndex device_index) {
  StreamState& state = GetStreamState();
  absl::MutexLock lock(state.mutex);
  const int64_t event_id = state.next_snapshot_id++;
  auto it = state.pending_futures.find(device_index);
  std::vector<StreamState::SharedFuture> futures =
      it != state.pending_futures.end()
          ? it->second
          : std::vector<StreamState::SharedFuture>{};
  state.event_snapshots.emplace(event_id, std::move(futures));
  return event_id;
}

absl::Status WaitEventSnapshot(int64_t event_id) {
  std::vector<StreamState::SharedFuture> futures;
  {
    StreamState& state = GetStreamState();
    absl::MutexLock lock(state.mutex);
    auto it = state.event_snapshots.find(event_id);
    if (it == state.event_snapshots.end()) {
      return TT_ERROR(error::kInvalidArgument)
             << "Unknown event snapshot id: " << event_id;
    }
    futures = it->second;
  }

  for (auto& future : futures) {
    if (future != nullptr && future->IsValid()) {
      TT_RETURN_IF_ERROR(future->Await());
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> QueryEventSnapshot(int64_t event_id) {
  std::vector<StreamState::SharedFuture> futures;
  {
    StreamState& state = GetStreamState();
    absl::MutexLock lock(state.mutex);
    auto it = state.event_snapshots.find(event_id);
    if (it == state.event_snapshots.end()) {
      return TT_ERROR(error::kInvalidArgument)
             << "Unknown event snapshot id: " << event_id;
    }
    futures = it->second;
  }
  for (const auto& future : futures) {
    if (future != nullptr && future->IsValid() && !future->IsReady()) {
      return false;
    }
  }
  return true;
}

void ReleaseEventSnapshot(int64_t event_id) {
  StreamState& state = GetStreamState();
  absl::MutexLock lock(state.mutex);
  state.event_snapshots.erase(event_id);
}

}  // namespace torch_tpu
