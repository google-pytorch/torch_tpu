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

#include "torch_tpu/ifrt/ifrt_state.h"

#include <memory>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"

namespace torch_tpu {

IfrtBackend& IfrtBackend::GetInstance() {
  static absl::NoDestructor<IfrtBackend> instance;
  return *instance;
}

absl::Status IfrtBackend::EnsureInitialized() {
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

absl::Status IfrtBackend::InitializeInternal() {
  TT_RETURN_IF_ERROR(PjrtBackend::GetInstance().EnsureInitialized());
  TT_ASSIGN_OR_RETURN(auto pjrt_client,
                      PjrtBackend::GetInstance().GetSharedClient());

  client_ = xla::ifrt::PjRtClient::Create(std::move(pjrt_client));
  ABSL_CHECK_NE(client_, nullptr)  // CRASH_OK
      << "IFRT Client should not be null after wrapping PjRtClient.";
  ABSL_CHECK(!client_->devices().empty())  // CRASH_OK
      << "No IFRT devices found after client initialization.";

  const auto& addressable_devices = client_->addressable_devices();
  ABSL_CHECK(!addressable_devices.empty())  // CRASH_OK
      << "No addressable IFRT devices found.";

  device_ = addressable_devices[0];

  TT_ASSIGN_OR_RETURN(global_device_count_,
                      PjrtBackend::GetInstance().GetGlobalDeviceCount());
  TT_ASSIGN_OR_RETURN(global_device_id_,
                      PjrtBackend::GetInstance().GetGlobalDeviceId());

  return absl::OkStatus();
}

bool IfrtBackend::IsInitialized() const {
  absl::ReaderMutexLock lock(mutex_);
  return client_ != nullptr;
}

xla::ifrt::Client* absl_nullable IfrtBackend::GetClient() {
  if (const absl::Status s = EnsureInitialized(); !s.ok()) {
    ABSL_LOG(ERROR) << "IfrtBackend::GetClient failed to initialize: " << s;
    return nullptr;
  }
  absl::ReaderMutexLock lock(mutex_);
  return client_.get();
}

absl::StatusOr<absl_nonnull std::shared_ptr<xla::ifrt::Client>>
IfrtBackend::GetSharedClient() {
  TT_RETURN_IF_ERROR(EnsureInitialized());
  absl::ReaderMutexLock lock(mutex_);
  return client_;
}

xla::ifrt::Device* absl_nullable IfrtBackend::GetDevice() {
  if (const absl::Status s = EnsureInitialized(); !s.ok()) {
    ABSL_LOG(ERROR) << "IfrtBackend::GetDevice failed to initialize: " << s;
    return nullptr;
  }
  absl::ReaderMutexLock lock(mutex_);
  return device_;
}

absl::StatusOr<int> IfrtBackend::GetGlobalDeviceCount() {
  TT_RETURN_IF_ERROR(EnsureInitialized());
  absl::ReaderMutexLock lock(mutex_);
  if (client_ == nullptr) {
    return TT_ERROR(error::kInternal) << "IFRT is not initialized";
  }
  return global_device_count_;
}

absl::StatusOr<int> IfrtBackend::GetGlobalDeviceId() {
  TT_RETURN_IF_ERROR(EnsureInitialized());
  absl::ReaderMutexLock lock(mutex_);
  if (client_ == nullptr) {
    return TT_ERROR(error::kInternal) << "IFRT is not initialized";
  }
  return global_device_id_;
}

void IfrtBackend::Shutdown() {
  absl::MutexLock lock(mutex_);
  if (!init_attempted_) {
    return;
  }
  ABSL_VLOG(1) << "ShutdownIFRT";

  client_.reset();
  device_ = nullptr;
  global_device_count_ = 0;
  global_device_id_ = -1;
  init_attempted_ = false;
  init_status_ = absl::OkStatus();
  ABSL_VLOG(1) << "IFRT shut down.";
}

}  // namespace torch_tpu
