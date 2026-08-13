// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TORCH_TPU_INTERNAL_SYNC_SYNC_BACKEND_BRIDGE_H_
#define TORCH_TPU_INTERNAL_SYNC_SYNC_BACKEND_BRIDGE_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

// Provides execution synchronization, multi-device barrier, and memory
// metrics through the active TTR backend.
class SyncBackendBridge {
 public:
  // Synchronizes all outstanding asynchronous device operations.
  static absl::Status SyncAllDevices();

  // Executes a global cluster-level barrier synchronization across all devices.
  static absl::Status ClusterBarrier();

  // Queries memory allocator statistics from the active backend.
  static absl::StatusOr<tsl::AllocatorStats> GetMemoryStats();
};

}  // namespace torch_tpu

#endif  // THIRD_PARTY_PY_TORCH_TPU__INTERNAL_SYNC_SYNC_BACKEND_BRIDGE_H_
