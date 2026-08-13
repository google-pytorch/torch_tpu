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

#include "torch_tpu/_internal/sync/sync_backend_bridge.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/backend_config.h"
#include "torch_tpu/runtime/cerebro/cerebro_backend_impl.h"
#include "torch_tpu/runtime/cerebro/cerebro_cluster_pipeline.h"
#include "torch_tpu/runtime/eager_runtime_adapter.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

absl::Status SyncBackendBridge::SyncAllDevices() {
  return EagerRuntimeAdapter::SynchronizeAllDevices();
}

absl::Status SyncBackendBridge::ClusterBarrier() {
  Backend* backend = Backend::GetInstance();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Backend pointer check
      backend != nullptr, error::kInternal)
      << "global backend instance is null";

  if (auto* cerebro = dynamic_cast<CerebroBackendImpl*>(backend)) {
    CerebroClusterPipeline pipeline(cerebro);
    return pipeline.Barrier();
  }

  return EagerRuntimeAdapter::SynchronizeAllDevices();
}

absl::StatusOr<tsl::AllocatorStats> SyncBackendBridge::GetMemoryStats() {
  return EagerRuntimeAdapter::GetAllocatorStats();
}

}  // namespace torch_tpu
