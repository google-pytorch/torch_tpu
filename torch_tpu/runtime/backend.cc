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

#include "torch_tpu/runtime/backend.h"

#include <memory>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/runtime/backend_config.h"
#include "torch_tpu/runtime/backend_registry.h"
#include "torch_tpu/runtime/cerebro/cerebro_backend_impl.h"
#include "torch_tpu/runtime/mock_backend.h"
#include "torch_tpu/runtime/pjrt/pjrt_backend_impl.h"

namespace torch_tpu {
namespace {

struct GlobalBackendState {
  absl::Mutex mutex;
  std::unique_ptr<Backend> active_backend;
};

GlobalBackendState& GetGlobalState() {
  static absl::NoDestructor<GlobalBackendState> state;
  return *state;
}

}  // namespace

absl::StatusOr<std::unique_ptr<Backend>> Backend::Create(
    BackendKind kind, const BackendOptions& options) {
  if (HasRegisteredBackendFactory(kind)) {
    return CreateRegisteredBackend(kind, options);
  }
  switch (kind) {
    case BackendKind::kMock:
      return std::make_unique<MockBackend>();
    case BackendKind::kPjrt:
      return std::make_unique<PjrtBackendImpl>(options);
    case BackendKind::kCerebro:
      return std::make_unique<CerebroBackendImpl>(options);
  }
  return TT_ERROR(  // ERROR_COV_INFEASIBLE=Unknown BackendKind enum value
             error::kInvalidArgument)
         << "unknown BackendKind requested";
}

absl::StatusOr<std::unique_ptr<Backend>> Backend::CreateForDevice(
    const std::string& device_type, BackendKind kind,
    const BackendOptions& options) {
  BackendOptions opts = options;
  if (device_type == "xla_cpu") {
    opts.platform_name = "CPU";
    return Backend::Create(BackendKind::kPjrt, opts);
  } else if (device_type == "xla_cuda") {
    opts.platform_name = "CUDA";
    return Backend::Create(BackendKind::kPjrt, opts);
  }
  return Backend::Create(kind, opts);
}

Backend* Backend::GetInstance() {
  GlobalBackendState& state = GetGlobalState();
  absl::MutexLock lock(state.mutex);
  if (state.active_backend == nullptr) {
    RuntimeBackendConfig config = GetGlobalBackendConfig();
    BackendOptions default_options;
    auto backend_or = Backend::Create(config.default_backend, default_options);
    if (backend_or.ok()) {
      state.active_backend = std::move(*backend_or);
      ABSL_LOG(INFO) << "[TorchTPU] Initialized runtime execution engine: "
                     << BackendKindToString(config.default_backend)
                     << " (cluster: " << config.cluster_name << ")";
    } else {
      // Default fallback to PJRT backend for zero-config startup.
      state.active_backend = std::make_unique<PjrtBackendImpl>(default_options);
      ABSL_LOG(WARNING) << "[TorchTPU] Failed to initialize "
                        << BackendKindToString(config.default_backend)
                        << ", falling back to pjrt: " << backend_or.status();
    }
  }
  return state.active_backend.get();
}

void Backend::SetInstance(std::unique_ptr<Backend> backend) {
  GlobalBackendState& state = GetGlobalState();
  absl::MutexLock lock(state.mutex);
  state.active_backend = std::move(backend);
}

void Backend::ResetInstance() {
  GlobalBackendState& state = GetGlobalState();
  absl::MutexLock lock(state.mutex);
  state.active_backend.reset();
}

}  // namespace torch_tpu
