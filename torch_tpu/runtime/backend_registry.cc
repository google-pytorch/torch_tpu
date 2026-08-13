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

#include "torch_tpu/runtime/backend_registry.h"

#include <functional>
#include <memory>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/runtime/backend.h"

namespace torch_tpu {
namespace {

struct RegistryState {
  absl::Mutex mutex;
  absl::flat_hash_map<BackendKind, BackendFactory> factories;
};

RegistryState& GetRegistryState() {
  static absl::NoDestructor<RegistryState> state;
  return *state;
}

}  // namespace

void RegisterBackendFactory(BackendKind kind, BackendFactory factory) {
  RegistryState& state = GetRegistryState();
  absl::MutexLock lock(state.mutex);
  state.factories[kind] = std::move(factory);
}

bool HasRegisteredBackendFactory(BackendKind kind) {
  RegistryState& state = GetRegistryState();
  absl::MutexLock lock(state.mutex);
  return state.factories.contains(kind);
}

absl::StatusOr<std::unique_ptr<Backend>> CreateRegisteredBackend(
    BackendKind kind, const BackendOptions& options) {
  RegistryState& state = GetRegistryState();
  absl::MutexLock lock(state.mutex);
  auto it = state.factories.find(kind);
  if (it == state.factories.end()) {
    return TT_ERROR(  // ERROR_COV_INFEASIBLE=Unregistered backend factory
               error::kNotFound)
           << "no registered backend factory found for backend kind";
  }
  return it->second(options);
}

void ResetRegisteredBackendFactories() {
  RegistryState& state = GetRegistryState();
  absl::MutexLock lock(state.mutex);
  state.factories.clear();
}

}  // namespace torch_tpu
