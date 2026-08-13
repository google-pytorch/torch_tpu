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

#include "torch_tpu/runtime/backend_config.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

#include "absl/base/no_destructor.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/runtime/backend.h"

namespace torch_tpu {
namespace {

struct ConfigState {
  absl::Mutex mutex;
  std::optional<RuntimeBackendConfig> explicit_config;
};

ConfigState& GetConfigState() {
  static absl::NoDestructor<ConfigState> state;
  return *state;
}

RuntimeBackendConfig LoadFromEnvironment() {
  RuntimeBackendConfig config;
  const char* backend_env =
      std::getenv("TORCH_TPU_BACKEND");  // GETENV_OK=Backend
  if (backend_env != nullptr) {
    auto parsed_kind = ParseBackendKind(backend_env);
    if (parsed_kind.ok()) {
      config.default_backend = *parsed_kind;
    }
  }

  const char* cluster_env =
      std::getenv("TORCH_TPU_CLUSTER_NAME");  // GETENV_OK=Cluster
  if (cluster_env != nullptr) {
    config.cluster_name = cluster_env;
  }

  const char* nodes_env =
      std::getenv("TORCH_TPU_CEREBRO_NODES");  // GETENV_OK=Nodes
  if (nodes_env != nullptr) {
    int parsed_nodes = 1;
    if (absl::SimpleAtoi(nodes_env, &parsed_nodes) && parsed_nodes > 0) {
      config.num_nodes = parsed_nodes;
    }
  }

  const char* chips_env =
      std::getenv("TORCH_TPU_CEREBRO_CHIPS_PER_NODE");  // GETENV_OK=Chips
  if (chips_env != nullptr) {
    int parsed_chips = 4;
    if (absl::SimpleAtoi(chips_env, &parsed_chips) && parsed_chips > 0) {
      config.chips_per_node = parsed_chips;
    }
  }

  return config;
}

}  // namespace

absl::StatusOr<BackendKind> ParseBackendKind(std::string_view name) {
  std::string lower_name = absl::AsciiStrToLower(name);
  if (lower_name == "pjrt") {
    return BackendKind::kPjrt;
  }
  if (lower_name == "mock") {
    return BackendKind::kMock;
  }
  if (lower_name == "cerebro") {
    return BackendKind::kCerebro;
  }
  return TT_ERROR(  // ERROR_COV_INFEASIBLE=Invalid backend name string
             error::kInvalidArgument)
         << "unknown backend name: " << name;
}

std::string_view BackendKindToString(BackendKind kind) {
  switch (kind) {
    case BackendKind::kPjrt:
      return "pjrt";
    case BackendKind::kMock:
      return "mock";
    case BackendKind::kCerebro:
      return "cerebro";
  }
  return "unknown";
}

RuntimeBackendConfig GetGlobalBackendConfig() {
  ConfigState& state = GetConfigState();
  absl::MutexLock lock(state.mutex);
  if (state.explicit_config.has_value()) {
    return *state.explicit_config;
  }
  return LoadFromEnvironment();
}

void SetGlobalBackendKind(BackendKind kind) {
  ConfigState& state = GetConfigState();
  absl::MutexLock lock(state.mutex);
  if (!state.explicit_config.has_value()) {
    state.explicit_config = LoadFromEnvironment();
  }
  state.explicit_config->default_backend = kind;
}

void SetGlobalBackendConfig(const RuntimeBackendConfig& config) {
  ConfigState& state = GetConfigState();
  absl::MutexLock lock(state.mutex);
  state.explicit_config = config;
}

void ResetGlobalBackendConfig() {
  ConfigState& state = GetConfigState();
  absl::MutexLock lock(state.mutex);
  state.explicit_config.reset();
}

}  // namespace torch_tpu
