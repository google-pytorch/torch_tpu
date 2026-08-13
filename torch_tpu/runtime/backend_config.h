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

#ifndef TORCH_TPU_RUNTIME_BACKEND_CONFIG_H_
#define TORCH_TPU_RUNTIME_BACKEND_CONFIG_H_

#include <optional>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "torch_tpu/runtime/backend.h"

namespace torch_tpu {

// Configuration holder for TorchTPU Runtime (TTR) backend selection and flags.
struct RuntimeBackendConfig {
  BackendKind default_backend = BackendKind::kCerebro;
  std::string cluster_name = "default";
  int num_nodes = 1;
  int chips_per_node = 4;
};

// Parses a backend kind string (e.g. "pjrt", "cerebro", "mock") into
// BackendKind.
absl::StatusOr<BackendKind> ParseBackendKind(std::string_view name);

// Converts a BackendKind enum into its canonical string representation.
std::string_view BackendKindToString(BackendKind kind);

// Retrieves the global runtime configuration, loaded from environment variables
// (such as TORCH_TPU_BACKEND) and programmatic overrides.
RuntimeBackendConfig GetGlobalBackendConfig();

// Sets a programmatic override for the global backend kind.
void SetGlobalBackendKind(BackendKind kind);

// Sets the global runtime configuration override.
void SetGlobalBackendConfig(const RuntimeBackendConfig& config);

// Resets global backend configuration to default environment-driven state.
void ResetGlobalBackendConfig();

}  // namespace torch_tpu

#endif  // THIRD_PARTY_PY_TORCH_TPU_RUNTIME_BACKEND_CONFIG_H_
