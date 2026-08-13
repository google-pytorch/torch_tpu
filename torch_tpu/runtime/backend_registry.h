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

#ifndef TORCH_TPU_RUNTIME_BACKEND_REGISTRY_H_
#define TORCH_TPU_RUNTIME_BACKEND_REGISTRY_H_

#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "torch_tpu/runtime/backend.h"

namespace torch_tpu {

// Function signature for dynamic backend creation factories.
using BackendFactory =
    absl::AnyInvocable<std::unique_ptr<Backend>(const BackendOptions&)>;

// Registers a dynamic backend factory for the specified BackendKind.
void RegisterBackendFactory(BackendKind kind, BackendFactory factory);

// Checks if a custom factory has been registered for the given BackendKind.
bool HasRegisteredBackendFactory(BackendKind kind);

// Instantiates a backend using a registered factory if available.
absl::StatusOr<std::unique_ptr<Backend>> CreateRegisteredBackend(
    BackendKind kind, const BackendOptions& options);

// Clears all custom registered factories (for testing teardown).
void ResetRegisteredBackendFactories();

}  // namespace torch_tpu

#endif  // THIRD_PARTY_PY_TORCH_TPU_RUNTIME_BACKEND_REGISTRY_H_
