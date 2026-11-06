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

#ifndef TORCH_TPU_CSRC_COMMON_COMPILATION_CACHE_TEST_UTILS_H_
#define TORCH_TPU_CSRC_COMMON_COMPILATION_CACHE_TEST_UTILS_H_

#include <memory>

#include "absl/status/statusor.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {

// Returns dummy unbounded dynamic execution metadata whose compatibility check
// accepts exactly one rank-2 shape.
UnboundedDynamicExecutionMetadata MakeDummyDynamicMetadata();

// Compiles a trivial identity computation with `client`. Matches the
// UnboundedDynamicKernel::BuildFn signature, so it can be registered directly
// with UnboundedDynamicKernelRegistry.
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
CreateDummyExecutable(xla::PjRtClient& client,
                      UniqueCompileOptions compile_options);

// Always fails. Matches the UnboundedDynamicKernel::BuildFn signature, for
// tests that need a registered kernel whose build fails.
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
CreateFailingExecutable(xla::PjRtClient& client,
                        UniqueCompileOptions compile_options);

// Registers a kernel built by `build` with `MakeDummyDynamicMetadata()` under
// `key`, ignoring kAlreadyExists so that tests sharing a process can register
// the same key repeatedly. The registry is process-wide and append-only, so
// tests must use keys that are unique to the behaviour they exercise.
void RegisterDummyKernel(
    const CompilationCacheKey& key,
    UnboundedDynamicKernel::BuildFn build = CreateDummyExecutable);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_COMPILATION_CACHE_TEST_UTILS_H_
