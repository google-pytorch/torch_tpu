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
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {

// Returns dummy unbounded dynamic execution metadata whose compatibility check
// accepts exactly one rank-2 shape.
UnboundedDynamicExecutionMetadata MakeDummyDynamicMetadata();

// Compiles a trivial identity computation with `client`. Matches the
// LoadedExecutableBuilder signature, so it can be passed directly to
// CompilationCache::AddUnboundedDynamicCacheEntry.
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
CreateDummyExecutable(xla::PjRtClient& client,
                      UniqueCompileOptions compile_options);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_COMPILATION_CACHE_TEST_UTILS_H_
