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

#include "csrc/common/unbounded_dynamic_kernel_registry.h"

#include <memory>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"

namespace torch_tpu {

UnboundedDynamicKernelRegistry& UnboundedDynamicKernelRegistry::GetInstance() {
  static absl::NoDestructor<UnboundedDynamicKernelRegistry> instance;
  return *instance;
}

absl::Status UnboundedDynamicKernelRegistry::RegisterKernel(
    UnboundedDynamicKernel kernel) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=unreachable from Python.
      !kernels_.contains(kernel.key), error::kAlreadyExists)
      << "an unbounded dynamic kernel is already registered for key "
      << kernel.key;
  const ShapelessKey shapeless_key = kernel.key.graph_key().shapeless_key();
  auto owned = std::make_unique<UnboundedDynamicKernel>(std::move(kernel));
  shapeless_map_[shapeless_key].push_back(owned.get());
  const CompilationCacheKey key = owned->key;
  kernels_.emplace(key, std::move(owned));
  return absl::OkStatus();
}

const UnboundedDynamicKernel* absl_nullable
UnboundedDynamicKernelRegistry::FindUnboundedDynamicKernel(
    const ShapelessKey& key, absl::Span<const Shape> input_shapes,
    absl::Span<const Shape> output_shapes) const {
  const auto it = shapeless_map_.find(key);
  if (it == shapeless_map_.end()) {
    return nullptr;
  }
  for (const UnboundedDynamicKernel* kernel : it->second) {
    const UnboundedDynamicExecutionMetadata::CompatibilityFunc& is_compatible =
        kernel->metadata.is_compatible;
    if (!is_compatible || is_compatible(input_shapes, output_shapes)) {
      return kernel;
    }
  }
  return nullptr;
}

const UnboundedDynamicExecutionMetadata* absl_nullable
UnboundedDynamicKernelRegistry::GetUnboundedDynamicMetadata(
    const CompilationCacheKey& key) const {
  const auto it = kernels_.find(key);
  return it == kernels_.end() ? nullptr : &it->second->metadata;
}

}  // namespace torch_tpu
