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
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/shape.h"

namespace torch_tpu {

UnboundedDynamicKernelRegistry& UnboundedDynamicKernelRegistry::GetInstance() {
  static absl::NoDestructor<UnboundedDynamicKernelRegistry> instance;
  return *instance;
}

void UnboundedDynamicKernelRegistry::RegisterKernel(
    UnboundedDynamicKernel kernel) {
  std::vector<std::unique_ptr<UnboundedDynamicKernel>>& kernels =
      kernels_[kernel.key.graph_key().shapeless_key()];
  kernels.push_back(
      std::make_unique<UnboundedDynamicKernel>(std::move(kernel)));
}

const UnboundedDynamicKernel* absl_nullable
UnboundedDynamicKernelRegistry::FindUnboundedDynamicKernel(
    const ShapelessKey& key, absl::Span<const Shape> input_shapes,
    absl::Span<const Shape> output_shapes) const {
  const auto it = kernels_.find(key);
  if (it == kernels_.end()) {
    return nullptr;
  }
  for (const std::unique_ptr<UnboundedDynamicKernel>& kernel : it->second) {
    const UnboundedDynamicExecutionMetadata::CompatibilityFunc is_compatible =
        kernel->metadata.is_compatible;
    if (!is_compatible || is_compatible(input_shapes, output_shapes)) {
      return kernel.get();
    }
  }
  return nullptr;
}

}  // namespace torch_tpu
