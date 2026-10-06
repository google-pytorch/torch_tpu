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

#include "csrc/experimental/precompiled_kernels/precompiled_kernels.h"

#include "csrc/common/compilation_cache.h"
#include "csrc/common/error_utils.h"
#include "csrc/experimental/precompiled_kernels/nonzero_indices.h"

namespace torch_tpu {

void LoadPrecompiledKernels() {
  TT_THROW_IF_ERROR(
      CompilationCache::GetInstance().AddUnboundedDynamicCacheEntry(
          NonZeroCacheKey(), BuildNonzeroIndicesExecutable,
          MakeNonzeroDynamicMetadata()));
}

}  // namespace torch_tpu
