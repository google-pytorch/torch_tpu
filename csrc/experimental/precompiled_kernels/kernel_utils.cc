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

#include "csrc/experimental/precompiled_kernels/kernel_utils.h"

#include "csrc/common/cache_key.h"
#include "csrc/common/compile_options_key.h"

namespace torch_tpu {

CompilationCacheKey MakeUnboundedDynamicKernelCacheKey(ShapelessKey key,
                                                       int id_key) {
  return CompilationCacheKey(GraphKey(key, DimensionsKey({})),
                             CompileOptionsKey({}), id_key);
}

}  // namespace torch_tpu
