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

// Registers the nonzero indices kernel with the unbounded dynamic kernel
// registry when this library is loaded. Nothing references this file, so it is
// built with `alwayslink = True`; link it only into TPU-only targets.

#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "csrc/experimental/precompiled_kernels/nonzero_indices.h"

namespace torch_tpu {
namespace {

const UnboundedDynamicKernelRegistrar kNonzeroIndicesRegistrar(
    NonZeroCacheKey(), BuildNonzeroIndicesExecutable,
    MakeNonzeroDynamicMetadata());

}  // namespace
}  // namespace torch_tpu
