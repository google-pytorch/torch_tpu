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

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_SORT_SORT_KERNELS_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_SORT_SORT_KERNELS_H_

#include <vector>

#include "csrc/common/unbounded_dynamic_kernel_registry.h"

namespace torch_tpu {

// Returns the Mosaic sort kernels, for registration with
// `UnboundedDynamicKernelRegistry`. They are opt-in, so they only run when
// TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS is "all".
//
// For now they serve `torch.sort(x)` and `torch.sort(x, stable=True)` of a
// 1-D int32 tensor of exactly 1024 elements: ascending, along dim -1. The sort
// is stable in both cases.
std::vector<UnboundedDynamicKernel> GetSortUnboundedDynamicKernels();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_SORT_SORT_KERNELS_H_
