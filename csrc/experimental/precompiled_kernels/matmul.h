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

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_MATMUL_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_MATMUL_H_

#include <vector>

#include "csrc/common/unbounded_dynamic_kernel_registry.h"

namespace torch_tpu {

// Returns the Mosaic matmul kernels, for registration with
// `UnboundedDynamicKernelRegistry`. They are opt-in, so they only run when
// TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS is "all".
//
// They serve `torch.mm(a, b)` of F32 or BF16 matrices at every matmul
// precision. Other dtypes and `torch.mm(a, b, out_dtype=...)` stay on static
// kernels.
//
// Buffer contract: `a[M,K]`, `b[K,N]` and the result `[M,N]`, which `Execute`
// allocates and donates, all have the row-major layout {1,0:T(8,128)}
// ({1,0:T(8,128)(2,1)} for BF16), with M, K, N >= 1. XLA defaults to another
// layout if that pads less, e.g. to a column-major one for F32[1000,48].
std::vector<UnboundedDynamicKernelRegistration>
GetMatmulUnboundedDynamicKernels();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_MATMUL_H_
