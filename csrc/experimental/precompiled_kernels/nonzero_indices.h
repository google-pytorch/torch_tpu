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

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_NONZERO_INDICES_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_NONZERO_INDICES_H_

#include "csrc/common/unbounded_dynamic_kernel_registry.h"

namespace torch_tpu {

// Returns the Mosaic nonzero indices kernel, keyed for registration with
// `UnboundedDynamicKernelRegistry` under the shapeless key of a 2-D F32 nonzero
// graph. Run through `Execute` with the F32[M,N] input and the logical output
// shape S64[K,2], its executable returns the (row, column) coordinates of the
// input's nonzero elements, ordered by row, then column. The predicate
// excludes only +0.0 and -0.0, matching torch.nonzero.
//
// Buffer contract (all payloads have no dynamic-metadata prefix):
// - Input: F32[M,N]{1,0:T(8,128)}, the only buffer the caller supplies.
// - Output: S64[K,2]{0,1:T(2,128)}. `Execute` allocates the destination itself
//   as the S32[R,128]{1,0:T(1,128)} view the Mosaic kernel was compiled against
//   (R = 4 * ceil(K/128); rows [0, R/2) hold interleaved row/col low words,
//   [R/2, R) hold zeroed high words), donates it, and bitcasts the result back
//   to S64[K,2]. The detour exists only because Mosaic does not support S64
//   memrefs; callers never see it.
//   NB: requires K > 2 because the XLA default layout for S64[K,2] switches to
//   {1,0:T(2,128)} when K <= 2.
//
// Caller-checked arithmetic bounds: 1 <= M <= INT32_MAX,
// 1 <= N <= INT32_MAX - 127, and 3 <= K <= min(M*N, INT32_MAX).
UnboundedDynamicKernel GetNonzeroUnboundedDynamicKernel();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_NONZERO_INDICES_H_
