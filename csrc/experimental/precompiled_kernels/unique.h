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

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_UNIQUE_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_UNIQUE_H_

#include <vector>

#include "csrc/common/unbounded_dynamic_kernel_registry.h"

namespace torch_tpu {

// Returns the unbounded dynamic Kiln kernels for
// `OpName::kUniqueValuesFromSorted` graphs with int32 inputs, which compact
// the sorted input into its unique values. There is one kernel for each of
// XLA's default 1-D input tiles B_in in {1024, 512, 256, 128}.
//
// Buffer contract:
// - Input: S32[N]{0:T(B_in)}, sorted in ascending order, 1 <= N <= INT32_MAX -
//   1024.
// - Output: S32[U]{0:T(B_out)} with XLA's default 1-D layout for U, 1 <= U <=
//   N. It is executed via a donated S32[R,128]{1,0:T(1,128)} view, R =
//   ceil(U/B_out) * (B_out/128), and bitcast back to S32[U].
[[nodiscard]] std::vector<UnboundedDynamicKernelRegistration>
GetUniqueUnboundedDynamicKernels();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_UNIQUE_H_
