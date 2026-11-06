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

#include "csrc/ops/has_compatible_shallow_copy_type/has_compatible_shallow_copy_type_aten_kernels.h"

#include "ATen/core/ATen_fwd.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"

namespace torch_tpu {

bool AtenHasCompatibleShallowCopyType(const at::Tensor& self,
                                      const at::Tensor& from) {
  TT_KERNEL(OpName::kHasCompatibleShallowCopyType, _, (self, from), {
    // We permit shallow copies between dense tensors when moving from CPU
    // to TPU. This supports module parameter tying.
    auto is_dense_cpu_or_tpu = [](c10::DispatchKeySet keyset) {
      constexpr auto cpu_or_tpu =
          c10::DispatchKeySet({c10::BackendComponent::CPUBit,
                               c10::BackendComponent::PrivateUse1Bit});
      constexpr auto dense_k = c10::DispatchKeySet(c10::DispatchKey::Dense);
      return keyset.has_any(dense_k) && keyset.has_any(cpu_or_tpu);
    };
    return is_dense_cpu_or_tpu(self.key_set()) &&
           is_dense_cpu_or_tpu(from.key_set());
  });
}

}  // namespace torch_tpu
