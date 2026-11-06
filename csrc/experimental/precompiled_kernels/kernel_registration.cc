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

#include <utility>

#include "absl/log/absl_check.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "csrc/experimental/precompiled_kernels/nonzero_indices.h"

namespace torch_tpu {
namespace {

// Registers a kernel with `UnboundedDynamicKernelRegistry` when constructed.
// Prefer `TT_REGISTER_DYNAMIC_KERNEL` over instantiating this class directly.
class UnboundedDynamicKernelRegistrar {
 public:
  explicit UnboundedDynamicKernelRegistrar(UnboundedDynamicKernel kernel) {
    ABSL_CHECK_OK(  // CRASH_OK=static-init registration bug
        UnboundedDynamicKernelRegistry::GetInstance().RegisterKernel(
            std::move(kernel)));
  }
};

#define TT_REGISTER_DYNAMIC_KERNEL_CONCAT_INNER_(x, y) x##y
#define TT_REGISTER_DYNAMIC_KERNEL_CONCAT_(x, y) \
  TT_REGISTER_DYNAMIC_KERNEL_CONCAT_INNER_(x, y)

// Registers an unbounded dynamic kernel with `UnboundedDynamicKernelRegistry`
// at static initialization time. Invoke at namespace scope in a library built
// with `alwayslink = True`:
//
//   TT_REGISTER_DYNAMIC_KERNEL(GetNonzeroUnboundedDynamicKernel());
//
// `kernel` is an `UnboundedDynamicKernel`, registered under its `key`. Each
// kernel defines its key next to its builder, e.g. with
// `MakeUnboundedDynamicKernelCacheKey`.
//
// The argument is evaluated during static initialization, so it must not
// depend on dynamically initialized globals from other translation units
// (flags, environment-derived settings, other registries, ...).
//
// Registering two kernels under the same key is a programming error and
// crashes at startup.
#define TT_REGISTER_DYNAMIC_KERNEL(kernel)                              \
  const ::torch_tpu::UnboundedDynamicKernelRegistrar                    \
  TT_REGISTER_DYNAMIC_KERNEL_CONCAT_(kUnboundedDynamicKernelRegistrar_, \
                                     __COUNTER__)((kernel))

TT_REGISTER_DYNAMIC_KERNEL(GetNonzeroUnboundedDynamicKernel());

}  // namespace
}  // namespace torch_tpu
