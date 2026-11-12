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

// Registers the nonzero indices kernels with the unbounded dynamic kernel
// registry when this library is loaded. Nothing references this file, so it is
// built with `alwayslink = True`; link it only into TPU-only targets.

#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "csrc/experimental/precompiled_kernels/nonzero_indices.h"
#include "csrc/experimental/precompiled_kernels/sort/sort_kernels.h"

namespace torch_tpu {
namespace {

// Registers kernels with `UnboundedDynamicKernelRegistry` when constructed.
// Prefer `TT_REGISTER_DYNAMIC_KERNEL` or `TT_REGISTER_DYNAMIC_KERNELS` over
// instantiating this class directly.
class UnboundedDynamicKernelRegistrar {
 public:
  explicit UnboundedDynamicKernelRegistrar(UnboundedDynamicKernel kernel) {
    Register(std::move(kernel));
  }

  // Registers `kernels` in order.
  explicit UnboundedDynamicKernelRegistrar(
      std::vector<UnboundedDynamicKernel> kernels) {
    for (UnboundedDynamicKernel& kernel : kernels) {
      Register(std::move(kernel));
    }
  }

 private:
  static void Register(UnboundedDynamicKernel kernel) {
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
//   TT_REGISTER_DYNAMIC_KERNEL(GetFooUnboundedDynamicKernel());
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

// Registers a list of unbounded dynamic kernels, in order, like
// `TT_REGISTER_DYNAMIC_KERNEL`. `kernels` is a
// `std::vector<UnboundedDynamicKernel>`:
//
//   TT_REGISTER_DYNAMIC_KERNELS(GetNonzeroUnboundedDynamicKernels());
#define TT_REGISTER_DYNAMIC_KERNELS(kernels)                            \
  const ::torch_tpu::UnboundedDynamicKernelRegistrar                    \
  TT_REGISTER_DYNAMIC_KERNEL_CONCAT_(kUnboundedDynamicKernelRegistrar_, \
                                     __COUNTER__)((kernels))

TT_REGISTER_DYNAMIC_KERNELS(GetNonzeroUnboundedDynamicKernels());
TT_REGISTER_DYNAMIC_KERNELS(GetSortUnboundedDynamicKernels());

}  // namespace
}  // namespace torch_tpu
