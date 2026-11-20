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

// Registers the precompiled unbounded dynamic kernels with the registry when
// this library is loaded. Nothing references this file, so it is built with
// `alwayslink = True`; link it only into TPU-only targets.

#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "csrc/experimental/precompiled_kernels/matmul.h"
#include "csrc/experimental/precompiled_kernels/nonzero_indices.h"
#include "csrc/experimental/precompiled_kernels/sort/sort.h"

namespace torch_tpu {
namespace {

// Registers kernels with `UnboundedDynamicKernelRegistry` when constructed.
// Prefer `TT_REGISTER_DYNAMIC_KERNEL` or `TT_REGISTER_DYNAMIC_KERNELS` over
// instantiating this class directly.
class UnboundedDynamicKernelRegistrar {
 public:
  explicit UnboundedDynamicKernelRegistrar(
      UnboundedDynamicKernelRegistration registration) {
    Register(std::move(registration));
  }

  // Registers `registrations` in order.
  explicit UnboundedDynamicKernelRegistrar(
      std::vector<UnboundedDynamicKernelRegistration> registrations) {
    for (UnboundedDynamicKernelRegistration& registration : registrations) {
      Register(std::move(registration));
    }
  }

 private:
  static void Register(UnboundedDynamicKernelRegistration registration) {
    ABSL_CHECK_OK(  // CRASH_OK=static-init registration bug
        UnboundedDynamicKernelRegistry::GetInstance().RegisterKernel(
            std::move(registration)));
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
// `registration` is an `UnboundedDynamicKernelRegistration`: an
// `UnboundedDynamicKernel`, registered under its `key`, which graphs with that
// key's extentless key find, and `additional_search_keys`, the extentless keys
// of other graphs that can also find it. Each kernel defines its key next to
// its builder, e.g. with `MakeUnboundedDynamicKernelCacheKey`.
//
// The argument is evaluated during static initialization, so it must not
// depend on dynamically initialized globals from other translation units
// (flags, environment-derived settings, other registries, ...).
//
// Registering two kernels under the same key is a programming error and
// crashes at startup.
#define TT_REGISTER_DYNAMIC_KERNEL(registration)                        \
  const ::torch_tpu::UnboundedDynamicKernelRegistrar                    \
  TT_REGISTER_DYNAMIC_KERNEL_CONCAT_(kUnboundedDynamicKernelRegistrar_, \
                                     __COUNTER__)((registration))

// Registers a list of unbounded dynamic kernels, in order, like
// `TT_REGISTER_DYNAMIC_KERNEL`. `registrations` is a
// `std::vector<UnboundedDynamicKernelRegistration>`:
//
//   TT_REGISTER_DYNAMIC_KERNELS(GetNonzeroUnboundedDynamicKernels());
#define TT_REGISTER_DYNAMIC_KERNELS(registrations)                      \
  const ::torch_tpu::UnboundedDynamicKernelRegistrar                    \
  TT_REGISTER_DYNAMIC_KERNEL_CONCAT_(kUnboundedDynamicKernelRegistrar_, \
                                     __COUNTER__)((registrations))

TT_REGISTER_DYNAMIC_KERNELS(GetMatmulUnboundedDynamicKernels());
TT_REGISTER_DYNAMIC_KERNELS(GetNonzeroUnboundedDynamicKernels());
TT_REGISTER_DYNAMIC_KERNEL(GetSortUnboundedDynamicKernel());

}  // namespace
}  // namespace torch_tpu
