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

#ifndef TORCH_TPU_CSRC_COMMON_UNBOUNDED_DYNAMIC_KERNEL_REGISTRY_H_
#define TORCH_TPU_CSRC_COMMON_UNBOUNDED_DYNAMIC_KERNEL_REGISTRY_H_

#include <memory>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/shape.h"
#include "csrc/ops/op_names.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {

// An unbounded-dynamic kernel that the compilation cache can load instead of
// compiling a traced computation.
struct UnboundedDynamicKernel {
  using BuildFn =
      absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> (*)(
          xla::PjRtClient& client, UniqueCompileOptions compile_options);

  BuildFn absl_nonnull build;
  UnboundedDynamicExecutionMetadata metadata;
  // Identifies the kernel, and its executable in the compilation cache.
  CompilationCacheKey key;
  // The op of the single-op graphs that the kernel serves, e.g. `kNonzero`,
  // through `key` and any additional search keys. Tests use it to find the ops
  // that have kernels.
  OpName op_name;
  // Whether a cache miss may load the kernel when
  // TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS is "default" (or unset).
  // Opt-in kernels (false) are only loaded when it is "all".
  bool enabled_by_default = false;
};

// A kernel to register. Graphs with the extentless key of `kernel.key` can find
// it, as can graphs with any of the extentless keys in
// `additional_search_keys`.
struct UnboundedDynamicKernelRegistration {
  UnboundedDynamicKernel kernel;
  std::vector<ExtentlessKey> additional_search_keys;
};

// Process-wide registry of unbounded dynamic kernels, grouped by extentless
// key. Several kernels may share an extentless key, e.g. with different
// compatibility predicates; a lookup returns the first compatible one in
// registration order.
class UnboundedDynamicKernelRegistry {
 public:
  static UnboundedDynamicKernelRegistry& GetInstance();

  UnboundedDynamicKernelRegistry(const UnboundedDynamicKernelRegistry&) =
      delete;
  UnboundedDynamicKernelRegistry& operator=(
      const UnboundedDynamicKernelRegistry&) = delete;

  // Registers `registration.kernel` under its `key`, that key's extentless key
  // and each of `registration.additional_search_keys`, so that graphs with any
  // of them can find it and share its executable. Must only be called during
  // static initialization; it is not synchronized with lookups. Returns
  // kAlreadyExists if a kernel is already registered under
  // `registration.kernel.key`.
  absl::Status RegisterKernel(UnboundedDynamicKernelRegistration registration);

  // Returns the first kernel registered under `key`, in registration order,
  // whose compatibility predicate accepts the given logical input and output
  // shapes, or null if there is none.
  [[nodiscard]] const UnboundedDynamicKernel* absl_nullable
  FindUnboundedDynamicKernel(const ExtentlessKey& key,
                             absl::Span<const Shape> input_shapes,
                             absl::Span<const Shape> output_shapes) const;

  // Returns the `op_name`s of the registered kernels, whether or not
  // TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS enables them.
  [[nodiscard]] absl::btree_set<OpName> GetOpNames() const;

 private:
  // Owns the kernels, indexed by their exact cache key.
  using KernelMap = absl::flat_hash_map<CompilationCacheKey,
                                        std::unique_ptr<UnboundedDynamicKernel>,
                                        CompilationCacheKey::Hash>;
  // Non-owning index from extentless key to the kernels registered under it, in
  // registration order. Pointers point into `KernelMap`'s heap-allocated
  // values, which are never freed or moved (the registry is append-only).
  using KeysMap =
      absl::flat_hash_map<ExtentlessKey,
                          std::vector<const UnboundedDynamicKernel*>,
                          ExtentlessKey::Hash>;

  friend class absl::NoDestructor<UnboundedDynamicKernelRegistry>;
  UnboundedDynamicKernelRegistry() = default;

  KernelMap kernels_;
  KeysMap extentless_map_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_UNBOUNDED_DYNAMIC_KERNEL_REGISTRY_H_
