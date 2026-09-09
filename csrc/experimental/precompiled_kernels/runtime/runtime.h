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

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_RUNTIME_RUNTIME_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_RUNTIME_RUNTIME_H_

#include <optional>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/compilation.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/traversal.h"
#include "csrc/experimental/precompiled_kernels/bundle.pb.h"
#include "csrc/pjrt/pjrt_utils.h"

namespace torch_tpu {

// The immutable bundle has process lifetime; entries also key the load cache.
class Catalog {
 public:
  explicit Catalog(const precompiled_kernels::proto::Bundle& bundle);

  // Finds all candidate kernels matching the given operation name.
  absl::Span<const precompiled_kernels::proto::Kernel* const> Find(
      std::string_view op) const;

  // Returns the underlying bundle reference.
  const precompiled_kernels::proto::Bundle& bundle() const { return bundle_; }

 private:
  const precompiled_kernels::proto::Bundle& bundle_;
  absl::flat_hash_map<std::string_view,
                      std::vector<const precompiled_kernels::proto::Kernel*>>
      entries_;
};

// A matched execution may decline before submission (returning nullopt). The
// signature uses direct eager types; the runtime does not depend on the
// optional registration hook.
using Execution =
    absl::AnyInvocable<absl::StatusOr<std::optional<PjRtBufferPointers>>(
        const CompiledKernel&, absl::Span<const DeviceBufferRef>) &&>;

// Inspects graph semantics and matches candidate kernels without loading code
// or resolving input buffers.
Execution MatchPrecompiledKernel(const Traversal& traversal,
                                 const Catalog& catalog);

// Clears cached executables and compilation failures.
void ClearPrecompiledKernelCache();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_RUNTIME_RUNTIME_H_
