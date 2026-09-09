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

#include <cstdlib>
#include <string_view>

#include "absl/base/no_destructor.h"
#include "csrc/eager/materialize_common.h"
#include "csrc/eager/traversal.h"
#include "csrc/experimental/precompiled_kernels/bundle.pb.h"
#include "csrc/experimental/precompiled_kernels/runtime/embedded_bundle.h"
#include "csrc/experimental/precompiled_kernels/runtime/runtime.h"

namespace torch_tpu {
namespace {
PrecompiledExecution Match(const Traversal& traversal) {
  static const absl::NoDestructor<precompiled_kernels::proto::Bundle> bundle(
      [] {
        precompiled_kernels::proto::Bundle result;
        const auto& data = *embedded_bundle_create();
        if (!result.ParseFromArray(data.data, data.size)) result.Clear();
        return result;
      }());
  static const absl::NoDestructor<Catalog> catalog(*bundle);
  return MatchPrecompiledKernel(traversal, *catalog);
}

// alwayslink retains this initializer only when the build flag links the
// plugin.
[[maybe_unused]] const bool registered = [] {
  const char* enabled =
      std::getenv("TORCH_TPU_EXPERIMENTAL_PRECOMPILED");  // GETENV_OK=plugin.
  if (enabled && std::string_view(enabled) == "1")
    RegisterPrecompiledExecution(Match, ClearPrecompiledKernelCache);
  return true;
}();
}  // namespace
}  // namespace torch_tpu
