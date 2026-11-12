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

#include "csrc/internal/mosaic/kiln/core/builder_scope.h"

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"

namespace torch_tpu::kiln::internal {
namespace {

// The active scope of the current thread, or null outside of BuildKernel().
thread_local BuilderScope*  // CPP_THREAD_LOCAL_OK=C++ scope
    tls_builder_scope = nullptr;

}  // namespace

BuilderScope::BuilderScope(mlir::ImplicitLocOpBuilder* absl_nonnull builder)
    : builder_(builder) {
  ABSL_CHECK(builder != nullptr)  // CRASH_OK
      << "BuilderScope requires a non-null builder";
  ABSL_CHECK(tls_builder_scope == nullptr)  // CRASH_OK
      << "BuilderScope cannot be nested";
  tls_builder_scope = this;
}

BuilderScope::~BuilderScope() { tls_builder_scope = nullptr; }

BuilderScope& CurrentBuilderScope() {
  ABSL_CHECK(tls_builder_scope != nullptr)  // CRASH_OK
      << "Kiln operations may only be called inside BuildKernel()";
  return *tls_builder_scope;
}

}  // namespace torch_tpu::kiln::internal
