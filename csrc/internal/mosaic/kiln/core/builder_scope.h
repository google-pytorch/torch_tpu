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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_BUILDER_SCOPE_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_BUILDER_SCOPE_H_

#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"

// Forward declared rather than included so that this target does not depend on
// MLIR.
namespace mlir {  // NON_TORCH_TPU_OK=ImplicitLocOpBuilder forward declaration
class ImplicitLocOpBuilder;
}  // namespace mlir

namespace torch_tpu::kiln::internal {

// RAII helper to install and clear the active thread-local scope, which holds
// the builder and the first error reported while the scope is active.
class BuilderScope {
 public:
  explicit BuilderScope(mlir::ImplicitLocOpBuilder* absl_nonnull builder);
  ~BuilderScope();

  BuilderScope(const BuilderScope&) = delete;
  BuilderScope& operator=(const BuilderScope&) = delete;

  mlir::ImplicitLocOpBuilder& builder() const { return *builder_; }

  // The first error reported with `ReportError()` while this was the active
  // scope, or OK if there was none.
  const absl::Status& error() const { return error_; }

 private:
  // Declared in error.h; the only way to record an error on a scope.
  friend void ReportError(absl::Status error);

  void RecordError(absl::Status error) { error_.Update(std::move(error)); }

  mlir::ImplicitLocOpBuilder* absl_nonnull builder_;
  absl::Status error_;
};

// Returns the active scope of the current thread.
// Must be called within an active BuildKernel() or IndexMap execution.
BuilderScope& CurrentBuilderScope();

}  // namespace torch_tpu::kiln::internal

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_BUILDER_SCOPE_H_
