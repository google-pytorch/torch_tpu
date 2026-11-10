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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_CONTEXT_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_CONTEXT_H_

#include <string_view>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/literal.h"
#include "csrc/internal/mosaic/kiln/core/value_handle.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

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
  friend void ReportError(absl::Status error);

  void RecordError(absl::Status error) { error_.Update(std::move(error)); }

  mlir::ImplicitLocOpBuilder* absl_nonnull builder_;
  absl::Status error_;
};

// Returns the active MLIR ImplicitLocOpBuilder for the current thread.
// Must be called within an active BuildKernel() or IndexMap execution.
mlir::ImplicitLocOpBuilder& CurrentBuilder();

// Reports `error` to the active scope of the current thread, which holds the
// first error reported to it; later errors are ignored, and `error` must not
// be OK. `BuildKernel()` returns the held error. Kiln handle types signal an
// error with a null `ValueHandle`; the error itself is reported here.
// Must be called within an active BuildKernel() or IndexMap execution.
void ReportError(absl::Status error);

mlir::Type ToMlirType(DType dtype, mlir::MLIRContext* ctx);
mlir::Type ToMlirType(DType dtype);

// CHECK-fails unless the element type of `handle` (or its type, if it isn't
// shaped) is the IR type of `dtype`. `type_name` names the Kiln type being
// constructed (e.g. "Scalar") in the failure message.
void CheckElementType(ValueHandle handle, DType dtype,
                      std::string_view type_name);

template <typename T>
inline mlir::Value unwrap(const T& value) {
  return value.handle().value();
}

template <typename T, typename... Args>
T wrap(mlir::Value value, Args&&... args) {
  return T(ValueHandle(value), std::forward<Args>(args)...);
}

mlir::TypedAttr LiteralToAttr(mlir::ImplicitLocOpBuilder& b, DType dtype,
                              Literal literal);

mlir::Value EmitCast(mlir::Value value, DType source, DType target);
mlir::Value EmitBitcast(mlir::Value value, DType source, DType target);

}  // namespace torch_tpu::kiln::internal

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_CONTEXT_H_
