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

#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/literal.h"
#include "csrc/internal/mosaic/kiln/core/value_base.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace torch_tpu::kiln::internal {

// RAII helper to install and restore the active thread-local builder.
class BuilderScope {
 public:
  explicit BuilderScope(mlir::ImplicitLocOpBuilder* builder);
  ~BuilderScope();

  BuilderScope(const BuilderScope&) = delete;
  BuilderScope& operator=(const BuilderScope&) = delete;

 private:
  mlir::ImplicitLocOpBuilder* prev_;
};

// Returns the active MLIR ImplicitLocOpBuilder for the current thread.
// Must be called within an active BuildKernel() or IndexMap execution.
mlir::ImplicitLocOpBuilder& CurrentBuilder();

struct ValueAccess {
  static mlir::Value Get(const ValueBase& v) {
    return mlir::Value::getFromOpaquePointer(v.impl());
  }
  template <typename T>
  static T Make(mlir::Value v) {
    return T(v.getAsOpaquePointer());
  }
};

inline mlir::Value unwrap(const ValueBase& v) { return ValueAccess::Get(v); }

template <typename T>
inline T wrap(mlir::Value v) {
  return ValueAccess::Make<T>(v);
}

mlir::Type ToMlirType(DType dtype, mlir::MLIRContext* ctx);
mlir::Type ToMlirType(DType dtype);
DType FromMlirType(mlir::Type type);

mlir::TypedAttr LiteralToAttr(mlir::ImplicitLocOpBuilder& b, DType dtype,
                              Literal literal);

mlir::Value EmitCast(mlir::Value value, DType source, DType target);
mlir::Value EmitBitcast(mlir::Value value, DType source, DType target);

}  // namespace torch_tpu::kiln::internal

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_CONTEXT_H_
