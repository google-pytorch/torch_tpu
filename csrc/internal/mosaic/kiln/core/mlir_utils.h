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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_MLIR_UTILS_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_MLIR_UTILS_H_

#include <optional>
#include <string_view>

#include "absl/base/nullability.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "csrc/internal/mosaic/kiln/core/specs.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

namespace torch_tpu::kiln::internal {

template <typename T>
inline mlir::MemRefType MemRefTypeOf(const T& val) {
  return llvm::cast<mlir::MemRefType>(unwrap(val).getType());
}

// Get the mlir::VectorType corresponding to the given Shape.
mlir::VectorType ToMlirVectorType(const Shape& shape);

// Get the mlir::MemRefType corresponding to the given Shape and MemorySpace.
mlir::MemRefType ToMlirMemRefType(const Shape& shape, MemorySpace space,
                                  mlir::MLIRContext* absl_nonnull ctx);

enum class IndexBounds {
  kNonNegative,
  kPositive,
};

// Validates that `arg` satisfies `bounds` when it is an integer literal or
// constant Scalar, and that any Scalar argument has bitwidth <= 32.
void ValidateI32IndexArg(const IndexArg& arg, IndexBounds bounds,
                         std::string_view name);

// Validates `arg` against `bounds` and converts it to an i32 mlir::Value.
mlir::Value ToValidatedI32Value(const IndexArg& arg, IndexBounds bounds,
                                std::string_view name);

// Converts an optional I32 IndexArg to an optional Scalar, validating the
// IndexArg in the process so literals are not rematerialized on each use.
std::optional<Scalar> OptionalI32Scalar(const std::optional<IndexArg>& arg,
                                        std::string_view name);

// Unwraps an optional Scalar to its underlying mlir::Value, or returns nullptr
// if it is std::nullopt.
mlir::Value UnwrapOptionalScalar(const std::optional<Scalar>& s);

mlir::Value OptionalI32Value(const std::optional<IndexArg>& arg,
                             std::string_view name);

}  // namespace torch_tpu::kiln::internal

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_MLIR_UTILS_H_
