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

#include "csrc/internal/mosaic/kiln/core/elementwise_impl.h"

#include <algorithm>
#include <cstdint>

#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"

namespace torch_tpu::kiln::internal {
namespace {

// The shape of a scalar or vector value; scalars are rank 0.
llvm::ArrayRef<int64_t> ShapeOf(mlir::Value value) {
  if (auto vec = llvm::dyn_cast<mlir::VectorType>(value.getType())) {
    return vec.getShape();
  }
  return {};
}

// numpy broadcasting rule over two static shapes; CHECK-fails if the shapes
// are not broadcastable.
Dimensions BroadcastShapes(llvm::ArrayRef<int64_t> lhs,
                           llvm::ArrayRef<int64_t> rhs) {
  const int64_t rank = std::max(lhs.size(), rhs.size());
  Dimensions result(rank, 1);
  for (int64_t i = 0; i < rank; ++i) {
    const int64_t l = i < rank - static_cast<int64_t>(lhs.size())
                          ? 1
                          : lhs[i - (rank - lhs.size())];
    const int64_t r = i < rank - static_cast<int64_t>(rhs.size())
                          ? 1
                          : rhs[i - (rank - rhs.size())];
    ABSL_CHECK(l == r || l == 1 || r == 1)  // CRASH_OK
        << "shapes are not broadcastable";
    result[i] = std::max(l, r);
  }
  return result;
}

}  // namespace

mlir::Value EmitBroadcast(mlir::Value value, llvm::ArrayRef<int64_t> shape) {
  if (llvm::isa<mlir::VectorType>(value.getType()) && ShapeOf(value) == shape) {
    return value;
  }
  ABSL_CHECK_EQ(BroadcastShapes(ShapeOf(value), shape), shape)  // CRASH_OK
      << "Cannot broadcast to target shape";
  return mlir::vector::BroadcastOp::create(
      CurrentBuilder(),
      mlir::VectorType::get(shape, mlir::getElementTypeOrSelf(value)), value);
}

void AlignShapes(llvm::MutableArrayRef<mlir::Value> values) {
  if (llvm::none_of(values, [](mlir::Value v) {
        return llvm::isa<mlir::VectorType>(v.getType());
      })) {
    return;
  }
  Dimensions shape;
  for (mlir::Value v : values) {
    shape = BroadcastShapes(shape, ShapeOf(v));
  }
  for (mlir::Value& v : values) {
    v = EmitBroadcast(v, shape);
  }
}

}  // namespace torch_tpu::kiln::internal
