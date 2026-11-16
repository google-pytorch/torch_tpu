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

#include "csrc/internal/mosaic/kiln/core/mlir_utils.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "csrc/internal/mosaic/kiln/core/specs.h"
#include "csrc/internal/mosaic/op_builders.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace torch_tpu::kiln::internal {
namespace {

enum class Signedness {
  kSigned,
  kUnsigned,
};

void ValidateConstantBound(const llvm::APInt& val, Signedness signedness,
                           IndexBounds bounds, std::string_view name) {
  const bool is_unsigned = signedness == Signedness::kUnsigned;
  if (bounds == IndexBounds::kPositive) {
    const bool is_positive =
        is_unsigned ? !val.isZero() : val.isStrictlyPositive();
    ABSL_CHECK(is_positive) << name << " must be positive";  // CRASH_OK
  } else {
    const bool is_non_negative = is_unsigned || !val.isNegative();
    ABSL_CHECK(is_non_negative) << name << " must be non-negative";  // CRASH_OK
  }
  const bool fits_i32 = is_unsigned
                            ? val.ule(std::numeric_limits<int32_t>::max())
                            : val.sle(std::numeric_limits<int32_t>::max());
  ABSL_CHECK(fits_i32) << name << " exceeds int32_t max";  // CRASH_OK
}

}  // namespace

mlir::VectorType ToMlirVectorType(const Shape& shape) {
  return mlir::VectorType::get(shape.dims(), ToMlirType(shape.dtype()));
}

mlir::MemRefType ToMlirMemRefType(const Shape& shape, MemorySpace space,
                                  mlir::MLIRContext* absl_nonnull ctx) {
  mlir::Type elem_type = ToMlirType(shape.dtype(), ctx);
  switch (space) {
    case MemorySpace::kVmem:
      return GetVmemMemRefType(ctx, shape.dims(), elem_type);
    case MemorySpace::kSmem:
      return GetSmemMemRefType(ctx, shape.dims(), elem_type);
    case MemorySpace::kHbm:
      return GetHbmMemRefType(ctx, shape.dims(), elem_type);
  }
}

void ValidateI32IndexArg(const IndexArg& arg, IndexBounds bounds,
                         std::string_view name) {
  if (arg.IsInt()) {
    ValidateConstantBound(llvm::APInt(64, arg.AsInt(), /*isSigned=*/true),
                          Signedness::kSigned, bounds, name);
  } else {
    Scalar s = arg.ToScalar();
    ABSL_CHECK_LE(s.dtype().GetIntOrFloatBitWidth(), 32u)  // CRASH_OK
        << name << " scalar bitwidth must be <= 32";
    if (auto const_op = unwrap(s).getDefiningOp<mlir::arith::ConstantOp>()) {
      if (auto attr = llvm::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
        const Signedness signedness = s.dtype().IsUnsigned()
                                          ? Signedness::kUnsigned
                                          : Signedness::kSigned;
        ValidateConstantBound(attr.getValue(), signedness, bounds, name);
      }
    }
  }
}

mlir::Value ToValidatedI32Value(const IndexArg& arg, IndexBounds bounds,
                                std::string_view name) {
  ValidateI32IndexArg(arg, bounds, name);
  return unwrap(arg.ToScalar().ToI32());
}

std::optional<Scalar> OptionalI32Scalar(const std::optional<IndexArg>& arg,
                                        std::string_view name) {
  if (!arg.has_value()) {
    return std::nullopt;
  }
  ValidateI32IndexArg(*arg, IndexBounds::kNonNegative, name);
  return arg->ToScalar().ToI32();
}

mlir::Value UnwrapOptionalScalar(const std::optional<Scalar>& s) {
  return s.has_value() ? unwrap(*s) : nullptr;
}

mlir::Value OptionalI32Value(const std::optional<IndexArg>& arg,
                             std::string_view name) {
  return UnwrapOptionalScalar(OptionalI32Scalar(arg, name));
}

mlir::Value ToIndex(mlir::ImplicitLocOpBuilder& b, const IndexArg& arg) {
  if (arg.IsInt()) {
    return mlir::arith::ConstantOp::create(b, b.getIndexAttr(arg.AsInt()));
  }
  const Scalar s = arg.ToScalar();
  if (s.dtype().IsUnsigned()) {
    return mlir::arith::IndexCastUIOp::create(b, b.getIndexType(), unwrap(s));
  }
  return mlir::arith::IndexCastOp::create(b, b.getIndexType(), unwrap(s));
}

}  // namespace torch_tpu::kiln::internal
