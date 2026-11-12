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

#include "csrc/internal/mosaic/kiln/core/context.h"

#include <string_view>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "csrc/internal/mosaic/kiln/core/builder_scope.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/literal.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "csrc/internal/mosaic/kiln/core/value_handle.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"

namespace torch_tpu::kiln::internal {
namespace {

static_assert(kDynamic == mlir::ShapedType::kDynamic,
              "kDynamic must match mlir::ShapedType::kDynamic");

mlir::Type WithElementType(mlir::Type type, DType target) {
  mlir::Type elem = ToMlirType(target, type.getContext());
  if (auto vec = llvm::dyn_cast<mlir::VectorType>(type)) {
    return vec.clone(elem);
  }
  return elem;
}

mlir::Value EmitIntCast(mlir::ImplicitLocOpBuilder& b, mlir::Value value,
                        DType source, DType target, mlir::Type dst_mlir_type) {
  const unsigned src_bits = source.GetIntOrFloatBitWidth();
  const unsigned dst_bits = target.GetIntOrFloatBitWidth();
  if (src_bits < dst_bits) {
    if (source.IsSigned()) {
      return mlir::arith::ExtSIOp::create(b, dst_mlir_type, value);
    }
    return mlir::arith::ExtUIOp::create(b, dst_mlir_type, value);
  }
  if (target.IsBool()) {
    return mlir::arith::CmpIOp::create(
        b, mlir::arith::CmpIPredicate::ne, value,
        mlir::arith::ConstantOp::create(b, b.getZeroAttr(value.getType())));
  }
  if (src_bits == dst_bits) {
    return value;
  }
  return mlir::arith::TruncIOp::create(b, dst_mlir_type, value);
}

}  // namespace

mlir::ImplicitLocOpBuilder& CurrentBuilder() {
  return CurrentBuilderScope().builder();
}

mlir::Type ToMlirType(DType dtype, mlir::MLIRContext* ctx) {
  switch (dtype.kind()) {
    case DType::Kind::kF16:
      return mlir::Float16Type::get(ctx);
    case DType::Kind::kBF16:
      return mlir::BFloat16Type::get(ctx);
    case DType::Kind::kF32:
      return mlir::Float32Type::get(ctx);
    default:
      ABSL_CHECK(dtype.IsInt() || dtype.IsBool())  // CRASH_OK
          << dtype << " is not a bool, int, or known float type";
      return mlir::IntegerType::get(ctx, dtype.GetIntOrFloatBitWidth());
  }
}

mlir::Type ToMlirType(DType dtype) {
  return ToMlirType(dtype, CurrentBuilder().getContext());
}

void CheckElementType(ValueHandle handle, DType dtype,
                      std::string_view type_name) {
  const mlir::Type element_type = mlir::getElementTypeOrSelf(handle.value());
  ABSL_CHECK(  // CRASH_OK
      element_type == ToMlirType(dtype, element_type.getContext()))
      << type_name << " dtype " << dtype
      << " does not match its IR element type "
      << mlir::debugString(element_type);
}

mlir::Value ValueOf(ValueHandle handle) {
  if (!handle.is_null()) {
    return handle.value();
  }
  const absl::Status& error = CurrentBuilderScope().error();
  if (error.ok()) {
    ABSL_LOG(FATAL)  // CRASH_OK=a failed operation has no value to give
        << "The value of a failed Kiln operation was used, with no error "
           "recorded on the active scope";
  }
  ABSL_LOG(FATAL)  // CRASH_OK=a failed operation has no value to give
      << "The value of a failed Kiln operation was used; the failure was: "
      << error;
}

mlir::TypedAttr LiteralToAttr(mlir::ImplicitLocOpBuilder& b, DType dtype,
                              Literal literal) {
  mlir::Type mlir_type = ToMlirType(dtype, b.getContext());
  if (dtype.IsFloat()) {
    return b.getFloatAttr(mlir_type, literal.AsDouble());
  }
  // Integer or boolean.
  return b.getIntegerAttr(mlir_type, literal.AsInt());
}

mlir::Value EmitCast(mlir::Value value, DType source, DType target) {
  if (source == target) {
    return value;
  }

  mlir::ImplicitLocOpBuilder& b = CurrentBuilder();
  mlir::Type dst_mlir_type = WithElementType(value.getType(), target);
  if (source.IsFloat() && target.IsFloat()) {
    const unsigned src_bits = source.GetIntOrFloatBitWidth();
    const unsigned dst_bits = target.GetIntOrFloatBitWidth();
    ABSL_CHECK_NE(src_bits, dst_bits) << "Unsupported float cast";  // CRASH_OK
    if (src_bits < dst_bits) {
      return mlir::arith::ExtFOp::create(b, dst_mlir_type, value);
    }
    return mlir::arith::TruncFOp::create(b, dst_mlir_type, value);
  }
  if (source.IsFloat()) {
    if (target.IsBool()) {
      return mlir::arith::CmpFOp::create(
          b, mlir::arith::CmpFPredicate::UNE, value,
          mlir::arith::ConstantOp::create(b, b.getZeroAttr(value.getType())));
    }
    if (target.IsUnsigned()) {
      return mlir::arith::FPToUIOp::create(b, dst_mlir_type, value);
    }
    return mlir::arith::FPToSIOp::create(b, dst_mlir_type, value);
  }
  if (target.IsFloat()) {
    if (source.IsSigned()) {
      return mlir::arith::SIToFPOp::create(b, dst_mlir_type, value);
    }
    return mlir::arith::UIToFPOp::create(b, dst_mlir_type, value);
  }
  return EmitIntCast(b, value, source, target, dst_mlir_type);
}

mlir::Value EmitBitcast(mlir::Value value, DType source, DType target) {
  ABSL_CHECK(source.IsIntOrFloat() && target.IsIntOrFloat())  // CRASH_OK
      << "Bitcast() requires integer or floating-point types";
  ABSL_CHECK_EQ(source.GetIntOrFloatBitWidth(),  // CRASH_OK
                target.GetIntOrFloatBitWidth())
      << "Bitcast() requires source and target types to have the same bit "
         "width";
  mlir::ImplicitLocOpBuilder& b = CurrentBuilder();
  mlir::Type dst_mlir_type = WithElementType(value.getType(), target);
  if (dst_mlir_type == value.getType()) {
    return value;
  }
  return mlir::arith::BitcastOp::create(b, dst_mlir_type, value);
}

}  // namespace torch_tpu::kiln::internal
