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

#include "csrc/internal/mosaic/kiln/core/scalar.h"

#include <numbers>

#include "absl/log/absl_check.h"
#include "csrc/common/libtpu_version.h"
#include "csrc/common/version.h"
#include "csrc/internal/mosaic/kiln/core/cond.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/literal.h"
#include "csrc/internal/mosaic/kiln/core/value_base.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace torch_tpu::kiln {
namespace {

namespace arith = mlir::arith;
namespace math = mlir::math;

template <typename FloatOp, typename IntOp>
Scalar ScalarArith(Scalar lhs, Scalar rhs) {
  ABSL_CHECK_EQ(lhs.dtype(), rhs.dtype())  // CRASH_OK
      << "Scalar arithmetic requires operands of the same type";
  ABSL_CHECK(lhs.IsIntOrFloat())  // CRASH_OK
      << "Scalar arithmetic requires integer or floating-point operands";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  if (lhs.IsFloat()) {
    return internal::wrap<Scalar>(
        FloatOp::create(b, internal::unwrap(lhs), internal::unwrap(rhs)));
  }
  return internal::wrap<Scalar>(
      IntOp::create(b, internal::unwrap(lhs), internal::unwrap(rhs)));
}

Cond ScalarCompare(Scalar lhs, Scalar rhs, arith::CmpFPredicate fpred,
                   arith::CmpIPredicate ipred) {
  ABSL_CHECK_EQ(lhs.dtype(), rhs.dtype())  // CRASH_OK
      << "Scalar comparison requires operands of the same type";
  ABSL_CHECK(lhs.IsIntOrFloat())  // CRASH_OK
      << "Scalar comparison requires integer or floating-point operands";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  if (lhs.IsFloat()) {
    return internal::wrap<Cond>(arith::CmpFOp::create(
        b, fpred, internal::unwrap(lhs), internal::unwrap(rhs)));
  }
  return internal::wrap<Cond>(arith::CmpIOp::create(
      b, ipred, internal::unwrap(lhs), internal::unwrap(rhs)));
}

template <typename Op>
Scalar BitwiseScalar(Scalar lhs, Scalar rhs) {
  ABSL_CHECK(lhs.IsInt() && rhs.IsInt())  // CRASH_OK
      << "Bitwise scalar operators require integer operands";
  ABSL_CHECK_EQ(lhs.dtype(), rhs.dtype())  // CRASH_OK
      << "Bitwise scalar operators require operands of the same type";
  return internal::wrap<Scalar>(Op::create(internal::CurrentBuilder(),
                                           internal::unwrap(lhs),
                                           internal::unwrap(rhs)));
}

template <typename Op>
Scalar UnaryFloat(Scalar s) {
  ABSL_CHECK(s.IsFloat())  // CRASH_OK
      << "Unary math operation requires a floating-point scalar";
  return internal::wrap<Scalar>(
      Op::create(internal::CurrentBuilder(), internal::unwrap(s)));
}

}  // namespace

Scalar::Scalar(const void* impl) : ValueBase(impl) {
  mlir::Value value = internal::unwrap(*this);
  ABSL_CHECK(!value.getType().isIndex())  // CRASH_OK
      << "Scalar must not hold an index";
  ABSL_CHECK(value.getType().isIntOrFloat())  // CRASH_OK
      << "Scalar must be an integer or floating-point value";
}

Scalar::Scalar(Cond cond)
    : Scalar(internal::unwrap(cond).getAsOpaquePointer()) {}

Scalar::operator Cond() const {
  ABSL_CHECK(IsBool()) << "Cond requires a bool scalar";  // CRASH_OK
  return internal::wrap<Cond>(internal::unwrap(*this));
}

DType Scalar::dtype() const {
  return internal::FromMlirType(internal::unwrap(*this).getType());
}

bool Scalar::IsBool() const { return dtype().IsBool(); }
bool Scalar::IsFloat() const { return dtype().IsFloat(); }
bool Scalar::IsInt() const { return dtype().IsInt(); }
bool Scalar::IsIntOrFloat() const { return dtype().IsIntOrFloat(); }

Scalar Scalar::AsType(DType target) const {
  return internal::wrap<Scalar>(
      internal::EmitCast(internal::unwrap(*this), dtype(), target));
}

Scalar Scalar::Bitcast(DType target) const {
  return internal::wrap<Scalar>(
      internal::EmitBitcast(internal::unwrap(*this), dtype(), target));
}

Scalar Scalar::ToI32() const { return AsType(Int32Type()); }
Scalar Scalar::ToF32() const { return AsType(Float32Type()); }
Scalar Scalar::ToBF16() const { return AsType(BFloat16Type()); }

Scalar Scalar::Constant(Literal value) const {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Scalar>(
      arith::ConstantOp::create(b, internal::LiteralToAttr(b, dtype(), value)));
}

Scalar I32(Literal value) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Scalar>(
      arith::ConstantOp::create(b, b.getI32IntegerAttr(value.AsInt())));
}

Scalar F32(Literal value) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Scalar>(
      arith::ConstantOp::create(b, b.getF32FloatAttr(value.AsDouble())));
}

Scalar operator-(Scalar s) {
  ABSL_CHECK(s.IsIntOrFloat())  // CRASH_OK
      << "Unary - requires an integer or floating-point scalar";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  if (s.IsFloat()) {
    return internal::wrap<Scalar>(
        arith::NegFOp::create(b, internal::unwrap(s)));
  }
  return s.Constant(0) - s;
}

Scalar operator+(Scalar lhs, Scalar rhs) {
  return ScalarArith<arith::AddFOp, arith::AddIOp>(lhs, rhs);
}
Scalar operator-(Scalar lhs, Scalar rhs) {
  return ScalarArith<arith::SubFOp, arith::SubIOp>(lhs, rhs);
}
Scalar operator*(Scalar lhs, Scalar rhs) {
  return ScalarArith<arith::MulFOp, arith::MulIOp>(lhs, rhs);
}
Scalar operator/(Scalar lhs, Scalar rhs) {
  return ScalarArith<arith::DivFOp, arith::DivSIOp>(lhs, rhs);
}
Scalar operator%(Scalar lhs, Scalar rhs) {
  ABSL_CHECK_EQ(lhs.dtype(), rhs.dtype())  // CRASH_OK
      << "Scalar % requires operands of the same type";
  ABSL_CHECK(lhs.IsInt())  // CRASH_OK
      << "Scalar % requires integer operands";
  return internal::wrap<Scalar>(
      arith::RemSIOp::create(internal::CurrentBuilder(), internal::unwrap(lhs),
                             internal::unwrap(rhs)));
}
Scalar operator&(Scalar lhs, Scalar rhs) {
  return BitwiseScalar<arith::AndIOp>(lhs, rhs);
}
Scalar operator|(Scalar lhs, Scalar rhs) {
  return BitwiseScalar<arith::OrIOp>(lhs, rhs);
}
Scalar operator^(Scalar lhs, Scalar rhs) {
  return BitwiseScalar<arith::XOrIOp>(lhs, rhs);
}
Scalar operator<<(Scalar lhs, Scalar rhs) {
  return BitwiseScalar<arith::ShLIOp>(lhs, rhs);
}
Scalar operator>>(Scalar lhs, Scalar rhs) {
  return BitwiseScalar<arith::ShRSIOp>(lhs, rhs);
}

Scalar LogicalRightShift(Scalar lhs, Scalar rhs) {
  return BitwiseScalar<arith::ShRUIOp>(lhs, rhs);
}

Cond operator==(Scalar lhs, Scalar rhs) {
  return ScalarCompare(lhs, rhs, arith::CmpFPredicate::OEQ,
                       arith::CmpIPredicate::eq);
}
Cond operator!=(Scalar lhs, Scalar rhs) {
  return ScalarCompare(lhs, rhs, arith::CmpFPredicate::ONE,
                       arith::CmpIPredicate::ne);
}
Cond operator<(Scalar lhs, Scalar rhs) {
  return ScalarCompare(lhs, rhs, arith::CmpFPredicate::OLT,
                       arith::CmpIPredicate::slt);
}
Cond operator<=(Scalar lhs, Scalar rhs) {
  return ScalarCompare(lhs, rhs, arith::CmpFPredicate::OLE,
                       arith::CmpIPredicate::sle);
}
Cond operator>(Scalar lhs, Scalar rhs) {
  return ScalarCompare(lhs, rhs, arith::CmpFPredicate::OGT,
                       arith::CmpIPredicate::sgt);
}
Cond operator>=(Scalar lhs, Scalar rhs) {
  return ScalarCompare(lhs, rhs, arith::CmpFPredicate::OGE,
                       arith::CmpIPredicate::sge);
}

Scalar Select(Cond cond, Scalar if_true, Scalar if_false) {
  ABSL_CHECK_EQ(if_true.dtype(), if_false.dtype())  // CRASH_OK
      << "Select requires branches of the same type";
  return internal::wrap<Scalar>(arith::SelectOp::create(
      internal::CurrentBuilder(), internal::unwrap(cond),
      internal::unwrap(if_true), internal::unwrap(if_false)));
}
Scalar Min(Scalar lhs, Scalar rhs) {
  return ScalarArith<arith::MinimumFOp, arith::MinSIOp>(lhs, rhs);
}
Scalar Max(Scalar lhs, Scalar rhs) {
  return ScalarArith<arith::MaximumFOp, arith::MaxSIOp>(lhs, rhs);
}

Scalar Exp(Scalar s) { return UnaryFloat<math::ExpOp>(s); }
Scalar Exp2(Scalar s) { return UnaryFloat<math::Exp2Op>(s); }
Scalar Log(Scalar s) { return UnaryFloat<math::LogOp>(s); }
Scalar Log2(Scalar s) {
  static const bool is_natively_supported =
      torch_tpu::GetLibtpuVersion() >= torch_tpu::Version(0, 0, 48);
  if (is_natively_supported) {
    return UnaryFloat<math::Log2Op>(s);
  }
  return Log(s) * std::numbers::log2e;
}
Scalar Sqrt(Scalar s) { return UnaryFloat<math::SqrtOp>(s); }
Scalar Rsqrt(Scalar s) { return UnaryFloat<math::RsqrtOp>(s); }
Scalar Abs(Scalar s) {
  ABSL_CHECK(s.IsIntOrFloat())  // CRASH_OK
      << "Abs() requires an integer or floating-point scalar";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  if (s.IsFloat()) {
    return internal::wrap<Scalar>(math::AbsFOp::create(b, internal::unwrap(s)));
  }
  return internal::wrap<Scalar>(math::AbsIOp::create(b, internal::unwrap(s)));
}
Scalar Tanh(Scalar s) { return UnaryFloat<math::TanhOp>(s); }

}  // namespace torch_tpu::kiln
