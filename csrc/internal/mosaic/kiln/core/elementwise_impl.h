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

// Definitions of `internal::ElementwiseOps` (see elementwise.h). Include this
// from the .cc of a handle type and explicitly instantiate the struct, e.g.
//
//   template struct internal::ElementwiseOps<Tile>;

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_ELEMENTWISE_IMPL_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_ELEMENTWISE_IMPL_H_

#include <cstdint>
#include <string_view>

#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/elementwise.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"

namespace torch_tpu::kiln::internal {

// Broadcasts `value` (a scalar or vector) to a vector of `shape`. Returns
// `value` unchanged if it already has that shape.
mlir::Value EmitBroadcast(mlir::Value value, llvm::ArrayRef<int64_t> shape);

// Brings the operands of a multi-operand elementwise op to a common shape.
// Vector operands are broadcast numpy-style; if all operands are scalars there
// is nothing to do.
void AlignShapes(llvm::MutableArrayRef<mlir::Value> values);

template <typename T>
void CheckSameDType(T lhs, T rhs, std::string_view what) {
  ABSL_CHECK_EQ(lhs.dtype(), rhs.dtype())  // CRASH_OK
      << what << " requires operands of the same dtype";
}

template <typename T>
void CheckNotBool(T v, std::string_view what) {
  ABSL_CHECK(!v.dtype().IsBool())  // CRASH_OK
      << what << " requires non-bool operands";
}

// Aligns the operand shapes and emits the binary `Op`. Callers validate the
// dtypes beforehand.
template <typename Op, typename T>
T EmitBinary(T lhs, T rhs) {
  mlir::Value values[] = {unwrap(lhs), unwrap(rhs)};
  AlignShapes(values);
  return wrap<T>(Op::create(CurrentBuilder(), values[0], values[1]),
                 lhs.dtype());
}

// Binary op on integer or floating-point operands, dispatching on the element
// type to `FloatOp` or `IntOp`.
template <typename FloatOp, typename SIntOp, typename UIntOp = SIntOp,
          typename T>
T Arith(T lhs, T rhs, std::string_view what) {
  CheckSameDType(lhs, rhs, what);
  // i1 type not supported by Mosaic arith lowerings.
  CheckNotBool(lhs, what);
  if (lhs.dtype().IsFloat()) {
    return EmitBinary<FloatOp>(lhs, rhs);
  }
  if (lhs.dtype().IsUnsigned()) {
    return EmitBinary<UIntOp>(lhs, rhs);
  }
  return EmitBinary<SIntOp>(lhs, rhs);
}

// Binary op on integer or bool (mask) operands.
template <typename Op, typename T>
T Bitwise(T lhs, T rhs, std::string_view what) {
  CheckSameDType(lhs, rhs, what);
  ABSL_CHECK(lhs.dtype().IsInt() || lhs.dtype().IsBool())  // CRASH_OK
      << what << " requires integer or bool operands";
  return EmitBinary<Op>(lhs, rhs);
}

// Binary op on integer or bool (mask) operands.
template <typename Op, typename T>
T Shift(T lhs, T rhs, std::string_view what) {
  CheckSameDType(lhs, rhs, what);
  CheckNotBool(lhs, what);
  ABSL_CHECK(lhs.dtype().IsInt())  // CRASH_OK
      << what << " requires integer operands";
  return EmitBinary<Op>(lhs, rhs);
}

template <typename T>
T Compare(T lhs, T rhs, mlir::arith::CmpFPredicate fpred,
          mlir::arith::CmpIPredicate ipred) {
  CheckSameDType(lhs, rhs, "Comparison");
  mlir::Value values[] = {unwrap(lhs), unwrap(rhs)};
  AlignShapes(values);
  mlir::ImplicitLocOpBuilder& b = CurrentBuilder();
  if (lhs.dtype().IsFloat()) {
    return wrap<T>(mlir::arith::CmpFOp::create(b, fpred, values[0], values[1]),
                   bool_);
  }
  return wrap<T>(mlir::arith::CmpIOp::create(b, ipred, values[0], values[1]),
                 bool_);
}

template <typename T>
T Compare(T lhs, T rhs, mlir::arith::CmpFPredicate fpred,
          mlir::arith::CmpIPredicate signed_ipred,
          mlir::arith::CmpIPredicate unsigned_ipred) {
  return Compare(lhs, rhs, fpred,
                 lhs.dtype().IsUnsigned() || lhs.dtype().IsBool()
                     ? unsigned_ipred
                     : signed_ipred);
}

template <typename Op, typename T>
T UnaryFloat(T v, std::string_view what) {
  ABSL_CHECK(v.dtype().IsFloat())  // CRASH_OK
      << what << " requires floating-point operands";
  return wrap<T>(Op::create(CurrentBuilder(), unwrap(v)), v.dtype());
}

template <typename T>
T ElementwiseOps<T>::AsType(T v, DType target) {
  return wrap<T>(EmitCast(unwrap(v), v.dtype(), target), target);
}

template <typename T>
T ElementwiseOps<T>::Bitcast(T v, DType target) {
  return wrap<T>(EmitBitcast(unwrap(v), v.dtype(), target), target);
}

template <typename T>
T ElementwiseOps<T>::Neg(T v) {
  CheckNotBool(v, "Unary -");
  if (v.dtype().IsFloat()) {
    return wrap<T>(mlir::arith::NegFOp::create(CurrentBuilder(), unwrap(v)),
                   v.dtype());
  }
  return 0 - v;
}

template <typename T>
T ElementwiseOps<T>::Add(T lhs, T rhs) {
  return Arith<mlir::arith::AddFOp, mlir::arith::AddIOp>(lhs, rhs, "+");
}

template <typename T>
T ElementwiseOps<T>::Sub(T lhs, T rhs) {
  return Arith<mlir::arith::SubFOp, mlir::arith::SubIOp>(lhs, rhs, "-");
}

template <typename T>
T ElementwiseOps<T>::Mul(T lhs, T rhs) {
  return Arith<mlir::arith::MulFOp, mlir::arith::MulIOp>(lhs, rhs, "*");
}

template <typename T>
T ElementwiseOps<T>::Div(T lhs, T rhs) {
  return Arith<mlir::arith::DivFOp, mlir::arith::DivSIOp, mlir::arith::DivUIOp>(
      lhs, rhs, "/");
}

template <typename T>
T ElementwiseOps<T>::Rem(T lhs, T rhs) {
  CheckSameDType(lhs, rhs, "%");
  ABSL_CHECK(lhs.dtype().IsInt()) << "% requires integer operands";  // CRASH_OK
  if (lhs.dtype().IsUnsigned()) {
    return EmitBinary<mlir::arith::RemUIOp>(lhs, rhs);
  }
  return EmitBinary<mlir::arith::RemSIOp>(lhs, rhs);
}

template <typename T>
T ElementwiseOps<T>::Min(T lhs, T rhs) {
  if (lhs.dtype().IsBool() && rhs.dtype().IsBool()) {
    return lhs & rhs;
  }
  return Arith<mlir::arith::MinimumFOp, mlir::arith::MinSIOp,
               mlir::arith::MinUIOp>(lhs, rhs, "Min()");
}

template <typename T>
T ElementwiseOps<T>::Max(T lhs, T rhs) {
  if (lhs.dtype().IsBool() && rhs.dtype().IsBool()) {
    return lhs | rhs;
  }
  return Arith<mlir::arith::MaximumFOp, mlir::arith::MaxSIOp,
               mlir::arith::MaxUIOp>(lhs, rhs, "Max()");
}

template <typename T>
T ElementwiseOps<T>::And(T lhs, T rhs) {
  return Bitwise<mlir::arith::AndIOp>(lhs, rhs, "&");
}

template <typename T>
T ElementwiseOps<T>::Or(T lhs, T rhs) {
  return Bitwise<mlir::arith::OrIOp>(lhs, rhs, "|");
}

template <typename T>
T ElementwiseOps<T>::Xor(T lhs, T rhs) {
  return Bitwise<mlir::arith::XOrIOp>(lhs, rhs, "^");
}

template <typename T>
T ElementwiseOps<T>::ShiftLeft(T lhs, T rhs) {
  return Shift<mlir::arith::ShLIOp>(lhs, rhs, "<<");
}

template <typename T>
T ElementwiseOps<T>::ShiftRight(T lhs, T rhs) {
  if (lhs.dtype().IsUnsigned()) {
    return Shift<mlir::arith::ShRUIOp>(lhs, rhs, ">>");
  }
  return Shift<mlir::arith::ShRSIOp>(lhs, rhs, ">>");
}

template <typename T>
T ElementwiseOps<T>::LogicalNot(T v) {
  ABSL_CHECK(v.dtype().IsBool()) << "! requires a bool operand";  // CRASH_OK
  return v ^ true;
}

template <typename T>
T ElementwiseOps<T>::Eq(T lhs, T rhs) {
  // Compare bools bitwise, as Min / Max / LogicalNot do.
  if (lhs.dtype().IsBool() && rhs.dtype().IsBool()) {
    return !(lhs ^ rhs);
  }
  return Compare(lhs, rhs, mlir::arith::CmpFPredicate::OEQ,
                 mlir::arith::CmpIPredicate::eq);
}

template <typename T>
T ElementwiseOps<T>::Ne(T lhs, T rhs) {
  // See `Eq`.
  if (lhs.dtype().IsBool() && rhs.dtype().IsBool()) {
    return lhs ^ rhs;
  }
  return Compare(lhs, rhs, mlir::arith::CmpFPredicate::ONE,
                 mlir::arith::CmpIPredicate::ne);
}

template <typename T>
T ElementwiseOps<T>::Lt(T lhs, T rhs) {
  return Compare(lhs, rhs, mlir::arith::CmpFPredicate::OLT,
                 mlir::arith::CmpIPredicate::slt,
                 mlir::arith::CmpIPredicate::ult);
}

template <typename T>
T ElementwiseOps<T>::Le(T lhs, T rhs) {
  return Compare(lhs, rhs, mlir::arith::CmpFPredicate::OLE,
                 mlir::arith::CmpIPredicate::sle,
                 mlir::arith::CmpIPredicate::ule);
}

template <typename T>
T ElementwiseOps<T>::Gt(T lhs, T rhs) {
  return Compare(lhs, rhs, mlir::arith::CmpFPredicate::OGT,
                 mlir::arith::CmpIPredicate::sgt,
                 mlir::arith::CmpIPredicate::ugt);
}

template <typename T>
T ElementwiseOps<T>::Ge(T lhs, T rhs) {
  return Compare(lhs, rhs, mlir::arith::CmpFPredicate::OGE,
                 mlir::arith::CmpIPredicate::sge,
                 mlir::arith::CmpIPredicate::uge);
}

template <typename T>
T ElementwiseOps<T>::Select(T mask, T if_true, T if_false) {
  CheckSameDType(if_true, if_false, "Select() branches");
  mlir::Value values[] = {unwrap(mask), unwrap(if_true), unwrap(if_false)};
  ABSL_CHECK(mlir::getElementTypeOrSelf(values[0]).isInteger(1))  // CRASH_OK
      << "Select() mask must be bool";
  AlignShapes(values);
  return wrap<T>(mlir::arith::SelectOp::create(CurrentBuilder(), values[0],
                                               values[1], values[2]),
                 if_true.dtype());
}

template <typename T>
T ElementwiseOps<T>::Exp(T v) {
  return UnaryFloat<mlir::math::ExpOp>(v, "Exp()");
}

template <typename T>
T ElementwiseOps<T>::Exp2(T v) {
  return UnaryFloat<mlir::math::Exp2Op>(v, "Exp2()");
}

template <typename T>
T ElementwiseOps<T>::Log(T v) {
  return UnaryFloat<mlir::math::LogOp>(v, "Log()");
}

template <typename T>
T ElementwiseOps<T>::Log2(T v) {
  return UnaryFloat<mlir::math::Log2Op>(v, "Log2()");
}

template <typename T>
T ElementwiseOps<T>::Sqrt(T v) {
  return UnaryFloat<mlir::math::SqrtOp>(v, "Sqrt()");
}

template <typename T>
T ElementwiseOps<T>::Rsqrt(T v) {
  return UnaryFloat<mlir::math::RsqrtOp>(v, "Rsqrt()");
}

template <typename T>
T ElementwiseOps<T>::Tanh(T v) {
  return UnaryFloat<mlir::math::TanhOp>(v, "Tanh()");
}

template <typename T>
T ElementwiseOps<T>::Abs(T v) {
  CheckNotBool(v, "Abs()");
  if (v.dtype().IsUnsigned()) {
    return v;
  }
  mlir::ImplicitLocOpBuilder& b = CurrentBuilder();
  if (v.dtype().IsFloat()) {
    return wrap<T>(mlir::math::AbsFOp::create(b, unwrap(v)), v.dtype());
  }
  return wrap<T>(mlir::math::AbsIOp::create(b, unwrap(v)), v.dtype());
}

}  // namespace torch_tpu::kiln::internal

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_ELEMENTWISE_IMPL_H_
