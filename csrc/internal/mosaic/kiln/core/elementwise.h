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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_ELEMENTWISE_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_ELEMENTWISE_H_

#include "csrc/internal/mosaic/kiln/core/dtype.h"

namespace torch_tpu::kiln {

namespace internal {

// Emitters for the elementwise operations shared by `Scalar` and `Tile`.
//
// The underlying MLIR `arith` / `math` ops are elementwise over both scalars
// and vectors, so a single implementation serves both handle types. The only
// shape-dependent step is aligning the operands of multi-operand ops: vector
// operands are broadcast numpy-style to a common shape, while scalar operands
// have no shape and are left untouched (see `AlignShapes`).
//
// `T` is the handle type (`Scalar` or `Tile`). `T` must provide
// `DType dtype() const` and support `Literal` operands.
//
// The members are defined in elementwise_impl.h and explicitly instantiated in
// scalar.cc and tile.cc, which keeps MLIR out of the public headers.
template <typename T>
struct ElementwiseOps {
  // Type conversion.
  static T AsType(T v, DType target);
  static T Bitcast(T v, DType target);

  // Arithmetic.
  static T Neg(T v);
  static T Add(T lhs, T rhs);
  static T Sub(T lhs, T rhs);
  static T Mul(T lhs, T rhs);
  static T Div(T lhs, T rhs);
  static T Rem(T lhs, T rhs);
  static T Min(T lhs, T rhs);
  static T Max(T lhs, T rhs);

  // Bitwise / logical.
  static T And(T lhs, T rhs);
  static T Or(T lhs, T rhs);
  static T Xor(T lhs, T rhs);
  static T ShiftLeft(T lhs, T rhs);
  static T ShiftRight(T lhs, T rhs);
  static T LogicalNot(T v);

  // Comparison.
  static T Eq(T lhs, T rhs);
  static T Ne(T lhs, T rhs);
  static T Lt(T lhs, T rhs);
  static T Le(T lhs, T rhs);
  static T Gt(T lhs, T rhs);
  static T Ge(T lhs, T rhs);

  static T Select(T mask, T if_true, T if_false);

  // Unary math.
  static T Exp(T v);
  static T Exp2(T v);
  static T Log(T v);
  static T Log2(T v);
  static T Sqrt(T v);
  static T Rsqrt(T v);
  static T Tanh(T v);
  static T Abs(T v);
};

}  // namespace internal

// CRTP mixin providing the elementwise operators and math functions shared by
// `Scalar` and `Tile`.
//
// Multi-operand ops require operands of the same dtype; element types are
// never promoted implicitly (use `AsType`). For tiles, operands are broadcast
// numpy-style to a common shape first. Comparisons produce a bool `Derived`.
template <typename Derived>
class WithElementwiseOps {
  using Ops = internal::ElementwiseOps<Derived>;

 public:
  // Element type conversion (float<->float, int<->float, int widths).
  Derived AsType(DType target) const { return Ops::AsType(derived(), target); }
  // Reinterprets the bit pattern of each element as `target` (must have the
  // same bit width).
  Derived Bitcast(DType target) const {
    return Ops::Bitcast(derived(), target);
  }
  Derived ToBool() const { return AsType(bool_); }
  Derived ToI32() const { return AsType(i32); }
  Derived ToU8() const { return AsType(u8); }
  Derived ToU16() const { return AsType(u16); }
  Derived ToU32() const { return AsType(u32); }
  Derived ToU64() const { return AsType(u64); }
  Derived ToF32() const { return AsType(f32); }
  Derived ToBF16() const { return AsType(bf16); }

  // Arithmetic on integer or floating-point operands.
  friend Derived operator-(Derived v) { return Ops::Neg(v); }
  friend Derived operator+(Derived lhs, Derived rhs) {
    return Ops::Add(lhs, rhs);
  }
  friend Derived operator-(Derived lhs, Derived rhs) {
    return Ops::Sub(lhs, rhs);
  }
  friend Derived operator*(Derived lhs, Derived rhs) {
    return Ops::Mul(lhs, rhs);
  }
  friend Derived operator/(Derived lhs, Derived rhs) {
    return Ops::Div(lhs, rhs);
  }
  // Remainder; integer operands only.
  friend Derived operator%(Derived lhs, Derived rhs) {
    return Ops::Rem(lhs, rhs);
  }

  friend Derived& operator+=(Derived& lhs, Derived rhs) {
    return lhs = lhs + rhs;
  }
  friend Derived& operator-=(Derived& lhs, Derived rhs) {
    return lhs = lhs - rhs;
  }
  friend Derived& operator*=(Derived& lhs, Derived rhs) {
    return lhs = lhs * rhs;
  }
  friend Derived& operator/=(Derived& lhs, Derived rhs) {
    return lhs = lhs / rhs;
  }
  friend Derived& operator%=(Derived& lhs, Derived rhs) {
    return lhs = lhs % rhs;
  }

  // Bitwise / logical ops on integer and bool (mask) operands. `>>` is an
  // arithmetic shift for signed types, and logical shift for unsigned types.
  friend Derived operator&(Derived lhs, Derived rhs) {
    return Ops::And(lhs, rhs);
  }
  friend Derived operator|(Derived lhs, Derived rhs) {
    return Ops::Or(lhs, rhs);
  }
  friend Derived operator^(Derived lhs, Derived rhs) {
    return Ops::Xor(lhs, rhs);
  }
  friend Derived operator<<(Derived lhs, Derived rhs) {
    return Ops::ShiftLeft(lhs, rhs);
  }
  friend Derived operator>>(Derived lhs, Derived rhs) {
    return Ops::ShiftRight(lhs, rhs);
  }

  friend Derived& operator&=(Derived& lhs, Derived rhs) {
    return lhs = lhs & rhs;
  }
  friend Derived& operator|=(Derived& lhs, Derived rhs) {
    return lhs = lhs | rhs;
  }
  friend Derived& operator^=(Derived& lhs, Derived rhs) {
    return lhs = lhs ^ rhs;
  }
  friend Derived& operator<<=(Derived& lhs, Derived rhs) {
    return lhs = lhs << rhs;
  }
  friend Derived& operator>>=(Derived& lhs, Derived rhs) {
    return lhs = lhs >> rhs;
  }
  friend Derived operator!(Derived v) { return Ops::LogicalNot(v); }

  // Comparisons on bool, integer, or floating-point operands.
  friend Derived operator==(Derived lhs, Derived rhs) {
    return Ops::Eq(lhs, rhs);
  }
  friend Derived operator!=(Derived lhs, Derived rhs) {
    return Ops::Ne(lhs, rhs);
  }
  friend Derived operator<(Derived lhs, Derived rhs) {
    return Ops::Lt(lhs, rhs);
  }
  friend Derived operator<=(Derived lhs, Derived rhs) {
    return Ops::Le(lhs, rhs);
  }
  friend Derived operator>(Derived lhs, Derived rhs) {
    return Ops::Gt(lhs, rhs);
  }
  friend Derived operator>=(Derived lhs, Derived rhs) {
    return Ops::Ge(lhs, rhs);
  }

  // Elementwise min / max / select. On bool operands `Min` / `Max` are
  // logical and / or.
  friend Derived Min(Derived lhs, Derived rhs) { return Ops::Min(lhs, rhs); }
  friend Derived Max(Derived lhs, Derived rhs) { return Ops::Max(lhs, rhs); }
  friend Derived Select(Derived mask, Derived if_true, Derived if_false) {
    return Ops::Select(mask, if_true, if_false);
  }

  // Unary math on floating-point operands (`Abs` also accepts integers).
  friend Derived Exp(Derived v) { return Ops::Exp(v); }
  friend Derived Exp2(Derived v) { return Ops::Exp2(v); }
  friend Derived Log(Derived v) { return Ops::Log(v); }
  friend Derived Log2(Derived v) { return Ops::Log2(v); }
  friend Derived Sqrt(Derived v) { return Ops::Sqrt(v); }
  friend Derived Rsqrt(Derived v) { return Ops::Rsqrt(v); }
  friend Derived Tanh(Derived v) { return Ops::Tanh(v); }
  friend Derived Abs(Derived v) { return Ops::Abs(v); }
  friend Derived Bitcast(Derived v, DType target) { return v.Bitcast(target); }

 private:
  const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_ELEMENTWISE_H_
