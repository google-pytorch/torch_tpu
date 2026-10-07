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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SCALAR_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SCALAR_H_

#include "csrc/internal/mosaic/kiln/core/cond.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/literal.h"
#include "csrc/internal/mosaic/kiln/core/value_base.h"

namespace torch_tpu::kiln {

// A scalar SSA value (boolean, integer, or floating-point). Grid indices,
// loop induction variables, SMEM scalar values, constants, and scalar
// statistics are Scalars.
class Scalar : public ValueBase, public WithLiteralOps<Scalar, Cond> {
 public:
  /* implicit */ Scalar(Cond cond);  // NOLINT(google-explicit-constructor)
  explicit operator Cond() const;

  Scalar MakeConstant(Literal value) const { return Constant(value); }

  DType dtype() const;
  bool IsBool() const;
  bool IsFloat() const;
  bool IsInt() const;
  bool IsIntOrFloat() const;

  Scalar AsType(DType target) const;
  Scalar Bitcast(DType target) const;
  Scalar ToI32() const;
  Scalar ToF32() const;
  Scalar ToBF16() const;

  // A constant with the same type as this scalar.
  Scalar Constant(Literal value) const;

  friend Scalar operator-(Scalar s);
  friend Scalar operator+(Scalar lhs, Scalar rhs);
  friend Scalar operator-(Scalar lhs, Scalar rhs);
  friend Scalar operator*(Scalar lhs, Scalar rhs);
  friend Scalar operator/(Scalar lhs, Scalar rhs);
  friend Scalar operator%(Scalar lhs, Scalar rhs);

  // Bitwise ops on integer scalars. `>>` is an arithmetic (signed) shift.
  friend Scalar operator&(Scalar lhs, Scalar rhs);
  friend Scalar operator|(Scalar lhs, Scalar rhs);
  friend Scalar operator^(Scalar lhs, Scalar rhs);
  friend Scalar operator<<(Scalar lhs, Scalar rhs);
  friend Scalar operator>>(Scalar lhs, Scalar rhs);

  friend Scalar LogicalRightShift(Scalar lhs, Scalar rhs);

  friend Cond operator==(Scalar lhs, Scalar rhs);
  friend Cond operator!=(Scalar lhs, Scalar rhs);
  friend Cond operator<(Scalar lhs, Scalar rhs);
  friend Cond operator<=(Scalar lhs, Scalar rhs);
  friend Cond operator>(Scalar lhs, Scalar rhs);
  friend Cond operator>=(Scalar lhs, Scalar rhs);

 private:
  friend struct internal::ValueAccess;
  explicit Scalar(const void* impl);
};

Scalar I32(Literal value);
Scalar F32(Literal value);

Scalar Select(Cond cond, Scalar if_true, Scalar if_false);
Scalar Min(Scalar lhs, Scalar rhs);
Scalar Max(Scalar lhs, Scalar rhs);

inline Scalar Bitcast(Scalar s, DType target) { return s.Bitcast(target); }

// Scalar elementwise math
// TODO(willfroom): Combine these with the tile implementations.
Scalar Exp(Scalar s);
Scalar Exp2(Scalar s);
Scalar Log(Scalar s);
Scalar Log2(Scalar s);
Scalar Sqrt(Scalar s);
Scalar Rsqrt(Scalar s);
Scalar Abs(Scalar s);
Scalar Tanh(Scalar s);

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SCALAR_H_
