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
#include "csrc/internal/mosaic/kiln/core/elementwise.h"
#include "csrc/internal/mosaic/kiln/core/literal.h"
#include "csrc/internal/mosaic/kiln/core/value_base.h"

namespace torch_tpu::kiln {

// A scalar SSA value (boolean, integer, or floating-point). Grid indices,
// loop induction variables, SMEM scalar values, constants, and scalar
// statistics are Scalars.
//
// Arithmetic, bitwise, comparison (yielding a `Cond`), min / max / select and
// unary math operators come from `WithElementwiseOps`; overloads taking a
// `Literal` on either side come from `WithLiteralOps`.
class Scalar : public ValueBase,
               public WithElementwiseOps<Scalar, Cond>,
               public WithLiteralOps<Scalar, Cond> {
 public:
  /* implicit */ Scalar(Cond cond);  // NOLINT(google-explicit-constructor)
  explicit operator Cond() const;

  Scalar MakeConstant(Literal value) const;

  DType dtype() const;
  bool IsBool() const;
  bool IsFloat() const;
  bool IsInt() const;
  bool IsIntOrFloat() const;

 private:
  friend struct internal::ValueAccess;
  explicit Scalar(const void* impl);
};

Scalar I32(Literal value);
Scalar F32(Literal value);

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SCALAR_H_
