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

#include <cstdint>

#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/cond.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/elementwise.h"
#include "csrc/internal/mosaic/kiln/core/elementwise_impl.h"  // IWYU pragma: keep
#include "csrc/internal/mosaic/kiln/core/literal.h"
#include "csrc/internal/mosaic/kiln/core/value_base.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace torch_tpu::kiln {

namespace {

Scalar MakeScalarConstant(DType dtype, Literal value) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Scalar>(
      mlir::arith::ConstantOp::create(b,
                                      internal::LiteralToAttr(b, dtype, value)),
      dtype);
}

}  // namespace

Scalar::Scalar(const void* impl, DType dtype) : ValueBase(impl, dtype) {
  mlir::Value value = internal::unwrap(*this);
  ABSL_CHECK(!value.getType().isIndex())  // CRASH_OK
      << "Scalar must not hold an index";
  ABSL_CHECK(value.getType().isIntOrFloat())  // CRASH_OK
      << "Scalar must be an integer or floating-point value";
}

Scalar::Scalar(Cond cond)
    : Scalar(internal::unwrap(cond).getAsOpaquePointer(), bool_) {}

Scalar::operator Cond() const {
  ABSL_CHECK(dtype().IsBool()) << "Cond requires a bool scalar";  // CRASH_OK
  return internal::wrap<Cond>(internal::unwrap(*this), bool_);
}

Scalar MultipleOf(Scalar value, int32_t multiple) {
  ABSL_CHECK(value.dtype().IsInt())  // CRASH_OK
      << "MultipleOf() requires an integer scalar";
  ABSL_CHECK_GT(multiple, 0)  // CRASH_OK
      << "MultipleOf() multiple must be positive";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Scalar>(
      mlir::tpu::AssumeMultipleOp::create(b, internal::unwrap(value), multiple),
      value.dtype());
}

Scalar Scalar::MakeConstant(Literal value) const {
  return MakeScalarConstant(dtype(), value);
}

Scalar I32(Literal value) { return MakeScalarConstant(i32, value); }
Scalar U32(Literal value) { return MakeScalarConstant(u32, value); }
Scalar F32(Literal value) { return MakeScalarConstant(f32, value); }

template struct internal::ElementwiseOps<Scalar, Cond>;

}  // namespace torch_tpu::kiln
