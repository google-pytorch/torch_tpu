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

namespace torch_tpu::kiln {

namespace arith = mlir::arith;

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

Scalar Scalar::MakeConstant(Literal value) const {
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

template struct internal::ElementwiseOps<Scalar, Cond>;

}  // namespace torch_tpu::kiln
