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

#include "csrc/internal/mosaic/kiln/core/cond.h"

#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/value_base.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace torch_tpu::kiln {

Cond::Cond(const void* impl) : ValueBase(impl) {
  ABSL_CHECK(internal::unwrap(*this).getType().isInteger(1))  // CRASH_OK
      << "Cond must be i1";
}

Cond operator&&(Cond lhs, Cond rhs) {
  return internal::wrap<Cond>(mlir::arith::AndIOp::create(
      internal::CurrentBuilder(), internal::unwrap(lhs),
      internal::unwrap(rhs)));
}

Cond operator||(Cond lhs, Cond rhs) {
  return internal::wrap<Cond>(
      mlir::arith::OrIOp::create(internal::CurrentBuilder(),
                                 internal::unwrap(lhs), internal::unwrap(rhs)));
}

Cond operator!(Cond cond) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  mlir::Value t = mlir::arith::ConstantOp::create(b, b.getBoolAttr(true));
  return internal::wrap<Cond>(
      mlir::arith::XOrIOp::create(b, internal::unwrap(cond), t));
}

}  // namespace torch_tpu::kiln
