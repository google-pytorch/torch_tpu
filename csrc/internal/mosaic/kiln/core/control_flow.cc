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

#include "csrc/internal/mosaic/kiln/core/control_flow.h"

#include "absl/functional/function_ref.h"
#include "csrc/internal/mosaic/kiln/core/cond.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace torch_tpu::kiln {
namespace {

mlir::Value ToIndex(mlir::ImplicitLocOpBuilder& b, const IndexArg& arg) {
  if (arg.IsInt()) {
    return mlir::arith::ConstantOp::create(b, b.getIndexAttr(arg.AsInt()));
  }
  return mlir::arith::IndexCastOp::create(b, b.getIndexType(),
                                          internal::unwrap(arg.ToScalar()));
}

}  // namespace

void If(Cond cond, absl::FunctionRef<void()> then_body) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  auto if_op = mlir::scf::IfOp::create(b, internal::unwrap(cond),
                                       /*withElseRegion=*/false);
  mlir::OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(if_op.thenBlock());
  then_body();
}

void If(Cond cond, absl::FunctionRef<void()> then_body,
        absl::FunctionRef<void()> else_body) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  auto if_op = mlir::scf::IfOp::create(b, internal::unwrap(cond),
                                       /*withElseRegion=*/true);
  mlir::OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(if_op.thenBlock());
  then_body();
  b.setInsertionPointToStart(if_op.elseBlock());
  else_body();
}

void For(IndexArg lower, IndexArg upper, absl::FunctionRef<void(Scalar)> body) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  mlir::Value step = mlir::arith::ConstantOp::create(b, b.getIndexAttr(1));
  auto for_op =
      mlir::scf::ForOp::create(b, ToIndex(b, lower), ToIndex(b, upper), step);
  mlir::OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(for_op.getBody());
  mlir::Value iv = mlir::arith::IndexCastOp::create(b, b.getI32Type(),
                                                    for_op.getInductionVar());
  body(internal::wrap<Scalar>(iv));
}

}  // namespace torch_tpu::kiln
