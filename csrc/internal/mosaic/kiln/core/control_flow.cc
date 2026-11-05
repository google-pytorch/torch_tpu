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

#include <cstddef>
#include <string_view>
#include <type_traits>
#include <variant>

#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "csrc/internal/mosaic/kiln/core/tile.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace torch_tpu::kiln {
namespace {

mlir::Value ToIndex(mlir::ImplicitLocOpBuilder& b, const IndexArg& arg) {
  if (arg.IsInt()) {
    return mlir::arith::ConstantOp::create(b, b.getIndexAttr(arg.AsInt()));
  }
  Scalar s = arg.ToScalar();
  if (s.dtype().IsUnsigned()) {
    return mlir::arith::IndexCastUIOp::create(b, b.getIndexType(),
                                              internal::unwrap(s));
  }
  return mlir::arith::IndexCastOp::create(b, b.getIndexType(),
                                          internal::unwrap(s));
}

void ValidateLoopStep(const IndexArg& step) {
  if (step.IsInt()) {
    ABSL_CHECK_GT(step.AsInt(), 0) << "Loop step must be positive";  // CRASH_OK
    return;
  }
  Scalar s = step.ToScalar();
  if (auto const_op =
          internal::unwrap(s).getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto attr = llvm::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
      const llvm::APInt val = attr.getValue();
      const bool is_positive =
          s.dtype().IsUnsigned() ? !val.isZero() : val.isStrictlyPositive();
      ABSL_CHECK(is_positive) << "Loop step must be positive";  // CRASH_OK
    }
  }
}

using internal::ControlFlowValue;
using internal::ControlFlowValues;

// The current values of `vars`.
ControlFlowValues Load(llvm::ArrayRef<LoopVar> vars) {
  ControlFlowValues values;
  for (const LoopVar& var : vars) {
    values.push_back(std::visit(
        [](auto* p) -> ControlFlowValue {
          ABSL_CHECK(p != nullptr)  // CRASH_OK
              << "For() loop variables must not be null";
          return *p;
        },
        var));
  }
  return values;
}

// Wraps each of `values` with the kind and dtype of the matching `prototypes`.
ControlFlowValues Wrap(mlir::ValueRange values,
                       llvm::ArrayRef<ControlFlowValue> prototypes) {
  ControlFlowValues wrapped;
  wrapped.reserve(prototypes.size());
  for (auto [value, prototype] : llvm::zip_equal(values, prototypes)) {
    wrapped.push_back(std::visit(
        [value](const auto& p) -> ControlFlowValue {
          return internal::wrap<std::decay_t<decltype(p)>>(value, p.dtype());
        },
        prototype));
  }
  return wrapped;
}

// Binds each of `vars` to the matching `values`, with the kind and dtype of the
// matching `prototypes`.
void Bind(llvm::ArrayRef<LoopVar> vars, mlir::ValueRange values,
          llvm::ArrayRef<ControlFlowValue> prototypes) {
  for (auto [var, value] : llvm::zip_equal(vars, Wrap(values, prototypes))) {
    std::visit(
        [&value](auto* p) {
          *p = std::get<std::remove_pointer_t<decltype(p)>>(value);
        },
        var);
  }
}

llvm::SmallVector<mlir::Value, 4> Unwrap(
    llvm::ArrayRef<ControlFlowValue> values) {
  llvm::SmallVector<mlir::Value, 4> unwrapped;
  for (const ControlFlowValue& v : values) {
    unwrapped.push_back(
        std::visit([](const auto& x) { return internal::unwrap(x); }, v));
  }
  return unwrapped;
}

// Checks that each of `next` has the dtype and shape of the matching `init`.
// `what` names the values in error messages, e.g. "For() loop variable".
void ValidateMatchingValues(llvm::ArrayRef<ControlFlowValue> init,
                            llvm::ArrayRef<ControlFlowValue> next,
                            std::string_view what) {
  for (size_t i = 0; i < init.size(); ++i) {
    if (const Scalar* want = std::get_if<Scalar>(&init[i])) {
      const DType got = std::get<Scalar>(next[i]).dtype();
      ABSL_CHECK(got == want->dtype())  // CRASH_OK
          << what << " " << i << " changed dtype: expected " << want->dtype()
          << ", got " << got;
    } else {
      const Shape want_shape = std::get<Tile>(init[i]).shape();
      const Shape got = std::get<Tile>(next[i]).shape();
      ABSL_CHECK(got == want_shape)  // CRASH_OK
          << what << " " << i << " changed shape: expected " << want_shape
          << ", got " << got;
    }
  }
}

}  // namespace

namespace internal {

ControlFlowValues IfWithResults(
    Scalar cond, absl::FunctionRef<ControlFlowValues()> then_body,
    absl::FunctionRef<ControlFlowValues()> else_body) {
  ABSL_CHECK(cond.dtype().IsBool())  // CRASH_OK
      << "If() condition must be a bool scalar";
  mlir::ImplicitLocOpBuilder& b = CurrentBuilder();
  ControlFlowValues then_values;
  // `scf.if` creates each block with `b`, which moves its insertion point there
  // while the matching callback traces the Kiln ops of the branch, and infers
  // its result types from the `scf.yield`s.
  auto if_op = mlir::scf::IfOp::create(
      b, unwrap(cond),
      [&then_body, &then_values](mlir::OpBuilder& builder, mlir::Location loc) {
        then_values = then_body();
        mlir::scf::YieldOp::create(builder, loc, Unwrap(then_values));
      },
      [&else_body, &then_values](mlir::OpBuilder& builder, mlir::Location loc) {
        // Both branches return the same C++ type, so the number and kinds of
        // the values match; only their dtypes and shapes can differ.
        const ControlFlowValues else_values = else_body();
        ValidateMatchingValues(then_values, else_values,
                               "If() else-branch result");
        mlir::scf::YieldOp::create(builder, loc, Unwrap(else_values));
      });
  return Wrap(if_op.getResults(), then_values);
}

}  // namespace internal

void If(Scalar cond, absl::FunctionRef<void()> then_body) {
  ABSL_CHECK(cond.dtype().IsBool())  // CRASH_OK
      << "If() condition must be a bool scalar";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  auto if_op = mlir::scf::IfOp::create(b, internal::unwrap(cond),
                                       /*withElseRegion=*/false);
  mlir::OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(if_op.thenBlock());
  then_body();
}

void If(Scalar cond, absl::FunctionRef<void()> then_body,
        absl::FunctionRef<void()> else_body) {
  ABSL_CHECK(cond.dtype().IsBool())  // CRASH_OK
      << "If() condition must be a bool scalar";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  auto if_op = mlir::scf::IfOp::create(b, internal::unwrap(cond),
                                       /*withElseRegion=*/true);
  mlir::OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(if_op.thenBlock());
  then_body();
  b.setInsertionPointToStart(if_op.elseBlock());
  else_body();
}

void For(IndexArg lower, IndexArg upper, IndexArg step,
         absl::FunctionRef<void(Scalar)> body) {
  ValidateLoopStep(step);
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  auto for_op = mlir::scf::ForOp::create(b, ToIndex(b, lower),
                                         ToIndex(b, upper), ToIndex(b, step));
  mlir::OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(for_op.getBody());
  mlir::Value iv = mlir::arith::IndexCastOp::create(b, b.getI32Type(),
                                                    for_op.getInductionVar());
  body(internal::wrap<Scalar>(iv, i32));
}

void For(IndexArg lower, IndexArg upper, IndexArg step,
         llvm::ArrayRef<LoopVar> vars, absl::FunctionRef<void(Scalar)> body) {
  ValidateLoopStep(step);
  const ControlFlowValues init = Load(vars);
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  auto for_op = mlir::scf::ForOp::create(
      b, ToIndex(b, lower), ToIndex(b, upper), ToIndex(b, step), Unwrap(init));
  {
    mlir::OpBuilder::InsertionGuard guard(b);
    mlir::Block* block = for_op.getBody();
    // Without `iter_args`, `scf.for` comes with an empty `scf.yield`.
    if (!block->empty()) block->back().erase();
    b.setInsertionPointToStart(block);
    mlir::Value iv = mlir::arith::IndexCastOp::create(b, b.getI32Type(),
                                                      for_op.getInductionVar());
    // `body` sees and reassigns the current iteration's values, then yields
    // whatever `vars` hold.
    Bind(vars, for_op.getRegionIterArgs(), init);
    body(internal::wrap<Scalar>(iv, i32));
    const ControlFlowValues next = Load(vars);
    ValidateMatchingValues(init, next, "For() loop variable");
    mlir::scf::YieldOp::create(b, Unwrap(next));
  }
  // After the loop, `vars` hold the final values.
  Bind(vars, for_op.getResults(), init);
}

void RunScoped(absl::FunctionRef<void()> body) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  auto region_op = mlir::tpu::RegionOp::create(b, mlir::TypeRange{});
  mlir::Block& block = region_op.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(&block);
  body();
  mlir::tpu::YieldOp::create(b, mlir::ValueRange{});
}

}  // namespace torch_tpu::kiln
