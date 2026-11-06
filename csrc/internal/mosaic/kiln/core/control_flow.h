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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_CONTROL_FLOW_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_CONTROL_FLOW_H_

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/functional/function_ref.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/tile.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace torch_tpu::kiln {

namespace internal {

// A value carried out of a control-flow region: a result of an `If` branch or a
// loop variable of a `For`.
using ControlFlowValue = std::variant<Scalar, Tile>;
using ControlFlowValues = llvm::SmallVector<ControlFlowValue, 4>;

// Builds an `scf.if` with one result per value that the branches return. The
// result types come from `then_body`; `else_body` must return as many values,
// each with the same kind (`Scalar` or `Tile`), dtype and shape.
ControlFlowValues IfWithResults(
    Scalar cond, absl::FunctionRef<ControlFlowValues()> then_body,
    absl::FunctionRef<ControlFlowValues()> else_body);

// Converts between the return type `T` of the branches of an `If` and
// `ControlFlowValues`. Only defined for `Tile`, `Scalar`, and tuples of them.
template <typename T>
struct IfResult;

template <typename T>
struct IfSingleResult {
  static ControlFlowValues Pack(const T& v) { return {ControlFlowValue(v)}; }
  static T Unpack(const ControlFlowValues& v) { return std::get<T>(v[0]); }
};

template <>
struct IfResult<Tile> : IfSingleResult<Tile> {};

template <>
struct IfResult<Scalar> : IfSingleResult<Scalar> {};

template <typename... Ts>
struct IfResult<std::tuple<Ts...>> {
  static_assert(sizeof...(Ts) > 0, "If() branches must return a value");
  static_assert(((std::is_same_v<Ts, Tile> || std::is_same_v<Ts, Scalar>) &&
                 ...),
                "If() branches may only return tuples of Tile and Scalar");

  static ControlFlowValues Pack(const std::tuple<Ts...>& t) {
    return std::apply(
        [](const Ts&... v) {
          return ControlFlowValues{ControlFlowValue(v)...};
        },
        t);
  }
  static std::tuple<Ts...> Unpack(const ControlFlowValues& v) {
    return [&v]<size_t... I>(std::index_sequence<I...>) {
      return std::tuple<Ts...>(std::get<Ts>(v[I])...);
    }(std::index_sequence_for<Ts...>{});
  }
};

}  // namespace internal

// A variable carried from one loop iteration to the next (see `For` below).
using LoopVar = std::variant<Scalar*, Tile*>;

// Structured control flow inside a kernel body.
//
// All callbacks below (`then_body`, `else_body`, and `body`) are invoked
// synchronously to trace the IR region before the call returns and are not
// retained afterwards, so capturing local variables by reference is safe.
void If(Scalar cond, absl::FunctionRef<void()> then_body);
void If(Scalar cond, absl::FunctionRef<void()> then_body,
        absl::FunctionRef<void()> else_body);

// Conditional with results (`scf.if` with results, Pallas `lax.cond`): returns
// the value(s) returned by `then_body()` when `cond` is true and by
// `else_body()` otherwise. Both branches must return the same type: a `Tile`, a
// `Scalar`, or a `std::tuple` of them, with matching dtypes and shapes, e.g.
//   Tile x = If(cond, [&] { return a; }, [&] { return b; });
//   auto [s, x] = If(
//       cond, [&] { return std::tuple{s0, a}; },
//       [&] { return std::tuple{s1, b}; });
template <typename ThenFn, typename ElseFn,
          typename Result = std::invoke_result_t<ThenFn&>>
  requires(!std::is_void_v<Result>)  // Void branches use the overload above.
Result If(Scalar cond, ThenFn&& then_body, ElseFn&& else_body) {
  static_assert(std::is_same_v<Result, std::invoke_result_t<ElseFn&>>,
                "If() branches must return the same type");
  using Converter = internal::IfResult<Result>;
  return Converter::Unpack(internal::IfWithResults(
      cond, [&then_body] { return Converter::Pack(then_body()); },
      [&else_body] { return Converter::Pack(else_body()); }));
}

// Loops from `lower` to `upper` (exclusive) by positive `step`, passing the
// current induction variable to `body` as a signed i32 `Scalar` (`scf.for`).
// `body` is invoked synchronously once before `For` returns.
void For(IndexArg lower, IndexArg upper, IndexArg step,
         absl::FunctionRef<void(Scalar)> body);
inline void For(IndexArg lower, IndexArg upper,
                absl::FunctionRef<void(Scalar)> body) {
  For(lower, upper, 1, body);
}
inline void For(IndexArg upper, absl::FunctionRef<void(Scalar)> body) {
  For(0, upper, 1, body);
}

// Counted loop with loop-carried SSA values (`scf.for` with `iter_args`, the
// carry of `lax.fori_loop`). Within `body`, each of `vars` holds its value for
// the current iteration, and `body` may reassign it, keeping its dtype and
// shape. After the loop, each holds its final value. The pointees must outlive
// the call.
void For(IndexArg lower, IndexArg upper, IndexArg step,
         llvm::ArrayRef<LoopVar> vars, absl::FunctionRef<void(Scalar)> body);

// Executes `body` synchronously inside a `tpu.region` (Pallas `pl.run_scoped`)
// before `RunScoped` returns.
void RunScoped(absl::FunctionRef<void()> body);

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_CONTROL_FLOW_H_
