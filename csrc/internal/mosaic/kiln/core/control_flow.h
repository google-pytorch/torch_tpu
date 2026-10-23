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

#include "absl/functional/function_ref.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"

namespace torch_tpu::kiln {

// Structured control flow inside a kernel body.
//
// All callbacks below (`then_body`, `else_body`, and `body`) are invoked
// synchronously to trace the IR region before the call returns and are not
// retained afterwards, so capturing local variables by reference is safe.
void If(Scalar cond, absl::FunctionRef<void()> then_body);
void If(Scalar cond, absl::FunctionRef<void()> then_body,
        absl::FunctionRef<void()> else_body);

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

// Executes `body` synchronously inside a `tpu.region` (Pallas `pl.run_scoped`)
// before `RunScoped` returns.
void RunScoped(absl::FunctionRef<void()> body);

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_CONTROL_FLOW_H_
