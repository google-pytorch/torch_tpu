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
#include "csrc/internal/mosaic/kiln/core/cond.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"

namespace torch_tpu::kiln {

// Structured control flow inside a kernel body.
void If(Cond cond, absl::FunctionRef<void()> then_body);
void If(Cond cond, absl::FunctionRef<void()> then_body,
        absl::FunctionRef<void()> else_body);
void For(IndexArg lower, IndexArg upper, absl::FunctionRef<void(Scalar)> body);
inline void For(IndexArg upper, absl::FunctionRef<void(Scalar)> body) {
  For(0, upper, body);
}

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_CONTROL_FLOW_H_
