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

#ifndef TORCH_TPU_CSRC_INTERNAL_COMPILE_DISPATCH_INVOKE_SUBGRAPH_H_
#define TORCH_TPU_CSRC_INTERNAL_COMPILE_DISPATCH_INVOKE_SUBGRAPH_H_

#include <memory>
#include <string_view>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "csrc/common/compilation.h"

namespace torch_tpu {

// Creates a deferred invoke_subgraph operation on TPU that lowers `inputs` via
// `func.call` to the cloned `@main` function of `subgraph_module`.
//
// Inputs whose runtime shapes fit within `@main`'s bounded-dynamic argument
// types are padded and bounded during lowering to match `@main`'s signature.
// Dynamic result types on `@main` are not supported.
//
// Repeated calls for the same subgraph module (matched by location-free MLIR
// fingerprint), device assignment, and identifier reuse the already injected
// private `func.func` symbol in the destination module instead of cloning the
// subgraph multiple times, keeping the first call's debug locations.
//
// If `@main` has zero outputs, inputs and `subgraph_module` are still
// validated, no deferred operation is emitted, and an empty vector is returned.
//
// Args:
//   inputs: The flat list of tensor operands passed to `subgraph_module`'s
//       `@main` function.
//   subgraph_module: The MLIR module containing `@main` (and any private helper
//       symbols) to invoke.
//   identifier: Optional human-readable identifier used as a prefix for the
//       injected MLIR function symbol name. Non-`[A-Za-z0-9_]` characters are
//       replaced with `_` (with a leading `_` prepended if the first character
//       is a digit), and an empty `identifier` defaults to `"invoke_subgraph"`.
//
// Returns:
//   A list of tensors matching `@main`'s result signature, or an empty vector
//   if `@main` produces no results.
std::vector<at::Tensor> PyCreateInvokeSubgraphOp(
    const std::vector<at::Tensor>& inputs,
    const std::shared_ptr<ContextedModule>& subgraph_module,
    std::string_view identifier = "");

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_INTERNAL_COMPILE_DISPATCH_INVOKE_SUBGRAPH_H_
