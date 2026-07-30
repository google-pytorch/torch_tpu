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

#ifndef TORCH_TPU_INTERNAL_COMPILE_DISPATCH_COND_H_
#define TORCH_TPU_INTERNAL_COMPILE_DISPATCH_COND_H_

#include <memory>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "torch_tpu/common/compilation.h"

namespace torch_tpu {

// Creates a StableHLO if operation on TPU.
//
// Args:
//   pred: A scalar or single-element bool tensor predicate.
//   operands: Flat tensor operands passed to both branch functions.
//   true_module: The MLIR sub-graph representing true_fn.
//   false_module: The MLIR sub-graph representing false_fn.
//   dummy_outputs: Flat meta tensors used to statically infer output shapes and
//       dtypes.
//
// Returns:
//   A flat list of tensors containing the selected branch outputs.
std::vector<at::Tensor> PyCreateCondOp(
    const at::Tensor& pred, const std::vector<at::Tensor>& operands,
    const std::shared_ptr<ContextedModule>& true_module,
    const std::shared_ptr<ContextedModule>& false_module,
    const std::vector<at::Tensor>& dummy_outputs);

}  // namespace torch_tpu

#endif  // TORCH_TPU_INTERNAL_COMPILE_DISPATCH_COND_H_
