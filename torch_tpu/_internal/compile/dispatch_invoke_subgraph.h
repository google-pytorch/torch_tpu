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

#ifndef TORCH_TPU_INTERNAL_COMPILE_DISPATCH_INVOKE_SUBGRAPH_H_
#define TORCH_TPU_INTERNAL_COMPILE_DISPATCH_INVOKE_SUBGRAPH_H_

#include <memory>
#include <string_view>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "torch_tpu/common/compilation.h"

namespace torch_tpu {

std::vector<at::Tensor> PyCreateInvokeSubgraphOp(
    const std::vector<at::Tensor>& inputs,
    const std::shared_ptr<ContextedModule>& body_module,
    const std::vector<at::Tensor>& dummy_outputs, std::string_view identifier);

}  // namespace torch_tpu

#endif  // TORCH_TPU_INTERNAL_COMPILE_DISPATCH_INVOKE_SUBGRAPH_H_
