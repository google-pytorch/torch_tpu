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

#ifndef TORCH_TPU_CSRC_COMMON_AUTOGRAD_UTILS_H_
#define TORCH_TPU_CSRC_COMMON_AUTOGRAD_UTILS_H_

#include <concepts>

#include "c10/util/intrusive_ptr.h"
#include "csrc/common/macro_utils.h"
#if TT_TORCH_VERSION_GE(2, 13)
#include "torch/csrc/autograd/node.h"
#else
#include <memory>

#include "torch/csrc/autograd/function.h"
#endif

namespace torch_tpu {

// Owning smart pointer for a `torch::autograd::Node` (`c10::intrusive_ptr` in
// PyTorch 2.13+, or `std::shared_ptr` in earlier OSS PyTorch versions).
#if TT_TORCH_VERSION_GE(2, 13)
template <typename T>
using AutogradNodePtr = c10::intrusive_ptr<T>;
#else
template <typename T>
using AutogradNodePtr = std::shared_ptr<T>;
#endif

// Creates an autograd `Node` smart pointer (`c10::intrusive_ptr<T>` in PyTorch
// 2.13+ where `Node` inherits from `c10::intrusive_ptr_target`, or
// `std::shared_ptr<T>` with `torch::autograd::deleteNode` in earlier PyTorch
// versions used by OSS builds).
// TODO: b/585211199 - Simplify to `c10::make_intrusive<T>()` once minimum OSS
// PyTorch is >= 2.13.0, where `Node` is a `c10::intrusive_ptr_target`.
template <std::derived_from<torch::autograd::Node> T>
[[nodiscard]] AutogradNodePtr<T> MakeAutogradNode() {
#if TT_TORCH_VERSION_GE(2, 13)
  return c10::make_intrusive<T>();
#else
  return std::shared_ptr<T>(new T(),
                            [](T* ptr) { torch::autograd::deleteNode(ptr); });
#endif
}

// Runs the callbacks of `torch.autograd.graph.node_creation_hook` (PyTorch
// 2.14+) on `node`, as generated kernels do once its edges and history are set.
template <std::derived_from<torch::autograd::Node> T>
void FireNodeCreationHooks([[maybe_unused]] const AutogradNodePtr<T>& node) {
#if TT_TORCH_VERSION_GE(2, 14)
  torch::autograd::fire_node_creation_hooks(node);
#endif
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_AUTOGRAD_UTILS_H_
