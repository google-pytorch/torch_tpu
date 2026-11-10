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

#ifndef TORCH_TPU_CSRC_COMMON_FAKE_TENSOR_UTILS_H_
#define TORCH_TPU_CSRC_COMMON_FAKE_TENSOR_UTILS_H_

#include "csrc/common/const_tensor.h"

namespace torch_tpu {

// Returns true if a `functorch` interpreter is active (e.g., inside
// `torch.vmap`, `torch.func.grad` or `make_fx`), i.e. the thread-local
// dispatch key set (`tls_local_dispatch_key_set`) contains
// `FuncTorchDynamicLayerFrontMode` or `FuncTorchDynamicLayerBackMode`. The
// interpreter must then intercept primitive ATen operations rather than opaque
// backend kernels.
[[nodiscard]] bool IsFunctorchActive();

// Returns true if `tensor` is a `FakeTensor`, `Meta` tensor, or wrapped inside
// a `functorch` dynamic layer (e.g., `torch.vmap`, `torch.func.grad`,
// `make_fx`).
//
// Checks two conditions:
//   1. A functorch interpreter is active (`IsFunctorchActive()`).
//   2. The tensor has no storage data (`!tensor.has_storage_data()`), which
//      holds for `FakeTensor` and `Meta` tensors during `torch.compile`
//      tracing. Note that this also holds for real tensors without any bytes,
//      e.g. zero-element CPU tensors.
//
// Never throws, and never guards on (and thereby specializes) symbolic shapes.
[[nodiscard]] bool IsFakeOrFunctorchTensor(const ConstTensor& tensor);

// Returns true if `tensor` is a `Meta` tensor, a `FakeTensor` (used by
// `torch.compile` and `torch.export`, backed by `DeviceType::Meta` storage),
// or an AOTAutograd `FunctionalTensor` (used during `torch.compile` under
// `torch.no_grad()` or training, whose storage sets `throw_on_mutable_data_ptr`
// while `TorchDispatchModeKey::FUNCTIONAL` is active).
//
// Never throws, and never guards on (and thereby specializes) symbolic shapes.
[[nodiscard]] bool IsMetaOrFakeOrFunctionalTensor(const ConstTensor& tensor);

// Returns true if `tensor` is a Python wrapper tensor subclass (such as
// `DTensor` or `NestedTensor`) that should delegate to upstream composite
// decompositions rather than opaque TPU `AutogradPrivateUse1` kernels.
// Distinguishes user/distributed wrapper subclasses (even when wrapped inside
// `aot_dispatch_subclass` under `FunctionalTensorMode`) from `FakeTensor` and
// AOTAutograd `FunctionalTensor`.
//
// Never throws, and never guards on (and thereby specializes) symbolic shapes.
[[nodiscard]] bool IsPythonWrapperSubclass(const ConstTensor& tensor);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_FAKE_TENSOR_UTILS_H_
