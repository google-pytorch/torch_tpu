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

#ifndef TORCH_TPU_CSRC_OPS_MATMUL_MATMUL_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_MATMUL_MATMUL_ATEN_KERNELS_H_

#include <array>
#include <tuple>

#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// PrivateUse1 kernel for `aten::matmul(self, other)`. Validates dtypes/shapes
// with upstream PyTorch parity and lowers across all 5 rank combinations
// (`1D x 1D`, `2D x 2D`, `ND x 1D`, `1D x ND`, `ND x ND`) with NumPy-style
// batch broadcasting.
at::Tensor AtenMatmul(const at::Tensor& self, const at::Tensor& other);

// AutogradPrivateUse1 wrapper for `aten::matmul(self, other)`. Attaches
// `MatmulAutogradNode` so backward passes dispatch directly to our native
// `aten::matmul_backward` TPU kernel for both real and complex tensors, while
// falling back to `DecomposeMatmul` for FakeTensors, functorch transforms,
// Python wrapper tensor subclasses, and forward-mode AD.
at::Tensor AtenMatmulAutograd(const at::Tensor& self, const at::Tensor& other);

// PrivateUse1 out-variant kernel for `aten::matmul.out(self, other, *, out)`.
at::Tensor& AtenMatmulOut(const at::Tensor& self, const at::Tensor& other,
                          at::Tensor& out);

// AutogradPrivateUse1 wrapper for `aten::matmul.out(self, other, *, out)`.
// Rejects calls where any operand or `out` requires gradients under `GradMode`
// (matching upstream PyTorch), falls back to `at::native::matmul_out` for
// functorch transforms, Python wrapper subclasses, and forward-mode AD, and
// otherwise redispatches below `Autograd` so `ADInplaceOrView` updates `out`'s
// version counter.
at::Tensor& AtenMatmulOutAutograd(const at::Tensor& self,
                                  const at::Tensor& other, at::Tensor& out);

// PrivateUse1 kernel for `aten::matmul_backward(grad, self, other, mask)`.
// Computes `(grad_self, grad_other)` gated by `mask = {compute_self,
// compute_other}`, omitting unused forward tensor buffers from XLA compilation
// when only a single gradient is requested and shapes are static.
std::tuple<at::Tensor, at::Tensor> AtenMatmulBackward(const at::Tensor& grad,
                                                      const at::Tensor& self,
                                                      const at::Tensor& other,
                                                      std::array<bool, 2> mask);

// PrivateUse1 out-variant kernel for `aten::matmul_backward.out`. Writes
// requested gradients into `grad_self` and/or `grad_other` according to `mask`.
std::tuple<at::Tensor&, at::Tensor&> AtenMatmulBackwardOut(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    std::array<bool, 2> mask, at::Tensor& grad_self, at::Tensor& grad_other);

// AutogradPrivateUse1 wrapper for `aten::matmul_backward`. Attaches
// `MatmulBackwardAutogradNode` to support higher-order (double-backward)
// gradients through `aten::matmul` and `aten::matmul_backward`.
std::tuple<at::Tensor, at::Tensor> AtenMatmulBackwardAutograd(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    std::array<bool, 2> mask);

// Meta kernel for `aten::matmul_backward`. Validates inputs and allocates
// symbolic/meta output tensors matching `self.sym_sizes()` and
// `other.sym_sizes()` when `mask[0]` / `mask[1]` are true, without dispatching
// to XLA.
std::tuple<at::Tensor, at::Tensor> MetaMatmulBackward(const at::Tensor& grad,
                                                      const at::Tensor& self,
                                                      const at::Tensor& other,
                                                      std::array<bool, 2> mask);

// Meta kernel for `aten::matmul_backward.out`. Validates inputs and resizes
// defined `grad_self` / `grad_other` meta tensors to match `self` / `other`
// symbolic dimensions when `mask[0]` / `mask[1]` are true.
std::tuple<at::Tensor&, at::Tensor&> MetaMatmulBackwardOut(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    std::array<bool, 2> mask, at::Tensor& grad_self, at::Tensor& grad_other);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_MATMUL_MATMUL_ATEN_KERNELS_H_
