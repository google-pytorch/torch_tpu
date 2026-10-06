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

// Performs matrix multiplication across arbitrary tensor ranks (1D, 2D, 3D,
// 4D+) with broadcasting of batch dimensions.
at::Tensor AtenMatmul(const at::Tensor& self, const at::Tensor& other);

// Autograd wrapper for `AtenMatmul` supporting both real and complex tensors.
at::Tensor AtenMatmulAutograd(const at::Tensor& self, const at::Tensor& other);

// Out-variant of `AtenMatmul`.
at::Tensor& AtenMatmulOut(const at::Tensor& self, const at::Tensor& other,
                          at::Tensor& out);

// Computes gradients with respect to `self` and `other` for `aten::matmul`.
std::tuple<at::Tensor, at::Tensor> AtenMatmulBackward(const at::Tensor& grad,
                                                      const at::Tensor& self,
                                                      const at::Tensor& other,
                                                      std::array<bool, 2> mask);

// Out-variant of `AtenMatmulBackward`.
std::tuple<at::Tensor&, at::Tensor&> AtenMatmulBackwardOut(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    std::array<bool, 2> mask, at::Tensor& grad_self, at::Tensor& grad_other);

// Autograd wrapper for `AtenMatmulBackward` supporting higher-order gradients.
std::tuple<at::Tensor, at::Tensor> AtenMatmulBackwardAutograd(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    std::array<bool, 2> mask);

// Meta implementation of `aten::matmul_backward`.
std::tuple<at::Tensor, at::Tensor> MetaMatmulBackward(const at::Tensor& grad,
                                                      const at::Tensor& self,
                                                      const at::Tensor& other,
                                                      std::array<bool, 2> mask);

// Meta implementation of `aten::matmul_backward.out`.
std::tuple<at::Tensor&, at::Tensor&> MetaMatmulBackwardOut(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    std::array<bool, 2> mask, at::Tensor& grad_self, at::Tensor& grad_other);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_MATMUL_MATMUL_ATEN_KERNELS_H_
