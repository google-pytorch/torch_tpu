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

#ifndef TORCH_TPU_CSRC_OPS_LINALG_LINALG_SVD_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_LINALG_LINALG_SVD_KERNELS_H_

#include <optional>
#include <tuple>

#include "ATen/core/TensorBody.h"
#include "c10/util/string_view.h"

namespace torch_tpu {

// Computes the singular value decomposition (SVD) of a matrix or batch of
// matrices: A = U @ diag(S) @ Vh.
//
// - U contains the left singular vectors (satisfying U^H @ U = I).
// - V contains the right singular vectors (satisfying V^H @ V = I), with Vh =
// V^H.
// - S is a vector of real non-negative singular values in descending order.
//
// Arguments:
//   self: Input tensor of shape (..., M, N) containing matrices to factorize.
//   full_matrices: If true, computes full U (..., M, M) and Vh (..., N, N).
//     If false, computes reduced U (..., M, K) and Vh (..., K, N) where K =
//     min(M, N).
//   compute_uv: If true, computes (U, S, Vh). If false, computes only s,
//     returning empty U and Vh.
//   driver: Name of the driver to use (only "gesvd" is supported).
//   u: Output tensor for left singular vectors U.
//   s: Output tensor for singular values S.
//   vh: Output tensor for right singular vectors Vh.
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> AtenLinalgSvdU(
    const at::Tensor& self, bool full_matrices, bool compute_uv,
    std::optional<c10::string_view> driver, at::Tensor& u, at::Tensor& s,
    at::Tensor& vh);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_LINALG_LINALG_SVD_KERNELS_H_
