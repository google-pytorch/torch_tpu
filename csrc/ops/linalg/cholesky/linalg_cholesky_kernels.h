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

#ifndef TORCH_TPU_CSRC_OPS_LINALG_CHOLESKY_LINALG_CHOLESKY_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_LINALG_CHOLESKY_LINALG_CHOLESKY_KERNELS_H_

#include <tuple>

#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// Computes the Cholesky decomposition of a symmetric/Hermitian
// positive-definite matrix (or batch of matrices) `self` of shape `(..., n,
// n)`. Writes the lower-triangular (`upper=false`) or upper-triangular
// (`upper=true`) factor into `l` and the LAPACK-style status code (`0` on
// success, or the 1-based index of the first failing leading minor) into
// `info`. When `check_errors` is true, raises a `LinAlgError` if any `info`
// entry is non-zero.
std::tuple<at::Tensor&, at::Tensor&> AtenLinalgCholeskyExL(
    const at::Tensor& self, bool upper, bool check_errors, at::Tensor& l,
    at::Tensor& info);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_LINALG_CHOLESKY_LINALG_CHOLESKY_KERNELS_H_
