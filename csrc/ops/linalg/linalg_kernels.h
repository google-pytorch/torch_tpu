#ifndef TORCH_TPU_OPS_LINALG_LINALG_KERNELS_H_
#define TORCH_TPU_OPS_LINALG_LINALG_KERNELS_H_

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

#include <tuple>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

// Solves a system of linear equations and returns solution,
// LU factorization of A, pivots, and singularity info.
// left: if true, solve A * X = B, otherwise solve X * A = B.
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
AtenLinalgSolveExOut(const at::Tensor& a, const at::Tensor& b, bool left,
                     bool check_errors, at::Tensor& result, at::Tensor& lu,
                     at::Tensor& pivots, at::Tensor& info);

// Computes inverse of a matrix or batch of matrices `A`.
// Returns inverse and singularity info.
std::tuple<at::Tensor&, at::Tensor&> AtenLinalgInvExInverse(const at::Tensor& a,
                                                            bool check_errors,
                                                            at::Tensor& inverse,
                                                            at::Tensor& info);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_LINALG_LINALG_KERNELS_H_
