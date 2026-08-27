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
#ifndef TORCH_TPU_OPS_LINALG_LU_LINALG_LU_KERNELS_H_
#define TORCH_TPU_OPS_LINALG_LU_LINALG_LU_KERNELS_H_

#include <tuple>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

// Computes LU factorization of a matrix or batch of matrices `A` using
// partial pivoting. Returns a tuple containing:
// - LU: tensor containing L and U factors. L is stored in the strictly lower
// part of the tensor (with implied 1s on the diagonal), and U is stored in the
// upper part, including the main diagonal.
// - pivots: tensor containing pivot indices (1-based indexing).
// - info: tensor containing for each batch, the index (1-based) of the first
// zero of the diagonal (or zero if no zeros are found).
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> AtenLinalgLuFactorExOut(
    const at::Tensor& a, bool pivot, bool check_errors, at::Tensor& lu,
    at::Tensor& pivots, at::Tensor& info);

// Unpacks LU factorization results into P, L, and U matrices.
// LU_data and LU_pivots are outputs from AtenLinalgLuFactorExOut.
// P is the permutation matrix, L is the lower triangular matrix, and U is the
// upper triangular matrix.
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> AtenLuUnpackOut(
    const at::Tensor& lu_data, const at::Tensor& lu_pivots, bool unpack_data,
    bool unpack_pivots, at::Tensor& p, at::Tensor& l, at::Tensor& u);

// Solves a system of linear equations given LU factorization of A.
// LU and pivots contain the LU factorization of A, B is the right-hand side.
// left: if true, solve C * X = B, otherwise solve X * C = B, where
// C = A if adjoint == false, and C = A^H if adjoing == true.
at::Tensor& AtenLinalgLuSolveOut(const at::Tensor& lu, const at::Tensor& pivots,
                                 const at::Tensor& b, bool left, bool adjoint,
                                 at::Tensor& out);

// Computes PLU factorization of a matrix or batch of matrices `A`.
// Returns P, L, U matrices. This is a combination of
// AtenLinalgLuFactorExOut and AtenLuUnpackOut.
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> AtenLinalgLuOut(
    const at::Tensor& a, bool pivot, at::Tensor& p, at::Tensor& l,
    at::Tensor& u);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_LINALG_LU_LINALG_LU_KERNELS_H_
