/*
 * Copyright 2025 Google LLC
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

#ifndef TORCH_TPU_OPS_EXPERIMENTAL_RAGGED_DOT_RAGGED_DOT_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_EXPERIMENTAL_RAGGED_DOT_RAGGED_DOT_ATEN_KERNELS_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// Ragged dot product of two arrays, useful for implementing Mixture-of-Experts
// models.
//
// This operation computes a ragged dot product of lhs and rhs, using
// group_sizes to specify the ragged structure.
//
// The ragged dot operation can be expressed as follows:
// Given lhs (m x k), rhs (g x k x n), and group_sizes (g),
// where m is the total number of samples, k is the feature dimension,
// g is the number of groups (experts), and n is the output dimension.
// The i-th element of group_sizes specifies the number of samples belonging
// to the i-th group. The sum of group_sizes must be equal to m.
//
// The result is an array of shape (m x n), computed by performing g
// dot products:
// result[start_i:end_i, :] = dot(lhs[start_i:end_i, :], rhs[i, :, :])
// where start_i = sum(group_sizes[:i]) and end_i = sum(group_sizes[:i+1]).
//
// Args:
//   lhs: A 2D tensor of shape (m, k).
//   rhs: A 3D tensor of shape (g, k, n).
//   group_sizes: A 1D tensor of shape (g), containing the size of each group.
//
// Returns:
//   A 2D tensor of shape (m, n).
at::Tensor AtenRaggedDot(const at::Tensor& lhs, const at::Tensor& rhs,
                         const at::Tensor& group_sizes);

// Ragged dot product of two arrays (out variant).
//
// This operation computes a ragged dot product of lhs and rhs, using
// group_sizes to specify the ragged structure and writes the result to out.
//
// Args:
//   lhs: A 2D tensor of shape (m, k).
//   rhs: A 3D tensor of shape (g, k, n).
//   group_sizes: A 1D tensor of shape (g), containing the size of each group.
//   out: A 2D tensor of shape (m, n) to store the result.
//
// Returns:
//   A 2D tensor of shape (m, n).
at::Tensor& AtenRaggedDotOut(const at::Tensor& lhs, const at::Tensor& rhs,
                             const at::Tensor& group_sizes, at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_EXPERIMENTAL_RAGGED_DOT_RAGGED_DOT_ATEN_KERNELS_H_
