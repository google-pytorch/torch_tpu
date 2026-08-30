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

#ifndef TORCH_TPU_OPS_EXPERIMENTAL_RAGGED_ALL_TO_ALL_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_EXPERIMENTAL_RAGGED_ALL_TO_ALL_ATEN_KERNELS_H_

#include <string_view>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// RaggedAllToAll collective operation.
//
// This operation communicates slices of ragged tensors across devices.
//
// Args:
//   operand: The input tensor.
//   output: The output tensor (used to specify shape and sharding).
//   input_offsets: Offsets for input slices.
//   send_sizes: Sizes of slices to send.
//   output_offsets: Offsets for output slices.
//   recv_sizes: Sizes of slices to receive.
//
// Returns:
//   A tensor containing the gathered data.
at::Tensor AtenRaggedAllToAll(const at::Tensor& operand,
                              const at::Tensor& output,
                              const at::Tensor& input_offsets,
                              const at::Tensor& send_sizes,
                              const at::Tensor& output_offsets,
                              const at::Tensor& recv_sizes,
                              std::string_view process_group_name);

// RaggedAllToAll collective operation (out variant).
at::Tensor& AtenRaggedAllToAllOut(
    const at::Tensor& operand, const at::Tensor& output,
    const at::Tensor& input_offsets, const at::Tensor& send_sizes,
    const at::Tensor& output_offsets, const at::Tensor& recv_sizes,
    std::string_view process_group_name, at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_EXPERIMENTAL_RAGGED_ALL_TO_ALL_ATEN_KERNELS_H_
