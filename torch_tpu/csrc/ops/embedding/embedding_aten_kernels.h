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

#ifndef TORCH_TPU_OPS_EMBEDDING_EMBEDDING_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_EMBEDDING_EMBEDDING_ATEN_KERNELS_H_

#include <cstdint>
#include <optional>
#include <tuple>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> AtenEmbeddingBag(
    const at::Tensor& weight, const at::Tensor& indices,
    const at::Tensor& offsets, bool scale_grad_by_freq, int64_t mode,
    bool sparse, const std::optional<at::Tensor>& per_sample_weights,
    bool include_last_offset, int64_t padding_idx);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenEmbeddingBagForwardOnly(const at::Tensor& weight, const at::Tensor& indices,
                            const at::Tensor& offsets, bool scale_grad_by_freq,
                            int64_t mode, bool sparse,
                            const std::optional<at::Tensor>& per_sample_weights,
                            bool include_last_offset, int64_t padding_idx);

at::Tensor& AtenEmbeddingRenorm_(at::Tensor& self, const at::Tensor& indices,
                                 double max_norm, double norm_type);

at::Tensor AtenEmbeddingBagBackward(
    const at::Tensor& grad, const at::Tensor& indices,
    const at::Tensor& offsets, const at::Tensor& offset2bag,
    const at::Tensor& bag_size, const at::Tensor& max_indices,
    at::SymInt num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse,
    const std::optional<at::Tensor>& per_sample_weights, int64_t padding_idx);

// This implementation is adapted from:
// https://github.com/pytorch/pytorch/blob/5b764267f4446e92e4072fb4faf128da6208cf8f/torch/_decomp/decompositions.py#L1247-L1273
at::Tensor AtenEmbeddingDenseBackward(const at::Tensor& grad_output,
                                      const at::Tensor& indices,
                                      at::SymInt num_weights,
                                      at::SymInt padding_idx,
                                      bool scale_grad_by_freq);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_EMBEDDING_EMBEDDING_ATEN_KERNELS_H_
