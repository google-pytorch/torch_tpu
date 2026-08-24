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

#ifndef TORCH_TPU_OPS_ALL_ANY_ALL_ANY_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_ALL_ANY_ALL_ANY_ATEN_KERNELS_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// Kernels for the 'all' operation
at::Tensor& AtenAllAllOut(const at::Tensor& self, at::Tensor& out);

at::Tensor& AtenAllOut(const at::Tensor& self, int64_t dim, bool keep_dim,
                       at::Tensor& out);

// Kernels for the 'any' operation
at::Tensor& AtenAnyAllOut(const at::Tensor& self, at::Tensor& out);

at::Tensor& AtenAnyOut(const at::Tensor& self, int64_t dim, bool keep_dim,
                       at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_ALL_ANY_ALL_ANY_ATEN_KERNELS_H_
