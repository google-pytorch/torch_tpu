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

#ifndef TORCH_TPU_OPS_A_MIN_MAX_A_MIN_MAX_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_A_MIN_MAX_A_MIN_MAX_ATEN_KERNELS_H_

#include <cstdint>
#include <tuple>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/Optional.h"

namespace torch_tpu {

at::Tensor& AtenAmaxOut(const at::Tensor& self, at::IntArrayRef dims,
                        bool keep_dim, at::Tensor& out);

at::Tensor& AtenAminOut(const at::Tensor& self, at::IntArrayRef dims,
                        bool keep_dim, at::Tensor& out);

std::tuple<at::Tensor&, at::Tensor&> AtenAminmaxOut(const at::Tensor& self,
                                                    c10::optional<int64_t> dim,
                                                    bool keep_dim,
                                                    at::Tensor& min,
                                                    at::Tensor& max);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_A_MIN_MAX_A_MIN_MAX_ATEN_KERNELS_H_
