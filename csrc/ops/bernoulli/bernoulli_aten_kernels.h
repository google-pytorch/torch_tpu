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

#ifndef TORCH_TPU_OPS_BERNOULLI_BERNOULLI_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_BERNOULLI_BERNOULLI_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

at::Tensor& AtenBernoulliOut(const at::Tensor& self,
                             std::optional<at::Generator> generator,
                             at::Tensor& out);

at::Tensor& AtenBernoulli_Float(at::Tensor& self, double p,
                                std::optional<at::Generator> generator);

at::Tensor& AtenBernoulli_Tensor(at::Tensor& self, const at::Tensor& p,
                                 std::optional<at::Generator> generator);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_BERNOULLI_BERNOULLI_ATEN_KERNELS_H_
