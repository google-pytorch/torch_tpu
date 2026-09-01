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

#ifndef TORCH_TPU_OPS_NORMAL_NORMAL_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_NORMAL_NORMAL_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/DeprecatedTypeProperties.h"

namespace torch_tpu {

at::Tensor& AtenNormal_(at::Tensor& self, double mean, double std,
                        std::optional<at::Generator> generator);

at::Tensor AtenNormalFloatTensor(double mean, const at::Tensor& std,
                                 std::optional<at::Generator> generator);

at::Tensor& AtenNormalFloatTensorOut(double mean, const at::Tensor& std,
                                     std::optional<at::Generator> generator,
                                     at::Tensor& out);

at::Tensor AtenNormalTensorFloat(const at::Tensor& mean, double std,
                                 std::optional<at::Generator> generator);

at::Tensor& AtenNormalTensorFloatOut(const at::Tensor& mean, double std,
                                     std::optional<at::Generator> generator,
                                     at::Tensor& out);

at::Tensor AtenNormalTensorTensor(const at::Tensor& mean, const at::Tensor& std,
                                  std::optional<at::Generator> generator);

at::Tensor& AtenNormalTensorTensorOut(const at::Tensor& mean,
                                      const at::Tensor& std,
                                      std::optional<at::Generator> generator,
                                      at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_NORMAL_NORMAL_ATEN_KERNELS_H_
