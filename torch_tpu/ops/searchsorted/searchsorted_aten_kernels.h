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

#ifndef TORCH_TPU_OPS_SEARCHSORTED_SEARCHSORTED_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_SEARCHSORTED_SEARCHSORTED_ATEN_KERNELS_H_

#include <optional>
#include <string_view>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

at::Tensor AtenSearchsortedTensor(const at::Tensor& sorted_sequence,
                                  const at::Tensor& self, bool out_int32,
                                  bool right,
                                  std::optional<std::string_view> side,
                                  const std::optional<at::Tensor>& sorter);

at::Tensor& AtenSearchsortedTensorOut(const at::Tensor& sorted_sequence,
                                      const at::Tensor& self, bool out_int32,
                                      bool right,
                                      std::optional<std::string_view> side,
                                      const std::optional<at::Tensor>& sorter,
                                      at::Tensor& out);

at::Tensor AtenSearchsortedScalar(const at::Tensor& sorted_sequence,
                                  const at::Scalar& self, bool out_int32,
                                  bool right,
                                  std::optional<std::string_view> side,
                                  const std::optional<at::Tensor>& sorter);

at::Tensor& AtenSearchsortedScalarOut(const at::Tensor& sorted_sequence,
                                      const at::Scalar& self, bool out_int32,
                                      bool right,
                                      std::optional<std::string_view> side,
                                      const std::optional<at::Tensor>& sorter,
                                      at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_SEARCHSORTED_SEARCHSORTED_ATEN_KERNELS_H_
