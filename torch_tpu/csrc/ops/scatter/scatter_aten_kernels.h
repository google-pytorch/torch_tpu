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

#ifndef TORCH_TPU_OPS_SCATTER_SCATTER_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_SCATTER_SCATTER_ATEN_KERNELS_H_

#include <cstdint>
#include <string_view>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/string_view.h"

namespace torch_tpu {

at::Tensor& AtenScatterSrcOut(const at::Tensor& self, int64_t dim,
                              const at::Tensor& index, const at::Tensor& src,
                              at::Tensor& out);

at::Tensor& AtenScatterValueOut(const at::Tensor& self, int64_t dim,
                                const at::Tensor& index,
                                const at::Scalar& value, at::Tensor& out);

at::Tensor& AtenScatterReduceOut(const at::Tensor& self, int64_t dim,
                                 const at::Tensor& index, const at::Tensor& src,
                                 std::string_view reduction_op,
                                 at::Tensor& out);

at::Tensor& AtenScatterReduceTwoOut(const at::Tensor& self, int64_t dim,
                                    const at::Tensor& index,
                                    const at::Tensor& src,
                                    c10::string_view reduction,
                                    bool include_self, at::Tensor& out);

at::Tensor& AtenScatterValueReduceOut(const at::Tensor& self, int64_t dim,
                                      const at::Tensor& index,
                                      const at::Scalar& value,
                                      std::string_view reduction_op,
                                      at::Tensor& out);

at::Tensor& AtenScatterAddOut(const at::Tensor& self, int64_t dim,
                              const at::Tensor& index, const at::Tensor& src,
                              at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_SCATTER_SCATTER_ATEN_KERNELS_H_
