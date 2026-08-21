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

#ifndef TORCH_TPU_OPS_MIN_MAX_MIN_MAX_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_MIN_MAX_MIN_MAX_ATEN_KERNELS_H_

#include <cstdint>
#include <optional>
#include <tuple>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/util/Optional.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/ops/min_max/min_max.h"
#include "torch_tpu/csrc/ops/op_names.h"

namespace torch_tpu {

// Computes the global minimum or maximum of the input tensor on the TPU
// natively. This is an internal helper designed for use within other native
// operators. Returns a `DeviceBufferRef` pointing to the 1-element result
// tensor.
absl::StatusOr<DeviceBufferRef> DispatchUnaryMinMax(
    const at::Tensor& self, MinMaxOp op,
    std::optional<OpName> op_name = std::nullopt);

at::Tensor& AtenArgmaxOut(const at::Tensor& self, c10::optional<int64_t> dim,
                          bool keep_dim, at::Tensor& out);

at::Tensor& AtenArgminOut(const at::Tensor& self, c10::optional<int64_t> dim,
                          bool keep_dim, at::Tensor& out);

at::Tensor AtenMax(const at::Tensor& self);

at::Tensor& AtenMaxUnaryOut(const at::Tensor& self, at::Tensor& out);

std::tuple<at::Tensor&, at::Tensor&> AtenMaxDimMax(const at::Tensor& self,
                                                   int64_t dim, bool keep_dim,
                                                   at::Tensor& max,
                                                   at::Tensor& max_indices);

at::Tensor AtenMin(const at::Tensor& self);

std::tuple<at::Tensor&, at::Tensor&> AtenMinDimMin(const at::Tensor& self,
                                                   int64_t dim, bool keep_dim,
                                                   at::Tensor& min,
                                                   at::Tensor& min_indices);

at::Tensor& AtenMinUnaryOut(const at::Tensor& self, at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_MIN_MAX_MIN_MAX_ATEN_KERNELS_H_
