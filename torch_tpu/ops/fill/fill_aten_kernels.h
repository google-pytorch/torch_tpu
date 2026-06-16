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

#ifndef TORCH_TPU_OPS_FILL_FILL_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_FILL_FILL_ATEN_KERNELS_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Optional.h"

namespace torch_tpu {
at::Tensor& AtenZero_(at::Tensor& self);

at::Tensor& AtenFillScalar_(at::Tensor& self, const at::Scalar& value);

at::Tensor& AtenFillTensor_(at::Tensor& self, const at::Tensor& fill_value);

absl::StatusOr<at::Tensor> MakeFullTensor(
    at::IntArrayRef size, at::Scalar fill_value,
    c10::optional<at::ScalarType> dtype_opt = c10::nullopt,
    c10::optional<at::Device> device_opt = c10::nullopt);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_FILL_FILL_ATEN_KERNELS_H_
