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

#ifndef TORCH_TPU_OPS_CLAMP_CLAMP_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_CLAMP_CLAMP_ATEN_KERNELS_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/Optional.h"

namespace torch_tpu {

at::Tensor& AtenClampOut(const at::Tensor& self,
                         const c10::optional<at::Scalar>& min,
                         const c10::optional<at::Scalar>& max, at::Tensor& out);

at::Tensor& AtenClampMinOut(const at::Tensor& self, const at::Scalar& min,
                            at::Tensor& out);

at::Tensor& AtenClampMaxOut(const at::Tensor& self, const at::Scalar& max,
                            at::Tensor& out);

at::Tensor& AtenClampTensorOut(const at::Tensor& self,
                               const c10::optional<at::Tensor>& min,
                               const c10::optional<at::Tensor>& max,
                               at::Tensor& out);

at::Tensor& AtenClampMinTensorOut(const at::Tensor& self, const at::Tensor& min,
                                  at::Tensor& out);

at::Tensor& AtenClampMaxTensorOut(const at::Tensor& self, const at::Tensor& max,
                                  at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_CLAMP_CLAMP_ATEN_KERNELS_H_
