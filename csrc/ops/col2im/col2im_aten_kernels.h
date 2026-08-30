// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TORCH_TPU_OPS_COL2IM_COL2IM_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_COL2IM_COL2IM_ATEN_KERNELS_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

at::Tensor AtenCol2Im(const at::Tensor& input, at::IntArrayRef output_size,
                      at::IntArrayRef kernel_size, at::IntArrayRef dilation,
                      at::IntArrayRef padding, at::IntArrayRef stride);

at::Tensor& AtenCol2ImOut(const at::Tensor& input, at::IntArrayRef output_size,
                          at::IntArrayRef kernel_size, at::IntArrayRef dilation,
                          at::IntArrayRef padding, at::IntArrayRef stride,
                          at::Tensor& output);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_COL2IM_COL2IM_ATEN_KERNELS_H_
