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

#ifndef TORCH_TPU_OPS_INDEX_FILL_INDEX_FILL_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_INDEX_FILL_INDEX_FILL_ATEN_KERNELS_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

at::Tensor& AtenIndexFillIntScalar_(at::Tensor& self, int64_t dim,
                                    const at::Tensor& index,
                                    const at::Scalar& value);

at::Tensor& AtenIndexFillIntTensor_(at::Tensor& self, int64_t dim,
                                    const at::Tensor& index,
                                    const at::Tensor& value);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_INDEX_FILL_INDEX_FILL_ATEN_KERNELS_H_
