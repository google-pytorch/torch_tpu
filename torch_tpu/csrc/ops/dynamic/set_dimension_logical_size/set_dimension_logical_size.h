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

#ifndef TORCH_TPU_OPS_DYNAMIC_SET_DIMENSION_LOGICAL_SIZE_H_
#define TORCH_TPU_OPS_DYNAMIC_SET_DIMENSION_LOGICAL_SIZE_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

// This op is a torch_tpu custom op for use in torch.compile() mode to handle
// dynamic tensor shapes on TPU. It lowers down to stablehlo.set_dimension_size
// which XLA uses to determine the runtime size of the padded tensor dimension.
// Args:
//   input: The input tensor to set the dimension size of.
//   dim: The padded dimension to set the size of. If negative, it counts from
//     the end of dimensions.
//   size: The size tensor that contains the runtime size of the padded
//     dimension. Must be a 0-D tensor.
// Returns:
//   The input tensor with the first `size` elements of the specified dimension
//   being valid and the rest being undefined.
at::Tensor SetDimensionLogicalSize(const at::Tensor& input, int64_t dim,
                                   const at::Tensor& size);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_DYNAMIC_SET_DIMENSION_LOGICAL_SIZE_H_
