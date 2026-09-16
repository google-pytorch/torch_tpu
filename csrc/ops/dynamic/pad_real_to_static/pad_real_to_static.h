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

#ifndef TORCH_TPU_CSRC_OPS_DYNAMIC_PAD_REAL_TO_STATIC_PAD_REAL_TO_STATIC_H_
#define TORCH_TPU_CSRC_OPS_DYNAMIC_PAD_REAL_TO_STATIC_PAD_REAL_TO_STATIC_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/List.h"

namespace torch_tpu {

// This op is a torch_tpu custom op for use in torch.compile() mode to pad
// dynamic tensors as part of the model module itself. For dynamic dimensions,
// the runtime size would be different from run to run and hence provided as
// scalar inputs to the op along with size for the static dimensions.
//
// Args:
//   input: The input tensor to pad.
//   runtime_shape: List of 0-D (scalar) int32 tensors containing the runtime
//     sizes for each dimension (0 to rank - 1).
//   static_shape: The static upper bound for the shape.
//   is_dynamic: List of booleans indicating which dimensions are dynamic.
// Returns:
//   The padded tensor, bounded to its dynamic shape.
at::Tensor PadRealToStatic(const at::Tensor& input,
                           at::TensorList runtime_shape,
                           at::IntArrayRef static_shape,
                           c10::List<bool> is_dynamic);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_DYNAMIC_PAD_REAL_TO_STATIC_PAD_REAL_TO_STATIC_H_
