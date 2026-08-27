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

#ifndef TORCH_TPU_OPS_DYNAMIC_DYNAMIC_RESHAPE_H_
#define TORCH_TPU_OPS_DYNAMIC_DYNAMIC_RESHAPE_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/List.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// Reshapes the input tensor to a dynamic output shape.
//
// This operator is used in torch.compile() mode to support reshaping
// tensors with bounded dynamism on TPU. It lowers to
// stablehlo.dynamic_reshape.
//
// Args:
//   input: The input tensor to reshape.
//   shape: List of 0-D (scalar) int32 tensors containing the runtime sizes of
//     the output dimensions.
//   static_shape: The static upper bound for the output shape. Used for static
//     allocation.
//   is_dynamic: List of booleans indicating which output dimensions are
//     dynamic.
//
// Returns:
//   The reshaped tensor, bounded to its dynamic shape.
at::Tensor DynamicReshape(const at::Tensor& input, at::TensorList shape,
                          at::IntArrayRef static_shape,
                          c10::List<bool> is_dynamic);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_DYNAMIC_DYNAMIC_RESHAPE_H_
