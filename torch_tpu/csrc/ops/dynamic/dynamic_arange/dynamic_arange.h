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

#ifndef TORCH_TPU_OPS_DYNAMIC_DYNAMIC_ARANGE_H_
#define TORCH_TPU_OPS_DYNAMIC_DYNAMIC_ARANGE_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

//
// This op is a torch_tpu custom op for use in torch.compile() mode to handle
// dynamic arange operations on TPU. It generates a sequence of numbers
// starting from `start`, up to (but not including) `end`, by `step`. Since
// the length of the sequence can be dynamic, it takes a `max_length` to
// allocate a static tensor of that size, and then uses
// `stablehlo.set_dimension_size` to set the runtime size.
//
// Args:
//   start: 0-D tensor containing the start value.
//   end: 0-D tensor containing the end value.
//   step: 0-D tensor containing the step value.
//   max_length: Static upper bound for the maximum number of elements. This is
//     used for static allocation.
//   dtype: The desired data type of the output tensor.
//
// Returns:
//   A 1-D tensor containing the arange sequence, bounded to its dynamic length.
at::Tensor DynamicArange(const at::Tensor& start, const at::Tensor& end,
                         const at::Tensor& step, int64_t max_length,
                         at::ScalarType dtype);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_DYNAMIC_DYNAMIC_ARANGE_H_
