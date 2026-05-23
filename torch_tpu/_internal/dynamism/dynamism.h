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

#ifndef TORCH_TPU__INTERNAL_DYNAMISM_DYNAMISM_H_
#define TORCH_TPU__INTERNAL_DYNAMISM_DYNAMISM_H_

#include <cstdint>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/common/shape.h"

namespace torch_tpu {

// Marks a dimension of the given tensor as dynamic.
// The dimension must be in bounds for the tensor.
// The lower and upper bounds must be in bounds for the dimension as well.
//
// If the input tensor is a view tensor, then the operation creates a
// DeferredOp for the view and returns a new tensor holding the DeferredOp,
// otherwise it returns the input tensor.
absl::StatusOr<at::Tensor> MarkDynamic(const at::Tensor& tensor,
                                       int64_t dimension, int64_t lower_bound,
                                       int64_t upper_bound);

absl::StatusOr<absl::Span<const BoundedDynamicDimension>> GetDynamismInfo(
    const at::Tensor& tensor);

}  // namespace torch_tpu

#endif  // TORCH_TPU__INTERNAL_DYNAMISM_DYNAMISM_H_
