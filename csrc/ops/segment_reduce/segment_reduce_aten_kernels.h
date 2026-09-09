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

#ifndef TORCH_TPU_CSRC_OPS_SEGMENT_REDUCE_SEGMENT_REDUCE_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_SEGMENT_REDUCE_SEGMENT_REDUCE_ATEN_KERNELS_H_

#include <cstdint>
#include <optional>

#include "ATen/core/Scalar.h"
#include "ATen/core/Tensor.h"
#include "c10/util/string_view.h"

namespace torch_tpu {

at::Tensor AtenSegmentReduce(const at::Tensor& data, c10::string_view reduce,
                             const std::optional<at::Tensor>& lengths,
                             const std::optional<at::Tensor>& indices,
                             const std::optional<at::Tensor>& offsets,
                             int64_t axis, bool unsafe,
                             const std::optional<at::Scalar>& initial);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SEGMENT_REDUCE_SEGMENT_REDUCE_ATEN_KERNELS_H_
