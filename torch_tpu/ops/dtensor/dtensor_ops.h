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

#ifndef TORCH_TPU_OPS_DTENSOR_DTENSOR_OPS_H_
#define TORCH_TPU_OPS_DTENSOR_DTENSOR_OPS_H_

#include <cstdint>
#include <string_view>

#include "ATen/core/ATen_fwd.h"
#include "c10/util/ArrayRef.h"

namespace torch_tpu {

at::Tensor SdySharding(const at::Tensor& x, std::string_view mesh_sdy_format,
                       std::string_view placements_sdy_format,
                       at::IntArrayRef target_shape, bool is_input);

at::Tensor SdyShardingConstraint(const at::Tensor& x,
                                 std::string_view mesh_sdy_format,
                                 std::string_view placements_sdy_format);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_DTENSOR_DTENSOR_OPS_H_
