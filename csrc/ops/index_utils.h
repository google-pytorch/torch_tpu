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

#ifndef TORCH_TPU_CSRC_OPS_INDEX_UTILS_H_
#define TORCH_TPU_CSRC_OPS_INDEX_UTILS_H_

#include <cstdint>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"

namespace torch_tpu {

// Resolves negative indices by converting them to positive indices
absl::Status ResolveNegativeIndices(std::vector<at::Tensor>& indices,
                                    at::IntArrayRef sizes,
                                    const Indices& dimensions);

absl::StatusOr<int64_t> ValidateIndexInputsAndGetDim(const at::Tensor& self,
                                                     int64_t dim,
                                                     const at::Tensor& index,
                                                     const at::Tensor& source);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_INDEX_UTILS_H_
