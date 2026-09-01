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

#ifndef TORCH_TPU_OPS_AS_STRIDED_AS_STRIDED_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_AS_STRIDED_AS_STRIDED_ATEN_KERNELS_H_

#include <cstdint>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Optional.h"

namespace torch_tpu {

at::Tensor AtenAsStrided(const at::Tensor& self, c10::SymIntArrayRef size_sym,
                         c10::SymIntArrayRef stride_sym,
                         c10::optional<c10::SymInt> storage_offset_sym_opt);

// Creates a view of the target tensor with the given size, stride, and
// storage offset.
absl::StatusOr<at::Tensor> AsStrided(const at::Tensor& self,
                                     c10::IntArrayRef size,
                                     c10::IntArrayRef stride,
                                     int64_t storage_offset);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_AS_STRIDED_AS_STRIDED_ATEN_KERNELS_H_
