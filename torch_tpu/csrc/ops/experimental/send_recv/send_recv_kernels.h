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

#ifndef TORCH_TPU_OPS_EXPERIMENTAL_SEND_RECV_SEND_RECV_KERNELS_H_
#define TORCH_TPU_OPS_EXPERIMENTAL_SEND_RECV_SEND_RECV_KERNELS_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/ivalue.h"

namespace torch_tpu {

c10::IValue TorchTpuExperimentalSend(at::ITensorListRef tensors, int64_t dst,
                                     int64_t tag);

c10::IValue TorchTpuExperimentalRecv(at::ITensorListRef tensors, int64_t src,
                                     int64_t tag);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_EXPERIMENTAL_SEND_RECV_SEND_RECV_KERNELS_H_
