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

#ifndef TORCH_TPU_CSRC_INTERNAL_DEVICE_VMEM_CAPACITY_H_
#define TORCH_TPU_CSRC_INTERNAL_DEVICE_VMEM_CAPACITY_H_

#include <cstdint>

namespace torch_tpu {

// VMEM capacity of the local device. Kernels choose how much of it to request
// as Mosaic's scoped VMEM limit.
int64_t GetDeviceVmemCapacityBytes();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_INTERNAL_DEVICE_VMEM_CAPACITY_H_
