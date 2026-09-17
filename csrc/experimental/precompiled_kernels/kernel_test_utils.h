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

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_KERNEL_TEST_UTILS_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_KERNEL_TEST_UTILS_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

// Load the test's pinned PJRT plugin and create a client for the attached TPU.
absl::StatusOr<std::unique_ptr<xla::PjRtClient>> CreateKernelTestClient();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_KERNEL_TEST_UTILS_H_
