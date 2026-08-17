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

#ifndef TORCH_TPU_COMMON_ERROR_UTILS_TEST_HELPER_H_
#define TORCH_TPU_COMMON_ERROR_UTILS_TEST_HELPER_H_

// A file that triggers an error in the header, used for testing the stack trace
// cleaning functionality.

#include "absl/status/status.h"
#include "torch_tpu/csrc/common/error_utils.h"

namespace torch_tpu {

inline absl::Status MakeErrorFromHeader() {
  return TT_ERROR(error::kInternal) << "error from header";
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_ERROR_UTILS_TEST_HELPER_H_
