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

#include "csrc/common/version.h"

#include <cstdint>

#include "absl/log/absl_log.h"

namespace torch_tpu {

void Version::DieOnNegativeComponent(int major, int minor, int patch,
                                     int subpatch, int64_t dev) {
  ABSL_LOG(FATAL)  // CRASH_OK=TorchTPU bug
      << "Version components must be non-negative, got major=" << major
      << ", minor=" << minor << ", patch=" << patch << ", subpatch=" << subpatch
      << ", dev=" << dev;
}

}  // namespace torch_tpu
