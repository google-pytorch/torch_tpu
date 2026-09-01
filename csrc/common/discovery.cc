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
#include "csrc/common/discovery.h"

#include <cstdint>

#include "absl/base/no_destructor.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"

namespace torch_tpu {

const absl::StatusOr<int64_t>& GetPremappedBufferSizeFromEnvOnce() {
  static const absl::NoDestructor<absl::StatusOr<int64_t>>
      premapped_buffer_size([]() -> absl::StatusOr<int64_t> {
        const auto& env_val = GetEnvOnce<kTpuPremappedBufferSizeEnvVar>();
        if (!env_val.has_value()) {
          SetEnv<kTpuPremappedBufferSizeEnvVar>("0");
          return 0;
        }
        int64_t size;
        TT_RET_CHECK(absl::SimpleAtoi(*env_val, &size),
                     error::kFailedPrecondition)
            << "the " << kTpuPremappedBufferSizeEnvVar
            << " environment variable is not a valid integer: " << *env_val;
        return size;
      }());
  return *premapped_buffer_size;
}

}  // namespace torch_tpu
