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
#include <cstdlib>
#include <optional>

#include "absl/base/no_destructor.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/premapped_buffer_defaults.h"
#include "csrc/common/warnings.h"

namespace torch_tpu {

const absl::StatusOr<int64_t>& GetPremappedBufferSizeFromEnvOnce() {
  static const absl::NoDestructor<absl::StatusOr<int64_t>>
      premapped_buffer_size([]() -> absl::StatusOr<int64_t> {
        // Read with std::getenv() rather than GetEnvOnce<>(), since the env var
        // is set below if it's unset. The result is memoized here instead.
        const char* const raw_val =
            std::getenv(  // GETENV_OK=the env var is set below.
                kTpuPremappedBufferSizeEnvVar);
        if (raw_val == nullptr) {
          const std::optional<int64_t> memory_limit_bytes =
              GetEffectiveMemoryLimitBytes();
          const int64_t local_workers = EstimateLocalWorkerCount();
          const int64_t default_bytes = ComputeDefaultPremappedBufferSizeBytes(
              memory_limit_bytes, local_workers);
          SafeSetEnv<kTpuPremappedBufferSizeEnvVar>(
              absl::StrCat(default_bytes));
          ABSL_LOG(INFO) << "TPU_PREMAPPED_BUFFER_SIZE is unset; defaulting to "
                         << default_bytes << " bytes (memory_limit_bytes="
                         << (memory_limit_bytes.has_value()
                                 ? absl::StrCat(*memory_limit_bytes)
                                 : "unknown")
                         << ", local_workers=" << local_workers << ").";
          if (default_bytes == 0) {
            TT_WARN(WarningCategory::kPerformance,
                    "TPU_PREMAPPED_BUFFER_SIZE is unset and its automatic "
                    "default is 0 because the host memory limit (",
                    (memory_limit_bytes.has_value()
                         ? absl::StrCat(*memory_limit_bytes, " bytes")
                         : "unknown"),
                    ") is too small to share among ", local_workers,
                    " local TorchTPU process(es), so transfers between host "
                    "and TPU use a slower path. To use a premapped buffer, "
                    "set TPU_PREMAPPED_BUFFER_SIZE to a size in bytes that "
                    "fits in host memory (e.g. 1073741824 for 1 GiB).");
          }
          return default_bytes;
        }
        int64_t env_val = 0;
        TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Status is swallowed by
                       // .value_or(0) in pjrt_state.cc; covered in C++
                       // discovery_test.
            absl::SimpleAtoi(raw_val, &env_val), error::kFailedPrecondition)
            << "the " << kTpuPremappedBufferSizeEnvVar
            << " environment variable is not a valid integer: " << raw_val;
        return env_val;
      }());
  return *premapped_buffer_size;
}

}  // namespace torch_tpu
