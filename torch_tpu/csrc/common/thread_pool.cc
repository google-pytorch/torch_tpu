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

#include "torch_tpu/csrc/common/thread_pool.h"

#include <string>

#include "absl/log/absl_check.h"
#include "xla/tsl/platform/env.h"

namespace torch_tpu {

ThreadPool::ThreadPool(const std::string& name, int num_threads,
                       bool low_latency_hint)
    : pool_(tsl::Env::Default(), tsl::ThreadOptions(), name, num_threads,
            low_latency_hint) {
  ABSL_CHECK_LE(name.size(), 12)  // CRASH_OK
      << "Thread name must be at most 12 characters, or it will be truncated.";
  ABSL_CHECK_GT(num_threads, 0)  // CRASH_OK
      << "Number of threads must be positive.";
}

}  // namespace torch_tpu
