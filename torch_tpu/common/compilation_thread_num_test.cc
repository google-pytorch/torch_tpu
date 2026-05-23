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

#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>  // NOLINT

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/compilation_cache.h"
#include "torch_tpu/common/flags.h"
#include "xla/xla.pb.h"

ABSL_DECLARE_FLAG(int, torch_tpu_internal_num_compilation_threads);

namespace torch_tpu {

int GetNumCompilationThreads(int num_procs);

namespace {

using testing::ExitedWithCode;

TEST(GetNumCompilationThreads, ReturnsHardwareConcurrencyWhenNprocNotSet) {
  EXPECT_EXIT(
      {
        ABSL_CHECK_EQ(
            (GetFlagOnce<int32_t,
                         &FLAGS_torch_tpu_internal_num_compilation_threads>()),
            0);
        ABSL_LOG(INFO) << "testing GetNumCompilationThreads";
        exit(GetNumCompilationThreads(0));
      },
      ExitedWithCode(std::thread::hardware_concurrency() - 1),
      "testing GetNumCompilationThreads");
}

TEST(GetNumCompilationThreads, ReturnsNprocTimesTwoWhenNprocSet) {
  EXPECT_EXIT(
      {
        ABSL_CHECK_EQ(
            (GetFlagOnce<int32_t,
                         &FLAGS_torch_tpu_internal_num_compilation_threads>()),
            0);
        ABSL_LOG(INFO) << "testing GetNumCompilationThreads";
        exit(GetNumCompilationThreads(10));
      },
      ExitedWithCode(36), "testing GetNumCompilationThreads");
}

// We use death tests to test the effect of setting flags in different
// subprocesses, because GetNumCompilationThreads() memoizes the flag value on
// the first call.

TEST(GetNumCompilationThreadsDeathTest, ReturnsFlagValueWhenSet) {
  EXPECT_EXIT(
      {
        absl::SetFlag(&FLAGS_torch_tpu_internal_num_compilation_threads, 42);
        // Make sure the SetFlag was effective.
        ABSL_CHECK_EQ(
            (GetFlagOnce<int32_t,
                         &FLAGS_torch_tpu_internal_num_compilation_threads>()),
            42);
        ABSL_LOG(INFO) << "testing GetNumCompilationThreads";
        exit(GetNumCompilationThreads());
      },
      ExitedWithCode(42), "testing GetNumCompilationThreads");
}

TEST(GetNumCompilationThreadsDeathTest,
     ReturnsHardwareConcurrencyWhenFlagSetToZero) {
  EXPECT_EXIT(
      {
        absl::SetFlag(&FLAGS_torch_tpu_internal_num_compilation_threads, 0);
        // Make sure the SetFlag was effective.
        ABSL_CHECK_EQ(
            (GetFlagOnce<int32_t,
                         &FLAGS_torch_tpu_internal_num_compilation_threads>()),
            0);
        ABSL_LOG(INFO) << "testing GetNumCompilationThreads";
        exit(GetNumCompilationThreads(0));
      },
      ExitedWithCode(std::thread::hardware_concurrency() - 1),
      "testing GetNumCompilationThreads");
}

TEST(GetNumCompilationThreadsDeathTest, FlagTakesPrecedenceOverNproc) {
  EXPECT_EXIT(
      {
        absl::SetFlag(&FLAGS_torch_tpu_internal_num_compilation_threads, 42);
        // Make sure the SetFlag was effective.
        ABSL_CHECK_EQ(
            (GetFlagOnce<int32_t,
                         &FLAGS_torch_tpu_internal_num_compilation_threads>()),
            42);
        ABSL_LOG(INFO) << "testing GetNumCompilationThreads";
        exit(GetNumCompilationThreads(10));
      },
      ExitedWithCode(42), "testing GetNumCompilationThreads");
}

}  // namespace
}  // namespace torch_tpu
