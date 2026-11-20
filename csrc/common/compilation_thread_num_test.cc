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

#include <sched.h>

#include <algorithm>
#include <cstdlib>
#include <string>

#include "absl/flags/declare.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/compilation_cache.h"
#include "csrc/common/flags.h"
#include "gtest/gtest.h"

ABSL_DECLARE_FLAG(int, torch_tpu_internal_num_compilation_threads);

namespace torch_tpu {

int GetNumCompilationThreads(int num_procs);

namespace {

using testing::ExitedWithCode;

// Returns the CPU affinity mask of this process.
cpu_set_t AffinityMask() {
  cpu_set_t mask;
  ABSL_CHECK_EQ(sched_getaffinity(0, sizeof(mask), &mask), 0);
  return mask;
}

// Restricts this process to the first `num_cpus` CPUs of its affinity mask.
void RestrictToCpus(int num_cpus) {
  const cpu_set_t allowed = AffinityMask();
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int cpu = 0; num_cpus > 0; ++cpu) {
    if (CPU_ISSET(cpu, &allowed)) {
      CPU_SET(cpu, &mask);
      --num_cpus;
    }
  }
  ABSL_CHECK_EQ(sched_setaffinity(0, sizeof(mask), &mask), 0);
}

// Logs `num_threads` and exits. The tests match the log instead of exiting
// with the count, as exit statuses only keep 8 bits.
[[noreturn]] void LogAndExit(int num_threads) {
  ABSL_LOG(INFO) << "threads=" << num_threads << ";";
  exit(0);
}

// Returns the log of `LogAndExit(num_threads)`.
std::string Logged(int num_threads) {
  return absl::StrCat("threads=", num_threads, ";");
}

// Each test runs GetNumCompilationThreads() in a death test's child process,
// so that it can set the flag, which GetNumCompilationThreads() memoizes on the
// first call, and the CPU affinity without affecting other tests.

TEST(GetNumCompilationThreadsDeathTest, DefaultUsesCpuAffinity) {
  const cpu_set_t mask = AffinityMask();
  EXPECT_EXIT(LogAndExit(GetNumCompilationThreads(0)), ExitedWithCode(0),
              Logged(std::max(1, CPU_COUNT(&mask) - 1)));
}

TEST(GetNumCompilationThreadsDeathTest, DefaultUsesNarrowedCpuAffinity) {
  if (const cpu_set_t mask = AffinityMask(); CPU_COUNT(&mask) < 3) {
    GTEST_SKIP() << "Needs at least 3 CPUs in the affinity mask.";
  }
  EXPECT_EXIT(
      {
        RestrictToCpus(3);
        LogAndExit(GetNumCompilationThreads(0));
      },
      ExitedWithCode(0), Logged(2));
}

TEST(GetNumCompilationThreadsDeathTest, DefaultUsesOneThreadOnOneCpu) {
  EXPECT_EXIT(
      {
        RestrictToCpus(1);
        LogAndExit(GetNumCompilationThreads(0));
      },
      ExitedWithCode(0), Logged(1));
}

TEST(GetNumCompilationThreadsDeathTest, NprocMinusOneTimesFour) {
  EXPECT_EXIT(LogAndExit(GetNumCompilationThreads(10)), ExitedWithCode(0),
              Logged(36));
}

TEST(GetNumCompilationThreadsDeathTest, FlagValue) {
  EXPECT_EXIT(
      {
        SafeSetFlag(&FLAGS_torch_tpu_internal_num_compilation_threads, 42);
        LogAndExit(GetNumCompilationThreads());
      },
      ExitedWithCode(0), Logged(42));
}

TEST(GetNumCompilationThreadsDeathTest, FlagTakesPrecedenceOverNproc) {
  EXPECT_EXIT(
      {
        SafeSetFlag(&FLAGS_torch_tpu_internal_num_compilation_threads, 42);
        LogAndExit(GetNumCompilationThreads(10));
      },
      ExitedWithCode(0), Logged(42));
}

}  // namespace
}  // namespace torch_tpu
