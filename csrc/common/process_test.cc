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

#include "csrc/common/process.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

TEST(ProcessTest, HasTorchTpuAncestorProcessReturnsFalseInParent) {
  EXPECT_FALSE(HasTorchTpuAncestorProcess());
  // Verify idempotency.
  EXPECT_FALSE(HasTorchTpuAncestorProcess());
}

TEST(ProcessTest, ForkedChildProcessReturnsTrue) {
  ASSERT_FALSE(HasTorchTpuAncestorProcess());

  const pid_t child_pid = fork();
  ASSERT_GE(child_pid, 0);
  if (child_pid == 0) {
    _exit(HasTorchTpuAncestorProcess() ? 0 : 1);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child_pid, &status, 0), child_pid);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(ProcessTest, ExecedChildProcessReturnsTrue) {
  if (HasTorchTpuAncestorProcess()) {
    // Running inside the re-execed child process image, which detected the
    // parent test process as its TorchTPU ancestor.
    return;
  }

  ASSERT_FALSE(HasTorchTpuAncestorProcess());

  const pid_t child_pid = fork();
  ASSERT_GE(child_pid, 0);
  if (child_pid == 0) {
    // Prevent the child test runner from overwriting the parent's output files.
    unsetenv("GUNIT_OUTPUT");
    unsetenv("TEST_INFRASTRUCTURE_FAILURE_FILE");
    unsetenv("TEST_LOGSPLITTER_OUTPUT_FILE");
    unsetenv("TEST_PREMATURE_EXIT_FILE");
    unsetenv("TEST_UNUSED_RUNFILES_LOG_FILE");
    unsetenv("TEST_WARNINGS_OUTPUT_FILE");
    unsetenv("XML_OUTPUT_FILE");
    setenv(  // STRING_LITERAL_ENV_VAR_NAME_OK=Bazel test filter env var.
        "TESTBRIDGE_TEST_ONLY", "ProcessTest.ExecedChildProcessReturnsTrue", 1);
    execl("/proc/self/exe", "/proc/self/exe", nullptr);
    _exit(127);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child_pid, &status, 0), child_pid);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

}  // namespace
}  // namespace torch_tpu
