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

#include "csrc/common/compile_options_key.h"

#include <string>

#include "absl/log/absl_check.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compilation_test_helper.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/fingerprint_utils.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gtest/gtest.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {
namespace {

using testing::ExitedWithCode;

class CompileOptionsKeyTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // This must be done before `MakeCompilerOptions` is called, as the latter
    // depends on the PjRt client.
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());
  }
};

constexpr FingerprintType kFastCompileDefaultFingerprint =
    7052654442258214116ULL;
constexpr FingerprintType kFastRuntimeDefaultFingerprint =
    10471044014925058086ULL;

[[nodiscard]] CompileOptionsKey GetCompileOptionsKey(
    const CompilationMode mode) {
  return GetCompilationSpec(mode).compile_options_key;
}

TEST_F(CompileOptionsKeyTest, StableFingerprint) {
  {
    const CompileOptionsKey key =
        GetCompileOptionsKey(CompilationMode::kFastCompile);
    EXPECT_EQ(key.key(), kFastCompileDefaultFingerprint)
        << "Fingerprint stability is vital for the compilation cache "
           "correctness. Do not change the expected value to make the test "
           "pass unless the code changes how `xla::CompileOptions` is "
           "fingerprinted.";
  }

  {
    const CompileOptionsKey key =
        GetCompileOptionsKey(CompilationMode::kFastRuntime);
    EXPECT_EQ(key.key(), kFastRuntimeDefaultFingerprint)
        << "Fingerprint stability is vital for the compilation cache "
           "correctness. Do not change the expected value to make the test "
           "pass unless the code changes how `xla::CompileOptions` is "
           "fingerprinted.";
  }
}

TEST_F(CompileOptionsKeyTest,
       DifferentCompilationModesYieldDifferentFingerprints) {
  EXPECT_NE(kFastCompileDefaultFingerprint, kFastRuntimeDefaultFingerprint);
}

TEST_F(CompileOptionsKeyTest, XlaExecutionEffortLevelOverrides) {
  {
    ScopedCompilerOptionOverrides overrides({{"xla_optimization_level", "O3"}});

    const CompileOptionsKey key =
        GetCompileOptionsKey(CompilationMode::kFastCompile);
    EXPECT_NE(key.key(), kFastCompileDefaultFingerprint);
    EXPECT_EQ(key.key(), 11690617673366428246ULL)
        << "Fingerprint stability is vital for the compilation cache "
           "correctness. Do not change the expected value to make the test "
           "pass unless the code changes how `xla::CompileOptions` is "
           "fingerprinted.";
  }

  {
    ScopedCompilerOptionOverrides overrides(
        {{"xla_memory_fitting_level", "O1"}});

    const CompileOptionsKey key =
        GetCompileOptionsKey(CompilationMode::kFastCompile);
    EXPECT_NE(key.key(), kFastCompileDefaultFingerprint);
    EXPECT_EQ(key.key(), 8369878376337229729ULL)
        << "Fingerprint stability is vital for the compilation cache "
           "correctness. Do not change the expected value to make the test "
           "pass unless the code changes how `xla::CompileOptions` is "
           "fingerprinted.";
  }
}

TEST_F(CompileOptionsKeyTest, EnvOptionOverrides) {
  ScopedCompilerOptionOverrides overrides(
      {{"xla_tpu_enable_deduplicated_calls", "DISABLED"}});

  const CompileOptionsKey key =
      GetCompileOptionsKey(CompilationMode::kFastCompile);
  EXPECT_NE(key.key(), kFastCompileDefaultFingerprint);
  EXPECT_EQ(key.key(), 6329271667438365873ULL)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test pass "
         "unless the code changes how `xla::CompileOptions` is fingerprinted.";
}

TEST_F(CompileOptionsKeyTest, ThreadLocalContextOverrides) {
  EXPECT_EQ(GetCompileOptionsKey(CompilationMode::kFastCompile).key(),
            kFastCompileDefaultFingerprint);

  {
    ScopedCompilerOptionOverrides overrides({{"xla_optimization_level", "O3"}});

    const CompileOptionsKey key =
        GetCompileOptionsKey(CompilationMode::kFastCompile);
    EXPECT_NE(key.key(), kFastCompileDefaultFingerprint);
    EXPECT_EQ(key.key(), 11690617673366428246ULL)
        << "Fingerprint stability is vital for the compilation cache "
           "correctness. Do not change the expected value to make the test "
           "pass unless the code changes how `xla::CompileOptions` is "
           "fingerprinted.";
  }

  EXPECT_EQ(GetCompileOptionsKey(CompilationMode::kFastCompile).key(),
            kFastCompileDefaultFingerprint);
}

TEST(CompileOptionsKeyDeathTest, LibtpuInitArgsAffectsFingerprint) {
  constexpr FingerprintType kDefaultOptionsFingerprint =
      15154592522507906076ULL;
  constexpr FingerprintType kLibtpuInitArgsFingerprint =
      13352940040894739080ULL;

  xla::CompileOptions options;
  options.executable_build_options.mutable_debug_options();

  EXPECT_EXIT(
      {
        setenv(kLibtpuInitArgsEnvVar, "--xla_tpu_scoped_vmem_limit_kib=1", 1);
        const CompileOptionsKey key = MakeCompileOptionsKey(options);
        ABSL_CHECK_NE(key.key(), kDefaultOptionsFingerprint);
        ABSL_CHECK_EQ(key.key(), kLibtpuInitArgsFingerprint)
            << "Fingerprint stability is vital for the compilation cache "
               "correctness. Do not change the expected value to make the test "
               "pass unless the code changes how `xla::CompileOptions` is "
               "fingerprinted.";
        _exit(0);
      },
      ExitedWithCode(0), "");

  const CompileOptionsKey key = MakeCompileOptionsKey(options);
  EXPECT_EQ(key.key(), kDefaultOptionsFingerprint)
      << "Fingerprint stability is vital for the compilation cache "
         "correctness. Do not change the expected value to make the test "
         "pass unless the code changes how `xla::CompileOptions` is "
         "fingerprinted.";
}

}  // namespace
}  // namespace torch_tpu
