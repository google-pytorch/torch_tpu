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

#include "torch_tpu/common/compile_options_key.h"

#include <string>

#include "absl/log/absl_check.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/compilation_test_helper.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "torch_tpu/pjrt/pjrt_state.h"

namespace torch_tpu {
namespace {

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
    11255405118680073921ULL;
constexpr FingerprintType kFastRuntimeDefaultFingerprint =
    10955459558934170598ULL;

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
    EXPECT_EQ(key.key(), 14056296875166197543ULL)
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
    EXPECT_EQ(key.key(), 12100832961370432928ULL)
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
  EXPECT_EQ(key.key(), 9003441201055756418ULL)
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
    EXPECT_EQ(key.key(), 14056296875166197543ULL)
        << "Fingerprint stability is vital for the compilation cache "
           "correctness. Do not change the expected value to make the test "
           "pass unless the code changes how `xla::CompileOptions` is "
           "fingerprinted.";
  }

  EXPECT_EQ(GetCompileOptionsKey(CompilationMode::kFastCompile).key(),
            kFastCompileDefaultFingerprint);
}

}  // namespace
}  // namespace torch_tpu
