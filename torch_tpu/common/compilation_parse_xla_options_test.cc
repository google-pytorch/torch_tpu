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

#include <string>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/compilation_test_helper.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace torch_tpu {

absl::Status ApplyCompilerOptionOverrides(CompilerOptionOverrides overrides,
                                          xla::CompileOptions& compile_options);

namespace {

using testing::Contains;
using testing::ElementsAre;
using testing::Pair;

class MakeCompilerOptionsTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // This must be done before MakeCompilerOptions() is called, as the latter
    // depends on the PjRt client.
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());
  }

  ~MakeCompilerOptionsTest() override {
    unsetenv("TORCH_TPU_INTERNAL_XLA_OPTIONS");
  }
};

TEST_F(MakeCompilerOptionsTest, ParsesXlaOptions) {
  setenv("TORCH_TPU_INTERNAL_XLA_OPTIONS",
         // Spaces are intentional to test parsing.
         " xla_optimization_level=O3  xla_tpu_enable_deduplicated_calls=AUTO ",
         1);

  TF_ASSERT_OK_AND_ASSIGN(CompilationSpecsByMode compilation_specs,
                          MakeCompilationSpecs(CompilationMode::kFastCompile));
  const auto& spec = compilation_specs.at(CompilationMode::kFastCompile);

  EXPECT_EQ(
      spec.xla_compile_options->executable_build_options.optimization_level(),
      xla::ExecutionOptions::EFFORT_O3);
  EXPECT_THAT(
      spec.xla_compile_options->env_option_overrides,
      Contains(Pair("xla_tpu_enable_deduplicated_calls", std::string("AUTO"))));
}

TEST_F(MakeCompilerOptionsTest, PythonContextManagerOverridesEnvVar) {
  setenv("TORCH_TPU_INTERNAL_XLA_OPTIONS",
         "xla_optimization_level=O3  xla_tpu_enable_deduplicated_calls=AUTO",
         1);
  ScopedCompilerOptionOverrides overrides({
      {"xla_optimization_level", "O2"},
  });

  TF_ASSERT_OK_AND_ASSIGN(CompilationSpecsByMode compilation_specs,
                          MakeCompilationSpecs(CompilationMode::kFastCompile));
  const auto& spec = compilation_specs.at(CompilationMode::kFastCompile);

  EXPECT_EQ(
      spec.xla_compile_options->executable_build_options.optimization_level(),
      xla::ExecutionOptions::EFFORT_O2);
  EXPECT_THAT(
      spec.xla_compile_options->env_option_overrides,
      Contains(Pair("xla_tpu_enable_deduplicated_calls", std::string("AUTO"))));
}

TEST(ApplyCompilerOptionOverrides, AppendsNonExistingOptions) {
  xla::CompileOptions options;
  options.env_option_overrides = {
      // go/keep-sorted start
      {"xla_abc", true},
      {"xla_xyz", 42},
      // go/keep-sorted end
  };
  CompilerOptionOverrides overrides = {
      // go/keep-sorted start
      {"xla_def", "AUTO"},
      {"xla_ghi", "SAFE"},
      // go/keep-sorted end
  };
  ASSERT_EQ(ApplyCompilerOptionOverrides(std::move(overrides), options),
            absl::OkStatus());
  EXPECT_THAT(options.env_option_overrides,
              ElementsAre(Pair("xla_abc", true), Pair("xla_xyz", 42),
                          Pair("xla_def", std::string("AUTO")),
                          Pair("xla_ghi", std::string("SAFE"))));
}

TEST(ApplyCompilerOptionOverrides, ReplacesExistingOptions) {
  xla::CompileOptions options;
  options.env_option_overrides = {
      // go/keep-sorted start
      {"xla_abc", true},
      {"xla_xyz", 42},
      // go/keep-sorted end
  };
  CompilerOptionOverrides overrides = {
      // go/keep-sorted start
      {"xla_abc", "AUTO"},
      {"xla_xyz", "SAFE"},
      // go/keep-sorted end
  };
  ASSERT_EQ(ApplyCompilerOptionOverrides(std::move(overrides), options),
            absl::OkStatus());
  EXPECT_THAT(options.env_option_overrides,
              ElementsAre(Pair("xla_abc", std::string("AUTO")),
                          Pair("xla_xyz", std::string("SAFE"))));
}

TEST(ApplyCompilerOptionOverrides, HandlesExistingAndNewOptions) {
  xla::CompileOptions options;
  options.env_option_overrides = {
      // go/keep-sorted start
      {"xla_abc", true},
      {"xla_xyz", 42},
      // go/keep-sorted end
  };
  CompilerOptionOverrides overrides = {
      // go/keep-sorted start
      {"xla_abc", "AUTO"},
      {"xla_def", "SAFE"},
      // go/keep-sorted end
  };
  ASSERT_EQ(ApplyCompilerOptionOverrides(std::move(overrides), options),
            absl::OkStatus());
  EXPECT_THAT(
      options.env_option_overrides,
      ElementsAre(Pair("xla_abc", std::string("AUTO")), Pair("xla_xyz", 42),
                  Pair("xla_def", std::string("SAFE"))));
}

}  // namespace
}  // namespace torch_tpu
