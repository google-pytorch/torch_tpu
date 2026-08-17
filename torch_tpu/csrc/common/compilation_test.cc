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

#include "torch_tpu/csrc/common/compilation.h"

#include <string>

#include "absl/log/absl_check.h"
#include "gtest/gtest.h"
#include "torch_tpu/csrc/common/compilation_spec.h"
#include "torch_tpu/csrc/common/compilation_test_helper.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/xla.pb.h"

namespace torch_tpu {
namespace {

class MakeCompilerOptionsTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // This must be done before MakeCompilerOptions() is called, as the latter
    // depends on the PjRt client.
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());
  }
};

UniqueCompileOptions GetCompileOptions(const CompilationMode mode) {
  return GetCompilationSpec(mode).xla_compile_options;
}

TEST_F(MakeCompilerOptionsTest, DefaultToO1ForEagerMode) {
  UniqueCompileOptions options =
      GetCompileOptions(CompilationMode::kFastCompile);
  EXPECT_EQ(options->executable_build_options.optimization_level(),
            xla::ExecutionOptions::EFFORT_O1);
}

TEST_F(MakeCompilerOptionsTest, DefaultToUnsetForTorchCompileMode) {
  UniqueCompileOptions options =
      GetCompileOptions(CompilationMode::kFastRuntime);
  EXPECT_EQ(options->executable_build_options.optimization_level(),
            xla::ExecutionOptions::EFFORT_UNKNOWN);
}

TEST_F(MakeCompilerOptionsTest, CompilerOptionOverrides) {
  {
    ScopedCompilerOptionOverrides overrides({{"xla_optimization_level", "O1"}});
    UniqueCompileOptions options =
        GetCompileOptions(CompilationMode::kFastCompile);
    EXPECT_EQ(options->executable_build_options.optimization_level(),
              xla::ExecutionOptions::EFFORT_O1);
  }

  {
    ScopedCompilerOptionOverrides overrides({{"xla_optimization_level", "O2"}});
    UniqueCompileOptions options =
        GetCompileOptions(CompilationMode::kFastCompile);
    EXPECT_EQ(options->executable_build_options.optimization_level(),
              xla::ExecutionOptions::EFFORT_O2);
  }
}

}  // namespace
}  // namespace torch_tpu
