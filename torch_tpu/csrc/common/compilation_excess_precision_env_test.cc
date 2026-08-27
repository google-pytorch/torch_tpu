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

#include <cstdlib>
#include <string_view>

#include "gtest/gtest.h"
#include "torch_tpu/csrc/common/excess_precision.h"

namespace torch_tpu {
namespace {

TEST(CompilationExcessPrecisionEnvTest, VerifiesEnvVarInfluence) {
  const char* expected_env = std::getenv("EXPECTED_VALUE");
  ASSERT_NE(expected_env, nullptr) << "EXPECTED_VALUE env var must be set";
  bool expected = (std::string_view(expected_env) == "true");
  EXPECT_EQ(GetAllowExcessPrecision(), expected);
}

}  // namespace
}  // namespace torch_tpu
