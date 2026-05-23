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

#include "torch_tpu/common/flags.h"

#include <string>

#include "absl/flags/flag.h"
#include "gtest/gtest.h"

ABSL_FLAG(bool, test_bool_flag, true, "Test bool flag");
ABSL_FLAG(int, test_int_flag, 42, "Test int flag");
ABSL_FLAG(std::string, test_string_flag, "default", "Test string flag");
ABSL_FLAG(int, multi_test_flag_1, 10, "Multi test flag 1");
ABSL_FLAG(int, multi_test_flag_2, 20, "Multi test flag 2");

namespace torch_tpu {
namespace {

TEST(FlagsTest, MemoizesBoolFlag) {
  // Initial value should be true.
  EXPECT_TRUE((GetFlagOnce<bool, &FLAGS_test_bool_flag>()));

  // Modify the flag.
  absl::SetFlag(&FLAGS_test_bool_flag, false);

  // GetFlagOnce should STILL return the initial value (true).
  EXPECT_TRUE((GetFlagOnce<bool, &FLAGS_test_bool_flag>()));
}

TEST(FlagsTest, MemoizesIntFlag) {
  EXPECT_EQ((GetFlagOnce<int, &FLAGS_test_int_flag>()), 42);
  absl::SetFlag(&FLAGS_test_int_flag, 100);
  EXPECT_EQ((GetFlagOnce<int, &FLAGS_test_int_flag>()), 42);
}

TEST(FlagsTest, MemoizesStringFlag) {
  EXPECT_EQ((GetFlagOnce<std::string, &FLAGS_test_string_flag>()), "default");
  absl::SetFlag(&FLAGS_test_string_flag, "new_value");
  EXPECT_EQ((GetFlagOnce<std::string, &FLAGS_test_string_flag>()), "default");
}

TEST(FlagsTest, HandlesMultipleFlagsOfSameType) {
  EXPECT_EQ((GetFlagOnce<int, &FLAGS_multi_test_flag_1>()), 10);
  EXPECT_EQ((GetFlagOnce<int, &FLAGS_multi_test_flag_2>()), 20);

  absl::SetFlag(&FLAGS_multi_test_flag_1, 100);
  absl::SetFlag(&FLAGS_multi_test_flag_2, 200);

  EXPECT_EQ((GetFlagOnce<int, &FLAGS_multi_test_flag_1>()), 10);
  EXPECT_EQ((GetFlagOnce<int, &FLAGS_multi_test_flag_2>()), 20);
}

}  // namespace
}  // namespace torch_tpu
