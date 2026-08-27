// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/csrc/ops/op_names.h"

#include <set>
#include <string>

#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

TEST(OpNamesTest, AllToStringResultsAreUnique) {
  std::set<std::string> seen_names;
  for (int i = static_cast<int>(OpName::kMinOpMinus1) + 1;
       i < static_cast<int>(OpName::kMaxOpPlus1); ++i) {
    const OpName op = static_cast<OpName>(i);
    const std::string name(ToString(op));
    EXPECT_TRUE(seen_names.insert(name).second)
        << "Duplicate OpName string: " << name << " for enum value " << i;
  }
}

}  // namespace
}  // namespace torch_tpu
