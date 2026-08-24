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

#include "torch_tpu/csrc/common/op_name_stack.h"

#include "gtest/gtest.h"
#include "torch_tpu/csrc/ops/op_names.h"

namespace torch_tpu {
namespace internal {
namespace {

TEST(OpNameStackTest, EmptyStackReturnsNullopt) {
  EXPECT_FALSE(OpNameStack::MaybeTop().has_value());
}

TEST(OpNameStackTest, ScopedOpNamePushesAndPops) {
  EXPECT_FALSE(OpNameStack::MaybeTop().has_value());
  {
    ScopedOpName guard(OpName::kAdd);
    EXPECT_EQ(OpNameStack::Top(), OpName::kAdd);
    EXPECT_EQ(OpNameStack::MaybeTop(), OpName::kAdd);
  }
  EXPECT_FALSE(OpNameStack::MaybeTop().has_value());
}

TEST(OpNameStackTest, NestedScopedOpNamesBehaveAsStack) {
  ScopedOpName guard1(OpName::kMaxPool2d);
  EXPECT_EQ(OpNameStack::Top(), OpName::kMaxPool2d);

  {
    ScopedOpName guard2(OpName::kSub);
    EXPECT_EQ(OpNameStack::Top(), OpName::kSub);
  }

  EXPECT_EQ(OpNameStack::Top(), OpName::kMaxPool2d);
}

}  // namespace
}  // namespace internal
}  // namespace torch_tpu
