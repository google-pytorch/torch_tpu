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

#include "torch_tpu/common/op_name_stack.h"

#include "gtest/gtest.h"
#include "torch_tpu/ops/op_names.h"

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
  ScopedOpName guard1(OpName::kAdd);
  EXPECT_EQ(OpNameStack::Top(), OpName::kAdd);

  {
    ScopedOpName guard2(OpName::kSub);
    EXPECT_EQ(OpNameStack::Top(), OpName::kSub);
  }

  EXPECT_EQ(OpNameStack::Top(), OpName::kAdd);
}

TEST(OpNameStackTest, ScopedForwardingGuardTransitionsState) {
  EXPECT_FALSE(ScopedForwardingGuard::IsForwardingActive());
  {
    ScopedForwardingGuard guard1;
    EXPECT_TRUE(ScopedForwardingGuard::IsForwardingActive());
    {
      ScopedForwardingGuard guard2;
      EXPECT_TRUE(ScopedForwardingGuard::IsForwardingActive());
    }
    EXPECT_TRUE(ScopedForwardingGuard::IsForwardingActive());
  }
  EXPECT_FALSE(ScopedForwardingGuard::IsForwardingActive());
}

#if GTEST_HAS_DEATH_TEST
TEST(OpNameStackTest, ForwardingGuardCorrectlyBypassesCompositeOpCheck) {
  // Setup: With the guard active, pushing a nested op under a non-whitelisted
  // parent (e.g. kSoftmaxOut) is allowed and bypasses the check.
  {
    ScopedOpName parent_op(OpName::kSoftmaxOut);
    {
      ScopedForwardingGuard guard;
      ScopedOpName nested_op(OpName::kSub);
      EXPECT_EQ(OpNameStack::Top(), OpName::kSub);
    }
    EXPECT_EQ(OpNameStack::Top(), OpName::kSoftmaxOut);
  }

  // Without the guard, pushing a nested op under a non-whitelisted parent
  // must trigger a raw CompositeOpCheck crash.
  EXPECT_DEATH(
      {
        ScopedOpName parent_op(OpName::kSoftmaxOut);
        ScopedOpName nested_op(OpName::kSub);
      },
      "CompositeOpCheck: Op _softmax.out is decomposed into sub");
}
#endif

}  // namespace
}  // namespace internal
}  // namespace torch_tpu
