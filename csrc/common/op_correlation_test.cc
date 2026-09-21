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

#include "csrc/common/op_correlation.h"

#include <cstdint>
#include <limits>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

TEST(OpCorrelationTest, NarrowCorrelationIdPreservesBitsAndMatchesShimCast) {
  EXPECT_EQ(NarrowCorrelationId(0), kNoCorrelationId);
  EXPECT_EQ(NarrowCorrelationId(42), 42);
  EXPECT_EQ(NarrowCorrelationId(std::numeric_limits<int32_t>::max()),
            std::numeric_limits<int32_t>::max());

  // Verify that values exceeding INT32_MAX narrow with the exact same
  // wrap-around semantics as kineto_shim.cpp
  // (static_cast<int32_t>(correlation_id)).
  const uint64_t overflow_id =
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1ULL;
  EXPECT_EQ(NarrowCorrelationId(overflow_id),
            std::numeric_limits<int32_t>::min());
  EXPECT_EQ(NarrowCorrelationId(std::numeric_limits<uint64_t>::max()), -1);
}

TEST(OpCorrelationTest, DispatchStackPushPopAndReset) {
  ResetOpCorrelationIds();
  EXPECT_EQ(GetCurrentOpCorrelationId(), kNoCorrelationId);

  // Popping an empty stack must be safe and not crash.
  PopOpCorrelationId();
  EXPECT_EQ(GetCurrentOpCorrelationId(), kNoCorrelationId);

  PushOpCorrelationId(101);
  EXPECT_EQ(GetCurrentOpCorrelationId(), 101);

  // Nested push: innermost must be active.
  PushOpCorrelationId(102);
  EXPECT_EQ(GetCurrentOpCorrelationId(), 102);

  // Pop innermost restores outer op.
  PopOpCorrelationId();
  EXPECT_EQ(GetCurrentOpCorrelationId(), 101);

  // Pop outer restores empty.
  PopOpCorrelationId();
  EXPECT_EQ(GetCurrentOpCorrelationId(), kNoCorrelationId);

  // Reset clears all stacked IDs.
  PushOpCorrelationId(201);
  PushOpCorrelationId(202);
  ResetOpCorrelationIds();
  EXPECT_EQ(GetCurrentOpCorrelationId(), kNoCorrelationId);
}

TEST(OpCorrelationTest, PushCounterMonotonicallyIncrementsAndResets) {
  ResetOpCorrelationPushCount();
  EXPECT_EQ(GetOpCorrelationPushCount(), 0);

  PushOpCorrelationId(1);
  EXPECT_EQ(GetOpCorrelationPushCount(), 1);

  PushOpCorrelationId(2);
  EXPECT_EQ(GetOpCorrelationPushCount(), 2);

  // Popping does not decrement the push counter (it measures pushes received).
  PopOpCorrelationId();
  EXPECT_EQ(GetOpCorrelationPushCount(), 2);

  ResetOpCorrelationPushCount();
  EXPECT_EQ(GetOpCorrelationPushCount(), 0);

  // Clean up thread-local stack for other tests.
  ResetOpCorrelationIds();
}

TEST(OpCorrelationTest, PushNarrowsIdsOnTheWayIn) {
  ResetOpCorrelationIds();

  // Narrowing has to happen on the push path, not just in the standalone
  // helper. libkineto hands us a uint64_t, but the activity map we will be
  // looked up in is keyed on the int32_t cast of that value, so whatever we
  // store must already be that cast. A saturating or clamping implementation
  // would compile, keep NarrowCorrelationId's own test green, and silently
  // stop every lookup above INT32_MAX from resolving.
  const uint64_t overflow_id =
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1ULL;
  PushOpCorrelationId(overflow_id);
  EXPECT_EQ(GetCurrentOpCorrelationId(), NarrowCorrelationId(overflow_id));
  EXPECT_EQ(GetCurrentOpCorrelationId(), std::numeric_limits<int32_t>::min());

  // The all-ones case must reach the stack as -1 rather than being dropped or
  // clamped to INT32_MAX.
  PushOpCorrelationId(std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(GetCurrentOpCorrelationId(), -1);

  // Unwinding a narrowed push restores the previous narrowed value.
  PopOpCorrelationId();
  EXPECT_EQ(GetCurrentOpCorrelationId(), std::numeric_limits<int32_t>::min());

  ResetOpCorrelationIds();
}

TEST(OpCorrelationTest, DispatchStackIsThreadLocal) {
  ResetOpCorrelationIds();
  PushOpCorrelationId(500);
  EXPECT_EQ(GetCurrentOpCorrelationId(), 500);

  std::thread worker([] {
    // Other thread starts with clean empty stack.
    EXPECT_EQ(GetCurrentOpCorrelationId(), kNoCorrelationId);
    PushOpCorrelationId(600);
    EXPECT_EQ(GetCurrentOpCorrelationId(), 600);
    ResetOpCorrelationIds();
  });
  worker.join();

  // Current thread was unaffected by worker thread operations.
  EXPECT_EQ(GetCurrentOpCorrelationId(), 500);
  ResetOpCorrelationIds();
}

TEST(OpCorrelationTest, ScopedLaunchCorrelationInstallsAndRestores) {
  // Outside any scope, launch correlation IDs must be empty.
  EXPECT_TRUE(GetCurrentLaunchCorrelationIds().empty());

  {
    ScopedLaunchCorrelation outer({10, 20, 30});
    absl::Span<const CorrelationId> outer_ids =
        GetCurrentLaunchCorrelationIds();
    ASSERT_EQ(outer_ids.size(), 3);
    EXPECT_EQ(outer_ids[0], 10);
    EXPECT_EQ(outer_ids[1], 20);
    EXPECT_EQ(outer_ids[2], 30);

    {
      // Inner nested scope shadows outer scope.
      ScopedLaunchCorrelation inner({999});
      absl::Span<const CorrelationId> inner_ids =
          GetCurrentLaunchCorrelationIds();
      ASSERT_EQ(inner_ids.size(), 1);
      EXPECT_EQ(inner_ids[0], 999);
    }

    // Outer scope restored upon inner destruction.
    outer_ids = GetCurrentLaunchCorrelationIds();
    ASSERT_EQ(outer_ids.size(), 3);
    EXPECT_EQ(outer_ids[0], 10);
    EXPECT_EQ(outer_ids[1], 20);
    EXPECT_EQ(outer_ids[2], 30);
  }

  // Restored to empty upon outer destruction.
  EXPECT_TRUE(GetCurrentLaunchCorrelationIds().empty());
}

TEST(OpCorrelationTest, EmptyLaunchScopeHidesEnclosingIds) {
  ScopedLaunchCorrelation outer({10, 20});
  ASSERT_EQ(GetCurrentLaunchCorrelationIds().size(), 2);

  {
    // A launch containing no correlated ops must present as empty, not fall
    // through to the enclosing scope. The enqueue-side TraceMe keys off
    // `empty()` to decide whether to stamp a correlation ID at all, so leaking
    // the outer IDs here would attribute this launch to an unrelated op.
    ScopedLaunchCorrelation inner{std::vector<CorrelationId>()};
    EXPECT_TRUE(GetCurrentLaunchCorrelationIds().empty());
  }

  // And the enclosing scope comes back intact afterwards.
  const absl::Span<const CorrelationId> restored =
      GetCurrentLaunchCorrelationIds();
  ASSERT_EQ(restored.size(), 2);
  EXPECT_EQ(restored[0], 10);
  EXPECT_EQ(restored[1], 20);
}

TEST(OpCorrelationTest, LaunchCorrelationIsThreadLocal) {
  ScopedLaunchCorrelation main_launch({111, 222});
  EXPECT_EQ(GetCurrentLaunchCorrelationIds().size(), 2);

  std::thread worker([] {
    // Other thread must see an empty launch correlation span.
    EXPECT_TRUE(GetCurrentLaunchCorrelationIds().empty());
    {
      ScopedLaunchCorrelation worker_launch({333});
      EXPECT_EQ(GetCurrentLaunchCorrelationIds().size(), 1);
    }
    EXPECT_TRUE(GetCurrentLaunchCorrelationIds().empty());
  });
  worker.join();

  EXPECT_EQ(GetCurrentLaunchCorrelationIds().size(), 2);
}

}  // namespace
}  // namespace torch_tpu
