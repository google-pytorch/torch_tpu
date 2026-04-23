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

#include "torch_tpu/eager/structured_log_buffer.h"

#include <atomic>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

TEST(StructuredLogBufferTest, PushDrainRoundtrip) {
  auto& buffer = StructuredLogBuffer::Get();
  const bool was_enabled = buffer.enabled();
  buffer.SetEnabledForTest(true);
  (void)buffer.Drain();  // Start clean

  EagerEvent event;
  event.name = "torchtpu_eager/test";
  event.ts_us = 42;
  buffer.Push(std::move(event));

  auto events = buffer.Drain();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].name, "torchtpu_eager/test");
  EXPECT_EQ(events[0].ts_us, 42);
  EXPECT_TRUE(buffer.Drain().empty());

  buffer.SetEnabledForTest(was_enabled);
}

TEST(StructuredLogBufferTest, DisabledBufferDropsEvents) {
  auto& buffer = StructuredLogBuffer::Get();
  const bool was_enabled = buffer.enabled();
  buffer.SetEnabledForTest(false);
  (void)buffer.Drain();  // Start clean

  EagerEvent event;
  event.name = "torchtpu_eager/test";
  buffer.Push(std::move(event));

  const auto events = buffer.Drain();
  EXPECT_TRUE(events.empty());

  buffer.SetEnabledForTest(was_enabled);
}

TEST(StructuredLogBufferTest, ConcurrentPushIsSafe) {
  auto& buffer = StructuredLogBuffer::Get();
  const bool was_enabled = buffer.enabled();
  buffer.SetEnabledForTest(true);
  (void)buffer.Drain();

  constexpr int kThreads = 8;
  constexpr int kPerThreadPushes = 200;
  std::atomic<bool> start_signal{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      while (!start_signal.load()) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kPerThreadPushes; ++i) {
        EagerEvent event;
        event.ts_us = i;
        buffer.Push(std::move(event));
      }
    });
  }

  start_signal.store(true);
  for (auto& th : threads) {
    th.join();
  }

  const auto events = buffer.Drain();
  // Buffer is capped at 1000, so with 8*200=1600 pushes we expect exactly
  // 1000 retained (the newest ones).
  EXPECT_EQ(events.size(), 1000u);

  buffer.SetEnabledForTest(was_enabled);
}

TEST(StructuredLogBufferTest, FIFOEviction) {
  auto& buffer = StructuredLogBuffer::Get();
  const bool was_enabled = buffer.enabled();
  buffer.SetEnabledForTest(true);
  (void)buffer.Drain();  // Start clean

  for (int i = 0; i < 1000; ++i) {
    EagerEvent event;
    event.ts_us = i;
    buffer.Push(std::move(event));
  }

  EagerEvent event;
  event.ts_us = 1000;
  buffer.Push(std::move(event));

  auto events = buffer.Drain();
  ASSERT_EQ(events.size(), 1000u);
  EXPECT_EQ(events[0].ts_us, 1);
  EXPECT_EQ(events[999].ts_us, 1000);

  buffer.SetEnabledForTest(was_enabled);
}

TEST(StructuredLogBufferTest, ReasonPreserved) {
  auto& buffer = StructuredLogBuffer::Get();
  const bool was_enabled = buffer.enabled();
  buffer.SetEnabledForTest(true);
  (void)buffer.Drain();  // Start clean

  EagerEvent event;
  event.name = "torchtpu_eager/test";
  event.reason = MaterializationReason::kCpuConversion;
  buffer.Push(std::move(event));

  auto events = buffer.Drain();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].reason, MaterializationReason::kCpuConversion);

  buffer.SetEnabledForTest(was_enabled);
}

}  // namespace
}  // namespace torch_tpu
