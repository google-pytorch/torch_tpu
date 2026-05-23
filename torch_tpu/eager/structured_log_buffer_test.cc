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
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/time/time.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

class StructuredLogBufferTest : public testing::Test {
 protected:
  StructuredLogBufferTest() {
    auto& buffer = StructuredLogBuffer::GetInstance();
    was_enabled_ = buffer.enabled();
    buffer.SetEnabledForTest(true);
    static_cast<void>(buffer.Drain());  // Start clean
  }

  ~StructuredLogBufferTest() override {
    auto& buffer = StructuredLogBuffer::GetInstance();
    buffer.SetEnabledForTest(was_enabled_);
  }

  bool was_enabled_;
};

TEST_F(StructuredLogBufferTest, PushDrainRoundtrip) {
  auto& buffer = StructuredLogBuffer::GetInstance();

  auto event = std::make_unique<StructuredLogEvent>();
  event->name = "torchtpu_eager/test";
  event->timestamp = absl::FromUnixMicros(42);
  buffer.Push(std::move(event));

  auto result = buffer.Drain();
  ASSERT_EQ(result.events.size(), 1u);
  EXPECT_EQ(result.events[0]->name, "torchtpu_eager/test");
  EXPECT_EQ(result.events[0]->timestamp, absl::FromUnixMicros(42));
  EXPECT_EQ(result.dropped_since_last_drain, 0u);
  EXPECT_TRUE(buffer.Drain().events.empty());
}

TEST_F(StructuredLogBufferTest, DisabledBufferDropsEvents) {
  auto& buffer = StructuredLogBuffer::GetInstance();
  buffer.SetEnabledForTest(false);

  auto event = std::make_unique<StructuredLogEvent>();
  event->name = "torchtpu_eager/test";
  buffer.Push(std::move(event));

  const auto result = buffer.Drain();
  EXPECT_TRUE(result.events.empty());
}

TEST_F(StructuredLogBufferTest, DrainWhileDisabledReturnsEmpty) {
  auto& buffer = StructuredLogBuffer::GetInstance();

  auto event = std::make_unique<StructuredLogEvent>();
  event->name = "torchtpu_eager/test";
  buffer.Push(std::move(event));

  buffer.SetEnabledForTest(false);

  auto result = buffer.Drain();
  EXPECT_TRUE(result.events.empty());

  // Re-enable to show the events were kept in the buffer.
  buffer.SetEnabledForTest(true);
  result = buffer.Drain();
  ASSERT_EQ(result.events.size(), 1u);
  EXPECT_EQ(result.events[0]->name, "torchtpu_eager/test");
  EXPECT_EQ(result.dropped_since_last_drain, 0u);
}

TEST_F(StructuredLogBufferTest, ConcurrentPushIsSafe) {
  auto& buffer = StructuredLogBuffer::GetInstance();

  constexpr int kThreads = 12;
  constexpr int kPerThreadPushes = 1000;
  std::atomic<bool> start_signal{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      while (!start_signal.load()) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kPerThreadPushes; ++i) {
        auto event = std::make_unique<StructuredLogEvent>();
        event->timestamp = absl::FromUnixMicros(i);
        buffer.Push(std::move(event));
      }
    });
  }

  start_signal.store(true);
  for (auto& th : threads) {
    th.join();
  }

  const auto result = buffer.Drain();
  // Buffer is capped at 10000, so with 12*1000=12000 pushes we expect exactly
  // 10000 retained (the newest ones) and 2000 dropped.
  EXPECT_EQ(result.events.size(), 10000u);
  EXPECT_EQ(result.dropped_since_last_drain, 2000u);
}

TEST_F(StructuredLogBufferTest, FifoEviction) {
  auto& buffer = StructuredLogBuffer::GetInstance();

  for (int i = 0; i < 10000; ++i) {
    auto event = std::make_unique<StructuredLogEvent>();
    event->timestamp = absl::FromUnixMicros(i);
    buffer.Push(std::move(event));
  }

  auto event = std::make_unique<StructuredLogEvent>();
  event->timestamp = absl::FromUnixMicros(10000);
  buffer.Push(std::move(event));

  auto result = buffer.Drain();
  ASSERT_EQ(result.events.size(), 10000u);
  EXPECT_EQ(result.events[0]->timestamp, absl::FromUnixMicros(1));
  EXPECT_EQ(result.events[9999]->timestamp, absl::FromUnixMicros(10000));
  EXPECT_EQ(result.dropped_since_last_drain, 1u);
}

}  // namespace
}  // namespace torch_tpu
