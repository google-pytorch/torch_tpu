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

#include "csrc/internal/profiler/xprof_callback_handler.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string_view>
#include <thread>
#include <vector>

#include "ATen/record_function.h"
#include "absl/container/flat_hash_set.h"
#include "gtest/gtest.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace torch_tpu {
namespace {

tensorflow::ProfileOptions MakeProfileOptions(int host_tracer_level) {
  tensorflow::ProfileOptions opts = tsl::ProfilerSession::DefaultOptions();
  opts.set_host_tracer_level(host_tracer_level);
  opts.set_device_tracer_level(0);
  return opts;
}

int CountHostCpuEvents(const tensorflow::profiler::XSpace& xspace,
                       std::string_view event_name) {
  int count = 0;
  for (const auto& plane : xspace.planes()) {
    if (plane.name() != "/host:CPU") {
      continue;
    }
    absl::flat_hash_set<int64_t> matching_ids;
    for (const auto& [meta_id, metadata] : plane.event_metadata()) {
      if (metadata.name() == event_name) {
        matching_ids.insert(meta_id);
      }
    }
    for (const auto& line : plane.lines()) {
      for (const auto& event : line.events()) {
        if (matching_ids.contains(event.metadata_id())) {
          ++count;
        }
      }
    }
  }
  return count;
}

TEST(XProfCallbackHandlerTest, StressTestThreadSafety) {
  ASSERT_FALSE(at::hasGlobalCallbacks());
  std::atomic<bool> stop_workers{false};

  // 1. Spawn worker threads executing RecordFunction continuously.
  std::vector<std::thread> workers;
  for (int i = 0; i < 8; ++i) {
    workers.emplace_back([&]() {
      while (!stop_workers.load(std::memory_order_relaxed)) {
        at::RecordFunction guard(at::RecordScope::FUNCTION);
        if (guard.isActive()) {
          // Simulate tiny work.
        }
      }
    });
  }

  // 2. Concurrently add and remove the callback repeatedly.
  for (int i = 0; i < 1000; ++i) {
    XProfCallbackHandler handler;
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }

  // 3. Clean up.
  stop_workers.store(true);
  for (auto& worker : workers) {
    worker.join();
  }
  EXPECT_FALSE(at::hasGlobalCallbacks());
}

TEST(XProfCallbackHandlerTest, BackToBackSessionsCaptureExactSingleEventEach) {
  // Static initialization + alwayslink registered the tracer factory, not a
  // global callback. Run Session 1 before any explicit RegisterTracerFactory()
  // call to verify static registration alone activates per-session tracing.
  ASSERT_FALSE(at::hasGlobalCallbacks());

  tensorflow::profiler::XSpace xspace1;
  {
    auto session1 = tsl::ProfilerSession::Create(MakeProfileOptions(2));
    ASSERT_NE(session1, nullptr);
    ASSERT_TRUE(session1->Status().ok());
    EXPECT_TRUE(at::hasGlobalCallbacks());
    {
      RECORD_USER_SCOPE("marker_session_1");
    }
    ASSERT_TRUE(session1->CollectData(&xspace1).ok());
    EXPECT_FALSE(at::hasGlobalCallbacks());
  }

  {
    RECORD_USER_SCOPE("between_sessions_marker");
  }

  // Repeated explicit calls must remain idempotent and never register duplicate
  // per-session callbacks.
  XProfCallbackHandler::RegisterTracerFactory();
  XProfCallbackHandler::RegisterTracerFactory();
  ASSERT_FALSE(at::hasGlobalCallbacks());

  tensorflow::profiler::XSpace xspace2;
  {
    auto session2 = tsl::ProfilerSession::Create(MakeProfileOptions(2));
    ASSERT_NE(session2, nullptr);
    ASSERT_TRUE(session2->Status().ok());
    EXPECT_TRUE(at::hasGlobalCallbacks());
    {
      RECORD_USER_SCOPE("marker_session_2");
    }
    ASSERT_TRUE(session2->CollectData(&xspace2).ok());
    EXPECT_FALSE(at::hasGlobalCallbacks());
  }

  EXPECT_EQ(CountHostCpuEvents(xspace1, "marker_session_1"), 1);
  EXPECT_EQ(CountHostCpuEvents(xspace1, "between_sessions_marker"), 0);
  EXPECT_EQ(CountHostCpuEvents(xspace1, "marker_session_2"), 0);

  EXPECT_EQ(CountHostCpuEvents(xspace2, "marker_session_2"), 1);
  EXPECT_EQ(CountHostCpuEvents(xspace2, "marker_session_1"), 0);
  EXPECT_EQ(CountHostCpuEvents(xspace2, "between_sessions_marker"), 0);
}

TEST(XProfCallbackHandlerTest, HostTracerLevelZeroSkipsCallbackRegistration) {
  ASSERT_FALSE(at::hasGlobalCallbacks());
  auto session = tsl::ProfilerSession::Create(MakeProfileOptions(0));
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(session->Status().ok());
  EXPECT_FALSE(at::hasGlobalCallbacks());
  {
    RECORD_USER_SCOPE("device_only_marker");
  }
  tensorflow::profiler::XSpace xspace;
  ASSERT_TRUE(session->CollectData(&xspace).ok());
  EXPECT_FALSE(at::hasGlobalCallbacks());
  EXPECT_EQ(CountHostCpuEvents(xspace, "device_only_marker"), 0);
}

TEST(XProfCallbackHandlerTest, UncollectedSessionDestructionRemovesCallback) {
  ASSERT_FALSE(at::hasGlobalCallbacks());
  {
    auto session = tsl::ProfilerSession::Create(MakeProfileOptions(2));
    ASSERT_NE(session, nullptr);
    ASSERT_TRUE(session->Status().ok());
    EXPECT_TRUE(at::hasGlobalCallbacks());
    RECORD_USER_SCOPE("uncollected_session_marker");
  }
  EXPECT_FALSE(at::hasGlobalCallbacks());
}

}  // namespace
}  // namespace torch_tpu
