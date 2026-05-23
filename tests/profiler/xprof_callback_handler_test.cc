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

#include "torch_tpu/_internal/profiler/xprof_callback_handler.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "ATen/record_function.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

TEST(XProfCallbackHandlerTest, StressTestThreadSafety) {
  std::atomic<bool> stop_workers{false};

  // 1. Spawn worker threads executing RecordFunction continuously
  std::vector<std::thread> workers;
  for (int i = 0; i < 8; ++i) {
    workers.emplace_back([&]() {
      while (!stop_workers.load(std::memory_order_relaxed)) {
        at::RecordFunction guard(at::RecordScope::FUNCTION);
        if (guard.isActive()) {
          // Simulate tiny work
        }
      }
    });
  }

  // 2. Concurrently add and remove the callback repeatedly
  for (int i = 0; i < 1000; ++i) {
    XProfCallbackHandler handler;
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }

  // 3. Clean up
  stop_workers.store(true);
  for (auto& worker : workers) {
    worker.join();
  }
}

}  // namespace
}  // namespace torch_tpu
