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

#include <optional>
#include <thread>  // NOLINT(build/c++11)

#include "absl/cleanup/cleanup.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

TEST(ContextStateAccess, IsAllowedByDefault) {
  EXPECT_EQ(GetContextState<EagerModeContextState>(), std::nullopt);
}

TEST(ContextStateAccessDeathTest, CrashOnGetContextStateIfDisallowed) {
  EXPECT_DEATH(
      {
        std::thread t([&]() {
          DisallowThisThreadToAccessContextState();
          GetContextState<EagerModeContextState>();
        });
        t.join();
      },
      "dispatch thread");
}

TEST(ContextStateMutation, IsAllowedByDefault) {
  PushContextState(EagerMode::kDeferAndFuse);
  absl::Cleanup cleanup =
      absl::MakeCleanup([]() { PopContextState<EagerModeContextState>(); });

  EXPECT_EQ(GetContextState<EagerModeContextState>(), EagerMode::kDeferAndFuse);
}

TEST(ContextStateMutationDeathTest, CrashOnPushAndPopContextStateIfDisallowed) {
  EXPECT_DEATH(
      {
        std::thread t([&]() {
          DisallowThisThreadToAccessContextState();
          PushContextState(EagerMode::kDeferAndFuse);
        });
        t.join();
      },
      "dispatch thread");

  EXPECT_DEATH(
      {
        std::thread t([&]() {
          DisallowThisThreadToAccessContextState();
          PopContextState<EagerModeContextState>();
        });
        t.join();
      },
      "dispatch thread");
}

TEST(ContextStateAccess, IsIsolatedAcrossThreads) {
  EXPECT_EQ(GetContextState<EagerModeContextState>(), std::nullopt);

  EXPECT_DEATH(
      {
        std::thread worker([&]() {
          DisallowThisThreadToAccessContextState();
          GetContextState<EagerModeContextState>();
        });
        worker.join();
      },
      "dispatch thread");

  EXPECT_EQ(GetContextState<EagerModeContextState>(), std::nullopt);
}

}  // namespace
}  // namespace torch_tpu
