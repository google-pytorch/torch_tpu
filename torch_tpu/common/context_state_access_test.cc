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
#include "gtest/gtest.h"
#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"

namespace torch_tpu {
namespace {

TEST(ContextStateAccessDeathTest, IsNotAllowedByDefault) {
  EXPECT_DEATH(
      {
        std::thread t([&]() { GetContextState<EagerModeContextState>(); });
        t.join();
      },
      "GetContextState must only be called from a dispatch thread");
}

TEST(ContextStateMutationDeathTest, IsNotAllowedByDefault) {
  EXPECT_DEATH(
      {
        std::thread t([&]() { PushContextState(EagerMode::kDeferAndFuse); });
        t.join();
      },
      "PushContextState must only be called from a dispatch thread");

  EXPECT_DEATH(
      {
        std::thread t([&]() { PopContextState<EagerModeContextState>(); });
        t.join();
      },
      "PopContextState must only be called from a dispatch thread");
}

TEST(AllowThisThreadToAccessContextState, AllowsContextStateAccessAndMutation) {
  AllowThisThreadToAccessContextState();
  EXPECT_EQ(GetContextState<EagerModeContextState>(), std::nullopt);

  PushContextState(EagerMode::kDeferAndFuse);
  auto cleanup =
      absl::MakeCleanup([]() { PopContextState<EagerModeContextState>(); });
  EXPECT_EQ(GetContextState<EagerModeContextState>(), EagerMode::kDeferAndFuse);
}

TEST(ContextStateAccessDeathTest, IsIsolatedAcrossThreads) {
  AllowThisThreadToAccessContextState();
  EXPECT_EQ(GetContextState<EagerModeContextState>(), std::nullopt);

  EXPECT_DEATH(
      {
        std::thread worker([&]() { GetContextState<EagerModeContextState>(); });
        worker.join();
      },
      "GetContextState must only be called from a dispatch thread");

  EXPECT_EQ(GetContextState<EagerModeContextState>(), std::nullopt);
}

}  // namespace
}  // namespace torch_tpu
