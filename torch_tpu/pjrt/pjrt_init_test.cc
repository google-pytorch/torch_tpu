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

#include <cstdlib>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "torch_tpu/pjrt/pjrt_state.h"

namespace torch_tpu {
namespace {

using testing::ExitedWithCode;

class PjRtInitTest : public ::testing::Test {
 protected:
  void TearDown() override { PjrtBackend::GetInstance().Shutdown(); }
};

TEST_F(PjRtInitTest, InitializePjRtIsLazy) {
  EXPECT_EXIT(
      {
        auto check_lazy = []() -> bool {
          if (PjrtBackend::GetInstance().IsInitialized()) {
            return false;
          }

          PjrtBackend::GetInstance().SetPjRtInitializationOptions(
              {.device_type = "xla_cpu"});
          if (PjrtBackend::GetInstance().IsInitialized()) {
            return false;
          }

          if (PjrtBackend::GetInstance().GetClient() == nullptr) {
            return false;
          }
          if (!PjrtBackend::GetInstance().IsInitialized()) {
            return false;
          }
          return true;
        };
        std::exit(check_lazy() ? 0 : 1);
      },
      ExitedWithCode(0), "");
}

TEST_F(PjRtInitTest, InitializePjRtSetsCorrectDeviceCount) {
  ASSERT_FALSE(PjrtBackend::GetInstance().IsInitialized());

  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "xla_cpu"});

  auto* client = PjrtBackend::GetInstance().GetClient();
  ASSERT_NE(client, nullptr);
  EXPECT_TRUE(PjrtBackend::GetInstance().IsInitialized());

  auto device_count_or = PjrtBackend::GetInstance().GetGlobalDeviceCount();
  ASSERT_EQ(device_count_or.status(), absl::OkStatus());
  EXPECT_EQ(device_count_or.value(), 1);
}

}  // namespace
}  // namespace torch_tpu
