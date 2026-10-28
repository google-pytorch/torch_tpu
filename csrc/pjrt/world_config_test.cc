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

#include "csrc/pjrt/world_config.h"

#include "absl/status/status_matchers.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using absl_testing::StatusIs;
using testing::AllOf;
using testing::HasSubstr;
using testing::Not;

TEST(ResolveWorldConfigTest, SingleProcessUsesOneDevice) {
  TT_ASSERT_OK_AND_ASSIGN(const WorldConfig config,
                          ResolveWorldConfig(/*world_size=*/1,
                                             /*device_count=*/8,
                                             /*virtual_multihost=*/false));
  EXPECT_EQ(config, (WorldConfig{.world_size = 1,
                                 .global_device_count = 1,
                                 .virtual_multihost = false}));
}

TEST(ResolveWorldConfigTest, WorldMatchingTheSliceIsRegular) {
  for (const bool virtual_multihost : {false, true}) {
    TT_ASSERT_OK_AND_ASSIGN(
        const WorldConfig config,
        ResolveWorldConfig(/*world_size=*/8, /*device_count=*/8,
                           virtual_multihost));
    EXPECT_EQ(config, (WorldConfig{.world_size = 8,
                                   .global_device_count = 8,
                                   .virtual_multihost = false}))
        << "virtual_multihost=" << virtual_multihost;
  }
}

TEST(ResolveWorldConfigTest, MultipleOfTheSliceIsRejectedWithoutTheOptIn) {
  EXPECT_THAT(ResolveWorldConfig(/*world_size=*/16, /*device_count=*/8,
                                 /*virtual_multihost=*/false),
              StatusIs(error::kInvalidArgument,
                       AllOf(HasSubstr("expected 1 or 8"), HasSubstr("got 16"),
                             HasSubstr(kTorchTpuVirtualMultiHostEnvVar))));
}

TEST(ResolveWorldConfigTest, MultipleOfTheSliceIsEmulatedWithTheOptIn) {
  TT_ASSERT_OK_AND_ASSIGN(const WorldConfig config,
                          ResolveWorldConfig(/*world_size=*/16,
                                             /*device_count=*/8,
                                             /*virtual_multihost=*/true));
  EXPECT_EQ(config, (WorldConfig{.world_size = 16,
                                 .global_device_count = 8,
                                 .virtual_multihost = true}));
  TT_ASSERT_OK_AND_ASSIGN(const WorldConfig four_slices,
                          ResolveWorldConfig(/*world_size=*/32,
                                             /*device_count=*/8,
                                             /*virtual_multihost=*/true));
  EXPECT_EQ(four_slices.global_device_count, 8);
  EXPECT_TRUE(four_slices.virtual_multihost);
}

TEST(ResolveWorldConfigTest, NonMultiplesAreRejectedEvenWithTheOptIn) {
  for (const int world_size : {3, 12, 15}) {
    EXPECT_THAT(ResolveWorldConfig(world_size, /*device_count=*/8,
                                   /*virtual_multihost=*/true),
                StatusIs(error::kInvalidArgument,
                         AllOf(HasSubstr("larger multiple"),
                               Not(HasSubstr("set TORCH_TPU")))))
        << "world_size=" << world_size;
  }
}

TEST(ResolveWorldConfigTest, SmallerWorldsAreRejected) {
  // A world smaller than the slice never was valid; the opt-in does not
  // change that and the hint is not offered (nothing to emulate).
  for (const bool virtual_multihost : {false, true}) {
    EXPECT_THAT(
        ResolveWorldConfig(/*world_size=*/4, /*device_count=*/8,
                           virtual_multihost),
        StatusIs(error::kInvalidArgument, Not(HasSubstr("set TORCH_TPU"))));
  }
}

TEST(ResolveWorldConfigTest, InvalidDeviceCountIsRejected) {
  EXPECT_THAT(ResolveWorldConfig(/*world_size=*/1, /*device_count=*/0,
                                 /*virtual_multihost=*/true),
              StatusIs(error::kInvalidArgument, HasSubstr("device count")));
}

}  // namespace
}  // namespace torch_tpu
