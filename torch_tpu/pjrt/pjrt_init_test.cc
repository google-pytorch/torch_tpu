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
#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/pjrt/pjrt_state.h"

namespace torch_tpu {
namespace {

using testing::ExitedWithCode;

class PjRtInitTest : public testing::Test {
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

  xla::PjRtClient* client = PjrtBackend::GetInstance().GetClient();
  ASSERT_NE(client, nullptr);
  EXPECT_TRUE(PjrtBackend::GetInstance().IsInitialized());

  auto device_count_or = PjrtBackend::GetInstance().GetGlobalDeviceCount();
  ASSERT_EQ(device_count_or.status(), absl::OkStatus());
  EXPECT_EQ(device_count_or.value(), 1);
}

TEST_F(PjRtInitTest, InitializeExpandsRankInEnvVars) {
  auto cleanup = absl::MakeCleanup([] {
    unsetenv(kXlaFlagsEnvVar);
    unsetenv(kLibtpuInitArgsEnvVar);
    unsetenv(kRankEnvVar);
  });

  // Use a valid XLA flag to avoid parsing errors in XLA initialization.
  SetEnv(kXlaFlagsEnvVar, "--xla_disable_hlo_passes=some_pass_${RANK}");
  SetEnv(kLibtpuInitArgsEnvVar, "some_args_${RANK}");
  SetEnv(kRankEnvVar, "1234");

  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "xla_cpu"});
  xla::PjRtClient* client = PjrtBackend::GetInstance().GetClient();
  ASSERT_NE(client, nullptr);

  const char* xla_flags = std::getenv(kXlaFlagsEnvVar);
  EXPECT_NE(xla_flags, nullptr);
  EXPECT_STREQ(xla_flags, "--xla_disable_hlo_passes=some_pass_1234");

  const char* libtpu_args = std::getenv(kLibtpuInitArgsEnvVar);
  EXPECT_NE(libtpu_args, nullptr);
  EXPECT_STREQ(libtpu_args, "some_args_1234");
}

TEST_F(PjRtInitTest, DeviceLockFailureClassification) {
  // Test false-positive rejections (generic errors):
  EXPECT_FALSE(IsDeviceAcquisitionFailure(absl::OkStatus()));
  EXPECT_FALSE(IsDeviceAcquisitionFailure(TT_ERROR(error::kInvalidArgument)
                                          << "invalid shape for tensor"));
  EXPECT_FALSE(IsDeviceAcquisitionFailure(TT_ERROR(error::kNotFound)
                                          << "file not found"));
  EXPECT_FALSE(
      IsDeviceAcquisitionFailure(TT_ERROR(error::kPermissionDenied)
                                 << "Permission denied reading config.json"));
  EXPECT_FALSE(IsDeviceAcquisitionFailure(
      TT_ERROR(error::kFailedPrecondition)
      << "TPU initialization failed: graph compilation error"));

  // Test false-positive rejections (EBUSY / EACCES on non-device resources):
  EXPECT_FALSE(IsDeviceAcquisitionFailure(
      TT_ERROR(error::kUnavailable)
      << "Failed to bind rendezvous port 8080: EBUSY"));
  EXPECT_FALSE(IsDeviceAcquisitionFailure(
      TT_ERROR(error::kUnavailable)
      << "Resource temporarily unavailable: Device or "
         "resource busy in lockfile /tmp/cache.lock"));
  EXPECT_FALSE(IsDeviceAcquisitionFailure(
      TT_ERROR(error::kPermissionDenied)
      << "Failed to create cache directory /tmp/xla_cache: EACCES"));
  EXPECT_FALSE(IsDeviceAcquisitionFailure(
      TT_ERROR(error::kPermissionDenied)
      << "Permission denied reading config /tmp/xla.json"));
  EXPECT_FALSE(IsDeviceAcquisitionFailure(
      TT_ERROR(error::kInternal) << "os error 16 occurred on socket stream"));

  // Test true collision detections (hardware device lock contention):
  EXPECT_TRUE(
      IsDeviceAcquisitionFailure(TT_ERROR(error::kFailedPrecondition)
                                 << "open(/dev/vfio/633): Device or resource "
                                    "busy: Couldn't open iommu group"));
  EXPECT_TRUE(IsDeviceAcquisitionFailure(
      TT_ERROR(error::kUnavailable)
      << "open(/dev/vfio/633): EBUSY (os error 16)"));
  EXPECT_TRUE(IsDeviceAcquisitionFailure(
      TT_ERROR(error::kPermissionDenied)
      << "Permission denied opening /dev/vfio/vfio"));
  EXPECT_TRUE(IsDeviceAcquisitionFailure(
      TT_ERROR(error::kPermissionDenied)
      << "open(/dev/accel0): Permission denied (EACCES)"));
  EXPECT_TRUE(IsDeviceAcquisitionFailure(
      TT_ERROR(error::kAborted)
      << "InitializePjrtPlugin failed: The TPU is already in "
         "use by process with pid 12345"));
  EXPECT_TRUE(IsDeviceAcquisitionFailure(
      TT_ERROR(error::kAborted)
      << "Internal error when accessing libtpu multi-process lockfile"));
  EXPECT_TRUE(IsDeviceAcquisitionFailure(
      TT_ERROR(error::kFailedPrecondition)
      << "open(/dev/vfio/vfio): vfio container cannot be opened"));
}

TEST_F(PjRtInitTest, EnrichDeviceAcquisitionStatus) {
  absl::Status raw_status = TT_ERROR(error::kFailedPrecondition)
                            << "open(/dev/vfio/0): Device or resource busy";
  absl::Status enriched = EnrichDeviceAcquisitionStatus(raw_status);

  EXPECT_EQ(enriched.code(), error::kFailedPrecondition);
  EXPECT_TRUE(absl::StrContains(
      enriched.message(), "[TorchTPU] Failed to acquire TPU hardware device:"));
  EXPECT_TRUE(absl::StrContains(enriched.message(),
                                "TPU_VISIBLE_DEVICES or LOCAL_RANK"));
}

TEST_F(PjRtInitTest, InitializationFailureCachesStatusAndReturnsNullptr) {
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "invalid_device_type"});
  xla::PjRtClient* client = PjrtBackend::GetInstance().GetClient();
  EXPECT_EQ(client, nullptr);
  EXPECT_FALSE(PjrtBackend::GetInstance().IsInitialized());
  EXPECT_EQ(PjrtBackend::GetInstance().GetInitializationStatus().code(),
            error::kInvalidArgument);
}

}  // namespace
}  // namespace torch_tpu
