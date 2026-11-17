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

#include "csrc/ops/copy_from/copy_common.h"

#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "csrc/common/macro_utils.h"
#include "csrc/eager/eager_mode.h"
#include "gtest/gtest.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

TEST(GetTransferSyncModeTest, DeferNever) {
  SetEagerMode(EagerMode::kDeferNever);
  EXPECT_EQ(GetTransferSyncMode(/*non_blocking=*/true),
            TransferSyncMode::kEagerNonBlocking);
  EXPECT_EQ(GetTransferSyncMode(/*non_blocking=*/false),
            TransferSyncMode::kEagerBlocking);
}

TEST(GetTransferSyncModeTest, DeferAndFuse) {
  SetEagerMode(EagerMode::kDeferAndFuse);
  EXPECT_EQ(GetTransferSyncMode(/*non_blocking=*/true),
            TransferSyncMode::kEagerNonBlocking);
  EXPECT_EQ(GetTransferSyncMode(/*non_blocking=*/false),
            TransferSyncMode::kEagerBlocking);
}

TEST(GetTransferSyncModeTest, InternalDeferAll) {
  SetEagerMode(EagerMode::kInternalDeferAll);
  EXPECT_EQ(GetTransferSyncMode(/*non_blocking=*/true),
            TransferSyncMode::kEagerNonBlocking);
  EXPECT_EQ(GetTransferSyncMode(/*non_blocking=*/false),
            TransferSyncMode::kEagerBlocking);
}

TEST(GetTransferSyncModeTest, InternalCompileFxGraph) {
  SetEagerMode(EagerMode::kInternalCompileFxGraph);
  // FX graph compile overrides non_blocking to kIsolated.
  EXPECT_EQ(GetTransferSyncMode(/*non_blocking=*/true),
            TransferSyncMode::kIsolated);
  EXPECT_EQ(GetTransferSyncMode(/*non_blocking=*/false),
            TransferSyncMode::kIsolated);
}

TEST(GetTransferSyncModeTest, DeferNeverAndLaunchBlocking) {
  SetEagerMode(EagerMode::kDeferNeverAndLaunchBlocking);
  // Launch blocking overrides non_blocking to kEagerBlocking.
  EXPECT_EQ(GetTransferSyncMode(/*non_blocking=*/true),
            TransferSyncMode::kEagerBlocking);
  EXPECT_EQ(GetTransferSyncMode(/*non_blocking=*/false),
            TransferSyncMode::kEagerBlocking);
}

TEST(ValidateShellDtypeCastTest, SameDtypeAllowed) {
  EXPECT_TRUE(
      ValidateShellDtypeCast(at::ScalarType::Bits8, at::ScalarType::Bits8)
          .ok());
  EXPECT_TRUE(
      ValidateShellDtypeCast(at::ScalarType::Float, at::ScalarType::Float)
          .ok());
}

TEST(ValidateShellDtypeCastTest, CastToOrFromShellDtypeRejected) {
  const std::vector<at::ScalarType> uncastable_shell_types = {
      at::ScalarType::Bits1x8,        at::ScalarType::Bits2x4,
      at::ScalarType::Bits4x2,        at::ScalarType::Bits8,
      at::ScalarType::Bits16,         at::ScalarType::UInt1,
      at::ScalarType::UInt2,          at::ScalarType::UInt3,
      at::ScalarType::UInt4,          at::ScalarType::UInt5,
      at::ScalarType::UInt6,          at::ScalarType::UInt7,
      at::ScalarType::Int1,           at::ScalarType::Int2,
      at::ScalarType::Int3,           at::ScalarType::Int4,
      at::ScalarType::Int5,           at::ScalarType::Int6,
      at::ScalarType::Int7,
#if TT_HAS_FLOAT8_E5M3FNU
      at::ScalarType::Float8_e5m3fnu,
#endif
  };
  for (at::ScalarType shell_type : uncastable_shell_types) {
    absl::Status status_from =
        ValidateShellDtypeCast(shell_type, at::ScalarType::Float);
    EXPECT_FALSE(status_from.ok());
    EXPECT_NE(status_from.message().find(
                  "is a shell dtype with no defined conversions"),
              std::string_view::npos);

    absl::Status status_to =
        ValidateShellDtypeCast(at::ScalarType::Float, shell_type);
    EXPECT_FALSE(status_to.ok());
    EXPECT_NE(status_to.message().find(
                  "is a shell dtype with no defined conversions"),
              std::string_view::npos);

    EXPECT_TRUE(ValidateShellDtypeCast(shell_type, shell_type).ok());
  }
}

TEST(ValidateShellDtypeCastTest, ShellToDifferentShellRejected) {
  EXPECT_FALSE(
      ValidateShellDtypeCast(at::ScalarType::Bits8, at::ScalarType::Bits16)
          .ok());
}

TEST(ValidateShellDtypeCastTest, NonShellDtypesAllowed) {
  EXPECT_TRUE(
      ValidateShellDtypeCast(at::ScalarType::Float, at::ScalarType::Int).ok());
  EXPECT_TRUE(
      ValidateShellDtypeCast(at::ScalarType::Float, at::ScalarType::BFloat16)
          .ok());
}

TEST(ValidateShellDtypeCastTest, Float4Allowed) {
  EXPECT_TRUE(ValidateShellDtypeCast(at::ScalarType::Float4_e2m1fn_x2,
                                     at::ScalarType::Float)
                  .ok());
  EXPECT_TRUE(ValidateShellDtypeCast(at::ScalarType::Float,
                                     at::ScalarType::Float4_e2m1fn_x2)
                  .ok());
}

}  // namespace
}  // namespace torch_tpu
