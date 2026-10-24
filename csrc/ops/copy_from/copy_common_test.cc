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

#include "csrc/eager/eager_mode.h"
#include "gtest/gtest.h"

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

}  // namespace
}  // namespace torch_tpu
