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

#include "csrc/pjrt/pjrt_state.h"

#include <cstdint>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Return;

// A client whose only working method is a mocked `GetDefaultLayout()`.
class FakePjRtClient : public xla::PjRtClient {
 public:
  MOCK_METHOD(absl::StatusOr<xla::Layout>, GetDefaultLayout,
              (xla::PrimitiveType element_type, absl::Span<const int64_t> dims),
              (override));

  int process_index() const override { return 0; }
  int device_count() const override { return 0; }
  int addressable_device_count() const override { return 0; }
  absl::Span<xla::PjRtDevice* const> devices() const override { return {}; }
  absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
    return {};
  }
  absl::Span<xla::PjRtMemorySpace* const> memory_spaces() const override {
    return {};
  }
  xla::PjRtPlatformId platform_id() const override { return 0; }
  std::string_view platform_name() const override { return "fake"; }
  std::string_view platform_version() const override { return "fake"; }
};

xla::Layout RowMajor() { return xla::LayoutUtil::MakeLayout({1, 0}); }
xla::Layout ColumnMajor() { return xla::LayoutUtil::MakeLayout({0, 1}); }

TEST(DefaultLayoutCacheTest, ReturnsClientDefaultLayout) {
  FakePjRtClient client;
  EXPECT_CALL(client, GetDefaultLayout(xla::BF16, ElementsAre(16, 128)))
      .WillOnce(Return(ColumnMajor()));
  DefaultLayoutCache cache;

  EXPECT_THAT(cache.Get(client, xla::BF16, {16, 128}),
              IsOkAndHolds(ColumnMajor()));
}

TEST(DefaultLayoutCacheTest, QueriesClientOncePerKey) {
  FakePjRtClient client;
  EXPECT_CALL(client, GetDefaultLayout(xla::BF16, ElementsAre(16, 128)))
      .WillOnce(Return(ColumnMajor()));
  DefaultLayoutCache cache;

  EXPECT_THAT(cache.Get(client, xla::BF16, {16, 128}),
              IsOkAndHolds(ColumnMajor()));
  EXPECT_THAT(cache.Get(client, xla::BF16, {16, 128}),
              IsOkAndHolds(ColumnMajor()));
}

TEST(DefaultLayoutCacheTest, KeysByElementTypeAndDims) {
  FakePjRtClient client;
  EXPECT_CALL(client, GetDefaultLayout(xla::BF16, ElementsAre(16, 128)))
      .WillOnce(Return(ColumnMajor()));
  EXPECT_CALL(client, GetDefaultLayout(xla::F32, ElementsAre(16, 128)))
      .WillOnce(Return(RowMajor()));
  EXPECT_CALL(client, GetDefaultLayout(xla::BF16, ElementsAre(128, 16)))
      .WillOnce(Return(RowMajor()));
  DefaultLayoutCache cache;

  EXPECT_THAT(cache.Get(client, xla::BF16, {16, 128}),
              IsOkAndHolds(ColumnMajor()));
  EXPECT_THAT(cache.Get(client, xla::F32, {16, 128}), IsOkAndHolds(RowMajor()));
  EXPECT_THAT(cache.Get(client, xla::BF16, {128, 16}),
              IsOkAndHolds(RowMajor()));
  EXPECT_THAT(cache.Get(client, xla::BF16, {16, 128}),
              IsOkAndHolds(ColumnMajor()));
}

TEST(DefaultLayoutCacheTest, DoesNotCacheErrors) {
  FakePjRtClient client;
  const absl::Status not_ready = TT_ERROR(error::kUnavailable) << "not ready";
  EXPECT_CALL(client, GetDefaultLayout(xla::BF16, ElementsAre(16, 128)))
      .WillOnce(Return(not_ready))
      .WillOnce(Return(ColumnMajor()));
  DefaultLayoutCache cache;

  EXPECT_THAT(cache.Get(client, xla::BF16, {16, 128}),
              StatusIs(error::kUnavailable));
  EXPECT_THAT(cache.Get(client, xla::BF16, {16, 128}),
              IsOkAndHolds(ColumnMajor()));
}

TEST(DefaultLayoutCacheTest, ClearDropsCachedLayouts) {
  FakePjRtClient old_client;
  FakePjRtClient new_client;
  EXPECT_CALL(old_client, GetDefaultLayout(xla::BF16, ElementsAre(16, 128)))
      .WillOnce(Return(ColumnMajor()));
  EXPECT_CALL(new_client, GetDefaultLayout(xla::BF16, ElementsAre(16, 128)))
      .WillOnce(Return(RowMajor()));
  DefaultLayoutCache cache;

  EXPECT_THAT(cache.Get(old_client, xla::BF16, {16, 128}),
              IsOkAndHolds(ColumnMajor()));
  cache.Clear();
  EXPECT_THAT(cache.Get(new_client, xla::BF16, {16, 128}),
              IsOkAndHolds(RowMajor()));
}

TEST(DefaultLayoutCacheTest, ClearsWhenFull) {
  FakePjRtClient client;
  EXPECT_CALL(client, GetDefaultLayout(xla::BF16, ElementsAre(1)))
      .Times(2)
      .WillRepeatedly(Return(RowMajor()));
  EXPECT_CALL(client, GetDefaultLayout(xla::BF16, ElementsAre(2)))
      .WillOnce(Return(RowMajor()));
  EXPECT_CALL(client, GetDefaultLayout(xla::BF16, ElementsAre(3)))
      .WillOnce(Return(RowMajor()));
  DefaultLayoutCache cache(/*max_entries=*/2);

  EXPECT_THAT(cache.Get(client, xla::BF16, {1}), IsOkAndHolds(RowMajor()));
  EXPECT_THAT(cache.Get(client, xla::BF16, {2}), IsOkAndHolds(RowMajor()));
  // The cache is full, so adding {3} drops {1} and {2}.
  EXPECT_THAT(cache.Get(client, xla::BF16, {3}), IsOkAndHolds(RowMajor()));
  EXPECT_THAT(cache.Get(client, xla::BF16, {1}), IsOkAndHolds(RowMajor()));
  EXPECT_THAT(cache.Get(client, xla::BF16, {3}), IsOkAndHolds(RowMajor()));
}

}  // namespace
}  // namespace torch_tpu
