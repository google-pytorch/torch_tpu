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

#include "csrc/pjrt/runtime_shape_metadata_pool.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

xla::Shape MakeMetadataShape(const int64_t rank, const int64_t tile = 128) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {rank});
  *shape.mutable_layout() =
      xla::Layout({0}, {xla::Tile({tile})})
          .set_memory_space(xla::Layout::kHostMemorySpace);
  return shape;
}

class RuntimeShapeMetadataPoolTest : public testing::Test {
 protected:
  void SetUp() override {
    PjrtBackend& backend = PjrtBackend::GetInstance();
    backend.SetPjRtInitializationOptions({.device_type = "tpu"});
    TT_ASSERT_OK_AND_ASSIGN(device_, backend.GetDevice());
    pool_ = RuntimeShapeMetadataPool::Create();
  }

  void TearDown() override {
    pool_.reset();
    PjrtBackend::GetInstance().Shutdown();
  }

  absl::StatusOr<std::shared_ptr<xla::PjRtBuffer>> Acquire(
      const std::vector<int32_t>& dimensions) {
    return pool_->Acquire(dimensions, MakeMetadataShape(dimensions.size()),
                          *device_);
  }

  static absl::StatusOr<std::vector<int32_t>> Read(xla::PjRtBuffer& buffer) {
    TT_ASSIGN_OR_RETURN(const std::shared_ptr<xla::Literal> literal,
                        WrapXlaError(buffer.ToLiteral().Await(),
                                     /*context=*/"failed to read the buffer"));
    const auto elements = literal->data<int32_t>();
    return std::vector<int32_t>(elements.begin(), elements.end());
  }

  xla::PjRtDevice* device_ = nullptr;
  std::shared_ptr<RuntimeShapeMetadataPool> pool_;
};

TEST_F(RuntimeShapeMetadataPoolTest, ReturnsPinnedHostBufferOfDimensions) {
  const xla::Shape shape = MakeMetadataShape(/*rank=*/2);
  TT_ASSERT_OK_AND_ASSIGN(const std::shared_ptr<xla::PjRtBuffer> buffer,
                          Acquire({3, 5}));

  EXPECT_EQ(buffer->memory_space()->kind(), "pinned_host");
  EXPECT_THAT(buffer->on_device_shape().dimensions(), ElementsAre(2));
  EXPECT_EQ(buffer->on_device_shape().layout().tiles(), shape.layout().tiles());
  EXPECT_THAT(Read(*buffer), IsOkAndHolds(ElementsAre(3, 5)));
}

TEST_F(RuntimeShapeMetadataPoolTest, OverwritesReleasedBuffer) {
  TT_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::PjRtBuffer> first,
                          Acquire({3, 5}));
  const xla::PjRtBuffer* const released = first.get();
  first.reset();

  // Other dimensions, unlike a cache keyed by them.
  TT_ASSERT_OK_AND_ASSIGN(const std::shared_ptr<xla::PjRtBuffer> second,
                          Acquire({7, 11}));

  EXPECT_EQ(second.get(), released);
  EXPECT_THAT(Read(*second), IsOkAndHolds(ElementsAre(7, 11)));
}

TEST_F(RuntimeShapeMetadataPoolTest, KeepsBufferUntilLastCopyIsDestroyed) {
  TT_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::PjRtBuffer> first,
                          Acquire({3, 5}));
  const xla::PjRtBuffer* const first_ptr = first.get();
  std::shared_ptr<xla::PjRtBuffer> copy = first;
  first.reset();

  // The first buffer is still in use, even for the same dimensions.
  TT_ASSERT_OK_AND_ASSIGN(const std::shared_ptr<xla::PjRtBuffer> second,
                          Acquire({3, 5}));

  EXPECT_NE(second.get(), first_ptr);
  EXPECT_THAT(Read(*copy), IsOkAndHolds(ElementsAre(3, 5)));
  copy.reset();
  // Once the last copy is gone, the first buffer returns to the pool.
  TT_ASSERT_OK_AND_ASSIGN(const std::shared_ptr<xla::PjRtBuffer> reused,
                          Acquire({13, 17}));
  EXPECT_EQ(reused.get(), first_ptr);
}

TEST_F(RuntimeShapeMetadataPoolTest, KeysIdleBuffersByRankAndLayout) {
  const xla::Shape other_shape = MakeMetadataShape(/*rank=*/2, /*tile=*/256);
  TT_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::PjRtBuffer> rank_2,
                          Acquire({3, 5}));
  const xla::PjRtBuffer* const idle = rank_2.get();
  rank_2.reset();

  TT_ASSERT_OK_AND_ASSIGN(const std::shared_ptr<xla::PjRtBuffer> rank_3,
                          Acquire({2, 3, 5}));
  TT_ASSERT_OK_AND_ASSIGN(const std::shared_ptr<xla::PjRtBuffer> other_tile,
                          pool_->Acquire({7, 11}, other_shape, *device_));

  EXPECT_NE(rank_3.get(), idle);
  EXPECT_NE(other_tile.get(), idle);
  EXPECT_THAT(Read(*rank_3), IsOkAndHolds(ElementsAre(2, 3, 5)));
  EXPECT_THAT(Read(*other_tile), IsOkAndHolds(ElementsAre(7, 11)));
  EXPECT_EQ(other_tile->on_device_shape().layout().tiles(),
            other_shape.layout().tiles());
}

TEST_F(RuntimeShapeMetadataPoolTest, RejectsMismatchedShape) {
  EXPECT_THAT(pool_->Acquire({3, 5}, MakeMetadataShape(/*rank=*/3), *device_),
              StatusIs(error::kInvalidArgument));
}

TEST_F(RuntimeShapeMetadataPoolTest, DestroysBufferReleasedAfterPoolIsGone) {
  TT_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::PjRtBuffer> in_use,
                          Acquire({3, 5}));

  pool_.reset();

  // The caller still owns its buffer and destroys it safely on scope exit.
  EXPECT_THAT(Read(*in_use), IsOkAndHolds(ElementsAre(3, 5)));
}

TEST_F(RuntimeShapeMetadataPoolTest, BackendResetsPoolOnShutdown) {
  PjrtBackend& backend = PjrtBackend::GetInstance();
  TT_ASSERT_OK_AND_ASSIGN(std::shared_ptr<RuntimeShapeMetadataPool> pool,
                          backend.GetRuntimeShapeMetadataPool());
  const std::weak_ptr<RuntimeShapeMetadataPool> weak_pool = pool;
  TT_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<xla::PjRtBuffer> buffer,
      pool->Acquire({3, 5}, MakeMetadataShape(/*rank=*/2), *device_));
  buffer.reset();
  pool.reset();

  backend.Shutdown();

  EXPECT_TRUE(weak_pool.expired());
}

}  // namespace
}  // namespace torch_tpu
