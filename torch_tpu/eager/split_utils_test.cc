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

#include "torch_tpu/eager/split_utils.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"

namespace torch_tpu {
namespace {

// A dummy MLIR op builder for testing purposes.
absl::StatusOr<DynamicMlirOpResults> DummyBuilder(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs) {
  return DynamicMlirOpResults{};
}

TEST(SplitUtilsTest, ApplySplitPointsSorted) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create a graph of:
  //       / -> c
  // a -> b
  //       \ -> d
  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or;
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kEmpty, DummyBuilder,
      /*inputs=*/{}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  DeviceBufferRef ref_a = refs_or.value()[0];

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  DeviceBufferRef ref_b = refs_or.value()[0];

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_b}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  DeviceBufferRef ref_c = refs_or.value()[0];

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_b}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  DeviceBufferRef ref_d = refs_or.value()[0];

  // Get the traversal of the graph.
  absl::StatusOr<absl_nonnull std::unique_ptr<Traversal>> traversal_or;
  traversal_or = Traversal::Create({ref_c, ref_d});
  ASSERT_TRUE(traversal_or.ok());
  auto& traversal = traversal_or.value();

  // Sort here so we can `use_sorted=true` in ApplySplitPoints later.
  traversal->SortByCreationOrder();

  // Set nodes b and d as split points.
  absl::flat_hash_set<const DeviceBufferList*> split_points = {
      ref_b.device_buffer_list().get(), ref_d.device_buffer_list().get()};

  absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
      traversals_or;
  traversals_or =
      ApplySplitPoints(std::move(traversal), split_points, /*use_sorted=*/true);
  ASSERT_TRUE(traversals_or.ok());
  auto& traversals = traversals_or.value();

  // We should get two traversals; one for {a, b} and one for {c, d}.
  // Node c is not in the graph for any split point; but, because we are using
  // `use_sorted=true`, it is included in the next traversal (for {d}).
  ASSERT_THAT(traversals, testing::SizeIs(2));
  EXPECT_THAT(traversals[0]->execution_order(), testing::SizeIs(2));
  ASSERT_THAT(traversals[0]->outputs(), testing::SizeIs(1));
  EXPECT_EQ(traversals[0]->outputs()[0], ref_b);

  ASSERT_THAT(traversals[1]->arguments(), testing::SizeIs(1));
  EXPECT_EQ(traversals[1]->arguments()[0], ref_b);
  EXPECT_THAT(traversals[1]->execution_order(), testing::SizeIs(2));
  ASSERT_THAT(traversals[1]->outputs(), testing::SizeIs(1));
  EXPECT_EQ(traversals[1]->outputs()[0], ref_d);
}

TEST(SplitUtilsTest, ApplySplitPointsUnsorted) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create a graph of:
  //       / -> c
  // a -> b
  //       \ -> d
  absl::StatusOr<std::vector<DeviceBufferRef>> refs_or;
  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kEmpty, DummyBuilder,
      /*inputs=*/{}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  DeviceBufferRef ref_a = refs_or.value()[0];

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_a}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  DeviceBufferRef ref_b = refs_or.value()[0];

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_b}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  DeviceBufferRef ref_c = refs_or.value()[0];

  refs_or = DeviceBufferList::CreateDeferred(
      OpName::kAdd, DummyBuilder, {ref_b}, OpParamCacheKeys::Empty(), {shape});
  ASSERT_TRUE(refs_or.ok());
  DeviceBufferRef ref_d = refs_or.value()[0];

  // Get the traversal of the graph.
  absl::StatusOr<absl_nonnull std::unique_ptr<Traversal>> traversal_or;
  traversal_or = Traversal::Create({ref_c, ref_d});
  ASSERT_TRUE(traversal_or.ok());
  auto& traversal = traversal_or.value();

  // Set nodes b and d as split points.
  absl::flat_hash_set<const DeviceBufferList*> split_points = {
      ref_b.device_buffer_list().get(), ref_d.device_buffer_list().get()};

  absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
      traversals_or;
  traversals_or = ApplySplitPoints(std::move(traversal), split_points,
                                   /*use_sorted=*/false);
  ASSERT_TRUE(traversals_or.ok());
  auto& traversals = traversals_or.value();

  // We should get two traversals; one for {a, b} and one for {d}.
  // Node c is not included since it is not a part of the graph for any split
  // point, and with `use_sorted=false`, we retrace the graph each time.
  ASSERT_THAT(traversals, testing::SizeIs(2));
  EXPECT_THAT(traversals[0]->execution_order(), testing::SizeIs(2));
  ASSERT_THAT(traversals[0]->outputs(), testing::SizeIs(1));
  EXPECT_EQ(traversals[0]->outputs()[0], ref_b);

  ASSERT_THAT(traversals[1]->arguments(), testing::SizeIs(1));
  EXPECT_EQ(traversals[1]->arguments()[0], ref_b);
  EXPECT_THAT(traversals[1]->execution_order(), testing::SizeIs(1));
  ASSERT_THAT(traversals[1]->outputs(), testing::SizeIs(1));
  EXPECT_EQ(traversals[1]->outputs()[0], ref_d);
}

}  // namespace
}  // namespace torch_tpu
