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

#include "csrc/eager/split_traversal.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/traversal.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

// A dummy MLIR op builder for testing purposes.
absl::StatusOr<DynamicMlirOpResults> DummyBuilder(
    mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs) {
  return DynamicMlirOpResults{};
}

TEST(SplitTraversalTest, ApplySplitPointsSorted) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // Create a graph of:
  //       / -> c
  // a -> b
  //       \ -> d
  // (letter indicates execution order)
  TT_ASSERT_OK_AND_ASSIGN(
      auto refs_a, DeviceBufferList::CreateDeferred(
                       OpName::kEmpty, DummyBuilder,
                       /*inputs=*/{}, OpParamCacheKeys::Empty(), {shape}));
  DeviceBufferRef ref_a = refs_a[0];

  TT_ASSERT_OK_AND_ASSIGN(auto refs_b, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_a},
                                           OpParamCacheKeys::Empty(), {shape}));
  DeviceBufferRef ref_b = refs_b[0];

  TT_ASSERT_OK_AND_ASSIGN(auto refs_c, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_b},
                                           OpParamCacheKeys::Empty(), {shape}));
  DeviceBufferRef ref_c = refs_c[0];

  TT_ASSERT_OK_AND_ASSIGN(auto refs_d, DeviceBufferList::CreateDeferred(
                                           OpName::kAdd, DummyBuilder, {ref_b},
                                           OpParamCacheKeys::Empty(), {shape}));
  DeviceBufferRef ref_d = refs_d[0];

  TT_ASSERT_OK_AND_ASSIGN(
      auto traversal,
      Traversal::CreateFromExecutionOrder(
          /*execution_order=*/{ref_a.device_buffer_list(),
                               ref_b.device_buffer_list(),
                               ref_c.device_buffer_list(),
                               ref_d.device_buffer_list()},
          /*nodes_to_materialize=*/{ref_c.device_buffer_list(),
                                    ref_d.device_buffer_list()}));

  absl::flat_hash_set<const DeviceBufferList*> scratch_set;

  auto traversals = SplitTraversal(std::move(traversal), scratch_set);

  // We should get three traversals.
  // c and d need to be split. Since b is common to both, it needs to also be
  // materialized.
  ASSERT_THAT(traversals, testing::SizeIs(3));

  // First traversal: executes a and b, materializes b.
  ASSERT_THAT(traversals[0]->execution_order(), testing::SizeIs(2));
  ASSERT_THAT(traversals[0]->outputs(), testing::SizeIs(1));
  EXPECT_EQ(traversals[0]->outputs()[0], ref_b);

  // Second traversal: executes c, materializes c.
  ASSERT_THAT(traversals[1]->arguments(), testing::SizeIs(1));
  EXPECT_EQ(traversals[1]->arguments()[0], ref_b);
  EXPECT_THAT(traversals[1]->execution_order(), testing::SizeIs(1));
  ASSERT_THAT(traversals[1]->outputs(), testing::SizeIs(1));
  EXPECT_EQ(traversals[1]->outputs()[0], ref_c);

  // Third traversal: executes d, materializes d.
  ASSERT_THAT(traversals[2]->arguments(), testing::SizeIs(1));
  EXPECT_EQ(traversals[2]->arguments()[0], ref_b);
  EXPECT_THAT(traversals[2]->execution_order(), testing::SizeIs(1));
  ASSERT_THAT(traversals[2]->outputs(), testing::SizeIs(1));
  EXPECT_EQ(traversals[2]->outputs()[0], ref_d);
}

}  // namespace
}  // namespace torch_tpu
