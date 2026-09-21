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

#include <cstdint>
#include <limits>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/op_correlation.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/materialize_common.h"
#include "csrc/eager/traversal.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

// Dummy builder for constructing DeferredOps and DeviceBufferLists without
// needing actual MLIR generation or TPU hardware.
absl::StatusOr<DynamicMlirOpResults> DummyBuilder(
    mlir::MlirBuilder& /*builder*/, absl::Span<mlir::MlirOp> /*inputs*/) {
  return DynamicMlirOpResults{};
}

class OpCorrelationEagerTest : public testing::Test {
 protected:
  void SetUp() override {
    ResetOpCorrelationIds();
    ResetOpCorrelationPushCount();
  }

  void TearDown() override {
    ResetOpCorrelationIds();
    ResetOpCorrelationPushCount();
  }

  ScopedPythonContextCapturer capturer_{OpName::kEmpty};
  Shape shape_{Dimensions{8}, mlir::ElementType::F32};
};

TEST_F(OpCorrelationEagerTest, DeferredOpSamplesNoCorrelationIdWhenStackEmpty) {
  DeferredOp op(OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(),
                {shape_});
  EXPECT_EQ(op.pt_correlation_id(), kNoCorrelationId);
}

TEST_F(OpCorrelationEagerTest, DeferredOpSamplesInnermostCorrelationId) {
  PushOpCorrelationId(101);
  DeferredOp op1(OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(),
                 {shape_});
  EXPECT_EQ(op1.pt_correlation_id(), 101);

  // Nested push: innermost must be sampled.
  PushOpCorrelationId(202);
  DeferredOp op2(OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(),
                 {shape_});
  EXPECT_EQ(op2.pt_correlation_id(), 202);

  // Pop innermost: outer op restored.
  PopOpCorrelationId();
  DeferredOp op3(OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(),
                 {shape_});
  EXPECT_EQ(op3.pt_correlation_id(), 101);
}

TEST_F(OpCorrelationEagerTest,
       DeferredOpSamplesNarrowedCorrelationIdAboveInt32Max) {
  const uint64_t large_id =
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 10ULL;
  PushOpCorrelationId(large_id);
  DeferredOp op(OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(),
                {shape_});
  EXPECT_EQ(op.pt_correlation_id(), NarrowCorrelationId(large_id));
}

TEST_F(OpCorrelationEagerTest, DeferredOpSamplingIsThreadLocal) {
  PushOpCorrelationId(777);

  std::thread worker([&] {
    // Worker thread starts with clean empty stack and its own python context.
    ScopedPythonContextCapturer thread_capturer(OpName::kEmpty);
    DeferredOp thread_op(OpName::kAdd, DummyBuilder, {},
                         OpParamCacheKeys::Empty(), {shape_});
    EXPECT_EQ(thread_op.pt_correlation_id(), kNoCorrelationId);
  });
  worker.join();

  // Main thread still has its pushed correlation ID.
  DeferredOp main_op(OpName::kAdd, DummyBuilder, {}, OpParamCacheKeys::Empty(),
                     {shape_});
  EXPECT_EQ(main_op.pt_correlation_id(), 777);
}

TEST_F(OpCorrelationEagerTest,
       FromExecutableExtractsEmptyWhenNoActiveCorrelation) {
  ExecutionTask task = ExecutionTask::FromExecutableForTesting("test_task");
  EXPECT_TRUE(task.pt_correlation_ids().empty());
}

TEST_F(OpCorrelationEagerTest, FromExecutableExtractsActiveCorrelationId) {
  PushOpCorrelationId(888);
  ExecutionTask task = ExecutionTask::FromExecutableForTesting("test_task");
  EXPECT_EQ(task.pt_correlation_ids(),
            absl::MakeConstSpan(std::vector<CorrelationId>{888}));
}

TEST_F(OpCorrelationEagerTest,
       ExtractCorrelationIdsFromTraversalWithDeferredOps) {
  PushOpCorrelationId(111);
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_a,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape_}));
  PopOpCorrelationId();

  PushOpCorrelationId(222);
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_b,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {refs_a[0]},
                                       OpParamCacheKeys::Empty(), {shape_}));
  PopOpCorrelationId();

  TT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Traversal> traversal,
                          Traversal::Create({refs_b[0]}));
  traversal->SortByCreationOrder();

  EXPECT_EQ(ExecutionTask::ExtractCorrelationIds(*traversal),
            (std::vector<CorrelationId>{111, 222}));
}

TEST_F(OpCorrelationEagerTest, ExtractCorrelationIdsSkipsUnstampedDeferredOps) {
  // First op is unstamped (no active correlation ID).
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_a,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape_}));

  // Second op is stamped.
  PushOpCorrelationId(333);
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_b,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {refs_a[0]},
                                       OpParamCacheKeys::Empty(), {shape_}));
  PopOpCorrelationId();

  TT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Traversal> traversal,
                          Traversal::Create({refs_b[0]}));
  traversal->SortByCreationOrder();

  EXPECT_EQ(ExecutionTask::ExtractCorrelationIds(*traversal),
            std::vector<CorrelationId>{333});
}

TEST_F(OpCorrelationEagerTest,
       ExtractCorrelationIdsDeduplicatesAndPreservesOrder) {
  // Construct a sequence of deferred ops with duplicated correlation IDs:
  // op_a (101) -> op_b (202) -> op_c (101 duplicate) -> op_d (303) -> op_e (202
  // duplicate)
  PushOpCorrelationId(101);
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_a,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {},
                                       OpParamCacheKeys::Empty(), {shape_}));
  PopOpCorrelationId();

  PushOpCorrelationId(202);
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_b,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {refs_a[0]},
                                       OpParamCacheKeys::Empty(), {shape_}));
  PopOpCorrelationId();

  PushOpCorrelationId(101);
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_c,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {refs_b[0]},
                                       OpParamCacheKeys::Empty(), {shape_}));
  PopOpCorrelationId();

  PushOpCorrelationId(303);
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_d,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {refs_c[0]},
                                       OpParamCacheKeys::Empty(), {shape_}));
  PopOpCorrelationId();

  PushOpCorrelationId(202);
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_e,
      DeviceBufferList::CreateDeferred(OpName::kAdd, DummyBuilder, {refs_d[0]},
                                       OpParamCacheKeys::Empty(), {shape_}));
  PopOpCorrelationId();

  TT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Traversal> traversal,
                          Traversal::Create({refs_e[0]}));
  traversal->SortByCreationOrder();

  // Must contain only {101, 202, 303} in first-seen order without duplicates.
  EXPECT_EQ(ExecutionTask::ExtractCorrelationIds(*traversal),
            (std::vector<CorrelationId>{101, 202, 303}));
}

TEST_F(OpCorrelationEagerTest, EmptyTraversalReturnsEmpty) {
  TT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Traversal> traversal,
                          Traversal::Create(std::vector<DeviceBufferRef>{}));
  EXPECT_TRUE(ExecutionTask::ExtractCorrelationIds(*traversal).empty());
}

TEST_F(OpCorrelationEagerTest,
       ExecutionTaskPropagatesCorrelationIdsToLaunchScope) {
  PushOpCorrelationId(999);
  ExecutionTask task = ExecutionTask::FromExecutableForTesting("test_task");
  EXPECT_EQ(task.pt_correlation_ids(),
            absl::MakeConstSpan(std::vector<CorrelationId>{999}));

  // Before launch scope, launch correlation is empty.
  EXPECT_TRUE(GetCurrentLaunchCorrelationIds().empty());

  {
    // Simulates the launch correlation scope installed inside
    // ExecutionTask::RunInternal().
    ScopedLaunchCorrelation launch(std::vector<CorrelationId>(
        task.pt_correlation_ids().begin(), task.pt_correlation_ids().end()));

    EXPECT_EQ(GetCurrentLaunchCorrelationIds(),
              absl::MakeConstSpan(std::vector<CorrelationId>{999}));
  }

  // After launch scope exits, restored to empty.
  EXPECT_TRUE(GetCurrentLaunchCorrelationIds().empty());
}

TEST_F(OpCorrelationEagerTest, PropagatedLaunchScopeIsThreadLocal) {
  PushOpCorrelationId(555);
  ExecutionTask task = ExecutionTask::FromExecutableForTesting("test_task");

  ScopedLaunchCorrelation launch(std::vector<CorrelationId>(
      task.pt_correlation_ids().begin(), task.pt_correlation_ids().end()));

  std::thread worker([] {
    // Other thread must see an empty launch correlation span.
    EXPECT_TRUE(GetCurrentLaunchCorrelationIds().empty());
  });
  worker.join();

  EXPECT_EQ(GetCurrentLaunchCorrelationIds(),
            absl::MakeConstSpan(std::vector<CorrelationId>{555}));
}

}  // namespace
}  // namespace torch_tpu
