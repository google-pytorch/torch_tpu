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

#include "csrc/eager/materialize.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_cache.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/context_states.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/execution_task.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/tpu_hooks.h"
#include "csrc/eager/traversal.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gtest/gtest.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape_util.h"

namespace torch_tpu {
namespace {

class MaterializeTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    const std::string device_type = "tpu";
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = device_type});
    ASSERT_EQ(AddTpuHooks(), absl::OkStatus());
    RegisterTpuAllocator();
    CompilationCache::GetInstance().SetOptions({});
  }

  static void TearDownTestSuite() {
    // Shuts down the runtime in the correct order to mimic the python bindings
    // runtime shutdown sequence (_shutdown_runtime):
    // 1. Terminate materialization worker threads so no new compilation or
    //    execution tasks are processed.
    // 2. Clear the compilation cache.
    // 3. Shut down the PjRtBackend (destroying client & devices).
    ShutDownMaterializationState();
    CompilationCache::ShutDown();
    PjrtBackend::GetInstance().Shutdown();
  }

  void SetUp() override { ClearAllStreams(); }
  void TearDown() override { ClearAllStreams(); }
};

TEST_F(MaterializeTest, EmptyListNoOpSuccess) {
  EXPECT_EQ(Materialize(absl::Span<const SharedDeviceBufferList>(),
                        MaterializationReason::kExplicitSync),
            absl::OkStatus());
  EXPECT_EQ(Materialize(absl::Span<const DeviceBufferRef>(),
                        MaterializationReason::kExplicitSync),
            absl::OkStatus());
}

TEST_F(MaterializeTest, MaterializedZeroSizeBufferSuccess) {
  const mlir::ElementType dtype = mlir::ElementType::F32;
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef ref,
                          CreateZeroSizeDeviceBufferRef({0}, dtype));
  EXPECT_TRUE(ref.is_deferred());
  EXPECT_EQ(Materialize(ref, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  EXPECT_TRUE(ref.AwaitBuffer().ok());
}

TEST_F(MaterializeTest, Create3DArgumentRelayoutBuffer) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);

  // 1. Placeholder buffer has no physical buffer, so it returns nullopt without
  // error.
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef placeholder,
                          DeviceBufferList::CreatePlaceholder(
                              Dimensions{2, 4, 8}, mlir::ElementType::F32));
  TT_ASSERT_OK_AND_ASSIGN(
      std::optional<RelayoutBufferResult> placeholder_relayout,
      MaybeCreate3DArgumentRelayoutBuffer(placeholder, {2, 1, 0}));
  EXPECT_FALSE(placeholder_relayout.has_value());

  // 2. Materialize a 3D buffer on device.
  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef buf,
      CreateEmptyDeviceBufferRef({2, 4, 8}, mlir::ElementType::F32));
  ASSERT_EQ(Materialize({buf}, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const pjrt_buf, buf.AwaitBuffer());
  ASSERT_NE(pjrt_buf, nullptr);

  // 3. Request a relayout to a layout that differs from current layout.
  const xla::Layout& current_layout = pjrt_buf->on_device_shape().layout();
  const Indices kTargetMtm =
      (current_layout.minor_to_major() == absl::Span<const int64_t>({2, 1, 0}))
          ? Indices{0, 1, 2}
          : Indices{2, 1, 0};

  TT_ASSERT_OK_AND_ASSIGN(std::optional<RelayoutBufferResult> relayout_ref,
                          MaybeCreate3DArgumentRelayoutBuffer(buf, kTargetMtm));
  ASSERT_TRUE(relayout_ref.has_value());
  RecordDeferredOpCreated(relayout_ref->buffer.device_buffer_list());
  ASSERT_EQ(Materialize({relayout_ref->buffer},
                        MaterializationReason::kCompileModeExecution,
                        GraphSplitMode::kSplitGraph),
            absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const relayout_pjrt_buf,
                          relayout_ref->buffer.AwaitBuffer());
  ASSERT_NE(relayout_pjrt_buf, nullptr);
  EXPECT_EQ(relayout_pjrt_buf->on_device_shape().layout().minor_to_major(),
            absl::Span<const int64_t>(kTargetMtm));

  // The original buffer remains immutable and untouched.
  EXPECT_EQ(pjrt_buf->on_device_shape().layout().minor_to_major(),
            current_layout.minor_to_major());

  // 4. Once layout matches on the relayout buffer, calling with the same target
  // layout returns nullopt.
  TT_ASSERT_OK_AND_ASSIGN(
      std::optional<RelayoutBufferResult> second_relayout,
      MaybeCreate3DArgumentRelayoutBuffer(relayout_ref->buffer, kTargetMtm));
  EXPECT_FALSE(second_relayout.has_value());
}

TEST_F(MaterializeTest, RelayoutDoesNotDataRaceWithConcurrentReaders) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);

  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef buf,
      CreateEmptyDeviceBufferRef({2, 4, 8}, mlir::ElementType::F32));
  ASSERT_EQ(Materialize({buf}, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const pjrt_buf, buf.AwaitBuffer());
  ASSERT_NE(pjrt_buf, nullptr);

  const xla::Layout& current_layout = pjrt_buf->on_device_shape().layout();
  const Indices kTargetMtm =
      (current_layout.minor_to_major() == absl::Span<const int64_t>({2, 1, 0}))
          ? Indices{0, 1, 2}
          : Indices{2, 1, 0};

  std::atomic<bool> done{false};
  std::vector<std::thread> readers;
  readers.reserve(4);
  for (int i = 0; i < 4; ++i) {
    readers.emplace_back([&buf, &done]() {
      while (!done.load(std::memory_order_relaxed)) {
        auto await_or = buf.AwaitBuffer();
        ASSERT_TRUE(await_or.ok());
        ASSERT_NE(*await_or, nullptr);
        EXPECT_EQ(buf.dimensions().size(), 3);
        auto shared_or = buf.GetSharedBuffer();
        ASSERT_TRUE(shared_or.ok());
        ASSERT_NE(*shared_or, nullptr);
      }
    });
  }

  // Concurrently perform relayout creation and materialization.
  for (int iter = 0; iter < 10; ++iter) {
    auto relayout_ref_or = MaybeCreate3DArgumentRelayoutBuffer(buf, kTargetMtm);
    ASSERT_TRUE(relayout_ref_or.ok());
    if (relayout_ref_or->has_value()) {
      RecordDeferredOpCreated((*relayout_ref_or)->buffer.device_buffer_list());
      ASSERT_EQ(Materialize({(*relayout_ref_or)->buffer},
                            MaterializationReason::kCompileModeExecution,
                            GraphSplitMode::kSplitGraph),
                absl::OkStatus());
    }
  }

  done.store(true, std::memory_order_relaxed);
  for (auto& reader : readers) {
    reader.join();
  }
}

TEST_F(MaterializeTest, RelayoutBufferIsAdoptedAndOldBufferDropped) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);

  // 1. Materialize a 3D weight buffer on device.
  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef weight_buf,
      CreateEmptyDeviceBufferRef({2, 4, 8}, mlir::ElementType::F32));
  ASSERT_EQ(Materialize({weight_buf}, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const pjrt_buf,
                          weight_buf.AwaitBuffer());
  ASSERT_NE(pjrt_buf, nullptr);

  // Capture shared_ptr to the old buffer.
  TT_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::PjRtBuffer> old_pjrt_buf,
                          weight_buf.GetSharedBuffer());
  ASSERT_NE(old_pjrt_buf, nullptr);
  // Refcount is 2: 1 in weight_buf's DeviceBufferList, 1 in old_pjrt_buf.
  EXPECT_EQ(old_pjrt_buf.use_count(), 2);

  const xla::Layout& current_layout = pjrt_buf->on_device_shape().layout();
  const Indices kTargetMtm =
      (current_layout.minor_to_major() == absl::Span<const int64_t>({2, 1, 0}))
          ? Indices{0, 1, 2}
          : Indices{2, 1, 0};

  // Step 1: Request relayout buffer.
  TT_ASSERT_OK_AND_ASSIGN(
      std::optional<RelayoutBufferResult> relayout1,
      MaybeCreate3DArgumentRelayoutBuffer(weight_buf, kTargetMtm));
  ASSERT_TRUE(relayout1.has_value());
  EXPECT_TRUE(relayout1->buffer.is_deferred());

  // Materialize relayout buffer.
  RecordDeferredOpCreated(relayout1->buffer.device_buffer_list());
  ASSERT_EQ(Materialize({relayout1->buffer},
                        MaterializationReason::kCompileModeExecution,
                        GraphSplitMode::kSplitGraph),
            absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const pjrt_relayout1,
                          relayout1->buffer.AwaitBuffer());
  ASSERT_NE(pjrt_relayout1, nullptr);

  // Synchronize both buffers before adoption to ensure no executions in flight.
  ASSERT_EQ(relayout1->buffer.Synchronize(), absl::OkStatus());
  ASSERT_EQ(weight_buf.Synchronize(), absl::OkStatus());

  // Adopt the relayout buffer into weight_buf.
  CustomLayout target_layout{.minor_to_major = kTargetMtm};
  ASSERT_EQ(
      weight_buf.AdoptMaterializedBufferFrom(relayout1->buffer, target_layout),
      absl::OkStatus());

  // The old buffer's refcount in DeviceBufferList drops!
  // Only our local old_pjrt_buf shared pointer remains.
  EXPECT_EQ(old_pjrt_buf.use_count(), 1);

  // The adopted buffer now backs weight_buf with the new layout.
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const new_pjrt_buf,
                          weight_buf.AwaitBuffer());
  EXPECT_EQ(new_pjrt_buf, pjrt_relayout1);
  ASSERT_TRUE(weight_buf.shape().layout().has_value());
  EXPECT_EQ(weight_buf.shape().layout()->minor_to_major, kTargetMtm);

  // Step 2: Second call to MaybeCreate3DArgumentRelayoutBuffer sees the adopted
  // layout matches target, returning nullopt (0 relayouts, no duplicates).
  TT_ASSERT_OK_AND_ASSIGN(
      std::optional<RelayoutBufferResult> relayout2,
      MaybeCreate3DArgumentRelayoutBuffer(weight_buf, kTargetMtm));
  EXPECT_FALSE(relayout2.has_value());
}

TEST_F(MaterializeTest, RelayoutBufferIsCachedAcrossExecutions) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);

  // 1. Materialize a 3D weight buffer on device.
  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef weight_buf,
      CreateEmptyDeviceBufferRef({2, 4, 8}, mlir::ElementType::F32));
  ASSERT_EQ(Materialize({weight_buf}, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const pjrt_buf,
                          weight_buf.AwaitBuffer());
  ASSERT_NE(pjrt_buf, nullptr);

  const xla::Layout& current_layout = pjrt_buf->on_device_shape().layout();
  const Indices kTargetMtm =
      (current_layout.minor_to_major() == absl::Span<const int64_t>({2, 1, 0}))
          ? Indices{0, 1, 2}
          : Indices{2, 1, 0};

  // 2. Step 1: First request creates a new deferred relayout buffer.
  TT_ASSERT_OK_AND_ASSIGN(
      std::optional<RelayoutBufferResult> relayout1,
      MaybeCreate3DArgumentRelayoutBuffer(weight_buf, kTargetMtm));
  ASSERT_TRUE(relayout1.has_value());
  EXPECT_FALSE(relayout1->cache_hit);
  EXPECT_TRUE(relayout1->buffer.is_deferred());

  // Materialize the relayout buffer (simulating first step execution).
  RecordDeferredOpCreated(relayout1->buffer.device_buffer_list());
  ASSERT_EQ(Materialize({relayout1->buffer},
                        MaterializationReason::kCompileModeExecution,
                        GraphSplitMode::kSplitGraph),
            absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const pjrt_relayout1,
                          relayout1->buffer.AwaitBuffer());
  ASSERT_NE(pjrt_relayout1, nullptr);
  EXPECT_TRUE(relayout1->buffer.is_materialized());
  weight_buf.SetCachedRelayoutBuffer(kTargetMtm, relayout1->buffer);

  // 3. Step 2: Second request with the same weight returns the cached relayout
  // buffer without creating a duplicate deferred op.
  TT_ASSERT_OK_AND_ASSIGN(
      std::optional<RelayoutBufferResult> relayout2,
      MaybeCreate3DArgumentRelayoutBuffer(weight_buf, kTargetMtm));
  ASSERT_TRUE(relayout2.has_value());
  EXPECT_TRUE(relayout2->cache_hit);
  EXPECT_EQ(relayout1->buffer.device_buffer_list(),
            relayout2->buffer.device_buffer_list());
  EXPECT_EQ(relayout1->buffer.index(), relayout2->buffer.index());
  EXPECT_FALSE(relayout2->buffer.is_deferred());
  EXPECT_TRUE(relayout2->buffer.is_materialized());

  // 4. Invalidation: Modifying or invalidating the underlying buffer clears
  // the cache, so subsequent calls produce a new relayout buffer.
  weight_buf.device_buffer_list()->InvalidateRelayoutCache();
  TT_ASSERT_OK_AND_ASSIGN(
      std::optional<RelayoutBufferResult> relayout3,
      MaybeCreate3DArgumentRelayoutBuffer(weight_buf, kTargetMtm));
  ASSERT_TRUE(relayout3.has_value());
  EXPECT_FALSE(relayout3->cache_hit);
  EXPECT_NE(relayout1->buffer.device_buffer_list(),
            relayout3->buffer.device_buffer_list());
  EXPECT_TRUE(relayout3->buffer.is_deferred());

  RecordDeferredOpCreated(relayout3->buffer.device_buffer_list());
  ASSERT_EQ(Materialize({relayout3->buffer},
                        MaterializationReason::kCompileModeExecution,
                        GraphSplitMode::kSplitGraph),
            absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const pjrt_relayout3,
                          relayout3->buffer.AwaitBuffer());
  ASSERT_NE(pjrt_relayout3, nullptr);
  EXPECT_TRUE(relayout3->buffer.is_materialized());
}

TEST_F(MaterializeTest, RelayoutCacheInvalidatedOnSetAsError) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);

  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef weight_buf,
      CreateEmptyDeviceBufferRef({2, 4, 8}, mlir::ElementType::F32));
  ASSERT_EQ(Materialize({weight_buf}, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  const Indices kTargetMtm = {0, 1, 2};
  TT_ASSERT_OK_AND_ASSIGN(
      std::optional<RelayoutBufferResult> relayout,
      MaybeCreate3DArgumentRelayoutBuffer(weight_buf, kTargetMtm));
  ASSERT_TRUE(relayout.has_value());
  EXPECT_FALSE(relayout->cache_hit);
  RecordDeferredOpCreated(relayout->buffer.device_buffer_list());
  ASSERT_EQ(Materialize({relayout->buffer},
                        MaterializationReason::kCompileModeExecution,
                        GraphSplitMode::kSplitGraph),
            absl::OkStatus());
  ASSERT_EQ(relayout->buffer.Synchronize(), absl::OkStatus());
  weight_buf.SetCachedRelayoutBuffer(kTargetMtm, relayout->buffer);
  EXPECT_TRUE(weight_buf.GetCachedRelayoutBuffer(kTargetMtm).has_value());

  // SetAsError clears the relayout cache.
  weight_buf.device_buffer_list()->SetAsError(
      TT_ERROR(error::kInternal) << "simulated materialization failure");
  EXPECT_FALSE(weight_buf.GetCachedRelayoutBuffer(kTargetMtm).has_value());
}

TEST_F(MaterializeTest, RelayoutCacheInvalidatedOnAdoptMaterializedBuffer) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);

  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef weight_buf,
      CreateEmptyDeviceBufferRef({2, 4, 8}, mlir::ElementType::F32));
  ASSERT_EQ(Materialize({weight_buf}, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  const Indices kTargetMtm = {0, 1, 2};
  TT_ASSERT_OK_AND_ASSIGN(
      std::optional<RelayoutBufferResult> relayout1,
      MaybeCreate3DArgumentRelayoutBuffer(weight_buf, kTargetMtm));
  ASSERT_TRUE(relayout1.has_value());
  EXPECT_FALSE(relayout1->cache_hit);
  RecordDeferredOpCreated(relayout1->buffer.device_buffer_list());
  ASSERT_EQ(Materialize({relayout1->buffer},
                        MaterializationReason::kCompileModeExecution,
                        GraphSplitMode::kSplitGraph),
            absl::OkStatus());
  ASSERT_EQ(relayout1->buffer.Synchronize(), absl::OkStatus());
  ASSERT_EQ(weight_buf.Synchronize(), absl::OkStatus());

  weight_buf.SetCachedRelayoutBuffer(kTargetMtm, relayout1->buffer);
  EXPECT_TRUE(weight_buf.GetCachedRelayoutBuffer(kTargetMtm).has_value());

  CustomLayout target_layout{.minor_to_major = kTargetMtm};
  ASSERT_EQ(
      weight_buf.AdoptMaterializedBufferFrom(relayout1->buffer, target_layout),
      absl::OkStatus());
  // Adopting clears the relayout cache.
  EXPECT_FALSE(weight_buf.GetCachedRelayoutBuffer(kTargetMtm).has_value());
}

TEST_F(MaterializeTest, FailedRelayoutBufferEvictedFromCache) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);

  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef weight_buf,
      CreateEmptyDeviceBufferRef({2, 4, 8}, mlir::ElementType::F32));
  ASSERT_EQ(Materialize({weight_buf}, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  const Indices kTargetMtm = {0, 1, 2};
  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef failed_relayout,
      CreateEmptyDeviceBufferRef({2, 4, 8}, mlir::ElementType::F32));
  failed_relayout.device_buffer_list()->SetAsError(
      TT_ERROR(error::kInternal) << "simulated relayout failure");

  weight_buf.SetCachedRelayoutBuffer(kTargetMtm, failed_relayout);
  // GetCachedRelayoutBuffer inspects the failed buffer and evicts it.
  EXPECT_FALSE(weight_buf.GetCachedRelayoutBuffer(kTargetMtm).has_value());
}

TEST_F(MaterializeTest, DeletedCachedRelayoutBufferEvictedFromCache) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);

  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef weight_buf,
      CreateEmptyDeviceBufferRef({2, 4, 8}, mlir::ElementType::F32));
  ASSERT_EQ(Materialize({weight_buf}, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  const Indices kTargetMtm = {0, 1, 2};
  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef relayout_buf,
      CreateEmptyDeviceBufferRef({2, 4, 8}, mlir::ElementType::F32));
  ASSERT_EQ(Materialize({relayout_buf}, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  ASSERT_EQ(relayout_buf.Synchronize(), absl::OkStatus());

  weight_buf.SetCachedRelayoutBuffer(kTargetMtm, relayout_buf);
  EXPECT_TRUE(weight_buf.GetCachedRelayoutBuffer(kTargetMtm).has_value());

  TT_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::PjRtBuffer> pjrt_buf,
                          relayout_buf.GetSharedBuffer());
  pjrt_buf->Delete();

  // GetCachedRelayoutBuffer detects deleted buffer, evicts, and returns
  // nullopt.
  EXPECT_FALSE(weight_buf.GetCachedRelayoutBuffer(kTargetMtm).has_value());
}

TEST_F(MaterializeTest, AlternatingTwoTargetLayoutsPreservesBothInCache) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);

  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef weight_buf,
      CreateEmptyDeviceBufferRef({2, 4, 8}, mlir::ElementType::F32));
  ASSERT_EQ(Materialize({weight_buf}, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  const Indices kTargetMtmA = {0, 1, 2};
  const Indices kTargetMtmB = {1, 0, 2};

  TT_ASSERT_OK_AND_ASSIGN(
      std::optional<RelayoutBufferResult> relayout_a,
      MaybeCreate3DArgumentRelayoutBuffer(weight_buf, kTargetMtmA));
  ASSERT_TRUE(relayout_a.has_value());
  EXPECT_FALSE(relayout_a->cache_hit);
  RecordDeferredOpCreated(relayout_a->buffer.device_buffer_list());
  ASSERT_EQ(Materialize({relayout_a->buffer},
                        MaterializationReason::kCompileModeExecution,
                        GraphSplitMode::kSplitGraph),
            absl::OkStatus());
  ASSERT_EQ(relayout_a->buffer.Synchronize(), absl::OkStatus());
  weight_buf.SetCachedRelayoutBuffer(kTargetMtmA, relayout_a->buffer);

  TT_ASSERT_OK_AND_ASSIGN(
      std::optional<RelayoutBufferResult> relayout_b,
      MaybeCreate3DArgumentRelayoutBuffer(weight_buf, kTargetMtmB));
  ASSERT_TRUE(relayout_b.has_value());
  EXPECT_FALSE(relayout_b->cache_hit);
  RecordDeferredOpCreated(relayout_b->buffer.device_buffer_list());
  ASSERT_EQ(Materialize({relayout_b->buffer},
                        MaterializationReason::kCompileModeExecution,
                        GraphSplitMode::kSplitGraph),
            absl::OkStatus());
  ASSERT_EQ(relayout_b->buffer.Synchronize(), absl::OkStatus());
  weight_buf.SetCachedRelayoutBuffer(kTargetMtmB, relayout_b->buffer);

  // Both layouts remain cached without clearing each other.
  TT_ASSERT_OK_AND_ASSIGN(
      std::optional<RelayoutBufferResult> cached_a,
      MaybeCreate3DArgumentRelayoutBuffer(weight_buf, kTargetMtmA));
  ASSERT_TRUE(cached_a.has_value());
  EXPECT_TRUE(cached_a->cache_hit);

  TT_ASSERT_OK_AND_ASSIGN(
      std::optional<RelayoutBufferResult> cached_b,
      MaybeCreate3DArgumentRelayoutBuffer(weight_buf, kTargetMtmB));
  ASSERT_TRUE(cached_b.has_value());
  EXPECT_TRUE(cached_b->cache_hit);
}

TEST_F(MaterializeTest, Donated3DParamInEnqueueExecutableSkipsAdoption) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);

  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef param_buf,
      CreateEmptyDeviceBufferRef({2, 4, 8}, mlir::ElementType::F32));
  ASSERT_EQ(Materialize({param_buf}, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  const Shape shape_3d(Dimensions{2, 4, 8}, mlir::ElementType::F32);
  const auto identity_builder = [](mlir::MlirBuilder&,
                                   absl::Span<mlir::MlirOp> inputs) {
    return DynamicMlirOpResults{inputs[0]};
  };

  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef placeholder,
                          DeviceBufferList::CreatePlaceholder(
                              shape_3d.dimensions(), shape_3d.dtype()));
  TT_ASSERT_OK_AND_ASSIGN(const std::vector<DeviceBufferRef> refs,
                          DeviceBufferList::CreateDeferred(
                              OpName::kToCopy, identity_builder, {placeholder},
                              OpParamCacheKeys::Empty(), {shape_3d}));
  TT_ASSERT_OK_AND_ASSIGN(auto traversal, Traversal::Create(refs));

  xla::Shape param_shape_custom =
      xla::ShapeUtil::MakeShape(xla::F32, {2, 4, 8});
  *param_shape_custom.mutable_layout() = xla::LayoutUtil::MakeLayout({0, 1, 2});
  CompilationSpec spec = GetCompilationSpec(CompilationMode::kFastCompile);
  spec.xla_compile_options->argument_layouts = {param_shape_custom};

  TT_ASSERT_OK_AND_ASSIGN(
      CompiledKernel kernel,
      traversal->Compile(spec, UnboundedKernelPolicy::kDisallow));
  TT_ASSERT_OK_AND_ASSIGN(SharedLoadedExecutableWithMetadata exec,
                          kernel.fixed_shape_kernel.get());

  // Mark parameter 0 as donated in LoadedExecutableWithMetadata.
  SharedLoadedExecutableWithMetadata exec_donated =
      LoadedExecutableWithMetadata::With3DParamLayouts(
          exec,
          {{.param_idx = 0, .minor_to_major = {0, 1, 2}, .is_donated = true}},
          exec->parameter_device_shapes(), /*donated_3d_param_indices=*/{0});

  EXPECT_TRUE(exec_donated->IsDonated3DParam(0));
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> outputs,
      EnqueueExecutable(exec_donated, {param_buf}, {shape_3d}, "donated_task"));

  // param_buf must NOT be adopted in place (layout stays default / unset).
  EXPECT_FALSE(param_buf.shape().layout().has_value());
  // Relayout cache must NOT retain a donated/deleted buffer.
  EXPECT_FALSE(param_buf.GetCachedRelayoutBuffer({0, 1, 2}).has_value());
}

TEST_F(MaterializeTest, EnqueueExecutableNonDefaultThenDefaultLayout) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);

  // 1. Create a 3D argument buffer on device (initially in default layout).
  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef param_buf,
      CreateEmptyDeviceBufferRef({2, 4, 8}, mlir::ElementType::F32));
  ASSERT_EQ(Materialize({param_buf}, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  const Shape shape_3d(Dimensions{2, 4, 8}, mlir::ElementType::F32);
  const auto identity_builder = [](mlir::MlirBuilder&,
                                   absl::Span<mlir::MlirOp> inputs) {
    return DynamicMlirOpResults{inputs[0]};
  };

  // 2. Compile executable 1 with argument_layouts = {param_shape_custom}.
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef placeholder1,
                          DeviceBufferList::CreatePlaceholder(
                              shape_3d.dimensions(), shape_3d.dtype()));
  TT_ASSERT_OK_AND_ASSIGN(const std::vector<DeviceBufferRef> refs1,
                          DeviceBufferList::CreateDeferred(
                              OpName::kToCopy, identity_builder, {placeholder1},
                              OpParamCacheKeys::Empty(), {shape_3d}));
  TT_ASSERT_OK_AND_ASSIGN(auto traversal1, Traversal::Create(refs1));

  xla::Shape param_shape_custom =
      xla::ShapeUtil::MakeShape(xla::F32, {2, 4, 8});
  *param_shape_custom.mutable_layout() = xla::LayoutUtil::MakeLayout({0, 1, 2});
  CompilationSpec spec1 = GetCompilationSpec(CompilationMode::kFastCompile);
  spec1.xla_compile_options->argument_layouts = {param_shape_custom};

  TT_ASSERT_OK_AND_ASSIGN(
      CompiledKernel kernel1,
      traversal1->Compile(spec1, UnboundedKernelPolicy::kDisallow));
  TT_ASSERT_OK_AND_ASSIGN(SharedLoadedExecutableWithMetadata exec1,
                          kernel1.fixed_shape_kernel.get());
  ASSERT_EQ(exec1->non_default_3d_param_layouts().size(), 1);
  EXPECT_EQ(exec1->non_default_3d_param_layouts()[0].minor_to_major,
            Indices({0, 1, 2}));

  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> outputs1,
      EnqueueExecutable(exec1, {param_buf}, {shape_3d}, "exec1_non_default"));
  ASSERT_EQ(outputs1[0].Synchronize(), absl::OkStatus());
  ASSERT_EQ(param_buf.Synchronize(), absl::OkStatus());

  // param_buf now has adopted non-default layout {0, 1, 2}.
  ASSERT_TRUE(param_buf.shape().layout().has_value());
  EXPECT_EQ(param_buf.shape().layout()->minor_to_major, Indices({0, 1, 2}));

  // 3. Compile executable 2 with default layout.
  CompilationCache::GetInstance().EvictAll();
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef placeholder2,
                          DeviceBufferList::CreatePlaceholder(
                              shape_3d.dimensions(), shape_3d.dtype()));
  TT_ASSERT_OK_AND_ASSIGN(const std::vector<DeviceBufferRef> refs2,
                          DeviceBufferList::CreateDeferred(
                              OpName::kToCopy, identity_builder, {placeholder2},
                              OpParamCacheKeys::Empty(), {shape_3d}));
  TT_ASSERT_OK_AND_ASSIGN(auto traversal2, Traversal::Create(refs2));
  CompilationSpec spec2 = GetCompilationSpec(CompilationMode::kFastCompile);
  TT_ASSERT_OK_AND_ASSIGN(
      CompiledKernel kernel2,
      traversal2->Compile(spec2, UnboundedKernelPolicy::kDisallow));
  TT_ASSERT_OK_AND_ASSIGN(SharedLoadedExecutableWithMetadata exec2,
                          kernel2.fixed_shape_kernel.get());
  EXPECT_TRUE(exec2->non_default_3d_param_layouts().empty());

  // EnqueueExecutable handles the layout discrepancy via relayout_cache_
  // without mutating param_buf or failing ValidateArgumentLayouts.
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> outputs2,
      EnqueueExecutable(exec2, {param_buf}, {shape_3d}, "exec2_default"));
  ASSERT_EQ(outputs2[0].Synchronize(), absl::OkStatus());
}

TEST_F(MaterializeTest, CompilerOptionsPropagateToMaterializeThread) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);

  const Shape shape(Dimensions{1}, mlir::ElementType::F32);
  const auto builder = [shape](mlir::MlirBuilder& builder,
                               absl::Span<mlir::MlirOp>) {
    return DynamicMlirOpResults{
        BuildFillUninitialized(builder, shape.dtype(), shape.dimensions())};
  };

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> refs,
      DeviceBufferList::CreateDeferred(OpName::kEmpty, builder,
                                       /*inputs=*/{}, OpParamCacheKeys::Empty(),
                                       {shape}));
  const DeviceBufferRef ref = refs[0];
  const at::Tensor t = MakeTensor(ref);

  // 1. Clear all cache entries to verify we trigger a fresh compilation.
  CompilationCache::GetInstance().EvictAll();

  // 2. Set compiler overrides on the main thread and calculate its expected
  // cache fingerprint.
  CompilerOptionOverrides overrides;
  overrides["xla_tpu_autofdo"] = "false";
  ASSERT_EQ(PushCompilerOptionOverrides(overrides), absl::OkStatus());
  const CompileOptionsKey expected_key =
      GetCompilationSpec(CompilationMode::kFastCompile).compile_options_key;

  absl::Cleanup cleanup = [] { PopCompilerOptionOverrides(); };

  // 3. Synchronously materialize/compile the buffer.
  ASSERT_EQ(Materialize(ref, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  EXPECT_TRUE(ref.AwaitBuffer().ok());

  // 4. Verify the compiled executable fingerprint in cache matches our
  // overridden key.
  const PerfStats stats = CompilationCache::GetInstance().GetCacheStats();
  ASSERT_EQ(stats.per_entry_stats.size(), 1);
  EXPECT_EQ(stats.per_entry_stats[0].key.compile_options_key(), expected_key);
}

TEST_F(MaterializeTest, OutOfOrderUnexecutedDeferred) {
  OpName op_name = OpName::kAdd;
  ScopedPythonContextCapturer capturer(op_name);

  // Create a deferred, non-constant buffer but do not register it with the
  // events queue.
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs,
      DeviceBufferList::CreateDeferred(
          op_name,
          [](mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs)
              -> absl::StatusOr<DynamicMlirOpResults> {
            return DynamicMlirOpResults{};
          },
          {}, OpParamCacheKeys::Empty(),
          {Shape{Dimensions{10}, mlir::ElementType::F32}}));
  auto never_executed_deferred = refs[0];

  // Create an op which depends on the deferred buffer and register it with the
  // events queue.
  TT_ASSERT_OK_AND_ASSIGN(
      refs, DeviceBufferList::CreateDeferred(
                op_name,
                [](mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs)
                    -> absl::StatusOr<DynamicMlirOpResults> {
                  auto ranked_tensor_type = mlir::makeTensorType(
                      builder.getContext(), {0}, mlir::ElementType::F32);
                  auto dense_elements_attr =
                      mlir::DenseElementsAttr::getFromRawBuffer(
                          ranked_tensor_type, {});
                  return DynamicMlirOpResults{
                      mlir::stablehlo::Constant(builder, dense_elements_attr)};
                },
                {never_executed_deferred}, OpParamCacheKeys::Empty(),
                {Shape{Dimensions{0}, mlir::ElementType::F32}}));
  auto depends_on_deferred = refs[0];
  RecordDeferredOpCreated(depends_on_deferred.device_buffer_list());

  // Try to materialize the dependent buffer in the background.
  // This will fail on synchronization when the execution thread detects that
  // the deferred buffer wasn't materialized first.
  Materialize(depends_on_deferred, MaterializationReason::kExplicitSync)
      .IgnoreError();
  auto finished_status = depends_on_deferred.Synchronize();
  EXPECT_EQ(finished_status.code(), error::kFailedPrecondition);
  EXPECT_NE(
      finished_status.message().find("detected an out of order execution"),
      std::string::npos);
}

TEST_F(MaterializeTest, OutOfOrderUnexecutedPending) {
  // Create a buffer in the pending state.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef pending_buffer,
                          DeviceBufferList::CreatePendingExecution(
                              Shape{Dimensions{10}, mlir::ElementType::F32},
                              device_index, stream_id));

  // Create an op which depends on the pending buffer and register it with the
  // events queue.
  OpName op_name = OpName::kAdd;
  ScopedPythonContextCapturer capturer(op_name);
  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<DeviceBufferRef> refs,
      DeviceBufferList::CreateDeferred(
          op_name,
          [](mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp> inputs)
              -> absl::StatusOr<DynamicMlirOpResults> {
            auto ranked_tensor_type = mlir::makeTensorType(
                builder.getContext(), {0}, mlir::ElementType::F32);
            auto dense_elements_attr =
                mlir::DenseElementsAttr::getFromRawBuffer(ranked_tensor_type,
                                                          {});
            return DynamicMlirOpResults{
                mlir::stablehlo::Constant(builder, dense_elements_attr)};
          },
          {pending_buffer}, OpParamCacheKeys::Empty(),
          {Shape{Dimensions{0}, mlir::ElementType::F32}}));
  auto depends_on_pending = refs[0];
  RecordDeferredOpCreated(depends_on_pending.device_buffer_list());

  // Try to materialize the dependent buffer. This will fail on synchronization
  // when the execution thread detects that the pending buffer wasn't
  // materialized first.
  Materialize(depends_on_pending, MaterializationReason::kExplicitSync)
      .IgnoreError();
  auto finished_status = depends_on_pending.Synchronize();
  EXPECT_EQ(finished_status.code(), error::kFailedPrecondition);
  EXPECT_NE(
      finished_status.message().find("detected an out of order execution"),
      std::string::npos);
}

// Creates a deferred op that fills an 8-element F32 tensor, handling its
// creation as dictated by the current eager mode.
absl::StatusOr<DeviceBufferRef> CreateFillOp(const bool allow_materialization) {
  internal::DeferredOpParams params{
      .op_name = OpName::kAdd,
      .op_builder = [](mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp>)
          -> absl::StatusOr<DynamicMlirOpResults> {
        return DynamicMlirOpResults{
            BuildFillUninitialized(builder, mlir::ElementType::F32, {8})};
      },
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
      .output_shapes = {Shape(Dimensions{8}, mlir::ElementType::F32)},
      .allow_materialization = allow_materialization,
  };
  TT_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferRef> refs,
      internal::CreateDeferredDeviceBufferList(std::move(params)));
  TT_RET_CHECK(refs.size() == 1, error::kInternal)
      << "expected 1 output, got " << refs.size();
  return std::move(refs[0]);
}

// Waits until `ref` is no longer deferred, i.e. until its materialization has
// started. Returns false if it is still deferred after `timeout`.
bool WaitUntilNotDeferred(const DeviceBufferRef& ref,
                          const absl::Duration timeout) {
  const absl::Time deadline = absl::Now() + timeout;
  while (ref.is_deferred()) {
    if (absl::Now() > deadline) {
      return false;
    }
    absl::SleepFor(absl::Milliseconds(1));
  }
  return true;
}

TEST_F(MaterializeTest, RunPostponedMaterializationExecutesOpsInDeferNever) {
  ClearAllStreams();
  const ScopedPythonContextCapturer capturer(OpName::kAdd);
  const EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kDeferNever);
  absl::Cleanup cleanup_mode = [prev_mode] { SetEagerMode(prev_mode); };

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef ref,
                          CreateFillOp(/*allow_materialization=*/false));
  EXPECT_TRUE(ref.is_deferred());

  // Running the postponed materialization executes the op.
  ASSERT_EQ(internal::RunPostponedMaterialization(), absl::OkStatus());
  ASSERT_TRUE(WaitUntilNotDeferred(ref, absl::Minutes(1)));
  EXPECT_EQ(ref.Synchronize(), absl::OkStatus());
}

TEST_F(MaterializeTest,
       RunPostponedMaterializationInsertsRepeatedOpsGraphBreakInDeferAndFuse) {
  ClearAllStreams();
  const ScopedPythonContextCapturer capturer(OpName::kAdd);
  const EagerMode prev_mode = GetEagerMode();
  SetEagerMode(EagerMode::kDeferAndFuse);
  absl::Cleanup cleanup_mode = [prev_mode] { SetEagerMode(prev_mode); };

  // Create four copies of an op with materialization disallowed: the repeated
  // ops heuristic requests a graph break on the fourth one, which must be
  // postponed.
  std::vector<DeviceBufferRef> refs;
  refs.reserve(4);
  for (int i = 0; i < 4; ++i) {
    TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef ref,
                            CreateFillOp(/*allow_materialization=*/false));
    refs.push_back(std::move(ref));
  }

  // Sleep for 100ms. This is enough that a background materialization should
  // have started if there was a materialization attempt.
  absl::SleepFor(absl::Milliseconds(100));
  for (const DeviceBufferRef& ref : refs) {
    EXPECT_TRUE(ref.is_deferred());
  }

  // Running the postponed materialization inserts the requested graph break,
  // which executes all the ops.
  ASSERT_EQ(internal::RunPostponedMaterialization(), absl::OkStatus());
  for (const DeviceBufferRef& ref : refs) {
    ASSERT_TRUE(WaitUntilNotDeferred(ref, absl::Minutes(1)));
    EXPECT_EQ(ref.Synchronize(), absl::OkStatus());
  }
}

absl::StatusOr<DeviceBufferRef> CreateIotaOp(const bool record_on_stream = true,
                                             const int64_t size = 8) {
  const ScopedPythonContextCapturer capturer(OpName::kAdd);
  TT_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferRef> refs,
      DeviceBufferList::CreateDeferred(
          OpName::kAdd,
          [size](mlir::MlirBuilder& builder, absl::Span<mlir::MlirOp>)
              -> absl::StatusOr<DynamicMlirOpResults> {
            const auto out_type = mlir::makeTensorType(
                builder.getContext(), {size}, mlir::ElementType::F32);
            return DynamicMlirOpResults{
                mlir::stablehlo::Iota(builder, out_type, /*iota_dimension=*/0)};
          },
          /*inputs=*/{}, OpParamCacheKeys::Empty(),
          {Shape(Dimensions{size}, mlir::ElementType::F32)}));
  if (record_on_stream) {
    RecordDeferredOpCreated(refs[0].device_buffer_list());
  }
  return std::move(refs[0]);
}

TEST_F(MaterializeTest, SnapshotAwaitsPriorTensorCoreOutputs) {
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  std::vector<DeviceBufferRef> refs;
  for (const int64_t size : {8, 64, 65536}) {
    TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef ref,
                            CreateIotaOp(/*record_on_stream=*/true, size));
    ASSERT_TRUE(InsertGraphBreak(device_index, stream_id,
                                 {ref.device_buffer_list()},
                                 MaterializationReason::kExplicitSync,
                                 GraphSplitMode::kSplitGraph)
                    .ok());
    refs.push_back(std::move(ref));
  }

  TT_ASSERT_OK_AND_ASSIGN(
      EventSnapshot snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph));
  ASSERT_EQ(snapshot.Synchronize(), absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(const bool completed, snapshot.Query());
  EXPECT_TRUE(completed);

  for (const DeviceBufferRef& ref : refs) {
    TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const pjrt_buf, ref.AwaitBuffer());
    xla::Future<void> ready = pjrt_buf->GetReadyFuture();
    EXPECT_TRUE(ready.IsKnownReady());
    EXPECT_EQ(ready.Await(), absl::OkStatus());
  }
}

TEST_F(MaterializeTest,
       SnapshotAwaitsLastDeviceOutputAcrossBatchesAndClearAllStreams) {
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef ref1,
                          CreateIotaOp(/*record_on_stream=*/true, 65536));
  ASSERT_EQ(Materialize(ref1, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  TT_ASSERT_OK_AND_ASSIGN(
      EventSnapshot snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph));
  ASSERT_EQ(snapshot.Synchronize(), absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const pjrt_buf1, ref1.AwaitBuffer());
  EXPECT_TRUE(pjrt_buf1->GetReadyFuture().IsKnownReady());
  EXPECT_EQ(pjrt_buf1->GetReadyFuture().Await(), absl::OkStatus());

  // Verify ClearAllStreams() drops StreamState's cached output ref.
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef tmp_ref, CreateIotaOp());
  std::optional<DeviceBufferRef> ref2 = std::move(tmp_ref);
  const DeviceBufferRef::Weak weak_ref2 = ref2->weak();
  ASSERT_EQ(Materialize(*ref2, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  ASSERT_EQ(ref2->Synchronize(), absl::OkStatus());
  ref2.reset();
  EXPECT_FALSE(weak_ref2.expired());

  ClearAllStreams();
  // `execute_thread_` may still briefly hold its ExecutionTask copy of `ref2`.
  const absl::Time deadline = absl::Now() + absl::Seconds(10);
  while (!weak_ref2.expired() && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  EXPECT_TRUE(weak_ref2.expired());
  TT_ASSERT_OK_AND_ASSIGN(
      EventSnapshot post_clear_snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph));
  ASSERT_EQ(post_clear_snapshot.Synchronize(), absl::OkStatus());
}

TEST_F(MaterializeTest, DeletedLastDeviceOutputResolvesSnapshotOk) {
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef ref, CreateIotaOp());
  ASSERT_EQ(Materialize(ref, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtBuffer* const pjrt_buf, ref.AwaitBuffer());
  ASSERT_EQ(pjrt_buf->GetReadyFuture().Await(), absl::OkStatus());
  pjrt_buf->Delete();
  ASSERT_FALSE(pjrt_buf->GetReadyFuture().Await().ok());

  TT_ASSERT_OK_AND_ASSIGN(
      EventSnapshot snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph));
  EXPECT_EQ(snapshot.Synchronize(), absl::OkStatus());
}

TEST_F(MaterializeTest, CompiledExecutionPreservesPriorLastDeviceOutput) {
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef tmp_tc, CreateIotaOp());
  std::optional<DeviceBufferRef> ref_tc = std::move(tmp_tc);
  const DeviceBufferRef::Weak weak_tc = ref_tc->weak();
  ASSERT_EQ(Materialize(*ref_tc, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  ASSERT_EQ(ref_tc->Synchronize(), absl::OkStatus());
  ref_tc.reset();
  ASSERT_FALSE(weak_tc.expired());

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef unrecorded_op,
                          CreateIotaOp(/*record_on_stream=*/false));
  TT_ASSERT_OK_AND_ASSIGN(
      const auto traversal,
      Traversal::Create(std::vector<DeviceBufferRef>{unrecorded_op}));
  TT_ASSERT_OK_AND_ASSIGN(
      CompiledKernel kernel,
      traversal->Compile(GetCompilationSpec(CompilationMode::kFastCompile),
                         UnboundedKernelPolicy::kDisallow));
  TT_ASSERT_OK_AND_ASSIGN(SharedLoadedExecutableWithMetadata executable,
                          kernel.fixed_shape_kernel.get());
  TT_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> compiled_outputs,
      EnqueueExecutable(std::move(executable), /*arguments=*/{},
                        {Shape(Dimensions{8}, mlir::ElementType::F32)},
                        "test_compiled"));
  ASSERT_EQ(compiled_outputs[0].Synchronize(), absl::OkStatus());
  EXPECT_FALSE(weak_tc.expired());

  TT_ASSERT_OK_AND_ASSIGN(
      EventSnapshot snapshot,
      MaterializeCurrentStream(MaterializationReason::kExplicitSync,
                               GraphSplitMode::kSplitGraph));
  ASSERT_EQ(snapshot.Synchronize(), absl::OkStatus());
  EXPECT_TRUE(weak_tc.expired());
}

TEST_F(MaterializeTest, ExecutionTaskPreservesTraversalCorePinningMode) {
  const auto mlir_context = MakeMlirContext();
  const CompilationSpec spec =
      GetCompilationSpec(CompilationMode::kFastCompile);

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef unpinned_op,
                          CreateIotaOp(/*record_on_stream=*/false));
  TT_ASSERT_OK_AND_ASSIGN(
      auto unpinned_traversal,
      Traversal::Create(std::vector<DeviceBufferRef>{unpinned_op}));
  TT_ASSERT_OK_AND_ASSIGN(
      ExecutionTask unpinned_task,
      ExecutionTask::FromTraversal(std::move(unpinned_traversal), *mlir_context,
                                   spec));
  EXPECT_EQ(unpinned_task.core_pinning_mode(), CorePinningMode::kUnpinned);
  ASSERT_EQ(unpinned_task.Run(), absl::OkStatus());
  ASSERT_EQ(unpinned_op.Synchronize(), absl::OkStatus());

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef pinned_op,
                          CreateIotaOp(/*record_on_stream=*/false));
  TT_ASSERT_OK_AND_ASSIGN(
      auto pinned_traversal,
      Traversal::Create(std::vector<DeviceBufferRef>{pinned_op}));
  pinned_traversal->SetCorePinningMode(CorePinningMode::kTensorCore);
  TT_ASSERT_OK_AND_ASSIGN(
      ExecutionTask pinned_task,
      ExecutionTask::FromTraversal(std::move(pinned_traversal), *mlir_context,
                                   spec));
  EXPECT_EQ(pinned_task.core_pinning_mode(), CorePinningMode::kTensorCore);
  ASSERT_EQ(pinned_task.Run(), absl::OkStatus());
  ASSERT_EQ(pinned_op.Synchronize(), absl::OkStatus());
}

TEST_F(MaterializeTest,
       ExecutionTaskFromTraversalStampsNonDefaultOutputLayoutOnCacheHit) {
  const ScopedPythonContextCapturer capturer(OpName::kEmpty);
  CompilationCache::GetInstance().SetAllowCacheMode(true);
  const auto mlir_context = MakeMlirContext();

  Shape shape_lhs(Dimensions{128, 128}, mlir::ElementType::BF16);
  Shape shape_rhs_3d(Dimensions{8, 128, 128}, mlir::ElementType::BF16);
  Shape shape_group_sizes(Dimensions{8}, mlir::ElementType::I32);
  Shape shape_bias_3d(Dimensions{8, 128, 128}, mlir::ElementType::BF16);
  Shape shape_dot_out(Dimensions{128, 128}, mlir::ElementType::BF16);
  Shape shape_out_3d(Dimensions{8, 128, 128}, mlir::ElementType::BF16);

  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef ref_lhs,
      CreateEmptyDeviceBufferRef(shape_lhs.dimensions(), shape_lhs.dtype()));
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef ref_rhs_3d,
                          CreateEmptyDeviceBufferRef(shape_rhs_3d.dimensions(),
                                                     shape_rhs_3d.dtype()));
  TT_ASSERT_OK_AND_ASSIGN(
      DeviceBufferRef ref_group_sizes,
      CreateEmptyDeviceBufferRef(shape_group_sizes.dimensions(),
                                 shape_group_sizes.dtype()));
  TT_ASSERT_OK_AND_ASSIGN(DeviceBufferRef ref_bias_3d,
                          CreateEmptyDeviceBufferRef(shape_bias_3d.dimensions(),
                                                     shape_bias_3d.dtype()));
  ASSERT_EQ(Materialize({ref_lhs, ref_rhs_3d, ref_group_sizes, ref_bias_3d},
                        MaterializationReason::kExplicitSync),
            absl::OkStatus());
  ASSERT_EQ(ref_lhs.Synchronize(), absl::OkStatus());
  ASSERT_EQ(ref_rhs_3d.Synchronize(), absl::OkStatus());
  ASSERT_EQ(ref_group_sizes.Synchronize(), absl::OkStatus());
  ASSERT_EQ(ref_bias_3d.Synchronize(), absl::OkStatus());

  auto make_eager_traversal =
      [&]() -> absl::StatusOr<std::unique_ptr<Traversal>> {
    MlirOpBuilder ragged_dot_builder = [](mlir::MlirBuilder& builder,
                                          absl::Span<mlir::MlirOp> inputs)
        -> absl::StatusOr<DynamicMlirOpResults> {
      mlir::MlirOp rhs_t = mlir::stablehlo::Transpose(inputs[1], {0, 2, 1});

      const auto dimension_numbers =
          mlir::chlo::RaggedDotDimensionNumbersAttr::get(
              &builder.getContext(), /*lhsBatchingDimensions=*/{},
              /*rhsBatchingDimensions=*/{},
              /*lhsContractingDimensions=*/{1},
              /*rhsContractingDimensions=*/{1},
              /*lhsRaggedDimensions=*/{0},
              /*rhsGroupDimensions=*/{0});
      auto precision_attr = mlir::chlo::PrecisionAttr::get(
          &builder.getContext(),
          mlir::chlo::Precision::DEFAULT);  // EXPLICIT_PRECISION_OK=test IR
                                            // construction
      auto precision_config = mlir::ArrayAttr::get(
          &builder.getContext(), {precision_attr, precision_attr});
      const mlir::Type out_type = mlir::makeTensorType(
          builder.getContext(), {128, 128}, mlir::ElementType::BF16);
      mlir::MlirOp ragged_dot =
          mlir::chlo::RaggedDot(out_type, inputs[0], rhs_t, inputs[2],
                                dimension_numbers, precision_config);

      mlir::MlirOp add_3d = mlir::stablehlo::Add(inputs[1], inputs[3]);

      return DynamicMlirOpResults{ragged_dot, add_3d};
    };

    TT_ASSIGN_OR_RETURN(
        std::vector<DeviceBufferRef> refs_out,
        DeviceBufferList::CreateDeferred(
            OpName::kRaggedDot, std::move(ragged_dot_builder),
            {ref_lhs, ref_rhs_3d, ref_group_sizes, ref_bias_3d},
            OpParamCacheKeys::Empty(), {shape_dot_out, shape_out_3d}));

    return Traversal::Create(refs_out);
  };

  const CompilationSpec spec =
      GetCompilationSpec(CompilationMode::kFastCompile);

  // 1. First execution task (cache miss - compiles and populates cache).
  TT_ASSERT_OK_AND_ASSIGN(auto tr1, make_eager_traversal());
  TT_ASSERT_OK_AND_ASSIGN(
      ExecutionTask task1,
      ExecutionTask::FromTraversal(std::move(tr1), *mlir_context, spec));
  ASSERT_EQ(task1.Run(), absl::OkStatus());

  // 2. Second execution task (cache hit - loads from cache).
  TT_ASSERT_OK_AND_ASSIGN(auto tr2, make_eager_traversal());
  std::vector<DeviceBufferRef> tr2_outputs(tr2->outputs().begin(),
                                           tr2->outputs().end());
  EXPECT_FALSE(tr2_outputs[0].shape().layout().has_value());
  EXPECT_FALSE(tr2_outputs[1].shape().layout().has_value());

  TT_ASSERT_OK_AND_ASSIGN(
      ExecutionTask task2,
      ExecutionTask::FromTraversal(std::move(tr2), *mlir_context, spec));

  // Output 0 has cleared default layout, leaving it unstamped.
  EXPECT_FALSE(tr2_outputs[0].shape().layout().has_value());
  // Output 1 has non-default {1, 2, 0}, stamped on eager compilation cache hit.
  ASSERT_TRUE(tr2_outputs[1].shape().layout().has_value());
  EXPECT_EQ(tr2_outputs[1].shape().layout()->minor_to_major,
            Indices({1, 2, 0}));
}

class PinnedHostPropagationTest : public testing::Test {
 protected:
  absl::StatusOr<DeviceBufferRef> CreateRoot() {
    return CreateOp(OpName::kEmpty);
  }

  absl::StatusOr<DeviceBufferRef> CreateView(const DeviceBufferRef& input) {
    return CreateOp(OpName::kView, {input});
  }

  absl::StatusOr<DeviceBufferRef> CreateComputeOp(
      std::vector<DeviceBufferRef> inputs) {
    return CreateOp(OpName::kAdd, std::move(inputs));
  }

 private:
  absl::StatusOr<DeviceBufferRef> CreateOp(
      OpName op_name, std::vector<DeviceBufferRef> inputs = {}) {
    ScopedPythonContextCapturer capturer(op_name);
    TT_ASSIGN_OR_RETURN(
        const std::vector<DeviceBufferRef> refs,
        DeviceBufferList::CreateDeferred(
            op_name,
            [](mlir::MlirBuilder&, absl::Span<mlir::MlirOp>) {
              return DynamicMlirOpResults{};
            },
            std::move(inputs), OpParamCacheKeys::Empty(), {shape_}));
    return refs[0];
  }

  const Shape shape_{Dimensions{10}, mlir::ElementType::F32};
};

// Connectivity graph:
//
//                 root (kEmpty)
//               /      \
//         (view)        (view)
//           /              \
//       view1              view2
//          |               |
//        (view)          (view)
//          |               |
//     view1_child     view2_child
TEST_F(PinnedHostPropagationTest, TreeOfViews) {
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef root, CreateRoot());
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef view1, CreateView(root));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef view1_child, CreateView(view1));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef view2, CreateView(root));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef view2_child, CreateView(view2));

  EXPECT_FALSE(root.is_pinned_host());
  EXPECT_FALSE(view1.is_pinned_host());
  EXPECT_FALSE(view1_child.is_pinned_host());
  EXPECT_FALSE(view2.is_pinned_host());
  EXPECT_FALSE(view2_child.is_pinned_host());

  // 1. Marking leaf view view1_child as pinned host propagates upstream to
  // ancestors (view1, root) and dynamically to sibling views (view2,
  // view2_child).
  view1_child.set_is_pinned_host();

  EXPECT_TRUE(view1_child.is_pinned_host());
  EXPECT_TRUE(view1.is_pinned_host());
  EXPECT_TRUE(root.is_pinned_host());
  EXPECT_TRUE(view2.is_pinned_host());
  EXPECT_TRUE(view2_child.is_pinned_host());

  // 2. New views created from pinned buffers (root or sibling view) inherit the
  // flag at creation.
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef new_view_from_root,
                          CreateView(root));
  EXPECT_TRUE(new_view_from_root.is_pinned_host());
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef new_view_from_view2,
                          CreateView(view2));
  EXPECT_TRUE(new_view_from_view2.is_pinned_host());
}

// Connectivity graph:
//
//                               root (kEmpty)
//                             /   |    \       \
//                       (view)  (view)  |     (kAdd)
//                         /       |     |        \
//                     view1     view2   |   root_compute_op
//                       |         \     |
//                     (kAdd)       \   /
//                       |         (kAdd)
//              view1_compute_op     |
//                         multi_input_compute_op
TEST_F(PinnedHostPropagationTest, ComputeOpIsolation) {
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef root, CreateRoot());
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef view1, CreateView(root));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef view1_compute_op,
                          CreateComputeOp({view1}));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef view2, CreateView(root));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef root_compute_op,
                          CreateComputeOp({root}));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef multi_input_compute_op,
                          CreateComputeOp({root, view2}));

  EXPECT_FALSE(root.is_pinned_host());
  EXPECT_FALSE(view1.is_pinned_host());
  EXPECT_FALSE(view1_compute_op.is_pinned_host());
  EXPECT_FALSE(view2.is_pinned_host());
  EXPECT_FALSE(root_compute_op.is_pinned_host());
  EXPECT_FALSE(multi_input_compute_op.is_pinned_host());

  // 1. Marking a non-metadata compute op (view1_compute_op) as pinned host
  // marks itself, but does NOT propagate upstream to its parent view (view1) or
  // root.
  view1_compute_op.set_is_pinned_host();
  EXPECT_TRUE(view1_compute_op.is_pinned_host());
  EXPECT_FALSE(view1.is_pinned_host());
  EXPECT_FALSE(root.is_pinned_host());

  // 2. Marking view view1 as pinned host propagates upstream to root and
  // sibling view view2, but does NOT propagate downstream to non-metadata
  // compute ops (root_compute_op, multi_input_compute_op).
  view1.set_is_pinned_host();
  EXPECT_TRUE(view1.is_pinned_host());
  EXPECT_TRUE(root.is_pinned_host());
  EXPECT_TRUE(view2.is_pinned_host());
  EXPECT_FALSE(root_compute_op.is_pinned_host());
  EXPECT_FALSE(multi_input_compute_op.is_pinned_host());

  // 3. New compute ops created from pinned buffers do not inherit the flag.
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef new_compute_op,
                          CreateComputeOp({root}));
  EXPECT_FALSE(new_compute_op.is_pinned_host());
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef new_multi_input_compute_op,
                          CreateComputeOp({root, view2}));
  EXPECT_FALSE(new_multi_input_compute_op.is_pinned_host());
}

}  // namespace
}  // namespace torch_tpu
