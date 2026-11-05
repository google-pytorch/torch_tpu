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

#include "csrc/eager/tpu_graph.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/synchronization/barrier.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "csrc/common/macro_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#if TT_TORCH_VERSION_GE(2, 12)
#include "ATen/core/GraphImplInterface.h"
#endif
#include "ATen/core/TensorBody.h"
#include "c10/core/AutogradState.h"
#include "c10/core/Stream.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation_cache.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/graph_capture_state.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/tpu_hooks.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/pjrt/pjrt_state.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

#if TT_TORCH_VERSION_GE(2, 12)

class TpuGraphTest : public testing::Test {
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
    ShutDownMaterializationState();
    CompilationCache::ShutDown();
    PjrtBackend::GetInstance().Shutdown();
  }

  void SetUp() override { ClearAllStreams(); }
  void TearDown() override { ClearAllStreams(); }
};

absl::StatusOr<at::Tensor> CreateFloatTensor(absl::Span<const float> values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(float));
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buf,
                      HostToDeviceTransfer::FromBytesVector(
                          std::move(bytes), mlir::ElementType::F32,
                          {static_cast<int64_t>(values.size())})
                          .Run());
  return MakeTensor(std::move(buf), /*device_idx=*/0);
}

absl::Status AssignFloatTensor(at::Tensor& tensor,
                               absl::Span<const float> values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(float));
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buf,
                      HostToDeviceTransfer::FromBytesVector(
                          std::move(bytes), mlir::ElementType::F32,
                          {static_cast<int64_t>(values.size())})
                          .Run());
  return AssignBufferToAtTensor(std::move(buf), tensor);
}

absl::StatusOr<std::vector<float>> ReadFloatTensor(const at::Tensor& tensor) {
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef buf,
      MaterializeAndReturn(tensor, MaterializationReason::kCpuTransfer));
  std::vector<float> out(tensor.numel());
  TT_RETURN_IF_ERROR(DeviceToHostTransfer::IntoBytes(std::move(buf), out.data())
                         .Run()
                         .Await());
  return out;
}

absl::StatusOr<at::Tensor> CaptureDoubleOp(
    const at::Tensor& input, OpSplitMode split_mode = OpSplitMode::kNone) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef in_buf, GetBuffer(input));
  Shape out_shape(CopyIntVector(in_buf.dimensions()), in_buf.element_type());
  auto op_builder = [](mlir::MlirBuilder&, absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(inputs.size() == 1, error::kInternal)
        << "expected 1 input, got " << inputs.size();
    return DynamicMlirOpResults{mlir::stablehlo::Add(inputs[0], inputs[0])};
  };
  TT_ASSIGN_OR_RETURN(std::vector<DeviceBufferRef> results,
                      internal::CreateDeferredDeviceBufferList({
                          .op_name = OpName::kAdd,
                          .op_builder = std::move(op_builder),
                          .op_param_cache_keys = OpParamCacheKeys::Empty(),
                          .inputs = {std::move(in_buf)},
                          .output_shapes = {std::move(out_shape)},
                          .split_mode = split_mode,
                      }));
  return MakeTensor(std::move(results[0]), /*device_idx=*/0);
}

TEST_F(TpuGraphTest, MultiStageCaptureAndReplay) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  TT_ASSERT_OK_AND_ASSIGN(at::Tensor x,
                          CreateFloatTensor({1.0f, 2.0f, 3.0f, 4.0f}));

  const c10::StreamId capture_stream = NextStreamId(/*device_index=*/0);
  const StreamId prev_stream =
      ExchangeCurrentStreamId(/*device_index=*/0, capture_stream);
  const absl::Cleanup restore_stream = [prev_stream] {
    ExchangeCurrentStreamId(/*device_index=*/0, prev_stream);
  };

  TpuGraphImpl graph(at::GraphImplArgs{.keep_graph = true});
  graph.capture_begin(/*pool=*/{0, 0}, at::GraphCaptureMode::Default);
  TT_ASSERT_OK_AND_ASSIGN(at::Tensor stage0,
                          CaptureDoubleOp(x, OpSplitMode::kSplitAfter));
  TT_ASSERT_OK_AND_ASSIGN(at::Tensor stage1,
                          CaptureDoubleOp(stage0, OpSplitMode::kSplitAfter));
  TT_ASSERT_OK_AND_ASSIGN(at::Tensor stage2,
                          CaptureDoubleOp(stage1, OpSplitMode::kNone));
  graph.capture_end();

  graph.replay();
  TT_ASSERT_OK_AND_ASSIGN(std::vector<float> out1, ReadFloatTensor(stage2));
  EXPECT_THAT(out1, testing::ElementsAre(8.0f, 16.0f, 24.0f, 32.0f));

  ASSERT_TRUE(AssignFloatTensor(x, {2.0f, -1.0f, 0.5f, 4.0f}).ok());
  graph.replay();
  TT_ASSERT_OK_AND_ASSIGN(std::vector<float> out2, ReadFloatTensor(stage2));
  EXPECT_THAT(out2, testing::ElementsAre(16.0f, -8.0f, 4.0f, 32.0f));
}

TEST_F(TpuGraphTest,
       ConcurrentWaitersSerializeOnOrphanAbortAndRestoreStorages) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  TT_ASSERT_OK_AND_ASSIGN(at::Tensor x,
                          CreateFloatTensor({1.0f, 2.0f, 3.0f, 4.0f}));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef init_x, GetBaseBuffer(x));

  const c10::StreamId capture_stream = NextStreamId(/*device_index=*/0);
  absl::Notification captured;
  absl::Notification graph_destroyed;
  std::unique_ptr<TpuGraphImpl> graph;
  at::Tensor created_out;  // UNINITIALIZED_TENSOR_OK=assigned by owner

  std::thread owner([&] {
    ScopedPythonContextCapturer thread_capturer(OpName::kAdd);
    const StreamId prev =
        ExchangeCurrentStreamId(/*device_index=*/0, capture_stream);
    const absl::Cleanup restore = [prev] {
      ExchangeCurrentStreamId(/*device_index=*/0, prev);
    };
    graph = std::make_unique<TpuGraphImpl>();
    graph->capture_begin(/*pool=*/{0, 0}, at::GraphCaptureMode::Default);
    TT_ASSERT_OK_AND_ASSIGN(created_out, CaptureDoubleOp(x));
    captured.Notify();
    graph_destroyed.WaitForNotification();
  });

  captured.WaitForNotification();
  // Destroying the graph on a non-owner thread leaves capture active/orphaned
  // while the owner thread is still alive.
  graph.reset();
  EXPECT_TRUE(IsStreamCapturing(/*device_index=*/0, capture_stream));
  MaybeEndOrphanedGraphCaptures();
  EXPECT_TRUE(IsStreamCapturing(/*device_index=*/0, capture_stream));

  graph_destroyed.Notify();
  owner.join();

  constexpr int kNumWaiters = 4;
  absl::Barrier ready_barrier(kNumWaiters);
  std::atomic<int> failures{0};
  std::vector<std::thread> waiters;
  waiters.reserve(kNumWaiters);
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&] {
      ready_barrier.Block();
      const absl::StatusOr<DeviceBufferRef> buf = GetBaseBuffer(x);
      if (!buf.ok() || *buf != init_x) {
        failures.fetch_add(1);
      }
    });
  }
  for (std::thread& t : waiters) {
    t.join();
  }

  EXPECT_EQ(failures.load(), 0);
  EXPECT_FALSE(IsStreamCapturing(/*device_index=*/0, capture_stream));
  const absl::StatusOr<std::vector<float>> read_created =
      ReadFloatTensor(created_out);
  ASSERT_FALSE(read_created.ok());
  EXPECT_THAT(
      read_created.status().message(),
      testing::HasSubstr(
          "cannot use a tensor created during an aborted graph capture"));
}

TEST_F(TpuGraphTest, OrphanedCaptureEndsWhenOwnerBeginsNextCapture) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  TT_ASSERT_OK_AND_ASSIGN(at::Tensor x,
                          CreateFloatTensor({1.0f, 2.0f, 3.0f, 4.0f}));
  const c10::StreamId capture_stream = NextStreamId(/*device_index=*/0);
  absl::Notification first_captured;
  absl::Notification first_destroyed;
  std::unique_ptr<TpuGraphImpl> first_graph;
  at::Tensor out;  // UNINITIALIZED_TENSOR_OK=assigned by owner

  std::thread owner([&] {
    ScopedPythonContextCapturer thread_capturer(OpName::kAdd);
    const StreamId prev =
        ExchangeCurrentStreamId(/*device_index=*/0, capture_stream);
    const absl::Cleanup restore = [prev] {
      ExchangeCurrentStreamId(/*device_index=*/0, prev);
    };
    first_graph = std::make_unique<TpuGraphImpl>();
    first_graph->capture_begin(/*pool=*/{0, 0}, at::GraphCaptureMode::Default);
    TT_ASSERT_OK_AND_ASSIGN(at::Tensor ignored, CaptureDoubleOp(x));
    (void)ignored;
    first_captured.Notify();
    first_destroyed.WaitForNotification();

    // Starting a new capture on the owner thread aborts the orphaned capture.
    TpuGraphImpl second_graph;
    second_graph.capture_begin(/*pool=*/{0, 0}, at::GraphCaptureMode::Default);
    TT_ASSERT_OK_AND_ASSIGN(out, CaptureDoubleOp(x));
    second_graph.capture_end();
    second_graph.replay();
  });

  first_captured.WaitForNotification();
  first_graph.reset();
  EXPECT_TRUE(IsStreamCapturing(/*device_index=*/0, capture_stream));
  first_destroyed.Notify();
  owner.join();

  TT_ASSERT_OK_AND_ASSIGN(std::vector<float> vals, ReadFloatTensor(out));
  EXPECT_THAT(vals, testing::ElementsAre(2.0f, 4.0f, 6.0f, 8.0f));
}

TEST_F(TpuGraphTest, KeepGraphConcurrentReplaysSerializeInstantiation) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  TT_ASSERT_OK_AND_ASSIGN(at::Tensor x,
                          CreateFloatTensor({1.0f, 2.0f, 3.0f, 4.0f}));

  const c10::StreamId capture_stream = NextStreamId(/*device_index=*/0);
  const StreamId prev_stream =
      ExchangeCurrentStreamId(/*device_index=*/0, capture_stream);
  const absl::Cleanup restore_stream = [prev_stream] {
    ExchangeCurrentStreamId(/*device_index=*/0, prev_stream);
  };

  TpuGraphImpl graph(at::GraphImplArgs{.keep_graph = true});
  graph.capture_begin(/*pool=*/{0, 0}, at::GraphCaptureMode::Default);
  TT_ASSERT_OK_AND_ASSIGN(at::Tensor stage0,
                          CaptureDoubleOp(x, OpSplitMode::kSplitAfter));
  TT_ASSERT_OK_AND_ASSIGN(at::Tensor stage1,
                          CaptureDoubleOp(stage0, OpSplitMode::kNone));
  graph.capture_end();
  EXPECT_NE(graph.pool().first, 0);

  constexpr int kNumThreads = 4;
  absl::Barrier barrier(kNumThreads);
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&] {
      ScopedPythonContextCapturer thread_capturer(OpName::kAdd);
      barrier.Block();
      try {
        graph.replay();
      } catch (...) {
        failures.fetch_add(1);
      }
    });
  }
  for (std::thread& t : threads) {
    t.join();
  }
  EXPECT_EQ(failures.load(), 0);
  TT_ASSERT_OK_AND_ASSIGN(std::vector<float> vals, ReadFloatTensor(stage1));
  EXPECT_THAT(vals, testing::ElementsAre(4.0f, 8.0f, 12.0f, 16.0f));
}

TEST_F(TpuGraphTest,
       ClearAllStreamsMidCaptureAbortsCaptureEndAndRestoresState) {
  ScopedPythonContextCapturer capturer(OpName::kAdd);
  TT_ASSERT_OK_AND_ASSIGN(at::Tensor x, CreateFloatTensor({1.0f, 2.0f}));
  TT_ASSERT_OK_AND_ASSIGN(at::Tensor other, CreateFloatTensor({3.0f, 4.0f}));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef init_x, GetBaseBuffer(x));
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef init_other,
                          GetBaseBuffer(other));

  const c10::StreamId capture_stream = NextStreamId(/*device_index=*/0);
  const StreamId prev_stream =
      ExchangeCurrentStreamId(/*device_index=*/0, capture_stream);
  const absl::Cleanup restore_stream = [prev_stream] {
    ExchangeCurrentStreamId(/*device_index=*/0, prev_stream);
  };

  TpuGraphImpl graph;
  graph.capture_begin(/*pool=*/{0, 0}, at::GraphCaptureMode::Default);
  TT_ASSERT_OK_AND_ASSIGN(at::Tensor y, CaptureDoubleOp(x));
  (void)y;
  EXPECT_FALSE(
      c10::AutogradState::get_tls_state().get_multithreading_enabled());

  ClearAllStreams();
  // After ClearAllStreams() resets stream capture state, reading `other` no
  // longer installs a capture placeholder.
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef peeked_other,
                          GetBaseBuffer(other));
  EXPECT_EQ(peeked_other, init_other);

  std::string err;
  try {
    graph.capture_end();
    FAIL() << "Expected capture_end() to fail after ClearAllStreams()";
  } catch (const std::exception& e) {
    err = e.what();
  }
  EXPECT_THAT(err, testing::HasSubstr("capture stream state was cleared"));
  EXPECT_TRUE(c10::AutogradState::get_tls_state().get_multithreading_enabled());
  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef restored_x, GetBaseBuffer(x));
  EXPECT_EQ(restored_x, init_x);
}

#endif  // TT_TORCH_VERSION_GE(2, 12)

}  // namespace
}  // namespace torch_tpu
