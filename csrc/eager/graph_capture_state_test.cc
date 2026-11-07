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

#include "csrc/eager/graph_capture_state.h"

#include <optional>
#include <thread>  // NOLINT(build/c++11)
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/events_queue.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using ::testing::HasSubstr;

// Begins a graph capture on a new stream of device 0 on the current thread,
// and returns the stream.
DeviceStreamId BeginCaptureOnNewStream() {
  const DeviceStreamId stream{.device_index = 0,
                              .stream_id = NextStreamId(/*device_index=*/0)};
  EXPECT_TRUE(
      TryBeginStreamGraphCapture(stream.device_index, stream.stream_id));
  SetCurrentThreadGraphCaptureStream(stream);
  return stream;
}

// Ends the graph capture on `stream`, and clears the index.
void EndCapture(const DeviceStreamId stream) {
  EndStreamGraphCapture(stream.device_index, stream.stream_id);
  ClearCurrentThreadGraphCaptureStream();
}

TEST(GraphCaptureStateTest, ThreadsCaptureNoGraphByDefault) {
  EXPECT_FALSE(IsCurrentThreadCapturingGraph());
  EXPECT_FALSE(GetCurrentThreadGraphCaptureStream().has_value());

  const absl::Status error = TT_ERROR(error::kInternal) << "unrelated error";
  EXPECT_EQ(InvalidateCurrentThreadGraphCapture(error), error);
}

TEST(GraphCaptureStateTest, TracksTheCaptureOfTheCurrentThread) {
  const DeviceStreamId stream = BeginCaptureOnNewStream();
  const absl::Cleanup end_capture = [stream] { EndCapture(stream); };

  EXPECT_TRUE(IsCurrentThreadCapturingGraph());
  const std::optional<DeviceStreamId> capture_stream =
      GetCurrentThreadGraphCaptureStream();
  ASSERT_TRUE(capture_stream.has_value());
  EXPECT_EQ(capture_stream->device_index, stream.device_index);
  EXPECT_EQ(capture_stream->stream_id, stream.stream_id);

  // Other threads do not own the capture, even if they record its stream.
  std::thread other_thread([stream] {
    EXPECT_FALSE(IsCurrentThreadCapturingGraph());
    SetCurrentThreadGraphCaptureStream(stream);
    EXPECT_FALSE(IsCurrentThreadCapturingGraph());
    EXPECT_FALSE(
        IsStreamCapturedByCurrentThread(stream.device_index, stream.stream_id));
  });
  other_thread.join();
  EXPECT_TRUE(IsCurrentThreadCapturingGraph());
}

TEST(GraphCaptureStateTest, InvalidationKeepsTheFirstErrorUntilCaptureEnds) {
  const DeviceStreamId stream = BeginCaptureOnNewStream();
  absl::Cleanup end_capture = [stream] { EndCapture(stream); };

  const absl::Status first_error = TT_ERROR(error::kInternal) << "first";
  const absl::Status second_error = TT_ERROR(error::kInternal) << "second";
  EXPECT_EQ(InvalidateCurrentThreadGraphCapture(first_error), first_error);
  EXPECT_EQ(InvalidateCurrentThreadGraphCapture(second_error), second_error);

  // An invalidated capture is still in progress until it ends.
  EXPECT_TRUE(IsCurrentThreadCapturingGraph());
  EXPECT_TRUE(IsStreamCapturing(stream.device_index, stream.stream_id));
  EXPECT_EQ(GetStreamGraphCaptureError(stream.device_index, stream.stream_id),
            first_error);

  std::move(end_capture).Invoke();
  EXPECT_FALSE(IsCurrentThreadCapturingGraph());
  EXPECT_TRUE(
      GetStreamGraphCaptureError(stream.device_index, stream.stream_id).ok());
}

TEST(GraphCaptureStateTest, ForgetsCapturesThatEnded) {
  const DeviceStreamId stream = BeginCaptureOnNewStream();
  EndStreamGraphCapture(stream.device_index, stream.stream_id);

  // The index is cleared lazily once the capture of its stream has ended.
  EXPECT_FALSE(IsCurrentThreadCapturingGraph());
  EXPECT_FALSE(GetCurrentThreadGraphCaptureStream().has_value());

  // A new capture on the same stream by another thread does not make the
  // current thread an owner again.
  std::thread other_thread([stream] {
    ASSERT_TRUE(
        TryBeginStreamGraphCapture(stream.device_index, stream.stream_id));
  });
  other_thread.join();
  SetCurrentThreadGraphCaptureStream(stream);
  EXPECT_FALSE(IsCurrentThreadCapturingGraph());
  EndCapture(stream);
}

TEST(GraphCaptureStateTest, ValidatesTheCaptureStream) {
  const DeviceStreamId capture_stream{.device_index = 0, .stream_id = 3};
  EXPECT_TRUE(ValidateGraphCaptureStream(capture_stream, capture_stream).ok());

  const absl::Status other_stream_status = ValidateGraphCaptureStream(
      capture_stream, DeviceStreamId{.device_index = 0, .stream_id = 4});
  EXPECT_EQ(other_stream_status.code(), error::kFailedPrecondition);
  EXPECT_THAT(other_stream_status.message(),
              HasSubstr("operations captured into a graph must run on the "
                        "stream the capture began on (device 0, stream 3), but "
                        "an operation ran on device 0, stream 4"));

  const absl::Status other_device_status = ValidateGraphCaptureStream(
      capture_stream, DeviceStreamId{.device_index = 1, .stream_id = 3});
  EXPECT_THAT(other_device_status.message(),
              HasSubstr("but an operation ran on device 1, stream 3"));
}

}  // namespace
}  // namespace torch_tpu
