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

#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "csrc/common/error_utils.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

TEST(GraphCaptureStateTest, DefaultWithoutActiveCapture) {
  EXPECT_EQ(GetCurrentThreadGraphCapture(), nullptr);
  EXPECT_FALSE(IsCurrentThreadCapturingGraph());
  EXPECT_TRUE(GetCurrentThreadGraphCaptureError().ok());
  const absl::Status no_op_error = TT_ERROR(error::kInternal) << "no-op";
  EXPECT_EQ(InvalidateCurrentThreadGraphCapture(no_op_error), no_op_error);
  EXPECT_TRUE(GetCurrentThreadGraphCaptureError().ok());
}

TEST(GraphCaptureStateTest, LifecycleThreadIsolationAndStickyError) {
  auto capture = std::make_shared<GraphCaptureState>();
  GraphCaptureState* const raw_capture = capture.get();
  PushCurrentThreadGraphCapture(std::move(capture));
  bool popped = false;
  const absl::Cleanup reset_capture = [&] {
    if (!popped) {
      PopCurrentThreadGraphCapture();
    }
  };

  // Before activation, the thread holds the state but is not actively
  // recording operations into the graph.
  EXPECT_EQ(GetCurrentThreadGraphCapture(), raw_capture);
  EXPECT_FALSE(IsCurrentThreadCapturingGraph());
  const absl::Status ignored_error = TT_ERROR(error::kInternal) << "ignored";
  EXPECT_EQ(InvalidateCurrentThreadGraphCapture(ignored_error), ignored_error);
  EXPECT_TRUE(GetCurrentThreadGraphCaptureError().ok());

  raw_capture->Activate();
  EXPECT_TRUE(IsCurrentThreadCapturingGraph());

  // Other threads do not inherit the capture context state.
  std::thread other([&] {
    EXPECT_EQ(GetCurrentThreadGraphCapture(), nullptr);
    EXPECT_FALSE(IsCurrentThreadCapturingGraph());
  });
  other.join();

  // Only the first invalidation error is retained.
  const absl::Status first_error = TT_ERROR(error::kInternal) << "first";
  const absl::Status second_error = TT_ERROR(error::kInternal) << "second";
  EXPECT_EQ(InvalidateCurrentThreadGraphCapture(first_error), first_error);
  EXPECT_EQ(InvalidateCurrentThreadGraphCapture(second_error), second_error);
  EXPECT_EQ(GetCurrentThreadGraphCaptureError(), first_error);

  PopCurrentThreadGraphCapture();
  popped = true;
  EXPECT_EQ(GetCurrentThreadGraphCapture(), nullptr);
  EXPECT_FALSE(IsCurrentThreadCapturingGraph());
}

}  // namespace
}  // namespace torch_tpu
