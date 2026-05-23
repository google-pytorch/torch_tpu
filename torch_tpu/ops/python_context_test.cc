/*
 * Copyright 2025 Google LLC
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

#include "torch_tpu/ops/python_context.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

using testing::ElementsAre;
using testing::HasSubstr;
using testing::Not;

TEST(CaptureCurrentPythonTrackback, CapturesPythonStack) {
  mlir::MLIRContext ctx;
  PythonContext python_context(
      {},
      CaptureCurrentPythonTrackback(PythonStackLocationOptions::kPythonOnly));
  auto loc = MakeMlirLocation(ctx, python_context);
  EXPECT_THAT(mlir::debugString(loc),
              Not(HasSubstr("CaptureCurrentPythonTrackback")));
}

TEST(CaptureCurrentPythonTrackback, CapturesCppStackWhenRequested) {
  mlir::MLIRContext ctx;
  PythonContext python_context(
      {},
      CaptureCurrentPythonTrackback(PythonStackLocationOptions::kPythonAndCpp));
  auto loc = MakeMlirLocation(ctx, python_context);
  EXPECT_THAT(mlir::debugString(loc),
              HasSubstr("CapturesCppStackWhenRequested"));
}

TEST(ScopedPythonContextCapturer, NoObjectIsAlive) {
  const auto& context = ScopedPythonContextCapturer::MaybeGetContext();
  EXPECT_FALSE(context.has_value());
}

TEST(ScopedPythonContextCapturer, OneObjectIsAlive) {
  {
    ScopedPythonContextCapturer capturer(OpName::kAdd);
    EXPECT_THAT(ScopedPythonContextCapturer::GetContext().op_call_chain(),
                ElementsAre("add"));
    const auto& context = ScopedPythonContextCapturer::MaybeGetContext();
    ASSERT_TRUE(context.has_value());
    EXPECT_THAT(context->op_call_chain(), ElementsAre("add"));
  }
  const auto& context = ScopedPythonContextCapturer::MaybeGetContext();
  EXPECT_FALSE(context.has_value());
}

TEST(ScopedPythonContextCapturer, TwoObjectsAreAlive) {
  {
    ScopedPythonContextCapturer capturer(OpName::kAdd);
    {
      ScopedPythonContextCapturer capturer(OpName::kBmm);
      EXPECT_THAT(ScopedPythonContextCapturer::GetContext().op_call_chain(),
                  ElementsAre("add", "bmm"));
      const auto& context = ScopedPythonContextCapturer::MaybeGetContext();
      ASSERT_TRUE(context.has_value());
      EXPECT_THAT(context->op_call_chain(), ElementsAre("add", "bmm"));
    }
    EXPECT_THAT(ScopedPythonContextCapturer::GetContext().op_call_chain(),
                ElementsAre("add"));
    const auto& context = ScopedPythonContextCapturer::MaybeGetContext();
    ASSERT_TRUE(context.has_value());
    EXPECT_THAT(context->op_call_chain(), ElementsAre("add"));
  }
  const auto& context = ScopedPythonContextCapturer::MaybeGetContext();
  EXPECT_FALSE(context.has_value());
}

class ScopedPythonContextCapturerTrackebackTest : public testing::Test {
 protected:
  ~ScopedPythonContextCapturerTrackebackTest() override {
    // Restore the default state.
    PopContextState<EnableTracebacksContextState>();
  }
};

TEST_F(ScopedPythonContextCapturerTrackebackTest,
       CapturesTracebackWhenEnabled) {
  PushContextState<EnableTracebacksContextState>(TracebackMode::kEnabled);
  {
    ScopedPythonContextCapturer capturer(OpName::kAdd);
    auto context = ScopedPythonContextCapturer::GetContext();

    // The traceback is not nullptr, but we don't check its contents because
    // this created context has no python frames (and hence, is empty).
    ASSERT_NE(context.traceback(), nullptr);
  }
}

TEST_F(ScopedPythonContextCapturerTrackebackTest,
       DoesNotCaptureTracebackWhenDisabled) {
  PushContextState<EnableTracebacksContextState>(TracebackMode::kDisabled);
  {
    ScopedPythonContextCapturer capturer(OpName::kAdd);
    auto context = ScopedPythonContextCapturer::GetContext();
    EXPECT_TRUE(context.traceback() == nullptr);
  }
}

TEST(ScopedPythonContextProvider, NoObjectIsAlive) {
  const auto& context = ScopedPythonContextProvider::MaybeGetContext();
  EXPECT_FALSE(context.has_value());
}

TEST(ScopedPythonContextProvider, OneObjectIsAlive) {
  mlir::MLIRContext ctx;
  mlir::Location loc = mlir::FileLineColLoc::get(&ctx, "add", 1, 2);
  mlir::MlirBuilder builder(ctx, loc);
  {
    ScopedPythonContextProvider provider(
        PythonContext({"add"}, CaptureCurrentPythonTrackback()), &builder);
    EXPECT_NE(builder.getLoc(), loc);
    const auto& context = ScopedPythonContextProvider::MaybeGetContext();
    ASSERT_TRUE(context.has_value());
    EXPECT_THAT(context->op_call_chain(), ElementsAre("add"));
  }
  EXPECT_EQ(builder.getLoc(), loc);
  const auto& context = ScopedPythonContextProvider::MaybeGetContext();
  EXPECT_FALSE(context.has_value());
}

TEST(ScopedPythonContextProvider, TwoObjectsAreAlive) {
  mlir::MLIRContext ctx;
  mlir::Location loc = mlir::FileLineColLoc::get(&ctx, "add", 1, 2);
  mlir::MlirBuilder builder(ctx, loc);
  {
    ScopedPythonContextProvider provider(
        PythonContext({"add"}, CaptureCurrentPythonTrackback()), &builder);
    {
      ScopedPythonContextProvider provider(
          PythonContext({"sub"}, CaptureCurrentPythonTrackback()), &builder);
      EXPECT_NE(builder.getLoc(), loc);
      const auto& context = ScopedPythonContextProvider::MaybeGetContext();
      ASSERT_TRUE(context.has_value());
      EXPECT_THAT(context->op_call_chain(), ElementsAre("sub"));
    }
    EXPECT_NE(builder.getLoc(), loc);
    const auto& context = ScopedPythonContextProvider::MaybeGetContext();
    ASSERT_TRUE(context.has_value());
    EXPECT_THAT(context->op_call_chain(), ElementsAre("add"));
  }
  EXPECT_EQ(builder.getLoc(), loc);
  const auto& context = ScopedPythonContextProvider::MaybeGetContext();
  EXPECT_FALSE(context.has_value());
}

TEST(GetOriginalOpName, ReturnsCurrentOpNameWhenNoContextIsAlive) {
  EXPECT_EQ(GetRootOpName(OpName::kAdd), "add");
}

TEST(GetOriginalOpName, ReturnsFirstOpNameWhenCapturerIsAlive) {
  {
    ScopedPythonContextCapturer capturer(OpName::kAdd);
    EXPECT_EQ(GetRootOpName(OpName::kSub), "add");
    {
      ScopedPythonContextCapturer capturer(OpName::kCatOut);
      EXPECT_EQ(GetRootOpName(OpName::kSub), "add");
    }
    EXPECT_EQ(GetRootOpName(OpName::kSub), "add");
  }
  EXPECT_EQ(GetRootOpName(OpName::kSub), "sub");
}

TEST(GetOriginalOpName, ReturnsFirstOpNameWhenProviderIsAlive) {
  mlir::MLIRContext ctx;
  mlir::Location loc = mlir::FileLineColLoc::get(&ctx, "add", 1, 2);
  mlir::MlirBuilder builder(ctx, loc);
  {
    ScopedPythonContextProvider provider(
        PythonContext({"add"}, CaptureCurrentPythonTrackback()), &builder);
    EXPECT_EQ(GetRootOpName(OpName::kSub), "add");
    {
      ScopedPythonContextProvider provider(
          PythonContext({"baz", "add"}, CaptureCurrentPythonTrackback()),
          &builder);
      EXPECT_EQ(GetRootOpName(OpName::kSub), "baz");
    }
    EXPECT_EQ(GetRootOpName(OpName::kSub), "add");
  }
  EXPECT_EQ(GetRootOpName(OpName::kSub), "sub");
}

}  // namespace
}  // namespace torch_tpu
