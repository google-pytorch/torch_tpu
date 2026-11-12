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

#include "csrc/internal/mosaic/kiln/core/error.h"

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/kiln/core/builder_scope.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

namespace torch_tpu::kiln::internal {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

struct TestScope {
  mlir::MLIRContext mlir_ctx;
  mlir::ImplicitLocOpBuilder builder{mlir::UnknownLoc::get(&mlir_ctx),
                                     &mlir_ctx};
  BuilderScope scope{&builder};
};

TEST(ErrorTest, ReportErrorRecordsOnlyTheFirstError) {
  TestScope env;

  ReportError(TT_ERROR(error::kInvalidArgument) << "first error");
  ReportError(TT_ERROR(error::kInternal) << "second error");

  EXPECT_THAT(env.scope.error(),
              StatusIs(error::kInvalidArgument, HasSubstr("first error")));
}

TEST(ErrorDeathTest, ReportErrorOutsideScopeDies) {
  EXPECT_DEATH(ReportError(TT_ERROR(error::kInternal) << "boom"),
               "Kiln operations may only be called inside BuildKernel");
}

TEST(ErrorDeathTest, ReportErrorWithOkStatusDies) {
  EXPECT_DEATH(
      {
        TestScope env;
        ReportError(absl::OkStatus());
      },
      "ReportError\\(\\) requires a non-OK status");
}

}  // namespace
}  // namespace torch_tpu::kiln::internal
