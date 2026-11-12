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

#include "csrc/internal/mosaic/kiln/core/context.h"

#include "absl/status/status_matchers.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/kiln/core/builder_scope.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/error.h"
#include "csrc/internal/mosaic/kiln/core/value_handle.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

namespace torch_tpu::kiln::internal {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class TestHandle {
 public:
  explicit TestHandle(ValueHandle handle, DType dtype)
      : handle_(handle), dtype_(dtype) {}

  ValueHandle handle() const { return handle_; }
  DType dtype() const { return dtype_; }

 private:
  ValueHandle handle_;
  DType dtype_;
};

struct TestScope {
  mlir::MLIRContext mlir_ctx;
  mlir::ImplicitLocOpBuilder builder{mlir::UnknownLoc::get(&mlir_ctx),
                                     &mlir_ctx};
  BuilderScope scope{&builder};
  mlir::Block block;

  mlir::Value CreateI32Arg() {
    return block.addArgument(mlir::IntegerType::get(&mlir_ctx, 32),
                             mlir::UnknownLoc::get(&mlir_ctx));
  }
};

TEST(ContextTest, WrapAndUnwrapNonNullValue) {
  TestScope env;
  const mlir::Value value = env.CreateI32Arg();

  const TestHandle wrapped = wrap<TestHandle>(value, i32);

  EXPECT_FALSE(wrapped.handle().is_null());
  EXPECT_EQ(wrapped.dtype(), i32);
  EXPECT_EQ(ValueOf(wrapped.handle()), value);
  EXPECT_EQ(unwrap(wrapped), value);
  EXPECT_THAT(env.scope.error(), IsOk());
}

TEST(ContextTest, WrapNullValueReportsInternalErrorAndReturnsNullHandle) {
  TestScope env;

  const TestHandle wrapped = wrap<TestHandle>(mlir::Value(), f32);

  EXPECT_TRUE(wrapped.handle().is_null());
  EXPECT_EQ(wrapped.dtype(), f32);
  EXPECT_THAT(env.scope.error(),
              StatusIs(error::kInternal,
                       HasSubstr("a Kiln operation produced no value")));
}

TEST(ContextDeathTest, ValueOfAndUnwrapNullDieWithScopeError) {
  EXPECT_DEATH(
      {
        TestScope env;
        ReportError(TT_ERROR(error::kInvalidArgument) << "bad op");
        static_cast<void>(ValueOf(ValueHandle::Null()));
      },
      "value of a failed Kiln operation was used.*bad op");

  EXPECT_DEATH(
      {
        TestScope env;
        ReportError(TT_ERROR(error::kInvalidArgument) << "bad handle");
        static_cast<void>(unwrap(CreateNullValue<TestHandle>(f32)));
      },
      "value of a failed Kiln operation was used.*bad handle");

  EXPECT_DEATH(
      {
        TestScope env;
        static_cast<void>(ValueOf(ValueHandle::Null()));
      },
      "value of a failed Kiln operation was used, with no error recorded");
}

}  // namespace
}  // namespace torch_tpu::kiln::internal
