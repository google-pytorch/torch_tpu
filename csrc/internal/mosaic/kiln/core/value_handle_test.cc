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

#include "csrc/internal/mosaic/kiln/core/value_handle.h"

#include <memory>
#include <utility>

#include "gtest/gtest.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

namespace torch_tpu::kiln::internal {
namespace {

TEST(ValueHandleTest, HoldsNonNullValue) {
  mlir::MLIRContext context;
  mlir::Block block;
  const mlir::Value value = block.addArgument(
      mlir::IntegerType::get(&context, 32), mlir::UnknownLoc::get(&context));

  const ValueHandle handle(value);

  EXPECT_FALSE(handle.is_null());
  EXPECT_EQ(handle.value(), value);
}

TEST(ValueHandleTest, NullGivesNullHandle) {
  const ValueHandle null_handle = ValueHandle::Null();
  EXPECT_TRUE(null_handle.is_null());

  const ValueHandle copied(null_handle);
  EXPECT_TRUE(copied.is_null());

  mlir::MLIRContext context;
  mlir::Block block;
  const mlir::Value value = block.addArgument(
      mlir::IntegerType::get(&context, 32), mlir::UnknownLoc::get(&context));
  ValueHandle assigned(value);
  ASSERT_FALSE(assigned.is_null());
  assigned = null_handle;
  EXPECT_TRUE(assigned.is_null());
}

TEST(ValueHandleDeathTest, NullValueDies) {
  EXPECT_DEATH(static_cast<void>(ValueHandle(mlir::Value())),
               "ValueHandle requires a non-null value");
}

TEST(ValueHandleDeathTest, ValueOfNullHandleDies) {
  const ValueHandle handle = ValueHandle::Null();

  EXPECT_DEATH(static_cast<void>(handle.value()), "ValueHandle is null");
}

class TestHandle {
 public:
  explicit TestHandle(ValueHandle h, int tag) : handle_(h), tag_(tag) {}

  ValueHandle handle() const { return handle_; }
  int tag() const { return tag_; }

 private:
  ValueHandle handle_;
  int tag_;
};

TEST(ValueHandleTest, AnyNullAndCreateNullValueForwardArgs) {
  mlir::MLIRContext context;
  mlir::Block block;
  const mlir::Value value = block.addArgument(
      mlir::IntegerType::get(&context, 32), mlir::UnknownLoc::get(&context));

  const TestHandle non_null(ValueHandle(value), 1);
  const TestHandle null = CreateNullValue<TestHandle>(7);

  EXPECT_TRUE(null.handle().is_null());
  EXPECT_EQ(null.tag(), 7);
  EXPECT_FALSE(AnyNull(non_null));
  EXPECT_FALSE(AnyNull(non_null, non_null));
  EXPECT_TRUE(AnyNull(null));
  EXPECT_TRUE(AnyNull(null, non_null));
  EXPECT_TRUE(AnyNull(non_null, null));
}

// A handle type that, like `Semaphore`, is constructed from the handle alone.
class BareTestHandle {
 public:
  explicit BareTestHandle(ValueHandle h) : handle_(h) {}

  ValueHandle handle() const { return handle_; }

 private:
  ValueHandle handle_;
};

// A handle type with a move-only constructor argument.
class OwningTestHandle {
 public:
  explicit OwningTestHandle(ValueHandle h, std::unique_ptr<int> tag)
      : handle_(h), tag_(std::move(tag)) {}

  ValueHandle handle() const { return handle_; }
  int tag() const { return *tag_; }

 private:
  ValueHandle handle_;
  std::unique_ptr<int> tag_;
};

TEST(ValueHandleTest, AnyNullAcceptsMixedHandleTypesAndNoHandles) {
  mlir::MLIRContext context;
  mlir::Block block;
  const mlir::Value value = block.addArgument(
      mlir::IntegerType::get(&context, 32), mlir::UnknownLoc::get(&context));

  const TestHandle non_null(ValueHandle(value), 1);
  const BareTestHandle bare_non_null{ValueHandle(value)};
  const BareTestHandle bare_null = CreateNullValue<BareTestHandle>();

  EXPECT_FALSE(AnyNull());
  EXPECT_FALSE(AnyNull(non_null, bare_non_null));
  EXPECT_TRUE(AnyNull(non_null, bare_non_null, bare_null));
}

TEST(ValueHandleTest, CreateNullValueWithoutExtraArgs) {
  EXPECT_TRUE(CreateNullValue<BareTestHandle>().handle().is_null());
}

TEST(ValueHandleTest, CreateNullValueForwardsMoveOnlyArg) {
  const OwningTestHandle null =
      CreateNullValue<OwningTestHandle>(std::make_unique<int>(7));

  EXPECT_TRUE(null.handle().is_null());
  EXPECT_EQ(null.tag(), 7);
}

}  // namespace
}  // namespace torch_tpu::kiln::internal
