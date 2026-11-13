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

#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/value_handle.h"
#include "gtest/gtest.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

namespace torch_tpu::kiln::internal {
namespace {

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

TEST(ContextTest, WrapAndUnwrapValue) {
  mlir::MLIRContext mlir_ctx;
  mlir::Block block;
  const mlir::Value value = block.addArgument(
      mlir::IntegerType::get(&mlir_ctx, 32), mlir::UnknownLoc::get(&mlir_ctx));

  const TestHandle wrapped = wrap<TestHandle>(value, i32);

  EXPECT_EQ(wrapped.dtype(), i32);
  EXPECT_EQ(unwrap(wrapped), value);
}

TEST(ContextDeathTest, WrapNullValueDies) {
  EXPECT_DEATH(static_cast<void>(wrap<TestHandle>(mlir::Value(), f32)),
               "ValueHandle requires a non-null value");
}

}  // namespace
}  // namespace torch_tpu::kiln::internal
