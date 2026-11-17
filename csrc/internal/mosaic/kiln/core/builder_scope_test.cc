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

#include "csrc/internal/mosaic/kiln/core/builder_scope.h"

#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

namespace torch_tpu::kiln::internal {
namespace {

using ::absl_testing::IsOk;

struct TestScope {
  mlir::MLIRContext mlir_ctx;
  mlir::ImplicitLocOpBuilder builder{mlir::UnknownLoc::get(&mlir_ctx),
                                     &mlir_ctx};
  BuilderScope scope{&builder};
};

TEST(BuilderScopeTest, InstallsItselfAsTheCurrentScope) {
  TestScope env;

  EXPECT_EQ(&CurrentBuilderScope(), &env.scope);
  EXPECT_EQ(&env.scope.builder(), &env.builder);
  EXPECT_THAT(env.scope.error(), IsOk());
}

TEST(BuilderScopeTest, ANewScopeCanBeInstalledAfterThePreviousOneEnds) {
  {
    TestScope first;
  }
  TestScope second;

  EXPECT_EQ(&CurrentBuilderScope(), &second.scope);
}

TEST(BuilderScopeDeathTest, CurrentBuilderScopeOutsideScopeDies) {
  EXPECT_DEATH(static_cast<void>(CurrentBuilderScope()),
               "Kiln operations may only be called inside BuildKernel");
}

TEST(BuilderScopeDeathTest, NestedBuilderScopeDies) {
  EXPECT_DEATH(
      {
        TestScope outer;
        TestScope inner;
      },
      "BuilderScope cannot be nested");
}

}  // namespace
}  // namespace torch_tpu::kiln::internal
