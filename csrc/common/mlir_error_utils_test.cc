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

#include "csrc/common/mlir_error_utils.h"

#include <string>

#include "csrc/common/error_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/mlir/utils/error_util.h"

namespace torch_tpu {
namespace {

using testing::EndsWith;
using testing::HasSubstr;

TEST(ConsumeMlirDiagnostics, NoDiagnostic) {
  mlir::MLIRContext context;
  mlir::BaseScopedDiagnosticHandler diag_handler(&context);

  EXPECT_EQ(ConsumeMlirDiagnostics(diag_handler,
                                   /* context= */ "failed to do something"),
            "failed to do something: no MLIR diagnostic was reported");
}

TEST(ConsumeMlirDiagnostics, WrapsDiagnostic) {
  mlir::MLIRContext context;
  mlir::BaseScopedDiagnosticHandler diag_handler(&context);
  mlir::emitError(mlir::UnknownLoc::get(&context), "Something went wrong.");

  const std::string message = ConsumeMlirDiagnostics(
      diag_handler, /* context= */ "failed to do something");

  EXPECT_THAT(message, HasSubstr(internal::kXlaErrorSentinel.marker));
  EXPECT_THAT(message, EndsWith("error: Something went wrong."));
  EXPECT_TRUE(diag_handler.ok());  // The diagnostic was consumed.
}

}  // namespace
}  // namespace torch_tpu
