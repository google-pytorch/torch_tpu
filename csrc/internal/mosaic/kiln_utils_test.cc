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

#include "csrc/internal/mosaic/kiln_utils.h"

#include "absl/status/status_matchers.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(ToKilnDTypeTest, SupportedElementTypes) {
  EXPECT_THAT(ToKilnDType(mlir::ElementType::PRED), IsOkAndHolds(kiln::bool_));
  EXPECT_THAT(ToKilnDType(mlir::ElementType::I8), IsOkAndHolds(kiln::i8));
  EXPECT_THAT(ToKilnDType(mlir::ElementType::I16), IsOkAndHolds(kiln::i16));
  EXPECT_THAT(ToKilnDType(mlir::ElementType::I32), IsOkAndHolds(kiln::i32));
  EXPECT_THAT(ToKilnDType(mlir::ElementType::I64), IsOkAndHolds(kiln::i64));
  EXPECT_THAT(ToKilnDType(mlir::ElementType::UI8), IsOkAndHolds(kiln::u8));
  EXPECT_THAT(ToKilnDType(mlir::ElementType::UI16), IsOkAndHolds(kiln::u16));
  EXPECT_THAT(ToKilnDType(mlir::ElementType::UI32), IsOkAndHolds(kiln::u32));
  EXPECT_THAT(ToKilnDType(mlir::ElementType::UI64), IsOkAndHolds(kiln::u64));
  EXPECT_THAT(ToKilnDType(mlir::ElementType::F16), IsOkAndHolds(kiln::f16));
  EXPECT_THAT(ToKilnDType(mlir::ElementType::BF16), IsOkAndHolds(kiln::bf16));
  EXPECT_THAT(ToKilnDType(mlir::ElementType::F32), IsOkAndHolds(kiln::f32));
}

TEST(ToKilnDTypeTest, UnsupportedElementTypeReturnsError) {
  EXPECT_THAT(ToKilnDType(mlir::ElementType::F64),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("unsupported Kiln dtype float64")));
}

}  // namespace
}  // namespace torch_tpu
