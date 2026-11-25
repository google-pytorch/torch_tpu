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

#include "csrc/common/aten_utils.h"

#include "ATen/ops/empty.h"
#include "absl/status/status_matchers.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/DimVector.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;
using testing::HasSubstr;

TEST(AtenUtilsTest, ConcreteSizeToDimensionsConvertsConcreteSizes) {
  const c10::SymDimVector sizes = {c10::SymInt(2), c10::SymInt(0),
                                   c10::SymInt(5)};
  TT_ASSERT_OK_AND_ASSIGN(const Dimensions dims,
                          ConcreteSizeToDimensions(sizes));
  EXPECT_EQ(dims, (Dimensions{2, 0, 5}));

  const c10::SymDimVector rank7_sizes = {
      c10::SymInt(1), c10::SymInt(2), c10::SymInt(3), c10::SymInt(4),
      c10::SymInt(5), c10::SymInt(6), c10::SymInt(7)};
  TT_ASSERT_OK_AND_ASSIGN(const Dimensions rank7_dims,
                          ConcreteSizeToDimensions(rank7_sizes));
  EXPECT_EQ(rank7_dims, (Dimensions{1, 2, 3, 4, 5, 6, 7}));

  TT_ASSERT_OK_AND_ASSIGN(const Dimensions empty_dims,
                          ConcreteSizeToDimensions(c10::SymIntArrayRef()));
  EXPECT_EQ(empty_dims, Dimensions());
}

TEST(AtenUtilsTest, InferComputationDtypeFloatingPointAndComplex) {
  EXPECT_EQ(InferComputationDtype(at::ScalarType::BFloat16),
            at::ScalarType::Float);
  TT_ASSERT_OK_AND_ASSIGN(const mlir::ElementType bf16_comp,
                          InferComputationDtype(mlir::ElementType::BF16));
  EXPECT_EQ(bf16_comp, mlir::ElementType::F32);

  EXPECT_EQ(InferComputationDtype(at::ScalarType::Half), at::ScalarType::Float);
  TT_ASSERT_OK_AND_ASSIGN(const mlir::ElementType f16_comp,
                          InferComputationDtype(mlir::ElementType::F16));
  EXPECT_EQ(f16_comp, mlir::ElementType::F32);

  EXPECT_EQ(InferComputationDtype(at::ScalarType::Float),
            at::ScalarType::Float);
  TT_ASSERT_OK_AND_ASSIGN(const mlir::ElementType f32_comp,
                          InferComputationDtype(mlir::ElementType::F32));
  EXPECT_EQ(f32_comp, mlir::ElementType::F32);

  EXPECT_EQ(InferComputationDtype(at::ScalarType::Double),
            at::ScalarType::Double);
  TT_ASSERT_OK_AND_ASSIGN(const mlir::ElementType f64_comp,
                          InferComputationDtype(mlir::ElementType::F64));
  EXPECT_EQ(f64_comp, mlir::ElementType::F64);

  EXPECT_EQ(InferComputationDtype(at::ScalarType::Float4_e2m1fn_x2),
            at::ScalarType::Float);
  TT_ASSERT_OK_AND_ASSIGN(const mlir::ElementType fp4_comp,
                          InferComputationDtype(mlir::ElementType::F4E2M1FN));
  EXPECT_EQ(fp4_comp, mlir::ElementType::F32);

  EXPECT_EQ(InferComputationDtype(at::ScalarType::ComplexFloat),
            at::ScalarType::ComplexFloat);
  TT_ASSERT_OK_AND_ASSIGN(const mlir::ElementType cf32_comp,
                          InferComputationDtype(mlir::ElementType::COMPLEXF32));
  EXPECT_EQ(cf32_comp, mlir::ElementType::COMPLEXF32);

  EXPECT_EQ(InferComputationDtype(at::ScalarType::ComplexDouble),
            at::ScalarType::ComplexDouble);
  TT_ASSERT_OK_AND_ASSIGN(const mlir::ElementType cf64_comp,
                          InferComputationDtype(mlir::ElementType::COMPLEXF64));
  EXPECT_EQ(cf64_comp, mlir::ElementType::COMPLEXF64);
}

TEST(AtenUtilsTest, InferComputationDtypeIntegralAndUnsigned) {
  for (const at::ScalarType scalar_type : {
           at::ScalarType::Bool,
           at::ScalarType::Int2,
           at::ScalarType::Int4,
           at::ScalarType::Char,
           at::ScalarType::Short,
           at::ScalarType::Int,
           at::ScalarType::Long,
           at::ScalarType::UInt2,
           at::ScalarType::UInt4,
           at::ScalarType::Byte,
           at::ScalarType::UInt16,
           at::ScalarType::UInt32,
           at::ScalarType::UInt64,
       }) {
    EXPECT_EQ(InferComputationDtype(scalar_type), scalar_type);
  }

  for (const mlir::ElementType elem_type : {
           mlir::ElementType::PRED,
           mlir::ElementType::I2,
           mlir::ElementType::I4,
           mlir::ElementType::I8,
           mlir::ElementType::I16,
           mlir::ElementType::I32,
           mlir::ElementType::I64,
           mlir::ElementType::UI2,
           mlir::ElementType::UI4,
           mlir::ElementType::UI8,
           mlir::ElementType::UI16,
           mlir::ElementType::UI32,
           mlir::ElementType::UI64,
       }) {
    TT_ASSERT_OK_AND_ASSIGN(const mlir::ElementType comp_type,
                            InferComputationDtype(elem_type));
    EXPECT_EQ(comp_type, elem_type);
  }
}

TEST(AtenUtilsTest, IsShellDtypeIdentifiesShellAndNonShellDtypes) {
  for (const at::ScalarType shell_dtype : {
           at::ScalarType::Float4_e2m1fn_x2,
           at::ScalarType::Bits1x8,
           at::ScalarType::Bits2x4,
           at::ScalarType::Bits4x2,
           at::ScalarType::Bits8,
           at::ScalarType::Bits16,
           at::ScalarType::UInt1,
           at::ScalarType::UInt2,
           at::ScalarType::UInt3,
           at::ScalarType::UInt4,
           at::ScalarType::UInt5,
           at::ScalarType::UInt6,
           at::ScalarType::UInt7,
           at::ScalarType::Int1,
           at::ScalarType::Int2,
           at::ScalarType::Int3,
           at::ScalarType::Int4,
           at::ScalarType::Int5,
           at::ScalarType::Int6,
           at::ScalarType::Int7,
       }) {
    EXPECT_TRUE(IsShellDtype(shell_dtype)) << shell_dtype;
  }

  for (const at::ScalarType non_shell_dtype : {
           at::ScalarType::Bool,
           at::ScalarType::Byte,
           at::ScalarType::Char,
           at::ScalarType::Short,
           at::ScalarType::Int,
           at::ScalarType::Long,
           at::ScalarType::UInt16,
           at::ScalarType::UInt32,
           at::ScalarType::UInt64,
           at::ScalarType::Half,
           at::ScalarType::BFloat16,
           at::ScalarType::Float,
           at::ScalarType::Double,
           at::ScalarType::Float8_e4m3fn,
           at::ScalarType::Float8_e5m2,
           at::ScalarType::Float8_e8m0fnu,
           at::ScalarType::ComplexFloat,
           at::ScalarType::ComplexDouble,
       }) {
    EXPECT_FALSE(IsShellDtype(non_shell_dtype)) << non_shell_dtype;
  }
}

TEST(AtenUtilsTest, ValidateOutDtypeShellDtypesAndOrdering) {
  // Same-dtype shell pass-through is allowed regardless of
  // allow_out_dtype_cast.
  for (const at::ScalarType shell_dtype : {
           at::ScalarType::Float4_e2m1fn_x2,
           at::ScalarType::Bits8,
           at::ScalarType::Int4,
           at::ScalarType::UInt2,
       }) {
    const at::Tensor out =
        at::empty({4}, at::TensorOptions().dtype(shell_dtype));
    EXPECT_THAT(
        ValidateOutDtype(out, shell_dtype, /*allow_out_dtype_cast=*/true),
        IsOk());
    EXPECT_THAT(
        ValidateOutDtype(out, shell_dtype, /*allow_out_dtype_cast=*/false),
        IsOk());
  }

  // Shell-as-output casts that c10::canCast permits raise NotImplementedError.
  EXPECT_THAT(
      ValidateOutDtype(at::empty({4}, at::TensorOptions().dtype(
                                          at::ScalarType::Float4_e2m1fn_x2)),
                       at::ScalarType::Float),
      StatusIs(
          error::kPythonNotImplementedError,
          HasSubstr(
              "casting from float32 to float4_e2m1fn_x2 is not supported")));
  EXPECT_THAT(
      ValidateOutDtype(
          at::empty({4}, at::TensorOptions().dtype(at::ScalarType::Bits8)),
          at::ScalarType::Float),
      StatusIs(error::kPythonNotImplementedError,
               HasSubstr("casting from float32 to bits8 is not supported")));
  EXPECT_THAT(
      ValidateOutDtype(
          at::empty({4}, at::TensorOptions().dtype(at::ScalarType::Int4)),
          at::ScalarType::Int),
      StatusIs(error::kPythonNotImplementedError,
               HasSubstr("casting from int32 to int4 is not supported")));
  EXPECT_THAT(
      ValidateOutDtype(
          at::empty({4}, at::TensorOptions().dtype(at::ScalarType::UInt2)),
          at::ScalarType::Int),
      StatusIs(error::kPythonNotImplementedError,
               HasSubstr("casting from int32 to uint2 is not supported")));

  EXPECT_THAT(
      ValidateOutDtype(
          at::empty({4}, at::TensorOptions().dtype(at::ScalarType::Int4)),
          at::ScalarType::Float),
      StatusIs(error::kPythonNotImplementedError,
               HasSubstr("casting from float32 to int4 is not supported")));

  // Shell-as-source casts that c10::canCast permits raise NotImplementedError.
  const at::Tensor float_out =
      at::empty({4}, at::TensorOptions().dtype(at::ScalarType::Float));
  EXPECT_THAT(
      ValidateOutDtype(float_out, at::ScalarType::Float4_e2m1fn_x2),
      StatusIs(
          error::kPythonNotImplementedError,
          HasSubstr(
              "casting from float4_e2m1fn_x2 to float32 is not supported")));
  EXPECT_THAT(
      ValidateOutDtype(float_out, at::ScalarType::Bits8),
      StatusIs(error::kPythonNotImplementedError,
               HasSubstr("casting from bits8 to float32 is not supported")));
  EXPECT_THAT(
      ValidateOutDtype(float_out, at::ScalarType::Int4),
      StatusIs(error::kPythonNotImplementedError,
               HasSubstr("casting from int4 to float32 is not supported")));

  // c10::canCast runs before the shell-dtype check, so casts forbidden by
  // canCast keep returning kInvalidArgument (RuntimeError).
  EXPECT_THAT(
      ValidateOutDtype(at::empty({4}, at::TensorOptions().dtype(
                                          at::ScalarType::Float4_e2m1fn_x2)),
                       at::ScalarType::ComplexFloat),
      StatusIs(error::kInvalidArgument, HasSubstr("can't be cast")));
  EXPECT_THAT(
      ValidateOutDtype(
          at::empty({4}, at::TensorOptions().dtype(at::ScalarType::Int)),
          at::ScalarType::Float4_e2m1fn_x2),
      StatusIs(error::kInvalidArgument, HasSubstr("can't be cast")));
  EXPECT_THAT(
      ValidateOutDtype(
          at::empty({4}, at::TensorOptions().dtype(at::ScalarType::Bool)),
          at::ScalarType::Float4_e2m1fn_x2),
      StatusIs(error::kInvalidArgument, HasSubstr("can't be cast")));
  EXPECT_THAT(
      ValidateOutDtype(
          at::empty({4}, at::TensorOptions().dtype(at::ScalarType::Int4)),
          at::ScalarType::ComplexFloat),
      StatusIs(error::kInvalidArgument, HasSubstr("can't be cast")));

  // allow_out_dtype_cast=false still enforces exact dtype equality first.
  EXPECT_THAT(
      ValidateOutDtype(at::empty({4}, at::TensorOptions().dtype(
                                          at::ScalarType::Float4_e2m1fn_x2)),
                       at::ScalarType::Float, /*allow_out_dtype_cast=*/false),
      StatusIs(error::kInvalidArgument,
               HasSubstr("expected the output dtype to be float32, got "
                         "float4_e2m1fn_x2")));
}

}  // namespace
}  // namespace torch_tpu
