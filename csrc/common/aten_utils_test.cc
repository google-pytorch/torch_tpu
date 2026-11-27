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

#include <cmath>
#include <cstdint>
#include <limits>

#include "ATen/core/ATen_fwd.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/DimVector.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/status_test_utils.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch/headeronly/util/BFloat16.h"
#include "torch/headeronly/util/complex.h"

namespace torch_tpu {
namespace {

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

TEST(AtenUtilsTest, CastScalarToDtypeHandlesRoundingOverflowAndWrapping) {
  EXPECT_EQ(
      CastScalarToDtype(at::Scalar(1.1), at::ScalarType::Float).toDouble(),
      static_cast<double>(static_cast<float>(1.1)));
  EXPECT_TRUE(std::isinf(
      CastScalarToDtype(at::Scalar(1e5), at::ScalarType::Half).toDouble()));
  EXPECT_GT(CastScalarToDtype(at::Scalar(1e5), at::ScalarType::Half).toDouble(),
            0.0);
  EXPECT_TRUE(std::isinf(
      CastScalarToDtype(at::Scalar(-1e5), at::ScalarType::Half).toDouble()));
  EXPECT_LT(
      CastScalarToDtype(at::Scalar(-1e5), at::ScalarType::Half).toDouble(),
      0.0);
  EXPECT_EQ(
      CastScalarToDtype(at::Scalar(1e40), at::ScalarType::Float).toDouble(),
      std::numeric_limits<double>::infinity());
  EXPECT_EQ(
      CastScalarToDtype(at::Scalar(-1e40), at::ScalarType::Float).toDouble(),
      -std::numeric_limits<double>::infinity());
  EXPECT_EQ(
      CastScalarToDtype(at::Scalar(1e40), at::ScalarType::BFloat16).toDouble(),
      std::numeric_limits<double>::infinity());
  // Values beyond the `float` range round to nearest: `FLT_MAX` below half a
  // unit in the last place, infinity from there on.
  constexpr double kFloatMax = std::numeric_limits<float>::max();
  EXPECT_EQ(
      CastScalarToDtype(at::Scalar(0x1.fffffe8p+127), at::ScalarType::Float)
          .toDouble(),
      kFloatMax);
  EXPECT_EQ(
      CastScalarToDtype(at::Scalar(-0x1.fffffe8p+127), at::ScalarType::Float)
          .toDouble(),
      -kFloatMax);
  EXPECT_EQ(
      CastScalarToDtype(at::Scalar(0x1.ffffffp+127), at::ScalarType::Float)
          .toDouble(),
      std::numeric_limits<double>::infinity());
  EXPECT_TRUE(std::isnan(
      CastScalarToDtype(at::Scalar(std::numeric_limits<double>::quiet_NaN()),
                        at::ScalarType::Half)
          .toDouble()));
  EXPECT_EQ(CastScalarToDtype(at::Scalar(int64_t{1000}), at::ScalarType::Char)
                .toLong(),
            -24);
  EXPECT_EQ(
      CastScalarToDtype(at::Scalar(int64_t{-1}), at::ScalarType::Byte).toLong(),
      255);
  EXPECT_TRUE(
      CastScalarToDtype(at::Scalar(true), at::ScalarType::Float).isBoolean());
  // `at::Scalar` stores unsigned values up to `INT64_MAX` as `int64`, so cast
  // a larger one to exercise `uint64` scalars.
  constexpr uint64_t kAboveInt64Max = (uint64_t{1} << 63) + 70000;
  ASSERT_EQ(at::Scalar(kAboveInt64Max).type(), at::ScalarType::UInt64);
  EXPECT_EQ(
      CastScalarToDtype(at::Scalar(kAboveInt64Max), at::ScalarType::UInt16)
          .toLong(),
      4464);
  EXPECT_EQ(CastScalarToDtype(at::Scalar(kAboveInt64Max), at::ScalarType::Long)
                .toLong(),
            std::numeric_limits<int64_t>::min() + 70000);
  EXPECT_EQ(CastScalarToDtype(at::Scalar(int64_t{-1}), at::ScalarType::UInt32)
                .toLong(),
            int64_t{std::numeric_limits<uint32_t>::max()});
  EXPECT_EQ(CastScalarToDtype(at::Scalar(int64_t{-1}), at::ScalarType::UInt64)
                .toUInt64(),
            std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(CastScalarToDtype(at::Scalar(int64_t{100000}), at::ScalarType::Half)
                .toDouble(),
            std::numeric_limits<double>::infinity());
  // Floating-point scalars aren't cast to integral dtypes.
  EXPECT_EQ(CastScalarToDtype(at::Scalar(1e40), at::ScalarType::Int).toDouble(),
            1e40);
  EXPECT_EQ(
      CastScalarToDtype(at::Scalar(1.1), at::ScalarType::BFloat16).toDouble(),
      static_cast<double>(c10::BFloat16(1.1)));
  EXPECT_TRUE(CastScalarToDtype(at::Scalar(c10::complex<double>(1.0, 2.0)),
                                at::ScalarType::Float)
                  .isComplex());
  EXPECT_EQ(
      CastScalarToDtype(at::Scalar(3.5), at::ScalarType::Double).toDouble(),
      3.5);
  EXPECT_EQ(
      CastScalarToDtype(at::Scalar(int64_t{5}), at::ScalarType::Bool).toLong(),
      5);
}

}  // namespace
}  // namespace torch_tpu
