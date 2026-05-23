// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/common/cache_key.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Dimname.h"
#include "ATen/core/symbol.h"
#include "ATen/ops/empty.h"
#include "absl/status/statusor.h"
#include "c10/core/ConstantSymNodeImpl.h"
#include "c10/core/Device.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/core/SymNodeImpl.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Optional.h"
#include "c10/util/intrusive_ptr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

using testing::ElementsAre;
using testing::IsEmpty;
using testing::Pair;

TEST(OpParamCacheKeys, DefaultIsEmpty) {
  OpParamCacheKeys params = OpParamCacheKeys::Empty();
  EXPECT_TRUE(params.begin() == params.end());
  EXPECT_THAT(params, IsEmpty());
  EXPECT_EQ(params.size(), 0);
}

TEST(OpParamCacheKeys, SetParamScalar) {
  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", at::Scalar(123));
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "123:Long")));
}

TEST(OpParamCacheKeys, SetParamMaybePromotedScalar_Promoted) {
  auto dummy_promoter =
      [](const at::Scalar&,
         std::optional<at::ScalarType>) -> absl::StatusOr<at::Tensor> {
    return TT_ERROR(error::kUnimplemented) << "Not implemented in test";
  };
  PromotedScalar ps(dummy_promoter, at::Scalar(5));
  MaybePromotedScalar mps(std::move(ps), ScalarValue::kZero, ScalarValue::kOne);

  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", mps);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), IsEmpty());
}

TEST(OpParamCacheKeys, SetParamMaybePromotedScalarExcludedZero) {
  auto dummy_promoter =
      [](const at::Scalar&,
         std::optional<at::ScalarType>) -> absl::StatusOr<at::Tensor> {
    return TT_ERROR(error::kUnimplemented) << "Not implemented in test";
  };
  PromotedScalar ps(dummy_promoter, at::Scalar(0));
  MaybePromotedScalar mps(std::move(ps), ScalarValue::kZero, ScalarValue::kOne);

  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", mps);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "0")));
}

TEST(OpParamCacheKeys, SetParamMaybePromotedScalarExcludedOne) {
  auto dummy_promoter =
      [](const at::Scalar&,
         std::optional<at::ScalarType>) -> absl::StatusOr<at::Tensor> {
    return TT_ERROR(error::kUnimplemented) << "Not implemented in test";
  };
  PromotedScalar ps(dummy_promoter, at::Scalar(1));
  MaybePromotedScalar mps(std::move(ps), ScalarValue::kZero, ScalarValue::kOne);

  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", mps);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "1")));
}

TEST(MaybePromotedScalar, IsZeroAndIsOne) {
  auto dummy_promoter =
      [](const at::Scalar&,
         std::optional<at::ScalarType>) -> absl::StatusOr<at::Tensor> {
    return TT_ERROR(error::kUnimplemented) << "Not implemented in test";
  };

  {
    PromotedScalar ps(dummy_promoter, at::Scalar(0));
    MaybePromotedScalar mps(std::move(ps), ScalarValue::kOne);
    EXPECT_TRUE(mps.IsZero());
    EXPECT_FALSE(mps.IsOne());
  }

  {
    PromotedScalar ps(dummy_promoter, at::Scalar(1));
    MaybePromotedScalar mps(std::move(ps), ScalarValue::kZero);
    EXPECT_FALSE(mps.IsZero());
    EXPECT_TRUE(mps.IsOne());
  }

  {
    PromotedScalar ps(dummy_promoter, at::Scalar(5));
    MaybePromotedScalar mps(std::move(ps), ScalarValue::kZero);
    EXPECT_FALSE(mps.IsZero());
    EXPECT_FALSE(mps.IsOne());
  }
}

TEST(OpParamCacheKeys, SetParamScalarType) {
  auto params_or =
      *OpParamCacheKeysBuilder().SetParam("foo", at::ScalarType::Float);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "Float")));
}

TEST(OpParamCacheKeysDeathTest, SetSameParamTwiceCrashes) {
  OpParamCacheKeys::Builder builder;
  builder.SetParam("foo", 1).SetParam("bar", 3);
  EXPECT_DEATH(builder.SetParam("foo", 2), "Duplicate parameter name 'foo'");
}

void KernelWithMaybePromotedScalar(at::Scalar s) {
  auto dummy_promoter =
      [](const at::Scalar&,
         std::optional<at::ScalarType>) -> absl::StatusOr<at::Tensor> {
    return TT_ERROR(error::kUnimplemented) << "Not implemented in test";
  };
  PromotedScalar ps(dummy_promoter, s);
  MaybePromotedScalar mps(std::move(ps), ScalarValue::kZero, ScalarValue::kOne);

  TT_KERNEL(OpName::kRelu, param_keys, (mps), {
    static_cast<void>(param_keys);
    // Do nothing
  });
}

// The "must call .GetTensor() on promoted scalar" check is only enabled in
// debug builds.
#ifndef NDEBUG
TEST(OpParamCacheKeysDeathTest, MaybePromotedScalarNotUsedCrashes) {
  EXPECT_DEATH(
      { KernelWithMaybePromotedScalar(5); },
      "The kernel didn't call \\.GetTensor\\(\\) on the promoted scalar");
}
#endif

TEST(MaybePromotedScalarDeathTest, GetTensorWhenValueMatchesExcludeCrashes) {
  auto dummy_promoter = [](const at::Scalar&, std::optional<at::ScalarType>)
      -> absl::StatusOr<at::Tensor> { return at::empty({}); };
  PromotedScalar ps(dummy_promoter, at::Scalar(0));
  MaybePromotedScalar mps(std::move(ps), ScalarValue::kZero);
  ASSERT_TRUE(mps.ValueMatchesExclude());
  EXPECT_DEATH(static_cast<void>(mps.GetTensor()), "GetTensor");
}

TEST(OpParamCacheKeys, SetParamScalarArray) {
  at::Scalar s1(123);
  at::Scalar s2(4.5);
  at::Scalar s3(true);
  at::Scalar scalars[] = {s1, s2, s3};
  auto params_or = *OpParamCacheKeysBuilder().SetParam(
      "foo", at::ArrayRef<at::Scalar>(scalars));
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(),
              ElementsAre(Pair("foo", "[123:Long,4.5:Double,1:Bool]")));

  auto params2_or =
      *OpParamCacheKeysBuilder().SetParam("foo", at::ArrayRef<at::Scalar>());
  ASSERT_TRUE(params2_or.ok());
  EXPECT_THAT(params2_or.value(), IsEmpty());
}

TEST(OpParamCacheKeys, SetParamReduceOp) {
  const c10d::ReduceOp reduce_op = c10d::ReduceOp::SUM;
  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", reduce_op);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "sum")));

  const c10d::ReduceOp reduce_op2 = c10d::ReduceOp::MAX;
  auto params2_or = *OpParamCacheKeysBuilder().SetParam("foo", reduce_op2);
  ASSERT_TRUE(params2_or.ok());
  EXPECT_THAT(params2_or.value(), ElementsAre(Pair("foo", "max")));
}

TEST(OpParamCacheKeys, SetParamMlirElementType) {
  auto params_or =
      *OpParamCacheKeysBuilder().SetParam("foo", mlir::ElementType::F32);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "f32")));
}

TEST(OpParamCacheKeys, SetParamStablehloPrecision) {
  auto params_or = *OpParamCacheKeysBuilder().SetParam(
      "foo", mlir::stablehlo::Precision::DEFAULT);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "DEFAULT")));

  auto params2_or = *OpParamCacheKeysBuilder().SetParam(
      "foo", mlir::stablehlo::Precision::HIGH);  // EXPLICIT_PRECISION_OK=unit
                                                 // test okay
  ASSERT_TRUE(params2_or.ok());
  EXPECT_THAT(params2_or.value(), ElementsAre(Pair("foo", "HIGH")));

  auto params3_or = *OpParamCacheKeysBuilder().SetParam(
      "foo",
      mlir::stablehlo::Precision::HIGHEST);  // EXPLICIT_PRECISION_OK=unit
                                             // test okay
  ASSERT_TRUE(params3_or.ok());
  EXPECT_THAT(params3_or.value(), ElementsAre(Pair("foo", "HIGHEST")));
}

TEST(OpParamCacheKeys, SetParamInteger) {
  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", 1234567890L);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "1234567890")));
}

TEST(OpParamCacheKeys, SetParamBool) {
  auto params_or = *OpParamCacheKeysBuilder().SetParam("bar", true);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("bar", "t")));

  auto params2_or = *OpParamCacheKeysBuilder().SetParam("bar", false);
  ASSERT_TRUE(params2_or.ok());
  EXPECT_THAT(params2_or.value(), ElementsAre(Pair("bar", "f")));
}

void Kernel1(int x, int y) {
  TT_KERNEL(OpName::kAdd, param_keys, (IgnoreInCacheKey(x, "testing"), y), {
    // x should be ignored in the cache keys, so only y should be there.
    EXPECT_THAT(param_keys, ElementsAre(Pair("y", "42")));
  });
}

// Verifies that TT_KERNEL() ignores the arguments marked by
// IgnoreInCacheKey() in the cache key.
TEST(OpParamCacheKeys, TtKernelIgnored) { Kernel1(9, 42); }

TEST(OpParamCacheKeys, SetParamString) {
  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", "a,bar=b");
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "a,bar=b")));

  auto params2_or = *OpParamCacheKeysBuilder().SetParam("foo", "\"\n");
  ASSERT_TRUE(params2_or.ok());
  EXPECT_THAT(params2_or.value(), ElementsAre(Pair("foo", "\"\n")));
}

TEST(OpParamCacheKeys, SetParamIntSpan) {
  auto params_or =
      *OpParamCacheKeysBuilder().SetParam("foo", Dimensions({1, 2, 3}));
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "[1,2,3]")));
}

TEST(OpParamCacheKeys, SetParamDouble) {
  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", 4.5);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "4.5")));
}

TEST(OpParamCacheKeys, SetParamNullopt) {
  const std::optional<at::Scalar> no_scalar = std::nullopt;
  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", no_scalar);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), IsEmpty());

  const std::optional<at::ScalarType> no_scalar_type = std::nullopt;
  auto params2_or = *OpParamCacheKeysBuilder().SetParam("foo", no_scalar_type);
  ASSERT_TRUE(params2_or.ok());
  EXPECT_THAT(params2_or.value(), IsEmpty());

  const std::optional<int64_t> no_int64 = std::nullopt;
  auto params3_or = *OpParamCacheKeysBuilder().SetParam("foo", no_int64);
  ASSERT_TRUE(params3_or.ok());
  EXPECT_THAT(params3_or.value(), IsEmpty());
}

TEST(OpParamCacheKeys, SetParamNewNulloptIsNoOp) {
  const std::optional<at::Scalar> no_scalar = std::nullopt;
  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", no_scalar);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), IsEmpty());
}

TEST(OpParamCacheKeys, SetParamLayout) {
  auto params_or = *OpParamCacheKeysBuilder()
                        .SetParam("foo", at::Layout::Strided)
                        .SetParam("bar", at::Layout::Sparse);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(),
              ElementsAre(Pair("bar", "Sparse"), Pair("foo", "Strided")));
}

TEST(OpParamCacheKeys, SetParamOptionalTensor) {
  c10::optional<at::Tensor> no_tensor = std::nullopt;
  c10::optional<at::Tensor> undefined_tensor = at::Tensor();
  c10::optional<at::Tensor> defined_tensor = at::empty({});
  auto params_or = *OpParamCacheKeysBuilder()
                        .SetParam("foo", no_tensor)
                        .SetParam("bar", undefined_tensor)
                        .SetParam("baz", defined_tensor);
  ASSERT_TRUE(params_or.ok());
  // foo should be omitted from the cache keys.
  // bar should be formatted as "u" to indicate the presence of an undefined
  // tensor. baz should be formatted as "t" to indicate the presence of a
  // defined tensor.
  EXPECT_THAT(params_or.value(),
              ElementsAre(Pair("bar", "u"), Pair("baz", "t")));
}

TEST(OpParamCacheKeys, SetParamMemoryFormat) {
  auto params_or = *OpParamCacheKeysBuilder()
                        .SetParam("foo", at::MemoryFormat::Contiguous)
                        .SetParam("bar", at::MemoryFormat::ChannelsLast3d);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("bar", "ChannelsLast3d"),
                                             Pair("foo", "Contiguous")));
}

TEST(OpParamCacheKeys, SetParamDimname) {
  auto params_or = *OpParamCacheKeysBuilder().SetParam(
      "foo", at::Dimname::fromSymbol(at::Symbol::dimname("N")));
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "dimname::N")));
}

TEST(OpParamCacheKeys, SetParamSymInt) {
  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", c10::SymInt(123));
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "123")));

#if defined(__has_feature) && __has_feature(hwaddress_sanitizer)
  // Skip the test as c10::SymInt pointer packing is incompatible with
  // HWASAN on ARM64.
#else
  c10::SymNode sym_node =
      c10::make_intrusive<c10::ConstantSymNodeImpl<int64_t> >(456);
  auto params2_or =
      *OpParamCacheKeysBuilder().SetParam("foo", c10::SymInt(sym_node));
  ASSERT_TRUE(params2_or.ok());
  EXPECT_THAT(params2_or.value(), ElementsAre(Pair("foo", "456")));
#endif
}

TEST(OpParamCacheKeys, SetParamSymIntArrayRef) {
#if defined(__has_feature) && __has_feature(hwaddress_sanitizer)
  // Skip the test as c10::SymInt pointer packing is incompatible with
  // HWASAN on ARM64.
#else
  c10::SymNode sym_node =
      c10::make_intrusive<c10::ConstantSymNodeImpl<int64_t> >(456);
  c10::SymInt si[] = {c10::SymInt(123), c10::SymInt(sym_node)};
  c10::SymIntArrayRef sir(si);
  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", sir);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "[123,456]")));
#endif

  c10::SymIntArrayRef empty_sym_int_array_ref;
  auto params2_or =
      *OpParamCacheKeysBuilder().SetParam("foo", empty_sym_int_array_ref);
  ASSERT_TRUE(params2_or.ok());
  EXPECT_THAT(params2_or.value(), IsEmpty());
}

TEST(OpParamCacheKeys, SetParamDevice) {
  auto params_or = *OpParamCacheKeysBuilder()
                        .SetParam("foo", at::Device("cpu"))
                        .SetParam("bar", at::Device("cuda:1"));
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(),
              ElementsAre(Pair("bar", "cuda:1"), Pair("foo", "cpu")));
}

TEST(OpParamCacheKeys, SetParamAllreduceOptions) {
  c10d::AllreduceOptions options;
  options.reduceOp = c10d::ReduceOp::SUM;
  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", options);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "sum")));
}

TEST(OpParamCacheKeys, SetParamReduceScatterOptions) {
  c10d::ReduceScatterOptions options;
  options.reduceOp = c10d::ReduceOp::PRODUCT;
  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", options);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "product")));
}

TEST(OpParamCacheKeys, SetParamBroadcastOptions) {
  c10d::BroadcastOptions options;
  options.rootRank = 1;
  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", options);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "1")));
}

TEST(OpParamCacheKeys, SetParamScatterOptions) {
  c10d::ScatterOptions options;
  options.rootRank = 2;
  auto params_or = *OpParamCacheKeysBuilder().SetParam("foo", options);
  ASSERT_TRUE(params_or.ok());
  EXPECT_THAT(params_or.value(), ElementsAre(Pair("foo", "2")));
}

}  // namespace
}  // namespace torch_tpu
