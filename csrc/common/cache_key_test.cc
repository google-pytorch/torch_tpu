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

#include "csrc/common/cache_key.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "ATen/core/ATen_fwd.h"
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
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fingerprint_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
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
  const at::Scalar scalar(123);
  TT_ASSERT_OK_AND_ASSIGN(auto params,
                          *OpParamCacheKeysBuilder().SetParam("foo", scalar));
  EXPECT_THAT(params, ElementsAre(Pair("foo", FingerprintCat("int64", 123))));
}

TEST(OpParamCacheKeys, SetParamMaybePromotedScalar_Promoted) {
  auto dummy_promoter =
      [](const at::Scalar&,
         std::optional<at::ScalarType>) -> absl::StatusOr<at::Tensor> {
    return TT_ERROR(error::kPythonNotImplementedError)
           << "Not implemented in test";
  };
  PromotedScalar ps(dummy_promoter, at::Scalar(5));
  MaybePromotedScalar mps(std::move(ps), ScalarValue::kZero, ScalarValue::kOne);

  TT_ASSERT_OK_AND_ASSIGN(auto params,
                          *OpParamCacheKeysBuilder().SetParam("foo", mps));
  EXPECT_THAT(params, IsEmpty());
}

TEST(OpParamCacheKeys, SetParamMaybePromotedScalarExcludedZero) {
  auto dummy_promoter =
      [](const at::Scalar&,
         std::optional<at::ScalarType>) -> absl::StatusOr<at::Tensor> {
    return TT_ERROR(error::kPythonNotImplementedError)
           << "Not implemented in test";
  };
  PromotedScalar ps(dummy_promoter, at::Scalar(0));
  MaybePromotedScalar mps(std::move(ps), ScalarValue::kZero, ScalarValue::kOne);

  TT_ASSERT_OK_AND_ASSIGN(auto params,
                          *OpParamCacheKeysBuilder().SetParam("foo", mps));
  EXPECT_THAT(params, ElementsAre(Pair("foo", Fingerprint("0"))));
}

TEST(OpParamCacheKeys, SetParamMaybePromotedScalarExcludedOne) {
  auto dummy_promoter =
      [](const at::Scalar&,
         std::optional<at::ScalarType>) -> absl::StatusOr<at::Tensor> {
    return TT_ERROR(error::kPythonNotImplementedError)
           << "Not implemented in test";
  };
  PromotedScalar ps(dummy_promoter, at::Scalar(1));
  MaybePromotedScalar mps(std::move(ps), ScalarValue::kZero, ScalarValue::kOne);

  TT_ASSERT_OK_AND_ASSIGN(auto params,
                          *OpParamCacheKeysBuilder().SetParam("foo", mps));
  EXPECT_THAT(params, ElementsAre(Pair("foo", Fingerprint("1"))));
}

TEST(OpParamCacheKeys, SetParamMaybePromotedScalarExcludedTwo) {
  auto dummy_promoter =
      [](const at::Scalar&,
         std::optional<at::ScalarType>) -> absl::StatusOr<at::Tensor> {
    return TT_ERROR(error::kPythonNotImplementedError)
           << "Not implemented in test";
  };
  PromotedScalar ps(dummy_promoter, at::Scalar(2.0));
  MaybePromotedScalar mps(std::move(ps), ScalarValue::kTwo);

  TT_ASSERT_OK_AND_ASSIGN(auto params,
                          *OpParamCacheKeysBuilder().SetParam("foo", mps));
  EXPECT_THAT(params, ElementsAre(Pair("foo", Fingerprint("2"))));
}

TEST(MaybePromotedScalar, IsZeroAndIsOneAndIsTwo) {
  auto dummy_promoter =
      [](const at::Scalar&,
         std::optional<at::ScalarType>) -> absl::StatusOr<at::Tensor> {
    return TT_ERROR(error::kPythonNotImplementedError)
           << "Not implemented in test";
  };

  {
    PromotedScalar ps(dummy_promoter, at::Scalar(0));
    MaybePromotedScalar mps(std::move(ps), ScalarValue::kOne);
    EXPECT_TRUE(mps.IsZero());
    EXPECT_FALSE(mps.IsOne());
    EXPECT_FALSE(mps.IsTwo());
  }

  {
    PromotedScalar ps(dummy_promoter, at::Scalar(1));
    MaybePromotedScalar mps(std::move(ps), ScalarValue::kZero);
    EXPECT_FALSE(mps.IsZero());
    EXPECT_TRUE(mps.IsOne());
    EXPECT_FALSE(mps.IsTwo());
  }

  {
    PromotedScalar ps(dummy_promoter, at::Scalar(2));
    MaybePromotedScalar mps(std::move(ps), ScalarValue::kTwo);
    EXPECT_FALSE(mps.IsZero());
    EXPECT_FALSE(mps.IsOne());
    EXPECT_TRUE(mps.IsTwo());
  }

  {
    PromotedScalar ps(dummy_promoter, at::Scalar(5));
    MaybePromotedScalar mps(std::move(ps), ScalarValue::kZero);
    EXPECT_FALSE(mps.IsZero());
    EXPECT_FALSE(mps.IsOne());
    EXPECT_FALSE(mps.IsTwo());
  }
}

TEST(OpParamCacheKeys, SetParamScalarType) {
  TT_ASSERT_OK_AND_ASSIGN(auto params, *OpParamCacheKeysBuilder().SetParam(
                                           "foo", at::ScalarType::Float));
  EXPECT_THAT(params, ElementsAre(Pair("foo", Fingerprint("float32"))));
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
    return TT_ERROR(error::kPythonNotImplementedError)
           << "Not implemented in test";
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
  // Empty array should be omitted from the cache keys.
  TT_ASSERT_OK_AND_ASSIGN(auto params0, *OpParamCacheKeysBuilder().SetParam(
                                            "foo", at::ArrayRef<at::Scalar>()));
  EXPECT_THAT(params0, IsEmpty());

  const at::Scalar s1(123);
  const at::Scalar s2(4.5);
  const at::Scalar s3(true);
  const at::Scalar scalars1[] = {s1};
  TT_ASSERT_OK_AND_ASSIGN(auto params1,
                          *OpParamCacheKeysBuilder().SetParam(
                              "foo", at::ArrayRef<at::Scalar>(scalars1)));
  EXPECT_THAT(params1,
              ElementsAre(Pair(
                  "foo", FingerprintCat(
                             "", internal::EncodeParamCacheKey(s1).value()))));

  const at::Scalar scalars2[] = {s1, s2};
  TT_ASSERT_OK_AND_ASSIGN(auto params2,
                          *OpParamCacheKeysBuilder().SetParam(
                              "foo", at::ArrayRef<at::Scalar>(scalars2)));
  EXPECT_THAT(
      params2,
      ElementsAre(Pair(
          "foo", FingerprintCat("", internal::EncodeParamCacheKey(s1).value(),
                                internal::EncodeParamCacheKey(s2).value()))));

  const at::Scalar scalars3[] = {s1, s2, s3};
  TT_ASSERT_OK_AND_ASSIGN(auto params3,
                          *OpParamCacheKeysBuilder().SetParam(
                              "foo", at::ArrayRef<at::Scalar>(scalars3)));
  EXPECT_THAT(
      params3,
      ElementsAre(Pair(
          "foo", FingerprintCat("", internal::EncodeParamCacheKey(s1).value(),
                                internal::EncodeParamCacheKey(s2).value(),
                                internal::EncodeParamCacheKey(s3).value()))));
}

TEST(OpParamCacheKeys, SetParamReduceOp) {
  const c10d::ReduceOp reduce_op = c10d::ReduceOp::SUM;
  TT_ASSERT_OK_AND_ASSIGN(
      auto params, *OpParamCacheKeysBuilder().SetParam("foo", reduce_op));
  EXPECT_THAT(params, ElementsAre(Pair("foo", Fingerprint("sum"))));

  const c10d::ReduceOp reduce_op2 = c10d::ReduceOp::MAX;
  TT_ASSERT_OK_AND_ASSIGN(
      auto params2, *OpParamCacheKeysBuilder().SetParam("foo", reduce_op2));
  EXPECT_THAT(params2, ElementsAre(Pair("foo", Fingerprint("max"))));
}

TEST(OpParamCacheKeys, SetParamMlirElementType) {
  TT_ASSERT_OK_AND_ASSIGN(auto params, *OpParamCacheKeysBuilder().SetParam(
                                           "foo", mlir::ElementType::F32));
  EXPECT_THAT(params, ElementsAre(Pair("foo", Fingerprint("f32"))));
}

TEST(OpParamCacheKeys, SetParamStablehloPrecision) {
  TT_ASSERT_OK_AND_ASSIGN(
      auto params,
      *OpParamCacheKeysBuilder().SetParam(
          "foo",
          mlir::stablehlo::Precision::DEFAULT));  // EXPLICIT_PRECISION_OK=unit
                                                  // test okay
  EXPECT_THAT(params, ElementsAre(Pair("foo", Fingerprint("DEFAULT"))));

  TT_ASSERT_OK_AND_ASSIGN(
      auto params2,
      *OpParamCacheKeysBuilder().SetParam(
          "foo",
          mlir::stablehlo::Precision::HIGH));  // EXPLICIT_PRECISION_OK=unit
                                               // test okay
  EXPECT_THAT(params2, ElementsAre(Pair("foo", Fingerprint("HIGH"))));

  TT_ASSERT_OK_AND_ASSIGN(
      auto params3,
      *OpParamCacheKeysBuilder().SetParam(
          "foo",
          mlir::stablehlo::Precision::HIGHEST));  // EXPLICIT_PRECISION_OK=unit
                                                  // test okay
  EXPECT_THAT(params3, ElementsAre(Pair("foo", Fingerprint("HIGHEST"))));
}

TEST(OpParamCacheKeys, SetParamInteger) {
  TT_ASSERT_OK_AND_ASSIGN(
      auto params, *OpParamCacheKeysBuilder().SetParam("foo", 1234567890L));
  EXPECT_THAT(params, ElementsAre(Pair("foo", 1234567890UL)));
}

TEST(OpParamCacheKeys, SetParamBool) {
  TT_ASSERT_OK_AND_ASSIGN(auto params,
                          *OpParamCacheKeysBuilder().SetParam("bar", true));
  EXPECT_THAT(params, ElementsAre(Pair("bar", 1)));

  TT_ASSERT_OK_AND_ASSIGN(auto params2,
                          *OpParamCacheKeysBuilder().SetParam("bar", false));
  EXPECT_THAT(params2, ElementsAre(Pair("bar", 0)));
}

void Kernel1(int x, int y) {
  TT_KERNEL(OpName::kAdd, param_keys, (IgnoreInCacheKey(x, "testing"), y), {
    // x should be ignored in the cache keys, so only y should be there.
    EXPECT_THAT(param_keys, ElementsAre(Pair("y", 42)));
  });
}

// Verifies that TT_KERNEL() ignores the arguments marked by
// IgnoreInCacheKey() in the cache key.
TEST(OpParamCacheKeys, TtKernelIgnored) { Kernel1(9, 42); }

TEST(OpParamCacheKeys, SetParamString) {
  TT_ASSERT_OK_AND_ASSIGN(
      auto params, *OpParamCacheKeysBuilder().SetParam("foo", "a,bar=b"));
  EXPECT_THAT(params, ElementsAre(Pair("foo", Fingerprint("a,bar=b"))));

  TT_ASSERT_OK_AND_ASSIGN(auto params2,
                          *OpParamCacheKeysBuilder().SetParam("foo", "\"\n"));
  EXPECT_THAT(params2, ElementsAre(Pair("foo", Fingerprint("\"\n"))));
}

TEST(OpParamCacheKeys, SetParamIntSpan) {
  TT_ASSERT_OK_AND_ASSIGN(auto params, *OpParamCacheKeysBuilder().SetParam(
                                           "foo", Dimensions({1, 2, 3})));
  EXPECT_THAT(params, ElementsAre(Pair("foo", FingerprintCat("", 1L, 2L, 3L))));
}

TEST(OpParamCacheKeys, SetParamDouble) {
  TT_ASSERT_OK_AND_ASSIGN(auto params,
                          *OpParamCacheKeysBuilder().SetParam("foo", 4.5));
  EXPECT_THAT(params,
              ElementsAre(Pair("foo", absl::bit_cast<FingerprintType>(4.5))));
}

TEST(OpParamCacheKeys, SetParamNullopt) {
  const std::optional<at::Scalar> no_scalar = std::nullopt;
  TT_ASSERT_OK_AND_ASSIGN(
      auto params, *OpParamCacheKeysBuilder().SetParam("foo", no_scalar));
  EXPECT_THAT(params, IsEmpty());

  const std::optional<at::ScalarType> no_scalar_type = std::nullopt;
  TT_ASSERT_OK_AND_ASSIGN(
      auto params2, *OpParamCacheKeysBuilder().SetParam("foo", no_scalar_type));
  EXPECT_THAT(params2, IsEmpty());

  const std::optional<int64_t> no_int64 = std::nullopt;
  TT_ASSERT_OK_AND_ASSIGN(auto params3,
                          *OpParamCacheKeysBuilder().SetParam("foo", no_int64));
  EXPECT_THAT(params3, IsEmpty());
}

TEST(OpParamCacheKeys, SetParamNewNulloptIsNoOp) {
  const std::optional<at::Scalar> no_scalar = std::nullopt;
  TT_ASSERT_OK_AND_ASSIGN(
      auto params, *OpParamCacheKeysBuilder().SetParam("foo", no_scalar));
  EXPECT_THAT(params, IsEmpty());
}

TEST(OpParamCacheKeys, SetParamLayout) {
  TT_ASSERT_OK_AND_ASSIGN(auto params,
                          *OpParamCacheKeysBuilder()
                               .SetParam("foo", at::Layout::Strided)
                               .SetParam("bar", at::Layout::Sparse));
  EXPECT_THAT(params, ElementsAre(Pair("bar", Fingerprint("Sparse")),
                                  Pair("foo", Fingerprint("Strided"))));
}

TEST(OpParamCacheKeys, SetParamOptionalTensor) {
  c10::optional<at::Tensor> no_tensor = std::nullopt;
  c10::optional<at::Tensor> undefined_tensor = at::Tensor();
  c10::optional<at::Tensor> defined_tensor = at::empty({});
  TT_ASSERT_OK_AND_ASSIGN(auto params, *OpParamCacheKeysBuilder()
                                            .SetParam("foo", no_tensor)
                                            .SetParam("bar", undefined_tensor)
                                            .SetParam("baz", defined_tensor));
  // Both foo and bar should be omitted from the cache keys.
  // baz should be formatted as "t" to indicate the presence of a
  // defined tensor.
  EXPECT_THAT(params, ElementsAre(Pair("baz", Fingerprint("t"))));
}

TEST(OpParamCacheKeys, SetParamMemoryFormat) {
  TT_ASSERT_OK_AND_ASSIGN(
      auto params, *OpParamCacheKeysBuilder()
                        .SetParam("foo", at::MemoryFormat::Contiguous)
                        .SetParam("bar", at::MemoryFormat::ChannelsLast3d));
  EXPECT_THAT(params, ElementsAre(Pair("bar", Fingerprint("ChannelsLast3d")),
                                  Pair("foo", Fingerprint("Contiguous"))));
}

TEST(OpParamCacheKeys, SetParamSymInt) {
  TT_ASSERT_OK_AND_ASSIGN(auto params, *OpParamCacheKeysBuilder().SetParam(
                                           "foo", c10::SymInt(123)));
  EXPECT_THAT(params, ElementsAre(Pair("foo", Fingerprint("123"))));

#if defined(__has_feature) && __has_feature(hwaddress_sanitizer)
  // Skip the test as c10::SymInt pointer packing is incompatible with
  // HWASAN on ARM64.
#else
  c10::SymNode sym_node =
      c10::make_intrusive<c10::ConstantSymNodeImpl<int64_t> >(456);
  TT_ASSERT_OK_AND_ASSIGN(auto params2, *OpParamCacheKeysBuilder().SetParam(
                                            "foo", c10::SymInt(sym_node)));
  EXPECT_THAT(params2, ElementsAre(Pair("foo", Fingerprint("456"))));
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
  TT_ASSERT_OK_AND_ASSIGN(auto params,
                          *OpParamCacheKeysBuilder().SetParam("foo", sir));
  EXPECT_THAT(params,
              ElementsAre(Pair("foo", FingerprintCat("", "123", "456"))));
#endif

  c10::SymIntArrayRef empty_sym_int_array_ref;
  TT_ASSERT_OK_AND_ASSIGN(auto params2, *OpParamCacheKeysBuilder().SetParam(
                                            "foo", empty_sym_int_array_ref));
  EXPECT_THAT(params2, IsEmpty());
}

TEST(OpParamCacheKeys, SetParamDevice) {
  TT_ASSERT_OK_AND_ASSIGN(auto params,
                          *OpParamCacheKeysBuilder()
                               .SetParam("foo", at::Device("cpu"))
                               .SetParam("bar", at::Device("cuda:1")));
  EXPECT_THAT(params, ElementsAre(Pair("bar", Fingerprint("cuda:1")),
                                  Pair("foo", Fingerprint("cpu"))));
}

TEST(OpParamCacheKeys, SetParamAllreduceOptions) {
  c10d::AllreduceOptions options;
  options.reduceOp = c10d::ReduceOp::SUM;
  TT_ASSERT_OK_AND_ASSIGN(auto params,
                          *OpParamCacheKeysBuilder().SetParam("foo", options));
  EXPECT_THAT(params, ElementsAre(Pair("foo", Fingerprint("sum"))));
}

TEST(OpParamCacheKeys, SetParamReduceScatterOptions) {
  c10d::ReduceScatterOptions options;
  options.reduceOp = c10d::ReduceOp::PRODUCT;
  TT_ASSERT_OK_AND_ASSIGN(auto params,
                          *OpParamCacheKeysBuilder().SetParam("foo", options));
  EXPECT_THAT(params, ElementsAre(Pair("foo", Fingerprint("product"))));
}

TEST(OpParamCacheKeys, SetParamBroadcastOptions) {
  c10d::BroadcastOptions options;
  options.rootRank = 1;
  TT_ASSERT_OK_AND_ASSIGN(auto params,
                          *OpParamCacheKeysBuilder().SetParam("foo", options));
  EXPECT_THAT(params, ElementsAre(Pair("foo", 1)));
}

TEST(OpParamCacheKeys, SetParamScatterOptions) {
  c10d::ScatterOptions options;
  options.rootRank = 2;
  TT_ASSERT_OK_AND_ASSIGN(auto params,
                          *OpParamCacheKeysBuilder().SetParam("foo", options));
  EXPECT_THAT(params, ElementsAre(Pair("foo", 2)));
}

TEST(DynamicPadModuleCacheKey, DistinguishesShapesBoundsDtypesAndPadModule) {
  Shape base({4, 16}, mlir::ElementType::I64);
  base.dynamic_dimensions().push_back(BoundedDynamicDimension{
      .dimension = 1, .lower_bound = 2, .upper_bound = 64});

  Shape same = base;
  EXPECT_EQ(DynamicPadModuleCacheKey({base}), DynamicPadModuleCacheKey({same}));
  EXPECT_NE(DynamicPadModuleCacheKey({base}),
            PadModuleCacheKey({base}, /*pad_only_module=*/true));
  EXPECT_NE(DynamicPadModuleCacheKey({base}),
            PadModuleCacheKey({base}, /*pad_only_module=*/false));

  Shape wider_bound({4, 16}, mlir::ElementType::I64);
  wider_bound.dynamic_dimensions().push_back(BoundedDynamicDimension{
      .dimension = 1, .lower_bound = 2, .upper_bound = 128});
  EXPECT_NE(DynamicPadModuleCacheKey({base}),
            DynamicPadModuleCacheKey({wider_bound}));

  Shape diff_input_size({4, 32}, mlir::ElementType::I64);
  diff_input_size.dynamic_dimensions().push_back(BoundedDynamicDimension{
      .dimension = 1, .lower_bound = 2, .upper_bound = 64});
  EXPECT_NE(DynamicPadModuleCacheKey({base}),
            DynamicPadModuleCacheKey({diff_input_size}));

  Shape diff_dtype({4, 16}, mlir::ElementType::I32);
  diff_dtype.dynamic_dimensions().push_back(BoundedDynamicDimension{
      .dimension = 1, .lower_bound = 2, .upper_bound = 64});
  EXPECT_NE(DynamicPadModuleCacheKey({base}),
            DynamicPadModuleCacheKey({diff_dtype}));

  // Same static and padded dimensions ({4, 64}), different dynamic axis.
  Shape at_bound_dim1({4, 64}, mlir::ElementType::I64);
  at_bound_dim1.dynamic_dimensions().push_back(BoundedDynamicDimension{
      .dimension = 1, .lower_bound = 2, .upper_bound = 64});
  Shape at_bound_dim0({4, 64}, mlir::ElementType::I64);
  at_bound_dim0.dynamic_dimensions().push_back(BoundedDynamicDimension{
      .dimension = 0, .lower_bound = 2, .upper_bound = 4});
  EXPECT_NE(DynamicPadModuleCacheKey({at_bound_dim1}),
            DynamicPadModuleCacheKey({at_bound_dim0}));

  // A static passthrough shape differs from a dynamic shape already at its
  // upper bound, even though both have the same static dimensions.
  Shape static_passthrough({4, 64}, mlir::ElementType::I64);
  EXPECT_NE(DynamicPadModuleCacheKey({static_passthrough}),
            DynamicPadModuleCacheKey({at_bound_dim1}));

  // Tensor order matters: the key must follow the argument order.
  EXPECT_EQ(DynamicPadModuleCacheKey({base, diff_dtype}),
            DynamicPadModuleCacheKey({base, diff_dtype}));
  EXPECT_NE(DynamicPadModuleCacheKey({base, diff_dtype}),
            DynamicPadModuleCacheKey({diff_dtype, base}));

  // Two dynamic dimensions on one tensor differ from either single-dimension
  // variant and from each other when the bounds are swapped.
  Shape two_dynamic_dims({4, 8, 16}, mlir::ElementType::I64);
  two_dynamic_dims.dynamic_dimensions().push_back(BoundedDynamicDimension{
      .dimension = 0, .lower_bound = 2, .upper_bound = 8});
  two_dynamic_dims.dynamic_dimensions().push_back(BoundedDynamicDimension{
      .dimension = 2, .lower_bound = 2, .upper_bound = 64});
  Shape only_dim0({4, 8, 16}, mlir::ElementType::I64);
  only_dim0.dynamic_dimensions().push_back(BoundedDynamicDimension{
      .dimension = 0, .lower_bound = 2, .upper_bound = 8});
  Shape only_dim2({4, 8, 16}, mlir::ElementType::I64);
  only_dim2.dynamic_dimensions().push_back(BoundedDynamicDimension{
      .dimension = 2, .lower_bound = 2, .upper_bound = 64});
  Shape swapped_bounds({4, 8, 16}, mlir::ElementType::I64);
  swapped_bounds.dynamic_dimensions().push_back(BoundedDynamicDimension{
      .dimension = 0, .lower_bound = 2, .upper_bound = 64});
  swapped_bounds.dynamic_dimensions().push_back(BoundedDynamicDimension{
      .dimension = 2, .lower_bound = 2, .upper_bound = 8});
  Shape two_dynamic_dims_copy = two_dynamic_dims;
  EXPECT_EQ(DynamicPadModuleCacheKey({two_dynamic_dims}),
            DynamicPadModuleCacheKey({two_dynamic_dims_copy}));
  EXPECT_NE(DynamicPadModuleCacheKey({two_dynamic_dims}),
            DynamicPadModuleCacheKey({only_dim0}));
  EXPECT_NE(DynamicPadModuleCacheKey({two_dynamic_dims}),
            DynamicPadModuleCacheKey({only_dim2}));
  EXPECT_NE(DynamicPadModuleCacheKey({two_dynamic_dims}),
            DynamicPadModuleCacheKey({swapped_bounds}));
}

}  // namespace
}  // namespace torch_tpu
