/*
 * Copyright 2025 Google LLC
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

#include "torch_tpu/ops/view_decomposition/view_sequence.h"

#include <cstdint>
#include <string>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/view_decomposition/bitcast_primitive.h"
#include "torch_tpu/ops/view_decomposition/broadcast_primitive.h"
#include "torch_tpu/ops/view_decomposition/conj_primitive.h"
#include "torch_tpu/ops/view_decomposition/pad_primitive.h"
#include "torch_tpu/ops/view_decomposition/reshape_primitive.h"
#include "torch_tpu/ops/view_decomposition/slice_primitive.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/transpose_primitive.h"
#include "torch_tpu/ops/view_decomposition/unfold_primitive.h"
#include "xla/tsl/platform/statusor.h"

namespace torch_tpu {
namespace {

using testing::ElementsAre;
using testing::Pair;

void SimplifyTest(absl::Span<const int64_t> contiguous_base_shape,
                  ViewSequence sequence, ViewSequenceSpan expected,
                  bool expect_physical_layout_change = true) {
  const auto unmodified_layout =
      MakeContiguousBaseLayout(contiguous_base_shape);

  // The simplified sequence should return the same final layout as the
  // original un-simplified sequence.
  auto expected_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  UpdateLayout(expected_layout, sequence);

  // If the simplified sequence is empty, then the simplified sequence should
  // not update the layout; otherwise, there should be a meaningful update.
  const bool expected_updated = !expected.empty();

  Simplify(sequence, contiguous_base_shape);
  EXPECT_EQ(sequence, expected);

  auto actual_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  auto updated_status = UpdateLayout(actual_layout, sequence);
  EXPECT_EQ(updated_status, expected_updated);
  EXPECT_EQ(actual_layout, expected_layout);

  // After simplification, a non-empty sequence should not be a net no-op,
  // while an empty sequence should be a no-op.
  const bool layout_changed = actual_layout != unmodified_layout;
  if (expect_physical_layout_change) {
    EXPECT_EQ(layout_changed, expected_updated);
  } else {
    EXPECT_EQ(layout_changed, false);
  }
}

TEST(Simplify, EmptyList) { SimplifyTest({54}, {}, {}); }

TEST(Simplify, NothingToSimplify) {
  Dimensions contiguous_base_shape = {54};
  // Equivalent to:
  // torch.arange(54).view(6, 9)[:, 1:].view(
  //   6, 4, 2, 1).permute(0, 3, 2, 1).expand(6, 999, 2, 4)
  // No no-ops, nothing to merge
  ViewSequence sequence = {
      ReshapePrimitive{.base_sizes = {54}, .new_sizes = {6, 9}},
      SlicePrimitive{
          .slice_dims = {{.start_index = 0, .limit_index = 6, .stride = 1},
                         {.start_index = 1, .limit_index = 9, .stride = 1}}},
      ReshapePrimitive{.base_sizes = {6, 8}, .new_sizes = {6, 4, 2, 1}},
      TransposePrimitive{.permutation = {0, 3, 2, 1}},
      BroadcastPrimitive{.base_shape = {6, 1, 2, 4},
                         .new_sizes = {6, 999, 2, 4},
                         .broadcast_dimensions = {0, 1, 2, 3}}};
  ViewSequence expected = sequence;  // Should not be modified

  SimplifyTest(contiguous_base_shape, std::move(sequence), std::move(expected));
}

TEST(Simplify, RemoveNoOpReshape) {
  SimplifyTest({6, 9},
               {ReshapePrimitive{.base_sizes = {6, 9}, .new_sizes = {6, 9}}},
               {});
}

TEST(Simplify, RemoveNoOpPermute) {
  SimplifyTest({1, 2, 3, 4}, {TransposePrimitive{.permutation = {0, 1, 2, 3}}},
               {});
}

TEST(Simplify, RemoveNoOpBroadcast) {
  SimplifyTest({6, 2, 4},
               {BroadcastPrimitive{.base_shape = {6, 2, 4},
                                   .new_sizes = {6, 2, 4},
                                   .broadcast_dimensions = {0, 1, 2}}},
               {});
}

TEST(Simplify, RemoveNoOpSlice) {
  SimplifyTest(
      {2, 3, 4},
      {SlicePrimitive{
          .slice_dims = {{.start_index = 0, .limit_index = 2, .stride = 1},
                         {.start_index = 0, .limit_index = 3, .stride = 1},
                         {.start_index = 0, .limit_index = 4, .stride = 1}}}},
      {});
}

TEST(Simplify, RemoveNoOpPad) {
  SimplifyTest(
      {6, 9},
      {PadPrimitive{
          .pad_dims =
              {{.low_padding = 0, .high_padding = 0, .interior_padding = 0},
               {.low_padding = 0, .high_padding = 0, .interior_padding = 0}}}},
      {});
}

TEST(Simplify, RemoveNoOpBitcast) {
  SimplifyTest({6, 9},
               {RealToRealBitcast{.from_type = mlir::ElementType::F32,
                                  .to_type = mlir::ElementType::F32}},
               {});
}

TEST(Simplify, RemoveNoOpUnfold) {
  SimplifyTest({6, 1, 9},
               {UnfoldPrimitive{.start_index = 0,
                                .limit_index = 9,
                                .window_stride = 999,
                                .window_size = 9}},
               {});
}

// Can't use SimplifyTest because it should be a non-no-op operation that
// doesn't change the layout.
TEST(Simplify, DoNotRemoveSameSizeBitcast) {
  const Dimensions contiguous_base_shape = {6, 9};
  ViewSequence sequence = {RealToRealBitcast{
      .from_type = mlir::ElementType::F32, .to_type = mlir::ElementType::UI32}};
  const ViewSequence expected = sequence;

  const auto unmodified_layout =
      MakeContiguousBaseLayout(contiguous_base_shape);

  auto expected_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  UpdateLayout(expected_layout, sequence);

  // Same-size bitcasts to a new type are not no-ops. They should not be
  // removed by Simplify.
  const bool expected_updated = true;

  Simplify(sequence, contiguous_base_shape);
  EXPECT_EQ(sequence, expected);

  // There is no layout change, even though the sequence is not a no-op,
  // because the values would be reinterpreted.
  auto actual_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  auto updated_status = UpdateLayout(actual_layout, sequence);
  EXPECT_EQ(updated_status, expected_updated);
  EXPECT_EQ(actual_layout, unmodified_layout);
}

TEST(Simplify, MergedReshape) {
  SimplifyTest({54},
               {ReshapePrimitive{.base_sizes = {54}, .new_sizes = {27, 2}},
                ReshapePrimitive{.base_sizes = {27, 2}, .new_sizes = {6, 9}}},
               {ReshapePrimitive{.base_sizes = {54}, .new_sizes = {6, 9}}});
}

TEST(Simplify, MergedSlice) {
  SimplifyTest(
      {6, 9},
      {SlicePrimitive{
           .slice_dims = {{.start_index = 0, .limit_index = 6, .stride = 1},
                          {.start_index = 1, .limit_index = 9, .stride = 1}}},
       SlicePrimitive{
           .slice_dims = {{.start_index = 0, .limit_index = 6, .stride = 2},
                          {.start_index = 0, .limit_index = 8, .stride = 1}}}},
      {SlicePrimitive{
          .slice_dims = {{.start_index = 0, .limit_index = 6, .stride = 2},
                         {.start_index = 1, .limit_index = 9, .stride = 1}}}});
}

TEST(Simplify, MergedReshapeAndSlice) {
  SimplifyTest({54},
               {
                   ReshapePrimitive{.base_sizes = {54}, .new_sizes = {27, 2}},
                   ReshapePrimitive{.base_sizes = {27, 2}, .new_sizes = {6, 9}},
                   SlicePrimitive{
                       .slice_dims =
                           {{.start_index = 0, .limit_index = 6, .stride = 1},
                            {.start_index = 0, .limit_index = 9, .stride = 3}}},
                   SlicePrimitive{
                       .slice_dims =
                           {{.start_index = 0, .limit_index = 6, .stride = 2},
                            {.start_index = 0, .limit_index = 3, .stride = 1}}},
               },
               {
                   ReshapePrimitive{.base_sizes = {54}, .new_sizes = {6, 9}},
                   SlicePrimitive{
                       .slice_dims =
                           {{.start_index = 0, .limit_index = 6, .stride = 2},
                            {.start_index = 0, .limit_index = 9, .stride = 3}}},
               });
}

TEST(Simplify, RecursiveSimplify) {
  // The slice and transpose are no ops that get removed.
  // Then we have 3 sequential reshapes that get merged.
  // The final remaining reshape is redundant and gets removed.
  SimplifyTest(
      {54},
      {ReshapePrimitive{.base_sizes = {54}, .new_sizes = {27, 2}},
       SlicePrimitive{
           .slice_dims = {{.start_index = 0, .limit_index = 27, .stride = 1},
                          {.start_index = 0, .limit_index = 2, .stride = 1}}},
       ReshapePrimitive{.base_sizes = {27, 2}, .new_sizes = {6, 9}},
       TransposePrimitive{.permutation = {0, 1}},
       ReshapePrimitive{.base_sizes = {6, 9}, .new_sizes = {54}}},
      {});
}

TEST(Simplify, ConjReal) {
  // ConjPrimitive isn't created for real types by decomposition. If one
  // manually creates it, it's treated as active is_set=true, and it will NOT be
  // simplified away. This is fine since conjugate on real is a no-op anyway.
  SimplifyTest({6, 9}, {ConjPrimitive{}}, {ConjPrimitive{}},
               /*expect_physical_layout_change=*/false);
}

TEST(Simplify, ConjComplex) {
  // Conjugate on complex is preserved.
  SimplifyTest({6, 9}, {ConjPrimitive{}}, {ConjPrimitive{}},
               /*expect_physical_layout_change=*/false);
}

TEST(Simplify, DoubleConj) {
  // Two conjugates on complex are merged into a no-op (identity) and removed.
  SimplifyTest({6, 9},
               {
                   ConjPrimitive{},
                   ConjPrimitive{},
               },
               {},
               /*expect_physical_layout_change=*/false);
}

TEST(Simplify, ReshapeConj) {
  // Reshape then Conj.
  SimplifyTest({54},
               {
                   ReshapePrimitive{.base_sizes = {54}, .new_sizes = {6, 9}},
                   ConjPrimitive{},
               },
               {
                   ReshapePrimitive{.base_sizes = {54}, .new_sizes = {6, 9}},
                   ConjPrimitive{},
               });
}

TEST(ViewPrimitiveEquality, ConjPrimitive) {
  ViewPrimitive c1 = ConjPrimitive{.is_set = true};
  ViewPrimitive c2 = ConjPrimitive{.is_set = true};
  ViewPrimitive c3 = ConjPrimitive{.is_set = false};

  EXPECT_EQ(c1, c2);
  EXPECT_NE(c1, c3);
}

TEST(ViewPrimitiveEquality, BroadcastPrimitive) {
  ViewPrimitive b1 = BroadcastPrimitive{.base_shape = {1, 2},
                                        .new_sizes = {2, 2},
                                        .broadcast_dimensions = {0, 1}};
  ViewPrimitive b2 = BroadcastPrimitive{.base_shape = {1, 2},
                                        .new_sizes = {2, 2},
                                        .broadcast_dimensions = {0, 1}};
  ViewPrimitive b3 = BroadcastPrimitive{.base_shape = {2, 2},
                                        .new_sizes = {2, 2},
                                        .broadcast_dimensions = {0, 1}};

  EXPECT_EQ(b1, b2);
  EXPECT_NE(b1, b3);
}

TEST(SymbolicViewPrimitive, ReshapeViewCacheKeys) {
  // Create a fallback tensor which is only used when symbolic keygen fails
  at::Tensor tensor = at::empty({4, 6}).reshape({6, 4});

  auto param_keys = OpParamCacheKeys::Empty();
  ViewSequence flatten = {
      ReshapePrimitive{.base_sizes = {2, 2}, .new_sizes = {4}}};
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys, ViewSequenceCacheKey(flatten, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys, ElementsAre(Pair("view", "reshape:flatten")));

  ViewSequence collapse = {
      ReshapePrimitive{.base_sizes = {2, 3, 4}, .new_sizes = {6, 4}}};
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys,
      ViewSequenceCacheKey(collapse, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys,
              ElementsAre(Pair("view", "reshape:collapse{0,1},{2}")));

  ViewSequence squeeze = {
      ReshapePrimitive{.base_sizes = {1, 4, 1, 4, 1}, .new_sizes = {4, 4}}};
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys, ViewSequenceCacheKey(squeeze, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys,
              ElementsAre(Pair("view", "reshape:collapse{0,1},{2,3,4}")));

  ViewSequence unsqueeze = {
      ReshapePrimitive{.base_sizes = {4, 4}, .new_sizes = {1, 4, 1, 4, 1}}};
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys,
      ViewSequenceCacheKey(unsqueeze, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(
      param_keys,
      ElementsAre(Pair("view", "reshape:expand{0,1},{2,3,4}d0=1,d2=1,d4=1")));

  ViewSequence scalar_unsqueeze = {
      ReshapePrimitive{.base_sizes = {}, .new_sizes = {1}}};
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys,
      ViewSequenceCacheKey(scalar_unsqueeze, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys,
              ElementsAre(Pair("view", "reshape:expand:scalar_unsqueeze(1)")));

  ViewSequence scalar_squeeze = {
      ReshapePrimitive{.base_sizes = {1}, .new_sizes = {}}};
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys,
      ViewSequenceCacheKey(scalar_squeeze, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys,
              ElementsAre(Pair("view", "reshape:collapse:scalar_squeeze")));

  // Unflatten encodes static dimensions when reassociation is has ambiguous
  // factorizations.
  //  {6} => {2,3} or {3,2}, use suffix `d0=2,d1=3`
  ViewSequence unflatten = {
      ReshapePrimitive{.base_sizes = {6, 4}, .new_sizes = {2, 3, 4}}};
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys,
      ViewSequenceCacheKey(unflatten, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys,
              ElementsAre(Pair("view", "reshape:expand{0,1},{2}d0=2,d1=3")));

  ViewSequence transpose_like = {
      ReshapePrimitive{.base_sizes = {1, 4, 6}, .new_sizes = {4, 1, 6}}};
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys,
      ViewSequenceCacheKey(transpose_like, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys,
              ElementsAre(Pair("view", "reshape:transpose_like:d1=1")));
}

TEST(SymbolicViewPrimitive, TransposeViewCacheKeys) {
  // Create a fallback tensor which is only used when symbolic keygen fails
  at::Tensor tensor = at::empty({4, 6}).reshape({6, 4});
  std::string no_sym_key = "cache_key{storage_offset:0, strides:[4,1]}";

  auto param_keys = OpParamCacheKeys::Empty();
  ViewSequence transpose = {TransposePrimitive{.permutation = {1, 0}}};
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys,
      ViewSequenceCacheKey(transpose, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys, ElementsAre(Pair("view", "transpose:[1,0]")));
}

TEST(SymbolicViewPrimitive, CastingViewCacheKeys) {
  // Create a fallback tensor which is only used when symbolic keygen fails
  at::Tensor tensor = at::empty({4, 6}).reshape({6, 4});

  auto param_keys = OpParamCacheKeys::Empty();
  ViewSequence conj = {ConjPrimitive{true}};
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys, ViewSequenceCacheKey(conj, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys, ElementsAre(Pair("view", "conj:1")));

  ViewSequence view_as_complex = {
      ViewAsComplex{ComplexElementType::kComplexFloat}};
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys,
      ViewSequenceCacheKey(view_as_complex, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys, ElementsAre(Pair("view", "view_as_complex:cfloat")));

  ViewSequence real_to_real_bitcast = {RealToRealBitcast{
      .from_type = mlir::ElementType::F32, .to_type = mlir::ElementType::UI32}};
  TF_ASSERT_OK_AND_ASSIGN(param_keys,
                          ViewSequenceCacheKey(real_to_real_bitcast,
                                               *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys, ElementsAre(Pair(
                              "view", "real_to_real_bitcast:float32->uint32")));

  ViewSequence complex_to_real_bitcast = {ComplexToRealBitcast{
      .complex_element_type = ComplexElementType::kComplexFloat,
      .bitcast_type = ComplexToRealBitcastType::kViewAsReal}};
  TF_ASSERT_OK_AND_ASSIGN(param_keys,
                          ViewSequenceCacheKey(complex_to_real_bitcast,
                                               *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(
      param_keys,
      ElementsAre(Pair("view", "complex_to_real_bitcast:cfloat:view_as_real")));
}

TEST(SymbolicViewPrimitive, PadViewCacheKeys) {
  // Create a fallback tensor which is only used when symbolic keygen fails
  at::Tensor tensor = at::empty({4, 6}).reshape({6, 4});

  auto param_keys = OpParamCacheKeys::Empty();
  ViewSequence pad = {PadPrimitive{
      .pad_dims = {
          {.low_padding = 0, .high_padding = 0, .interior_padding = 0}}}};
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys, ViewSequenceCacheKey(pad, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys, ElementsAre(Pair("view", "pad:[{l0,h0,i0}]")));
}

TEST(SymbolicViewPrimitive, BroadcastViewCacheKeys) {
  // Create a fallback tensor which is only used when symbolic keygen fails
  at::Tensor tensor = at::empty({4, 6}).reshape({6, 4});

  auto param_keys = OpParamCacheKeys::Empty();
  ViewSequence bcast_size_one = {BroadcastPrimitive{
      .base_shape{1, 1}, .new_sizes = {2, 2}, .broadcast_dimensions = {0, 1}}};
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys,
      ViewSequenceCacheKey(bcast_size_one, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys, ElementsAre(Pair("view", "broadcast:[2,2]")));

  ViewSequence bcast_expand_replicate = {BroadcastPrimitive{
      .base_shape{4}, .new_sizes = {4, 2, 1}, .broadcast_dimensions = {0}}};
  TF_ASSERT_OK_AND_ASSIGN(param_keys,
                          ViewSequenceCacheKey(bcast_expand_replicate,
                                               *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys, ElementsAre(Pair("view", "broadcast:[in0,2,1]")));
}

TEST(SymbolicViewPrimitive, MultipleViewCacheKeys) {
  // Create a fallback tensor which is only used when symbolic keygen fails
  at::Tensor tensor = at::empty({4, 6}).reshape({6, 4});

  auto param_keys = OpParamCacheKeys::Empty();
  ViewSequence reshape_transpose = {
      ReshapePrimitive{.base_sizes = {2, 2, 2}, .new_sizes = {4, 2}},
      TransposePrimitive{.permutation = {1, 0}},
  };
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys,
      ViewSequenceCacheKey(reshape_transpose, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(
      param_keys,
      ElementsAre(Pair("view", "reshape:collapse{0,1},{2};transpose:[1,0]")));

  ViewSequence transpose_transpose = {
      TransposePrimitive{{1, 0}},
      TransposePrimitive{{1, 0}},
  };
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys,
      ViewSequenceCacheKey(transpose_transpose, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys,
              ElementsAre(Pair("view", "transpose:[1,0];transpose:[1,0]")));
}

TEST(SymbolicViewPrimitive, UnsupportedViewCacheKeys) {
  // Create a fallback tensor which is only used when symbolic keygen fails
  at::Tensor tensor = at::empty({4, 6}).reshape({6, 4});

  auto param_keys = OpParamCacheKeys::Empty();

  // Unfolds are mostly static shape, symbolic keygen not very useful.
  ViewSequence unfold = {UnfoldPrimitive{.start_index = 0,
                                         .limit_index = 9,
                                         .window_stride = 999,
                                         .window_size = 9}};
  TF_ASSERT_OK_AND_ASSIGN(
      param_keys, ViewSequenceCacheKey(unfold, *tensor.unsafeGetTensorImpl()));
  EXPECT_THAT(param_keys, ElementsAre(Pair("storage_offset", "0"),
                                      Pair("strides", "[4,1]")));
}

}  // namespace
}  // namespace torch_tpu
