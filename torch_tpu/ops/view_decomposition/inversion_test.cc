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

#include "torch_tpu/ops/view_decomposition/inversion.h"

#include <utility>
#include <variant>

#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/ops/view_decomposition/bitcast_primitive.h"
#include "torch_tpu/ops/view_decomposition/conj_primitive.h"
#include "torch_tpu/ops/view_decomposition/pad_primitive.h"
#include "torch_tpu/ops/view_decomposition/reshape_primitive.h"
#include "torch_tpu/ops/view_decomposition/slice_primitive.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/transpose_primitive.h"
#include "torch_tpu/ops/view_decomposition/view_sequence.h"

namespace torch_tpu {
namespace {
using absl_testing::StatusIs;
using testing::HasSubstr;

TEST(ComputeInverseViewOperation, ScalarNoOp) {
  const Dimensions contiguous_base_shape = {};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const Dimensions view_shape = {};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);

  ASSERT_TRUE(inverse_view_operation.ok());
  InverseViewOperation expected = {
      .base_transform = {},
      .bitcast_view = {},
      .stages = {InverseViewStage()},
      .final_shape = Shape({}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, TensorNoOp) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const Dimensions view_shape = {2, 3, 4};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);

  ASSERT_TRUE(inverse_view_operation.ok());
  InverseViewOperation expected = {
      .base_transform = {},
      .bitcast_view = {},
      .stages = {InverseViewStage()},
      .final_shape = Shape({2, 3, 4}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, ReshapeScalarToTensor) {
  const Dimensions contiguous_base_shape = {};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const Dimensions view_shape = {1, 1};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);

  ASSERT_TRUE(inverse_view_operation.ok());
  InverseViewOperation expected = {
      .base_transform = {ReshapePrimitive{.new_sizes = {1, 1}}},
      .bitcast_view = {},
      .stages = {InverseViewStage()},
      .final_shape = Shape({1, 1}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, ReshapeTensorToScalar) {
  const Dimensions contiguous_base_shape = {1, 1};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const Dimensions view_shape = {};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);

  ASSERT_TRUE(inverse_view_operation.ok());
  InverseViewOperation expected = {
      .base_transform = {ReshapePrimitive{.base_sizes = {1, 1},
                                          .new_sizes = {}}},
      .bitcast_view = {},
      .stages = {InverseViewStage()},
      .final_shape = Shape({}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, ReshapeTensorToTensor) {
  const Dimensions contiguous_base_shape = {27, 2};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const Dimensions view_shape = {6, 9};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);

  ASSERT_TRUE(inverse_view_operation.ok());
  InverseViewOperation expected = {
      .base_transform = {ReshapePrimitive{.base_sizes = {27, 2},
                                          .new_sizes = {6, 9}}},
      .bitcast_view = {},
      .stages = {InverseViewStage()},
      .final_shape = Shape({6, 9}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, PermuteTensor) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const StridedLayout view_layout = {
      .strided_dims = {{.size = 3, .stride = 4},
                       {.size = 2, .stride = 12},
                       {.size = 4, .stride = 1}}};
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);

  ASSERT_TRUE(inverse_view_operation.ok());
  InverseViewOperation expected = {
      .base_transform = {},
      .bitcast_view = {},
      .stages = {InverseViewStage{
          .forward = {TransposePrimitive{.permutation = {1, 0, 2}}},
          .inverse = {TransposePrimitive{.permutation = {1, 0, 2}}}}},
      .final_shape = Shape({2, 3, 4}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, SliceTensorLow) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const StridedLayout view_layout = {.strided_dims = {{.size = 2, .stride = 12},
                                                      {.size = 2, .stride = 4},
                                                      {.size = 3, .stride = 1}},
                                     .storage_offset = 5};
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);

  ASSERT_TRUE(inverse_view_operation.ok());
  InverseViewOperation expected = {
      .base_transform = {},
      .bitcast_view = {},
      .stages = {InverseViewStage{
                     .slice = SlicePrimitive{.slice_dims = {{.start_index = 0,
                                                             .limit_index = 2,
                                                             .stride = 1},
                                                            {.start_index = 1,
                                                             .limit_index = 3,
                                                             .stride = 1},
                                                            {.start_index = 1,
                                                             .limit_index = 4,
                                                             .stride = 1}}}},
                 InverseViewStage()},
      .final_shape = Shape({2, 3, 4}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, SliceTensorHigh) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const StridedLayout view_layout = {.strided_dims = {{.size = 1, .stride = 12},
                                                      {.size = 2, .stride = 4},
                                                      {.size = 3, .stride = 1}},
                                     .storage_offset = 0};
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);

  ASSERT_TRUE(inverse_view_operation.ok());
  InverseViewOperation expected = {
      .base_transform = {ReshapePrimitive{.base_sizes = {2, 3, 4},
                                          .new_sizes = {6, 4}}},
      .bitcast_view = {},
      .stages = {InverseViewStage{
                     .slice = SlicePrimitive{.slice_dims = {{.start_index = 0,
                                                             .limit_index = 2,
                                                             .stride = 1},
                                                            {.start_index = 0,
                                                             .limit_index = 3,
                                                             .stride = 1}}}},
                 InverseViewStage{
                     .forward = {ReshapePrimitive{.base_sizes = {2, 3},
                                                  .new_sizes = {1, 2, 3}}},
                     .inverse = {ReshapePrimitive{.base_sizes = {1, 2, 3},
                                                  .new_sizes = {2, 3}}}}},
      .final_shape = Shape({6, 4}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, SliceTensorStridedNonContiguous) {
  const Dimensions contiguous_base_shape = {7};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 3}, {.size = 2, .stride = 2}},
      .storage_offset = 1};
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  EXPECT_TRUE(inverse_view_operation.ok());

  // Forward original view sequence: torch.ones(7)[1:].view(2, 3)[:, ::2]
  // Gets rewritten by ReplaceStridedSlices to:
  // ```
  //   x = torch.ones(7)[1:].view(2, 3)
  //   x = torch.nn.functional.pad(x, (0, 0, 0, 1)).view(2, 2, 2)
  //   x = x[:, :, 0].view(2, 2)
  // ```
  InverseViewOperation expected = {
      .base_transform = {},
      .bitcast_view = {},
      .stages = {InverseViewStage{
                     .slice = SlicePrimitive{.slice_dims = {{.start_index = 1,
                                                             .limit_index = 7,
                                                             .stride = 1}}}},
                 InverseViewStage{
                     .forward = {ReshapePrimitive{.base_sizes = {6},
                                                  .new_sizes = {2, 3}},
                                 PadPrimitive{
                                     .pad_dims = {PadDimension(),
                                                  PadDimension{.high_padding =
                                                                   1}}},
                                 ReshapePrimitive{.base_sizes = {2, 4},
                                                  .new_sizes = {2, 2, 2}}},
                     .slice = SlicePrimitive{.slice_dims = {{.start_index = 0,
                                                             .limit_index = 2,
                                                             .stride = 1},
                                                            {.start_index = 0,
                                                             .limit_index = 2,
                                                             .stride = 1},
                                                            {.start_index = 0,
                                                             .limit_index = 1,
                                                             .stride = 1}}},
                     .inverse = {ReshapePrimitive{.base_sizes = {2, 2, 2},
                                                  .new_sizes = {2, 4}},
                                 SlicePrimitive{
                                     .slice_dims = {{.start_index = 0,
                                                     .limit_index = 2,
                                                     .stride = 1},
                                                    {.start_index = 0,
                                                     .limit_index = 3,
                                                     .stride = 1}}},
                                 ReshapePrimitive{.base_sizes = {2, 3},
                                                  .new_sizes = {6}}}},
                 InverseViewStage{.forward = {ReshapePrimitive{
                                      .base_sizes =
                                          {2, 2, 1},
                                      .new_sizes = {2, 2}}},
                                  .inverse = {ReshapePrimitive{
                                      .base_sizes =
                                          {2, 2},
                                      .new_sizes = {2, 2, 1}}}}},
      .final_shape = Shape({7}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, SliceTensorStridedContiguous) {
  const Dimensions contiguous_base_shape = {2, 4, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const StridedLayout view_layout = {.strided_dims = {{.size = 1, .stride = 16},
                                                      {.size = 2, .stride = 8},
                                                      {.size = 2, .stride = 2}},
                                     .storage_offset = 0};
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());

  InverseViewOperation expected =
      {.base_transform = {ReshapePrimitive{.base_sizes = {2, 4, 4},
                                           .new_sizes = {4, 4, 2}}},
       .bitcast_view = {},
       .stages =
           {InverseViewStage{
                .slice =
                    SlicePrimitive{
                        .slice_dims =
                            {{.start_index = 0, .limit_index = 2, .stride = 1},
                             {.start_index = 0, .limit_index = 2, .stride = 1},
                             {.start_index = 0, .limit_index = 1, .stride = 1}},
                    }},
            InverseViewStage{
                .forward = {ReshapePrimitive{.base_sizes = {2, 2, 1},
                                             .new_sizes = {1, 2, 2}}},
                .inverse = {ReshapePrimitive{.base_sizes = {1, 2, 1},
                                             .new_sizes = {2, 2, 1}}}}},
       .final_shape = Shape({4, 4, 2}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, SliceTensorStridedWithLowAndHighIndex) {
  const Dimensions contiguous_base_shape = {30};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  // View layout: torch.ones(30)[5:20:6], elements are 5, 11, 17.
  const StridedLayout view_layout = {.strided_dims = {{.size = 3, .stride = 6}},
                                     .storage_offset = 5};
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());

  InverseViewOperation expected = {
      .base_transform = {ReshapePrimitive{.base_sizes = {30},
                                          .new_sizes = {5, 6}}},
      .bitcast_view = {},
      .stages = {InverseViewStage{
                     .slice = SlicePrimitive{.slice_dims = {{.start_index = 0,
                                                             .limit_index = 3,
                                                             .stride = 1},
                                                            {.start_index = 5,
                                                             .limit_index = 6,
                                                             .stride = 1}}}},
                 InverseViewStage{
                     .forward = {ReshapePrimitive{.base_sizes = {3, 1},
                                                  .new_sizes = {3}}},
                     .inverse = {ReshapePrimitive{.base_sizes = {3},
                                                  .new_sizes = {3, 1}}}}},
      .final_shape = Shape({5, 6}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, NoDtypeChange) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const Dimensions view_shape = {4, 3, 2};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());
  InverseViewOperation expected = {
      .base_transform = {ReshapePrimitive{.base_sizes = {2, 3, 4},
                                          .new_sizes = {4, 3, 2}}},
      .bitcast_view = {},
      .stages = {InverseViewStage()},
      .final_shape = Shape({4, 3, 2}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, RealToRealSameSize) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const Dimensions view_shape = {4, 3, 2};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::UI32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());
  // Prefers casting into the base dtype.
  InverseViewOperation expected = {
      .base_transform = {ReshapePrimitive{.base_sizes = {2, 3, 4},
                                          .new_sizes = {4, 3, 2}}},
      .bitcast_view = {RealToRealBitcast{.from_type = mlir::ElementType::UI32,
                                         .to_type = mlir::ElementType::F32}},
      .stages = {InverseViewStage()},
      .final_shape = Shape({4, 3, 2}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, RealToRealLargerSize) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const Dimensions view_shape = {3, 2, 2};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::UI64;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());
  InverseViewOperation expected = {
      .base_transform = {ReshapePrimitive{.base_sizes = {2, 3, 4},
                                          .new_sizes = {3, 2, 2, 2}}},
      .bitcast_view = {RealToRealBitcast{.from_type = mlir::ElementType::UI64,
                                         .to_type = mlir::ElementType::F32}},
      .stages = {InverseViewStage()},
      .final_shape = Shape({3, 2, 2, 2}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape.dimensions(),
            expected.final_shape.dimensions());
  EXPECT_EQ(inverse_view_operation->final_shape.dtype(),
            expected.final_shape.dtype());
}

TEST(ComputeInverseViewOperation, RealToRealSmallerSize) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::UI64;
  const Dimensions view_shape = {4, 3, 4};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());
  InverseViewOperation expected = {
      .base_transform = {RealToRealBitcast{.from_type = mlir::ElementType::UI64,
                                           .to_type = mlir::ElementType::F32},
                         ReshapePrimitive{.base_sizes = {2, 3, 4, 2},
                                          .new_sizes = {4, 3, 4}}},
      .bitcast_view = {},
      .stages = {InverseViewStage()},
      .final_shape = Shape({4, 3, 4}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, BoolToBool) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::PRED;
  const Dimensions view_shape = {4, 3, 2};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::PRED;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());
  // bool-to-bool is also a no-op, same as any other no-op typecast.
  InverseViewOperation expected = {
      .base_transform = {ReshapePrimitive{.base_sizes = {2, 3, 4},
                                          .new_sizes = {4, 3, 2}}},
      .bitcast_view = {},
      .stages = {InverseViewStage()},
      .final_shape = Shape({4, 3, 2}, mlir::ElementType::PRED)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, BoolToByte) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::PRED;
  const Dimensions view_shape = {4, 3, 2};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::UI8;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());
  // Writes to booleans force the base to be stored as UI8.
  InverseViewOperation expected = {
      .base_transform = {RealToRealBitcast{.from_type = mlir::ElementType::PRED,
                                           .to_type = mlir::ElementType::UI8},
                         ReshapePrimitive{.base_sizes = {2, 3, 4},
                                          .new_sizes = {4, 3, 2}}},
      .bitcast_view = {},
      .stages = {InverseViewStage()},
      .final_shape = Shape({4, 3, 2}, mlir::ElementType::UI8)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, BoolToLargerSize) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::PRED;
  const Dimensions view_shape = {3, 2};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());
  // Writes to booleans force the base to be stored as UI8, and we also need
  // to cast the write to UI8.
  InverseViewOperation expected = {
      .base_transform = {RealToRealBitcast{.from_type = mlir::ElementType::PRED,
                                           .to_type = mlir::ElementType::UI8},
                         ReshapePrimitive{.base_sizes = {2, 3, 4},
                                          .new_sizes = {3, 2, 4}}},
      .bitcast_view = {RealToRealBitcast{.from_type = mlir::ElementType::F32,
                                         .to_type = mlir::ElementType::UI8}},
      .stages = {InverseViewStage()},
      .final_shape = Shape({3, 2, 4}, mlir::ElementType::UI8)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape.dimensions(),
            expected.final_shape.dimensions());
  EXPECT_EQ(inverse_view_operation->final_shape.dtype(),
            expected.final_shape.dtype());
}

TEST(ComputeInverseViewOperation, ByteToBool) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::UI8;
  const Dimensions view_shape = {4, 3, 2};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::PRED;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());
  // Base is already stored as UI8, so no bitcasting is needed.
  InverseViewOperation expected = {
      .base_transform = {ReshapePrimitive{.base_sizes = {2, 3, 4},
                                          .new_sizes = {4, 3, 2}}},
      .bitcast_view = {RealToRealBitcast{.from_type = mlir::ElementType::PRED,
                                         .to_type = mlir::ElementType::UI8}},
      .stages = {InverseViewStage()},
      .final_shape = Shape({4, 3, 2}, mlir::ElementType::UI8)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, ViewAsReal) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  const Dimensions view_shape = {2, 2, 4, 3};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());
  InverseViewOperation expected = {
      .base_transform = {ComplexToRealBitcast{
                             .complex_element_type =
                                 ComplexElementType::kComplexFloat,
                             .bitcast_type =
                                 ComplexToRealBitcastType::kViewAsReal},
                         ReshapePrimitive{.base_sizes = {2, 3, 4, 2},
                                          .new_sizes = {2, 2, 4, 3}}},
      .bitcast_view = {},
      .stages = {InverseViewStage()},
      .final_shape = Shape({2, 2, 4, 3}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, RealPart) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;

  const StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 24},
                       {.size = 4, .stride = 2},
                       {.size = 3, .stride = 8}},
      .storage_offset = 0,
  };
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());
  // Base is converted to F32s so that we can preserve the imaginary part.
  // Inverse adds a pad operation add the imaginary part back in.
  SlicePrimitive expected_slice;
  expected_slice.slice_dims.push_back(
      SliceDimension{.start_index = 0, .limit_index = 2, .stride = 1});
  expected_slice.slice_dims.push_back(
      SliceDimension{.start_index = 0, .limit_index = 3, .stride = 1});
  expected_slice.slice_dims.push_back(
      SliceDimension{.start_index = 0, .limit_index = 4, .stride = 1});
  expected_slice.slice_dims.push_back(
      SliceDimension{.start_index = 0, .limit_index = 1, .stride = 1});

  InverseViewOperation expected = {
      .base_transform = {ComplexToRealBitcast{
          .complex_element_type = ComplexElementType::kComplexFloat,
          .bitcast_type = ComplexToRealBitcastType::kViewAsReal}},
      .bitcast_view = {},
      .stages = {InverseViewStage{.slice = std::move(expected_slice)},
                 InverseViewStage{
                     .forward = {ReshapePrimitive{.base_sizes = {2, 3, 4, 1},
                                                  .new_sizes = {2, 3, 4}},
                                 TransposePrimitive{.permutation = {0, 2, 1}}},
                     .inverse = {TransposePrimitive{.permutation = {0, 2, 1}},
                                 ReshapePrimitive{.base_sizes = {2, 3, 4},
                                                  .new_sizes = {2, 3, 4, 1}}}}},
      .final_shape = Shape({2, 3, 4, 2}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, ImaginaryPart) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;

  const StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 24},
                       {.size = 4, .stride = 2},
                       {.size = 3, .stride = 8}},
      .storage_offset = 1,
  };
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());
  // Base is converted to F32s so that we can preserve the real part.
  // Inverse adds a pad operation add the real part back in.

  SlicePrimitive expected_slice;
  expected_slice.slice_dims.push_back(
      SliceDimension{.start_index = 0, .limit_index = 2, .stride = 1});
  expected_slice.slice_dims.push_back(
      SliceDimension{.start_index = 0, .limit_index = 3, .stride = 1});
  expected_slice.slice_dims.push_back(
      SliceDimension{.start_index = 0, .limit_index = 4, .stride = 1});
  expected_slice.slice_dims.push_back(
      SliceDimension{.start_index = 1, .limit_index = 2, .stride = 1});

  InverseViewOperation expected = {
      .base_transform = {ComplexToRealBitcast{
          .complex_element_type = ComplexElementType::kComplexFloat,
          .bitcast_type = ComplexToRealBitcastType::kViewAsReal}},
      .bitcast_view = {},
      .stages = {InverseViewStage{.slice = std::move(expected_slice)},
                 InverseViewStage{
                     .forward = {ReshapePrimitive{.base_sizes = {2, 3, 4, 1},
                                                  .new_sizes = {2, 3, 4}},
                                 TransposePrimitive{.permutation = {0, 2, 1}}},
                     .inverse = {TransposePrimitive{.permutation = {0, 2, 1}},
                                 ReshapePrimitive{.base_sizes = {2, 3, 4},
                                                  .new_sizes = {2, 3, 4, 1}}}}},
      .final_shape = Shape({2, 3, 4, 2}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, ViewAsComplex) {
  const Dimensions contiguous_base_shape = {2, 3, 4, 2};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const Dimensions view_shape = {2, 4, 3};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::COMPLEXF32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());
  InverseViewOperation expected = {
      .base_transform = {ReshapePrimitive{.base_sizes = {2, 3, 4, 2},
                                          .new_sizes = {2, 4, 3, 2}}},
      .bitcast_view = {ComplexToRealBitcast{
          .complex_element_type = ComplexElementType::kComplexFloat,
          .bitcast_type = ComplexToRealBitcastType::kViewAsReal}},
      .stages = {InverseViewStage()},
      .final_shape = Shape({2, 4, 3, 2}, mlir::ElementType::F32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, ComplexF32ToComplexF64) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  const Dimensions view_shape = {2, 3, 2};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::COMPLEXF64;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());
  // View gets cast from C64 -> F64 -> F32 -> C32.
  InverseViewOperation expected = {
      .base_transform = {ReshapePrimitive{.base_sizes = {2, 3, 4},
                                          .new_sizes = {2, 3, 2, 2, 1}}},
      .bitcast_view = {ComplexToRealBitcast{
                           .complex_element_type =
                               ComplexElementType::kComplexDouble,
                           .bitcast_type =
                               ComplexToRealBitcastType::kViewAsReal},
                       RealToRealBitcast{.from_type = mlir::ElementType::F64,
                                         .to_type = mlir::ElementType::F32},
                       ViewAsComplex{.complex_element_type =
                                         ComplexElementType::kComplexFloat}},
      .stages = {InverseViewStage()},
      .final_shape = Shape({2, 3, 2, 2, 1}, mlir::ElementType::COMPLEXF32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, ComplexF64ToComplexF32) {
  const Dimensions contiguous_base_shape = {2, 3, 2};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF64;
  const Dimensions view_shape = {4, 3, 2};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::COMPLEXF32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  ASSERT_TRUE(inverse_view_operation.ok());
  // Base gets cast from C64 -> F64 -> F32 -> C32.
  InverseViewOperation expected = {
      .base_transform = {ComplexToRealBitcast{
                             .complex_element_type =
                                 ComplexElementType::kComplexDouble,
                             .bitcast_type =
                                 ComplexToRealBitcastType::kViewAsReal},
                         RealToRealBitcast{.from_type = mlir::ElementType::F64,
                                           .to_type = mlir::ElementType::F32},
                         ViewAsComplex{.complex_element_type =
                                           ComplexElementType::kComplexFloat},
                         ReshapePrimitive{.base_sizes = {2, 3, 2, 2, 1},
                                          .new_sizes = {4, 3, 2}}},
      .bitcast_view = {},
      .stages = {InverseViewStage()},
      .final_shape = Shape({4, 3, 2}, mlir::ElementType::COMPLEXF32)};
  EXPECT_EQ(inverse_view_operation->base_transform, expected.base_transform);
  EXPECT_EQ(inverse_view_operation->bitcast_view, expected.bitcast_view);
  EXPECT_EQ(inverse_view_operation->stages, expected.stages);
  EXPECT_EQ(inverse_view_operation->final_shape, expected.final_shape);
}

TEST(ComputeInverseViewOperation, InvertConj) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  const Dimensions view_shape = {2, 3, 4};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::COMPLEXF32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype,
      /*is_conj=*/true);
  ASSERT_TRUE(inverse_view_operation.ok());
  // Base to view should have a ConjPrimitive, so view to base should also have
  // a ConjPrimitive.
  ASSERT_EQ(inverse_view_operation->stages.size(), 1);
  ASSERT_EQ(inverse_view_operation->stages[0].inverse.size(), 1);
  EXPECT_TRUE(std::holds_alternative<ConjPrimitive>(
      inverse_view_operation->stages[0].inverse[0]));
}

TEST(ComputeInverseViewOperation, InvertConjReal) {
  const Dimensions contiguous_base_shape = {2, 3, 4};
  const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  const Dimensions view_shape = {2, 3, 4};
  const StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  const mlir::ElementType view_dtype = mlir::ElementType::F32;

  auto inverse_view_operation = ComputeInverseViewOperation(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype,
      /*is_conj=*/true);
  ASSERT_TRUE(inverse_view_operation.ok());
  // Conjugation on real types is a no-op, so it should be removed.
  ASSERT_EQ(inverse_view_operation->stages.size(), 1);
  EXPECT_TRUE(inverse_view_operation->stages[0].forward.empty());
  EXPECT_FALSE(inverse_view_operation->stages[0].slice.has_value());
  EXPECT_TRUE(inverse_view_operation->stages[0].inverse.empty());
}

}  // namespace
}  // namespace torch_tpu
