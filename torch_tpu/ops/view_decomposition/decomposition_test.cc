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

#include "torch_tpu/ops/view_decomposition/decomposition.h"

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/view_sequence.h"

namespace torch_tpu {
namespace {

// Validates that applying the view sequence to the contiguous base shape
// produces the desired view layout. Returns an error if it does not.
absl::Status ValidateViewSequence(
    absl::Span<const ViewPrimitive> view_sequence,
    absl::Span<const int64_t> contiguous_base_shape,
    const StridedLayout& view_layout) {
  auto layout = MakeContiguousBaseLayout(contiguous_base_shape);
  for (const auto& primitive : view_sequence) {
    UpdateLayout(layout, primitive);
  }

  TT_RET_CHECK(layout == view_layout, error::kInvalidArgument)
      << "Validation failed:\n"
      << "contiguous_base_shape: " << ToString(contiguous_base_shape)
      << "\nview_sequence: " << ToString(view_sequence)
      << "\nvalidation error: view_sequence does not produce the desired view "
         "layout. Expected: "
      << view_layout << " but got: " << layout;
  return absl::OkStatus();
}

void DecompositionTest(
    absl::Span<const int64_t> contiguous_base_shape,
    const StridedLayout& view_layout,
    const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32,
    const mlir::ElementType view_dtype = mlir::ElementType::F32,
    bool is_conj = false) {
  absl::StatusOr<ViewSequence> sequence_status =
      DecomposeIntoViewSequence(contiguous_base_shape, contiguous_base_dtype,
                                view_layout, view_dtype, is_conj);

  ASSERT_TRUE(sequence_status.ok());
  ViewSequence sequence = std::move(sequence_status).value();

  // Un-simplified sequence should be valid.
  EXPECT_TRUE(
      ValidateViewSequence(sequence, contiguous_base_shape, view_layout).ok());

  // Sequence should be simplifiable.
  Simplify(sequence, contiguous_base_shape);

  // Simplified sequence should be also valid.
  EXPECT_TRUE(
      ValidateViewSequence(sequence, contiguous_base_shape, view_layout).ok());
}

TEST(DecomposeIntoViewSequence, ScalarNoOp) {
  Dimensions contiguous_base_shape = {};
  StridedLayout view_layout = {
      .strided_dims = {},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, TensorNoOp) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, ReshapeScalarToTensor) {
  Dimensions contiguous_base_shape = {};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 1}, {.size = 1, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, ReshapeTensorToScalar) {
  Dimensions contiguous_base_shape = {1, 1};
  StridedLayout view_layout = {
      .strided_dims = {},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, FlattenTensor) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 24, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, FlattenAndBroadcastTensor) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 999, .stride = 0},
                       {.size = 24, .stride = 1},
                       {.size = 1, .stride = 999}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, ReshapeTensorToTensor) {
  Dimensions contiguous_base_shape = {27, 2};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 6, .stride = 9}, {.size = 9, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, UnsqueezeTensor) {
  Dimensions contiguous_base_shape = {27, 2};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 999},
                       {.size = 27, .stride = 2},
                       {.size = 1, .stride = 999},
                       {.size = 2, .stride = 1},
                       {.size = 1, .stride = 999}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, PermuteTensor) {
  Dimensions contiguous_base_shape = {1, 2, 3, 4};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 24},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1},
                       {.size = 2, .stride = 12}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, ExpandTensor) {
  Dimensions contiguous_base_shape = {1024, 1, 128};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1024, .stride = 128},
                       {.size = 512, .stride = 0},
                       {.size = 128, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, ExpandSingleElementTensor) {
  Dimensions contiguous_base_shape = {128};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 999, .stride = 0}, {.size = 1, .stride = 1}},
      .storage_offset = 64};
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, BroadcastScalar) {
  Dimensions contiguous_base_shape = {};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1024, .stride = 0},
                       {.size = 512, .stride = 0},
                       {.size = 128, .stride = 0}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, BroadcastTensor) {
  Dimensions contiguous_base_shape = {1024, 128};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1024, .stride = 128},
                       {.size = 512, .stride = 0},
                       {.size = 128, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, BroadcastAndPermuteTensor) {
  Dimensions contiguous_base_shape = {1024, 128};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 128, .stride = 1},
                       {.size = 512, .stride = 0},
                       {.size = 1024, .stride = 128}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, SliceTensorLow) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  // Equivalent to torch.ones(2, 3, 4)[1:, 1:, 1:]
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 12},
                       {.size = 2, .stride = 4},
                       {.size = 3, .stride = 1}},
      .storage_offset = 9,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, SliceTensorHigh) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  // Equivalent to torch.ones(2, 3, 4)[:1, :2, :3]
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 12},
                       {.size = 2, .stride = 4},
                       {.size = 3, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, SliceTensorStridedNonContiguous) {
  Dimensions contiguous_base_shape = {7};
  // Equivalent to torch.ones(7)[1:].view(2, 3)[:, ::2]
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 3}, {.size = 2, .stride = 2}},
      .storage_offset = 1,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, SliceTensorStridedContiguous) {
  Dimensions contiguous_base_shape = {2, 4, 4};
  // Equivalent to torch.ones(2, 4, 4)[::2, ::2, ::2]
  StridedLayout view_layout = {
      .strided_dims = {{.size = 1, .stride = 32},
                       {.size = 2, .stride = 8},
                       {.size = 2, .stride = 2}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, PaddingRequired) {
  Dimensions contiguous_base_shape = {5};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 3}, {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, ScalarViewAsReal) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::F32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorViewAsReal) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 24},
                       {.size = 3, .stride = 8},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::F32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, ScalarRealHalf) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::F32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, ScalarImagHalf) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {},
      .storage_offset = 1,
  };
  mlir::ElementType view_dtype = mlir::ElementType::F32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorRealHalf) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 24},
                       {.size = 3, .stride = 8},
                       {.size = 4, .stride = 2}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::F32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorImagHalf) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 24},
                       {.size = 3, .stride = 8},
                       {.size = 4, .stride = 2}},
      .storage_offset = 1,
  };
  mlir::ElementType view_dtype = mlir::ElementType::F32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, ScalarRealToSmallerReal) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::UI64;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::UI16;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorRealToSmallerReal) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::UI64;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 48},
                       {.size = 3, .stride = 16},
                       {.size = 4, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::UI16;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, ScalarRealToEqualSizedReal) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::PRED;
  StridedLayout view_layout = {
      .strided_dims = {},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::I8;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorRealToEqualSizedReal) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::PRED;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::I8;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorRealToLargerReal) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::UI16;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 3}, {.size = 3, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::UI64;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorViewAsComplex) {
  Dimensions contiguous_base_shape = {2, 3, 4, 2};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::COMPLEXF32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, ScalarCDoubleToCFloat) {
  Dimensions contiguous_base_shape = {};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF64;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::COMPLEXF32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorCDoubleToCFloat) {
  Dimensions contiguous_base_shape = {2, 3, 4};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF64;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 24},
                       {.size = 3, .stride = 8},
                       {.size = 4, .stride = 2},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::COMPLEXF32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, TensorCFloatToCDouble) {
  Dimensions contiguous_base_shape = {2, 3, 4, 2};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::COMPLEXF32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 3, .stride = 4},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::COMPLEXF64;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, MultiStepAsStridedAndBitcast) {
  Dimensions contiguous_base_shape = {54};
  mlir::ElementType contiguous_base_dtype = mlir::ElementType::UI32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 999, .stride = 0},
                       {.size = 2, .stride = 1},
                       {.size = 4, .stride = 2}},
      .storage_offset = 1,
  };
  mlir::ElementType view_dtype = mlir::ElementType::F32;
  DecompositionTest(contiguous_base_shape, view_layout, contiguous_base_dtype,
                    view_dtype);
}

TEST(DecomposeIntoViewSequence, OverlappingMinorDimension) {
  Dimensions contiguous_base_shape = {55};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 6, .stride = 9},
                       {.size = 999, .stride = 0},
                       {.size = 3, .stride = 1},
                       {.size = 4, .stride = 2}},
      .storage_offset = 1,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, OverlappingMajorDimension) {
  Dimensions contiguous_base_shape = {44};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 6, .stride = 7},
                       {.size = 999, .stride = 0},
                       {.size = 2, .stride = 1},
                       {.size = 4, .stride = 2}},
      .storage_offset = 1,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, OverlappingMajorAndMinorDimension) {
  Dimensions contiguous_base_shape = {45};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 6, .stride = 7},
                       {.size = 999, .stride = 0},
                       {.size = 3, .stride = 1},
                       {.size = 4, .stride = 2}},
      .storage_offset = 1,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, Conjugate_ComplexBase_ComplexView) {
  Dimensions contiguous_base_shape = {2, 3};
  StridedLayout view_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  DecompositionTest(contiguous_base_shape, view_layout,
                    mlir::ElementType::COMPLEXF32,
                    mlir::ElementType::COMPLEXF32, /*is_conj=*/true);
}

TEST(DecomposeIntoViewSequence, Conjugate_ComplexBase_RealView) {
  Dimensions contiguous_base_shape = {2, 3};
  // Complex -> Real (ViewAsReal) implies output shape {2, 3, 2}
  Dimensions view_shape = {2, 3, 2};
  StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  DecompositionTest(contiguous_base_shape, view_layout,
                    mlir::ElementType::COMPLEXF32, mlir::ElementType::F32,
                    /*is_conj=*/true);
}

TEST(DecomposeIntoViewSequence, Conjugate_RealBase_ComplexView) {
  Dimensions contiguous_base_shape = {2, 3, 2};
  // Real -> Complex (ViewAsComplex) implies output shape {2, 3}
  Dimensions view_shape = {2, 3};
  StridedLayout view_layout = MakeContiguousBaseLayout(view_shape);
  DecompositionTest(contiguous_base_shape, view_layout, mlir::ElementType::F32,
                    mlir::ElementType::COMPLEXF32, /*is_conj=*/true);
}

TEST(DecomposeIntoViewSequence, Conjugate_RealBase_RealView) {
  Dimensions contiguous_base_shape = {2, 3};
  StridedLayout view_layout = MakeContiguousBaseLayout(contiguous_base_shape);
  DecompositionTest(contiguous_base_shape, view_layout, mlir::ElementType::F32,
                    mlir::ElementType::F32, /*is_conj=*/true);
}

void GetContiguousBaseShapeTest(const StridedLayout& view_layout,
                                absl::Span<const int64_t> expected_base_shape) {
  absl::StatusOr<Dimensions> base_shape = GetContiguousBaseShape(view_layout);
  ASSERT_TRUE(base_shape.ok());

  EXPECT_THAT(*base_shape, testing::ElementsAreArray(expected_base_shape));

  DecompositionTest(*base_shape, view_layout);
}

TEST(GetContiguousBaseShape, ScalarView) {
  StridedLayout view_layout = {
      .strided_dims = {},
      .storage_offset = 0,
  };
  Dimensions expected_base_shape = {};
  GetContiguousBaseShapeTest(view_layout, expected_base_shape);
}

TEST(GetContiguousBaseShape, ScalarViewWithOffset) {
  StridedLayout view_layout = {
      .strided_dims = {},
      .storage_offset = 7,
  };
  Dimensions expected_base_shape = {8};
  GetContiguousBaseShapeTest(view_layout, expected_base_shape);
}

TEST(GetContiguousBaseShape, OneDimensionContiguousBase) {
  StridedLayout view_layout = {
      .strided_dims = {{.size = 7, .stride = 1}},
      .storage_offset = 0,
  };
  Dimensions expected_base_shape = {7};
  GetContiguousBaseShapeTest(view_layout, expected_base_shape);
}

TEST(GetContiguousBaseShape, OneDimensionDenseSlice) {
  StridedLayout view_layout = {
      .strided_dims = {{.size = 7, .stride = 1}},
      .storage_offset = 1,
  };
  Dimensions expected_base_shape = {8};
  GetContiguousBaseShapeTest(view_layout, expected_base_shape);
}

TEST(GetContiguousBaseShape, OneDimensionStridedSlice) {
  StridedLayout view_layout = {
      .strided_dims = {{.size = 7, .stride = 2}},
      .storage_offset = 0,
  };
  Dimensions expected_base_shape = {13};
  GetContiguousBaseShapeTest(view_layout, expected_base_shape);
}

TEST(GetContiguousBaseShape, TwoDimensionContiguousBase) {
  StridedLayout view_layout = {
      .strided_dims = {{.size = 3, .stride = 2}, {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  Dimensions expected_base_shape = {3, 2};
  GetContiguousBaseShapeTest(view_layout, expected_base_shape);
}

TEST(GetContiguousBaseShape, TwoDimensionPermuted) {
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 1}, {.size = 3, .stride = 2}},
      .storage_offset = 0,
  };
  Dimensions expected_base_shape = {3, 2};
  GetContiguousBaseShapeTest(view_layout, expected_base_shape);
}

TEST(GetContiguousBaseShape, TwoDimensionDenseSlice) {
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 2}, {.size = 2, .stride = 1}},
      .storage_offset = 2,
  };
  Dimensions expected_base_shape = {3, 2};
  GetContiguousBaseShapeTest(view_layout, expected_base_shape);
}

TEST(GetContiguousBaseShape, TwoDimensionStridedSlice) {
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 4}, {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  // This could be a strided slice like torch.ones(3, 2)[::2, :], but the
  // decomposition prefers using padded dense slices, by padding from
  // (6,) -> (8,), reshaping to (2, 4), and then slicing to (2, 2).
  Dimensions expected_base_shape = {6};
  GetContiguousBaseShapeTest(view_layout, expected_base_shape);
}

TEST(GetContiguousBaseShape, TwoDimensionBroadcast) {
  StridedLayout view_layout = {
      .strided_dims = {{.size = 3, .stride = 1}, {.size = 999, .stride = 0}},
      .storage_offset = 0,
  };
  Dimensions expected_base_shape = {3};
  GetContiguousBaseShapeTest(view_layout, expected_base_shape);
}

TEST(GetContiguousBaseShape, TwoDimensionOverlapping) {
  StridedLayout view_layout = {
      .strided_dims = {{.size = 3, .stride = 2}, {.size = 3, .stride = 1}},
      .storage_offset = 0,
  };
  // The unsqueezed 1 dimension is so that the unfold can behave like
  // ```
  //   x = torch.ones(1, 7)
  //   y = torch.concat([x[:, 0:3], x[:, 2:5], x[:, 4:7]], dim=0)
  // ```
  Dimensions expected_base_shape = {1, 7};
  GetContiguousBaseShapeTest(view_layout, expected_base_shape);
}

}  // namespace
}  // namespace torch_tpu
