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

#include "csrc/ops/view_decomposition/decomposition.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/stride/stride_helper.h"
#include "csrc/ops/view_decomposition/bitcast_primitive.h"
#include "csrc/ops/view_decomposition/broadcast_primitive.h"
#include "csrc/ops/view_decomposition/conj_primitive.h"
#include "csrc/ops/view_decomposition/pad_primitive.h"
#include "csrc/ops/view_decomposition/reshape_primitive.h"
#include "csrc/ops/view_decomposition/slice_primitive.h"
#include "csrc/ops/view_decomposition/strided_layout.h"
#include "csrc/ops/view_decomposition/transpose_primitive.h"
#include "csrc/ops/view_decomposition/unfold_primitive.h"
#include "csrc/ops/view_decomposition/view_sequence.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

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
  TT_ASSERT_OK_AND_ASSIGN(
      ViewSequence sequence,
      DecomposeIntoViewSequence(contiguous_base_shape, contiguous_base_dtype,
                                view_layout, view_dtype, is_conj));

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

TEST(DecomposeIntoViewSequence, SelectMajorAxis2D) {
  // torch.ones(3, 4)[1, :]
  Dimensions contiguous_base_shape = {3, 4};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 4, .stride = 1}},
      .storage_offset = 4,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, SelectMinorAxis2D) {
  // torch.ones(3, 4)[:, 1]
  Dimensions contiguous_base_shape = {3, 4};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 3, .stride = 4}},
      .storage_offset = 1,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, SelectMajorAxis3D) {
  // torch.ones(3, 4, 5)[1, :, :]
  Dimensions contiguous_base_shape = {3, 4, 5};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 4, .stride = 5}, {.size = 5, .stride = 1}},
      .storage_offset = 20,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, SelectMiddleAxis3D) {
  // torch.ones(3, 4, 5)[:, 1, :]
  Dimensions contiguous_base_shape = {3, 4, 5};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 3, .stride = 20}, {.size = 5, .stride = 1}},
      .storage_offset = 5,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, SelectMinorAxis3D) {
  // torch.ones(3, 4, 5)[:, :, 1]
  Dimensions contiguous_base_shape = {3, 4, 5};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 3, .stride = 20}, {.size = 4, .stride = 5}},
      .storage_offset = 1,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, SelectTwoAxes) {
  // torch.ones(3, 4, 5)[1, :, 1]
  Dimensions contiguous_base_shape = {3, 4, 5};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 4, .stride = 5}},
      .storage_offset = 21,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

TEST(DecomposeIntoViewSequence, SelectSingleElement) {
  // torch.ones(3, 4, 5)[1, 1, 1]
  Dimensions contiguous_base_shape = {3, 4, 5};
  StridedLayout view_layout = {
      .strided_dims = {},
      .storage_offset = 26,
  };
  DecompositionTest(contiguous_base_shape, view_layout);
}

void GetContiguousBaseShapeTest(const StridedLayout& view_layout,
                                absl::Span<const int64_t> expected_base_shape) {
  TT_ASSERT_OK_AND_ASSIGN(Dimensions base_shape,
                          GetContiguousBaseShape(view_layout));

  EXPECT_THAT(base_shape, testing::ElementsAreArray(expected_base_shape));

  DecompositionTest(base_shape, view_layout);
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

// Returns the number of elements of the layout immediately before the first
// bitcast in `sequence`, or -1 if there is no bitcast.
int64_t NumelBeforeFirstBitcast(absl::Span<const ViewPrimitive> sequence,
                                absl::Span<const int64_t> base_shape) {
  StridedLayout layout = MakeContiguousBaseLayout(base_shape);
  for (const auto& primitive : sequence) {
    if (std::holds_alternative<RealToRealBitcast>(primitive)) {
      int64_t numel = 1;
      for (const auto& dim : layout.strided_dims) numel *= dim.size;
      return numel;
    }
    UpdateLayout(layout, primitive);
  }
  return -1;
}

// A wide view (f32) of a narrow base (u8) must bitcast only the bytes covered
// by the view, not the whole base, so that the work scales with the view.
void NarrowBaseBitcastScalesWithViewTest(const StridedLayout& view_layout,
                                         int64_t view_numel,
                                         const Dimensions& base_shape = {
                                             1 << 20}) {
  TT_ASSERT_OK_AND_ASSIGN(
      ViewSequence sequence,
      DecomposeIntoViewSequence(base_shape, mlir::ElementType::UI8, view_layout,
                                mlir::ElementType::F32,
                                /*is_conj=*/false));
  Simplify(sequence, base_shape);
  const absl::Status valid =
      ValidateViewSequence(sequence, base_shape, view_layout);
  EXPECT_TRUE(valid.ok()) << valid;
  const int64_t numel_before_bitcast =
      NumelBeforeFirstBitcast(sequence, base_shape);
  EXPECT_GT(numel_before_bitcast, 0) << ToString(sequence);
  // The view covers 4 * view_numel bytes.
  EXPECT_LE(numel_before_bitcast, 4 * view_numel) << ToString(sequence);
}

TEST(DecomposeIntoViewSequence, NarrowBaseWideView1DBitcastsOnlyView) {
  // 16 KiB float32 view (4096 f32 elements = 16,384 bytes) centered in a
  // 1 MiB (1 << 20 byte = 262,144 f32 element) uint8 base.
  constexpr int64_t kViewNumel = 4096;
  StridedLayout view_layout = {
      .strided_dims = {{.size = kViewNumel, .stride = 1}},
      .storage_offset = ((1 << 20) / 4 - kViewNumel) / 2,
  };
  NarrowBaseBitcastScalesWithViewTest(view_layout, kViewNumel);
}

TEST(DecomposeIntoViewSequence, NarrowBaseWideView2DBitcastsOnlyView) {
  // 16 KiB 64x64 float32 view (4096 f32 elements) centered in a 1 MiB uint8
  // base.
  constexpr int64_t kViewNumel = 64 * 64;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 64, .stride = 64}, {.size = 64, .stride = 1}},
      .storage_offset = ((1 << 20) / 4 - kViewNumel) / 2,
  };
  NarrowBaseBitcastScalesWithViewTest(view_layout, kViewNumel);
}

TEST(DecomposeIntoViewSequence,
     NarrowBaseWideViewStridedColumnBitcastsOnlyView) {
  // torch.ones(4096, dtype=torch.uint8).view(32, 128).view(torch.float32)[:, 1]
  // Spans all 32 rows of the base, but only covers 32 float32 elements
  // (128 bytes).
  constexpr int64_t kViewNumel = 32;
  StridedLayout view_layout = {
      .strided_dims = {{.size = kViewNumel, .stride = 32}},
      .storage_offset = 1,
  };
  NarrowBaseBitcastScalesWithViewTest(view_layout, kViewNumel,
                                      /*base_shape=*/{4096});
  NarrowBaseBitcastScalesWithViewTest(view_layout, kViewNumel,
                                      /*base_shape=*/{32, 128});
}

TEST(DecomposeIntoViewSequence,
     NarrowBaseWideViewTransposedAndBroadcastBitcastsOnlyView) {
  // Transposed 64x64 float32 view centered in a 1 MiB uint8 base.
  constexpr int64_t kViewNumel = 64 * 64;
  StridedLayout transposed_view_layout = {
      .strided_dims = {{.size = 64, .stride = 1}, {.size = 64, .stride = 64}},
      .storage_offset = ((1 << 20) / 4 - kViewNumel) / 2,
  };
  NarrowBaseBitcastScalesWithViewTest(transposed_view_layout, kViewNumel);

  // Broadcasted 64x128 float32 view (backed by 64 f32 elements) centered in a
  // 1 MiB uint8 base.
  StridedLayout broadcast_view_layout = {
      .strided_dims = {{.size = 64, .stride = 1}, {.size = 128, .stride = 0}},
      .storage_offset = ((1 << 20) / 4 - 64) / 2,
  };
  NarrowBaseBitcastScalesWithViewTest(broadcast_view_layout, /*view_numel=*/64);
}

TEST(DecomposeIntoViewSequence, NarrowBase3DWideView2DBitcastsOnlyViewRows) {
  // The same 1 MiB uint8 base, after a write through a float32 view has
  // reshaped it to [256, 1024, 4] (256 rows of 1024 f32 elements = 4 KiB per
  // row). Both a 64x64 view and a row-aligned 4x1024 view bitcast only the
  // 4 * 4096 bytes covered by the view, and the row-aligned 4x1024 view
  // decomposes to [slice, bitcast] with no reshape.
  constexpr int64_t kViewNumel = 64 * 64;
  StridedLayout view_layout_64x64 = {
      .strided_dims = {{.size = 64, .stride = 64}, {.size = 64, .stride = 1}},
      .storage_offset = ((1 << 20) / 4 - kViewNumel) / 2,
  };
  NarrowBaseBitcastScalesWithViewTest(view_layout_64x64, kViewNumel,
                                      /*base_shape=*/{256, 1024, 4});

  StridedLayout view_layout_rows = {
      .strided_dims = {{.size = 4, .stride = 1024},
                       {.size = 1024, .stride = 1}},
      .storage_offset = 126 * 1024,
  };
  const Dimensions base_shape = {256, 1024, 4};
  NarrowBaseBitcastScalesWithViewTest(view_layout_rows, kViewNumel, base_shape);
  TT_ASSERT_OK_AND_ASSIGN(
      ViewSequence sequence,
      DecomposeIntoViewSequence(base_shape, mlir::ElementType::UI8,
                                view_layout_rows, mlir::ElementType::F32,
                                /*is_conj=*/false));
  Simplify(sequence, base_shape);
  for (const auto& primitive : sequence) {
    EXPECT_FALSE(std::holds_alternative<ReshapePrimitive>(primitive))
        << ToString(sequence);
  }
}

TEST(DecomposeIntoViewSequence, NarrowBase3DCompiledArgumentPrefix) {
  // torch.compile passes a view as the minimal contiguous prefix of its
  // storage: here the first 32 rows (32 * 1024 f32 elements = 128 KiB) of a
  // [256, 1024, 4] uint8 base, read as float32 from offset 0.
  constexpr int64_t kViewNumel = 32 * 1024;
  StridedLayout view_layout = {
      .strided_dims = {{.size = kViewNumel, .stride = 1}},
      .storage_offset = 0,
  };
  NarrowBaseBitcastScalesWithViewTest(view_layout, kViewNumel,
                                      /*base_shape=*/{256, 1024, 4});
}

TEST(DecomposeIntoViewSequence, EqualSizeRealViewBitcastsOnlyView) {
  // Same-size real-to-real bitcast (e.g. int32 -> float32): a 4096-element
  // float32 view centered in a 262,144-element int32 base should slice the
  // base down to 4096 elements before bitcasting.
  constexpr int64_t kBaseNumel = 1 << 18;
  constexpr int64_t kViewNumel = 4096;
  const Dimensions base_shape = {kBaseNumel};
  StridedLayout view_layout = {
      .strided_dims = {{.size = kViewNumel, .stride = 1}},
      .storage_offset = (kBaseNumel - kViewNumel) / 2,
  };
  TT_ASSERT_OK_AND_ASSIGN(
      ViewSequence sequence,
      DecomposeIntoViewSequence(base_shape, mlir::ElementType::I32, view_layout,
                                mlir::ElementType::F32, /*is_conj=*/false));
  Simplify(sequence, base_shape);
  const absl::Status valid =
      ValidateViewSequence(sequence, base_shape, view_layout);
  EXPECT_TRUE(valid.ok()) << valid;
  EXPECT_EQ(NumelBeforeFirstBitcast(sequence, base_shape), kViewNumel)
      << ToString(sequence);
}

TEST(DecomposeIntoViewSequence, RemoveExtraDimWithBitcast) {
  Dimensions base_shape = {8};
  mlir::ElementType base_dtype = mlir::ElementType::PRED;
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 2}},
      .storage_offset = 0,
  };
  mlir::ElementType view_dtype = mlir::ElementType::UI8;
  DecompositionTest(base_shape, view_layout, base_dtype, view_dtype);
}

// The storage indices of the elements of a tensor, in row-major order. An
// element that comes from padding has storage index -1.
struct StorageIndices {
  Dimensions sizes;
  std::vector<int64_t> values;  // INT_VEC_OK
};

// Calls fn with each index of a tensor of the given sizes, in row-major order.
void ForEachIndex(absl::Span<const int64_t> sizes,
                  absl::FunctionRef<void(absl::Span<const int64_t>)> fn) {
  if (absl::c_linear_search(sizes, 0)) {
    return;
  }
  Dimensions index(sizes.size(), 0);
  while (true) {
    fn(index);
    int64_t i = static_cast<int64_t>(sizes.size()) - 1;
    while (i >= 0 && ++index[i] == sizes[i]) {
      index[i] = 0;
      --i;
    }
    if (i < 0) {
      return;
    }
  }
}

// Returns a tensor of the given sizes. Its element at each index is the element
// of input at the index set by get_input_index, or -1 if get_input_index
// returns false.
StorageIndices Gather(const StorageIndices& input, Dimensions sizes,
                      absl::FunctionRef<bool(absl::Span<const int64_t> index,
                                             Dimensions& input_index)>
                          get_input_index) {
  StorageIndices result = {.sizes = std::move(sizes)};
  Dimensions input_index(input.sizes.size());
  ForEachIndex(result.sizes, [&](absl::Span<const int64_t> index) {
    if (!get_input_index(index, input_index)) {
      result.values.push_back(-1);
      return;
    }
    int64_t position = 0;
    for (int64_t i = 0; i < input.sizes.size(); ++i) {
      if (input_index[i] < 0 || input_index[i] >= input.sizes[i]) {
        ADD_FAILURE() << "expected an index in " << ToString(input.sizes)
                      << ", got " << ToString(input_index);
        result.values.push_back(-2);
        return;
      }
      position = position * input.sizes[i] + input_index[i];
    }
    result.values.push_back(input.values[position]);
  });
  return result;
}

StorageIndices Apply(const StorageIndices& input,
                     const ReshapePrimitive& reshape) {
  EXPECT_EQ(input.sizes, reshape.base_sizes);
  const int64_t numel =
      absl::c_accumulate(reshape.new_sizes, int64_t{1}, std::multiplies<>());
  if (numel != static_cast<int64_t>(input.values.size())) {
    ADD_FAILURE() << "expected " << input.values.size()
                  << " elements, got reshape to "
                  << ToString(reshape.new_sizes);
    return input;
  }
  return {.sizes = reshape.new_sizes, .values = input.values};
}

StorageIndices Apply(const StorageIndices& input,
                     const TransposePrimitive& transpose) {
  Dimensions sizes;
  for (const int64_t dim : transpose.permutation) {
    sizes.push_back(input.sizes[dim]);
  }
  return Gather(input, std::move(sizes),
                [&](absl::Span<const int64_t> index, Dimensions& input_index) {
                  for (int64_t i = 0; i < index.size(); ++i) {
                    input_index[transpose.permutation[i]] = index[i];
                  }
                  return true;
                });
}

StorageIndices Apply(const StorageIndices& input,
                     const BroadcastPrimitive& broadcast) {
  EXPECT_EQ(input.sizes, broadcast.base_shape);
  return Gather(input, broadcast.new_sizes,
                [&](absl::Span<const int64_t> index, Dimensions& input_index) {
                  for (int64_t i = 0; i < input.sizes.size(); ++i) {
                    input_index[i] =
                        input.sizes[i] == 1
                            ? 0
                            : index[broadcast.broadcast_dimensions[i]];
                  }
                  return true;
                });
}

StorageIndices Apply(const StorageIndices& input, const SlicePrimitive& slice) {
  Dimensions sizes;
  for (int64_t i = 0; i < slice.slice_dims.size(); ++i) {
    const SliceDimension& dim = slice.slice_dims[i];
    EXPECT_LE(dim.limit_index, input.sizes[i]) << slice;
    sizes.push_back((dim.limit_index - dim.start_index + dim.stride - 1) /
                    dim.stride);
  }
  return Gather(input, std::move(sizes),
                [&](absl::Span<const int64_t> index, Dimensions& input_index) {
                  for (int64_t i = 0; i < index.size(); ++i) {
                    input_index[i] = slice.slice_dims[i].start_index +
                                     index[i] * slice.slice_dims[i].stride;
                  }
                  return true;
                });
}

StorageIndices Apply(const StorageIndices& input, const PadPrimitive& pad) {
  Dimensions sizes;
  for (int64_t i = 0; i < pad.pad_dims.size(); ++i) {
    const PadDimension& dim = pad.pad_dims[i];
    sizes.push_back(input.sizes[i] + dim.low_padding + dim.high_padding +
                    std::max<int64_t>(input.sizes[i] - 1, 0) *
                        dim.interior_padding);
  }
  return Gather(input, std::move(sizes),
                [&](absl::Span<const int64_t> index, Dimensions& input_index) {
                  for (int64_t i = 0; i < index.size(); ++i) {
                    const PadDimension& dim = pad.pad_dims[i];
                    const int64_t offset = index[i] - dim.low_padding;
                    if (offset < 0 ||
                        offset % (dim.interior_padding + 1) != 0 ||
                        offset / (dim.interior_padding + 1) >= input.sizes[i]) {
                      return false;
                    }
                    input_index[i] = offset / (dim.interior_padding + 1);
                  }
                  return true;
                });
}

StorageIndices Apply(const StorageIndices& input,
                     const UnfoldPrimitive& unfold) {
  const int64_t rank = input.sizes.size();
  EXPECT_LE(unfold.limit_index, input.sizes[rank - 1]) << unfold;
  Dimensions sizes = input.sizes;
  sizes[rank - 2] =
      1 + (unfold.limit_index - unfold.start_index - unfold.window_size) /
              unfold.window_stride;
  sizes[rank - 1] = unfold.window_size;
  return Gather(input, std::move(sizes),
                [&](absl::Span<const int64_t> index, Dimensions& input_index) {
                  absl::c_copy(index, input_index.begin());
                  input_index[rank - 2] = 0;
                  input_index[rank - 1] =
                      unfold.start_index +
                      index[rank - 2] * unfold.window_stride + index[rank - 1];
                  return true;
                });
}

StorageIndices Apply(const StorageIndices& input, const ConjPrimitive& conj) {
  return input;
}

// A bitcast to a narrower type splits each element along a new last dimension.
// The storage indices then count elements of the narrower type, like the
// storage offset and strides of a view with that type.
StorageIndices Apply(const StorageIndices& input,
                     const RealToRealBitcast& bitcast) {
  const int64_t from_bitwidth = TorchEquivalentBitwidth(bitcast.from_type);
  const int64_t to_bitwidth = TorchEquivalentBitwidth(bitcast.to_type);
  if (from_bitwidth < to_bitwidth) {
    ADD_FAILURE() << "bitcasts to a wider type are not supported, got "
                  << bitcast;
    return input;
  }
  const int64_t ratio = from_bitwidth / to_bitwidth;
  StorageIndices result = {.sizes = input.sizes};
  if (ratio > 1) {
    result.sizes.push_back(ratio);
  }
  for (const int64_t value : input.values) {
    for (int64_t i = 0; i < ratio; ++i) {
      result.values.push_back(value < 0 ? value : value * ratio + i);
    }
  }
  return result;
}

// Complex bitcasts change the number of elements per storage index in ways
// that StorageIndices does not model.
template <typename T>
StorageIndices Apply(const StorageIndices& input, const T& bitcast) {
  ADD_FAILURE() << "complex bitcasts are not supported, got " << bitcast;
  return input;
}

// Applies the view sequence to the storage indices of a contiguous base.
StorageIndices ApplyToStorageIndices(
    ViewSequenceSpan view_sequence,
    absl::Span<const int64_t> contiguous_base_shape) {
  StorageIndices indices = {
      .sizes = Dimensions(contiguous_base_shape.begin(),
                          contiguous_base_shape.end()),
  };
  indices.values.resize(absl::c_accumulate(contiguous_base_shape, int64_t{1},
                                           std::multiplies<>()));
  absl::c_iota(indices.values, int64_t{0});
  for (const auto& primitive : view_sequence) {
    indices = std::visit(
        [&](const auto& primitive) { return Apply(indices, primitive); },
        primitive);
  }
  return indices;
}

// Returns the storage indices of the elements of the view.
StorageIndices GetViewStorageIndices(const StridedLayout& view_layout) {
  StorageIndices indices = {.sizes = GetSizes(view_layout)};
  ForEachIndex(indices.sizes, [&](absl::Span<const int64_t> index) {
    int64_t storage_index = view_layout.storage_offset;
    for (int64_t i = 0; i < index.size(); ++i) {
      storage_index += index[i] * view_layout.strided_dims[i].stride;
    }
    indices.values.push_back(storage_index);
  });
  return indices;
}

// Checks that the view sequence reads the elements of the view from the
// contiguous base. Unlike ValidateViewSequence, this also works for view
// sequences with a pad after a reshape, since a pad resets the layout.
void ExpectReadsViewElements(ViewSequenceSpan view_sequence,
                             absl::Span<const int64_t> contiguous_base_shape,
                             const StridedLayout& view_layout) {
  const StorageIndices actual =
      ApplyToStorageIndices(view_sequence, contiguous_base_shape);
  const StorageIndices expected = GetViewStorageIndices(view_layout);
  EXPECT_EQ(actual.sizes, expected.sizes) << ToString(view_sequence);
  EXPECT_EQ(actual.values, expected.values) << ToString(view_sequence);
}

// Decomposes a view of a contiguous base, and checks that the view sequence
// reads the elements of the view before and after simplification. Returns the
// simplified view sequence.
ViewSequence DecomposeAndCheckElements(
    absl::Span<const int64_t> contiguous_base_shape,
    const StridedLayout& view_layout,
    const mlir::ElementType contiguous_base_dtype = mlir::ElementType::F32,
    const mlir::ElementType view_dtype = mlir::ElementType::F32) {
  absl::StatusOr<ViewSequence> sequence = DecomposeIntoViewSequence(
      contiguous_base_shape, contiguous_base_dtype, view_layout, view_dtype);
  if (!sequence.ok()) {
    ADD_FAILURE() << sequence.status();
    return {};
  }
  ExpectReadsViewElements(*sequence, contiguous_base_shape, view_layout);
  Simplify(*sequence, contiguous_base_shape);
  ExpectReadsViewElements(*sequence, contiguous_base_shape, view_layout);
  return *std::move(sequence);
}

// Returns true if the view sequence has an unfold or a broadcast. These
// primitives repeat elements, so writes to the view can't be inverted.
bool HasRepeatingPrimitive(ViewSequenceSpan view_sequence) {
  return absl::c_any_of(view_sequence, [](const ViewPrimitive& primitive) {
    return std::holds_alternative<UnfoldPrimitive>(primitive) ||
           std::holds_alternative<BroadcastPrimitive>(primitive);
  });
}

TEST(DecomposeIntoViewSequence, StridedSliceOfMiddleDimension) {
  // torch.ones(2, 3, 4)[:, ::2]
  // The size of the middle dimension (3) is not a multiple of the step (2), so
  // its size times its stride (2 * 8) is larger than the outer stride (12).
  // Its elements span only 12, so the view doesn't overlap.
  Dimensions contiguous_base_shape = {2, 3, 4};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 12},
                       {.size = 2, .stride = 8},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_FALSE(HasRepeatingPrimitive(
      DecomposeAndCheckElements(contiguous_base_shape, view_layout)));
}

TEST(DecomposeIntoViewSequence, StridedSliceOfMiddleDimensionWithUnitDim) {
  // torch.ones(12, 4, 8, 1)[:, 0:4:3]
  Dimensions contiguous_base_shape = {12, 4, 8, 1};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 12, .stride = 32},
                       {.size = 2, .stride = 24},
                       {.size = 8, .stride = 1},
                       {.size = 1, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_FALSE(HasRepeatingPrimitive(
      DecomposeAndCheckElements(contiguous_base_shape, view_layout)));
}

TEST(DecomposeIntoViewSequence, StridedSliceOfMiddleDimensionWithOffset) {
  // torch.ones(12, 5, 8)[:, 1:5:3]
  Dimensions contiguous_base_shape = {12, 5, 8};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 12, .stride = 40},
                       {.size = 2, .stride = 24},
                       {.size = 8, .stride = 1}},
      .storage_offset = 8,
  };
  EXPECT_FALSE(HasRepeatingPrimitive(
      DecomposeAndCheckElements(contiguous_base_shape, view_layout)));

  // The same strides with storage offset 9. The slice of each row can absorb
  // only 8 of it, so the head keeps a slice of the base.
  contiguous_base_shape = {481};
  view_layout.storage_offset = 9;
  EXPECT_FALSE(HasRepeatingPrimitive(
      DecomposeAndCheckElements(contiguous_base_shape, view_layout)));
}

TEST(DecomposeIntoViewSequence, StridedViewsWithTwoPads) {
  // In each view, the rows of two dimensions are narrower than the view width
  // of the next dimension, so the body pads both of them.
  const auto count_pads = [](ViewSequenceSpan view_sequence) {
    return absl::c_count_if(view_sequence, [](const ViewPrimitive& primitive) {
      return std::holds_alternative<PadPrimitive>(primitive);
    });
  };

  // torch.ones(2, 3, 2, 3, 4)[:, ::2, :, ::2]
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 72},
                       {.size = 2, .stride = 48},
                       {.size = 2, .stride = 12},
                       {.size = 2, .stride = 8},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  ViewSequence sequence =
      DecomposeAndCheckElements({2, 3, 2, 3, 4}, view_layout);
  EXPECT_FALSE(HasRepeatingPrimitive(sequence));
  EXPECT_EQ(count_pads(sequence), 2) << ToString(sequence);

  // torch.ones(26).as_strided((2, 2, 2, 2), (13, 7, 4, 1))
  view_layout = {
      .strided_dims = {{.size = 2, .stride = 13},
                       {.size = 2, .stride = 7},
                       {.size = 2, .stride = 4},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  sequence = DecomposeAndCheckElements({26}, view_layout);
  EXPECT_FALSE(HasRepeatingPrimitive(sequence));
  EXPECT_EQ(count_pads(sequence), 2) << ToString(sequence);
}

TEST(DecomposeIntoViewSequence, StridedSliceOfDivisibleMiddleDimension) {
  // torch.ones(2, 4, 4)[:, ::2]
  Dimensions contiguous_base_shape = {2, 4, 4};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 16},
                       {.size = 2, .stride = 8},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_FALSE(HasRepeatingPrimitive(
      DecomposeAndCheckElements(contiguous_base_shape, view_layout)));
}

TEST(DecomposeIntoViewSequence, StridedSliceOfMinorDimension) {
  // torch.ones(2, 3, 5)[:, :, ::2]
  Dimensions contiguous_base_shape = {2, 3, 5};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 15},
                       {.size = 3, .stride = 5},
                       {.size = 3, .stride = 2}},
      .storage_offset = 0,
  };
  EXPECT_FALSE(HasRepeatingPrimitive(
      DecomposeAndCheckElements(contiguous_base_shape, view_layout)));
}

TEST(DecomposeIntoViewSequence, OverlappingWindowsUseUnfold) {
  // torch.arange(7).unfold(0, 3, 2)
  StridedLayout view_layout = {
      .strided_dims = {{.size = 3, .stride = 2}, {.size = 3, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_TRUE(
      HasRepeatingPrimitive(DecomposeAndCheckElements({7}, view_layout)));

  // Dimension 1 overlaps the dimensions after it, while the rows of dimension 0
  // need a pad, like the strided slices above.
  view_layout = {
      .strided_dims = {{.size = 2, .stride = 9},
                       {.size = 2, .stride = 4},
                       {.size = 2, .stride = 3},
                       {.size = 2, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_TRUE(
      HasRepeatingPrimitive(DecomposeAndCheckElements({18}, view_layout)));
}

TEST(DecomposeIntoViewSequence, ExpandedDimensionUsesBroadcast) {
  // torch.ones(2, 1, 4).expand(2, 3, 4)
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 4},
                       {.size = 3, .stride = 0},
                       {.size = 4, .stride = 1}},
      .storage_offset = 0,
  };
  EXPECT_TRUE(
      HasRepeatingPrimitive(DecomposeAndCheckElements({8}, view_layout)));
}

TEST(DecomposeIntoViewSequence, StridedViewOfTwoDimensionalBase) {
  // x.view(-1)[1:11].view(2, 5).unfold(1, 2, 3) for x of shape (6, 2). The rows
  // of dimension 0 need a pad, so the head slice of the flattened base keeps
  // the storage offset: [1:11]. Simplify can't move that slice before the
  // reshape of the base, since dimension 1 of the base would need an empty
  // slice.
  const Dimensions contiguous_base_shape = {6, 2};
  StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 5},
                       {.size = 2, .stride = 3},
                       {.size = 2, .stride = 1}},
      .storage_offset = 1,
  };
  EXPECT_FALSE(HasRepeatingPrimitive(
      DecomposeAndCheckElements(contiguous_base_shape, view_layout)));

  // The same view after .permute(2, 1, 0).
  view_layout.strided_dims = {{.size = 2, .stride = 1},
                              {.size = 2, .stride = 3},
                              {.size = 2, .stride = 5}};
  EXPECT_FALSE(HasRepeatingPrimitive(
      DecomposeAndCheckElements(contiguous_base_shape, view_layout)));
}

TEST(DecomposeIntoViewSequence, StridedNarrowViewOfWideBase) {
  // torch.zeros(6).view(torch.bfloat16).as_strided((2, 2, 2), (1, 3, 5), 1).
  // The bitcast to bfloat16 turns the base into shape (6, 2), as above.
  const StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 1},
                       {.size = 2, .stride = 3},
                       {.size = 2, .stride = 5}},
      .storage_offset = 1,
  };
  EXPECT_FALSE(HasRepeatingPrimitive(DecomposeAndCheckElements(
      {6}, view_layout, mlir::ElementType::F32, mlir::ElementType::BF16)));
}

TEST(DecomposeIntoViewSequence, StridedSliceOfTwoDimensionalBase) {
  // torch.ones(5, 4)[2::2, 2]. Simplify can't move the head slice of the
  // flattened base, [3:19], before the reshape of the base, since dimension 1
  // of the base would need an empty slice.
  const StridedLayout view_layout = {
      .strided_dims = {{.size = 2, .stride = 8}},
      .storage_offset = 10,
  };
  DecomposeAndCheckElements({5, 4}, view_layout);
}

// Calls fn with every view layout with up to 3 dimensions of size <= 3, stride
// <= 7 and storage offset <= 1, and the number of elements of the smallest
// contiguous base that holds the view.
void ForEachSmallLayout(
    absl::FunctionRef<void(const StridedLayout&, int64_t)> fn) {
  for (int64_t rank = 0; rank <= 3; ++rank) {
    ForEachIndex(Dimensions(rank, 4), [&](absl::Span<const int64_t> sizes) {
      if (absl::c_linear_search(sizes, 0)) {
        return;
      }
      ForEachIndex(Dimensions(rank, 8), [&](absl::Span<const int64_t> strides) {
        for (int64_t storage_offset = 0; storage_offset <= 1;
             ++storage_offset) {
          StridedLayout view_layout = {.storage_offset = storage_offset};
          int64_t numel = storage_offset + 1;
          for (int64_t i = 0; i < sizes.size(); ++i) {
            view_layout.strided_dims.push_back(
                {.size = sizes[i], .stride = strides[i]});
            numel += (sizes[i] - 1) * strides[i];
          }
          fn(view_layout, numel);
        }
      });
    });
  }
}

// Checks the decomposition of the view over the contiguous base. A view that
// isn't overlapping must not repeat elements, so that writes to it can be
// inverted.
void CheckSmallLayout(absl::Span<const int64_t> contiguous_base_shape,
                      const StridedLayout& view_layout) {
  SCOPED_TRACE(testing::Message()
               << "contiguous_base_shape: " << ToString(contiguous_base_shape)
               << ", view_layout: " << view_layout);
  const ViewSequence sequence =
      DecomposeAndCheckElements(contiguous_base_shape, view_layout);
  if (!IsOverlapping(GetSizes(view_layout), GetStrides(view_layout))) {
    EXPECT_FALSE(HasRepeatingPrimitive(sequence)) << ToString(sequence);
  }
}

TEST(DecomposeIntoViewSequence, SmallLayouts) {
  // The smallest 1-D base that holds the view, and a larger one.
  ForEachSmallLayout([](const StridedLayout& view_layout, int64_t numel) {
    CheckSmallLayout({numel}, view_layout);
    CheckSmallLayout({numel + 5}, view_layout);
  });
}

TEST(DecomposeIntoViewSequence, SmallLayoutsOfTwoDimensionalBases) {
  // Eager storage keeps the shape of the tensor that created it, so the base
  // often has more than one dimension.
  ForEachSmallLayout([](const StridedLayout& view_layout, int64_t numel) {
    CheckSmallLayout({(numel + 1) / 2, 2}, view_layout);
    CheckSmallLayout({(numel + 2) / 3, 3}, view_layout);
  });
}

TEST(GetContiguousBaseShape, StridedSliceOfMiddleDimension) {
  // torch.ones(12, 4, 8, 1)[:, 0:4:3]
  StridedLayout view_layout = {
      .strided_dims = {{.size = 12, .stride = 32},
                       {.size = 2, .stride = 24},
                       {.size = 8, .stride = 1},
                       {.size = 1, .stride = 1}},
      .storage_offset = 0,
  };
  TT_ASSERT_OK_AND_ASSIGN(Dimensions base_shape,
                          GetContiguousBaseShape(view_layout));
  EXPECT_THAT(base_shape, testing::ElementsAre(12, 32));
  EXPECT_FALSE(HasRepeatingPrimitive(
      DecomposeAndCheckElements(base_shape, view_layout)));
}

}  // namespace
}  // namespace torch_tpu
