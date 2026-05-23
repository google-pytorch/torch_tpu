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

#include "torch_tpu/ops/nonzero/nonzero.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/sort/sort.h"
#include "torch_tpu/ops/stride/stride_helper.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

mlir::MlirOp GetInputMask(mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);

  mlir::MlirOp input_mask;
  if (IsBooleanType(input_type)) {
    return input;
  }

  // not a boolean type, so create a boolean mask
  // create a zero tensor of same shape as input
  mlir::MlirOp zero = MakeConstantLike(input, 0);

  // compare input with zero_val to get a tensor of booleans
  // where input is not zero, the corresponding boolean is true.
  return stablehlo::Compare(input, zero, stablehlo::ComparisonDirection::NE);
}

absl::StatusOr<mlir::MlirOp> FlattenIfNeeded(mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  if (input_type.getRank() == 1) {
    // Already flat, no reshape needed.
    return input;
  }

  const int64_t num_elements = input_type.getNumElements();
  return stablehlo::Reshape(input, {num_elements});
}

absl::StatusOr<mlir::MlirOp> UnravelIndices(mlir::MlirOp input,
                                            mlir::MlirOp linear_indices,
                                            mlir::ElementType element_type) {
  mlir::MlirBuilder& builder = input.getBuilder();

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);

  Strides strides =
      CalculateStridesContiguous(CopyIntVector(input_type.getShape()));

  std::vector<mlir::MlirOp> dim_indices(input_type.getRank());

  for (int64_t dim = 0; dim < input_type.getRank(); ++dim) {
    // Support constant with bounded dynamism
    mlir::MlirOp stride_constant =
        MakeConstantLike(linear_indices, strides[dim]);
    mlir::MlirOp div_res = stablehlo::Div(linear_indices, stride_constant);
    mlir::MlirOp dim_constant =
        MakeConstantLike(div_res, input_type.getShape()[dim]);
    mlir::MlirOp rem_res = stablehlo::Rem(div_res, dim_constant);
    TT_ASSIGN_OR_RETURN(dim_indices[dim], Unsqueeze(rem_res, /*dim=*/1));
  }

  mlir::MlirOp indices =
      stablehlo::Concatenate(builder, dim_indices, /*dim=*/1);
  ABSL_VLOG(3) << "BuildNonzeroShlo indices: " << indices.ToString();
  return indices;
}

}  // namespace

// Algorithm: Nonzero Get Output Size Calculation
//
// This algorithm finds the number of non-zero elements in an input tensor.
//
// 1. Mask: A boolean mask is generated, with True values marking the positions
// of non-zero elements.
//
// 2. Count Non-Zero Elements: A sum reduction is performed on the boolean
// mask over all dimensions. The result is the number of non-zero elements.
//
// Example: Input Tensor A (Shape: 2x5):
// [[0, 8, 9, 0, 6],
//  [0, 0, 7, 0, 0]]
//
// Number of non-zero elements (S) = 4.
//
// Boolean Mask:
// [[F, T, T, F, T],
//  [F, F, T, F, F]]
// Numeric: [[0, 1, 1, 0, 1],
//           [0, 0, 1, 0, 0]]
//
// Sum of Mask: 4
//
absl::StatusOr<mlir::MlirOp> BuildNonzeroGetOutputSizeShlo(
    mlir::ElementType element_type, mlir::MlirOp input) {
  ABSL_VLOG(3) << "BuildNonzeroGetOutputSizeShlo input: " << input.ToString();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);

  mlir::MlirOp input_mask = GetInputMask(input);

  mlir::MlirOp zero_val =
      MakeScalarConstant(input.getBuilder(), 0, element_type);
  mlir::Type mlir_type =
      mlir::getElementType(input.getBuilder().getContext(), element_type);

  auto reduce_fn = [mlir_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<stablehlo::AddOp>(
        mlir_type, rb.getRegion(), rb.getOpBuilder());
  };

  Dimensions dims(input_type.getRank());
  absl::c_iota(dims, 0);
  mlir::MlirOp output_size =
      BuildReductionShlo(input_mask, dims, mlir_type, zero_val, reduce_fn,
                         ReductionMode::kDropDims);
  ABSL_VLOG(3) << "BuildNonzeroGetOutputSizeShlo output_size: "
               << output_size.ToString();
  return output_size;
}

//
// Algorithm: Nonzero Indices Calculation
//
// This algorithm finds the multi-dimensional indices of all non-zero elements
// in an input tensor.
//
// 1. Flatten and Mask: The input tensor is first flattened into a 1D array. A
// boolean mask is generated, with True values marking the positions of non-zero
// elements.
//
// 2. Stable Sort: Perform stable sort on the flattened mask to get the indices
// of the true values in the mask to the beginning of the array.
//
// 3. Slice: Slice to take the first output_size indices.
//
// 4. Unravel Indices: These linear indices are then converted back into
// multi-dimensional coordinates. This is done by repeatedly applying modulo and
// floor division operations with the dimension sizes of the original input
// tensor, typically in reverse order of dimensions (from last to first).
// The unraveled indices for each dimension (each now a 1D
// tensor of size S) are stacked along a new axis to produce the final output
// tensor of shape (S, rank).

// Example:
//
// Input Tensor A (Shape: 2x5):
// [[0, 8, 9, 0, 6],
//  [0, 0, 7, 0, 0]]
//
// Number of non-zero elements (S) = 4.
// Expected Output: [[0, 1], [0, 2], [0, 4], [1, 2]]
//
// Flattened Mask:
// [F, T, T, F, T, F, F, T, F, F]
// Numeric: [0, 1, 1, 0, 1, 0, 0, 1, 0, 0]
//
// Stable Sort:
// [T, T, T, T, F, F, F, F, F, F]
// [1, 2, 4, 7, 3, 4, 6, 8, 9, 10]
//
// Slice (S=4):
// [1, 2, 4, 7]
//
// Unravel Indices [1, 2, 4, 7] (Original Shape 2x5):
//
// Dimension 1 (size 5): [1%5, 2%5, 4%5, 7%5] = [1, 2, 4, 2]
// Dimension 0 (size 2): [1//5, 2//5, 4//5, 7//5] = [0, 0, 0, 1]
// Format Output: Stack Dim 0 and Dim 1 indices:
// Dim 0: [[0], [0], [0], [1]]
// Dim 1: [[1], [2], [4], [2]]
// Concatenated Result: [[0, 1], [0, 2], [0, 4], [1, 2]]
//
absl::StatusOr<mlir::MlirOp> BuildNonzeroShlo(int64_t output_size,
                                              mlir::MlirOp input) {
  mlir::MlirBuilder& builder = input.getBuilder();

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);

  if (output_size == 0 || input_type.getRank() == 0) {
    return MakeConstant(builder, 0, mlir::ElementType::I64,
                        {output_size, input_type.getRank()});
  }

  // STEP 1:
  // The input tensor is first flattened into a 1D array. A
  // boolean mask is generated, with True values marking the positions of
  // non-zero elements.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp input_flattened, FlattenIfNeeded(input));
  mlir::MlirOp input_mask = GetInputMask(input_flattened);

  // STEP 2:
  // Perform stable sort on the flattened mask to get the indices of the true
  // values in the mask.
  // TODO(mkkhanna): Reconsider switching to topk once it is made optimal
  // for small k or large mask. Currently, it does the sort on the entire mask
  // and then takes the first output_size indices.
  SortShloOutputs sort_shlo_outputs = BuildSortShlo(
      input_mask, /*stable=*/true, /*dim=*/0, /*descending=*/true);

  // STEP 3:
  // Slice to take the first output_size indices.
  mlir::MlirOp input_linear_indices =
      stablehlo::Slice(sort_shlo_outputs.indices, /*start_indices=*/{0},
                       /*limit_indices=*/{output_size}, /*stride_indices=*/{1});

  // STEP 4:
  // The linear indices are then converted back into multi-dimensional
  // coordinates.
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp indices,
      UnravelIndices(input, input_linear_indices, mlir::ElementType::I32));
  return indices;
}

}  // namespace torch_tpu
