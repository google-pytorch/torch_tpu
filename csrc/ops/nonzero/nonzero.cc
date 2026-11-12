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

#include "csrc/ops/nonzero/nonzero.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/sort/sort.h"
#include "csrc/ops/stride/stride_helper.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

absl::StatusOr<mlir::MlirOp> FlattenIfNeeded(mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  if (input_type.getRank() == 1) {
    // Already flat, no reshape needed.
    return input;
  }

  const int64_t num_elements = input_type.getNumElements();
  return stablehlo::Reshape(input, {num_elements});
}

// Sorts 1D boolean `mask_1d` descending (`is_stable = true`) and returns
// `i32` indices of shape `mask_1d.shape()`.
mlir::MlirOp SortMaskIndicesI32(mlir::MlirOp mask_1d) {
  const mlir::RankedTensorType mask_type = GetTensorTypeOrDie(mask_1d);
  mlir::MlirBuilder& builder = mask_1d.getBuilder();
  mlir::MlirOp iota_i32 = stablehlo::Iota(
      builder,
      mlir::makeTensorType(builder.getContext(), mask_type.getShape(),
                           mlir::ElementType::I32),
      /*dim=*/0);
  auto comparator = [mask_type](mlir::RegionBuilder& rb) {
    mlir::OpBuilder& op_builder = rb.getOpBuilder();
    stablehlo::buildSortComparisonBody(
        {mask_type.getElementType(), op_builder.getI32Type()},
        stablehlo::ComparisonDirection::GT,
        /*compare_type=*/std::nullopt, &rb.getRegion(), &op_builder);
  };
  auto outputs = stablehlo::Sort(builder, {mask_1d, iota_i32}, comparator,
                                 /*dim=*/0, /*is_stable=*/true);
  return outputs[1];
}

// Unravels 1D `linear_indices` (i32 or i64) into 1D coordinate vectors for each
// dimension of contiguous `shape`.
std::vector<mlir::MlirOp> UnravelLinearIndices1D(
    absl::Span<const int64_t> shape, mlir::MlirOp linear_indices) {
  Strides strides = CalculateStridesContiguous(shape);
  std::vector<mlir::MlirOp> coords;
  coords.reserve(shape.size());

  for (int64_t dim = 0; dim < static_cast<int64_t>(shape.size()); ++dim) {
    mlir::MlirOp stride_constant =
        MakeConstantLike(linear_indices, strides[dim]);
    mlir::MlirOp div_res = stablehlo::Div(linear_indices, stride_constant);
    mlir::MlirOp dim_constant = MakeConstantLike(div_res, shape[dim]);
    coords.push_back(stablehlo::Rem(div_res, dim_constant));
  }
  return coords;
}

absl::StatusOr<mlir::MlirOp> UnravelIndices(absl::Span<const int64_t> shape,
                                            mlir::MlirOp linear_indices) {
  mlir::MlirBuilder& builder = linear_indices.getBuilder();
  std::vector<mlir::MlirOp> dim_coords =
      UnravelLinearIndices1D(shape, linear_indices);
  std::vector<mlir::MlirOp> dim_cols(dim_coords.size());
  for (size_t dim = 0; dim < dim_coords.size(); ++dim) {
    TT_ASSIGN_OR_RETURN(dim_cols[dim], Unsqueeze(dim_coords[dim], /*dim=*/1));
  }
  return stablehlo::Concatenate(builder, dim_cols, /*dim=*/1);
}

bool FitsWithinI32(const mlir::RankedTensorType& type) {
  return type.hasStaticShape() && type.getNumElements() > 0 &&
         type.getNumElements() <= std::numeric_limits<int32_t>::max();
}

absl::StatusOr<mlir::MlirOp> BuildNonzeroGeneralShlo(int64_t output_size,
                                                     mlir::MlirOp input,
                                                     mlir::MlirOp input_mask) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask_1d, FlattenIfNeeded(input_mask));

  const bool use_i32 = FitsWithinI32(input_type);

  mlir::MlirOp sorted_indices;
  if (use_i32) {
    sorted_indices = SortMaskIndicesI32(mask_1d);
  } else {
    SortShloOutputs sort_outputs =
        BuildSortShlo(mask_1d, /*stable=*/true, /*dim=*/0, /*descending=*/true);
    sorted_indices = sort_outputs.indices;
  }

  mlir::MlirOp selected_linear_indices =
      stablehlo::Slice(sorted_indices, /*start_indices=*/{0},
                       /*limit_indices=*/{output_size}, /*stride_indices=*/{1});

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp indices,
      UnravelIndices(input_type.getShape(), selected_linear_indices));
  if (use_i32) {
    indices = stablehlo::ConvertElementType(indices, mlir::ElementType::I64);
  }
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
absl::StatusOr<mlir::MlirOp> BuildNonzeroGetOutputSizeShlo(
    mlir::ElementType element_type, mlir::MlirOp input) {
  ABSL_VLOG(3) << "BuildNonzeroGetOutputSizeShlo input: " << input.ToString();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);

  mlir::MlirOp input_mask = GetNonzeroMask(input);

  const bool reduce_in_i32 =
      element_type == mlir::ElementType::I64 && FitsWithinI32(input_type);
  const mlir::ElementType accum_element_type =
      reduce_in_i32 ? mlir::ElementType::I32 : element_type;

  mlir::MlirOp zero_val =
      MakeScalarConstant(input.getBuilder(), 0, accum_element_type);
  mlir::Type accum_mlir_type =
      mlir::getElementType(input.getBuilder().getContext(), accum_element_type);

  auto reduce_fn = [accum_mlir_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<stablehlo::AddOp>(
        accum_mlir_type, rb.getRegion(), rb.getOpBuilder());
  };

  Dimensions dims(input_type.getRank());
  absl::c_iota(dims, 0);
  mlir::MlirOp output_size =
      BuildReductionShlo(input_mask, dims, accum_mlir_type, zero_val, reduce_fn,
                         ReductionMode::kDropDims);
  if (reduce_in_i32) {
    output_size = stablehlo::ConvertElementType(output_size, element_type);
  }
  ABSL_VLOG(3) << "BuildNonzeroGetOutputSizeShlo output_size: "
               << output_size.ToString();
  return output_size;
}

absl::StatusOr<mlir::MlirOp> BuildNonzeroShlo(int64_t output_size,
                                              mlir::MlirOp input) {
  mlir::MlirBuilder& builder = input.getBuilder();

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const int64_t rank = input_type.getRank();

  if (output_size == 0 || rank == 0) {
    return MakeConstant(builder, 0, mlir::ElementType::I64,
                        {output_size, rank});
  }

  mlir::MlirOp input_mask = GetNonzeroMask(input);
  return BuildNonzeroGeneralShlo(output_size, input, input_mask);
}

absl::StatusOr<mlir::MlirOp> BuildNonzeroStaticShlo(int64_t size,
                                                    int64_t fill_value,
                                                    mlir::MlirOp input) {
  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const int64_t rank = input_type.getRank();
  const int64_t num_elements = input_type.getNumElements();

  if (size == 0 || rank == 0) {
    return MakeConstant(builder, int64_t{0},
                        mlir::makeTensorType(builder.getContext(), {size, rank},
                                             mlir::ElementType::I64));
  }

  if (num_elements == 0) {
    return MakeConstant(builder, fill_value,
                        mlir::makeTensorType(builder.getContext(), {size, rank},
                                             mlir::ElementType::I64));
  }

  TT_ASSIGN_OR_RETURN(mlir::MlirOp input_flattened, FlattenIfNeeded(input));
  mlir::MlirOp input_mask = GetNonzeroMask(input_flattened);

  SortShloOutputs sort_shlo_outputs = BuildSortShlo(
      input_mask, /*stable=*/true, /*dim=*/0, /*descending=*/true);

  const int64_t slice_len = std::min(size, num_elements);
  mlir::MlirOp sorted_mask =
      stablehlo::Slice(sort_shlo_outputs.values, /*start_indices=*/{0},
                       /*limit_indices=*/{slice_len}, /*stride_indices=*/{1});
  mlir::MlirOp input_linear_indices =
      stablehlo::Slice(sort_shlo_outputs.indices, /*start_indices=*/{0},
                       /*limit_indices=*/{slice_len}, /*stride_indices=*/{1});

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp unraveled_indices,
      UnravelIndices(input_type.getShape(), input_linear_indices));

  mlir::RankedTensorType mask_2d_type =
      mlir::makeTensorType(builder.getContext(), {slice_len, rank},
                           GetTensorTypeOrDie(sorted_mask).getElementType());
  mlir::MlirOp mask_2d =
      stablehlo::BroadcastInDim(mask_2d_type, sorted_mask, {0});
  mlir::MlirOp fill_slice = MakeConstantLike(unraveled_indices, fill_value);
  mlir::MlirOp result =
      stablehlo::Select(mask_2d, unraveled_indices, fill_slice);

  if (size > slice_len) {
    mlir::MlirOp pad = MakeConstant(
        builder, fill_value,
        mlir::makeTensorType(builder.getContext(), {size - slice_len, rank},
                             mlir::ElementType::I64));
    result = stablehlo::Concatenate(builder, {result, pad}, /*dim=*/0);
  }
  return result;
}

}  // namespace torch_tpu
