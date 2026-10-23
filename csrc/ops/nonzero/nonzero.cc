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

constexpr int64_t kMinUniformRowWidth = 256;
constexpr int64_t kMinUniformElements = 4096;
constexpr int64_t kMinUniformRows = 4;

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
  Strides strides =
      CalculateStridesContiguous(Dimensions(shape.begin(), shape.end()));
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

bool IsWideRowTensorWithinI32(const mlir::RankedTensorType& type) {
  return type.hasStaticShape() && type.getRank() >= 3 &&
         type.getShape().back() >= kMinUniformRowWidth &&
         type.getNumElements() >= kMinUniformElements &&
         type.getNumElements() <= std::numeric_limits<int32_t>::max();
}

absl::StatusOr<mlir::MlirOp> BuildNonzeroGeneralShlo(int64_t output_size,
                                                     mlir::MlirOp input,
                                                     mlir::MlirOp input_mask) {
  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask_1d, FlattenIfNeeded(input_mask));

  const bool use_i32 = IsWideRowTensorWithinI32(input_type);

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

  std::vector<mlir::MlirOp> dim_coords =
      UnravelLinearIndices1D(input_type.getShape(), selected_linear_indices);
  std::vector<mlir::MlirOp> dim_cols(dim_coords.size());
  for (size_t dim = 0; dim < dim_coords.size(); ++dim) {
    TT_ASSIGN_OR_RETURN(dim_cols[dim], Unsqueeze(dim_coords[dim], /*dim=*/1));
  }

  mlir::MlirOp indices = stablehlo::Concatenate(builder, dim_cols, /*dim=*/1);
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

  const bool reduce_in_i32 = element_type == mlir::ElementType::I64 &&
                             IsWideRowTensorWithinI32(input_type);
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

  if (rank >= 3 && input_type.hasStaticShape() &&
      input_type.getNumElements() >= kMinUniformElements &&
      input_type.getNumElements() <= std::numeric_limits<int32_t>::max()) {
    const mlir::ArrayRef<int64_t> input_shape = input_type.getShape();
    const int64_t row_width = input_shape[rank - 1];
    const int64_t num_rows = input_type.getNumElements() / row_width;

    if (row_width >= kMinUniformRowWidth && num_rows >= kMinUniformRows &&
        output_size >= row_width && output_size % row_width == 0) {
      const int64_t selected_rows = output_size / row_width;
      const mlir::Type i1_type = builder.getOpBuilder().getI1Type();
      mlir::MlirOp col0;
      mlir::MlirOp is_row_uniform =
          SliceMaskFirstColumnAndCheckRowUniform(input_mask, col0);

      const mlir::RankedTensorType out_type = mlir::makeTensorType(
          builder.getContext(), {output_size, rank}, mlir::ElementType::I64);
      auto if_op = stablehlo::IfOp::create(
          builder.getOpBuilder(), builder.getLoc(), mlir::TypeRange{out_type},
          is_row_uniform.getValue());

      {
        mlir::RegionBuilder true_rb(builder, if_op->getRegion(0));
        mlir::MlirOp true_col0(true_rb, col0.getValue());
        mlir::MlirOp col0_1d = stablehlo::Reshape(
            mlir::makeTensorType(true_rb.getContext(), {num_rows}, i1_type),
            true_col0);
        mlir::MlirOp sorted_rows = SortMaskIndicesI32(col0_1d);
        mlir::MlirOp selected_row_indices = stablehlo::Slice(
            sorted_rows, /*start_indices=*/{0},
            /*limit_indices=*/{selected_rows}, /*stride_indices=*/{1});

        std::vector<mlir::MlirOp> row_coords = UnravelLinearIndices1D(
            absl::MakeConstSpan(input_shape.data(), rank - 1),
            selected_row_indices);

        const mlir::RankedTensorType row_grid_i32_type = mlir::makeTensorType(
            true_rb.getContext(), {selected_rows, row_width},
            mlir::ElementType::I32);
        const mlir::RankedTensorType col_out_i32_type = mlir::makeTensorType(
            true_rb.getContext(), {output_size, 1}, mlir::ElementType::I32);

        std::vector<mlir::MlirOp> all_cols;
        all_cols.reserve(rank);
        for (mlir::MlirOp& row_coord : row_coords) {
          mlir::MlirOp expanded = stablehlo::BroadcastInDim(
              row_grid_i32_type, row_coord, llvm::ArrayRef<int64_t>{0});
          all_cols.push_back(stablehlo::Reshape(col_out_i32_type, expanded));
        }

        mlir::MlirOp col_iota =
            stablehlo::Iota(true_rb, row_grid_i32_type, /*dim=*/1);
        all_cols.push_back(stablehlo::Reshape(col_out_i32_type, col_iota));

        mlir::MlirOp concat_i32 =
            stablehlo::Concatenate(true_rb, all_cols, /*dim=*/1);
        mlir::MlirOp true_out =
            stablehlo::ConvertElementType(concat_i32, mlir::ElementType::I64);
        stablehlo::Return(true_rb, {true_out});
      }

      {
        mlir::RegionBuilder false_rb(builder, if_op->getRegion(1));
        TT_ASSIGN_OR_RETURN(
            mlir::MlirOp false_out,
            BuildNonzeroGeneralShlo(
                output_size, mlir::MlirOp(false_rb, input.getValue()),
                mlir::MlirOp(false_rb, input_mask.getValue())));
        stablehlo::Return(false_rb, {false_out});
      }

      return mlir::MlirOp(builder, if_op.getResult(0));
    }
  }

  return BuildNonzeroGeneralShlo(output_size, input, input_mask);
}

}  // namespace torch_tpu
