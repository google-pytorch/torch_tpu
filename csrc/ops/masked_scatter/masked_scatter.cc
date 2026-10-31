// Copyright 2026 Google LLC
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

#include "csrc/ops/masked_scatter/masked_scatter.h"

#include <cstddef>
#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "csrc/ops/cumsum/cumsum.h"
#include "csrc/ops/index_select/index_select.h"
#include "csrc/ops/op_builder_utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// A helper struct describing how a mask in the masked_scatter operation
// should partition the input tensor into rows for more efficient
// implementation.
struct RowPartition {
  // Number of trailing dimensions folded into one row.
  int64_t num_folded_trailing_dims;
  // Number of elements per row, i.e. the product of the folded dimensions.
  int64_t row_size;
  // Number of rows, the number of elements divided by `row_size`.
  int64_t num_rows;
};

bool IsBroadcastOrUnitOperandDim(const mlir::ArrayRef<int64_t> operand_shape,
                                 const mlir::ArrayRef<int64_t> bcast_dims,
                                 const int64_t mask_dim) {
  for (size_t i = 0; i < bcast_dims.size(); ++i) {
    if (bcast_dims[i] == mask_dim) {
      return operand_shape[i] == 1;
    }
  }
  return true;
}

// Narrows trailing dimensions of `mask` that were broadcast from size 1 (e.g.
// via `expand_as(inputs_embeds)` lowered to `stablehlo.broadcast_in_dim` in
// `torch.compile`), restoring their size to 1 so `ComputeRowPartition` folds
// whole feature rows instead of decomposing into scalar gathers.
mlir::MlirOp NarrowTrailingBroadcastMask(mlir::MlirOp mask) {
  auto bcast = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
      mask.getValue().getDefiningOp());
  if (!bcast) {
    return mask;
  }
  const mlir::RankedTensorType mask_type = GetTensorTypeOrDie(mask);
  const mlir::RankedTensorType operand_type =
      mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
  if (!operand_type || !mask_type.hasStaticShape() ||
      !operand_type.hasStaticShape()) {
    return mask;
  }
  const mlir::ArrayRef<int64_t> mask_shape = mask_type.getShape();
  const mlir::ArrayRef<int64_t> operand_shape = operand_type.getShape();
  const mlir::ArrayRef<int64_t> bcast_dims = bcast.getBroadcastDimensions();
  const int64_t mask_rank = mask_shape.size();

  Dimensions start_indices(mask_rank, 0);
  Dimensions limit_indices = CopyIntVector(mask_shape);
  Strides strides(mask_rank, 1);
  bool narrowed = false;

  for (int64_t d = mask_rank - 1; d >= 0; --d) {
    if (mask_shape[d] <= 1) {
      continue;
    }
    if (!IsBroadcastOrUnitOperandDim(operand_shape, bcast_dims, d)) {
      break;
    }
    limit_indices[d] = 1;
    narrowed = true;
  }

  if (!narrowed) {
    return mask;
  }
  return mlir::stablehlo::Slice(mask, start_indices, limit_indices, strides);
}

// Calculates the RowPartitions of a tensor of shape `input_shape` into the
// rows that a mask of shape `mask_shape` selects.
RowPartition ComputeRowPartition(const mlir::ArrayRef<int64_t> input_shape,
                                 const mlir::ArrayRef<int64_t> mask_shape) {
  const int64_t rank = input_shape.size();
  const int64_t mask_rank = mask_shape.size();
  int64_t num_folded_trailing_dims = 0;
  int64_t row_size = 1;
  for (int64_t i = 0; i < rank; ++i) {
    const int64_t mask_dim = i < mask_rank ? mask_shape[mask_rank - 1 - i] : 1;
    if (mask_dim != 1) {
      break;
    }
    row_size *= input_shape[rank - 1 - i];
    ++num_folded_trailing_dims;
  }
  int64_t numel = 1;
  for (const int64_t dim : input_shape) {
    numel *= dim;
  }
  return RowPartition{.num_folded_trailing_dims = num_folded_trailing_dims,
                      .row_size = row_size,
                      .num_rows = numel / row_size};
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildMaskedScatterShlo(mlir::MlirOp input,
                                                    mlir::MlirOp mask,
                                                    mlir::MlirOp source) {
  mlir::MlirBuilder& builder = input.getBuilder();
  mask = NarrowTrailingBroadcastMask(mask);
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const mlir::RankedTensorType mask_type = GetTensorTypeOrDie(mask);
  const mlir::RankedTensorType source_type = GetTensorTypeOrDie(source);

  const int64_t rank = input_type.getRank();
  const int64_t numel = input_type.getNumElements();
  if (numel == 0) {
    return input;
  }

  // 1. Partition `input` and `source` into rows of the same size.
  const mlir::ArrayRef<int64_t> input_shape = input_type.getShape();
  const RowPartition input_partition =
      ComputeRowPartition(input_shape, mask_type.getShape());
  const int64_t source_numel = source_type.getNumElements();
  const int64_t num_source_rows = source_numel / input_partition.row_size;
  if (num_source_rows == 0) {
    // `source` holds less than a row, so selecting any row would already
    // overrun it, and the gather below needs a non-empty operand.
    return input;
  }

  // 2. Reshape the input into rows and the mask into one element per row,
  // broadcasting the mask only along the dimensions that are not folded.
  Dimensions row_mask_shape = CopyIntVector(input_shape);
  for (int64_t i = rank - input_partition.num_folded_trailing_dims; i < rank;
       ++i) {
    row_mask_shape[i] = 1;
  }
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp broadcast_mask,
      BroadcastIfNeeded(mask, absl::MakeConstSpan(row_mask_shape)));

  mlir::MlirOp input_rows = mlir::stablehlo::Reshape(
      mlir::makeTensorType(builder.getContext(),
                           {input_partition.num_rows, input_partition.row_size},
                           input_type.getElementType()),
      input);
  mlir::MlirOp row_mask = mlir::stablehlo::Reshape(
      mlir::makeTensorType(builder.getContext(), {input_partition.num_rows},
                           mask_type.getElementType()),
      broadcast_mask);

  // 3. Reshape the source into rows, dropping a partial trailing one. Rows of
  // `source` are consumed in order, one per selected row, so a legal call never
  // reads past its last whole row.
  mlir::MlirOp source_rows = source;
  if (num_source_rows * input_partition.row_size != source_numel) {
    mlir::MlirOp flat_source = mlir::stablehlo::Reshape(
        mlir::makeTensorType(builder.getContext(), {source_numel},
                             source_type.getElementType()),
        source_rows);
    source_rows = mlir::stablehlo::Slice(
        flat_source, Dimensions{0},
        Dimensions{num_source_rows * input_partition.row_size}, Dimensions{1});
  }
  source_rows = mlir::stablehlo::Reshape(
      mlir::makeTensorType(builder.getContext(),
                           {num_source_rows, input_partition.row_size},
                           source_type.getElementType()),
      source_rows);

  // 4. Convert the row mask from bool to i32 and prefix sum it. Keep the source
  // offsets in i32: TPU vector registers cannot materialize the i64 values with
  // `Unsupported conversion from vmreg/vreg to U64`.
  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  mlir::MlirOp row_mask_i32 =
      mlir::stablehlo::ConvertElementType(row_mask, i32_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp selected_row_count,
                      BuildCumsumShlo(0, mlir::ElementType::I32, row_mask_i32));

  // 5. Subtract one to make the source row indices zero based. Unselected rows
  // keep the index consumed before them, or -1 before the first selected row;
  // `stablehlo.gather` clamps that and the select below discards the row.
  mlir::MlirOp ones =
      MakeConstantLike(selected_row_count, static_cast<int32_t>(1));
  mlir::MlirOp source_row_indices =
      mlir::stablehlo::Subtract(selected_row_count, ones);

  // 6. Gather one source row per row of the input.
  mlir::MlirOp gathered_rows;
  const mlir::Type elem_type = input_type.getElementType();
  if (input_partition.row_size >= 128 && num_source_rows > 0 &&
      num_source_rows <= 1024 &&
      (elem_type.isBF16() || elem_type.isF16() || elem_type.isF32())) {
    const auto mat_i32_type = mlir::makeTensorType(
        builder.getContext(), {input_partition.num_rows, num_source_rows},
        i32_type);
    mlir::MlirOp indices_2d = mlir::stablehlo::BroadcastInDim(
        mat_i32_type, source_row_indices, llvm::ArrayRef<int64_t>{0});
    mlir::MlirOp iota_2d = mlir::stablehlo::Iota(builder, mat_i32_type, 1);
    mlir::MlirOp eq_mask = mlir::stablehlo::Compare(
        indices_2d, iota_2d, mlir::stablehlo::ComparisonDirection::EQ);
    mlir::MlirOp indicator =
        mlir::stablehlo::ConvertElementType(eq_mask, elem_type);
    auto dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        &builder.getContext(),
        /*lhs_batching_dimensions=*/{},
        /*rhs_batching_dimensions=*/{},
        /*lhs_contracting_dimensions=*/{1},
        /*rhs_contracting_dimensions=*/{0});
    mlir::ArrayAttr precision_config;
    if (elem_type.isF16() || elem_type.isF32()) {
      const mlir::stablehlo::Precision precisions[2] = {
          mlir::stablehlo::Precision::
              HIGHEST,  // EXPLICIT_PRECISION_OK=one_hot_lhs
          mlir::stablehlo::Precision::
              HIGHEST,  // EXPLICIT_PRECISION_OK=exact_row_gather
      };
      precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
          &builder.getContext(), precisions);
    }
    gathered_rows = mlir::stablehlo::DotGeneral(
        indicator, source_rows, dot_dims, /*precisionConfig=*/precision_config);
  } else {
    gathered_rows = BuildIndexSelectShlo(source_rows, 0, source_row_indices);
  }

  // 7. Take the gathered rows where the mask is true, and reshape back.
  mlir::MlirOp row_mask_2d = mlir::stablehlo::BroadcastInDim(
      mlir::makeTensorType(builder.getContext(),
                           {input_partition.num_rows, input_partition.row_size},
                           mask_type.getElementType()),
      row_mask, llvm::ArrayRef<int64_t>{0});
  mlir::MlirOp output_rows =
      mlir::stablehlo::Select(row_mask_2d, gathered_rows, input_rows);

  return mlir::stablehlo::Reshape(input_type, output_rows);
}

}  // namespace torch_tpu
