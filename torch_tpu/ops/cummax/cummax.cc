/*
 * Copyright 2026 Google LLC
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

#include "torch_tpu/ops/cummax/cummax.h"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>

#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace {

namespace stablehlo = mlir::stablehlo;

struct ReduceWindowAttributes {
  llvm::SmallVector<int64_t> window_dimensions;
  mlir::DenseI64ArrayAttr window_strides;
  mlir::DenseI64ArrayAttr base_dilations;
  mlir::DenseI64ArrayAttr window_dilations;
  mlir::DenseIntElementsAttr padding;
};

absl::StatusOr<ReduceWindowAttributes> GetReduceWindowAttributes(
    mlir::MlirBuilder& builder, mlir::MlirOp input, int64_t dim) {
  mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const int64_t rank = input_type.getRank();
  TT_ASSIGN_OR_RETURN(const int64_t normalized_dim, SafeWrapDim(dim, rank));

  stablehlo::Dimensions input_dims = GetDimensions(input);
  int64_t dim_size = input_dims[normalized_dim].size;

  llvm::SmallVector<int64_t> window_dimensions(rank, 1);
  window_dimensions[normalized_dim] = dim_size;

  llvm::SmallVector<int64_t> window_strides(rank, 1);
  llvm::SmallVector<int64_t> base_dilations(rank, 1);
  llvm::SmallVector<int64_t> window_dilations(rank, 1);

  // Padding: (dim - 1) values to low padding on `dim`.
  llvm::SmallVector<int64_t> pad_values(rank * 2, 0);
  pad_values[2 * normalized_dim] = dim_size - 1;

  return ReduceWindowAttributes{
      .window_dimensions = window_dimensions,
      .window_strides =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), window_strides),
      .base_dilations =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), base_dilations),
      .window_dilations =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), window_dilations),
      .padding = mlir::DenseIntElementsAttr::get(
          makeTensorType(builder.getContext(), {rank, 2},
                         builder.getOpBuilder().getI64Type()),
          pad_values)};
}

}  // namespace

absl::StatusOr<CummaxOutputs> BuildCummaxShlo(const int64_t dim,
                                              mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const int64_t rank = input_type.getRank();
  TT_ASSIGN_OR_RETURN(const int64_t normalized_dim, SafeWrapDim(dim, rank));

  mlir::MlirOp current_input = input;
  int64_t current_dim = normalized_dim;
  llvm::SmallVector<int64_t> transpose_perm(rank);
  llvm::SmallVector<int64_t> inv_transpose_perm(rank);

  bool needs_transpose = (normalized_dim != rank - 1);
  if (needs_transpose) {
    std::iota(transpose_perm.begin(), transpose_perm.end(), 0);
    std::swap(transpose_perm[normalized_dim], transpose_perm[rank - 1]);

    for (int i = 0; i < rank; ++i) {
      inv_transpose_perm[transpose_perm[i]] = i;
    }
    current_input = mlir::stablehlo::Transpose(current_input, transpose_perm);
    current_dim = rank - 1;
  }
  const mlir::RankedTensorType current_input_type =
      GetTensorTypeOrDie(current_input);
  mlir::Type element_type = current_input_type.getElementType();
  mlir::MlirBuilder& builder = input.getBuilder();

  stablehlo::Dimensions current_input_dims = GetDimensions(current_input);

  mlir::MlirOp indices =
      stablehlo::IotaLike(current_input, current_dim, mlir::ElementType::I32);
  for (size_t i = 0; i < current_input_dims.size(); ++i) {
    if (current_input_dims[i].boundOp.has_value()) {
      mlir::MlirOp boundOp =
          mlir::MlirOp(builder, *current_input_dims[i].boundOp);
      auto dimSize = stablehlo::GetDimensionSize(
          boundOp, current_input_dims[i].boundOpDim);
      indices = stablehlo::SetDimensionSize(indices, dimSize, i);
    }
  }
  const mlir::Type indices_element_type =
      GetTensorTypeOrDie(indices).getElementType();

  mlir::Type original_element_type = element_type;
  bool needs_cast_back = false;
  if (element_type.isF64()) {
    element_type = builder.getOpBuilder().getF32Type();
    current_input =
        mlir::stablehlo::ConvertElementType(current_input, element_type);
    needs_cast_back = true;
  } else if (element_type.isInteger(64)) {
    element_type = builder.getOpBuilder().getI32Type();
    current_input =
        mlir::stablehlo::ConvertElementType(current_input, element_type);
    needs_cast_back = true;
  }

  mlir::Attribute max_init_attr =
      GetMinFiniteValueAttr(element_type, builder.getOpBuilder());
  mlir::DenseElementsAttr value_init_attr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({}, element_type), max_init_attr);
  mlir::MlirOp value_init = stablehlo::Constant(builder, value_init_attr);

  mlir::MlirOp index_init =
      MakeScalarConstant(builder, 0, mlir::ElementType::I32);

  TT_ASSIGN_OR_RETURN(
      (auto [window_dimensions, window_strides, base_dilations,
             window_dilations, padding]),
      GetReduceWindowAttributes(builder, current_input, current_dim));

  auto reduce_body = [&](mlir::RegionBuilder& body) {
    mlir::OpBuilder op_builder = body.getOpBuilder();
    mlir::Region& region = body.getRegion();
    if (region.getBlocks().empty()) op_builder.createBlock(&region);
    mlir::Block* block = &region.getBlocks().front();

    mlir::Type value_type = mlir::RankedTensorType::get({}, element_type);
    mlir::Type index_type =
        mlir::RankedTensorType::get({}, indices_element_type);
    mlir::Location loc = body.getLoc();
    block->addArguments({value_type, index_type}, {loc, loc});
    block->addArguments({value_type, index_type}, {loc, loc});

    auto lhs_value = block->getArgument(0);
    auto lhs_index = block->getArgument(1);
    auto rhs_value = block->getArgument(2);
    auto rhs_index = block->getArgument(3);

    auto gt_pred =
        stablehlo::CompareOp::create(op_builder, loc, lhs_value, rhs_value,
                                     stablehlo::ComparisonDirection::GT)
            .getResult();

    auto eq_pred =
        stablehlo::CompareOp::create(op_builder, loc, lhs_value, rhs_value,
                                     stablehlo::ComparisonDirection::EQ)
            .getResult();
    auto max_index =
        stablehlo::MaxOp::create(op_builder, loc, lhs_index, rhs_index)
            .getResult();

    auto selected_value = stablehlo::SelectOp::create(op_builder, loc, gt_pred,
                                                      lhs_value, rhs_value)
                              .getResult();

    auto argmax_index = stablehlo::SelectOp::create(op_builder, loc, gt_pred,
                                                    lhs_index, rhs_index)
                            .getResult();

    auto final_index = stablehlo::SelectOp::create(op_builder, loc, eq_pred,
                                                   max_index, argmax_index)
                           .getResult();

    stablehlo::ReturnOp::create(op_builder, loc,
                                mlir::ValueRange{selected_value, final_index});
  };

  auto reduce_window_op = stablehlo::ReduceWindow(
      builder, /*inputs=*/{current_input, indices},
      /*init_values=*/{value_init, index_init}, reduce_body, window_dimensions,
      window_strides, base_dilations, window_dilations, padding);

  mlir::MlirOp result_values = reduce_window_op[0];
  mlir::MlirOp result_indices = reduce_window_op[1];

  if (needs_transpose) {
    result_values =
        mlir::stablehlo::Transpose(result_values, inv_transpose_perm);
    result_indices =
        mlir::stablehlo::Transpose(result_indices, inv_transpose_perm);
  }
  if (needs_cast_back) {
    result_values = mlir::stablehlo::ConvertElementType(result_values,
                                                        original_element_type);
  }

  // TPU ReduceWindow does not support int64 reduction yet, so we use I32 for
  // indices inside the reduction block, and explicitly cast back to I64 here.
  mlir::MlirOp final_indices = mlir::stablehlo::ConvertElementType(
      result_indices, mlir::ElementType::I64);

  return CummaxOutputs{.values = result_values, .indices = final_indices};
}

}  // namespace torch_tpu
