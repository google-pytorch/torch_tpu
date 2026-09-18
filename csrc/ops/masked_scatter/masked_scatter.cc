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
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/cat/cat.h"
#include "csrc/ops/cumsum/cumsum.h"
#include "csrc/ops/index_select/index_select.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/scan_builder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

mlir::MlirOp FlattenTensor(mlir::MlirOp input) {
  const auto type = GetTensorTypeOrDie(input);
  if (type.hasStaticShape()) {
    return Flatten(input);
  }
  if (type.getRank() <= 1) {
    return input;
  }
  auto& builder = input.getBuilder();
  mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  mlir::stablehlo::Dimensions dims = GetDimensions(input);
  int64_t total_bound = 1;
  mlir::MlirOp total_numel = MakeScalarConstant(builder, 1, i32_type);
  for (size_t i = 0; i < dims.size(); ++i) {
    total_bound *= dims[i].size;
    mlir::MlirOp dim_size;
    if (dims[i].boundOp.has_value()) {
      mlir::MlirOp bound_op(input.getBuilder(), *dims[i].boundOp);
      dim_size =
          mlir::stablehlo::GetDimensionSize(bound_op, dims[i].boundOpDim);
    } else {
      dim_size = MakeScalarConstant(builder, static_cast<int32_t>(dims[i].size),
                                    i32_type);
    }
    total_numel = mlir::stablehlo::Mul(total_numel, dim_size);
  }

  mlir::MlirOp shape_1d_numel = mlir::stablehlo::Reshape(total_numel, {1});
  auto bounds_attr = mlir::stablehlo::TypeExtensionsAttr::get(
      &builder.getContext(), {total_bound});
  auto flat_type = mlir::RankedTensorType::get(
      {mlir::ShapedType::kDynamic}, type.getElementType(), bounds_attr);
  return mlir::stablehlo::DynamicReshape(flat_type, input, shape_1d_numel);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildMaskedScatterShlo(mlir::MlirOp input,
                                                    mlir::MlirOp mask,
                                                    mlir::MlirOp source) {
  auto& builder = input.getBuilder();

  // 0. Apply broadcasting between input and mask if needed
  TT_ASSIGN_OR_RETURN((auto [broadcasted_input, broadcasted_mask]),
                      ApplyBroadcastIfNeeded(input, mask));
  const auto broadcasted_input_type = GetTensorTypeOrDie(broadcasted_input);

  mlir::Type i32_type = builder.getOpBuilder().getI32Type();

  mlir::MlirOp flat_input;
  mlir::MlirOp flat_mask;
  mlir::MlirOp flat_source = FlattenTensor(source);

  mlir::stablehlo::Dimensions broadcasted_dims =
      GetDimensions(broadcasted_input);
  const bool is_dynamic = !broadcasted_input_type.hasStaticShape();

  std::vector<mlir::MlirOp> shape_1d_ops;
  shape_1d_ops.reserve(broadcasted_dims.size());

  if (is_dynamic) {
    int64_t total_bound = 1;
    mlir::MlirOp total_numel = MakeScalarConstant(builder, 1, i32_type);
    for (size_t i = 0; i < broadcasted_dims.size(); ++i) {
      total_bound *= broadcasted_dims[i].size;
      mlir::MlirOp dim_size;
      if (broadcasted_dims[i].boundOp.has_value()) {
        mlir::MlirOp bound_op(input.getBuilder(), *broadcasted_dims[i].boundOp);
        dim_size = mlir::stablehlo::GetDimensionSize(
            bound_op, broadcasted_dims[i].boundOpDim);
      } else {
        dim_size = MakeScalarConstant(
            builder, static_cast<int32_t>(broadcasted_dims[i].size), i32_type);
      }
      total_numel = mlir::stablehlo::Mul(total_numel, dim_size);
      shape_1d_ops.push_back(mlir::stablehlo::Reshape(dim_size, {1}));
    }

    mlir::MlirOp shape_1d_numel = mlir::stablehlo::Reshape(total_numel, {1});
    auto bounds_attr = mlir::stablehlo::TypeExtensionsAttr::get(
        &builder.getContext(), {total_bound});

    auto flat_input_type = mlir::RankedTensorType::get(
        {mlir::ShapedType::kDynamic}, broadcasted_input_type.getElementType(),
        bounds_attr);
    auto flat_mask_type = mlir::RankedTensorType::get(
        {mlir::ShapedType::kDynamic}, builder.getOpBuilder().getI1Type(),
        bounds_attr);

    flat_input = mlir::stablehlo::DynamicReshape(
        flat_input_type, broadcasted_input, shape_1d_numel);
    flat_mask = mlir::stablehlo::DynamicReshape(
        flat_mask_type, broadcasted_mask, shape_1d_numel);
  } else {
    flat_input = Flatten(broadcasted_input);
    flat_mask = Flatten(broadcasted_mask);
  }

  // 2. Convert flat_mask from bool to i32. Keep the source offsets in i32:
  // TPU vector registers cannot materialize the i64 values with
  // `Unsupported conversion from vmreg/vreg to U64`.
  mlir::MlirOp mask_i32 =
      mlir::stablehlo::ConvertElementType(flat_mask, i32_type);

  // 3. Compute prefix sum (cumsum) of mask values on device
  mlir::MlirOp prefix_sum;
  if (is_dynamic) {
    mlir::MlirOp carry_init = MakeScalarConstant(builder, 0, i32_type);
    MultiInputScanBodyBuilder body_builder =
        [](mlir::OpBuilder& op_builder, mlir::Location loc,
           mlir::ValueRange input_slices, mlir::Value /*index*/,
           mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
      llvm::SmallVector<mlir::Value> sum = {
          mlir::stablehlo::AddOp::create(op_builder, loc, input_slices[0],
                                         carries[0])
              .getResult()};
      return ScanBodyResults{sum, sum};
    };
    TT_ASSIGN_OR_RETURN(
        const DynamicMlirOpResults results,
        BuildScanShlo(
            builder, {mask_i32}, 0, /*num_scan_inputs=*/1,
            /*carry_inits=*/{carry_init}, /*output_inits=*/{mask_i32},
            std::move(body_builder),
            ScanOptions{.should_squeeze = true, .is_associative = false}));
    prefix_sum = results[1];
  } else {
    TT_ASSIGN_OR_RETURN(prefix_sum,
                        BuildCumsumShlo(0, mlir::ElementType::I32, mask_i32));
  }

  // 4. Compute indices for gather: gather_indices = prefix_sum - 1
  mlir::MlirOp constant_1 =
      MakeScalarConstant(builder, static_cast<int32_t>(1), i32_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_1,
                      BroadcastIfNeeded(constant_1, prefix_sum));
  mlir::MlirOp gather_indices =
      mlir::stablehlo::Subtract(prefix_sum, broadcasted_1);

  // 5. Perform dynamic gather from source using BuildIndexSelectShlo helper
  // along dim 0 (returns raw MlirOp!)
  mlir::MlirOp gathered_source =
      BuildIndexSelectShlo(flat_source, 0, gather_indices);

  // 6. Elementwise selection: if mask is True, take gathered_source, else keep
  // flat_input
  mlir::MlirOp flat_output =
      mlir::stablehlo::Select(flat_mask, gathered_source, flat_input);

  // 7. Reshape flat_output back to the original ranked shape of broadcasted
  // input
  if (is_dynamic) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp target_shape_tensor,
                        BuildCatShlo(absl::MakeSpan(shape_1d_ops), 0));
    return mlir::stablehlo::DynamicReshape(broadcasted_input_type, flat_output,
                                           target_shape_tensor);
  }

  return mlir::stablehlo::Reshape(broadcasted_input_type, flat_output);
}

}  // namespace torch_tpu
