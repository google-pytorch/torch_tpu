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

#include "csrc/ops/sort/sort.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/Builders.h"
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

bool IsInt32BoundedSortKey(mlir::Value val, int depth = 0) {
  if (!val || depth > 8) {
    return false;
  }
  auto* def = val.getDefiningOp();
  if (!def) {
    return false;
  }
  if (auto conv = mlir::dyn_cast<stablehlo::ConvertOp>(def)) {
    mlir::Type src_type =
        mlir::getElementTypeOrSelf(conv.getOperand().getType());
    if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(src_type)) {
      return int_type.getWidth() <= 32 &&
             (int_type.getWidth() < 32 || !int_type.isUnsigned());
    }
    return false;
  }
  if (auto iota = mlir::dyn_cast<stablehlo::IotaOp>(def)) {
    auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(iota.getType());
    const uint64_t iota_dim = iota.getIotaDimension();
    if (out_type && out_type.hasStaticShape() &&
        iota_dim < static_cast<uint64_t>(out_type.getRank())) {
      return out_type.getDimSize(iota_dim) <=
             static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    }
    return false;
  }
  if (auto cst = mlir::dyn_cast<stablehlo::ConstantOp>(def)) {
    if (auto dense =
            mlir::dyn_cast<mlir::DenseIntElementsAttr>(cst.getValue())) {
      for (const llvm::APInt& v : dense.getValues<llvm::APInt>()) {
        const int64_t s = v.getSExtValue();
        if (s < std::numeric_limits<int32_t>::min() ||
            s > std::numeric_limits<int32_t>::max()) {
          return false;
        }
      }
      return true;
    }
    return false;
  }
  if (auto res = mlir::dyn_cast<mlir::OpResult>(val)) {
    if (auto sort_op = mlir::dyn_cast<stablehlo::SortOp>(def)) {
      if (res.getResultNumber() == 1) {
        auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
        const uint64_t sdim = sort_op.getDimension();
        if (out_type && out_type.hasStaticShape() &&
            sdim < static_cast<uint64_t>(out_type.getRank())) {
          return out_type.getDimSize(sdim) <=
                 static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
        }
      }
      return false;
    }
  }
  if (auto reshape = mlir::dyn_cast<stablehlo::ReshapeOp>(def)) {
    return IsInt32BoundedSortKey(reshape.getOperand(), depth + 1);
  }
  if (auto transpose = mlir::dyn_cast<stablehlo::TransposeOp>(def)) {
    return IsInt32BoundedSortKey(transpose.getOperand(), depth + 1);
  }
  if (auto bcast = mlir::dyn_cast<stablehlo::BroadcastInDimOp>(def)) {
    return IsInt32BoundedSortKey(bcast.getOperand(), depth + 1);
  }
  if (auto slice = mlir::dyn_cast<stablehlo::SliceOp>(def)) {
    return IsInt32BoundedSortKey(slice.getOperand(), depth + 1);
  }
  return false;
}

}  // namespace

SortShloOutputs BuildSortShlo(mlir::MlirOp input_op, bool stable, int64_t dim,
                              bool descending) {
  const mlir::RankedTensorType inputType = GetTensorTypeOrDie(input_op);
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const bool use_i32_indices =
      dim >= 0 && dim < inputType.getRank() &&
      inputType.getShape()[dim] != mlir::ShapedType::kDynamic &&
      inputType.getShape()[dim] <=
          static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  const bool use_i32_key =
      GetElementTypeOrDie(input_op) == mlir::ElementType::I64 &&
      IsInt32BoundedSortKey(input_op.getValue());

  mlir::MlirOp sort_key = use_i32_key ? stablehlo::ConvertElementType(
                                            input_op, mlir::ElementType::I32)
                                      : input_op;
  const mlir::Type sort_key_elem_type =
      GetTensorTypeOrDie(sort_key).getElementType();
  const mlir::ElementType index_elem_type =
      use_i32_indices ? mlir::ElementType::I32 : mlir::ElementType::I64;

  mlir::MlirOp indices =
      stablehlo::Iota(builder,
                      makeTensorType(builder.getContext(), inputType.getShape(),
                                     index_elem_type),
                      dim);
  auto comparator = [sort_key_elem_type, use_i32_indices,
                     descending](mlir::RegionBuilder& rb) {
    mlir::OpBuilder& op_builder = rb.getOpBuilder();
    std::optional<llvm::StringRef> compare_type = std::nullopt;
    if (mlir::isa<mlir::FloatType>(sort_key_elem_type)) {
      compare_type = "TOTALORDER";
    }
    const mlir::Type idx_mlir_type =
        use_i32_indices ? op_builder.getI32Type() : op_builder.getI64Type();
    stablehlo::buildSortComparisonBody(
        {sort_key_elem_type, idx_mlir_type},
        descending ? stablehlo::ComparisonDirection::GT
                   : stablehlo::ComparisonDirection::LT,
        compare_type, &rb.getRegion(), &op_builder);
  };
  auto outputs =
      stablehlo::Sort(builder, {sort_key, indices}, comparator, dim, stable);
  mlir::MlirOp sorted_values =
      use_i32_key
          ? stablehlo::ConvertElementType(outputs[0], mlir::ElementType::I64)
          : outputs[0];
  mlir::MlirOp sorted_indices =
      use_i32_indices
          ? stablehlo::ConvertElementType(outputs[1], mlir::ElementType::I64)
          : outputs[1];
  return SortShloOutputs{.values = sorted_values, .indices = sorted_indices};
}

}  // namespace torch_tpu
