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

#include "csrc/mlir/passes/canonicalize_index_arithmetic/canonicalize_index_arithmetic.h"  // NOLINT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // NOLINT
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"  // NOLINT
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"  // NOLINT
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// GENERIC: hardware TPU vector register height.
[[maybe_unused]] constexpr int64_t kTpuSublaneCount = 8;
// GENERIC: hardware TPU vector register width.
[[maybe_unused]] constexpr int64_t kTpuLaneWidth = 128;

constexpr int64_t kMaxSafeI32Bound = std::numeric_limits<int32_t>::max() / 2;
constexpr int64_t kMaxI32Extent = std::numeric_limits<int32_t>::max();
constexpr unsigned kInt32BitWidth = 32;
constexpr int kMaxAnalysisDepth = 8;
constexpr int kMaxRecursionDepth = 16;
constexpr int64_t kMinRankForUnitDimSqueeze = 2;

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::Block& block : func.getBlocks()) {
      for (mlir::Operation& op :
           llvm::make_early_inc_range(llvm::reverse(block))) {
        if (!op.hasTrait<mlir::OpTrait::IsTerminator>() &&
            mlir::isOpTriviallyDead(&op)) {
          op.erase();
          changed = true;
        }
      }
    }
  }
}

[[nodiscard]] std::optional<int64_t> CheckedMulPositive(int64_t lhs,
                                                        int64_t rhs) {
  if (lhs <= 0 || rhs <= 0) {
    return std::nullopt;
  }
  const __int128 prod = __int128{lhs} * __int128{rhs};
  if (prod > __int128{std::numeric_limits<int64_t>::max()}) {
    return std::nullopt;
  }
  return static_cast<int64_t>(prod);
}

[[nodiscard]] std::optional<int64_t> CheckedShapeProduct(
    llvm::ArrayRef<int64_t> shape) {
  int64_t prod = 1;
  for (const int64_t dim : shape) {
    std::optional<int64_t> next = CheckedMulPositive(prod, dim);
    if (!next.has_value()) {
      return std::nullopt;
    }
    prod = *next;
  }
  return prod;
}

[[nodiscard]] bool HasPositiveStaticShape(mlir::RankedTensorType type) {
  if (!type || !type.hasStaticShape()) {
    return false;
  }
  return CheckedShapeProduct(type.getShape()).has_value();
}

[[nodiscard]] bool IsStaticRankedTensor(mlir::Type type,
                                        mlir::RankedTensorType* out = nullptr) {
  auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!rtt || !rtt.hasStaticShape()) {
    return false;
  }
  if (out != nullptr) {
    *out = rtt;
  }
  return true;
}

[[nodiscard]] bool IsSupportedFloatType(mlir::Type elem_type) {
  return elem_type.isBF16() || elem_type.isF16() || elem_type.isF32();
}

[[nodiscard]] bool IsSupportedHloIndexIntType(mlir::Type elem_type) {
  auto int_type = llvm::dyn_cast<mlir::IntegerType>(elem_type);
  return int_type && (int_type.isSignless() || int_type.isUnsigned());
}

[[nodiscard]] bool IsPositiveZeroConstant(mlir::stablehlo::ConstantOp cst) {
  auto dense_fp = llvm::dyn_cast<mlir::DenseFPElementsAttr>(cst.getValue());
  if (!dense_fp || (!dense_fp.isSplat() && dense_fp.getNumElements() != 1)) {
    return false;
  }
  if (dense_fp.getNumElements() == 0) {
    return false;
  }
  llvm::APFloat fp_val = dense_fp.isSplat()
                             ? dense_fp.getSplatValue<llvm::APFloat>()
                             : *dense_fp.value_begin<llvm::APFloat>();
  return fp_val.isZero() && !fp_val.isNegative();
}

[[nodiscard]] mlir::Value UnwrapZeroPreservingOp(mlir::Operation* def_op) {
  if (auto bcast = llvm::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(def_op)) {
    return bcast.getOperand();
  }
  if (auto reshape = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(def_op)) {
    return reshape.getOperand();
  }
  if (auto convert = llvm::dyn_cast<mlir::stablehlo::ConvertOp>(def_op)) {
    auto in_type =
        llvm::dyn_cast<mlir::RankedTensorType>(convert.getOperand().getType());
    if (in_type && llvm::isa<mlir::FloatType>(in_type.getElementType())) {
      return convert.getOperand();
    }
  }
  return nullptr;
}

[[nodiscard]] bool IsPositiveZeroSplat(mlir::Value val, int depth = 0) {
  for (int d = depth; d <= kMaxAnalysisDepth && val; ++d) {
    mlir::Operation* def_op = val.getDefiningOp();
    if (!def_op) {
      return false;
    }
    if (auto cst = llvm::dyn_cast<mlir::stablehlo::ConstantOp>(def_op)) {
      return IsPositiveZeroConstant(cst);
    }
    val = UnwrapZeroPreservingOp(def_op);
  }
  return false;
}

[[nodiscard]] bool IsScalarAddRegion(mlir::Region& region) {
  if (!region.hasOneBlock()) {
    return false;
  }
  mlir::Block& block = region.front();
  if (block.getNumArguments() != 2 ||
      !llvm::hasSingleElement(block.without_terminator())) {
    return false;
  }
  auto ret_op =
      llvm::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  if (!ret_op || ret_op.getNumOperands() != 1) {
    return false;
  }
  auto add_op = ret_op.getOperand(0).getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add_op) {
    return false;
  }
  mlir::Value arg0 = block.getArgument(0);
  mlir::Value arg1 = block.getArgument(1);
  mlir::Value lhs = add_op.getLhs();
  mlir::Value rhs = add_op.getRhs();
  return (lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0);
}

[[nodiscard]] std::optional<int64_t> MatchSplatIntConstant(mlir::Value val) {
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    val = bcast.getOperand();
  }
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  auto dense_attr =
      mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense_attr || !dense_attr.isSplat()) {
    return std::nullopt;
  }
  auto int_type =
      mlir::dyn_cast<mlir::IntegerType>(dense_attr.getElementType());
  if (!int_type || int_type.getWidth() > 64) {
    return std::nullopt;
  }
  llvm::APInt ap_val = dense_attr.getSplatValue<llvm::APInt>();
  return ap_val.getSExtValue();
}

mlir::Value CreateSplatInt(mlir::OpBuilder& builder, mlir::Location loc,
                           llvm::ArrayRef<int64_t> shape, mlir::Type elem_type,
                           int64_t value) {
  auto tensor_type = mlir::RankedTensorType::get(shape, elem_type);
  auto int_type = llvm::cast<mlir::IntegerType>(elem_type);
  llvm::APInt ap_val(int_type.getWidth(), value, !int_type.isUnsigned());
  auto attr = mlir::DenseElementsAttr::get(tensor_type, ap_val);
  return builder.create<mlir::stablehlo::ConstantOp>(loc, attr);
}

mlir::Value CreateScalarFloat(mlir::OpBuilder& builder, mlir::Location loc,
                              mlir::Type elem_type, float value) {
  auto scalar_type = mlir::RankedTensorType::get({}, elem_type);
  llvm::APFloat ap_val(value);
  bool loses_info = false;
  if (elem_type.isBF16()) {
    ap_val.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                   &loses_info);
  } else if (elem_type.isF16()) {
    ap_val.convert(llvm::APFloat::IEEEhalf(),
                   llvm::APFloat::rmNearestTiesToEven, &loses_info);
  } else {
    ap_val.convert(llvm::APFloat::IEEEsingle(),
                   llvm::APFloat::rmNearestTiesToEven, &loses_info);
  }
  auto attr = mlir::DenseElementsAttr::get(scalar_type, ap_val);
  return builder.create<mlir::stablehlo::ConstantOp>(loc, attr);
}

// ============================================================================
// Mechanism 1: Canonicalize Additive Row Scatters to Reductions or Contractions
// ============================================================================

struct AdditiveRowScatterInfo {
  mlir::stablehlo::ScatterOp op;
  mlir::Value base;
  mlir::Value indices;
  mlir::Value updates;
  mlir::RankedTensorType base_type;
  mlir::RankedTensorType indices_type;
  mlir::RankedTensorType updates_type;
  mlir::Type elem_type;
  int64_t num_rows = 0;
  int64_t row_width = 0;
  llvm::ArrayRef<int64_t> window_shape;
  int64_t scatter_rank = 0;
  int64_t num_updates = 0;
  llvm::ArrayRef<int64_t> scatter_batch_shape;
};

[[nodiscard]] bool CheckScatterShapesAndTypes(mlir::stablehlo::ScatterOp op,
                                              AdditiveRowScatterInfo& info) {
  if (op.getInputs().size() != 1 || op.getUpdates().size() != 1 ||
      op.getNumResults() != 1) {
    return false;
  }
  info.op = op;
  info.base = op.getInputs().front();
  info.indices = op.getScatterIndices();
  info.updates = op.getUpdates().front();

  info.base_type = llvm::dyn_cast<mlir::RankedTensorType>(info.base.getType());
  info.indices_type =
      llvm::dyn_cast<mlir::RankedTensorType>(info.indices.getType());
  info.updates_type =
      llvm::dyn_cast<mlir::RankedTensorType>(info.updates.getType());
  auto result_type =
      llvm::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());

  if (!HasPositiveStaticShape(info.base_type) ||
      !HasPositiveStaticShape(info.indices_type) ||
      !HasPositiveStaticShape(info.updates_type) ||
      result_type != info.base_type || info.base_type.getRank() < 1) {
    return false;
  }

  info.elem_type = info.base_type.getElementType();
  if (!IsSupportedFloatType(info.elem_type) ||
      info.updates_type.getElementType() != info.elem_type) {
    return false;
  }

  info.num_rows = info.base_type.getDimSize(0);
  info.window_shape = info.base_type.getShape().drop_front(1);
  std::optional<int64_t> row_width = CheckedShapeProduct(info.window_shape);
  if (!row_width.has_value()) {
    return false;
  }
  info.row_width = *row_width;

  return info.num_rows >= 1 &&
         info.num_rows <
             static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

[[nodiscard]] bool IsContiguousAscendingRange(llvm::ArrayRef<int64_t> dims,
                                              int64_t start, int64_t count) {
  if (static_cast<int64_t>(dims.size()) != count) {
    return false;
  }
  for (int64_t i = 0; i < count; ++i) {
    if (dims[i] != start + i) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CheckScatterDimensionNumbers(mlir::stablehlo::ScatterOp op,
                                                AdditiveRowScatterInfo& info) {
  auto dnums = op.getScatterDimensionNumbers();
  if (!dnums.getInputBatchingDims().empty() ||
      !dnums.getScatterIndicesBatchingDims().empty() ||
      dnums.getInsertedWindowDims() != llvm::ArrayRef<int64_t>{0} ||
      dnums.getScatterDimsToOperandDims() != llvm::ArrayRef<int64_t>{0}) {
    return false;
  }

  const int64_t idx_rank = info.indices_type.getRank();
  const int64_t ivd = dnums.getIndexVectorDim();
  if (ivd == idx_rank) {
    info.scatter_rank = idx_rank;
  } else if (ivd == idx_rank - 1 && info.indices_type.getDimSize(ivd) == 1) {
    info.scatter_rank = idx_rank - 1;
  } else {
    return false;
  }

  if (info.scatter_rank < 1 ||
      !IsSupportedHloIndexIntType(info.indices_type.getElementType())) {
    return false;
  }

  info.scatter_batch_shape =
      info.indices_type.getShape().take_front(info.scatter_rank);
  std::optional<int64_t> num_updates =
      CheckedShapeProduct(info.scatter_batch_shape);
  if (!num_updates.has_value() ||
      !CheckedMulPositive(info.num_rows, *num_updates).has_value()) {
    return false;
  }
  info.num_updates = *num_updates;

  const int64_t window_rank = info.window_shape.size();
  return info.updates_type.getRank() == info.scatter_rank + window_rank &&
         IsContiguousAscendingRange(dnums.getUpdateWindowDims(),
                                    info.scatter_rank, window_rank) &&
         info.updates_type.getShape().take_front(info.scatter_rank) ==
             info.scatter_batch_shape &&
         info.updates_type.getShape().drop_front(info.scatter_rank) ==
             info.window_shape;
}

[[nodiscard]] bool MatchAdditiveRowScatter(mlir::stablehlo::ScatterOp op,
                                           AdditiveRowScatterInfo& info) {
  return CheckScatterShapesAndTypes(op, info) &&
         IsPositiveZeroSplat(info.base) &&
         IsScalarAddRegion(op.getUpdateComputation()) &&
         CheckScatterDimensionNumbers(op, info);
}

[[nodiscard]] bool IotaFitsInSignedInt32(mlir::stablehlo::IotaOp iota) {
  auto rtt = llvm::dyn_cast<mlir::RankedTensorType>(iota.getType());
  if (!rtt || !rtt.hasStaticShape()) {
    return false;
  }
  const int64_t iota_dim = static_cast<int64_t>(iota.getIotaDimension());
  return iota_dim >= 0 && iota_dim < rtt.getRank() &&
         rtt.getDimSize(iota_dim) <=
             static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

[[nodiscard]] bool ConvertFitsInSignedInt32(
    mlir::stablehlo::ConvertOp convert) {
  auto in_type =
      llvm::dyn_cast<mlir::RankedTensorType>(convert.getOperand().getType());
  if (!in_type) {
    return false;
  }
  auto in_int = llvm::dyn_cast<mlir::IntegerType>(in_type.getElementType());
  if (!in_int) {
    return false;
  }
  return in_int.getWidth() < kInt32BitWidth ||
         (in_int.isSignless() && in_int.getWidth() == kInt32BitWidth);
}

[[nodiscard]] bool ConstantFitsInSignedInt32(mlir::stablehlo::ConstantOp cst) {
  auto dense_int = llvm::dyn_cast<mlir::DenseIntElementsAttr>(cst.getValue());
  if (!dense_int) {
    return false;
  }
  if (dense_int.isSplat()) {
    const llvm::APInt val = dense_int.getSplatValue<llvm::APInt>();
    return val.isSignedIntN(kInt32BitWidth);  // MLIR_SIGNED_INT_OK
  }
  return llvm::all_of(
      dense_int.getValues<llvm::APInt>(), [](const llvm::APInt& ap) {
        return ap.isSignedIntN(kInt32BitWidth);  // MLIR_SIGNED_INT_OK
      });
}

[[nodiscard]] mlir::Value UnwrapRangePreservingOp(mlir::Operation* def_op) {
  if (auto resh = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(def_op)) {
    return resh.getOperand();
  }
  if (auto bcast = llvm::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(def_op)) {
    return bcast.getOperand();
  }
  if (auto trans = llvm::dyn_cast<mlir::stablehlo::TransposeOp>(def_op)) {
    return trans.getOperand();
  }
  if (auto slice = llvm::dyn_cast<mlir::stablehlo::SliceOp>(def_op)) {
    return slice.getOperand();
  }
  return nullptr;
}

[[nodiscard]] bool FitsInSignedInt32(mlir::Value val, int depth = 0) {
  if (depth > kMaxAnalysisDepth || !val) {
    return false;
  }
  mlir::Operation* def_op = val.getDefiningOp();
  if (!def_op) {
    return false;
  }
  if (auto iota = llvm::dyn_cast<mlir::stablehlo::IotaOp>(def_op)) {
    return IotaFitsInSignedInt32(iota);
  }
  if (auto convert = llvm::dyn_cast<mlir::stablehlo::ConvertOp>(def_op)) {
    return ConvertFitsInSignedInt32(convert);
  }
  if (auto cst = llvm::dyn_cast<mlir::stablehlo::ConstantOp>(def_op)) {
    return ConstantFitsInSignedInt32(cst);
  }
  if (mlir::Value inner = UnwrapRangePreservingOp(def_op)) {
    return FitsInSignedInt32(inner, depth + 1);
  }
  if (auto select = llvm::dyn_cast<mlir::stablehlo::SelectOp>(def_op)) {
    return FitsInSignedInt32(select.getOnTrue(), depth + 1) &&
           FitsInSignedInt32(select.getOnFalse(), depth + 1);
  }
  return false;
}

[[nodiscard]] bool IsSplatZeroInt(mlir::Value val) {
  auto cst = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!cst) {
    return false;
  }
  auto dense_int = llvm::dyn_cast<mlir::DenseIntElementsAttr>(cst.getValue());
  if (!dense_int) {
    return false;
  }
  if (dense_int.isSplat()) {
    return dense_int.getSplatValue<llvm::APInt>().isZero();
  }
  return llvm::all_of(dense_int.getValues<llvm::APInt>(),
                      [](const llvm::APInt& ap) { return ap.isZero(); });
}

mlir::Value NormalizeIndicesToFlatI32(mlir::OpBuilder& builder,
                                      mlir::Location loc,
                                      const AdditiveRowScatterInfo& info) {
  mlir::Value flat_idx = info.indices;
  auto idx_elem = info.indices_type.getElementType();
  if (info.indices_type.getRank() != 1 ||
      info.indices_type.getDimSize(0) != info.num_updates) {
    auto flat_type = mlir::RankedTensorType::get({info.num_updates}, idx_elem);
    flat_idx = builder.create<mlir::stablehlo::ReshapeOp>(loc, flat_type,
                                                          info.indices);
  }

  auto int_type = llvm::cast<mlir::IntegerType>(idx_elem);
  const unsigned bit_width = int_type.getWidth();
  auto target_type =
      mlir::RankedTensorType::get({info.num_updates}, builder.getI32Type());

  if (int_type.isSignless() && bit_width == kInt32BitWidth) {
    return flat_idx;
  }
  if (bit_width < kInt32BitWidth) {
    return builder.create<mlir::stablehlo::ConvertOp>(loc, target_type,
                                                      flat_idx);
  }
  if (int_type.isUnsigned()) {
    mlir::Value upper_bound = CreateSplatInt(builder, loc, {info.num_updates},
                                             idx_elem, info.num_rows);
    mlir::Value clamped =
        builder.create<mlir::stablehlo::MinOp>(loc, flat_idx, upper_bound);
    return builder.create<mlir::stablehlo::ConvertOp>(loc, target_type,
                                                      clamped);
  }

  if (FitsInSignedInt32(flat_idx)) {
    return builder.create<mlir::stablehlo::ConvertOp>(loc, target_type,
                                                      flat_idx);
  }
  mlir::Value lower_bound =
      CreateSplatInt(builder, loc, {info.num_updates}, idx_elem, -1);
  mlir::Value upper_bound =
      CreateSplatInt(builder, loc, {info.num_updates}, idx_elem, info.num_rows);
  mlir::Value clamped = builder.create<mlir::stablehlo::ClampOp>(
      loc, lower_bound, flat_idx, upper_bound);
  return builder.create<mlir::stablehlo::ConvertOp>(loc, target_type, clamped);
}

mlir::Value EmitSingleRowReduction(mlir::OpBuilder& builder, mlir::Location loc,
                                   const AdditiveRowScatterInfo& info,
                                   mlir::Value flat_i32,
                                   mlir::Value updates_2d) {
  mlir::Value masked_2d = updates_2d;
  if (!IsSplatZeroInt(flat_i32)) {
    mlir::Value scalar_zero_i32 =
        CreateSplatInt(builder, loc, {}, builder.getI32Type(), 0);
    auto zero_i32_1d = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc,
        mlir::RankedTensorType::get({info.num_updates}, builder.getI32Type()),
        scalar_zero_i32, builder.getDenseI64ArrayAttr({}));
    auto eq_zero_1d = builder.create<mlir::stablehlo::CompareOp>(
        loc, flat_i32, zero_i32_1d, mlir::stablehlo::ComparisonDirection::EQ);

    auto i1_2d_type = mlir::RankedTensorType::get(
        {info.num_updates, info.row_width}, builder.getI1Type());
    auto eq_zero_2d = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, i1_2d_type, eq_zero_1d, builder.getDenseI64ArrayAttr({0}));

    mlir::Value scalar_zero_f =
        CreateScalarFloat(builder, loc, info.elem_type, +0.0f);
    auto zero_f_2d_type = mlir::RankedTensorType::get(
        {info.num_updates, info.row_width}, info.elem_type);
    auto zero_f_2d = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, zero_f_2d_type, scalar_zero_f, builder.getDenseI64ArrayAttr({}));

    masked_2d = builder.create<mlir::stablehlo::SelectOp>(
        loc, eq_zero_2d, updates_2d, zero_f_2d);
  }

  mlir::Value init_scalar =
      CreateScalarFloat(builder, loc, info.elem_type, +0.0f);
  auto sum_1d_type =
      mlir::RankedTensorType::get({info.row_width}, info.elem_type);
  auto reduce_op = builder.create<mlir::stablehlo::ReduceOp>(
      loc, mlir::TypeRange{sum_1d_type}, mlir::ValueRange{masked_2d},
      mlir::ValueRange{init_scalar}, builder.getDenseI64ArrayAttr({0}));
  mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
      info.elem_type, reduce_op.getBody(), builder);
  mlir::Value sum_1d = reduce_op.getResult(0);

  if (sum_1d.getType() != info.base_type) {
    return builder.create<mlir::stablehlo::ReshapeOp>(loc, info.base_type,
                                                      sum_1d);
  }
  return sum_1d;
}

void CanonicalizeAdditiveRowScatters(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::ScatterOp> scatter_ops;
  func.walk([&scatter_ops](mlir::stablehlo::ScatterOp op) {
    scatter_ops.push_back(op);
  });

  for (mlir::stablehlo::ScatterOp scatter_op : scatter_ops) {
    AdditiveRowScatterInfo info;
    if (!MatchAdditiveRowScatter(scatter_op, info)) {
      continue;
    }
    // Single-row additive scatter is always profitable to lower to masked
    // reduce-sum: it replaces atomic memory RMW scatter with in-register VPU
    // select + reduce across the update batch dimension (profitable for all
    // num_updates >= 1). Multi-row indicator contraction is dropped and
    // deferred to a follow-up that resolves non-finite propagation semantics.
    if (info.num_rows != 1) {
      continue;
    }
    mlir::OpBuilder builder(scatter_op);
    mlir::Location loc = scatter_op.getLoc();
    mlir::Value flat_i32 = NormalizeIndicesToFlatI32(builder, loc, info);
    mlir::Value updates_2d = info.updates;
    if (info.updates_type.getRank() != 2 ||
        info.updates_type.getDimSize(0) != info.num_updates ||
        info.updates_type.getDimSize(1) != info.row_width) {
      auto updates_2d_type = mlir::RankedTensorType::get(
          {info.num_updates, info.row_width}, info.elem_type);
      updates_2d = builder.create<mlir::stablehlo::ReshapeOp>(
          loc, updates_2d_type, info.updates);
    }

    mlir::Value replacement =
        EmitSingleRowReduction(builder, loc, info, flat_i32, updates_2d);
    scatter_op.replaceAllUsesWith(mlir::ValueRange{replacement});
    scatter_op.erase();
  }
}

// ============================================================================
// Mechanism 2: Canonicalize Unit Gather Dims & Narrow Negative Index Wraps
// ============================================================================

[[nodiscard]] bool IsEntryParameterOrReshape(mlir::Value raw_idx) {
  mlir::Value val = raw_idx;
  while (auto reshape_op = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    if (val != raw_idx && !val.hasOneUse()) {
      return false;
    }
    val = reshape_op.getOperand();
  }
  auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(val);
  return block_arg && block_arg.getOwner()->isEntryBlock() &&
         mlir::isa<mlir::func::FuncOp>(block_arg.getOwner()->getParentOp());
}

[[nodiscard]] bool IsNegativeIndexCompare(mlir::stablehlo::CompareOp cmp_op,
                                          mlir::Value raw_idx) {
  // PyTorch negative index wrapping tests signed negativity: `idx < 0` or `idx
  // <= -1`. Unsigned comparisons (or comparisons on unsigned types) must never
  // be treated as negative checks.
  auto cmp_type = cmp_op.getCompareType();
  if (cmp_type == mlir::stablehlo::ComparisonType::UNSIGNED) {
    return false;
  }
  auto raw_type = mlir::dyn_cast<mlir::RankedTensorType>(raw_idx.getType());
  if (!raw_type) {
    return false;
  }
  auto int_type = mlir::dyn_cast<mlir::IntegerType>(raw_type.getElementType());
  if (!int_type || int_type.isUnsigned()) {
    return false;
  }
  const auto dir = cmp_op.getComparisonDirection();
  if (cmp_op.getLhs() == raw_idx) {
    std::optional<int64_t> rhs_cst = MatchSplatIntConstant(cmp_op.getRhs());
    if (!rhs_cst.has_value()) {
      return false;
    }
    return (dir == mlir::stablehlo::ComparisonDirection::LT && *rhs_cst == 0) ||
           (dir == mlir::stablehlo::ComparisonDirection::LE && *rhs_cst == -1);
  }
  if (cmp_op.getRhs() == raw_idx) {
    std::optional<int64_t> lhs_cst = MatchSplatIntConstant(cmp_op.getLhs());
    if (!lhs_cst.has_value()) {
      return false;
    }
    return (dir == mlir::stablehlo::ComparisonDirection::GT && *lhs_cst == 0) ||
           (dir == mlir::stablehlo::ComparisonDirection::GE && *lhs_cst == -1);
  }
  return false;
}

[[nodiscard]] bool IsAdditiveExtentShift(mlir::stablehlo::AddOp add_op,
                                         mlir::Value raw_idx,
                                         int64_t table_extent) {
  mlir::Value extent_operand;
  if (add_op.getLhs() == raw_idx) {
    extent_operand = add_op.getRhs();
  } else if (add_op.getRhs() == raw_idx) {
    extent_operand = add_op.getLhs();
  } else {
    return false;
  }
  std::optional<int64_t> add_cst = MatchSplatIntConstant(extent_operand);
  return add_cst.has_value() && *add_cst == table_extent;
}

[[nodiscard]] mlir::Value EmitNarrowedI32IndexWrap(
    mlir::stablehlo::SelectOp select_op, mlir::Value raw_idx,
    int64_t table_extent, mlir::stablehlo::ReshapeOp outer_reshape,
    mlir::RankedTensorType indices_type) {
  auto raw_type = mlir::cast<mlir::RankedTensorType>(raw_idx.getType());
  mlir::OpBuilder builder(select_op);
  mlir::Location loc = select_op.getLoc();
  mlir::Type i32_elem_type = builder.getI32Type();
  auto raw_i32_type =
      mlir::RankedTensorType::get(raw_type.getShape(), i32_elem_type);

  mlir::Value idx_to_convert = raw_idx;
  if (!FitsInSignedInt32(raw_idx)) {
    mlir::Value lower_bound =
        CreateSplatInt(builder, loc, raw_type.getShape(),
                       raw_type.getElementType(), -table_extent - 1);
    mlir::Value upper_bound =
        CreateSplatInt(builder, loc, raw_type.getShape(),
                       raw_type.getElementType(), table_extent);
    idx_to_convert = builder.create<mlir::stablehlo::ClampOp>(
        loc, lower_bound, raw_idx, upper_bound);
  }

  mlir::Value idx_i32 = builder.create<mlir::stablehlo::ConvertOp>(
      loc, raw_i32_type, idx_to_convert);
  mlir::Value zero_i32 =
      CreateSplatInt(builder, loc, raw_type.getShape(), i32_elem_type, 0);
  mlir::Value extent_i32 = CreateSplatInt(builder, loc, raw_type.getShape(),
                                          i32_elem_type, table_extent);

  auto cmp_i32 = builder.create<mlir::stablehlo::CompareOp>(
      loc, idx_i32, zero_i32, mlir::stablehlo::ComparisonDirection::LT);
  auto add_i32 = builder.create<mlir::stablehlo::AddOp>(loc, raw_i32_type,
                                                        idx_i32, extent_i32);
  mlir::Value wrapped_i32 = builder.create<mlir::stablehlo::SelectOp>(
      loc, raw_i32_type, cmp_i32, add_i32, idx_i32);

  if (outer_reshape) {
    auto reshaped_i32_type =
        mlir::RankedTensorType::get(indices_type.getShape(), i32_elem_type);
    return builder.create<mlir::stablehlo::ReshapeOp>(
        outer_reshape.getLoc(), reshaped_i32_type, wrapped_i32);
  }
  return wrapped_i32;
}

[[nodiscard]] mlir::Value TryNarrowEntryNegativeIndexWrapToI32(
    mlir::Value start_indices, int64_t table_extent) {
  if (table_extent <= 0 || table_extent > kMaxSafeI32Bound) {
    return start_indices;
  }
  auto indices_type =
      mlir::dyn_cast<mlir::RankedTensorType>(start_indices.getType());
  if (!indices_type || !indices_type.hasStaticShape()) {
    return start_indices;
  }
  auto int_type =
      mlir::dyn_cast<mlir::IntegerType>(indices_type.getElementType());
  // PyTorch indexing contract: tensor indices are signed integers. Unsigned
  // 64-bit tensors (ui64) cannot represent negative indices and must remain
  // unchanged.
  if (!int_type || int_type.getWidth() != 64 || !int_type.isSignless()) {
    return start_indices;
  }

  mlir::stablehlo::ReshapeOp outer_reshape;
  mlir::Value cur = start_indices;
  if (auto reshape_op = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    if (!cur.hasOneUse()) {
      return start_indices;
    }
    outer_reshape = reshape_op;
    cur = reshape_op.getOperand();
  }

  auto select_op = cur.getDefiningOp<mlir::stablehlo::SelectOp>();
  if (!select_op) {
    return start_indices;
  }
  if (!outer_reshape && !select_op->hasOneUse()) {
    return start_indices;
  }

  mlir::Value raw_idx = select_op.getOnFalse();
  if (!IsEntryParameterOrReshape(raw_idx) && !FitsInSignedInt32(raw_idx)) {
    return start_indices;
  }

  auto cmp_op = select_op.getPred().getDefiningOp<mlir::stablehlo::CompareOp>();
  auto add_op = select_op.getOnTrue().getDefiningOp<mlir::stablehlo::AddOp>();
  if (!cmp_op || !add_op || !select_op.getPred().hasOneUse() ||
      !select_op.getOnTrue().hasOneUse()) {
    return start_indices;
  }

  if (!IsNegativeIndexCompare(cmp_op, raw_idx) ||
      !IsAdditiveExtentShift(add_op, raw_idx, table_extent)) {
    return start_indices;
  }

  return EmitNarrowedI32IndexWrap(select_op, raw_idx, table_extent,
                                  outer_reshape, indices_type);
}

[[nodiscard]] bool IsEligible1DGather(mlir::stablehlo::GatherOp gather_op,
                                      mlir::RankedTensorType& operand_type,
                                      mlir::RankedTensorType& indices_type,
                                      mlir::RankedTensorType& result_type,
                                      int64_t& indexed_dim,
                                      int64_t& table_extent) {
  auto op_type =
      mlir::dyn_cast<mlir::RankedTensorType>(gather_op.getOperand().getType());
  auto idx_type = mlir::dyn_cast<mlir::RankedTensorType>(
      gather_op.getStartIndices().getType());
  auto res_type =
      mlir::dyn_cast<mlir::RankedTensorType>(gather_op.getResult().getType());
  if (!op_type || !idx_type || !res_type || !op_type.hasStaticShape() ||
      !idx_type.hasStaticShape() || !res_type.hasStaticShape()) {
    return false;
  }

  mlir::stablehlo::GatherDimensionNumbersAttr dnums =
      gather_op.getDimensionNumbers();
  if (!dnums.getOperandBatchingDims().empty() ||
      !dnums.getStartIndicesBatchingDims().empty() ||
      dnums.getStartIndexMap().size() != 1 ||
      dnums.getCollapsedSliceDims().size() != 1 ||
      dnums.getStartIndexMap()[0] != dnums.getCollapsedSliceDims()[0]) {
    return false;
  }

  indexed_dim = dnums.getStartIndexMap()[0];
  if (indexed_dim < 0 || indexed_dim >= op_type.getRank()) {
    return false;
  }
  table_extent = op_type.getDimSize(indexed_dim);
  operand_type = op_type;
  indices_type = idx_type;
  result_type = res_type;
  return true;
}

[[nodiscard]] mlir::Value TrySqueezeUnitIndexVectorDim(
    mlir::stablehlo::GatherOp gather_op, mlir::Value indices,
    int64_t& index_vector_dim,
    llvm::SmallVectorImpl<mlir::Operation*>& dead_roots) {
  auto indices_type = mlir::cast<mlir::RankedTensorType>(indices.getType());
  if (index_vector_dim < 0 || index_vector_dim >= indices_type.getRank() ||
      indices_type.getRank() < kMinRankForUnitDimSqueeze ||
      indices_type.getDimSize(index_vector_dim) != 1) {
    return indices;
  }
  llvm::SmallVector<int64_t> squeezed_shape;
  squeezed_shape.reserve(indices_type.getRank() - 1);
  for (int64_t i = 0; i < indices_type.getRank(); ++i) {
    if (i != index_vector_dim) {
      squeezed_shape.push_back(indices_type.getDimSize(i));
    }
  }
  auto squeezed_type = mlir::RankedTensorType::get(
      squeezed_shape, indices_type.getElementType());
  mlir::Value result_indices;
  if (auto reshape_op = indices.getDefiningOp<mlir::stablehlo::ReshapeOp>();
      reshape_op && reshape_op.getOperand().getType() == squeezed_type) {
    result_indices = reshape_op.getOperand();
    dead_roots.push_back(reshape_op.getOperation());
  } else {
    mlir::OpBuilder builder(gather_op);
    result_indices = builder.create<mlir::stablehlo::ReshapeOp>(
        gather_op.getLoc(), squeezed_type, indices);
  }
  index_vector_dim = squeezed_type.getRank();
  return result_indices;
}

bool TryCanonicalizeGatherIndices(
    mlir::stablehlo::GatherOp gather_op,
    llvm::SmallVectorImpl<mlir::Operation*>& dead_roots) {
  mlir::RankedTensorType operand_type;
  mlir::RankedTensorType indices_type;
  mlir::RankedTensorType result_type;
  int64_t indexed_dim = 0;
  int64_t table_extent = 0;
  if (!IsEligible1DGather(gather_op, operand_type, indices_type, result_type,
                          indexed_dim, table_extent)) {
    return false;
  }

  mlir::Value start_indices = gather_op.getStartIndices();
  mlir::Value new_indices =
      TryNarrowEntryNegativeIndexWrapToI32(start_indices, table_extent);
  mlir::stablehlo::GatherDimensionNumbersAttr dnums =
      gather_op.getDimensionNumbers();
  int64_t index_vector_dim = dnums.getIndexVectorDim();
  new_indices = TrySqueezeUnitIndexVectorDim(gather_op, new_indices,
                                             index_vector_dim, dead_roots);
  if (new_indices == start_indices &&
      index_vector_dim == dnums.getIndexVectorDim()) {
    return false;
  }

  mlir::OpBuilder builder(gather_op);
  auto new_dnums = mlir::stablehlo::GatherDimensionNumbersAttr::get(
      builder.getContext(), dnums.getOffsetDims(),
      dnums.getCollapsedSliceDims(), dnums.getOperandBatchingDims(),
      dnums.getStartIndicesBatchingDims(), dnums.getStartIndexMap(),
      index_vector_dim);
  auto new_gather = builder.create<mlir::stablehlo::GatherOp>(
      gather_op.getLoc(), result_type, gather_op.getOperand(), new_indices,
      new_dnums, gather_op.getSliceSizesAttr(),
      gather_op.getIndicesAreSortedAttr());
  gather_op.replaceAllUsesWith(new_gather.getResult());
  dead_roots.push_back(gather_op.getOperation());
  return true;
}

void CanonicalizeUnitGatherDimsAndIndexWrap(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::GatherOp> gather_ops;
  func.walk([&gather_ops](mlir::stablehlo::GatherOp op) {
    gather_ops.push_back(op);
  });
  llvm::SmallVector<mlir::Operation*> dead_roots;
  for (mlir::stablehlo::GatherOp gather_op : gather_ops) {
    if (gather_op && !mlir::isOpTriviallyDead(gather_op)) {
      TryCanonicalizeGatherIndices(gather_op, dead_roots);
    }
  }
  for (mlir::Operation* op : dead_roots) {
    if (op && mlir::isOpTriviallyDead(op)) {
      op->erase();
    }
  }
  EraseTriviallyDeadOps(func);
}

// ============================================================================
// Mechanism 3: Narrow Bounded i64 Index Subgraphs to i32 via Interval Analysis
// ============================================================================

struct IntInterval {
  int64_t lo = 0;
  int64_t hi = -1;
  bool valid = false;
};

bool IsWithinSafeI32(IntInterval iv) {
  return iv.valid && iv.lo <= iv.hi && iv.lo >= -kMaxSafeI32Bound &&
         iv.hi <= kMaxSafeI32Bound;
}

bool TryGetConstantFloatBounds(mlir::Value val, double* min_out,
                               double* max_out) {
  while (mlir::Operation* def = val.getDefiningOp()) {
    if (mlir::isa<mlir::stablehlo::BroadcastInDimOp,
                  mlir::stablehlo::ReshapeOp>(def)) {
      val = def->getOperand(0);
    } else {
      break;
    }
  }
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto attr = mlir::dyn_cast<mlir::DenseFPElementsAttr>(const_op.getValue());
  if (!attr) {
    return false;
  }
  if (attr.isSplat()) {
    double v = attr.getSplatValue<llvm::APFloat>().convertToDouble();
    if (!std::isfinite(v)) {
      return false;
    }
    *min_out = v;
    *max_out = v;
    return true;
  }
  return false;
}

bool TryComputeFloatInterval(
    mlir::Value val, const llvm::DenseMap<mlir::Value, IntInterval>& intervals,
    mlir::Value guarded_int_val, int64_t guarded_int_lo, double* lo_out,
    double* hi_out, int depth);

bool TryComputeFloatBinaryOp(mlir::Operation* op, double a_lo, double a_hi,
                             double b_lo, double b_hi, double* lo_out,
                             double* hi_out) {
  if (mlir::isa<mlir::stablehlo::AddOp>(op)) {
    *lo_out = a_lo + b_lo;
    *hi_out = a_hi + b_hi;
    return std::isfinite(*lo_out) && std::isfinite(*hi_out);
  }
  if (mlir::isa<mlir::stablehlo::SubtractOp>(op)) {
    *lo_out = a_lo - b_hi;
    *hi_out = a_hi - b_lo;
    return std::isfinite(*lo_out) && std::isfinite(*hi_out);
  }
  if (mlir::isa<mlir::stablehlo::MulOp>(op)) {
    double p1 = a_lo * b_lo;
    double p2 = a_lo * b_hi;
    double p3 = a_hi * b_lo;
    double p4 = a_hi * b_hi;
    if (!std::isfinite(p1) || !std::isfinite(p2) || !std::isfinite(p3) ||
        !std::isfinite(p4)) {
      return false;
    }
    *lo_out = std::min({p1, p2, p3, p4});
    *hi_out = std::max({p1, p2, p3, p4});
    return true;
  }
  if (mlir::isa<mlir::stablehlo::DivOp>(op)) {
    if (b_lo <= 0.0 && b_hi >= 0.0) {
      return false;
    }
    double q1 = a_lo / b_lo;
    double q2 = a_lo / b_hi;
    double q3 = a_hi / b_lo;
    double q4 = a_hi / b_hi;
    if (!std::isfinite(q1) || !std::isfinite(q2) || !std::isfinite(q3) ||
        !std::isfinite(q4)) {
      return false;
    }
    *lo_out = std::min({q1, q2, q3, q4});
    *hi_out = std::max({q1, q2, q3, q4});
    return true;
  }
  return false;
}

bool TryComputeFloatFromConvert(
    mlir::stablehlo::ConvertOp conv,
    const llvm::DenseMap<mlir::Value, IntInterval>& intervals,
    mlir::Value guarded_int_val, int64_t guarded_int_lo, double* lo_out,
    double* hi_out, int depth) {
  mlir::Value opnd = conv.getOperand();
  auto it = intervals.find(opnd);
  if (it != intervals.end() && it->second.valid) {
    int64_t lo_i = it->second.lo;
    int64_t hi_i = it->second.hi;
    if (opnd == guarded_int_val) {
      lo_i = std::max(lo_i, guarded_int_lo);
    }
    if (lo_i > hi_i) {
      return false;
    }
    *lo_out = static_cast<double>(lo_i);
    *hi_out = static_cast<double>(hi_i);
    return true;
  }
  return TryComputeFloatInterval(opnd, intervals, guarded_int_val,
                                 guarded_int_lo, lo_out, hi_out, depth + 1);
}

bool TryComputeFloatClamp(
    mlir::stablehlo::ClampOp clamp,
    const llvm::DenseMap<mlir::Value, IntInterval>& intervals,
    mlir::Value guarded_int_val, int64_t guarded_int_lo, double* lo_out,
    double* hi_out, int depth) {
  double lo_a = 0.0, hi_a = 0.0, lo_b = 0.0, hi_b = 0.0;
  double x_lo = 0.0, x_hi = 0.0;
  if (TryGetConstantFloatBounds(clamp.getMin(), &lo_a, &hi_a) &&
      TryGetConstantFloatBounds(clamp.getMax(), &lo_b, &hi_b) && lo_a <= hi_b &&
      TryComputeFloatInterval(clamp.getOperand(), intervals, guarded_int_val,
                              guarded_int_lo, &x_lo, &x_hi, depth + 1)) {
    *lo_out = std::min(hi_b, std::max(lo_a, x_lo));
    *hi_out = std::max(lo_a, std::min(hi_b, x_hi));
    return std::isfinite(*lo_out) && std::isfinite(*hi_out) &&
           *lo_out <= *hi_out;
  }
  return false;
}

bool TryComputeFloatLog(
    mlir::stablehlo::LogOp log_op,
    const llvm::DenseMap<mlir::Value, IntInterval>& intervals,
    mlir::Value guarded_int_val, int64_t guarded_int_lo, double* lo_out,
    double* hi_out, int depth) {
  double a_lo = 0.0, a_hi = 0.0;
  if (!TryComputeFloatInterval(log_op.getOperand(), intervals, guarded_int_val,
                               guarded_int_lo, &a_lo, &a_hi, depth + 1) ||
      a_lo <= 0.0 || a_hi < a_lo) {
    return false;
  }
  *lo_out = std::log(a_lo);
  *hi_out = std::log(a_hi);
  return std::isfinite(*lo_out) && std::isfinite(*hi_out);
}

bool TryComputeFloatInterval(
    mlir::Value val, const llvm::DenseMap<mlir::Value, IntInterval>& intervals,
    mlir::Value guarded_int_val, int64_t guarded_int_lo, double* lo_out,
    double* hi_out, int depth = 0) {
  if (depth > kMaxRecursionDepth) {
    return false;
  }
  if (TryGetConstantFloatBounds(val, lo_out, hi_out)) {
    return true;
  }
  mlir::Operation* def = val.getDefiningOp();
  if (def == nullptr) {
    return false;
  }
  if (mlir::isa<mlir::stablehlo::BroadcastInDimOp, mlir::stablehlo::ReshapeOp>(
          def)) {
    return TryComputeFloatInterval(def->getOperand(0), intervals,
                                   /*guarded_int_val=*/nullptr, guarded_int_lo,
                                   lo_out, hi_out, depth + 1);
  }
  if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(def)) {
    return TryComputeFloatFromConvert(conv, intervals, guarded_int_val,
                                      guarded_int_lo, lo_out, hi_out, depth);
  }
  if (auto clamp = mlir::dyn_cast<mlir::stablehlo::ClampOp>(def)) {
    return TryComputeFloatClamp(clamp, intervals, guarded_int_val,
                                guarded_int_lo, lo_out, hi_out, depth);
  }
  if (auto log_op = mlir::dyn_cast<mlir::stablehlo::LogOp>(def)) {
    return TryComputeFloatLog(log_op, intervals, guarded_int_val,
                              guarded_int_lo, lo_out, hi_out, depth);
  }
  if (def->getNumOperands() == 2) {
    double a_lo = 0.0, a_hi = 0.0, b_lo = 0.0, b_hi = 0.0;
    if (!TryComputeFloatInterval(def->getOperand(0), intervals, guarded_int_val,
                                 guarded_int_lo, &a_lo, &a_hi, depth + 1) ||
        !TryComputeFloatInterval(def->getOperand(1), intervals, guarded_int_val,
                                 guarded_int_lo, &b_lo, &b_hi, depth + 1)) {
      return false;
    }
    return TryComputeFloatBinaryOp(def, a_lo, a_hi, b_lo, b_hi, lo_out, hi_out);
  }
  return false;
}

bool TryGetConstantIntBounds(mlir::stablehlo::ConstantOp const_op,
                             IntInterval* iv_out) {
  auto attr = mlir::dyn_cast<mlir::DenseIntElementsAttr>(const_op.getValue());
  if (!attr) {
    return false;
  }
  if (attr.isSplat()) {
    llvm::APInt v = attr.getSplatValue<llvm::APInt>();
    if (v.getSignificantBits() > 63) {
      return false;
    }
    int64_t sv = v.getSExtValue();
    *iv_out = {sv, sv, true};
    return IsWithinSafeI32(*iv_out);
  }
  int64_t lo = std::numeric_limits<int64_t>::max();
  int64_t hi = std::numeric_limits<int64_t>::min();
  for (const llvm::APInt& v : attr.getValues<llvm::APInt>()) {
    if (v.getSignificantBits() > 63) {
      return false;
    }
    int64_t sv = v.getSExtValue();
    lo = std::min(lo, sv);
    hi = std::max(hi, sv);
  }
  *iv_out = {lo, hi, true};
  return IsWithinSafeI32(*iv_out);
}

bool TryInferFloatToIntInterval(
    mlir::stablehlo::ConvertOp conv_op,
    const llvm::DenseSet<mlir::Operation*>& comp_ops,
    const llvm::DenseMap<mlir::Value, IntInterval>& intervals,
    IntInterval* iv_out) {
  mlir::Value guarded_val;
  int64_t guarded_lo = 0;

  mlir::Operation* cur_op = conv_op.getOperation();
  while (cur_op != nullptr && cur_op->hasOneUse()) {
    mlir::Operation* next_op = *cur_op->getUsers().begin();
    if (!comp_ops.contains(next_op)) {
      break;
    }
    if (auto sel_op = mlir::dyn_cast<mlir::stablehlo::SelectOp>(next_op)) {
      if (sel_op.getOnFalse() == cur_op->getResult(0)) {
        if (auto cmp_op =
                sel_op.getPred().getDefiningOp<mlir::stablehlo::CompareOp>()) {
          if (cmp_op.getComparisonDirection() ==
              mlir::stablehlo::ComparisonDirection::LT) {
            auto rhs_it = intervals.find(cmp_op.getRhs());
            if (rhs_it != intervals.end() && rhs_it->second.valid) {
              guarded_val = cmp_op.getLhs();
              guarded_lo = rhs_it->second.lo;
            }
          }
        }
      }
      break;
    }
    if (mlir::isa<mlir::stablehlo::AddOp, mlir::stablehlo::MinOp,
                  mlir::stablehlo::ClampOp>(next_op)) {
      cur_op = next_op;
    } else {
      break;
    }
  }

  double lo_f = 0.0;
  double hi_f = 0.0;
  if (!TryComputeFloatInterval(conv_op.getOperand(), intervals, guarded_val,
                               guarded_lo, &lo_f, &hi_f)) {
    return false;
  }
  if (lo_f < -static_cast<double>(kMaxSafeI32Bound) ||
      hi_f > static_cast<double>(kMaxSafeI32Bound)) {
    return false;
  }
  *iv_out = {static_cast<int64_t>(std::floor(lo_f)),
             static_cast<int64_t>(std::ceil(hi_f)), true};
  return IsWithinSafeI32(*iv_out);
}

void HoistIotaFloatConverts(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::ConvertOp> float_convs;
  func.walk([&float_convs](mlir::stablehlo::ConvertOp op) {
    auto out_ty = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (out_ty && mlir::isa<mlir::FloatType>(out_ty.getElementType())) {
      float_convs.push_back(op);
    }
  });
  for (mlir::stablehlo::ConvertOp conv : float_convs) {
    mlir::RankedTensorType out_ty;
    if (!IsStaticRankedTensor(conv.getType(), &out_ty)) {
      continue;
    }
    // ALGEBRAIC: Hoisting ConvertOp(float) before BroadcastInDimOp/ReshapeOp
    // converts L elements instead of L * B elements, saving (B - 1) * L
    // float conversions for any downstream consumer.
    llvm::SmallVector<mlir::Operation*> view_chain;
    mlir::Value cur = conv.getOperand();
    while (mlir::Operation* def = cur.getDefiningOp()) {
      if (mlir::isa<mlir::stablehlo::BroadcastInDimOp,
                    mlir::stablehlo::ReshapeOp>(def)) {
        view_chain.push_back(def);
        cur = def->getOperand(0);
      } else {
        break;
      }
    }
    auto iota = cur.getDefiningOp<mlir::stablehlo::IotaOp>();
    if (!iota) {
      continue;
    }
    mlir::RankedTensorType iota_ty;
    if (!IsStaticRankedTensor(iota.getType(), &iota_ty) ||
        !iota_ty.getElementType().isSignlessInteger(64) ||
        out_ty.getNumElements() <= iota_ty.getNumElements()) {
      continue;
    }
    const int64_t iota_dim = static_cast<int64_t>(iota.getIotaDimension());
    if (iota_ty.getDimSize(iota_dim) > kMaxI32Extent) {
      continue;
    }
    mlir::OpBuilder builder(iota);
    auto iota_i32_ty =
        mlir::RankedTensorType::get(iota_ty.getShape(), builder.getI32Type());
    auto iota_i32 = builder.create<mlir::stablehlo::IotaOp>(
        iota.getLoc(), iota_i32_ty, iota.getIotaDimensionAttr());
    auto iota_flt_ty = mlir::RankedTensorType::get(iota_ty.getShape(),
                                                   out_ty.getElementType());
    mlir::Value flt_val = builder.create<mlir::stablehlo::ConvertOp>(
        conv.getLoc(), iota_flt_ty, iota_i32);
    builder.setInsertionPoint(conv);
    for (auto it = view_chain.rbegin(); it != view_chain.rend(); ++it) {
      mlir::Operation* vop = *it;
      auto v_ty =
          mlir::cast<mlir::RankedTensorType>(vop->getResult(0).getType());
      auto new_v_ty =
          mlir::RankedTensorType::get(v_ty.getShape(), out_ty.getElementType());
      if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(vop)) {
        flt_val = builder.create<mlir::stablehlo::BroadcastInDimOp>(
            bcast.getLoc(), new_v_ty, flt_val,
            bcast.getBroadcastDimensionsAttr());
      } else if (auto res = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(vop)) {
        flt_val = builder.create<mlir::stablehlo::ReshapeOp>(res.getLoc(),
                                                             new_v_ty, flt_val);
      }
    }
    conv.replaceAllUsesWith(flt_val);
  }
  EraseTriviallyDeadOps(func);
}

void BuildI64AdjacencyGraph(
    llvm::ArrayRef<mlir::Operation*> i64_ops,
    const llvm::DenseSet<mlir::Operation*>& i64_op_set,
    llvm::DenseMap<mlir::Operation*, llvm::SmallVector<mlir::Operation*>>&
        adj) {
  auto add_edge = [&i64_op_set, &adj](mlir::Operation* a, mlir::Operation* b) {
    if (a != nullptr && b != nullptr && a != b && i64_op_set.contains(a) &&
        i64_op_set.contains(b)) {
      adj[a].push_back(b);
      adj[b].push_back(a);
    }
  };
  for (mlir::Operation* op : i64_ops) {
    for (mlir::Value opnd : op->getOperands()) {
      add_edge(op, opnd.getDefiningOp());
    }
    for (mlir::Operation* user : op->getUsers()) {
      if (i64_op_set.contains(user)) {
        add_edge(op, user);
      } else if (auto cmp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(user)) {
        add_edge(cmp.getLhs().getDefiningOp(), cmp.getRhs().getDefiningOp());
      }
    }
  }
}

llvm::DenseSet<mlir::Operation*> CollectConnectedComponent(
    mlir::Operation* seed,
    const llvm::DenseMap<mlir::Operation*, llvm::SmallVector<mlir::Operation*>>&
        adj) {
  llvm::DenseSet<mlir::Operation*> comp_set;
  llvm::SmallVector<mlir::Operation*> queue = {seed};
  comp_set.insert(seed);
  while (!queue.empty()) {
    mlir::Operation* cur = queue.pop_back_val();
    auto it = adj.find(cur);
    if (it == adj.end()) {
      continue;
    }
    for (mlir::Operation* nxt : it->second) {
      if (comp_set.insert(nxt).second) {
        queue.push_back(nxt);
      }
    }
  }
  return comp_set;
}

bool IsValidComponentOpAndSinks(
    mlir::Operation* op, const llvm::DenseSet<mlir::Operation*>& comp_set) {
  for (mlir::Value opnd : op->getOperands()) {
    auto opnd_ty = mlir::dyn_cast<mlir::RankedTensorType>(opnd.getType());
    if (opnd_ty && opnd_ty.getElementType().isSignlessInteger(64)) {
      mlir::Operation* def = opnd.getDefiningOp();
      if (def == nullptr || !comp_set.contains(def)) {
        return false;
      }
    }
  }

  for (mlir::Operation* user : op->getUsers()) {
    if (comp_set.contains(user)) {
      continue;
    }
    if (auto cmp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(user)) {
      if (!comp_set.contains(cmp.getLhs().getDefiningOp()) ||
          !comp_set.contains(cmp.getRhs().getDefiningOp())) {
        return false;
      }
    } else if (mlir::isa<mlir::stablehlo::ConvertOp>(user)) {
      // Valid sink.
    } else if (auto gather = mlir::dyn_cast<mlir::stablehlo::GatherOp>(user)) {
      if (gather.getStartIndices() != op->getResult(0) ||
          comp_set.contains(gather.getOperand().getDefiningOp())) {
        return false;
      }
      mlir::RankedTensorType table_ty;
      if (!IsStaticRankedTensor(gather.getOperand().getType(), &table_ty)) {
        return false;
      }
      for (int64_t d : gather.getDimensionNumbers().getStartIndexMap()) {
        if (table_ty.getDimSize(d) > kMaxI32Extent) {
          return false;
        }
      }
    } else {
      return false;
    }
  }
  return true;
}

bool TryComputeOpIntInterval(
    mlir::Operation* op, const llvm::DenseSet<mlir::Operation*>& comp_set,
    const llvm::DenseMap<mlir::Value, IntInterval>& intervals,
    IntInterval* iv_out) {
  if (auto iota = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op)) {
    auto rtt = mlir::cast<mlir::RankedTensorType>(iota.getType());
    int64_t ext = rtt.getDimSize(static_cast<int64_t>(iota.getIotaDimension()));
    if (ext <= 0 || ext > kMaxSafeI32Bound) {
      return false;
    }
    *iv_out = {0, ext - 1, true};
    return true;
  }
  if (auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
    return TryGetConstantIntBounds(cst, iv_out);
  }
  if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
    mlir::RankedTensorType in_ty;
    if (!IsStaticRankedTensor(conv.getOperand().getType(), &in_ty)) {
      return false;
    }
    mlir::Type el_ty = in_ty.getElementType();
    if (el_ty.isInteger(1)) {
      *iv_out = {0, 1, true};
      return true;
    }
    if (el_ty.isInteger(8)) {
      *iv_out = {-128, 255, true};
      return true;
    }
    if (el_ty.isInteger(16)) {
      *iv_out = {-32768, 65535, true};
      return true;
    }
    if (mlir::isa<mlir::FloatType>(el_ty)) {
      return TryInferFloatToIntInterval(conv, comp_set, intervals, iv_out);
    }
    return false;
  }
  if (mlir::isa<mlir::stablehlo::ReshapeOp, mlir::stablehlo::BroadcastInDimOp,
                mlir::stablehlo::TransposeOp, mlir::stablehlo::SliceOp>(op)) {
    auto it = intervals.find(op->getOperand(0));
    if (it == intervals.end()) {
      return false;
    }
    *iv_out = it->second;
    return true;
  }
  if (auto concat = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(op)) {
    auto it = intervals.find(concat.getInputs()[0]);
    if (it == intervals.end()) {
      return false;
    }
    *iv_out = it->second;
    for (size_t i = 1; i < concat.getInputs().size(); ++i) {
      auto nxt_it = intervals.find(concat.getInputs()[i]);
      if (nxt_it == intervals.end()) {
        return false;
      }
      iv_out->lo = std::min(iv_out->lo, nxt_it->second.lo);
      iv_out->hi = std::max(iv_out->hi, nxt_it->second.hi);
      iv_out->valid = iv_out->valid && nxt_it->second.valid;
    }
    return true;
  }
  if (auto add_op = mlir::dyn_cast<mlir::stablehlo::AddOp>(op)) {
    auto a_it = intervals.find(add_op.getLhs());
    auto b_it = intervals.find(add_op.getRhs());
    if (a_it == intervals.end() || b_it == intervals.end()) {
      return false;
    }
    *iv_out = {a_it->second.lo + b_it->second.lo,
               a_it->second.hi + b_it->second.hi,
               a_it->second.valid && b_it->second.valid};
    return true;
  }
  if (auto sub_op = mlir::dyn_cast<mlir::stablehlo::SubtractOp>(op)) {
    auto a_it = intervals.find(sub_op.getLhs());
    auto b_it = intervals.find(sub_op.getRhs());
    if (a_it == intervals.end() || b_it == intervals.end()) {
      return false;
    }
    *iv_out = {a_it->second.lo - b_it->second.hi,
               a_it->second.hi - b_it->second.lo,
               a_it->second.valid && b_it->second.valid};
    return true;
  }
  if (auto mul_op = mlir::dyn_cast<mlir::stablehlo::MulOp>(op)) {
    auto a_it = intervals.find(mul_op.getLhs());
    auto b_it = intervals.find(mul_op.getRhs());
    if (a_it == intervals.end() || b_it == intervals.end() ||
        !a_it->second.valid || !b_it->second.valid) {
      return false;
    }
    int64_t p1 = 0;
    int64_t p2 = 0;
    int64_t p3 = 0;
    int64_t p4 = 0;
    if (__builtin_mul_overflow(a_it->second.lo, b_it->second.lo, &p1) ||
        __builtin_mul_overflow(a_it->second.lo, b_it->second.hi, &p2) ||
        __builtin_mul_overflow(a_it->second.hi, b_it->second.lo, &p3) ||
        __builtin_mul_overflow(a_it->second.hi, b_it->second.hi, &p4)) {
      return false;
    }
    *iv_out = {std::min({p1, p2, p3, p4}), std::max({p1, p2, p3, p4}), true};
    return true;
  }
  if (auto min_op = mlir::dyn_cast<mlir::stablehlo::MinOp>(op)) {
    auto a_it = intervals.find(min_op.getLhs());
    auto b_it = intervals.find(min_op.getRhs());
    if (a_it == intervals.end() || b_it == intervals.end()) {
      return false;
    }
    *iv_out = {std::min(a_it->second.lo, b_it->second.lo),
               std::min(a_it->second.hi, b_it->second.hi),
               a_it->second.valid && b_it->second.valid};
    return true;
  }
  if (auto max_op = mlir::dyn_cast<mlir::stablehlo::MaxOp>(op)) {
    auto a_it = intervals.find(max_op.getLhs());
    auto b_it = intervals.find(max_op.getRhs());
    if (a_it == intervals.end() || b_it == intervals.end()) {
      return false;
    }
    *iv_out = {std::max(a_it->second.lo, b_it->second.lo),
               std::max(a_it->second.hi, b_it->second.hi),
               a_it->second.valid && b_it->second.valid};
    return true;
  }
  if (auto clamp_op = mlir::dyn_cast<mlir::stablehlo::ClampOp>(op)) {
    auto lo_it = intervals.find(clamp_op.getMin());
    auto x_it = intervals.find(clamp_op.getOperand());
    auto hi_it = intervals.find(clamp_op.getMax());
    if (lo_it == intervals.end() || x_it == intervals.end() ||
        hi_it == intervals.end()) {
      return false;
    }
    const IntInterval lo_iv = lo_it->second;
    const IntInterval x_iv = x_it->second;
    const IntInterval hi_iv = hi_it->second;
    if (!lo_iv.valid || !x_iv.valid || !hi_iv.valid || lo_iv.lo > hi_iv.hi) {
      return false;
    }
    const int64_t lo = std::min(std::max(x_iv.lo, lo_iv.lo), hi_iv.lo);
    const int64_t hi = std::min(std::max(x_iv.hi, lo_iv.hi), hi_iv.hi);
    if (lo > hi) {
      return false;
    }
    *iv_out = {lo, hi, true};
    return true;
  }
  if (auto neg_op = mlir::dyn_cast<mlir::stablehlo::NegOp>(op)) {
    auto a_it = intervals.find(neg_op.getOperand());
    if (a_it == intervals.end()) {
      return false;
    }
    *iv_out = {-a_it->second.hi, -a_it->second.lo, a_it->second.valid};
    return true;
  }
  if (auto abs_op = mlir::dyn_cast<mlir::stablehlo::AbsOp>(op)) {
    auto a_it = intervals.find(abs_op.getOperand());
    if (a_it == intervals.end()) {
      return false;
    }
    const IntInterval a = a_it->second;
    int64_t max_abs = std::max(std::abs(a.lo), std::abs(a.hi));
    int64_t min_abs =
        (a.lo <= 0 && a.hi >= 0) ? 0 : std::min(std::abs(a.lo), std::abs(a.hi));
    *iv_out = {min_abs, max_abs, a.valid};
    return true;
  }
  if (auto sel_op = mlir::dyn_cast<mlir::stablehlo::SelectOp>(op)) {
    auto a_it = intervals.find(sel_op.getOnTrue());
    auto b_it = intervals.find(sel_op.getOnFalse());
    if (a_it == intervals.end() || b_it == intervals.end()) {
      return false;
    }
    *iv_out = {std::min(a_it->second.lo, b_it->second.lo),
               std::max(a_it->second.hi, b_it->second.hi),
               a_it->second.valid && b_it->second.valid};
    return true;
  }
  return false;
}

mlir::Value CloneComponentOpAsI32(
    mlir::Operation* op, mlir::RankedTensorType new_ty,
    const llvm::DenseMap<mlir::Value, mlir::Value>& i32_map,
    mlir::OpBuilder& builder) {
  mlir::Location loc = op->getLoc();
  if (auto iota = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op)) {
    return builder.create<mlir::stablehlo::IotaOp>(loc, new_ty,
                                                   iota.getIotaDimensionAttr());
  }
  if (auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
    auto attr = mlir::cast<mlir::DenseIntElementsAttr>(cst.getValue());
    mlir::ElementsAttr new_attr;
    if (attr.isSplat()) {
      int64_t sv = attr.getSplatValue<llvm::APInt>().getSExtValue();
      new_attr = mlir::DenseElementsAttr::get(
          new_ty, llvm::APInt(32, static_cast<uint64_t>(sv),
                              /*isSigned=*/true));
    } else {
      llvm::SmallVector<int32_t> vals;
      vals.reserve(attr.getNumElements());
      for (const llvm::APInt& v : attr.getValues<llvm::APInt>()) {
        vals.push_back(static_cast<int32_t>(v.getSExtValue()));
      }
      new_attr = mlir::DenseIntElementsAttr::get(new_ty, vals);
    }
    return builder.create<mlir::stablehlo::ConstantOp>(cst.getLoc(), new_ty,
                                                       new_attr);
  }
  if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
    if (conv.getOperand().getType() == new_ty) {
      return conv.getOperand();
    }
    return builder.create<mlir::stablehlo::ConvertOp>(conv.getLoc(), new_ty,
                                                      conv.getOperand());
  }
  if (auto res = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
    return builder.create<mlir::stablehlo::ReshapeOp>(
        res.getLoc(), new_ty, i32_map.lookup(res.getOperand()));
  }
  if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
    return builder.create<mlir::stablehlo::BroadcastInDimOp>(
        bcast.getLoc(), new_ty, i32_map.lookup(bcast.getOperand()),
        bcast.getBroadcastDimensionsAttr());
  }
  if (auto tr = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op)) {
    return builder.create<mlir::stablehlo::TransposeOp>(
        tr.getLoc(), new_ty, i32_map.lookup(tr.getOperand()),
        tr.getPermutationAttr());
  }
  if (auto sl = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op)) {
    return builder.create<mlir::stablehlo::SliceOp>(
        sl.getLoc(), new_ty, i32_map.lookup(sl.getOperand()),
        sl.getStartIndicesAttr(), sl.getLimitIndicesAttr(),
        sl.getStridesAttr());
  }
  if (auto cat = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(op)) {
    llvm::SmallVector<mlir::Value> new_inps;
    for (mlir::Value inp : cat.getInputs()) {
      new_inps.push_back(i32_map.lookup(inp));
    }
    return builder.create<mlir::stablehlo::ConcatenateOp>(
        cat.getLoc(), new_ty, new_inps, cat.getDimension());
  }
  if (auto add_op = mlir::dyn_cast<mlir::stablehlo::AddOp>(op)) {
    return builder.create<mlir::stablehlo::AddOp>(
        add_op.getLoc(), new_ty, i32_map.lookup(add_op.getLhs()),
        i32_map.lookup(add_op.getRhs()));
  }
  if (auto sub_op = mlir::dyn_cast<mlir::stablehlo::SubtractOp>(op)) {
    return builder.create<mlir::stablehlo::SubtractOp>(
        sub_op.getLoc(), new_ty, i32_map.lookup(sub_op.getLhs()),
        i32_map.lookup(sub_op.getRhs()));
  }
  if (auto mul_op = mlir::dyn_cast<mlir::stablehlo::MulOp>(op)) {
    return builder.create<mlir::stablehlo::MulOp>(
        mul_op.getLoc(), new_ty, i32_map.lookup(mul_op.getLhs()),
        i32_map.lookup(mul_op.getRhs()));
  }
  if (auto min_op = mlir::dyn_cast<mlir::stablehlo::MinOp>(op)) {
    return builder.create<mlir::stablehlo::MinOp>(
        min_op.getLoc(), new_ty, i32_map.lookup(min_op.getLhs()),
        i32_map.lookup(min_op.getRhs()));
  }
  if (auto max_op = mlir::dyn_cast<mlir::stablehlo::MaxOp>(op)) {
    return builder.create<mlir::stablehlo::MaxOp>(
        max_op.getLoc(), new_ty, i32_map.lookup(max_op.getLhs()),
        i32_map.lookup(max_op.getRhs()));
  }
  if (auto clamp_op = mlir::dyn_cast<mlir::stablehlo::ClampOp>(op)) {
    return builder.create<mlir::stablehlo::ClampOp>(
        clamp_op.getLoc(), new_ty, i32_map.lookup(clamp_op.getMin()),
        i32_map.lookup(clamp_op.getOperand()),
        i32_map.lookup(clamp_op.getMax()));
  }
  if (auto neg_op = mlir::dyn_cast<mlir::stablehlo::NegOp>(op)) {
    return builder.create<mlir::stablehlo::NegOp>(
        neg_op.getLoc(), new_ty, i32_map.lookup(neg_op.getOperand()));
  }
  if (auto abs_op = mlir::dyn_cast<mlir::stablehlo::AbsOp>(op)) {
    return builder.create<mlir::stablehlo::AbsOp>(
        abs_op.getLoc(), new_ty, i32_map.lookup(abs_op.getOperand()));
  }
  if (auto sel_op = mlir::dyn_cast<mlir::stablehlo::SelectOp>(op)) {
    return builder.create<mlir::stablehlo::SelectOp>(
        sel_op.getLoc(), new_ty, sel_op.getPred(),
        i32_map.lookup(sel_op.getOnTrue()),
        i32_map.lookup(sel_op.getOnFalse()));
  }
  return nullptr;
}

void NarrowBoundedI64IndexSubgraphs(mlir::func::FuncOp func) {
  HoistIotaFloatConverts(func);

  if (func.getBody().empty()) {
    return;
  }
  mlir::Block& block = func.getBody().front();
  llvm::SmallVector<mlir::Operation*> i64_ops;
  llvm::DenseSet<mlir::Operation*> i64_op_set;
  for (mlir::Operation& op : block) {
    if (op.getNumResults() != 1) {
      continue;
    }
    mlir::RankedTensorType rtt;
    if (IsStaticRankedTensor(op.getResult(0).getType(), &rtt) &&
        rtt.getElementType().isSignlessInteger(64)) {
      i64_ops.push_back(&op);
      i64_op_set.insert(&op);
    }
  }
  if (i64_ops.empty()) {
    return;
  }

  llvm::DenseMap<mlir::Operation*, llvm::SmallVector<mlir::Operation*>> adj;
  BuildI64AdjacencyGraph(i64_ops, i64_op_set, adj);

  llvm::DenseSet<mlir::Operation*> visited;
  for (mlir::Operation* seed : i64_ops) {
    if (visited.contains(seed)) {
      continue;
    }
    llvm::DenseSet<mlir::Operation*> comp_set =
        CollectConnectedComponent(seed, adj);
    for (mlir::Operation* op : comp_set) {
      visited.insert(op);
    }

    llvm::SmallVector<mlir::Operation*> comp_topo;
    bool has_target_op = false;
    for (mlir::Operation* op : i64_ops) {
      if (comp_set.contains(op)) {
        comp_topo.push_back(op);
        if (mlir::isa<mlir::stablehlo::IotaOp, mlir::stablehlo::AddOp,
                      mlir::stablehlo::SubtractOp, mlir::stablehlo::MulOp,
                      mlir::stablehlo::MinOp, mlir::stablehlo::MaxOp,
                      mlir::stablehlo::ClampOp, mlir::stablehlo::AbsOp,
                      mlir::stablehlo::NegOp, mlir::stablehlo::SelectOp>(op)) {
          has_target_op = true;
        }
      }
    }
    if (!has_target_op) {
      continue;
    }

    bool valid_comp = true;
    llvm::DenseMap<mlir::Value, IntInterval> intervals;
    for (mlir::Operation* op : comp_topo) {
      if (!IsValidComponentOpAndSinks(op, comp_set)) {
        valid_comp = false;
        break;
      }
      IntInterval iv;
      if (!TryComputeOpIntInterval(op, comp_set, intervals, &iv) ||
          !IsWithinSafeI32(iv)) {
        valid_comp = false;
        break;
      }
      intervals[op->getResult(0)] = iv;
    }
    if (!valid_comp) {
      continue;
    }

    llvm::DenseMap<mlir::Value, mlir::Value> i32_map;
    for (mlir::Operation* op : comp_topo) {
      mlir::OpBuilder builder(op);
      auto old_ty =
          mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
      auto new_ty =
          mlir::RankedTensorType::get(old_ty.getShape(), builder.getI32Type());
      mlir::Value new_val = CloneComponentOpAsI32(op, new_ty, i32_map, builder);
      if (new_val) {
        i32_map[op->getResult(0)] = new_val;
      }
    }

    for (mlir::Operation* op : comp_topo) {
      mlir::Value old_v = op->getResult(0);
      mlir::Value new_v = i32_map.lookup(old_v);
      if (!new_v) {
        continue;
      }
      llvm::SmallVector<mlir::OpOperand*> external_uses;
      for (mlir::OpOperand& use : old_v.getUses()) {
        if (!comp_set.contains(use.getOwner())) {
          external_uses.push_back(&use);
        }
      }
      for (mlir::OpOperand* use : external_uses) {
        use->set(new_v);
      }
    }

    for (auto it = comp_topo.rbegin(); it != comp_topo.rend(); ++it) {
      mlir::Operation* op = *it;
      if (op->use_empty()) {
        op->erase();
      }
    }
  }

  EraseTriviallyDeadOps(func);
}

// ============================================================================
// Pass Registration
// ============================================================================

struct CanonicalizeIndexArithmeticPass
    : public mlir::PassWrapper<CanonicalizeIndexArithmeticPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeIndexArithmeticPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-canonicalize-index-arithmetic";
  }
  llvm::StringRef getDescription() const override {
    return "Canonicalizes index arithmetic patterns (additive row scatters, "
           "unit gather dimensions, negative index wraps, and bounded i64 "
           "subgraphs) in StableHLO.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    CanonicalizeAdditiveRowScatters(func);
    CanonicalizeUnitGatherDimsAndIndexWrap(func);
    NarrowBoundedI64IndexSubgraphs(func);
    EraseTriviallyDeadOps(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCanonicalizeIndexArithmeticPass() {
  return std::make_unique<CanonicalizeIndexArithmeticPass>();
}

}  // namespace torch_tpu
