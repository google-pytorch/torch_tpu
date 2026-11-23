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

#include "csrc/mlir/passes/narrow_index_width.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

constexpr int64_t kMaxSmallConstantElementsToNarrow = 64;

bool IsSignlessI32(mlir::Type type) { return type.isSignlessInteger(32); }

bool IsSignlessI64(mlir::Type type) { return type.isSignlessInteger(64); }

bool FitsInSignedInt32(int64_t value) {
  return value >= std::numeric_limits<int32_t>::min() &&
         value <= std::numeric_limits<int32_t>::max();
}

mlir::RankedTensorType WithElementType(mlir::RankedTensorType type,
                                       mlir::Type new_element_type) {
  return mlir::RankedTensorType::get(type.getShape(), new_element_type,
                                     type.getEncoding());
}

// Returns the static dimension size or the StableHLO bounded-dynamic upper
// bound for `dim`. Returns `std::nullopt` for unbounded dynamic dimensions.
std::optional<int64_t> GetDimUpperBound(mlir::RankedTensorType type,
                                        int64_t dim) {
  if (dim < 0 || dim >= type.getRank()) {
    return std::nullopt;
  }
  const int64_t dim_size = type.getDimSize(dim);
  if (!mlir::ShapedType::isDynamic(dim_size)) {
    return dim_size;
  }
  auto bounds_attr =
      mlir::dyn_cast_or_null<mlir::stablehlo::TypeExtensionsAttr>(
          type.getEncoding());
  if (!bounds_attr) {
    return std::nullopt;
  }
  const llvm::ArrayRef<int64_t> bounds = bounds_attr.getBounds();
  if (dim >= static_cast<int64_t>(bounds.size())) {
    return std::nullopt;
  }
  const int64_t bound = bounds[dim];
  if (mlir::ShapedType::isDynamic(bound)) {
    return std::nullopt;
  }
  return bound;
}

bool DimFitsInSignedInt32(mlir::RankedTensorType type, int64_t dim) {
  const std::optional<int64_t> upper = GetDimUpperBound(type, dim);
  return upper.has_value() && *upper >= 0 &&
         *upper <= std::numeric_limits<int32_t>::max();
}

bool AllDimsFitInSignedInt32(mlir::RankedTensorType type) {
  for (int64_t d = 0; d < type.getRank(); ++d) {
    if (!DimFitsInSignedInt32(type, d)) {
      return false;
    }
  }
  return true;
}

// Matches `stablehlo.convert(x : i32 -> i64)` (sign-extension) and returns `x`.
mlir::Value MatchSignExtendedI32(mlir::Value value) {
  auto dst_type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!dst_type || !IsSignlessI64(dst_type.getElementType())) {
    return nullptr;
  }
  auto conv = value.getDefiningOp<mlir::stablehlo::ConvertOp>();
  if (!conv) {
    return nullptr;
  }
  auto src_type =
      mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
  if (!src_type || !IsSignlessI32(src_type.getElementType())) {
    return nullptr;
  }
  return conv.getOperand();
}

// Returns true when `value` is a small or splat signless `i64` constant whose
// elements all fit strictly within `[INT32_MIN, INT32_MAX]`.
bool IsNarrowableI64Constant(mlir::Value value) {
  auto const_op = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(const_op.getType());
  if (!tensor_type || !tensor_type.hasStaticShape() ||
      !IsSignlessI64(tensor_type.getElementType())) {
    return false;
  }
  auto dense_attr =
      mlir::dyn_cast<mlir::DenseIntElementsAttr>(const_op.getValue());
  if (!dense_attr) {
    return false;
  }
  if (dense_attr.isSplat()) {
    return FitsInSignedInt32(dense_attr.getSplatValue<int64_t>());
  }
  if (tensor_type.getNumElements() > kMaxSmallConstantElementsToNarrow) {
    return false;
  }
  for (const llvm::APInt& elem : dense_attr.getValues<llvm::APInt>()) {
    if (!FitsInSignedInt32(elem.getSExtValue())) {
      return false;
    }
  }
  return true;
}

// Materializes the `i32` equivalent of either a `convert(i32 -> i64)` value or
// a verified in-range `i64` constant.
mlir::Value MaterializeNarrowedI32(mlir::PatternRewriter& rewriter,
                                   mlir::Location loc, mlir::Value value) {
  if (mlir::Value src = MatchSignExtendedI32(value)) {
    return src;
  }
  auto const_op = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(const_op.getType());
  auto i32_tensor_type = WithElementType(tensor_type, rewriter.getI32Type());
  auto dense_attr = mlir::cast<mlir::DenseIntElementsAttr>(const_op.getValue());
  if (dense_attr.isSplat()) {
    const int32_t splat_val =
        static_cast<int32_t>(dense_attr.getSplatValue<int64_t>());
    auto new_attr = mlir::DenseElementsAttr::get(i32_tensor_type, splat_val);
    return mlir::stablehlo::ConstantOp::create(rewriter, loc, new_attr);
  }
  llvm::SmallVector<int32_t> vals;
  vals.reserve(tensor_type.getNumElements());
  for (const llvm::APInt& elem : dense_attr.getValues<llvm::APInt>()) {
    vals.push_back(static_cast<int32_t>(elem.getSExtValue()));
  }
  auto new_attr = mlir::DenseIntElementsAttr::get(i32_tensor_type, vals);
  return mlir::stablehlo::ConstantOp::create(rewriter, loc, new_attr);
}

bool IsSignedComparison(mlir::stablehlo::CompareOp compare_op) {
  std::optional<mlir::stablehlo::ComparisonType> compare_type =
      compare_op.getCompareType();
  return !compare_type.has_value() ||
         *compare_type == mlir::stablehlo::ComparisonType::SIGNED;
}

bool IsNarrowableIotaConsumer(mlir::Operation* user) {
  return mlir::isa<mlir::stablehlo::ReshapeOp, mlir::stablehlo::TransposeOp,
                   mlir::stablehlo::BroadcastInDimOp, mlir::stablehlo::SliceOp,
                   mlir::stablehlo::ReverseOp, mlir::stablehlo::ConcatenateOp,
                   mlir::stablehlo::PadOp, mlir::stablehlo::SelectOp,
                   mlir::stablehlo::GatherOp, mlir::stablehlo::ScatterOp,
                   mlir::stablehlo::DynamicSliceOp,
                   mlir::stablehlo::DynamicUpdateSliceOp,
                   mlir::stablehlo::CompareOp, mlir::stablehlo::SortOp,
                   mlir::stablehlo::ConvertOp>(user);
}

// Rewrites `convert(convert(x : i32 -> i64) : i64 -> i32)` directly to `x`.
struct CancelI32RoundTripPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ConvertOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto dst_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!dst_type || !IsSignlessI32(dst_type.getElementType())) {
      return mlir::failure();
    }
    mlir::Value inner = MatchSignExtendedI32(op.getOperand());
    if (!inner || inner.getType() != op.getType()) {
      return mlir::failure();
    }
    rewriter.replaceOp(op, inner);
    return mlir::success();
  }
};

// Rewrites `stablehlo.iota : tensor<...xi64>` whose iota dimension fits in
// `[0, INT32_MAX]` into `convert(stablehlo.iota : tensor<...xi32> -> i64)`,
// when consumed by at least one downstream op that can propagate/absorb `i32`.
struct NarrowIotaPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::IotaOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::IotaOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!result_type || !IsSignlessI64(result_type.getElementType())) {
      return mlir::failure();
    }
    const int64_t iota_dim = static_cast<int64_t>(op.getIotaDimension());
    if (!DimFitsInSignedInt32(result_type, iota_dim)) {
      return mlir::failure();
    }
    bool has_narrowable_user = false;
    for (mlir::Operation* user : op->getUsers()) {
      if (IsNarrowableIotaConsumer(user)) {
        has_narrowable_user = true;
        break;
      }
    }
    if (!has_narrowable_user) {
      return mlir::failure();
    }
    auto i32_type = WithElementType(result_type, rewriter.getI32Type());
    mlir::Value narrow_iota = mlir::stablehlo::IotaOp::create(
        rewriter, op.getLoc(), i32_type, op.getIotaDimensionAttr());
    mlir::Value widened = mlir::stablehlo::ConvertOp::create(
        rewriter, op.getLoc(), result_type, narrow_iota);
    rewriter.replaceOp(op, widened);
    return mlir::success();
  }
};

// Sinks `convert(i32 -> i64)` past `stablehlo.reshape`.
struct SinkConvertThroughReshapePattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReshapeOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Value src = MatchSignExtendedI32(op.getOperand());
    if (!src) {
      return mlir::failure();
    }
    auto dst_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!dst_type || !IsSignlessI64(dst_type.getElementType())) {
      return mlir::failure();
    }
    auto i32_type = WithElementType(dst_type, rewriter.getI32Type());
    mlir::Value narrow = mlir::stablehlo::ReshapeOp::create(
        rewriter, op.getLoc(), i32_type, src);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvertOp>(op, dst_type,
                                                            narrow);
    return mlir::success();
  }
};

// Sinks `convert(i32 -> i64)` past `stablehlo.transpose`.
struct SinkConvertThroughTransposePattern
    : public mlir::OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::TransposeOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Value src = MatchSignExtendedI32(op.getOperand());
    if (!src) {
      return mlir::failure();
    }
    auto dst_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!dst_type || !IsSignlessI64(dst_type.getElementType())) {
      return mlir::failure();
    }
    auto i32_type = WithElementType(dst_type, rewriter.getI32Type());
    mlir::Value narrow = mlir::stablehlo::TransposeOp::create(
        rewriter, op.getLoc(), i32_type, src, op.getPermutationAttr());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvertOp>(op, dst_type,
                                                            narrow);
    return mlir::success();
  }
};

// Sinks `convert(i32 -> i64)` past `stablehlo.broadcast_in_dim`.
struct SinkConvertThroughBroadcastPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::BroadcastInDimOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Value src = MatchSignExtendedI32(op.getOperand());
    if (!src) {
      return mlir::failure();
    }
    auto dst_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!dst_type || !IsSignlessI64(dst_type.getElementType())) {
      return mlir::failure();
    }
    auto i32_type = WithElementType(dst_type, rewriter.getI32Type());
    mlir::Value narrow = mlir::stablehlo::BroadcastInDimOp::create(
        rewriter, op.getLoc(), i32_type, src, op.getBroadcastDimensionsAttr());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvertOp>(op, dst_type,
                                                            narrow);
    return mlir::success();
  }
};

// Sinks `convert(i32 -> i64)` past `stablehlo.slice`.
struct SinkConvertThroughSlicePattern
    : public mlir::OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::SliceOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Value src = MatchSignExtendedI32(op.getOperand());
    if (!src) {
      return mlir::failure();
    }
    auto dst_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!dst_type || !IsSignlessI64(dst_type.getElementType())) {
      return mlir::failure();
    }
    auto i32_type = WithElementType(dst_type, rewriter.getI32Type());
    mlir::Value narrow = mlir::stablehlo::SliceOp::create(
        rewriter, op.getLoc(), i32_type, src, op.getStartIndicesAttr(),
        op.getLimitIndicesAttr(), op.getStridesAttr());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvertOp>(op, dst_type,
                                                            narrow);
    return mlir::success();
  }
};

// Sinks `convert(i32 -> i64)` past `stablehlo.reverse`.
struct SinkConvertThroughReversePattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ReverseOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReverseOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Value src = MatchSignExtendedI32(op.getOperand());
    if (!src) {
      return mlir::failure();
    }
    auto dst_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!dst_type || !IsSignlessI64(dst_type.getElementType())) {
      return mlir::failure();
    }
    auto i32_type = WithElementType(dst_type, rewriter.getI32Type());
    mlir::Value narrow = mlir::stablehlo::ReverseOp::create(
        rewriter, op.getLoc(), i32_type, src, op.getDimensionsAttr());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvertOp>(op, dst_type,
                                                            narrow);
    return mlir::success();
  }
};

// Sinks `convert(i32 -> i64)` past `stablehlo.concatenate` when every operand
// is either a sign-extended `i32` or a small/splat in-range `i64` constant.
struct SinkConvertThroughConcatenatePattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ConcatenateOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto dst_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!dst_type || !IsSignlessI64(dst_type.getElementType())) {
      return mlir::failure();
    }
    bool has_convert = false;
    for (mlir::Value operand : op.getInputs()) {
      if (MatchSignExtendedI32(operand)) {
        has_convert = true;
      } else if (!IsNarrowableI64Constant(operand)) {
        return mlir::failure();
      }
    }
    if (!has_convert) {
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Value> narrow_inputs;
    narrow_inputs.reserve(op.getInputs().size());
    for (mlir::Value operand : op.getInputs()) {
      narrow_inputs.push_back(
          MaterializeNarrowedI32(rewriter, op.getLoc(), operand));
    }
    auto i32_type = WithElementType(dst_type, rewriter.getI32Type());
    mlir::Value narrow = mlir::stablehlo::ConcatenateOp::create(
        rewriter, op.getLoc(), i32_type, narrow_inputs, op.getDimensionAttr());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvertOp>(op, dst_type,
                                                            narrow);
    return mlir::success();
  }
};

// Sinks `convert(i32 -> i64)` past `stablehlo.pad`.
struct SinkConvertThroughPadPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::PadOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto dst_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!dst_type || !IsSignlessI64(dst_type.getElementType())) {
      return mlir::failure();
    }
    mlir::Value operand = op.getOperand();
    mlir::Value padding = op.getPaddingValue();
    const bool has_convert = MatchSignExtendedI32(operand) != nullptr ||
                             MatchSignExtendedI32(padding) != nullptr;
    if (!has_convert) {
      return mlir::failure();
    }
    if (!MatchSignExtendedI32(operand) && !IsNarrowableI64Constant(operand)) {
      return mlir::failure();
    }
    if (!MatchSignExtendedI32(padding) && !IsNarrowableI64Constant(padding)) {
      return mlir::failure();
    }
    auto i32_type = WithElementType(dst_type, rewriter.getI32Type());
    mlir::Value narrow = mlir::stablehlo::PadOp::create(
        rewriter, op.getLoc(), i32_type,
        MaterializeNarrowedI32(rewriter, op.getLoc(), operand),
        MaterializeNarrowedI32(rewriter, op.getLoc(), padding),
        op.getEdgePaddingLowAttr(), op.getEdgePaddingHighAttr(),
        op.getInteriorPaddingAttr());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvertOp>(op, dst_type,
                                                            narrow);
    return mlir::success();
  }
};

// Sinks `convert(i32 -> i64)` past `stablehlo.select`.
struct SinkConvertThroughSelectPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::SelectOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto dst_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!dst_type || !IsSignlessI64(dst_type.getElementType())) {
      return mlir::failure();
    }
    mlir::Value on_true = op.getOnTrue();
    mlir::Value on_false = op.getOnFalse();
    const bool true_is_conv = MatchSignExtendedI32(on_true) != nullptr;
    const bool false_is_conv = MatchSignExtendedI32(on_false) != nullptr;
    if (!true_is_conv && !false_is_conv) {
      return mlir::failure();
    }
    if (!true_is_conv && !IsNarrowableI64Constant(on_true)) {
      return mlir::failure();
    }
    if (!false_is_conv && !IsNarrowableI64Constant(on_false)) {
      return mlir::failure();
    }
    auto i32_type = WithElementType(dst_type, rewriter.getI32Type());
    mlir::Value narrow = mlir::stablehlo::SelectOp::create(
        rewriter, op.getLoc(), i32_type, op.getPred(),
        MaterializeNarrowedI32(rewriter, op.getLoc(), on_true),
        MaterializeNarrowedI32(rewriter, op.getLoc(), on_false));
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvertOp>(op, dst_type,
                                                            narrow);
    return mlir::success();
  }
};

// Replaces `start_indices` in `stablehlo.gather` with its un-widened `i32`
// source whenever every indexed operand dimension fits in `[0, INT32_MAX]`.
struct NarrowGatherStartIndicesPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::GatherOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Value narrow_idx = MatchSignExtendedI32(op.getStartIndices());
    if (!narrow_idx) {
      return mlir::failure();
    }
    auto operand_type =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand().getType());
    if (!operand_type) {
      return mlir::failure();
    }
    for (int64_t dim : op.getDimensionNumbers().getStartIndexMap()) {
      if (!DimFitsInSignedInt32(operand_type, dim)) {
        return mlir::failure();
      }
    }
    rewriter.modifyOpInPlace(op, [&op, narrow_idx]() {
      op.getStartIndicesMutable().assign(narrow_idx);
    });
    return mlir::success();
  }
};

// Replaces `scatter_indices` in `stablehlo.scatter` with its un-widened `i32`
// source whenever every indexed input dimension fits in `[0, INT32_MAX]`.
struct NarrowScatterIndicesPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ScatterOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Value narrow_idx = MatchSignExtendedI32(op.getScatterIndices());
    if (!narrow_idx) {
      return mlir::failure();
    }
    const llvm::ArrayRef<int64_t> indexed_dims =
        op.getScatterDimensionNumbers().getScatterDimsToOperandDims();
    for (mlir::Value input : op.getInputs()) {
      auto input_type = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
      if (!input_type) {
        return mlir::failure();
      }
      for (int64_t dim : indexed_dims) {
        if (!DimFitsInSignedInt32(input_type, dim)) {
          return mlir::failure();
        }
      }
    }
    rewriter.modifyOpInPlace(op, [&op, narrow_idx]() {
      op.getScatterIndicesMutable().assign(narrow_idx);
    });
    return mlir::success();
  }
};

// Replaces `start_indices` in `stablehlo.dynamic_slice` with `i32` when every
// index is either `convert(i32 -> i64)` or an in-range `i64` scalar constant.
struct NarrowDynamicSliceIndicesPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::DynamicSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::DynamicSliceOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto operand_type =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand().getType());
    if (!operand_type || !AllDimsFitInSignedInt32(operand_type)) {
      return mlir::failure();
    }
    bool has_convert = false;
    for (mlir::Value idx : op.getStartIndices()) {
      if (MatchSignExtendedI32(idx)) {
        has_convert = true;
      } else if (!IsNarrowableI64Constant(idx)) {
        return mlir::failure();
      }
    }
    if (!has_convert) {
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Value> narrow_indices;
    narrow_indices.reserve(op.getStartIndices().size());
    for (mlir::Value idx : op.getStartIndices()) {
      narrow_indices.push_back(
          MaterializeNarrowedI32(rewriter, op.getLoc(), idx));
    }
    rewriter.replaceOpWithNewOp<mlir::stablehlo::DynamicSliceOp>(
        op, op.getType(), op.getOperand(), narrow_indices,
        op.getSliceSizesAttr());
    return mlir::success();
  }
};

// Replaces `start_indices` in `stablehlo.dynamic_update_slice` with `i32` when
// every index is either `convert(i32 -> i64)` or an in-range `i64` scalar.
struct NarrowDynamicUpdateSliceIndicesPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::DynamicUpdateSliceOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto operand_type =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand().getType());
    if (!operand_type || !AllDimsFitInSignedInt32(operand_type)) {
      return mlir::failure();
    }
    bool has_convert = false;
    for (mlir::Value idx : op.getStartIndices()) {
      if (MatchSignExtendedI32(idx)) {
        has_convert = true;
      } else if (!IsNarrowableI64Constant(idx)) {
        return mlir::failure();
      }
    }
    if (!has_convert) {
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Value> narrow_indices;
    narrow_indices.reserve(op.getStartIndices().size());
    for (mlir::Value idx : op.getStartIndices()) {
      narrow_indices.push_back(
          MaterializeNarrowedI32(rewriter, op.getLoc(), idx));
    }
    rewriter.replaceOpWithNewOp<mlir::stablehlo::DynamicUpdateSliceOp>(
        op, op.getType(), op.getOperand(), op.getUpdate(), narrow_indices);
    return mlir::success();
  }
};

// Rewrites signed `stablehlo.compare(sext(a), sext(b))` -> `compare(a, b)`
// (also allowing one operand to be an in-range `i64` constant).
struct NarrowComparePattern
    : public mlir::OpRewritePattern<mlir::stablehlo::CompareOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::CompareOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto lhs_type =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getLhs().getType());
    if (!lhs_type || !IsSignlessI64(lhs_type.getElementType()) ||
        !IsSignedComparison(op)) {
      return mlir::failure();
    }
    const bool lhs_is_conv = MatchSignExtendedI32(op.getLhs()) != nullptr;
    const bool rhs_is_conv = MatchSignExtendedI32(op.getRhs()) != nullptr;
    if (!lhs_is_conv && !rhs_is_conv) {
      return mlir::failure();
    }
    if (!lhs_is_conv && !IsNarrowableI64Constant(op.getLhs())) {
      return mlir::failure();
    }
    if (!rhs_is_conv && !IsNarrowableI64Constant(op.getRhs())) {
      return mlir::failure();
    }
    mlir::Value new_lhs =
        MaterializeNarrowedI32(rewriter, op.getLoc(), op.getLhs());
    mlir::Value new_rhs =
        MaterializeNarrowedI32(rewriter, op.getLoc(), op.getRhs());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::CompareOp>(
        op, op.getType(), new_lhs, new_rhs, op.getComparisonDirectionAttr(),
        op.getCompareTypeAttr());
    return mlir::success();
  }
};

// Returns true if `user` is a signed `stablehlo.compare` directly comparing
// `arg_a` and `arg_b` against each other (in either order).
bool IsDirectSignedPairwiseCompare(mlir::Operation* user,
                                   mlir::BlockArgument arg_a,
                                   mlir::BlockArgument arg_b) {
  auto cmp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(user);
  if (!cmp || !IsSignedComparison(cmp)) {
    return false;
  }
  return (cmp.getLhs() == arg_a && cmp.getRhs() == arg_b) ||
         (cmp.getLhs() == arg_b && cmp.getRhs() == arg_a);
}

bool AllUsersAreDirectSignedPairwiseCompares(mlir::BlockArgument primary,
                                             mlir::BlockArgument peer) {
  for (mlir::Operation* user : primary.getUsers()) {
    if (!IsDirectSignedPairwiseCompare(user, primary, peer)) {
      return false;
    }
  }
  return true;
}

// Returns true if comparator block arguments `arg_a` and `arg_b` for slot `k`
// can safely be narrowed from `tensor<i64>` to `tensor<i32>` inside `sort`'s
// comparator region (either unused payload args, or only compared directly
// against each other with a signed `stablehlo.compare`).
bool CanNarrowSortComparatorArgs(mlir::BlockArgument arg_a,
                                 mlir::BlockArgument arg_b) {
  return AllUsersAreDirectSignedPairwiseCompares(arg_a, arg_b) &&
         AllUsersAreDirectSignedPairwiseCompares(arg_b, arg_a);
}

struct NarrowedSortColumn {
  mlir::Value input;
  mlir::Type result_type;
};

// Returns the narrowed `i32` input and output type for column `col` of `op`
// when safe to narrow, or `std::nullopt` otherwise.
std::optional<NarrowedSortColumn> TryNarrowSortColumn(
    mlir::stablehlo::SortOp op, mlir::Block& comp_block, size_t col,
    mlir::Type i32_type) {
  mlir::Value src = MatchSignExtendedI32(op.getInputs()[col]);
  if (!src) {
    return std::nullopt;
  }
  if (!CanNarrowSortComparatorArgs(comp_block.getArgument(2 * col),
                                   comp_block.getArgument(2 * col + 1))) {
    return std::nullopt;
  }
  auto orig_res_type =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(col).getType());
  if (!orig_res_type || !IsSignlessI64(orig_res_type.getElementType())) {
    return std::nullopt;
  }
  return NarrowedSortColumn{
      .input = src,
      .result_type = WithElementType(orig_res_type, i32_type),
  };
}

void CloneSortComparatorWithNarrowedArgs(mlir::PatternRewriter& rewriter,
                                         mlir::Block& old_block,
                                         mlir::stablehlo::SortOp new_sort,
                                         llvm::ArrayRef<bool> narrowed) {
  auto scalar_i32_type = mlir::RankedTensorType::get({}, rewriter.getI32Type());
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  mlir::Block* new_block = rewriter.createBlock(&new_sort.getComparator());
  mlir::IRMapping mapping;
  for (size_t k = 0; k < narrowed.size(); ++k) {
    mlir::BlockArgument old_a = old_block.getArgument(2 * k);
    mlir::BlockArgument old_b = old_block.getArgument(2 * k + 1);
    mlir::Type arg_type = narrowed[k] ? scalar_i32_type : old_a.getType();
    mapping.map(old_a, new_block->addArgument(arg_type, old_a.getLoc()));
    mapping.map(old_b, new_block->addArgument(arg_type, old_b.getLoc()));
  }
  rewriter.setInsertionPointToEnd(new_block);
  for (mlir::Operation& inner : old_block) {
    rewriter.clone(inner, mapping);
  }
}

llvm::SmallVector<mlir::Value> BuildNarrowedSortReplacements(
    mlir::PatternRewriter& rewriter, mlir::stablehlo::SortOp old_sort,
    mlir::stablehlo::SortOp new_sort, llvm::ArrayRef<bool> narrowed) {
  llvm::SmallVector<mlir::Value> replacements;
  replacements.reserve(narrowed.size());
  for (size_t k = 0; k < narrowed.size(); ++k) {
    mlir::Value result = new_sort.getResult(k);
    if (narrowed[k]) {
      result = mlir::stablehlo::ConvertOp::create(
          rewriter, old_sort.getLoc(), old_sort.getResult(k).getType(), result);
    }
    replacements.push_back(result);
  }
  return replacements;
}

// Narrows `stablehlo.sort` operands fed by `convert(i32 -> i64)` (both unused
// payload columns and key columns with direct signed pairwise comparisons) so
// sorting runs natively in `i32`; wraps only narrowed results in `convert`.
struct NarrowSortOperandPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::SortOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::SortOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Region& comparator = op.getComparator();
    if (!comparator.hasOneBlock()) {
      return mlir::failure();
    }
    mlir::Block& comp_block = comparator.front();
    const size_t num_inputs = op.getInputs().size();
    if (comp_block.getNumArguments() != num_inputs * 2) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value> new_inputs(op.getInputs().begin(),
                                              op.getInputs().end());
    llvm::SmallVector<mlir::Type> new_result_types(op.getResultTypes().begin(),
                                                   op.getResultTypes().end());
    llvm::SmallVector<bool> narrowed(num_inputs, false);
    bool any_narrowed = false;

    const mlir::Type i32_type = rewriter.getI32Type();
    for (size_t k = 0; k < num_inputs; ++k) {
      auto narrowed_col = TryNarrowSortColumn(op, comp_block, k, i32_type);
      if (!narrowed_col.has_value()) {
        continue;
      }
      new_inputs[k] = narrowed_col->input;
      new_result_types[k] = narrowed_col->result_type;
      narrowed[k] = true;
      any_narrowed = true;
    }
    if (!any_narrowed) {
      return mlir::failure();
    }

    auto new_sort = mlir::stablehlo::SortOp::create(
        rewriter, op.getLoc(), new_result_types, new_inputs,
        op.getDimensionAttr(), op.getIsStableAttr());
    CloneSortComparatorWithNarrowedArgs(rewriter, comp_block, new_sort,
                                        narrowed);
    rewriter.setInsertionPointAfter(new_sort);
    rewriter.replaceOp(
        op, BuildNarrowedSortReplacements(rewriter, op, new_sort, narrowed));
    return mlir::success();
  }
};

struct NarrowIndexWidthPass
    : public mlir::PassWrapper<NarrowIndexWidthPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NarrowIndexWidthPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-narrow-index-width";
  }
  llvm::StringRef getDescription() const override {
    return "Eliminates redundant i32->i64 index widenings across StableHLO "
           "indexing chains when bounds fit in signed 32-bit integers.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<
        CancelI32RoundTripPattern, NarrowIotaPattern,
        SinkConvertThroughReshapePattern, SinkConvertThroughTransposePattern,
        SinkConvertThroughBroadcastPattern, SinkConvertThroughSlicePattern,
        SinkConvertThroughReversePattern, SinkConvertThroughConcatenatePattern,
        SinkConvertThroughPadPattern, SinkConvertThroughSelectPattern,
        NarrowGatherStartIndicesPattern, NarrowScatterIndicesPattern,
        NarrowDynamicSliceIndicesPattern,
        NarrowDynamicUpdateSliceIndicesPattern, NarrowComparePattern,
        NarrowSortOperandPattern>(context);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createNarrowIndexWidthPass() {
  return std::make_unique<NarrowIndexWidthPass>();
}

}  // namespace torch_tpu
