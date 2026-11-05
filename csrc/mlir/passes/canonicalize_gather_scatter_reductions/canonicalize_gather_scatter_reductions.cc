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

#include "csrc/mlir/passes/canonicalize_gather_scatter_reductions/canonicalize_gather_scatter_reductions.h"  // NOLINT(whitespace/line_length)

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"      // NOLINT(whitespace/line_length)
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // NOLINT(whitespace/line_length)
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"  // NOLINT(whitespace/line_length)
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"  // NOLINT(whitespace/line_length)
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // NOLINT(whitespace/line_length)
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// Returns true if dims contains [0, rank) in order, excluding excluded_dim.
bool IsSequentialDimsExcept(llvm::ArrayRef<int64_t> dims, int64_t rank,
                            int64_t excluded_dim) {
  int64_t pos = 0;
  for (int64_t d = 0; d < rank; ++d) {
    if (d == excluded_dim) {
      continue;
    }
    if (pos >= static_cast<int64_t>(dims.size()) || dims[pos] != d) {
      return false;
    }
    ++pos;
  }
  return pos == static_cast<int64_t>(dims.size());
}

// Returns true if dims contains [0, rank) in order.
bool IsIdentityDims(llvm::ArrayRef<int64_t> dims, int64_t rank) {
  return IsSequentialDimsExcept(dims, rank, /*excluded_dim=*/-1);
}

// Returns the mapped operand axis if bcast broadcasts an axis without
// expanding it.
std::optional<int64_t> MapBroadcastOutputAxisToInput(
    mlir::stablehlo::BroadcastInDimOp bcast, int64_t axis) {
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
  if (!in_type || !out_type || !in_type.hasStaticShape() ||
      !out_type.hasStaticShape() || axis < 0 || axis >= out_type.getRank()) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> bcast_dims = bcast.getBroadcastDimensions();
  for (int64_t i = 0; i < static_cast<int64_t>(bcast_dims.size()); ++i) {
    if (bcast_dims[i] == axis) {
      if (in_type.getDimSize(i) == out_type.getDimSize(axis)) {
        return i;
      }
      return std::nullopt;
    }
  }
  return std::nullopt;
}

// Returns true if reshape only adds trailing unit dimensions.
bool IsTrailingUnitDimReshape(mlir::stablehlo::ReshapeOp reshape,
                              int64_t axis) {
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(reshape.getOperand().getType());
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(reshape.getType());
  if (!in_type || !out_type || !in_type.hasStaticShape() ||
      !out_type.hasStaticShape()) {
    return false;
  }
  const int64_t in_rank = in_type.getRank();
  const int64_t out_rank = out_type.getRank();
  if (in_rank >= out_rank || axis < 0 || axis >= in_rank) {
    return false;
  }
  if (in_type.getShape() != out_type.getShape().take_front(in_rank)) {
    return false;
  }
  for (int64_t d = in_rank; d < out_rank; ++d) {
    if (out_type.getDimSize(d) != 1) {
      return false;
    }
  }
  return true;
}

// Returns true if convert from src to dst preserves all values in [0, extent).
bool IsValuePreservingConvertForExtent(mlir::Type src, mlir::Type dst,
                                       int64_t extent) {
  auto src_int = mlir::dyn_cast<mlir::IntegerType>(src);
  auto dst_int = mlir::dyn_cast<mlir::IntegerType>(dst);
  if (!src_int || !dst_int) {
    return false;
  }
  const unsigned src_w = src_int.getWidth();
  const unsigned dst_w = dst_int.getWidth();
  if (dst_w >= src_w && src_int.getSignedness() == dst_int.getSignedness()) {
    return true;
  }
  const int64_t max_val = extent - 1;
  if (max_val < 0) {
    return false;
  }
  if (src_w < 64) {
    const int64_t src_max = (int64_t{1} << (src_w - 1)) - 1;
    if (max_val > src_max) {
      return false;
    }
  }
  if (dst_w < 64) {
    const int64_t dst_max = (int64_t{1} << (dst_w - 1)) - 1;
    if (max_val > dst_max) {
      return false;
    }
  }
  return true;
}

struct IotaStep {
  mlir::Value value;
  int64_t axis = 0;
};

// Returns the predecessor value and updated axis if cur is a transparent index
// transform (Convert, BroadcastInDim, or trailing unit Reshape) along axis.
std::optional<IotaStep> StepIotaTransform(mlir::Value cur, int64_t axis) {
  if (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    auto ranked_type = mlir::dyn_cast<mlir::RankedTensorType>(cur.getType());
    auto src_type =
        mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
    if (!ranked_type || !src_type || !ranked_type.hasStaticShape() ||
        axis < 0 || axis >= ranked_type.getRank()) {
      return std::nullopt;
    }
    const int64_t extent = ranked_type.getDimSize(axis);
    if (!IsValuePreservingConvertForExtent(
            src_type.getElementType(), ranked_type.getElementType(), extent)) {
      return std::nullopt;
    }
    return IotaStep{conv.getOperand(), axis};
  }
  if (auto bcast = cur.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    std::optional<int64_t> mapped_axis =
        MapBroadcastOutputAxisToInput(bcast, axis);
    if (!mapped_axis.has_value()) {
      return std::nullopt;
    }
    return IotaStep{bcast.getOperand(), *mapped_axis};
  }
  if (auto reshape = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    if (IsTrailingUnitDimReshape(reshape, axis)) {
      return IotaStep{reshape.getOperand(), axis};
    }
  }
  return std::nullopt;
}

// Returns true if val represents an iota along axis.
bool IsIotaAlongAxis(mlir::Value val, int64_t axis) {
  mlir::Value cur = val;
  while (cur) {
    if (auto iota = cur.getDefiningOp<mlir::stablehlo::IotaOp>()) {
      return static_cast<int64_t>(iota.getIotaDimension()) == axis;
    }
    auto step = StepIotaTransform(cur, axis);
    if (!step.has_value()) {
      return false;
    }
    cur = step->value;
    axis = step->axis;
  }
  return false;
}

// Matches concatenated iota indices where exactly one dimension is non-iota.
std::optional<int64_t> MatchConcatIotaGatherDim(mlir::Value start_indices,
                                                int64_t rank) {
  auto concat = start_indices.getDefiningOp<mlir::stablehlo::ConcatenateOp>();
  if (!concat || static_cast<int64_t>(concat.getDimension()) != rank ||
      static_cast<int64_t>(concat.getInputs().size()) != rank) {
    return std::nullopt;
  }
  for (mlir::Value input : concat.getInputs()) {
    auto in_type = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!in_type || !in_type.hasStaticShape() ||
        in_type.getRank() != rank + 1 || in_type.getDimSize(rank) != 1) {
      return std::nullopt;
    }
  }
  std::optional<int64_t> non_iota_dim;
  for (int64_t d = 0; d < rank; ++d) {
    if (IsIotaAlongAxis(concat.getInputs()[d], d)) {
      continue;
    }
    if (non_iota_dim.has_value()) {
      return std::nullopt;
    }
    non_iota_dim = d;
  }
  return non_iota_dim;
}

// Resolves the gather dimension from dimension numbers.
std::optional<int64_t> ResolveGatherDimFromDnums(
    mlir::stablehlo::GatherOp gather_op, int64_t rank) {
  mlir::stablehlo::GatherDimensionNumbersAttr dnums =
      gather_op.getDimensionNumbers();
  llvm::ArrayRef<int64_t> offset_dims = dnums.getOffsetDims();
  llvm::ArrayRef<int64_t> collapsed = dnums.getCollapsedSliceDims();
  llvm::ArrayRef<int64_t> start_index_map = dnums.getStartIndexMap();
  llvm::ArrayRef<int64_t> op_batch = dnums.getOperandBatchingDims();
  llvm::ArrayRef<int64_t> idx_batch = dnums.getStartIndicesBatchingDims();

  if (offset_dims.empty() && llvm::hasSingleElement(collapsed) &&
      llvm::hasSingleElement(start_index_map)) {
    const int64_t dim = collapsed[0];
    if (dim >= 0 && dim < rank && start_index_map[0] == dim &&
        IsSequentialDimsExcept(op_batch, rank, dim) && op_batch == idx_batch) {
      return dim;
    }
    return std::nullopt;
  }

  if (offset_dims.empty() && op_batch.empty() && idx_batch.empty() &&
      dnums.getIndexVectorDim() == rank && IsIdentityDims(collapsed, rank) &&
      IsIdentityDims(start_index_map, rank)) {
    if (rank == 1) {
      return 0;
    }
    return MatchConcatIotaGatherDim(gather_op.getStartIndices(), rank);
  }

  return std::nullopt;
}

// Matches a single-element-per-slice GatherOp and returns its gather dimension.
std::optional<int64_t> MatchGatherDim(mlir::stablehlo::GatherOp gather_op) {
  auto table_type =
      mlir::dyn_cast<mlir::RankedTensorType>(gather_op.getOperand().getType());
  auto indices_type = mlir::dyn_cast<mlir::RankedTensorType>(
      gather_op.getStartIndices().getType());
  auto result_type =
      mlir::dyn_cast<mlir::RankedTensorType>(gather_op.getResult().getType());

  if (!table_type || !indices_type || !result_type ||
      !table_type.hasStaticShape() || !indices_type.hasStaticShape() ||
      !result_type.hasStaticShape()) {
    return std::nullopt;
  }

  const int64_t rank = table_type.getRank();
  if (rank < 1 || result_type.getRank() != rank) {
    return std::nullopt;
  }

  llvm::ArrayRef<int64_t> slice_sizes = gather_op.getSliceSizes();
  if (static_cast<int64_t>(slice_sizes.size()) != rank) {
    return std::nullopt;
  }

  const int64_t idx_rank = indices_type.getRank();
  mlir::stablehlo::GatherDimensionNumbersAttr dnums =
      gather_op.getDimensionNumbers();
  if (dnums.getIndexVectorDim() != idx_rank &&
      dnums.getIndexVectorDim() != idx_rank - 1) {
    return std::nullopt;
  }

  std::optional<int64_t> gather_dim =
      ResolveGatherDimFromDnums(gather_op, rank);
  if (!gather_dim.has_value()) {
    return std::nullopt;
  }

  for (int64_t d = 0; d < rank; ++d) {
    if (slice_sizes[d] != 1) {
      return std::nullopt;
    }
    if (d != *gather_dim &&
        result_type.getDimSize(d) != table_type.getDimSize(d)) {
      return std::nullopt;
    }
  }

  const int64_t result_dim = result_type.getDimSize(*gather_dim);
  const int64_t table_dim = table_type.getDimSize(*gather_dim);
  mlir::Operation* producer = gather_op.getOperand().getDefiningOp();
  if (producer && !producer->hasOneUse()) {
    if (result_dim >= table_dim) {
      return std::nullopt;
    }
  } else {
    if (result_dim > table_dim) {
      return std::nullopt;
    }
  }

  return *gather_dim;
}

// Returns true if broadcast operand does not expand along gather_dim.
bool IsBroadcastInvariantAlongDim(mlir::stablehlo::BroadcastInDimOp bcast,
                                  int64_t gather_dim) {
  auto src_type =
      mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
  if (!src_type || !src_type.hasStaticShape()) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = bcast.getBroadcastDimensions();
  for (int64_t i = 0; i < static_cast<int64_t>(dims.size()); ++i) {
    if (dims[i] == gather_dim && src_type.getDimSize(i) != 1) {
      return false;
    }
  }
  return true;
}

// Returns true if type is per-axis quantized along the specified dimension.
bool HasPerAxisQuantizedAlongDim(mlir::Type type, int64_t dim) {
  if (auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    if (auto per_axis =
            mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(
                tensor_type.getElementType())) {
      return per_axis.getQuantizedDimension() == dim;
    }
  }
  return false;
}

// Returns true if val is a splat constant.
bool IsSplatConstant(mlir::Value val) {
  if (auto cst = val.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
    if (auto elements =
            mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue())) {
      return !elements.empty() && elements.isSplat();
    }
  }
  return false;
}

// Recreates a splat constant with target_type.
mlir::Value CreateGatheredSplatConstant(mlir::PatternRewriter& rewriter,
                                        mlir::Location loc, mlir::Value val,
                                        mlir::RankedTensorType target_type) {
  auto cst = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  auto elements = mlir::cast<mlir::DenseElementsAttr>(cst.getValue());
  mlir::Type elem_type = target_type.getElementType();
  if (llvm::isa<mlir::FloatType>(elem_type)) {
    return rewriter.create<mlir::stablehlo::ConstantOp>(
        loc, mlir::DenseElementsAttr::get(
                 target_type, elements.getSplatValue<llvm::APFloat>()));
  }
  if (llvm::isa<mlir::IntegerType>(elem_type)) {
    return rewriter.create<mlir::stablehlo::ConstantOp>(
        loc, mlir::DenseElementsAttr::get(
                 target_type, elements.getSplatValue<llvm::APInt>()));
  }
  return rewriter.create<mlir::stablehlo::ConstantOp>(
      loc, mlir::DenseElementsAttr::get(
               target_type, elements.getSplatValue<mlir::Attribute>()));
}

// Creates an invariant broadcast operand matching target_type.
mlir::Value CreateGatheredBroadcastOperand(
    mlir::PatternRewriter& rewriter, mlir::Location loc,
    mlir::stablehlo::BroadcastInDimOp bcast,
    mlir::RankedTensorType target_type) {
  return rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, target_type, bcast.getOperand(), bcast.getBroadcastDimensions());
}

// Rewrites redundant batch-dimension reshape-convert sequences:
// reshape(convert(reshape(x))) -> convert(x) when the outer reshape restores
// the original rank and shape, or pushes reshape before convert.
struct CanonicalizeReshapeConvertPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  explicit CanonicalizeReshapeConvertPattern(mlir::MLIRContext* context)
      : mlir::OpRewritePattern<mlir::stablehlo::ReshapeOp>(context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReshapeOp reshape,
      mlir::PatternRewriter& rewriter) const override {
    auto conv =
        reshape.getOperand().getDefiningOp<mlir::stablehlo::ConvertOp>();
    if (!conv) {
      return mlir::failure();
    }
    mlir::Value inner = conv.getOperand();
    auto target_type =
        mlir::dyn_cast<mlir::RankedTensorType>(reshape.getType());
    if (!target_type || !target_type.hasStaticShape()) {
      return mlir::failure();
    }
    if (auto inner_reshape =
            inner.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      auto orig_type = mlir::dyn_cast<mlir::RankedTensorType>(
          inner_reshape.getOperand().getType());
      if (orig_type && orig_type.hasStaticShape() &&
          orig_type.getShape() == target_type.getShape()) {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvertOp>(
            reshape, target_type, inner_reshape.getOperand());
        return mlir::success();
      }
    } else if (conv->hasOneUse()) {
      auto inner_type = mlir::dyn_cast<mlir::RankedTensorType>(inner.getType());
      if (inner_type && inner_type.hasStaticShape() &&
          inner_type.getNumElements() == target_type.getNumElements() &&
          inner_type.getShape() != target_type.getShape()) {
        auto reshaped_base_type = mlir::RankedTensorType::get(
            target_type.getShape(), inner_type.getElementType());
        mlir::Value base_reshaped = rewriter.create<mlir::stablehlo::ReshapeOp>(
            reshape.getLoc(), reshaped_base_type, inner);
        rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvertOp>(
            reshape, target_type, base_reshaped);
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

// Returns true if any operand or result of op has per-axis quantization along
// dim.
bool HasPerAxisQuantizedAlongDim(mlir::Operation* op, int64_t dim) {
  for (mlir::Type t : op->getOperandTypes()) {
    if (HasPerAxisQuantizedAlongDim(t, dim)) {
      return true;
    }
  }
  for (mlir::Type t : op->getResultTypes()) {
    if (HasPerAxisQuantizedAlongDim(t, dim)) {
      return true;
    }
  }
  return false;
}

// Returns true if all operands of producer are either table operands matching
// table_shape or invariant along gather_dim, with at least one table operand.
bool CanSinkGatherThroughOperands(mlir::Operation* producer, int64_t gather_dim,
                                  llvm::ArrayRef<int64_t> table_shape) {
  int num_table_operands = 0;
  for (mlir::Value opnd : producer->getOperands()) {
    auto opnd_type = mlir::dyn_cast<mlir::RankedTensorType>(opnd.getType());
    if (!opnd_type || !opnd_type.hasStaticShape() ||
        opnd_type.getShape() != table_shape) {
      return false;
    }
    if (auto bcast = opnd.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      if (!IsBroadcastInvariantAlongDim(bcast, gather_dim)) {
        return false;
      }
      continue;
    }
    if (IsSplatConstant(opnd)) {
      continue;
    }
    ++num_table_operands;
  }
  return num_table_operands > 0;
}

// Returns the sunken/gathered operand corresponding to opnd.
mlir::Value CreateGatheredOperand(mlir::PatternRewriter& rewriter,
                                  mlir::stablehlo::GatherOp gather_op,
                                  mlir::Value opnd,
                                  llvm::ArrayRef<int64_t> result_shape) {
  auto opnd_type = mlir::cast<mlir::RankedTensorType>(opnd.getType());
  auto target_type =
      mlir::RankedTensorType::get(result_shape, opnd_type.getElementType());
  const mlir::Location loc = gather_op.getLoc();
  if (IsSplatConstant(opnd)) {
    return CreateGatheredSplatConstant(rewriter, loc, opnd, target_type);
  }
  if (auto bcast = opnd.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    return CreateGatheredBroadcastOperand(rewriter, loc, bcast, target_type);
  }
  return rewriter.create<mlir::stablehlo::GatherOp>(
      loc, target_type, opnd, gather_op.getStartIndices(),
      gather_op.getDimensionNumbersAttr(), gather_op.getSliceSizesAttr(),
      gather_op.getIndicesAreSortedAttr());
}

// Rewrites GatherOp across elementwise operations.
struct SinkGatherThroughElementwisePattern
    : public mlir::OpRewritePattern<mlir::stablehlo::GatherOp> {
  explicit SinkGatherThroughElementwisePattern(mlir::MLIRContext* context)
      : mlir::OpRewritePattern<mlir::stablehlo::GatherOp>(context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::GatherOp gather_op,
      mlir::PatternRewriter& rewriter) const override {
    std::optional<int64_t> maybe_dim = MatchGatherDim(gather_op);
    if (!maybe_dim.has_value()) {
      return mlir::failure();
    }
    const int64_t gather_dim = *maybe_dim;

    mlir::Value table = gather_op.getOperand();
    auto table_type = mlir::cast<mlir::RankedTensorType>(table.getType());
    auto result_type =
        mlir::cast<mlir::RankedTensorType>(gather_op.getResult().getType());
    llvm::ArrayRef<int64_t> result_shape = result_type.getShape();

    mlir::Operation* producer = table.getDefiningOp();
    if (!producer || producer->getNumResults() != 1 ||
        !producer->hasTrait<mlir::OpTrait::Elementwise>()) {
      return mlir::failure();
    }

    if (HasPerAxisQuantizedAlongDim(producer, gather_dim) ||
        !CanSinkGatherThroughOperands(producer, gather_dim,
                                      table_type.getShape())) {
      return mlir::failure();
    }

    mlir::IRMapping mapping;

    // When the elementwise producer has other uses (e.g., intermediate
    // log_probs returned or used in other reductions), sink the gather anyway
    // so the gathered path benefits from avoiding full-table materialization,
    // while preserving the original op for its other consumers.
    for (mlir::Value opnd : producer->getOperands()) {
      if (mapping.contains(opnd)) {
        continue;
      }
      mapping.map(
          opnd, CreateGatheredOperand(rewriter, gather_op, opnd, result_shape));
    }

    mlir::Operation* new_op = rewriter.clone(*producer, mapping);
    for (mlir::OpResult res : new_op->getOpResults()) {
      auto orig_res_type = mlir::cast<mlir::RankedTensorType>(res.getType());
      res.setType(mlir::RankedTensorType::get(result_shape,
                                              orig_res_type.getElementType()));
    }

    rewriter.replaceOp(gather_op, new_op->getResults());
    return mlir::success();
  }
};

// ----------------- Scatter Reduction Folding Pattern -----------------

bool IsZeroAttr(mlir::ElementsAttr attr) {
  if (!attr || attr.empty() || !attr.isSplat()) {
    return false;
  }
  mlir::Type elem_type = attr.getElementType();
  if (llvm::isa<mlir::FloatType>(elem_type)) {
    return attr.getSplatValue<llvm::APFloat>().isZero();
  }
  if (llvm::isa<mlir::IntegerType>(elem_type)) {
    return attr.getSplatValue<llvm::APInt>().isZero();
  }
  return false;
}

bool IsZeroValue(mlir::Value val) {
  mlir::Value cur = val;
  while (cur) {
    if (auto const_op = cur.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
      return IsZeroAttr(
          mlir::dyn_cast<mlir::ElementsAttr>(const_op.getValue()));
    }
    if (auto bcast = cur.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      cur = bcast.getOperand();
      continue;
    }
    if (auto reshape = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      cur = reshape.getOperand();
      continue;
    }
    if (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      cur = conv.getOperand();
      continue;
    }
    break;
  }
  return false;
}

bool IsAddRegion(mlir::Region& region, bool allow_rhs_overwrite) {
  if (region.empty()) {
    return false;
  }
  mlir::Block& block = region.front();
  if (block.getNumArguments() != 2) {
    return false;
  }
  auto ret_op =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  if (!ret_op || !llvm::hasSingleElement(ret_op.getOperands())) {
    return false;
  }
  mlir::Value ret_val = ret_op.getOperands()[0];
  mlir::Value arg0 = block.getArgument(0);
  mlir::Value arg1 = block.getArgument(1);
  if (allow_rhs_overwrite && ret_val == arg1) {
    return true;
  }
  auto add_op = ret_val.getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add_op) {
    return false;
  }
  mlir::Value lhs = add_op.getLhs();
  mlir::Value rhs = add_op.getRhs();
  return (lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0);
}

// Returns true if tensor shapes match unit scatter invariants along dim.
bool HasUnitScatterShapesAlongDim(mlir::RankedTensorType base_type,
                                  mlir::RankedTensorType indices_type,
                                  mlir::RankedTensorType updates_type,
                                  int64_t dim) {
  const int64_t rank = base_type.getRank();
  if (dim < 0 || dim >= rank || updates_type.getRank() != rank ||
      indices_type.getRank() != rank + 1 || updates_type.getDimSize(dim) != 1 ||
      indices_type.getDimSize(dim) != 1) {
    return false;
  }
  for (int64_t d = 0; d < rank; ++d) {
    if (d != dim && (updates_type.getDimSize(d) != base_type.getDimSize(d) ||
                     indices_type.getDimSize(d) != base_type.getDimSize(d))) {
      return false;
    }
  }
  return true;
}

// Extracts the scatter indices along dim from concatenated iotas.
std::optional<mlir::Value> ExtractConcatScatterDimIndices(mlir::Value indices,
                                                          int64_t rank,
                                                          int64_t dim) {
  auto concat = indices.getDefiningOp<mlir::stablehlo::ConcatenateOp>();
  if (!concat || static_cast<int64_t>(concat.getDimension()) != rank ||
      static_cast<int64_t>(concat.getInputs().size()) != rank) {
    return std::nullopt;
  }
  for (mlir::Value input : concat.getInputs()) {
    auto in_type = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!in_type || !in_type.hasStaticShape() ||
        in_type.getRank() != rank + 1 || in_type.getDimSize(rank) != 1) {
      return std::nullopt;
    }
  }
  for (int64_t d = 0; d < rank; ++d) {
    if (d != dim && !IsIotaAlongAxis(concat.getInputs()[d], d)) {
      return std::nullopt;
    }
  }
  mlir::Value dim_idx = concat.getInputs()[dim];
  auto dim_idx_type = mlir::dyn_cast<mlir::RankedTensorType>(dim_idx.getType());
  if (!dim_idx_type || !dim_idx_type.hasStaticShape() ||
      dim_idx_type.getRank() != rank + 1 || dim_idx_type.getDimSize(dim) != 1 ||
      dim_idx_type.getDimSize(rank) != 1) {
    return std::nullopt;
  }
  return dim_idx;
}

// Extracts the indices for the scattered unit-slice dimension.
std::optional<mlir::Value> ExtractUnitScatterDimIndices(
    mlir::stablehlo::ScatterOp scatter_op, int64_t dim) {
  if (!llvm::hasSingleElement(scatter_op.getInputs()) ||
      !llvm::hasSingleElement(scatter_op.getUpdates()) ||
      !llvm::hasSingleElement(scatter_op.getResults())) {
    return std::nullopt;
  }

  mlir::Value base = scatter_op.getInputs()[0];
  mlir::Value indices = scatter_op.getScatterIndices();
  mlir::Value updates = scatter_op.getUpdates()[0];

  auto base_type = mlir::dyn_cast<mlir::RankedTensorType>(base.getType());
  auto indices_type = mlir::dyn_cast<mlir::RankedTensorType>(indices.getType());
  auto updates_type = mlir::dyn_cast<mlir::RankedTensorType>(updates.getType());
  if (!base_type || !indices_type || !updates_type ||
      !base_type.hasStaticShape() || !indices_type.hasStaticShape() ||
      !updates_type.hasStaticShape() || base_type.getNumElements() <= 0) {
    return std::nullopt;
  }

  const int64_t rank = base_type.getRank();
  if (!HasUnitScatterShapesAlongDim(base_type, indices_type, updates_type,
                                    dim) ||
      !IsZeroValue(base) ||
      !IsAddRegion(scatter_op.getUpdateComputation(),
                   /*allow_rhs_overwrite=*/true)) {
    return std::nullopt;
  }

  mlir::stablehlo::ScatterDimensionNumbersAttr dnums =
      scatter_op.getScatterDimensionNumbers();
  if (!dnums.getUpdateWindowDims().empty() ||
      dnums.getIndexVectorDim() != rank) {
    return std::nullopt;
  }

  llvm::ArrayRef<int64_t> inserted = dnums.getInsertedWindowDims();
  llvm::ArrayRef<int64_t> scatter_map = dnums.getScatterDimsToOperandDims();
  llvm::ArrayRef<int64_t> in_batch = dnums.getInputBatchingDims();
  llvm::ArrayRef<int64_t> idx_batch = dnums.getScatterIndicesBatchingDims();

  if (llvm::hasSingleElement(inserted) && llvm::hasSingleElement(scatter_map) &&
      inserted[0] == dim && scatter_map[0] == dim &&
      IsSequentialDimsExcept(in_batch, rank, dim) && in_batch == idx_batch &&
      indices_type.getDimSize(rank) == 1) {
    return indices;
  }

  if (in_batch.empty() && idx_batch.empty() && IsIdentityDims(inserted, rank) &&
      IsIdentityDims(scatter_map, rank) &&
      indices_type.getDimSize(rank) == rank) {
    if (rank == 1) {
      return indices;
    }
    return ExtractConcatScatterDimIndices(indices, rank, dim);
  }

  return std::nullopt;
}

// Collects element types of an unbroken chain of ConvertOps.
bool CollectConvertChainTypes(
    mlir::Value& cur, mlir::RankedTensorType reduce_in_type,
    llvm::SmallVectorImpl<mlir::Type>& rev_convert_types) {
  while (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    auto src_type =
        mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
    auto dst_type = mlir::dyn_cast<mlir::RankedTensorType>(conv.getType());
    if (!src_type || !dst_type || !src_type.hasStaticShape() ||
        src_type.getShape() != reduce_in_type.getShape()) {
      return false;
    }
    rev_convert_types.push_back(dst_type.getElementType());
    cur = conv.getOperand();
  }
  return true;
}

// Replays a chain of ConvertOps on val toward result_type.
mlir::Value ApplyConvertChain(mlir::PatternRewriter& rewriter,
                              mlir::Location loc, mlir::Value val,
                              llvm::ArrayRef<int64_t> result_shape,
                              llvm::ArrayRef<mlir::Type> rev_convert_types,
                              mlir::RankedTensorType result_type) {
  for (mlir::Type dst_elem_type : llvm::reverse(rev_convert_types)) {
    auto cur_val_type = mlir::cast<mlir::RankedTensorType>(val.getType());
    if (cur_val_type.getElementType() != dst_elem_type) {
      val = rewriter.create<mlir::stablehlo::ConvertOp>(
          loc, mlir::RankedTensorType::get(result_shape, dst_elem_type), val);
    }
  }
  if (val.getType() != result_type) {
    val = rewriter.create<mlir::stablehlo::ConvertOp>(loc, result_type, val);
  }
  return val;
}

// Builds an in-bounds predicate mask for squeezed_idx if bounds check is
// needed. Returns std::nullopt if all representable index values are guaranteed
// in bounds.
std::optional<mlir::Value> BuildInBoundsMaskIfNeeded(
    mlir::PatternRewriter& rewriter, mlir::Location loc,
    mlir::Value squeezed_idx, mlir::RankedTensorType squeezed_idx_type,
    mlir::IntegerType idx_elem_type, int64_t dim_extent) {
  const unsigned bitwidth = idx_elem_type.getWidth();
  const uint64_t max_val =
      idx_elem_type.isUnsigned()
          ? llvm::APInt::getMaxValue(bitwidth).getZExtValue()
          : static_cast<uint64_t>(
                llvm::APInt::getSignedMaxValue(bitwidth).getSExtValue());

  if (static_cast<uint64_t>(dim_extent) > max_val) {
    if (idx_elem_type.isUnsigned()) {
      return std::nullopt;
    }
    mlir::Value zero_const = rewriter.create<mlir::stablehlo::ConstantOp>(
        loc, mlir::DenseElementsAttr::get(squeezed_idx_type,
                                          llvm::APInt(bitwidth, 0)));
    return rewriter.create<mlir::stablehlo::CompareOp>(
        loc, squeezed_idx, zero_const,
        mlir::stablehlo::ComparisonDirection::GE);
  }

  mlir::Value limit_const = rewriter.create<mlir::stablehlo::ConstantOp>(
      loc, mlir::DenseElementsAttr::get(
               squeezed_idx_type,
               llvm::APInt(bitwidth, static_cast<uint64_t>(dim_extent))));

  if (idx_elem_type.isUnsigned()) {
    return rewriter.create<mlir::stablehlo::CompareOp>(
        loc, squeezed_idx, limit_const,
        mlir::stablehlo::ComparisonDirection::LT);
  }

  mlir::Value zero_const = rewriter.create<mlir::stablehlo::ConstantOp>(
      loc, mlir::DenseElementsAttr::get(squeezed_idx_type,
                                        llvm::APInt(bitwidth, 0)));
  mlir::Value ge_zero = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, squeezed_idx, zero_const, mlir::stablehlo::ComparisonDirection::GE);
  mlir::Value lt_limit = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, squeezed_idx, limit_const, mlir::stablehlo::ComparisonDirection::LT);
  return rewriter.create<mlir::stablehlo::AndOp>(loc, ge_zero, lt_limit);
}

// Rewrite pattern that folds additive reductions over zero-base
// single-element-per-slice scatters into in-bounds selections.
struct FoldReduceOverUnitScatterPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ReduceOp> {
  explicit FoldReduceOverUnitScatterPattern(mlir::MLIRContext* context)
      : mlir::OpRewritePattern<mlir::stablehlo::ReduceOp>(context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReduceOp reduce_op,
      mlir::PatternRewriter& rewriter) const override {
    if (!llvm::hasSingleElement(reduce_op.getInputs()) ||
        !llvm::hasSingleElement(reduce_op.getInitValues()) ||
        !llvm::hasSingleElement(reduce_op.getResults()) ||
        !llvm::hasSingleElement(reduce_op.getDimensions())) {
      return mlir::failure();
    }

    const int64_t dim = reduce_op.getDimensions()[0];
    mlir::Value init_val = reduce_op.getInitValues()[0];
    if (!IsZeroValue(init_val) ||
        !IsAddRegion(reduce_op.getBody(), /*allow_rhs_overwrite=*/false)) {
      return mlir::failure();
    }

    auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(
        reduce_op.getResult(0).getType());
    auto reduce_in_type = mlir::dyn_cast<mlir::RankedTensorType>(
        reduce_op.getInputs()[0].getType());
    if (!result_type || !reduce_in_type || !result_type.hasStaticShape() ||
        !reduce_in_type.hasStaticShape()) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Type, 4> rev_convert_types;
    mlir::Value cur = reduce_op.getInputs()[0];
    if (!CollectConvertChainTypes(cur, reduce_in_type, rev_convert_types)) {
      return mlir::failure();
    }

    auto scatter_op = cur.getDefiningOp<mlir::stablehlo::ScatterOp>();
    if (!scatter_op) {
      return mlir::failure();
    }

    std::optional<mlir::Value> maybe_dim_indices =
        ExtractUnitScatterDimIndices(scatter_op, dim);
    if (!maybe_dim_indices.has_value()) {
      return mlir::failure();
    }

    const mlir::Location loc = reduce_op.getLoc();
    llvm::ArrayRef<int64_t> result_shape = result_type.getShape();
    auto scatter_in_type =
        mlir::cast<mlir::RankedTensorType>(scatter_op.getInputs()[0].getType());
    const int64_t dim_extent = scatter_in_type.getDimSize(dim);

    mlir::Value raw_indices = *maybe_dim_indices;
    auto raw_idx_type =
        mlir::cast<mlir::RankedTensorType>(raw_indices.getType());
    auto idx_elem_type =
        mlir::dyn_cast<mlir::IntegerType>(raw_idx_type.getElementType());
    if (!idx_elem_type) {
      return mlir::failure();
    }

    auto squeezed_idx_type =
        mlir::RankedTensorType::get(result_shape, idx_elem_type);
    mlir::Value squeezed_idx = rewriter.create<mlir::stablehlo::ReshapeOp>(
        loc, squeezed_idx_type, raw_indices);

    mlir::Value updates = scatter_op.getUpdates()[0];
    auto updates_type = mlir::cast<mlir::RankedTensorType>(updates.getType());
    mlir::Value val = rewriter.create<mlir::stablehlo::ReshapeOp>(
        loc,
        mlir::RankedTensorType::get(result_shape,
                                    updates_type.getElementType()),
        updates);
    val = ApplyConvertChain(rewriter, loc, val, result_shape, rev_convert_types,
                            result_type);

    std::optional<mlir::Value> in_bounds =
        BuildInBoundsMaskIfNeeded(rewriter, loc, squeezed_idx,
                                  squeezed_idx_type, idx_elem_type, dim_extent);
    if (!in_bounds.has_value()) {
      rewriter.replaceOp(reduce_op, val);
      return mlir::success();
    }

    mlir::Value zero_out = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, result_type, init_val, rewriter.getDenseI64ArrayAttr({}));
    mlir::Value selected = rewriter.create<mlir::stablehlo::SelectOp>(
        loc, result_type, *in_bounds, val, zero_out);
    rewriter.replaceOp(reduce_op, selected);
    return mlir::success();
  }
};

// Pass that canonicalizes gather and scatter reductions in StableHLO.
class CanonicalizeGatherScatterReductionsPass
    : public mlir::PassWrapper<CanonicalizeGatherScatterReductionsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CanonicalizeGatherScatterReductionsPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-canonicalize-gather-scatter-reductions";
  }

  llvm::StringRef getDescription() const override {
    return "Hoists gathers across invariant-broadcast elementwise DAGs "
           "and folds additive reductions over single-element-per-slice "
           "zero-base scatters.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<CanonicalizeReshapeConvertPattern,
                 SinkGatherThroughElementwisePattern,
                 FoldReduceOverUnitScatterPattern>(context);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCanonicalizeGatherScatterReductionsPass() {
  return std::make_unique<CanonicalizeGatherScatterReductionsPass>();
}

}  // namespace torch_tpu
