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

#include "csrc/mlir/passes/trim_convolution_padding_overhang.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

bool IsStaticRankedTensor(mlir::Type type, mlir::RankedTensorType* out_type) {
  auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!ranked || !ranked.hasStaticShape()) {
    return false;
  }
  if (out_type != nullptr) {
    *out_type = ranked;
  }
  return true;
}

bool IsUnitStrideSlice(mlir::stablehlo::SliceOp slice_op) {
  for (int64_t s : slice_op.getStrides()) {
    if (s != 1) {  // GENERIC: semantic unit-stride slice check
      return false;
    }
  }
  return true;
}

bool IsIdentitySlice(llvm::ArrayRef<int64_t> starts,
                     llvm::ArrayRef<int64_t> limits,
                     llvm::ArrayRef<int64_t> shape) {
  if (starts.size() != shape.size() || limits.size() != shape.size()) {
    return false;
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (starts[i] != 0 || limits[i] != shape[i]) {
      return false;
    }
  }
  return true;
}

mlir::Value BuildUnitStrideSlice(mlir::OpBuilder& builder, mlir::Location loc,
                                 mlir::Value input,
                                 llvm::ArrayRef<int64_t> starts,
                                 llvm::ArrayRef<int64_t> limits) {
  auto in_type = llvm::cast<mlir::RankedTensorType>(input.getType());
  if (IsIdentitySlice(starts, limits, in_type.getShape())) {
    return input;
  }
  llvm::SmallVector<int64_t, 4> strides(starts.size(),
                                        1);  // GENERIC: semantic unit stride
  return builder.create<mlir::stablehlo::SliceOp>(
      loc, input, builder.getDenseI64ArrayAttr(starts),
      builder.getDenseI64ArrayAttr(limits),
      builder.getDenseI64ArrayAttr(strides));
}

// Canonicalizes unit-dimension weight-norm reductions:
//   %keep  = stablehlo.reshape %red_1d : (tensor<Cxf32>) ->
//   tensor<1x..xCx..xf32> %vnorm = stablehlo.sqrt %keep : tensor<1x..xCx..xf32>
//   %inv   = stablehlo.divide %g_aligned, %vnorm : tensor<1x..xCx..xf32>
//   %bcast = stablehlo.broadcast_in_dim %inv, dims = [0, ..., R-1]
// into 1D tensor<Cxf32> operations and a single 1D BroadcastInDim(dims = [dim])
// so XLA does not emit a 2D-to-1D ReduceOneSizedDimension kReduce kernel.
void CanonicalizeUnitDimWeightNormBroadcasts(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::BroadcastInDimOp> bcast_ops;
  func.walk(
      [&](mlir::stablehlo::BroadcastInDimOp op) { bcast_ops.push_back(op); });

  for (mlir::stablehlo::BroadcastInDimOp bcast_op : bcast_ops) {
    mlir::RankedTensorType out_type;
    mlir::RankedTensorType inv_type;
    if (!IsStaticRankedTensor(bcast_op.getType(), &out_type) ||
        !IsStaticRankedTensor(bcast_op.getOperand().getType(), &inv_type)) {
      continue;
    }
    const int64_t rank = out_type.getRank();
    // GENERIC: semantic multi-dimensional weight tensor rank >= 2
    if (rank < 2 || inv_type.getRank() != rank) {
      continue;
    }
    llvm::ArrayRef<int64_t> bdims = bcast_op.getBroadcastDimensions();
    if (static_cast<int64_t>(bdims.size()) != rank) {
      continue;
    }
    bool identity_bdims = true;
    for (int64_t i = 0; i < rank; ++i) {
      if (bdims[i] != i) {
        identity_bdims = false;
        break;
      }
    }
    if (!identity_bdims) {
      continue;
    }

    auto div_op = bcast_op.getOperand().getDefiningOp<mlir::stablehlo::DivOp>();
    if (!div_op || !div_op->hasOneUse()) {
      continue;
    }
    auto sqrt_op = div_op.getRhs().getDefiningOp<mlir::stablehlo::SqrtOp>();
    if (!sqrt_op) {
      continue;
    }
    auto keep_reshape =
        sqrt_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!keep_reshape) {
      continue;
    }
    mlir::Value red_1d = keep_reshape.getOperand();
    mlir::RankedTensorType red_type;
    if (!IsStaticRankedTensor(red_1d.getType(), &red_type) ||
        red_type.getRank() != 1) {  // GENERIC: semantic 1D channel norm vector
      continue;
    }
    const int64_t channels = red_type.getDimSize(0);
    if (channels <= 0) {
      continue;
    }

    // Identify the unique channel axis matching red_1d where all other axes of
    // inv_type have unit size 1.
    int64_t channel_dim = -1;
    bool valid_unit_shape = true;
    for (int64_t d = 0; d < rank; ++d) {
      const int64_t sz = inv_type.getDimSize(d);
      if (sz == channels && out_type.getDimSize(d) == channels) {
        if (channel_dim != -1) {
          valid_unit_shape = false;  // GENERIC: semantic unique channel axis
          break;
        }
        channel_dim = d;
      } else if (sz != 1) {  // GENERIC: semantic unit broadcast dimension
        valid_unit_shape = false;
        break;
      }
    }
    if (!valid_unit_shape || channel_dim < 0) {
      continue;
    }

    mlir::OpBuilder sqrt_builder(sqrt_op);
    mlir::Value vnorm_1d = sqrt_builder.create<mlir::stablehlo::SqrtOp>(
        sqrt_op.getLoc(), red_type, red_1d);

    mlir::OpBuilder builder(div_op);
    mlir::Location loc = div_op.getLoc();
    mlir::Value g_val = div_op.getLhs();
    if (auto g_resh = g_val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      if (g_resh.getOperand().getType() == red_type) {
        g_val = g_resh.getOperand();
      }
    } else if (auto g_conv =
                   g_val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      if (auto g_resh =
              g_conv.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
        mlir::RankedTensorType g_in_type;
        if (IsStaticRankedTensor(g_resh.getOperand().getType(), &g_in_type) &&
            g_in_type.getRank() == 1 &&
            g_in_type.getDimSize(0) == channels) {  // GENERIC: semantic 1D g
          g_val = builder.create<mlir::stablehlo::ConvertOp>(
              loc, red_type, g_resh.getOperand());
        }
      }
    }
    if (g_val.getType() != red_type) {
      g_val = builder.create<mlir::stablehlo::ReshapeOp>(loc, red_type, g_val);
    }

    mlir::Value inv_1d =
        builder.create<mlir::stablehlo::DivOp>(loc, red_type, g_val, vnorm_1d);
    mlir::Value new_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        bcast_op.getLoc(), out_type, inv_1d,
        builder.getDenseI64ArrayAttr({channel_dim}));
    bcast_op.replaceAllUsesWith(new_bcast);
    bcast_op.erase();
    div_op.erase();

    if (!sqrt_op.use_empty()) {
      mlir::Value vnorm_aligned =
          sqrt_builder.create<mlir::stablehlo::ReshapeOp>(sqrt_op.getLoc(),
                                                          inv_type, vnorm_1d);
      sqrt_op.replaceAllUsesWith(vnorm_aligned);
    }
    sqrt_op.erase();
    if (keep_reshape.use_empty()) {
      keep_reshape.erase();
    }
  }
}

bool TryPushSingleSliceUpstream(
    mlir::stablehlo::SliceOp slice_op,
    llvm::SmallVectorImpl<mlir::stablehlo::SliceOp>& worklist) {
  if (!slice_op || slice_op->getBlock() == nullptr ||
      !IsUnitStrideSlice(slice_op)) {
    return false;
  }
  mlir::Value operand = slice_op.getOperand();
  mlir::RankedTensorType in_type;
  mlir::RankedTensorType out_type;
  if (!IsStaticRankedTensor(operand.getType(), &in_type) ||
      !IsStaticRankedTensor(slice_op.getType(), &out_type)) {
    return false;
  }
  llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
  if (IsIdentitySlice(starts, limits, in_type.getShape())) {
    slice_op.replaceAllUsesWith(operand);
    slice_op.erase();
    return true;
  }

  // Deduplicate identical sibling SliceOps consuming operand directly against
  // slice_op to safely collapse diamonds without use-after-free hazards.
  for (mlir::Operation* user : llvm::make_early_inc_range(operand.getUsers())) {
    auto other = llvm::dyn_cast<mlir::stablehlo::SliceOp>(user);
    if (!other || other == slice_op ||
        other->getBlock() != slice_op->getBlock() ||
        !IsUnitStrideSlice(other)) {
      continue;
    }
    if (other.getStartIndices() == starts &&
        other.getLimitIndices() == limits &&
        other.getStrides() == slice_op.getStrides()) {
      if (other->isBeforeInBlock(slice_op)) {
        slice_op.replaceAllUsesWith(other.getResult());
        slice_op.erase();
        if (!llvm::is_contained(worklist, other)) {
          worklist.push_back(other);
        }
        return true;
      }
      other.replaceAllUsesWith(slice_op.getResult());
      for (auto& item : worklist) {
        if (item == other) {
          item = nullptr;
        }
      }
      other.erase();
    }
  }

  mlir::Operation* def_op = operand.getDefiningOp();
  if (def_op == nullptr) {
    return false;
  }

  mlir::OpBuilder builder(slice_op);
  mlir::Location loc = slice_op.getLoc();

  // Case 1: Splat ConstantOp.
  if (auto const_op = llvm::dyn_cast<mlir::stablehlo::ConstantOp>(def_op)) {
    auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    if (dense && dense.isSplat()) {
      mlir::Value new_const = builder.create<mlir::stablehlo::ConstantOp>(
          loc, out_type, dense.resizeSplat(out_type));
      slice_op.replaceAllUsesWith(new_const);
      slice_op.erase();
      if (const_op.use_empty()) {
        const_op.erase();
      }
      return true;
    }
    return false;
  }

  // Case 2: BroadcastInDimOp (can be pushed through even with multiple uses).
  if (auto bcast_op =
          llvm::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(def_op)) {
    mlir::Value bcast_in = bcast_op.getOperand();
    mlir::RankedTensorType bcast_in_type;
    if (!IsStaticRankedTensor(bcast_in.getType(), &bcast_in_type)) {
      return false;
    }
    llvm::ArrayRef<int64_t> bdims = bcast_op.getBroadcastDimensions();
    llvm::SmallVector<int64_t, 4> op_starts(bcast_in_type.getRank(), 0);
    llvm::SmallVector<int64_t, 4> op_limits(bcast_in_type.getShape().begin(),
                                            bcast_in_type.getShape().end());
    for (int64_t j = 0; j < bcast_in_type.getRank(); ++j) {
      const int64_t d = bdims[j];
      if (bcast_in_type.getDimSize(j) == 1) {
        op_starts[j] = 0;
        op_limits[j] = 1;  // GENERIC: semantic size-1 broadcast axis
      } else {
        op_starts[j] = starts[d];
        op_limits[j] = limits[d];
      }
    }
    mlir::Value sliced_bcast_in =
        BuildUnitStrideSlice(builder, loc, bcast_in, op_starts, op_limits);
    if (auto new_s =
            sliced_bcast_in.getDefiningOp<mlir::stablehlo::SliceOp>()) {
      worklist.push_back(new_s);
    }
    mlir::Value new_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, out_type, sliced_bcast_in, bcast_op.getBroadcastDimensionsAttr());
    slice_op.replaceAllUsesWith(new_bcast);
    slice_op.erase();
    if (bcast_op.use_empty()) {
      bcast_op.erase();
    }
    return true;
  }

  // Remaining ops require operand to have a single use so we shrink work rather
  // than duplicating it.
  if (!operand.hasOneUse()) {
    return false;
  }

  // Case 3: Unary elementwise ops.
  if (llvm::isa<mlir::stablehlo::ConvertOp, mlir::stablehlo::AbsOp,
                mlir::stablehlo::NegOp, mlir::stablehlo::SqrtOp,
                mlir::stablehlo::RsqrtOp, mlir::stablehlo::ExpOp>(def_op)) {
    mlir::Value in_val = def_op->getOperand(0);
    mlir::Value sliced_in =
        BuildUnitStrideSlice(builder, loc, in_val, starts, limits);
    if (auto new_s = sliced_in.getDefiningOp<mlir::stablehlo::SliceOp>()) {
      worklist.push_back(new_s);
    }
    mlir::OperationState state(def_op->getLoc(), def_op->getName());
    state.addOperands(sliced_in);
    state.addTypes(out_type);
    state.addAttributes(def_op->getAttrs());
    mlir::Operation* new_op = builder.create(state);
    slice_op.replaceAllUsesWith(new_op->getResult(0));
    slice_op.erase();
    def_op->erase();
    return true;
  }

  // Case 4: Binary elementwise ops.
  if (llvm::isa<mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
                mlir::stablehlo::MulOp, mlir::stablehlo::DivOp,
                mlir::stablehlo::MaxOp, mlir::stablehlo::MinOp>(def_op)) {
    mlir::Value lhs = def_op->getOperand(0);
    mlir::Value rhs = def_op->getOperand(1);
    mlir::Value sliced_lhs =
        BuildUnitStrideSlice(builder, loc, lhs, starts, limits);
    mlir::Value sliced_rhs =
        (lhs == rhs) ? sliced_lhs
                     : BuildUnitStrideSlice(builder, loc, rhs, starts, limits);
    if (auto s_lhs = sliced_lhs.getDefiningOp<mlir::stablehlo::SliceOp>()) {
      worklist.push_back(s_lhs);
    }
    if (lhs != rhs) {
      if (auto s_rhs = sliced_rhs.getDefiningOp<mlir::stablehlo::SliceOp>()) {
        worklist.push_back(s_rhs);
      }
    }
    mlir::OperationState state(def_op->getLoc(), def_op->getName());
    state.addOperands({sliced_lhs, sliced_rhs});
    state.addTypes(out_type);
    state.addAttributes(def_op->getAttrs());
    mlir::Operation* new_op = builder.create(state);
    slice_op.replaceAllUsesWith(new_op->getResult(0));
    slice_op.erase();
    def_op->erase();
    return true;
  }

  // Case 5: ReshapeOp that only inserts or removes unit-sized (size == 1) dims.
  if (auto reshape_op = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(def_op)) {
    mlir::Value r_in = reshape_op.getOperand();
    mlir::RankedTensorType r_in_type;
    if (!IsStaticRankedTensor(r_in.getType(), &r_in_type)) {
      return false;
    }
    llvm::SmallVector<int64_t, 4> in_non_unit;
    llvm::SmallVector<int64_t, 4> out_non_unit;
    for (int64_t d = 0; d < r_in_type.getRank(); ++d) {
      if (r_in_type.getDimSize(d) != 1) {  // GENERIC: semantic non-unit axis
        in_non_unit.push_back(d);
      }
    }
    for (int64_t d = 0; d < in_type.getRank(); ++d) {
      if (in_type.getDimSize(d) != 1) {  // GENERIC: semantic non-unit axis
        out_non_unit.push_back(d);
      }
    }
    if (in_non_unit.size() != out_non_unit.size()) {
      return false;
    }
    llvm::SmallVector<int64_t, 4> op_starts(r_in_type.getRank(), 0);
    llvm::SmallVector<int64_t, 4> op_limits(r_in_type.getShape().begin(),
                                            r_in_type.getShape().end());
    for (size_t k = 0; k < in_non_unit.size(); ++k) {
      const int64_t in_d = in_non_unit[k];
      const int64_t out_d = out_non_unit[k];
      if (r_in_type.getDimSize(in_d) != in_type.getDimSize(out_d)) {
        return false;
      }
      op_starts[in_d] = starts[out_d];
      op_limits[in_d] = limits[out_d];
    }
    mlir::Value sliced_r_in =
        BuildUnitStrideSlice(builder, loc, r_in, op_starts, op_limits);
    if (auto new_s = sliced_r_in.getDefiningOp<mlir::stablehlo::SliceOp>()) {
      worklist.push_back(new_s);
    }
    mlir::Value new_reshape =
        builder.create<mlir::stablehlo::ReshapeOp>(loc, out_type, sliced_r_in);
    slice_op.replaceAllUsesWith(new_reshape);
    slice_op.erase();
    reshape_op.erase();
    return true;
  }

  // Case 6: Single-result ReduceOp (slicing unreduced output axes slices the
  // corresponding non-reduced input axes before the reduction).
  if (auto reduce_op = llvm::dyn_cast<mlir::stablehlo::ReduceOp>(def_op)) {
    if (reduce_op.getNumResults() != 1 || reduce_op.getInputs().size() != 1) {
      return false;  // GENERIC: semantic single-input single-result reduction
    }
    mlir::Value red_in = reduce_op.getInputs()[0];
    mlir::RankedTensorType red_in_type;
    if (!IsStaticRankedTensor(red_in.getType(), &red_in_type)) {
      return false;
    }
    llvm::DenseSet<int64_t> reduced_dims;
    for (int64_t d : reduce_op.getDimensions()) {
      reduced_dims.insert(d);
    }
    llvm::SmallVector<int64_t, 4> unreduced_in_dims;
    for (int64_t d = 0; d < red_in_type.getRank(); ++d) {
      if (!reduced_dims.contains(d)) {
        unreduced_in_dims.push_back(d);
      }
    }
    if (static_cast<int64_t>(unreduced_in_dims.size()) != in_type.getRank()) {
      return false;
    }
    llvm::SmallVector<int64_t, 4> op_starts(red_in_type.getRank(), 0);
    llvm::SmallVector<int64_t, 4> op_limits(red_in_type.getShape().begin(),
                                            red_in_type.getShape().end());
    for (int64_t k = 0; k < in_type.getRank(); ++k) {
      const int64_t in_d = unreduced_in_dims[k];
      op_starts[in_d] = starts[k];
      op_limits[in_d] = limits[k];
    }
    mlir::Value sliced_red_in =
        BuildUnitStrideSlice(builder, loc, red_in, op_starts, op_limits);
    if (auto new_s = sliced_red_in.getDefiningOp<mlir::stablehlo::SliceOp>()) {
      worklist.push_back(new_s);
    }
    reduce_op->moveBefore(slice_op);
    reduce_op->setOperand(0, sliced_red_in);
    reduce_op.getResult(0).setType(out_type);
    slice_op.replaceAllUsesWith(reduce_op.getResult(0));
    slice_op.erase();
    return true;
  }

  return false;
}

void PropagateSlicesUpstream(
    llvm::SmallVectorImpl<mlir::stablehlo::SliceOp>& worklist) {
  while (!worklist.empty()) {
    mlir::stablehlo::SliceOp slice_op = worklist.pop_back_val();
    if (!slice_op) {
      continue;
    }
    TryPushSingleSliceUpstream(slice_op, worklist);
  }
}

// Folds a single-user unit-stride spatial SliceOp consuming a ConvolutionOp
// directly into the ConvolutionOp's padding attribute.
bool FoldOutputSpatialSliceIntoConvolution(
    mlir::stablehlo::ConvolutionOp conv_op) {
  if (!conv_op->hasOneUse() || conv_op.getBatchGroupCount() !=
                                   1) {  // GENERIC: semantic single batch group
    return false;
  }
  auto slice_op =
      llvm::dyn_cast<mlir::stablehlo::SliceOp>(*conv_op->getUsers().begin());
  if (!slice_op || !IsUnitStrideSlice(slice_op)) {
    return false;
  }

  mlir::RankedTensorType conv_type;
  mlir::RankedTensorType slice_type;
  if (!IsStaticRankedTensor(conv_op.getType(), &conv_type) ||
      !IsStaticRankedTensor(slice_op.getType(), &slice_type)) {
    return false;
  }

  auto dnums = conv_op.getDimensionNumbers();
  const int64_t out_batch_dim = dnums.getOutputBatchDimension();
  const int64_t out_feat_dim = dnums.getOutputFeatureDimension();
  llvm::ArrayRef<int64_t> out_spatial_dims = dnums.getOutputSpatialDimensions();
  const size_t num_spatial = out_spatial_dims.size();

  llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
  if (starts[out_batch_dim] != 0 ||
      limits[out_batch_dim] != conv_type.getDimSize(out_batch_dim) ||
      starts[out_feat_dim] != 0 ||
      limits[out_feat_dim] != conv_type.getDimSize(out_feat_dim)) {
    return false;
  }

  if (auto lhs_dil = conv_op.getLhsDilation()) {
    for (int64_t d : *lhs_dil) {
      if (d != 1) {  // GENERIC: semantic unit lhs dilation
        return false;
      }
    }
  }
  if (auto win_rev = conv_op.getWindowReversal()) {
    for (bool rev : *win_rev) {
      if (rev) {
        return false;
      }
    }
  }

  llvm::SmallVector<int64_t, 3> strides(
      num_spatial,
      1);  // GENERIC: semantic default stride 1
  if (auto ws = conv_op.getWindowStrides()) {
    strides.assign(ws->begin(), ws->end());
  }
  llvm::SmallVector<int64_t, 6> padding(num_spatial * 2, 0);
  if (auto pad_attr = conv_op.getPadding()) {
    auto vals = pad_attr->getValues<int64_t>();
    padding.assign(vals.begin(), vals.end());
  }

  bool any_sliced = false;
  for (size_t i = 0; i < num_spatial; ++i) {
    const int64_t od = out_spatial_dims[i];
    const int64_t n_out = conv_type.getDimSize(od);
    const int64_t o_start = starts[od];
    const int64_t o_end = limits[od];
    if (o_start < 0 || o_end > n_out || o_start >= o_end) {
      return false;
    }
    if (o_start > 0 || o_end < n_out) {
      const int64_t pad_lo_sub = o_start * strides[i];
      const int64_t pad_hi_sub = (n_out - o_end) * strides[i];
      if (padding[2 * i] < pad_lo_sub || padding[2 * i + 1] < pad_hi_sub) {
        return false;  // GENERIC: semantic non-negative convolution padding
                       // check
      }
      padding[2 * i] -= pad_lo_sub;
      padding[2 * i + 1] -= pad_hi_sub;
      any_sliced = true;
    }
  }
  if (!any_sliced) {
    return false;
  }

  mlir::OpBuilder builder(conv_op);
  auto pad_type = mlir::RankedTensorType::get(
      {static_cast<int64_t>(num_spatial), 2}, builder.getI64Type());
  conv_op.setPaddingAttr(mlir::DenseIntElementsAttr::get(pad_type, padding));
  conv_op.getResult().setType(slice_type);
  slice_op.replaceAllUsesWith(conv_op.getResult());
  slice_op.erase();
  return true;
}

// Trims kernel spatial coordinates that only ever multiply zero-padded input
// positions for all output coordinates o in [0, N_out).
std::optional<mlir::stablehlo::SliceOp> TrimConvolutionKernelOverhang(
    mlir::stablehlo::ConvolutionOp conv_op) {
  mlir::RankedTensorType lhs_type;
  mlir::RankedTensorType rhs_type;
  mlir::RankedTensorType out_type;
  if (!IsStaticRankedTensor(conv_op.getLhs().getType(), &lhs_type) ||
      !IsStaticRankedTensor(conv_op.getRhs().getType(), &rhs_type) ||
      !IsStaticRankedTensor(conv_op.getType(), &out_type)) {
    return std::nullopt;
  }

  if (auto lhs_dil = conv_op.getLhsDilation()) {
    for (int64_t d : *lhs_dil) {
      if (d != 1) {  // GENERIC: semantic unit lhs dilation
        return std::nullopt;
      }
    }
  }
  if (auto win_rev = conv_op.getWindowReversal()) {
    for (bool rev : *win_rev) {
      if (rev) {
        return std::nullopt;
      }
    }
  }

  auto dnums = conv_op.getDimensionNumbers();
  llvm::ArrayRef<int64_t> in_sp = dnums.getInputSpatialDimensions();
  llvm::ArrayRef<int64_t> ker_sp = dnums.getKernelSpatialDimensions();
  llvm::ArrayRef<int64_t> out_sp = dnums.getOutputSpatialDimensions();
  const size_t num_spatial = in_sp.size();
  if (num_spatial == 0 || ker_sp.size() != num_spatial ||
      out_sp.size() != num_spatial) {
    return std::nullopt;
  }

  llvm::SmallVector<int64_t, 3> strides(
      num_spatial,
      1);  // GENERIC: semantic default stride 1
  if (auto ws = conv_op.getWindowStrides()) {
    strides.assign(ws->begin(), ws->end());
  }
  llvm::SmallVector<int64_t, 3> rhs_dil(
      num_spatial,
      1);  // GENERIC: semantic default dilation 1
  if (auto rd = conv_op.getRhsDilation()) {
    rhs_dil.assign(rd->begin(), rd->end());
  }
  llvm::SmallVector<int64_t, 6> padding(num_spatial * 2, 0);
  if (auto pad_attr = conv_op.getPadding()) {
    auto vals = pad_attr->getValues<int64_t>();
    padding.assign(vals.begin(), vals.end());
  }

  llvm::SmallVector<int64_t, 4> rhs_starts(rhs_type.getRank(), 0);
  llvm::SmallVector<int64_t, 4> rhs_limits(rhs_type.getShape().begin(),
                                           rhs_type.getShape().end());
  bool any_trimmed = false;

  for (size_t d = 0; d < num_spatial; ++d) {
    const int64_t n_in = lhs_type.getDimSize(in_sp[d]);
    const int64_t n_out = out_type.getDimSize(out_sp[d]);
    const int64_t k_size = rhs_type.getDimSize(ker_sp[d]);
    const int64_t s = strides[d];
    const int64_t w = rhs_dil[d];
    const int64_t pad_lo = padding[2 * d];
    const int64_t pad_hi = padding[2 * d + 1];

    if (n_in <= 0 || n_out <= 0 || k_size <= 0 || s <= 0 || w <= 0) {
      return std::nullopt;
    }

    // Active kernel index k must satisfy:
    //   0 <= o * s + k * w - pad_lo <= n_in - 1  for some o in [0, n_out - 1].
    const int64_t num_min = pad_lo - (n_out - 1) * s;
    const int64_t k_min =
        (num_min > 0) ? ((num_min + w - 1) / w) : 0;  // GENERIC: semantic ceil
    const int64_t num_max = n_in - 1 + pad_lo;
    if (num_max < 0) {
      return std::nullopt;
    }
    const int64_t k_max = std::min(k_size - 1, num_max / w);
    if (k_min < 0 || k_min > k_max || k_max >= k_size) {
      return std::nullopt;
    }

    if (k_min > 0 || k_max < k_size - 1) {
      const int64_t new_pad_lo = pad_lo - k_min * w;
      const int64_t new_pad_hi = pad_hi - (k_size - 1 - k_max) * w;
      if (new_pad_lo < 0 || new_pad_hi < 0) {
        return std::nullopt;  // GENERIC: semantic non-negative convolution
                              // padding check
      }
      rhs_starts[ker_sp[d]] = k_min;
      rhs_limits[ker_sp[d]] = k_max + 1;
      padding[2 * d] = new_pad_lo;
      padding[2 * d + 1] = new_pad_hi;
      any_trimmed = true;
    }
  }

  if (!any_trimmed) {
    return std::nullopt;
  }

  mlir::OpBuilder builder(conv_op);
  mlir::Value sliced_rhs = BuildUnitStrideSlice(
      builder, conv_op.getLoc(), conv_op.getRhs(), rhs_starts, rhs_limits);
  auto pad_type = mlir::RankedTensorType::get(
      {static_cast<int64_t>(num_spatial), 2}, builder.getI64Type());
  conv_op.setPaddingAttr(mlir::DenseIntElementsAttr::get(pad_type, padding));
  conv_op->setOperand(1, sliced_rhs);
  if (auto s = sliced_rhs.getDefiningOp<mlir::stablehlo::SliceOp>()) {
    return s;
  }
  return std::nullopt;
}

void EraseDeadPureOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation*> dead_ops;
    func.walk([&](mlir::Operation* op) {
      if (op != func.getOperation() && op->use_empty() &&
          (mlir::isOpTriviallyDead(op) ||
           llvm::isa<mlir::stablehlo::ReduceOp>(op))) {
        dead_ops.push_back(op);
      }
    });
    for (mlir::Operation* op : dead_ops) {
      if (op->getBlock() != nullptr) {
        op->erase();
        changed = true;
      }
    }
  }
}

struct TrimConvolutionPaddingOverhangPass
    : public mlir::PassWrapper<TrimConvolutionPaddingOverhangPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TrimConvolutionPaddingOverhangPass)

  llvm::StringRef getArgument() const override {
    return "trim-convolution-padding-overhang";
  }
  llvm::StringRef getDescription() const override {
    return "Folds spatial output slices into convolution padding, trims "
           "overhanging kernel coordinates that only multiply zero padding, "
           "and canonicalizes unit-dim weight-norm reductions to 1D.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    // Step 1: Canonicalize unit-dim weight-norm reductions to 1D.
    CanonicalizeUnitDimWeightNormBroadcasts(func);
    EraseDeadPureOps(func);

    // Step 2: Push any slice consuming a convolution+bias AddOp onto the
    // ConvolutionOp so FoldOutputSpatialSliceIntoConvolution sees it directly.
    llvm::SmallVector<mlir::stablehlo::SliceOp> initial_slices;
    func.walk([&](mlir::stablehlo::SliceOp s) {
      if (auto add_op =
              s.getOperand().getDefiningOp<mlir::stablehlo::AddOp>()) {
        if (add_op.getLhs().getDefiningOp<mlir::stablehlo::ConvolutionOp>() ||
            add_op.getRhs().getDefiningOp<mlir::stablehlo::ConvolutionOp>()) {
          initial_slices.push_back(s);
        }
      }
    });
    PropagateSlicesUpstream(initial_slices);

    // Step 3: Fold spatial output slices into ConvolutionOp padding and trim
    // overhanging kernel coordinates, then propagate kernel slices upstream.
    llvm::SmallVector<mlir::stablehlo::ConvolutionOp> conv_ops;
    func.walk(
        [&](mlir::stablehlo::ConvolutionOp op) { conv_ops.push_back(op); });

    llvm::SmallVector<mlir::stablehlo::SliceOp> kernel_slices;
    for (mlir::stablehlo::ConvolutionOp conv_op : conv_ops) {
      FoldOutputSpatialSliceIntoConvolution(conv_op);
      if (auto rhs_slice = TrimConvolutionKernelOverhang(conv_op)) {
        kernel_slices.push_back(*rhs_slice);
      }
    }
    PropagateSlicesUpstream(kernel_slices);
    EraseDeadPureOps(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createTrimConvolutionPaddingOverhangPass() {
  return std::make_unique<TrimConvolutionPaddingOverhangPass>();
}

}  // namespace torch_tpu
