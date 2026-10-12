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

#include "csrc/mlir/passes/fuse_rotary_half_rotation.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

constexpr int64_t kMinRotaryHeadDim = 32;
constexpr int64_t kMaxRotaryHeadDim = 256;

mlir::Value SkipBroadcastInDim(mlir::Value val) {
  while (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    val = bcast.getOperand();
  }
  return val;
}

bool IsSplatFloatEqual(mlir::Value val, double target) {
  auto const_op =
      SkipBroadcastInDim(val).getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr || !attr.isSplat() ||
      !mlir::isa<mlir::FloatType>(attr.getElementType())) {
    return false;
  }
  return attr.getSplatValue<llvm::APFloat>().isExactlyValue(target);
}

mlir::Value MatchNegatedValue(mlir::Value val) {
  if (auto neg_op = val.getDefiningOp<mlir::stablehlo::NegOp>()) {
    return neg_op.getOperand();
  }
  if (auto mul_op = val.getDefiningOp<mlir::stablehlo::MulOp>()) {
    if (IsSplatFloatEqual(mul_op.getRhs(), -1.0)) {
      return mul_op.getLhs();
    }
    if (IsSplatFloatEqual(mul_op.getLhs(), -1.0)) {
      return mul_op.getRhs();
    }
  }
  if (auto sub_op = val.getDefiningOp<mlir::stablehlo::SubtractOp>()) {
    if (IsSplatFloatEqual(sub_op.getLhs(), 0.0)) {
      return sub_op.getRhs();
    }
  }
  return mlir::Value();
}

bool IsSupportedRotaryOutputType(mlir::RankedTensorType out_type,
                                 uint64_t concat_dim) {
  if (!out_type || !out_type.hasStaticShape() || out_type.getRank() < 2 ||
      !mlir::isa<mlir::FloatType>(out_type.getElementType())) {
    return false;
  }
  const int64_t last_axis = out_type.getRank() - 1;
  if (concat_dim != static_cast<uint64_t>(last_axis)) {
    return false;
  }
  const int64_t d_full = out_type.getDimSize(last_axis);
  return d_full % 2 == 0 && d_full >= kMinRotaryHeadDim &&
         d_full <= kMaxRotaryHeadDim;
}

bool MatchHalfRotationBranches(mlir::stablehlo::ConcatenateOp concat_op,
                               mlir::RankedTensorType& out_type,
                               mlir::Value& out_lo_branch,
                               mlir::Value& out_hi_branch,
                               bool& out_negate_upper) {
  if (concat_op.getInputs().size() != 2) {
    return false;
  }
  out_type = mlir::dyn_cast<mlir::RankedTensorType>(concat_op.getType());
  if (!IsSupportedRotaryOutputType(out_type, concat_op.getDimension())) {
    return false;
  }
  const mlir::Value first = concat_op.getInputs()[0];
  const mlir::Value second = concat_op.getInputs()[1];
  const mlir::Value neg_first = MatchNegatedValue(first);
  const mlir::Value neg_second = MatchNegatedValue(second);
  if (neg_first && !neg_second) {
    out_hi_branch = neg_first;
    out_lo_branch = second;
    out_negate_upper = true;
    return true;
  }
  if (!neg_first && neg_second) {
    out_hi_branch = first;
    out_lo_branch = neg_second;
    out_negate_upper = false;
    return true;
  }
  return false;
}

bool IsLastDimHalfSlice(mlir::Value val, int64_t expected_start,
                        int64_t expected_limit, mlir::Value& out_src) {
  auto slice_op = val.getDefiningOp<mlir::stablehlo::SliceOp>();
  if (!slice_op) {
    return false;
  }
  auto src_type =
      mlir::dyn_cast<mlir::RankedTensorType>(slice_op.getOperand().getType());
  if (!src_type || !src_type.hasStaticShape() || src_type.getRank() < 2) {
    return false;
  }
  const int64_t rank = src_type.getRank();
  if (src_type.getDimSize(rank - 1) < expected_limit) {
    return false;
  }
  const llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  const llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
  const llvm::ArrayRef<int64_t> strides = slice_op.getStrides();
  for (int64_t d = 0; d < rank - 1; ++d) {
    if (starts[d] != 0 || limits[d] != src_type.getDimSize(d) ||
        strides[d] != 1) {
      return false;
    }
  }
  if (starts[rank - 1] != expected_start ||
      limits[rank - 1] != expected_limit || strides[rank - 1] != 1) {
    return false;
  }
  out_src = slice_op.getOperand();
  return true;
}

mlir::Value SliceLastDimPrefixIfNeeded(mlir::PatternRewriter& rewriter,
                                       mlir::Location loc, mlir::Value src,
                                       int64_t d_full) {
  auto src_type = mlir::cast<mlir::RankedTensorType>(src.getType());
  const int64_t rank = src_type.getRank();
  if (src_type.getDimSize(rank - 1) == d_full) {
    return src;
  }
  llvm::SmallVector<int64_t> starts(rank, 0);
  llvm::SmallVector<int64_t> limits(src_type.getShape().begin(),
                                    src_type.getShape().end());
  limits[rank - 1] = d_full;
  llvm::SmallVector<int64_t> strides(rank, 1);
  auto sliced_type =
      mlir::RankedTensorType::get(limits, src_type.getElementType());
  return rewriter.create<mlir::stablehlo::SliceOp>(
      loc, sliced_type, src, rewriter.getDenseI64ArrayAttr(starts),
      rewriter.getDenseI64ArrayAttr(limits),
      rewriter.getDenseI64ArrayAttr(strides));
}

bool MatchHoistableTransposeInputs(
    mlir::stablehlo::TransposeOp t_lo, mlir::stablehlo::TransposeOp t_hi,
    int64_t d_full, mlir::Value& out_src,
    mlir::RankedTensorType& out_full_reshape_type) {
  const llvm::ArrayRef<int64_t> perm = t_lo.getPermutation();
  if (perm != t_hi.getPermutation() || perm.empty()) {
    return false;
  }
  const int64_t t_in_axis = perm.back();
  mlir::Value s_lo_val = t_lo.getOperand();
  mlir::Value s_hi_val = t_hi.getOperand();
  out_full_reshape_type = nullptr;

  auto r_lo = s_lo_val.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  auto r_hi = s_hi_val.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (r_lo && r_hi && r_lo.getType() == r_hi.getType()) {
    auto r_type = mlir::cast<mlir::RankedTensorType>(r_lo.getType());
    if (t_in_axis < 0 || t_in_axis >= r_type.getRank() ||
        r_type.getDimSize(t_in_axis) != d_full / 2) {
      return false;
    }
    llvm::SmallVector<int64_t> full_shape(r_type.getShape().begin(),
                                          r_type.getShape().end());
    full_shape[t_in_axis] = d_full;
    out_full_reshape_type =
        mlir::RankedTensorType::get(full_shape, r_type.getElementType());
    s_lo_val = r_lo.getOperand();
    s_hi_val = r_hi.getOperand();
  } else if (t_in_axis != static_cast<int64_t>(perm.size()) - 1) {
    return false;
  }

  mlir::Value src_lo;
  mlir::Value src_hi;
  const int64_t half_d = d_full / 2;
  if (!IsLastDimHalfSlice(s_lo_val, 0, half_d, src_lo) ||
      !IsLastDimHalfSlice(s_hi_val, half_d, d_full, src_hi) ||
      src_lo != src_hi) {
    return false;
  }
  if (out_full_reshape_type) {
    auto s_lo_type = mlir::cast<mlir::RankedTensorType>(s_lo_val.getType());
    if (s_lo_type.getNumElements() * 2 !=
        out_full_reshape_type.getNumElements()) {
      return false;
    }
  }
  out_src = src_lo;
  return true;
}

mlir::Value EmitHoistedHalfSliceConcat(mlir::PatternRewriter& rewriter,
                                       mlir::Location loc,
                                       mlir::Value transposed_full,
                                       mlir::RankedTensorType out_type,
                                       bool negate_upper) {
  const int64_t rank = out_type.getRank();
  const int64_t d_full = out_type.getDimSize(rank - 1);
  const int64_t half_d = d_full / 2;

  llvm::SmallVector<int64_t> starts_lo(rank, 0);
  llvm::SmallVector<int64_t> limits_lo(out_type.getShape().begin(),
                                       out_type.getShape().end());
  limits_lo[rank - 1] = half_d;

  llvm::SmallVector<int64_t> starts_hi(rank, 0);
  starts_hi[rank - 1] = half_d;
  llvm::SmallVector<int64_t> limits_hi(out_type.getShape().begin(),
                                       out_type.getShape().end());
  llvm::SmallVector<int64_t> strides(rank, 1);

  auto half_type =
      mlir::RankedTensorType::get(limits_lo, out_type.getElementType());
  mlir::Value new_s_lo = rewriter.create<mlir::stablehlo::SliceOp>(
      loc, half_type, transposed_full, rewriter.getDenseI64ArrayAttr(starts_lo),
      rewriter.getDenseI64ArrayAttr(limits_lo),
      rewriter.getDenseI64ArrayAttr(strides));
  mlir::Value new_s_hi = rewriter.create<mlir::stablehlo::SliceOp>(
      loc, half_type, transposed_full, rewriter.getDenseI64ArrayAttr(starts_hi),
      rewriter.getDenseI64ArrayAttr(limits_hi),
      rewriter.getDenseI64ArrayAttr(strides));

  mlir::Value first =
      negate_upper
          ? rewriter.create<mlir::stablehlo::NegOp>(loc, half_type, new_s_hi)
                .getResult()
          : new_s_hi;
  mlir::Value second =
      negate_upper
          ? new_s_lo
          : rewriter.create<mlir::stablehlo::NegOp>(loc, half_type, new_s_lo)
                .getResult();
  return rewriter.create<mlir::stablehlo::ConcatenateOp>(
      loc, out_type, mlir::ValueRange{first, second},
      rewriter.getI64IntegerAttr(rank - 1));
}

mlir::Value CreateHalfRotationMatrixConstant(mlir::OpBuilder& builder,
                                             mlir::Location loc, int64_t dim_d,
                                             mlir::FloatType float_type,
                                             bool negate_upper) {
  const int64_t half_d = dim_d / 2;
  const llvm::fltSemantics& semantics = float_type.getFloatSemantics();
  const llvm::APFloat zero(semantics, 0);
  llvm::APFloat pos_one(1.0);
  llvm::APFloat neg_one(-1.0);
  bool loses_info = false;
  pos_one.convert(semantics, llvm::APFloat::rmNearestTiesToEven, &loses_info);
  neg_one.convert(semantics, llvm::APFloat::rmNearestTiesToEven, &loses_info);

  llvm::SmallVector<llvm::APFloat> values(dim_d * dim_d, zero);
  const llvm::APFloat lower_left_val = negate_upper ? neg_one : pos_one;
  const llvm::APFloat upper_right_val = negate_upper ? pos_one : neg_one;
  for (int64_t i = 0; i < half_d; ++i) {
    values[(i + half_d) * dim_d + i] = lower_left_val;
    values[i * dim_d + (i + half_d)] = upper_right_val;
  }
  auto mat_type = mlir::RankedTensorType::get({dim_d, dim_d}, float_type);
  auto dense_attr = mlir::DenseElementsAttr::get(mat_type, values);
  return builder.create<mlir::stablehlo::ConstantOp>(loc, mat_type, dense_attr);
}

struct HoistTransposeAboveHalfSliceConcatPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern<mlir::stablehlo::ConcatenateOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ConcatenateOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::RankedTensorType out_type;
    mlir::Value lo_branch;
    mlir::Value hi_branch;
    bool negate_upper = false;
    if (!MatchHalfRotationBranches(op, out_type, lo_branch, hi_branch,
                                   negate_upper)) {
      return mlir::failure();
    }
    auto t_lo = lo_branch.getDefiningOp<mlir::stablehlo::TransposeOp>();
    auto t_hi = hi_branch.getDefiningOp<mlir::stablehlo::TransposeOp>();
    if (!t_lo || !t_hi) {
      return mlir::failure();
    }
    const int64_t d_full = out_type.getDimSize(out_type.getRank() - 1);
    mlir::Value src;
    mlir::RankedTensorType full_reshape_type;
    if (!MatchHoistableTransposeInputs(t_lo, t_hi, d_full, src,
                                       full_reshape_type)) {
      return mlir::failure();
    }
    const mlir::Location loc = op.getLoc();
    mlir::Value pre_transpose =
        SliceLastDimPrefixIfNeeded(rewriter, loc, src, d_full);
    if (full_reshape_type) {
      pre_transpose = rewriter.create<mlir::stablehlo::ReshapeOp>(
          loc, full_reshape_type, pre_transpose);
    }
    mlir::Value transposed_full = rewriter.create<mlir::stablehlo::TransposeOp>(
        loc, out_type, pre_transpose, t_lo.getPermutationAttr());
    mlir::Value new_concat = EmitHoistedHalfSliceConcat(
        rewriter, loc, transposed_full, out_type, negate_upper);
    rewriter.replaceOp(op, new_concat);
    return mlir::success();
  }
};

struct RewriteHalfRotationToDotGeneralPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern<mlir::stablehlo::ConcatenateOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ConcatenateOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::RankedTensorType out_type;
    mlir::Value lo_branch;
    mlir::Value hi_branch;
    bool negate_upper = false;
    if (!MatchHalfRotationBranches(op, out_type, lo_branch, hi_branch,
                                   negate_upper)) {
      return mlir::failure();
    }
    const int64_t rank = out_type.getRank();
    const int64_t d_full = out_type.getDimSize(rank - 1);
    const int64_t half_d = d_full / 2;
    mlir::Value src_lo;
    mlir::Value src_hi;
    if (!IsLastDimHalfSlice(lo_branch, 0, half_d, src_lo) ||
        !IsLastDimHalfSlice(hi_branch, half_d, d_full, src_hi) ||
        src_lo != src_hi) {
      return mlir::failure();
    }
    const mlir::Location loc = op.getLoc();
    mlir::Value rotary_input =
        SliceLastDimPrefixIfNeeded(rewriter, loc, src_lo, d_full);
    auto float_type = mlir::cast<mlir::FloatType>(out_type.getElementType());
    mlir::Value j_mat = CreateHalfRotationMatrixConstant(
        rewriter, loc, d_full, float_type, negate_upper);
    auto dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*lhsBatchingDimensions=*/{},
        /*rhsBatchingDimensions=*/{},
        /*lhsContractingDimensions=*/{rank - 1},
        /*rhsContractingDimensions=*/{0});
    rewriter.replaceOpWithNewOp<mlir::stablehlo::DotGeneralOp>(
        op, out_type, rotary_input, j_mat, dot_dims,
        /*precision_config=*/mlir::ArrayAttr{},
        /*algorithm=*/mlir::stablehlo::DotAlgorithmAttr{});
    return mlir::success();
  }
};

struct FuseRotaryHalfRotationPass
    : public mlir::PassWrapper<FuseRotaryHalfRotationPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseRotaryHalfRotationPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-fuse-rotary-half-rotation";
  }
  llvm::StringRef getDescription() const override {
    return "Fuses rotary half-rotation slice/negate/concatenate patterns into "
           "stablehlo.dot_general.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<HoistTransposeAboveHalfSliceConcatPattern,
                 RewriteHalfRotationToDotGeneralPattern>(ctx);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createFuseRotaryHalfRotationPass() {
  return std::make_unique<FuseRotaryHalfRotationPass>();
}

}  // namespace torch_tpu
