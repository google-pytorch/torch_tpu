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

#include "csrc/mlir/passes/cancel_inverse_view_transposes/cancel_inverse_view_transposes.h"  // NOLINT

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // NOLINT
#include "mlir/IR/BuiltinAttributes.h"     // NOLINT
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"  // NOLINT
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // NOLINT
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

[[nodiscard]] bool IsIdentityPerm(llvm::ArrayRef<int64_t> perm) {
  for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
    if (perm[i] != i) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] llvm::SmallVector<int64_t> GetNonUnitDimSizes(
    llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t> sizes;
  for (const int64_t d : shape) {
    if (d > 1) {
      sizes.push_back(d);
    }
  }
  return sizes;
}

[[nodiscard]] bool ComputeSqueezedPermutation(
    llvm::ArrayRef<int64_t> in_shape, llvm::ArrayRef<int64_t> perm,
    llvm::SmallVectorImpl<int64_t>& squeezed_perm) {
  if (in_shape.size() != perm.size()) {
    return false;
  }
  llvm::SmallVector<int64_t> axis_to_squeezed(in_shape.size(), -1);
  int64_t next_idx = 0;
  for (size_t i = 0; i < in_shape.size(); ++i) {
    if (in_shape[i] > 1) {
      axis_to_squeezed[i] = next_idx++;
    }
  }
  squeezed_perm.clear();
  squeezed_perm.reserve(next_idx);
  for (const int64_t d : perm) {
    if (d < 0 || d >= static_cast<int64_t>(in_shape.size())) {
      return false;
    }
    if (axis_to_squeezed[d] != -1) {
      squeezed_perm.push_back(axis_to_squeezed[d]);
    }
  }
  return static_cast<int64_t>(squeezed_perm.size()) == next_idx;
}

[[nodiscard]] mlir::Value PeelUnitDimPreservingReshapes(
    mlir::Value val, llvm::ArrayRef<int64_t> non_unit_sizes) {
  while (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    auto in_type =
        mlir::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
    if (!in_type || !in_type.hasStaticShape() ||
        GetNonUnitDimSizes(in_type.getShape()) != non_unit_sizes) {
      break;
    }
    val = resh.getOperand();
  }
  return val;
}

// Reshapes `val` to `dst_type` if `val.getType() != dst_type`.
// In StableHLO, reshape ∘ reshape is always valid and folded by standard
// canonicalization.
[[nodiscard]] mlir::Value EmitReshapeIfNeeded(mlir::PatternRewriter& rewriter,
                                              mlir::Location loc,
                                              mlir::Value val,
                                              mlir::RankedTensorType dst_type) {
  if (val.getType() == dst_type) {
    return val;
  }
  return rewriter.create<mlir::stablehlo::ReshapeOp>(loc, dst_type, val);
}

// Cancels `TransposeOp(views(TransposeOp(x, perm1)), perm2)` across
// layout-invariant operations (a single slice with arbitrary strides, or a
// chain of unit-stride slices, plus unit-dimension-preserving reshapes) when
// `perm1` and `perm2` are squeezed inverses.
bool MatchAndCancelTransposeSlice(mlir::stablehlo::TransposeOp tr2,
                                  mlir::PatternRewriter& rewriter) {
  auto tr2_type = mlir::dyn_cast<mlir::RankedTensorType>(tr2.getType());
  auto tr2_in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(tr2.getOperand().getType());
  if (!tr2_type || !tr2_type.hasStaticShape() || !tr2_in_type ||
      !tr2_in_type.hasStaticShape()) {
    return false;
  }

  // 1. Peel unit-dim reshapes on the output side of the slice chain (before
  // tr2).
  llvm::SmallVector<int64_t> non_unit_before_tr2 =
      GetNonUnitDimSizes(tr2_in_type.getShape());
  mlir::Value post_slice_peeled =
      PeelUnitDimPreservingReshapes(tr2.getOperand(), non_unit_before_tr2);

  auto outer_slice =
      post_slice_peeled.getDefiningOp<mlir::stablehlo::SliceOp>();
  if (!outer_slice) {
    return false;
  }
  auto outer_slice_type =
      mlir::dyn_cast<mlir::RankedTensorType>(outer_slice.getType());
  if (!outer_slice_type || !outer_slice_type.hasStaticShape()) {
    return false;
  }
  const int64_t slice_rank = outer_slice_type.getRank();

  llvm::SmallVector<int64_t> cum_starts(outer_slice.getStartIndices().begin(),
                                        outer_slice.getStartIndices().end());
  llvm::SmallVector<int64_t> cum_limits(outer_slice.getLimitIndices().begin(),
                                        outer_slice.getLimitIndices().end());
  llvm::SmallVector<int64_t> cum_strides(outer_slice.getStrides().begin(),
                                         outer_slice.getStrides().end());
  mlir::Value slice_src = outer_slice.getOperand();

  // Accumulate chained unit-stride slices. Strided slices cannot be chained.
  while (auto prev_slice =
             slice_src.getDefiningOp<mlir::stablehlo::SliceOp>()) {
    bool all_unit_strides = true;
    for (const int64_t st : cum_strides) {
      if (st != 1) {
        all_unit_strides = false;
        break;
      }
    }
    for (const int64_t st : prev_slice.getStrides()) {
      if (st != 1) {
        all_unit_strides = false;
        break;
      }
    }
    if (!all_unit_strides) {
      break;
    }
    llvm::ArrayRef<int64_t> p_starts = prev_slice.getStartIndices();
    for (int64_t i = 0; i < slice_rank; ++i) {
      cum_starts[i] += p_starts[i];
      cum_limits[i] += p_starts[i];
    }
    slice_src = prev_slice.getOperand();
  }

  // If there are unabsorbed SliceOps in slice_src (e.g. strided slice chained
  // into another slice), do not cancel.
  if (slice_src.getDefiningOp<mlir::stablehlo::SliceOp>()) {
    return false;
  }

  auto slice_in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(slice_src.getType());
  if (!slice_in_type || !slice_in_type.hasStaticShape()) {
    return false;
  }

  // 2. Peel unit-dim reshapes between the slice chain and tr1.
  llvm::SmallVector<int64_t> non_unit_slice_in =
      GetNonUnitDimSizes(slice_in_type.getShape());
  mlir::Value pre_slice_peeled =
      PeelUnitDimPreservingReshapes(slice_src, non_unit_slice_in);

  auto tr1 = pre_slice_peeled.getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!tr1) {
    return false;
  }
  auto x = tr1.getOperand();
  auto x_type = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
  auto tr1_type = mlir::dyn_cast<mlir::RankedTensorType>(tr1.getType());
  if (!x_type || !x_type.hasStaticShape() || !tr1_type ||
      !tr1_type.hasStaticShape()) {
    return false;
  }

  // 3. Compute squeezed permutations for tr1 and tr2.
  llvm::SmallVector<int64_t> squeezed_perm1;
  if (!ComputeSqueezedPermutation(x_type.getShape(), tr1.getPermutation(),
                                  squeezed_perm1)) {
    return false;
  }
  llvm::SmallVector<int64_t> squeezed_perm2;
  if (!ComputeSqueezedPermutation(tr2_in_type.getShape(), tr2.getPermutation(),
                                  squeezed_perm2)) {
    return false;
  }
  // TODO(gunhyun): Slices that shrink a non-unit dimension down to size 1
  // reduce the squeezed rank of tr2. Support aligning squeezed axes across
  // size-1 slices in a follow-up.
  if (squeezed_perm1.size() != squeezed_perm2.size()) {
    return false;
  }
  const size_t num_squeezed = squeezed_perm1.size();
  llvm::SmallVector<int64_t> composed_perm(num_squeezed);
  for (size_t i = 0; i < num_squeezed; ++i) {
    composed_perm[i] = squeezed_perm1[squeezed_perm2[i]];
  }
  if (!IsIdentityPerm(composed_perm)) {
    return false;
  }

  // 4. Map slice bounds and strides through the squeezed permutation.
  llvm::SmallVector<int64_t> non_unit_axes_slice_in;
  for (int64_t a = 0; a < slice_in_type.getRank(); ++a) {
    if (slice_in_type.getShape()[a] > 1) {
      non_unit_axes_slice_in.push_back(a);
    } else {
      // Unit-dim slice must cover the full unit dimension (0:1:1).
      if (cum_starts[a] != 0 || cum_limits[a] != 1 || cum_strides[a] != 1) {
        return false;
      }
    }
  }
  if (non_unit_axes_slice_in.size() != num_squeezed) {
    return false;
  }

  llvm::SmallVector<int64_t> non_unit_axes_x;
  for (int64_t i = 0; i < x_type.getRank(); ++i) {
    if (x_type.getShape()[i] > 1) {
      non_unit_axes_x.push_back(i);
    }
  }
  if (non_unit_axes_x.size() != num_squeezed) {
    return false;
  }

  const int64_t x_rank = x_type.getRank();
  llvm::SmallVector<int64_t> orig_starts(x_rank, 0);
  llvm::SmallVector<int64_t> orig_limits(x_rank);
  for (int64_t i = 0; i < x_rank; ++i) {
    orig_limits[i] = x_type.getShape()[i];
  }
  llvm::SmallVector<int64_t> orig_strides(x_rank, 1);

  llvm::SmallVector<int64_t> sliced_shape(x_type.getShape().begin(),
                                          x_type.getShape().end());

  for (size_t m = 0; m < num_squeezed; ++m) {
    const int64_t k = squeezed_perm1[m];
    const int64_t axis_slice = non_unit_axes_slice_in[m];
    const int64_t axis_x = non_unit_axes_x[k];

    orig_starts[axis_x] = cum_starts[axis_slice];
    orig_limits[axis_x] = cum_limits[axis_slice];
    orig_strides[axis_x] = cum_strides[axis_slice];

    const int64_t diff = orig_limits[axis_x] - orig_starts[axis_x];
    const int64_t st = orig_strides[axis_x];
    sliced_shape[axis_x] = (diff + st - 1) / st;
  }

  auto sliced_type =
      mlir::RankedTensorType::get(sliced_shape, x_type.getElementType());
  mlir::Value sliced = rewriter.create<mlir::stablehlo::SliceOp>(
      tr2.getLoc(), sliced_type, x, rewriter.getDenseI64ArrayAttr(orig_starts),
      rewriter.getDenseI64ArrayAttr(orig_limits),
      rewriter.getDenseI64ArrayAttr(orig_strides));

  // 5. Emit reshape to tr2_type if needed.
  mlir::Value rep =
      EmitReshapeIfNeeded(rewriter, tr2.getLoc(), sliced, tr2_type);
  rewriter.replaceOp(tr2, rep);
  return true;
}

// Cancels `TransposeOp([unit-dim ReshapeOp]*(TransposeOp(x, perm1)), perm2)`
// when the composition of permutations on non-unit dimensions is the identity.
bool MatchAndCancelTransposeReshape(mlir::stablehlo::TransposeOp tr2,
                                    mlir::PatternRewriter& rewriter) {
  auto tr2_type = mlir::dyn_cast<mlir::RankedTensorType>(tr2.getType());
  auto tr2_in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(tr2.getOperand().getType());
  if (!tr2_type || !tr2_type.hasStaticShape() || !tr2_in_type ||
      !tr2_in_type.hasStaticShape()) {
    return false;
  }

  llvm::SmallVector<int64_t> non_unit_before_tr2 =
      GetNonUnitDimSizes(tr2_in_type.getShape());
  llvm::SmallVector<int64_t> squeezed_perm2;
  if (!ComputeSqueezedPermutation(tr2_in_type.getShape(), tr2.getPermutation(),
                                  squeezed_perm2) ||
      // If tr2 is already an identity on non-unit dims, MLIR
      // canonicalizer handles it.
      IsIdentityPerm(squeezed_perm2)) {
    return false;
  }

  mlir::Value peeled =
      PeelUnitDimPreservingReshapes(tr2.getOperand(), non_unit_before_tr2);
  auto tr1 = peeled.getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!tr1) {
    return false;
  }
  auto tr1_in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(tr1.getOperand().getType());
  if (!tr1_in_type || !tr1_in_type.hasStaticShape()) {
    return false;
  }

  llvm::SmallVector<int64_t> squeezed_perm1;
  if (!ComputeSqueezedPermutation(tr1_in_type.getShape(), tr1.getPermutation(),
                                  squeezed_perm1) ||
      squeezed_perm1.size() != squeezed_perm2.size() ||
      // If tr1 is already an identity on non-unit dims, MLIR
      // canonicalizer handles it.
      IsIdentityPerm(squeezed_perm1)) {
    return false;
  }

  llvm::SmallVector<int64_t> composed_perm(squeezed_perm2.size());
  for (size_t i = 0; i < squeezed_perm2.size(); ++i) {
    composed_perm[i] = squeezed_perm1[squeezed_perm2[i]];
  }
  if (!IsIdentityPerm(composed_perm)) {
    return false;
  }

  mlir::Value rep =
      EmitReshapeIfNeeded(rewriter, tr2.getLoc(), tr1.getOperand(), tr2_type);
  rewriter.replaceOp(tr2, rep);
  return true;
}

class CancelInverseViewTransposesPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::TransposeOp> {
 public:
  explicit CancelInverseViewTransposesPattern(mlir::MLIRContext* context)
      : mlir::OpRewritePattern<mlir::stablehlo::TransposeOp>(context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::TransposeOp tr2,
      mlir::PatternRewriter& rewriter) const override {
    if (MatchAndCancelTransposeSlice(tr2, rewriter)) {
      return mlir::success();
    }
    if (MatchAndCancelTransposeReshape(tr2, rewriter)) {
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct CancelInverseViewTransposesPass
    : public mlir::PassWrapper<CancelInverseViewTransposesPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CancelInverseViewTransposesPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-cancel-inverse-view-transposes";
  }
  llvm::StringRef getDescription() const override {
    return "Cancels inverse transpose pairs separated by layout-invariant ops "
           "(slice and unit-dim-preserving reshape).";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<CancelInverseViewTransposesPattern>(context);

    mlir::GreedyRewriteConfig config;
    config.setMaxIterations(10);
    if (mlir::failed(
            mlir::applyPatternsGreedily(func, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCancelInverseViewTransposesPass() {
  return std::make_unique<CancelInverseViewTransposesPass>();
}

}  // namespace torch_tpu
