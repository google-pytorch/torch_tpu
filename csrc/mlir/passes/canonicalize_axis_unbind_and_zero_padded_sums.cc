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

#include "csrc/mlir/passes/canonicalize_axis_unbind_and_zero_padded_sums.h"

#include <cstddef>
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
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

// GENERIC: hardware TPU TensorCore vector register lane width (128)
constexpr int64_t kTpuLaneWidth = 128;
constexpr int kMaxRecursionDepth = 8;

// Returns true if `v` is provably an all-zero tensor.
[[nodiscard]] bool IsZeroTensor(mlir::Value v, int recursion_depth = 0) {
  if (!v || recursion_depth > kMaxRecursionDepth) {
    return false;
  }
  if (auto const_op = v.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
    auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    if (!attr || !attr.isSplat()) {
      return false;
    }
    const mlir::Type elem_type = attr.getElementType();
    if (mlir::isa<mlir::FloatType>(elem_type)) {
      const llvm::APFloat val = attr.getSplatValue<llvm::APFloat>();
      return val.isZero() && !val.isNegative();
    }
    if (mlir::isa<mlir::IntegerType>(elem_type)) {
      const llvm::APInt val = attr.getSplatValue<llvm::APInt>();
      return val.isZero();
    }
    return false;
  }
  if (auto bcast_op = v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    return IsZeroTensor(bcast_op.getOperand(), recursion_depth + 1);
  }
  if (auto reshape_op = v.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    return IsZeroTensor(reshape_op.getOperand(), recursion_depth + 1);
  }
  if (auto slice_op = v.getDefiningOp<mlir::stablehlo::SliceOp>()) {
    return IsZeroTensor(slice_op.getOperand(), recursion_depth + 1);
  }
  if (auto transpose_op = v.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    return IsZeroTensor(transpose_op.getOperand(), recursion_depth + 1);
  }
  if (auto pad_op = v.getDefiningOp<mlir::stablehlo::PadOp>()) {
    return IsZeroTensor(pad_op.getOperand(), recursion_depth + 1) &&
           IsZeroTensor(pad_op.getPaddingValue(), recursion_depth + 1);
  }
  return false;
}

// Creates a splat zero constant tensor of `type`.
mlir::Value CreateZeroTensor(mlir::PatternRewriter& rewriter,
                             mlir::Location loc, mlir::RankedTensorType type) {
  auto zero_attr =
      mlir::dyn_cast_or_null<mlir::ElementsAttr>(rewriter.getZeroAttr(type));
  return rewriter.create<mlir::stablehlo::ConstantOp>(loc, type, zero_attr);
}

// Extracts a constant integer value from a 0D scalar constant tensor.
[[nodiscard]] std::optional<int64_t> GetConstantScalarIndex(mlir::Value v) {
  auto const_op = v.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr || !attr.isSplat()) {
    return std::nullopt;
  }
  if (!mlir::isa<mlir::IntegerType>(attr.getElementType())) {
    return std::nullopt;
  }
  return attr.getSplatValue<llvm::APInt>().getSExtValue();
}

// Returns true if `perm` moves `perm[dst_axis]` to `dst_axis` while keeping
// all other axes in strictly increasing relative order.
[[nodiscard]] bool IsSingleAxisOrderPreservingPerm(llvm::ArrayRef<int64_t> perm,
                                                   int64_t dst_axis) {
  const int64_t rank = perm.size();
  if (dst_axis < 0 || dst_axis >= rank) {
    return false;
  }
  int64_t prev = -1;
  for (int64_t a = 0; a < rank; ++a) {
    if (a == dst_axis) {
      continue;
    }
    if (perm[a] <= prev) {
      return false;
    }
    prev = perm[a];
  }
  return true;
}

// Unwraps a chain of ReshapeOps to the underlying non-reshape value.
[[nodiscard]] mlir::Value UnwrapReshapes(mlir::Value v) {
  while (auto reshape_op = v.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    v = reshape_op.getOperand();
  }
  return v;
}

// Creates a ReshapeOp to `target_type`, or returns `raw` directly when its
// type already equals `target_type`.
mlir::Value CreateOrFoldReshape(mlir::PatternRewriter& rewriter,
                                mlir::Location loc,
                                mlir::RankedTensorType target_type,
                                mlir::Value v) {
  mlir::Value raw = UnwrapReshapes(v);
  if (raw.getType() == target_type) {
    return raw;
  }
  return rewriter.create<mlir::stablehlo::ReshapeOp>(loc, target_type, raw);
}

// Returns true if `v` (of rank `rank`) can be flattened to `flat_type`
// (merging the last two dimensions `rank - 2` and `rank - 1`) without
// introducing a new ReshapeOp on a non-reshape value.
[[nodiscard]] bool CanFlattenDirectlyWithoutNewReshape(
    mlir::Value v, mlir::RankedTensorType flat_type, int64_t rank,
    int recursion_depth = 0) {
  if (!v || recursion_depth > kMaxRecursionDepth) {
    return false;
  }
  if (auto reshape_op = v.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    mlir::Value src = UnwrapReshapes(reshape_op.getOperand());
    return src.getType() == flat_type || reshape_op->hasOneUse();
  }
  if (auto const_op = v.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
    auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    return attr && attr.isSplat();
  }
  if (auto bcast_op = v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    // GENERIC: capability prefix/scalar broadcasts do not depend on the merged
    // trailing axes `rank - 2` and `rank - 1`.
    return llvm::all_of(bcast_op.getBroadcastDimensions(),
                        [rank](int64_t d) { return d < rank - 2; });
  }
  if (!v.hasOneUse()) {
    return false;
  }
  mlir::Operation* def_op = v.getDefiningOp();
  if (!def_op) {
    return false;
  }
  if (mlir::isa<mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
                mlir::stablehlo::MulOp, mlir::stablehlo::DivOp,
                mlir::stablehlo::MaxOp, mlir::stablehlo::MinOp>(def_op)) {
    mlir::Value lhs = def_op->getOperand(0);
    mlir::Value rhs = def_op->getOperand(1);
    return CanFlattenDirectlyWithoutNewReshape(lhs, flat_type, rank,
                                               recursion_depth + 1) ||
           CanFlattenDirectlyWithoutNewReshape(rhs, flat_type, rank,
                                               recursion_depth + 1);
  }
  return false;
}

// Flattens the last two dimensions (`rank - 2` and `rank - 1`) of `v` into
// `flat_type`, commuting across single-use binary elementwise ops when at
// least one branch flattens directly without a new ReshapeOp.
mlir::Value FlattenLastTwoDims(mlir::Value v, mlir::RankedTensorType flat_type,
                               int64_t rank, mlir::PatternRewriter& rewriter,
                               int recursion_depth = 0) {
  const mlir::Location loc = v.getLoc();
  if (recursion_depth <= kMaxRecursionDepth) {
    if (auto reshape_op = v.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      return CreateOrFoldReshape(rewriter, loc, flat_type,
                                 reshape_op.getOperand());
    }
    if (auto const_op = v.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
      auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
      if (attr && attr.isSplat()) {
        auto flat_attr = mlir::DenseElementsAttr::get(
            flat_type, attr.getSplatValue<mlir::Attribute>());
        return rewriter.create<mlir::stablehlo::ConstantOp>(loc, flat_type,
                                                            flat_attr);
      }
    }
    if (auto bcast_op = v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      // GENERIC: capability prefix/scalar broadcasts map identically into the
      // flattened tensor type.
      if (llvm::all_of(bcast_op.getBroadcastDimensions(),
                       [rank](int64_t d) { return d < rank - 2; })) {
        return rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, flat_type, bcast_op.getOperand(),
            bcast_op.getBroadcastDimensionsAttr());
      }
    }
    if (recursion_depth == 0 || v.hasOneUse()) {
      mlir::Operation* def_op = v.getDefiningOp();
      if (def_op &&
          mlir::isa<mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
                    mlir::stablehlo::MulOp, mlir::stablehlo::DivOp,
                    mlir::stablehlo::MaxOp, mlir::stablehlo::MinOp>(def_op)) {
        mlir::Value lhs = def_op->getOperand(0);
        mlir::Value rhs = def_op->getOperand(1);
        if (CanFlattenDirectlyWithoutNewReshape(lhs, flat_type, rank,
                                                recursion_depth + 1) ||
            CanFlattenDirectlyWithoutNewReshape(rhs, flat_type, rank,
                                                recursion_depth + 1)) {
          mlir::Value lhs_flat = FlattenLastTwoDims(
              lhs, flat_type, rank, rewriter, recursion_depth + 1);
          mlir::Value rhs_flat = FlattenLastTwoDims(
              rhs, flat_type, rank, rewriter, recursion_depth + 1);
          if (mlir::isa<mlir::stablehlo::AddOp>(def_op)) {
            return rewriter.create<mlir::stablehlo::AddOp>(loc, flat_type,
                                                           lhs_flat, rhs_flat);
          }
          if (mlir::isa<mlir::stablehlo::SubtractOp>(def_op)) {
            return rewriter.create<mlir::stablehlo::SubtractOp>(
                loc, flat_type, lhs_flat, rhs_flat);
          }
          if (mlir::isa<mlir::stablehlo::MulOp>(def_op)) {
            return rewriter.create<mlir::stablehlo::MulOp>(loc, flat_type,
                                                           lhs_flat, rhs_flat);
          }
          if (mlir::isa<mlir::stablehlo::DivOp>(def_op)) {
            return rewriter.create<mlir::stablehlo::DivOp>(loc, flat_type,
                                                           lhs_flat, rhs_flat);
          }
          if (mlir::isa<mlir::stablehlo::MaxOp>(def_op)) {
            return rewriter.create<mlir::stablehlo::MaxOp>(loc, flat_type,
                                                           lhs_flat, rhs_flat);
          }
          return rewriter.create<mlir::stablehlo::MinOp>(loc, flat_type,
                                                         lhs_flat, rhs_flat);
        }
      }
    }
  }
  return CreateOrFoldReshape(rewriter, loc, flat_type, v);
}

// Pattern 1: Commutes unit-width SliceOps through a single-axis-moving
// TransposeOp so the TransposeOp is eliminated:
//   Slice_{dst_axis=[i:i+1]}(Transpose(X, perm))
//     -> Reshape(Slice_{src_axis=[i:i+1]}(X))
struct SliceOfSingleAxisTransposePattern
    : public mlir::OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern<mlir::stablehlo::TransposeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::TransposeOp transpose_op,
      mlir::PatternRewriter& rewriter) const override {
    if (transpose_op->use_empty()) {
      return mlir::failure();
    }
    auto x_type = mlir::dyn_cast<mlir::RankedTensorType>(
        transpose_op.getOperand().getType());
    auto t_type =
        mlir::dyn_cast<mlir::RankedTensorType>(transpose_op.getType());
    if (!x_type || !x_type.hasStaticShape() || !t_type ||
        !t_type.hasStaticShape()) {
      return mlir::failure();
    }
    const int64_t rank = t_type.getRank();
    // GENERIC: capability transpose requires at least rank 2.
    if (rank < 2) {
      return mlir::failure();
    }

    llvm::ArrayRef<int64_t> perm = transpose_op.getPermutation();
    llvm::SmallVector<mlir::stablehlo::SliceOp> slice_users;
    for (mlir::Operation* user : transpose_op->getUsers()) {
      auto slice_op = mlir::dyn_cast<mlir::stablehlo::SliceOp>(user);
      if (!slice_op) {
        return mlir::failure();
      }
      if (!llvm::all_of(slice_op.getStrides(),
                        [](int64_t s) { return s == 1; })) {
        return mlir::failure();
      }
      slice_users.push_back(slice_op);
    }

    int64_t matched_dst_axis = -1;
    for (int64_t dst_axis = 0; dst_axis < rank; ++dst_axis) {
      if (perm[dst_axis] == dst_axis ||
          !IsSingleAxisOrderPreservingPerm(perm, dst_axis)) {
        continue;
      }
      const bool all_unit_along_dst =
          llvm::all_of(slice_users, [dst_axis](mlir::stablehlo::SliceOp s_op) {
            return s_op.getLimitIndices()[dst_axis] -
                       s_op.getStartIndices()[dst_axis] ==
                   1;
          });
      if (all_unit_along_dst) {
        matched_dst_axis = dst_axis;
        break;
      }
    }
    if (matched_dst_axis < 0) {
      return mlir::failure();
    }

    mlir::Value x = transpose_op.getOperand();
    const mlir::Type elem_type = t_type.getElementType();
    for (mlir::stablehlo::SliceOp s_op : slice_users) {
      llvm::ArrayRef<int64_t> starts = s_op.getStartIndices();
      llvm::ArrayRef<int64_t> limits = s_op.getLimitIndices();
      llvm::SmallVector<int64_t> x_starts(rank, 0);
      llvm::SmallVector<int64_t> x_limits(rank, 0);
      llvm::SmallVector<int64_t> x_strides(rank, 1);
      llvm::SmallVector<int64_t> x_slice_shape(rank, 0);
      for (int64_t a = 0; a < rank; ++a) {
        const int64_t src_a = perm[a];
        x_starts[src_a] = starts[a];
        x_limits[src_a] = limits[a];
        x_slice_shape[src_a] = limits[a] - starts[a];
      }
      rewriter.setInsertionPoint(s_op);
      auto x_slice_type = mlir::RankedTensorType::get(x_slice_shape, elem_type);
      mlir::Value x_slice = rewriter.create<mlir::stablehlo::SliceOp>(
          s_op.getLoc(), x_slice_type, x,
          rewriter.getDenseI64ArrayAttr(x_starts),
          rewriter.getDenseI64ArrayAttr(x_limits),
          rewriter.getDenseI64ArrayAttr(x_strides));
      mlir::Value reshaped = CreateOrFoldReshape(
          rewriter, s_op.getLoc(),
          mlir::cast<mlir::RankedTensorType>(s_op.getType()), x_slice);
      rewriter.replaceOp(s_op, reshaped);
    }
    if (transpose_op->use_empty()) {
      rewriter.eraseOp(transpose_op);
    }
    return mlir::success();
  }
};

// Pattern 2: Canonicalizes partitioned unit-slice squeezes (`select` /
// `unbind` along `rank - 2`) on `[D0..., K, D]` into lane-aligned minor-axis
// slices on `[D0..., K * D]`:
//   Reshape(Slice(Y, [0:D0..., i:i+1, c0:c1]), R_out)
//     -> Reshape(Slice(Flatten(Y), [0:D0..., i*D + c0 : i*D + c1]), R_out)
struct CanonicalizeAxisUnbindPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern<mlir::stablehlo::SliceOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::SliceOp slice_op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Value y = slice_op.getOperand();
    auto y_type = mlir::dyn_cast<mlir::RankedTensorType>(y.getType());
    if (!y_type || !y_type.hasStaticShape()) {
      return mlir::failure();
    }
    const int64_t rank = y_type.getRank();
    // GENERIC: capability rank >= 3 has at least one leading dimension plus
    // second-minor and minor axes.
    if (rank < 3) {
      return mlir::failure();
    }
    const int64_t num_parts = y_type.getDimSize(rank - 2);
    const int64_t part_width = y_type.getDimSize(rank - 1);
    // GENERIC: semantic at least 2 partitions along rank - 2, and hardware
    // lane-tile alignment (128) along the minor dimension.
    if (num_parts < 2 || part_width <= 0 || part_width % kTpuLaneWidth != 0) {
      return mlir::failure();
    }

    llvm::ArrayRef<int64_t> leading_prefix =
        y_type.getShape().take_front(rank - 2);
    llvm::SmallVector<mlir::stablehlo::SliceOp> slice_users;
    for (mlir::Operation* user : y.getUsers()) {
      auto s_op = mlir::dyn_cast<mlir::stablehlo::SliceOp>(user);
      if (!s_op || s_op->use_empty()) {
        return mlir::failure();
      }
      if (!llvm::all_of(s_op.getStrides(), [](int64_t s) { return s == 1; })) {
        return mlir::failure();
      }
      llvm::ArrayRef<int64_t> starts = s_op.getStartIndices();
      llvm::ArrayRef<int64_t> limits = s_op.getLimitIndices();
      for (int64_t a = 0; a < rank - 2; ++a) {
        if (starts[a] != 0 || limits[a] != y_type.getDimSize(a)) {
          return mlir::failure();
        }
      }
      if (limits[rank - 2] - starts[rank - 2] != 1) {
        return mlir::failure();
      }
      const int64_t c0 = starts[rank - 1];
      const int64_t c1 = limits[rank - 1];
      // GENERIC: hardware minor-axis slice offset and width must align to the
      // 128-lane vector register width.
      if (c1 <= c0 || c0 % kTpuLaneWidth != 0 ||
          (c1 - c0) % kTpuLaneWidth != 0) {
        return mlir::failure();
      }
      for (mlir::Operation* s_user : s_op->getUsers()) {
        auto r_op = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(s_user);
        if (!r_op) {
          return mlir::failure();
        }
        auto r_type = mlir::dyn_cast<mlir::RankedTensorType>(r_op.getType());
        // GENERIC: semantic squeezed output preserves leading dimensions.
        if (!r_type || !r_type.hasStaticShape() || r_type == s_op.getType() ||
            r_type.getRank() < rank - 1 ||
            r_type.getShape().take_front(rank - 2) != leading_prefix) {
          return mlir::failure();
        }
      }
      slice_users.push_back(s_op);
    }

    // GENERIC: semantic at least two unit slices from the partitioned producer.
    if (slice_users.size() < 2) {
      return mlir::failure();
    }

    llvm::SmallVector<int64_t> flat_shape(leading_prefix.begin(),
                                          leading_prefix.end());
    flat_shape.push_back(num_parts * part_width);
    const mlir::Type elem_type = y_type.getElementType();
    auto flat_type = mlir::RankedTensorType::get(flat_shape, elem_type);

    rewriter.setInsertionPointAfterValue(y);
    mlir::Value y_flat = FlattenLastTwoDims(y, flat_type, rank, rewriter);

    for (mlir::stablehlo::SliceOp s_op : slice_users) {
      llvm::ArrayRef<int64_t> starts = s_op.getStartIndices();
      llvm::ArrayRef<int64_t> limits = s_op.getLimitIndices();
      const int64_t part_idx = starts[rank - 2];
      const int64_t c0 = starts[rank - 1];
      const int64_t c1 = limits[rank - 1];

      llvm::SmallVector<int64_t> flat_starts(rank - 2, 0);
      flat_starts.push_back(part_idx * part_width + c0);

      llvm::SmallVector<int64_t> flat_limits(leading_prefix.begin(),
                                             leading_prefix.end());
      flat_limits.push_back(part_idx * part_width + c1);

      llvm::SmallVector<int64_t> flat_strides(rank - 1, 1);

      llvm::SmallVector<int64_t> slice_out_shape(leading_prefix.begin(),
                                                 leading_prefix.end());
      slice_out_shape.push_back(c1 - c0);
      auto slice_out_type =
          mlir::RankedTensorType::get(slice_out_shape, elem_type);

      rewriter.setInsertionPoint(s_op);
      mlir::Value flat_slice = rewriter.create<mlir::stablehlo::SliceOp>(
          s_op.getLoc(), slice_out_type, y_flat,
          rewriter.getDenseI64ArrayAttr(flat_starts),
          rewriter.getDenseI64ArrayAttr(flat_limits),
          rewriter.getDenseI64ArrayAttr(flat_strides));

      llvm::SmallVector<mlir::stablehlo::ReshapeOp> reshape_users;
      for (mlir::Operation* s_user : s_op->getUsers()) {
        reshape_users.push_back(mlir::cast<mlir::stablehlo::ReshapeOp>(s_user));
      }
      for (mlir::stablehlo::ReshapeOp r_op : reshape_users) {
        rewriter.setInsertionPoint(r_op);
        auto r_type = mlir::cast<mlir::RankedTensorType>(r_op.getType());
        mlir::Value repl =
            CreateOrFoldReshape(rewriter, r_op.getLoc(), r_type, flat_slice);
        rewriter.replaceOp(r_op, repl);
      }
      if (s_op->use_empty()) {
        rewriter.eraseOp(s_op);
      }
    }
    return mlir::success();
  }
};

// Represents a contiguous slice interval [start, end) on a single axis.
struct IntervalPiece {
  int64_t start = 0;
  int64_t end = 0;
  mlir::Value value;
};

// Represents a tensor embedding with zero padding along a single axis.
struct ZeroPaddedEmbedding {
  mlir::RankedTensorType inner_type;
  int64_t axis = -1;
  llvm::SmallVector<IntervalPiece, 2> pieces;
};

// Matches a single-axis zero-padded slice embedding constructed via PadOp,
// DynamicUpdateSliceOp, or ConcatenateOp, optionally wrapped in a ReshapeOp.
[[nodiscard]] std::optional<ZeroPaddedEmbedding> MatchZeroPaddedEmbedding(
    mlir::Value v) {
  mlir::Value inner_val = v;
  if (!v.getDefiningOp<mlir::stablehlo::PadOp>() &&
      !v.getDefiningOp<mlir::stablehlo::DynamicUpdateSliceOp>() &&
      !v.getDefiningOp<mlir::stablehlo::ConcatenateOp>()) {
    if (auto reshape_op = v.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      inner_val = reshape_op.getOperand();
    }
  }
  auto inner_type = mlir::dyn_cast<mlir::RankedTensorType>(inner_val.getType());
  if (!inner_type || !inner_type.hasStaticShape() || inner_type.getRank() < 1) {
    return std::nullopt;
  }
  const int64_t rank = inner_type.getRank();

  if (auto pad_op = inner_val.getDefiningOp<mlir::stablehlo::PadOp>()) {
    if (!IsZeroTensor(pad_op.getPaddingValue()) ||
        !llvm::all_of(pad_op.getInteriorPadding(),
                      [](int64_t p) { return p == 0; })) {
      return std::nullopt;
    }
    auto opnd_type =
        mlir::dyn_cast<mlir::RankedTensorType>(pad_op.getOperand().getType());
    if (!opnd_type || !opnd_type.hasStaticShape()) {
      return std::nullopt;
    }
    llvm::ArrayRef<int64_t> low = pad_op.getEdgePaddingLow();
    llvm::ArrayRef<int64_t> high = pad_op.getEdgePaddingHigh();
    int64_t pad_axis = -1;
    for (int64_t a = 0; a < rank; ++a) {
      if (low[a] < 0 || high[a] < 0) {
        return std::nullopt;
      }
      if (low[a] > 0 || high[a] > 0) {
        if (pad_axis != -1) {
          return std::nullopt;
        }
        pad_axis = a;
      }
    }
    if (pad_axis < 0) {
      return std::nullopt;
    }
    const int64_t len = opnd_type.getDimSize(pad_axis);
    if (len <= 0) {
      return std::nullopt;
    }
    ZeroPaddedEmbedding emb;
    emb.inner_type = inner_type;
    emb.axis = pad_axis;
    emb.pieces.push_back(
        IntervalPiece{low[pad_axis], low[pad_axis] + len, pad_op.getOperand()});
    return emb;
  }

  if (auto dus_op =
          inner_val.getDefiningOp<mlir::stablehlo::DynamicUpdateSliceOp>()) {
    if (!IsZeroTensor(dus_op.getOperand())) {
      return std::nullopt;
    }
    auto update_type =
        mlir::dyn_cast<mlir::RankedTensorType>(dus_op.getUpdate().getType());
    if (!update_type || !update_type.hasStaticShape() ||
        static_cast<int64_t>(dus_op.getStartIndices().size()) != rank) {
      return std::nullopt;
    }
    int64_t dus_axis = -1;
    int64_t start_idx = 0;
    for (int64_t a = 0; a < rank; ++a) {
      std::optional<int64_t> idx =
          GetConstantScalarIndex(dus_op.getStartIndices()[a]);
      if (!idx.has_value() || *idx < 0 ||
          *idx + update_type.getDimSize(a) > inner_type.getDimSize(a)) {
        return std::nullopt;
      }
      if (*idx != 0 || update_type.getDimSize(a) != inner_type.getDimSize(a)) {
        if (dus_axis != -1) {
          return std::nullopt;
        }
        dus_axis = a;
        start_idx = *idx;
      }
    }
    if (dus_axis < 0) {
      return std::nullopt;
    }
    const int64_t len = update_type.getDimSize(dus_axis);
    if (len <= 0 || len >= inner_type.getDimSize(dus_axis)) {
      return std::nullopt;
    }
    ZeroPaddedEmbedding emb;
    emb.inner_type = inner_type;
    emb.axis = dus_axis;
    emb.pieces.push_back(
        IntervalPiece{start_idx, start_idx + len, dus_op.getUpdate()});
    return emb;
  }

  if (auto concat_op =
          inner_val.getDefiningOp<mlir::stablehlo::ConcatenateOp>()) {
    const int64_t axis = concat_op.getDimension();
    if (axis < 0 || axis >= rank) {
      return std::nullopt;
    }
    ZeroPaddedEmbedding emb;
    emb.inner_type = inner_type;
    emb.axis = axis;
    int64_t pos = 0;
    int64_t zero_count = 0;
    int64_t nonzero_count = 0;
    for (mlir::Value inp : concat_op.getInputs()) {
      auto inp_type = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
      if (!inp_type || !inp_type.hasStaticShape()) {
        return std::nullopt;
      }
      const int64_t len = inp_type.getDimSize(axis);
      if (len <= 0) {
        continue;
      }
      if (IsZeroTensor(inp)) {
        ++zero_count;
      } else {
        emb.pieces.push_back(IntervalPiece{pos, pos + len, inp});
        ++nonzero_count;
      }
      pos += len;
    }
    if (zero_count < 1 || nonzero_count < 1) {
      return std::nullopt;
    }
    return emb;
  }

  return std::nullopt;
}

// Flattens a single-use associative AddOp chain into its leaf operands.
void CollectAddLeaves(mlir::Value v,
                      llvm::SmallVectorImpl<mlir::Value>& leaves) {
  if (auto add_op = v.getDefiningOp<mlir::stablehlo::AddOp>()) {
    if (add_op->hasOneUse()) {
      CollectAddLeaves(add_op.getLhs(), leaves);
      CollectAddLeaves(add_op.getRhs(), leaves);
      return;
    }
  }
  leaves.push_back(v);
}

// Pattern 3: Folds sums of disjoint zero-padded slice embeddings along a
// shared axis into a single ConcatenateOp (or minor-axis ConcatenateOp when
// concatenating unit slices along `rank - 2` with a lane-aligned minor axis).
struct CanonicalizeZeroPaddedSumPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::AddOp> {
  using OpRewritePattern<mlir::stablehlo::AddOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::AddOp root_add,
      mlir::PatternRewriter& rewriter) const override {
    if (root_add->hasOneUse() &&
        mlir::isa<mlir::stablehlo::AddOp>(*root_add->user_begin())) {
      return mlir::failure();
    }
    auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(root_add.getType());
    if (!out_type || !out_type.hasStaticShape() || out_type.getRank() < 1) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value, 8> leaves;
    CollectAddLeaves(root_add.getLhs(), leaves);
    CollectAddLeaves(root_add.getRhs(), leaves);
    // GENERIC: semantic at least two additive terms to combine.
    if (leaves.size() < 2) {
      return mlir::failure();
    }

    struct EmbeddingGroup {
      mlir::RankedTensorType inner_type;
      int64_t axis = -1;
      llvm::SmallVector<size_t, 4> leaf_indices;
      llvm::SmallVector<IntervalPiece, 4> pieces;
    };
    llvm::SmallVector<EmbeddingGroup, 2> groups;

    for (size_t i = 0; i < leaves.size(); ++i) {
      std::optional<ZeroPaddedEmbedding> emb =
          MatchZeroPaddedEmbedding(leaves[i]);
      if (!emb.has_value()) {
        continue;
      }
      EmbeddingGroup* found = nullptr;
      for (EmbeddingGroup& g : groups) {
        if (g.inner_type == emb->inner_type && g.axis == emb->axis) {
          found = &g;
          break;
        }
      }
      if (!found) {
        groups.push_back(EmbeddingGroup{emb->inner_type, emb->axis, {}, {}});
        found = &groups.back();
      }
      found->leaf_indices.push_back(i);
      for (const IntervalPiece& p : emb->pieces) {
        found->pieces.push_back(p);
      }
    }

    const mlir::Location loc = root_add.getLoc();
    llvm::SmallVector<bool, 8> consumed(leaves.size(), false);
    llvm::SmallVector<mlir::Value, 4> new_terms;
    bool any_folded = false;

    for (EmbeddingGroup& g : groups) {
      // GENERIC: semantic at least two zero-padded embeddings in the group.
      if (g.leaf_indices.size() < 2) {
        continue;
      }
      llvm::stable_sort(g.pieces,
                        [](const IntervalPiece& a, const IntervalPiece& b) {
                          if (a.start != b.start) {
                            return a.start < b.start;
                          }
                          return a.end < b.end;
                        });

      llvm::SmallVector<IntervalPiece, 4> merged;
      bool has_partial_overlap = false;
      for (const IntervalPiece& p : g.pieces) {
        if (!merged.empty() && merged.back().start == p.start &&
            merged.back().end == p.end) {
          merged.back().value = rewriter.create<mlir::stablehlo::AddOp>(
              loc, merged.back().value.getType(), merged.back().value, p.value);
        } else if (!merged.empty() && merged.back().end > p.start) {
          has_partial_overlap = true;
          break;
        } else {
          merged.push_back(p);
        }
      }
      if (has_partial_overlap || merged.empty()) {
        continue;
      }

      const mlir::RankedTensorType inner_type = g.inner_type;
      const int64_t inner_rank = inner_type.getRank();
      const int64_t axis = g.axis;
      const int64_t total_dim = inner_type.getDimSize(axis);
      const mlir::Type elem_type = inner_type.getElementType();

      mlir::Value combined_inner;
      // Check if this is a full unit-slice partition along `inner_rank - 2`
      // with a lane-aligned minor axis `inner_rank - 1`.
      // GENERIC: capability rank >= 3 has leading dims plus second-minor and
      // minor axes; hardware lane alignment is kTpuLaneWidth (128).
      bool can_flatten_minor =
          inner_rank >= 3 && axis == inner_rank - 2 && total_dim >= 2 &&
          inner_type.getDimSize(inner_rank - 1) > 0 &&
          inner_type.getDimSize(inner_rank - 1) % kTpuLaneWidth == 0 &&
          static_cast<int64_t>(merged.size()) == total_dim;
      if (can_flatten_minor) {
        for (int64_t i = 0; i < total_dim; ++i) {
          if (merged[i].start != i || merged[i].end != i + 1) {
            can_flatten_minor = false;
            break;
          }
        }
      }

      if (can_flatten_minor) {
        const int64_t part_width = inner_type.getDimSize(inner_rank - 1);
        llvm::ArrayRef<int64_t> prefix =
            inner_type.getShape().take_front(inner_rank - 2);
        llvm::SmallVector<int64_t> part_flat_shape(prefix.begin(),
                                                   prefix.end());
        part_flat_shape.push_back(part_width);
        auto part_flat_type =
            mlir::RankedTensorType::get(part_flat_shape, elem_type);

        llvm::SmallVector<mlir::Value, 8> flat_parts;
        flat_parts.reserve(total_dim);
        for (const IntervalPiece& p : merged) {
          flat_parts.push_back(
              CreateOrFoldReshape(rewriter, loc, part_flat_type, p.value));
        }

        llvm::SmallVector<int64_t> full_flat_shape(prefix.begin(),
                                                   prefix.end());
        full_flat_shape.push_back(total_dim * part_width);
        auto full_flat_type =
            mlir::RankedTensorType::get(full_flat_shape, elem_type);
        mlir::Value flat_concat =
            rewriter.create<mlir::stablehlo::ConcatenateOp>(
                loc, full_flat_type, flat_parts, inner_rank - 2);
        combined_inner =
            CreateOrFoldReshape(rewriter, loc, inner_type, flat_concat);
      } else {
        llvm::SmallVector<mlir::Value, 8> concat_inputs;
        int64_t cur = 0;
        for (const IntervalPiece& p : merged) {
          if (p.start > cur) {
            llvm::SmallVector<int64_t> gap_shape(inner_type.getShape().begin(),
                                                 inner_type.getShape().end());
            gap_shape[axis] = p.start - cur;
            auto gap_type = mlir::RankedTensorType::get(gap_shape, elem_type);
            concat_inputs.push_back(CreateZeroTensor(rewriter, loc, gap_type));
          }
          concat_inputs.push_back(p.value);
          cur = p.end;
        }
        if (cur < total_dim) {
          llvm::SmallVector<int64_t> gap_shape(inner_type.getShape().begin(),
                                               inner_type.getShape().end());
          gap_shape[axis] = total_dim - cur;
          auto gap_type = mlir::RankedTensorType::get(gap_shape, elem_type);
          concat_inputs.push_back(CreateZeroTensor(rewriter, loc, gap_type));
        }
        if (concat_inputs.size() == 1) {
          combined_inner = concat_inputs[0];
        } else {
          combined_inner = rewriter.create<mlir::stablehlo::ConcatenateOp>(
              loc, inner_type, concat_inputs, axis);
        }
      }

      mlir::Value combined_out =
          CreateOrFoldReshape(rewriter, loc, out_type, combined_inner);
      new_terms.push_back(combined_out);
      for (size_t idx : g.leaf_indices) {
        consumed[idx] = true;
      }
      any_folded = true;
    }

    if (!any_folded) {
      return mlir::failure();
    }

    for (size_t i = 0; i < leaves.size(); ++i) {
      if (!consumed[i]) {
        new_terms.push_back(leaves[i]);
      }
    }

    mlir::Value acc = new_terms[0];
    for (size_t i = 1; i < new_terms.size(); ++i) {
      acc = rewriter.create<mlir::stablehlo::AddOp>(loc, out_type, acc,
                                                    new_terms[i]);
    }
    rewriter.replaceOp(root_add, acc);
    return mlir::success();
  }
};

// Pattern 4: Commutes a single-axis-moving TransposeOp through a
// ConcatenateOp whose non-zero inputs have unit length along the concatenated
// axis, eliminating the TransposeOp:
//   Transpose(Concatenate_{src_axis}(U_0, ..., U_{K-1}), perm)
//     -> Concatenate_{dst_axis}(Reshape(U_0), ..., Reshape(U_{K-1}))
struct TransposeOfUnitConcatenatePattern
    : public mlir::OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern<mlir::stablehlo::TransposeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::TransposeOp transpose_op,
      mlir::PatternRewriter& rewriter) const override {
    auto concat_op = transpose_op.getOperand()
                         .getDefiningOp<mlir::stablehlo::ConcatenateOp>();
    if (!concat_op || !concat_op->hasOneUse()) {
      return mlir::failure();
    }
    auto c_type = mlir::dyn_cast<mlir::RankedTensorType>(concat_op.getType());
    auto t_type =
        mlir::dyn_cast<mlir::RankedTensorType>(transpose_op.getType());
    if (!c_type || !c_type.hasStaticShape() || !t_type ||
        !t_type.hasStaticShape()) {
      return mlir::failure();
    }
    const int64_t rank = t_type.getRank();
    // GENERIC: capability transpose requires at least rank 2.
    if (rank < 2) {
      return mlir::failure();
    }

    const int64_t src_axis = concat_op.getDimension();
    llvm::ArrayRef<int64_t> perm = transpose_op.getPermutation();
    int64_t dst_axis = -1;
    for (int64_t a = 0; a < rank; ++a) {
      if (perm[a] == src_axis) {
        dst_axis = a;
        break;
      }
    }
    if (dst_axis < 0 || dst_axis == src_axis ||
        !IsSingleAxisOrderPreservingPerm(perm, dst_axis)) {
      return mlir::failure();
    }

    bool all_unit = true;
    for (mlir::Value inp : concat_op.getInputs()) {
      auto inp_type = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
      if (!inp_type || !inp_type.hasStaticShape()) {
        return mlir::failure();
      }
      if (inp_type.getDimSize(src_axis) != 1) {
        all_unit = false;
        if (!IsZeroTensor(inp)) {
          return mlir::failure();
        }
      }
    }

    const mlir::Location loc = transpose_op.getLoc();
    const mlir::Type elem_type = t_type.getElementType();
    const int64_t num_parts = t_type.getDimSize(dst_axis);
    const int64_t part_width = t_type.getDimSize(rank - 1);

    // GENERIC: capability rank >= 3 has leading dims plus second-minor and
    // minor axes; hardware lane alignment is kTpuLaneWidth (128).
    if (all_unit && rank >= 3 && dst_axis == rank - 2 && num_parts >= 2 &&
        part_width > 0 && part_width % kTpuLaneWidth == 0) {
      llvm::ArrayRef<int64_t> prefix = t_type.getShape().take_front(rank - 2);
      llvm::SmallVector<int64_t> part_flat_shape(prefix.begin(), prefix.end());
      part_flat_shape.push_back(part_width);
      auto part_flat_type =
          mlir::RankedTensorType::get(part_flat_shape, elem_type);

      llvm::SmallVector<mlir::Value, 8> flat_parts;
      flat_parts.reserve(concat_op.getInputs().size());
      for (mlir::Value inp : concat_op.getInputs()) {
        flat_parts.push_back(
            CreateOrFoldReshape(rewriter, loc, part_flat_type, inp));
      }

      llvm::SmallVector<int64_t> full_flat_shape(prefix.begin(), prefix.end());
      full_flat_shape.push_back(num_parts * part_width);
      auto full_flat_type =
          mlir::RankedTensorType::get(full_flat_shape, elem_type);
      mlir::Value flat_concat = rewriter.create<mlir::stablehlo::ConcatenateOp>(
          loc, full_flat_type, flat_parts, rank - 2);
      mlir::Value repl =
          CreateOrFoldReshape(rewriter, loc, t_type, flat_concat);
      rewriter.replaceOp(transpose_op, repl);
      return mlir::success();
    }

    llvm::SmallVector<mlir::Value, 8> reshaped_inputs;
    reshaped_inputs.reserve(concat_op.getInputs().size());
    for (mlir::Value inp : concat_op.getInputs()) {
      auto inp_type = mlir::cast<mlir::RankedTensorType>(inp.getType());
      llvm::SmallVector<int64_t> new_inp_shape(t_type.getShape().begin(),
                                               t_type.getShape().end());
      new_inp_shape[dst_axis] = inp_type.getDimSize(src_axis);
      auto new_inp_type = mlir::RankedTensorType::get(new_inp_shape, elem_type);
      reshaped_inputs.push_back(
          CreateOrFoldReshape(rewriter, loc, new_inp_type, inp));
    }
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConcatenateOp>(
        transpose_op, t_type, reshaped_inputs, dst_axis);
    return mlir::success();
  }
};

// Returns true if `v` is produced by a ConcatenateOp (or elementwise op on a
// ConcatenateOp) along the minor dimension.
[[nodiscard]] bool IsMinorConcatTree(mlir::Value v, int recursion_depth = 0) {
  if (!v || recursion_depth > kMaxRecursionDepth) {
    return false;
  }
  auto v_type = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  // GENERIC: capability tensor requires at least rank 2 to have a distinct
  // minor axis.
  if (!v_type || !v_type.hasStaticShape() || v_type.getRank() < 2) {
    return false;
  }
  if (auto concat_op = v.getDefiningOp<mlir::stablehlo::ConcatenateOp>()) {
    return concat_op.getDimension() == v_type.getRank() - 1;
  }
  mlir::Operation* def_op = v.getDefiningOp();
  if (!def_op) {
    return false;
  }
  if (mlir::isa<mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
                mlir::stablehlo::MulOp, mlir::stablehlo::DivOp>(def_op)) {
    return IsMinorConcatTree(def_op->getOperand(0), recursion_depth + 1) ||
           IsMinorConcatTree(def_op->getOperand(1), recursion_depth + 1);
  }
  return false;
}

// Pattern 5: Sinks an expanding ReshapeOp `[D0..., K * D] -> [D0..., K, D]`
// (wrapping a minor-axis ConcatenateOp) below a single-use binary
// elementwise op whose other operand flattens directly without a new
// ReshapeOp, so downstream `[D0..., K * D]` consumers avoid round-tripping
// through `[D0..., K, D]`.
struct SinkExpandReshapeThroughElementwisePattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern<mlir::stablehlo::ReshapeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReshapeOp reshape_op,
      mlir::PatternRewriter& rewriter) const override {
    if (!reshape_op->hasOneUse()) {
      return mlir::failure();
    }
    mlir::Value flat_src = reshape_op.getOperand();
    auto flat_type = mlir::dyn_cast<mlir::RankedTensorType>(flat_src.getType());
    auto exp_type =
        mlir::dyn_cast<mlir::RankedTensorType>(reshape_op.getType());
    if (!flat_type || !flat_type.hasStaticShape() || !exp_type ||
        !exp_type.hasStaticShape()) {
      return mlir::failure();
    }
    const int64_t rank = exp_type.getRank();
    // GENERIC: capability rank >= 3 has leading dims plus second-minor and
    // minor axes.
    if (rank < 3 || flat_type.getRank() != rank - 1 ||
        exp_type.getShape().take_front(rank - 2) !=
            flat_type.getShape().take_front(rank - 2)) {
      return mlir::failure();
    }
    if (!IsMinorConcatTree(flat_src)) {
      return mlir::failure();
    }

    mlir::Operation* user_op = *reshape_op->user_begin();
    if (!mlir::isa<mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
                   mlir::stablehlo::MulOp, mlir::stablehlo::DivOp>(user_op)) {
      return mlir::failure();
    }
    const bool is_lhs = (user_op->getOperand(0) == reshape_op.getResult());
    mlir::Value other = user_op->getOperand(is_lhs ? 1 : 0);
    if (!CanFlattenDirectlyWithoutNewReshape(other, flat_type, rank)) {
      return mlir::failure();
    }

    rewriter.setInsertionPoint(user_op);
    const mlir::Location loc = user_op->getLoc();
    mlir::Value other_flat =
        FlattenLastTwoDims(other, flat_type, rank, rewriter);
    mlir::Value lhs_flat = is_lhs ? flat_src : other_flat;
    mlir::Value rhs_flat = is_lhs ? other_flat : flat_src;

    mlir::Value flat_elem;
    if (mlir::isa<mlir::stablehlo::AddOp>(user_op)) {
      flat_elem = rewriter.create<mlir::stablehlo::AddOp>(loc, flat_type,
                                                          lhs_flat, rhs_flat);
    } else if (mlir::isa<mlir::stablehlo::SubtractOp>(user_op)) {
      flat_elem = rewriter.create<mlir::stablehlo::SubtractOp>(
          loc, flat_type, lhs_flat, rhs_flat);
    } else if (mlir::isa<mlir::stablehlo::MulOp>(user_op)) {
      flat_elem = rewriter.create<mlir::stablehlo::MulOp>(loc, flat_type,
                                                          lhs_flat, rhs_flat);
    } else {
      flat_elem = rewriter.create<mlir::stablehlo::DivOp>(loc, flat_type,
                                                          lhs_flat, rhs_flat);
    }

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(user_op, exp_type,
                                                            flat_elem);
    if (reshape_op->use_empty()) {
      rewriter.eraseOp(reshape_op);
    }
    return mlir::success();
  }
};

// Pattern 6: Folds adjacent `ReshapeOp(ReshapeOp(x))` chains exposed by
// earlier patterns within the same greedy rewrite pass.
struct ComposeReshapePattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern<mlir::stablehlo::ReshapeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReshapeOp reshape_op,
      mlir::PatternRewriter& rewriter) const override {
    if (reshape_op.getOperand().getType() == reshape_op.getType()) {
      rewriter.replaceOp(reshape_op, reshape_op.getOperand());
      return mlir::success();
    }
    auto inner_reshape =
        reshape_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!inner_reshape) {
      return mlir::failure();
    }
    mlir::Value src = UnwrapReshapes(inner_reshape.getOperand());
    if (src.getType() == reshape_op.getType()) {
      rewriter.replaceOp(reshape_op, src);
    } else {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(
          reshape_op, reshape_op.getType(), src);
    }
    return mlir::success();
  }
};

struct CanonicalizeAxisUnbindAndZeroPaddedSumsPass
    : public mlir::PassWrapper<CanonicalizeAxisUnbindAndZeroPaddedSumsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CanonicalizeAxisUnbindAndZeroPaddedSumsPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-canonicalize-axis-unbind-and-zero-padded-sums";
  }
  llvm::StringRef getDescription() const override {
    return "Canonicalizes partitioned unit-slice squeezes into minor-axis "
           "slices and folds sums of disjoint zero-padded slice embeddings "
           "into concatenations.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<
        SliceOfSingleAxisTransposePattern, CanonicalizeAxisUnbindPattern,
        CanonicalizeZeroPaddedSumPattern, TransposeOfUnitConcatenatePattern,
        SinkExpandReshapeThroughElementwisePattern, ComposeReshapePattern>(
        context);

    mlir::GreedyRewriteConfig config;
    config.setMaxIterations(10);
    if (mlir::failed(
            mlir::applyPatternsGreedily(func, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createCanonicalizeAxisUnbindAndZeroPaddedSumsPass() {
  return std::make_unique<CanonicalizeAxisUnbindAndZeroPaddedSumsPass>();
}

}  // namespace torch_tpu
