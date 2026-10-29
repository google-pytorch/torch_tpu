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

#include "csrc/mlir/passes/canonicalize_broadcast_invariant_and_bounded_index.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// Half of INT32_MAX so any pairwise addition/subtraction of bounded intervals
// stays strictly inside signed 32-bit integer range without overflow.
// Caution: Do not use this as an arbitrary upper bound for unanalyzed inputs.
constexpr int64_t kMaxSafeI32Bound =  // GENERIC: semantic half of INT32_MAX
    std::numeric_limits<int32_t>::max() / 2;
constexpr int64_t kMaxI32Extent =  // GENERIC: semantic signed 32-bit max
    std::numeric_limits<int32_t>::max();
constexpr int64_t kUnitDimExtent = 1;   // GENERIC: semantic unit axis extent
constexpr int64_t kLeadingDimIdx = 0;   // GENERIC: semantic leading axis index
constexpr int64_t kRank1 = 1;           // GENERIC: semantic 1D tensor rank
constexpr int64_t kRank2 = 2;           // GENERIC: semantic 2D tensor rank
constexpr size_t kMinConcatInputs = 2;  // GENERIC: semantic binary concat min
constexpr int kMaxRecursionDepth = 16;  // GENERIC: capability traversal bound

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation*> dead_ops;
    func.walk([&](mlir::Operation* op) {
      if (!mlir::isa<mlir::func::ReturnOp>(op) && mlir::isOpTriviallyDead(op)) {
        dead_ops.push_back(op);
      }
    });
    for (auto it = dead_ops.rbegin(); it != dead_ops.rend(); ++it) {
      (*it)->erase();
      changed = true;
    }
  }
}

bool IsStaticRankedTensor(mlir::Type type,
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

bool IsIdentityBroadcastDims(llvm::ArrayRef<int64_t> dims, int64_t rank) {
  if (static_cast<int64_t>(dims.size()) != rank) {
    return false;
  }
  for (int64_t i = 0; i < rank; ++i) {
    if (dims[i] != i) {
      return false;
    }
  }
  return true;
}

mlir::Value StripIdentityBroadcasts(mlir::Value val) {
  while (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    auto in_ty =
        mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
    auto out_ty = mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
    if (!in_ty || !out_ty || in_ty != out_ty) {
      break;
    }
    if (!IsIdentityBroadcastDims(bcast.getBroadcastDimensions(),
                                 out_ty.getRank())) {
      break;
    }
    val = bcast.getOperand();
  }
  return val;
}

void FoldIdentityBroadcasts(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::BroadcastInDimOp> bcasts;
  func.walk(
      [&](mlir::stablehlo::BroadcastInDimOp op) { bcasts.push_back(op); });
  for (mlir::stablehlo::BroadcastInDimOp bcast : bcasts) {
    auto in_ty =
        mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
    auto out_ty = mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
    if (in_ty && out_ty && in_ty == out_ty &&
        IsIdentityBroadcastDims(bcast.getBroadcastDimensions(),
                                out_ty.getRank())) {
      bcast.replaceAllUsesWith(bcast.getOperand());
    }
  }
}

// Strips unit dimensions from `BroadcastInDimOp(ReshapeOp(x))` so that
// redundant intermediate singleton axes (such as `f32[C, 1]` broadcast to
// `f32[C, H, 1]` along `{0, 2}`) canonicalize to `f32[C]` broadcast along
// `{0}`.
void StripUnitDimsFromBroadcastOfReshape(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::BroadcastInDimOp> bcasts;
  func.walk(
      [&](mlir::stablehlo::BroadcastInDimOp op) { bcasts.push_back(op); });
  for (mlir::stablehlo::BroadcastInDimOp bcast : bcasts) {
    auto res_op =
        bcast.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!res_op) {
      continue;
    }
    mlir::RankedTensorType mid_ty;
    mlir::RankedTensorType out_ty;
    if (!IsStaticRankedTensor(res_op.getType(), &mid_ty) ||
        !IsStaticRankedTensor(bcast.getType(), &out_ty)) {
      continue;
    }
    if (mid_ty.getRank() < kRank2) {
      continue;
    }
    auto bcast_dims = bcast.getBroadcastDimensions();
    if (static_cast<int64_t>(bcast_dims.size()) != mid_ty.getRank()) {
      continue;
    }
    llvm::SmallVector<int64_t> kept_shape;
    llvm::SmallVector<int64_t> kept_dims;
    bool has_unit = false;
    for (int64_t i = 0; i < mid_ty.getRank(); ++i) {
      if (mid_ty.getDimSize(i) == kUnitDimExtent) {
        has_unit = true;
      } else {
        kept_shape.push_back(mid_ty.getDimSize(i));
        kept_dims.push_back(bcast_dims[i]);
      }
    }
    if (!has_unit || kept_shape.empty()) {
      continue;
    }
    mlir::OpBuilder builder(bcast);
    auto squeezed_ty =
        mlir::RankedTensorType::get(kept_shape, mid_ty.getElementType());
    auto squeezed_res = builder.create<mlir::stablehlo::ReshapeOp>(
        res_op.getLoc(), squeezed_ty, res_op.getOperand());
    auto new_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        bcast.getLoc(), out_ty, squeezed_res,
        builder.getDenseI64ArrayAttr(kept_dims));
    bcast.replaceAllUsesWith(new_bcast.getResult());
  }
}

bool IsSingletonExpandedOf1D(mlir::RankedTensorType nd_ty, int64_t extent_1d) {
  if (!nd_ty || !nd_ty.hasStaticShape() || nd_ty.getRank() < kRank2) {
    return false;
  }
  int non_unit_count = 0;
  for (int64_t d : nd_ty.getShape()) {
    if (d == kUnitDimExtent) {
      continue;
    }
    if (d != extent_1d) {
      return false;
    }
    ++non_unit_count;
  }
  return non_unit_count == 1 && extent_1d > kUnitDimExtent;
}

// Collapses singleton-expanded N-D elementwise normalization chains
// `Reshape_to_1D(DivOp(g_nd, SqrtOp(Reshape_to_ND(r_1d))))` into 1D `[C]`
// operations (`s_1d = SqrtOp(r_1d); DivOp(Reshape_to_1D(g_nd), s_1d)`),
// avoiding 3D layout expansion and unfused standalone sqrt/fdiv kernels.
void CollapseSingletonExpandedElementwise(mlir::func::FuncOp func) {
  FoldIdentityBroadcasts(func);
  EraseTriviallyDeadOps(func);
  StripUnitDimsFromBroadcastOfReshape(func);
  EraseTriviallyDeadOps(func);

  llvm::SmallVector<mlir::stablehlo::ReshapeOp> reshapes;
  func.walk([&](mlir::stablehlo::ReshapeOp op) { reshapes.push_back(op); });

  for (mlir::stablehlo::ReshapeOp res_to_1d : reshapes) {
    mlir::RankedTensorType ty_1d;
    if (!IsStaticRankedTensor(res_to_1d.getType(), &ty_1d) ||
        ty_1d.getRank() != kRank1) {
      continue;
    }
    const int64_t c_ext = ty_1d.getDimSize(kLeadingDimIdx);
    if (c_ext <= kUnitDimExtent) {
      continue;
    }
    mlir::Value src = StripIdentityBroadcasts(res_to_1d.getOperand());
    auto div_op = src.getDefiningOp<mlir::stablehlo::DivOp>();
    auto mul_op = src.getDefiningOp<mlir::stablehlo::MulOp>();
    mlir::Operation* bin_op =
        div_op ? div_op.getOperation() : mul_op.getOperation();
    if (bin_op == nullptr || !bin_op->hasOneUse()) {
      continue;
    }
    mlir::RankedTensorType nd_ty;
    if (!IsStaticRankedTensor(bin_op->getResult(0).getType(), &nd_ty) ||
        !IsSingletonExpandedOf1D(nd_ty, c_ext)) {
      continue;
    }
    mlir::Value lhs_nd = bin_op->getOperand(0);
    mlir::Value rhs_nd = bin_op->getOperand(1);

    auto match_unary_of_1d = [&](mlir::Value v, mlir::Value* r_1d_out,
                                 mlir::Operation** unary_out) -> bool {
      if (auto sqrt_op = v.getDefiningOp<mlir::stablehlo::SqrtOp>()) {
        if (auto r_res = sqrt_op.getOperand()
                             .getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
          if (r_res.getOperand().getType() == ty_1d) {
            *r_1d_out = r_res.getOperand();
            *unary_out = sqrt_op.getOperation();
            return true;
          }
        }
      }
      if (auto rsqrt_op = v.getDefiningOp<mlir::stablehlo::RsqrtOp>()) {
        if (auto r_res = rsqrt_op.getOperand()
                             .getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
          if (r_res.getOperand().getType() == ty_1d) {
            *r_1d_out = r_res.getOperand();
            *unary_out = rsqrt_op.getOperation();
            return true;
          }
        }
      }
      return false;
    };

    mlir::Value r_1d;
    mlir::Operation* unary_op = nullptr;
    if (!match_unary_of_1d(rhs_nd, &r_1d, &unary_op)) {
      continue;
    }

    mlir::OpBuilder builder(unary_op);
    mlir::Value s_1d;
    if (mlir::isa<mlir::stablehlo::SqrtOp>(unary_op)) {
      s_1d = builder.create<mlir::stablehlo::SqrtOp>(unary_op->getLoc(), ty_1d,
                                                     r_1d);
    } else {
      s_1d = builder.create<mlir::stablehlo::RsqrtOp>(unary_op->getLoc(), ty_1d,
                                                      r_1d);
    }
    auto s_nd_replacement = builder.create<mlir::stablehlo::ReshapeOp>(
        unary_op->getLoc(), nd_ty, s_1d);
    unary_op->getResult(0).replaceAllUsesWith(s_nd_replacement.getResult());

    mlir::OpBuilder bin_builder(res_to_1d);
    mlir::Value lhs_1d;
    if (auto conv_op = lhs_nd.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      mlir::RankedTensorType conv_in_ty;
      if (IsStaticRankedTensor(conv_op.getOperand().getType(), &conv_in_ty) &&
          IsSingletonExpandedOf1D(conv_in_ty, c_ext)) {
        auto in_1d_ty =
            mlir::RankedTensorType::get({c_ext}, conv_in_ty.getElementType());
        auto in_1d = bin_builder.create<mlir::stablehlo::ReshapeOp>(
            conv_op.getLoc(), in_1d_ty, conv_op.getOperand());
        lhs_1d = bin_builder.create<mlir::stablehlo::ConvertOp>(
            conv_op.getLoc(), ty_1d, in_1d);
      }
    }
    if (!lhs_1d) {
      lhs_1d = bin_builder.create<mlir::stablehlo::ReshapeOp>(bin_op->getLoc(),
                                                              ty_1d, lhs_nd);
    }

    mlir::Value res_1d;
    if (div_op) {
      res_1d = bin_builder.create<mlir::stablehlo::DivOp>(bin_op->getLoc(),
                                                          ty_1d, lhs_1d, s_1d);
    } else {
      res_1d = bin_builder.create<mlir::stablehlo::MulOp>(bin_op->getLoc(),
                                                          ty_1d, lhs_1d, s_1d);
    }
    res_to_1d.replaceAllUsesWith(res_1d);
  }
}

// Returns true if `val` is provably broadcast-invariant along axis `dim`
// (i.e. all slices along `dim` are identical).
bool IsInvariantAlongDim(mlir::Value val, int64_t dim, int depth = 0) {
  if (depth > kMaxRecursionDepth) {
    return false;
  }
  mlir::RankedTensorType rtt;
  if (!IsStaticRankedTensor(val.getType(), &rtt) || dim < 0 ||
      dim >= rtt.getRank()) {
    return false;
  }
  if (rtt.getDimSize(dim) == kUnitDimExtent) {
    return true;
  }
  mlir::Operation* def = val.getDefiningOp();
  if (def == nullptr) {
    return false;
  }
  if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(def)) {
    auto bdims = bcast.getBroadcastDimensions();
    for (size_t i = 0; i < bdims.size(); ++i) {
      if (bdims[i] == dim) {
        return IsInvariantAlongDim(bcast.getOperand(), static_cast<int64_t>(i),
                                   depth + 1);
      }
    }
    return true;
  }
  if (auto reshape = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(def)) {
    mlir::RankedTensorType in_ty;
    if (!IsStaticRankedTensor(reshape.getOperand().getType(), &in_ty)) {
      return false;
    }
    if (dim == kLeadingDimIdx && in_ty.getRank() >= kRank1 &&
        in_ty.getDimSize(kLeadingDimIdx) == rtt.getDimSize(kLeadingDimIdx)) {
      return IsInvariantAlongDim(reshape.getOperand(), kLeadingDimIdx,
                                 depth + 1);
    }
    return false;
  }
  if (auto transpose = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(def)) {
    auto perm = transpose.getPermutation();
    if (dim < static_cast<int64_t>(perm.size())) {
      return IsInvariantAlongDim(transpose.getOperand(), perm[dim], depth + 1);
    }
    return false;
  }
  if (auto concat = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(def)) {
    if (static_cast<int64_t>(concat.getDimension()) == dim) {
      return false;
    }
    for (mlir::Value inp : concat.getInputs()) {
      if (!IsInvariantAlongDim(inp, dim, depth + 1)) {
        return false;
      }
    }
    return true;
  }
  if (mlir::isa<mlir::stablehlo::ConvertOp, mlir::stablehlo::CosineOp,
                mlir::stablehlo::SineOp, mlir::stablehlo::NegOp,
                mlir::stablehlo::ExpOp, mlir::stablehlo::LogOp,
                mlir::stablehlo::SqrtOp, mlir::stablehlo::RsqrtOp,
                mlir::stablehlo::AbsOp>(def)) {
    return IsInvariantAlongDim(def->getOperand(0), dim, depth + 1);
  }
  if (mlir::isa<mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
                mlir::stablehlo::MulOp, mlir::stablehlo::DivOp,
                mlir::stablehlo::MinOp, mlir::stablehlo::MaxOp>(def)) {
    return IsInvariantAlongDim(def->getOperand(0), dim, depth + 1) &&
           IsInvariantAlongDim(def->getOperand(1), dim, depth + 1);
  }
  if (auto dot = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(def)) {
    auto dnums = dot.getDotDimensionNumbers();
    auto lhs_batch = dnums.getLhsBatchingDimensions();
    auto rhs_batch = dnums.getRhsBatchingDimensions();
    if (dim < static_cast<int64_t>(lhs_batch.size()) &&
        lhs_batch.size() == rhs_batch.size()) {
      return IsInvariantAlongDim(dot.getLhs(), lhs_batch[dim], depth + 1) &&
             IsInvariantAlongDim(dot.getRhs(), rhs_batch[dim], depth + 1);
    }
    return false;
  }
  return false;
}

// Projects `val` along its leading dimension 0 to `[0:1, ...]`, pushing the
// unit slice up to the `BroadcastInDimOp` roots when `IsInvariantAlongDim(val,
// 0)` holds.
mlir::Value ProjectLeadingZeroSlice(
    mlir::Value val, mlir::OpBuilder& builder,
    llvm::DenseMap<mlir::Value, mlir::Value>& memo) {
  auto it = memo.find(val);
  if (it != memo.end()) {
    return it->second;
  }
  auto rtt = mlir::cast<mlir::RankedTensorType>(val.getType());
  if (rtt.getDimSize(kLeadingDimIdx) == kUnitDimExtent) {
    memo[val] = val;
    return val;
  }
  llvm::SmallVector<int64_t> new_shape(rtt.getShape().begin(),
                                       rtt.getShape().end());
  new_shape[kLeadingDimIdx] = kUnitDimExtent;
  auto new_ty = mlir::RankedTensorType::get(new_shape, rtt.getElementType());

  mlir::Operation* def = val.getDefiningOp();
  if (def == nullptr) {
    return val;
  }
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(def);
  mlir::Value result;

  if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(def)) {
    auto bdims = bcast.getBroadcastDimensions();
    int64_t src_dim_for_0 = -1;
    for (size_t i = 0; i < bdims.size(); ++i) {
      if (bdims[i] == kLeadingDimIdx) {
        src_dim_for_0 = static_cast<int64_t>(i);
        break;
      }
    }
    if (src_dim_for_0 < 0) {
      result = builder.create<mlir::stablehlo::BroadcastInDimOp>(
          bcast.getLoc(), new_ty, bcast.getOperand(),
          bcast.getBroadcastDimensionsAttr());
    } else if (src_dim_for_0 == kLeadingDimIdx) {
      mlir::Value op_slice =
          ProjectLeadingZeroSlice(bcast.getOperand(), builder, memo);
      result = builder.create<mlir::stablehlo::BroadcastInDimOp>(
          bcast.getLoc(), new_ty, op_slice, bcast.getBroadcastDimensionsAttr());
    }
  } else if (auto reshape = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(def)) {
    mlir::RankedTensorType in_ty;
    if (IsStaticRankedTensor(reshape.getOperand().getType(), &in_ty) &&
        in_ty.getRank() >= kRank1 &&
        in_ty.getDimSize(kLeadingDimIdx) == rtt.getDimSize(kLeadingDimIdx)) {
      mlir::Value op_slice =
          ProjectLeadingZeroSlice(reshape.getOperand(), builder, memo);
      result = builder.create<mlir::stablehlo::ReshapeOp>(reshape.getLoc(),
                                                          new_ty, op_slice);
    }
  } else if (auto transpose =
                 mlir::dyn_cast<mlir::stablehlo::TransposeOp>(def)) {
    auto perm = transpose.getPermutation();
    if (!perm.empty() && perm[kLeadingDimIdx] == kLeadingDimIdx) {
      mlir::Value op_slice =
          ProjectLeadingZeroSlice(transpose.getOperand(), builder, memo);
      result = builder.create<mlir::stablehlo::TransposeOp>(
          transpose.getLoc(), new_ty, op_slice, transpose.getPermutationAttr());
    }
  } else if (auto concat =
                 mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(def)) {
    llvm::SmallVector<mlir::Value> new_inputs;
    new_inputs.reserve(concat.getInputs().size());
    for (mlir::Value inp : concat.getInputs()) {
      new_inputs.push_back(ProjectLeadingZeroSlice(inp, builder, memo));
    }
    result = builder.create<mlir::stablehlo::ConcatenateOp>(
        concat.getLoc(), new_ty, new_inputs, concat.getDimension());
  } else if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(def)) {
    mlir::Value op_slice =
        ProjectLeadingZeroSlice(conv.getOperand(), builder, memo);
    result = builder.create<mlir::stablehlo::ConvertOp>(conv.getLoc(), new_ty,
                                                        op_slice);
  } else if (auto cos_op = mlir::dyn_cast<mlir::stablehlo::CosineOp>(def)) {
    mlir::Value op_slice =
        ProjectLeadingZeroSlice(cos_op.getOperand(), builder, memo);
    result = builder.create<mlir::stablehlo::CosineOp>(cos_op.getLoc(), new_ty,
                                                       op_slice);
  } else if (auto sin_op = mlir::dyn_cast<mlir::stablehlo::SineOp>(def)) {
    mlir::Value op_slice =
        ProjectLeadingZeroSlice(sin_op.getOperand(), builder, memo);
    result = builder.create<mlir::stablehlo::SineOp>(sin_op.getLoc(), new_ty,
                                                     op_slice);
  } else if (auto neg_op = mlir::dyn_cast<mlir::stablehlo::NegOp>(def)) {
    mlir::Value op_slice =
        ProjectLeadingZeroSlice(neg_op.getOperand(), builder, memo);
    result = builder.create<mlir::stablehlo::NegOp>(neg_op.getLoc(), new_ty,
                                                    op_slice);
  } else if (auto dot = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(def)) {
    auto dnums = dot.getDotDimensionNumbers();
    auto lhs_batch = dnums.getLhsBatchingDimensions();
    auto rhs_batch = dnums.getRhsBatchingDimensions();
    if (!lhs_batch.empty() && !rhs_batch.empty() &&
        lhs_batch[kLeadingDimIdx] == kLeadingDimIdx &&
        rhs_batch[kLeadingDimIdx] == kLeadingDimIdx) {
      mlir::Value lhs_slice =
          ProjectLeadingZeroSlice(dot.getLhs(), builder, memo);
      mlir::Value rhs_slice =
          ProjectLeadingZeroSlice(dot.getRhs(), builder, memo);
      result = builder.create<mlir::stablehlo::DotGeneralOp>(
          dot.getLoc(), new_ty, lhs_slice, rhs_slice,
          dot.getDotDimensionNumbersAttr(), dot.getPrecisionConfigAttr(),
          dot.getAlgorithmAttr());
    }
  }

  if (!result) {
    llvm::SmallVector<int64_t> starts(rtt.getRank(), 0);
    llvm::SmallVector<int64_t> limits(rtt.getShape().begin(),
                                      rtt.getShape().end());
    limits[kLeadingDimIdx] = kUnitDimExtent;
    llvm::SmallVector<int64_t> strides(rtt.getRank(), kUnitDimExtent);
    result = builder.create<mlir::stablehlo::SliceOp>(
        def->getLoc(), new_ty, val, builder.getDenseI64ArrayAttr(starts),
        builder.getDenseI64ArrayAttr(limits),
        builder.getDenseI64ArrayAttr(strides));
  }

  memo[val] = result;
  return result;
}

void DemoteBroadcastInvariantDotAndFoldSlices(mlir::func::FuncOp func) {
  // 1. Demote one-sided broadcast-invariant leading batch dimensions in
  // DotGeneralOp: DotGeneral(lhs [B, M, K], BroadcastInDim(rhs [K, N]),
  // batch={0}) -> DotGeneral(lhs [B, M, K], rhs [K, N], batch={}).
  llvm::SmallVector<mlir::stablehlo::DotGeneralOp> dots;
  func.walk([&](mlir::stablehlo::DotGeneralOp op) { dots.push_back(op); });
  for (mlir::stablehlo::DotGeneralOp dot : dots) {
    mlir::RankedTensorType lhs_ty;
    mlir::RankedTensorType rhs_ty;
    mlir::RankedTensorType out_ty;
    if (!IsStaticRankedTensor(dot.getLhs().getType(), &lhs_ty) ||
        !IsStaticRankedTensor(dot.getRhs().getType(), &rhs_ty) ||
        !IsStaticRankedTensor(dot.getType(), &out_ty)) {
      continue;
    }
    auto dnums = dot.getDotDimensionNumbers();
    auto lhs_batch = dnums.getLhsBatchingDimensions();
    auto rhs_batch = dnums.getRhsBatchingDimensions();
    auto lhs_contract = dnums.getLhsContractingDimensions();
    auto rhs_contract = dnums.getRhsContractingDimensions();
    if (lhs_batch.size() != 1 || rhs_batch.size() != 1 ||
        lhs_contract.size() != 1 || rhs_contract.size() != 1) {
      continue;
    }
    if (lhs_batch[0] != kLeadingDimIdx || rhs_batch[0] != kLeadingDimIdx ||
        lhs_ty.getDimSize(kLeadingDimIdx) <= kUnitDimExtent) {
      continue;
    }
    auto rhs_bcast =
        dot.getRhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!rhs_bcast || IsInvariantAlongDim(dot.getLhs(), kLeadingDimIdx)) {
      continue;
    }
    mlir::RankedTensorType rhs_in_ty;
    if (!IsStaticRankedTensor(rhs_bcast.getOperand().getType(), &rhs_in_ty) ||
        rhs_in_ty.getRank() != rhs_ty.getRank() - 1) {
      continue;
    }
    auto bdims = rhs_bcast.getBroadcastDimensions();
    bool is_drop_0 =
        (static_cast<int64_t>(bdims.size()) == rhs_in_ty.getRank());
    for (int64_t i = 0; is_drop_0 && i < rhs_in_ty.getRank(); ++i) {
      if (bdims[i] != i + 1) {
        is_drop_0 = false;
      }
    }
    if (!is_drop_0 || rhs_contract[0] == kLeadingDimIdx) {
      continue;
    }
    mlir::OpBuilder builder(dot);
    auto new_dnums = mlir::stablehlo::DotDimensionNumbersAttr::get(
        dot.getContext(),
        /*lhsBatchingDimensions=*/{},
        /*rhsBatchingDimensions=*/{},
        /*lhsContractingDimensions=*/lhs_contract,
        /*rhsContractingDimensions=*/{rhs_contract[0] - 1});
    auto new_dot = builder.create<mlir::stablehlo::DotGeneralOp>(
        dot.getLoc(), out_ty, dot.getLhs(), rhs_bcast.getOperand(), new_dnums,
        dot.getPrecisionConfigAttr(), dot.getAlgorithmAttr());
    dot.replaceAllUsesWith(new_dot.getResult());
  }

  // 2. Normalize SliceOp start offsets along broadcast-invariant dimensions:
  // Slice(v, d=[k : k + m]) -> Slice(v, d=[0 : m]) when v is invariant on d.
  llvm::SmallVector<mlir::stablehlo::SliceOp> slices;
  func.walk([&](mlir::stablehlo::SliceOp op) { slices.push_back(op); });
  for (mlir::stablehlo::SliceOp slice : slices) {
    mlir::RankedTensorType in_ty;
    if (!IsStaticRankedTensor(slice.getOperand().getType(), &in_ty)) {
      continue;
    }
    auto starts = slice.getStartIndices();
    auto limits = slice.getLimitIndices();
    auto strides = slice.getStrides();
    llvm::SmallVector<int64_t> new_starts(starts.begin(), starts.end());
    llvm::SmallVector<int64_t> new_limits(limits.begin(), limits.end());
    bool changed = false;
    for (int64_t d = 0; d < in_ty.getRank(); ++d) {
      if (strides[d] == kUnitDimExtent && starts[d] > 0 &&
          IsInvariantAlongDim(slice.getOperand(), d)) {
        new_limits[d] = limits[d] - starts[d];
        new_starts[d] = 0;
        changed = true;
      }
    }
    if (changed) {
      mlir::OpBuilder builder(slice);
      auto new_slice = builder.create<mlir::stablehlo::SliceOp>(
          slice.getLoc(), slice.getType(), slice.getOperand(),
          builder.getDenseI64ArrayAttr(new_starts),
          builder.getDenseI64ArrayAttr(new_limits), slice.getStridesAttr());
      slice.replaceAllUsesWith(new_slice.getResult());
    }
  }

  // 3. Coalesce ConcatenateOp of contiguous SliceOp (or ReshapeOp(SliceOp))
  // operands on the same source tensor into a single SliceOp (+ ReshapeOp).
  llvm::SmallVector<mlir::stablehlo::ConcatenateOp> concats;
  func.walk([&](mlir::stablehlo::ConcatenateOp op) { concats.push_back(op); });
  for (mlir::stablehlo::ConcatenateOp concat : concats) {
    auto inputs = concat.getInputs();
    if (inputs.size() < kMinConcatInputs) {
      continue;
    }
    mlir::RankedTensorType out_ty;
    if (!IsStaticRankedTensor(concat.getType(), &out_ty)) {
      continue;
    }
    const int64_t concat_dim = static_cast<int64_t>(concat.getDimension());

    struct SlicePart {
      mlir::Value src;
      llvm::SmallVector<int64_t> starts;
      llvm::SmallVector<int64_t> limits;
      int64_t src_concat_dim = -1;
      bool has_reshape = false;
    };

    auto match_part = [&](mlir::Value v, SlicePart* part) -> bool {
      mlir::stablehlo::SliceOp slice_op =
          v.getDefiningOp<mlir::stablehlo::SliceOp>();
      bool has_res = false;
      if (!slice_op) {
        auto res_op = v.getDefiningOp<mlir::stablehlo::ReshapeOp>();
        if (!res_op) {
          return false;
        }
        slice_op =
            res_op.getOperand().getDefiningOp<mlir::stablehlo::SliceOp>();
        if (!slice_op) {
          return false;
        }
        has_res = true;
      }
      mlir::RankedTensorType s_ty;
      mlir::RankedTensorType v_ty;
      if (!IsStaticRankedTensor(slice_op.getType(), &s_ty) ||
          !IsStaticRankedTensor(v.getType(), &v_ty)) {
        return false;
      }
      for (int64_t st : slice_op.getStrides()) {
        if (st != kUnitDimExtent) {
          return false;
        }
      }
      int64_t mapped_dim = concat_dim;
      if (has_res) {
        // Map non-unit dimensions (or unit dimensions in order) between s_ty
        // and v_ty when ReshapeOp only squeezes unit dimensions.
        if (v_ty.getRank() >= s_ty.getRank()) {
          return false;
        }
        int64_t s_idx = 0;
        int64_t v_idx = 0;
        mapped_dim = -1;
        while (s_idx < s_ty.getRank() && v_idx < v_ty.getRank()) {
          if (s_ty.getDimSize(s_idx) == v_ty.getDimSize(v_idx)) {
            if (v_idx == concat_dim) {
              mapped_dim = s_idx;
            }
            ++s_idx;
            ++v_idx;
          } else if (s_ty.getDimSize(s_idx) == kUnitDimExtent) {
            ++s_idx;
          } else {
            return false;
          }
        }
        while (s_idx < s_ty.getRank()) {
          if (s_ty.getDimSize(s_idx) != kUnitDimExtent) {
            return false;
          }
          ++s_idx;
        }
        if (v_idx != v_ty.getRank() || mapped_dim < 0) {
          return false;
        }
      }
      part->src = slice_op.getOperand();
      part->starts.assign(slice_op.getStartIndices().begin(),
                          slice_op.getStartIndices().end());
      part->limits.assign(slice_op.getLimitIndices().begin(),
                          slice_op.getLimitIndices().end());
      part->src_concat_dim = mapped_dim;
      part->has_reshape = has_res;
      return true;
    };

    SlicePart first;
    if (!match_part(inputs[0], &first)) {
      continue;
    }
    int64_t cur_limit = first.limits[first.src_concat_dim];
    bool valid = true;
    for (size_t i = 1; i < inputs.size(); ++i) {
      SlicePart next;
      if (!match_part(inputs[i], &next) || next.src != first.src ||
          next.src_concat_dim != first.src_concat_dim ||
          next.has_reshape != first.has_reshape) {
        valid = false;
        break;
      }
      for (size_t d = 0; d < first.starts.size(); ++d) {
        if (static_cast<int64_t>(d) == first.src_concat_dim) {
          if (next.starts[d] != cur_limit) {
            valid = false;
            break;
          }
          cur_limit = next.limits[d];
        } else if (next.starts[d] != first.starts[d] ||
                   next.limits[d] != first.limits[d]) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        break;
      }
    }
    if (!valid) {
      continue;
    }

    llvm::SmallVector<int64_t> merged_limits = first.limits;
    merged_limits[first.src_concat_dim] = cur_limit;
    llvm::SmallVector<int64_t> merged_shape;
    for (size_t d = 0; d < first.starts.size(); ++d) {
      merged_shape.push_back(merged_limits[d] - first.starts[d]);
    }
    llvm::SmallVector<int64_t> strides(first.starts.size(), kUnitDimExtent);
    mlir::OpBuilder builder(concat);
    auto slice_ty =
        mlir::RankedTensorType::get(merged_shape, out_ty.getElementType());
    mlir::Value merged = builder.create<mlir::stablehlo::SliceOp>(
        concat.getLoc(), slice_ty, first.src,
        builder.getDenseI64ArrayAttr(first.starts),
        builder.getDenseI64ArrayAttr(merged_limits),
        builder.getDenseI64ArrayAttr(strides));
    if (slice_ty != out_ty) {
      merged = builder.create<mlir::stablehlo::ReshapeOp>(concat.getLoc(),
                                                          out_ty, merged);
    }
    concat.replaceAllUsesWith(merged);
  }

  // 4. Push unit leading-dimension SliceOp `[0:1, 0:D1, ..., 0:Dk]` up through
  // broadcast-invariant chains (`ConvertOp`, `CosineOp`/`SineOp`,
  // `ConcatenateOp`, `TransposeOp`, `ReshapeOp`, `DotGeneralOp`) to fold with
  // `BroadcastInDimOp`.
  llvm::DenseMap<mlir::Value, mlir::Value> memo;
  llvm::SmallVector<mlir::stablehlo::SliceOp> unit_slices;
  func.walk([&](mlir::stablehlo::SliceOp op) { unit_slices.push_back(op); });
  for (mlir::stablehlo::SliceOp slice : unit_slices) {
    mlir::RankedTensorType in_ty;
    if (!IsStaticRankedTensor(slice.getOperand().getType(), &in_ty) ||
        in_ty.getRank() < kRank2 ||
        in_ty.getDimSize(kLeadingDimIdx) <= kUnitDimExtent) {
      continue;
    }
    auto starts = slice.getStartIndices();
    auto limits = slice.getLimitIndices();
    auto strides = slice.getStrides();
    if (starts[kLeadingDimIdx] != 0 ||
        limits[kLeadingDimIdx] != kUnitDimExtent ||
        strides[kLeadingDimIdx] != kUnitDimExtent) {
      continue;
    }
    bool full_other_dims = true;
    for (int64_t d = 1; d < in_ty.getRank(); ++d) {
      if (starts[d] != 0 || limits[d] != in_ty.getDimSize(d) ||
          strides[d] != kUnitDimExtent) {
        full_other_dims = false;
        break;
      }
    }
    if (!full_other_dims ||
        !IsInvariantAlongDim(slice.getOperand(), kLeadingDimIdx)) {
      continue;
    }
    mlir::OpBuilder builder(slice);
    mlir::Value projected =
        ProjectLeadingZeroSlice(slice.getOperand(), builder, memo);
    slice.replaceAllUsesWith(projected);
  }
}

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
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      return TryGetConstantFloatBounds(bcast.getOperand(), min_out, max_out);
    }
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

// Checks whether a float value feeding `ConvertOp(float -> i64)` is bounded
// by an upstream `ClampOp`.
bool TryInferFloatToIntInterval(
    mlir::stablehlo::ConvertOp conv_op,
    const llvm::DenseSet<mlir::Operation*>& /*comp_ops*/, IntInterval* iv_out) {
  mlir::Value src = conv_op.getOperand();
  while (auto res = src.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    src = res.getOperand();
  }
  if (auto clamp = src.getDefiningOp<mlir::stablehlo::ClampOp>()) {
    double lo_a = 0.0, hi_a = 0.0, lo_b = 0.0, hi_b = 0.0;
    if (TryGetConstantFloatBounds(clamp.getMin(), &lo_a, &hi_a) &&
        TryGetConstantFloatBounds(clamp.getMax(), &lo_b, &hi_b)) {
      if (lo_a >= -static_cast<double>(kMaxSafeI32Bound) &&
          hi_b <= static_cast<double>(kMaxSafeI32Bound)) {
        *iv_out = {static_cast<int64_t>(std::floor(lo_a)),
                   static_cast<int64_t>(std::ceil(hi_b)), true};
        return IsWithinSafeI32(*iv_out);
      }
    }
  }
  return false;
}

void NarrowBoundedI64IndexSubgraphs(mlir::func::FuncOp func) {
  // Step A: Hoist and narrow `IotaOp(i64) -> [ReshapeOp/BroadcastInDimOp]* ->
  // ConvertOp(float)` so 1D `IotaOp` converts in `i32` before broadcasting.
  llvm::SmallVector<mlir::stablehlo::ConvertOp> float_convs;
  func.walk([&](mlir::stablehlo::ConvertOp op) {
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
        !iota_ty.getElementType().isInteger(64)) {
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

  // Step B: Narrow closed bounded `i64` connected components in the entry block
  // to `i32`.
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
        rtt.getElementType().isInteger(64)) {
      i64_ops.push_back(&op);
      i64_op_set.insert(&op);
    }
  }
  if (i64_ops.empty()) {
    return;
  }

  llvm::DenseMap<mlir::Operation*, llvm::SmallVector<mlir::Operation*>> adj;
  auto add_edge = [&](mlir::Operation* a, mlir::Operation* b) {
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

  llvm::DenseSet<mlir::Operation*> visited;
  for (mlir::Operation* seed : i64_ops) {
    if (visited.contains(seed)) {
      continue;
    }
    llvm::DenseSet<mlir::Operation*> comp_set;
    llvm::SmallVector<mlir::Operation*> queue = {seed};
    visited.insert(seed);
    comp_set.insert(seed);
    while (!queue.empty()) {
      mlir::Operation* cur = queue.pop_back_val();
      for (mlir::Operation* nxt : adj[cur]) {
        if (visited.insert(nxt).second) {
          comp_set.insert(nxt);
          queue.push_back(nxt);
        }
      }
    }

    // Order ops in `comp_set` in block topological order.
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

    // Validate all ops, external inputs, and external sinks, and compute
    // static 64-bit intervals.
    bool valid_comp = true;
    llvm::DenseMap<mlir::Value, IntInterval> intervals;

    for (mlir::Operation* op : comp_topo) {
      // Check external i64 operands (e.g., block arguments).
      for (mlir::Value opnd : op->getOperands()) {
        auto opnd_ty = mlir::dyn_cast<mlir::RankedTensorType>(opnd.getType());
        if (opnd_ty && opnd_ty.getElementType().isInteger(64)) {
          mlir::Operation* def = opnd.getDefiningOp();
          if (def == nullptr || !comp_set.contains(def)) {
            valid_comp = false;
            break;
          }
        }
      }
      if (!valid_comp) {
        break;
      }

      // Check external users (sinks).
      for (mlir::Operation* user : op->getUsers()) {
        if (comp_set.contains(user)) {
          continue;
        }
        if (auto cmp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(user)) {
          if (!comp_set.contains(cmp.getLhs().getDefiningOp()) ||
              !comp_set.contains(cmp.getRhs().getDefiningOp())) {
            valid_comp = false;
            break;
          }
        } else if (mlir::isa<mlir::stablehlo::ConvertOp>(user)) {
          // Valid sink: converts i64 to non-i64 type.
        } else if (auto gather =
                       mlir::dyn_cast<mlir::stablehlo::GatherOp>(user)) {
          if (gather.getStartIndices() != op->getResult(0) ||
              comp_set.contains(gather.getOperand().getDefiningOp())) {
            valid_comp = false;
            break;
          }
          mlir::RankedTensorType table_ty;
          if (!IsStaticRankedTensor(gather.getOperand().getType(), &table_ty)) {
            valid_comp = false;
            break;
          }
          for (int64_t d : gather.getDimensionNumbers().getStartIndexMap()) {
            if (table_ty.getDimSize(d) > kMaxI32Extent) {
              valid_comp = false;
              break;
            }
          }
          if (!valid_comp) {
            break;
          }
        } else {
          valid_comp = false;
          break;
        }
      }
      if (!valid_comp) {
        break;
      }

      // Compute interval for `op`.
      IntInterval iv;
      if (auto iota = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op)) {
        auto rtt = mlir::cast<mlir::RankedTensorType>(iota.getType());
        int64_t ext =
            rtt.getDimSize(static_cast<int64_t>(iota.getIotaDimension()));
        if (ext <= 0 || ext > kMaxSafeI32Bound) {
          valid_comp = false;
          break;
        }
        iv = {0, ext - 1, true};
      } else if (auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
        if (!TryGetConstantIntBounds(cst, &iv)) {
          valid_comp = false;
          break;
        }
      } else if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
        mlir::RankedTensorType in_ty;
        if (!IsStaticRankedTensor(conv.getOperand().getType(), &in_ty)) {
          valid_comp = false;
          break;
        }
        mlir::Type el_ty = in_ty.getElementType();
        if (el_ty.isInteger(1)) {
          iv = {0, 1, true};
        } else if (el_ty.isInteger(8)) {
          iv = {-128, 255, true};
        } else if (el_ty.isInteger(16)) {
          iv = {-32768, 65535, true};
        } else if (el_ty.isInteger(32)) {
          valid_comp = false;
          break;
        } else if (mlir::isa<mlir::FloatType>(el_ty)) {
          if (!TryInferFloatToIntInterval(conv, comp_set, &iv)) {
            valid_comp = false;
            break;
          }
        } else {
          valid_comp = false;
          break;
        }
      } else if (mlir::isa<mlir::stablehlo::ReshapeOp,
                           mlir::stablehlo::BroadcastInDimOp,
                           mlir::stablehlo::TransposeOp,
                           mlir::stablehlo::SliceOp>(op)) {
        iv = intervals[op->getOperand(0)];
      } else if (auto concat =
                     mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(op)) {
        iv = intervals[concat.getInputs()[0]];
        for (size_t i = 1; i < concat.getInputs().size(); ++i) {
          IntInterval nxt = intervals[concat.getInputs()[i]];
          iv.lo = std::min(iv.lo, nxt.lo);
          iv.hi = std::max(iv.hi, nxt.hi);
          iv.valid = iv.valid && nxt.valid;
        }
      } else if (auto add_op = mlir::dyn_cast<mlir::stablehlo::AddOp>(op)) {
        IntInterval a = intervals[add_op.getLhs()];
        IntInterval b = intervals[add_op.getRhs()];
        iv = {a.lo + b.lo, a.hi + b.hi, a.valid && b.valid};
      } else if (auto sub_op =
                     mlir::dyn_cast<mlir::stablehlo::SubtractOp>(op)) {
        IntInterval a = intervals[sub_op.getLhs()];
        IntInterval b = intervals[sub_op.getRhs()];
        iv = {a.lo - b.hi, a.hi - b.lo, a.valid && b.valid};
      } else if (auto mul_op = mlir::dyn_cast<mlir::stablehlo::MulOp>(op)) {
        IntInterval a = intervals[mul_op.getLhs()];
        IntInterval b = intervals[mul_op.getRhs()];
        if (!a.valid || !b.valid) {
          valid_comp = false;
          break;
        }
        int64_t p1 = 0;
        int64_t p2 = 0;
        int64_t p3 = 0;
        int64_t p4 = 0;
        if (__builtin_mul_overflow(a.lo, b.lo, &p1) ||
            __builtin_mul_overflow(a.lo, b.hi, &p2) ||
            __builtin_mul_overflow(a.hi, b.lo, &p3) ||
            __builtin_mul_overflow(a.hi, b.hi, &p4)) {
          valid_comp = false;
          break;
        }
        iv = {std::min({p1, p2, p3, p4}), std::max({p1, p2, p3, p4}), true};
      } else if (auto min_op = mlir::dyn_cast<mlir::stablehlo::MinOp>(op)) {
        IntInterval a = intervals[min_op.getLhs()];
        IntInterval b = intervals[min_op.getRhs()];
        iv = {std::min(a.lo, b.lo), std::min(a.hi, b.hi), a.valid && b.valid};
      } else if (auto max_op = mlir::dyn_cast<mlir::stablehlo::MaxOp>(op)) {
        IntInterval a = intervals[max_op.getLhs()];
        IntInterval b = intervals[max_op.getRhs()];
        iv = {std::max(a.lo, b.lo), std::max(a.hi, b.hi), a.valid && b.valid};
      } else if (auto clamp_op = mlir::dyn_cast<mlir::stablehlo::ClampOp>(op)) {
        IntInterval lo_iv = intervals[clamp_op.getMin()];
        IntInterval x_iv = intervals[clamp_op.getOperand()];
        IntInterval hi_iv = intervals[clamp_op.getMax()];
        iv = {std::max(lo_iv.lo, x_iv.lo), std::min(hi_iv.hi, x_iv.hi),
              lo_iv.valid && x_iv.valid && hi_iv.valid};
      } else if (auto neg_op = mlir::dyn_cast<mlir::stablehlo::NegOp>(op)) {
        IntInterval a = intervals[neg_op.getOperand()];
        iv = {-a.hi, -a.lo, a.valid};
      } else if (auto abs_op = mlir::dyn_cast<mlir::stablehlo::AbsOp>(op)) {
        IntInterval a = intervals[abs_op.getOperand()];
        int64_t max_abs = std::max(std::abs(a.lo), std::abs(a.hi));
        int64_t min_abs = (a.lo <= 0 && a.hi >= 0)
                              ? 0
                              : std::min(std::abs(a.lo), std::abs(a.hi));
        iv = {min_abs, max_abs, a.valid};
      } else if (auto sel_op = mlir::dyn_cast<mlir::stablehlo::SelectOp>(op)) {
        IntInterval a = intervals[sel_op.getOnTrue()];
        IntInterval b = intervals[sel_op.getOnFalse()];
        iv = {std::min(a.lo, b.lo), std::max(a.hi, b.hi), a.valid && b.valid};
      } else {
        valid_comp = false;
        break;
      }

      if (!IsWithinSafeI32(iv)) {
        valid_comp = false;
        break;
      }
      intervals[op->getResult(0)] = iv;
    }

    if (!valid_comp) {
      continue;
    }

    // Rewrite every op in `comp_topo` from i64 to i32.
    llvm::DenseMap<mlir::Value, mlir::Value> i32_map;
    for (mlir::Operation* op : comp_topo) {
      mlir::OpBuilder builder(op);
      auto old_ty =
          mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
      auto new_ty =
          mlir::RankedTensorType::get(old_ty.getShape(), builder.getI32Type());
      mlir::Value new_val;

      if (auto iota = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op)) {
        new_val = builder.create<mlir::stablehlo::IotaOp>(
            iota.getLoc(), new_ty, iota.getIotaDimensionAttr());
      } else if (auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
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
        new_val = builder.create<mlir::stablehlo::ConstantOp>(cst.getLoc(),
                                                              new_ty, new_attr);
      } else if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
        if (conv.getOperand().getType() == new_ty) {
          new_val = conv.getOperand();
        } else {
          new_val = builder.create<mlir::stablehlo::ConvertOp>(
              conv.getLoc(), new_ty, conv.getOperand());
        }
      } else if (auto res = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
        new_val = builder.create<mlir::stablehlo::ReshapeOp>(
            res.getLoc(), new_ty, i32_map[res.getOperand()]);
      } else if (auto bcast =
                     mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
        new_val = builder.create<mlir::stablehlo::BroadcastInDimOp>(
            bcast.getLoc(), new_ty, i32_map[bcast.getOperand()],
            bcast.getBroadcastDimensionsAttr());
      } else if (auto tr = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op)) {
        new_val = builder.create<mlir::stablehlo::TransposeOp>(
            tr.getLoc(), new_ty, i32_map[tr.getOperand()],
            tr.getPermutationAttr());
      } else if (auto sl = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op)) {
        new_val = builder.create<mlir::stablehlo::SliceOp>(
            sl.getLoc(), new_ty, i32_map[sl.getOperand()],
            sl.getStartIndicesAttr(), sl.getLimitIndicesAttr(),
            sl.getStridesAttr());
      } else if (auto cat =
                     mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(op)) {
        llvm::SmallVector<mlir::Value> new_inps;
        for (mlir::Value inp : cat.getInputs()) {
          new_inps.push_back(i32_map[inp]);
        }
        new_val = builder.create<mlir::stablehlo::ConcatenateOp>(
            cat.getLoc(), new_ty, new_inps, cat.getDimension());
      } else if (auto add_op = mlir::dyn_cast<mlir::stablehlo::AddOp>(op)) {
        new_val = builder.create<mlir::stablehlo::AddOp>(
            add_op.getLoc(), new_ty, i32_map[add_op.getLhs()],
            i32_map[add_op.getRhs()]);
      } else if (auto sub_op =
                     mlir::dyn_cast<mlir::stablehlo::SubtractOp>(op)) {
        new_val = builder.create<mlir::stablehlo::SubtractOp>(
            sub_op.getLoc(), new_ty, i32_map[sub_op.getLhs()],
            i32_map[sub_op.getRhs()]);
      } else if (auto mul_op = mlir::dyn_cast<mlir::stablehlo::MulOp>(op)) {
        new_val = builder.create<mlir::stablehlo::MulOp>(
            mul_op.getLoc(), new_ty, i32_map[mul_op.getLhs()],
            i32_map[mul_op.getRhs()]);
      } else if (auto min_op = mlir::dyn_cast<mlir::stablehlo::MinOp>(op)) {
        new_val = builder.create<mlir::stablehlo::MinOp>(
            min_op.getLoc(), new_ty, i32_map[min_op.getLhs()],
            i32_map[min_op.getRhs()]);
      } else if (auto max_op = mlir::dyn_cast<mlir::stablehlo::MaxOp>(op)) {
        new_val = builder.create<mlir::stablehlo::MaxOp>(
            max_op.getLoc(), new_ty, i32_map[max_op.getLhs()],
            i32_map[max_op.getRhs()]);
      } else if (auto clamp_op = mlir::dyn_cast<mlir::stablehlo::ClampOp>(op)) {
        new_val = builder.create<mlir::stablehlo::ClampOp>(
            clamp_op.getLoc(), new_ty, i32_map[clamp_op.getMin()],
            i32_map[clamp_op.getOperand()], i32_map[clamp_op.getMax()]);
      } else if (auto neg_op = mlir::dyn_cast<mlir::stablehlo::NegOp>(op)) {
        new_val = builder.create<mlir::stablehlo::NegOp>(
            neg_op.getLoc(), new_ty, i32_map[neg_op.getOperand()]);
      } else if (auto abs_op = mlir::dyn_cast<mlir::stablehlo::AbsOp>(op)) {
        new_val = builder.create<mlir::stablehlo::AbsOp>(
            abs_op.getLoc(), new_ty, i32_map[abs_op.getOperand()]);
      } else if (auto sel_op = mlir::dyn_cast<mlir::stablehlo::SelectOp>(op)) {
        new_val = builder.create<mlir::stablehlo::SelectOp>(
            sel_op.getLoc(), new_ty, sel_op.getPred(),
            i32_map[sel_op.getOnTrue()], i32_map[sel_op.getOnFalse()]);
      }
      i32_map[op->getResult(0)] = new_val;
    }

    // Redirect all external sinks to use the new `i32` values.
    for (mlir::Operation* op : comp_topo) {
      mlir::Value old_v = op->getResult(0);
      mlir::Value new_v = i32_map[old_v];
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
  }
}

struct CanonicalizeBroadcastInvariantAndBoundedIndexPass
    : public mlir::PassWrapper<
          CanonicalizeBroadcastInvariantAndBoundedIndexPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CanonicalizeBroadcastInvariantAndBoundedIndexPass)

  llvm::StringRef getArgument() const override {
    return "canonicalize-broadcast-invariant-and-bounded-index";
  }
  llvm::StringRef getDescription() const override {
    return "Demotes broadcast-invariant DotGeneralOp batch dims, folds "
           "invariant SliceOp/ConcatenateOp chains, narrows bounded i64 "
           "IotaOp index subgraphs to i32, and collapses singleton-expanded "
           "elementwise normalizations.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    CollapseSingletonExpandedElementwise(func);
    NarrowBoundedI64IndexSubgraphs(func);
    DemoteBroadcastInvariantDotAndFoldSlices(func);
    EraseTriviallyDeadOps(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
createCanonicalizeBroadcastInvariantAndBoundedIndexPass() {
  return std::make_unique<CanonicalizeBroadcastInvariantAndBoundedIndexPass>();
}

}  // namespace torch_tpu
