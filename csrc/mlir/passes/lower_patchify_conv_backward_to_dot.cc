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

#include "csrc/mlir/passes/lower_patchify_conv_backward_to_dot.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// GENERIC: hardware TPU vector register sublane tile height is 8.
constexpr int64_t kTpuSublaneCount = 8;

bool EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool any_changed = false;
  bool local_changed = true;
  while (local_changed) {
    local_changed = false;
    for (auto& block : func.getBody()) {
      for (auto& op : llvm::make_early_inc_range(llvm::reverse(block))) {
        if (&op != func.getOperation() && op.use_empty() &&
            mlir::isOpTriviallyDead(&op)) {
          op.erase();
          local_changed = true;
          any_changed = true;
        }
      }
    }
  }
  return any_changed;
}

mlir::Value UnwrapReshapes(mlir::Value val) {
  while (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    auto in_t =
        llvm::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
    if (!in_t || !in_t.hasStaticShape()) {
      break;
    }
    val = resh.getOperand();
  }
  return val;
}

mlir::Value EmitReshapeIfNeeded(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value val,
                                mlir::RankedTensorType dst_type) {
  val = UnwrapReshapes(val);
  if (val.getType() == dst_type) {
    return val;
  }
  return builder.create<mlir::stablehlo::ReshapeOp>(loc, dst_type, val);
}

llvm::SmallVector<int64_t, 8> GetNonUnitDimSizes(
    llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t, 8> out;
  for (int64_t d : shape) {
    if (d > 1) {
      out.push_back(d);
    }
  }
  return out;
}

bool IsIdentityPerm(llvm::ArrayRef<int64_t> perm) {
  for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
    if (perm[i] != i) {
      return false;
    }
  }
  return true;
}

bool ComputeSqueezedPermutation(llvm::ArrayRef<int64_t> in_shape,
                                llvm::ArrayRef<int64_t> perm,
                                llvm::SmallVectorImpl<int64_t>& squeezed_perm) {
  if (in_shape.size() != perm.size()) {
    return false;
  }
  llvm::SmallVector<int64_t, 8> axis_to_squeezed(in_shape.size(), -1);
  int64_t next_idx = 0;
  for (size_t i = 0; i < in_shape.size(); ++i) {
    if (in_shape[i] > 1) {
      axis_to_squeezed[i] = next_idx++;
    }
  }
  squeezed_perm.clear();
  squeezed_perm.reserve(next_idx);
  for (int64_t d : perm) {
    if (d < 0 || d >= static_cast<int64_t>(in_shape.size())) {
      return false;
    }
    if (axis_to_squeezed[d] != -1) {
      squeezed_perm.push_back(axis_to_squeezed[d]);
    }
  }
  return static_cast<int64_t>(squeezed_perm.size()) == next_idx;
}

mlir::Value PeelUnitDimPreservingReshapes(
    mlir::Value val, llvm::ArrayRef<int64_t> non_unit_sizes) {
  mlir::Value best = val;
  while (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    val = resh.getOperand();
    auto in_type = llvm::dyn_cast<mlir::RankedTensorType>(val.getType());
    if (!in_type || !in_type.hasStaticShape()) {
      break;
    }
    if (llvm::ArrayRef<int64_t>(GetNonUnitDimSizes(in_type.getShape())) ==
        non_unit_sizes) {
      best = val;
    } else {
      break;
    }
  }
  return best;
}

bool IsPatchifyDotOutput(mlir::Value v) {
  v = UnwrapReshapes(v);
  if (auto dot = v.getDefiningOp<mlir::stablehlo::DotGeneralOp>()) {
    auto dnums = dot.getDotDimensionNumbers();
    return dnums.getLhsBatchingDimensions().empty() &&
           dnums.getRhsBatchingDimensions().empty() &&
           dnums.getLhsContractingDimensions().size() == 1 &&
           dnums.getRhsContractingDimensions().size() == 1 &&
           dnums.getLhsContractingDimensions()[0] == 1 &&
           dnums.getRhsContractingDimensions()[0] == 1;
  }
  if (auto add = v.getDefiningOp<mlir::stablehlo::AddOp>()) {
    for (int side = 0; side < 2; ++side) {
      mlir::Value a = (side == 0) ? add.getLhs() : add.getRhs();
      mlir::Value b = (side == 0) ? add.getRhs() : add.getLhs();
      auto bcast = b.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
      if (bcast && bcast.getBroadcastDimensions().size() == 1 &&
          bcast.getBroadcastDimensions()[0] == 1 && IsPatchifyDotOutput(a)) {
        return true;
      }
    }
  }
  return false;
}

bool CancelPatchifyDotOutputTransposes(mlir::func::FuncOp func) {
  bool any_changed = false;
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::stablehlo::TransposeOp, 16> tr_ops;
    func.walk([&](mlir::stablehlo::TransposeOp op) { tr_ops.push_back(op); });

    for (mlir::stablehlo::TransposeOp tr2 : tr_ops) {
      if (!tr2 || !tr2->getBlock() || tr2->use_empty()) {
        continue;
      }
      auto tr2_type = llvm::dyn_cast<mlir::RankedTensorType>(tr2.getType());
      auto tr2_in_type =
          llvm::dyn_cast<mlir::RankedTensorType>(tr2.getOperand().getType());
      if (!tr2_type || !tr2_type.hasStaticShape() || !tr2_in_type ||
          !tr2_in_type.hasStaticShape()) {
        continue;
      }

      llvm::SmallVector<int64_t, 8> non_unit_before_tr2 =
          GetNonUnitDimSizes(tr2_in_type.getShape());
      llvm::SmallVector<int64_t, 8> squeezed_perm2;
      if (non_unit_before_tr2.size() >= 2 &&
          ComputeSqueezedPermutation(tr2_in_type.getShape(),
                                     tr2.getPermutation(), squeezed_perm2) &&
          !IsIdentityPerm(squeezed_perm2)) {
        mlir::Value peeled = PeelUnitDimPreservingReshapes(tr2.getOperand(),
                                                           non_unit_before_tr2);
        if (auto tr1 = peeled.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
          auto tr1_in_type = llvm::dyn_cast<mlir::RankedTensorType>(
              tr1.getOperand().getType());
          llvm::SmallVector<int64_t, 8> squeezed_perm1;
          if (tr1_in_type && tr1_in_type.hasStaticShape() &&
              ComputeSqueezedPermutation(tr1_in_type.getShape(),
                                         tr1.getPermutation(),
                                         squeezed_perm1) &&
              squeezed_perm1.size() == squeezed_perm2.size() &&
              !IsIdentityPerm(squeezed_perm1)) {
            llvm::SmallVector<int64_t, 8> composed_perm(squeezed_perm2.size());
            for (size_t i = 0; i < squeezed_perm2.size(); ++i) {
              composed_perm[i] = squeezed_perm1[squeezed_perm2[i]];
            }
            if (IsIdentityPerm(composed_perm)) {
              llvm::SmallVector<int64_t, 8> orig_non_unit =
                  GetNonUnitDimSizes(tr1_in_type.getShape());
              mlir::Value root_val = PeelUnitDimPreservingReshapes(
                  tr1.getOperand(), orig_non_unit);
              if (!IsPatchifyDotOutput(root_val)) {
                continue;
              }
              mlir::OpBuilder builder(tr2);
              mlir::Value rep = EmitReshapeIfNeeded(builder, tr2.getLoc(),
                                                    root_val, tr2_type);
              tr2.replaceAllUsesWith(rep);
              changed = true;
              any_changed = true;
              continue;
            }
          }
        }
      }
    }
    if (changed) {
      EraseTriviallyDeadOps(func);
    }
  }
  return any_changed;
}

bool ExtractSpatialPadding(mlir::stablehlo::ConvolutionOp conv, int64_t n_sp,
                           llvm::SmallVectorImpl<int64_t>& pad_lo,
                           llvm::SmallVectorImpl<int64_t>& pad_hi,
                           bool& has_non_zero_pad) {
  pad_lo.assign(n_sp, 0);
  pad_hi.assign(n_sp, 0);
  has_non_zero_pad = false;
  auto pad = conv.getPadding();
  if (!pad) {
    return true;
  }
  llvm::SmallVector<int64_t, 8> vals;
  for (llvm::APInt v : pad->getValues<llvm::APInt>()) {
    vals.push_back(v.getSExtValue());
  }
  if (static_cast<int64_t>(vals.size()) != 2 * n_sp) {
    return false;
  }
  for (int64_t i = 0; i < n_sp; ++i) {
    pad_lo[i] = vals[2 * i];
    pad_hi[i] = vals[2 * i + 1];
    if (pad_lo[i] != 0 || pad_hi[i] != 0) {
      has_non_zero_pad = true;
    }
  }
  return true;
}

// Applies window-aligned spatial slicing and zero-padding to `raw_lhs`
// (`[B, C_in, In_0, ...]`) so each spatial axis `i` has exact extent
// `grid_dims[i] * kernel_dims[i]`.
mlir::Value ApplyWindowAlignedSpatialSliceAndPad(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value raw_lhs,
    llvm::ArrayRef<int64_t> grid_dims, llvm::ArrayRef<int64_t> kernel_dims,
    llvm::ArrayRef<int64_t> pad_lo) {
  auto lhs_t = llvm::dyn_cast<mlir::RankedTensorType>(raw_lhs.getType());
  if (!lhs_t || !lhs_t.hasStaticShape()) {
    return mlir::Value();
  }
  llvm::ArrayRef<int64_t> lhs_shape = lhs_t.getShape();
  const int64_t n_sp = grid_dims.size();
  if (static_cast<int64_t>(lhs_shape.size()) != 2 + n_sp) {
    return mlir::Value();
  }

  llvm::SmallVector<int64_t, 6> slice_starts = {0, 0};
  llvm::SmallVector<int64_t, 6> slice_limits = {lhs_shape[0], lhs_shape[1]};
  llvm::SmallVector<int64_t, 6> slice_strides(2 + n_sp, 1);
  llvm::SmallVector<int64_t, 6> sliced_shape = {lhs_shape[0], lhs_shape[1]};
  llvm::SmallVector<int64_t, 6> low_pads = {0, 0};
  llvm::SmallVector<int64_t, 6> high_pads = {0, 0};
  llvm::SmallVector<int64_t, 6> interior_pads(2 + n_sp, 0);
  llvm::SmallVector<int64_t, 6> padded_shape = {lhs_shape[0], lhs_shape[1]};

  bool needs_slice = false;
  bool needs_pad = false;
  for (int64_t i = 0; i < n_sp; ++i) {
    const int64_t in_i = lhs_shape[2 + i];
    const int64_t target_i = grid_dims[i] * kernel_dims[i];
    const int64_t plo = pad_lo[i];
    const int64_t start_i = std::max<int64_t>(0, -plo);
    const int64_t limit_i = std::min<int64_t>(in_i, target_i - plo);
    if (start_i >= limit_i || start_i < 0 || limit_i > in_i) {
      return mlir::Value();
    }
    const int64_t sliced_len = limit_i - start_i;
    const int64_t lpad = std::max<int64_t>(0, plo);
    const int64_t hpad = target_i - (lpad + sliced_len);
    if (lpad < 0 || hpad < 0) {
      return mlir::Value();
    }
    if (start_i > 0 || limit_i < in_i) {
      needs_slice = true;
    }
    if (lpad > 0 || hpad > 0) {
      needs_pad = true;
    }
    slice_starts.push_back(start_i);
    slice_limits.push_back(limit_i);
    sliced_shape.push_back(sliced_len);
    low_pads.push_back(lpad);
    high_pads.push_back(hpad);
    padded_shape.push_back(target_i);
  }

  mlir::Type elem_t = lhs_t.getElementType();
  mlir::Value cur = raw_lhs;
  if (needs_slice) {
    auto sliced_t = mlir::RankedTensorType::get(sliced_shape, elem_t);
    cur = builder.create<mlir::stablehlo::SliceOp>(
        loc, sliced_t, cur, builder.getDenseI64ArrayAttr(slice_starts),
        builder.getDenseI64ArrayAttr(slice_limits),
        builder.getDenseI64ArrayAttr(slice_strides));
  }
  if (needs_pad) {
    auto scalar_t = mlir::RankedTensorType::get({}, elem_t);
    mlir::Value zero_val = builder.create<mlir::stablehlo::ConstantOp>(
        loc, scalar_t,
        llvm::cast<mlir::ElementsAttr>(builder.getZeroAttr(scalar_t)));
    auto padded_t = mlir::RankedTensorType::get(padded_shape, elem_t);
    cur = builder.create<mlir::stablehlo::PadOp>(
        loc, padded_t, cur, zero_val, builder.getDenseI64ArrayAttr(low_pads),
        builder.getDenseI64ArrayAttr(high_pads),
        builder.getDenseI64ArrayAttr(interior_pads));
  }
  return cur;
}

// Extracts the 2D patch matrix `[M_total, K_total]` (`M_total = B * prod(G_i)`,
// `K_total = C_in * prod(K_i)`) from `raw_lhs` (`[B, C_in, In_0, ...]`).
mlir::Value BuildPatchMatrix2D(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::Value raw_lhs, int64_t b, int64_t c_in,
                               llvm::ArrayRef<int64_t> grid_dims,
                               llvm::ArrayRef<int64_t> kernel_dims,
                               llvm::ArrayRef<int64_t> pad_lo,
                               bool has_non_zero_pad, int64_t m_total,
                               int64_t k_total, mlir::Type elem_t) {
  const int64_t n_sp = grid_dims.size();
  bool lhs_from_btchw = false;
  mlir::Value src_lhs = raw_lhs;
  if (!has_non_zero_pad && n_sp == 3 && src_lhs.hasOneUse()) {
    if (auto lhs_tr = src_lhs.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
      auto p = lhs_tr.getPermutation();
      if (p.size() == 5 && p[0] == 0 && p[1] == 2 && p[2] == 1 && p[3] == 3 &&
          p[4] == 4) {
        src_lhs = lhs_tr.getOperand();
        lhs_from_btchw = true;
      }
    }
  }

  if (!lhs_from_btchw && has_non_zero_pad) {
    src_lhs = ApplyWindowAlignedSpatialSliceAndPad(
        builder, loc, src_lhs, grid_dims, kernel_dims, pad_lo);
    if (!src_lhs) {
      return mlir::Value();
    }
  }

  llvm::SmallVector<int64_t, 8> split_shape;
  llvm::SmallVector<int64_t, 8> full_perm;
  if (lhs_from_btchw) {
    split_shape = {b,
                   grid_dims[0],
                   kernel_dims[0],
                   c_in,
                   grid_dims[1],
                   kernel_dims[1],
                   grid_dims[2],
                   kernel_dims[2]};
    full_perm = {0, 1, 4, 6, 3, 2, 5, 7};
  } else {
    split_shape = {b, c_in};
    for (int64_t i = 0; i < n_sp; ++i) {
      split_shape.push_back(grid_dims[i]);
      split_shape.push_back(kernel_dims[i]);
    }
    full_perm = {0};
    for (int64_t i = 0; i < n_sp; ++i) {
      full_perm.push_back(2 + 2 * i);
    }
    full_perm.push_back(1);
    for (int64_t i = 0; i < n_sp; ++i) {
      full_perm.push_back(3 + 2 * i);
    }
  }

  llvm::SmallVector<int64_t, 8> sq_split_shape =
      GetNonUnitDimSizes(split_shape);
  llvm::SmallVector<int64_t, 8> sq_perm;
  if (!ComputeSqueezedPermutation(split_shape, full_perm, sq_perm)) {
    return mlir::Value();
  }

  auto sq_in_type = mlir::RankedTensorType::get(sq_split_shape, elem_t);
  mlir::Value lhs_sq = EmitReshapeIfNeeded(builder, loc, src_lhs, sq_in_type);
  mlir::Value lhs_perm = lhs_sq;
  if (!IsIdentityPerm(sq_perm)) {
    llvm::SmallVector<int64_t, 8> sq_out_shape(sq_perm.size());
    for (size_t i = 0; i < sq_perm.size(); ++i) {
      sq_out_shape[i] = sq_split_shape[sq_perm[i]];
    }
    auto sq_out_type = mlir::RankedTensorType::get(sq_out_shape, elem_t);
    lhs_perm = builder.create<mlir::stablehlo::TransposeOp>(
        loc, sq_out_type, lhs_sq, builder.getDenseI64ArrayAttr(sq_perm));
  }

  auto lhs_2d_type = mlir::RankedTensorType::get({m_total, k_total}, elem_t);
  return EmitReshapeIfNeeded(builder, loc, lhs_perm, lhs_2d_type);
}

mlir::Value ReconstructConvOutput(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::Value dot_result, int64_t b,
                                  int64_t num_patches_per_batch,
                                  int64_t m_total, int64_t c_out,
                                  mlir::Type elem_t,
                                  mlir::RankedTensorType out_t) {
  if (b == 1) {
    auto cm_type = mlir::RankedTensorType::get({c_out, m_total}, elem_t);
    mlir::Value tr_cm = builder.create<mlir::stablehlo::TransposeOp>(
        loc, cm_type, dot_result, builder.getDenseI64ArrayAttr({1, 0}));
    return EmitReshapeIfNeeded(builder, loc, tr_cm, out_t);
  }
  auto bmc_type =
      mlir::RankedTensorType::get({b, num_patches_per_batch, c_out}, elem_t);
  mlir::Value r_bmc = EmitReshapeIfNeeded(builder, loc, dot_result, bmc_type);
  auto bcm_type =
      mlir::RankedTensorType::get({b, c_out, num_patches_per_batch}, elem_t);
  mlir::Value tr_bcm = builder.create<mlir::stablehlo::TransposeOp>(
      loc, bcm_type, r_bmc, builder.getDenseI64ArrayAttr({0, 2, 1}));
  return EmitReshapeIfNeeded(builder, loc, tr_bcm, out_t);
}

void LowerForwardConvWithPatchMatrix(mlir::stablehlo::ConvolutionOp conv,
                                     mlir::Value lhs_2d, int64_t b,
                                     int64_t num_patches_per_batch,
                                     int64_t m_total, int64_t k_total,
                                     int64_t c_out, mlir::Type elem_t,
                                     mlir::RankedTensorType out_t) {
  mlir::OpBuilder builder(conv);
  mlir::Location loc = conv.getLoc();
  auto rhs_2d_type = mlir::RankedTensorType::get({c_out, k_total}, elem_t);
  mlir::Value rhs_2d =
      EmitReshapeIfNeeded(builder, loc, conv.getRhs(), rhs_2d_type);

  auto dot_dnums = mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(), /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{}, /*lhsContractingDimensions=*/{1},
      /*rhsContractingDimensions=*/{1});
  auto dot_2d_type = mlir::RankedTensorType::get({m_total, c_out}, elem_t);
  mlir::Value dot_val = builder.create<mlir::stablehlo::DotGeneralOp>(
      loc, dot_2d_type, lhs_2d, rhs_2d, dot_dnums,
      conv.getPrecisionConfigAttr(),
      /*algorithm=*/mlir::stablehlo::DotAlgorithmAttr{});

  if (conv->hasOneUse()) {
    if (auto add_user =
            llvm::dyn_cast<mlir::stablehlo::AddOp>(*conv->getUsers().begin())) {
      mlir::Value other = add_user.getLhs() == conv.getResult()
                              ? add_user.getRhs()
                              : add_user.getLhs();
      if (auto bcast =
              other.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
        llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
        auto bias_t = llvm::dyn_cast<mlir::RankedTensorType>(
            bcast.getOperand().getType());
        if (bias_t && bias_t.hasStaticShape() && bias_t.getRank() == 1 &&
            bias_t.getDimSize(0) == c_out && bdims.size() == 1 &&
            bdims[0] == 1) {
          builder.setInsertionPoint(add_user);
          mlir::Value bias_2d =
              builder.create<mlir::stablehlo::BroadcastInDimOp>(
                  add_user.getLoc(), dot_2d_type, bcast.getOperand(),
                  builder.getDenseI64ArrayAttr({1}));
          mlir::Value dot_plus_bias = builder.create<mlir::stablehlo::AddOp>(
              add_user.getLoc(), dot_val, bias_2d);
          mlir::Value out_val = ReconstructConvOutput(
              builder, add_user.getLoc(), dot_plus_bias, b,
              num_patches_per_batch, m_total, c_out, elem_t, out_t);
          add_user.replaceAllUsesWith(out_val);
          return;
        }
      }
    }
  }

  mlir::Value out_val =
      ReconstructConvOutput(builder, loc, dot_val, b, num_patches_per_batch,
                            m_total, c_out, elem_t, out_t);
  conv.replaceAllUsesWith(out_val);
}

bool MatchForwardPatchifyConv(mlir::stablehlo::ConvolutionOp conv,
                              llvm::SmallVectorImpl<int64_t>& grid_dims,
                              llvm::SmallVectorImpl<int64_t>& kernel_dims,
                              llvm::SmallVectorImpl<int64_t>& pad_lo,
                              bool& has_non_zero_pad) {
  if (!conv || !conv->getBlock() || conv->use_empty()) {
    return false;
  }
  if (conv.getFeatureGroupCount() != 1 || conv.getBatchGroupCount() != 1) {
    return false;
  }
  auto lhs_t = llvm::dyn_cast<mlir::RankedTensorType>(conv.getLhs().getType());
  auto rhs_t = llvm::dyn_cast<mlir::RankedTensorType>(conv.getRhs().getType());
  auto out_t = llvm::dyn_cast<mlir::RankedTensorType>(conv.getType());
  if (!lhs_t || !rhs_t || !out_t || !lhs_t.hasStaticShape() ||
      !rhs_t.hasStaticShape() || !out_t.hasStaticShape()) {
    return false;
  }
  mlir::Type elem_t = out_t.getElementType();
  if (lhs_t.getElementType() != elem_t || rhs_t.getElementType() != elem_t) {
    return false;
  }

  auto dnums = conv.getDimensionNumbers();
  llvm::ArrayRef<int64_t> in_sp = dnums.getInputSpatialDimensions();
  llvm::ArrayRef<int64_t> ker_sp = dnums.getKernelSpatialDimensions();
  llvm::ArrayRef<int64_t> out_sp = dnums.getOutputSpatialDimensions();
  const int64_t n_sp = in_sp.size();
  if ((n_sp != 2 && n_sp != 3) || ker_sp.size() != in_sp.size() ||
      out_sp.size() != in_sp.size()) {
    return false;
  }
  if (dnums.getInputBatchDimension() != 0 ||
      dnums.getInputFeatureDimension() != 1 ||
      dnums.getOutputBatchDimension() != 0 ||
      dnums.getOutputFeatureDimension() != 1 ||
      dnums.getKernelOutputFeatureDimension() != 0 ||
      dnums.getKernelInputFeatureDimension() != 1) {
    return false;
  }
  for (int64_t i = 0; i < n_sp; ++i) {
    if (in_sp[i] != 2 + i || ker_sp[i] != 2 + i || out_sp[i] != 2 + i) {
      return false;
    }
  }
  if (auto ld = conv.getLhsDilation()) {
    if (!llvm::all_of(*ld, [](int64_t v) { return v == 1; })) {
      return false;
    }
  }
  if (auto rd = conv.getRhsDilation()) {
    if (!llvm::all_of(*rd, [](int64_t v) { return v == 1; })) {
      return false;
    }
  }
  if (auto wr = conv.getWindowReversal()) {
    if (!llvm::all_of(*wr, [](bool v) { return !v; })) {
      return false;
    }
  }
  auto strides_opt = conv.getWindowStrides();
  if (!strides_opt.has_value() ||
      static_cast<int64_t>(strides_opt->size()) != n_sp) {
    return false;
  }
  llvm::SmallVector<int64_t, 4> pad_hi;
  if (!ExtractSpatialPadding(conv, n_sp, pad_lo, pad_hi, has_non_zero_pad)) {
    return false;
  }

  llvm::ArrayRef<int64_t> strides = *strides_opt;
  llvm::ArrayRef<int64_t> lhs_shape = lhs_t.getShape();
  llvm::ArrayRef<int64_t> rhs_shape = rhs_t.getShape();
  llvm::ArrayRef<int64_t> out_shape = out_t.getShape();

  grid_dims.clear();
  kernel_dims.clear();
  int64_t num_patches_per_batch = 1;
  int64_t patch_area = 1;
  for (int64_t i = 0; i < n_sp; ++i) {
    const int64_t k_i = rhs_shape[2 + i];
    const int64_t s_i = strides[i];
    const int64_t in_i = lhs_shape[2 + i];
    const int64_t out_i = out_shape[2 + i];
    if (k_i <= 0 || out_i <= 0 || s_i != k_i) {
      return false;
    }
    const int64_t padded_in_i = in_i + pad_lo[i] + pad_hi[i];
    if (padded_in_i < out_i * k_i || padded_in_i >= (out_i + 1) * k_i) {
      return false;
    }
    grid_dims.push_back(out_i);
    kernel_dims.push_back(k_i);
    num_patches_per_batch *= out_i;
    patch_area *= k_i;
  }
  if (patch_area <= 1) {
    return false;
  }

  const int64_t b = lhs_shape[0];
  const int64_t c_in = lhs_shape[1];
  const int64_t c_out = out_shape[1];
  const int64_t m_total = b * num_patches_per_batch;
  const int64_t k_total = c_in * patch_area;
  // GENERIC: hardware - only lower input-channel-sublane-underutilized
  // patchify stems (c_in <= kTpuSublaneCount).
  if (c_in <= 0 || c_in > kTpuSublaneCount || c_out <= 1 || k_total <= 1 ||
      m_total <= 1) {
    return false;
  }
  return true;
}

bool MatchBackwardPatchifyWeightGradConv(
    mlir::stablehlo::ConvolutionOp conv,
    llvm::SmallVectorImpl<int64_t>& grid_dims,
    llvm::SmallVectorImpl<int64_t>& kernel_dims,
    llvm::SmallVectorImpl<int64_t>& pad_lo, bool& has_non_zero_pad) {
  if (!conv || !conv->getBlock() || conv->use_empty()) {
    return false;
  }
  if (conv.getFeatureGroupCount() != 1 || conv.getBatchGroupCount() != 1) {
    return false;
  }
  auto lhs_t = llvm::dyn_cast<mlir::RankedTensorType>(conv.getLhs().getType());
  auto rhs_t = llvm::dyn_cast<mlir::RankedTensorType>(conv.getRhs().getType());
  auto out_t = llvm::dyn_cast<mlir::RankedTensorType>(conv.getType());
  if (!lhs_t || !rhs_t || !out_t || !lhs_t.hasStaticShape() ||
      !rhs_t.hasStaticShape() || !out_t.hasStaticShape()) {
    return false;
  }
  mlir::Type elem_t = out_t.getElementType();
  if (lhs_t.getElementType() != elem_t || rhs_t.getElementType() != elem_t) {
    return false;
  }

  auto dnums = conv.getDimensionNumbers();
  llvm::ArrayRef<int64_t> in_sp = dnums.getInputSpatialDimensions();
  llvm::ArrayRef<int64_t> ker_sp = dnums.getKernelSpatialDimensions();
  llvm::ArrayRef<int64_t> out_sp = dnums.getOutputSpatialDimensions();
  const int64_t n_sp = in_sp.size();
  if ((n_sp != 2 && n_sp != 3) || ker_sp.size() != in_sp.size() ||
      out_sp.size() != in_sp.size()) {
    return false;
  }
  if (dnums.getInputBatchDimension() != 1 ||
      dnums.getInputFeatureDimension() != 0 ||
      dnums.getKernelInputFeatureDimension() != 0 ||
      dnums.getKernelOutputFeatureDimension() != 1 ||
      dnums.getOutputBatchDimension() != 1 ||
      dnums.getOutputFeatureDimension() != 0) {
    return false;
  }
  for (int64_t i = 0; i < n_sp; ++i) {
    if (in_sp[i] != 2 + i || ker_sp[i] != 2 + i || out_sp[i] != 2 + i) {
      return false;
    }
  }
  if (auto strides = conv.getWindowStrides()) {
    if (!llvm::all_of(*strides, [](int64_t v) { return v == 1; })) {
      return false;
    }
  }
  if (auto ld = conv.getLhsDilation()) {
    if (!llvm::all_of(*ld, [](int64_t v) { return v == 1; })) {
      return false;
    }
  }
  if (auto wr = conv.getWindowReversal()) {
    if (!llvm::all_of(*wr, [](bool v) { return !v; })) {
      return false;
    }
  }
  auto rd_opt = conv.getRhsDilation();
  if (!rd_opt.has_value() || static_cast<int64_t>(rd_opt->size()) != n_sp) {
    return false;
  }
  llvm::SmallVector<int64_t, 4> pad_hi;
  if (!ExtractSpatialPadding(conv, n_sp, pad_lo, pad_hi, has_non_zero_pad)) {
    return false;
  }

  llvm::ArrayRef<int64_t> rd = *rd_opt;
  llvm::ArrayRef<int64_t> lhs_shape = lhs_t.getShape();
  llvm::ArrayRef<int64_t> rhs_shape = rhs_t.getShape();
  llvm::ArrayRef<int64_t> out_shape = out_t.getShape();
  if (lhs_shape[0] != rhs_shape[0] || lhs_shape[1] != out_shape[1] ||
      rhs_shape[1] != out_shape[0]) {
    return false;
  }

  grid_dims.clear();
  kernel_dims.clear();
  int64_t num_patches_per_batch = 1;
  int64_t patch_area = 1;
  for (int64_t i = 0; i < n_sp; ++i) {
    const int64_t g_i = rhs_shape[2 + i];
    const int64_t k_i = out_shape[2 + i];
    const int64_t in_i = lhs_shape[2 + i];
    if (g_i <= 0 || k_i <= 0 || rd[i] != k_i) {
      return false;
    }
    if (in_i + pad_lo[i] + pad_hi[i] != g_i * k_i) {
      return false;
    }
    grid_dims.push_back(g_i);
    kernel_dims.push_back(k_i);
    num_patches_per_batch *= g_i;
    patch_area *= k_i;
  }
  if (patch_area <= 1) {
    return false;
  }

  const int64_t b = lhs_shape[0];
  const int64_t c_in = lhs_shape[1];
  const int64_t c_out = out_shape[0];
  const int64_t m_total = b * num_patches_per_batch;
  const int64_t k_total = c_in * patch_area;
  // GENERIC: hardware - only lower input-channel-sublane-underutilized
  // patchify stems (c_in <= kTpuSublaneCount).
  if (c_in <= 0 || c_in > kTpuSublaneCount || c_out <= 1 || k_total <= 1 ||
      m_total <= 1) {
    return false;
  }
  return true;
}

// Converts `grad_output` (`[B, C_out, G_0, ...]`) into 2D `[M_total, C_out]`,
// canceling any upstream inverse `TransposeOp` (`[B, G_0, ..., C_out] ->
// [B, C_out, G_0, ...]`) and rewriting any sibling bias-gradient `ReduceOp` on
// `grad_output` across `{0, 2, ...}` to reduce `dy_2d` across `{0}`.
mlir::Value BuildGradOutput2DAndFoldBiasReduce(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value grad_out,
    int64_t m_total, int64_t c_out, int64_t n_sp, mlir::Type elem_t) {
  auto grad_t = llvm::dyn_cast<mlir::RankedTensorType>(grad_out.getType());
  llvm::ArrayRef<int64_t> grad_shape = grad_t.getShape();
  auto dy_2d_type = mlir::RankedTensorType::get({m_total, c_out}, elem_t);

  llvm::SmallVector<int64_t, 6> expected_red_dims = {0};
  for (int64_t i = 0; i < n_sp; ++i) {
    expected_red_dims.push_back(2 + i);
  }
  mlir::Operation* earliest_pt = &*builder.getInsertionPoint();
  llvm::SmallVector<mlir::stablehlo::ReduceOp, 4> red_users;
  for (mlir::Operation* user : grad_out.getUsers()) {
    if (auto red = llvm::dyn_cast<mlir::stablehlo::ReduceOp>(user)) {
      if (red->getBlock() == earliest_pt->getBlock() &&
          red.getInputs().size() == 1 && red.getInitValues().size() == 1 &&
          red->getNumResults() == 1 &&
          red.getDimensions() == llvm::ArrayRef<int64_t>(expected_red_dims)) {
        red_users.push_back(red);
        if (red->isBeforeInBlock(earliest_pt)) {
          earliest_pt = red.getOperation();
        }
      }
    }
  }
  mlir::OpBuilder dy_builder(earliest_pt);

  // Try canceling an upstream inverse `TransposeOp` either at the flattened
  // `[B, C_out, M_patches]` shape (standard ViT `transpose(1, 2) -> reshape`)
  // or at the unflattened `[B, C_out, G_0, ...]` shape.
  const int64_t b = grad_shape[0];
  const int64_t num_patches = m_total / b;
  llvm::SmallVector<int64_t, 4> flat_3d_shape = {b, c_out, num_patches};
  llvm::SmallVector<int64_t, 4> perm_3d = {0, 2, 1};

  llvm::SmallVector<int64_t, 8> full_perm = {0};
  for (int64_t i = 0; i < n_sp; ++i) {
    full_perm.push_back(2 + i);
  }
  full_perm.push_back(1);

  auto try_cancel_upstream_transpose =
      [&](llvm::ArrayRef<int64_t> candidate_shape,
          llvm::ArrayRef<int64_t> candidate_perm) -> mlir::Value {
    llvm::SmallVector<int64_t, 8> sq_in = GetNonUnitDimSizes(candidate_shape);
    llvm::SmallVector<int64_t, 8> sq_p;
    if (sq_in.size() < 2 ||
        !ComputeSqueezedPermutation(candidate_shape, candidate_perm, sq_p) ||
        IsIdentityPerm(sq_p)) {
      return mlir::Value();
    }
    mlir::Value peeled = PeelUnitDimPreservingReshapes(grad_out, sq_in);
    auto tr1 = peeled.getDefiningOp<mlir::stablehlo::TransposeOp>();
    if (!tr1) {
      return mlir::Value();
    }
    auto tr1_in_t =
        llvm::dyn_cast<mlir::RankedTensorType>(tr1.getOperand().getType());
    llvm::SmallVector<int64_t, 8> sq_perm1;
    if (!tr1_in_t || !tr1_in_t.hasStaticShape() ||
        !ComputeSqueezedPermutation(tr1_in_t.getShape(), tr1.getPermutation(),
                                    sq_perm1) ||
        sq_perm1.size() != sq_p.size()) {
      return mlir::Value();
    }
    llvm::SmallVector<int64_t, 8> composed(sq_p.size());
    for (size_t i = 0; i < sq_p.size(); ++i) {
      composed[i] = sq_perm1[sq_p[i]];
    }
    if (!IsIdentityPerm(composed)) {
      return mlir::Value();
    }
    llvm::SmallVector<int64_t, 8> orig_non_unit =
        GetNonUnitDimSizes(tr1_in_t.getShape());
    mlir::Value root_val =
        PeelUnitDimPreservingReshapes(tr1.getOperand(), orig_non_unit);
    return EmitReshapeIfNeeded(dy_builder, loc, root_val, dy_2d_type);
  };

  mlir::Value dy_2d = try_cancel_upstream_transpose(flat_3d_shape, perm_3d);
  if (!dy_2d) {
    dy_2d = try_cancel_upstream_transpose(grad_shape, full_perm);
  }
  if (!dy_2d) {
    llvm::SmallVector<int64_t, 8> sq_in_shape =
        GetNonUnitDimSizes(flat_3d_shape);
    llvm::SmallVector<int64_t, 8> sq_perm;
    if (sq_in_shape.size() >= 2 &&
        ComputeSqueezedPermutation(flat_3d_shape, perm_3d, sq_perm) &&
        !IsIdentityPerm(sq_perm)) {
      auto sq_in_t = mlir::RankedTensorType::get(sq_in_shape, elem_t);
      mlir::Value g_sq =
          EmitReshapeIfNeeded(dy_builder, loc, grad_out, sq_in_t);
      llvm::SmallVector<int64_t, 8> sq_out_shape(sq_perm.size());
      for (size_t i = 0; i < sq_perm.size(); ++i) {
        sq_out_shape[i] = sq_in_shape[sq_perm[i]];
      }
      auto sq_out_t = mlir::RankedTensorType::get(sq_out_shape, elem_t);
      mlir::Value g_tr = dy_builder.create<mlir::stablehlo::TransposeOp>(
          loc, sq_out_t, g_sq, dy_builder.getDenseI64ArrayAttr(sq_perm));
      dy_2d = EmitReshapeIfNeeded(dy_builder, loc, g_tr, dy_2d_type);
    } else {
      dy_2d = EmitReshapeIfNeeded(dy_builder, loc, grad_out, dy_2d_type);
    }
  }

  for (mlir::stablehlo::ReduceOp red : red_users) {
    mlir::OpBuilder red_builder(red);
    auto new_red = red_builder.create<mlir::stablehlo::ReduceOp>(
        red.getLoc(), red.getResultTypes(), mlir::ValueRange{dy_2d},
        red.getInitValues(), red_builder.getDenseI64ArrayAttr({0}));
    mlir::IRMapping mapping;
    red.getBody().cloneInto(&new_red.getBody(), mapping);
    red.replaceAllUsesWith(new_red.getResults());
  }

  return dy_2d;
}

// Sinks `ReshapeOp(bwd_dot[C_out, K_total] -> [C_out, C_in, K_0, ...])` across
// direct elementwise users whose other operand is either `out_4d` itself or a
// scalar `BroadcastInDimOp`. This keeps `bwd_dot` consumed by 2D elementwise
// ops rather than a direct 4D/5D `ReshapeOp`, preventing XLA's
// `TpuBroadcastRewriter::CanTransformDot` from un-flattening `bwd_dot` into a
// multi-dimensional dot / padded spatial convolution.
void SinkBackwardDotReshapeThroughElementwiseUsers(
    mlir::Value bwd_dot, mlir::Value out_4d, mlir::RankedTensorType dot_2d_type,
    mlir::RankedTensorType out_t) {
  llvm::SmallVector<mlir::Operation*, 8> users(out_4d.getUsers().begin(),
                                               out_4d.getUsers().end());
  for (mlir::Operation* user : users) {
    if (!user || user->getNumResults() != 1 ||
        user->getResult(0).getType() != out_t) {
      continue;
    }
    if (!llvm::isa<mlir::stablehlo::MulOp, mlir::stablehlo::AddOp,
                   mlir::stablehlo::SubtractOp, mlir::stablehlo::DivOp>(user)) {
      continue;
    }
    mlir::Value lhs = user->getOperand(0);
    mlir::Value rhs = user->getOperand(1);
    mlir::OpBuilder builder(user);
    mlir::Location uloc = user->getLoc();

    auto to_2d_operand = [&](mlir::Value v) -> mlir::Value {
      if (v == out_4d) {
        return bwd_dot;
      }
      if (auto bcast = v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
        if (bcast.getBroadcastDimensions().empty()) {
          return builder.create<mlir::stablehlo::BroadcastInDimOp>(
              uloc, dot_2d_type, bcast.getOperand(),
              builder.getDenseI64ArrayAttr({}));
        }
      }
      return mlir::Value();
    };

    mlir::Value lhs_2d = to_2d_operand(lhs);
    mlir::Value rhs_2d = to_2d_operand(rhs);
    if (!lhs_2d || !rhs_2d) {
      continue;
    }

    mlir::Value op_2d;
    if (llvm::isa<mlir::stablehlo::MulOp>(user)) {
      op_2d = builder.create<mlir::stablehlo::MulOp>(uloc, dot_2d_type, lhs_2d,
                                                     rhs_2d);
    } else if (llvm::isa<mlir::stablehlo::AddOp>(user)) {
      op_2d = builder.create<mlir::stablehlo::AddOp>(uloc, dot_2d_type, lhs_2d,
                                                     rhs_2d);
    } else if (llvm::isa<mlir::stablehlo::SubtractOp>(user)) {
      op_2d = builder.create<mlir::stablehlo::SubtractOp>(uloc, dot_2d_type,
                                                          lhs_2d, rhs_2d);
    } else if (llvm::isa<mlir::stablehlo::DivOp>(user)) {
      op_2d = builder.create<mlir::stablehlo::DivOp>(uloc, dot_2d_type, lhs_2d,
                                                     rhs_2d);
    }
    if (op_2d) {
      mlir::Value resh =
          builder.create<mlir::stablehlo::ReshapeOp>(uloc, out_t, op_2d);
      user->getResult(0).replaceAllUsesWith(resh);
    }
  }
}

bool RewriteBackwardAndPaddedPatchifyConvsToDot(mlir::func::FuncOp func) {
  bool any_changed = false;
  llvm::SmallVector<mlir::stablehlo::ConvolutionOp, 16> conv_ops;
  func.walk([&](mlir::stablehlo::ConvolutionOp op) { conv_ops.push_back(op); });

  // Map from input tensor `X` (`conv.getLhs()`) to a paired forward patchify
  // ConvolutionOp in the same block so forward and backward share a single
  // `X_2D` patch matrix.
  llvm::DenseMap<mlir::Value, mlir::stablehlo::ConvolutionOp> fwd_by_lhs;
  llvm::SmallVector<mlir::stablehlo::ConvolutionOp, 8> padded_fwd_convs;
  for (mlir::stablehlo::ConvolutionOp conv : conv_ops) {
    llvm::SmallVector<int64_t, 4> grid_dims;
    llvm::SmallVector<int64_t, 4> kernel_dims;
    llvm::SmallVector<int64_t, 4> pad_lo;
    bool has_non_zero_pad = false;
    if (MatchForwardPatchifyConv(conv, grid_dims, kernel_dims, pad_lo,
                                 has_non_zero_pad)) {
      fwd_by_lhs[conv.getLhs()] = conv;
      if (has_non_zero_pad) {
        padded_fwd_convs.push_back(conv);
      }
    }
  }

  llvm::DenseMap<mlir::Operation*, mlir::Value> lowered_fwd_x2d;

  for (mlir::stablehlo::ConvolutionOp conv_bwd : conv_ops) {
    llvm::SmallVector<int64_t, 4> grid_dims;
    llvm::SmallVector<int64_t, 4> kernel_dims;
    llvm::SmallVector<int64_t, 4> pad_lo;
    bool has_non_zero_pad = false;
    if (!MatchBackwardPatchifyWeightGradConv(conv_bwd, grid_dims, kernel_dims,
                                             pad_lo, has_non_zero_pad)) {
      continue;
    }

    auto lhs_t =
        llvm::dyn_cast<mlir::RankedTensorType>(conv_bwd.getLhs().getType());
    auto out_t = llvm::dyn_cast<mlir::RankedTensorType>(conv_bwd.getType());
    mlir::Type elem_t = out_t.getElementType();
    const int64_t b = lhs_t.getDimSize(0);
    const int64_t c_in = lhs_t.getDimSize(1);
    const int64_t c_out = out_t.getDimSize(0);
    const int64_t n_sp = grid_dims.size();
    int64_t num_patches_per_batch = 1;
    int64_t patch_area = 1;
    for (int64_t i = 0; i < n_sp; ++i) {
      num_patches_per_batch *= grid_dims[i];
      patch_area *= kernel_dims[i];
    }
    const int64_t m_total = b * num_patches_per_batch;
    const int64_t k_total = c_in * patch_area;

    // Check if there is a paired forward patchify convolution on the same input
    // tensor in the same block so `x_2d` is emitted once at the earlier op and
    // shared by both forward and backward.
    mlir::stablehlo::ConvolutionOp paired_fwd;
    auto it = fwd_by_lhs.find(conv_bwd.getLhs());
    if (it != fwd_by_lhs.end() && it->second &&
        it->second->getBlock() == conv_bwd->getBlock()) {
      llvm::SmallVector<int64_t, 4> f_grid;
      llvm::SmallVector<int64_t, 4> f_ker;
      llvm::SmallVector<int64_t, 4> f_pad_lo;
      bool f_non_zero = false;
      if (MatchForwardPatchifyConv(it->second, f_grid, f_ker, f_pad_lo,
                                   f_non_zero) &&
          f_grid == grid_dims && f_ker == kernel_dims && f_pad_lo == pad_lo) {
        paired_fwd = it->second;
      }
    }

    mlir::Operation* insert_pt = conv_bwd.getOperation();
    if (paired_fwd && paired_fwd->isBeforeInBlock(conv_bwd.getOperation())) {
      insert_pt = paired_fwd.getOperation();
    }
    mlir::OpBuilder x_builder(insert_pt);
    mlir::Value x_2d = BuildPatchMatrix2D(
        x_builder, insert_pt->getLoc(), conv_bwd.getLhs(), b, c_in, grid_dims,
        kernel_dims, pad_lo, has_non_zero_pad, m_total, k_total, elem_t);
    if (!x_2d) {
      continue;
    }

    if (paired_fwd && !lowered_fwd_x2d.count(paired_fwd.getOperation())) {
      auto fwd_out_t =
          llvm::dyn_cast<mlir::RankedTensorType>(paired_fwd.getType());
      LowerForwardConvWithPatchMatrix(paired_fwd, x_2d, b,
                                      num_patches_per_batch, m_total, k_total,
                                      c_out, elem_t, fwd_out_t);
      lowered_fwd_x2d[paired_fwd.getOperation()] = x_2d;
    }

    mlir::OpBuilder bwd_builder(conv_bwd);
    mlir::Location bwd_loc = conv_bwd.getLoc();
    mlir::Value dy_2d = BuildGradOutput2DAndFoldBiasReduce(
        bwd_builder, bwd_loc, conv_bwd.getRhs(), m_total, c_out, n_sp, elem_t);

    auto dot_dnums = mlir::stablehlo::DotDimensionNumbersAttr::get(
        bwd_builder.getContext(), /*lhsBatchingDimensions=*/{},
        /*rhsBatchingDimensions=*/{}, /*lhsContractingDimensions=*/{0},
        /*rhsContractingDimensions=*/{0});
    auto dot_2d_type = mlir::RankedTensorType::get({c_out, k_total}, elem_t);
    mlir::Value bwd_dot = bwd_builder.create<mlir::stablehlo::DotGeneralOp>(
        bwd_loc, dot_2d_type, dy_2d, x_2d, dot_dnums,
        conv_bwd.getPrecisionConfigAttr(),
        /*algorithm=*/mlir::stablehlo::DotAlgorithmAttr{});

    mlir::Value out_4d =
        EmitReshapeIfNeeded(bwd_builder, bwd_loc, bwd_dot, out_t);
    conv_bwd.replaceAllUsesWith(out_4d);
    if (out_4d != bwd_dot) {
      SinkBackwardDotReshapeThroughElementwiseUsers(bwd_dot, out_4d,
                                                    dot_2d_type, out_t);
    }
    any_changed = true;
  }

  // Also lower standalone padded forward patchify convolutions (where
  // `has_non_zero_pad` is true) that had no backward convolution in the same
  // function.
  for (mlir::stablehlo::ConvolutionOp conv_fwd : padded_fwd_convs) {
    if (!conv_fwd || !conv_fwd->getBlock() || conv_fwd->use_empty() ||
        lowered_fwd_x2d.count(conv_fwd.getOperation())) {
      continue;
    }
    llvm::SmallVector<int64_t, 4> grid_dims;
    llvm::SmallVector<int64_t, 4> kernel_dims;
    llvm::SmallVector<int64_t, 4> pad_lo;
    bool has_non_zero_pad = false;
    if (!MatchForwardPatchifyConv(conv_fwd, grid_dims, kernel_dims, pad_lo,
                                  has_non_zero_pad)) {
      continue;
    }
    auto lhs_t =
        llvm::dyn_cast<mlir::RankedTensorType>(conv_fwd.getLhs().getType());
    auto out_t = llvm::dyn_cast<mlir::RankedTensorType>(conv_fwd.getType());
    mlir::Type elem_t = out_t.getElementType();
    const int64_t b = lhs_t.getDimSize(0);
    const int64_t c_in = lhs_t.getDimSize(1);
    const int64_t c_out = out_t.getDimSize(1);
    int64_t num_patches_per_batch = 1;
    int64_t patch_area = 1;
    for (size_t i = 0; i < grid_dims.size(); ++i) {
      num_patches_per_batch *= grid_dims[i];
      patch_area *= kernel_dims[i];
    }
    const int64_t m_total = b * num_patches_per_batch;
    const int64_t k_total = c_in * patch_area;

    mlir::OpBuilder builder(conv_fwd);
    mlir::Value x_2d = BuildPatchMatrix2D(
        builder, conv_fwd.getLoc(), conv_fwd.getLhs(), b, c_in, grid_dims,
        kernel_dims, pad_lo, has_non_zero_pad, m_total, k_total, elem_t);
    if (!x_2d) {
      continue;
    }
    LowerForwardConvWithPatchMatrix(conv_fwd, x_2d, b, num_patches_per_batch,
                                    m_total, k_total, c_out, elem_t, out_t);
    lowered_fwd_x2d[conv_fwd.getOperation()] = x_2d;
    any_changed = true;
  }

  if (any_changed) {
    EraseTriviallyDeadOps(func);
  }
  return any_changed;
}

class LowerPatchifyConvBackwardToDotPass
    : public mlir::PassWrapper<LowerPatchifyConvBackwardToDotPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LowerPatchifyConvBackwardToDotPass)

  llvm::StringRef getArgument() const final {
    return "lower-patchify-conv-backward-to-dot";
  }

  llvm::StringRef getDescription() const final {
    return "Lowers non-overlapping 2D/3D patchify backward weight-gradient "
           "convolutions (sharing the extracted patch matrix with any paired "
           "forward patchify convolution) and padded non-overlapping forward "
           "patchify convolutions to 2D DotGeneralOp contractions.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    if (RewriteBackwardAndPaddedPatchifyConvsToDot(func)) {
      CancelPatchifyDotOutputTransposes(func);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createLowerPatchifyConvBackwardToDotPass() {
  return std::make_unique<LowerPatchifyConvBackwardToDotPass>();
}

}  // namespace torch_tpu
