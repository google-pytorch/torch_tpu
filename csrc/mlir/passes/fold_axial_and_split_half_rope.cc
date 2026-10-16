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

#include "csrc/mlir/passes/fold_axial_and_split_half_rope.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
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

bool IsDotOrDotGeneral(mlir::Operation* op) {
  return op && (mlir::isa<mlir::stablehlo::DotOp>(op) ||
                mlir::isa<mlir::stablehlo::DotGeneralOp>(op));
}

bool Is0213Permutation(llvm::ArrayRef<int64_t> perm) {
  return perm.size() == 4 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1 &&
         perm[3] == 3;
}

bool Is3102Permutation(llvm::ArrayRef<int64_t> perm) {
  return perm.size() == 4 && perm[0] == 3 && perm[1] == 1 && perm[2] == 0 &&
         perm[3] == 2;
}

llvm::SmallVector<int64_t> GetNonUnitDimSizes(llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t> sizes;
  for (int64_t d : shape) {
    if (d > 1) {
      sizes.push_back(d);
    }
  }
  return sizes;
}

mlir::Value EmitReshapeIfNeeded(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value val,
                                mlir::RankedTensorType dst_type) {
  if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    auto op_type =
        mlir::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
    if (op_type && op_type.hasStaticShape() &&
        op_type.getNumElements() == dst_type.getNumElements()) {
      val = resh.getOperand();
    }
  }
  if (val.getType() == dst_type) {
    return val;
  }
  return builder.create<mlir::stablehlo::ReshapeOp>(loc, dst_type, val);
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  for (mlir::Block& block : llvm::reverse(func.getBlocks())) {
    for (mlir::Operation& op :
         llvm::make_early_inc_range(llvm::reverse(block))) {
      if (&op != func.getOperation() && op.use_empty() &&
          mlir::isOpTriviallyDead(&op)) {
        op.erase();
      }
    }
  }
}

// Checks whether `val` is a 4D `BroadcastInDimOp` into `[B, S, H, D]` that
// broadcasts across the head dimension `dim 2` (i.e. `dim 1` and `dim 3` are in
// `broadcast_dimensions` and `dim 2` is not in `broadcast_dimensions`).
bool IsBshdHeadBroadcastTrig4D(mlir::Value val) {
  auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast) {
    return false;
  }
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
  if (!out_type || !out_type.hasStaticShape() || out_type.getRank() != 4) {
    return false;
  }
  llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
  bool has_dim1 = false;
  bool has_dim2 = false;
  bool has_dim3 = false;
  for (int64_t d : bdims) {
    if (d == 1) {
      has_dim1 = true;
    } else if (d == 2) {
      has_dim2 = true;
    } else if (d == 3) {
      has_dim3 = true;
    }
  }
  return has_dim1 && !has_dim2 && has_dim3;
}

enum class HeadOrderKind {
  kNone,      // Already in target 4D layout (no transpose).
  kBSHD0213,  // Reshape([B, S, H, D]) -> Transpose({0, 2, 1, 3}) -> [B, H, S,
              // D]
  kSHD13102,  // Reshape([S, H, D, 1]) -> Transpose({3, 1, 0, 2}) -> [1, H, S,
              // D]
};

// Describes a contiguous minor-dimension slice from a rotary input tensor.
struct ContiguousSliceDesc {
  mlir::Value root_src;
  HeadOrderKind order = HeadOrderKind::kNone;
  int64_t d_full = 0;
  int64_t start = 0;
  int64_t limit = 0;
};

// Matches a contiguous minor-dimension slice `[start : limit]` producing a 4D
// tensor of shape `[B, D1, D2, w]`, either sliced directly from a 4D tensor
// `[B, D1, D2, D_full]` (`order == kNone`) or sliced before a head-split
// transpose (`kBSHD0213` or `kSHD13102`).
bool MatchContiguousHeadSlice4D(mlir::Value val, int64_t expected_w,
                                ContiguousSliceDesc& desc) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!out_type || !out_type.hasStaticShape() || out_type.getRank() != 4 ||
      out_type.getDimSize(3) != expected_w) {
    return false;
  }
  const int64_t b = out_type.getDimSize(0);
  const int64_t d1 = out_type.getDimSize(1);
  const int64_t d2 = out_type.getDimSize(2);

  // Case 1: Direct 4D slice on `[B, D1, D2, D_full]`.
  if (auto slice_op = val.getDefiningOp<mlir::stablehlo::SliceOp>()) {
    auto in_type =
        mlir::dyn_cast<mlir::RankedTensorType>(slice_op.getOperand().getType());
    if (in_type && in_type.hasStaticShape() && in_type.getRank() == 4 &&
        in_type.getDimSize(0) == b && in_type.getDimSize(1) == d1 &&
        in_type.getDimSize(2) == d2) {
      llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
      llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
      llvm::ArrayRef<int64_t> strides = slice_op.getStrides();
      if (starts[0] == 0 && limits[0] == b && strides[0] == 1 &&
          starts[1] == 0 && limits[1] == d1 && strides[1] == 1 &&
          starts[2] == 0 && limits[2] == d2 && strides[2] == 1 &&
          strides[3] == 1 && limits[3] - starts[3] == expected_w) {
        desc = {slice_op.getOperand(), HeadOrderKind::kNone,
                in_type.getDimSize(3), starts[3], limits[3]};
        return true;
      }
    }
  }

  auto tr_op = val.getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!tr_op) {
    return false;
  }
  llvm::ArrayRef<int64_t> perm = tr_op.getPermutation();

  // Case 2: `Transpose(Reshape(Slice(Reshape(root, [S, H, D_full]), ...),
  // [S, H, w, 1]), {3, 1, 0, 2})` with `b == 1, h == d1, s == d2`.
  if (b == 1 && Is3102Permutation(perm)) {
    const int64_t h = d1;
    const int64_t s = d2;
    auto r4 = tr_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!r4) {
      return false;
    }
    auto r4_type = mlir::dyn_cast<mlir::RankedTensorType>(r4.getType());
    if (!r4_type ||
        r4_type.getShape() != llvm::ArrayRef<int64_t>({s, h, expected_w, 1})) {
      return false;
    }
    auto slice_op = r4.getOperand().getDefiningOp<mlir::stablehlo::SliceOp>();
    if (!slice_op) {
      return false;
    }
    auto r3 = slice_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!r3) {
      return false;
    }
    auto r3_type = mlir::dyn_cast<mlir::RankedTensorType>(r3.getType());
    if (!r3_type || !r3_type.hasStaticShape() || r3_type.getRank() != 3 ||
        r3_type.getDimSize(0) != s || r3_type.getDimSize(1) != h) {
      return false;
    }
    llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
    llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
    llvm::ArrayRef<int64_t> strides = slice_op.getStrides();
    if (starts[0] != 0 || limits[0] != s || strides[0] != 1 || starts[1] != 0 ||
        limits[1] != h || strides[1] != 1 || strides[2] != 1 ||
        limits[2] - starts[2] != expected_w) {
      return false;
    }
    desc = {r3.getOperand(), HeadOrderKind::kSHD13102, r3_type.getDimSize(2),
            starts[2], limits[2]};
    return true;
  }

  // Case 3: `Transpose(Slice(Reshape(root, [B, S, H, D_full]), ...),
  // {0, 2, 1, 3})` with `h == d1, s == d2`.
  if (Is0213Permutation(perm)) {
    const int64_t h = d1;
    const int64_t s = d2;
    auto slice_op =
        tr_op.getOperand().getDefiningOp<mlir::stablehlo::SliceOp>();
    if (!slice_op) {
      return false;
    }
    auto r4 = slice_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!r4) {
      return false;
    }
    auto r4_type = mlir::dyn_cast<mlir::RankedTensorType>(r4.getType());
    if (!r4_type || !r4_type.hasStaticShape() || r4_type.getRank() != 4 ||
        r4_type.getDimSize(0) != b || r4_type.getDimSize(1) != s ||
        r4_type.getDimSize(2) != h) {
      return false;
    }
    llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
    llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
    llvm::ArrayRef<int64_t> strides = slice_op.getStrides();
    if (starts[0] != 0 || limits[0] != b || strides[0] != 1 || starts[1] != 0 ||
        limits[1] != s || strides[1] != 1 || starts[2] != 0 || limits[2] != h ||
        strides[2] != 1 || strides[3] != 1 ||
        limits[3] - starts[3] != expected_w) {
      return false;
    }
    desc = {r4.getOperand(), HeadOrderKind::kBSHD0213, r4_type.getDimSize(3),
            starts[3], limits[3]};
    return true;
  }

  return false;
}

// Matches a stride-2 even (`parity == 0`) or odd (`parity == 1`) slice
// covering pairs `[pair_start : pair_limit]` of `root_src`:
// `Transpose(Slice(Reshape(root_src, [S, H, D_full/2, 2]), [0:S, 0:H,
// pair_start:pair_limit, parity:parity+1]), {3, 1, 0, 2})` -> `[1, H, S, w/2]`.
bool MatchPairwiseParityHeadSlice4D(mlir::Value val, int64_t expected_half_w,
                                    int64_t parity, ContiguousSliceDesc& desc) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!out_type || !out_type.hasStaticShape() || out_type.getRank() != 4 ||
      out_type.getDimSize(0) != 1 ||
      out_type.getDimSize(3) != expected_half_w) {
    return false;
  }
  const int64_t h = out_type.getDimSize(1);
  const int64_t s = out_type.getDimSize(2);
  auto tr_op = val.getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!tr_op || !Is3102Permutation(tr_op.getPermutation())) {
    return false;
  }
  auto slice_op = tr_op.getOperand().getDefiningOp<mlir::stablehlo::SliceOp>();
  if (!slice_op) {
    return false;
  }
  auto r4 = slice_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!r4) {
    return false;
  }
  auto r4_type = mlir::dyn_cast<mlir::RankedTensorType>(r4.getType());
  if (!r4_type || !r4_type.hasStaticShape() || r4_type.getRank() != 4 ||
      r4_type.getDimSize(0) != s || r4_type.getDimSize(1) != h ||
      r4_type.getDimSize(3) != 2) {
    return false;
  }
  llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
  llvm::ArrayRef<int64_t> strides = slice_op.getStrides();
  if (starts[0] != 0 || limits[0] != s || strides[0] != 1 || starts[1] != 0 ||
      limits[1] != h || strides[1] != 1 || strides[2] != 1 ||
      limits[2] - starts[2] != expected_half_w || starts[3] != parity ||
      limits[3] != parity + 1 || strides[3] != 1) {
    return false;
  }
  desc = {r4.getOperand(), HeadOrderKind::kSHD13102, r4_type.getDimSize(2) * 2,
          starts[2] * 2, limits[2] * 2};
  return true;
}

// Matches `Reshape(Concatenate(Reshape(Neg(odd_slice), [..., w/2, 1]),
// Reshape(even_slice, [..., w/2, 1]), dim=4), [1, H, S, w])`.
bool MatchPairwiseRotatedBlock4D(mlir::Value val, int64_t expected_w,
                                 ContiguousSliceDesc& desc) {
  if (expected_w < 2 || expected_w % 2 != 0) {
    return false;
  }
  auto resh_out = val.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!resh_out) {
    return false;
  }
  auto cat_op =
      resh_out.getOperand().getDefiningOp<mlir::stablehlo::ConcatenateOp>();
  if (!cat_op || cat_op.getInputs().size() != 2 || cat_op.getDimension() != 4) {
    return false;
  }
  auto r_odd =
      cat_op.getInputs()[0].getDefiningOp<mlir::stablehlo::ReshapeOp>();
  auto r_even =
      cat_op.getInputs()[1].getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!r_odd || !r_even) {
    return false;
  }
  auto neg_odd = r_odd.getOperand().getDefiningOp<mlir::stablehlo::NegOp>();
  if (!neg_odd) {
    return false;
  }
  ContiguousSliceDesc odd_desc;
  ContiguousSliceDesc even_desc;
  if (!MatchPairwiseParityHeadSlice4D(neg_odd.getOperand(), expected_w / 2,
                                      /*parity=*/1, odd_desc) ||
      !MatchPairwiseParityHeadSlice4D(r_even.getOperand(), expected_w / 2,
                                      /*parity=*/0, even_desc)) {
    return false;
  }
  if (odd_desc.root_src != even_desc.root_src ||
      odd_desc.order != even_desc.order ||
      odd_desc.d_full != even_desc.d_full ||
      odd_desc.start != even_desc.start || odd_desc.limit != even_desc.limit) {
    return false;
  }
  desc = odd_desc;
  return true;
}

// Matches `Concatenate(Neg(slice_hi), slice_lo, dim=3)` of width `expected_w`
// covering `[start : start + expected_w]` of `root_src`.
bool MatchHalfRotatedBlock4D(mlir::Value val, int64_t expected_w,
                             ContiguousSliceDesc& desc) {
  if (expected_w < 2 || expected_w % 2 != 0) {
    return false;
  }
  const int64_t half_w = expected_w / 2;
  auto cat_op = val.getDefiningOp<mlir::stablehlo::ConcatenateOp>();
  if (!cat_op || cat_op.getInputs().size() != 2 || cat_op.getDimension() != 3) {
    return false;
  }
  auto neg_hi = cat_op.getInputs()[0].getDefiningOp<mlir::stablehlo::NegOp>();
  if (!neg_hi) {
    return false;
  }
  ContiguousSliceDesc hi_desc;
  ContiguousSliceDesc lo_desc;
  if (!MatchContiguousHeadSlice4D(neg_hi.getOperand(), half_w, hi_desc) ||
      !MatchContiguousHeadSlice4D(cat_op.getInputs()[1], half_w, lo_desc)) {
    return false;
  }
  if (hi_desc.root_src != lo_desc.root_src || hi_desc.order != lo_desc.order ||
      hi_desc.d_full != lo_desc.d_full || lo_desc.limit != hi_desc.start ||
      hi_desc.limit - lo_desc.start != expected_w) {
    return false;
  }
  desc = {lo_desc.root_src, lo_desc.order, lo_desc.d_full, lo_desc.start,
          hi_desc.limit};
  return true;
}

enum class AxialBlockKind {
  kPairwiseRotary,
  kHalfSplitRotary,
  kPassThrough,
};

// Describes a single rotary or pass-through slice along the head dimension.
struct AxialBlockDesc {
  AxialBlockKind kind = AxialBlockKind::kPassThrough;
  int64_t start = 0;
  int64_t limit = 0;
  mlir::Value cos_val;
  mlir::Value sin_val;
};

bool MatchRotaryAddBlock4D(mlir::Value block_val,
                           ContiguousSliceDesc& out_slice_desc,
                           AxialBlockDesc& out_block) {
  auto block_type = mlir::dyn_cast<mlir::RankedTensorType>(block_val.getType());
  if (!block_type || !block_type.hasStaticShape() ||
      block_type.getRank() != 4) {
    return false;
  }
  const int64_t w = block_type.getDimSize(3);

  // Check if this block is an unrotated pass-through slice.
  if (MatchContiguousHeadSlice4D(block_val, w, out_slice_desc)) {
    out_block = {AxialBlockKind::kPassThrough, out_slice_desc.start,
                 out_slice_desc.limit, nullptr, nullptr};
    return true;
  }

  auto add_op = block_val.getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add_op) {
    return false;
  }
  auto mul_a = add_op.getLhs().getDefiningOp<mlir::stablehlo::MulOp>();
  auto mul_b = add_op.getRhs().getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul_a || !mul_b) {
    return false;
  }

  auto try_match_pair = [&](mlir::stablehlo::MulOp cos_mul,
                            mlir::stablehlo::MulOp sin_mul) -> bool {
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
      mlir::Value x_part = c_idx == 0 ? cos_mul.getLhs() : cos_mul.getRhs();
      mlir::Value cos_part = c_idx == 0 ? cos_mul.getRhs() : cos_mul.getLhs();
      ContiguousSliceDesc base_desc;
      if (!MatchContiguousHeadSlice4D(x_part, w, base_desc)) {
        continue;
      }
      for (int s_idx = 0; s_idx < 2; ++s_idx) {
        mlir::Value rot_part = s_idx == 0 ? sin_mul.getLhs() : sin_mul.getRhs();
        mlir::Value sin_part = s_idx == 0 ? sin_mul.getRhs() : sin_mul.getLhs();
        ContiguousSliceDesc rot_desc;
        if (MatchPairwiseRotatedBlock4D(rot_part, w, rot_desc) &&
            rot_desc.root_src == base_desc.root_src &&
            rot_desc.order == base_desc.order &&
            rot_desc.d_full == base_desc.d_full &&
            rot_desc.start == base_desc.start &&
            rot_desc.limit == base_desc.limit) {
          out_slice_desc = base_desc;
          out_block = {AxialBlockKind::kPairwiseRotary, base_desc.start,
                       base_desc.limit, cos_part, sin_part};
          return true;
        }
        if (MatchHalfRotatedBlock4D(rot_part, w, rot_desc) &&
            rot_desc.root_src == base_desc.root_src &&
            rot_desc.order == base_desc.order &&
            rot_desc.d_full == base_desc.d_full &&
            rot_desc.start == base_desc.start &&
            rot_desc.limit == base_desc.limit) {
          out_slice_desc = base_desc;
          out_block = {AxialBlockKind::kHalfSplitRotary, base_desc.start,
                       base_desc.limit, cos_part, sin_part};
          return true;
        }
      }
    }
    return false;
  };

  return try_match_pair(mul_a, mul_b) || try_match_pair(mul_b, mul_a);
}

mlir::Value BuildFullHeadTensor4D(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::Value root_src, HeadOrderKind order,
                                  mlir::RankedTensorType out_type) {
  if (order == HeadOrderKind::kNone) {
    return EmitReshapeIfNeeded(builder, loc, root_src, out_type);
  }
  const int64_t b = out_type.getDimSize(0);
  const int64_t h = out_type.getDimSize(1);
  const int64_t s = out_type.getDimSize(2);
  const int64_t d = out_type.getDimSize(3);
  if (order == HeadOrderKind::kSHD13102) {
    auto r4_type =
        mlir::RankedTensorType::get({s, h, d, 1}, out_type.getElementType());
    mlir::Value r4 = EmitReshapeIfNeeded(builder, loc, root_src, r4_type);
    return builder.create<mlir::stablehlo::TransposeOp>(
        loc, out_type, r4, builder.getDenseI64ArrayAttr({3, 1, 0, 2}));
  }
  auto r4_type =
      mlir::RankedTensorType::get({b, s, h, d}, out_type.getElementType());
  mlir::Value r4 = EmitReshapeIfNeeded(builder, loc, root_src, r4_type);
  return builder.create<mlir::stablehlo::TransposeOp>(
      loc, out_type, r4, builder.getDenseI64ArrayAttr({0, 2, 1, 3}));
}

// Peels redundant `Reshape(BroadcastInDim(Reshape(x)))` chains on a 1D/2D/3D
// rotary `cos`/`sin` tensor before its final 4D `BroadcastInDim`.
mlir::Value CanonicalizeRotaryTrigSource(mlir::OpBuilder& builder,
                                         mlir::Location loc, mlir::Value src) {
  auto src_type = mlir::dyn_cast<mlir::RankedTensorType>(src.getType());
  if (!src_type || !src_type.hasStaticShape()) {
    return src;
  }
  mlir::Value cur = src;
  while (true) {
    if (auto r = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      auto in_t =
          mlir::dyn_cast<mlir::RankedTensorType>(r.getOperand().getType());
      if (in_t && in_t.hasStaticShape() &&
          GetNonUnitDimSizes(in_t.getShape()) ==
              GetNonUnitDimSizes(src_type.getShape())) {
        cur = r.getOperand();
        continue;
      }
    }
    if (auto b = cur.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      if (!llvm::is_sorted(b.getBroadcastDimensions())) {
        break;
      }
      auto in_t =
          mlir::dyn_cast<mlir::RankedTensorType>(b.getOperand().getType());
      auto out_t = mlir::dyn_cast<mlir::RankedTensorType>(b.getType());
      if (in_t && out_t && in_t.hasStaticShape() && out_t.hasStaticShape() &&
          in_t.getNumElements() == out_t.getNumElements() &&
          GetNonUnitDimSizes(in_t.getShape()) ==
              GetNonUnitDimSizes(src_type.getShape())) {
        cur = b.getOperand();
        continue;
      }
    }
    break;
  }
  if (cur == src) {
    return src;
  }
  return EmitReshapeIfNeeded(builder, loc, cur, src_type);
}

void SetInsertionPointAfterLatestOperand(mlir::OpBuilder& builder,
                                         mlir::Operation* fallback_op,
                                         llvm::ArrayRef<mlir::Value> vals) {
  mlir::Operation* latest_op = nullptr;
  mlir::Block* target_block = fallback_op->getBlock();
  for (mlir::Value v : vals) {
    if (!v) {
      continue;
    }
    if (mlir::Operation* def = v.getDefiningOp()) {
      if (def->getBlock() == target_block) {
        if (!latest_op || latest_op->isBeforeInBlock(def)) {
          latest_op = def;
        }
      }
    }
  }
  if (latest_op) {
    builder.setInsertionPointAfter(latest_op);
  } else {
    builder.setInsertionPoint(fallback_op);
  }
}

mlir::Value BuildCombinedRotaryTrig4D(mlir::OpBuilder& builder,
                                      mlir::Operation* anchor_op,
                                      mlir::Location loc,
                                      mlir::RankedTensorType out_type,
                                      llvm::ArrayRef<AxialBlockDesc> blocks,
                                      bool is_cos) {
  mlir::Type elem_type = out_type.getElementType();
  auto float_type = mlir::cast<mlir::FloatType>(elem_type);
  llvm::APFloat fill_val(float_type.getFloatSemantics(), is_cos ? 1 : 0);
  auto scalar_type = mlir::RankedTensorType::get({}, float_type);

  bool can_hoist_bcast = true;
  llvm::SmallVector<int64_t> common_bdims;
  llvm::SmallVector<int64_t> common_leading_shape;
  llvm::SmallVector<mlir::Value> raw_sources;

  for (const AxialBlockDesc& blk : blocks) {
    if (blk.kind == AxialBlockKind::kPassThrough) {
      continue;
    }
    mlir::Value v = is_cos ? blk.cos_val : blk.sin_val;
    auto bcast = v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!bcast) {
      can_hoist_bcast = false;
      break;
    }
    llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
    auto src_t =
        mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
    if (!src_t || !src_t.hasStaticShape() || src_t.getRank() < 1 ||
        bdims.empty() || bdims.back() != 3) {
      can_hoist_bcast = false;
      break;
    }
    llvm::SmallVector<int64_t> leading(src_t.getShape().drop_back(1).begin(),
                                       src_t.getShape().drop_back(1).end());
    if (common_bdims.empty()) {
      common_bdims.assign(bdims.begin(), bdims.end());
      common_leading_shape = leading;
    } else if (common_bdims != bdims || common_leading_shape != leading) {
      can_hoist_bcast = false;
      break;
    }
    raw_sources.push_back(bcast.getOperand());
  }

  if (can_hoist_bcast && !common_bdims.empty()) {
    mlir::OpBuilder hoisted_builder(anchor_op);
    SetInsertionPointAfterLatestOperand(hoisted_builder, anchor_op,
                                        raw_sources);
    llvm::SmallVector<mlir::Value> canon_sources;
    for (mlir::Value s : raw_sources) {
      canon_sources.push_back(
          CanonicalizeRotaryTrigSource(hoisted_builder, loc, s));
    }
    SetInsertionPointAfterLatestOperand(hoisted_builder, anchor_op,
                                        canon_sources);

    llvm::SmallVector<mlir::Value> cat_parts;
    size_t src_idx = 0;
    const int64_t src_rank = common_leading_shape.size() + 1;
    for (const AxialBlockDesc& blk : blocks) {
      const int64_t w = blk.limit - blk.start;
      if (blk.kind == AxialBlockKind::kPassThrough) {
        llvm::SmallVector<int64_t> pass_shape = common_leading_shape;
        pass_shape.push_back(w);
        auto pass_type = mlir::RankedTensorType::get(pass_shape, float_type);
        mlir::Value c_scalar =
            hoisted_builder.create<mlir::stablehlo::ConstantOp>(
                loc, scalar_type,
                mlir::DenseElementsAttr::get(scalar_type, fill_val));
        mlir::Value pass_bcast =
            hoisted_builder.create<mlir::stablehlo::BroadcastInDimOp>(
                loc, pass_type, c_scalar,
                hoisted_builder.getDenseI64ArrayAttr({}));
        cat_parts.push_back(pass_bcast);
      } else {
        cat_parts.push_back(canon_sources[src_idx++]);
      }
    }
    llvm::SmallVector<int64_t> full_src_shape = common_leading_shape;
    full_src_shape.push_back(out_type.getDimSize(3));
    auto full_src_type =
        mlir::RankedTensorType::get(full_src_shape, float_type);
    mlir::Value cat_src =
        hoisted_builder.create<mlir::stablehlo::ConcatenateOp>(
            loc, full_src_type, cat_parts,
            hoisted_builder.getI64IntegerAttr(src_rank - 1));
    return hoisted_builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, out_type, cat_src,
        hoisted_builder.getDenseI64ArrayAttr(common_bdims));
  }

  llvm::SmallVector<mlir::Value> parts_4d;
  for (const AxialBlockDesc& blk : blocks) {
    const int64_t w = blk.limit - blk.start;
    if (blk.kind == AxialBlockKind::kPassThrough) {
      auto pass_type = mlir::RankedTensorType::get(
          {out_type.getDimSize(0), out_type.getDimSize(1),
           out_type.getDimSize(2), w},
          float_type);
      mlir::Value c_scalar = builder.create<mlir::stablehlo::ConstantOp>(
          loc, scalar_type,
          mlir::DenseElementsAttr::get(scalar_type, fill_val));
      parts_4d.push_back(builder.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, pass_type, c_scalar, builder.getDenseI64ArrayAttr({})));
    } else {
      parts_4d.push_back(is_cos ? blk.cos_val : blk.sin_val);
    }
  }
  return builder.create<mlir::stablehlo::ConcatenateOp>(
      loc, out_type, parts_4d, builder.getI64IntegerAttr(3));
}

mlir::Value BuildAxialRotationMatrix(mlir::OpBuilder& builder,
                                     mlir::Operation* anchor_op,
                                     mlir::Location loc, mlir::Type elem_type,
                                     int64_t d_full,
                                     llvm::ArrayRef<AxialBlockDesc> blocks) {
  mlir::OpBuilder top_builder(anchor_op);
  top_builder.setInsertionPointToStart(anchor_op->getBlock());
  auto float_type = mlir::cast<mlir::FloatType>(elem_type);
  llvm::APFloat zero(float_type.getFloatSemantics(), 0);
  llvm::APFloat pos_one(float_type.getFloatSemantics(), 1);
  llvm::APFloat neg_one = pos_one;
  neg_one.changeSign();

  llvm::SmallVector<llvm::APFloat> vals(d_full * d_full, zero);
  for (const AxialBlockDesc& blk : blocks) {
    const int64_t w = blk.limit - blk.start;
    if (blk.kind == AxialBlockKind::kPairwiseRotary) {
      for (int64_t p = 0; p < w / 2; ++p) {
        const int64_t c0 = blk.start + 2 * p;
        const int64_t c1 = blk.start + 2 * p + 1;
        vals[c1 * d_full + c0] = neg_one;
        vals[c0 * d_full + c1] = pos_one;
      }
    } else if (blk.kind == AxialBlockKind::kHalfSplitRotary) {
      const int64_t half_w = w / 2;
      for (int64_t m = 0; m < half_w; ++m) {
        const int64_t c_lo = blk.start + m;
        const int64_t c_hi = blk.start + half_w + m;
        vals[c_hi * d_full + c_lo] = neg_one;
        vals[c_lo * d_full + c_hi] = pos_one;
      }
    }
  }
  auto mat_type = mlir::RankedTensorType::get({d_full, d_full}, float_type);
  return top_builder.create<mlir::stablehlo::ConstantOp>(
      loc, mlir::DenseElementsAttr::get(mat_type, vals));
}

mlir::stablehlo::DotGeneralOp EmitMinorDimContraction(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::RankedTensorType out_type, mlir::Value x, mlir::Value mat) {
  const int64_t rank = out_type.getRank();
  auto dnums = mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(), /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{}, /*lhsContractingDimensions=*/{rank - 1},
      /*rhsContractingDimensions=*/{0});
  return builder.create<mlir::stablehlo::DotGeneralOp>(
      loc, out_type, x, mat, dnums, /*precision_config=*/nullptr,
      /*algorithm=*/nullptr);
}

// Family 1: Multi-block axial and partial RoPE concatenation along dimension 3
// (`Concatenate(rot_0, rot_1, [rot_2, pass_3], dim=3)`).
bool TryFoldMultiBlockAxialRotaryConcat(
    mlir::stablehlo::ConcatenateOp concat_op,
    llvm::SmallVectorImpl<mlir::stablehlo::DotGeneralOp>& created_rope_dots) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(concat_op.getType());
  if (!out_type || !out_type.hasStaticShape() || out_type.getRank() != 4 ||
      concat_op.getDimension() != 3 ||
      !mlir::isa<mlir::FloatType>(out_type.getElementType())) {
    return false;
  }
  const size_t num_inputs = concat_op.getInputs().size();
  if (num_inputs < 2) {
    return false;
  }
  const int64_t d_full = out_type.getDimSize(3);
  if (d_full <= 1) {
    return false;
  }

  mlir::Value common_root;
  HeadOrderKind common_order = HeadOrderKind::kNone;
  int64_t expected_offset = 0;
  int64_t num_rotary_blocks = 0;
  bool all_rotary_bshd_bcast = true;
  llvm::SmallVector<AxialBlockDesc, 4> blocks;

  for (size_t i = 0; i < num_inputs; ++i) {
    ContiguousSliceDesc slice_desc;
    AxialBlockDesc blk_desc;
    if (!MatchRotaryAddBlock4D(concat_op.getInputs()[i], slice_desc,
                               blk_desc)) {
      return false;
    }
    if (slice_desc.d_full != d_full || slice_desc.start != expected_offset) {
      return false;
    }
    if (i == 0) {
      common_root = slice_desc.root_src;
      common_order = slice_desc.order;
    } else if (slice_desc.root_src != common_root ||
               slice_desc.order != common_order) {
      return false;
    }
    if (blk_desc.kind != AxialBlockKind::kPassThrough) {
      ++num_rotary_blocks;
      if (!IsBshdHeadBroadcastTrig4D(blk_desc.cos_val) ||
          !IsBshdHeadBroadcastTrig4D(blk_desc.sin_val)) {
        all_rotary_bshd_bcast = false;
      }
    }
    expected_offset = slice_desc.limit;
    blocks.push_back(blk_desc);
  }

  if (expected_offset != d_full || num_rotary_blocks < 1) {
    return false;
  }
  if (num_rotary_blocks == 1 && !all_rotary_bshd_bcast) {
    return false;
  }

  mlir::OpBuilder builder(concat_op);
  mlir::Location loc = concat_op.getLoc();
  mlir::Value x_full =
      BuildFullHeadTensor4D(builder, loc, common_root, common_order, out_type);
  mlir::Value cos_full =
      BuildCombinedRotaryTrig4D(builder, concat_op.getOperation(), loc,
                                out_type, blocks, /*is_cos=*/true);
  mlir::Value sin_full =
      BuildCombinedRotaryTrig4D(builder, concat_op.getOperation(), loc,
                                out_type, blocks, /*is_cos=*/false);
  mlir::Value p_mat =
      BuildAxialRotationMatrix(builder, concat_op.getOperation(), loc,
                               out_type.getElementType(), d_full, blocks);
  auto rot_x = EmitMinorDimContraction(builder, loc, out_type, x_full, p_mat);
  created_rope_dots.push_back(rot_x);
  mlir::Value mul_cos =
      builder.create<mlir::stablehlo::MulOp>(loc, x_full, cos_full);
  mlir::Value mul_sin =
      builder.create<mlir::stablehlo::MulOp>(loc, rot_x.getResult(), sin_full);
  mlir::Value add_out =
      builder.create<mlir::stablehlo::AddOp>(loc, mul_cos, mul_sin);
  concat_op.replaceAllUsesWith(add_out);
  return true;
}

// Family 2: Split-half complex RoPE
// `Concatenate(Sub(Mul(x_lo, cos), Mul(x_hi, sin)),
//              Add(Mul(x_hi, cos), Mul(x_lo, sin)), dim=3)`.
bool TryFoldSplitHalfComplexRotaryConcat(
    mlir::stablehlo::ConcatenateOp concat_op,
    llvm::SmallVectorImpl<mlir::stablehlo::DotGeneralOp>& created_rope_dots) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(concat_op.getType());
  if (!out_type || !out_type.hasStaticShape() || out_type.getRank() != 4 ||
      concat_op.getInputs().size() != 2 || concat_op.getDimension() != 3 ||
      !mlir::isa<mlir::FloatType>(out_type.getElementType())) {
    return false;
  }
  const int64_t d_full = out_type.getDimSize(3);
  if (d_full < 2 || d_full % 2 != 0) {
    return false;
  }
  const int64_t half_d = d_full / 2;

  auto sub_op =
      concat_op.getInputs()[0].getDefiningOp<mlir::stablehlo::SubtractOp>();
  auto add_op =
      concat_op.getInputs()[1].getDefiningOp<mlir::stablehlo::AddOp>();
  if (!sub_op || !add_op) {
    return false;
  }
  auto sub_lhs = sub_op.getLhs().getDefiningOp<mlir::stablehlo::MulOp>();
  auto sub_rhs = sub_op.getRhs().getDefiningOp<mlir::stablehlo::MulOp>();
  auto add_lhs = add_op.getLhs().getDefiningOp<mlir::stablehlo::MulOp>();
  auto add_rhs = add_op.getRhs().getDefiningOp<mlir::stablehlo::MulOp>();
  if (!sub_lhs || !sub_rhs || !add_lhs || !add_rhs) {
    return false;
  }

  ContiguousSliceDesc lo_desc;
  ContiguousSliceDesc hi_desc;
  mlir::Value cos_half;
  mlir::Value sin_half;
  bool matched = false;

  for (int i0 = 0; i0 < 2 && !matched; ++i0) {
    mlir::Value cand_lo = i0 == 0 ? sub_lhs.getLhs() : sub_lhs.getRhs();
    mlir::Value cand_cos = i0 == 0 ? sub_lhs.getRhs() : sub_lhs.getLhs();
    if (!MatchContiguousHeadSlice4D(cand_lo, half_d, lo_desc) ||
        lo_desc.start != 0 || lo_desc.limit != half_d ||
        lo_desc.d_full != d_full) {
      continue;
    }
    for (int i1 = 0; i1 < 2 && !matched; ++i1) {
      mlir::Value cand_hi = i1 == 0 ? sub_rhs.getLhs() : sub_rhs.getRhs();
      mlir::Value cand_sin = i1 == 0 ? sub_rhs.getRhs() : sub_rhs.getLhs();
      if (!MatchContiguousHeadSlice4D(cand_hi, half_d, hi_desc) ||
          hi_desc.root_src != lo_desc.root_src ||
          hi_desc.order != lo_desc.order || hi_desc.start != half_d ||
          hi_desc.limit != d_full) {
        continue;
      }
      auto matches_mul = [](mlir::stablehlo::MulOp m, mlir::Value a,
                            mlir::Value b) {
        return (m.getLhs() == a && m.getRhs() == b) ||
               (m.getLhs() == b && m.getRhs() == a);
      };
      if ((matches_mul(add_lhs, cand_hi, cand_cos) &&
           matches_mul(add_rhs, cand_lo, cand_sin)) ||
          (matches_mul(add_rhs, cand_hi, cand_cos) &&
           matches_mul(add_lhs, cand_lo, cand_sin))) {
        cos_half = cand_cos;
        sin_half = cand_sin;
        matched = true;
      }
    }
  }
  if (!matched) {
    return false;
  }

  llvm::SmallVector<AxialBlockDesc, 2> blocks = {
      {AxialBlockKind::kHalfSplitRotary, 0, half_d, cos_half, sin_half},
      {AxialBlockKind::kHalfSplitRotary, half_d, d_full, cos_half, sin_half},
  };
  llvm::SmallVector<AxialBlockDesc, 1> single_full_block = {
      {AxialBlockKind::kHalfSplitRotary, 0, d_full, nullptr, nullptr},
  };

  mlir::OpBuilder builder(concat_op);
  mlir::Location loc = concat_op.getLoc();
  mlir::Value x_full = BuildFullHeadTensor4D(builder, loc, lo_desc.root_src,
                                             lo_desc.order, out_type);
  mlir::Value cos_full =
      BuildCombinedRotaryTrig4D(builder, concat_op.getOperation(), loc,
                                out_type, blocks, /*is_cos=*/true);
  mlir::Value sin_full =
      BuildCombinedRotaryTrig4D(builder, concat_op.getOperation(), loc,
                                out_type, blocks, /*is_cos=*/false);
  mlir::Value j_mat = BuildAxialRotationMatrix(
      builder, concat_op.getOperation(), loc, out_type.getElementType(), d_full,
      single_full_block);
  auto rot_x = EmitMinorDimContraction(builder, loc, out_type, x_full, j_mat);
  created_rope_dots.push_back(rot_x);
  mlir::Value mul_cos =
      builder.create<mlir::stablehlo::MulOp>(loc, x_full, cos_full);
  mlir::Value mul_sin =
      builder.create<mlir::stablehlo::MulOp>(loc, rot_x.getResult(), sin_full);
  mlir::Value add_out =
      builder.create<mlir::stablehlo::AddOp>(loc, mul_cos, mul_sin);
  concat_op.replaceAllUsesWith(add_out);
  return true;
}

bool MatchMinorHalfSlice(mlir::Value val, int64_t half_d, bool is_upper,
                         mlir::Value& base_val) {
  auto val_type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!val_type || !val_type.hasStaticShape() || val_type.getRank() < 2 ||
      val_type.getDimSize(val_type.getRank() - 1) != half_d) {
    return false;
  }
  llvm::SmallVector<int64_t> leading_non_unit =
      GetNonUnitDimSizes(val_type.getShape().drop_back(1));

  mlir::Value cur = val;
  if (auto resh = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    auto r_in_type =
        mlir::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
    if (r_in_type && r_in_type.hasStaticShape() && r_in_type.getRank() >= 2 &&
        r_in_type.getDimSize(r_in_type.getRank() - 1) == half_d &&
        GetNonUnitDimSizes(r_in_type.getShape().drop_back(1)) ==
            leading_non_unit) {
      cur = resh.getOperand();
    }
  }
  auto slice_op = cur.getDefiningOp<mlir::stablehlo::SliceOp>();
  if (!slice_op) {
    return false;
  }
  auto src_type =
      mlir::dyn_cast<mlir::RankedTensorType>(slice_op.getOperand().getType());
  if (!src_type || !src_type.hasStaticShape() || src_type.getRank() < 2 ||
      src_type.getDimSize(src_type.getRank() - 1) != 2 * half_d ||
      GetNonUnitDimSizes(src_type.getShape().drop_back(1)) !=
          leading_non_unit) {
    return false;
  }
  const int64_t s_rank = src_type.getRank();
  llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
  llvm::ArrayRef<int64_t> strides = slice_op.getStrides();
  for (int64_t i = 0; i < s_rank - 1; ++i) {
    if (starts[i] != 0 || limits[i] != src_type.getDimSize(i) ||
        strides[i] != 1) {
      return false;
    }
  }
  const int64_t expected_start = is_upper ? half_d : 0;
  const int64_t expected_limit = is_upper ? 2 * half_d : half_d;
  if (starts[s_rank - 1] != expected_start ||
      limits[s_rank - 1] != expected_limit || strides[s_rank - 1] != 1) {
    return false;
  }
  base_val = slice_op.getOperand();
  return true;
}

// Family 3: Pre-transpose 4D `[B, S, H, D]` single-block `rotate_half`
// `Concatenate(Neg(x_hi), x_lo, dim = 3)` consumed by a rotary `MulOp(...,
// sin)` where `sin` is a 4D `BroadcastInDimOp` broadcasting across the head
// dimension `dim 2`.
bool TryFoldBshdHalfSplitRotateHalfConcat(
    mlir::stablehlo::ConcatenateOp concat_op,
    llvm::SmallVectorImpl<mlir::stablehlo::DotGeneralOp>& created_rope_dots) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(concat_op.getType());
  if (!out_type || !out_type.hasStaticShape() || out_type.getRank() != 4 ||
      concat_op.getInputs().size() != 2 || concat_op.getDimension() != 3) {
    return false;
  }
  mlir::Type elem_type = out_type.getElementType();
  if (!mlir::isa<mlir::FloatType>(elem_type)) {
    return false;
  }
  const int64_t d = out_type.getDimSize(3);
  if (d < 2 || d % 2 != 0) {
    return false;
  }

  // Verify that this `rotate_half` is in pre-transpose `[B, S, H, D]` layout by
  // checking that it is multiplied by a rotary `sin` broadcast across head dim
  // 2 (`dims = [..., 1, 3]`, omitting `2`).
  bool has_bshd_trig_user = false;
  for (mlir::Operation* user : concat_op->getUsers()) {
    if (auto mul_op = mlir::dyn_cast<mlir::stablehlo::MulOp>(user)) {
      mlir::Value other = mul_op.getLhs() == concat_op.getResult()
                              ? mul_op.getRhs()
                              : mul_op.getLhs();
      if (IsBshdHeadBroadcastTrig4D(other)) {
        has_bshd_trig_user = true;
        break;
      }
    }
  }
  if (!has_bshd_trig_user) {
    return false;
  }

  const int64_t half_d = d / 2;
  mlir::Value lhs = concat_op.getInputs()[0];
  mlir::Value rhs = concat_op.getInputs()[1];
  auto neg_lhs = lhs.getDefiningOp<mlir::stablehlo::NegOp>();
  if (!neg_lhs) {
    return false;
  }
  mlir::Value base_hi;
  mlir::Value base_lo;
  if (!MatchMinorHalfSlice(neg_lhs.getOperand(), half_d, /*is_upper=*/true,
                           base_hi) ||
      !MatchMinorHalfSlice(rhs, half_d, /*is_upper=*/false, base_lo) ||
      base_hi != base_lo) {
    return false;
  }

  llvm::SmallVector<AxialBlockDesc, 1> single_block = {
      {AxialBlockKind::kHalfSplitRotary, 0, d, nullptr, nullptr},
  };
  mlir::OpBuilder builder(concat_op);
  mlir::Location loc = concat_op.getLoc();
  mlir::Value x = EmitReshapeIfNeeded(builder, loc, base_lo, out_type);
  mlir::Value j_mat = BuildAxialRotationMatrix(
      builder, concat_op.getOperation(), loc, elem_type, d, single_block);
  auto replacement = EmitMinorDimContraction(builder, loc, out_type, x, j_mat);
  created_rope_dots.push_back(replacement);
  concat_op.replaceAllUsesWith(replacement.getResult());
  return true;
}

// Sinks any newly unified full-head Q/K projection
// `Reshape([S, H*D] -> [S, H, D, 1]) -> Transpose({3, 1, 0, 2})` or
// `Reshape([B*S, H*D] -> [B, S, H, D]) -> Transpose({0, 2, 1, 3})`
// directly consumed by our folded RoPE `DotGeneralOp` past `Add(bias, dot)`
// onto `dot`.
void SinkFoldedRopeProjections(
    mlir::func::FuncOp func,
    llvm::ArrayRef<mlir::stablehlo::DotGeneralOp> created_rope_dots) {
  for (mlir::stablehlo::DotGeneralOp rope_dot : created_rope_dots) {
    if (!rope_dot || rope_dot->getParentOp() == nullptr ||
        rope_dot->use_empty()) {
      continue;
    }
    auto tr_op =
        rope_dot.getLhs().getDefiningOp<mlir::stablehlo::TransposeOp>();
    if (!tr_op) {
      continue;
    }
    auto tr_type = mlir::dyn_cast<mlir::RankedTensorType>(tr_op.getType());
    auto r4 = tr_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!tr_type || !tr_type.hasStaticShape() || tr_type.getRank() != 4 ||
        !r4 || !r4->hasOneUse()) {
      continue;
    }
    const int64_t b = tr_type.getDimSize(0);
    const int64_t h = tr_type.getDimSize(1);
    const int64_t s = tr_type.getDimSize(2);
    const int64_t d = tr_type.getDimSize(3);
    if (d <= 1) {
      continue;
    }
    llvm::ArrayRef<int64_t> perm = tr_op.getPermutation();
    if (!(b == 1 && Is3102Permutation(perm)) && !Is0213Permutation(perm)) {
      continue;
    }
    auto add_op = r4.getOperand().getDefiningOp<mlir::stablehlo::AddOp>();
    if (!add_op || !add_op->hasOneUse()) {
      continue;
    }
    const int64_t feature_n = h * d;
    mlir::Value bias_1d;
    mlir::Value dot_val;
    if (auto bcast = add_op.getLhs()
                         .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      auto b_t =
          mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
      auto bcast_out_t =
          mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
      if (b_t && b_t.getRank() == 1 && b_t.getDimSize(0) == feature_n &&
          bcast_out_t &&
          bcast.getBroadcastDimensions() ==
              llvm::ArrayRef<int64_t>({bcast_out_t.getRank() - 1})) {
        bias_1d = bcast.getOperand();
        dot_val = add_op.getRhs();
      }
    }
    if (!bias_1d) {
      if (auto bcast =
              add_op.getRhs()
                  .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
        auto b_t = mlir::dyn_cast<mlir::RankedTensorType>(
            bcast.getOperand().getType());
        auto bcast_out_t =
            mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
        if (b_t && b_t.getRank() == 1 && b_t.getDimSize(0) == feature_n &&
            bcast_out_t &&
            bcast.getBroadcastDimensions() ==
                llvm::ArrayRef<int64_t>({bcast_out_t.getRank() - 1})) {
          bias_1d = bcast.getOperand();
          dot_val = add_op.getLhs();
        }
      }
    }
    if (!bias_1d || !dot_val || !IsDotOrDotGeneral(dot_val.getDefiningOp()) ||
        !dot_val.hasOneUse()) {
      continue;
    }

    mlir::OpBuilder builder(tr_op);
    mlir::Location loc = tr_op.getLoc();
    auto canonical_4d_type =
        mlir::RankedTensorType::get({b, s, h, d}, tr_type.getElementType());
    mlir::Value dot_4d = builder.create<mlir::stablehlo::ReshapeOp>(
        loc, canonical_4d_type, dot_val);
    mlir::Value trans_dot = builder.create<mlir::stablehlo::TransposeOp>(
        loc, tr_type, dot_4d, builder.getDenseI64ArrayAttr({0, 2, 1, 3}));
    auto bias_2d_type =
        mlir::RankedTensorType::get({h, d}, tr_type.getElementType());
    mlir::Value bias_2d =
        builder.create<mlir::stablehlo::ReshapeOp>(loc, bias_2d_type, bias_1d);
    mlir::Value bias_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, tr_type, bias_2d, builder.getDenseI64ArrayAttr({1, 3}));
    mlir::Value new_add =
        builder.create<mlir::stablehlo::AddOp>(loc, bias_bcast, trans_dot);
    tr_op.replaceAllUsesWith(new_add);
  }
  EraseTriviallyDeadOps(func);
}

struct FoldAxialAndSplitHalfRopePass
    : public mlir::PassWrapper<FoldAxialAndSplitHalfRopePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAxialAndSplitHalfRopePass)

  llvm::StringRef getArgument() const override {
    return "fold-axial-and-split-half-rope";
  }
  llvm::StringRef getDescription() const override {
    return "Folds multi-block axial, split-half complex, and pre-transpose "
           "BSHD rotary embedding concatenations into skew-symmetric "
           "contractions and sinks pre-rotary head projections.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::stablehlo::DotGeneralOp> created_rope_dots;

    llvm::SmallVector<mlir::stablehlo::ConcatenateOp> concat_ops;
    func.walk(
        [&](mlir::stablehlo::ConcatenateOp op) { concat_ops.push_back(op); });
    for (mlir::stablehlo::ConcatenateOp op : concat_ops) {
      if (!op || op->use_empty() || op->getParentOp() == nullptr) {
        continue;
      }
      if (TryFoldMultiBlockAxialRotaryConcat(op, created_rope_dots)) {
        continue;
      }
      if (TryFoldSplitHalfComplexRotaryConcat(op, created_rope_dots)) {
        continue;
      }
    }
    EraseTriviallyDeadOps(func);

    llvm::SmallVector<mlir::stablehlo::ConcatenateOp> remaining_concats;
    func.walk([&](mlir::stablehlo::ConcatenateOp op) {
      remaining_concats.push_back(op);
    });
    for (mlir::stablehlo::ConcatenateOp op : remaining_concats) {
      if (op && !op->use_empty() && op->getParentOp() != nullptr) {
        TryFoldBshdHalfSplitRotateHalfConcat(op, created_rope_dots);
      }
    }
    EraseTriviallyDeadOps(func);

    SinkFoldedRopeProjections(func, created_rope_dots);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createFoldAxialAndSplitHalfRopePass() {
  return std::make_unique<FoldAxialAndSplitHalfRopePass>();
}

}  // namespace torch_tpu
