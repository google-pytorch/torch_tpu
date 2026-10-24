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

#include "csrc/mlir/passes/factor_head_norm_and_complex_rope_sublanes.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/log/absl_check.h"
#include "llvm/ADT/APFloat.h"
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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// GENERIC: hardware TPU TensorCore sublane tile extent (8 sublanes).
constexpr int64_t kTpuSublaneTile = 8;
// GENERIC: hardware TPU TensorCore lane tile extent (128 lanes).
constexpr int64_t kTpuLaneTile = 128;
// GENERIC: semantic complex real/imaginary pair factor.
constexpr int64_t kComplexPairSize = 2;

bool IsEligibleHeadSplitShape(const int64_t m, const int64_t h,
                              const int64_t d) {
  // GENERIC: hardware M must span multiple 8-sublane tiles, H must fill at
  // least one 8-sublane tile, and D must be a multiple of the 128-lane tile.
  return m > kTpuSublaneTile && m % kTpuSublaneTile == 0 &&
         h >= kTpuSublaneTile && d >= kTpuLaneTile && d % kTpuLaneTile == 0;
}

bool IsStride2EvenOddSlice(mlir::stablehlo::SliceOp slice_op, const int64_t m,
                           const int64_t h, const int64_t d,
                           const int64_t start_offset) {
  if (!slice_op) return false;
  auto in_ty =
      mlir::dyn_cast<mlir::RankedTensorType>(slice_op.getOperand().getType());
  // GENERIC: semantic rank-3 [M, H, D] head-split tensor check.
  if (!in_ty || !in_ty.hasStaticShape() || in_ty.getRank() != 3 ||
      in_ty.getDimSize(0) != m || in_ty.getDimSize(1) != h ||
      in_ty.getDimSize(2) != d) {
    return false;
  }
  llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
  llvm::ArrayRef<int64_t> strides = slice_op.getStrides();
  // GENERIC: semantic stride-2 slice along the last dimension.
  return starts.size() == 3 && limits.size() == 3 && strides.size() == 3 &&
         starts[0] == 0 && starts[1] == 0 && starts[2] == start_offset &&
         limits[0] == m && limits[1] == h && limits[2] == d &&
         strides[0] == 1 && strides[1] == 1 && strides[2] == kComplexPairSize;
}

mlir::Value UnwrapSameElementViews(mlir::Value v) {
  while (v) {
    if (auto r = v.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      v = r.getOperand();
      continue;
    }
    if (auto b = v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      auto in_ty =
          mlir::dyn_cast<mlir::RankedTensorType>(b.getOperand().getType());
      auto out_ty = mlir::dyn_cast<mlir::RankedTensorType>(b.getType());
      if (in_ty && out_ty && in_ty.hasStaticShape() &&
          out_ty.hasStaticShape() &&
          in_ty.getNumElements() == out_ty.getNumElements()) {
        v = b.getOperand();
        continue;
      }
    }
    break;
  }
  return v;
}

// Returns the block-diagonal pairwise skew matrix for RoPE rotation.
mlir::DenseElementsAttr BuildBlockDiagonalSkewMatrix(mlir::Type elem_ty,
                                                     const int64_t d) {
  auto mat_ty = mlir::RankedTensorType::get({d, d}, elem_ty);
  const llvm::fltSemantics& sem =
      mlir::cast<mlir::FloatType>(elem_ty).getFloatSemantics();
  llvm::APFloat zero(sem, 0);
  llvm::APFloat pos_one(sem, 1);
  llvm::APFloat neg_one = pos_one;
  neg_one.changeSign();

  llvm::SmallVector<llvm::APFloat> vals(d * d, zero);
  const int64_t half_d = d / kComplexPairSize;
  for (int64_t j = 0; j < half_d; ++j) {
    const int64_t even_idx = kComplexPairSize * j;
    const int64_t odd_idx = even_idx + 1;
    vals[even_idx * d + odd_idx] = pos_one;
    vals[odd_idx * d + even_idx] = neg_one;
  }
  return mlir::DenseElementsAttr::get(mat_ty, vals);
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation*> dead_ops;
    func.walk([&](mlir::Operation* op) {
      if (mlir::isOpTriviallyDead(op)) {
        dead_ops.push_back(op);
      }
    });
    for (auto it = dead_ops.rbegin(); it != dead_ops.rend(); ++it) {
      if (mlir::isOpTriviallyDead(*it)) {
        (*it)->erase();
        changed = true;
      }
    }
  }
}

// Folds stride-2 complex RoPE into stride-2-free real rotation on [M, H, D]:
//   out = x * cos_3d +/- convert(dot_general(x_bf16, P_skew)) * sin_3d
void FoldStride2ComplexRope(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::ReshapeOp> candidates;
  func.walk([&](mlir::stablehlo::ReshapeOp op) { candidates.push_back(op); });

  for (mlir::stablehlo::ReshapeOp out_reshape : candidates) {
    auto out_ty = mlir::dyn_cast<mlir::RankedTensorType>(out_reshape.getType());
    // GENERIC: semantic rank-3 [M, H, D] output of interleaved real/imag pair.
    if (!out_ty || !out_ty.hasStaticShape() || out_ty.getRank() != 3 ||
        !out_ty.getElementType().isF32()) {
      continue;
    }
    const int64_t m = out_ty.getDimSize(0);
    const int64_t h = out_ty.getDimSize(1);
    const int64_t d = out_ty.getDimSize(2);
    if (!IsEligibleHeadSplitShape(m, h, d)) continue;
    const int64_t half_d = d / kComplexPairSize;

    auto cat = out_reshape.getOperand()
                   .getDefiningOp<mlir::stablehlo::ConcatenateOp>();
    // GENERIC: semantic rank-4 [M, H, D/2, 2] real/imag concatenation along 3.
    if (!cat || cat.getDimension() != 3 || cat.getInputs().size() != 2) {
      continue;
    }
    auto cat_ty = mlir::dyn_cast<mlir::RankedTensorType>(cat.getType());
    // GENERIC: semantic rank-4 [M, H, D/2, 2] shape check.
    if (!cat_ty || !cat_ty.hasStaticShape() || cat_ty.getRank() != 4 ||
        cat_ty.getDimSize(0) != m || cat_ty.getDimSize(1) != h ||
        cat_ty.getDimSize(2) != half_d ||
        cat_ty.getDimSize(3) != kComplexPairSize) {
      continue;
    }

    auto r_real =
        cat.getInputs()[0].getDefiningOp<mlir::stablehlo::ReshapeOp>();
    auto r_imag =
        cat.getInputs()[1].getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!r_real || !r_imag) continue;

    auto real_op = r_real.getOperand().getDefiningOp<mlir::stablehlo::RealOp>();
    auto imag_op = r_imag.getOperand().getDefiningOp<mlir::stablehlo::ImagOp>();
    if (!real_op || !imag_op || real_op.getOperand() != imag_op.getOperand()) {
      continue;
    }

    auto mul_op = real_op.getOperand().getDefiningOp<mlir::stablehlo::MulOp>();
    if (!mul_op) continue;

    auto match_input_complex = [&](mlir::Value v) -> mlir::Value {
      auto c_op = v.getDefiningOp<mlir::stablehlo::ComplexOp>();
      if (!c_op) return nullptr;
      auto s_even = c_op.getLhs().getDefiningOp<mlir::stablehlo::SliceOp>();
      auto s_odd = c_op.getRhs().getDefiningOp<mlir::stablehlo::SliceOp>();
      if (!IsStride2EvenOddSlice(s_even, m, h, d, 0) ||
          !IsStride2EvenOddSlice(s_odd, m, h, d, 1) ||
          s_even.getOperand() != s_odd.getOperand()) {
        return nullptr;
      }
      return s_even.getOperand();
    };

    mlir::Value x_f32 = match_input_complex(mul_op.getLhs());
    mlir::Value z_cis = mul_op.getRhs();
    if (!x_f32) {
      x_f32 = match_input_complex(mul_op.getRhs());
      z_cis = mul_op.getLhs();
    }
    if (!x_f32) continue;

    auto cis_bcast = z_cis.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!cis_bcast) continue;
    llvm::ArrayRef<int64_t> bdims = cis_bcast.getBroadcastDimensions();
    // GENERIC: semantic [M, D/2] broadcast along dims {0, 2} to [M, H, D/2].
    if (bdims.size() != 2 || bdims[0] != 0 || bdims[1] != 2) continue;

    mlir::Value base_cis = UnwrapSameElementViews(cis_bcast.getOperand());
    bool sin_negated = false;
    if (auto conj_op = base_cis.getDefiningOp<mlir::stablehlo::ComplexOp>()) {
      auto c_re = conj_op.getLhs().getDefiningOp<mlir::stablehlo::RealOp>();
      auto c_neg = conj_op.getRhs().getDefiningOp<mlir::stablehlo::NegOp>();
      auto c_im =
          c_neg ? c_neg.getOperand().getDefiningOp<mlir::stablehlo::ImagOp>()
                : nullptr;
      if (c_re && c_im && c_re.getOperand() == c_im.getOperand()) {
        sin_negated = true;
        base_cis = UnwrapSameElementViews(c_re.getOperand());
      }
    }

    auto cis_ty = mlir::dyn_cast<mlir::RankedTensorType>(base_cis.getType());
    // GENERIC: semantic rank-2 [M, D/2] complex rotary frequency table.
    if (!cis_ty || !cis_ty.hasStaticShape() || cis_ty.getRank() != 2 ||
        cis_ty.getDimSize(0) != m || cis_ty.getDimSize(1) != half_d) {
      continue;
    }
    auto complex_ty =
        mlir::dyn_cast<mlir::ComplexType>(cis_ty.getElementType());
    if (!complex_ty || !complex_ty.getElementType().isF32()) continue;

    mlir::OpBuilder builder(out_reshape);
    mlir::MLIRContext* ctx = builder.getContext();
    const mlir::Location loc = out_reshape.getLoc();
    const mlir::Type f32_ty = builder.getF32Type();

    auto expand_half_to_3d = [&](mlir::Value half_2d) -> mlir::Value {
      auto pair_ty =
          mlir::RankedTensorType::get({m, half_d, kComplexPairSize}, f32_ty);
      auto b1 = builder.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, pair_ty, half_2d, builder.getDenseI64ArrayAttr({0, 1}));
      auto flat_ty = mlir::RankedTensorType::get({m, d}, f32_ty);
      auto r1 = builder.create<mlir::stablehlo::ReshapeOp>(loc, flat_ty, b1);
      return builder.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, out_ty, r1, builder.getDenseI64ArrayAttr({0, 2}));
    };

    auto half_2d_ty = mlir::RankedTensorType::get({m, half_d}, f32_ty);
    mlir::Value cos_2d =
        builder.create<mlir::stablehlo::RealOp>(loc, half_2d_ty, base_cis);
    mlir::Value sin_2d =
        builder.create<mlir::stablehlo::ImagOp>(loc, half_2d_ty, base_cis);
    mlir::Value cos_3d = expand_half_to_3d(cos_2d);
    mlir::Value sin_3d = expand_half_to_3d(sin_2d);

    mlir::Value dot_lhs = x_f32;
    if (auto conv = x_f32.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      auto in_ty =
          mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
      if (in_ty && in_ty.hasStaticShape() &&
          in_ty.getShape() == out_ty.getShape() &&
          (in_ty.getElementType().isBF16() || in_ty.getElementType().isF16())) {
        dot_lhs = conv.getOperand();
      }
    }
    auto dot_lhs_ty = mlir::cast<mlir::RankedTensorType>(dot_lhs.getType());
    const mlir::Type dot_elem_ty = dot_lhs_ty.getElementType();

    mlir::DenseElementsAttr pskew_attr =
        BuildBlockDiagonalSkewMatrix(dot_elem_ty, d);
    mlir::Value pskew =
        builder.create<mlir::stablehlo::ConstantOp>(loc, pskew_attr);

    mlir::ArrayAttr precision_config = nullptr;
    if (dot_elem_ty.isF32()) {
      precision_config = builder.getArrayAttr({
          mlir::stablehlo::PrecisionAttr::get(
              ctx,
              mlir::stablehlo::Precision::
                  HIGHEST),  // EXPLICIT_PRECISION_OK=f32_rope_permutation
          mlir::stablehlo::PrecisionAttr::get(
              ctx,
              mlir::stablehlo::Precision::
                  HIGHEST),  // EXPLICIT_PRECISION_OK=f32_rope_permutation
      });
    }

    auto dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        ctx, /*lhsBatchingDimensions=*/{}, /*rhsBatchingDimensions=*/{},
        /*lhsContractingDimensions=*/{2}, /*rhsContractingDimensions=*/{0});
    mlir::Value xrot = builder.create<mlir::stablehlo::DotGeneralOp>(
        loc, dot_lhs_ty, dot_lhs, pskew, dot_dims, precision_config,
        /*algorithm=*/nullptr);
    mlir::Value xrot_f32 = xrot;
    if (dot_elem_ty != f32_ty) {
      xrot_f32 = builder.create<mlir::stablehlo::ConvertOp>(loc, out_ty, xrot);
    }

    mlir::Value term1 =
        builder.create<mlir::stablehlo::MulOp>(loc, out_ty, x_f32, cos_3d);
    mlir::Value term2 =
        builder.create<mlir::stablehlo::MulOp>(loc, out_ty, xrot_f32, sin_3d);
    mlir::Value new_out =
        sin_negated
            ? builder
                  .create<mlir::stablehlo::SubtractOp>(loc, out_ty, term1,
                                                       term2)
                  .getResult()
            : builder.create<mlir::stablehlo::AddOp>(loc, out_ty, term1, term2)
                  .getResult();
    out_reshape.replaceAllUsesWith(new_out);
  }

  EraseTriviallyDeadOps(func);
}

enum class InternalShapeKind {
  kNone,
  kRank3Full,     // [M, H, D] -> [M / 8, 8, H, D]
  kRank3Unit,     // [M, H, 1] -> [M / 8, 8, H, 1]
  kRank2Reduced,  // [M, H]    -> [M / 8, 8, H]
};

InternalShapeKind ClassifyInternalShape(mlir::Type type, const int64_t m,
                                        const int64_t h, const int64_t d) {
  auto r_ty = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!r_ty || !r_ty.hasStaticShape() ||
      !mlir::isa<mlir::FloatType>(r_ty.getElementType())) {
    return InternalShapeKind::kNone;
  }
  // GENERIC: semantic rank-3 [M, H, D] or [M, H, 1] internal tensor check.
  if (r_ty.getRank() == 3 && r_ty.getDimSize(0) == m &&
      r_ty.getDimSize(1) == h) {
    if (r_ty.getDimSize(2) == d) return InternalShapeKind::kRank3Full;
    if (r_ty.getDimSize(2) == 1) return InternalShapeKind::kRank3Unit;
  }
  // GENERIC: semantic rank-2 [M, H] reduced tensor check.
  if (r_ty.getRank() == 2 && r_ty.getDimSize(0) == m &&
      r_ty.getDimSize(1) == h) {
    return InternalShapeKind::kRank2Reduced;
  }
  return InternalShapeKind::kNone;
}

// Returns the 4D factored tensor type for the given internal shape.
mlir::RankedTensorType GetFactoredType(mlir::Type type, const int64_t m,
                                       const int64_t h, const int64_t d) {
  auto r_ty = mlir::cast<mlir::RankedTensorType>(type);
  const mlir::Type elem_ty = r_ty.getElementType();
  const int64_t m_outer = m / kTpuSublaneTile;
  switch (ClassifyInternalShape(type, m, h, d)) {
    case InternalShapeKind::kRank3Full:
      return mlir::RankedTensorType::get({m_outer, kTpuSublaneTile, h, d},
                                         elem_ty);
    case InternalShapeKind::kRank3Unit:
      return mlir::RankedTensorType::get({m_outer, kTpuSublaneTile, h, 1},
                                         elem_ty);
    case InternalShapeKind::kRank2Reduced:
      return mlir::RankedTensorType::get({m_outer, kTpuSublaneTile, h},
                                         elem_ty);
    case InternalShapeKind::kNone:
      return nullptr;
  }
  return nullptr;
}

bool IsValidExternalBroadcast(mlir::stablehlo::BroadcastInDimOp bcast,
                              const int64_t m, const int64_t h,
                              const int64_t d) {
  if (!bcast) return false;
  const InternalShapeKind out_kind =
      ClassifyInternalShape(bcast.getType(), m, h, d);
  if (out_kind == InternalShapeKind::kNone) return false;

  auto in_ty =
      mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
  if (!in_ty || !in_ty.hasStaticShape() ||
      ClassifyInternalShape(in_ty, m, h, d) != InternalShapeKind::kNone) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = bcast.getBroadcastDimensions();
  if (in_ty.getRank() == 0) {
    return dims.empty();
  }
  if (out_kind != InternalShapeKind::kRank3Full) {
    return false;
  }
  if (in_ty.getRank() == 1) {
    return (in_ty.getDimSize(0) == d && dims.size() == 1 && dims[0] == 2) ||
           (in_ty.getDimSize(0) == h && dims.size() == 1 && dims[0] == 1);
  }
  // GENERIC: semantic rank-2 [M, D] or [H, D] broadcast into [M, H, D].
  if (in_ty.getRank() == 2) {
    return (in_ty.getDimSize(0) == m && in_ty.getDimSize(1) == d &&
            dims.size() == 2 && dims[0] == 0 && dims[1] == 2) ||
           (in_ty.getDimSize(0) == h && in_ty.getDimSize(1) == d &&
            dims.size() == 2 && dims[0] == 1 && dims[1] == 2);
  }
  // GENERIC: semantic rank-3 [M, 1, D] broadcast into [M, H, D].
  if (in_ty.getRank() == 3) {
    return in_ty.getDimSize(0) == m && in_ty.getDimSize(1) == 1 &&
           in_ty.getDimSize(2) == d && dims.size() == 3 && dims[0] == 0 &&
           dims[1] == 1 && dims[2] == 2;
  }
  return false;
}

mlir::Value EmitFactoredExternalBroadcast(
    mlir::OpBuilder& builder, mlir::stablehlo::BroadcastInDimOp bcast,
    const int64_t m, const int64_t h, const int64_t d) {
  const mlir::Location loc = bcast.getLoc();
  mlir::Value operand = bcast.getOperand();
  auto in_ty = mlir::cast<mlir::RankedTensorType>(operand.getType());
  auto out_ty = GetFactoredType(bcast.getType(), m, h, d);
  llvm::ArrayRef<int64_t> dims = bcast.getBroadcastDimensions();
  const int64_t m_outer = m / kTpuSublaneTile;

  if (in_ty.getRank() == 0) {
    return builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, out_ty, operand, builder.getDenseI64ArrayAttr({}));
  }
  if (in_ty.getRank() == 1) {
    const int64_t new_dim = (dims[0] == 2) ? 3 : 2;
    return builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, out_ty, operand, builder.getDenseI64ArrayAttr({new_dim}));
  }
  // GENERIC: semantic rank-2 [M, D] or [H, D] broadcast into [M/8, 8, H, D].
  if (in_ty.getRank() == 2) {
    if (dims[0] == 0 && dims[1] == 2) {
      auto r_ty = mlir::RankedTensorType::get({m_outer, kTpuSublaneTile, d},
                                              in_ty.getElementType());
      mlir::Value r_val =
          builder.create<mlir::stablehlo::ReshapeOp>(loc, r_ty, operand);
      return builder.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, out_ty, r_val, builder.getDenseI64ArrayAttr({0, 1, 3}));
    }
    return builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, out_ty, operand, builder.getDenseI64ArrayAttr({2, 3}));
  }
  auto r_ty = mlir::RankedTensorType::get({m_outer, kTpuSublaneTile, 1, d},
                                          in_ty.getElementType());
  mlir::Value r_val =
      builder.create<mlir::stablehlo::ReshapeOp>(loc, r_ty, operand);
  return builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, out_ty, r_val, builder.getDenseI64ArrayAttr({0, 1, 2, 3}));
}

bool IsValidExternalConstant(mlir::stablehlo::ConstantOp c_op, const int64_t m,
                             const int64_t h, const int64_t d) {
  if (!c_op || ClassifyInternalShape(c_op.getType(), m, h, d) ==
                   InternalShapeKind::kNone) {
    return false;
  }
  return mlir::isa<mlir::DenseElementsAttr>(c_op.getValue());
}

bool IsEntry2DReshape(mlir::Operation* op, const int64_t m, const int64_t h,
                      const int64_t d) {
  auto r_op = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op);
  if (!r_op || ClassifyInternalShape(r_op.getType(), m, h, d) !=
                   InternalShapeKind::kRank3Full) {
    return false;
  }
  auto in_ty =
      mlir::dyn_cast<mlir::RankedTensorType>(r_op.getOperand().getType());
  // GENERIC: semantic rank-2 [M, H*D] linear projection output shape.
  return in_ty && in_ty.hasStaticShape() && in_ty.getRank() == 2 &&
         in_ty.getDimSize(0) == m && in_ty.getDimSize(1) == h * d;
}

mlir::Value MatchEntry4DTransposeSource(mlir::Operation* op, const int64_t m,
                                        const int64_t h, const int64_t d) {
  auto r_op = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op);
  if (!r_op || ClassifyInternalShape(r_op.getType(), m, h, d) !=
                   InternalShapeKind::kRank3Full) {
    return nullptr;
  }
  auto t_op = r_op.getOperand().getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!t_op) return nullptr;
  auto t_out_ty = mlir::dyn_cast<mlir::RankedTensorType>(t_op.getType());
  auto t_in_ty =
      mlir::dyn_cast<mlir::RankedTensorType>(t_op.getOperand().getType());
  // GENERIC: semantic rank-4 [1, H, M, D] attention tensor shape.
  if (!t_out_ty || !t_in_ty || !t_out_ty.hasStaticShape() ||
      !t_in_ty.hasStaticShape() || t_out_ty.getRank() != 4 ||
      t_in_ty.getRank() != 4 || t_in_ty.getDimSize(0) != 1 ||
      t_in_ty.getDimSize(1) != h || t_in_ty.getDimSize(2) != m ||
      t_in_ty.getDimSize(3) != d) {
    return nullptr;
  }
  llvm::ArrayRef<int64_t> perm = t_op.getPermutation();
  // GENERIC: semantic 4D transpose permutations {2, 1, 3, 0} or {0, 2, 1, 3}.
  if (perm.size() != 4) return nullptr;
  const bool is_mhd1 =
      t_out_ty.getDimSize(0) == m && t_out_ty.getDimSize(1) == h &&
      t_out_ty.getDimSize(2) == d && t_out_ty.getDimSize(3) == 1 &&
      perm[0] == 2 && perm[1] == 1 && perm[2] == 3 && perm[3] == 0;
  const bool is_1mhd =
      t_out_ty.getDimSize(0) == 1 && t_out_ty.getDimSize(1) == m &&
      t_out_ty.getDimSize(2) == h && t_out_ty.getDimSize(3) == d &&
      perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3;
  return (is_mhd1 || is_1mhd) ? t_op.getOperand() : nullptr;
}

bool IsExit2DReshape(mlir::Operation* op, const int64_t m, const int64_t h,
                     const int64_t d) {
  auto r_op = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op);
  if (!r_op || ClassifyInternalShape(r_op.getOperand().getType(), m, h, d) !=
                   InternalShapeKind::kRank3Full) {
    return false;
  }
  auto out_ty = mlir::dyn_cast<mlir::RankedTensorType>(r_op.getType());
  // GENERIC: semantic rank-2 [M, H*D] linear backward input shape.
  return out_ty && out_ty.hasStaticShape() && out_ty.getRank() == 2 &&
         out_ty.getDimSize(0) == m && out_ty.getDimSize(1) == h * d;
}

bool IsExit4DReshapeTranspose(mlir::Operation* op, const int64_t m,
                              const int64_t h, const int64_t d) {
  auto r_op = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op);
  if (!r_op || r_op.use_empty() ||
      ClassifyInternalShape(r_op.getOperand().getType(), m, h, d) !=
          InternalShapeKind::kRank3Full) {
    return false;
  }
  auto r_ty = mlir::dyn_cast<mlir::RankedTensorType>(r_op.getType());
  // GENERIC: semantic rank-4 [M, H, D, 1] or [1, M, H, D] intermediate shape.
  if (!r_ty || !r_ty.hasStaticShape() || r_ty.getRank() != 4) return false;
  const bool is_mhd1 = r_ty.getDimSize(0) == m && r_ty.getDimSize(1) == h &&
                       r_ty.getDimSize(2) == d && r_ty.getDimSize(3) == 1;
  const bool is_1mhd = r_ty.getDimSize(0) == 1 && r_ty.getDimSize(1) == m &&
                       r_ty.getDimSize(2) == h && r_ty.getDimSize(3) == d;
  if (!is_mhd1 && !is_1mhd) return false;

  for (mlir::Operation* user : r_op->getUsers()) {
    auto t_op = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(user);
    if (!t_op) return false;
    auto t_ty = mlir::dyn_cast<mlir::RankedTensorType>(t_op.getType());
    // GENERIC: semantic rank-4 [1, H, M, D] attention operand shape.
    if (!t_ty || !t_ty.hasStaticShape() || t_ty.getRank() != 4 ||
        t_ty.getDimSize(0) != 1 || t_ty.getDimSize(1) != h ||
        t_ty.getDimSize(2) != m || t_ty.getDimSize(3) != d) {
      return false;
    }
    llvm::ArrayRef<int64_t> perm = t_op.getPermutation();
    // GENERIC: semantic 4D transpose permutations {3, 1, 0, 2} or {0, 2, 1, 3}.
    if (perm.size() != 4) return false;
    if (is_mhd1 &&
        !(perm[0] == 3 && perm[1] == 1 && perm[2] == 0 && perm[3] == 2)) {
      return false;
    }
    if (is_1mhd &&
        !(perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3)) {
      return false;
    }
  }
  return true;
}

bool IsExitWeightGradReduce(mlir::Operation* op, const int64_t m,
                            const int64_t h, const int64_t d) {
  auto red_op = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op);
  if (!red_op || red_op.getInputs().size() != 1 ||
      red_op.getInitValues().size() != 1 ||
      ClassifyInternalShape(red_op.getInputs()[0].getType(), m, h, d) !=
          InternalShapeKind::kRank3Full) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = red_op.getDimensions();
  auto out_ty =
      mlir::dyn_cast<mlir::RankedTensorType>(red_op.getResult(0).getType());
  // GENERIC: semantic reduction over [M, H] (dims {0, 1}) to 1D [D] weight.
  return dims.size() == 2 && dims[0] == 0 && dims[1] == 1 && out_ty &&
         out_ty.hasStaticShape() && out_ty.getRank() == 1 &&
         out_ty.getDimSize(0) == d;
}

bool ValidateInternalOp(mlir::Operation* op,
                        const llvm::DenseSet<mlir::Operation*>& comp_set,
                        const int64_t m, const int64_t h, const int64_t d) {
  if (op->getNumResults() != 1) return false;
  const InternalShapeKind out_kind =
      ClassifyInternalShape(op->getResult(0).getType(), m, h, d);
  if (out_kind == InternalShapeKind::kNone) return false;

  if (IsEntry2DReshape(op, m, h, d) ||
      MatchEntry4DTransposeSource(op, m, h, d) != nullptr) {
    return true;
  }

  auto check_internal_or_ext_bcast = [&](mlir::Value v) -> bool {
    if (auto* def_op = v.getDefiningOp()) {
      if (comp_set.contains(def_op)) return true;
      if (auto bcast =
              mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(def_op)) {
        return IsValidExternalBroadcast(bcast, m, h, d);
      }
      if (auto c_op = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(def_op)) {
        return IsValidExternalConstant(c_op, m, h, d);
      }
    }
    return false;
  };

  if (auto r_op = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
    mlir::Value in_val = r_op.getOperand();
    return ClassifyInternalShape(in_val.getType(), m, h, d) !=
               InternalShapeKind::kNone &&
           check_internal_or_ext_bcast(in_val);
  }

  if (auto b_op = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
    mlir::Value in_val = b_op.getOperand();
    const InternalShapeKind in_kind =
        ClassifyInternalShape(in_val.getType(), m, h, d);
    if (in_kind == InternalShapeKind::kNone ||
        !check_internal_or_ext_bcast(in_val)) {
      return false;
    }
    llvm::ArrayRef<int64_t> dims = b_op.getBroadcastDimensions();
    // GENERIC: semantic rank-2 [M, H] broadcast along {0, 1}.
    if (in_kind == InternalShapeKind::kRank2Reduced) {
      return (out_kind == InternalShapeKind::kRank3Full ||
              out_kind == InternalShapeKind::kRank3Unit) &&
             dims.size() == 2 && dims[0] == 0 && dims[1] == 1;
    }
    // GENERIC: semantic rank-3 [M, H, 1] or [M, H, D] broadcast along {0,1,2}.
    return dims.size() == 3 && dims[0] == 0 && dims[1] == 1 && dims[2] == 2;
  }

  if (auto red_op = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
    if (red_op.getInputs().size() != 1 || red_op.getInitValues().size() != 1 ||
        out_kind != InternalShapeKind::kRank2Reduced) {
      return false;
    }
    mlir::Value in_val = red_op.getInputs()[0];
    llvm::ArrayRef<int64_t> dims = red_op.getDimensions();
    // GENERIC: semantic reduction over head_dim axis 2 of [M, H, D].
    return ClassifyInternalShape(in_val.getType(), m, h, d) ==
               InternalShapeKind::kRank3Full &&
           check_internal_or_ext_bcast(in_val) && dims.size() == 1 &&
           dims[0] == 2;
  }

  if (auto dot_op = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
    if (out_kind != InternalShapeKind::kRank3Full) return false;
    mlir::Value lhs = dot_op.getLhs();
    mlir::Value rhs = dot_op.getRhs();
    auto rhs_ty = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
    auto dnums = dot_op.getDotDimensionNumbers();
    // GENERIC: semantic unbatched contraction of [M, H, D] with [D, D].
    return ClassifyInternalShape(lhs.getType(), m, h, d) ==
               InternalShapeKind::kRank3Full &&
           check_internal_or_ext_bcast(lhs) && rhs_ty &&
           rhs_ty.hasStaticShape() && rhs_ty.getRank() == 2 &&
           rhs_ty.getDimSize(0) == d && rhs_ty.getDimSize(1) == d &&
           dnums.getLhsBatchingDimensions().empty() &&
           dnums.getRhsBatchingDimensions().empty() &&
           dnums.getLhsContractingDimensions().size() == 1 &&
           dnums.getLhsContractingDimensions()[0] == 2 &&
           dnums.getRhsContractingDimensions().size() == 1 &&
           dnums.getRhsContractingDimensions()[0] == 0;
  }

  if (mlir::isa<mlir::stablehlo::ConvertOp, mlir::stablehlo::RsqrtOp,
                mlir::stablehlo::NegOp, mlir::stablehlo::ExpOp>(op)) {
    mlir::Value in_val = op->getOperand(0);
    return ClassifyInternalShape(in_val.getType(), m, h, d) == out_kind &&
           check_internal_or_ext_bcast(in_val);
  }

  if (mlir::isa<mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
                mlir::stablehlo::MulOp, mlir::stablehlo::DivOp>(op)) {
    mlir::Value lhs = op->getOperand(0);
    mlir::Value rhs = op->getOperand(1);
    return ClassifyInternalShape(lhs.getType(), m, h, d) == out_kind &&
           ClassifyInternalShape(rhs.getType(), m, h, d) == out_kind &&
           check_internal_or_ext_bcast(lhs) && check_internal_or_ext_bcast(rhs);
  }

  return false;
}

// Factors [M, H, D] head-split normalization and rotary subgraphs into
// [M / 8, 8, H, D] so the 8-sublane tile stays on the M token axis across the
// 2D projection boundary and only transposes to [1, H, M, D] in bf16 at the
// attention boundary.
void FactorHeadSplitNormSublanes(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::ReduceOp> seeds;
  func.walk([&](mlir::stablehlo::ReduceOp op) { seeds.push_back(op); });

  llvm::DenseSet<mlir::Operation*> visited_seeds;
  for (mlir::stablehlo::ReduceOp seed : seeds) {
    if (visited_seeds.contains(seed.getOperation())) continue;
    if (seed.getInputs().size() != 1 || seed.getInitValues().size() != 1) {
      continue;
    }
    llvm::ArrayRef<int64_t> red_dims = seed.getDimensions();
    if (red_dims.size() != 1 || red_dims[0] != 2) continue;

    auto in_ty =
        mlir::dyn_cast<mlir::RankedTensorType>(seed.getInputs()[0].getType());
    // GENERIC: semantic rank-3 [M, H, D] input to head-dim reduction.
    if (!in_ty || !in_ty.hasStaticShape() || in_ty.getRank() != 3 ||
        !mlir::isa<mlir::FloatType>(in_ty.getElementType())) {
      continue;
    }
    const int64_t m = in_ty.getDimSize(0);
    const int64_t h = in_ty.getDimSize(1);
    const int64_t d = in_ty.getDimSize(2);
    if (!IsEligibleHeadSplitShape(m, h, d)) continue;
    if (ClassifyInternalShape(seed.getResult(0).getType(), m, h, d) !=
        InternalShapeKind::kRank2Reduced) {
      continue;
    }

    mlir::Block* seed_block = seed->getBlock();
    llvm::DenseSet<mlir::Operation*> comp_set;
    llvm::SmallVector<mlir::Operation*> queue;
    auto try_enqueue = [&](mlir::Operation* op) {
      if (!op || op->getBlock() != seed_block || comp_set.contains(op)) return;
      if (op->getNumResults() != 1 ||
          ClassifyInternalShape(op->getResult(0).getType(), m, h, d) ==
              InternalShapeKind::kNone) {
        return;
      }
      if (mlir::isa<mlir::stablehlo::ConstantOp>(op)) {
        return;
      }
      if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
        if (ClassifyInternalShape(bcast.getOperand().getType(), m, h, d) ==
            InternalShapeKind::kNone) {
          return;
        }
      }
      comp_set.insert(op);
      queue.push_back(op);
    };

    try_enqueue(seed.getOperation());
    while (!queue.empty()) {
      mlir::Operation* curr = queue.pop_back_val();
      if (!IsEntry2DReshape(curr, m, h, d) &&
          MatchEntry4DTransposeSource(curr, m, h, d) == nullptr) {
        for (mlir::Value v : curr->getOperands()) {
          if (auto* def_op = v.getDefiningOp()) {
            try_enqueue(def_op);
          }
        }
      }
      for (mlir::Operation* user : curr->getResult(0).getUsers()) {
        try_enqueue(user);
      }
    }

    llvm::SmallVector<mlir::Operation*> ordered_ops;
    llvm::SmallVector<mlir::Operation*> exit_ops;
    llvm::DenseSet<mlir::Operation*> exit_set;
    bool valid = true;

    for (mlir::Operation& op : *seed_block) {
      if (!comp_set.contains(&op)) continue;
      if (mlir::isa<mlir::stablehlo::ReduceOp>(op)) {
        visited_seeds.insert(&op);
      }
      ordered_ops.push_back(&op);
      if (!ValidateInternalOp(&op, comp_set, m, h, d)) {
        valid = false;
        break;
      }
    }
    if (!valid) continue;

    for (mlir::Operation& op : *seed_block) {
      if (!comp_set.contains(&op)) continue;
      for (mlir::Operation* user : op.getResult(0).getUsers()) {
        if (comp_set.contains(user) || exit_set.contains(user)) continue;
        if (user->getBlock() != seed_block ||
            (!IsExit2DReshape(user, m, h, d) &&
             !IsExit4DReshapeTranspose(user, m, h, d) &&
             !IsExitWeightGradReduce(user, m, h, d))) {
          valid = false;
          break;
        }
        exit_set.insert(user);
        exit_ops.push_back(user);
      }
      if (!valid) break;
    }
    if (!valid || exit_ops.empty()) continue;

    llvm::DenseMap<mlir::Value, mlir::Value> mapped_val;
    auto get_mapped = [&](mlir::OpBuilder& builder,
                          mlir::Value v) -> mlir::Value {
      if (auto it = mapped_val.find(v); it != mapped_val.end()) {
        return it->second;
      }
      if (auto c_op = v.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
        auto attr = mlir::cast<mlir::DenseElementsAttr>(c_op.getValue());
        auto new_ty = GetFactoredType(c_op.getType(), m, h, d);
        mlir::Value factored = builder.create<mlir::stablehlo::ConstantOp>(
            c_op.getLoc(), attr.reshape(new_ty));
        mapped_val[v] = factored;
        return factored;
      }
      auto bcast = v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
      ABSL_CHECK(bcast != nullptr);
      mlir::Value factored =
          EmitFactoredExternalBroadcast(builder, bcast, m, h, d);
      mapped_val[v] = factored;
      return factored;
    };

    const int64_t m_outer = m / kTpuSublaneTile;
    for (mlir::Operation* op : ordered_ops) {
      mlir::OpBuilder builder(op);
      const mlir::Location loc = op->getLoc();
      mlir::Value old_res = op->getResult(0);
      auto new_ty = GetFactoredType(old_res.getType(), m, h, d);

      if (IsEntry2DReshape(op, m, h, d)) {
        auto r_op = mlir::cast<mlir::stablehlo::ReshapeOp>(op);
        mapped_val[old_res] = builder.create<mlir::stablehlo::ReshapeOp>(
            loc, new_ty, r_op.getOperand());
        continue;
      }

      if (mlir::Value src_4d = MatchEntry4DTransposeSource(op, m, h, d)) {
        auto h_m_ty = mlir::RankedTensorType::get(
            {h, m_outer, kTpuSublaneTile, d}, new_ty.getElementType());
        mlir::Value r_val =
            builder.create<mlir::stablehlo::ReshapeOp>(loc, h_m_ty, src_4d);
        mapped_val[old_res] = builder.create<mlir::stablehlo::TransposeOp>(
            loc, new_ty, r_val, builder.getDenseI64ArrayAttr({1, 2, 0, 3}));
        continue;
      }

      if (auto r_op = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
        mlir::Value in_mapped = get_mapped(builder, r_op.getOperand());
        mapped_val[old_res] = (in_mapped.getType() == new_ty)
                                  ? in_mapped
                                  : builder
                                        .create<mlir::stablehlo::ReshapeOp>(
                                            loc, new_ty, in_mapped)
                                        .getResult();
        continue;
      }

      if (auto b_op = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
        mlir::Value in_mapped = get_mapped(builder, b_op.getOperand());
        const InternalShapeKind in_kind =
            ClassifyInternalShape(b_op.getOperand().getType(), m, h, d);
        if (in_kind == InternalShapeKind::kRank2Reduced) {
          mapped_val[old_res] =
              builder.create<mlir::stablehlo::BroadcastInDimOp>(
                  loc, new_ty, in_mapped,
                  builder.getDenseI64ArrayAttr({0, 1, 2}));
        } else {
          mapped_val[old_res] =
              builder.create<mlir::stablehlo::BroadcastInDimOp>(
                  loc, new_ty, in_mapped,
                  builder.getDenseI64ArrayAttr({0, 1, 2, 3}));
        }
        continue;
      }

      if (auto red_op = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
        mlir::Value in_mapped = get_mapped(builder, red_op.getInputs()[0]);
        auto new_red = builder.create<mlir::stablehlo::ReduceOp>(
            loc, mlir::TypeRange{new_ty}, mlir::ValueRange{in_mapped},
            mlir::ValueRange{red_op.getInitValues()[0]},
            builder.getDenseI64ArrayAttr({3}));
        mlir::IRMapping mapping;
        red_op.getBody().cloneInto(&new_red.getBody(), mapping);
        mapped_val[old_res] = new_red.getResult(0);
        continue;
      }

      if (auto dot_op = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
        mlir::Value lhs_mapped = get_mapped(builder, dot_op.getLhs());
        auto dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
            builder.getContext(), /*lhsBatchingDimensions=*/{},
            /*rhsBatchingDimensions=*/{},
            /*lhsContractingDimensions=*/{3},
            /*rhsContractingDimensions=*/{0});
        mapped_val[old_res] = builder.create<mlir::stablehlo::DotGeneralOp>(
            loc, new_ty, lhs_mapped, dot_op.getRhs(), dot_dims,
            dot_op.getPrecisionConfigAttr(), dot_op.getAlgorithmAttr());
        continue;
      }

      if (mlir::isa<mlir::stablehlo::ConvertOp>(op)) {
        mapped_val[old_res] = builder.create<mlir::stablehlo::ConvertOp>(
            loc, new_ty, get_mapped(builder, op->getOperand(0)));
        continue;
      }
      if (mlir::isa<mlir::stablehlo::RsqrtOp>(op)) {
        mapped_val[old_res] = builder.create<mlir::stablehlo::RsqrtOp>(
            loc, new_ty, get_mapped(builder, op->getOperand(0)));
        continue;
      }
      if (mlir::isa<mlir::stablehlo::NegOp>(op)) {
        mapped_val[old_res] = builder.create<mlir::stablehlo::NegOp>(
            loc, new_ty, get_mapped(builder, op->getOperand(0)));
        continue;
      }
      if (mlir::isa<mlir::stablehlo::ExpOp>(op)) {
        mapped_val[old_res] = builder.create<mlir::stablehlo::ExpOp>(
            loc, new_ty, get_mapped(builder, op->getOperand(0)));
        continue;
      }
      if (mlir::isa<mlir::stablehlo::AddOp>(op)) {
        mapped_val[old_res] = builder.create<mlir::stablehlo::AddOp>(
            loc, new_ty, get_mapped(builder, op->getOperand(0)),
            get_mapped(builder, op->getOperand(1)));
        continue;
      }
      if (mlir::isa<mlir::stablehlo::SubtractOp>(op)) {
        mapped_val[old_res] = builder.create<mlir::stablehlo::SubtractOp>(
            loc, new_ty, get_mapped(builder, op->getOperand(0)),
            get_mapped(builder, op->getOperand(1)));
        continue;
      }
      if (mlir::isa<mlir::stablehlo::MulOp>(op)) {
        mapped_val[old_res] = builder.create<mlir::stablehlo::MulOp>(
            loc, new_ty, get_mapped(builder, op->getOperand(0)),
            get_mapped(builder, op->getOperand(1)));
        continue;
      }
      if (mlir::isa<mlir::stablehlo::DivOp>(op)) {
        mapped_val[old_res] = builder.create<mlir::stablehlo::DivOp>(
            loc, new_ty, get_mapped(builder, op->getOperand(0)),
            get_mapped(builder, op->getOperand(1)));
        continue;
      }
    }

    for (mlir::Operation* exit_op : exit_ops) {
      mlir::OpBuilder builder(exit_op);
      const mlir::Location loc = exit_op->getLoc();
      if (IsExit2DReshape(exit_op, m, h, d)) {
        auto r_op = mlir::cast<mlir::stablehlo::ReshapeOp>(exit_op);
        mlir::Value in_mapped = get_mapped(builder, r_op.getOperand());
        mlir::Value new_exit = builder.create<mlir::stablehlo::ReshapeOp>(
            loc, r_op.getType(), in_mapped);
        r_op.replaceAllUsesWith(new_exit);
        continue;
      }
      if (IsExit4DReshapeTranspose(exit_op, m, h, d)) {
        auto r_op = mlir::cast<mlir::stablehlo::ReshapeOp>(exit_op);
        mlir::Value in_mapped = get_mapped(builder, r_op.getOperand());
        auto in_mapped_ty =
            mlir::cast<mlir::RankedTensorType>(in_mapped.getType());
        auto t_ty = mlir::RankedTensorType::get(
            {h, m_outer, kTpuSublaneTile, d}, in_mapped_ty.getElementType());
        mlir::Value t_val = builder.create<mlir::stablehlo::TransposeOp>(
            loc, t_ty, in_mapped, builder.getDenseI64ArrayAttr({2, 0, 1, 3}));
        auto out_4d_ty = mlir::RankedTensorType::get(
            {1, h, m, d}, in_mapped_ty.getElementType());
        mlir::Value new_4d =
            builder.create<mlir::stablehlo::ReshapeOp>(loc, out_4d_ty, t_val);
        llvm::SmallVector<mlir::Operation*> t_users(r_op->getUsers().begin(),
                                                    r_op->getUsers().end());
        for (mlir::Operation* t_user : t_users) {
          t_user->getResult(0).replaceAllUsesWith(new_4d);
        }
        continue;
      }
      if (IsExitWeightGradReduce(exit_op, m, h, d)) {
        auto red_op = mlir::cast<mlir::stablehlo::ReduceOp>(exit_op);
        mlir::Value in_mapped = get_mapped(builder, red_op.getInputs()[0]);
        auto new_red = builder.create<mlir::stablehlo::ReduceOp>(
            loc, red_op.getResultTypes(), mlir::ValueRange{in_mapped},
            mlir::ValueRange{red_op.getInitValues()[0]},
            builder.getDenseI64ArrayAttr({0, 1, 2}));
        mlir::IRMapping mapping;
        red_op.getBody().cloneInto(&new_red.getBody(), mapping);
        red_op.getResult(0).replaceAllUsesWith(new_red.getResult(0));
        continue;
      }
    }
  }

  EraseTriviallyDeadOps(func);
}

struct FactorHeadNormAndComplexRopeSublanesPass
    : public mlir::PassWrapper<FactorHeadNormAndComplexRopeSublanesPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      FactorHeadNormAndComplexRopeSublanesPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-factor-head-norm-and-complex-rope-sublanes";
  }

  llvm::StringRef getDescription() const override {
    return "Folds stride-2 complex rotary embeddings into real contractions "
           "and factors the token sublane dimension in head-split "
           "normalization subgraphs.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    FoldStride2ComplexRope(func);
    FactorHeadSplitNormSublanes(func);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createFactorHeadNormAndComplexRopeSublanesPass() {
  return std::make_unique<FactorHeadNormAndComplexRopeSublanesPass>();
}

}  // namespace torch_tpu
