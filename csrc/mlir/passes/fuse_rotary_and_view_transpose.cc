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

#include "csrc/mlir/passes/fuse_rotary_and_view_transpose.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

struct HalfSliceBranch {
  mlir::Value base_or_src;
  llvm::SmallVector<int64_t, 4> pre_transpose_full_shape;
  llvm::SmallVector<int64_t, 4> perm;
  bool pre_transpose = false;
  bool is_upper_half = false;
  bool negated = false;
};

bool IsUnitStrides(llvm::ArrayRef<int64_t> strides) {
  for (int64_t s : strides) {
    if (s != 1) {
      return false;
    }
  }
  return true;
}

bool IsFullPrefixSlice(llvm::ArrayRef<int64_t> starts,
                       llvm::ArrayRef<int64_t> limits,
                       llvm::ArrayRef<int64_t> shape, int64_t axis) {
  for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
    if (i == axis) {
      continue;
    }
    if (starts[i] != 0 || limits[i] != shape[i]) {
      return false;
    }
  }
  return true;
}

std::optional<bool> MatchHalfBounds(int64_t start, int64_t limit,
                                    int64_t half_dim, int64_t full_dim) {
  if (start == 0 && limit == half_dim) {
    return false;
  }
  if (start == half_dim && limit == full_dim) {
    return true;
  }
  return std::nullopt;
}

bool HasUnitTrailingDimsAfter(mlir::RankedTensorType type, int64_t dim) {
  for (int64_t i = dim + 1; i < type.getRank(); ++i) {
    if (type.getDimSize(i) != 1) {
      return false;
    }
  }
  return true;
}

// Form 1: `[Neg](Slice(base, [..., start:limit]))` on final transposed shape.
std::optional<HalfSliceBranch> MatchPostTransposeHalfSlice(
    mlir::stablehlo::SliceOp slice, int64_t full_last_dim, bool negated) {
  const int64_t half_dim = full_last_dim / 2;
  auto base_type =
      mlir::dyn_cast<mlir::RankedTensorType>(slice.getOperand().getType());
  if (!base_type || !base_type.hasStaticShape()) {
    return std::nullopt;
  }
  const int64_t rank = base_type.getRank();
  if (rank < 2 || base_type.getDimSize(rank - 1) != full_last_dim ||
      !IsUnitStrides(slice.getStrides())) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> starts = slice.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice.getLimitIndices();
  if (!IsFullPrefixSlice(starts, limits, base_type.getShape(), rank - 1)) {
    return std::nullopt;
  }
  auto is_upper = MatchHalfBounds(starts[rank - 1], limits[rank - 1], half_dim,
                                  full_last_dim);
  if (!is_upper.has_value()) {
    return std::nullopt;
  }

  HalfSliceBranch b;
  b.base_or_src = slice.getOperand();
  b.pre_transpose = false;
  b.is_upper_half = *is_upper;
  b.negated = negated;
  return b;
}

// Form 2: `[Neg](Transpose(Reshape(Slice(src, [..., start:limit])), perm))`
// emitted when slicing a non-contiguous view (e.g. `query_states[..., :D/2]`).
std::optional<HalfSliceBranch> MatchPreTransposeHalfSlice(
    mlir::stablehlo::TransposeOp transpose, int64_t full_last_dim,
    bool negated) {
  const int64_t half_dim = full_last_dim / 2;
  llvm::ArrayRef<int64_t> perm = transpose.getPermutation();
  const int64_t out_rank = perm.size();
  if (out_rank < 2) {
    return std::nullopt;
  }
  const int64_t mapped_dim = perm[out_rank - 1];

  auto reshape =
      transpose.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!reshape) {
    return std::nullopt;
  }
  auto r_type = mlir::dyn_cast<mlir::RankedTensorType>(reshape.getType());
  if (!r_type || !r_type.hasStaticShape() || r_type.getRank() != out_rank ||
      r_type.getDimSize(mapped_dim) != half_dim ||
      !HasUnitTrailingDimsAfter(r_type, mapped_dim)) {
    return std::nullopt;
  }

  auto slice = reshape.getOperand().getDefiningOp<mlir::stablehlo::SliceOp>();
  if (!slice) {
    return std::nullopt;
  }
  auto src_type =
      mlir::dyn_cast<mlir::RankedTensorType>(slice.getOperand().getType());
  if (!src_type || !src_type.hasStaticShape() ||
      !IsUnitStrides(slice.getStrides())) {
    return std::nullopt;
  }
  const int64_t src_rank = src_type.getRank();
  llvm::ArrayRef<int64_t> starts = slice.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice.getLimitIndices();
  if (src_rank < 2 || src_type.getDimSize(src_rank - 1) != full_last_dim ||
      !IsFullPrefixSlice(starts, limits, src_type.getShape(), src_rank - 1)) {
    return std::nullopt;
  }
  auto is_upper = MatchHalfBounds(starts[src_rank - 1], limits[src_rank - 1],
                                  half_dim, full_last_dim);
  if (!is_upper.has_value()) {
    return std::nullopt;
  }

  HalfSliceBranch b;
  b.base_or_src = slice.getOperand();
  b.pre_transpose = true;
  b.pre_transpose_full_shape.assign(r_type.getShape().begin(),
                                    r_type.getShape().end());
  b.pre_transpose_full_shape[mapped_dim] = full_last_dim;
  b.perm.assign(perm.begin(), perm.end());
  b.is_upper_half = *is_upper;
  b.negated = negated;
  return b;
}

std::optional<HalfSliceBranch> MatchHalfSliceBranch(mlir::Value val,
                                                    int64_t full_last_dim) {
  bool negated = false;
  if (auto neg = val.getDefiningOp<mlir::stablehlo::NegOp>()) {
    negated = true;
    val = neg.getOperand();
  }
  if (auto slice = val.getDefiningOp<mlir::stablehlo::SliceOp>()) {
    return MatchPostTransposeHalfSlice(slice, full_last_dim, negated);
  }
  if (auto transpose = val.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    return MatchPreTransposeHalfSlice(transpose, full_last_dim, negated);
  }
  return std::nullopt;
}

bool AreCompatibleHalfBranches(const HalfSliceBranch& lhs,
                               const HalfSliceBranch& rhs) {
  return lhs.pre_transpose == rhs.pre_transpose &&
         lhs.base_or_src == rhs.base_or_src &&
         lhs.is_upper_half != rhs.is_upper_half &&
         lhs.pre_transpose_full_shape == rhs.pre_transpose_full_shape &&
         lhs.perm == rhs.perm;
}

mlir::Value BuildHalfRotationMatrixConstant(mlir::OpBuilder& builder,
                                            mlir::Location loc, int64_t d,
                                            mlir::FloatType elem_type,
                                            const HalfSliceBranch& lhs_b,
                                            const HalfSliceBranch& rhs_b) {
  const int64_t half = d / 2;
  const float lhs_sign = lhs_b.negated ? -1.0f : 1.0f;
  const float rhs_sign = rhs_b.negated ? -1.0f : 1.0f;

  llvm::SmallVector<llvm::APFloat> j_vals(
      d * d, llvm::APFloat::getZero(elem_type.getFloatSemantics()));
  for (int64_t i = 0; i < half; ++i) {
    const int64_t lhs_row = lhs_b.is_upper_half ? (i + half) : i;
    const int64_t rhs_row = rhs_b.is_upper_half ? (i + half) : i;
    llvm::APFloat v_lhs(lhs_sign);
    llvm::APFloat v_rhs(rhs_sign);
    bool loses_info = false;
    v_lhs.convert(elem_type.getFloatSemantics(),
                  llvm::APFloat::rmNearestTiesToEven, &loses_info);
    v_rhs.convert(elem_type.getFloatSemantics(),
                  llvm::APFloat::rmNearestTiesToEven, &loses_info);
    j_vals[lhs_row * d + i] = v_lhs;
    j_vals[rhs_row * d + (i + half)] = v_rhs;
  }

  auto j_type = mlir::RankedTensorType::get({d, d}, elem_type);
  auto j_attr = mlir::DenseElementsAttr::get(j_type, j_vals);
  return mlir::stablehlo::ConstantOp::create(builder, loc, j_attr);
}

bool TryRewriteHalfRotationConcatenate(
    mlir::stablehlo::ConcatenateOp concat_op) {
  if (concat_op.getInputs().size() != 2) {
    return false;
  }
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(concat_op.getType());
  if (!out_type || !out_type.hasStaticShape()) {
    return false;
  }
  auto elem_type = mlir::dyn_cast<mlir::FloatType>(out_type.getElementType());
  if (!elem_type) {
    return false;
  }
  const int64_t rank = out_type.getRank();
  if (rank < 2 || concat_op.getDimension() != rank - 1) {
    return false;
  }
  const int64_t d = out_type.getDimSize(rank - 1);
  if (d < 32 || d > 256 || (d % 2) != 0) {
    return false;
  }

  auto lhs_b = MatchHalfSliceBranch(concat_op.getInputs()[0], d);
  auto rhs_b = MatchHalfSliceBranch(concat_op.getInputs()[1], d);
  if (!lhs_b.has_value() || !rhs_b.has_value() ||
      !AreCompatibleHalfBranches(*lhs_b, *rhs_b)) {
    return false;
  }

  mlir::OpBuilder builder(concat_op);
  mlir::Location loc = concat_op.getLoc();

  mlir::Value full_x = lhs_b->base_or_src;
  if (lhs_b->pre_transpose) {
    auto r_full_type =
        mlir::RankedTensorType::get(lhs_b->pre_transpose_full_shape, elem_type);
    mlir::Value reshaped = mlir::stablehlo::ReshapeOp::create(
        builder, loc, r_full_type, lhs_b->base_or_src);
    full_x = mlir::stablehlo::TransposeOp::create(
        builder, loc, out_type, reshaped,
        builder.getDenseI64ArrayAttr(lhs_b->perm));
  }

  mlir::Value j_const = BuildHalfRotationMatrixConstant(
      builder, loc, d, elem_type, *lhs_b, *rhs_b);

  auto dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(),
      /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{},
      /*lhsContractingDimensions=*/{rank - 1},
      /*rhsContractingDimensions=*/{0});

  auto dot = mlir::stablehlo::DotGeneralOp::create(
      builder, loc, out_type, full_x, j_const, dot_dims, mlir::ArrayAttr{},
      mlir::stablehlo::DotAlgorithmAttr{});

  concat_op.replaceAllUsesWith(dot.getResult());
  concat_op->erase();
  return true;
}

bool IsConstantFloatValue(mlir::Value val, double target, double tol = 1e-6) {
  mlir::DenseElementsAttr attr;
  if (!mlir::matchPattern(val, mlir::m_Constant(&attr)) || !attr.isSplat()) {
    return false;
  }
  if (auto f = llvm::dyn_cast<mlir::FloatAttr>(
          attr.getSplatValue<mlir::Attribute>())) {
    return std::abs(f.getValueAsDouble() - target) <= tol;
  }
  return false;
}

// Maps a 4D type inside `[1, H, S, ...]` (`[1, H, S, D]`, `[1, H, S, 1]`,
// `[1, H, S]`, `[D]`, or `[]`) to its contiguous pre-transpose `[S, H, ...]`
// type.
std::optional<mlir::RankedTensorType> Map4DNormTypeTo3D(mlir::Type ty,
                                                        int64_t h_dim,
                                                        int64_t s_dim,
                                                        int64_t d_dim) {
  auto rt = mlir::dyn_cast<mlir::RankedTensorType>(ty);
  if (!rt || !rt.hasStaticShape()) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> sh = rt.getShape();
  if (sh.empty() || (sh.size() == 1 && sh[0] == d_dim)) {
    return rt;
  }
  if (sh.size() == 4 && sh[0] == 1 && sh[1] == h_dim && sh[2] == s_dim &&
      (sh[3] == d_dim || sh[3] == 1)) {
    return mlir::RankedTensorType::get({s_dim, h_dim, sh[3]},
                                       rt.getElementType());
  }
  if (sh.size() == 3 && sh[0] == 1 && sh[1] == h_dim && sh[2] == s_dim) {
    return mlir::RankedTensorType::get({s_dim, h_dim}, rt.getElementType());
  }
  return std::nullopt;
}

struct NormSubgraphContext {
  int64_t h_dim;
  int64_t s_dim;
  int64_t d_dim;
  mlir::stablehlo::TransposeOp found_t_in = nullptr;
  llvm::SmallVector<mlir::Operation*, 16> topo_ops;
  llvm::DenseMap<mlir::Value, bool> visited;
};

bool IsMatchingInputTranspose(mlir::stablehlo::TransposeOp tin,
                              NormSubgraphContext& ctx) {
  if (tin.getPermutation() != llvm::ArrayRef<int64_t>({3, 1, 0, 2})) {
    return false;
  }
  auto rin = tin.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!rin ||
      mlir::cast<mlir::RankedTensorType>(rin.getType()).getShape() !=
          llvm::ArrayRef<int64_t>({ctx.s_dim, ctx.h_dim, ctx.d_dim, 1})) {
    return false;
  }
  if (ctx.found_t_in && ctx.found_t_in != tin) {
    return false;
  }
  ctx.found_t_in = tin;
  return true;
}

bool IsSupportedRowwiseNormOp(mlir::Operation* def) {
  if (auto red = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(def)) {
    return red.getDimensions() == llvm::ArrayRef<int64_t>({3}) &&
           red.getInputs().size() == 1;
  }
  if (auto bc = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(def)) {
    llvm::ArrayRef<int64_t> bdims = bc.getBroadcastDimensions();
    return bdims.empty() || bdims == llvm::ArrayRef<int64_t>({3}) ||
           bdims == llvm::ArrayRef<int64_t>({0, 1, 2});
  }
  return mlir::isa<mlir::stablehlo::ConvertOp, mlir::stablehlo::MulOp,
                   mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
                   mlir::stablehlo::DivOp, mlir::stablehlo::RsqrtOp,
                   mlir::stablehlo::ReshapeOp>(def);
}

std::optional<bool> TryMapLeafOrConstantOp(mlir::Operation* def,
                                           NormSubgraphContext& ctx) {
  if (!def || def->getNumResults() != 1) {
    return false;
  }
  if (auto tin = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(def)) {
    return IsMatchingInputTranspose(tin, ctx);
  }
  if (auto c = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(def)) {
    bool ok = mlir::cast<mlir::DenseElementsAttr>(c.getValue()).isSplat();
    if (ok) {
      ctx.topo_ops.push_back(def);
    }
    return ok;
  }
  return std::nullopt;
}

bool CanMapNormSubgraphValue(mlir::Value v, NormSubgraphContext& ctx) {
  if (auto it = ctx.visited.find(v); it != ctx.visited.end()) {
    return it->second;
  }
  if (!Map4DNormTypeTo3D(v.getType(), ctx.h_dim, ctx.s_dim, ctx.d_dim)
           .has_value()) {
    return ctx.visited[v] = false;
  }
  if (mlir::cast<mlir::RankedTensorType>(v.getType()).getRank() <= 1) {
    return ctx.visited[v] = true;
  }
  mlir::Operation* def = v.getDefiningOp();
  if (auto leaf = TryMapLeafOrConstantOp(def, ctx)) {
    return ctx.visited[v] = *leaf;
  }
  if (!IsSupportedRowwiseNormOp(def)) {
    return ctx.visited[v] = false;
  }

  ctx.visited[v] = true;
  for (mlir::Value operand : def->getOperands()) {
    if (!CanMapNormSubgraphValue(operand, ctx)) {
      return ctx.visited[v] = false;
    }
  }
  ctx.topo_ops.push_back(def);
  return true;
}

void CloneNormOpTo3D(mlir::OpBuilder& builder, mlir::Location loc,
                     mlir::Operation* op, mlir::RankedTensorType new_ty,
                     mlir::IRMapping& mapping) {
  if (auto c = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
    auto splat = mlir::cast<mlir::DenseElementsAttr>(c.getValue());
    auto new_attr = mlir::DenseElementsAttr::get(
        new_ty, splat.getSplatValue<mlir::Attribute>());
    mapping.map(c.getResult(),
                mlir::stablehlo::ConstantOp::create(builder, loc, new_attr));
    return;
  }
  if (auto res = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
    mapping.map(
        res.getResult(),
        mlir::stablehlo::ReshapeOp::create(
            builder, loc, new_ty, mapping.lookupOrDefault(res.getOperand())));
    return;
  }
  if (auto bc = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
    llvm::ArrayRef<int64_t> old_bdims = bc.getBroadcastDimensions();
    llvm::SmallVector<int64_t, 3> new_bdims;
    if (old_bdims == llvm::ArrayRef<int64_t>({3})) {
      new_bdims.push_back(2);
    } else if (old_bdims == llvm::ArrayRef<int64_t>({0, 1, 2})) {
      new_bdims = {0, 1};
    }
    mapping.map(bc.getResult(), mlir::stablehlo::BroadcastInDimOp::create(
                                    builder, loc, new_ty,
                                    mapping.lookupOrDefault(bc.getOperand()),
                                    builder.getDenseI64ArrayAttr(new_bdims)));
    return;
  }
  if (auto red = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
    auto new_red = mlir::stablehlo::ReduceOp::create(
        builder, loc, mlir::TypeRange{new_ty},
        mlir::ValueRange{mapping.lookupOrDefault(red.getInputs()[0])},
        mlir::ValueRange{mapping.lookupOrDefault(red.getInitValues()[0])},
        builder.getDenseI64ArrayAttr({2}));
    mlir::IRMapping reg_map;
    builder.cloneRegionBefore(red.getBody(), new_red.getBody(),
                              new_red.getBody().end(), reg_map);
    mapping.map(red.getResult(0), new_red.getResult(0));
    return;
  }
  mlir::Operation* cloned = builder.clone(*op, mapping);
  cloned->getResult(0).setType(new_ty);
  mapping.map(op->getResult(0), cloned->getResult(0));
}

// Commutes inverse 4D transpose pairs (`[S, H, D, 1] -> [1, H, S, D] ->
// row-wise norm/elementwise DAG -> [S, H, D, 1] -> [S, H, D]`) directly onto
// `[S, H, D]`.
bool TryCommuteInverseTransposeRmsNorm(mlir::stablehlo::ReshapeOp out_reshape) {
  auto out_3d_type =
      mlir::dyn_cast<mlir::RankedTensorType>(out_reshape.getType());
  if (!out_3d_type || !out_3d_type.hasStaticShape() ||
      out_3d_type.getRank() != 3) {
    return false;
  }
  const int64_t s_dim = out_3d_type.getDimSize(0);
  const int64_t h_dim = out_3d_type.getDimSize(1);
  const int64_t d_dim = out_3d_type.getDimSize(2);

  auto t_out =
      out_reshape.getOperand().getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!t_out || !t_out.getResult().hasOneUse() ||
      t_out.getPermutation() != llvm::ArrayRef<int64_t>({2, 1, 3, 0})) {
    return false;
  }

  NormSubgraphContext ctx{h_dim, s_dim, d_dim};
  if (!CanMapNormSubgraphValue(t_out.getOperand(), ctx) || !ctx.found_t_in) {
    return false;
  }

  mlir::OpBuilder builder(out_reshape);
  mlir::Location loc = out_reshape.getLoc();
  mlir::IRMapping mapping;

  auto r_in =
      ctx.found_t_in.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
  mlir::Value x_3d = mlir::stablehlo::ReshapeOp::create(
      builder, loc, out_3d_type, r_in.getOperand());
  mapping.map(ctx.found_t_in.getResult(), x_3d);

  for (mlir::Operation* op : ctx.topo_ops) {
    auto new_ty =
        *Map4DNormTypeTo3D(op->getResult(0).getType(), h_dim, s_dim, d_dim);
    CloneNormOpTo3D(builder, loc, op, new_ty, mapping);
  }

  out_reshape.replaceAllUsesWith(mapping.lookup(t_out.getOperand()));
  out_reshape->erase();
  return true;
}

class FuseRotaryAndViewTransposePass
    : public mlir::PassWrapper<FuseRotaryAndViewTransposePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseRotaryAndViewTransposePass)

  mlir::StringRef getArgument() const final {
    return "fuse-rotary-and-view-transpose";
  }

  mlir::StringRef getDescription() const final {
    return "Rewrite half-rotation slice-negate-concatenate into exact MXU "
           "DotGeneral and commute inverse transposes around last-axis "
           "RMSNorm.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    // 1. Replace `pow(x, 2.0)` with `mul(x, x)` prior to norm matching.
    llvm::SmallVector<mlir::stablehlo::PowOp, 16> pow_ops;
    func.walk([&](mlir::stablehlo::PowOp op) { pow_ops.push_back(op); });
    for (mlir::stablehlo::PowOp op : pow_ops) {
      if (mlir::isa<mlir::FloatType>(
              mlir::cast<mlir::ShapedType>(op.getType()).getElementType()) &&
          IsConstantFloatValue(op.getRhs(), 2.0)) {
        mlir::OpBuilder b(op);
        mlir::Value sq = mlir::stablehlo::MulOp::create(
            b, op.getLoc(), op.getLhs(), op.getLhs());
        op.replaceAllUsesWith(sq);
        op->erase();
      }
    }

    // 2. Commute inverse 4D transpose pairs around last-axis RMSNorm first so
    // downstream RoPE sees the clean contiguous 3D `[S, H, D]` tensor.
    llvm::SmallVector<mlir::stablehlo::ReshapeOp, 16> reshapes;
    func.walk([&](mlir::stablehlo::ReshapeOp op) { reshapes.push_back(op); });
    for (mlir::stablehlo::ReshapeOp op : reshapes) {
      if (op->getBlock() != nullptr) {
        TryCommuteInverseTransposeRmsNorm(op);
      }
    }

    // 3. Rewrite forward/backward half-rotation `Concatenate` into exact MXU
    // `DotGeneral(X, J_D)`.
    llvm::SmallVector<mlir::stablehlo::ConcatenateOp, 16> concats;
    func.walk(
        [&](mlir::stablehlo::ConcatenateOp op) { concats.push_back(op); });
    for (mlir::stablehlo::ConcatenateOp op : concats) {
      if (op->getBlock() != nullptr) {
        TryRewriteHalfRotationConcatenate(op);
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createFuseRotaryAndViewTransposePass() {
  return std::make_unique<FuseRotaryAndViewTransposePass>();
}

}  // namespace torch_tpu
