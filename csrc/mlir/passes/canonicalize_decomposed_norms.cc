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

#include "csrc/mlir/passes/canonicalize_decomposed_norms.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
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

mlir::Value SkipBroadcastsAndReshapes(mlir::Value val) {
  while (val) {
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      val = bcast.getOperand();
    } else if (auto reshape = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      val = reshape.getOperand();
    } else {
      break;
    }
  }
  return val;
}

std::optional<double> ExtractSplatFloatValue(mlir::Value val) {
  mlir::Value unwrapped = SkipBroadcastsAndReshapes(val);
  if (auto conv = unwrapped.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    unwrapped = SkipBroadcastsAndReshapes(conv.getOperand());
  }
  auto const_op = unwrapped.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr || attr.empty() || !attr.isSplat()) {
    return std::nullopt;
  }
  if (mlir::isa<mlir::FloatType>(attr.getElementType())) {
    return attr.getSplatValue<llvm::APFloat>().convertToDouble();
  }
  if (mlir::isa<mlir::IntegerType>(attr.getElementType())) {
    return static_cast<double>(
        attr.getSplatValue<llvm::APInt>().getSExtValue());
  }
  return std::nullopt;
}

mlir::Value CreateSplatFloatConstant(mlir::OpBuilder& builder,
                                     mlir::Location loc,
                                     mlir::RankedTensorType tensor_type,
                                     double value) {
  auto float_type =
      mlir::dyn_cast<mlir::FloatType>(tensor_type.getElementType());
  if (!float_type) {
    return nullptr;
  }
  llvm::APFloat ap_val(value);
  bool loses_info = false;
  ap_val.convert(float_type.getFloatSemantics(),
                 llvm::APFloat::rmNearestTiesToEven, &loses_info);
  if (!ap_val.isFinite()) {
    return nullptr;
  }
  auto attr = mlir::DenseElementsAttr::get(tensor_type, ap_val);
  return builder.create<mlir::stablehlo::ConstantOp>(loc, tensor_type, attr);
}

bool IsAddReduceBody(mlir::Region& region) {
  if (region.empty()) {
    return false;
  }
  mlir::Block& block = region.front();
  if (block.getNumArguments() != 2) {
    return false;
  }
  auto ret_op =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  if (!ret_op || ret_op.getOperands().size() != 1) {
    return false;
  }
  auto add_op = ret_op.getOperands()[0].getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add_op) {
    return false;
  }
  mlir::Value arg0 = block.getArgument(0);
  mlir::Value arg1 = block.getArgument(1);
  mlir::Value lhs = add_op.getLhs();
  mlir::Value rhs = add_op.getRhs();
  return (lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0);
}

bool IsSumReduceOp(mlir::stablehlo::ReduceOp reduce_op) {
  if (!reduce_op || reduce_op.getNumResults() != 1 ||
      reduce_op.getInputs().size() != 1 ||
      reduce_op.getInitValues().size() != 1) {
    return false;
  }
  return IsAddReduceBody(reduce_op.getBody());
}

bool IsProducedBySumReduce(mlir::Value val) {
  mlir::Value base = SkipBroadcastsAndReshapes(val);
  if (auto conv = base.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    base = SkipBroadcastsAndReshapes(conv.getOperand());
  }
  return IsSumReduceOp(base.getDefiningOp<mlir::stablehlo::ReduceOp>());
}

bool IsExactPowerOfTwo(double val) {
  if (!std::isfinite(val) || val <= 0.0) {
    return false;
  }
  int exp = 0;
  return std::frexp(val, &exp) == 0.5;
}

// Rewrites stablehlo.power(x, c) for float tensors with c in {1.0, 2.0, 3.0}
// into identity or multiplication chains.
struct LowerSmallIntegerPowPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::PowOp> {
  using OpRewritePattern<mlir::stablehlo::PowOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::PowOp pow_op,
      mlir::PatternRewriter& rewriter) const override {
    auto res_type = mlir::dyn_cast<mlir::RankedTensorType>(pow_op.getType());
    if (!res_type || !mlir::isa<mlir::FloatType>(res_type.getElementType())) {
      return mlir::failure();
    }
    const std::optional<double> exp_val =
        ExtractSplatFloatValue(pow_op.getRhs());
    if (!exp_val.has_value()) {
      return mlir::failure();
    }
    mlir::Value lhs = pow_op.getLhs();
    const mlir::Location loc = pow_op.getLoc();
    if (*exp_val == 1.0) {
      rewriter.replaceOp(pow_op, lhs);
      return mlir::success();
    }
    if (*exp_val == 2.0) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::MulOp>(pow_op, res_type, lhs,
                                                          lhs);
      return mlir::success();
    }
    if (*exp_val == 3.0) {
      mlir::Value sq =
          rewriter.create<mlir::stablehlo::MulOp>(loc, res_type, lhs, lhs);
      rewriter.replaceOpWithNewOp<mlir::stablehlo::MulOp>(pow_op, res_type, sq,
                                                          lhs);
      return mlir::success();
    }
    return mlir::failure();
  }
};

// Rewrites stablehlo.divide(lhs, c) into stablehlo.multiply(lhs, 1.0 / c) when
// lhs is produced by a sum reduction or c is an exact power of two.
struct ReplaceMeanDivWithReciprocalMulPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::DivOp> {
  using OpRewritePattern<mlir::stablehlo::DivOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::DivOp div_op,
      mlir::PatternRewriter& rewriter) const override {
    auto div_type = mlir::dyn_cast<mlir::RankedTensorType>(div_op.getType());
    if (!div_type || !div_type.hasStaticShape() ||
        !mlir::isa<mlir::FloatType>(div_type.getElementType())) {
      return mlir::failure();
    }
    const std::optional<double> divisor =
        ExtractSplatFloatValue(div_op.getRhs());
    if (!divisor.has_value() || !std::isfinite(*divisor) || *divisor <= 0.0) {
      return mlir::failure();
    }
    if (!IsProducedBySumReduce(div_op.getLhs()) &&
        !IsExactPowerOfTwo(*divisor)) {
      return mlir::failure();
    }
    mlir::Value inv_const = CreateSplatFloatConstant(rewriter, div_op.getLoc(),
                                                     div_type, 1.0 / *divisor);
    if (!inv_const || ExtractSplatFloatValue(inv_const).value_or(0.0) == 0.0) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<mlir::stablehlo::MulOp>(
        div_op, div_type, div_op.getLhs(), inv_const);
    return mlir::success();
  }
};

mlir::Value MatchOnePlusWeightAddOperand(mlir::stablehlo::AddOp add_op) {
  if (!add_op) {
    return nullptr;
  }
  mlir::Value weight;
  if (ExtractSplatFloatValue(add_op.getLhs()) == 1.0) {
    weight = add_op.getRhs();
  } else if (ExtractSplatFloatValue(add_op.getRhs()) == 1.0) {
    weight = add_op.getLhs();
  } else {
    return nullptr;
  }
  auto raw_type = mlir::dyn_cast<mlir::RankedTensorType>(
      SkipBroadcastsAndReshapes(weight).getType());
  if (!raw_type || raw_type.getRank() > 1) {
    return nullptr;
  }
  return weight;
}

bool TryPushConvertIntoOnePlusWeight(mlir::stablehlo::ConvertOp conv_op,
                                     mlir::RankedTensorType dst_type,
                                     mlir::PatternRewriter& rewriter) {
  auto add_op = conv_op.getOperand().getDefiningOp<mlir::stablehlo::AddOp>();
  mlir::Value weight = MatchOnePlusWeightAddOperand(add_op);
  if (!weight) {
    return false;
  }
  const mlir::Location loc = conv_op.getLoc();
  mlir::Value one_dst = CreateSplatFloatConstant(rewriter, loc, dst_type, 1.0);
  if (!one_dst) {
    return false;
  }
  mlir::Value weight_dst =
      rewriter.create<mlir::stablehlo::ConvertOp>(loc, dst_type, weight);
  rewriter.replaceOpWithNewOp<mlir::stablehlo::AddOp>(conv_op, dst_type,
                                                      one_dst, weight_dst);
  return true;
}

bool IsNormalizedActivation(mlir::Value val) {
  auto mul_op = val.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul_op) {
    return false;
  }
  mlir::Value lhs = SkipBroadcastsAndReshapes(mul_op.getLhs());
  mlir::Value rhs = SkipBroadcastsAndReshapes(mul_op.getRhs());
  return lhs.getDefiningOp<mlir::stablehlo::RsqrtOp>() != nullptr ||
         rhs.getDefiningOp<mlir::stablehlo::RsqrtOp>() != nullptr;
}

bool TryEliminateNormNarrowingRoundTripConvert(
    mlir::stablehlo::ConvertOp conv_op, mlir::RankedTensorType dst_type,
    mlir::PatternRewriter& rewriter) {
  auto inner_conv =
      conv_op.getOperand().getDefiningOp<mlir::stablehlo::ConvertOp>();
  if (!inner_conv) {
    return false;
  }
  mlir::Value orig_val = inner_conv.getOperand();
  if (orig_val.getType() != dst_type || !IsNormalizedActivation(orig_val)) {
    return false;
  }
  auto mid_type = mlir::dyn_cast<mlir::RankedTensorType>(inner_conv.getType());
  if (!mid_type) {
    return false;
  }
  auto mid_float = mlir::dyn_cast<mlir::FloatType>(mid_type.getElementType());
  auto dst_float = mlir::dyn_cast<mlir::FloatType>(dst_type.getElementType());
  if (!mid_float || !dst_float ||
      mid_float.getWidth() >= dst_float.getWidth()) {
    return false;
  }
  rewriter.replaceOp(conv_op, orig_val);
  return true;
}

struct CanonicalizeNormConvertPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::ConvertOp> {
  using OpRewritePattern<mlir::stablehlo::ConvertOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ConvertOp conv_op,
      mlir::PatternRewriter& rewriter) const override {
    auto dst_type = mlir::dyn_cast<mlir::RankedTensorType>(conv_op.getType());
    if (!dst_type || !dst_type.hasStaticShape() ||
        !mlir::isa<mlir::FloatType>(dst_type.getElementType())) {
      return mlir::failure();
    }
    if (TryPushConvertIntoOnePlusWeight(conv_op, dst_type, rewriter)) {
      return mlir::success();
    }
    if (TryEliminateNormNarrowingRoundTripConvert(conv_op, dst_type,
                                                  rewriter)) {
      return mlir::success();
    }
    return mlir::failure();
  }
};

bool AreInversePermutations(llvm::ArrayRef<int64_t> perm_in,
                            llvm::ArrayRef<int64_t> perm_out) {
  if (perm_in.size() != perm_out.size() || perm_in.empty()) {
    return false;
  }
  const int64_t rank = static_cast<int64_t>(perm_in.size());
  for (int64_t i = 0; i < rank; ++i) {
    if (perm_in[i] < 0 || perm_in[i] >= rank || perm_out[perm_in[i]] != i) {
      return false;
    }
  }
  return true;
}

bool IsTrailingUnitReshape(mlir::RankedTensorType base_type,
                           mlir::RankedTensorType reshaped_type) {
  const int64_t base_rank = base_type.getRank();
  const int64_t reshaped_rank = reshaped_type.getRank();
  if (reshaped_rank <= base_rank) {
    return false;
  }
  for (int64_t i = 0; i < base_rank; ++i) {
    if (reshaped_type.getDimSize(i) != base_type.getDimSize(i)) {
      return false;
    }
  }
  for (int64_t i = base_rank; i < reshaped_rank; ++i) {
    if (reshaped_type.getDimSize(i) != 1) {
      return false;
    }
  }
  return true;
}

struct MatchedTransposeNorm {
  mlir::Value orig_x;
  mlir::RankedTensorType orig_type;
  mlir::Type comp_elem_type;
  mlir::Type out_elem_type;
  mlir::Value reduce_init;
  int64_t d_orig = -1;
  double variance_scale = 0.0;
  double eps = 0.0;
  mlir::Value weight_1d;
  mlir::Operation* target_op_to_replace = nullptr;
};

mlir::Value StripOptionalSingleWeightMul(mlir::Value val,
                                         mlir::Value& out_weight_1d,
                                         int64_t& out_weight_dim) {
  out_weight_1d = nullptr;
  out_weight_dim = -1;
  auto mul_op = val.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul_op) {
    return val;
  }
  for (int i = 0; i < 2; ++i) {
    mlir::Value cand_w = i == 0 ? mul_op.getRhs() : mul_op.getLhs();
    mlir::Value cand_norm = i == 0 ? mul_op.getLhs() : mul_op.getRhs();
    auto bcast = cand_w.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!bcast || bcast.getBroadcastDimensions().size() != 1) {
      continue;
    }
    auto w_type =
        mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
    if (w_type && w_type.getRank() == 1) {
      out_weight_1d = bcast.getOperand();
      out_weight_dim = bcast.getBroadcastDimensions()[0];
      return cand_norm;
    }
  }
  return val;
}

bool MatchVarianceScaleAndReduce(mlir::Value var_val, double& out_scale,
                                 mlir::stablehlo::ReduceOp& out_reduce_op) {
  mlir::Value sum_part;
  if (auto div_op = var_val.getDefiningOp<mlir::stablehlo::DivOp>()) {
    const std::optional<double> div_val =
        ExtractSplatFloatValue(div_op.getRhs());
    if (!div_val.has_value() || *div_val <= 0.0) {
      return false;
    }
    out_scale = 1.0 / *div_val;
    sum_part = div_op.getLhs();
  } else if (auto mul_op = var_val.getDefiningOp<mlir::stablehlo::MulOp>()) {
    if (const std::optional<double> c =
            ExtractSplatFloatValue(mul_op.getRhs())) {
      out_scale = *c;
      sum_part = mul_op.getLhs();
    } else if (const std::optional<double> c =
                   ExtractSplatFloatValue(mul_op.getLhs())) {
      out_scale = *c;
      sum_part = mul_op.getRhs();
    } else {
      return false;
    }
  } else {
    return false;
  }
  out_reduce_op = SkipBroadcastsAndReshapes(sum_part)
                      .getDefiningOp<mlir::stablehlo::ReduceOp>();
  return IsSumReduceOp(out_reduce_op) &&
         out_reduce_op.getDimensions().size() == 1;
}

bool MatchRsqrtVarianceChain(mlir::Value rstd_bcast_val, double& out_scale,
                             double& out_eps,
                             mlir::stablehlo::ReduceOp& out_reduce_op) {
  auto rsqrt_op = SkipBroadcastsAndReshapes(rstd_bcast_val)
                      .getDefiningOp<mlir::stablehlo::RsqrtOp>();
  if (!rsqrt_op) {
    return false;
  }
  auto add_op = rsqrt_op.getOperand().getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add_op) {
    return false;
  }
  mlir::Value var_val;
  if (const std::optional<double> eps =
          ExtractSplatFloatValue(add_op.getRhs())) {
    out_eps = *eps;
    var_val = add_op.getLhs();
  } else if (const std::optional<double> eps =
                 ExtractSplatFloatValue(add_op.getLhs())) {
    out_eps = *eps;
    var_val = add_op.getRhs();
  } else {
    return false;
  }
  return MatchVarianceScaleAndReduce(var_val, out_scale, out_reduce_op);
}

bool IsSquareOfValue(mlir::Value sq_val, mlir::Value x_comp) {
  if (auto mul_op = sq_val.getDefiningOp<mlir::stablehlo::MulOp>()) {
    return mul_op.getLhs() == x_comp && mul_op.getRhs() == x_comp;
  }
  if (auto pow_op = sq_val.getDefiningOp<mlir::stablehlo::PowOp>()) {
    return pow_op.getLhs() == x_comp &&
           ExtractSplatFloatValue(pow_op.getRhs()) == 2.0;
  }
  return false;
}

bool TryPopulateUnitReshapeOrigin(mlir::stablehlo::TransposeOp t_in,
                                  mlir::stablehlo::TransposeOp t_out,
                                  int64_t mapped_axis,
                                  MatchedTransposeNorm& info) {
  auto r_in = t_in.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!r_in || !t_out->hasOneUse()) {
    return false;
  }
  auto r_out = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(*t_out->user_begin());
  if (!r_out) {
    return false;
  }
  auto x_type =
      mlir::dyn_cast<mlir::RankedTensorType>(r_in.getOperand().getType());
  auto r_in_type = mlir::dyn_cast<mlir::RankedTensorType>(r_in.getType());
  auto r_out_type = mlir::dyn_cast<mlir::RankedTensorType>(r_out.getType());
  if (!x_type || !r_in_type || !r_out_type ||
      x_type.getShape() != r_out_type.getShape() ||
      !IsTrailingUnitReshape(x_type, r_in_type) ||
      mapped_axis >= x_type.getRank()) {
    return false;
  }
  info.orig_x = r_in.getOperand();
  info.orig_type = x_type;
  info.d_orig = mapped_axis;
  info.target_op_to_replace = r_out.getOperation();
  return true;
}

bool PopulateSandwichOrigin(mlir::stablehlo::TransposeOp t_in,
                            mlir::stablehlo::TransposeOp t_out, int64_t d_red,
                            MatchedTransposeNorm& info) {
  llvm::ArrayRef<int64_t> perm_in = t_in.getPermutation();
  llvm::ArrayRef<int64_t> perm_out = t_out.getPermutation();
  if (!AreInversePermutations(perm_in, perm_out) || d_red < 0 ||
      d_red >= static_cast<int64_t>(perm_in.size())) {
    return false;
  }
  const int64_t mapped_axis = perm_in[d_red];
  if (TryPopulateUnitReshapeOrigin(t_in, t_out, mapped_axis, info)) {
    return true;
  }
  auto orig_type =
      mlir::dyn_cast<mlir::RankedTensorType>(t_in.getOperand().getType());
  if (!orig_type || !orig_type.hasStaticShape()) {
    return false;
  }
  info.orig_x = t_in.getOperand();
  info.orig_type = orig_type;
  info.d_orig = mapped_axis;
  info.target_op_to_replace = t_out.getOperation();
  return true;
}

mlir::Value StripOptionalConvert(mlir::Value val) {
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    return conv.getOperand();
  }
  return val;
}

bool MatchTransposeNormSubgraph(mlir::stablehlo::TransposeOp t_out,
                                MatchedTransposeNorm& info) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(t_out.getType());
  if (!out_type || !out_type.hasStaticShape()) {
    return false;
  }
  info.out_elem_type = out_type.getElementType();
  mlir::Value cur = StripOptionalConvert(t_out.getOperand());
  int64_t weight_dim = -1;
  cur = StripOptionalSingleWeightMul(cur, info.weight_1d, weight_dim);
  cur = StripOptionalConvert(cur);
  auto norm_mul = cur.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!norm_mul) {
    return false;
  }
  mlir::Value x_comp = norm_mul.getLhs();
  mlir::Value rstd_bcast = norm_mul.getRhs();
  mlir::stablehlo::ReduceOp reduce_op;
  if (!MatchRsqrtVarianceChain(rstd_bcast, info.variance_scale, info.eps,
                               reduce_op)) {
    std::swap(x_comp, rstd_bcast);
    if (!MatchRsqrtVarianceChain(rstd_bcast, info.variance_scale, info.eps,
                                 reduce_op)) {
      return false;
    }
  }
  if (!IsSquareOfValue(reduce_op.getInputs()[0], x_comp)) {
    return false;
  }
  const int64_t d_red = reduce_op.getDimensions()[0];
  if (info.weight_1d && weight_dim != d_red) {
    return false;
  }
  auto t_in = StripOptionalConvert(x_comp)
                  .getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!t_in) {
    return false;
  }
  info.comp_elem_type =
      mlir::cast<mlir::RankedTensorType>(x_comp.getType()).getElementType();
  info.reduce_init = reduce_op.getInitValues()[0];
  return PopulateSandwichOrigin(t_in, t_out, d_red, info);
}

mlir::Value ConvertIfNeeded(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value val, mlir::RankedTensorType dst_type) {
  if (val.getType() == dst_type) {
    return val;
  }
  return builder.create<mlir::stablehlo::ConvertOp>(loc, dst_type, val);
}

mlir::Value EmitCommutedNormSubgraph(mlir::PatternRewriter& rewriter,
                                     mlir::Location loc,
                                     const MatchedTransposeNorm& info) {
  const llvm::ArrayRef<int64_t> orig_shape = info.orig_type.getShape();
  const int64_t rank = info.orig_type.getRank();
  auto orig_comp_type =
      mlir::RankedTensorType::get(orig_shape, info.comp_elem_type);
  mlir::Value x_comp =
      ConvertIfNeeded(rewriter, loc, info.orig_x, orig_comp_type);
  mlir::Value sq = rewriter.create<mlir::stablehlo::MulOp>(loc, orig_comp_type,
                                                           x_comp, x_comp);

  llvm::SmallVector<int64_t> reduced_shape;
  llvm::SmallVector<int64_t> keepdim_shape(orig_shape.begin(),
                                           orig_shape.end());
  llvm::SmallVector<int64_t> all_axes;
  keepdim_shape[info.d_orig] = 1;
  for (int64_t i = 0; i < rank; ++i) {
    all_axes.push_back(i);
    if (i != info.d_orig) {
      reduced_shape.push_back(orig_shape[i]);
    }
  }
  auto reduced_type =
      mlir::RankedTensorType::get(reduced_shape, info.comp_elem_type);
  auto keepdim_type =
      mlir::RankedTensorType::get(keepdim_shape, info.comp_elem_type);

  auto reduce_op = rewriter.create<mlir::stablehlo::ReduceOp>(
      loc, mlir::TypeRange{reduced_type}, mlir::ValueRange{sq},
      mlir::ValueRange{info.reduce_init},
      rewriter.getDenseI64ArrayAttr({info.d_orig}));
  mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
      info.comp_elem_type, reduce_op.getBody(), rewriter);

  mlir::Value keepdim = rewriter.create<mlir::stablehlo::ReshapeOp>(
      loc, keepdim_type, reduce_op.getResult(0));
  mlir::Value scale_const = CreateSplatFloatConstant(
      rewriter, loc, keepdim_type, info.variance_scale);
  mlir::Value var = rewriter.create<mlir::stablehlo::MulOp>(
      loc, keepdim_type, keepdim, scale_const);
  mlir::Value eps_const =
      CreateSplatFloatConstant(rewriter, loc, keepdim_type, info.eps);
  mlir::Value var_eps = rewriter.create<mlir::stablehlo::AddOp>(
      loc, keepdim_type, var, eps_const);
  mlir::Value rstd =
      rewriter.create<mlir::stablehlo::RsqrtOp>(loc, keepdim_type, var_eps);
  mlir::Value bcast_rstd = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, orig_comp_type, rstd, rewriter.getDenseI64ArrayAttr(all_axes));
  mlir::Value norm_comp = rewriter.create<mlir::stablehlo::MulOp>(
      loc, orig_comp_type, x_comp, bcast_rstd);

  auto orig_out_type =
      mlir::RankedTensorType::get(orig_shape, info.out_elem_type);
  mlir::Value out_val =
      ConvertIfNeeded(rewriter, loc, norm_comp, orig_out_type);
  if (info.weight_1d) {
    mlir::Value w_cast =
        ConvertIfNeeded(rewriter, loc, info.weight_1d,
                        mlir::RankedTensorType::get({orig_shape[info.d_orig]},
                                                    info.out_elem_type));
    mlir::Value bcast_w = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, orig_out_type, w_cast,
        rewriter.getDenseI64ArrayAttr({info.d_orig}));
    out_val = rewriter.create<mlir::stablehlo::MulOp>(loc, orig_out_type,
                                                      out_val, bcast_w);
  }
  return out_val;
}

bool TryFoldInverseTransposeConvertSandwich(mlir::stablehlo::TransposeOp t_out,
                                            mlir::PatternRewriter& rewriter) {
  if (auto t_in =
          t_out.getOperand().getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    if (AreInversePermutations(t_in.getPermutation(), t_out.getPermutation())) {
      rewriter.replaceOp(t_out, t_in.getOperand());
      return true;
    }
  }
  auto conv_op = t_out.getOperand().getDefiningOp<mlir::stablehlo::ConvertOp>();
  if (!conv_op) {
    return false;
  }
  auto t_in =
      conv_op.getOperand().getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!t_in ||
      !AreInversePermutations(t_in.getPermutation(), t_out.getPermutation())) {
    return false;
  }
  rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvertOp>(
      t_out, t_out.getType(), t_in.getOperand());
  return true;
}

struct CommuteInverseTransposeNormPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern<mlir::stablehlo::TransposeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::TransposeOp t_out,
      mlir::PatternRewriter& rewriter) const override {
    if (TryFoldInverseTransposeConvertSandwich(t_out, rewriter)) {
      return mlir::success();
    }
    MatchedTransposeNorm info;
    if (!MatchTransposeNormSubgraph(t_out, info)) {
      return mlir::failure();
    }
    mlir::Value new_out =
        EmitCommutedNormSubgraph(rewriter, t_out.getLoc(), info);
    if (info.target_op_to_replace != t_out.getOperation()) {
      rewriter.replaceOp(info.target_op_to_replace, new_out);
      rewriter.eraseOp(t_out);
    } else {
      rewriter.replaceOp(t_out, new_out);
    }
    return mlir::success();
  }
};

struct CanonicalizeDecomposedNormsPass
    : public mlir::PassWrapper<CanonicalizeDecomposedNormsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeDecomposedNormsPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-canonicalize-decomposed-norms";
  }
  llvm::StringRef getDescription() const override {
    return "Canonicalizes decomposed normalization and activation patterns in "
           "StableHLO.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<
        LowerSmallIntegerPowPattern, ReplaceMeanDivWithReciprocalMulPattern,
        CanonicalizeNormConvertPattern, CommuteInverseTransposeNormPattern>(
        context);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createCanonicalizeDecomposedNormsPass() {
  return std::make_unique<CanonicalizeDecomposedNormsPass>();
}

}  // namespace torch_tpu
