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

#include "csrc/mlir/passes/canonicalize_rms_norm_and_layer_norm_backward.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
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

// Returns the underlying value after stripping BroadcastInDimOp and ReshapeOp
// wrappers.
mlir::Value StripBroadcastAndReshape(mlir::Value val) {
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

// Peels all shape-preserving floating-point ConvertOp wrappers. Used during
// pattern matching to see through `f32 -> bf16 -> f32` roundtrips emitted by
// `.float()` / `.type_as()` / `BuildSumShlo` / `BuildLayerNormShlo`.
mlir::Value PeelAllFloatConverts(mlir::Value val) {
  while (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    auto in_type =
        mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
    auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(conv.getType());
    if (in_type && out_type && in_type.hasStaticShape() &&
        out_type.hasStaticShape() &&
        in_type.getShape() == out_type.getShape() &&
        mlir::isa<mlir::FloatType>(in_type.getElementType()) &&
        mlir::isa<mlir::FloatType>(out_type.getElementType())) {
      val = conv.getOperand();
    } else {
      break;
    }
  }
  return val;
}

// Peels at most one shape-preserving floating-point upcast ConvertOp (e.g.
// `bf16 -> f32` or `f16 -> f32`), never peeling a downcast (`f32 -> bf16`).
mlir::Value PeelSingleUpcastConvert(mlir::Value val) {
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    auto in_type =
        mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
    auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(conv.getType());
    if (in_type && out_type && in_type.hasStaticShape() &&
        out_type.hasStaticShape() &&
        in_type.getShape() == out_type.getShape()) {
      auto in_flt = mlir::dyn_cast<mlir::FloatType>(in_type.getElementType());
      auto out_flt = mlir::dyn_cast<mlir::FloatType>(out_type.getElementType());
      if (in_flt && out_flt && in_flt.getWidth() < out_flt.getWidth()) {
        return conv.getOperand();
      }
    }
  }
  return val;
}

// Collects `root` and all transitive shape-preserving floating-point ConvertOp
// results reachable from `root` in the same block.
void CollectTransitiveFloatConvertUsers(
    mlir::Value root, llvm::SmallVectorImpl<mlir::Value>& out) {
  if (!root) return;
  auto root_type = mlir::dyn_cast<mlir::RankedTensorType>(root.getType());
  if (!root_type || !root_type.hasStaticShape()) return;
  const unsigned start_idx = out.size();
  if (!llvm::is_contained(out, root)) {
    out.push_back(root);
  }
  for (unsigned idx = start_idx; idx < out.size(); ++idx) {
    mlir::Value cur = out[idx];
    for (mlir::Operation* user : cur.getUsers()) {
      auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(user);
      if (!conv) continue;
      auto conv_type = mlir::dyn_cast<mlir::RankedTensorType>(conv.getType());
      if (!conv_type || !conv_type.hasStaticShape() ||
          conv_type.getShape() != root_type.getShape() ||
          !mlir::isa<mlir::FloatType>(conv_type.getElementType())) {
        continue;
      }
      mlir::Value res = conv.getResult();
      if (!llvm::is_contained(out, res)) {
        out.push_back(res);
      }
    }
  }
}

// Extracts the double value of a floating-point splat constant (optionally
// wrapped in BroadcastInDimOp / ReshapeOp / ConvertOp).
std::optional<double> GetFloatSplatConstantValue(mlir::Value val) {
  if (!val) return std::nullopt;
  val = StripBroadcastAndReshape(PeelAllFloatConverts(val));
  if (!val) return std::nullopt;
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) return std::nullopt;
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr || attr.empty() || !attr.isSplat()) return std::nullopt;
  if (!mlir::isa<mlir::FloatType>(attr.getElementType())) return std::nullopt;
  return attr.getSplatValue<llvm::APFloat>().convertToDouble();
}

// Checks whether `val` is a floating-point zero splat constant.
bool IsZeroFloatSplat(mlir::Value val) {
  if (!val) return false;
  val = StripBroadcastAndReshape(PeelAllFloatConverts(val));
  if (!val) return false;
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) return false;
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr || attr.empty() || !attr.isSplat()) return false;
  if (!mlir::isa<mlir::FloatType>(attr.getElementType())) return false;
  return attr.getSplatValue<llvm::APFloat>().isZero();
}

// Checks whether `val` is a floating-point splat constant (optionally wrapped
// in BroadcastInDimOp / ReshapeOp / ConvertOp).
bool IsFloatSplatConstant(mlir::Value val) {
  return GetFloatSplatConstantValue(val).has_value();
}

// Builds a scalar ConstantOp with value `val` and broadcasts it to
// `target_type`.
mlir::Value MakeBroadcastFloatConstant(mlir::PatternRewriter& rewriter,
                                       mlir::Location loc,
                                       mlir::RankedTensorType target_type,
                                       double val) {
  auto elem_type = mlir::cast<mlir::FloatType>(target_type.getElementType());
  llvm::APFloat apf(val);
  bool loses_info = false;
  apf.convert(elem_type.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
              &loses_info);
  auto scalar_type = mlir::RankedTensorType::get({}, elem_type);
  auto scalar_const = rewriter.create<mlir::stablehlo::ConstantOp>(
      loc, scalar_type, mlir::DenseElementsAttr::get(scalar_type, apf));
  if (target_type.getRank() == 0) {
    return scalar_const;
  }
  return rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, target_type, scalar_const, rewriter.getDenseI64ArrayAttr({}));
}

// Checks whether `reduce_op` is a single-input sum reduction with a 0.0 init
// value over `expected_dims`.
bool IsSumReduceOverDims(mlir::stablehlo::ReduceOp reduce_op,
                         llvm::ArrayRef<int64_t> expected_dims) {
  // GENERIC: semantic Single-input reduction check on StableHLO ReduceOp.
  if (!reduce_op || reduce_op.getInputs().size() != 1 ||
      reduce_op.getInitValues().size() != 1 ||
      reduce_op.getResults().size() != 1) {
    return false;
  }
  if (reduce_op.getDimensions() != expected_dims) {
    return false;
  }
  if (!IsZeroFloatSplat(reduce_op.getInitValues()[0])) {
    return false;
  }
  mlir::Region& body = reduce_op.getBody();
  if (!body.hasOneBlock()) return false;
  mlir::Block& block = body.front();
  // GENERIC: semantic Binary reducer block has exactly 2 scalar arguments.
  if (block.getNumArguments() != 2) return false;
  auto ret_op =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  // GENERIC: semantic Single-result reducer return check.
  if (!ret_op || ret_op.getResults().size() != 1) return false;
  auto add_op = ret_op.getResults()[0].getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add_op) return false;
  mlir::Value arg0 = block.getArgument(0);
  mlir::Value arg1 = block.getArgument(1);
  return (add_op.getLhs() == arg0 && add_op.getRhs() == arg1) ||
         (add_op.getLhs() == arg1 && add_op.getRhs() == arg0);
}

struct ScaledSumReduceMatch {
  mlir::stablehlo::ReduceOp reduce_op;
  mlir::Value reduced_input;
  double scale_val;
};

// Matches `Mul(Reduce(input, dims=expected_dims), inv_n)` (or either operand
// order).
std::optional<ScaledSumReduceMatch> MatchScaledSumReduce(
    mlir::Value val, llvm::ArrayRef<int64_t> expected_dims) {
  auto mul_op = val.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul_op) return std::nullopt;

  auto try_pair =
      [&](mlir::Value red_cand,  // REF_CAPTURE_OK=called locally
          mlir::Value scale_cand) -> std::optional<ScaledSumReduceMatch> {
    auto red_op = red_cand.getDefiningOp<mlir::stablehlo::ReduceOp>();
    if (!IsSumReduceOverDims(red_op, expected_dims)) return std::nullopt;
    std::optional<double> s = GetFloatSplatConstantValue(scale_cand);
    if (!s.has_value()) return std::nullopt;
    return ScaledSumReduceMatch{red_op, red_op.getInputs()[0], *s};
  };

  if (auto m = try_pair(mul_op.getLhs(), mul_op.getRhs())) return m;
  return try_pair(mul_op.getRhs(), mul_op.getLhs());
}

// Verifies that `bcast_op` broadcasts a per-batch stats tensor (mean or rstd)
// to `full_type`, where leading `drop_k` batch dimensions and any trailing
// normalized dimensions on the operand have size 1.
bool IsValidStatsBroadcast(mlir::stablehlo::BroadcastInDimOp bcast_op,
                           mlir::RankedTensorType full_type, int64_t batch_len,
                           int64_t drop_k, llvm::ArrayRef<int64_t> batch_dims,
                           llvm::ArrayRef<int64_t> all_dims) {
  if (!bcast_op || bcast_op.getType() != full_type) return false;
  llvm::ArrayRef<int64_t> dims = bcast_op.getBroadcastDimensions();
  if (dims != batch_dims && dims != all_dims) return false;

  auto op_type =
      mlir::dyn_cast<mlir::RankedTensorType>(bcast_op.getOperand().getType());
  if (!op_type || !op_type.hasStaticShape()) return false;
  llvm::ArrayRef<int64_t> op_shape = op_type.getShape();
  if (static_cast<int64_t>(op_shape.size()) < batch_len) return false;

  for (int64_t i = 0; i < drop_k; ++i) {
    // GENERIC: semantic Leading squeezed batch dimensions must have size 1.
    if (op_shape[i] != 1) return false;
  }
  for (int64_t i = drop_k; i < batch_len; ++i) {
    if (op_shape[i] != full_type.getDimSize(i)) return false;
  }
  for (int64_t i = batch_len; i < static_cast<int64_t>(op_shape.size()); ++i) {
    // GENERIC: semantic Trailing unsqueezed norm dimensions must have size 1.
    if (op_shape[i] != 1) return false;
  }
  return true;
}

// Verifies that `bcast_op` broadcasts a per-channel scale/gamma tensor to
// `expected_type` across `norm_dims` (plus optional leading size-1 batch dims
// `< batch_len`, as in AdaLN modulation `[1, D]` or `[1, 1, D] -> [1, S, D]`).
bool IsValidGammaBroadcast(mlir::stablehlo::BroadcastInDimOp bcast_op,
                           mlir::RankedTensorType expected_type,
                           int64_t batch_len,
                           llvm::ArrayRef<int64_t> norm_dims) {
  if (!bcast_op) return false;
  auto bcast_type = mlir::dyn_cast<mlir::RankedTensorType>(bcast_op.getType());
  if (!bcast_type || bcast_type.getShape() != expected_type.getShape()) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = bcast_op.getBroadcastDimensions();
  const int64_t norm_len = norm_dims.size();
  const int64_t dims_len = dims.size();
  if (dims_len < norm_len) return false;

  auto op_type =
      mlir::dyn_cast<mlir::RankedTensorType>(bcast_op.getOperand().getType());
  if (!op_type || !op_type.hasStaticShape() || op_type.getRank() != dims_len) {
    return false;
  }
  if (dims.drop_front(dims_len - norm_len) != norm_dims) {
    return false;
  }
  for (int64_t i = 0; i < norm_len; ++i) {
    if (op_type.getDimSize(dims_len - norm_len + i) !=
        expected_type.getDimSize(norm_dims[i])) {
      return false;
    }
  }
  for (int64_t i = 0; i < dims_len - norm_len; ++i) {
    // GENERIC: semantic Leading dims on gamma broadcast must be unit batch.
    if (dims[i] < 0 || dims[i] >= batch_len || op_type.getDimSize(i) != 1) {
      return false;
    }
  }
  return true;
}

// Squeezes leading unit dimensions of `val` into `squeezed_type`. If `val` is
// produced by an upcast `ConvertOp(src)` (`bf16/f16 -> f32`), squeezes `src`
// before converting so upstream reshapes on `src` cancel with the squeeze.
mlir::Value SqueezeInputTensor(mlir::PatternRewriter& rewriter,
                               mlir::Location loc, mlir::Value val,
                               mlir::RankedTensorType full_type,
                               mlir::RankedTensorType squeezed_type) {
  if (full_type == squeezed_type && val.getType() == squeezed_type) {
    return val;
  }
  auto squeeze_same_elem =
      [&](mlir::Value v,  // REF_CAPTURE_OK=called locally
          mlir::RankedTensorType target_sq_type) -> mlir::Value {
    if (v.getType() == target_sq_type) return v;
    if (auto resh = v.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      if (resh.getOperand().getType() == target_sq_type) {
        return resh.getOperand();
      }
    }
    return rewriter.create<mlir::stablehlo::ReshapeOp>(loc, target_sq_type, v);
  };

  mlir::Value src = PeelSingleUpcastConvert(val);
  auto src_type = mlir::cast<mlir::RankedTensorType>(src.getType());
  if (src_type.getElementType() != squeezed_type.getElementType()) {
    auto sq_src_type = mlir::RankedTensorType::get(squeezed_type.getShape(),
                                                   src_type.getElementType());
    mlir::Value sq_src = squeeze_same_elem(src, sq_src_type);
    return rewriter.create<mlir::stablehlo::ConvertOp>(loc, squeezed_type,
                                                       sq_src);
  }
  return squeeze_same_elem(src, squeezed_type);
}

// Builds a new single-input float sum ReduceOp with a 0.0 scalar init value.
mlir::Value BuildFloatSumReduce(mlir::PatternRewriter& rewriter,
                                mlir::Location loc, mlir::Value input,
                                llvm::ArrayRef<int64_t> reduce_dims,
                                mlir::RankedTensorType result_type) {
  auto elem_type = mlir::cast<mlir::FloatType>(result_type.getElementType());
  auto scalar_type = mlir::RankedTensorType::get({}, elem_type);
  mlir::Value zero = MakeBroadcastFloatConstant(rewriter, loc, scalar_type, 0);
  auto reduce_op = rewriter.create<mlir::stablehlo::ReduceOp>(
      loc, mlir::TypeRange{result_type}, mlir::ValueRange{input},
      mlir::ValueRange{zero}, rewriter.getDenseI64ArrayAttr(reduce_dims));
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  mlir::Region& region = reduce_op.getBody();
  mlir::Block* block = rewriter.createBlock(&region);
  block->addArgument(scalar_type, loc);
  block->addArgument(scalar_type, loc);
  rewriter.setInsertionPointToStart(block);
  auto sum = rewriter.create<mlir::stablehlo::AddOp>(
      loc, scalar_type, block->getArgument(0), block->getArgument(1));
  rewriter.create<mlir::stablehlo::ReturnOp>(loc, mlir::ValueRange{sum});
  return reduce_op.getResult(0);
}

// Replaces `old_reduce` with a sum reduction over `new_input` across
// `new_dims` producing `squeezed_out_type`, followed by optional ConvertOp /
// ReshapeOp to match `old_reduce`'s result type.
void ReplaceWithSqueezedSumReduce(mlir::PatternRewriter& rewriter,
                                  mlir::stablehlo::ReduceOp old_reduce,
                                  mlir::Value new_input,
                                  llvm::ArrayRef<int64_t> new_dims,
                                  mlir::RankedTensorType squeezed_out_type) {
  mlir::Location loc = old_reduce.getLoc();
  auto res_type =
      mlir::cast<mlir::RankedTensorType>(old_reduce.getResult(0).getType());
  mlir::Value new_red = BuildFloatSumReduce(rewriter, loc, new_input, new_dims,
                                            squeezed_out_type);
  if (res_type.getElementType() != squeezed_out_type.getElementType()) {
    auto cast_type = mlir::RankedTensorType::get(squeezed_out_type.getShape(),
                                                 res_type.getElementType());
    new_red =
        rewriter.create<mlir::stablehlo::ConvertOp>(loc, cast_type, new_red);
  }
  if (new_red.getType() != res_type) {
    new_red =
        rewriter.create<mlir::stablehlo::ReshapeOp>(loc, res_type, new_red);
  }
  rewriter.replaceOp(old_reduce, new_red);
}

// Checks whether all `inputs` precede `target_op` in the same block.
bool AllInputsPrecedeOp(llvm::ArrayRef<mlir::Value> inputs,
                        mlir::Operation* target_op) {
  if (!target_op || !target_op->getBlock()) return false;
  mlir::Block* block = target_op->getBlock();
  for (mlir::Value v : inputs) {
    if (!v) continue;
    if (mlir::Operation* def = v.getDefiningOp()) {
      if (def->getBlock() == block && !def->isBeforeInBlock(target_op)) {
        return false;
      }
    }
  }
  return true;
}

// Checks whether `cand` is an `x_hat = (x_f32 - mean_bcast) * rstd_bcast`
// computation equivalent to `x_hat` (peeling any float ConvertOp wrappers).
bool IsEquivalentLayerNormXHat(mlir::Value cand, mlir::Value x_hat,
                               mlir::Value x_f32, mlir::Value mean_raw,
                               mlir::Value rstd_raw,
                               mlir::RankedTensorType full_type,
                               int64_t batch_len, int64_t drop_k,
                               llvm::ArrayRef<int64_t> batch_dims,
                               llvm::ArrayRef<int64_t> all_dims) {
  if (cand == x_hat) return true;
  cand = PeelAllFloatConverts(cand);
  if (cand == x_hat) return true;
  auto mul_op = cand.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul_op || mul_op.getType() != full_type) return false;

  mlir::Value x_base = PeelAllFloatConverts(x_f32);
  auto try_order = [&](mlir::Value sub_val,  // REF_CAPTURE_OK=called locally
                       mlir::Value rstd_val) -> bool {
    sub_val = PeelAllFloatConverts(sub_val);
    rstd_val = PeelAllFloatConverts(rstd_val);
    auto sub_op = sub_val.getDefiningOp<mlir::stablehlo::SubtractOp>();
    auto rstd_b = rstd_val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!sub_op || !rstd_b || PeelAllFloatConverts(sub_op.getLhs()) != x_base) {
      return false;
    }
    auto mean_b = PeelAllFloatConverts(sub_op.getRhs())
                      .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!mean_b) return false;
    if (!IsValidStatsBroadcast(mean_b, full_type, batch_len, drop_k, batch_dims,
                               all_dims) ||
        !IsValidStatsBroadcast(rstd_b, full_type, batch_len, drop_k, batch_dims,
                               all_dims)) {
      return false;
    }
    return StripBroadcastAndReshape(mean_b.getOperand()) == mean_raw &&
           StripBroadcastAndReshape(rstd_b.getOperand()) == rstd_raw;
  };

  return try_order(mul_op.getLhs(), mul_op.getRhs()) ||
         try_order(mul_op.getRhs(), mul_op.getLhs());
}

// Checks whether `cand` is an `x_hat = x_f32 * rstd_bcast` computation for
// RMSNorm with `x_base` and `rstd_raw` (peeling any float ConvertOp wrappers).
bool IsEquivalentRmsNormXHat(mlir::Value cand, mlir::Value x_f32,
                             mlir::Value x_base, mlir::Value rstd_raw,
                             mlir::RankedTensorType full_type,
                             int64_t batch_len, int64_t drop_k,
                             llvm::ArrayRef<int64_t> batch_dims,
                             llvm::ArrayRef<int64_t> all_dims) {
  cand = PeelAllFloatConverts(cand);
  auto mul_op = cand.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul_op || mul_op.getType() != full_type) return false;

  auto try_order = [&](mlir::Value x_val,  // REF_CAPTURE_OK=called locally
                       mlir::Value rstd_val) -> bool {
    if (x_val != x_f32 && PeelAllFloatConverts(x_val) != x_base) return false;
    rstd_val = PeelAllFloatConverts(rstd_val);
    auto rstd_b = rstd_val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!IsValidStatsBroadcast(rstd_b, full_type, batch_len, drop_k, batch_dims,
                               all_dims)) {
      return false;
    }
    return StripBroadcastAndReshape(rstd_b.getOperand()) == rstd_raw;
  };

  return try_order(mul_op.getLhs(), mul_op.getRhs()) ||
         try_order(mul_op.getRhs(), mul_op.getLhs());
}

// =============================================================================
// Pattern 1: Canonicalize Decomposed Functorch RMSNorm Backward
// (`infinitely_differentiable_native_rms_norm_backward`)
//
// Under `torch.func.grad_and_value` (`create_graph=True`), PyTorch's
// `_FusedRmsNormBackward0` takes `infinitely_differentiable_native_rms_norm_
// backward` instead of `_fused_rms_norm_backward`, emitting:
//   term1 = rstd_bcast * convert(dy * broadcast(w))
//   term2 = ((x_f32 * broadcast(pow(rstd, 3))) / D) *
//           broadcast(convert(reduce(x_bf16 * (dy * broadcast(w)), dims=norm)))
//   dx    = convert(term1 - term2)
// while `dgamma` separately computes `reduce(convert(dy) * (x_f32 * rstd))`.
//
// This pattern rewrites `dx` and `dgamma` into the canonical factored form on
// the squeezed shape (dropping any leading unit batch dimensions):
//   x_hat  = x_f32_sq * rstd_bcast_sq
//   dgamma = reduce(dy_f32_sq * x_hat, dims=sq_batch_dims)
//   a2     = reduce(dy_w_f32_sq * x_hat, dims=sq_norm_dims)
//   dx     = (dy_w_f32_sq - (x_hat * broadcast(a2)) * (1/D)) * rstd_bcast_sq
// eliminating `pow(rstd, 3)` and the `[..., D]` vector division, and sharing
// `(dy_f32_sq, x_hat)` between `dgamma` and `a2` so XLA fuses them into one
// multi-output reduction fusion.
// =============================================================================
struct CanonicalizeRmsNormBackwardPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::SubtractOp> {
  using OpRewritePattern<mlir::stablehlo::SubtractOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::SubtractOp dx_sub,
      mlir::PatternRewriter& rewriter) const override {
    auto full_type = mlir::dyn_cast<mlir::RankedTensorType>(dx_sub.getType());
    // GENERIC: semantic Require rank >= 2 (>= 1 batch dim and >= 1 norm dim).
    if (!full_type || !full_type.hasStaticShape() ||
        !mlir::isa<mlir::FloatType>(full_type.getElementType()) ||
        full_type.getRank() < 2) {
      return mlir::failure();
    }
    const int64_t rank = full_type.getRank();

    auto term1 = dx_sub.getLhs().getDefiningOp<mlir::stablehlo::MulOp>();
    auto term2 = dx_sub.getRhs().getDefiningOp<mlir::stablehlo::MulOp>();
    if (!term1 || !term2) return mlir::failure();

    // Match term2 = Mul(div_val, dot_bcast_val) in either operand order.
    mlir::Value div_val;
    mlir::stablehlo::BroadcastInDimOp top_dot_bcast;
    mlir::stablehlo::ReduceOp dot_reduce;

    auto try_extract_dot =
        [&](mlir::Value cand_bcast,  // REF_CAPTURE_OK=called locally
            mlir::Value cand_div) -> bool {
      auto bcast =
          cand_bcast.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
      if (!bcast) return false;
      mlir::Value inner = StripBroadcastAndReshape(bcast.getOperand());
      while (auto conv = inner.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
        inner = StripBroadcastAndReshape(conv.getOperand());
      }
      auto red = inner.getDefiningOp<mlir::stablehlo::ReduceOp>();
      // GENERIC: semantic Single-input reduction check on dot_reduce.
      if (!red || red.getInputs().size() != 1 ||
          red.getInitValues().size() != 1 || red.getResults().size() != 1) {
        return false;
      }
      top_dot_bcast = bcast;
      dot_reduce = red;
      div_val = cand_div;
      return true;
    };

    if (!try_extract_dot(term2.getRhs(), term2.getLhs()) &&
        !try_extract_dot(term2.getLhs(), term2.getRhs())) {
      return mlir::failure();
    }

    llvm::ArrayRef<int64_t> norm_dims = dot_reduce.getDimensions();
    const int64_t norm_len = norm_dims.size();
    const int64_t batch_len = rank - norm_len;
    // GENERIC: semantic Require >= 1 batch dim and >= 1 norm dim.
    if (norm_len < 1 || batch_len < 1) return mlir::failure();
    for (int64_t i = 0; i < norm_len; ++i) {
      if (norm_dims[i] != batch_len + i) return mlir::failure();
    }
    if (!IsSumReduceOverDims(dot_reduce, norm_dims)) {
      return mlir::failure();
    }

    llvm::SmallVector<int64_t, 4> batch_dims;
    llvm::SmallVector<int64_t, 6> all_dims;
    for (int64_t i = 0; i < rank; ++i) {
      all_dims.push_back(i);
      if (i < batch_len) batch_dims.push_back(i);
    }

    int64_t drop_k = 0;
    // GENERIC: semantic Squeeze leading unit batch dims while keeping >= 1
    // batch dim.
    while (drop_k < batch_len - 1 && full_type.getDimSize(drop_k) == 1) {
      ++drop_k;
    }

    if (!IsValidStatsBroadcast(top_dot_bcast, full_type, batch_len, drop_k,
                               batch_dims, all_dims)) {
      return mlir::failure();
    }

    int64_t norm_numel = 1;
    for (int64_t d : norm_dims) {
      norm_numel *= full_type.getDimSize(d);
    }
    if (norm_numel <= 0) return mlir::failure();

    // Match div_val = Div(x_rstd3, norm_numel) or Mul(x_rstd3, 1.0/norm_numel).
    mlir::Value x_rstd3_val;
    if (auto div_op = div_val.getDefiningOp<mlir::stablehlo::DivOp>()) {
      std::optional<double> denom = GetFloatSplatConstantValue(div_op.getRhs());
      if (!denom.has_value() ||
          std::abs(*denom - static_cast<double>(norm_numel)) >
              1e-3 * static_cast<double>(norm_numel)) {
        return mlir::failure();
      }
      x_rstd3_val = div_op.getLhs();
    } else if (auto mul_div = div_val.getDefiningOp<mlir::stablehlo::MulOp>()) {
      const double expected_inv = 1.0 / static_cast<double>(norm_numel);
      if (auto s = GetFloatSplatConstantValue(mul_div.getRhs());
          s.has_value() && std::abs(*s - expected_inv) <= 1e-3 * expected_inv) {
        x_rstd3_val = mul_div.getLhs();
      } else if (auto s2 = GetFloatSplatConstantValue(mul_div.getLhs());
                 s2.has_value() &&
                 std::abs(*s2 - expected_inv) <= 1e-3 * expected_inv) {
        x_rstd3_val = mul_div.getRhs();
      } else {
        return mlir::failure();
      }
    } else {
      return mlir::failure();
    }

    // Match x_rstd3_val = Mul(x_f32, Broadcast(Pow(rstd, 3.0))) in either
    // operand order.
    auto x_rstd3_mul = x_rstd3_val.getDefiningOp<mlir::stablehlo::MulOp>();
    if (!x_rstd3_mul) return mlir::failure();

    mlir::Value x_f32;
    mlir::Value rstd_raw;
    auto try_extract_pow =
        [&](mlir::Value cand_x,  // REF_CAPTURE_OK=called locally
            mlir::Value cand_pow_bcast) -> bool {
      auto pow_bcast =
          cand_pow_bcast.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
      if (!IsValidStatsBroadcast(pow_bcast, full_type, batch_len, drop_k,
                                 batch_dims, all_dims)) {
        return false;
      }
      mlir::Value inner = StripBroadcastAndReshape(
          PeelAllFloatConverts(pow_bcast.getOperand()));
      auto pow_op = inner.getDefiningOp<mlir::stablehlo::PowOp>();
      if (!pow_op) return false;
      std::optional<double> exp_val =
          GetFloatSplatConstantValue(pow_op.getRhs());
      // GENERIC: semantic Cubic exponent 3.0 from d/dx(rstd) = -x * rstd^3 / N.
      if (!exp_val.has_value() || std::abs(*exp_val - 3.0) > 1e-4) {
        return false;
      }
      x_f32 = cand_x;
      rstd_raw =
          StripBroadcastAndReshape(PeelAllFloatConverts(pow_op.getLhs()));
      return true;
    };

    if (!try_extract_pow(x_rstd3_mul.getLhs(), x_rstd3_mul.getRhs()) &&
        !try_extract_pow(x_rstd3_mul.getRhs(), x_rstd3_mul.getLhs())) {
      return mlir::failure();
    }

    // Match term1 = Mul(rstd_bcast_1, dy_gamma_f32) in either operand order.
    mlir::stablehlo::BroadcastInDimOp rstd_bcast_1;
    mlir::Value dy_gamma_f32;
    auto try_extract_term1 =
        [&](mlir::Value cand_rstd,  // REF_CAPTURE_OK=called locally
            mlir::Value cand_dy_gamma) -> bool {
      auto bcast = cand_rstd.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
      if (!IsValidStatsBroadcast(bcast, full_type, batch_len, drop_k,
                                 batch_dims, all_dims)) {
        return false;
      }
      if (StripBroadcastAndReshape(PeelAllFloatConverts(bcast.getOperand())) !=
          rstd_raw) {
        return false;
      }
      rstd_bcast_1 = bcast;
      dy_gamma_f32 = cand_dy_gamma;
      return true;
    };

    if (!try_extract_term1(term1.getLhs(), term1.getRhs()) &&
        !try_extract_term1(term1.getRhs(), term1.getLhs())) {
      return mlir::failure();
    }

    // Verify dot_reduce reduces Mul(x, dy_gamma) (optionally wrapped in
    // ConvertOp by BuildReductionShlo).
    mlir::Value dot_red_in = PeelAllFloatConverts(dot_reduce.getInputs()[0]);
    auto dot_mul = dot_red_in.getDefiningOp<mlir::stablehlo::MulOp>();
    if (!dot_mul) return mlir::failure();

    mlir::Value x_base = PeelAllFloatConverts(x_f32);
    mlir::Value dy_gamma_raw = PeelAllFloatConverts(dy_gamma_f32);
    mlir::Value dot_lhs_base = PeelAllFloatConverts(dot_mul.getLhs());
    mlir::Value dot_rhs_base = PeelAllFloatConverts(dot_mul.getRhs());
    if (!((dot_lhs_base == x_base && dot_rhs_base == dy_gamma_raw) ||
          (dot_rhs_base == x_base && dot_lhs_base == dy_gamma_raw))) {
      return mlir::failure();
    }

    // Extract dy_src and optional weight_val from dy_gamma_raw.
    mlir::Value dy_src = dy_gamma_f32;
    mlir::Value weight_val;
    mlir::Type dyg_elem_type = full_type.getElementType();
    if (auto dyg_mul = dy_gamma_raw.getDefiningOp<mlir::stablehlo::MulOp>()) {
      auto dyg_type = mlir::cast<mlir::RankedTensorType>(dyg_mul.getType());
      auto try_match_weight =
          [&](mlir::Value cand_dy,  // REF_CAPTURE_OK=called locally
              mlir::Value cand_w) -> bool {
        mlir::Value w_peeled = PeelAllFloatConverts(cand_w);
        auto wb = w_peeled.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
        if (!IsValidGammaBroadcast(wb, dyg_type, batch_len, norm_dims)) {
          return false;
        }
        dy_src = cand_dy;
        weight_val = PeelSingleUpcastConvert(wb.getOperand());
        dyg_elem_type = dyg_type.getElementType();
        return true;
      };
      if (!try_match_weight(dyg_mul.getLhs(), dyg_mul.getRhs()) &&
          !try_match_weight(dyg_mul.getRhs(), dyg_mul.getLhs())) {
        // Not a multiplication with a broadcast weight.
      }
    }

    // Find companion dgamma = Reduce(Mul(dy, x_hat), batch_dims) reductions so
    // dgamma and a2 share the squeezed f32 (dy_sq, x_hat_sq).
    llvm::ArrayRef<int64_t> non_unit_batch_dims =
        llvm::ArrayRef(batch_dims).drop_front(drop_k);
    mlir::Value dy_base = PeelAllFloatConverts(dy_src);
    llvm::SmallVector<mlir::Value, 8> dy_cands;
    CollectTransitiveFloatConvertUsers(dy_base, dy_cands);

    llvm::SmallVector<mlir::stablehlo::ReduceOp, 2> dgamma_reduces;
    for (mlir::Value dy_cand : dy_cands) {
      for (mlir::Operation* user : dy_cand.getUsers()) {
        if (user->getBlock() != dx_sub->getBlock()) continue;
        auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(user);
        if (!mul) continue;
        mlir::Value other =
            (mul.getLhs() == dy_cand) ? mul.getRhs() : mul.getLhs();
        if (!IsEquivalentRmsNormXHat(other, x_f32, x_base, rstd_raw, full_type,
                                     batch_len, drop_k, batch_dims, all_dims)) {
          continue;
        }
        llvm::SmallVector<mlir::Value, 4> mul_cands;
        CollectTransitiveFloatConvertUsers(mul.getResult(), mul_cands);
        for (mlir::Value mul_cand : mul_cands) {
          for (mlir::Operation* mul_user : mul_cand.getUsers()) {
            if (mul_user->getBlock() != dx_sub->getBlock()) continue;
            if (auto red =
                    mlir::dyn_cast<mlir::stablehlo::ReduceOp>(mul_user)) {
              if ((IsSumReduceOverDims(red, batch_dims) ||
                   (drop_k > 0 &&
                    IsSumReduceOverDims(red, non_unit_batch_dims))) &&
                  !llvm::is_contained(dgamma_reduces, red)) {
                dgamma_reduces.push_back(red);
              }
            }
          }
        }
      }
    }

    // Set insertion point before the earliest of dx_sub and dgamma_reduces if
    // all required inputs precede it.
    mlir::Operation* earliest_op = dx_sub.getOperation();
    llvm::SmallVector<mlir::Value, 6> required_inputs = {
        PeelSingleUpcastConvert(x_f32), PeelSingleUpcastConvert(dy_src),
        rstd_raw, weight_val};
    for (mlir::stablehlo::ReduceOp red : dgamma_reduces) {
      if (red->getBlock() == earliest_op->getBlock() &&
          red->isBeforeInBlock(earliest_op)) {
        earliest_op = red.getOperation();
      }
    }
    if (earliest_op != dx_sub.getOperation()) {
      if (!AllInputsPrecedeOp(required_inputs, earliest_op)) {
        return mlir::failure();
      }
      rewriter.setInsertionPoint(earliest_op);
    } else {
      rewriter.setInsertionPoint(dx_sub);
    }

    // Build squeezed types and dimension lists.
    mlir::Location loc = dx_sub.getLoc();
    mlir::Type elem_type = full_type.getElementType();
    llvm::ArrayRef<int64_t> squeezed_shape =
        full_type.getShape().drop_front(drop_k);
    llvm::ArrayRef<int64_t> squeezed_batch_shape =
        full_type.getShape().slice(drop_k, batch_len - drop_k);
    llvm::ArrayRef<int64_t> squeezed_norm_shape =
        full_type.getShape().drop_front(batch_len);
    auto squeezed_type = mlir::RankedTensorType::get(squeezed_shape, elem_type);
    auto squeezed_batch_type =
        mlir::RankedTensorType::get(squeezed_batch_shape, elem_type);
    auto squeezed_norm_type =
        mlir::RankedTensorType::get(squeezed_norm_shape, elem_type);

    const int64_t sq_rank = rank - drop_k;
    const int64_t sq_batch_len = batch_len - drop_k;
    llvm::SmallVector<int64_t, 4> sq_batch_dims;
    llvm::SmallVector<int64_t, 4> sq_norm_dims;
    for (int64_t i = 0; i < sq_rank; ++i) {
      if (i < sq_batch_len) {
        sq_batch_dims.push_back(i);
      } else {
        sq_norm_dims.push_back(i);
      }
    }
    auto sq_batch_dims_attr = rewriter.getDenseI64ArrayAttr(sq_batch_dims);
    auto sq_norm_dims_attr = rewriter.getDenseI64ArrayAttr(sq_norm_dims);

    // Squeeze x, dy, rstd and compute shared x_hat_sq = x_sq * rstd_bcast_sq.
    mlir::Value x_sq =
        SqueezeInputTensor(rewriter, loc, x_f32, full_type, squeezed_type);

    mlir::Value rstd_sq = rstd_raw;
    if (rstd_raw.getType() != squeezed_batch_type) {
      rstd_sq = rewriter.create<mlir::stablehlo::ReshapeOp>(
          loc, squeezed_batch_type, rstd_raw);
    }
    auto rstd_bcast_sq = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, squeezed_type, rstd_sq, sq_batch_dims_attr);
    auto x_hat_sq = rewriter.create<mlir::stablehlo::MulOp>(
        loc, squeezed_type, x_sq, rstd_bcast_sq);

    mlir::Value dy_sq;
    mlir::Value dy_gamma_sq;
    if (weight_val && dyg_elem_type != elem_type) {
      auto full_dyg_type =
          mlir::RankedTensorType::get(full_type.getShape(), dyg_elem_type);
      auto squeezed_dyg_type =
          mlir::RankedTensorType::get(squeezed_shape, dyg_elem_type);
      auto squeezed_norm_dyg_type =
          mlir::RankedTensorType::get(squeezed_norm_shape, dyg_elem_type);
      mlir::Value dy_sq_raw = SqueezeInputTensor(
          rewriter, loc, dy_src, full_dyg_type, squeezed_dyg_type);
      dy_sq = rewriter.create<mlir::stablehlo::ConvertOp>(loc, squeezed_type,
                                                          dy_sq_raw);
      mlir::Value w_raw = weight_val;
      auto w_type = mlir::cast<mlir::RankedTensorType>(weight_val.getType());
      if (w_type.getElementType() != dyg_elem_type) {
        auto w_raw_type =
            mlir::RankedTensorType::get(w_type.getShape(), dyg_elem_type);
        w_raw = rewriter.create<mlir::stablehlo::ConvertOp>(loc, w_raw_type,
                                                            weight_val);
      }
      if (w_raw.getType() != squeezed_norm_dyg_type) {
        w_raw = rewriter.create<mlir::stablehlo::ReshapeOp>(
            loc, squeezed_norm_dyg_type, w_raw);
      }
      auto w_bcast_sq_raw = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, squeezed_dyg_type, w_raw, sq_norm_dims_attr);
      auto dy_gamma_sq_raw = rewriter.create<mlir::stablehlo::MulOp>(
          loc, squeezed_dyg_type, dy_sq_raw, w_bcast_sq_raw);
      dy_gamma_sq = rewriter.create<mlir::stablehlo::ConvertOp>(
          loc, squeezed_type, dy_gamma_sq_raw);
    } else {
      dy_sq =
          SqueezeInputTensor(rewriter, loc, dy_src, full_type, squeezed_type);
      dy_gamma_sq = dy_sq;
      if (weight_val) {
        mlir::Value w_f32 = weight_val;
        auto w_type = mlir::cast<mlir::RankedTensorType>(weight_val.getType());
        if (w_type.getElementType() != elem_type) {
          auto w_f32_type =
              mlir::RankedTensorType::get(w_type.getShape(), elem_type);
          w_f32 = rewriter.create<mlir::stablehlo::ConvertOp>(loc, w_f32_type,
                                                              weight_val);
        }
        if (w_f32.getType() != squeezed_norm_type) {
          w_f32 = rewriter.create<mlir::stablehlo::ReshapeOp>(
              loc, squeezed_norm_type, w_f32);
        }
        auto w_bcast_sq = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, squeezed_type, w_f32, sq_norm_dims_attr);
        dy_gamma_sq = rewriter.create<mlir::stablehlo::MulOp>(
            loc, squeezed_type, dy_sq, w_bcast_sq);
      }
    }

    if (!dgamma_reduces.empty()) {
      auto dgamma_full_sq = rewriter.create<mlir::stablehlo::MulOp>(
          loc, squeezed_type, dy_sq, x_hat_sq);
      for (mlir::stablehlo::ReduceOp dgamma_red : dgamma_reduces) {
        ReplaceWithSqueezedSumReduce(rewriter, dgamma_red, dgamma_full_sq,
                                     sq_batch_dims, squeezed_norm_type);
      }
    }

    auto a2_full_sq = rewriter.create<mlir::stablehlo::MulOp>(
        loc, squeezed_type, dy_gamma_sq, x_hat_sq);
    mlir::Value a2_sq = BuildFloatSumReduce(rewriter, loc, a2_full_sq,
                                            sq_norm_dims, squeezed_batch_type);

    const double inv_n = 1.0 / static_cast<double>(norm_numel);
    mlir::Value corr_sq;
    // GENERIC: semantic XLA RwbPreliminaryCandidateCheck only rewrites
    // Reduce->Broadcast into ReduceWindow for rank > 2; in rank 2, multiplying
    // by 1/N after BroadcastInDim fuses the broadcast into dx.
    if (sq_rank == 2) {
      auto a2_bcast_sq = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, squeezed_type, a2_sq, sq_batch_dims_attr);
      auto x_hat_a2_sq = rewriter.create<mlir::stablehlo::MulOp>(
          loc, squeezed_type, x_hat_sq, a2_bcast_sq);
      mlir::Value inv_n_2d =
          MakeBroadcastFloatConstant(rewriter, loc, squeezed_type, inv_n);
      corr_sq = rewriter.create<mlir::stablehlo::MulOp>(loc, squeezed_type,
                                                        x_hat_a2_sq, inv_n_2d);
    } else {
      mlir::Value inv_n_batch =
          MakeBroadcastFloatConstant(rewriter, loc, squeezed_batch_type, inv_n);
      auto a2_scaled_sq = rewriter.create<mlir::stablehlo::MulOp>(
          loc, squeezed_batch_type, a2_sq, inv_n_batch);
      auto a2_bcast_sq = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, squeezed_type, a2_scaled_sq, sq_batch_dims_attr);
      corr_sq = rewriter.create<mlir::stablehlo::MulOp>(loc, squeezed_type,
                                                        x_hat_sq, a2_bcast_sq);
    }

    auto diff_sq = rewriter.create<mlir::stablehlo::SubtractOp>(
        loc, squeezed_type, dy_gamma_sq, corr_sq);
    auto dx_sq = rewriter.create<mlir::stablehlo::MulOp>(
        loc, squeezed_type, diff_sq, rstd_bcast_sq);

    if (drop_k > 0) {
      llvm::SmallVector<mlir::stablehlo::ConvertOp, 2> dx_converts;
      for (mlir::Operation* user : dx_sub->getUsers()) {
        if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(user)) {
          if (!llvm::is_contained(dx_converts, conv)) {
            dx_converts.push_back(conv);
          }
        }
      }
      for (mlir::stablehlo::ConvertOp conv : dx_converts) {
        auto conv_type = mlir::dyn_cast<mlir::RankedTensorType>(conv.getType());
        if (!conv_type) continue;
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(conv);
        auto sq_conv_type = mlir::RankedTensorType::get(
            squeezed_shape, conv_type.getElementType());
        auto conv_sq = rewriter.create<mlir::stablehlo::ConvertOp>(
            conv.getLoc(), sq_conv_type, dx_sq);
        auto reshaped_conv = rewriter.create<mlir::stablehlo::ReshapeOp>(
            conv.getLoc(), conv_type, conv_sq);
        rewriter.replaceOp(conv, reshaped_conv);
      }
    }

    if (!dx_sub.use_empty()) {
      mlir::Value replacement = dx_sq;
      if (drop_k > 0) {
        replacement =
            rewriter.create<mlir::stablehlo::ReshapeOp>(loc, full_type, dx_sq);
      }
      rewriter.replaceOp(dx_sub, replacement);
    } else {
      rewriter.eraseOp(dx_sub);
    }
    return mlir::success();
  }
};

// =============================================================================
// Pattern 2: Canonicalize Leading-Unit-Batch LayerNorm Backward
// (`native_layer_norm_backward`), Including AdaLN Scale/Shift Modulation
//
// Squeezes leading unit batch dimensions (e.g. [1, ..., 1, S, D] -> [S, D])
// across `x`, `dy`, `mean`, `rstd`, `dbeta`/`dshift`, `dgamma`/`dscale`,
// `a1`, `a2`, and `dx` so all four LayerNorm backward reductions operate at the
// same rank and fuse into a single multi-output reduction fusion.
// =============================================================================
struct CanonicalizeLayerNormBackwardPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::MulOp> {
  using OpRewritePattern<mlir::stablehlo::MulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::MulOp dx_mul,
      mlir::PatternRewriter& rewriter) const override {
    auto full_type = mlir::dyn_cast<mlir::RankedTensorType>(dx_mul.getType());
    if (!full_type || !full_type.hasStaticShape() ||
        // GENERIC: semantic Require rank >= 3 so squeezing at least one
        // leading unit batch dim leaves >= 1 batch dim and >= 1 norm dim.
        !mlir::isa<mlir::FloatType>(full_type.getElementType()) ||
        full_type.getRank() < 3) {
      return mlir::failure();
    }
    const int64_t rank = full_type.getRank();

    // Match dx_mul = Mul(dx_sub2, rstd_bcast) in either operand order.
    mlir::stablehlo::SubtractOp dx_sub2;
    mlir::stablehlo::BroadcastInDimOp rstd_bcast;
    if ((dx_sub2 =
             dx_mul.getLhs().getDefiningOp<mlir::stablehlo::SubtractOp>()) &&
        (rstd_bcast =
             dx_mul.getRhs()
                 .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>())) {
    } else if ((dx_sub2 = dx_mul.getRhs()
                              .getDefiningOp<mlir::stablehlo::SubtractOp>()) &&
               (rstd_bcast =
                    dx_mul.getLhs()
                        .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>())) {
    } else {
      return mlir::failure();
    }

    // Match dx_sub2 = Subtract(dx_sub1, x_hat_a2) and
    // dx_sub1 = Subtract(dy_gamma, a1_bcast).
    auto dx_sub1 =
        dx_sub2.getLhs().getDefiningOp<mlir::stablehlo::SubtractOp>();
    auto x_hat_a2 = dx_sub2.getRhs().getDefiningOp<mlir::stablehlo::MulOp>();
    if (!dx_sub1 || !x_hat_a2) return mlir::failure();

    mlir::Value dy_gamma = dx_sub1.getLhs();
    auto a1_bcast =
        dx_sub1.getRhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!a1_bcast) return mlir::failure();

    llvm::ArrayRef<int64_t> batch_dims = a1_bcast.getBroadcastDimensions();
    const int64_t batch_len = batch_dims.size();
    // GENERIC: semantic At least 2 batch dims are required so squeezing a
    // leading unit batch dim leaves >= 1 batch dim and >= 1 norm dim.
    if (batch_len < 2 || batch_len >= rank) return mlir::failure();
    for (int64_t i = 0; i < batch_len; ++i) {
      if (batch_dims[i] != i) return mlir::failure();
    }

    int64_t drop_k = 0;
    // GENERIC: semantic Squeeze leading unit batch dims while keeping >= 1
    // batch dim.
    while (drop_k < batch_len - 1 && full_type.getDimSize(drop_k) == 1) {
      ++drop_k;
    }
    if (drop_k == 0) return mlir::failure();

    llvm::SmallVector<int64_t, 4> norm_dims;
    llvm::SmallVector<int64_t, 6> all_dims;
    for (int64_t i = 0; i < rank; ++i) {
      all_dims.push_back(i);
      if (i >= batch_len) norm_dims.push_back(i);
    }

    // Match x_hat_a2 = Mul(x_hat, a2_bcast) in either operand order.
    mlir::Value x_hat;
    mlir::stablehlo::BroadcastInDimOp a2_bcast;
    if (auto b = x_hat_a2.getRhs()
                     .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
        b && b.getBroadcastDimensions() == batch_dims) {
      a2_bcast = b;
      x_hat = x_hat_a2.getLhs();
    } else if (auto b2 =
                   x_hat_a2.getLhs()
                       .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
               b2 && b2.getBroadcastDimensions() == batch_dims) {
      a2_bcast = b2;
      x_hat = x_hat_a2.getRhs();
    } else {
      return mlir::failure();
    }

    // Match a1_scaled = Mul(a1_reduce(dy_gamma), inv_n) and
    // a2_scaled = Mul(a2_reduce(Mul(dy_gamma, x_hat)), inv_n).
    auto a1_match = MatchScaledSumReduce(a1_bcast.getOperand(), norm_dims);
    auto a2_match = MatchScaledSumReduce(a2_bcast.getOperand(), norm_dims);
    if (!a1_match || !a2_match) return mlir::failure();
    if (a1_match->reduced_input != dy_gamma) return mlir::failure();

    auto a2_full_mul =
        a2_match->reduced_input.getDefiningOp<mlir::stablehlo::MulOp>();
    if (!a2_full_mul) return mlir::failure();
    if (!((a2_full_mul.getLhs() == dy_gamma && a2_full_mul.getRhs() == x_hat) ||
          (a2_full_mul.getRhs() == dy_gamma &&
           a2_full_mul.getLhs() == x_hat))) {
      return mlir::failure();
    }

    // Match x_hat = Mul(Subtract(x_f32, mean_bcast), rstd_bcast).
    auto x_hat_mul = x_hat.getDefiningOp<mlir::stablehlo::MulOp>();
    if (!x_hat_mul) return mlir::failure();

    mlir::Value rstd_raw =
        StripBroadcastAndReshape(PeelAllFloatConverts(rstd_bcast.getOperand()));
    mlir::stablehlo::SubtractOp x_sub;
    mlir::stablehlo::BroadcastInDimOp rstd_bcast_xhat;
    if ((x_sub =
             x_hat_mul.getLhs().getDefiningOp<mlir::stablehlo::SubtractOp>()) &&
        (rstd_bcast_xhat =
             x_hat_mul.getRhs()
                 .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>())) {
    } else if ((x_sub = x_hat_mul.getRhs()
                            .getDefiningOp<mlir::stablehlo::SubtractOp>()) &&
               (rstd_bcast_xhat =
                    x_hat_mul.getLhs()
                        .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>())) {
    } else {
      return mlir::failure();
    }

    if (StripBroadcastAndReshape(
            PeelAllFloatConverts(rstd_bcast_xhat.getOperand())) != rstd_raw) {
      return mlir::failure();
    }

    mlir::Value x_f32 = x_sub.getLhs();
    auto mean_bcast =
        x_sub.getRhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!IsValidStatsBroadcast(mean_bcast, full_type, batch_len, drop_k,
                               batch_dims, all_dims) ||
        !IsValidStatsBroadcast(rstd_bcast, full_type, batch_len, drop_k,
                               batch_dims, all_dims)) {
      return mlir::failure();
    }
    mlir::Value mean_raw =
        StripBroadcastAndReshape(PeelAllFloatConverts(mean_bcast.getOperand()));

    // Match optional gamma_bcast in dy_gamma = Mul(dy, gamma_bcast), supporting
    // both 1D gamma (dims={2}) and AdaLN scale ([1, D] or [1, 1, D]).
    mlir::Value dy_gamma_raw = PeelAllFloatConverts(dy_gamma);
    mlir::Value dy_src = dy_gamma;
    mlir::Value gamma_val;
    if (auto dy_mul = dy_gamma_raw.getDefiningOp<mlir::stablehlo::MulOp>()) {
      auto dyg_type = mlir::cast<mlir::RankedTensorType>(dy_mul.getType());
      auto try_match_gamma =
          [&](mlir::Value cand_dy,  // REF_CAPTURE_OK=called locally
              mlir::Value cand_gamma) -> bool {
        mlir::Value g_peeled = PeelAllFloatConverts(cand_gamma);
        auto gb = g_peeled.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
        if (!IsValidGammaBroadcast(gb, dyg_type, batch_len, norm_dims)) {
          return false;
        }
        dy_src = cand_dy;
        gamma_val = PeelSingleUpcastConvert(gb.getOperand());
        return true;
      };
      if (!try_match_gamma(dy_mul.getLhs(), dy_mul.getRhs()) &&
          !try_match_gamma(dy_mul.getRhs(), dy_mul.getLhs())) {
        // dy_gamma is not a multiplication with gamma broadcast.
      }
    }

    // Find companion dbeta/dshift = Reduce(dy) and
    // dgamma/dscale = Reduce(Mul(dy, x_hat_equiv)) reductions over either
    // `batch_dims` or `batch_dims.drop_front(drop_k)`.
    llvm::ArrayRef<int64_t> non_unit_batch_dims = batch_dims.drop_front(drop_k);
    mlir::Value dy_base = PeelAllFloatConverts(dy_src);
    llvm::SmallVector<mlir::Value, 8> dy_cands;
    CollectTransitiveFloatConvertUsers(dy_base, dy_cands);

    llvm::SmallVector<mlir::stablehlo::ReduceOp, 2> dbeta_reduces;
    llvm::SmallVector<mlir::stablehlo::ReduceOp, 2> dgamma_reduces;
    for (mlir::Value dy_cand : dy_cands) {
      for (mlir::Operation* user : dy_cand.getUsers()) {
        if (user->getBlock() != dx_mul->getBlock()) continue;
        if (auto red = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(user)) {
          if ((IsSumReduceOverDims(red, batch_dims) ||
               IsSumReduceOverDims(red, non_unit_batch_dims)) &&
              !llvm::is_contained(dbeta_reduces, red)) {
            dbeta_reduces.push_back(red);
          }
        } else if (auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(user)) {
          mlir::Value other =
              (mul.getLhs() == dy_cand) ? mul.getRhs() : mul.getLhs();
          if (!IsEquivalentLayerNormXHat(other, x_hat, x_f32, mean_raw,
                                         rstd_raw, full_type, batch_len, drop_k,
                                         batch_dims, all_dims)) {
            continue;
          }
          llvm::SmallVector<mlir::Value, 4> mul_cands;
          CollectTransitiveFloatConvertUsers(mul.getResult(), mul_cands);
          for (mlir::Value mul_cand : mul_cands) {
            for (mlir::Operation* mul_user : mul_cand.getUsers()) {
              if (mul_user->getBlock() != dx_mul->getBlock()) continue;
              if (auto red =
                      mlir::dyn_cast<mlir::stablehlo::ReduceOp>(mul_user)) {
                if ((IsSumReduceOverDims(red, batch_dims) ||
                     IsSumReduceOverDims(red, non_unit_batch_dims)) &&
                    !llvm::is_contained(dgamma_reduces, red)) {
                  dgamma_reduces.push_back(red);
                }
              }
            }
          }
        }
      }
    }

    // Set insertion point before the earliest of dx_mul, dbeta_reduces, and
    // dgamma_reduces if all required inputs precede it.
    mlir::Operation* earliest_op = dx_mul.getOperation();
    llvm::SmallVector<mlir::Value, 6> required_inputs = {
        PeelSingleUpcastConvert(x_f32), PeelSingleUpcastConvert(dy_src),
        mean_raw, rstd_raw, gamma_val};
    for (mlir::stablehlo::ReduceOp red : dbeta_reduces) {
      if (red->getBlock() == earliest_op->getBlock() &&
          red->isBeforeInBlock(earliest_op)) {
        earliest_op = red.getOperation();
      }
    }
    for (mlir::stablehlo::ReduceOp red : dgamma_reduces) {
      if (red->getBlock() == earliest_op->getBlock() &&
          red->isBeforeInBlock(earliest_op)) {
        earliest_op = red.getOperation();
      }
    }
    if (earliest_op != dx_mul.getOperation()) {
      if (!AllInputsPrecedeOp(required_inputs, earliest_op)) {
        return mlir::failure();
      }
      rewriter.setInsertionPoint(earliest_op);
    } else {
      rewriter.setInsertionPoint(dx_mul);
    }

    // Build squeezed types and dimension lists.
    mlir::Location loc = dx_mul.getLoc();
    mlir::Type elem_type = full_type.getElementType();
    llvm::ArrayRef<int64_t> squeezed_shape =
        full_type.getShape().drop_front(drop_k);
    llvm::ArrayRef<int64_t> squeezed_batch_shape =
        full_type.getShape().slice(drop_k, batch_len - drop_k);
    llvm::ArrayRef<int64_t> squeezed_norm_shape =
        full_type.getShape().drop_front(batch_len);
    auto squeezed_type = mlir::RankedTensorType::get(squeezed_shape, elem_type);
    auto squeezed_batch_type =
        mlir::RankedTensorType::get(squeezed_batch_shape, elem_type);
    auto squeezed_norm_type =
        mlir::RankedTensorType::get(squeezed_norm_shape, elem_type);

    const int64_t sq_rank = rank - drop_k;
    const int64_t sq_batch_len = batch_len - drop_k;
    llvm::SmallVector<int64_t, 4> sq_batch_dims;
    llvm::SmallVector<int64_t, 4> sq_norm_dims;
    for (int64_t i = 0; i < sq_rank; ++i) {
      if (i < sq_batch_len) {
        sq_batch_dims.push_back(i);
      } else {
        sq_norm_dims.push_back(i);
      }
    }

    // Squeeze inputs x, dy, mean, rstd.
    mlir::Value x_sq =
        SqueezeInputTensor(rewriter, loc, x_f32, full_type, squeezed_type);
    mlir::Value dy_sq =
        SqueezeInputTensor(rewriter, loc, dy_src, full_type, squeezed_type);
    mlir::Value mean_sq = mean_raw;
    if (mean_raw.getType() != squeezed_batch_type) {
      mean_sq = rewriter.create<mlir::stablehlo::ReshapeOp>(
          loc, squeezed_batch_type, mean_raw);
    }
    mlir::Value rstd_sq = rstd_raw;
    if (rstd_raw.getType() != squeezed_batch_type) {
      rstd_sq = rewriter.create<mlir::stablehlo::ReshapeOp>(
          loc, squeezed_batch_type, rstd_raw);
    }

    auto sq_batch_dims_attr = rewriter.getDenseI64ArrayAttr(sq_batch_dims);
    auto sq_norm_dims_attr = rewriter.getDenseI64ArrayAttr(sq_norm_dims);

    auto mean_bcast_sq = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, squeezed_type, mean_sq, sq_batch_dims_attr);
    auto rstd_bcast_sq = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, squeezed_type, rstd_sq, sq_batch_dims_attr);

    auto x_sub_sq = rewriter.create<mlir::stablehlo::SubtractOp>(
        loc, squeezed_type, x_sq, mean_bcast_sq);
    auto x_hat_sq = rewriter.create<mlir::stablehlo::MulOp>(
        loc, squeezed_type, x_sub_sq, rstd_bcast_sq);

    mlir::Value dy_gamma_sq = dy_sq;
    if (gamma_val) {
      mlir::Value gamma_f32 = gamma_val;
      auto gamma_type = mlir::cast<mlir::RankedTensorType>(gamma_val.getType());
      if (gamma_type.getElementType() != elem_type) {
        auto gamma_f32_type =
            mlir::RankedTensorType::get(gamma_type.getShape(), elem_type);
        gamma_f32 = rewriter.create<mlir::stablehlo::ConvertOp>(
            loc, gamma_f32_type, gamma_val);
      }
      mlir::Value gamma_sq = gamma_f32;
      if (gamma_f32.getType() != squeezed_norm_type) {
        gamma_sq = rewriter.create<mlir::stablehlo::ReshapeOp>(
            loc, squeezed_norm_type, gamma_f32);
      }
      auto gamma_bcast_sq = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, squeezed_type, gamma_sq, sq_norm_dims_attr);
      dy_gamma_sq = rewriter.create<mlir::stablehlo::MulOp>(
          loc, squeezed_type, dy_sq, gamma_bcast_sq);
    }

    for (mlir::stablehlo::ReduceOp dbeta_red : dbeta_reduces) {
      ReplaceWithSqueezedSumReduce(rewriter, dbeta_red, dy_sq, sq_batch_dims,
                                   squeezed_norm_type);
    }

    if (!dgamma_reduces.empty()) {
      auto dgamma_full_sq = rewriter.create<mlir::stablehlo::MulOp>(
          loc, squeezed_type, dy_sq, x_hat_sq);
      for (mlir::stablehlo::ReduceOp dgamma_red : dgamma_reduces) {
        ReplaceWithSqueezedSumReduce(rewriter, dgamma_red, dgamma_full_sq,
                                     sq_batch_dims, squeezed_norm_type);
      }
    }

    // Build squeezed a1, a2, and dx.
    mlir::Value a1_sq = BuildFloatSumReduce(rewriter, loc, dy_gamma_sq,
                                            sq_norm_dims, squeezed_batch_type);
    auto a2_full_sq = rewriter.create<mlir::stablehlo::MulOp>(
        loc, squeezed_type, dy_gamma_sq, x_hat_sq);
    mlir::Value a2_sq = BuildFloatSumReduce(rewriter, loc, a2_full_sq,
                                            sq_norm_dims, squeezed_batch_type);

    mlir::Value dx_sub2_sq;
    const bool same_scale =
        std::abs(a1_match->scale_val - a2_match->scale_val) <=
        1e-6 * std::abs(a1_match->scale_val);
    // GENERIC: semantic XLA RwbPreliminaryCandidateCheck only rewrites
    // Reduce->Broadcast into ReduceWindow for rank > 2; in rank 2, factoring
    // the 1/N scale after BroadcastInDim fuses both broadcasts into dx.
    if (sq_rank == 2 && same_scale) {
      auto a1_bcast_sq = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, squeezed_type, a1_sq, sq_batch_dims_attr);
      auto a2_bcast_sq = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, squeezed_type, a2_sq, sq_batch_dims_attr);
      auto x_hat_a2_sq = rewriter.create<mlir::stablehlo::MulOp>(
          loc, squeezed_type, x_hat_sq, a2_bcast_sq);
      auto sum_corr_sq = rewriter.create<mlir::stablehlo::AddOp>(
          loc, squeezed_type, a1_bcast_sq, x_hat_a2_sq);
      mlir::Value inv_n_2d = MakeBroadcastFloatConstant(
          rewriter, loc, squeezed_type, a1_match->scale_val);
      auto corr_sq = rewriter.create<mlir::stablehlo::MulOp>(
          loc, squeezed_type, sum_corr_sq, inv_n_2d);
      dx_sub2_sq = rewriter.create<mlir::stablehlo::SubtractOp>(
          loc, squeezed_type, dy_gamma_sq, corr_sq);
    } else {
      mlir::Value inv_n_1_sq = MakeBroadcastFloatConstant(
          rewriter, loc, squeezed_batch_type, a1_match->scale_val);
      mlir::Value inv_n_2_sq =
          same_scale
              ? inv_n_1_sq
              : MakeBroadcastFloatConstant(rewriter, loc, squeezed_batch_type,
                                           a2_match->scale_val);
      auto a1_scaled_sq = rewriter.create<mlir::stablehlo::MulOp>(
          loc, squeezed_batch_type, a1_sq, inv_n_1_sq);
      auto a2_scaled_sq = rewriter.create<mlir::stablehlo::MulOp>(
          loc, squeezed_batch_type, a2_sq, inv_n_2_sq);
      auto a1_bcast_sq = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, squeezed_type, a1_scaled_sq, sq_batch_dims_attr);
      auto a2_bcast_sq = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, squeezed_type, a2_scaled_sq, sq_batch_dims_attr);
      auto dx_sub1_sq = rewriter.create<mlir::stablehlo::SubtractOp>(
          loc, squeezed_type, dy_gamma_sq, a1_bcast_sq);
      auto x_hat_a2_sq = rewriter.create<mlir::stablehlo::MulOp>(
          loc, squeezed_type, x_hat_sq, a2_bcast_sq);
      dx_sub2_sq = rewriter.create<mlir::stablehlo::SubtractOp>(
          loc, squeezed_type, dx_sub1_sq, x_hat_a2_sq);
    }
    auto dx_sq = rewriter.create<mlir::stablehlo::MulOp>(
        loc, squeezed_type, dx_sub2_sq, rstd_bcast_sq);

    llvm::SmallVector<mlir::stablehlo::ConvertOp, 2> dx_converts;
    for (mlir::Operation* user : dx_mul->getUsers()) {
      if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(user)) {
        if (!llvm::is_contained(dx_converts, conv)) {
          dx_converts.push_back(conv);
        }
      }
    }
    for (mlir::stablehlo::ConvertOp conv : dx_converts) {
      auto conv_type = mlir::dyn_cast<mlir::RankedTensorType>(conv.getType());
      if (!conv_type) continue;
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(conv);
      auto sq_conv_type = mlir::RankedTensorType::get(
          squeezed_shape, conv_type.getElementType());
      auto conv_sq = rewriter.create<mlir::stablehlo::ConvertOp>(
          conv.getLoc(), sq_conv_type, dx_sq);
      auto reshaped_conv = rewriter.create<mlir::stablehlo::ReshapeOp>(
          conv.getLoc(), conv_type, conv_sq);
      rewriter.replaceOp(conv, reshaped_conv);
    }

    if (!dx_mul.use_empty()) {
      auto dx_reshaped =
          rewriter.create<mlir::stablehlo::ReshapeOp>(loc, full_type, dx_sq);
      rewriter.replaceOp(dx_mul, dx_reshaped);
    } else {
      rewriter.eraseOp(dx_mul);
    }
    return mlir::success();
  }
};

struct CanonicalizeRmsNormAndLayerNormBackwardPass
    : public mlir::PassWrapper<CanonicalizeRmsNormAndLayerNormBackwardPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CanonicalizeRmsNormAndLayerNormBackwardPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-canonicalize-rms-norm-and-layer-norm-backward";
  }
  llvm::StringRef getDescription() const override {
    return "Canonicalizes decomposed RMSNorm backward and leading-unit-batch "
           "LayerNorm backward subgraphs.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<CanonicalizeRmsNormBackwardPattern,
                 CanonicalizeLayerNormBackwardPattern>(context);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createCanonicalizeRmsNormAndLayerNormBackwardPass() {
  return std::make_unique<CanonicalizeRmsNormAndLayerNormBackwardPass>();
}

}  // namespace torch_tpu
