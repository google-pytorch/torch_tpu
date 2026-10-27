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

#include "csrc/mlir/passes/canonicalize_pointwise_conv_and_finite_sdpa_biases.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// GENERIC: hardware TPU TensorCore sublane count is 8; T <= 8 avoids
// Large2ndMinor T(16, 128) relayouts and has <= 1 SpaceToBatch step.
constexpr int64_t kTpuSublaneCount = 8;

// GENERIC: hardware TPU TensorCore lane and MXU systolic width is 128; T a
// multiple of 128 fills full 128-row MXU passes so MSA prefetches weights.
constexpr int64_t kTpuLaneWidth = 128;

// GENERIC: capability bounded recursion depth for SSA definition-chain walks.
constexpr int kMaxWalkDepth = 32;

// GENERIC: semantic fixed-point iteration cap for local canonicalizations.
constexpr int kMaxPassIterations = 6;

bool IsStaticRankedTensor(mlir::Type type,
                          mlir::RankedTensorType* out = nullptr) {
  auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!ranked || !ranked.hasStaticShape()) {
    return false;
  }
  if (out != nullptr) {
    *out = ranked;
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

mlir::Value PeelIdentityBroadcast(mlir::Value val) {
  while (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    mlir::RankedTensorType in_t;
    mlir::RankedTensorType out_t;
    if (!IsStaticRankedTensor(bcast.getOperand().getType(), &in_t) ||
        !IsStaticRankedTensor(bcast.getType(), &out_t) ||
        in_t.getShape() != out_t.getShape() ||
        !IsIdentityBroadcastDims(bcast.getBroadcastDimensions(),
                                 out_t.getRank())) {
      break;
    }
    val = bcast.getOperand();
  }
  return val;
}

std::optional<double> ExtractSplatFloatValue(mlir::Value val, int depth = 0) {
  if (depth > kMaxWalkDepth) {
    return std::nullopt;
  }
  val = PeelIdentityBroadcast(val);
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    return ExtractSplatFloatValue(bcast.getOperand(), depth + 1);
  }
  if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    return ExtractSplatFloatValue(resh.getOperand(), depth + 1);
  }
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    auto in_val = ExtractSplatFloatValue(conv.getOperand(), depth + 1);
    if (!in_val.has_value()) {
      return std::nullopt;
    }
    if (auto r_t = mlir::dyn_cast<mlir::RankedTensorType>(conv.getType())) {
      if (auto flt_t = mlir::dyn_cast<mlir::FloatType>(r_t.getElementType())) {
        llvm::APFloat apf(*in_val);
        bool loses_info = false;
        apf.convert(flt_t.getFloatSemantics(),
                    llvm::APFloat::rmNearestTiesToEven, &loses_info);
        if (!apf.isFinite()) {
          return std::nullopt;
        }
        return apf.convertToDouble();
      }
    }
    return in_val;
  }
  auto cst = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!cst) {
    return std::nullopt;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat() ||
      !mlir::isa<mlir::FloatType>(dense.getElementType())) {
    return std::nullopt;
  }
  return dense.getSplatValue<llvm::APFloat>().convertToDouble();
}

bool IsFloatConstantCloseTo(mlir::Value val, double target, double rel_tol) {
  auto v = ExtractSplatFloatValue(val);
  if (!v.has_value() || !std::isfinite(*v)) {
    return false;
  }
  return std::abs(*v - target) <= rel_tol * std::max(1.0, std::abs(target));
}

bool IsNegativeInfinitySplat(mlir::Value val, int depth = 0) {
  if (depth > kMaxWalkDepth) {
    return false;
  }
  val = PeelIdentityBroadcast(val);
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    return IsNegativeInfinitySplat(bcast.getOperand(), depth + 1);
  }
  if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    return IsNegativeInfinitySplat(resh.getOperand(), depth + 1);
  }
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    return IsNegativeInfinitySplat(conv.getOperand(), depth + 1);
  }
  auto cst = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!cst) {
    return false;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat() ||
      !mlir::isa<mlir::FloatType>(dense.getElementType())) {
    return false;
  }
  llvm::APFloat apf = dense.getSplatValue<llvm::APFloat>();
  return apf.isNegInfinity();
}

bool IsLowestFiniteFloatSplat(mlir::Value val, int depth = 0) {
  if (depth > kMaxWalkDepth) {
    return false;
  }
  val = PeelIdentityBroadcast(val);
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    return IsLowestFiniteFloatSplat(bcast.getOperand(), depth + 1);
  }
  if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    return IsLowestFiniteFloatSplat(resh.getOperand(), depth + 1);
  }
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    return IsLowestFiniteFloatSplat(conv.getOperand(), depth + 1);
  }
  auto cst = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!cst) {
    return false;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat()) {
    return false;
  }
  auto flt_type = mlir::dyn_cast<mlir::FloatType>(dense.getElementType());
  if (!flt_type) {
    return false;
  }
  llvm::APFloat apf = dense.getSplatValue<llvm::APFloat>();
  if (!apf.isFinite() || !apf.isNegative()) {
    return false;
  }
  llvm::APFloat largest_neg = llvm::APFloat::getLargest(
      flt_type.getFloatSemantics(), /*Negative=*/true);
  // Also recognize f32 lowest cast or rounded to bf16/f32 (< -1e37).
  double d = apf.convertToDouble();
  return apf.bitwiseIsEqual(largest_neg) || d <= -1.0e37;
}

bool IsBoolSplatConstant(mlir::Value val, bool expected, int depth = 0) {
  if (depth > kMaxWalkDepth) {
    return false;
  }
  val = PeelIdentityBroadcast(val);
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    return IsBoolSplatConstant(bcast.getOperand(), expected, depth + 1);
  }
  if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    return IsBoolSplatConstant(resh.getOperand(), expected, depth + 1);
  }
  auto cst = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!cst) {
    return false;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat() || !dense.getElementType().isInteger(1)) {
    return false;
  }
  return dense.getSplatValue<bool>() == expected;
}

mlir::Value CreateBoolSplatConstant(mlir::OpBuilder& builder,
                                    mlir::Location loc,
                                    mlir::RankedTensorType type, bool value) {
  auto i1_type = builder.getI1Type();
  auto out_type = mlir::RankedTensorType::get(type.getShape(), i1_type);
  auto attr = mlir::DenseElementsAttr::get(out_type, value);
  return builder.create<mlir::stablehlo::ConstantOp>(loc, out_type, attr);
}

static bool TryStepBroadcastDim(mlir::stablehlo::BroadcastInDimOp bcast,
                                int64_t cur_dim, int64_t expected_len,
                                mlir::Value& out_cur, int64_t& out_next_dim) {
  mlir::RankedTensorType in_t;
  if (!IsStaticRankedTensor(bcast.getOperand().getType(), &in_t)) {
    return false;
  }
  llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
  int64_t next_dim = -1;
  for (int64_t i = 0; i < static_cast<int64_t>(bdims.size()); ++i) {
    if (bdims[i] == cur_dim) {
      next_dim = i;
    } else if (in_t.getDimSize(i) != 1) {
      return false;
    }
  }
  if (next_dim < 0 || in_t.getDimSize(next_dim) != expected_len) {
    return false;
  }
  out_cur = bcast.getOperand();
  out_next_dim = next_dim;
  return true;
}

static bool TryStepReshapeDim(mlir::stablehlo::ReshapeOp resh,
                              mlir::RankedTensorType cur_t, int64_t cur_dim,
                              int64_t expected_len, mlir::Value& out_cur,
                              int64_t& out_next_dim) {
  mlir::RankedTensorType in_t;
  if (!IsStaticRankedTensor(resh.getOperand().getType(), &in_t)) {
    return false;
  }
  for (int64_t i = 0; i < cur_t.getRank(); ++i) {
    if (i != cur_dim && cur_t.getDimSize(i) != 1) {
      return false;
    }
  }
  int64_t next_dim = -1;
  for (int64_t i = 0; i < in_t.getRank(); ++i) {
    if (in_t.getDimSize(i) == expected_len && next_dim < 0) {
      next_dim = i;
    } else if (in_t.getDimSize(i) != 1) {
      return false;
    }
  }
  if (next_dim < 0) {
    return false;
  }
  out_cur = resh.getOperand();
  out_next_dim = next_dim;
  return true;
}

// Extracts a 1D tensor of length `expected_len` broadcast along `expected_dim`
// of a static ranked tensor, peeling through unit-dimension ReshapeOp and
// BroadcastInDimOp chains.
mlir::Value Extract1DBroadcastAlongDim(mlir::Value val, int64_t expected_dim,
                                       int64_t expected_len) {
  mlir::RankedTensorType out_t;
  if (!IsStaticRankedTensor(val.getType(), &out_t) || expected_dim < 0 ||
      expected_dim >= out_t.getRank() ||
      out_t.getDimSize(expected_dim) != expected_len) {
    return nullptr;
  }
  mlir::Value cur = PeelIdentityBroadcast(val);
  int64_t cur_dim = expected_dim;
  for (int depth = 0; depth < kMaxWalkDepth; ++depth) {
    cur = PeelIdentityBroadcast(cur);
    mlir::RankedTensorType cur_t;
    if (!IsStaticRankedTensor(cur.getType(), &cur_t)) {
      return nullptr;
    }
    // GENERIC: semantic 1D bias tensor has rank 1.
    if (cur_t.getRank() == 1 && cur_dim == 0 &&
        cur_t.getDimSize(0) == expected_len) {
      return cur;
    }
    if (auto bcast = cur.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      if (!TryStepBroadcastDim(bcast, cur_dim, expected_len, cur, cur_dim)) {
        return nullptr;
      }
      continue;
    }
    if (auto resh = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      if (!TryStepReshapeDim(resh, cur_t, cur_dim, expected_len, cur,
                             cur_dim)) {
        return nullptr;
      }
      continue;
    }
    break;
  }
  return nullptr;
}

static mlir::Value MatchUnitBatchTransposeReshape(
    mlir::stablehlo::TransposeOp tr, mlir::Value opnd, int64_t d2, int64_t d1) {
  if (tr.getPermutation() != llvm::ArrayRef<int64_t>({2, 1, 0})) {
    return nullptr;
  }
  mlir::RankedTensorType in_t;
  if (!IsStaticRankedTensor(opnd.getType(), &in_t) ||
      in_t.getShape() != llvm::ArrayRef<int64_t>({d2, d1, 1})) {
    return nullptr;
  }
  auto r = opnd.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!r) {
    return nullptr;
  }
  mlir::Value r_in = PeelIdentityBroadcast(r.getOperand());
  mlir::RankedTensorType r_in_t;
  if (IsStaticRankedTensor(r_in.getType(), &r_in_t) &&
      r_in_t.getShape() == llvm::ArrayRef<int64_t>({1, d2, d1})) {
    return r_in;
  }
  return nullptr;
}

static mlir::Value MatchUnitBatchReshapeTransposeReshape(mlir::Value val,
                                                         int64_t d2,
                                                         int64_t d1) {
  auto r_out = val.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!r_out) {
    return nullptr;
  }
  mlir::Value mid = PeelIdentityBroadcast(r_out.getOperand());
  auto tr2 = mid.getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!tr2 || tr2.getPermutation() != llvm::ArrayRef<int64_t>({1, 0})) {
    return nullptr;
  }
  mlir::Value r_in_val = PeelIdentityBroadcast(tr2.getOperand());
  auto r_in = r_in_val.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!r_in) {
    return nullptr;
  }
  mlir::Value x = PeelIdentityBroadcast(r_in.getOperand());
  mlir::RankedTensorType x_t;
  if (IsStaticRankedTensor(x.getType(), &x_t) &&
      x_t.getShape() == llvm::ArrayRef<int64_t>({1, d2, d1})) {
    return x;
  }
  return nullptr;
}

// Matches a 3D last-two-axes swap `[B, D2, D1] -> [B, D1, D2]` (either
// `TransposeOp(x, {0, 2, 1})` or, when `B == 1`,
// `TransposeOp(ReshapeOp(x:[1, D2, D1] -> [D2, D1, 1]), {2, 1, 0})` or
// `ReshapeOp(TransposeOp(ReshapeOp(x:[1, D2, D1] -> [D2, D1]), {1, 0}) ->
// [1, D1, D2])`).
mlir::Value MatchSwapLastTwoDims3D(mlir::Value val) {
  val = PeelIdentityBroadcast(val);
  mlir::RankedTensorType out_t;
  // GENERIC: semantic 1D activation tensor has rank 3 [B, D1, D2].
  if (!IsStaticRankedTensor(val.getType(), &out_t) || out_t.getRank() != 3) {
    return nullptr;
  }
  const int64_t b = out_t.getDimSize(0);
  const int64_t d1 = out_t.getDimSize(1);
  const int64_t d2 = out_t.getDimSize(2);

  if (auto tr = val.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    llvm::ArrayRef<int64_t> perm = tr.getPermutation();
    mlir::Value opnd = PeelIdentityBroadcast(tr.getOperand());
    mlir::RankedTensorType in_t;
    // GENERIC: semantic 1D activation tensor has rank 3 [B, D2, D1].
    if (!IsStaticRankedTensor(opnd.getType(), &in_t) || in_t.getRank() != 3) {
      return nullptr;
    }
    if (perm == llvm::ArrayRef<int64_t>({0, 2, 1}) &&
        in_t.getShape() == llvm::ArrayRef<int64_t>({b, d2, d1})) {
      return opnd;
    }
    // GENERIC: semantic unit batch dimension (B == 1) allows view_decomposition
    // to lower .transpose(1, 2) via Reshape([D2, D1, 1]) -> Transpose({2,1,0}).
    if (b == 1) {
      if (mlir::Value r_in = MatchUnitBatchTransposeReshape(tr, opnd, d2, d1)) {
        return r_in;
      }
    }
  }
  if (b == 1) {
    return MatchUnitBatchReshapeTransposeReshape(val, d2, d1);
  }
  return nullptr;
}

static bool HasUnitStrides(mlir::stablehlo::ConvolutionOp conv_op) {
  auto strides = conv_op.getWindowStrides();
  return !strides || (strides->size() == 1 && (*strides)[0] == 1);
}

static bool HasZeroPadding(mlir::stablehlo::ConvolutionOp conv_op) {
  auto padding = conv_op.getPaddingAttr();
  if (!padding) {
    return true;
  }
  for (int64_t p : padding.getValues<int64_t>()) {
    if (p != 0) {
      return false;
    }
  }
  return true;
}

static bool HasUnitDilations(mlir::stablehlo::ConvolutionOp conv_op) {
  auto lhs_dil = conv_op.getLhsDilation();
  if (lhs_dil && (lhs_dil->size() != 1 || (*lhs_dil)[0] != 1)) {
    return false;
  }
  auto rhs_dil = conv_op.getRhsDilation();
  return !rhs_dil || (rhs_dil->size() == 1 && (*rhs_dil)[0] == 1);
}

static bool HasNoReversal(mlir::stablehlo::ConvolutionOp conv_op) {
  auto rev = conv_op.getWindowReversal();
  if (!rev) {
    return true;
  }
  for (bool r : *rev) {
    if (r) {
      return false;
    }
  }
  return true;
}

bool HasUnitWindowParams1D(mlir::stablehlo::ConvolutionOp conv_op) {
  // GENERIC: semantic ungrouped 1D pointwise convolution has unit group counts,
  // unit stride, zero padding, unit dilation, and no window reversal.
  if (conv_op.getFeatureGroupCount() != 1 ||
      conv_op.getBatchGroupCount() != 1) {
    return false;
  }
  return HasUnitStrides(conv_op) && HasZeroPadding(conv_op) &&
         HasUnitDilations(conv_op) && HasNoReversal(conv_op);
}

// Pattern 1: Lowers unit-batch 1D pointwise ConvolutionOp (`lhs: [1, C_in, T]`,
// `rhs: [C_out, C_in, 1] -> out: [1, C_out, T]` with `dim_labels=bf0_oi0->bf0`
// and `window={size=1}`) into a channel-last `DotGeneralOp` on `[1, T, C_in]`
// and `[C_out, C_in]` followed by `TransposeOp([1, T, C_out], {0, 2, 1})`.
bool TryLowerUnitBatch1DPointwiseConvToDot(
    mlir::stablehlo::ConvolutionOp conv_op) {
  if (!HasUnitWindowParams1D(conv_op)) {
    return false;
  }
  mlir::RankedTensorType lhs_t;
  mlir::RankedTensorType rhs_t;
  mlir::RankedTensorType out_t;
  // GENERIC: semantic 1D convolution operates on rank-3 [B, C, T] tensors.
  if (!IsStaticRankedTensor(conv_op.getLhs().getType(), &lhs_t) ||
      !IsStaticRankedTensor(conv_op.getRhs().getType(), &rhs_t) ||
      !IsStaticRankedTensor(conv_op.getType(), &out_t) ||
      // GENERIC: semantic 1D convolution operands and output have rank 3.
      lhs_t.getRank() != 3 || rhs_t.getRank() != 3 || out_t.getRank() != 3 ||
      lhs_t.getElementType() != rhs_t.getElementType() ||
      lhs_t.getElementType() != out_t.getElementType()) {
    return false;
  }
  auto dnums = conv_op.getDimensionNumbers();
  if (dnums.getInputBatchDimension() != 0 ||
      dnums.getInputFeatureDimension() != 1 ||
      dnums.getInputSpatialDimensions() != llvm::ArrayRef<int64_t>({2}) ||
      dnums.getKernelOutputFeatureDimension() != 0 ||
      dnums.getKernelInputFeatureDimension() != 1 ||
      dnums.getKernelSpatialDimensions() != llvm::ArrayRef<int64_t>({2}) ||
      dnums.getOutputBatchDimension() != 0 ||
      dnums.getOutputFeatureDimension() != 1 ||
      dnums.getOutputSpatialDimensions() != llvm::ArrayRef<int64_t>({2})) {
    return false;
  }
  const int64_t b = lhs_t.getDimSize(0);
  const int64_t c_in = lhs_t.getDimSize(1);
  const int64_t t = lhs_t.getDimSize(2);
  const int64_t c_out = rhs_t.getDimSize(0);
  // GENERIC: semantic unit batch (b == 1) and pointwise kernel (size == 1),
  // with hardware TPU sublane/lane alignment (t <= kTpuSublaneCount or t a
  // multiple of kTpuLaneWidth) so DotGeneralOp avoids intermediate
  // Large2ndMinor T(16, 128) <-> T(8, 128) relayout copies and maintains VMEM
  // weight prefetch.
  const bool has_aligned_token_dim =
      (t <= kTpuSublaneCount) || (t >= kTpuLaneWidth && t % kTpuLaneWidth == 0);
  if (b != 1 || c_in <= 0 || t <= 0 || !has_aligned_token_dim || c_out <= 0 ||
      rhs_t.getDimSize(1) != c_in || rhs_t.getDimSize(2) != 1 ||
      out_t.getShape() != llvm::ArrayRef<int64_t>({1, c_out, t})) {
    return false;
  }

  mlir::OpBuilder builder(conv_op);
  mlir::Location loc = conv_op.getLoc();
  mlir::Type elem_t = out_t.getElementType();

  mlir::Value x_btc = MatchSwapLastTwoDims3D(conv_op.getLhs());
  if (!x_btc) {
    auto btc_in_type = mlir::RankedTensorType::get({1, t, c_in}, elem_t);
    x_btc = builder.create<mlir::stablehlo::TransposeOp>(
        loc, btc_in_type, conv_op.getLhs(),
        builder.getDenseI64ArrayAttr({0, 2, 1}));
  }

  auto w_2d_type = mlir::RankedTensorType::get({c_out, c_in}, elem_t);
  mlir::Value w_2d = builder.create<mlir::stablehlo::ReshapeOp>(
      loc, w_2d_type, conv_op.getRhs());

  auto dot_btc_type = mlir::RankedTensorType::get({1, t, c_out}, elem_t);
  auto dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(),
      /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{},
      /*lhsContractingDimensions=*/{2},
      /*rhsContractingDimensions=*/{1});
  auto dot_btc = builder.create<mlir::stablehlo::DotGeneralOp>(
      loc, dot_btc_type, x_btc, w_2d, dot_dims,
      conv_op.getPrecisionConfigAttr(),
      /*algorithm=*/mlir::stablehlo::DotAlgorithmAttr{});

  mlir::Value out_bct = builder.create<mlir::stablehlo::TransposeOp>(
      loc, out_t, dot_btc, builder.getDenseI64ArrayAttr({0, 2, 1}));
  conv_op.replaceAllUsesWith(out_bct);
  conv_op.erase();
  return true;
}

// Pattern 2a: Pushes a 3D last-two-axes swap `TransposeOp(x:[B, T, C],
// {0,2,1})` downstream past a 1D channel bias `AddOp(tr,
// BroadcastInDim(bias:[C], {1}))` so `AddOp` runs in `[B, T, C]` layout and
// `TransposeOp` sits at the output.
bool TryPushChannelTransposeAcrossBiasAdd(mlir::stablehlo::AddOp add_op) {
  mlir::RankedTensorType out_t;
  // GENERIC: semantic 1D activation tensor has rank 3 [B, C, T].
  if (!IsStaticRankedTensor(add_op.getType(), &out_t) || out_t.getRank() != 3) {
    return false;
  }
  const int64_t b = out_t.getDimSize(0);
  const int64_t c = out_t.getDimSize(1);
  const int64_t t = out_t.getDimSize(2);

  for (int side = 0; side < 2; ++side) {
    mlir::Value tr_val = (side == 0) ? add_op.getLhs() : add_op.getRhs();
    mlir::Value bias_val = (side == 0) ? add_op.getRhs() : add_op.getLhs();
    if (!tr_val.hasOneUse()) {
      continue;
    }
    mlir::Value x_btc = MatchSwapLastTwoDims3D(tr_val);
    if (!x_btc) {
      continue;
    }
    mlir::Value bias_1d = Extract1DBroadcastAlongDim(bias_val, 1, c);
    if (!bias_1d) {
      continue;
    }
    mlir::OpBuilder builder(add_op);
    mlir::Location loc = add_op.getLoc();
    auto btc_type =
        mlir::RankedTensorType::get({b, t, c}, out_t.getElementType());
    mlir::Value bcast_btc = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, btc_type, bias_1d, builder.getDenseI64ArrayAttr({2}));
    mlir::Value add_btc =
        builder.create<mlir::stablehlo::AddOp>(loc, btc_type, x_btc, bcast_btc);
    mlir::Value out_bct = builder.create<mlir::stablehlo::TransposeOp>(
        loc, out_t, add_btc, builder.getDenseI64ArrayAttr({0, 2, 1}));
    add_op.replaceAllUsesWith(out_bct);
    add_op.erase();
    return true;
  }
  return false;
}

static bool TryCancelDoubleSwap3D(mlir::Operation* op,
                                  llvm::DenseSet<mlir::Operation*>& erased) {
  mlir::Value x = MatchSwapLastTwoDims3D(op->getResult(0));
  if (!x) {
    return false;
  }
  mlir::Value orig = MatchSwapLastTwoDims3D(x);
  if (!orig || orig.getType() != op->getResult(0).getType()) {
    return false;
  }
  op->getResult(0).replaceAllUsesWith(orig);
  erased.insert(op);
  op->erase();
  if (auto prev_op = x.getDefiningOp()) {
    if (prev_op->use_empty()) {
      erased.insert(prev_op);
      prev_op->erase();
    }
  }
  return true;
}

static bool TrySimplifySandwichedSlice(
    mlir::Operation* op, llvm::DenseSet<mlir::Operation*>& erased) {
  auto tr_out = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op);
  if (!tr_out) {
    return false;
  }
  auto slice_op = tr_out.getOperand().getDefiningOp<mlir::stablehlo::SliceOp>();
  if (!slice_op) {
    return false;
  }
  auto tr_in =
      slice_op.getOperand().getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!tr_in) {
    return false;
  }
  llvm::ArrayRef<int64_t> perm_in = tr_in.getPermutation();
  llvm::ArrayRef<int64_t> perm_out = tr_out.getPermutation();
  const int64_t rank = perm_in.size();
  if (rank <= 0 || static_cast<int64_t>(perm_out.size()) != rank) {
    return false;
  }
  for (int64_t i = 0; i < rank; ++i) {
    if (perm_out[i] < 0 || perm_out[i] >= rank || perm_in[perm_out[i]] != i) {
      return false;
    }
  }
  llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
  llvm::ArrayRef<int64_t> strides = slice_op.getStrides();
  llvm::SmallVector<int64_t, 4> new_starts(rank);
  llvm::SmallVector<int64_t, 4> new_limits(rank);
  llvm::SmallVector<int64_t, 4> new_strides(rank);
  for (int64_t i = 0; i < rank; ++i) {
    int64_t d = perm_in[i];
    new_starts[d] = starts[i];
    new_limits[d] = limits[i];
    new_strides[d] = strides[i];
  }
  mlir::OpBuilder builder(tr_out);
  mlir::Value new_slice = builder.create<mlir::stablehlo::SliceOp>(
      tr_out.getLoc(), tr_out.getType(), tr_in.getOperand(),
      builder.getDenseI64ArrayAttr(new_starts),
      builder.getDenseI64ArrayAttr(new_limits),
      builder.getDenseI64ArrayAttr(new_strides));
  tr_out.replaceAllUsesWith(new_slice);
  erased.insert(tr_out.getOperation());
  tr_out.erase();
  if (slice_op->use_empty()) {
    erased.insert(slice_op.getOperation());
    slice_op->erase();
  }
  if (tr_in->use_empty()) {
    erased.insert(tr_in.getOperation());
    tr_in->erase();
  }
  return true;
}

// Pattern 2b: Cancels inverse 3D last-two-axes swaps and simplifies
// `TransposeOp(SliceOp(TransposeOp(x, perm)), inv_perm)` -> `SliceOp(x)`.
bool TrySimplifyTransposePairsAndSlices(
    mlir::Operation* op, llvm::DenseSet<mlir::Operation*>& erased) {
  if (op->getNumResults() != 1) {
    return false;
  }
  if (TryCancelDoubleSwap3D(op, erased)) {
    return true;
  }
  return TrySimplifySandwichedSlice(op, erased);
}

struct GluSubgraphMatch {
  mlir::Value root_val;
  mlir::Value raw_in;
  mlir::Value comp_in;
  mlir::Type comp_elem_t;
  mlir::Type out_elem_t;
  int64_t b = 0;
  int64_t c = 0;
  int64_t t = 0;
  bool is_logistic = false;
};

static mlir::Value MatchGluSigmoidBranch(mlir::Value sig_val,
                                         bool& is_logistic) {
  if (auto div = sig_val.getDefiningOp<mlir::stablehlo::DivOp>()) {
    if (!IsFloatConstantCloseTo(div.getLhs(), 1.0, 0.0)) {
      return nullptr;
    }
    auto add = div.getRhs().getDefiningOp<mlir::stablehlo::AddOp>();
    if (!add) {
      return nullptr;
    }
    mlir::Value exp_val;
    if (IsFloatConstantCloseTo(add.getLhs(), 1.0, 0.0)) {
      exp_val = add.getRhs();
    } else if (IsFloatConstantCloseTo(add.getRhs(), 1.0, 0.0)) {
      exp_val = add.getLhs();
    } else {
      return nullptr;
    }
    auto exp_op = exp_val.getDefiningOp<mlir::stablehlo::ExpOp>();
    if (!exp_op) {
      return nullptr;
    }
    auto neg_op = exp_op.getOperand().getDefiningOp<mlir::stablehlo::NegOp>();
    if (!neg_op) {
      return nullptr;
    }
    is_logistic = false;
    return neg_op.getOperand();
  }
  if (auto log_op = sig_val.getDefiningOp<mlir::stablehlo::LogisticOp>()) {
    is_logistic = true;
    return log_op.getOperand();
  }
  return nullptr;
}

static bool MatchGluSlices(mlir::Value s0_val, mlir::Value s1_val, int64_t b,
                           int64_t c, int64_t t, mlir::Value& comp_in) {
  auto s0 = s0_val.getDefiningOp<mlir::stablehlo::SliceOp>();
  auto s1 = s1_val.getDefiningOp<mlir::stablehlo::SliceOp>();
  if (!s0 || !s1 || s0.getOperand() != s1.getOperand()) {
    return false;
  }
  if (s0.getStartIndices() != llvm::ArrayRef<int64_t>({0, 0, 0}) ||
      s0.getLimitIndices() != llvm::ArrayRef<int64_t>({b, c, t}) ||
      s0.getStrides() != llvm::ArrayRef<int64_t>({1, 1, 1}) ||
      s1.getStartIndices() != llvm::ArrayRef<int64_t>({0, c, 0}) ||
      s1.getLimitIndices() != llvm::ArrayRef<int64_t>({b, 2 * c, t}) ||
      s1.getStrides() != llvm::ArrayRef<int64_t>({1, 1, 1})) {
    return false;
  }
  comp_in = s0.getOperand();
  return true;
}

// Matches a 3D channel-first GLU subgraph along axis 1 (`[B, 2*C, T] ->
// [B, C, T]`): `Slice(0:C) * (1 / (1 + Exp(Neg(Slice(C:2C)))))`.
std::optional<GluSubgraphMatch> MatchChannelFirstGlu3D(mlir::Operation* op) {
  if (op->use_empty()) {
    return std::nullopt;
  }
  mlir::Value root_val;
  mlir::Value mul_val;
  if (auto conv_out = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
    if (!conv_out.getOperand().hasOneUse()) {
      return std::nullopt;
    }
    root_val = conv_out.getResult();
    mul_val = conv_out.getOperand();
  } else if (auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(op)) {
    if (mul->hasOneUse() &&
        mlir::isa<mlir::stablehlo::ConvertOp>(*mul->user_begin())) {
      return std::nullopt;
    }
    root_val = mul.getResult();
    mul_val = mul.getResult();
  } else {
    return std::nullopt;
  }

  auto mul_op = mul_val.getDefiningOp<mlir::stablehlo::MulOp>();
  mlir::RankedTensorType root_t;
  mlir::RankedTensorType mul_t;
  // GENERIC: semantic 1D activation tensor has rank 3 [B, C, T].
  if (!mul_op || !IsStaticRankedTensor(root_val.getType(), &root_t) ||
      !IsStaticRankedTensor(mul_val.getType(), &mul_t) ||
      root_t.getRank() != 3 || root_t.getShape() != mul_t.getShape()) {
    return std::nullopt;
  }
  const int64_t b = root_t.getDimSize(0);
  const int64_t c = root_t.getDimSize(1);
  const int64_t t = root_t.getDimSize(2);
  if (b <= 0 || c <= 0 || t <= 0) {
    return std::nullopt;
  }

  for (int side = 0; side < 2; ++side) {
    mlir::Value s0_val = (side == 0) ? mul_op.getLhs() : mul_op.getRhs();
    mlir::Value sig_val = (side == 0) ? mul_op.getRhs() : mul_op.getLhs();
    bool is_logistic = false;
    mlir::Value s1_val = MatchGluSigmoidBranch(sig_val, is_logistic);
    if (!s1_val) {
      continue;
    }
    mlir::Value comp_in;
    if (!MatchGluSlices(s0_val, s1_val, b, c, t, comp_in)) {
      continue;
    }
    mlir::Value raw_in = comp_in;
    if (auto conv_in = comp_in.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      raw_in = conv_in.getOperand();
    }
    GluSubgraphMatch m;
    m.root_val = root_val;
    m.raw_in = raw_in;
    m.comp_in = comp_in;
    m.comp_elem_t = mul_t.getElementType();
    m.out_elem_t = root_t.getElementType();
    m.b = b;
    m.c = c;
    m.t = t;
    m.is_logistic = is_logistic;
    return m;
  }
  return std::nullopt;
}

// Pattern 2c: Commutes a 3D last-two-axes swap `TransposeOp(x:[B, T, 2*C],
// {0, 2, 1})` downstream across a channel-first GLU subgraph so GLU executes
// in `[B, T, C]` layout and fuses into the upstream `DotGeneralOp`.
bool TryCommuteChannelTransposeDownstreamAcrossGlu(
    mlir::Operation* op, llvm::DenseSet<mlir::Operation*>& erased) {
  auto match = MatchChannelFirstGlu3D(op);
  if (!match.has_value()) {
    return false;
  }
  mlir::Value x_btc = MatchSwapLastTwoDims3D(match->raw_in);
  if (!x_btc) {
    return false;
  }
  mlir::RankedTensorType x_btc_t;
  if (!IsStaticRankedTensor(x_btc.getType(), &x_btc_t) ||
      x_btc_t.getShape() !=
          llvm::ArrayRef<int64_t>({match->b, match->t, 2 * match->c})) {
    return false;
  }

  mlir::OpBuilder builder(op);
  mlir::Location loc = op->getLoc();
  mlir::Value in_comp_btc = x_btc;
  if (x_btc_t.getElementType() != match->comp_elem_t) {
    auto comp_in_t = mlir::RankedTensorType::get(
        {match->b, match->t, 2 * match->c}, match->comp_elem_t);
    in_comp_btc =
        builder.create<mlir::stablehlo::ConvertOp>(loc, comp_in_t, x_btc);
  }
  auto half_comp_t = mlir::RankedTensorType::get({match->b, match->t, match->c},
                                                 match->comp_elem_t);
  mlir::Value s0_btc = builder.create<mlir::stablehlo::SliceOp>(
      loc, half_comp_t, in_comp_btc, builder.getDenseI64ArrayAttr({0, 0, 0}),
      builder.getDenseI64ArrayAttr({match->b, match->t, match->c}),
      builder.getDenseI64ArrayAttr({1, 1, 1}));
  mlir::Value s1_btc = builder.create<mlir::stablehlo::SliceOp>(
      loc, half_comp_t, in_comp_btc,
      builder.getDenseI64ArrayAttr({0, 0, match->c}),
      builder.getDenseI64ArrayAttr({match->b, match->t, 2 * match->c}),
      builder.getDenseI64ArrayAttr({1, 1, 1}));
  mlir::Value sig_btc;
  if (match->is_logistic) {
    sig_btc =
        builder.create<mlir::stablehlo::LogisticOp>(loc, half_comp_t, s1_btc);
  } else {
    mlir::Value neg_btc =
        builder.create<mlir::stablehlo::NegOp>(loc, half_comp_t, s1_btc);
    mlir::Value exp_btc =
        builder.create<mlir::stablehlo::ExpOp>(loc, half_comp_t, neg_btc);
    auto flt_t = mlir::cast<mlir::FloatType>(match->comp_elem_t);
    auto one_scalar_t = mlir::RankedTensorType::get({}, flt_t);
    mlir::Value one_scalar = builder.create<mlir::stablehlo::ConstantOp>(
        loc, one_scalar_t,
        mlir::DenseElementsAttr::get(
            one_scalar_t, llvm::APFloat(flt_t.getFloatSemantics(), 1)));
    mlir::Value one_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, half_comp_t, one_scalar, builder.getDenseI64ArrayAttr({}));
    mlir::Value add_btc = builder.create<mlir::stablehlo::AddOp>(
        loc, half_comp_t, exp_btc, one_bcast);
    sig_btc = builder.create<mlir::stablehlo::DivOp>(loc, half_comp_t,
                                                     one_bcast, add_btc);
  }
  mlir::Value mul_btc =
      builder.create<mlir::stablehlo::MulOp>(loc, half_comp_t, s0_btc, sig_btc);
  mlir::Value out_btc = mul_btc;
  if (match->out_elem_t != match->comp_elem_t) {
    auto out_btc_t = mlir::RankedTensorType::get({match->b, match->t, match->c},
                                                 match->out_elem_t);
    out_btc =
        builder.create<mlir::stablehlo::ConvertOp>(loc, out_btc_t, mul_btc);
  }
  auto out_bct_t = mlir::RankedTensorType::get({match->b, match->c, match->t},
                                               match->out_elem_t);
  mlir::Value out_bct = builder.create<mlir::stablehlo::TransposeOp>(
      loc, out_bct_t, out_btc, builder.getDenseI64ArrayAttr({0, 2, 1}));
  match->root_val.replaceAllUsesWith(out_bct);
  erased.insert(op);
  op->erase();
  return true;
}

struct SiluSubgraphMatch {
  mlir::Value raw_in;
  mlir::Type comp_elem_t;
  mlir::Type out_elem_t;
  bool is_logistic = false;
};

static mlir::Value MatchDivSilu(mlir::Value core_val) {
  auto div = core_val.getDefiningOp<mlir::stablehlo::DivOp>();
  if (!div) {
    return nullptr;
  }
  auto add = div.getRhs().getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add) {
    return nullptr;
  }
  mlir::Value exp_val;
  if (IsFloatConstantCloseTo(add.getLhs(), 1.0, 0.0)) {
    exp_val = add.getRhs();
  } else if (IsFloatConstantCloseTo(add.getRhs(), 1.0, 0.0)) {
    exp_val = add.getLhs();
  } else {
    return nullptr;
  }
  auto exp_op = exp_val.getDefiningOp<mlir::stablehlo::ExpOp>();
  if (!exp_op) {
    return nullptr;
  }
  auto neg_op = exp_op.getOperand().getDefiningOp<mlir::stablehlo::NegOp>();
  if (!neg_op || neg_op.getOperand() != div.getLhs()) {
    return nullptr;
  }
  return div.getLhs();
}

static mlir::Value MatchMulSilu(mlir::Value core_val) {
  auto mul = core_val.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul) {
    return nullptr;
  }
  for (int side = 0; side < 2; ++side) {
    mlir::Value a = (side == 0) ? mul.getLhs() : mul.getRhs();
    mlir::Value b = (side == 0) ? mul.getRhs() : mul.getLhs();
    if (auto log_op = b.getDefiningOp<mlir::stablehlo::LogisticOp>()) {
      if (log_op.getOperand() == a) {
        return a;
      }
    }
  }
  return nullptr;
}

// Matches a SiLU subgraph (`x / (1 + Exp(Neg(x)))` or `x * Logistic(x)`),
// optionally wrapped in `ConvertOp(f32) ... ConvertOp(bf16)`.
std::optional<SiluSubgraphMatch> MatchSiluSubgraph(mlir::Value val) {
  mlir::RankedTensorType out_t;
  if (!IsStaticRankedTensor(val.getType(), &out_t)) {
    return std::nullopt;
  }
  mlir::Value core_val = val;
  if (auto conv_out = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    core_val = conv_out.getOperand();
  }
  mlir::RankedTensorType comp_t;
  if (!IsStaticRankedTensor(core_val.getType(), &comp_t) ||
      comp_t.getShape() != out_t.getShape()) {
    return std::nullopt;
  }

  mlir::Value x_comp;
  bool is_logistic = false;
  if (mlir::Value x_div = MatchDivSilu(core_val)) {
    x_comp = x_div;
    is_logistic = false;
  } else if (mlir::Value x_mul = MatchMulSilu(core_val)) {
    x_comp = x_mul;
    is_logistic = true;
  } else {
    return std::nullopt;
  }

  mlir::Value raw_in = x_comp;
  if (auto conv_in = x_comp.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    raw_in = conv_in.getOperand();
  }
  SiluSubgraphMatch m;
  m.raw_in = raw_in;
  m.comp_elem_t = comp_t.getElementType();
  m.out_elem_t = out_t.getElementType();
  m.is_logistic = is_logistic;
  return m;
}

// Pattern 2d: Commutes a 3D last-two-axes swap `[B, C, T] -> [B, T, C]`
// upstream across a SiLU subgraph when its input is itself a `[B, T, C] ->
// [B, C, T]` swap (so commuting eliminates both transposes).
bool TryCommuteTransposeUpstreamAcrossCancellingSilu(mlir::Operation* op) {
  if (op->getNumResults() != 1) {
    return false;
  }

  mlir::Value bct_in = MatchSwapLastTwoDims3D(op->getResult(0));
  if (!bct_in) {
    return false;
  }
  mlir::RankedTensorType btc_t;
  mlir::RankedTensorType bct_t;
  if (!IsStaticRankedTensor(op->getResult(0).getType(), &btc_t) ||
      !IsStaticRankedTensor(bct_in.getType(), &bct_t) ||
      // GENERIC: semantic 1D activation tensor has rank 3 [B, T, C].
      btc_t.getRank() != 3 || bct_t.getRank() != 3) {
    return false;
  }
  const int64_t b = btc_t.getDimSize(0);
  const int64_t t = btc_t.getDimSize(1);
  const int64_t c = btc_t.getDimSize(2);

  // `bct_in` is produced by a SiLU subgraph on `[B, C, T]` whose input
  // is itself a `[B, T, C] -> [B, C, T]` swap (so commuting eliminates both
  // transposes).
  if (bct_in.hasOneUse()) {
    if (auto silu = MatchSiluSubgraph(bct_in)) {
      mlir::Value in_btc = MatchSwapLastTwoDims3D(silu->raw_in);
      if (in_btc) {
        mlir::OpBuilder builder(op);
        mlir::Location loc = op->getLoc();
        auto raw_in_t = mlir::cast<mlir::RankedTensorType>(in_btc.getType());
        auto comp_btc_t =
            mlir::RankedTensorType::get({b, t, c}, silu->comp_elem_t);
        mlir::Value x_comp = in_btc;
        if (raw_in_t.getElementType() != silu->comp_elem_t) {
          x_comp = builder.create<mlir::stablehlo::ConvertOp>(loc, comp_btc_t,
                                                              in_btc);
        }
        mlir::Value silu_val;
        if (silu->is_logistic) {
          mlir::Value log_v = builder.create<mlir::stablehlo::LogisticOp>(
              loc, comp_btc_t, x_comp);
          silu_val = builder.create<mlir::stablehlo::MulOp>(loc, comp_btc_t,
                                                            x_comp, log_v);
        } else {
          mlir::Value neg_v =
              builder.create<mlir::stablehlo::NegOp>(loc, comp_btc_t, x_comp);
          mlir::Value exp_v =
              builder.create<mlir::stablehlo::ExpOp>(loc, comp_btc_t, neg_v);
          auto flt_t = mlir::cast<mlir::FloatType>(silu->comp_elem_t);
          auto one_scalar_t = mlir::RankedTensorType::get({}, flt_t);
          mlir::Value one_scalar = builder.create<mlir::stablehlo::ConstantOp>(
              loc, one_scalar_t,
              mlir::DenseElementsAttr::get(
                  one_scalar_t, llvm::APFloat(flt_t.getFloatSemantics(), 1)));
          mlir::Value one_bcast =
              builder.create<mlir::stablehlo::BroadcastInDimOp>(
                  loc, comp_btc_t, one_scalar,
                  builder.getDenseI64ArrayAttr({}));
          mlir::Value add_v = builder.create<mlir::stablehlo::AddOp>(
              loc, comp_btc_t, exp_v, one_bcast);
          silu_val = builder.create<mlir::stablehlo::DivOp>(loc, comp_btc_t,
                                                            x_comp, add_v);
        }
        mlir::Value out_btc = silu_val;
        if (silu->out_elem_t != silu->comp_elem_t) {
          out_btc =
              builder.create<mlir::stablehlo::ConvertOp>(loc, btc_t, silu_val);
        }
        op->getResult(0).replaceAllUsesWith(out_btc);
        op->erase();
        return true;
      }
    }
  }
  return false;
}

// Forward declarations for affine iota comparison evaluation.
bool MatchAffineIota1D(mlir::Value val, int64_t& out_dim, int64_t& out_start,
                       int64_t& out_step, int depth = 0);

std::optional<int64_t> ExtractSplatIntValue(mlir::Value val, int depth = 0) {
  if (depth > kMaxWalkDepth) {
    return std::nullopt;
  }
  val = PeelIdentityBroadcast(val);
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    return ExtractSplatIntValue(bcast.getOperand(), depth + 1);
  }
  if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    return ExtractSplatIntValue(resh.getOperand(), depth + 1);
  }
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    return ExtractSplatIntValue(conv.getOperand(), depth + 1);
  }
  auto cst = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!cst) {
    return std::nullopt;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat() ||
      !mlir::isa<mlir::IntegerType>(dense.getElementType()) ||
      dense.getElementType().isInteger(1)) {
    return std::nullopt;
  }
  return dense.getSplatValue<llvm::APInt>().getSExtValue();
}

bool EvalIntComparison(mlir::stablehlo::ComparisonDirection dir, int64_t lhs,
                       int64_t rhs) {
  switch (dir) {
    case mlir::stablehlo::ComparisonDirection::EQ:
      return lhs == rhs;
    case mlir::stablehlo::ComparisonDirection::NE:
      return lhs != rhs;
    case mlir::stablehlo::ComparisonDirection::LT:
      return lhs < rhs;
    case mlir::stablehlo::ComparisonDirection::LE:
      return lhs <= rhs;
    case mlir::stablehlo::ComparisonDirection::GT:
      return lhs > rhs;
    case mlir::stablehlo::ComparisonDirection::GE:
      return lhs >= rhs;
  }
  return false;
}

static bool CheckAffineCompareBounds(mlir::IntegerType int_ty, int64_t len,
                                     int64_t s, int64_t st, int64_t& v_first,
                                     int64_t& v_last) {
  const unsigned bitwidth = int_ty.getWidth();
  if (bitwidth == 0 || bitwidth > 64) {
    return false;
  }
  const int64_t min_val =
      llvm::APInt::getSignedMinValue(bitwidth).getSExtValue();
  const int64_t max_val =
      llvm::APInt::getSignedMaxValue(bitwidth).getSExtValue();
  if (s < min_val || s > max_val) {
    return false;
  }
  int64_t step_span = 0;
  if (__builtin_mul_overflow(st, len - 1, &step_span)) {
    return false;
  }
  if (__builtin_add_overflow(s, step_span, &v_last)) {
    return false;
  }
  if (v_last < min_val || v_last > max_val) {
    return false;
  }
  v_first = s;
  return true;
}

bool EvaluateAffineCompareUniformly(mlir::stablehlo::CompareOp cmp_op,
                                    bool& out_uniform_value, int depth = 0) {
  if (depth > kMaxWalkDepth) {
    return false;
  }
  mlir::RankedTensorType lhs_t;
  if (!IsStaticRankedTensor(cmp_op.getLhs().getType(), &lhs_t)) {
    return false;
  }
  int64_t d = -1, s = 0, st = 0;
  std::optional<int64_t> c_val;
  bool iota_on_lhs = false;
  if (MatchAffineIota1D(cmp_op.getLhs(), d, s, st, depth + 1)) {
    c_val = ExtractSplatIntValue(cmp_op.getRhs(), depth + 1);
    iota_on_lhs = true;
  } else if (MatchAffineIota1D(cmp_op.getRhs(), d, s, st, depth + 1)) {
    c_val = ExtractSplatIntValue(cmp_op.getLhs(), depth + 1);
    iota_on_lhs = false;
  }
  if (!c_val.has_value() || d < 0 || d >= lhs_t.getRank() ||
      lhs_t.getDimSize(d) <= 0) {
    return false;
  }
  const auto dir = cmp_op.getComparisonDirection();
  if (dir == mlir::stablehlo::ComparisonDirection::EQ ||
      dir == mlir::stablehlo::ComparisonDirection::NE) {
    return false;
  }
  auto int_ty = mlir::dyn_cast<mlir::IntegerType>(lhs_t.getElementType());
  if (!int_ty || int_ty.isUnsigned()) {
    return false;
  }
  if (cmp_op.getCompareType() &&
      *cmp_op.getCompareType() != mlir::stablehlo::ComparisonType::SIGNED) {
    return false;
  }
  int64_t v_first = 0;
  int64_t v_last = 0;
  if (!CheckAffineCompareBounds(int_ty, lhs_t.getDimSize(d), s, st, v_first,
                                v_last)) {
    return false;
  }
  const bool b_first = iota_on_lhs ? EvalIntComparison(dir, v_first, *c_val)
                                   : EvalIntComparison(dir, *c_val, v_first);
  const bool b_last = iota_on_lhs ? EvalIntComparison(dir, v_last, *c_val)
                                  : EvalIntComparison(dir, *c_val, v_last);
  if (b_first != b_last) {
    return false;
  }
  out_uniform_value = b_first;
  return true;
}

static bool MatchAffineConvert(mlir::stablehlo::ConvertOp conv,
                               mlir::RankedTensorType type, int64_t& out_dim,
                               int64_t& out_start, int64_t& out_step,
                               int depth) {
  mlir::RankedTensorType in_t;
  if (!IsStaticRankedTensor(conv.getOperand().getType(), &in_t) ||
      !mlir::isa<mlir::IntegerType>(in_t.getElementType()) ||
      in_t.getElementTypeBitWidth() > type.getElementTypeBitWidth()) {
    return false;
  }
  auto in_int = mlir::cast<mlir::IntegerType>(in_t.getElementType());
  if (in_int.isUnsigned()) {
    return false;
  }
  int64_t op_dim = -1, op_start = 0, op_step = 0;
  if (!MatchAffineIota1D(conv.getOperand(), op_dim, op_start, op_step,
                         depth + 1)) {
    return false;
  }
  const unsigned in_bits = in_int.getWidth();
  if (in_bits == 0 || in_bits > 64) {
    return false;
  }
  const int64_t in_min = llvm::APInt::getSignedMinValue(in_bits).getSExtValue();
  const int64_t in_max = llvm::APInt::getSignedMaxValue(in_bits).getSExtValue();
  if (op_start < in_min || op_start > in_max) {
    return false;
  }
  const int64_t len = in_t.getDimSize(op_dim);
  int64_t span = 0;
  if (__builtin_mul_overflow(op_step, len - 1, &span)) {
    return false;
  }
  int64_t op_last = 0;
  if (__builtin_add_overflow(op_start, span, &op_last)) {
    return false;
  }
  if (op_last < in_min || op_last > in_max) {
    return false;
  }
  out_dim = op_dim;
  out_start = op_start;
  out_step = op_step;
  return true;
}

static bool MatchAffineBroadcast(mlir::stablehlo::BroadcastInDimOp bcast,
                                 mlir::RankedTensorType type, int64_t& out_dim,
                                 int64_t& out_start, int64_t& out_step,
                                 int depth) {
  mlir::RankedTensorType in_t;
  if (!IsStaticRankedTensor(bcast.getOperand().getType(), &in_t)) {
    return false;
  }
  int64_t op_dim = -1, op_start = 0, op_step = 0;
  if (!MatchAffineIota1D(bcast.getOperand(), op_dim, op_start, op_step,
                         depth + 1)) {
    return false;
  }
  llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
  if (op_dim < 0 || op_dim >= static_cast<int64_t>(bdims.size())) {
    return false;
  }
  int64_t mapped = bdims[op_dim];
  if (mapped < 0 || mapped >= type.getRank() ||
      in_t.getDimSize(op_dim) != type.getDimSize(mapped)) {
    return false;
  }
  out_dim = mapped;
  out_start = op_start;
  out_step = op_step;
  return true;
}

static bool MatchAffineAdd(mlir::stablehlo::AddOp add, int64_t& out_dim,
                           int64_t& out_start, int64_t& out_step, int depth) {
  int64_t d = -1, s = 0, st = 0;
  if (MatchAffineIota1D(add.getLhs(), d, s, st, depth + 1)) {
    if (auto c = ExtractSplatIntValue(add.getRhs(), depth + 1)) {
      int64_t res = 0;
      if (!__builtin_add_overflow(s, *c, &res)) {
        out_dim = d;
        out_start = res;
        out_step = st;
        return true;
      }
    }
  }
  if (MatchAffineIota1D(add.getRhs(), d, s, st, depth + 1)) {
    if (auto c = ExtractSplatIntValue(add.getLhs(), depth + 1)) {
      int64_t res = 0;
      if (!__builtin_add_overflow(*c, s, &res)) {
        out_dim = d;
        out_start = res;
        out_step = st;
        return true;
      }
    }
  }
  return false;
}

static bool MatchAffineSub(mlir::stablehlo::SubtractOp sub, int64_t& out_dim,
                           int64_t& out_start, int64_t& out_step, int depth) {
  int64_t d = -1, s = 0, st = 0;
  if (MatchAffineIota1D(sub.getLhs(), d, s, st, depth + 1)) {
    if (auto c = ExtractSplatIntValue(sub.getRhs(), depth + 1)) {
      int64_t res = 0;
      if (!__builtin_sub_overflow(s, *c, &res)) {
        out_dim = d;
        out_start = res;
        out_step = st;
        return true;
      }
    }
  }
  if (MatchAffineIota1D(sub.getRhs(), d, s, st, depth + 1)) {
    if (auto c = ExtractSplatIntValue(sub.getLhs(), depth + 1)) {
      int64_t res_start = 0;
      int64_t res_step = 0;
      if (!__builtin_sub_overflow(*c, s, &res_start) &&
          !__builtin_sub_overflow(0, st, &res_step)) {
        out_dim = d;
        out_start = res_start;
        out_step = res_step;
        return true;
      }
    }
  }
  return false;
}

static bool MatchAffineMul(mlir::stablehlo::MulOp mul, int64_t& out_dim,
                           int64_t& out_start, int64_t& out_step, int depth) {
  int64_t d = -1, s = 0, st = 0;
  if (MatchAffineIota1D(mul.getLhs(), d, s, st, depth + 1)) {
    if (auto c = ExtractSplatIntValue(mul.getRhs(), depth + 1)) {
      int64_t res_start = 0;
      int64_t res_step = 0;
      if (!__builtin_mul_overflow(s, *c, &res_start) &&
          !__builtin_mul_overflow(st, *c, &res_step)) {
        out_dim = d;
        out_start = res_start;
        out_step = res_step;
        return true;
      }
    }
  }
  if (MatchAffineIota1D(mul.getRhs(), d, s, st, depth + 1)) {
    if (auto c = ExtractSplatIntValue(mul.getLhs(), depth + 1)) {
      int64_t res_start = 0;
      int64_t res_step = 0;
      if (!__builtin_mul_overflow(*c, s, &res_start) &&
          !__builtin_mul_overflow(*c, st, &res_step)) {
        out_dim = d;
        out_start = res_start;
        out_step = res_step;
        return true;
      }
    }
  }
  return false;
}

bool MatchAffineIota1D(mlir::Value val, int64_t& out_dim, int64_t& out_start,
                       int64_t& out_step, int depth) {
  if (depth > kMaxWalkDepth) {
    return false;
  }
  val = PeelIdentityBroadcast(val);
  mlir::RankedTensorType type;
  if (!IsStaticRankedTensor(val.getType(), &type) || type.getRank() < 1 ||
      !mlir::isa<mlir::IntegerType>(type.getElementType()) ||
      type.getElementType().isInteger(1)) {
    return false;
  }
  if (auto iota = val.getDefiningOp<mlir::stablehlo::IotaOp>()) {
    out_dim = static_cast<int64_t>(iota.getIotaDimension());
    out_start = 0;
    out_step = 1;
    return out_dim >= 0 && out_dim < type.getRank();
  }
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    return MatchAffineConvert(conv, type, out_dim, out_start, out_step, depth);
  }
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    return MatchAffineBroadcast(bcast, type, out_dim, out_start, out_step,
                                depth);
  }
  if (auto add = val.getDefiningOp<mlir::stablehlo::AddOp>()) {
    return MatchAffineAdd(add, out_dim, out_start, out_step, depth);
  }
  if (auto sub = val.getDefiningOp<mlir::stablehlo::SubtractOp>()) {
    return MatchAffineSub(sub, out_dim, out_start, out_step, depth);
  }
  if (auto mul = val.getDefiningOp<mlir::stablehlo::MulOp>()) {
    return MatchAffineMul(mul, out_dim, out_start, out_step, depth);
  }
  return false;
}

constexpr double kMaxSafeFiniteMagnitude = 1.0e4;

bool IsProvablyFiniteAndAboveLowestFloat(
    mlir::Value val, llvm::DenseMap<mlir::Value, bool>& memo, int depth = 0);

static bool IsBoundedFloatConstant(mlir::stablehlo::ConstantOp cst) {
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !mlir::isa<mlir::FloatType>(dense.getElementType())) {
    return false;
  }
  if (dense.isSplat()) {
    double d = dense.getSplatValue<llvm::APFloat>().convertToDouble();
    return std::isfinite(d) && std::abs(d) <= kMaxSafeFiniteMagnitude;
  }
  if (dense.getNumElements() > 4096) {
    return false;
  }
  for (const llvm::APFloat& apf : dense.getValues<llvm::APFloat>()) {
    double d = apf.convertToDouble();
    if (!std::isfinite(d) || std::abs(d) > kMaxSafeFiniteMagnitude) {
      return false;
    }
  }
  return true;
}

static bool IsBoundedFloatConvert(mlir::stablehlo::ConvertOp conv,
                                  llvm::DenseMap<mlir::Value, bool>& memo,
                                  int depth) {
  mlir::RankedTensorType in_t;
  mlir::RankedTensorType out_t;
  if (!IsStaticRankedTensor(conv.getOperand().getType(), &in_t) ||
      !IsStaticRankedTensor(conv.getType(), &out_t)) {
    return false;
  }
  auto out_flt = mlir::dyn_cast<mlir::FloatType>(out_t.getElementType());
  if (!out_flt) {
    return false;
  }
  if (auto in_int = mlir::dyn_cast<mlir::IntegerType>(in_t.getElementType())) {
    return in_int.getWidth() <= 16 ||
           (in_int.getWidth() <= 32 && out_flt.getWidth() >= 32);
  }
  if (auto in_flt = mlir::dyn_cast<mlir::FloatType>(in_t.getElementType())) {
    if (in_flt.getWidth() <= out_flt.getWidth()) {
      return IsProvablyFiniteAndAboveLowestFloat(conv.getOperand(), memo,
                                                 depth + 1);
    }
  }
  return false;
}

static bool IsBoundedFloatSelect(mlir::stablehlo::SelectOp sel,
                                 llvm::DenseMap<mlir::Value, bool>& memo,
                                 int depth) {
  if (IsBoolSplatConstant(sel.getPred(), true)) {
    return IsProvablyFiniteAndAboveLowestFloat(sel.getOnTrue(), memo,
                                               depth + 1);
  }
  if (IsBoolSplatConstant(sel.getPred(), false)) {
    return IsProvablyFiniteAndAboveLowestFloat(sel.getOnFalse(), memo,
                                               depth + 1);
  }
  if (auto cmp = sel.getPred().getDefiningOp<mlir::stablehlo::CompareOp>()) {
    bool uniform_val = false;
    if (EvaluateAffineCompareUniformly(cmp, uniform_val)) {
      mlir::Value chosen = uniform_val ? sel.getOnTrue() : sel.getOnFalse();
      return IsProvablyFiniteAndAboveLowestFloat(chosen, memo, depth + 1);
    }
  }
  return IsProvablyFiniteAndAboveLowestFloat(sel.getOnTrue(), memo,
                                             depth + 1) &&
         IsProvablyFiniteAndAboveLowestFloat(sel.getOnFalse(), memo, depth + 1);
}

static bool IsSafeDotOperand(mlir::Value val,
                             llvm::DenseMap<mlir::Value, bool>& memo,
                             int depth) {
  val = PeelIdentityBroadcast(val);
  while (true) {
    if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      val = PeelIdentityBroadcast(resh.getOperand());
      continue;
    }
    if (auto tr = val.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
      val = PeelIdentityBroadcast(tr.getOperand());
      continue;
    }
    if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      val = PeelIdentityBroadcast(conv.getOperand());
      continue;
    }
    if (auto mul = val.getDefiningOp<mlir::stablehlo::MulOp>()) {
      if (ExtractSplatFloatValue(mul.getRhs()).has_value()) {
        val = PeelIdentityBroadcast(mul.getLhs());
        continue;
      }
      if (ExtractSplatFloatValue(mul.getLhs()).has_value()) {
        val = PeelIdentityBroadcast(mul.getRhs());
        continue;
      }
    }
    break;
  }
  if (mlir::isa<mlir::BlockArgument>(val)) {
    return true;
  }
  return IsProvablyFiniteAndAboveLowestFloat(val, memo, depth + 1);
}

// Recursively verifies that a floating-point value `val` is provably finite
// (`!= -inf`, `!= +inf`, `!= NaN`) and bounded below by `-FLT_MAX` across
// relative-positional bias subgraphs.
bool IsProvablyFiniteAndAboveLowestFloat(
    mlir::Value val, llvm::DenseMap<mlir::Value, bool>& memo, int depth) {
  if (depth > kMaxWalkDepth) {
    return false;
  }
  val = PeelIdentityBroadcast(val);
  auto it = memo.find(val);
  if (it != memo.end()) {
    return it->second;
  }
  auto record = [&](bool res) {
    memo[val] = res;
    return res;
  };

  if (auto v = ExtractSplatFloatValue(val)) {
    return record(std::isfinite(*v) && std::abs(*v) <= kMaxSafeFiniteMagnitude);
  }
  if (auto cst = val.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
    return record(IsBoundedFloatConstant(cst));
  }
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    return record(IsProvablyFiniteAndAboveLowestFloat(bcast.getOperand(), memo,
                                                      depth + 1));
  }
  if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    return record(IsProvablyFiniteAndAboveLowestFloat(resh.getOperand(), memo,
                                                      depth + 1));
  }
  if (auto tr = val.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    return record(
        IsProvablyFiniteAndAboveLowestFloat(tr.getOperand(), memo, depth + 1));
  }
  if (auto slice = val.getDefiningOp<mlir::stablehlo::SliceOp>()) {
    return record(IsProvablyFiniteAndAboveLowestFloat(slice.getOperand(), memo,
                                                      depth + 1));
  }
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    return record(IsBoundedFloatConvert(conv, memo, depth));
  }
  if (auto concat = val.getDefiningOp<mlir::stablehlo::ConcatenateOp>()) {
    for (mlir::Value opnd : concat.getOperands()) {
      if (!IsProvablyFiniteAndAboveLowestFloat(opnd, memo, depth + 1)) {
        return record(false);
      }
    }
    return record(true);
  }
  if (auto pad = val.getDefiningOp<mlir::stablehlo::PadOp>()) {
    return record(IsProvablyFiniteAndAboveLowestFloat(pad.getOperand(), memo,
                                                      depth + 1) &&
                  IsProvablyFiniteAndAboveLowestFloat(pad.getPaddingValue(),
                                                      memo, depth + 1));
  }
  if (auto add = val.getDefiningOp<mlir::stablehlo::AddOp>()) {
    return record(
        IsProvablyFiniteAndAboveLowestFloat(add.getLhs(), memo, depth + 1) &&
        IsProvablyFiniteAndAboveLowestFloat(add.getRhs(), memo, depth + 1));
  }
  if (auto sub = val.getDefiningOp<mlir::stablehlo::SubtractOp>()) {
    return record(
        IsProvablyFiniteAndAboveLowestFloat(sub.getLhs(), memo, depth + 1) &&
        IsProvablyFiniteAndAboveLowestFloat(sub.getRhs(), memo, depth + 1));
  }
  if (auto mul = val.getDefiningOp<mlir::stablehlo::MulOp>()) {
    return record(
        IsProvablyFiniteAndAboveLowestFloat(mul.getLhs(), memo, depth + 1) &&
        IsProvablyFiniteAndAboveLowestFloat(mul.getRhs(), memo, depth + 1));
  }
  if (auto sel = val.getDefiningOp<mlir::stablehlo::SelectOp>()) {
    return record(IsBoundedFloatSelect(sel, memo, depth));
  }
  if (auto gather = val.getDefiningOp<mlir::stablehlo::GatherOp>()) {
    return record(IsProvablyFiniteAndAboveLowestFloat(gather.getOperand(), memo,
                                                      depth + 1));
  }
  if (auto dot = val.getDefiningOp<mlir::stablehlo::DotOp>()) {
    return record(IsSafeDotOperand(dot.getLhs(), memo, depth + 1) &&
                  IsSafeDotOperand(dot.getRhs(), memo, depth + 1));
  }
  if (auto dot_gen = val.getDefiningOp<mlir::stablehlo::DotGeneralOp>()) {
    return record(IsSafeDotOperand(dot_gen.getLhs(), memo, depth + 1) &&
                  IsSafeDotOperand(dot_gen.getRhs(), memo, depth + 1));
  }
  if (auto sin_op = val.getDefiningOp<mlir::stablehlo::SineOp>()) {
    return record(IsProvablyFiniteAndAboveLowestFloat(sin_op.getOperand(), memo,
                                                      depth + 1));
  }
  if (auto cos_op = val.getDefiningOp<mlir::stablehlo::CosineOp>()) {
    return record(IsProvablyFiniteAndAboveLowestFloat(cos_op.getOperand(), memo,
                                                      depth + 1));
  }
  return record(false);
}

bool IsSingleDimAndReduce(mlir::stablehlo::ReduceOp reduce_op) {
  // GENERIC: semantic single-input single-result boolean AndOp reduction.
  if (!reduce_op || reduce_op.getInputs().size() != 1 ||
      reduce_op.getResults().size() != 1 ||
      reduce_op.getInitValues().size() != 1) {
    return false;
  }
  if (!IsBoolSplatConstant(reduce_op.getInitValues()[0], true)) {
    return false;
  }
  mlir::Region& body = reduce_op.getBody();
  if (!body.hasOneBlock()) {
    return false;
  }
  mlir::Block& block = body.front();
  // GENERIC: semantic binary reduction combiner takes 2 scalar arguments.
  if (block.getNumArguments() != 2) {
    return false;
  }
  auto ret_op =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  if (!ret_op || ret_op.getOperands().size() != 1) {
    return false;
  }
  return mlir::isa_and_nonnull<mlir::stablehlo::AndOp>(
      ret_op.getOperands()[0].getDefiningOp());
}

static bool TryFoldCompareAgainstNegInf(
    mlir::stablehlo::CompareOp cmp, mlir::RankedTensorType cmp_t,
    llvm::DenseMap<mlir::Value, bool>& finite_memo) {
  if (cmp.getComparisonDirection() !=
      mlir::stablehlo::ComparisonDirection::EQ) {
    return false;
  }
  mlir::Value mask_val;
  if (IsNegativeInfinitySplat(cmp.getRhs())) {
    mask_val = cmp.getLhs();
  } else if (IsNegativeInfinitySplat(cmp.getLhs())) {
    mask_val = cmp.getRhs();
  }
  if (mask_val && IsProvablyFiniteAndAboveLowestFloat(mask_val, finite_memo)) {
    mlir::OpBuilder builder(cmp);
    mlir::Value c_false =
        CreateBoolSplatConstant(builder, cmp.getLoc(), cmp_t, false);
    cmp.replaceAllUsesWith(c_false);
    cmp.erase();
    return true;
  }
  return false;
}

static bool TryFoldAndReduction(mlir::stablehlo::ReduceOp red) {
  mlir::RankedTensorType red_t;
  mlir::RankedTensorType in_t;
  if (!IsSingleDimAndReduce(red) ||
      !IsStaticRankedTensor(red.getResult(0).getType(), &red_t) ||
      !IsStaticRankedTensor(red.getInputs()[0].getType(), &in_t) ||
      !IsBoolSplatConstant(red.getInputs()[0], false)) {
    return false;
  }
  for (int64_t d : red.getDimensions()) {
    if (d < 0 || d >= in_t.getRank() || in_t.getDimSize(d) <= 0) {
      return false;
    }
  }
  mlir::OpBuilder builder(red);
  mlir::Value c_false =
      CreateBoolSplatConstant(builder, red.getLoc(), red_t, false);
  red.getResult(0).replaceAllUsesWith(c_false);
  red.erase();
  return true;
}

static bool TryFoldTrivialSelect(mlir::stablehlo::SelectOp sel) {
  if (IsBoolSplatConstant(sel.getPred(), true)) {
    sel.replaceAllUsesWith(sel.getOnTrue());
    sel.erase();
    return true;
  }
  if (IsBoolSplatConstant(sel.getPred(), false)) {
    sel.replaceAllUsesWith(sel.getOnFalse());
    sel.erase();
    return true;
  }
  return false;
}

static bool TryFoldMaxAgainstLowestFloat(
    mlir::stablehlo::MaxOp max_op,
    llvm::DenseMap<mlir::Value, bool>& finite_memo) {
  mlir::Value mask_val;
  if (IsLowestFiniteFloatSplat(max_op.getRhs())) {
    mask_val = max_op.getLhs();
  } else if (IsLowestFiniteFloatSplat(max_op.getLhs())) {
    mask_val = max_op.getRhs();
  }
  if (mask_val && IsProvablyFiniteAndAboveLowestFloat(mask_val, finite_memo)) {
    max_op.replaceAllUsesWith(mask_val);
    max_op.erase();
    return true;
  }
  return false;
}

// Pattern 3: Folds uniform affine-iota CompareOp, folds `CompareOp(mask, -inf,
// EQ)` to `false` and `MaxOp(mask, -FLT_MAX)` to `mask` when `mask` is provably
// finite (`>= -FLT_MAX`), and simplifies `ReduceOp(false, true, AndOp)` and
// `SelectOp(const_bool, on_true, on_false)`.
bool TryFoldFiniteSdpaBiasAndBooleanIdentities(
    mlir::Operation* op, llvm::DenseMap<mlir::Value, bool>& finite_memo) {
  if (auto cmp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op)) {
    mlir::RankedTensorType cmp_t;
    if (!IsStaticRankedTensor(cmp.getType(), &cmp_t)) {
      return false;
    }
    bool uniform_val = false;
    if (EvaluateAffineCompareUniformly(cmp, uniform_val)) {
      mlir::OpBuilder builder(cmp);
      mlir::Value c =
          CreateBoolSplatConstant(builder, cmp.getLoc(), cmp_t, uniform_val);
      cmp.replaceAllUsesWith(c);
      cmp.erase();
      return true;
    }
    return TryFoldCompareAgainstNegInf(cmp, cmp_t, finite_memo);
  }
  if (auto red = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
    return TryFoldAndReduction(red);
  }
  if (auto sel = mlir::dyn_cast<mlir::stablehlo::SelectOp>(op)) {
    return TryFoldTrivialSelect(sel);
  }
  if (auto max_op = mlir::dyn_cast<mlir::stablehlo::MaxOp>(op)) {
    return TryFoldMaxAgainstLowestFloat(max_op, finite_memo);
  }
  return false;
}

struct ScaledDotPathMatch {
  mlir::stablehlo::MulOp mul_op;
  double scale = 0.0;
  mlir::Value unscaled_input;
};

static std::optional<ScaledDotPathMatch> MatchPreDotScaling(
    mlir::Operation* user, mlir::Value cur, mlir::Type start_elem_t) {
  auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(user);
  if (!mul || !mul.getResult().hasOneUse()) {
    return std::nullopt;
  }
  auto mul_t = mlir::dyn_cast<mlir::RankedTensorType>(mul.getType());
  if (!mul_t || mul_t.getElementType() != start_elem_t) {
    return std::nullopt;
  }
  mlir::Value other = (mul.getLhs() == cur) ? mul.getRhs() : mul.getLhs();
  auto s = ExtractSplatFloatValue(other);
  if (!s.has_value()) {
    return std::nullopt;
  }
  if (!mlir::isa<mlir::stablehlo::DotGeneralOp, mlir::stablehlo::DotOp>(
          *mul.getResult().user_begin())) {
    return std::nullopt;
  }
  return ScaledDotPathMatch{mul, *s, cur};
}

static std::optional<ScaledDotPathMatch> MatchPostDotScaling(
    mlir::Operation* dot_user, mlir::Type start_elem_t) {
  mlir::Value post = dot_user->getResult(0);
  for (int k = 0; k < kMaxWalkDepth; ++k) {
    if (!post.hasOneUse()) {
      return std::nullopt;
    }
    mlir::Operation* post_user = *post.user_begin();
    if (mlir::isa<mlir::stablehlo::ReshapeOp, mlir::stablehlo::TransposeOp>(
            post_user)) {
      post = post_user->getResult(0);
      continue;
    }
    if (auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(post_user)) {
      auto mul_t = mlir::dyn_cast<mlir::RankedTensorType>(mul.getType());
      if (!mul_t || mul_t.getElementType() != start_elem_t) {
        return std::nullopt;
      }
      mlir::Value other = (mul.getLhs() == post) ? mul.getRhs() : mul.getLhs();
      auto s = ExtractSplatFloatValue(other);
      if (!s.has_value()) {
        return std::nullopt;
      }
      return ScaledDotPathMatch{mul, *s, post};
    }
    return std::nullopt;
  }
  return std::nullopt;
}

// Matches a single-use Reshape/Transpose chain from `start_val` to either:
// (a) `MulOp(view, scale) -> DotGeneralOp` (pre-dot scaling), or
// (b) `DotGeneralOp(view, ...) -> Reshape/Transpose -> MulOp(..., scale)`
//     (post-dot scaling).
std::optional<ScaledDotPathMatch> MatchScaledDotPath(mlir::Value start_val) {
  auto start_t = mlir::dyn_cast<mlir::RankedTensorType>(start_val.getType());
  if (!start_t) {
    return std::nullopt;
  }
  mlir::Type start_elem_t = start_t.getElementType();
  mlir::Value cur = start_val;
  for (int depth = 0; depth < 6; ++depth) {
    if (!cur.hasOneUse()) {
      return std::nullopt;
    }
    mlir::Operation* user = *cur.user_begin();
    if (mlir::isa<mlir::stablehlo::ReshapeOp, mlir::stablehlo::TransposeOp>(
            user)) {
      cur = user->getResult(0);
      continue;
    }
    if (auto pre = MatchPreDotScaling(user, cur, start_elem_t)) {
      return pre;
    }
    if (mlir::isa<mlir::stablehlo::DotGeneralOp, mlir::stablehlo::DotOp>(
            user)) {
      return MatchPostDotScaling(user, start_elem_t);
    }
    return std::nullopt;
  }
  return std::nullopt;
}

// Pattern 4: When a query tensor `Q` has two `ReshapeOp` users that both feed
// `DotGeneralOp` contractions scaled by the same scalar constant `scale`,
// hoists `Q_scaled = MulOp(Q, BroadcastInDim(scale))` before both `ReshapeOp`s
// and eliminates the two downstream `MulOp`s.
bool TryHoistSharedScalarMulBeforeDualDotViews(
    mlir::Operation* op, llvm::DenseSet<mlir::Operation*>& erased) {
  auto r_anchor = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op);
  if (!r_anchor) {
    return false;
  }
  mlir::Value q = r_anchor.getOperand();
  mlir::RankedTensorType q_t;
  if (!IsStaticRankedTensor(q.getType(), &q_t) ||
      !mlir::isa<mlir::FloatType>(q_t.getElementType())) {
    return false;
  }
  llvm::SmallVector<mlir::stablehlo::ReshapeOp, 2> reshapes;
  for (mlir::Operation* user : q.getUsers()) {
    auto r = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(user);
    if (!r) {
      return false;
    }
    reshapes.push_back(r);
  }
  // GENERIC: semantic dual-branch query consumer (content Q@K^T + positional
  // Q@R^T) has 2 ReshapeOp views.
  if (reshapes.size() != 2) {
    return false;
  }
  auto m0 = MatchScaledDotPath(reshapes[0].getResult());
  auto m1 = MatchScaledDotPath(reshapes[1].getResult());
  if (!m0.has_value() || !m1.has_value() || m0->mul_op == m1->mul_op ||
      m0->scale != m1->scale) {
    return false;
  }

  mlir::OpBuilder builder(r_anchor->getContext());
  if (mlir::Operation* def_op = q.getDefiningOp()) {
    builder.setInsertionPointAfter(def_op);
  } else if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(q)) {
    builder.setInsertionPointToStart(arg.getOwner());
  } else {
    builder.setInsertionPointToStart(r_anchor->getBlock());
  }
  mlir::Location loc = r_anchor.getLoc();
  auto flt_t = mlir::cast<mlir::FloatType>(q_t.getElementType());
  auto scalar_t = mlir::RankedTensorType::get({}, flt_t);
  llvm::APFloat scale_apf(m0->scale);
  bool loses_info = false;
  scale_apf.convert(flt_t.getFloatSemantics(),
                    llvm::APFloat::rmNearestTiesToEven, &loses_info);
  mlir::Value scale_cst = builder.create<mlir::stablehlo::ConstantOp>(
      loc, scalar_t, mlir::DenseElementsAttr::get(scalar_t, scale_apf));
  mlir::Value scale_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, q_t, scale_cst, builder.getDenseI64ArrayAttr({}));
  mlir::Value q_scaled =
      builder.create<mlir::stablehlo::MulOp>(loc, q_t, q, scale_bcast);

  reshapes[0]->setOperand(0, q_scaled);
  reshapes[1]->setOperand(0, q_scaled);

  for (ScaledDotPathMatch m : {*m0, *m1}) {
    m.mul_op.replaceAllUsesWith(m.unscaled_input);
    erased.insert(m.mul_op.getOperation());
    m.mul_op.erase();
  }
  return true;
}

static bool TryApplyRewritesToOp(
    mlir::Operation* op, llvm::DenseSet<mlir::Operation*>& erased,
    llvm::DenseMap<mlir::Value, bool>& finite_memo) {
  if (auto add_op = mlir::dyn_cast<mlir::stablehlo::AddOp>(op)) {
    if (TryPushChannelTransposeAcrossBiasAdd(add_op)) {
      erased.insert(add_op.getOperation());
      return true;
    }
  }
  if (TrySimplifyTransposePairsAndSlices(op, erased)) {
    return true;
  }
  if (TryCommuteChannelTransposeDownstreamAcrossGlu(op, erased)) {
    return true;
  }
  if (TryCommuteTransposeUpstreamAcrossCancellingSilu(op)) {
    erased.insert(op);
    return true;
  }
  if (TryFoldFiniteSdpaBiasAndBooleanIdentities(op, finite_memo)) {
    erased.insert(op);
    return true;
  }
  if (TryHoistSharedScalarMulBeforeDualDotViews(op, erased)) {
    return true;
  }
  return false;
}

class CanonicalizePointwiseConvAndFiniteSdpaBiasesPass
    : public mlir::PassWrapper<CanonicalizePointwiseConvAndFiniteSdpaBiasesPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CanonicalizePointwiseConvAndFiniteSdpaBiasesPass)

  llvm::StringRef getArgument() const final {
    return "canonicalize-pointwise-conv-and-finite-sdpa-biases";
  }

  llvm::StringRef getDescription() const final {
    return "Lowers unit-batch 1D pointwise ConvolutionOp to DotGeneralOp, "
           "commutes 3D channel transposes across GLU/SiLU, "
           "and folds finite relative-positional SDPA mask checks.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    // Step 1: Lower unit-batch 1D pointwise ConvolutionOp to DotGeneralOp.
    llvm::SmallVector<mlir::stablehlo::ConvolutionOp, 16> conv_ops;
    func.walk(
        [&](mlir::stablehlo::ConvolutionOp op) { conv_ops.push_back(op); });
    for (mlir::stablehlo::ConvolutionOp conv_op : conv_ops) {
      if (conv_op && conv_op->getBlock()) {
        TryLowerUnitBatch1DPointwiseConvToDot(conv_op);
      }
    }

    // Step 2: Fixed-point iteration for transpose commutation, bias folding,
    // finite SDPA mask canonicalization, and shared query scale hoisting.
    for (int iter = 0; iter < kMaxPassIterations; ++iter) {
      bool changed = false;
      llvm::DenseMap<mlir::Value, bool> finite_memo;
      llvm::DenseSet<mlir::Operation*> erased;
      llvm::SmallVector<mlir::Operation*, 64> ops;
      func.walk([&](mlir::Operation* op) {
        if (!mlir::isa<mlir::func::FuncOp, mlir::func::ReturnOp,
                       mlir::stablehlo::ReturnOp>(op)) {
          ops.push_back(op);
        }
      });
      for (mlir::Operation* op : ops) {
        if (!op || erased.contains(op)) {
          continue;
        }
        if (TryApplyRewritesToOp(op, erased, finite_memo)) {
          changed = true;
          finite_memo.clear();
        }
      }
      if (!changed) {
        break;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
createCanonicalizePointwiseConvAndFiniteSdpaBiasesPass() {
  return std::make_unique<CanonicalizePointwiseConvAndFiniteSdpaBiasesPass>();
}

}  // namespace torch_tpu
