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

#include "csrc/ops/scaled_dot_product_attention/helpers.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBase.h"
#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/ArrayRef.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

// Batch dim (un)flattening is a valid reshape reassociation as long as at most
// one non-1 batch dim exists when a batch dim is dynamic, which
// PrepareAttentionLogits enforces, so failures here are internal errors.
mlir::MlirOp ReshapeBatchDimsOrDie(mlir::MlirOp mlir_op,
                                   absl::Span<const int64_t> shape) {
  absl::StatusOr<mlir::MlirOp> reshaped =
      ReshapeToStaticDimensions(mlir_op, CopyIntVector(shape));
  ABSL_CHECK_OK(reshaped.status());  // CRASH_OK
  return *reshaped;
}

// Recursion depth bound when walking MLIR producer defs.
constexpr int kMaxMaskAnalysisDepth = 32;

std::optional<bool> EvalPredicateTautology(mlir::Value pred, int depth = 0);
std::optional<double> ExtractSplatFloatValue(mlir::Value val, int depth = 0);

std::optional<double> ExtractSelectSplatFloatValue(
    mlir::stablehlo::SelectOp select, int depth) {
  auto on_true = ExtractSplatFloatValue(select.getOnTrue(), depth + 1);
  auto on_false = ExtractSplatFloatValue(select.getOnFalse(), depth + 1);
  if (on_true.has_value() && on_false.has_value() && *on_true == *on_false) {
    return *on_true;
  }
  auto tautology = EvalPredicateTautology(select.getPred(), depth + 1);
  if (tautology.has_value()) {
    return *tautology ? on_true : on_false;
  }
  return std::nullopt;
}

std::optional<double> ExtractConstantSplatFloatValue(
    mlir::stablehlo::ConstantOp cst) {
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || (!dense.isSplat() && dense.getNumElements() != 1)) {
    return std::nullopt;
  }
  mlir::Attribute elem_attr =
      dense.isSplat() ? dense.getSplatValue<mlir::Attribute>()
                      : (*dense.getValues<mlir::Attribute>().begin());
  if (auto float_attr = mlir::dyn_cast<mlir::FloatAttr>(elem_attr)) {
    return float_attr.getValueAsDouble();
  }
  return std::nullopt;
}

std::optional<double> ExtractConvertSplatFloatValue(
    mlir::stablehlo::ConvertOp conv, int depth) {
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(conv.getType());
  if (!in_type || !out_type ||
      !mlir::isa<mlir::FloatType>(in_type.getElementType()) ||
      !mlir::isa<mlir::FloatType>(out_type.getElementType())) {
    return std::nullopt;
  }
  auto in_float = mlir::cast<mlir::FloatType>(in_type.getElementType());
  auto out_float = mlir::cast<mlir::FloatType>(out_type.getElementType());
  if (in_float.getIntOrFloatBitWidth() < out_float.getIntOrFloatBitWidth() ||
      llvm::APFloat::semanticsMaxExponent(in_float.getFloatSemantics()) <
          llvm::APFloat::semanticsMaxExponent(out_float.getFloatSemantics())) {
    return std::nullopt;
  }
  return ExtractSplatFloatValue(conv.getOperand(), depth + 1);
}

std::optional<double> ExtractSplatFloatValue(mlir::Value val, int depth) {
  if (depth > kMaxMaskAnalysisDepth || !val.getDefiningOp()) {
    return std::nullopt;
  }
  mlir::Operation* op = val.getDefiningOp();
  if (auto select = mlir::dyn_cast<mlir::stablehlo::SelectOp>(op)) {
    return ExtractSelectSplatFloatValue(select, depth);
  }
  if (auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
    return ExtractConstantSplatFloatValue(cst);
  }
  if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
    return ExtractConvertSplatFloatValue(conv, depth);
  }
  if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
    return ExtractSplatFloatValue(bcast.getOperand(), depth + 1);
  }
  if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastOp>(op)) {
    return ExtractSplatFloatValue(bcast.getOperand(), depth + 1);
  }
  if (auto reshape = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
    return ExtractSplatFloatValue(reshape.getOperand(), depth + 1);
  }
  return std::nullopt;
}

bool IsZeroSplat(mlir::Value val) {
  std::optional<double> v = ExtractSplatFloatValue(val);
  return v.has_value() && *v == 0.0;
}

bool IsLargeNegativeMaskSplat(mlir::Value val) {
  std::optional<double> v = ExtractSplatFloatValue(val);
  if (!v.has_value()) {
    return false;
  }
  if (std::isinf(*v) && *v < 0.0) {
    return true;
  }
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!type) {
    return false;
  }
  mlir::Type elem_type = type.getElementType();
  if (elem_type.isF32()) {
    return *v <= -1e9;
  }
  if (elem_type.isBF16()) {
    return *v <= -3.38e38;
  }
  if (elem_type.isF16()) {
    return *v <= -65504.0;
  }
  return false;
}

struct IndexExpr {
  llvm::SmallVector<int64_t, 4> coeffs;
  int64_t offset = 0;
};

bool AffineRangeFitsInElementType(const IndexExpr& expr,
                                  mlir::RankedTensorType type) {
  mlir::Type elem_type = type.getElementType();
  if (!elem_type.isInteger() || elem_type.isUnsignedInteger()) {
    return false;
  }
  const unsigned bitwidth = elem_type.getIntOrFloatBitWidth();
  if (bitwidth != 32 && bitwidth != 64) {
    return false;
  }
  const __int128 min_bound =
      (bitwidth == 32)
          ? static_cast<__int128>(std::numeric_limits<int32_t>::min())
          : static_cast<__int128>(std::numeric_limits<int64_t>::min());
  const __int128 max_bound =
      (bitwidth == 32)
          ? static_cast<__int128>(std::numeric_limits<int32_t>::max())
          : static_cast<__int128>(std::numeric_limits<int64_t>::max());

  __int128 min_val = static_cast<__int128>(expr.offset);
  __int128 max_val = static_cast<__int128>(expr.offset);
  const int64_t rank = type.getRank();
  if (static_cast<int64_t>(expr.coeffs.size()) != rank) {
    return false;
  }
  for (int64_t d = 0; d < rank; ++d) {
    const int64_t dim_size = type.getDimSize(d);
    if (dim_size <= 0) {
      return false;
    }
    const __int128 max_idx = static_cast<__int128>(dim_size - 1);
    const __int128 c = static_cast<__int128>(expr.coeffs[d]);
    if (c > 0) {
      max_val += c * max_idx;
    } else if (c < 0) {
      min_val += c * max_idx;
    }
  }
  return min_val >= min_bound && max_val <= max_bound;
}

std::optional<IndexExpr> EvalIndexExprImpl(mlir::Value val, int depth);

std::optional<IndexExpr> EvalIndexExpr(mlir::Value val, int depth = 0) {
  if (depth > kMaxMaskAnalysisDepth || !val.getDefiningOp()) {
    return std::nullopt;
  }
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!type || !type.hasStaticShape()) {
    return std::nullopt;
  }
  auto expr = EvalIndexExprImpl(val, depth);
  if (!expr.has_value() || !AffineRangeFitsInElementType(*expr, type)) {
    return std::nullopt;
  }
  return expr;
}

std::optional<IndexExpr> EvalIotaIndexExpr(mlir::stablehlo::IotaOp iota,
                                           mlir::RankedTensorType type) {
  mlir::Type elem_type = type.getElementType();
  if (!elem_type.isInteger() || elem_type.isUnsignedInteger() ||
      elem_type.getIntOrFloatBitWidth() < 32) {
    return std::nullopt;
  }
  IndexExpr expr;
  expr.coeffs.assign(type.getRank(), 0);
  const int64_t iota_dim = iota.getIotaDimension();
  if (type.getDimSize(iota_dim) > 1) {
    expr.coeffs[iota_dim] = 1;
  }
  expr.offset = 0;
  return expr;
}

std::optional<IndexExpr> EvalConstantIndexExpr(mlir::stablehlo::ConstantOp cst,
                                               mlir::RankedTensorType type) {
  mlir::Type elem_type = type.getElementType();
  if (!elem_type.isInteger() || elem_type.isUnsignedInteger() ||
      elem_type.getIntOrFloatBitWidth() < 32) {
    return std::nullopt;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || (!dense.isSplat() && dense.getNumElements() != 1)) {
    return std::nullopt;
  }
  mlir::Attribute elem_attr =
      dense.isSplat() ? dense.getSplatValue<mlir::Attribute>()
                      : (*dense.getValues<mlir::Attribute>().begin());
  auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(elem_attr);
  if (!int_attr) {
    return std::nullopt;
  }
  IndexExpr expr;
  expr.coeffs.assign(type.getRank(), 0);
  expr.offset = int_attr.getInt();
  return expr;
}

std::optional<IndexExpr> EvalConvertIndexExpr(mlir::stablehlo::ConvertOp conv,
                                              mlir::RankedTensorType type,
                                              int depth) {
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
  if (!in_type) {
    return std::nullopt;
  }
  mlir::Type in_elem = in_type.getElementType();
  mlir::Type out_elem = type.getElementType();
  if (!in_elem.isInteger() || in_elem.isUnsignedInteger() ||
      in_elem.getIntOrFloatBitWidth() < 32) {
    return std::nullopt;
  }
  if (!out_elem.isInteger() || out_elem.isUnsignedInteger() ||
      out_elem.getIntOrFloatBitWidth() < in_elem.getIntOrFloatBitWidth()) {
    return std::nullopt;
  }
  return EvalIndexExpr(conv.getOperand(), depth + 1);
}

std::optional<IndexExpr> EvalBroadcastInDimIndexExpr(
    mlir::stablehlo::BroadcastInDimOp bcast, mlir::RankedTensorType type,
    int depth) {
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
  if (!in_type || !in_type.hasStaticShape()) {
    return std::nullopt;
  }
  auto sub = EvalIndexExpr(bcast.getOperand(), depth + 1);
  if (!sub.has_value()) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> bcast_dims = bcast.getBroadcastDimensions();
  IndexExpr expr;
  expr.coeffs.assign(type.getRank(), 0);
  expr.offset = sub->offset;
  for (size_t i = 0; i < bcast_dims.size(); ++i) {
    if (in_type.getDimSize(i) > 1) {
      expr.coeffs[bcast_dims[i]] = sub->coeffs[i];
    }
  }
  return expr;
}

std::optional<IndexExpr> EvalReshapeIndexExpr(
    mlir::stablehlo::ReshapeOp reshape, mlir::RankedTensorType type,
    int depth) {
  auto sub = EvalIndexExpr(reshape.getOperand(), depth + 1);
  if (!sub.has_value()) {
    return std::nullopt;
  }
  auto in_type =
      mlir::cast<mlir::RankedTensorType>(reshape.getOperand().getType());
  llvm::ArrayRef<int64_t> in_shape = in_type.getShape();
  llvm::ArrayRef<int64_t> out_shape = type.getShape();
  IndexExpr expr;
  const int64_t rank = type.getRank();
  expr.coeffs.assign(rank, 0);
  expr.offset = sub->offset;
  int64_t in_idx = 0;
  int64_t out_idx = 0;
  while (in_idx < in_type.getRank() && out_idx < rank) {
    if (in_shape[in_idx] == out_shape[out_idx]) {
      expr.coeffs[out_idx] = sub->coeffs[in_idx];
      ++in_idx;
      ++out_idx;
    } else if (in_shape[in_idx] == 1 && sub->coeffs[in_idx] == 0) {
      ++in_idx;
    } else if (out_shape[out_idx] == 1) {
      ++out_idx;
    } else {
      return std::nullopt;
    }
  }
  while (in_idx < in_type.getRank()) {
    if (in_shape[in_idx] != 1 || sub->coeffs[in_idx] != 0) {
      return std::nullopt;
    }
    ++in_idx;
  }
  return expr;
}

std::optional<IndexExpr> EvalAddIndexExpr(mlir::stablehlo::AddOp add,
                                          mlir::RankedTensorType type,
                                          int depth) {
  auto lhs = EvalIndexExpr(add.getLhs(), depth + 1);
  auto rhs = EvalIndexExpr(add.getRhs(), depth + 1);
  if (!lhs.has_value() || !rhs.has_value()) {
    return std::nullopt;
  }
  const int64_t rank = type.getRank();
  for (int64_t i = 0; i < rank; ++i) {
    if (__builtin_add_overflow(lhs->coeffs[i], rhs->coeffs[i],
                               &lhs->coeffs[i])) {
      return std::nullopt;
    }
  }
  if (__builtin_add_overflow(lhs->offset, rhs->offset, &lhs->offset)) {
    return std::nullopt;
  }
  return lhs;
}

std::optional<IndexExpr> EvalSubIndexExpr(mlir::stablehlo::SubtractOp sub,
                                          mlir::RankedTensorType type,
                                          int depth) {
  auto lhs = EvalIndexExpr(sub.getLhs(), depth + 1);
  auto rhs = EvalIndexExpr(sub.getRhs(), depth + 1);
  if (!lhs.has_value() || !rhs.has_value()) {
    return std::nullopt;
  }
  const int64_t rank = type.getRank();
  for (int64_t i = 0; i < rank; ++i) {
    if (__builtin_sub_overflow(lhs->coeffs[i], rhs->coeffs[i],
                               &lhs->coeffs[i])) {
      return std::nullopt;
    }
  }
  if (__builtin_sub_overflow(lhs->offset, rhs->offset, &lhs->offset)) {
    return std::nullopt;
  }
  return lhs;
}

std::optional<IndexExpr> EvalNegIndexExpr(mlir::stablehlo::NegOp neg,
                                          mlir::RankedTensorType type,
                                          int depth) {
  auto sub = EvalIndexExpr(neg.getOperand(), depth + 1);
  if (!sub.has_value()) {
    return std::nullopt;
  }
  const int64_t rank = type.getRank();
  for (int64_t i = 0; i < rank; ++i) {
    if (__builtin_sub_overflow(0, sub->coeffs[i], &sub->coeffs[i])) {
      return std::nullopt;
    }
  }
  if (__builtin_sub_overflow(0, sub->offset, &sub->offset)) {
    return std::nullopt;
  }
  return sub;
}

std::optional<IndexExpr> ScaleIndexExpr(const IndexExpr& base, int64_t scalar,
                                        int64_t rank) {
  IndexExpr expr;
  expr.coeffs.assign(rank, 0);
  for (int64_t i = 0; i < rank; ++i) {
    if (__builtin_mul_overflow(base.coeffs[i], scalar, &expr.coeffs[i])) {
      return std::nullopt;
    }
  }
  if (__builtin_mul_overflow(base.offset, scalar, &expr.offset)) {
    return std::nullopt;
  }
  return expr;
}

std::optional<IndexExpr> EvalMulIndexExpr(mlir::stablehlo::MulOp mul,
                                          mlir::RankedTensorType type,
                                          int depth) {
  auto lhs = EvalIndexExpr(mul.getLhs(), depth + 1);
  auto rhs = EvalIndexExpr(mul.getRhs(), depth + 1);
  if (!lhs.has_value() || !rhs.has_value()) {
    return std::nullopt;
  }
  const int64_t rank = type.getRank();
  const bool lhs_const =
      absl::c_all_of(lhs->coeffs, [](int64_t c) { return c == 0; });
  const bool rhs_const =
      absl::c_all_of(rhs->coeffs, [](int64_t c) { return c == 0; });
  if (lhs_const) {
    return ScaleIndexExpr(*rhs, lhs->offset, rank);
  }
  if (rhs_const) {
    return ScaleIndexExpr(*lhs, rhs->offset, rank);
  }
  return std::nullopt;
}

std::optional<IndexExpr> EvalIndexExprImpl(mlir::Value val, int depth) {
  auto type = mlir::cast<mlir::RankedTensorType>(val.getType());
  mlir::Operation* op = val.getDefiningOp();
  if (auto iota = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op)) {
    return EvalIotaIndexExpr(iota, type);
  }
  if (auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
    return EvalConstantIndexExpr(cst, type);
  }
  if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
    return EvalConvertIndexExpr(conv, type, depth);
  }
  if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
    return EvalBroadcastInDimIndexExpr(bcast, type, depth);
  }
  if (auto reshape = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
    return EvalReshapeIndexExpr(reshape, type, depth);
  }
  if (auto add = mlir::dyn_cast<mlir::stablehlo::AddOp>(op)) {
    return EvalAddIndexExpr(add, type, depth);
  }
  if (auto sub = mlir::dyn_cast<mlir::stablehlo::SubtractOp>(op)) {
    return EvalSubIndexExpr(sub, type, depth);
  }
  if (auto neg = mlir::dyn_cast<mlir::stablehlo::NegOp>(op)) {
    return EvalNegIndexExpr(neg, type, depth);
  }
  if (auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(op)) {
    return EvalMulIndexExpr(mul, type, depth);
  }
  return std::nullopt;
}

bool IsAllNonNegative(llvm::ArrayRef<int64_t> coeffs) {
  return absl::c_all_of(coeffs, [](int64_t c) { return c >= 0; });
}

bool IsAllZeroCoeffs(llvm::ArrayRef<int64_t> coeffs) {
  return absl::c_all_of(coeffs, [](int64_t c) { return c == 0; });
}

bool IsSupportedComparisonType(
    std::optional<mlir::stablehlo::ComparisonType> cmp_type) {
  return !cmp_type.has_value() ||
         *cmp_type == mlir::stablehlo::ComparisonType::SIGNED ||
         *cmp_type == mlir::stablehlo::ComparisonType::NOTYPE;
}

bool CheckNonNegGeZero(mlir::stablehlo::ComparisonDirection dir,
                       int64_t lhs_offset, int64_t rhs_offset) {
  using Dir = mlir::stablehlo::ComparisonDirection;
  return (dir == Dir::GE && lhs_offset >= rhs_offset) ||
         (dir == Dir::GT && lhs_offset > rhs_offset);
}

bool CheckZeroLeNonNeg(mlir::stablehlo::ComparisonDirection dir,
                       int64_t lhs_offset, int64_t rhs_offset) {
  using Dir = mlir::stablehlo::ComparisonDirection;
  return (dir == Dir::LE && lhs_offset <= rhs_offset) ||
         (dir == Dir::LT && lhs_offset < rhs_offset);
}

std::optional<bool> EvalComparePredicateTautology(
    mlir::stablehlo::CompareOp cmp, int depth) {
  if (!IsSupportedComparisonType(cmp.getCompareType())) {
    return std::nullopt;
  }
  auto lhs = EvalIndexExpr(cmp.getLhs(), depth + 1);
  auto rhs = EvalIndexExpr(cmp.getRhs(), depth + 1);
  if (!lhs.has_value() || !rhs.has_value()) {
    return std::nullopt;
  }
  const auto dir = cmp.getComparisonDirection();
  if (IsAllNonNegative(lhs->coeffs) && IsAllZeroCoeffs(rhs->coeffs)) {
    if (CheckNonNegGeZero(dir, lhs->offset, rhs->offset)) {
      return true;
    }
  }
  if (IsAllZeroCoeffs(lhs->coeffs) && IsAllNonNegative(rhs->coeffs)) {
    if (CheckZeroLeNonNeg(dir, lhs->offset, rhs->offset)) {
      return true;
    }
  }
  return std::nullopt;
}

std::optional<bool> EvalNotPredicateTautology(mlir::stablehlo::NotOp not_op,
                                              int depth) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(not_op.getType());
  if (!type || !type.getElementType().isInteger(1)) {
    return std::nullopt;
  }
  auto sub = EvalPredicateTautology(not_op.getOperand(), depth + 1);
  if (!sub.has_value()) {
    return std::nullopt;
  }
  return !*sub;
}

std::optional<bool> EvalConvertPredicateTautology(
    mlir::stablehlo::ConvertOp conv, int depth) {
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
  if (!in_type || !in_type.getElementType().isInteger(1)) {
    return std::nullopt;
  }
  return EvalPredicateTautology(conv.getOperand(), depth + 1);
}

std::optional<bool> EvalPredicateTautology(mlir::Value pred, int depth) {
  if (depth > kMaxMaskAnalysisDepth || !pred.getDefiningOp()) {
    return std::nullopt;
  }
  mlir::Operation* op = pred.getDefiningOp();
  if (auto not_op = mlir::dyn_cast<mlir::stablehlo::NotOp>(op)) {
    return EvalNotPredicateTautology(not_op, depth);
  }
  if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
    return EvalConvertPredicateTautology(conv, depth);
  }
  if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
    return EvalPredicateTautology(bcast.getOperand(), depth + 1);
  }
  if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastOp>(op)) {
    return EvalPredicateTautology(bcast.getOperand(), depth + 1);
  }
  if (auto reshape = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
    return EvalPredicateTautology(reshape.getOperand(), depth + 1);
  }
  if (auto cmp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op)) {
    return EvalComparePredicateTautology(cmp, depth);
  }
  return std::nullopt;
}

bool IsValidCausalBroadcastInDim(mlir::stablehlo::BroadcastInDimOp bcast) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
  if (!out_type || !in_type || out_type.getRank() < 2 ||
      in_type.getRank() < 2) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = bcast.getBroadcastDimensions();
  const int64_t in_rank = in_type.getRank();
  const int64_t out_rank = out_type.getRank();
  return dims[in_rank - 2] == out_rank - 2 &&
         dims[in_rank - 1] == out_rank - 1 &&
         in_type.getDimSize(in_rank - 2) == out_type.getDimSize(out_rank - 2) &&
         in_type.getDimSize(in_rank - 1) == out_type.getDimSize(out_rank - 1);
}

bool IsValidCausalReshape(mlir::stablehlo::ReshapeOp reshape) {
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(reshape.getOperand().getType());
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(reshape.getType());
  if (!in_type || !out_type || in_type.getRank() < 2 ||
      out_type.getRank() < 2) {
    return false;
  }
  const int64_t in_rank = in_type.getRank();
  const int64_t out_rank = out_type.getRank();
  return in_type.getDimSize(in_rank - 2) == out_type.getDimSize(out_rank - 2) &&
         in_type.getDimSize(in_rank - 1) == out_type.getDimSize(out_rank - 1);
}

bool PrefixCoeffsMatch(const IndexExpr& lhs, const IndexExpr& rhs,
                       int64_t count) {
  for (int64_t d = 0; d < count; ++d) {
    if (lhs.coeffs[d] != rhs.coeffs[d]) {
      return false;
    }
  }
  return true;
}

std::optional<bool> MatchCausalForward(mlir::stablehlo::ComparisonDirection dir,
                                       int64_t d0) {
  using Dir = mlir::stablehlo::ComparisonDirection;
  if ((dir == Dir::GE && d0 == 0) || (dir == Dir::GT && d0 == 1)) {
    return true;
  }
  if ((dir == Dir::LT && d0 == 0) || (dir == Dir::LE && d0 == 1)) {
    return false;
  }
  return std::nullopt;
}

std::optional<bool> MatchCausalReversed(
    mlir::stablehlo::ComparisonDirection dir, int64_t d0) {
  using Dir = mlir::stablehlo::ComparisonDirection;
  if ((dir == Dir::LE && d0 == 0) || (dir == Dir::LT && d0 == -1)) {
    return true;
  }
  if ((dir == Dir::GT && d0 == 0) || (dir == Dir::GE && d0 == -1)) {
    return false;
  }
  return std::nullopt;
}

std::optional<bool> MatchCausalDirection(
    mlir::stablehlo::ComparisonDirection dir, int64_t di, int64_t dj,
    int64_t d0) {
  if (di == 1 && dj == -1) {
    return MatchCausalForward(dir, d0);
  }
  if (di == -1 && dj == 1) {
    return MatchCausalReversed(dir, d0);
  }
  return std::nullopt;
}

std::optional<bool> MatchCausalCompare(mlir::stablehlo::CompareOp cmp,
                                       int depth) {
  if (!IsSupportedComparisonType(cmp.getCompareType())) {
    return std::nullopt;
  }
  auto lhs = EvalIndexExpr(cmp.getLhs(), depth + 1);
  auto rhs = EvalIndexExpr(cmp.getRhs(), depth + 1);
  if (!lhs.has_value() || !rhs.has_value()) {
    return std::nullopt;
  }
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(cmp.getType());
  if (!type || type.getRank() < 2) {
    return std::nullopt;
  }
  const int64_t rank = type.getRank();
  if (!PrefixCoeffsMatch(*lhs, *rhs, rank - 2)) {
    return std::nullopt;
  }
  const int64_t di = lhs->coeffs[rank - 2] - rhs->coeffs[rank - 2];
  const int64_t dj = lhs->coeffs[rank - 1] - rhs->coeffs[rank - 1];
  const int64_t d0 = lhs->offset - rhs->offset;
  return MatchCausalDirection(cmp.getComparisonDirection(), di, dj, d0);
}

std::optional<bool> MatchCausalPredicate(mlir::Value pred, int depth = 0) {
  if (depth > kMaxMaskAnalysisDepth || !pred.getDefiningOp()) {
    return std::nullopt;
  }
  mlir::Operation* op = pred.getDefiningOp();
  if (auto not_op = mlir::dyn_cast<mlir::stablehlo::NotOp>(op)) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(not_op.getType());
    if (!type || !type.getElementType().isInteger(1)) {
      return std::nullopt;
    }
    auto sub = MatchCausalPredicate(not_op.getOperand(), depth + 1);
    if (!sub.has_value()) {
      return std::nullopt;
    }
    return !*sub;
  }
  if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
    if (!IsValidCausalBroadcastInDim(bcast)) {
      return std::nullopt;
    }
    return MatchCausalPredicate(bcast.getOperand(), depth + 1);
  }
  if (auto reshape = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
    if (!IsValidCausalReshape(reshape)) {
      return std::nullopt;
    }
    return MatchCausalPredicate(reshape.getOperand(), depth + 1);
  }
  if (auto cmp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op)) {
    return MatchCausalCompare(cmp, depth);
  }
  return std::nullopt;
}

}  // namespace

mlir::MlirOp flatten_batch_dims(mlir::MlirOp mlir_op, int batch_size,
                                int num_batch_dims) {
  const Dimensions shape = GetStaticShape(mlir_op);
  int rank = shape.size();
  int num_non_batch_dims = rank - num_batch_dims;
  Dimensions new_dims(num_non_batch_dims + 1);
  new_dims[0] = batch_size;
  for (int i = 0; i < num_non_batch_dims; i++)
    new_dims[i + 1] = shape[i + num_batch_dims];

  return ReshapeBatchDimsOrDie(mlir_op, new_dims);
}

mlir::MlirOp unflatten_batch_dims(mlir::MlirOp mlir_op,
                                  absl::Span<const int64_t> shape) {
  return ReshapeBatchDimsOrDie(mlir_op, shape);
}

mlir::MlirOp unflatten_batch_dims(mlir::MlirOp mlir_op,
                                  mlir::MlirOp mlir_op_with_target_dims) {
  return ReshapeBatchDimsOrDie(mlir_op,
                               GetStaticShape(mlir_op_with_target_dims));
}

int get_batch_size(absl::Span<const int64_t> shape) {
  ABSL_CHECK(shape.size() >= 3)  // CRASH_OK
      << "Shape must have at least 3 dimensions";
  int batch_size = 1;
  for (int i = 0; i < shape.size() - 3; i++) {
    batch_size *= shape[i];
  }
  return batch_size;
}

mlir::MlirOp GetScaleDefaulted(mlir::MlirBuilder& builder,
                               std::optional<double> maybe_scale,
                               int64_t head_dim, mlir::Type element_type) {
  double scale = maybe_scale.value_or(1.0 / std::sqrt(head_dim));

  if (element_type == nullptr) {
    element_type = builder.getOpBuilder().getF32Type();
  }
  return MakeScalarConstant(builder, scale, element_type);
}

Strides EmptyLikeStrides(const at::TensorBase& tensor) {
  if (tensor.is_non_overlapping_and_dense()) {
    return CopyIntVector(tensor.strides());
  }
  // `at::infer_dense_strides` returns a `std::vector`.
  const std::vector<int64_t> strides =  // INT_VEC_OK
      at::infer_dense_strides(tensor.sizes(), tensor.strides());
  return CopyIntVector(strides);
}

Strides MatchingLayoutStrides(const at::TensorBase& query,
                              absl::Span<const int64_t> result_sizes) {
  const at::IntArrayRef query_strides = query.strides();
  ABSL_DCHECK_EQ(query_strides.size(), result_sizes.size());
  if (absl::c_equal(query.sizes(), result_sizes)) {
    return EmptyLikeStrides(query);
  }

  // Fill the dims from innermost to outermost in ascending order of query
  // stride. Zero-stride (expanded) dims go last, i.e. outermost, and the stable
  // sort makes the lower dim innermost on ties.
  const auto fill_key = [&query_strides](int dim) {
    return query_strides[dim] == 0 ? std::numeric_limits<int64_t>::max()
                                   : query_strides[dim];
  };
  std::vector<int> fill_order(result_sizes.size());
  absl::c_iota(fill_order, 0);
  absl::c_stable_sort(fill_order, [&fill_key](int a, int b) {
    return fill_key(a) < fill_key(b);
  });

  Strides result_strides(result_sizes.size());
  int64_t stride = 1;
  for (const int dim : fill_order) {
    result_strides[dim] = stride;
    stride *= result_sizes[dim];
  }
  return result_strides;
}

mlir::stablehlo::BroadcastInDimOp FindUnderlyingBroadcastInDim(
    mlir::Value cur) {
  while (auto reshape = mlir::dyn_cast_or_null<mlir::stablehlo::ReshapeOp>(
             cur.getDefiningOp())) {
    cur = reshape.getOperand();
    if (mlir::isa_and_nonnull<mlir::stablehlo::BroadcastInDimOp>(
            cur.getDefiningOp())) {
      break;
    }
  }
  return mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
      cur.getDefiningOp());
}

int64_t ComputePrefixProduct(llvm::ArrayRef<int64_t> shape, int64_t count) {
  int64_t prod = 1;
  for (int64_t d = 0; d < count; ++d) {
    prod *= shape[d];
  }
  return prod;
}

bool CheckBroadcastDimensionMapping(llvm::ArrayRef<int64_t> bcast_dims,
                                    mlir::RankedTensorType bcast_in_type,
                                    llvm::ArrayRef<int64_t> out_shape,
                                    int64_t out_rank) {
  if (!llvm::is_sorted(bcast_dims)) {
    return false;
  }
  llvm::SmallVector<int64_t, 6> in_size_for_out_dim(out_rank, 1);
  for (size_t i = 0; i < bcast_dims.size(); ++i) {
    if (bcast_dims[i] < 0 || bcast_dims[i] >= out_rank) {
      return false;
    }
    in_size_for_out_dim[bcast_dims[i]] = bcast_in_type.getDimSize(i);
  }
  for (int64_t d = 0; d < out_rank; ++d) {
    const int64_t expected = (d == out_rank - 3) ? 1 : out_shape[d];
    if (in_size_for_out_dim[d] != expected) {
      return false;
    }
  }
  return true;
}

bool IsValidRepeatedKvBroadcast(mlir::stablehlo::BroadcastInDimOp bcast,
                                int64_t num_head_q, int64_t batch_size,
                                int64_t seq_len, int64_t head_dim,
                                int64_t& num_head_kv) {
  auto bcast_out_type = mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
  auto bcast_in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
  if (!bcast_out_type || !bcast_in_type || !bcast_out_type.hasStaticShape() ||
      !bcast_in_type.hasStaticShape() || bcast_out_type.getRank() < 4) {
    return false;
  }
  const int64_t out_rank = bcast_out_type.getRank();
  llvm::ArrayRef<int64_t> out_shape = bcast_out_type.getShape();
  num_head_kv = out_shape[out_rank - 4];
  const int64_t group_size = out_shape[out_rank - 3];
  if (num_head_kv <= 0 || group_size <= 1 ||
      num_head_kv * group_size != num_head_q ||
      out_shape[out_rank - 2] != seq_len ||
      out_shape[out_rank - 1] != head_dim) {
    return false;
  }
  if (ComputePrefixProduct(out_shape, out_rank - 4) != batch_size) {
    return false;
  }
  return CheckBroadcastDimensionMapping(bcast.getBroadcastDimensions(),
                                        bcast_in_type, out_shape, out_rank);
}

std::optional<mlir::MlirOp> UnwrapBroadcastBaseValue(
    mlir::MlirOp kv_op, mlir::Value base_val, const Dimensions& target_shape) {
  while (auto reshape = mlir::dyn_cast_or_null<mlir::stablehlo::ReshapeOp>(
             base_val.getDefiningOp())) {
    auto rtype =
        mlir::dyn_cast<mlir::RankedTensorType>(reshape.getOperand().getType());
    if (!rtype || !rtype.hasStaticShape()) {
      break;
    }
    if (absl::c_equal(rtype.getShape(), target_shape)) {
      return mlir::MlirOp(kv_op.getBuilder(), reshape.getOperand());
    }
    base_val = reshape.getOperand();
  }
  auto base_type = mlir::dyn_cast<mlir::RankedTensorType>(base_val.getType());
  if (!base_type || !base_type.hasStaticShape()) {
    return std::nullopt;
  }
  if (absl::c_equal(base_type.getShape(), target_shape)) {
    return mlir::MlirOp(kv_op.getBuilder(), base_val);
  }
  return mlir::stablehlo::Reshape(mlir::MlirOp(kv_op.getBuilder(), base_val),
                                  target_shape);
}

std::optional<mlir::MlirOp> UnwrapRepeatedKvHead(mlir::MlirOp kv_op,
                                                 int64_t num_head_q) {
  auto kv_type = mlir::dyn_cast<mlir::RankedTensorType>(kv_op.getType());
  if (!kv_type || !kv_type.hasStaticShape() || kv_type.getRank() < 4) {
    return std::nullopt;
  }
  const int64_t rank = kv_type.getRank();
  llvm::ArrayRef<int64_t> kv_shape = kv_type.getShape();
  if (kv_shape[rank - 3] != num_head_q) {
    return std::nullopt;
  }
  const int64_t batch_size = get_batch_size(kv_shape);
  const int64_t seq_len = kv_shape[rank - 2];
  const int64_t head_dim = kv_shape[rank - 1];

  auto bcast = FindUnderlyingBroadcastInDim(kv_op.getValue());
  if (!bcast) {
    return std::nullopt;
  }
  int64_t num_head_kv = 0;
  if (!IsValidRepeatedKvBroadcast(bcast, num_head_q, batch_size, seq_len,
                                  head_dim, num_head_kv)) {
    return std::nullopt;
  }

  Dimensions target_shape = CopyIntVector(kv_shape);
  target_shape[rank - 3] = num_head_kv;
  return UnwrapBroadcastBaseValue(kv_op, bcast.getOperand(), target_shape);
}

bool IsAllZeroAttnBias(mlir::MlirOp bias_op) {
  auto bias_type = mlir::dyn_cast<mlir::RankedTensorType>(bias_op.getType());
  if (!bias_type || !bias_type.hasStaticShape() ||
      !mlir::isa<mlir::FloatType>(bias_type.getElementType())) {
    return false;
  }
  return IsZeroSplat(bias_op.getValue());
}

std::optional<bool> CheckCausalBiasSelect(mlir::stablehlo::SelectOp select,
                                          int depth) {
  std::optional<bool> is_lower_tri =
      MatchCausalPredicate(select.getPred(), depth + 1);
  if (!is_lower_tri.has_value()) {
    return false;
  }
  return *is_lower_tri ? (IsZeroSplat(select.getOnTrue()) &&
                          IsLargeNegativeMaskSplat(select.getOnFalse()))
                       : (IsLargeNegativeMaskSplat(select.getOnTrue()) &&
                          IsZeroSplat(select.getOnFalse()));
}

bool IsValidCausalBiasConvert(mlir::stablehlo::ConvertOp conv) {
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(conv.getType());
  if (!in_type || !out_type ||
      !mlir::isa<mlir::FloatType>(in_type.getElementType()) ||
      !mlir::isa<mlir::FloatType>(out_type.getElementType())) {
    return false;
  }
  auto in_float = mlir::cast<mlir::FloatType>(in_type.getElementType());
  auto out_float = mlir::cast<mlir::FloatType>(out_type.getElementType());
  return in_float.getIntOrFloatBitWidth() >=
             out_float.getIntOrFloatBitWidth() &&
         llvm::APFloat::semanticsMaxExponent(in_float.getFloatSemantics()) >=
             llvm::APFloat::semanticsMaxExponent(out_float.getFloatSemantics());
}

bool IsValidCausalBiasBroadcast(mlir::stablehlo::BroadcastOp bcast,
                                int64_t q_seq_len, int64_t kv_seq_len) {
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
  return in_type && in_type.getRank() >= 2 &&
         in_type.getDimSize(in_type.getRank() - 2) == q_seq_len &&
         in_type.getDimSize(in_type.getRank() - 1) == kv_seq_len;
}

bool IsValidCausalBiasBroadcastInDim(mlir::stablehlo::BroadcastInDimOp bcast,
                                     int64_t q_seq_len, int64_t kv_seq_len) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
  if (!out_type || !in_type || out_type.getRank() < 2 ||
      in_type.getRank() < 2) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = bcast.getBroadcastDimensions();
  const int64_t in_rank = in_type.getRank();
  const int64_t out_rank = out_type.getRank();
  return dims[in_rank - 2] == out_rank - 2 &&
         dims[in_rank - 1] == out_rank - 1 &&
         out_type.getDimSize(out_rank - 2) == q_seq_len &&
         out_type.getDimSize(out_rank - 1) == kv_seq_len &&
         in_type.getDimSize(in_rank - 2) == q_seq_len &&
         in_type.getDimSize(in_rank - 1) == kv_seq_len;
}

bool IsValidCausalBiasReshape(mlir::stablehlo::ReshapeOp reshape,
                              int64_t q_seq_len, int64_t kv_seq_len) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(reshape.getType());
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(reshape.getOperand().getType());
  return out_type && in_type && out_type.getRank() >= 2 &&
         in_type.getRank() >= 2 &&
         out_type.getDimSize(out_type.getRank() - 2) == q_seq_len &&
         out_type.getDimSize(out_type.getRank() - 1) == kv_seq_len &&
         in_type.getDimSize(in_type.getRank() - 2) == q_seq_len &&
         in_type.getDimSize(in_type.getRank() - 1) == kv_seq_len;
}

bool IsValidCausalBiasSlice(mlir::stablehlo::SliceOp slice, int64_t q_seq_len,
                            int64_t kv_seq_len) {
  for (int64_t s : slice.getStrides()) {
    if (s != 1) {
      return false;
    }
  }
  llvm::ArrayRef<int64_t> starts = slice.getStartIndices();
  const int64_t r = starts.size();
  if (r < 2 || starts[r - 2] != 0 || starts[r - 1] != 0) {
    return false;
  }
  auto slice_out = mlir::dyn_cast<mlir::RankedTensorType>(slice.getType());
  return slice_out && slice_out.getRank() >= 2 &&
         slice_out.getDimSize(r - 2) == q_seq_len &&
         slice_out.getDimSize(r - 1) == kv_seq_len;
}

enum class BiasStepResult { kStop, kMatchedTrue, kMatchedFalse, kAdvance };

std::optional<mlir::Value> TryAdvanceCausalBiasView(mlir::Operation* op,
                                                    int64_t q_seq_len,
                                                    int64_t kv_seq_len,
                                                    bool& has_seen_slice) {
  if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastOp>(op)) {
    if (IsValidCausalBiasBroadcast(bcast, q_seq_len, kv_seq_len)) {
      return bcast.getOperand();
    }
    return std::nullopt;
  }
  if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
    if (IsValidCausalBiasBroadcastInDim(bcast, q_seq_len, kv_seq_len)) {
      return bcast.getOperand();
    }
    return std::nullopt;
  }
  if (auto reshape = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
    if (!has_seen_slice &&
        IsValidCausalBiasReshape(reshape, q_seq_len, kv_seq_len)) {
      return reshape.getOperand();
    }
    return std::nullopt;
  }
  if (auto slice = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op)) {
    if (IsValidCausalBiasSlice(slice, q_seq_len, kv_seq_len)) {
      has_seen_slice = true;
      return slice.getOperand();
    }
    return std::nullopt;
  }
  return std::nullopt;
}

BiasStepResult StepCausalBiasAnalysis(mlir::Operation* op, int64_t q_seq_len,
                                      int64_t kv_seq_len, int depth,
                                      bool& has_seen_slice,
                                      mlir::Value& next_val) {
  if (auto select = mlir::dyn_cast<mlir::stablehlo::SelectOp>(op)) {
    return CheckCausalBiasSelect(select, depth).value_or(false)
               ? BiasStepResult::kMatchedTrue
               : BiasStepResult::kMatchedFalse;
  }
  if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
    if (IsValidCausalBiasConvert(conv)) {
      next_val = conv.getOperand();
      return BiasStepResult::kAdvance;
    }
    return BiasStepResult::kStop;
  }
  auto view_operand =
      TryAdvanceCausalBiasView(op, q_seq_len, kv_seq_len, has_seen_slice);
  if (view_operand.has_value()) {
    next_val = *view_operand;
    return BiasStepResult::kAdvance;
  }
  return BiasStepResult::kStop;
}

bool IsValidCausalBiasType(mlir::MlirOp bias_op, int64_t q_seq_len,
                           int64_t kv_seq_len) {
  if (q_seq_len != kv_seq_len || q_seq_len <= 1) {
    return false;
  }
  auto bias_type = mlir::dyn_cast<mlir::RankedTensorType>(bias_op.getType());
  if (!bias_type || !bias_type.hasStaticShape() || bias_type.getRank() < 2 ||
      !mlir::isa<mlir::FloatType>(bias_type.getElementType())) {
    return false;
  }
  const int64_t rank = bias_type.getRank();
  return bias_type.getDimSize(rank - 2) == q_seq_len &&
         bias_type.getDimSize(rank - 1) == kv_seq_len;
}

bool IsPureCausalAttnBias(mlir::MlirOp bias_op, int64_t q_seq_len,
                          int64_t kv_seq_len) {
  if (!IsValidCausalBiasType(bias_op, q_seq_len, kv_seq_len)) {
    return false;
  }
  bool has_seen_slice = false;
  mlir::Value cur = bias_op.getValue();
  for (int depth = 0; depth <= kMaxMaskAnalysisDepth && cur.getDefiningOp();
       ++depth) {
    mlir::Value next_val;
    const BiasStepResult res =
        StepCausalBiasAnalysis(cur.getDefiningOp(), q_seq_len, kv_seq_len,
                               depth, has_seen_slice, next_val);
    if (res == BiasStepResult::kMatchedTrue) {
      return true;
    }
    if (res == BiasStepResult::kAdvance) {
      cur = next_val;
      continue;
    }
    return false;
  }
  return false;
}

}  // namespace torch_tpu
