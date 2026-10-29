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

#include "csrc/mlir/passes/canonicalize_rect_sdpa_and_shift_invariant_softmax.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/absl_check.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
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
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// GENERIC: capability bounded recursion depth for SSA definition-chain walks.
constexpr int kMaxWalkDepth = 20;

bool IsDropLastDimBroadcast(llvm::ArrayRef<int64_t> dims, int64_t rank) {
  if (static_cast<int64_t>(dims.size()) != rank - 1) {
    return false;
  }
  for (int64_t i = 0; i < rank - 1; ++i) {
    if (dims[i] != i) {
      return false;
    }
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

mlir::DenseI64ArrayAttr GetDropLastDimBroadcastAttr(mlir::OpBuilder& builder,
                                                    int64_t rank) {
  ABSL_DCHECK_GE(rank, 1);
  llvm::SmallVector<int64_t, 4> dims;
  dims.reserve(rank - 1);
  for (int64_t i = 0; i < rank - 1; ++i) {
    dims.push_back(i);
  }
  return builder.getDenseI64ArrayAttr(dims);
}

llvm::SmallVector<int64_t, 4> GetNonUnitDimSizes(llvm::ArrayRef<int64_t> dims) {
  llvm::SmallVector<int64_t, 4> result;
  for (int64_t d : dims) {
    // GENERIC: semantic unit dimensions (size == 1) carry a single coordinate
    // and can be inserted/squeezed by ReshapeOp without moving data.
    if (d > 1) {
      result.push_back(d);
    }
  }
  return result;
}

// Maps a dimension ordinal across a reshape between shapes with matching
// non-unit dimensions. Returns the matching dimension index in dst_shape, or
// -1 if src_dim is a unit dimension (size == 1), or -2 if unmappable.
int64_t MapDimAcrossReshape(llvm::ArrayRef<int64_t> src_shape,
                            llvm::ArrayRef<int64_t> dst_shape,
                            int64_t src_dim) {
  if (src_dim < 0 || src_dim >= static_cast<int64_t>(src_shape.size())) {
    return -2;
  }
  // GENERIC: semantic unit-sized axis carries only index 0.
  if (src_shape[src_dim] == 1) {
    return -1;
  }
  if (GetNonUnitDimSizes(src_shape) != GetNonUnitDimSizes(dst_shape)) {
    return -2;
  }
  int64_t non_unit_ordinal = 0;
  for (int64_t i = 0; i < src_dim; ++i) {
    if (src_shape[i] > 1) {
      ++non_unit_ordinal;
    }
  }
  int64_t seen = 0;
  for (int64_t j = 0; j < static_cast<int64_t>(dst_shape.size()); ++j) {
    if (dst_shape[j] > 1) {
      if (seen == non_unit_ordinal) {
        return j;
      }
      ++seen;
    }
  }
  return -2;
}

mlir::Value PeelReshapeBroadcastChain(mlir::Value val,
                                      llvm::ArrayRef<int64_t> outer_shape) {
  const int64_t rank = outer_shape.size();
  // GENERIC: semantic broadcast of a reduced tensor requires rank >= 2.
  if (rank < 2) {
    return nullptr;
  }
  llvm::ArrayRef<int64_t> prefix_shape = outer_shape.drop_back(1);
  auto matches_prefix_or_unit_last =
      [rank, prefix_shape](mlir::RankedTensorType t) -> bool {
    if (!t || !t.hasStaticShape()) {
      return false;
    }
    if (t.getRank() == rank - 1) {
      return t.getShape() == prefix_shape;
    }
    // GENERIC: semantic unsqueezed trailing reduction dimension has size 1.
    if (t.getRank() == rank && t.getDimSize(rank - 1) == 1) {
      return t.getShape().drop_back(1) == prefix_shape;
    }
    return false;
  };

  mlir::Value cur = val;
  bool saw_bcast = false;
  while (true) {
    if (auto bcast = cur.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      auto in_type =
          mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
      auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
      if (!in_type || !out_type || !in_type.hasStaticShape() ||
          !out_type.hasStaticShape()) {
        break;
      }
      llvm::ArrayRef<int64_t> dims = bcast.getBroadcastDimensions();
      if (out_type.getRank() == rank && in_type.getRank() == rank - 1 &&
          IsDropLastDimBroadcast(dims, rank) &&
          in_type.getShape() == prefix_shape) {
        cur = bcast.getOperand();
        saw_bcast = true;
        continue;
      }
      if (out_type.getRank() == rank && in_type.getRank() == rank &&
          IsIdentityBroadcastDims(dims, rank) &&
          matches_prefix_or_unit_last(in_type)) {
        cur = bcast.getOperand();
        if (out_type.getDimSize(rank - 1) == outer_shape[rank - 1]) {
          saw_bcast = true;
        }
        continue;
      }
      break;
    }
    if (auto resh = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      auto in_type =
          mlir::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
      if (matches_prefix_or_unit_last(in_type)) {
        cur = resh.getOperand();
        continue;
      }
      break;
    }
    break;
  }
  if (!saw_bcast) {
    return nullptr;
  }
  auto cur_type = mlir::dyn_cast<mlir::RankedTensorType>(cur.getType());
  if (!matches_prefix_or_unit_last(cur_type)) {
    return nullptr;
  }
  if (auto resh = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    auto in_type =
        mlir::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
    if (in_type && in_type.hasStaticShape() && in_type.getRank() == rank - 1 &&
        in_type.getShape() == prefix_shape) {
      cur = resh.getOperand();
    }
  }
  return cur;
}

template <typename BinaryOpTy>
bool IsSingleDimReduceWithOp(mlir::Value val, int64_t expected_dim,
                             mlir::stablehlo::ReduceOp* out_reduce = nullptr) {
  auto reduce_op = val.getDefiningOp<mlir::stablehlo::ReduceOp>();
  // GENERIC: semantic single-input single-result reduction along one axis.
  if (!reduce_op || reduce_op.getInputs().size() != 1 ||
      reduce_op.getResults().size() != 1 ||
      reduce_op.getInitValues().size() != 1) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = reduce_op.getDimensions();
  if (dims.size() != 1 || dims[0] != expected_dim) {
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
  auto bin_op = ret_op.getOperands()[0].getDefiningOp<BinaryOpTy>();
  if (!bin_op) {
    return false;
  }
  mlir::Value arg0 = block.getArgument(0);
  mlir::Value arg1 = block.getArgument(1);
  mlir::Value lhs = bin_op.getOperand(0);
  mlir::Value rhs = bin_op.getOperand(1);
  if (!((lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0))) {
    return false;
  }
  if (out_reduce) {
    *out_reduce = reduce_op;
  }
  return true;
}

// Matches either a single-output `MaxOp` reduction or result 0 of a dual-output
// `(max_val, argmax_idx)` reduction along `expected_dim`.
bool IsSingleDimMaxOrArgmaxValueReduce(
    mlir::Value val, int64_t expected_dim,
    mlir::stablehlo::ReduceOp* out_reduce = nullptr) {
  if (IsSingleDimReduceWithOp<mlir::stablehlo::MaxOp>(val, expected_dim,
                                                      out_reduce)) {
    return true;
  }
  auto reduce_op = val.getDefiningOp<mlir::stablehlo::ReduceOp>();
  // GENERIC: semantic dual-output (values, indices) reduction from aten::max.
  if (!reduce_op || reduce_op.getInputs().size() != 2 ||
      reduce_op.getResults().size() != 2 ||
      reduce_op.getInitValues().size() != 2 || val != reduce_op.getResult(0)) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = reduce_op.getDimensions();
  if (dims.size() != 1 || dims[0] != expected_dim) {
    return false;
  }
  mlir::Region& body = reduce_op.getBody();
  if (!body.hasOneBlock()) {
    return false;
  }
  mlir::Block& block = body.front();
  // GENERIC: semantic 2-input combiner takes 4 scalar block arguments.
  if (block.getNumArguments() != 4) {
    return false;
  }
  auto ret_op =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  // GENERIC: semantic dual-output combiner returns 2 values.
  if (!ret_op || ret_op.getOperands().size() != 2) {
    return false;
  }
  auto max_op = ret_op.getOperands()[0].getDefiningOp<mlir::stablehlo::MaxOp>();
  if (!max_op) {
    return false;
  }
  mlir::Value arg0 = block.getArgument(0);
  mlir::Value arg2 = block.getArgument(2);
  mlir::Value lhs = max_op.getLhs();
  mlir::Value rhs = max_op.getRhs();
  if (!((lhs == arg0 && rhs == arg2) || (lhs == arg2 && rhs == arg0))) {
    return false;
  }
  if (out_reduce) {
    *out_reduce = reduce_op;
  }
  return true;
}

mlir::Value UnwrapViewAndConvertChain(mlir::Value val) {
  for (int i = 0; i < kMaxWalkDepth; ++i) {
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      val = bcast.getOperand();
      continue;
    }
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastOp>()) {
      val = bcast.getOperand();
      continue;
    }
    if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      val = resh.getOperand();
      continue;
    }
    if (auto tr = val.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
      val = tr.getOperand();
      continue;
    }
    if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      val = conv.getOperand();
      continue;
    }
    break;
  }
  return val;
}

bool ExtractFloatSplatAttr(mlir::Value val, llvm::APFloat& out_apf) {
  val = UnwrapViewAndConvertChain(val);
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr || attr.empty() || !attr.isSplat() ||
      !mlir::isa<mlir::FloatType>(attr.getElementType())) {
    return false;
  }
  out_apf = attr.getSplatValue<llvm::APFloat>();
  return true;
}

bool IsZeroSplatConstant(mlir::Value val) {
  llvm::APFloat apf(0.0f);
  return ExtractFloatSplatAttr(val, apf) && apf.isZero();
}

bool IsFiniteFloatSplatConstant(mlir::Value val) {
  llvm::APFloat apf(0.0f);
  return ExtractFloatSplatAttr(val, apf) && apf.isFinite();
}

bool IsNegInfSplatConstant(mlir::Value val) {
  llvm::APFloat apf(0.0f);
  return ExtractFloatSplatAttr(val, apf) && apf.isInfinity() &&
         apf.isNegative();
}

bool IsBoolSplatConstant(mlir::Value val, bool expected) {
  val = UnwrapViewAndConvertChain(val);
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr || attr.empty() || !attr.isSplat() ||
      !attr.getElementType().isInteger(1)) {
    return false;
  }
  return attr.getSplatValue<bool>() == expected;
}

mlir::Value CreateSplatFloatConst(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::RankedTensorType type, float value) {
  auto float_type = mlir::cast<mlir::FloatType>(type.getElementType());
  llvm::APFloat apf(value);
  bool loses_info = false;
  apf.convert(float_type.getFloatSemantics(),
              llvm::APFloat::rmNearestTiesToEven, &loses_info);
  auto attr = mlir::DenseElementsAttr::get(type, apf);
  return builder.create<mlir::stablehlo::ConstantOp>(loc, type, attr);
}

mlir::Value CreateSplatBoolConst(mlir::OpBuilder& builder, mlir::Location loc,
                                 mlir::RankedTensorType type, bool value) {
  auto attr = mlir::DenseElementsAttr::get(type, value);
  return builder.create<mlir::stablehlo::ConstantOp>(loc, type, attr);
}

bool EvalIntComparison(mlir::stablehlo::ComparisonDirection dir, int64_t lhs,
                       int64_t rhs) {
  switch (dir) {
    case mlir::stablehlo::ComparisonDirection::EQ:
      return lhs == rhs;
    case mlir::stablehlo::ComparisonDirection::NE:
      return lhs != rhs;
    case mlir::stablehlo::ComparisonDirection::GE:
      return lhs >= rhs;
    case mlir::stablehlo::ComparisonDirection::GT:
      return lhs > rhs;
    case mlir::stablehlo::ComparisonDirection::LE:
      return lhs <= rhs;
    case mlir::stablehlo::ComparisonDirection::LT:
      return lhs < rhs;
  }
  return false;
}

bool MatchAffineIota1D(mlir::Value val, int64_t& out_dim, int64_t& out_start,
                       int64_t& out_step, int depth = 0);

std::optional<int64_t> ExtractSplatIntValue(mlir::Value val, int depth = 0) {
  if (depth > kMaxWalkDepth) {
    return std::nullopt;
  }
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!type || !type.hasStaticShape() ||
      !mlir::isa<mlir::IntegerType>(type.getElementType())) {
    return std::nullopt;
  }
  if (auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
    auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    if (!attr || attr.empty() || !attr.isSplat()) {
      return std::nullopt;
    }
    if (attr.getElementType().isInteger(1)) {
      return attr.getSplatValue<bool>() ? 1 : 0;
    }
    if (mlir::isa<mlir::IntegerType>(attr.getElementType())) {
      return attr.getSplatValue<llvm::APInt>().getSExtValue();
    }
    return std::nullopt;
  }
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    return ExtractSplatIntValue(bcast.getOperand(), depth + 1);
  }
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastOp>()) {
    return ExtractSplatIntValue(bcast.getOperand(), depth + 1);
  }
  if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    return ExtractSplatIntValue(resh.getOperand(), depth + 1);
  }
  if (auto tr = val.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    return ExtractSplatIntValue(tr.getOperand(), depth + 1);
  }
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    return ExtractSplatIntValue(conv.getOperand(), depth + 1);
  }
  if (auto slice = val.getDefiningOp<mlir::stablehlo::SliceOp>()) {
    if (auto sub_val = ExtractSplatIntValue(slice.getOperand(), depth + 1)) {
      return sub_val;
    }
    int64_t op_dim = -1, op_start = 0, op_step = 0;
    if (MatchAffineIota1D(slice.getOperand(), op_dim, op_start, op_step,
                          depth + 1)) {
      llvm::ArrayRef<int64_t> starts = slice.getStartIndices();
      llvm::ArrayRef<int64_t> limits = slice.getLimitIndices();
      // GENERIC: semantic unit-length slice along the only varying axis fixes
      // the coordinate to starts[op_dim], yielding a constant splat.
      if (op_dim >= 0 && op_dim < static_cast<int64_t>(starts.size()) &&
          limits[op_dim] - starts[op_dim] == 1) {
        return op_start + op_step * starts[op_dim];
      }
    }
    return std::nullopt;
  }
  if (auto add = val.getDefiningOp<mlir::stablehlo::AddOp>()) {
    auto lhs = ExtractSplatIntValue(add.getLhs(), depth + 1);
    auto rhs = ExtractSplatIntValue(add.getRhs(), depth + 1);
    if (lhs.has_value() && rhs.has_value()) {
      return *lhs + *rhs;
    }
    return std::nullopt;
  }
  if (auto sub = val.getDefiningOp<mlir::stablehlo::SubtractOp>()) {
    auto lhs = ExtractSplatIntValue(sub.getLhs(), depth + 1);
    auto rhs = ExtractSplatIntValue(sub.getRhs(), depth + 1);
    if (lhs.has_value() && rhs.has_value()) {
      return *lhs - *rhs;
    }
    int64_t d1 = -1, s1 = 0, st1 = 0;
    int64_t d2 = -1, s2 = 0, st2 = 0;
    if (MatchAffineIota1D(sub.getLhs(), d1, s1, st1, depth + 1) &&
        MatchAffineIota1D(sub.getRhs(), d2, s2, st2, depth + 1) && d1 == d2 &&
        st1 == st2) {
      return s1 - s2;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

bool EvaluateAffineCompareUniformly(mlir::stablehlo::CompareOp cmp_op,
                                    bool& out_uniform_value, int depth) {
  if (depth > kMaxWalkDepth) {
    return false;
  }
  auto lhs_t =
      mlir::dyn_cast<mlir::RankedTensorType>(cmp_op.getLhs().getType());
  if (!lhs_t || !lhs_t.hasStaticShape()) {
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
  const int64_t len = lhs_t.getDimSize(d);
  const int64_t v_first = s;
  const int64_t v_last = s + st * (len - 1);
  const auto dir = cmp_op.getComparisonDirection();
  if (dir == mlir::stablehlo::ComparisonDirection::EQ ||
      dir == mlir::stablehlo::ComparisonDirection::NE) {
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

bool MatchAffineIota1D(mlir::Value val, int64_t& out_dim, int64_t& out_start,
                       int64_t& out_step, int depth) {
  if (depth > kMaxWalkDepth) {
    return false;
  }
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
      !mlir::isa<mlir::IntegerType>(type.getElementType())) {
    return false;
  }
  if (auto iota = val.getDefiningOp<mlir::stablehlo::IotaOp>()) {
    out_dim = static_cast<int64_t>(iota.getIotaDimension());
    out_start = 0;
    out_step = 1;
    return out_dim >= 0 && out_dim < type.getRank();
  }
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    return MatchAffineIota1D(conv.getOperand(), out_dim, out_start, out_step,
                             depth + 1);
  }
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    auto in_type =
        mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
    if (!in_type || !in_type.hasStaticShape()) {
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
    int64_t mapped_dim = bdims[op_dim];
    if (mapped_dim < 0 || mapped_dim >= type.getRank() ||
        in_type.getDimSize(op_dim) != type.getDimSize(mapped_dim)) {
      return false;
    }
    out_dim = mapped_dim;
    out_start = op_start;
    out_step = op_step;
    return true;
  }
  if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    auto in_type =
        mlir::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
    if (!in_type || !in_type.hasStaticShape()) {
      return false;
    }
    int64_t op_dim = -1, op_start = 0, op_step = 0;
    if (!MatchAffineIota1D(resh.getOperand(), op_dim, op_start, op_step,
                           depth + 1)) {
      return false;
    }
    int64_t mapped_dim =
        MapDimAcrossReshape(in_type.getShape(), type.getShape(), op_dim);
    if (mapped_dim < 0) {
      return false;
    }
    out_dim = mapped_dim;
    out_start = op_start;
    out_step = op_step;
    return true;
  }
  if (auto slice = val.getDefiningOp<mlir::stablehlo::SliceOp>()) {
    int64_t op_dim = -1, op_start = 0, op_step = 0;
    if (!MatchAffineIota1D(slice.getOperand(), op_dim, op_start, op_step,
                           depth + 1)) {
      return false;
    }
    llvm::ArrayRef<int64_t> starts = slice.getStartIndices();
    llvm::ArrayRef<int64_t> strides = slice.getStrides();
    if (op_dim < 0 || op_dim >= static_cast<int64_t>(starts.size())) {
      return false;
    }
    out_dim = op_dim;
    out_start = op_start + op_step * starts[op_dim];
    out_step = op_step * strides[op_dim];
    return true;
  }
  if (auto add = val.getDefiningOp<mlir::stablehlo::AddOp>()) {
    int64_t d = -1, s = 0, st = 0;
    if (MatchAffineIota1D(add.getLhs(), d, s, st, depth + 1)) {
      if (auto c = ExtractSplatIntValue(add.getRhs(), depth + 1)) {
        out_dim = d;
        out_start = s + *c;
        out_step = st;
        return true;
      }
    }
    if (MatchAffineIota1D(add.getRhs(), d, s, st, depth + 1)) {
      if (auto c = ExtractSplatIntValue(add.getLhs(), depth + 1)) {
        out_dim = d;
        out_start = *c + s;
        out_step = st;
        return true;
      }
    }
    return false;
  }
  if (auto sub = val.getDefiningOp<mlir::stablehlo::SubtractOp>()) {
    int64_t d = -1, s = 0, st = 0;
    if (MatchAffineIota1D(sub.getLhs(), d, s, st, depth + 1)) {
      if (auto c = ExtractSplatIntValue(sub.getRhs(), depth + 1)) {
        out_dim = d;
        out_start = s - *c;
        out_step = st;
        return true;
      }
    }
    if (MatchAffineIota1D(sub.getRhs(), d, s, st, depth + 1)) {
      if (auto c = ExtractSplatIntValue(sub.getLhs(), depth + 1)) {
        out_dim = d;
        out_start = *c - s;
        out_step = -st;
        return true;
      }
    }
    return false;
  }
  if (auto sel = val.getDefiningOp<mlir::stablehlo::SelectOp>()) {
    if (IsBoolSplatConstant(sel.getPred(), false)) {
      return MatchAffineIota1D(sel.getOnFalse(), out_dim, out_start, out_step,
                               depth + 1);
    }
    if (IsBoolSplatConstant(sel.getPred(), true)) {
      return MatchAffineIota1D(sel.getOnTrue(), out_dim, out_start, out_step,
                               depth + 1);
    }
    if (auto cmp = sel.getPred().getDefiningOp<mlir::stablehlo::CompareOp>()) {
      bool uniform_val = false;
      if (EvaluateAffineCompareUniformly(cmp, uniform_val, depth + 1)) {
        mlir::Value chosen = uniform_val ? sel.getOnTrue() : sel.getOnFalse();
        return MatchAffineIota1D(chosen, out_dim, out_start, out_step,
                                 depth + 1);
      }
    }
    return false;
  }
  return false;
}

// Represents a peeled mask select operation with tracked query and key
// dimension indices and sequence length.
struct PeeledMaskSelect {
  mlir::Value keep_pred;
  int64_t q_dim = -1;
  int64_t k_dim = -1;
  int64_t seq_len = 0;
};

std::optional<PeeledMaskSelect> PeelMaskToSelectPredicate(
    mlir::Value mask_val) {
  auto full_type = mlir::dyn_cast<mlir::RankedTensorType>(mask_val.getType());
  // GENERIC: semantic 2D query-key matrix requires rank >= 2.
  if (!full_type || !full_type.hasStaticShape() || full_type.getRank() < 2) {
    return std::nullopt;
  }
  const int64_t full_rank = full_type.getRank();
  const int64_t q_len = full_type.getDimSize(full_rank - 2);
  const int64_t k_len = full_type.getDimSize(full_rank - 1);
  if (q_len <= 0 || q_len != k_len) {
    return std::nullopt;
  }

  mlir::Value cur = mask_val;
  llvm::SmallVector<int64_t> cur_to_full(full_rank);
  for (int64_t i = 0; i < full_rank; ++i) {
    cur_to_full[i] = i;
  }

  for (int depth = 0; depth < kMaxWalkDepth; ++depth) {
    if (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      cur = conv.getOperand();
      continue;
    }
    if (auto bcast = cur.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      auto in_t =
          mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
      auto out_t = mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
      if (!in_t || !out_t || !in_t.hasStaticShape() ||
          !out_t.hasStaticShape()) {
        break;
      }
      llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
      if (static_cast<int64_t>(bdims.size()) != in_t.getRank()) {
        break;
      }
      llvm::SmallVector<int64_t> next_to_full(in_t.getRank(), -1);
      bool valid = true;
      for (int64_t i = 0; i < in_t.getRank(); ++i) {
        int64_t o = bdims[i];
        if (o < 0 || o >= out_t.getRank()) {
          valid = false;
          break;
        }
        if (in_t.getDimSize(i) == out_t.getDimSize(o)) {
          next_to_full[i] = cur_to_full[o];
        } else if (in_t.getDimSize(i) == 1) {
          next_to_full[i] = -1;
        } else {
          valid = false;
          break;
        }
      }
      if (!valid) {
        break;
      }
      cur = bcast.getOperand();
      cur_to_full = std::move(next_to_full);
      continue;
    }
    if (auto resh = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      auto in_t =
          mlir::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
      auto out_t = mlir::dyn_cast<mlir::RankedTensorType>(resh.getType());
      if (!in_t || !out_t || !in_t.hasStaticShape() ||
          !out_t.hasStaticShape() ||
          GetNonUnitDimSizes(in_t.getShape()) !=
              GetNonUnitDimSizes(out_t.getShape())) {
        break;
      }
      llvm::SmallVector<int64_t> non_unit_to_full;
      for (int64_t o = 0; o < out_t.getRank(); ++o) {
        if (out_t.getDimSize(o) > 1) {
          non_unit_to_full.push_back(cur_to_full[o]);
        }
      }
      llvm::SmallVector<int64_t> next_to_full(in_t.getRank(), -1);
      size_t idx = 0;
      for (int64_t i = 0; i < in_t.getRank(); ++i) {
        if (in_t.getDimSize(i) > 1) {
          next_to_full[i] = non_unit_to_full[idx++];
        }
      }
      cur = resh.getOperand();
      cur_to_full = std::move(next_to_full);
      continue;
    }
    break;
  }

  auto sel_op = cur.getDefiningOp<mlir::stablehlo::SelectOp>();
  if (!sel_op) {
    return std::nullopt;
  }
  auto pred_t =
      mlir::dyn_cast<mlir::RankedTensorType>(sel_op.getPred().getType());
  auto sel_t = mlir::dyn_cast<mlir::RankedTensorType>(sel_op.getType());
  if (!pred_t || !sel_t || !pred_t.hasStaticShape() ||
      pred_t.getShape() != sel_t.getShape()) {
    return std::nullopt;
  }
  if (!IsFiniteFloatSplatConstant(sel_op.getOnTrue()) ||
      !IsNegInfSplatConstant(sel_op.getOnFalse())) {
    return std::nullopt;
  }
  int64_t root_q_dim = -1;
  int64_t root_k_dim = -1;
  for (int64_t i = 0; i < pred_t.getRank(); ++i) {
    if (cur_to_full[i] == full_rank - 2) {
      root_q_dim = i;
    } else if (cur_to_full[i] == full_rank - 1) {
      root_k_dim = i;
    }
  }
  if (root_q_dim < 0 || root_k_dim < 0) {
    return std::nullopt;
  }
  return PeeledMaskSelect{sel_op.getPred(), root_q_dim, root_k_dim, q_len};
}

bool EvalDiagonalLinearForm(mlir::Value val, int64_t q_dim, int64_t k_dim,
                            int64_t& out_slope, int64_t& out_offset,
                            int depth = 0) {
  if (depth > kMaxWalkDepth) {
    return false;
  }
  if (auto c = ExtractSplatIntValue(val, depth + 1)) {
    out_slope = 0;
    out_offset = *c;
    return true;
  }
  int64_t iota_dim = -1, iota_start = 0, iota_step = 0;
  if (MatchAffineIota1D(val, iota_dim, iota_start, iota_step, depth + 1)) {
    if (iota_dim == q_dim || iota_dim == k_dim) {
      out_slope = iota_step;
      out_offset = iota_start;
      return true;
    }
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
    if (t && t.hasStaticShape() && iota_dim >= 0 && iota_dim < t.getRank() &&
        t.getDimSize(iota_dim) == 1) {
      out_slope = 0;
      out_offset = iota_start;
      return true;
    }
    return false;
  }
  if (auto add = val.getDefiningOp<mlir::stablehlo::AddOp>()) {
    int64_t s1 = 0, o1 = 0, s2 = 0, o2 = 0;
    if (EvalDiagonalLinearForm(add.getLhs(), q_dim, k_dim, s1, o1, depth + 1) &&
        EvalDiagonalLinearForm(add.getRhs(), q_dim, k_dim, s2, o2, depth + 1)) {
      out_slope = s1 + s2;
      out_offset = o1 + o2;
      return true;
    }
    return false;
  }
  if (auto sub = val.getDefiningOp<mlir::stablehlo::SubtractOp>()) {
    int64_t s1 = 0, o1 = 0, s2 = 0, o2 = 0;
    if (EvalDiagonalLinearForm(sub.getLhs(), q_dim, k_dim, s1, o1, depth + 1) &&
        EvalDiagonalLinearForm(sub.getRhs(), q_dim, k_dim, s2, o2, depth + 1)) {
      out_slope = s1 - s2;
      out_offset = o1 - o2;
      return true;
    }
    return false;
  }
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    return EvalDiagonalLinearForm(conv.getOperand(), q_dim, k_dim, out_slope,
                                  out_offset, depth + 1);
  }
  if (auto abs_op = val.getDefiningOp<mlir::stablehlo::AbsOp>()) {
    int64_t s = 0, o = 0;
    if (EvalDiagonalLinearForm(abs_op.getOperand(), q_dim, k_dim, s, o,
                               depth + 1) &&
        s == 0) {
      out_slope = 0;
      out_offset = std::abs(o);
      return true;
    }
    return false;
  }
  return false;
}

bool MapDimsThroughBroadcastAndReshape(mlir::Value& cur, int64_t& q_dim,
                                       int64_t& k_dim) {
  for (int depth = 0; depth < kMaxWalkDepth; ++depth) {
    if (auto bcast = cur.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      auto in_t =
          mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
      auto out_t = mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
      if (!in_t || !out_t || !in_t.hasStaticShape() ||
          !out_t.hasStaticShape()) {
        return false;
      }
      llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
      int64_t next_q = -1;
      int64_t next_k = -1;
      for (int64_t i = 0; i < in_t.getRank(); ++i) {
        if (q_dim >= 0 && bdims[i] == q_dim &&
            in_t.getDimSize(i) == out_t.getDimSize(q_dim)) {
          next_q = i;
        }
        if (k_dim >= 0 && bdims[i] == k_dim &&
            in_t.getDimSize(i) == out_t.getDimSize(k_dim)) {
          next_k = i;
        }
      }
      q_dim = next_q;
      k_dim = next_k;
      cur = bcast.getOperand();
      continue;
    }
    if (auto resh = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      auto in_t =
          mlir::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
      auto out_t = mlir::dyn_cast<mlir::RankedTensorType>(resh.getType());
      if (!in_t || !out_t || !in_t.hasStaticShape() ||
          !out_t.hasStaticShape()) {
        return false;
      }
      int64_t next_q = q_dim >= 0 ? MapDimAcrossReshape(out_t.getShape(),
                                                        in_t.getShape(), q_dim)
                                  : -1;
      int64_t next_k = k_dim >= 0 ? MapDimAcrossReshape(out_t.getShape(),
                                                        in_t.getShape(), k_dim)
                                  : -1;
      if (next_q == -2 || next_k == -2) {
        return false;
      }
      q_dim = next_q;
      k_dim = next_k;
      cur = resh.getOperand();
      continue;
    }
    break;
  }
  return true;
}

// Checks whether `CompareOp(EQ, lhs, rhs)` compares two gathers from the exact
// same source tensor whose gather index vectors are identical on the diagonal
// `q_idx == k_idx` (so `T[b, i] == T[b, i]` holds reflexively without modifying
// `T`).
bool IsReflexiveDiagonalGatherEquality(mlir::stablehlo::CompareOp cmp_op,
                                       int64_t q_dim, int64_t k_dim) {
  if (cmp_op.getComparisonDirection() !=
      mlir::stablehlo::ComparisonDirection::EQ) {
    return false;
  }
  mlir::Value lhs = cmp_op.getLhs();
  mlir::Value rhs = cmp_op.getRhs();
  int64_t q_l = q_dim, k_l = k_dim;
  int64_t q_r = q_dim, k_r = k_dim;
  if (!MapDimsThroughBroadcastAndReshape(lhs, q_l, k_l) ||
      !MapDimsThroughBroadcastAndReshape(rhs, q_r, k_r)) {
    return false;
  }
  if (lhs == rhs && q_l == q_r && k_l == k_r) {
    return true;
  }
  auto g_l = lhs.getDefiningOp<mlir::stablehlo::GatherOp>();
  auto g_r = rhs.getDefiningOp<mlir::stablehlo::GatherOp>();
  if (!g_l || !g_r || g_l.getOperand() != g_r.getOperand() ||
      g_l.getDimensionNumbers() != g_r.getDimensionNumbers() ||
      g_l.getSliceSizes() != g_r.getSliceSizes()) {
    return false;
  }
  auto dnums = g_l.getDimensionNumbers();
  if (!dnums.getOffsetDims().empty()) {
    return false;
  }
  auto idx_l_t =
      mlir::dyn_cast<mlir::RankedTensorType>(g_l.getStartIndices().getType());
  auto idx_r_t =
      mlir::dyn_cast<mlir::RankedTensorType>(g_r.getStartIndices().getType());
  if (!idx_l_t || !idx_r_t || !idx_l_t.hasStaticShape() ||
      !idx_r_t.hasStaticShape()) {
    return false;
  }
  const int64_t idx_vec_dim = dnums.getIndexVectorDim();
  if (idx_vec_dim != idx_l_t.getRank() - 1 ||
      idx_vec_dim != idx_r_t.getRank() - 1) {
    return false;
  }
  auto cat_l =
      g_l.getStartIndices().getDefiningOp<mlir::stablehlo::ConcatenateOp>();
  auto cat_r =
      g_r.getStartIndices().getDefiningOp<mlir::stablehlo::ConcatenateOp>();
  if (!cat_l || !cat_r ||
      static_cast<int64_t>(cat_l.getDimension()) != idx_vec_dim ||
      static_cast<int64_t>(cat_r.getDimension()) != idx_vec_dim ||
      cat_l.getInputs().size() != cat_r.getInputs().size()) {
    return false;
  }
  for (size_t i = 0; i < cat_l.getInputs().size(); ++i) {
    mlir::Value in_l = cat_l.getInputs()[i];
    mlir::Value in_r = cat_r.getInputs()[i];
    int64_t s_l = 0, o_l = 0, s_r = 0, o_r = 0;
    if (EvalDiagonalLinearForm(in_l, q_l, k_l, s_l, o_l) &&
        EvalDiagonalLinearForm(in_r, q_r, k_r, s_r, o_r)) {
      if (s_l != s_r || o_l != o_r) {
        return false;
      }
      continue;
    }
    int64_t d_l = -1, st_l = 0, step_l = 0;
    int64_t d_r = -1, st_r = 0, step_r = 0;
    if (MatchAffineIota1D(in_l, d_l, st_l, step_l) &&
        MatchAffineIota1D(in_r, d_r, st_r, step_r) && d_l == d_r &&
        d_l != q_l && d_l != k_l && d_r != q_r && d_r != k_r && st_l == st_r &&
        step_l == step_r) {
      continue;
    }
    return false;
  }
  return true;
}

bool IsDiagonalPredicateTrue(mlir::Value pred, int64_t q_dim, int64_t k_dim,
                             int64_t seq_len, int depth = 0) {
  if (depth > kMaxWalkDepth || seq_len <= 0) {
    return false;
  }
  if (!MapDimsThroughBroadcastAndReshape(pred, q_dim, k_dim)) {
    return false;
  }
  if (IsBoolSplatConstant(pred, true)) {
    return true;
  }
  if (auto and_op = pred.getDefiningOp<mlir::stablehlo::AndOp>()) {
    return IsDiagonalPredicateTrue(and_op.getLhs(), q_dim, k_dim, seq_len,
                                   depth + 1) &&
           IsDiagonalPredicateTrue(and_op.getRhs(), q_dim, k_dim, seq_len,
                                   depth + 1);
  }
  if (auto or_op = pred.getDefiningOp<mlir::stablehlo::OrOp>()) {
    return IsDiagonalPredicateTrue(or_op.getLhs(), q_dim, k_dim, seq_len,
                                   depth + 1) ||
           IsDiagonalPredicateTrue(or_op.getRhs(), q_dim, k_dim, seq_len,
                                   depth + 1);
  }
  if (auto cmp_op = pred.getDefiningOp<mlir::stablehlo::CompareOp>()) {
    if (IsReflexiveDiagonalGatherEquality(cmp_op, q_dim, k_dim)) {
      return true;
    }
    int64_t s_l = 0, o_l = 0, s_r = 0, o_r = 0;
    if (!EvalDiagonalLinearForm(cmp_op.getLhs(), q_dim, k_dim, s_l, o_l) ||
        !EvalDiagonalLinearForm(cmp_op.getRhs(), q_dim, k_dim, s_r, o_r)) {
      return false;
    }
    const int64_t m = s_l - s_r;
    const int64_t d = o_l - o_r;
    const int64_t end_val = m * (seq_len - 1) + d;
    if (cmp_op.getComparisonDirection() ==
        mlir::stablehlo::ComparisonDirection::NE) {
      return (d > 0 && end_val > 0) || (d < 0 && end_val < 0);
    }
    return EvalIntComparison(cmp_op.getComparisonDirection(), d, 0) &&
           EvalIntComparison(cmp_op.getComparisonDirection(), end_val, 0);
  }
  return false;
}

bool HasAtLeastOneUnmaskedKeyPerRow(mlir::Value pred, int64_t q_dim,
                                    int64_t k_dim, int64_t seq_len) {
  if (!MapDimsThroughBroadcastAndReshape(pred, q_dim, k_dim)) {
    return false;
  }
  if (auto rev_op = pred.getDefiningOp<mlir::stablehlo::ReverseOp>()) {
    // Only allow ReverseOp if all reversed dimensions are independent batch
    // dimensions (d != q_dim && d != k_dim); reversing q_dim or k_dim flips
    // the tested diagonal into an anti-diagonal.
    for (int64_t d : rev_op.getDimensions()) {
      if (d == q_dim || d == k_dim) {
        return false;
      }
    }
    pred = rev_op.getOperand();
  }
  return IsDiagonalPredicateTrue(pred, q_dim, k_dim, seq_len);
}

bool IsRowAllMaskedProvablyFalse(mlir::Value row_mask_pred_3d) {
  mlir::stablehlo::ReduceOp reduce_op;
  auto red_type =
      mlir::dyn_cast<mlir::RankedTensorType>(row_mask_pred_3d.getType());
  if (!red_type || !red_type.hasStaticShape()) {
    return false;
  }
  const int64_t in_rank = red_type.getRank() + 1;
  if (!IsSingleDimReduceWithOp<mlir::stablehlo::AndOp>(
          row_mask_pred_3d, in_rank - 1, &reduce_op) ||
      !IsBoolSplatConstant(reduce_op.getInitValues()[0], true)) {
    return false;
  }
  auto cmp_op =
      reduce_op.getInputs()[0].getDefiningOp<mlir::stablehlo::CompareOp>();
  if (!cmp_op || cmp_op.getComparisonDirection() !=
                     mlir::stablehlo::ComparisonDirection::EQ) {
    return false;
  }
  mlir::Value mask_val;
  if (IsNegInfSplatConstant(cmp_op.getRhs())) {
    mask_val = cmp_op.getLhs();
  } else if (IsNegInfSplatConstant(cmp_op.getLhs())) {
    mask_val = cmp_op.getRhs();
  } else {
    return false;
  }
  auto peeled = PeelMaskToSelectPredicate(mask_val);
  if (!peeled) {
    return false;
  }
  return HasAtLeastOneUnmaskedKeyPerRow(peeled->keep_pred, peeled->q_dim,
                                        peeled->k_dim, peeled->seq_len);
}

// 1. Folds standalone `row_all_masked = ReduceOp(CompareOp(mask, -inf, EQ),
// true, dims={rank-1}, AndOp)` reductions to `splat(false)` (and simplifies
// `SelectOp(Broadcast(row_all_masked), on_true, on_false)` to `on_false`) when
// every query row has a provably unmasked diagonal entry `q_idx == k_idx`.
void RewriteStandaloneRowAllMaskedReductions(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::ReduceOp> reduce_ops;
  func.walk([&reduce_ops](mlir::stablehlo::ReduceOp op) {
    reduce_ops.push_back(op);
  });

  bool folded_any = false;
  for (mlir::stablehlo::ReduceOp reduce_op : reduce_ops) {
    if (!reduce_op || reduce_op->getParentOp() == nullptr ||
        reduce_op->use_empty() || reduce_op.getResults().size() != 1) {
      continue;
    }
    mlir::Value res = reduce_op.getResult(0);
    auto red_type = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());
    if (!red_type || !red_type.hasStaticShape() ||
        !red_type.getElementType().isInteger(1)) {
      continue;
    }
    if (!IsRowAllMaskedProvablyFalse(res)) {
      continue;
    }
    mlir::OpBuilder builder(reduce_op);
    mlir::Value false_splat =
        CreateSplatBoolConst(builder, reduce_op.getLoc(), red_type, false);
    res.replaceAllUsesWith(false_splat);
    folded_any = true;
  }

  if (!folded_any) {
    return;
  }

  llvm::SmallVector<mlir::stablehlo::SelectOp> select_ops;
  func.walk([&select_ops](mlir::stablehlo::SelectOp op) {
    select_ops.push_back(op);
  });
  for (mlir::stablehlo::SelectOp sel_op : select_ops) {
    if (!sel_op || sel_op->getParentOp() == nullptr || sel_op->use_empty()) {
      continue;
    }
    if (IsBoolSplatConstant(sel_op.getPred(), false)) {
      sel_op.replaceAllUsesWith(sel_op.getOnFalse());
    } else if (IsBoolSplatConstant(sel_op.getPred(), true)) {
      sel_op.replaceAllUsesWith(sel_op.getOnTrue());
    }
  }
}

// Returns true when a tensor has static shape, rank >= 2, float element type,
// and reduction along the last dimension.
bool IsTargetSoftmaxReductionShape(mlir::RankedTensorType type) {
  // GENERIC: semantic reduction tensors have rank >= 2 and float elements.
  if (!type || !type.hasStaticShape() || type.getRank() < 2 ||
      !mlir::isa<mlir::FloatType>(type.getElementType())) {
    return false;
  }
  const int64_t rank = type.getRank();
  return type.getDimSize(rank - 1) > 0;
}

// Holds matched operations for an exponential of shifted logits: logits value,
// maximum reduction, and the pre-subtraction operation.
struct ShiftedLogitsMatch {
  mlir::Value masked_logits;
  mlir::stablehlo::ReduceOp max_reduce;
  mlir::stablehlo::SubtractOp shifted_sub;
};

std::optional<ShiftedLogitsMatch> MatchExpOfShiftedLogits(mlir::Value exp_val) {
  auto exp_op = exp_val.getDefiningOp<mlir::stablehlo::ExpOp>();
  if (!exp_op) {
    return std::nullopt;
  }
  mlir::Value logits = exp_op.getOperand();
  if (auto conv = logits.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    logits = conv.getOperand();
  }
  if (auto sub_op = logits.getDefiningOp<mlir::stablehlo::SubtractOp>()) {
    auto sub_type = mlir::dyn_cast<mlir::RankedTensorType>(sub_op.getType());
    // GENERIC: semantic reduction score tensors have rank >= 2.
    if (!sub_type || !sub_type.hasStaticShape() || sub_type.getRank() < 2) {
      return std::nullopt;
    }
    const int64_t rank = sub_type.getRank();
    mlir::Value max_peeled =
        PeelReshapeBroadcastChain(sub_op.getRhs(), sub_type.getShape());
    if (!max_peeled) {
      return std::nullopt;
    }
    mlir::stablehlo::ReduceOp max_reduce;
    if (!IsSingleDimReduceWithOp<mlir::stablehlo::MaxOp>(max_peeled, rank - 1,
                                                         &max_reduce) ||
        max_reduce.getInputs()[0] != sub_op.getLhs()) {
      return std::nullopt;
    }
    return ShiftedLogitsMatch{sub_op.getLhs(), max_reduce, sub_op};
  }
  if (auto add_op = logits.getDefiningOp<mlir::stablehlo::AddOp>()) {
    auto add_type = mlir::dyn_cast<mlir::RankedTensorType>(add_op.getType());
    // GENERIC: semantic batched attention score tensors have rank >= 3.
    if (!add_type || !add_type.hasStaticShape() || add_type.getRank() < 3) {
      return std::nullopt;
    }
    const int64_t rank = add_type.getRank();
    mlir::Value neg_peeled =
        PeelReshapeBroadcastChain(add_op.getRhs(), add_type.getShape());
    if (!neg_peeled) {
      return std::nullopt;
    }
    auto neg_op = neg_peeled.getDefiningOp<mlir::stablehlo::NegOp>();
    if (!neg_op) {
      return std::nullopt;
    }
    mlir::stablehlo::ReduceOp max_reduce;
    if (!IsSingleDimReduceWithOp<mlir::stablehlo::MaxOp>(
            neg_op.getOperand(), rank - 1, &max_reduce) ||
        max_reduce.getInputs()[0] != add_op.getLhs()) {
      return std::nullopt;
    }
    return ShiftedLogitsMatch{add_op.getLhs(), max_reduce, nullptr};
  }
  return std::nullopt;
}

void RewriteSubtractMaxInPlace(ShiftedLogitsMatch& match) {
  mlir::stablehlo::SubtractOp sub_op = match.shifted_sub;
  if (!sub_op || sub_op->getParentOp() == nullptr || sub_op->use_empty()) {
    return;
  }
  auto sub_type = mlir::cast<mlir::RankedTensorType>(sub_op.getType());
  const int64_t rank = sub_type.getRank();
  mlir::OpBuilder builder(sub_op);
  mlir::Location loc = sub_op.getLoc();
  mlir::Value neg_max = builder.create<mlir::stablehlo::NegOp>(
      loc, match.max_reduce.getResult(0));
  mlir::Value neg_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, sub_type, neg_max, GetDropLastDimBroadcastAttr(builder, rank));
  mlir::Value add_val = builder.create<mlir::stablehlo::AddOp>(
      loc, match.masked_logits, neg_bcast);
  sub_op.replaceAllUsesWith(add_val);
}

bool MatchesSumExpOfExpVal(
    mlir::Value val, mlir::Value exp_val, int64_t expected_dim,
    mlir::stablehlo::ReduceOp* out_sum_reduce = nullptr) {
  mlir::Value cur = val;
  while (true) {
    if (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      cur = conv.getOperand();
      continue;
    }
    if (auto resh = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      cur = resh.getOperand();
      continue;
    }
    if (auto sel = cur.getDefiningOp<mlir::stablehlo::SelectOp>()) {
      if (IsZeroSplatConstant(sel.getOnTrue())) {
        cur = sel.getOnFalse();
        continue;
      }
    }
    break;
  }
  mlir::stablehlo::ReduceOp sum_reduce;
  if (!IsSingleDimReduceWithOp<mlir::stablehlo::AddOp>(cur, expected_dim,
                                                       &sum_reduce) ||
      sum_reduce.getInputs()[0] != exp_val) {
    return false;
  }
  if (out_sum_reduce) {
    *out_sum_reduce = sum_reduce;
  }
  return true;
}

bool HasPreNormalizedReciprocalMulUser(mlir::stablehlo::ExpOp exp_op) {
  auto exp_type = mlir::dyn_cast<mlir::RankedTensorType>(exp_op.getType());
  // GENERIC: semantic batched attention score tensors have rank >= 3.
  if (!exp_type || !exp_type.hasStaticShape() || exp_type.getRank() < 3) {
    return false;
  }
  for (mlir::Operation* u : exp_op.getResult().getUsers()) {
    if (mlir::isa<mlir::stablehlo::DivOp>(u) && !u->use_empty()) {
      return false;
    }
  }
  const int64_t rank = exp_type.getRank();
  for (mlir::Operation* u : exp_op.getResult().getUsers()) {
    auto mul_op = mlir::dyn_cast<mlir::stablehlo::MulOp>(u);
    if (!mul_op) {
      continue;
    }
    mlir::Value other = mul_op.getLhs() == exp_op.getResult() ? mul_op.getRhs()
                                                              : mul_op.getLhs();
    mlir::Value inv_peeled =
        PeelReshapeBroadcastChain(other, exp_type.getShape());
    if (!inv_peeled) {
      continue;
    }
    if (auto sel = inv_peeled.getDefiningOp<mlir::stablehlo::SelectOp>()) {
      if (IsZeroSplatConstant(sel.getOnTrue())) {
        inv_peeled = sel.getOnFalse();
      }
    }
    auto inv_div = inv_peeled.getDefiningOp<mlir::stablehlo::DivOp>();
    if (!inv_div) {
      continue;
    }
    mlir::Value denom = inv_div.getRhs();
    if (auto safe_sel = denom.getDefiningOp<mlir::stablehlo::SelectOp>()) {
      auto cmp_op =
          safe_sel.getPred().getDefiningOp<mlir::stablehlo::CompareOp>();
      if (cmp_op && cmp_op.getComparisonDirection() ==
                        mlir::stablehlo::ComparisonDirection::EQ) {
        denom = safe_sel.getOnFalse();
      }
    }
    if (MatchesSumExpOfExpVal(denom, exp_op.getResult(), rank - 1)) {
      return true;
    }
  }
  return false;
}

bool IsPreNormalizedMulProbability(mlir::Value val) {
  mlir::Value cur = val;
  if (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    cur = conv.getOperand();
  }
  if (auto sel = cur.getDefiningOp<mlir::stablehlo::SelectOp>()) {
    if (IsZeroSplatConstant(sel.getOnTrue())) {
      cur = sel.getOnFalse();
    }
  }
  auto mul_op = cur.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul_op) {
    return false;
  }
  if (auto exp_lhs = mul_op.getLhs().getDefiningOp<mlir::stablehlo::ExpOp>()) {
    if (HasPreNormalizedReciprocalMulUser(exp_lhs)) {
      return true;
    }
  }
  if (auto exp_rhs = mul_op.getRhs().getDefiningOp<mlir::stablehlo::ExpOp>()) {
    if (HasPreNormalizedReciprocalMulUser(exp_rhs)) {
      return true;
    }
  }
  return false;
}

bool IsTargetBackwardRowsumSubtract(mlir::Value rhs_peeled, int64_t rank,
                                    mlir::stablehlo::SubtractOp sub_op) {
  mlir::stablehlo::ReduceOp red_op;
  if (IsSingleDimReduceWithOp<mlir::stablehlo::AddOp>(rhs_peeled, rank - 1,
                                                      &red_op)) {
    auto prod_mul =
        red_op.getInputs()[0].getDefiningOp<mlir::stablehlo::MulOp>();
    if (!prod_mul) {
      return false;
    }
    mlir::Value dP = sub_op.getLhs();
    mlir::Value probs;
    if (prod_mul.getLhs() == dP) {
      probs = prod_mul.getRhs();
    } else if (prod_mul.getRhs() == dP) {
      probs = prod_mul.getLhs();
    } else {
      return false;
    }
    return IsPreNormalizedMulProbability(probs);
  }
  if (auto dg = rhs_peeled.getDefiningOp<mlir::stablehlo::DotGeneralOp>()) {
    auto dnums = dg.getDotDimensionNumbers();
    llvm::ArrayRef<int64_t> lhs_batch = dnums.getLhsBatchingDimensions();
    llvm::ArrayRef<int64_t> rhs_batch = dnums.getRhsBatchingDimensions();
    llvm::ArrayRef<int64_t> lhs_cont = dnums.getLhsContractingDimensions();
    llvm::ArrayRef<int64_t> rhs_cont = dnums.getRhsContractingDimensions();
    if (static_cast<int64_t>(lhs_batch.size()) != rank - 1 ||
        lhs_batch != rhs_batch ||
        lhs_cont != llvm::ArrayRef<int64_t>{rank - 1} ||
        rhs_cont != llvm::ArrayRef<int64_t>{rank - 1}) {
      return false;
    }
    for (int64_t i = 0; i < rank - 1; ++i) {
      if (lhs_batch[i] != i) {
        return false;
      }
    }
    mlir::Value dP = sub_op.getLhs();
    mlir::Value probs;
    if (dg.getLhs() == dP) {
      probs = dg.getRhs();
    } else if (dg.getRhs() == dP) {
      probs = dg.getLhs();
    } else {
      return false;
    }
    return IsPreNormalizedMulProbability(probs);
  }
  return false;
}

mlir::stablehlo::ExpOp GetSingleExpUser(mlir::stablehlo::SubtractOp sub_op) {
  if (!sub_op->hasOneUse()) {
    return nullptr;
  }
  mlir::Operation* u = *sub_op->getUsers().begin();
  if (auto exp_op = mlir::dyn_cast<mlir::stablehlo::ExpOp>(u)) {
    return exp_op;
  }
  if (auto conv_op = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(u)) {
    if (conv_op->hasOneUse()) {
      return mlir::dyn_cast<mlir::stablehlo::ExpOp>(
          *conv_op->getUsers().begin());
    }
  }
  return nullptr;
}

// 2. Canonicalizes SDPA forward and backward reductions to
// `Add(logits, Broadcast(Neg(max)))`, `Mul(exp, Broadcast(Div(1, sum)))`, and
// backward `Add(dP, Broadcast(Neg(rowsum)))`.
void RewriteRectangularSdpaReductions(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::DivOp> div_ops;
  func.walk([&div_ops](mlir::stablehlo::DivOp op) { div_ops.push_back(op); });

  for (mlir::stablehlo::DivOp div_op : div_ops) {
    if (!div_op || div_op->getParentOp() == nullptr || div_op->use_empty()) {
      continue;
    }
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(div_op.getType());
    if (!IsTargetSoftmaxReductionShape(type)) {
      continue;
    }
    const int64_t rank = type.getRank();
    mlir::Value exp_val = div_op.getLhs();
    auto shifted_match = MatchExpOfShiftedLogits(exp_val);
    if (!shifted_match) {
      continue;
    }
    mlir::Value denom_peeled =
        PeelReshapeBroadcastChain(div_op.getRhs(), type.getShape());
    if (!denom_peeled) {
      continue;
    }
    auto denom_type =
        mlir::dyn_cast<mlir::RankedTensorType>(denom_peeled.getType());
    if (!denom_type || denom_type.getRank() != rank - 1) {
      continue;
    }
    bool matched_denom = false;
    if (auto safe_select =
            denom_peeled.getDefiningOp<mlir::stablehlo::SelectOp>()) {
      auto cmp_op =
          safe_select.getPred().getDefiningOp<mlir::stablehlo::CompareOp>();
      if (cmp_op &&
          cmp_op.getComparisonDirection() ==
              mlir::stablehlo::ComparisonDirection::EQ &&
          MatchesSumExpOfExpVal(safe_select.getOnFalse(), exp_val, rank - 1)) {
        matched_denom = true;
      }
    } else if (MatchesSumExpOfExpVal(denom_peeled, exp_val, rank - 1)) {
      matched_denom = true;
    }
    if (!matched_denom) {
      continue;
    }

    RewriteSubtractMaxInPlace(*shifted_match);

    mlir::OpBuilder builder(div_op);
    mlir::Location loc = div_op.getLoc();
    mlir::DenseI64ArrayAttr bcast_dims_attr =
        GetDropLastDimBroadcastAttr(builder, rank);
    mlir::Value one_3d = CreateSplatFloatConst(builder, loc, denom_type, 1.0f);
    mlir::Value inv_sum_3d =
        builder.create<mlir::stablehlo::DivOp>(loc, one_3d, denom_peeled);
    mlir::Value inv_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, type, inv_sum_3d, bcast_dims_attr);
    mlir::Value mul_val =
        builder.create<mlir::stablehlo::MulOp>(loc, exp_val, inv_bcast);
    div_op.replaceAllUsesWith(mul_val);
  }

  llvm::SmallVector<mlir::stablehlo::SubtractOp> sub_ops;
  func.walk(
      [&sub_ops](mlir::stablehlo::SubtractOp op) { sub_ops.push_back(op); });

  for (mlir::stablehlo::SubtractOp sub_op : sub_ops) {
    if (!sub_op || sub_op->getParentOp() == nullptr || sub_op->use_empty()) {
      continue;
    }
    auto sub_type = mlir::dyn_cast<mlir::RankedTensorType>(sub_op.getType());
    if (!IsTargetSoftmaxReductionShape(sub_type)) {
      continue;
    }
    const int64_t rank = sub_type.getRank();
    mlir::Value rhs_peeled =
        PeelReshapeBroadcastChain(sub_op.getRhs(), sub_type.getShape());
    if (!rhs_peeled) {
      continue;
    }
    auto rhs_peeled_type =
        mlir::dyn_cast<mlir::RankedTensorType>(rhs_peeled.getType());
    if (!rhs_peeled_type || rhs_peeled_type.getRank() != rank - 1) {
      continue;
    }

    if (IsTargetBackwardRowsumSubtract(rhs_peeled, rank, sub_op)) {
      mlir::OpBuilder builder(sub_op);
      mlir::Location loc = sub_op.getLoc();
      mlir::Value neg_rowsum =
          builder.create<mlir::stablehlo::NegOp>(loc, rhs_peeled);
      mlir::Value neg_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, sub_type, neg_rowsum,
          GetDropLastDimBroadcastAttr(builder, rank));
      mlir::Value add_val = builder.create<mlir::stablehlo::AddOp>(
          loc, sub_op.getLhs(), neg_bcast);
      sub_op.replaceAllUsesWith(add_val);
      continue;
    }

    mlir::stablehlo::ReduceOp max_reduce;
    if (!IsSingleDimReduceWithOp<mlir::stablehlo::MaxOp>(rhs_peeled, rank - 1,
                                                         &max_reduce) ||
        max_reduce.getInputs()[0] != sub_op.getLhs()) {
      continue;
    }
    mlir::stablehlo::ExpOp exp_op = GetSingleExpUser(sub_op);
    if (!exp_op || !HasPreNormalizedReciprocalMulUser(exp_op)) {
      continue;
    }
    mlir::OpBuilder builder(sub_op);
    mlir::Location loc = sub_op.getLoc();
    mlir::Value neg_max =
        builder.create<mlir::stablehlo::NegOp>(loc, max_reduce.getResult(0));
    mlir::Value neg_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, sub_type, neg_max, GetDropLastDimBroadcastAttr(builder, rank));
    mlir::Value add_val =
        builder.create<mlir::stablehlo::AddOp>(loc, sub_op.getLhs(), neg_bcast);
    sub_op.replaceAllUsesWith(add_val);
  }
}

bool IsNormalizedSoftmaxProbabilityAlongLastDim(mlir::Value val, int64_t rank) {
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    val = conv.getOperand();
  }
  if (IsPreNormalizedMulProbability(val)) {
    return true;
  }
  if (auto div_op = val.getDefiningOp<mlir::stablehlo::DivOp>()) {
    auto div_t = mlir::dyn_cast<mlir::RankedTensorType>(div_op.getType());
    if (!div_t || !div_t.hasStaticShape() || div_t.getRank() != rank) {
      return false;
    }
    mlir::Value denom_peeled =
        PeelReshapeBroadcastChain(div_op.getRhs(), div_t.getShape());
    return denom_peeled &&
           MatchesSumExpOfExpVal(denom_peeled, div_op.getLhs(), rank - 1);
  }
  return false;
}

bool IsRowsumOfProbsAndGradAlongLastDim(mlir::Value rowsum, mlir::Value probs,
                                        mlir::Value dP, int64_t rank) {
  if (auto dg = rowsum.getDefiningOp<mlir::stablehlo::DotGeneralOp>()) {
    if (!((dg.getLhs() == probs && dg.getRhs() == dP) ||
          (dg.getLhs() == dP && dg.getRhs() == probs))) {
      return false;
    }
    auto dnums = dg.getDotDimensionNumbers();
    llvm::ArrayRef<int64_t> lhs_batch = dnums.getLhsBatchingDimensions();
    llvm::ArrayRef<int64_t> rhs_batch = dnums.getRhsBatchingDimensions();
    llvm::ArrayRef<int64_t> lhs_cont = dnums.getLhsContractingDimensions();
    llvm::ArrayRef<int64_t> rhs_cont = dnums.getRhsContractingDimensions();
    if (static_cast<int64_t>(lhs_batch.size()) != rank - 1 ||
        lhs_batch != rhs_batch ||
        lhs_cont != llvm::ArrayRef<int64_t>{rank - 1} ||
        rhs_cont != llvm::ArrayRef<int64_t>{rank - 1}) {
      return false;
    }
    for (int64_t i = 0; i < rank - 1; ++i) {
      if (lhs_batch[i] != i) {
        return false;
      }
    }
    return true;
  }
  mlir::stablehlo::ReduceOp red_op;
  if (IsSingleDimReduceWithOp<mlir::stablehlo::AddOp>(rowsum, rank - 1,
                                                      &red_op)) {
    if (auto mul =
            red_op.getInputs()[0].getDefiningOp<mlir::stablehlo::MulOp>()) {
      return (mul.getLhs() == probs && mul.getRhs() == dP) ||
             (mul.getLhs() == dP && mul.getRhs() == probs);
    }
  }
  return false;
}

bool IsSoftmaxBackwardJacobianAlongLastDim(mlir::Value val, int64_t rank) {
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    val = conv.getOperand();
  }
  auto mul_op = val.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul_op) {
    return false;
  }
  auto mul_t = mlir::dyn_cast<mlir::RankedTensorType>(mul_op.getType());
  if (!mul_t || !mul_t.hasStaticShape() || mul_t.getRank() != rank) {
    return false;
  }
  auto check_sides = [rank, mul_t](mlir::Value probs,
                                   mlir::Value sub_val) -> bool {
    if (!IsNormalizedSoftmaxProbabilityAlongLastDim(probs, rank)) {
      return false;
    }
    if (auto sub_op = sub_val.getDefiningOp<mlir::stablehlo::SubtractOp>()) {
      mlir::Value rowsum_peeled =
          PeelReshapeBroadcastChain(sub_op.getRhs(), mul_t.getShape());
      return rowsum_peeled && IsRowsumOfProbsAndGradAlongLastDim(
                                  rowsum_peeled, probs, sub_op.getLhs(), rank);
    }
    return false;
  };
  return check_sides(mul_op.getLhs(), mul_op.getRhs()) ||
         check_sides(mul_op.getRhs(), mul_op.getLhs());
}

bool IsZeroSumSoftmaxBackwardScatter(mlir::Value scatter_val,
                                     mlir::Value d_softmax_out) {
  auto scatter_op = scatter_val.getDefiningOp<mlir::stablehlo::ScatterOp>();
  // GENERIC: semantic single-tensor scatter into zero-initialized buffer.
  if (!scatter_op || scatter_op.getInputs().size() != 1 ||
      scatter_op.getUpdates().size() != 1 ||
      !IsZeroSplatConstant(scatter_op.getInputs()[0])) {
    return false;
  }
  mlir::Region& update_comp = scatter_op.getUpdateComputation();
  if (!update_comp.hasOneBlock()) {
    return false;
  }
  mlir::Block& block = update_comp.front();
  // GENERIC: semantic binary reduction combiner takes 2 scalar arguments.
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
  if (!((lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0))) {
    return false;
  }
  auto out_t = mlir::dyn_cast<mlir::RankedTensorType>(d_softmax_out.getType());
  // GENERIC: semantic softmax backward tensor has rank >= 2.
  if (!out_t || !out_t.hasStaticShape() || out_t.getRank() < 2) {
    return false;
  }
  const int64_t rank = out_t.getRank();
  mlir::Value update = scatter_op.getUpdates()[0];
  for (int i = 0; i < kMaxWalkDepth; ++i) {
    if (auto resh = update.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      update = resh.getOperand();
      continue;
    }
    if (auto bcast =
            update.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      update = bcast.getOperand();
      continue;
    }
    if (auto conv = update.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      update = conv.getOperand();
      continue;
    }
    if (auto neg = update.getDefiningOp<mlir::stablehlo::NegOp>()) {
      update = neg.getOperand();
      continue;
    }
    break;
  }
  mlir::stablehlo::ReduceOp sum_reduce;
  if (!IsSingleDimReduceWithOp<mlir::stablehlo::AddOp>(update, rank - 1,
                                                       &sum_reduce) ||
      !IsZeroSplatConstant(sum_reduce.getInitValues()[0])) {
    return false;
  }
  mlir::Value sum_in = sum_reduce.getInputs()[0];
  if (auto neg = sum_in.getDefiningOp<mlir::stablehlo::NegOp>()) {
    sum_in = neg.getOperand();
  }
  if (sum_in != d_softmax_out && UnwrapViewAndConvertChain(sum_in) !=
                                     UnwrapViewAndConvertChain(d_softmax_out)) {
    return false;
  }
  return IsSoftmaxBackwardJacobianAlongLastDim(d_softmax_out, rank);
}

// 3. Bypasses redundant pre-softmax max-subtraction `y = x - Broadcast(max(x))`
// when `y` only feeds a shifted softmax along the same axis, and folds the
// corresponding backward `Add(d_softmax, Scatter(0, argmax, -Sum(d_softmax)))`
// to `d_softmax` because the row sum of the softmax backward Jacobian is
// algebraically zero.
void RewriteShiftInvariantPreSoftmaxAndZeroArgmaxScatter(
    mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::SubtractOp> sub_ops;
  func.walk(
      [&sub_ops](mlir::stablehlo::SubtractOp op) { sub_ops.push_back(op); });

  for (mlir::stablehlo::SubtractOp sub_op : sub_ops) {
    if (!sub_op || sub_op->getParentOp() == nullptr || sub_op->use_empty()) {
      continue;
    }
    auto sub_type = mlir::dyn_cast<mlir::RankedTensorType>(sub_op.getType());
    // GENERIC: semantic softmax reduction requires rank >= 2.
    if (!sub_type || !sub_type.hasStaticShape() || sub_type.getRank() < 2 ||
        !mlir::isa<mlir::FloatType>(sub_type.getElementType())) {
      continue;
    }
    const int64_t rank = sub_type.getRank();
    mlir::Value max_peeled =
        PeelReshapeBroadcastChain(sub_op.getRhs(), sub_type.getShape());
    if (!max_peeled) {
      continue;
    }
    mlir::stablehlo::ReduceOp max_reduce;
    if (!IsSingleDimMaxOrArgmaxValueReduce(max_peeled, rank - 1, &max_reduce) ||
        max_reduce.getInputs()[0] != sub_op.getLhs()) {
      continue;
    }

    mlir::Value y_f32 = sub_op.getResult();
    if (sub_op->hasOneUse()) {
      if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(
              *sub_op->getUsers().begin())) {
        y_f32 = conv.getResult();
      }
    }
    auto y_f32_type = mlir::dyn_cast<mlir::RankedTensorType>(y_f32.getType());
    if (!y_f32_type || !y_f32_type.hasStaticShape() ||
        y_f32_type.getShape() != sub_type.getShape()) {
      continue;
    }

    mlir::stablehlo::ReduceOp softmax_max_reduce;
    mlir::stablehlo::SubtractOp softmax_sub;
    bool all_users_are_softmax = true;
    for (mlir::Operation* u : y_f32.getUsers()) {
      if (auto red = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(u)) {
        if (red.getResults().size() == 1 &&
            IsSingleDimReduceWithOp<mlir::stablehlo::MaxOp>(red.getResult(0),
                                                            rank - 1)) {
          softmax_max_reduce = red;
          continue;
        }
      }
      if (auto s2 = mlir::dyn_cast<mlir::stablehlo::SubtractOp>(u)) {
        if (s2.getLhs() == y_f32) {
          softmax_sub = s2;
          continue;
        }
      }
      all_users_are_softmax = false;
      break;
    }
    if (!all_users_are_softmax || !softmax_max_reduce || !softmax_sub) {
      continue;
    }
    mlir::Value max2_peeled =
        PeelReshapeBroadcastChain(softmax_sub.getRhs(), y_f32_type.getShape());
    if (max2_peeled != softmax_max_reduce.getResult(0)) {
      continue;
    }
    sub_op.replaceAllUsesWith(sub_op.getLhs());
  }

  llvm::SmallVector<mlir::stablehlo::AddOp> add_ops;
  func.walk([&add_ops](mlir::stablehlo::AddOp op) { add_ops.push_back(op); });
  for (mlir::stablehlo::AddOp add_op : add_ops) {
    if (!add_op || add_op->getParentOp() == nullptr || add_op->use_empty()) {
      continue;
    }
    if (IsZeroSumSoftmaxBackwardScatter(add_op.getRhs(), add_op.getLhs())) {
      add_op.replaceAllUsesWith(add_op.getLhs());
    } else if (IsZeroSumSoftmaxBackwardScatter(add_op.getLhs(),
                                               add_op.getRhs())) {
      add_op.replaceAllUsesWith(add_op.getRhs());
    }
  }
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  mlir::Operation* func_op = func.getOperation();
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation*> dead_ops;
    func.walk([&dead_ops, func_op](mlir::Operation* op) {
      if (op != func_op && op->use_empty() &&
          (mlir::isOpTriviallyDead(op) ||
           mlir::isa<mlir::stablehlo::ReduceOp, mlir::stablehlo::ScatterOp>(
               op))) {
        dead_ops.push_back(op);
      }
    });
    for (mlir::Operation* op : dead_ops) {
      op->erase();
      changed = true;
    }
  }
}

struct CanonicalizeRectangularSdpaAndShiftInvariantSoftmaxPass
    : public mlir::PassWrapper<
          CanonicalizeRectangularSdpaAndShiftInvariantSoftmaxPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CanonicalizeRectangularSdpaAndShiftInvariantSoftmaxPass)

  llvm::StringRef getArgument() const override {
    return "canonicalize-rectangular-sdpa-and-shift-invariant-softmax";
  }
  llvm::StringRef getDescription() const override {
    return "Canonicalizes rectangular lane-aligned-key SDPA reductions, "
           "eliminates shift-invariant pre-softmax max-subtraction and "
           "zero-sum backward argmax scatters, and folds standalone "
           "diagonal-reflexive row_all_masked reductions.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    RewriteStandaloneRowAllMaskedReductions(func);
    EraseTriviallyDeadOps(func);
    RewriteShiftInvariantPreSoftmaxAndZeroArgmaxScatter(func);
    EraseTriviallyDeadOps(func);
    RewriteRectangularSdpaReductions(func);
    EraseTriviallyDeadOps(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
createCanonicalizeRectangularSdpaAndShiftInvariantSoftmaxPass() {
  return std::make_unique<
      CanonicalizeRectangularSdpaAndShiftInvariantSoftmaxPass>();
}

}  // namespace torch_tpu
