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

#include "csrc/mlir/passes/canonicalize_gqa_and_masked_online_softmax.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
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
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// GENERIC: hardware TPU vector register lane count and MXU contracting width is
// 128; XLA FlashAttention online-softmax requires 128-aligned sequence lengths.
constexpr int64_t kTpuLaneWidth = 128;

// GENERIC: cost-model + hardware TPU 16 MiB VMEM capacity (4M f32 elements);
// streaming online-softmax amortizes sub-lane padding once scores reach 16 MiB.
constexpr int64_t kTpuScopedVmemLimitBytes = 16 * 1024 * 1024;

// GENERIC: capability bound on recursive expression-tree pattern matching.
constexpr int kMaxWalkDepth = 12;

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

llvm::SmallVector<int64_t> GetNonUnitDimSizes(llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t> out;
  for (int64_t d : shape) {
    // GENERIC: semantic unit extent (size == 1) carries no coordinate variation
    if (d > 1) {
      out.push_back(d);
    }
  }
  return out;
}

// Maps dimension `from_dim` of `from_shape` to the corresponding dimension in
// `to_shape` when a reshape preserves the extent, prefix product, and suffix
// product of `from_dim`. Returns -1 if `from_dim` has unit extent (or < 0), and
// -2 if the reshape splits or merges `from_dim` with a non-unit dimension.
int64_t MapDimAcrossReshape(llvm::ArrayRef<int64_t> from_shape,
                            llvm::ArrayRef<int64_t> to_shape,
                            int64_t from_dim) {
  if (from_dim < 0 || from_dim >= static_cast<int64_t>(from_shape.size())) {
    return -1;
  }
  const int64_t extent = from_shape[from_dim];
  // GENERIC: semantic unit-extent axis has constant coordinate 0
  if (extent <= 1) {
    return -1;
  }
  int64_t from_prefix = 1;
  for (int64_t i = 0; i < from_dim; ++i) {
    from_prefix *= from_shape[i];
  }
  int64_t from_suffix = 1;
  for (int64_t i = from_dim + 1; i < static_cast<int64_t>(from_shape.size());
       ++i) {
    from_suffix *= from_shape[i];
  }
  int64_t to_prefix = 1;
  int64_t total_to = 1;
  for (int64_t d : to_shape) {
    total_to *= d;
  }
  if (total_to != from_prefix * extent * from_suffix) {
    return -2;
  }
  for (int64_t j = 0; j < static_cast<int64_t>(to_shape.size()); ++j) {
    const int64_t d = to_shape[j];
    const int64_t to_suffix = total_to / (to_prefix * d);
    if (d == extent && to_prefix == from_prefix && to_suffix == from_suffix) {
      return j;
    }
    to_prefix *= d;
  }
  return -2;
}

mlir::Value PeelReshapeBroadcastChain(mlir::Value val,
                                      llvm::ArrayRef<int64_t> outer_shape) {
  const int64_t rank = static_cast<int64_t>(outer_shape.size());
  // GENERIC: semantic rank-1 or scalar tensor has no prefix dimensions
  if (rank < 2) {
    return nullptr;
  }
  llvm::ArrayRef<int64_t> prefix_shape = outer_shape.drop_back(1);
  auto matches_prefix_or_unit_last = [&](mlir::RankedTensorType t) -> bool {
    if (!t || !t.hasStaticShape()) {
      return false;
    }
    if (t.getRank() == rank - 1) {
      return t.getShape() == prefix_shape;
    }
    // GENERIC: semantic unit trailing dimension before broadcast expansion
    if (t.getRank() == rank && t.getDimSize(rank - 1) == 1) {
      return t.getShape().drop_back(1) == prefix_shape;
    }
    return false;
  };

  mlir::Value cur = val;
  bool saw_bcast = false;
  for (int depth = 0; depth < kMaxWalkDepth; ++depth) {
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
  return cur;
}

template <typename BinaryOpTy>
bool IsSingleDimReduceWithOp(mlir::Value val, int64_t expected_dim,
                             mlir::stablehlo::ReduceOp* out_reduce = nullptr) {
  auto reduce_op = val.getDefiningOp<mlir::stablehlo::ReduceOp>();
  // GENERIC: semantic single-input, single-output, single-axis reduction
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
  // GENERIC: semantic binary reduction combiner takes 2 scalar arguments
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

mlir::Value UnwrapViewAndConvertChain(mlir::Value val) {
  for (int depth = 0; depth < kMaxWalkDepth; ++depth) {
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

std::optional<llvm::APFloat> ExtractSplatFloatValue(mlir::Value val) {
  val = UnwrapViewAndConvertChain(val);
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr || attr.empty() || !attr.isSplat() ||
      !mlir::isa<mlir::FloatType>(attr.getElementType())) {
    return std::nullopt;
  }
  return attr.getSplatValue<llvm::APFloat>();
}

bool IsZeroSplatConstant(mlir::Value val) {
  std::optional<llvm::APFloat> apf = ExtractSplatFloatValue(val);
  return apf.has_value() && apf->isZero();
}

bool IsNegInfSplatConstant(mlir::Value val) {
  std::optional<llvm::APFloat> apf = ExtractSplatFloatValue(val);
  return apf.has_value() && apf->isInfinity() && apf->isNegative();
}

bool IsFiniteFloatSplatConstant(mlir::Value val) {
  std::optional<llvm::APFloat> apf = ExtractSplatFloatValue(val);
  return apf.has_value() && !apf->isInfinity() && !apf->isNaN();
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

mlir::Value CreateSplatIntConst(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::RankedTensorType type, int64_t value) {
  auto int_type = mlir::cast<mlir::IntegerType>(type.getElementType());
  llvm::APInt api(int_type.getWidth(), static_cast<uint64_t>(value),
                  /*isSigned=*/true);
  auto attr = mlir::DenseElementsAttr::get(type, api);
  return builder.create<mlir::stablehlo::ConstantOp>(loc, type, attr);
}

mlir::Value ConvertIfNeeded(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value val, mlir::Type target_elem_type) {
  auto ranked = mlir::cast<mlir::RankedTensorType>(val.getType());
  if (ranked.getElementType() == target_elem_type) {
    return val;
  }
  auto new_type =
      mlir::RankedTensorType::get(ranked.getShape(), target_elem_type);
  return builder.create<mlir::stablehlo::ConvertOp>(loc, new_type, val);
}

// Forward declaration for mutual recursion with ExtractSplatIntValue.
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
    int64_t op_dim = -1;
    int64_t op_start = 0;
    int64_t op_step = 0;
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

// Matches an integer tensor whose value at index `(i_0, ..., i_{r-1})` is
// `out_start + out_step * i_{out_dim}` (varying along a single dimension).
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
  if (auto cat = val.getDefiningOp<mlir::stablehlo::ConcatenateOp>()) {
    const int64_t c_dim = static_cast<int64_t>(cat.getDimension());
    if (c_dim < 0 || c_dim >= type.getRank()) {
      return false;
    }
    bool found_step = false;
    int64_t base_start = 0;
    int64_t step = 0;
    int64_t offset = 0;
    for (mlir::Value inp : cat.getInputs()) {
      auto in_t = mlir::dyn_cast<mlir::RankedTensorType>(inp.getType());
      if (!in_t || !in_t.hasStaticShape()) {
        return false;
      }
      int64_t d = -1, s = 0, st = 0;
      if (MatchAffineIota1D(inp, d, s, st, depth + 1) && d == c_dim) {
        found_step = true;
        step = st;
        base_start = s - offset * st;
        break;
      }
      offset += in_t.getDimSize(c_dim);
    }
    if (!found_step) {
      return false;
    }
    offset = 0;
    for (mlir::Value inp : cat.getInputs()) {
      auto in_t = mlir::cast<mlir::RankedTensorType>(inp.getType());
      const int64_t len = in_t.getDimSize(c_dim);
      const int64_t expected_start = base_start + offset * step;
      int64_t d = -1, s = 0, st = 0;
      if (MatchAffineIota1D(inp, d, s, st, depth + 1)) {
        if (d != c_dim || st != step || s != expected_start) {
          return false;
        }
      } else if (len == 1) {
        // GENERIC: semantic single-element slice along concat_dim matches the
        // affine progression when its scalar splat equals expected_start.
        auto c = ExtractSplatIntValue(inp, depth + 1);
        if (!c.has_value() || *c != expected_start) {
          return false;
        }
      } else {
        return false;
      }
      offset += len;
    }
    out_dim = c_dim;
    out_start = base_start;
    out_step = step;
    return true;
  }
  return false;
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

bool IsAdditiveAssociativeScan(mlir::chlo::ScanOp scan_op) {
  // GENERIC: semantic single-input single-carry associative prefix-sum scan
  if (scan_op.getInputs().size() != 1 || scan_op.getInits().size() != 1 ||
      scan_op.getResults().size() != 2) {
    return false;
  }
  mlir::Region& body = scan_op.getBody();
  if (!body.hasOneBlock()) {
    return false;
  }
  mlir::Block& block = body.front();
  // GENERIC: semantic binary scan combiner takes 2 arguments and returns 2
  if (block.getNumArguments() != 2) {
    return false;
  }
  auto ret_op =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  // GENERIC: semantic scan return yields (scan_elem, next_carry)
  if (!ret_op || ret_op.getOperands().size() != 2 ||
      ret_op.getOperands()[0] != ret_op.getOperands()[1]) {
    return false;
  }
  auto add_op = ret_op.getOperands()[0].getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add_op) {
    return false;
  }
  mlir::Value a0 = block.getArgument(0);
  mlir::Value a1 = block.getArgument(1);
  return (add_op.getLhs() == a0 && add_op.getRhs() == a1) ||
         (add_op.getLhs() == a1 && add_op.getRhs() == a0);
}

// 1. Folds constant-step affine IotaOp differences `Subtract(AffineIota1D,
// AffineIota1D)` feeding `CompareOp -> ConvertOp -> chlo.scan(0, 0, add) ->
// GatherOp -> CompareOp(EQ) -> AndOp` (such as unpacked sequence-ID prefix-sum
// masks). Scoped strictly to chains rooted at an affine-iota difference or
// zero-splat scan so unrelated comparisons and gathers are untouched.
void FoldAffineIotaDiffsAndZeroSplatScans(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::CompareOp> cmp_ops;
  func.walk([&](mlir::stablehlo::CompareOp op) { cmp_ops.push_back(op); });

  bool folded_any_diff_cmp = false;
  for (mlir::stablehlo::CompareOp cmp_op : cmp_ops) {
    if (!cmp_op || cmp_op->getParentOp() == nullptr || cmp_op->use_empty()) {
      continue;
    }
    auto cmp_type = mlir::dyn_cast<mlir::RankedTensorType>(cmp_op.getType());
    if (!cmp_type || !cmp_type.hasStaticShape()) {
      continue;
    }
    const bool has_sub =
        cmp_op.getLhs().getDefiningOp<mlir::stablehlo::SubtractOp>() !=
            nullptr ||
        cmp_op.getRhs().getDefiningOp<mlir::stablehlo::SubtractOp>() != nullptr;
    if (!has_sub) {
      continue;
    }
    auto lhs_val = ExtractSplatIntValue(cmp_op.getLhs());
    auto rhs_val = ExtractSplatIntValue(cmp_op.getRhs());
    if (!lhs_val.has_value() || !rhs_val.has_value()) {
      continue;
    }
    mlir::OpBuilder builder(cmp_op);
    const bool result_bool =
        EvalIntComparison(cmp_op.getComparisonDirection(), *lhs_val, *rhs_val);
    mlir::Value cst =
        CreateSplatBoolConst(builder, cmp_op.getLoc(), cmp_type, result_bool);
    cmp_op.replaceAllUsesWith(cst);
    folded_any_diff_cmp = true;
  }

  llvm::SmallVector<mlir::chlo::ScanOp> scan_ops;
  func.walk([&](mlir::chlo::ScanOp op) { scan_ops.push_back(op); });

  llvm::DenseSet<mlir::Value> folded_scan_outputs;
  for (mlir::chlo::ScanOp scan_op : scan_ops) {
    if (!scan_op || scan_op->getParentOp() == nullptr || scan_op->use_empty() ||
        !IsAdditiveAssociativeScan(scan_op)) {
      continue;
    }
    auto in_val = ExtractSplatIntValue(scan_op.getInputs()[0]);
    auto init_val = ExtractSplatIntValue(scan_op.getInits()[0]);
    if (!in_val.has_value() || *in_val != 0 || !init_val.has_value() ||
        *init_val != 0) {
      continue;
    }
    auto res0_t =
        mlir::dyn_cast<mlir::RankedTensorType>(scan_op.getResult(0).getType());
    auto res1_t =
        mlir::dyn_cast<mlir::RankedTensorType>(scan_op.getResult(1).getType());
    if (!res0_t || !res1_t || !res0_t.hasStaticShape() ||
        !res1_t.hasStaticShape()) {
      continue;
    }
    mlir::OpBuilder builder(scan_op);
    mlir::Location loc = scan_op.getLoc();
    mlir::Value zero0 = CreateSplatIntConst(builder, loc, res0_t, 0);
    mlir::Value zero1 = CreateSplatIntConst(builder, loc, res1_t, 0);
    scan_op.getResult(0).replaceAllUsesWith(zero0);
    scan_op.getResult(1).replaceAllUsesWith(zero1);
    folded_scan_outputs.insert(zero0);
    folded_scan_outputs.insert(zero1);
    scan_op.erase();
  }

  if (folded_scan_outputs.empty() && !folded_any_diff_cmp) {
    return;
  }

  llvm::DenseSet<mlir::Value> folded_gather_outputs;
  llvm::SmallVector<mlir::stablehlo::GatherOp> gather_ops;
  func.walk([&](mlir::stablehlo::GatherOp op) { gather_ops.push_back(op); });
  for (mlir::stablehlo::GatherOp gather_op : gather_ops) {
    if (!gather_op || gather_op->getParentOp() == nullptr ||
        gather_op->use_empty()) {
      continue;
    }
    mlir::Value root_opnd = UnwrapViewAndConvertChain(gather_op.getOperand());
    if (!folded_scan_outputs.contains(root_opnd)) {
      continue;
    }
    auto splat_val = ExtractSplatIntValue(gather_op.getOperand());
    auto g_type = mlir::dyn_cast<mlir::RankedTensorType>(gather_op.getType());
    if (!splat_val.has_value() || !g_type || !g_type.hasStaticShape()) {
      continue;
    }
    mlir::OpBuilder builder(gather_op);
    mlir::Value cst =
        CreateSplatIntConst(builder, gather_op.getLoc(), g_type, *splat_val);
    gather_op.replaceAllUsesWith(cst);
    folded_gather_outputs.insert(cst);
  }

  if (folded_gather_outputs.empty()) {
    return;
  }

  llvm::DenseSet<mlir::Value> folded_true_preds;
  llvm::SmallVector<mlir::stablehlo::CompareOp> eq_ops;
  func.walk([&](mlir::stablehlo::CompareOp op) { eq_ops.push_back(op); });
  for (mlir::stablehlo::CompareOp eq_op : eq_ops) {
    if (!eq_op || eq_op->getParentOp() == nullptr || eq_op->use_empty()) {
      continue;
    }
    mlir::Value lhs_root = UnwrapViewAndConvertChain(eq_op.getLhs());
    mlir::Value rhs_root = UnwrapViewAndConvertChain(eq_op.getRhs());
    if (!folded_gather_outputs.contains(lhs_root) &&
        !folded_gather_outputs.contains(rhs_root)) {
      continue;
    }
    auto lhs_val = ExtractSplatIntValue(eq_op.getLhs());
    auto rhs_val = ExtractSplatIntValue(eq_op.getRhs());
    auto eq_type = mlir::dyn_cast<mlir::RankedTensorType>(eq_op.getType());
    if (!lhs_val.has_value() || !rhs_val.has_value() || !eq_type ||
        !eq_type.hasStaticShape()) {
      continue;
    }
    const bool res =
        EvalIntComparison(eq_op.getComparisonDirection(), *lhs_val, *rhs_val);
    mlir::OpBuilder builder(eq_op);
    mlir::Value cst =
        CreateSplatBoolConst(builder, eq_op.getLoc(), eq_type, res);
    eq_op.replaceAllUsesWith(cst);
    if (res) {
      folded_true_preds.insert(cst);
    }
  }

  llvm::SmallVector<mlir::stablehlo::AndOp> and_ops;
  func.walk([&](mlir::stablehlo::AndOp op) { and_ops.push_back(op); });
  for (mlir::stablehlo::AndOp and_op : and_ops) {
    if (!and_op || and_op->getParentOp() == nullptr || and_op->use_empty()) {
      continue;
    }
    mlir::Value lhs_root = UnwrapViewAndConvertChain(and_op.getLhs());
    mlir::Value rhs_root = UnwrapViewAndConvertChain(and_op.getRhs());
    if (folded_true_preds.contains(rhs_root) &&
        IsBoolSplatConstant(and_op.getRhs(), true)) {
      and_op.replaceAllUsesWith(and_op.getLhs());
    } else if (folded_true_preds.contains(lhs_root) &&
               IsBoolSplatConstant(and_op.getLhs(), true)) {
      and_op.replaceAllUsesWith(and_op.getRhs());
    }
  }
}

struct PeeledMaskSelect {
  mlir::Value keep_pred;
  int64_t q_dim = -1;
  int64_t k_dim = -1;
  int64_t seq_len = 0;
  bool finite_is_zero = false;
};

// Peels `BroadcastInDimOp` and unit-dimension `ReshapeOp` from a 4D mask tensor
// `[B, H, S_q, S_kv]` down to `SelectOp(keep_pred, finite_splat, -inf)` (or
// `SelectOp(masked_pred, -inf, finite_splat)`), tracking which dimensions of
// `keep_pred` correspond to the query axis (`rank - 2`) and key axis
// (`rank - 1`).
std::optional<PeeledMaskSelect> PeelMaskToSelectPredicate(
    mlir::Value mask_val) {
  auto full_type = mlir::dyn_cast<mlir::RankedTensorType>(mask_val.getType());
  // GENERIC: semantic 2D query-key matrix requires rank >= 2
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
  return PeeledMaskSelect{sel_op.getPred(), root_q_dim, root_k_dim, q_len,
                          IsZeroSplatConstant(sel_op.getOnTrue())};
}

// Evaluates an integer tensor `val` along the diagonal `i_{q_dim} = i_{k_dim} =
// i` as an affine function `out_slope * i + out_offset`.
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

// Returns true if `pred` is provably true on at least one column `k_idx` in
// every row `q_idx`. Peels top-level `ReverseOp` along `q_dim` or `k_dim`
// (which permutes rows or columns without changing row-wise satisfiability)
// before testing the diagonal `q_idx == k_idx`.
bool HasAtLeastOneUnmaskedKeyPerRow(mlir::Value pred, int64_t q_dim,
                                    int64_t k_dim, int64_t seq_len) {
  if (!MapDimsThroughBroadcastAndReshape(pred, q_dim, k_dim)) {
    return false;
  }
  if (auto rev_op = pred.getDefiningOp<mlir::stablehlo::ReverseOp>()) {
    for (int64_t d : rev_op.getDimensions()) {
      if (d != q_dim && d != k_dim) {
        return false;
      }
    }
    pred = rev_op.getOperand();
  }
  return IsDiagonalPredicateTrue(pred, q_dim, k_dim, seq_len);
}

// Returns true if `pred` is provably true for EVERY `(q_idx, k_idx)` entry in
// the matrix (so `Select(pred, 0.0, -inf)` is identically zero everywhere).
bool IsPredicateAlwaysTrueEverywhere(mlir::Value pred, int64_t q_dim,
                                     int64_t k_dim, int64_t seq_len) {
  if (!MapDimsThroughBroadcastAndReshape(pred, q_dim, k_dim)) {
    return false;
  }
  if (IsBoolSplatConstant(pred, true)) {
    return true;
  }
  // If `pred` does not depend on `k_dim` (`k_dim == -1`) and is true along
  // `q_dim` for all `0 <= q_idx < seq_len`, then it is true on every entry.
  if (k_dim == -1) {
    return IsDiagonalPredicateTrue(pred, q_dim, /*k_dim=*/-1, seq_len);
  }
  return false;
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

// If `masked_logits` is `AddOp(scaled_logits, MaxOp(mask_val, -FLT_MAX))` where
// `mask_val` is `SelectOp(pred, 0.0, -inf)` and `pred` is provably true on
// every matrix entry, returns `scaled_logits`.
mlir::Value TrySimplifyTriviallyZeroClampedMaskAdd(mlir::Value masked_logits) {
  auto add_op = masked_logits.getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add_op) {
    return nullptr;
  }
  auto try_side = [&](mlir::Value logits_side,
                      mlir::Value clamp_side) -> mlir::Value {
    auto max_op = clamp_side.getDefiningOp<mlir::stablehlo::MaxOp>();
    if (!max_op) {
      return nullptr;
    }
    mlir::Value mask_val;
    if (IsFiniteFloatSplatConstant(max_op.getRhs())) {
      mask_val = max_op.getLhs();
    } else if (IsFiniteFloatSplatConstant(max_op.getLhs())) {
      mask_val = max_op.getRhs();
    } else {
      return nullptr;
    }
    auto peeled = PeelMaskToSelectPredicate(mask_val);
    if (!peeled || !peeled->finite_is_zero) {
      return nullptr;
    }
    if (IsPredicateAlwaysTrueEverywhere(peeled->keep_pred, peeled->q_dim,
                                        peeled->k_dim, peeled->seq_len)) {
      return logits_side;
    }
    return nullptr;
  };
  if (mlir::Value simplified = try_side(add_op.getLhs(), add_op.getRhs())) {
    return simplified;
  }
  return try_side(add_op.getRhs(), add_op.getLhs());
}

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
  auto sub_op = logits.getDefiningOp<mlir::stablehlo::SubtractOp>();
  if (!sub_op) {
    return std::nullopt;
  }
  auto sub_type = mlir::dyn_cast<mlir::RankedTensorType>(sub_op.getType());
  // GENERIC: semantic batched attention logits have rank >= 3
  if (!sub_type || !sub_type.hasStaticShape() || sub_type.getRank() < 3) {
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
                                                       &max_reduce)) {
    return std::nullopt;
  }
  if (max_reduce.getInputs()[0] != sub_op.getLhs()) {
    return std::nullopt;
  }
  return ShiftedLogitsMatch{sub_op.getLhs(), max_reduce, sub_op};
}

bool MatchRowAllMaskedSelect(mlir::Value val, mlir::Value& out_row_mask_pred_3d,
                             mlir::stablehlo::SelectOp& out_select_op) {
  auto sel = val.getDefiningOp<mlir::stablehlo::SelectOp>();
  if (!sel || !IsZeroSplatConstant(sel.getOnTrue())) {
    return false;
  }
  auto sel_type = mlir::dyn_cast<mlir::RankedTensorType>(sel.getType());
  // GENERIC: semantic 4D [B, H, Q, K] attention score tensor
  if (!sel_type || !sel_type.hasStaticShape() || sel_type.getRank() != 4) {
    return false;
  }
  auto bcast = sel.getPred().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast || !IsDropLastDimBroadcast(bcast.getBroadcastDimensions(), 4)) {
    return false;
  }
  mlir::Value pred_3d = bcast.getOperand();
  if (!IsSingleDimReduceWithOp<mlir::stablehlo::AndOp>(pred_3d, 3)) {
    return false;
  }
  out_row_mask_pred_3d = pred_3d;
  out_select_op = sel;
  return true;
}

struct UpstreamGqaQkDotMatch {
  mlir::stablehlo::ReshapeOp logits_4d_reshape;
  mlir::stablehlo::DotGeneralOp qk_dot;
  mlir::Value q_4d;
  mlir::Value k_4d;
  int64_t h_kv = 0;
  int64_t g = 0;
  int64_t qk_head_dim = 0;
};

std::optional<UpstreamGqaQkDotMatch> FindUpstreamGqaQkDot(
    mlir::Value masked_logits, int64_t b_size, int64_t h_q, int64_t q_len,
    int64_t kv_len) {
  auto try_candidate =
      [&](mlir::Value cand) -> std::optional<UpstreamGqaQkDotMatch> {
    while (auto conv = cand.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      if (!conv->hasOneUse()) {
        return std::nullopt;
      }
      cand = conv.getOperand();
    }
    auto logits_resh = cand.getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!logits_resh || !logits_resh->hasOneUse()) {
      return std::nullopt;
    }
    auto resh_t = mlir::dyn_cast<mlir::RankedTensorType>(logits_resh.getType());
    // GENERIC: semantic 4D [B, H_q, S_q, S_kv] reshaped from 5D GQA dot
    if (!resh_t || !resh_t.hasStaticShape() || resh_t.getRank() != 4 ||
        resh_t.getDimSize(0) != b_size || resh_t.getDimSize(1) != h_q ||
        resh_t.getDimSize(2) != q_len || resh_t.getDimSize(3) != kv_len) {
      return std::nullopt;
    }
    auto qk_dot =
        logits_resh.getOperand().getDefiningOp<mlir::stablehlo::DotGeneralOp>();
    if (!qk_dot || !qk_dot->hasOneUse()) {
      return std::nullopt;
    }
    auto dot_t = mlir::dyn_cast<mlir::RankedTensorType>(qk_dot.getType());
    // GENERIC: semantic 5D [B, H_kv, G, S_q, S_kv] GQA logits dot product
    if (!dot_t || !dot_t.hasStaticShape() || dot_t.getRank() != 5 ||
        dot_t.getDimSize(0) != b_size || dot_t.getDimSize(1) <= 0 ||
        dot_t.getDimSize(2) <= 0 ||
        dot_t.getDimSize(1) * dot_t.getDimSize(2) != h_q ||
        dot_t.getDimSize(3) != q_len || dot_t.getDimSize(4) != kv_len) {
      return std::nullopt;
    }
    auto dnums = qk_dot.getDotDimensionNumbers();
    if (llvm::ArrayRef<int64_t>(dnums.getLhsBatchingDimensions()) !=
            llvm::ArrayRef<int64_t>{0, 1} ||
        llvm::ArrayRef<int64_t>(dnums.getRhsBatchingDimensions()) !=
            llvm::ArrayRef<int64_t>{0, 1} ||
        llvm::ArrayRef<int64_t>(dnums.getLhsContractingDimensions()) !=
            llvm::ArrayRef<int64_t>{4} ||
        llvm::ArrayRef<int64_t>(dnums.getRhsContractingDimensions()) !=
            llvm::ArrayRef<int64_t>{3}) {
      return std::nullopt;
    }
    const int64_t h_kv = dot_t.getDimSize(1);
    const int64_t g = dot_t.getDimSize(2);
    auto q_5d_t =
        mlir::dyn_cast<mlir::RankedTensorType>(qk_dot.getLhs().getType());
    auto k_4d_t =
        mlir::dyn_cast<mlir::RankedTensorType>(qk_dot.getRhs().getType());
    // GENERIC: semantic 5D grouped query and 4D KV head tensors
    if (!q_5d_t || !k_4d_t || !q_5d_t.hasStaticShape() ||
        !k_4d_t.hasStaticShape() || q_5d_t.getRank() != 5 ||
        k_4d_t.getRank() != 4) {
      return std::nullopt;
    }
    const int64_t qk_head_dim = q_5d_t.getDimSize(4);
    if (qk_head_dim <= 0 || k_4d_t.getDimSize(0) != b_size ||
        k_4d_t.getDimSize(1) != h_kv || k_4d_t.getDimSize(2) != kv_len ||
        k_4d_t.getDimSize(3) != qk_head_dim) {
      return std::nullopt;
    }
    mlir::Value q_4d;
    if (auto q_resh =
            qk_dot.getLhs().getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      auto q_4d_t =
          mlir::dyn_cast<mlir::RankedTensorType>(q_resh.getOperand().getType());
      // GENERIC: semantic 4D [B, H_q, S_q, D] ungrouped query tensor
      if (q_4d_t && q_4d_t.hasStaticShape() && q_4d_t.getRank() == 4 &&
          q_4d_t.getDimSize(0) == b_size && q_4d_t.getDimSize(1) == h_q &&
          q_4d_t.getDimSize(2) == q_len &&
          q_4d_t.getDimSize(3) == qk_head_dim) {
        q_4d = q_resh.getOperand();
      }
    }
    return UpstreamGqaQkDotMatch{
        logits_resh, qk_dot, q_4d, qk_dot.getRhs(), h_kv, g, qk_head_dim};
  };

  if (auto m = try_candidate(masked_logits)) {
    return m;
  }
  if (auto sel = masked_logits.getDefiningOp<mlir::stablehlo::SelectOp>()) {
    if (auto m = try_candidate(sel.getOnFalse())) {
      return m;
    }
    if (auto m = try_candidate(sel.getOnTrue())) {
      return m;
    }
  }
  if (auto add = masked_logits.getDefiningOp<mlir::stablehlo::AddOp>()) {
    if (auto m = try_candidate(add.getLhs())) {
      return m;
    }
    if (auto m = try_candidate(add.getRhs())) {
      return m;
    }
  }
  return std::nullopt;
}

struct DownstreamOutputDotMatch {
  mlir::stablehlo::DotGeneralOp dot_op;
  mlir::stablehlo::ReshapeOp gqa_probs_reshape;
  mlir::stablehlo::ReshapeOp gqa_out_reshape;
  mlir::stablehlo::ConvertOp probs_convert;
};

std::optional<DownstreamOutputDotMatch> FindDownstreamAttentionOutputDot(
    mlir::Value softmax_val) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(softmax_val.getType());
  // GENERIC: semantic 4D [B, H_q, S_q, S_kv] attention probability tensor
  if (!type || !type.hasStaticShape() || type.getRank() != 4) {
    return std::nullopt;
  }
  const int64_t b_size = type.getDimSize(0);
  const int64_t h_q = type.getDimSize(1);
  const int64_t q_len = type.getDimSize(2);
  const int64_t kv_len = type.getDimSize(3);

  auto inspect_operation = [&](mlir::Operation* op, mlir::Value val,
                               mlir::stablehlo::ConvertOp probs_convert)
      -> std::optional<DownstreamOutputDotMatch> {
    if (auto dg = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
      if (dg.getLhs() == val) {
        auto dnums = dg.getDotDimensionNumbers();
        if (llvm::ArrayRef<int64_t>(dnums.getLhsBatchingDimensions()) ==
                llvm::ArrayRef<int64_t>{0, 1} &&
            llvm::ArrayRef<int64_t>(dnums.getRhsBatchingDimensions()) ==
                llvm::ArrayRef<int64_t>{0, 1} &&
            llvm::ArrayRef<int64_t>(dnums.getLhsContractingDimensions()) ==
                llvm::ArrayRef<int64_t>{3} &&
            llvm::ArrayRef<int64_t>(dnums.getRhsContractingDimensions()) ==
                llvm::ArrayRef<int64_t>{2}) {
          return DownstreamOutputDotMatch{dg, nullptr, nullptr, probs_convert};
        }
      }
    }
    if (auto resh = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
      auto r_type = mlir::dyn_cast<mlir::RankedTensorType>(resh.getType());
      // GENERIC: semantic 5D [B, H_kv, G, S_q, S_kv] GQA probability reshape
      if (r_type && r_type.hasStaticShape() && r_type.getRank() == 5 &&
          resh->hasOneUse() && r_type.getDimSize(0) == b_size &&
          r_type.getDimSize(1) > 0 && r_type.getDimSize(2) > 0 &&
          r_type.getDimSize(1) * r_type.getDimSize(2) == h_q &&
          r_type.getDimSize(3) == q_len && r_type.getDimSize(4) == kv_len &&
          !resh->use_empty()) {
        if (auto dg = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(
                *resh->getUsers().begin())) {
          if (dg.getLhs() == resh.getResult() && dg->hasOneUse() &&
              !dg->use_empty()) {
            auto dnums = dg.getDotDimensionNumbers();
            if (llvm::ArrayRef<int64_t>(dnums.getLhsBatchingDimensions()) ==
                    llvm::ArrayRef<int64_t>{0, 1} &&
                llvm::ArrayRef<int64_t>(dnums.getRhsBatchingDimensions()) ==
                    llvm::ArrayRef<int64_t>{0, 1} &&
                llvm::ArrayRef<int64_t>(dnums.getLhsContractingDimensions()) ==
                    llvm::ArrayRef<int64_t>{4} &&
                llvm::ArrayRef<int64_t>(dnums.getRhsContractingDimensions()) ==
                    llvm::ArrayRef<int64_t>{2}) {
              if (auto out_resh = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(
                      *dg->getUsers().begin())) {
                return DownstreamOutputDotMatch{dg, resh, out_resh,
                                                probs_convert};
              }
            }
          }
        }
      }
    }
    return std::nullopt;
  };

  for (mlir::Operation* user : softmax_val.getUsers()) {
    if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(user)) {
      if (conv->hasOneUse()) {
        for (mlir::Operation* u2 : conv.getResult().getUsers()) {
          if (auto match = inspect_operation(u2, conv.getResult(), conv)) {
            return match;
          }
        }
      }
    } else {
      if (auto match = inspect_operation(user, softmax_val, nullptr)) {
        return match;
      }
    }
  }
  return std::nullopt;
}

// 2. Normalizes forward-only 5D GQA `Q @ K^T` and `probs @ V` contractions to
// 4D `[B, H_q, ...]`, eliminates `row_all_masked` `Select` when every query row
// has an unmasked diagonal key (or sinks `Select` past `DotGeneral` otherwise),
// and simplifies trivially-zero clamped masks so XLA's FlashAttention pass
// fuses GQA and masked attention into `online-softmax`.
void RewriteForwardGqaAndMaskedOnlineSoftmax(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::DivOp> div_ops;
  func.walk([&](mlir::stablehlo::DivOp op) { div_ops.push_back(op); });

  for (mlir::stablehlo::DivOp div_op : div_ops) {
    if (!div_op || div_op->getParentOp() == nullptr || !div_op->hasOneUse()) {
      continue;
    }
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(div_op.getType());
    // GENERIC: semantic 4D [B, H_q, S_q, S_kv] attention score matrix
    if (!type || !type.hasStaticShape() || type.getRank() != 4 ||
        !mlir::isa<mlir::FloatType>(type.getElementType())) {
      continue;
    }
    const int64_t b_size = type.getDimSize(0);
    const int64_t h_q = type.getDimSize(1);
    const int64_t q_len = type.getDimSize(2);
    const int64_t kv_len = type.getDimSize(3);
    if (q_len <= 0 || kv_len <= 0 || (q_len % kTpuLaneWidth) != 0 ||
        (kv_len % kTpuLaneWidth) != 0) {
      continue;
    }

    mlir::Value exp_val = div_op.getLhs();
    auto bcast_op =
        div_op.getRhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!bcast_op || !bcast_op->hasOneUse() ||
        !IsDropLastDimBroadcast(bcast_op.getBroadcastDimensions(), 4)) {
      continue;
    }
    mlir::stablehlo::ReduceOp sum_reduce;
    if (!IsSingleDimReduceWithOp<mlir::stablehlo::AddOp>(bcast_op.getOperand(),
                                                         3, &sum_reduce) ||
        sum_reduce.getInputs()[0] != exp_val || !sum_reduce->hasOneUse()) {
      continue;
    }
    bool exp_only_used_by_forward = true;
    for (mlir::Operation* u : exp_val.getUsers()) {
      if (u != div_op.getOperation() && u != sum_reduce.getOperation()) {
        exp_only_used_by_forward = false;
        break;
      }
    }
    if (!exp_only_used_by_forward) {
      continue;
    }
    auto shifted_match = MatchExpOfShiftedLogits(exp_val);
    if (!shifted_match || !shifted_match->max_reduce->hasOneUse()) {
      continue;
    }

    mlir::Operation* root_softmax_op = div_op.getOperation();
    mlir::stablehlo::SelectOp row_mask_select;
    mlir::Value row_mask_pred_3d;
    mlir::Operation* user = *div_op->getUsers().begin();
    if (user->getNumResults() > 0) {
      mlir::stablehlo::SelectOp masked_sel;
      mlir::Value pred_3d;
      if (MatchRowAllMaskedSelect(user->getResult(0), pred_3d, masked_sel)) {
        root_softmax_op = masked_sel.getOperation();
        row_mask_select = masked_sel;
        row_mask_pred_3d = pred_3d;
      }
    }
    if (!root_softmax_op->hasOneUse()) {
      continue;
    }

    auto dot_match =
        FindDownstreamAttentionOutputDot(root_softmax_op->getResult(0));
    if (!dot_match) {
      continue;
    }
    auto qk_dot_match = FindUpstreamGqaQkDot(shifted_match->masked_logits,
                                             b_size, h_q, q_len, kv_len);
    if (!row_mask_select && !dot_match->gqa_probs_reshape && !qk_dot_match) {
      continue;
    }

    mlir::Value v_4d = dot_match->dot_op.getRhs();
    auto v_type = mlir::dyn_cast<mlir::RankedTensorType>(v_4d.getType());
    // GENERIC: semantic 4D [B, H_kv, S_kv, D_v] value tensor
    if (!v_type || !v_type.hasStaticShape() || v_type.getRank() != 4 ||
        v_type.getDimSize(0) != b_size || v_type.getDimSize(2) != kv_len) {
      continue;
    }
    if (dot_match->gqa_probs_reshape) {
      auto resh_type = mlir::cast<mlir::RankedTensorType>(
          dot_match->gqa_probs_reshape.getType());
      if (v_type.getDimSize(1) != resh_type.getDimSize(1)) {
        continue;
      }
    } else if (v_type.getDimSize(1) != h_q) {
      continue;
    }
    const int64_t head_dim = v_type.getDimSize(3);
    if (head_dim <= 0) {
      continue;
    }
    const int64_t single_head_matrix_f32_bytes =
        q_len * kv_len * static_cast<int64_t>(sizeof(float));
    // When head_dim is a multiple of the 128-lane TPU MXU tile width, every
    // vector tile is 100% lane-utilized; when head_dim < 128, require the
    // single-head score matrix to reach the 16 MiB VMEM capacity so streaming
    // online-softmax amortizes sub-lane padding.
    const bool is_lane_aligned_head_dim = (head_dim % kTpuLaneWidth) == 0;
    if (!is_lane_aligned_head_dim &&
        (head_dim > kTpuLaneWidth ||
         single_head_matrix_f32_bytes < kTpuScopedVmemLimitBytes)) {
      continue;
    }

    mlir::Operation* old_out_op =
        dot_match->gqa_out_reshape ? dot_match->gqa_out_reshape.getOperation()
                                   : dot_match->dot_op.getOperation();
    if (old_out_op->getNumResults() == 0) {
      continue;
    }
    auto out_4d_type = mlir::dyn_cast<mlir::RankedTensorType>(
        old_out_op->getResult(0).getType());
    // GENERIC: semantic 4D [B, H_q, S_q, D_v] attention output tensor
    if (!out_4d_type || !out_4d_type.hasStaticShape() ||
        out_4d_type.getRank() != 4 || out_4d_type.getDimSize(0) != b_size ||
        out_4d_type.getDimSize(1) != h_q ||
        out_4d_type.getDimSize(2) != q_len ||
        out_4d_type.getDimSize(3) != head_dim ||
        out_4d_type.getNumElements() > type.getNumElements()) {
      continue;
    }

    // Rewrite upstream 5D GQA `Q @ K^T` to 4D `[B, H_q, S_q, S_kv]`.
    if (qk_dot_match) {
      auto logits_4d_type = mlir::cast<mlir::RankedTensorType>(
          qk_dot_match->logits_4d_reshape.getType());
      auto k_type =
          mlir::cast<mlir::RankedTensorType>(qk_dot_match->k_4d.getType());
      auto k_rep_type = mlir::RankedTensorType::get(
          {b_size, h_q, kv_len, qk_dot_match->qk_head_dim},
          k_type.getElementType());
      if (k_rep_type.getNumElements() > type.getNumElements()) {
        continue;
      }
      mlir::OpBuilder qk_builder(qk_dot_match->logits_4d_reshape);
      mlir::Location qk_loc = qk_dot_match->logits_4d_reshape.getLoc();
      mlir::Value q_4d = qk_dot_match->q_4d;
      if (!q_4d) {
        auto q_type = mlir::cast<mlir::RankedTensorType>(
            qk_dot_match->qk_dot.getLhs().getType());
        auto q_4d_type = mlir::RankedTensorType::get(
            {b_size, h_q, q_len, qk_dot_match->qk_head_dim},
            q_type.getElementType());
        q_4d = qk_builder.create<mlir::stablehlo::ReshapeOp>(
            qk_loc, q_4d_type, qk_dot_match->qk_dot.getLhs());
      }
      auto k_5d_type = mlir::RankedTensorType::get(
          {b_size, qk_dot_match->h_kv, qk_dot_match->g, kv_len,
           qk_dot_match->qk_head_dim},
          k_type.getElementType());
      mlir::Value k_5d = qk_builder.create<mlir::stablehlo::BroadcastInDimOp>(
          qk_loc, k_5d_type, qk_dot_match->k_4d,
          qk_builder.getDenseI64ArrayAttr({0, 1, 3, 4}));
      mlir::Value k_rep_4d = qk_builder.create<mlir::stablehlo::ReshapeOp>(
          qk_loc, k_rep_type, k_5d);
      auto qk_dnums = mlir::stablehlo::DotDimensionNumbersAttr::get(
          qk_builder.getContext(), /*lhsBatchingDimensions=*/{0, 1},
          /*rhsBatchingDimensions=*/{0, 1}, /*lhsContractingDimensions=*/{3},
          /*rhsContractingDimensions=*/{3});
      mlir::Value new_logits_4d =
          qk_builder.create<mlir::stablehlo::DotGeneralOp>(
              qk_loc, logits_4d_type, q_4d, k_rep_4d, qk_dnums,
              qk_dot_match->qk_dot.getPrecisionConfigAttr(),
              qk_dot_match->qk_dot.getAlgorithmAttr());
      qk_dot_match->logits_4d_reshape.replaceAllUsesWith(new_logits_4d);
    }

    // Simplify `Add(scaled_logits, Max(Select(true, 0.0, -inf), -FLT_MAX))` ->
    // `scaled_logits` when the mask predicate is provably true everywhere.
    if (mlir::Value unmasked_logits = TrySimplifyTriviallyZeroClampedMaskAdd(
            shifted_match->masked_logits)) {
      shifted_match->masked_logits.replaceAllUsesWith(unmasked_logits);
    }

    mlir::OpBuilder builder(old_out_op);
    mlir::Location loc = old_out_op->getLoc();

    mlir::Value unmasked_probs = div_op.getResult();
    mlir::Value probs_for_dot =
        ConvertIfNeeded(builder, loc, unmasked_probs, v_type.getElementType());

    mlir::Value v_for_dot = v_4d;
    if (dot_match->gqa_probs_reshape) {
      const int64_t h_kv = v_type.getDimSize(1);
      if (h_kv <= 0 || (h_q % h_kv) != 0) {
        continue;
      }
      const int64_t g = h_q / h_kv;
      auto v_rep_type = mlir::RankedTensorType::get(
          {b_size, h_q, kv_len, head_dim}, v_type.getElementType());
      if (v_rep_type.getNumElements() > type.getNumElements()) {
        continue;
      }
      auto v_5d_type = mlir::RankedTensorType::get(
          {b_size, h_kv, g, kv_len, head_dim}, v_type.getElementType());
      mlir::Value v_5d = builder.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, v_5d_type, v_4d, builder.getDenseI64ArrayAttr({0, 1, 3, 4}));
      v_for_dot =
          builder.create<mlir::stablehlo::ReshapeOp>(loc, v_rep_type, v_5d);
    }

    auto dnums = mlir::stablehlo::DotDimensionNumbersAttr::get(
        builder.getContext(), /*lhsBatchingDimensions=*/{0, 1},
        /*rhsBatchingDimensions=*/{0, 1}, /*lhsContractingDimensions=*/{3},
        /*rhsContractingDimensions=*/{2});
    mlir::Value new_out = builder.create<mlir::stablehlo::DotGeneralOp>(
        loc, out_4d_type, probs_for_dot, v_for_dot, dnums,
        dot_match->dot_op.getPrecisionConfigAttr(),
        dot_match->dot_op.getAlgorithmAttr());

    if (row_mask_pred_3d && !IsRowAllMaskedProvablyFalse(row_mask_pred_3d)) {
      auto pred_4d_type = mlir::RankedTensorType::get(out_4d_type.getShape(),
                                                      builder.getI1Type());
      mlir::Value pred_4d = builder.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, pred_4d_type, row_mask_pred_3d,
          builder.getDenseI64ArrayAttr({0, 1, 2}));
      mlir::Value zero_out =
          CreateSplatFloatConst(builder, loc, out_4d_type, 0.0f);
      new_out = builder.create<mlir::stablehlo::SelectOp>(loc, pred_4d,
                                                          zero_out, new_out);
    }
    old_out_op->replaceAllUsesWith(mlir::ValueRange{new_out});
  }
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation*> dead_ops;
    func.walk([&](mlir::Operation* op) {
      if (op != func.getOperation() && op->use_empty() &&
          (mlir::isOpTriviallyDead(op) ||
           mlir::isa<mlir::stablehlo::ReduceOp, mlir::chlo::ScanOp>(op))) {
        dead_ops.push_back(op);
      }
    });
    for (mlir::Operation* op : dead_ops) {
      op->erase();
      changed = true;
    }
  }
}

struct CanonicalizeGqaAndMaskedOnlineSoftmaxPass
    : public mlir::PassWrapper<CanonicalizeGqaAndMaskedOnlineSoftmaxPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CanonicalizeGqaAndMaskedOnlineSoftmaxPass)

  llvm::StringRef getArgument() const override {
    return "canonicalize-gqa-and-masked-online-softmax";
  }
  llvm::StringRef getDescription() const override {
    return "Normalizes forward 5D GQA Q@K^T and probs@V contractions to 4D, "
           "folds diagonal-unmasked row-mask reductions, and folds affine "
           "iota packed-sequence scan masks for XLA online-softmax fusion.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    EraseTriviallyDeadOps(func);
    FoldAffineIotaDiffsAndZeroSplatScans(func);
    EraseTriviallyDeadOps(func);
    RewriteForwardGqaAndMaskedOnlineSoftmax(func);
    EraseTriviallyDeadOps(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCanonicalizeGqaAndMaskedOnlineSoftmaxPass() {
  return std::make_unique<CanonicalizeGqaAndMaskedOnlineSoftmaxPass>();
}

}  // namespace torch_tpu
