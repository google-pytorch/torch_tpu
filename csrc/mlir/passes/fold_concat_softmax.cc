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

#include "csrc/mlir/passes/fold_concat_softmax.h"

#include <cstdint>
#include <memory>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // NOLINT
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"      // NOLINT
#include "mlir/IR/BuiltinTypeInterfaces.h"  // NOLINT
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"  // NOLINT
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

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

bool IsDropDimBroadcastDims(llvm::ArrayRef<int64_t> dims, int64_t rank,
                            int64_t dropped_dim) {
  if (static_cast<int64_t>(dims.size()) != rank - 1) {
    return false;
  }
  int64_t idx = 0;
  for (int64_t i = 0; i < rank; ++i) {
    if (i == dropped_dim) {
      continue;
    }
    if (dims[idx++] != i) {
      return false;
    }
  }
  return true;
}

bool IsLosslessFloatWideningOrSame(mlir::Type narrow, mlir::Type wide) {
  if (!mlir::isa<mlir::FloatType>(narrow) ||
      !mlir::isa<mlir::FloatType>(wide)) {
    return false;
  }
  if (narrow == wide) {
    return true;
  }
  if (wide.isF32() && (narrow.isBF16() || narrow.isF16())) {
    return true;
  }
  if (wide.isF64() && (narrow.isF32() || narrow.isBF16() || narrow.isF16())) {
    return true;
  }
  return false;
}

mlir::Value StripOptionalConvert(mlir::Value val) {
  while (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    auto in_type =
        mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
    auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(conv.getType());
    if (!in_type || !out_type ||
        !IsLosslessFloatWideningOrSame(in_type.getElementType(),
                                       out_type.getElementType())) {
      break;
    }
    val = conv.getOperand();
  }
  return val;
}

mlir::Value StripOptionalFloatUpcast(mlir::Value val,
                                     mlir::Type outer_elem_type) {
  auto upcast = val.getDefiningOp<mlir::stablehlo::ConvertOp>();
  if (!upcast) {
    return val;
  }
  auto src_type =
      mlir::dyn_cast<mlir::RankedTensorType>(upcast.getOperand().getType());
  if (src_type && src_type.hasStaticShape() &&
      mlir::isa<mlir::FloatType>(src_type.getElementType()) &&
      src_type.getElementType().getIntOrFloatBitWidth() <=
          outer_elem_type.getIntOrFloatBitWidth()) {
    return upcast.getOperand();
  }
  return val;
}

bool IsSplatFloatConstantValue(mlir::Value val, double target) {
  while (true) {
    if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      val = resh.getOperand();
      continue;
    }
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      val = bcast.getOperand();
      continue;
    }
    if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      val = conv.getOperand();
      continue;
    }
    break;
  }
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto dense_attr =
      mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense_attr || !dense_attr.isSplat()) {
    return false;
  }
  mlir::Type elem_type = dense_attr.getElementType();
  if (mlir::isa<mlir::FloatType>(elem_type)) {
    double v = dense_attr.getSplatValue<llvm::APFloat>().convertToDouble();
    return v == target;
  }
  if (mlir::isa<mlir::IntegerType>(elem_type)) {
    return dense_attr.getSplatValue<llvm::APInt>().getSExtValue() ==
           static_cast<int64_t>(target);
  }
  return false;
}

bool IsNegativeInfinityConstant(mlir::Value val) {
  while (true) {
    if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      val = resh.getOperand();
      continue;
    }
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      val = bcast.getOperand();
      continue;
    }
    if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      val = conv.getOperand();
      continue;
    }
    break;
  }
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto dense_attr =
      mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense_attr || !dense_attr.isSplat()) {
    return false;
  }
  if (!mlir::isa<mlir::FloatType>(dense_attr.getElementType())) {
    return false;
  }
  llvm::APFloat ap_val = dense_attr.getSplatValue<llvm::APFloat>();
  return ap_val.isInfinity() && ap_val.isNegative();
}

bool IsZeroConstantTensor(mlir::Value val) {
  return IsSplatFloatConstantValue(val, 0.0);
}

bool IsOneConstantTensor(mlir::Value val) {
  return IsSplatFloatConstantValue(val, 1.0);
}

bool IsBinaryReduceBody(mlir::Region& region, bool is_max) {
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
  mlir::Operation* body_op = ret_op.getOperands()[0].getDefiningOp();
  if (!body_op || body_op->getNumOperands() != 2) {
    return false;
  }
  const bool op_matches = is_max ? mlir::isa<mlir::stablehlo::MaxOp>(body_op)
                                 : mlir::isa<mlir::stablehlo::AddOp>(body_op);
  if (!op_matches) {
    return false;
  }
  mlir::Value arg0 = block.getArgument(0);
  mlir::Value arg1 = block.getArgument(1);
  mlir::Value lhs = body_op->getOperand(0);
  mlir::Value rhs = body_op->getOperand(1);
  return (lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0);
}

bool IsMaxWithTwoResultsReduceBody(mlir::Region& region) {
  if (region.empty()) {
    return false;
  }
  mlir::Block& block = region.front();
  if (block.getNumArguments() != 4) {
    return false;
  }
  auto ret_op =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  if (!ret_op || ret_op.getOperands().size() != 2) {
    return false;
  }
  mlir::Operation* max_op = ret_op.getOperands()[0].getDefiningOp();
  if (!max_op || !mlir::isa<mlir::stablehlo::MaxOp>(max_op) ||
      max_op->getNumOperands() != 2) {
    return false;
  }
  mlir::Value arg0 = block.getArgument(0);
  mlir::Value arg2 = block.getArgument(2);
  mlir::Value lhs = max_op->getOperand(0);
  mlir::Value rhs = max_op->getOperand(1);
  return (lhs == arg0 && rhs == arg2) || (lhs == arg2 && rhs == arg0);
}

bool MatchSingleDimReduceOp(
    mlir::Value val, int64_t dim, mlir::Value& reduce_input, bool is_max,
    mlir::stablehlo::ReduceOp* out_reduce_op = nullptr) {
  val = StripOptionalConvert(val);
  auto reduce_op = val.getDefiningOp<mlir::stablehlo::ReduceOp>();
  if (!reduce_op) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = reduce_op.getDimensions();
  if (dims.size() != 1 || dims[0] != dim) {
    return false;
  }
  if (reduce_op.getNumResults() == 1 && reduce_op.getInputs().size() == 1 &&
      reduce_op.getInitValues().size() == 1 &&
      IsBinaryReduceBody(reduce_op.getBody(), is_max)) {
    reduce_input = reduce_op.getInputs()[0];
    if (out_reduce_op != nullptr) {
      *out_reduce_op = reduce_op;
    }
    return true;
  }
  if (is_max && reduce_op.getNumResults() == 2 &&
      reduce_op.getInputs().size() == 2 &&
      reduce_op.getInitValues().size() == 2 && val == reduce_op.getResult(0) &&
      IsMaxWithTwoResultsReduceBody(reduce_op.getBody())) {
    reduce_input = reduce_op.getInputs()[0];
    if (out_reduce_op != nullptr) {
      *out_reduce_op = reduce_op;
    }
    return true;
  }
  return false;
}

bool IsBroadcastOrUnsqueezeAlongDim(mlir::RankedTensorType outer_type,
                                    int64_t dim,
                                    mlir::RankedTensorType cur_type) {
  if (!cur_type || !cur_type.hasStaticShape()) {
    return false;
  }
  const int64_t rank = outer_type.getRank();
  if (cur_type.getRank() == rank) {
    for (int64_t i = 0; i < rank; ++i) {
      if (i == dim) {
        if (cur_type.getDimSize(i) != 1 &&
            cur_type.getDimSize(i) != outer_type.getDimSize(i)) {
          return false;
        }
      } else if (cur_type.getDimSize(i) != outer_type.getDimSize(i)) {
        return false;
      }
    }
    return true;
  }
  if (cur_type.getRank() == rank - 1) {
    int64_t idx = 0;
    for (int64_t i = 0; i < rank; ++i) {
      if (i == dim) {
        continue;
      }
      if (cur_type.getDimSize(idx++) != outer_type.getDimSize(i)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

mlir::Value PeelBroadcastAndUnsqueezeAlongDim(mlir::Value val,
                                              mlir::RankedTensorType outer_type,
                                              int64_t dim) {
  const int64_t rank = outer_type.getRank();
  while (true) {
    if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      auto in_type =
          mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
      auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(conv.getType());
      if (!in_type || !out_type ||
          !IsLosslessFloatWideningOrSame(in_type.getElementType(),
                                         out_type.getElementType())) {
        break;
      }
      val = conv.getOperand();
      continue;
    }
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      auto bcast_type = mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
      auto in_type =
          mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
      if (!bcast_type || !in_type ||
          !IsBroadcastOrUnsqueezeAlongDim(outer_type, dim, in_type)) {
        break;
      }
      llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
      if (bcast_type.getRank() == rank && in_type.getRank() == rank - 1 &&
          IsDropDimBroadcastDims(bdims, rank, dim)) {
        val = bcast.getOperand();
        continue;
      }
      if (IsIdentityBroadcastDims(bdims, bcast_type.getRank())) {
        val = bcast.getOperand();
        continue;
      }
      break;
    }
    if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      auto in_type =
          mlir::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
      if (in_type && IsBroadcastOrUnsqueezeAlongDim(outer_type, dim, in_type)) {
        val = resh.getOperand();
        continue;
      }
      break;
    }
    break;
  }
  return val;
}

bool MatchBroadcastedSingleDimReduce(
    mlir::Value val, int64_t dim, mlir::Value& reduce_input, bool is_max,
    mlir::stablehlo::ReduceOp* out_reduce_op = nullptr) {
  auto val_type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!val_type || !val_type.hasStaticShape() || dim < 0 ||
      dim >= val_type.getRank()) {
    return false;
  }
  mlir::Value unwrapped = PeelBroadcastAndUnsqueezeAlongDim(val, val_type, dim);
  return MatchSingleDimReduceOp(unwrapped, dim, reduce_input, is_max,
                                out_reduce_op);
}

bool MatchBroadcastedReciprocalSingleDimSum(
    mlir::Value val, int64_t dim, mlir::Value& reduce_input,
    mlir::stablehlo::ReduceOp* out_reduce_op = nullptr) {
  auto val_type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!val_type || !val_type.hasStaticShape() || dim < 0 ||
      dim >= val_type.getRank()) {
    return false;
  }
  mlir::Value unwrapped = PeelBroadcastAndUnsqueezeAlongDim(val, val_type, dim);
  auto div_op = unwrapped.getDefiningOp<mlir::stablehlo::DivOp>();
  if (!div_op || !IsOneConstantTensor(div_op.getLhs())) {
    return false;
  }
  mlir::Value denom =
      PeelBroadcastAndUnsqueezeAlongDim(div_op.getRhs(), val_type, dim);
  return MatchSingleDimReduceOp(denom, dim, reduce_input, /*is_max=*/false,
                                out_reduce_op);
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation*> dead_ops;
    func.walk([&dead_ops](mlir::Operation* op) {
      if (op->use_empty() && (mlir::isOpTriviallyDead(op) ||
                              mlir::isa<mlir::stablehlo::ReduceOp>(op))) {
        dead_ops.push_back(op);
      }
    });
    for (auto it = dead_ops.rbegin(); it != dead_ops.rend(); ++it) {
      if ((*it)->use_empty()) {
        (*it)->erase();
        changed = true;
      }
    }
  }
}

mlir::Value ConvertIfNeeded(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value val, mlir::RankedTensorType dst_type) {
  if (val.getType() == dst_type) {
    return val;
  }
  return builder.create<mlir::stablehlo::ConvertOp>(loc, dst_type, val);
}

void BuildReducedShapeAndAxes(mlir::RankedTensorType type, int64_t dim,
                              llvm::SmallVectorImpl<int64_t>& reduced_shape,
                              llvm::SmallVectorImpl<int64_t>& non_dim_axes) {
  const int64_t rank = type.getRank();
  reduced_shape.reserve(rank - 1);
  non_dim_axes.reserve(rank - 1);
  for (int64_t i = 0; i < rank; ++i) {
    if (i != dim) {
      reduced_shape.push_back(type.getDimSize(i));
      non_dim_axes.push_back(i);
    }
  }
}

template <typename BinaryOp>
mlir::Value CreateSingleDimReduce(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::RankedTensorType reduced_type,
                                  mlir::Value input, mlir::Value init,
                                  int64_t dim) {
  auto reduce_op = builder.create<mlir::stablehlo::ReduceOp>(
      loc, mlir::TypeRange{reduced_type}, mlir::ValueRange{input},
      mlir::ValueRange{init}, builder.getDenseI64ArrayAttr({dim}));
  mlir::stablehlo::buildReduceBody<BinaryOp>(reduced_type.getElementType(),
                                             reduce_op.getBody(), builder);
  return reduce_op.getResult(0);
}

template <typename BinaryOp>
mlir::Value CreateSingleDimReduceOrReshape(mlir::OpBuilder& builder,
                                           mlir::Location loc,
                                           mlir::RankedTensorType reduced_type,
                                           mlir::Value input,
                                           mlir::RankedTensorType input_type,
                                           mlir::Value init, int64_t dim) {
  if (input_type.getDimSize(dim) == 1) {
    return builder.create<mlir::stablehlo::ReshapeOp>(loc, reduced_type, input);
  }
  return CreateSingleDimReduce<BinaryOp>(builder, loc, reduced_type, input,
                                         init, dim);
}

bool TryFoldShiftInvariantPreSubtractionAlongDim(
    mlir::stablehlo::SubtractOp shifted_outer,
    mlir::RankedTensorType outer_type, int64_t dim) {
  mlir::Value x = shifted_outer.getLhs();
  mlir::Value reduce2_in;
  mlir::stablehlo::ReduceOp reduce2_op;
  if (!MatchBroadcastedSingleDimReduce(shifted_outer.getRhs(), dim, reduce2_in,
                                       /*is_max=*/true, &reduce2_op) ||
      reduce2_in != x) {
    return false;
  }
  if (reduce2_op.getInitValues().size() != 1 ||
      !IsNegativeInfinityConstant(reduce2_op.getInitValues()[0])) {
    return false;
  }

  mlir::Type outer_elem_type = outer_type.getElementType();
  mlir::Value inner_val = StripOptionalFloatUpcast(x, outer_elem_type);
  auto inner_sub = inner_val.getDefiningOp<mlir::stablehlo::SubtractOp>();
  if (!inner_sub) {
    return false;
  }
  mlir::Value base = inner_sub.getLhs();
  auto base_type = mlir::dyn_cast<mlir::RankedTensorType>(base.getType());
  if (!base_type || !mlir::isa<mlir::FloatType>(base_type.getElementType())) {
    return false;
  }

  mlir::Value reduce1_in;
  mlir::stablehlo::ReduceOp reduce1_op;
  if (!MatchBroadcastedSingleDimReduce(inner_sub.getRhs(), dim, reduce1_in,
                                       /*is_max=*/true, &reduce1_op) ||
      reduce1_in != base) {
    return false;
  }
  if (reduce1_op.getInitValues().empty() ||
      !IsNegativeInfinityConstant(reduce1_op.getInitValues()[0])) {
    return false;
  }

  mlir::OpBuilder builder(shifted_outer);
  mlir::Location loc = shifted_outer.getLoc();
  mlir::Value base_in_outer = ConvertIfNeeded(builder, loc, base, outer_type);

  llvm::SmallVector<int64_t> reduced_shape;
  llvm::SmallVector<int64_t> non_dim_axes;
  BuildReducedShapeAndAxes(outer_type, dim, reduced_shape, non_dim_axes);
  auto reduced_type =
      mlir::RankedTensorType::get(reduced_shape, outer_elem_type);
  mlir::Value max_init =
      ConvertIfNeeded(builder, loc, reduce2_op.getInitValues()[0],
                      mlir::RankedTensorType::get({}, outer_elem_type));

  mlir::Value new_max = CreateSingleDimReduce<mlir::stablehlo::MaxOp>(
      builder, loc, reduced_type, base_in_outer, max_init, dim);
  mlir::Value new_bcast_max = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, outer_type, new_max, builder.getDenseI64ArrayAttr(non_dim_axes));
  mlir::Value new_shifted = builder.create<mlir::stablehlo::SubtractOp>(
      loc, outer_type, base_in_outer, new_bcast_max);
  shifted_outer.replaceAllUsesWith(new_shifted);
  return true;
}

// Eliminates redundant shift-invariant pre-subtractions before softmax.
// When an attention pattern computes `x - max(x)` before calling softmax,
// the pre-subtraction is mathematically redundant with softmax's own max
// reduction (`softmax(x - c) == softmax(x)`), so we fold the shift into
// the downstream softmax and drop the redundant reduction.
void FoldShiftInvariantPreSubtraction(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::SubtractOp> sub_ops;
  func.walk(
      [&sub_ops](mlir::stablehlo::SubtractOp op) { sub_ops.push_back(op); });

  for (mlir::stablehlo::SubtractOp shifted_outer : sub_ops) {
    auto outer_type =
        mlir::dyn_cast<mlir::RankedTensorType>(shifted_outer.getType());
    if (!outer_type || !outer_type.hasStaticShape() ||
        !mlir::isa<mlir::FloatType>(outer_type.getElementType())) {
      continue;
    }
    for (int64_t dim = 0; dim < outer_type.getRank(); ++dim) {
      if (TryFoldShiftInvariantPreSubtractionAlongDim(shifted_outer, outer_type,
                                                      dim)) {
        break;
      }
    }
  }
  EraseTriviallyDeadOps(func);
}

bool IsFullUnitStrideSlice(llvm::ArrayRef<int64_t> start,
                           llvm::ArrayRef<int64_t> limit,
                           llvm::ArrayRef<int64_t> strides,
                           mlir::RankedTensorType part_type) {
  for (int64_t i = 0; i < part_type.getRank(); ++i) {
    if (start[i] != 0 || limit[i] != part_type.getDimSize(i) ||
        strides[i] != 1) {
      return false;
    }
  }
  return true;
}

void ReplaceSliceWithPart(mlir::stablehlo::SliceOp slice, mlir::Value part_val,
                          llvm::ArrayRef<int64_t> start,
                          llvm::ArrayRef<int64_t> limit) {
  auto part_type = mlir::cast<mlir::RankedTensorType>(part_val.getType());
  if (IsFullUnitStrideSlice(start, limit, slice.getStrides(), part_type)) {
    slice.replaceAllUsesWith(part_val);
    return;
  }
  mlir::OpBuilder builder(slice);
  auto new_slice = builder.create<mlir::stablehlo::SliceOp>(
      slice.getLoc(), part_val, builder.getDenseI64ArrayAttr(start),
      builder.getDenseI64ArrayAttr(limit), slice.getStridesAttr());
  slice.replaceAllUsesWith(new_slice.getResult());
}

void SimplifyConcatSliceUsers(mlir::stablehlo::ConcatenateOp concat_op,
                              mlir::Value lhs_val, mlir::Value rhs_val,
                              int64_t dim) {
  auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs_val.getType());
  const int64_t lhs_dim_size = lhs_type.getDimSize(dim);

  llvm::SmallVector<mlir::Operation*> users(concat_op->user_begin(),
                                            concat_op->user_end());
  for (mlir::Operation* user : users) {
    auto slice = mlir::dyn_cast<mlir::stablehlo::SliceOp>(user);
    if (!slice) {
      continue;
    }
    llvm::ArrayRef<int64_t> start = slice.getStartIndices();
    llvm::ArrayRef<int64_t> limit = slice.getLimitIndices();
    if (limit[dim] <= lhs_dim_size) {
      ReplaceSliceWithPart(slice, lhs_val, start, limit);
    } else if (start[dim] >= lhs_dim_size) {
      llvm::SmallVector<int64_t> new_start(start.begin(), start.end());
      llvm::SmallVector<int64_t> new_limit(limit.begin(), limit.end());
      new_start[dim] -= lhs_dim_size;
      new_limit[dim] -= lhs_dim_size;
      ReplaceSliceWithPart(slice, rhs_val, new_start, new_limit);
    }
  }
}

bool MatchConcatInputAlongDim(mlir::Value raw_in, int64_t dim, mlir::Value& lhs,
                              mlir::Value& rhs,
                              mlir::RankedTensorType& lhs_type,
                              mlir::RankedTensorType& rhs_type) {
  auto concat_op = raw_in.getDefiningOp<mlir::stablehlo::ConcatenateOp>();
  if (!concat_op || concat_op.getInputs().size() != 2 ||
      static_cast<int64_t>(concat_op.getDimension()) != dim) {
    return false;
  }
  lhs = concat_op.getInputs()[0];
  rhs = concat_op.getInputs()[1];
  lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  rhs_type = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  if (!lhs_type || !rhs_type || !lhs_type.hasStaticShape() ||
      !rhs_type.hasStaticShape()) {
    return false;
  }
  return lhs_type.getDimSize(dim) >= 1 && rhs_type.getDimSize(dim) >= 1;
}

mlir::stablehlo::ConvertOp GetSingleRankedConvertUser(mlir::Operation* op) {
  if (!op->hasOneUse()) {
    return nullptr;
  }
  auto convert_out =
      mlir::dyn_cast<mlir::stablehlo::ConvertOp>(*op->user_begin());
  if (convert_out && mlir::isa<mlir::RankedTensorType>(convert_out.getType())) {
    return convert_out;
  }
  return nullptr;
}

bool IsConfinedSlice(mlir::Operation* op, int64_t dim, int64_t lhs_dim_size) {
  auto slice = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op);
  if (!slice) {
    return false;
  }
  llvm::ArrayRef<int64_t> start = slice.getStartIndices();
  llvm::ArrayRef<int64_t> limit = slice.getLimitIndices();
  if (dim < 0 || dim >= static_cast<int64_t>(start.size()) ||
      dim >= static_cast<int64_t>(limit.size())) {
    return false;
  }
  return limit[dim] <= lhs_dim_size || start[dim] >= lhs_dim_size;
}

bool MatchSoftmaxBackwardDataPattern(
    mlir::Value val, int64_t dim, mlir::Value& probs_val, mlir::Value& grad_val,
    mlir::stablehlo::DotGeneralOp& dot_op,
    mlir::stablehlo::SubtractOp* out_sub_op = nullptr);

bool AreTargetUsersConfinedOrBackwardConfined(mlir::Operation* target_op,
                                              int64_t dim, int64_t lhs_dim_size,
                                              bool allow_backward_pattern) {
  if (target_op->use_empty()) {
    return false;
  }
  for (mlir::Operation* user : target_op->getUsers()) {
    if (IsConfinedSlice(user, dim, lhs_dim_size)) {
      continue;
    }
    if (allow_backward_pattern) {
      mlir::Value bwd_probs;
      mlir::Value bwd_grad;
      mlir::stablehlo::DotGeneralOp bwd_dot;
      mlir::stablehlo::MulOp bwd_mul;
      if (auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(user)) {
        if (MatchSoftmaxBackwardDataPattern(mul.getResult(), dim, bwd_probs,
                                            bwd_grad, bwd_dot) &&
            bwd_probs == target_op->getResult(0)) {
          bwd_mul = mul;
        }
      } else if (auto dot =
                     mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(user)) {
        for (mlir::Operation* dot_user : dot->getUsers()) {
          mlir::Operation* bcast_op = dot_user;
          if (auto conv =
                  mlir::dyn_cast<mlir::stablehlo::ConvertOp>(dot_user)) {
            if (conv->hasOneUse()) {
              bcast_op = *conv->user_begin();
            }
          }
          auto bcast =
              mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(bcast_op);
          if (!bcast) {
            continue;
          }
          for (mlir::Operation* bcast_user : bcast->getUsers()) {
            auto sub = mlir::dyn_cast<mlir::stablehlo::SubtractOp>(bcast_user);
            if (!sub) {
              continue;
            }
            for (mlir::Operation* sub_user : sub->getUsers()) {
              auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(sub_user);
              if (mul &&
                  MatchSoftmaxBackwardDataPattern(
                      mul.getResult(), dim, bwd_probs, bwd_grad, bwd_dot) &&
                  bwd_probs == target_op->getResult(0)) {
                bwd_mul = mul;
                break;
              }
            }
          }
        }
      }
      if (bwd_mul) {
        mlir::Operation* bwd_target = bwd_mul.getOperation();
        if (auto conv = GetSingleRankedConvertUser(bwd_mul.getOperation())) {
          bwd_target = conv.getOperation();
        }
        if (!bwd_target->use_empty()) {
          bool all_bwd_users_confined = true;
          for (mlir::Operation* bwd_user : bwd_target->getUsers()) {
            if (!IsConfinedSlice(bwd_user, dim, lhs_dim_size)) {
              all_bwd_users_confined = false;
              break;
            }
          }
          if (all_bwd_users_confined) {
            continue;
          }
        }
      }
    }
    return false;
  }
  return true;
}

bool HasOnlyExpectedConcatUsers(mlir::Value concat_val,
                                llvm::ArrayRef<mlir::Operation*> expected_ops) {
  auto concat_op = concat_val.getDefiningOp<mlir::stablehlo::ConcatenateOp>();
  if (!concat_op) {
    return false;
  }
  llvm::DenseSet<mlir::Operation*> expected_set(expected_ops.begin(),
                                                expected_ops.end());
  for (mlir::Operation* user : concat_op->getUsers()) {
    if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(user)) {
      for (mlir::Operation* conv_user : conv->getUsers()) {
        if (!expected_set.contains(conv_user)) {
          return false;
        }
      }
      continue;
    }
    if (!expected_set.contains(user)) {
      return false;
    }
  }
  return true;
}

mlir::Value CreateScalarFloatConst(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Type elem_type, double val) {
  auto float_type = mlir::cast<mlir::FloatType>(elem_type);
  llvm::APFloat ap_val(val);
  bool loses_info = false;
  ap_val.convert(float_type.getFloatSemantics(),
                 llvm::APFloat::rmNearestTiesToEven, &loses_info);
  auto scalar_type = mlir::RankedTensorType::get({}, float_type);
  auto attr = mlir::DenseElementsAttr::get(scalar_type, ap_val);
  return builder.create<mlir::stablehlo::ConstantOp>(loc, attr);
}

void EmitPartitionedConcatSoftmax(mlir::Operation* norm_op,
                                  bool use_reciprocal_mul,
                                  mlir::Type comp_elem_type, int64_t dim,
                                  mlir::Value lhs, mlir::Value rhs,
                                  mlir::RankedTensorType lhs_type,
                                  mlir::RankedTensorType rhs_type,
                                  mlir::stablehlo::ReduceOp max_reduce_op,
                                  mlir::stablehlo::ReduceOp sum_reduce_op) {
  mlir::stablehlo::ConvertOp convert_out = GetSingleRankedConvertUser(norm_op);
  mlir::Operation* target_op =
      convert_out ? convert_out.getOperation() : norm_op;
  mlir::OpBuilder builder(target_op);
  mlir::Location loc = target_op->getLoc();

  auto lhs_comp_type =
      mlir::RankedTensorType::get(lhs_type.getShape(), comp_elem_type);
  auto rhs_comp_type =
      mlir::RankedTensorType::get(rhs_type.getShape(), comp_elem_type);
  mlir::Value lhs_comp = ConvertIfNeeded(builder, loc, lhs, lhs_comp_type);
  mlir::Value rhs_comp = ConvertIfNeeded(builder, loc, rhs, rhs_comp_type);

  llvm::SmallVector<int64_t> reduced_shape;
  llvm::SmallVector<int64_t> non_dim_axes;
  BuildReducedShapeAndAxes(lhs_type, dim, reduced_shape, non_dim_axes);
  auto reduced_type =
      mlir::RankedTensorType::get(reduced_shape, comp_elem_type);
  auto scalar_comp_type = mlir::RankedTensorType::get({}, comp_elem_type);

  mlir::Value max_init = ConvertIfNeeded(
      builder, loc, max_reduce_op.getInitValues()[0], scalar_comp_type);
  mlir::Value sum_init = ConvertIfNeeded(
      builder, loc, sum_reduce_op.getInitValues()[0], scalar_comp_type);

  mlir::Value max_lhs = CreateSingleDimReduce<mlir::stablehlo::MaxOp>(
      builder, loc, reduced_type, lhs_comp, max_init, dim);
  mlir::Value max_rhs = CreateSingleDimReduceOrReshape<mlir::stablehlo::MaxOp>(
      builder, loc, reduced_type, rhs_comp, rhs_type, max_init, dim);
  mlir::Value max_total =
      builder.create<mlir::stablehlo::MaxOp>(loc, max_lhs, max_rhs);
  mlir::DenseI64ArrayAttr non_dim_attr =
      builder.getDenseI64ArrayAttr(non_dim_axes);

  mlir::Value bcast_max_lhs = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, lhs_comp_type, max_total, non_dim_attr);
  mlir::Value bcast_max_rhs = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, rhs_comp_type, max_total, non_dim_attr);

  mlir::Value exp_lhs = builder.create<mlir::stablehlo::ExpOp>(
      loc, builder.create<mlir::stablehlo::SubtractOp>(loc, lhs_comp,
                                                       bcast_max_lhs));
  mlir::Value exp_rhs = builder.create<mlir::stablehlo::ExpOp>(
      loc, builder.create<mlir::stablehlo::SubtractOp>(loc, rhs_comp,
                                                       bcast_max_rhs));

  mlir::Value sum_lhs = CreateSingleDimReduce<mlir::stablehlo::AddOp>(
      builder, loc, reduced_type, exp_lhs, sum_init, dim);
  mlir::Value sum_rhs = CreateSingleDimReduceOrReshape<mlir::stablehlo::AddOp>(
      builder, loc, reduced_type, exp_rhs, rhs_type, sum_init, dim);
  mlir::Value sum_total =
      builder.create<mlir::stablehlo::AddOp>(loc, sum_lhs, sum_rhs);

  mlir::Value final_lhs;
  mlir::Value final_rhs;
  if (use_reciprocal_mul) {
    mlir::Value one_scalar =
        CreateScalarFloatConst(builder, loc, comp_elem_type, 1.0);
    mlir::Value one_reduced = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, reduced_type, one_scalar, builder.getDenseI64ArrayAttr({}));
    mlir::Value inv_sum_total =
        builder.create<mlir::stablehlo::DivOp>(loc, one_reduced, sum_total);
    mlir::Value bcast_inv_lhs =
        builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, lhs_comp_type, inv_sum_total, non_dim_attr);
    mlir::Value bcast_inv_rhs =
        builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, rhs_comp_type, inv_sum_total, non_dim_attr);
    final_lhs =
        builder.create<mlir::stablehlo::MulOp>(loc, exp_lhs, bcast_inv_lhs);
    final_rhs =
        builder.create<mlir::stablehlo::MulOp>(loc, exp_rhs, bcast_inv_rhs);
  } else {
    mlir::Value bcast_sum_lhs =
        builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, lhs_comp_type, sum_total, non_dim_attr);
    mlir::Value bcast_sum_rhs =
        builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, rhs_comp_type, sum_total, non_dim_attr);
    final_lhs =
        builder.create<mlir::stablehlo::DivOp>(loc, exp_lhs, bcast_sum_lhs);
    final_rhs =
        builder.create<mlir::stablehlo::DivOp>(loc, exp_rhs, bcast_sum_rhs);
  }

  if (convert_out) {
    mlir::Type res_elem_type =
        mlir::cast<mlir::RankedTensorType>(convert_out.getType())
            .getElementType();
    final_lhs = builder.create<mlir::stablehlo::ConvertOp>(
        loc, mlir::RankedTensorType::get(lhs_type.getShape(), res_elem_type),
        final_lhs);
    final_rhs = builder.create<mlir::stablehlo::ConvertOp>(
        loc, mlir::RankedTensorType::get(rhs_type.getShape(), res_elem_type),
        final_rhs);
  }

  auto new_concat = builder.create<mlir::stablehlo::ConcatenateOp>(
      loc, mlir::ValueRange{final_lhs, final_rhs}, dim);
  target_op->replaceAllUsesWith(new_concat.getOperation());
  SimplifyConcatSliceUsers(new_concat, final_lhs, final_rhs, dim);
}

bool TryPartitionReciprocalConcatSoftmaxAlongDim(mlir::stablehlo::MulOp mul_op,
                                                 mlir::stablehlo::ExpOp exp_op,
                                                 mlir::Value inv_bcast_val,
                                                 mlir::Type comp_elem_type,
                                                 int64_t dim) {
  auto shifted_op =
      exp_op.getOperand().getDefiningOp<mlir::stablehlo::SubtractOp>();
  if (!shifted_op) {
    return false;
  }
  mlir::Value sum_in;
  mlir::stablehlo::ReduceOp sum_reduce_op;
  if (!MatchBroadcastedReciprocalSingleDimSum(inv_bcast_val, dim, sum_in,
                                              &sum_reduce_op) ||
      sum_in != exp_op.getResult()) {
    return false;
  }
  if (sum_reduce_op.getInitValues().size() != 1 ||
      !IsZeroConstantTensor(sum_reduce_op.getInitValues()[0])) {
    return false;
  }

  mlir::Value in_val = shifted_op.getLhs();
  mlir::Value max_in;
  mlir::stablehlo::ReduceOp max_reduce_op;
  if (!MatchBroadcastedSingleDimReduce(shifted_op.getRhs(), dim, max_in,
                                       /*is_max=*/true, &max_reduce_op) ||
      max_in != in_val) {
    return false;
  }
  if (max_reduce_op.getInitValues().empty() ||
      !IsNegativeInfinityConstant(max_reduce_op.getInitValues()[0])) {
    return false;
  }

  mlir::Value raw_in = StripOptionalFloatUpcast(in_val, comp_elem_type);
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::RankedTensorType lhs_type;
  mlir::RankedTensorType rhs_type;
  if (!MatchConcatInputAlongDim(raw_in, dim, lhs, rhs, lhs_type, rhs_type)) {
    return false;
  }

  mlir::stablehlo::ConvertOp convert_out =
      GetSingleRankedConvertUser(mul_op.getOperation());
  mlir::Operation* target_op =
      convert_out ? convert_out.getOperation() : mul_op.getOperation();
  if (!AreTargetUsersConfinedOrBackwardConfined(
          target_op, dim, lhs_type.getDimSize(dim),
          /*allow_backward_pattern=*/true)) {
    return false;
  }
  if (!HasOnlyExpectedConcatUsers(
          raw_in, {max_reduce_op.getOperation(), shifted_op.getOperation()})) {
    return false;
  }

  EmitPartitionedConcatSoftmax(mul_op.getOperation(),
                               /*use_reciprocal_mul=*/true, comp_elem_type, dim,
                               lhs, rhs, lhs_type, rhs_type, max_reduce_op,
                               sum_reduce_op);
  return true;
}

void PartitionReciprocalConcatSoftmax(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::MulOp> mul_ops;
  func.walk([&mul_ops](mlir::stablehlo::MulOp op) { mul_ops.push_back(op); });

  for (mlir::stablehlo::MulOp mul_op : mul_ops) {
    if (!mul_op || mul_op->use_empty()) {
      continue;
    }
    auto mul_type = mlir::dyn_cast<mlir::RankedTensorType>(mul_op.getType());
    if (!mul_type || !mul_type.hasStaticShape() || mul_type.getRank() < 1 ||
        !mlir::isa<mlir::FloatType>(mul_type.getElementType())) {
      continue;
    }
    auto try_operands = [&mul_op, &mul_type](mlir::Value cand_exp,
                                             mlir::Value cand_inv) -> bool {
      auto exp_op = cand_exp.getDefiningOp<mlir::stablehlo::ExpOp>();
      if (!exp_op) {
        return false;
      }
      for (int64_t dim = 0; dim < mul_type.getRank(); ++dim) {
        if (TryPartitionReciprocalConcatSoftmaxAlongDim(
                mul_op, exp_op, cand_inv, mul_type.getElementType(), dim)) {
          return true;
        }
      }
      return false;
    };
    if (!try_operands(mul_op.getLhs(), mul_op.getRhs())) {
      try_operands(mul_op.getRhs(), mul_op.getLhs());
    }
  }
  EraseTriviallyDeadOps(func);
}

bool TryPartitionDivConcatSoftmaxAlongDim(mlir::stablehlo::DivOp div_op,
                                          mlir::stablehlo::ExpOp exp_op,
                                          mlir::Type comp_elem_type,
                                          int64_t dim) {
  auto shifted_op =
      exp_op.getOperand().getDefiningOp<mlir::stablehlo::SubtractOp>();
  if (!shifted_op) {
    return false;
  }
  mlir::Value sum_in;
  mlir::stablehlo::ReduceOp sum_reduce_op;
  if (!MatchBroadcastedSingleDimReduce(div_op.getRhs(), dim, sum_in,
                                       /*is_max=*/false, &sum_reduce_op) ||
      sum_in != exp_op.getResult()) {
    return false;
  }
  if (sum_reduce_op.getInitValues().size() != 1 ||
      !IsZeroConstantTensor(sum_reduce_op.getInitValues()[0])) {
    return false;
  }

  mlir::Value in_val = shifted_op.getLhs();
  mlir::Value max_in;
  mlir::stablehlo::ReduceOp max_reduce_op;
  if (!MatchBroadcastedSingleDimReduce(shifted_op.getRhs(), dim, max_in,
                                       /*is_max=*/true, &max_reduce_op) ||
      max_in != in_val) {
    return false;
  }
  if (max_reduce_op.getInitValues().empty() ||
      !IsNegativeInfinityConstant(max_reduce_op.getInitValues()[0])) {
    return false;
  }

  mlir::Value raw_in = StripOptionalFloatUpcast(in_val, comp_elem_type);
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::RankedTensorType lhs_type;
  mlir::RankedTensorType rhs_type;
  if (!MatchConcatInputAlongDim(raw_in, dim, lhs, rhs, lhs_type, rhs_type)) {
    return false;
  }

  mlir::stablehlo::ConvertOp convert_out =
      GetSingleRankedConvertUser(div_op.getOperation());
  mlir::Operation* target_op =
      convert_out ? convert_out.getOperation() : div_op.getOperation();
  if (!AreTargetUsersConfinedOrBackwardConfined(
          target_op, dim, lhs_type.getDimSize(dim),
          /*allow_backward_pattern=*/true)) {
    return false;
  }
  if (!HasOnlyExpectedConcatUsers(
          raw_in, {max_reduce_op.getOperation(), shifted_op.getOperation()})) {
    return false;
  }

  EmitPartitionedConcatSoftmax(div_op.getOperation(),
                               /*use_reciprocal_mul=*/false, comp_elem_type,
                               dim, lhs, rhs, lhs_type, rhs_type, max_reduce_op,
                               sum_reduce_op);
  return true;
}

void PartitionDivConcatSoftmax(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::DivOp> div_ops;
  func.walk([&div_ops](mlir::stablehlo::DivOp op) { div_ops.push_back(op); });

  for (mlir::stablehlo::DivOp div_op : div_ops) {
    if (!div_op || div_op->use_empty()) {
      continue;
    }
    auto div_type = mlir::dyn_cast<mlir::RankedTensorType>(div_op.getType());
    if (!div_type || !div_type.hasStaticShape() || div_type.getRank() < 1 ||
        !mlir::isa<mlir::FloatType>(div_type.getElementType())) {
      continue;
    }
    auto exp_op = div_op.getLhs().getDefiningOp<mlir::stablehlo::ExpOp>();
    if (!exp_op) {
      continue;
    }
    for (int64_t dim = 0; dim < div_type.getRank(); ++dim) {
      if (TryPartitionDivConcatSoftmaxAlongDim(
              div_op, exp_op, div_type.getElementType(), dim)) {
        break;
      }
    }
  }
  EraseTriviallyDeadOps(func);
}

bool MatchSoftmaxBackwardDataPattern(mlir::Value val, int64_t dim,
                                     mlir::Value& probs_val,
                                     mlir::Value& grad_val,
                                     mlir::stablehlo::DotGeneralOp& dot_op,
                                     mlir::stablehlo::SubtractOp* out_sub_op) {
  mlir::Value unwrapped = StripOptionalConvert(val);
  auto mul_op = unwrapped.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul_op) {
    return false;
  }
  auto mul_type = mlir::dyn_cast<mlir::RankedTensorType>(mul_op.getType());
  if (!mul_type || !mul_type.hasStaticShape() || dim < 0 ||
      dim >= mul_type.getRank()) {
    return false;
  }
  const int64_t rank = mul_type.getRank();

  auto try_match = [&mul_type, dim, rank, &probs_val, &grad_val, &dot_op,
                    out_sub_op](mlir::Value cand_probs,
                                mlir::Value cand_sub) -> bool {
    auto sub_op = cand_sub.getDefiningOp<mlir::stablehlo::SubtractOp>();
    if (!sub_op) {
      return false;
    }
    mlir::Value cand_grad = sub_op.getLhs();
    mlir::Value bcast_val =
        PeelBroadcastAndUnsqueezeAlongDim(sub_op.getRhs(), mul_type, dim);
    if (auto conv = bcast_val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      auto in_type =
          mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
      auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(conv.getType());
      if (in_type && out_type &&
          mlir::isa<mlir::FloatType>(in_type.getElementType()) &&
          mlir::isa<mlir::FloatType>(out_type.getElementType())) {
        bcast_val = conv.getOperand();
      }
    }
    auto dg = bcast_val.getDefiningOp<mlir::stablehlo::DotGeneralOp>();
    if (!dg) {
      return false;
    }
    const auto& dnums = dg.getDotDimensionNumbers();
    llvm::ArrayRef<int64_t> lhs_c = dnums.getLhsContractingDimensions();
    llvm::ArrayRef<int64_t> rhs_c = dnums.getRhsContractingDimensions();
    llvm::ArrayRef<int64_t> lhs_b = dnums.getLhsBatchingDimensions();
    llvm::ArrayRef<int64_t> rhs_b = dnums.getRhsBatchingDimensions();
    if (lhs_c.size() != 1 || rhs_c.size() != 1 || lhs_c[0] != dim ||
        rhs_c[0] != dim || static_cast<int64_t>(lhs_b.size()) != rank - 1 ||
        lhs_b != rhs_b || !IsDropDimBroadcastDims(lhs_b, rank, dim)) {
      return false;
    }
    const bool operands_match =
        (dg.getLhs() == cand_probs && dg.getRhs() == cand_grad) ||
        (dg.getLhs() == cand_grad && dg.getRhs() == cand_probs);
    if (!operands_match) {
      return false;
    }
    probs_val = cand_probs;
    grad_val = cand_grad;
    dot_op = dg;
    if (out_sub_op != nullptr) {
      *out_sub_op = sub_op;
    }
    return true;
  };

  return try_match(mul_op.getLhs(), mul_op.getRhs()) ||
         try_match(mul_op.getRhs(), mul_op.getLhs());
}

bool TryPartitionConcatSoftmaxBackwardOp(mlir::stablehlo::MulOp mul_op) {
  auto mul_type = mlir::dyn_cast<mlir::RankedTensorType>(mul_op.getType());
  if (!mul_type || !mul_type.hasStaticShape() || mul_type.getRank() < 1 ||
      !mlir::isa<mlir::FloatType>(mul_type.getElementType())) {
    return false;
  }
  mlir::Type comp_elem_type = mul_type.getElementType();

  for (int64_t dim = 0; dim < mul_type.getRank(); ++dim) {
    mlir::Value probs_val;
    mlir::Value grad_val;
    mlir::stablehlo::DotGeneralOp dot_op;
    mlir::stablehlo::SubtractOp sub_op;
    if (!MatchSoftmaxBackwardDataPattern(mul_op.getResult(), dim, probs_val,
                                         grad_val, dot_op, &sub_op)) {
      continue;
    }
    mlir::Value raw_probs = StripOptionalFloatUpcast(probs_val, comp_elem_type);
    mlir::Value raw_grad = StripOptionalFloatUpcast(grad_val, comp_elem_type);
    mlir::Value probs_lhs;
    mlir::Value probs_rhs;
    mlir::RankedTensorType lhs_type;
    mlir::RankedTensorType rhs_type;
    mlir::Value grad_lhs;
    mlir::Value grad_rhs;
    mlir::RankedTensorType grad_lhs_type;
    mlir::RankedTensorType grad_rhs_type;
    if (!MatchConcatInputAlongDim(raw_probs, dim, probs_lhs, probs_rhs,
                                  lhs_type, rhs_type) ||
        !MatchConcatInputAlongDim(raw_grad, dim, grad_lhs, grad_rhs,
                                  grad_lhs_type, grad_rhs_type) ||
        lhs_type.getShape() != grad_lhs_type.getShape() ||
        rhs_type.getShape() != grad_rhs_type.getShape()) {
      continue;
    }

    mlir::stablehlo::ConvertOp convert_out =
        GetSingleRankedConvertUser(mul_op.getOperation());
    mlir::Operation* target_op =
        convert_out ? convert_out.getOperation() : mul_op.getOperation();
    if (!AreTargetUsersConfinedOrBackwardConfined(
            target_op, dim, lhs_type.getDimSize(dim),
            /*allow_backward_pattern=*/false)) {
      continue;
    }
    if (!HasOnlyExpectedConcatUsers(
            raw_probs, {dot_op.getOperation(), mul_op.getOperation()}) ||
        !HasOnlyExpectedConcatUsers(
            raw_grad, {dot_op.getOperation(), sub_op.getOperation()})) {
      continue;
    }

    mlir::OpBuilder builder(target_op);
    mlir::Location loc = target_op->getLoc();

    auto lhs_comp_type =
        mlir::RankedTensorType::get(lhs_type.getShape(), comp_elem_type);
    auto rhs_comp_type =
        mlir::RankedTensorType::get(rhs_type.getShape(), comp_elem_type);
    mlir::Value probs_lhs_comp =
        ConvertIfNeeded(builder, loc, probs_lhs, lhs_comp_type);
    mlir::Value probs_rhs_comp =
        ConvertIfNeeded(builder, loc, probs_rhs, rhs_comp_type);
    mlir::Value grad_lhs_comp =
        ConvertIfNeeded(builder, loc, grad_lhs, lhs_comp_type);
    mlir::Value grad_rhs_comp =
        ConvertIfNeeded(builder, loc, grad_rhs, rhs_comp_type);

    llvm::SmallVector<int64_t> reduced_shape;
    llvm::SmallVector<int64_t> non_dim_axes;
    BuildReducedShapeAndAxes(lhs_type, dim, reduced_shape, non_dim_axes);

    auto dot_res_type =
        mlir::dyn_cast<mlir::RankedTensorType>(dot_op.getType());
    if (!dot_res_type) {
      continue;
    }
    mlir::Type dot_elem_type = dot_res_type.getElementType();
    auto dot_reduced_type =
        mlir::RankedTensorType::get(reduced_shape, dot_elem_type);

    mlir::Value dot_total;
    if (IsZeroConstantTensor(grad_lhs)) {
      dot_total = builder.create<mlir::stablehlo::DotGeneralOp>(
          loc, dot_reduced_type, probs_rhs_comp, grad_rhs_comp,
          dot_op.getDotDimensionNumbersAttr(), dot_op.getPrecisionConfigAttr(),
          dot_op.getAlgorithmAttr());
    } else if (IsZeroConstantTensor(grad_rhs)) {
      dot_total = builder.create<mlir::stablehlo::DotGeneralOp>(
          loc, dot_reduced_type, probs_lhs_comp, grad_lhs_comp,
          dot_op.getDotDimensionNumbersAttr(), dot_op.getPrecisionConfigAttr(),
          dot_op.getAlgorithmAttr());
    } else {
      mlir::Value dot_lhs = builder.create<mlir::stablehlo::DotGeneralOp>(
          loc, dot_reduced_type, probs_lhs_comp, grad_lhs_comp,
          dot_op.getDotDimensionNumbersAttr(), dot_op.getPrecisionConfigAttr(),
          dot_op.getAlgorithmAttr());
      mlir::Value dot_rhs = builder.create<mlir::stablehlo::DotGeneralOp>(
          loc, dot_reduced_type, probs_rhs_comp, grad_rhs_comp,
          dot_op.getDotDimensionNumbersAttr(), dot_op.getPrecisionConfigAttr(),
          dot_op.getAlgorithmAttr());
      dot_total = builder.create<mlir::stablehlo::AddOp>(loc, dot_lhs, dot_rhs);
    }
    dot_total = ConvertIfNeeded(
        builder, loc, dot_total,
        mlir::RankedTensorType::get(reduced_shape, comp_elem_type));

    mlir::DenseI64ArrayAttr non_dim_attr =
        builder.getDenseI64ArrayAttr(non_dim_axes);
    mlir::Value bcast_dot_lhs =
        builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, lhs_comp_type, dot_total, non_dim_attr);
    mlir::Value bcast_dot_rhs =
        builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, rhs_comp_type, dot_total, non_dim_attr);

    mlir::Value dx_lhs = builder.create<mlir::stablehlo::MulOp>(
        loc, probs_lhs_comp,
        builder.create<mlir::stablehlo::SubtractOp>(loc, grad_lhs_comp,
                                                    bcast_dot_lhs));
    mlir::Value dx_rhs = builder.create<mlir::stablehlo::MulOp>(
        loc, probs_rhs_comp,
        builder.create<mlir::stablehlo::SubtractOp>(loc, grad_rhs_comp,
                                                    bcast_dot_rhs));
    if (convert_out) {
      mlir::Type res_elem_type =
          mlir::cast<mlir::RankedTensorType>(convert_out.getType())
              .getElementType();
      dx_lhs = builder.create<mlir::stablehlo::ConvertOp>(
          loc, mlir::RankedTensorType::get(lhs_type.getShape(), res_elem_type),
          dx_lhs);
      dx_rhs = builder.create<mlir::stablehlo::ConvertOp>(
          loc, mlir::RankedTensorType::get(rhs_type.getShape(), res_elem_type),
          dx_rhs);
    }

    auto new_concat = builder.create<mlir::stablehlo::ConcatenateOp>(
        loc, mlir::ValueRange{dx_lhs, dx_rhs}, dim);
    target_op->replaceAllUsesWith(new_concat.getOperation());
    SimplifyConcatSliceUsers(new_concat, dx_lhs, dx_rhs, dim);
    return true;
  }
  return false;
}

void PartitionConcatSoftmaxBackward(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::MulOp> mul_ops;
  func.walk([&mul_ops](mlir::stablehlo::MulOp op) { mul_ops.push_back(op); });
  for (mlir::stablehlo::MulOp mul_op : mul_ops) {
    if (mul_op && !mul_op->use_empty()) {
      TryPartitionConcatSoftmaxBackwardOp(mul_op);
    }
  }
  EraseTriviallyDeadOps(func);
}

struct FoldConcatSoftmaxPass
    : public mlir::PassWrapper<FoldConcatSoftmaxPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldConcatSoftmaxPass)

  llvm::StringRef getArgument() const override { return "fold-concat-softmax"; }
  llvm::StringRef getDescription() const override {
    return "Folds shift-invariant softmax pre-subtractions and partitions "
           "softmax across concatenations along the reduction axis.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    FoldShiftInvariantPreSubtraction(func);
    PartitionReciprocalConcatSoftmax(func);
    PartitionDivConcatSoftmax(func);
    PartitionConcatSoftmaxBackward(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createFoldConcatSoftmaxPass() {
  return std::make_unique<FoldConcatSoftmaxPass>();
}

}  // namespace torch_tpu
