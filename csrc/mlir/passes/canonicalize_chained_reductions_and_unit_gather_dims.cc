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

#include "csrc/mlir/passes/canonicalize_chained_reductions_and_unit_gather_dims.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
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

// GENERIC: capability maximum table extent where idx + extent cannot overflow
// signed 32-bit integer arithmetic when narrowing negative-index wraps.
constexpr int64_t kMaxSafeI32Bound = std::numeric_limits<int32_t>::max() / 2;

// GENERIC: semantic binary reduction region has 2 scalar block arguments.
constexpr int64_t kBinaryReducerNumArgs = 2;

// GENERIC: semantic minimum tensor rank to squeeze one unit dimension while
// keeping at least 1 batch dimension in start_indices.
constexpr int64_t kMinRankForUnitDimSqueeze = 2;

void RecursivelyEraseDeadOps(llvm::ArrayRef<mlir::Operation*> roots) {
  llvm::SmallVector<mlir::Operation*> worklist;
  llvm::SmallPtrSet<mlir::Operation*, 16> queued;
  for (mlir::Operation* root : roots) {
    if (root && mlir::isOpTriviallyDead(root) && queued.insert(root).second) {
      worklist.push_back(root);
    }
  }
  while (!worklist.empty()) {
    mlir::Operation* op = worklist.pop_back_val();
    if (!mlir::isOpTriviallyDead(op)) {
      continue;
    }
    llvm::SmallVector<mlir::Value> operands(op->getOperands());
    op->erase();
    for (mlir::Value operand : operands) {
      if (mlir::Operation* def_op = operand.getDefiningOp()) {
        if (mlir::isOpTriviallyDead(def_op) && queued.insert(def_op).second) {
          worklist.push_back(def_op);
        }
      }
    }
  }
}

std::optional<double> MatchSplatFloatConstant(mlir::Value val) {
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    val = bcast.getOperand();
  }
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  auto dense_attr =
      mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense_attr || !dense_attr.isSplat()) {
    return std::nullopt;
  }
  if (!mlir::isa<mlir::FloatType>(dense_attr.getElementType())) {
    return std::nullopt;
  }
  llvm::APFloat ap_val = dense_attr.getSplatValue<llvm::APFloat>();
  if (!ap_val.isFinite()) {
    return std::nullopt;
  }
  return ap_val.convertToDouble();
}

std::optional<int64_t> MatchSplatIntConstant(mlir::Value val) {
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    val = bcast.getOperand();
  }
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  auto dense_attr =
      mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense_attr || !dense_attr.isSplat()) {
    return std::nullopt;
  }
  auto int_type =
      mlir::dyn_cast<mlir::IntegerType>(dense_attr.getElementType());
  if (!int_type || int_type.getWidth() > 64) {
    return std::nullopt;
  }
  llvm::APInt ap_val = dense_attr.getSplatValue<llvm::APInt>();
  return ap_val.getSExtValue();
}

bool IsZeroFloatConstant(mlir::Value val) {
  std::optional<double> cst = MatchSplatFloatConstant(val);
  return cst.has_value() && *cst == 0.0;
}

bool IsAddReducerBody(mlir::Region& region) {
  if (region.empty()) {
    return false;
  }
  mlir::Block& block = region.front();
  if (block.getNumArguments() != kBinaryReducerNumArgs) {
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

bool IsAdditiveSingleFloatReduce(mlir::stablehlo::ReduceOp reduce_op) {
  if (!reduce_op || reduce_op.getNumResults() != 1 ||
      reduce_op.getInputs().size() != 1 ||
      reduce_op.getInitValues().size() != 1 ||
      reduce_op.getDimensions().empty()) {
    return false;
  }
  auto in_type = mlir::dyn_cast<mlir::RankedTensorType>(
      reduce_op.getInputs()[0].getType());
  auto out_type =
      mlir::dyn_cast<mlir::RankedTensorType>(reduce_op.getResult(0).getType());
  if (!in_type || !out_type || !in_type.hasStaticShape() ||
      !out_type.hasStaticShape()) {
    return false;
  }
  auto float_type = mlir::dyn_cast<mlir::FloatType>(in_type.getElementType());
  if (!float_type || out_type.getElementType() != float_type) {
    return false;
  }
  return IsZeroFloatConstant(reduce_op.getInitValues()[0]) &&
         IsAddReducerBody(reduce_op.getBody());
}

mlir::Value CreateSplatFloatConst(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::RankedTensorType tensor_type,
                                  double value) {
  auto float_type = mlir::cast<mlir::FloatType>(tensor_type.getElementType());
  llvm::APFloat ap_val(value);
  bool loses_info = false;
  ap_val.convert(float_type.getFloatSemantics(),
                 llvm::APFloat::rmNearestTiesToEven, &loses_info);
  auto attr = mlir::DenseElementsAttr::get(tensor_type, ap_val);
  return builder.create<mlir::stablehlo::ConstantOp>(loc, tensor_type, attr);
}

mlir::Value CreateSplatInt32Const(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::RankedTensorType tensor_type,
                                  int32_t value) {
  auto attr = mlir::DenseElementsAttr::get(tensor_type, value);
  return builder.create<mlir::stablehlo::ConstantOp>(loc, tensor_type, attr);
}

// Maps reduction dimensions of `reduce2` back to original axes of `x_type`
// (the input of `reduce1`), supporting both drop-dims and keep-dims shapes
// between `reduce1` and `reduce2`.
std::optional<llvm::SmallVector<int64_t>> MapChainedReductionDims(
    mlir::RankedTensorType x_type, llvm::ArrayRef<int64_t> dims1,
    mlir::RankedTensorType r1_out_type, mlir::RankedTensorType r2_in_type,
    llvm::ArrayRef<int64_t> dims2) {
  llvm::DenseSet<int64_t> dims1_set(dims1.begin(), dims1.end());
  llvm::SmallVector<int64_t> non_reduced1;
  for (int64_t a = 0; a < x_type.getRank(); ++a) {
    if (!dims1_set.contains(a)) {
      non_reduced1.push_back(a);
    }
  }

  llvm::DenseSet<int64_t> combined_set(dims1.begin(), dims1.end());
  if (r2_in_type.getShape() == r1_out_type.getShape()) {
    for (int64_t d2 : dims2) {
      if (d2 < 0 || d2 >= static_cast<int64_t>(non_reduced1.size())) {
        return std::nullopt;
      }
      combined_set.insert(non_reduced1[d2]);
    }
  } else if (r2_in_type.getRank() == x_type.getRank()) {
    for (int64_t a = 0; a < x_type.getRank(); ++a) {
      const int64_t expected = dims1_set.contains(a) ? 1 : x_type.getDimSize(a);
      if (r2_in_type.getDimSize(a) != expected) {
        return std::nullopt;
      }
    }
    for (int64_t d2 : dims2) {
      if (d2 < 0 || d2 >= x_type.getRank()) {
        return std::nullopt;
      }
      combined_set.insert(d2);
    }
  } else {
    return std::nullopt;
  }

  if (combined_set.size() <= dims1_set.size()) {
    return std::nullopt;
  }
  llvm::SmallVector<int64_t> combined_dims(combined_set.begin(),
                                           combined_set.end());
  llvm::sort(combined_dims);
  for (int64_t d : combined_dims) {
    if (x_type.getDimSize(d) <= 0) {
      return std::nullopt;
    }
  }
  return combined_dims;
}

bool TryFuseChainedAdditiveReduction(
    mlir::stablehlo::ReduceOp reduce2,
    llvm::SmallVectorImpl<mlir::Operation*>& dead_roots) {
  if (!IsAdditiveSingleFloatReduce(reduce2)) {
    return false;
  }
  mlir::Value r2_in = reduce2.getInputs()[0];
  auto r2_in_type = mlir::cast<mlir::RankedTensorType>(r2_in.getType());
  auto elem_type = mlir::cast<mlir::FloatType>(r2_in_type.getElementType());

  mlir::Value cur = r2_in;
  double div_scale = 1.0;
  double mul_scale = 1.0;
  bool has_div_scale = false;
  bool has_mul_scale = false;

  while (cur.hasOneUse() && !cur.getDefiningOp<mlir::stablehlo::ReduceOp>()) {
    if (auto reshape_op = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      cur = reshape_op.getOperand();
      continue;
    }
    if (auto conv_outer = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      mlir::Value mid = conv_outer.getOperand();
      auto mid_type = mlir::dyn_cast<mlir::RankedTensorType>(mid.getType());
      if (!mid_type || !mid_type.hasStaticShape()) {
        return false;
      }
      if (mid_type.getElementType() == elem_type) {
        cur = mid;
        continue;
      }
      auto mid_float =
          mlir::dyn_cast<mlir::FloatType>(mid_type.getElementType());
      auto conv_inner = mid.getDefiningOp<mlir::stablehlo::ConvertOp>();
      if (!mid_float || !mid.hasOneUse() || !conv_inner ||
          mid_float.getWidth() > elem_type.getWidth()) {
        return false;
      }
      mlir::Value src = conv_inner.getOperand();
      auto src_type = mlir::dyn_cast<mlir::RankedTensorType>(src.getType());
      if (!src_type || !src_type.hasStaticShape() ||
          src_type.getElementType() != elem_type) {
        return false;
      }
      cur = src;
      continue;
    }
    if (auto div_op = cur.getDefiningOp<mlir::stablehlo::DivOp>()) {
      std::optional<double> c = MatchSplatFloatConstant(div_op.getRhs());
      if (!c.has_value() || *c == 0.0) {
        return false;
      }
      div_scale *= *c;
      has_div_scale = true;
      cur = div_op.getLhs();
      continue;
    }
    if (auto mul_op = cur.getDefiningOp<mlir::stablehlo::MulOp>()) {
      if (std::optional<double> c = MatchSplatFloatConstant(mul_op.getRhs());
          c.has_value() && *c != 0.0) {
        mul_scale *= *c;
        has_mul_scale = true;
        cur = mul_op.getLhs();
        continue;
      }
      if (std::optional<double> c = MatchSplatFloatConstant(mul_op.getLhs());
          c.has_value() && *c != 0.0) {
        mul_scale *= *c;
        has_mul_scale = true;
        cur = mul_op.getRhs();
        continue;
      }
      return false;
    }
    return false;
  }

  auto reduce1 = cur.getDefiningOp<mlir::stablehlo::ReduceOp>();
  if (!reduce1 || !reduce1.getResult(0).hasOneUse() ||
      !IsAdditiveSingleFloatReduce(reduce1)) {
    return false;
  }

  mlir::Value x = reduce1.getInputs()[0];
  auto x_type = mlir::cast<mlir::RankedTensorType>(x.getType());
  auto r1_out_type =
      mlir::cast<mlir::RankedTensorType>(reduce1.getResult(0).getType());
  if (x_type.getElementType() != elem_type) {
    return false;
  }

  std::optional<llvm::SmallVector<int64_t>> combined_dims =
      MapChainedReductionDims(x_type, reduce1.getDimensions(), r1_out_type,
                              r2_in_type, reduce2.getDimensions());
  if (!combined_dims.has_value()) {
    return false;
  }

  llvm::DenseSet<int64_t> combined_set(combined_dims->begin(),
                                       combined_dims->end());
  llvm::SmallVector<int64_t> reduced_shape;
  for (int64_t a = 0; a < x_type.getRank(); ++a) {
    if (!combined_set.contains(a)) {
      reduced_shape.push_back(x_type.getDimSize(a));
    }
  }
  auto reduced_type = mlir::RankedTensorType::get(reduced_shape, elem_type);

  mlir::Operation* target_op = reduce2.getOperation();
  if (reduce2.getResult(0).hasOneUse()) {
    if (auto post_reshape = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(
            *reduce2.getResult(0).getUsers().begin())) {
      auto post_type =
          mlir::dyn_cast<mlir::RankedTensorType>(post_reshape.getType());
      if (post_type && post_type.hasStaticShape()) {
        target_op = post_reshape.getOperation();
      }
    }
  }

  mlir::OpBuilder builder(target_op);
  mlir::Location loc = target_op->getLoc();
  auto new_reduce = builder.create<mlir::stablehlo::ReduceOp>(
      loc, mlir::TypeRange{reduced_type}, mlir::ValueRange{x},
      mlir::ValueRange{reduce1.getInitValues()[0]},
      builder.getDenseI64ArrayAttr(*combined_dims));
  mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
      elem_type, new_reduce.getBody(), builder);

  mlir::Value result_val = new_reduce.getResult(0);
  auto target_type =
      mlir::cast<mlir::RankedTensorType>(target_op->getResult(0).getType());
  if (result_val.getType() != target_type) {
    result_val = builder.create<mlir::stablehlo::ReshapeOp>(loc, target_type,
                                                            result_val);
  }

  // Merge with a downstream DivOp when the first reduction was a mean.
  if (has_div_scale && !has_mul_scale && target_op->getResult(0).hasOneUse()) {
    if (auto post_div = mlir::dyn_cast<mlir::stablehlo::DivOp>(
            *target_op->getResult(0).getUsers().begin())) {
      if (post_div.getLhs() == target_op->getResult(0)) {
        std::optional<double> c2 = MatchSplatFloatConstant(post_div.getRhs());
        if (c2.has_value() && *c2 != 0.0) {
          builder.setInsertionPoint(post_div);
          mlir::Value combined_cst = CreateSplatFloatConst(
              builder, post_div.getLoc(), target_type, div_scale * (*c2));
          mlir::Value new_div = builder.create<mlir::stablehlo::DivOp>(
              post_div.getLoc(), target_type, result_val, combined_cst);
          post_div.replaceAllUsesWith(new_div);
          dead_roots.push_back(post_div.getOperation());
          return true;
        }
      }
    }
  }

  if (has_div_scale) {
    mlir::Value div_cst =
        CreateSplatFloatConst(builder, loc, target_type, div_scale);
    result_val = builder.create<mlir::stablehlo::DivOp>(loc, target_type,
                                                        result_val, div_cst);
  }
  if (has_mul_scale) {
    mlir::Value mul_cst =
        CreateSplatFloatConst(builder, loc, target_type, mul_scale);
    result_val = builder.create<mlir::stablehlo::MulOp>(loc, target_type,
                                                        result_val, mul_cst);
  }

  target_op->getResult(0).replaceAllUsesWith(result_val);
  dead_roots.push_back(target_op);
  return true;
}

bool IsEntryParameterOrReshape(mlir::Value raw_idx) {
  mlir::Value val = raw_idx;
  while (auto reshape_op = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    if (val != raw_idx && !val.hasOneUse()) {
      return false;
    }
    val = reshape_op.getOperand();
  }
  auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(val);
  return block_arg && block_arg.getOwner()->isEntryBlock() &&
         mlir::isa<mlir::func::FuncOp>(block_arg.getOwner()->getParentOp());
}

bool IsNegativeIndexCompare(mlir::stablehlo::CompareOp cmp_op,
                            mlir::Value raw_idx) {
  const auto dir = cmp_op.getComparisonDirection();
  if (cmp_op.getLhs() == raw_idx) {
    std::optional<int64_t> rhs_cst = MatchSplatIntConstant(cmp_op.getRhs());
    if (!rhs_cst.has_value()) {
      return false;
    }
    return (dir == mlir::stablehlo::ComparisonDirection::LT && *rhs_cst == 0) ||
           (dir == mlir::stablehlo::ComparisonDirection::LE && *rhs_cst == -1);
  }
  if (cmp_op.getRhs() == raw_idx) {
    std::optional<int64_t> lhs_cst = MatchSplatIntConstant(cmp_op.getLhs());
    if (!lhs_cst.has_value()) {
      return false;
    }
    return (dir == mlir::stablehlo::ComparisonDirection::GT && *lhs_cst == 0) ||
           (dir == mlir::stablehlo::ComparisonDirection::GE && *lhs_cst == -1);
  }
  return false;
}

// Narrows a 64-bit negative-index wrap:
//   SelectOp(CompareOp(idx_i64, 0, LT), AddOp(idx_i64, V), idx_i64)
// to i32 when `idx_i64` is an entry parameter (or ReshapeOp of an entry
// parameter) and `V == table_extent <= kMaxSafeI32Bound`.
mlir::Value TryNarrowEntryNegativeIndexWrapToI32(mlir::Value start_indices,
                                                 int64_t table_extent) {
  if (table_extent <= 0 || table_extent > kMaxSafeI32Bound) {
    return start_indices;
  }
  auto indices_type =
      mlir::dyn_cast<mlir::RankedTensorType>(start_indices.getType());
  if (!indices_type || !indices_type.hasStaticShape()) {
    return start_indices;
  }
  auto int_type =
      mlir::dyn_cast<mlir::IntegerType>(indices_type.getElementType());
  if (!int_type || int_type.getWidth() != 64) {
    return start_indices;
  }

  mlir::stablehlo::ReshapeOp outer_reshape;
  mlir::Value cur = start_indices;
  if (auto reshape_op = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    if (!cur.hasOneUse()) {
      return start_indices;
    }
    outer_reshape = reshape_op;
    cur = reshape_op.getOperand();
  }

  auto select_op = cur.getDefiningOp<mlir::stablehlo::SelectOp>();
  if (!select_op) {
    return start_indices;
  }

  mlir::Value raw_idx = select_op.getOnFalse();
  if (!IsEntryParameterOrReshape(raw_idx)) {
    return start_indices;
  }

  auto cmp_op = select_op.getPred().getDefiningOp<mlir::stablehlo::CompareOp>();
  auto add_op = select_op.getOnTrue().getDefiningOp<mlir::stablehlo::AddOp>();
  if (!cmp_op || !add_op || !select_op.getPred().hasOneUse() ||
      !select_op.getOnTrue().hasOneUse()) {
    return start_indices;
  }

  // Verify CompareOp(raw_idx, 0, LT) or canonicalized CompareOp(raw_idx, -1,
  // LE).
  if (!IsNegativeIndexCompare(cmp_op, raw_idx)) {
    return start_indices;
  }

  // Verify AddOp(raw_idx, table_extent).
  mlir::Value extent_operand;
  if (add_op.getLhs() == raw_idx) {
    extent_operand = add_op.getRhs();
  } else if (add_op.getRhs() == raw_idx) {
    extent_operand = add_op.getLhs();
  } else {
    return start_indices;
  }
  std::optional<int64_t> add_cst = MatchSplatIntConstant(extent_operand);
  if (!add_cst.has_value() || *add_cst != table_extent) {
    return start_indices;
  }

  auto raw_type = mlir::cast<mlir::RankedTensorType>(raw_idx.getType());
  mlir::OpBuilder builder(select_op);
  mlir::Location loc = select_op.getLoc();
  mlir::Type i32_elem_type = builder.getI32Type();
  auto raw_i32_type =
      mlir::RankedTensorType::get(raw_type.getShape(), i32_elem_type);

  mlir::Value idx_i32 =
      builder.create<mlir::stablehlo::ConvertOp>(loc, raw_i32_type, raw_idx);
  mlir::Value zero_i32 = CreateSplatInt32Const(builder, loc, raw_i32_type, 0);
  mlir::Value extent_i32 = CreateSplatInt32Const(
      builder, loc, raw_i32_type, static_cast<int32_t>(table_extent));

  auto cmp_i32 = builder.create<mlir::stablehlo::CompareOp>(
      loc, idx_i32, zero_i32, mlir::stablehlo::ComparisonDirection::LT);
  auto add_i32 = builder.create<mlir::stablehlo::AddOp>(loc, raw_i32_type,
                                                        idx_i32, extent_i32);
  mlir::Value wrapped_i32 = builder.create<mlir::stablehlo::SelectOp>(
      loc, raw_i32_type, cmp_i32, add_i32, idx_i32);

  if (outer_reshape) {
    auto reshaped_i32_type =
        mlir::RankedTensorType::get(indices_type.getShape(), i32_elem_type);
    return builder.create<mlir::stablehlo::ReshapeOp>(
        outer_reshape.getLoc(), reshaped_i32_type, wrapped_i32);
  }
  return wrapped_i32;
}

bool TryCanonicalizeGatherIndices(
    mlir::stablehlo::GatherOp gather_op,
    llvm::SmallVectorImpl<mlir::Operation*>& dead_roots) {
  auto operand_type =
      mlir::dyn_cast<mlir::RankedTensorType>(gather_op.getOperand().getType());
  auto indices_type = mlir::dyn_cast<mlir::RankedTensorType>(
      gather_op.getStartIndices().getType());
  auto result_type =
      mlir::dyn_cast<mlir::RankedTensorType>(gather_op.getResult().getType());
  if (!operand_type || !indices_type || !result_type ||
      !operand_type.hasStaticShape() || !indices_type.hasStaticShape() ||
      !result_type.hasStaticShape()) {
    return false;
  }

  mlir::stablehlo::GatherDimensionNumbersAttr dnums =
      gather_op.getDimensionNumbers();
  if (!dnums.getOperandBatchingDims().empty() ||
      !dnums.getStartIndicesBatchingDims().empty() ||
      dnums.getStartIndexMap().size() != 1 ||
      dnums.getCollapsedSliceDims().size() != 1 ||
      dnums.getStartIndexMap()[0] != dnums.getCollapsedSliceDims()[0]) {
    return false;
  }

  const int64_t indexed_dim = dnums.getStartIndexMap()[0];
  if (indexed_dim < 0 || indexed_dim >= operand_type.getRank()) {
    return false;
  }
  const int64_t table_extent = operand_type.getDimSize(indexed_dim);
  mlir::Value start_indices = gather_op.getStartIndices();

  mlir::Value new_indices =
      TryNarrowEntryNegativeIndexWrapToI32(start_indices, table_extent);
  bool changed = (new_indices != start_indices);
  auto cur_indices_type =
      mlir::cast<mlir::RankedTensorType>(new_indices.getType());

  int64_t index_vector_dim = dnums.getIndexVectorDim();
  if (index_vector_dim >= 0 && index_vector_dim < cur_indices_type.getRank() &&
      cur_indices_type.getRank() >= kMinRankForUnitDimSqueeze &&
      cur_indices_type.getDimSize(index_vector_dim) == 1) {
    llvm::SmallVector<int64_t> squeezed_shape;
    squeezed_shape.reserve(cur_indices_type.getRank() - 1);
    for (int64_t i = 0; i < cur_indices_type.getRank(); ++i) {
      if (i != index_vector_dim) {
        squeezed_shape.push_back(cur_indices_type.getDimSize(i));
      }
    }
    auto squeezed_type = mlir::RankedTensorType::get(
        squeezed_shape, cur_indices_type.getElementType());
    if (auto reshape_op =
            new_indices.getDefiningOp<mlir::stablehlo::ReshapeOp>();
        reshape_op && reshape_op.getOperand().getType() == squeezed_type) {
      new_indices = reshape_op.getOperand();
    } else {
      mlir::OpBuilder builder(gather_op);
      new_indices = builder.create<mlir::stablehlo::ReshapeOp>(
          gather_op.getLoc(), squeezed_type, new_indices);
    }
    index_vector_dim = squeezed_type.getRank();
    changed = true;
  }

  if (!changed) {
    return false;
  }

  mlir::OpBuilder builder(gather_op);
  auto new_dnums = mlir::stablehlo::GatherDimensionNumbersAttr::get(
      builder.getContext(), dnums.getOffsetDims(),
      dnums.getCollapsedSliceDims(), dnums.getOperandBatchingDims(),
      dnums.getStartIndicesBatchingDims(), dnums.getStartIndexMap(),
      index_vector_dim);
  auto new_gather = builder.create<mlir::stablehlo::GatherOp>(
      gather_op.getLoc(), result_type, gather_op.getOperand(), new_indices,
      new_dnums, gather_op.getSliceSizesAttr(),
      gather_op.getIndicesAreSortedAttr());
  gather_op.replaceAllUsesWith(new_gather.getResult());
  dead_roots.push_back(gather_op.getOperation());
  return true;
}

// Pass that fuses chained additive reductions and canonicalizes gather indices.
class CanonicalizeChainedReductionsAndUnitGatherDimsPass
    : public mlir::PassWrapper<
          CanonicalizeChainedReductionsAndUnitGatherDimsPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CanonicalizeChainedReductionsAndUnitGatherDimsPass)

  llvm::StringRef getArgument() const final {
    return "torch-tpu-canonicalize-chained-reductions-and-unit-gather-dims";
  }

  llvm::StringRef getDescription() const final {
    return "Fuse sequential additive reductions and canonicalize unit-vector "
           "gather indices in StableHLO";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::Operation*> dead_roots;

    llvm::SmallVector<mlir::stablehlo::ReduceOp> reduce_ops;
    func.walk([&](mlir::stablehlo::ReduceOp op) { reduce_ops.push_back(op); });
    for (mlir::stablehlo::ReduceOp reduce_op : reduce_ops) {
      if (reduce_op && !mlir::isOpTriviallyDead(reduce_op)) {
        TryFuseChainedAdditiveReduction(reduce_op, dead_roots);
      }
    }

    llvm::SmallVector<mlir::stablehlo::GatherOp> gather_ops;
    func.walk([&](mlir::stablehlo::GatherOp op) { gather_ops.push_back(op); });
    for (mlir::stablehlo::GatherOp gather_op : gather_ops) {
      if (gather_op && !mlir::isOpTriviallyDead(gather_op)) {
        TryCanonicalizeGatherIndices(gather_op, dead_roots);
      }
    }

    RecursivelyEraseDeadOps(dead_roots);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
createCanonicalizeChainedReductionsAndUnitGatherDimsPass() {
  return std::make_unique<CanonicalizeChainedReductionsAndUnitGatherDimsPass>();
}

}  // namespace torch_tpu
