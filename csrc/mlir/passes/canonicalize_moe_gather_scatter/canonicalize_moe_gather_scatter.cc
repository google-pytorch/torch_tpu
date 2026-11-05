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

#include "csrc/mlir/passes/canonicalize_moe_gather_scatter/canonicalize_moe_gather_scatter.h"  // NOLINT

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // NOLINT
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"  // NOLINT
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

constexpr int kMaxAnalysisRecursionDepth = 8;
constexpr int64_t kMaxSignedInt32 = std::numeric_limits<int32_t>::max();

struct SortOutputRef {
  mlir::stablehlo::SortOp sort_op;
  unsigned result_index = 0;
};

enum class RowWiseStepKind {
  kWidenFloatConvert,
  kRowScaleMul,
  kRowMaskZeroSelect,
};

struct RowWiseStep {
  RowWiseStepKind kind;
  mlir::Operation* op = nullptr;
  mlir::Type dst_elem_type;
  mlir::Value row_vec_1d;
  mlir::TypedAttr zero_scalar_attr;
  bool prev_is_lhs = true;
  bool zero_on_true = true;
};

enum class CoordKind {
  kConstant,
  kIdentityAxis,
};

struct CoordClassification {
  CoordKind kind;
  int64_t value_or_axis = 0;
};

std::optional<mlir::TypedAttr> GetSplatZeroScalarAttr(mlir::Value val) {
  while (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
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
  mlir::Type elem_type = dense_attr.getType().getElementType();
  if (elem_type.isInteger()) {
    llvm::APInt int_val = dense_attr.getSplatValue<llvm::APInt>();
    if (int_val.isZero()) {
      return mlir::IntegerAttr::get(elem_type, int_val);
    }
  } else if (mlir::isa<mlir::FloatType>(elem_type)) {
    llvm::APFloat flt_val = dense_attr.getSplatValue<llvm::APFloat>();
    if (flt_val.isZero()) {
      return mlir::FloatAttr::get(elem_type, flt_val);
    }
  }
  return std::nullopt;
}

bool IsZeroTensor(mlir::Value val) {
  return GetSplatZeroScalarAttr(val).has_value();
}

bool HasStaticNumElements(mlir::Value val, int64_t expected_size) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  return type && type.hasStaticShape() &&
         type.getNumElements() == expected_size;
}

std::optional<mlir::Value> TryUnwrapTupleElement(mlir::Value val,
                                                 int64_t expected_size) {
  if (auto get_tuple =
          val.getDefiningOp<mlir::stablehlo::GetTupleElementOp>()) {
    const int32_t idx = get_tuple.getIndex();
    if (idx < 0) {
      return std::nullopt;
    }
    if (auto tuple_op =
            get_tuple.getOperand().getDefiningOp<mlir::stablehlo::TupleOp>()) {
      if (static_cast<size_t>(idx) < tuple_op.getVal().size()) {
        mlir::Value elem = tuple_op.getVal()[idx];
        if (HasStaticNumElements(elem, expected_size)) {
          return elem;
        }
      }
    }
  }
  return std::nullopt;
}

std::optional<mlir::Value> Unwrap1DIndexStep(mlir::Value val,
                                             int64_t expected_size) {
  if (auto unwrapped = TryUnwrapTupleElement(val, expected_size)) {
    return unwrapped;
  }
  if (auto reshape = val.getDefiningOp<mlir::stablehlo::ReshapeOp>();
      reshape && HasStaticNumElements(reshape.getOperand(), expected_size)) {
    return reshape.getOperand();
  }
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
      bcast && HasStaticNumElements(bcast.getOperand(), expected_size)) {
    return bcast.getOperand();
  }
  if (auto convert = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    auto in_type =
        mlir::dyn_cast<mlir::RankedTensorType>(convert.getOperand().getType());
    auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(convert.getType());
    if (HasStaticNumElements(convert.getOperand(), expected_size) && out_type &&
        in_type.getElementType().isInteger() &&
        out_type.getElementType().isInteger() &&
        out_type.getElementType().getIntOrFloatBitWidth() >= 32) {
      return convert.getOperand();
    }
  }
  return std::nullopt;
}

mlir::Value Unwrap1DIndexTensor(mlir::Value val, int64_t expected_size) {
  for (int step = 0; step < kMaxAnalysisRecursionDepth; ++step) {
    auto next = Unwrap1DIndexStep(val, expected_size);
    if (!next.has_value()) {
      break;
    }
    val = *next;
  }
  return val;
}

bool UnwrapsTo1DIota(mlir::Value val, int64_t expected_size) {
  val = Unwrap1DIndexTensor(val, expected_size);
  auto iota = val.getDefiningOp<mlir::stablehlo::IotaOp>();
  if (!iota || iota.getIotaDimension() != 0) {
    return false;
  }
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(iota.getType());
  return type && type.getRank() == 1 && type.hasStaticShape() &&
         type.getDimSize(0) == expected_size;
}

bool Is1DSingleElementScatterDnums(
    mlir::stablehlo::ScatterDimensionNumbersAttr dnums) {
  return dnums.getUpdateWindowDims().empty() &&
         dnums.getInsertedWindowDims() == llvm::ArrayRef<int64_t>{0} &&
         dnums.getScatterDimsToOperandDims() == llvm::ArrayRef<int64_t>{0} &&
         dnums.getIndexVectorDim() == 1 &&
         dnums.getInputBatchingDims().empty() &&
         dnums.getScatterIndicesBatchingDims().empty();
}

bool IsScalarPassThroughRhsRegion(mlir::Region& region) {
  if (region.empty()) {
    return false;
  }
  mlir::Block& block = region.front();
  if (block.getNumArguments() != 2) {
    return false;
  }
  auto ret = mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  return ret && ret.getResults().size() == 1 &&
         ret.getResults()[0] == block.getArgument(1);
}

bool IsScalarAddRegion(mlir::Region& region) {
  if (region.empty()) {
    return false;
  }
  mlir::Block& block = region.front();
  if (block.getNumArguments() != 2) {
    return false;
  }
  auto ret = mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  if (!ret || ret.getResults().size() != 1) {
    return false;
  }
  auto add_op = ret.getResults()[0].getDefiningOp<mlir::stablehlo::AddOp>();
  return add_op && ((add_op.getLhs() == block.getArgument(0) &&
                     add_op.getRhs() == block.getArgument(1)) ||
                    (add_op.getLhs() == block.getArgument(1) &&
                     add_op.getRhs() == block.getArgument(0)));
}

std::optional<SortOutputRef> MatchSortOutput(mlir::Value val);

std::optional<SortOutputRef> MatchSortOutputFromGte(
    mlir::stablehlo::GetTupleElementOp gte) {
  const int32_t idx = gte.getIndex();
  if (idx < 0) {
    return std::nullopt;
  }
  mlir::Value tuple_val = gte.getOperand();
  if (auto sort_op = tuple_val.getDefiningOp<mlir::stablehlo::SortOp>()) {
    if (static_cast<size_t>(idx) < sort_op.getInputs().size()) {
      return SortOutputRef{sort_op, static_cast<unsigned>(idx)};
    }
  }
  if (auto tuple_op = tuple_val.getDefiningOp<mlir::stablehlo::TupleOp>()) {
    if (static_cast<size_t>(idx) < tuple_op.getVal().size()) {
      return MatchSortOutput(tuple_op.getVal()[idx]);
    }
  }
  return std::nullopt;
}

std::optional<SortOutputRef> MatchSortOutput(mlir::Value val) {
  if (auto t = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
      t && t.hasStaticShape()) {
    val = Unwrap1DIndexTensor(val, t.getNumElements());
  }
  while (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    val = conv.getOperand();
  }
  if (auto gte = val.getDefiningOp<mlir::stablehlo::GetTupleElementOp>()) {
    return MatchSortOutputFromGte(gte);
  }
  auto op_res = mlir::dyn_cast<mlir::OpResult>(val);
  auto sort_op =
      op_res ? mlir::dyn_cast<mlir::stablehlo::SortOp>(op_res.getOwner())
             : nullptr;
  if (sort_op && op_res.getResultNumber() < sort_op.getInputs().size()) {
    return SortOutputRef{sort_op, op_res.getResultNumber()};
  }
  return std::nullopt;
}

bool IsConstantPermutationOfIota(mlir::Value val, int64_t n) {
  val = Unwrap1DIndexTensor(val, n);
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto dense_attr =
      mlir::dyn_cast<mlir::DenseIntElementsAttr>(const_op.getValue());
  if (!dense_attr || dense_attr.getNumElements() != n) {
    return false;
  }
  llvm::SmallVector<bool> seen(n, false);
  for (const llvm::APInt& elem : dense_attr) {
    int64_t idx = elem.getSExtValue();
    if (idx < 0 || idx >= n || seen[idx]) {
      return false;
    }
    seen[idx] = true;
  }
  return true;
}

// Verify that the tensor is provably a permutation of iota(0..n-1).
bool Is1DPermutationOfIota(mlir::Value perm, int64_t n) {
  perm = Unwrap1DIndexTensor(perm, n);
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(perm.getType());
  if (!type || !type.hasStaticShape() || type.getNumElements() != n) {
    return false;
  }
  if (!type.getElementType().isInteger()) {
    return false;
  }
  if (UnwrapsTo1DIota(perm, n)) {
    return true;
  }
  if (IsConstantPermutationOfIota(perm, n)) {
    return true;
  }
  auto sort_ref = MatchSortOutput(perm);
  if (!sort_ref.has_value() || sort_ref->sort_op.getDimension() != 0 ||
      sort_ref->result_index != 1 || sort_ref->sort_op.getInputs().size() < 2) {
    return false;
  }
  mlir::Value iota_input = sort_ref->sort_op.getInputs()[1];
  auto iota_type = mlir::dyn_cast<mlir::RankedTensorType>(iota_input.getType());
  if (!iota_type || iota_type.getRank() != 1 ||
      !UnwrapsTo1DIota(iota_input, n)) {
    return false;
  }
  mlir::Value key_input = sort_ref->sort_op.getInputs()[0];
  auto key_type = mlir::dyn_cast<mlir::RankedTensorType>(key_input.getType());
  if (!key_type || key_type.getRank() != 1 ||
      !key_type.getElementType().isInteger()) {
    return false;
  }
  mlir::Region& comp = sort_ref->sort_op.getComparator();
  if (comp.empty()) {
    return false;
  }
  mlir::Block& comp_block = comp.front();
  if (comp_block.getNumArguments() < 2) {
    return false;
  }
  auto ret =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(comp_block.getTerminator());
  if (!ret || ret.getResults().size() != 1) {
    return false;
  }
  auto cmp = ret.getResults()[0].getDefiningOp<mlir::stablehlo::CompareOp>();
  if (!cmp || cmp.getComparisonDirection() !=
                  mlir::stablehlo::ComparisonDirection::LT) {
    return false;
  }
  if (cmp.getLhs() != comp_block.getArgument(0) ||
      cmp.getRhs() != comp_block.getArgument(1)) {
    return false;
  }
  return true;
}

std::optional<mlir::Value> MatchInversePermutation1D(mlir::Value val,
                                                     int64_t n) {
  val = Unwrap1DIndexTensor(val, n);
  if (!HasStaticNumElements(val, n)) {
    return std::nullopt;
  }
  if (auto scatter_op = val.getDefiningOp<mlir::stablehlo::ScatterOp>()) {
    if (scatter_op.getInputs().size() == 1 &&
        scatter_op.getUpdates().size() == 1 &&
        scatter_op.getNumResults() == 1 &&
        HasStaticNumElements(scatter_op.getInputs()[0], n) &&
        HasStaticNumElements(scatter_op.getScatterIndices(), n) &&
        Is1DSingleElementScatterDnums(
            scatter_op.getScatterDimensionNumbers()) &&
        IsScalarPassThroughRhsRegion(scatter_op.getUpdateComputation()) &&
        UnwrapsTo1DIota(scatter_op.getUpdates()[0], n)) {
      mlir::Value perm = Unwrap1DIndexTensor(scatter_op.getScatterIndices(), n);
      if (HasStaticNumElements(perm, n) && Is1DPermutationOfIota(perm, n)) {
        return perm;
      }
    }
  }
  if (auto sort_ref = MatchSortOutput(val)) {
    if (sort_ref->result_index == 1 && sort_ref->sort_op.getDimension() == 0 &&
        sort_ref->sort_op.getInputs().size() >= 2 &&
        HasStaticNumElements(sort_ref->sort_op.getInputs()[0], n)) {
      mlir::Value iota_input = sort_ref->sort_op.getInputs()[1];
      auto iota_type =
          mlir::dyn_cast<mlir::RankedTensorType>(iota_input.getType());
      if (!iota_type || iota_type.getRank() != 1 ||
          !UnwrapsTo1DIota(iota_input, n)) {
        return std::nullopt;
      }
      mlir::Value keys = sort_ref->sort_op.getInputs()[0];
      auto keys_type = mlir::dyn_cast<mlir::RankedTensorType>(keys.getType());
      if (!keys_type || keys_type.getRank() != 1 ||
          !keys_type.getElementType().isInteger()) {
        return std::nullopt;
      }
      mlir::Region& comp = sort_ref->sort_op.getComparator();
      if (comp.empty()) {
        return std::nullopt;
      }
      mlir::Block& comp_block = comp.front();
      if (comp_block.getNumArguments() < 2) {
        return std::nullopt;
      }
      auto ret =
          mlir::dyn_cast<mlir::stablehlo::ReturnOp>(comp_block.getTerminator());
      if (!ret || ret.getResults().size() != 1) {
        return std::nullopt;
      }
      auto cmp =
          ret.getResults()[0].getDefiningOp<mlir::stablehlo::CompareOp>();
      if (!cmp ||
          cmp.getComparisonDirection() !=
              mlir::stablehlo::ComparisonDirection::LT ||
          cmp.getLhs() != comp_block.getArgument(0) ||
          cmp.getRhs() != comp_block.getArgument(1)) {
        return std::nullopt;
      }
      if (!Is1DPermutationOfIota(keys, n)) {
        return std::nullopt;
      }
      mlir::Value perm = Unwrap1DIndexTensor(keys, n);
      if (HasStaticNumElements(perm, n)) {
        return perm;
      }
    }
  }
  return std::nullopt;
}

// Collect memory-effect-free dependencies in topological postorder so
// producers can be hoisted before consumers in the same block.
bool CollectSameBlockDepsBefore(
    mlir::Operation* def_op, mlir::Operation* target_op,
    llvm::SmallVectorImpl<mlir::Operation*>& topo_order) {
  enum class State { kVisiting, kVisited };
  llvm::DenseMap<mlir::Operation*, State> visited;
  llvm::SmallVector<mlir::Operation*, 16> postorder;

  std::function<bool(mlir::Operation*)> dfs =
      [&target_op, &visited, &postorder, &dfs](mlir::Operation* cur) -> bool {
    if (cur == target_op || cur->getNumRegions() > 0) {
      return false;
    }
    if (!mlir::isMemoryEffectFree(cur)) {
      return false;
    }
    auto [it, inserted] = visited.try_emplace(cur, State::kVisiting);
    if (!inserted) {
      return it->second == State::kVisited;
    }
    for (mlir::Value operand : cur->getOperands()) {
      mlir::Operation* op_def = operand.getDefiningOp();
      if (op_def && op_def->getBlock() == target_op->getBlock() &&
          !op_def->isBeforeInBlock(target_op)) {
        if (!dfs(op_def)) {
          return false;
        }
      }
    }
    it->second = State::kVisited;
    postorder.push_back(cur);
    return true;
  };

  if (!dfs(def_op)) {
    return false;
  }
  topo_order.assign(postorder.begin(), postorder.end());
  return true;
}

bool EnsureAvailableBefore(mlir::Value val, mlir::Operation* target_op) {
  mlir::Operation* def_op = val.getDefiningOp();
  if (!def_op || def_op->getBlock() != target_op->getBlock() ||
      def_op->isBeforeInBlock(target_op)) {
    return true;
  }
  llvm::SmallVector<mlir::Operation*, 16> topo_order;
  if (!CollectSameBlockDepsBefore(def_op, target_op, topo_order)) {
    return false;
  }
  for (mlir::Operation* op : topo_order) {
    if (!op->isBeforeInBlock(target_op)) {
      op->moveBefore(target_op);
    }
  }
  return true;
}

void EraseDeadOpsUpward(llvm::ArrayRef<mlir::Value> roots) {
  llvm::SmallVector<mlir::Operation*, 16> worklist;
  llvm::SmallPtrSet<mlir::Operation*, 16> in_worklist;
  auto push_op = [&worklist, &in_worklist](mlir::Operation* op) {
    if (op && in_worklist.insert(op).second) {
      worklist.push_back(op);
    }
  };
  for (mlir::Value root : roots) {
    if (mlir::Operation* def = root.getDefiningOp()) {
      push_op(def);
    }
  }
  while (!worklist.empty()) {
    mlir::Operation* op = worklist.pop_back_val();
    in_worklist.erase(op);
    if (!op || !op->use_empty() || !mlir::isMemoryEffectFree(op) ||
        op->hasTrait<mlir::OpTrait::IsTerminator>()) {
      continue;
    }
    llvm::SmallVector<mlir::Value> operands(op->getOperands().begin(),
                                            op->getOperands().end());
    op->erase();
    for (mlir::Value operand : operands) {
      if (mlir::Operation* def = operand.getDefiningOp()) {
        push_op(def);
      }
    }
  }
}

mlir::Value ConvertOrReshapeToI32(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::Value val,
                                  llvm::ArrayRef<int64_t> target_shape) {
  auto i32_tensor_type =
      mlir::RankedTensorType::get(target_shape, builder.getI32Type());
  if (val.getType() == i32_tensor_type) {
    return val;
  }
  int64_t num_elems = 1;
  for (const int64_t d : target_shape) {
    num_elems *= d;
  }
  mlir::Value unwrapped = Unwrap1DIndexTensor(val, num_elems);
  auto val_type = mlir::cast<mlir::RankedTensorType>(unwrapped.getType());
  mlir::Value i32_same_shape =
      val_type.getElementType().isInteger(32)
          ? unwrapped
          : builder
                .create<mlir::stablehlo::ConvertOp>(
                    loc,
                    mlir::RankedTensorType::get(val_type.getShape(),
                                                builder.getI32Type()),
                    unwrapped)
                .getResult();
  if (i32_same_shape.getType() == i32_tensor_type) {
    return i32_same_shape;
  }
  return builder
      .create<mlir::stablehlo::ReshapeOp>(loc, i32_tensor_type, i32_same_shape)
      .getResult();
}

std::optional<CoordClassification> ClassifyConstant(
    mlir::stablehlo::ConstantOp const_op, mlir::IntegerType int_type) {
  auto dense_attr =
      mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense_attr || !dense_attr.isSplat()) {
    return std::nullopt;
  }
  llvm::APInt val = dense_attr.getSplatValue<llvm::APInt>();
  if (!int_type.isUnsigned() && val.isNegative()) {
    return std::nullopt;
  }
  if (val.getActiveBits() >= 63) {
    return std::nullopt;
  }
  return CoordClassification{CoordKind::kConstant,
                             static_cast<int64_t>(val.getZExtValue())};
}

std::optional<CoordClassification> ClassifyIota(mlir::stablehlo::IotaOp iota_op,
                                                mlir::RankedTensorType type,
                                                mlir::IntegerType int_type) {
  int64_t axis = static_cast<int64_t>(iota_op.getIotaDimension());
  int64_t dim_size = type.getDimSize(axis);
  if (dim_size == 1) {
    return CoordClassification{CoordKind::kConstant, 0};
  }
  uint64_t max_val = static_cast<uint64_t>(dim_size - 1);
  int usable_bits =
      int_type.isUnsigned() ? int_type.getWidth() : (int_type.getWidth() - 1);
  if (usable_bits < 63 && max_val >= (uint64_t{1} << usable_bits)) {
    return std::nullopt;
  }
  return CoordClassification{CoordKind::kIdentityAxis, axis};
}

std::optional<CoordClassification> ClassifyReshape(
    mlir::stablehlo::ReshapeOp reshape_op, mlir::RankedTensorType type,
    const CoordClassification& inner) {
  if (inner.kind == CoordKind::kConstant) {
    return inner;
  }
  auto src_type =
      mlir::dyn_cast<mlir::RankedTensorType>(reshape_op.getOperand().getType());
  llvm::ArrayRef<int64_t> src_shape = src_type.getShape();
  llvm::ArrayRef<int64_t> dst_shape = type.getShape();
  int64_t src_axis = inner.value_or_axis;
  int64_t target_len = src_shape[src_axis];
  int64_t src_suffix_prod = 1;
  for (int64_t j = src_axis + 1; j < static_cast<int64_t>(src_shape.size());
       ++j) {
    src_suffix_prod *= src_shape[j];
  }
  int64_t dst_suffix_prod = 1;
  for (int64_t k = static_cast<int64_t>(dst_shape.size()) - 1; k >= 0; --k) {
    if (dst_shape[k] == target_len && dst_suffix_prod == src_suffix_prod) {
      return CoordClassification{CoordKind::kIdentityAxis, k};
    }
    dst_suffix_prod *= dst_shape[k];
  }
  return std::nullopt;
}

std::optional<CoordClassification> ClassifyCoordinate(mlir::Value v,
                                                      int depth = 0) {
  if (depth > kMaxAnalysisRecursionDepth || !v) {
    return std::nullopt;
  }
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!type || !type.hasStaticShape() || type.getNumElements() == 0) {
    return std::nullopt;
  }
  auto int_type = mlir::dyn_cast<mlir::IntegerType>(type.getElementType());
  if (!int_type || int_type.getWidth() <= 1) {
    return std::nullopt;
  }
  if (auto const_op = v.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
    return ClassifyConstant(const_op, int_type);
  }
  if (auto iota_op = v.getDefiningOp<mlir::stablehlo::IotaOp>()) {
    return ClassifyIota(iota_op, type, int_type);
  }
  if (auto convert_op = v.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    auto inner = ClassifyCoordinate(convert_op.getOperand(), depth + 1);
    if (!inner) {
      return std::nullopt;
    }
    uint64_t max_val =
        (inner->kind == CoordKind::kConstant)
            ? static_cast<uint64_t>(inner->value_or_axis)
            : static_cast<uint64_t>(type.getDimSize(inner->value_or_axis) - 1);
    int usable_bits =
        int_type.isUnsigned() ? int_type.getWidth() : (int_type.getWidth() - 1);
    if (usable_bits < 63 && max_val >= (uint64_t{1} << usable_bits)) {
      return std::nullopt;
    }
    return inner;
  }
  if (auto bcast_op = v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    auto inner = ClassifyCoordinate(bcast_op.getOperand(), depth + 1);
    if (!inner) {
      return std::nullopt;
    }
    if (inner->kind == CoordKind::kConstant) {
      return inner;
    }
    llvm::ArrayRef<int64_t> bcast_dims = bcast_op.getBroadcastDimensions();
    return CoordClassification{CoordKind::kIdentityAxis,
                               bcast_dims[inner->value_or_axis]};
  }
  if (auto reshape_op = v.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    auto inner = ClassifyCoordinate(reshape_op.getOperand(), depth + 1);
    if (!inner) {
      return std::nullopt;
    }
    return ClassifyReshape(reshape_op, type, *inner);
  }
  return std::nullopt;
}

// Mechanism 1: Fold identity or constant gather.
bool TryEliminateIdentityOrConstantGather(mlir::stablehlo::GatherOp gather_op,
                                          mlir::IRRewriter& rewriter) {
  auto operand_type =
      mlir::dyn_cast<mlir::RankedTensorType>(gather_op.getOperand().getType());
  auto indices_type = mlir::dyn_cast<mlir::RankedTensorType>(
      gather_op.getStartIndices().getType());
  auto result_type =
      mlir::dyn_cast<mlir::RankedTensorType>(gather_op.getResult().getType());
  if (!operand_type || !operand_type.hasStaticShape() || !indices_type ||
      !indices_type.hasStaticShape() || !result_type ||
      !result_type.hasStaticShape() || operand_type.getNumElements() == 0 ||
      result_type.getNumElements() == 0) {
    return false;
  }

  mlir::stablehlo::GatherDimensionNumbersAttr dnums =
      gather_op.getDimensionNumbers();
  if (!dnums.getOperandBatchingDims().empty() ||
      !dnums.getStartIndicesBatchingDims().empty()) {
    return false;
  }

  llvm::ArrayRef<int64_t> start_index_map = dnums.getStartIndexMap();
  const int64_t num_coords = static_cast<int64_t>(start_index_map.size());
  if (num_coords < 1) {
    return false;
  }

  const int64_t indices_rank = indices_type.getRank();
  const int64_t index_vector_dim = dnums.getIndexVectorDim();
  if (index_vector_dim < 0 || index_vector_dim > indices_rank) {
    return false;
  }

  llvm::SmallVector<int64_t, 6> batch_shape;
  for (int64_t d = 0; d < indices_rank; ++d) {
    if (d != index_vector_dim) {
      batch_shape.push_back(indices_type.getDimSize(d));
    }
  }

  llvm::SmallVector<mlir::Value, 4> coords;
  if (index_vector_dim == indices_rank) {
    if (num_coords != 1) {
      return false;
    }
    coords.push_back(gather_op.getStartIndices());
  } else {
    if (indices_type.getDimSize(index_vector_dim) != num_coords) {
      return false;
    }
    if (num_coords == 1) {
      coords.push_back(gather_op.getStartIndices());
    } else {
      auto concat_op = gather_op.getStartIndices()
                           .getDefiningOp<mlir::stablehlo::ConcatenateOp>();
      if (!concat_op ||
          static_cast<int64_t>(concat_op.getDimension()) != index_vector_dim ||
          static_cast<int64_t>(concat_op.getInputs().size()) != num_coords) {
        return false;
      }
      for (mlir::Value input : concat_op.getInputs()) {
        auto in_type = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
        if (!in_type || !in_type.hasStaticShape() ||
            in_type.getDimSize(index_vector_dim) != 1) {
          return false;
        }
        coords.push_back(input);
      }
    }
  }

  const int64_t operand_rank = operand_type.getRank();
  llvm::ArrayRef<int64_t> slice_sizes = gather_op.getSliceSizes();
  llvm::DenseSet<int64_t> collapsed_set(dnums.getCollapsedSliceDims().begin(),
                                        dnums.getCollapsedSliceDims().end());

  llvm::SmallVector<int64_t, 6> const_index_for_src_dim(operand_rank, -1);
  llvm::SmallVector<int64_t, 6> iota_batch_for_src_dim(operand_rank, -1);
  llvm::DenseSet<int64_t> seen_src_dims;
  llvm::DenseSet<int64_t> seen_batch_dims;

  for (int64_t k = 0; k < num_coords; ++k) {
    int64_t src_dim = start_index_map[k];
    if (src_dim < 0 || src_dim >= operand_rank ||
        !seen_src_dims.insert(src_dim).second) {
      return false;
    }
    if (slice_sizes[src_dim] != 1 || !collapsed_set.contains(src_dim)) {
      return false;
    }

    std::optional<CoordClassification> classified =
        ClassifyCoordinate(coords[k]);
    if (!classified) {
      return false;
    }
    if (classified->kind == CoordKind::kConstant) {
      int64_t c = classified->value_or_axis;
      if (c < 0 || c >= operand_type.getDimSize(src_dim)) {
        return false;
      }
      const_index_for_src_dim[src_dim] = c;
    } else {
      int64_t coord_axis = classified->value_or_axis;
      int64_t batch_axis = coord_axis;
      if (index_vector_dim < indices_rank) {
        if (coord_axis == index_vector_dim) {
          return false;
        }
        batch_axis =
            (coord_axis < index_vector_dim) ? coord_axis : (coord_axis - 1);
      }
      if (batch_axis < 0 ||
          batch_axis >= static_cast<int64_t>(batch_shape.size()) ||
          !seen_batch_dims.insert(batch_axis).second ||
          operand_type.getDimSize(src_dim) != batch_shape[batch_axis]) {
        return false;
      }
      iota_batch_for_src_dim[src_dim] = batch_axis;
    }
  }

  for (int64_t d = 0; d < operand_rank; ++d) {
    if (collapsed_set.contains(d)) {
      if (!seen_src_dims.contains(d)) {
        if (slice_sizes[d] != 1) {
          return false;
        }
        const_index_for_src_dim[d] = 0;
      }
    } else {
      if (slice_sizes[d] != operand_type.getDimSize(d)) {
        return false;
      }
    }
  }

  llvm::ArrayRef<int64_t> offset_dims = dnums.getOffsetDims();
  llvm::DenseSet<int64_t> offset_set(offset_dims.begin(), offset_dims.end());
  llvm::SmallVector<int64_t, 6> batch_out_dims;
  for (int64_t out_d = 0; out_d < result_type.getRank(); ++out_d) {
    if (!offset_set.contains(out_d)) {
      batch_out_dims.push_back(out_d);
    }
  }
  if (batch_out_dims.size() != batch_shape.size()) {
    return false;
  }

  rewriter.setInsertionPoint(gather_op);
  mlir::Location loc = gather_op.getLoc();
  mlir::Value current = gather_op.getOperand();

  bool needs_slice = false;
  llvm::SmallVector<int64_t, 6> slice_starts(operand_rank, 0);
  llvm::SmallVector<int64_t, 6> slice_limits(operand_type.getShape().begin(),
                                             operand_type.getShape().end());
  llvm::SmallVector<int64_t, 6> slice_strides(operand_rank, 1);
  for (int64_t d = 0; d < operand_rank; ++d) {
    if (const_index_for_src_dim[d] >= 0) {
      slice_starts[d] = const_index_for_src_dim[d];
      slice_limits[d] = const_index_for_src_dim[d] + 1;
      if (operand_type.getDimSize(d) > 1) {
        needs_slice = true;
      }
    }
  }
  if (needs_slice) {
    current = rewriter.create<mlir::stablehlo::SliceOp>(
        loc, current, rewriter.getDenseI64ArrayAttr(slice_starts),
        rewriter.getDenseI64ArrayAttr(slice_limits),
        rewriter.getDenseI64ArrayAttr(slice_strides));
  }

  llvm::SmallVector<int64_t, 6> squeezed_shape;
  llvm::SmallVector<int64_t, 6> dest_dims;
  int64_t offset_idx = 0;
  for (int64_t d = 0; d < operand_rank; ++d) {
    if (const_index_for_src_dim[d] >= 0) {
      continue;
    }
    squeezed_shape.push_back(operand_type.getDimSize(d));
    if (iota_batch_for_src_dim[d] >= 0) {
      dest_dims.push_back(batch_out_dims[iota_batch_for_src_dim[d]]);
    } else {
      dest_dims.push_back(offset_dims[offset_idx++]);
    }
  }

  auto cur_type = mlir::cast<mlir::RankedTensorType>(current.getType());
  if (cur_type.getShape() != llvm::ArrayRef<int64_t>(squeezed_shape)) {
    auto squeezed_type = mlir::RankedTensorType::get(
        squeezed_shape, operand_type.getElementType());
    current = rewriter.create<mlir::stablehlo::ReshapeOp>(loc, squeezed_type,
                                                          current);
  }

  if (!llvm::is_sorted(dest_dims)) {
    const int64_t rem_rank = static_cast<int64_t>(dest_dims.size());
    llvm::SmallVector<int64_t, 6> perm(rem_rank);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&dest_dims](int64_t a, int64_t b) {
      return dest_dims[a] < dest_dims[b];
    });
    llvm::SmallVector<int64_t, 6> transposed_shape(rem_rank);
    llvm::SmallVector<int64_t, 6> sorted_dest_dims(rem_rank);
    for (int64_t i = 0; i < rem_rank; ++i) {
      transposed_shape[i] = squeezed_shape[perm[i]];
      sorted_dest_dims[i] = dest_dims[perm[i]];
    }
    auto transposed_type = mlir::RankedTensorType::get(
        transposed_shape, operand_type.getElementType());
    current = rewriter.create<mlir::stablehlo::TransposeOp>(
        loc, transposed_type, current, rewriter.getDenseI64ArrayAttr(perm));
    dest_dims = std::move(sorted_dest_dims);
  }

  // BroadcastInDimOp requires destination dimensions to be strictly increasing.
  for (size_t i = 1; i < dest_dims.size(); ++i) {
    if (dest_dims[i] <= dest_dims[i - 1]) {
      return false;
    }
  }

  if (current.getType() != result_type) {
    current = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, result_type, current, rewriter.getDenseI64ArrayAttr(dest_dims));
  }

  rewriter.replaceOp(gather_op, current);
  return true;
}

// Mechanism 2: Row-wise gather sinking.
bool IsRowGatherAlongDim0(mlir::stablehlo::GatherOp gather_op, int64_t& out_m,
                          int64_t& out_n,
                          llvm::SmallVectorImpl<int64_t>& slice_shape) {
  auto operand_type =
      mlir::dyn_cast<mlir::RankedTensorType>(gather_op.getOperand().getType());
  auto res_type =
      mlir::dyn_cast<mlir::RankedTensorType>(gather_op.getResult().getType());
  if (!operand_type || !res_type || !operand_type.hasStaticShape() ||
      !res_type.hasStaticShape() || operand_type.getRank() < 2 ||
      res_type.getRank() != operand_type.getRank() ||
      operand_type.getElementType() != res_type.getElementType()) {
    return false;
  }
  const int64_t rank = operand_type.getRank();
  const int64_t m = operand_type.getDimSize(0);
  const int64_t n = res_type.getDimSize(0);
  if (m < 1 || n < 1 || !HasStaticNumElements(gather_op.getStartIndices(), n)) {
    return false;
  }
  slice_shape.clear();
  for (int64_t d = 1; d < rank; ++d) {
    if (res_type.getDimSize(d) != operand_type.getDimSize(d)) {
      return false;
    }
    slice_shape.push_back(operand_type.getDimSize(d));
  }
  mlir::stablehlo::GatherDimensionNumbersAttr dnums =
      gather_op.getDimensionNumbers();
  llvm::SmallVector<int64_t, 6> expected_offset_dims;
  for (int64_t d = 1; d < rank; ++d) {
    expected_offset_dims.push_back(d);
  }
  if (dnums.getOffsetDims() != llvm::ArrayRef<int64_t>(expected_offset_dims) ||
      dnums.getCollapsedSliceDims() != llvm::ArrayRef<int64_t>{0} ||
      !dnums.getOperandBatchingDims().empty() ||
      !dnums.getStartIndicesBatchingDims().empty() ||
      dnums.getStartIndexMap() != llvm::ArrayRef<int64_t>{0}) {
    return false;
  }
  llvm::SmallVector<int64_t, 6> expected_slice_sizes;
  expected_slice_sizes.push_back(1);
  for (int64_t d = 1; d < rank; ++d) {
    expected_slice_sizes.push_back(operand_type.getDimSize(d));
  }
  if (gather_op.getSliceSizes() !=
      llvm::ArrayRef<int64_t>(expected_slice_sizes)) {
    return false;
  }
  out_m = m;
  out_n = n;
  return true;
}

std::optional<mlir::Value> TryUnwrapRowBroadcastTo1D(
    mlir::Value val, int64_t expected_rows, mlir::Operation* target_op) {
  auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast) {
    return std::nullopt;
  }
  mlir::Value src = bcast.getOperand();
  auto src_type = mlir::dyn_cast<mlir::RankedTensorType>(src.getType());
  if (!src_type || !src_type.hasStaticShape()) {
    return std::nullopt;
  }
  const bool valid_bcast =
      (src_type.getRank() == 1 && src_type.getDimSize(0) == expected_rows &&
       bcast.getBroadcastDimensions() == llvm::ArrayRef<int64_t>{0}) ||
      (src_type.getRank() == 2 && src_type.getDimSize(0) == expected_rows &&
       src_type.getDimSize(1) == 1 &&
       bcast.getBroadcastDimensions() == llvm::ArrayRef<int64_t>{0, 1});
  if (!valid_bcast) {
    return std::nullopt;
  }
  for (int step = 0; step < kMaxAnalysisRecursionDepth; ++step) {
    if (auto reshape = src.getDefiningOp<mlir::stablehlo::ReshapeOp>();
        reshape && HasStaticNumElements(reshape.getOperand(), expected_rows)) {
      src = reshape.getOperand();
      continue;
    }
    if (auto inner_bcast =
            src.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
        inner_bcast &&
        HasStaticNumElements(inner_bcast.getOperand(), expected_rows)) {
      src = inner_bcast.getOperand();
      continue;
    }
    break;
  }
  auto peeled_type = mlir::dyn_cast<mlir::RankedTensorType>(src.getType());
  if (!peeled_type || !peeled_type.hasStaticShape() ||
      peeled_type.getNumElements() != expected_rows ||
      !EnsureAvailableBefore(src, target_op)) {
    return std::nullopt;
  }
  return src;
}

bool TryMatchRowScaleMulStep(mlir::stablehlo::MulOp mul, int64_t rows,
                             mlir::Type elem_type, mlir::Operation* gather_op,
                             RowWiseStep& out_step, mlir::Value& out_prev) {
  if (auto rhs_1d = TryUnwrapRowBroadcastTo1D(mul.getRhs(), rows, gather_op)) {
    out_step = RowWiseStep{
        RowWiseStepKind::kRowScaleMul, mul.getOperation(), elem_type, *rhs_1d,
        /*zero_scalar_attr=*/nullptr,
        /*prev_is_lhs=*/true,
        /*zero_on_true=*/true};
    out_prev = mul.getLhs();
    return true;
  }
  if (auto lhs_1d = TryUnwrapRowBroadcastTo1D(mul.getLhs(), rows, gather_op)) {
    out_step = RowWiseStep{
        RowWiseStepKind::kRowScaleMul, mul.getOperation(), elem_type, *lhs_1d,
        /*zero_scalar_attr=*/nullptr,
        /*prev_is_lhs=*/false,
        /*zero_on_true=*/true};
    out_prev = mul.getRhs();
    return true;
  }
  return false;
}

bool TryMatchRowMaskSelectStep(mlir::stablehlo::SelectOp sel, int64_t rows,
                               mlir::Type elem_type, mlir::Operation* gather_op,
                               RowWiseStep& out_step, mlir::Value& out_prev) {
  auto mask_1d = TryUnwrapRowBroadcastTo1D(sel.getPred(), rows, gather_op);
  if (!mask_1d.has_value()) {
    return false;
  }
  if (auto zero_attr = GetSplatZeroScalarAttr(sel.getOnTrue())) {
    out_step = RowWiseStep{RowWiseStepKind::kRowMaskZeroSelect,
                           sel.getOperation(),
                           elem_type,
                           *mask_1d,
                           *zero_attr,
                           /*prev_is_lhs=*/true,
                           /*zero_on_true=*/true};
    out_prev = sel.getOnFalse();
    return true;
  }
  if (auto zero_attr = GetSplatZeroScalarAttr(sel.getOnFalse())) {
    out_step = RowWiseStep{RowWiseStepKind::kRowMaskZeroSelect,
                           sel.getOperation(),
                           elem_type,
                           *mask_1d,
                           *zero_attr,
                           /*prev_is_lhs=*/true,
                           /*zero_on_true=*/false};
    out_prev = sel.getOnTrue();
    return true;
  }
  return false;
}

bool TryMatchSingleRowWiseStep(mlir::Value cur, int64_t rows,
                               llvm::ArrayRef<int64_t> slice_shape,
                               mlir::Block& block, mlir::Operation* gather_op,
                               RowWiseStep& out_step, mlir::Value& out_prev) {
  if (!cur.hasOneUse()) {
    return false;
  }
  mlir::Operation* op = cur.getDefiningOp();
  auto cur_type = mlir::dyn_cast<mlir::RankedTensorType>(cur.getType());
  if (!op || op->getBlock() != &block || !cur_type ||
      !cur_type.hasStaticShape() ||
      cur_type.getRank() != static_cast<int64_t>(slice_shape.size()) + 1 ||
      cur_type.getDimSize(0) != rows) {
    return false;
  }
  for (size_t i = 0; i < slice_shape.size(); ++i) {
    if (cur_type.getDimSize(i + 1) != slice_shape[i]) {
      return false;
    }
  }
  if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
    mlir::Value prev = conv.getOperand();
    auto prev_type = mlir::dyn_cast<mlir::RankedTensorType>(prev.getType());
    auto prev_flt =
        prev_type ? mlir::dyn_cast<mlir::FloatType>(prev_type.getElementType())
                  : nullptr;
    auto cur_flt = mlir::dyn_cast<mlir::FloatType>(cur_type.getElementType());
    if (!prev_flt || !cur_flt || !prev_type.hasStaticShape() ||
        prev_type.getShape() != cur_type.getShape() ||
        prev_flt.getWidth() >= cur_flt.getWidth()) {
      return false;
    }
    out_step = RowWiseStep{RowWiseStepKind::kWidenFloatConvert,
                           op,
                           cur_flt,
                           /*row_vec_1d=*/nullptr,
                           /*zero_scalar_attr=*/nullptr,
                           /*prev_is_lhs=*/true,
                           /*zero_on_true=*/true};
    out_prev = prev;
    return true;
  }
  if (auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(op)) {
    return TryMatchRowScaleMulStep(mul, rows, cur_type.getElementType(),
                                   gather_op, out_step, out_prev);
  }
  if (auto sel = mlir::dyn_cast<mlir::stablehlo::SelectOp>(op)) {
    return TryMatchRowMaskSelectStep(sel, rows, cur_type.getElementType(),
                                     gather_op, out_step, out_prev);
  }
  return false;
}

mlir::Value GatherOrUnwrapInversePerm1D(mlir::OpBuilder& builder,
                                        mlir::Location loc, mlir::Value v_1d,
                                        mlir::Value gather_indices, int64_t m,
                                        int64_t n, int64_t index_vector_dim) {
  auto v_type = mlir::cast<mlir::RankedTensorType>(v_1d.getType());
  mlir::Type elem_type = v_type.getElementType();
  auto vec_m_type = mlir::RankedTensorType::get({m}, elem_type);
  if (v_1d.getType() != vec_m_type) {
    v_1d = builder.create<mlir::stablehlo::ReshapeOp>(loc, vec_m_type, v_1d)
               .getResult();
  }

  if (m == n) {
    if (auto perm_1d = MatchInversePermutation1D(gather_indices, n)) {
      if (auto v_gather = v_1d.getDefiningOp<mlir::stablehlo::GatherOp>();
          v_gather && HasStaticNumElements(v_gather.getResult(), n) &&
          Unwrap1DIndexTensor(v_gather.getStartIndices(), n) == *perm_1d &&
          v_gather.getOperand().getType() == vec_m_type) {
        return v_gather.getOperand();
      }
      auto v_sort = MatchSortOutput(v_1d);
      auto p_sort = MatchSortOutput(*perm_1d);
      if (v_sort.has_value() && p_sort.has_value() &&
          v_sort->sort_op == p_sort->sort_op &&
          v_sort->sort_op.getDimension() == 0 &&
          UnwrapsTo1DIota(v_sort->sort_op.getInputs()[p_sort->result_index],
                          n)) {
        mlir::Value orig_1d = v_sort->sort_op.getInputs()[v_sort->result_index];
        if (orig_1d.getType() == vec_m_type) {
          return orig_1d;
        }
      }
    }
  }

  auto out_1d_type = mlir::RankedTensorType::get({n}, elem_type);
  auto dnums = mlir::stablehlo::GatherDimensionNumbersAttr::get(
      builder.getContext(),
      /*offsetDims=*/{},
      /*collapsedSliceDims=*/{0},
      /*operandBatchingDims=*/{},
      /*startIndicesBatchingDims=*/{},
      /*startIndexMap=*/{0},
      /*indexVectorDim=*/index_vector_dim);
  return builder
      .create<mlir::stablehlo::GatherOp>(loc, out_1d_type, v_1d, gather_indices,
                                         dnums,
                                         builder.getDenseI64ArrayAttr({1}),
                                         /*indices_are_sorted=*/false)
      .getResult();
}

mlir::Value ReplayRowWiseStepsOnGathered(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value base_val,
    mlir::stablehlo::GatherOp gather_op, int64_t m, int64_t n,
    llvm::ArrayRef<int64_t> slice_shape, llvm::ArrayRef<RowWiseStep> steps) {
  auto base_type = mlir::cast<mlir::RankedTensorType>(base_val.getType());
  llvm::SmallVector<int64_t, 6> gathered_shape;
  gathered_shape.push_back(n);
  gathered_shape.append(slice_shape.begin(), slice_shape.end());
  auto gathered_base_type =
      mlir::RankedTensorType::get(gathered_shape, base_type.getElementType());

  mlir::Value acc =
      builder
          .create<mlir::stablehlo::GatherOp>(
              loc, gathered_base_type, base_val, gather_op.getStartIndices(),
              gather_op.getDimensionNumbersAttr(),
              gather_op.getSliceSizesAttr(), gather_op.getIndicesAreSorted())
          .getResult();

  const int64_t index_vector_dim =
      gather_op.getDimensionNumbers().getIndexVectorDim();
  for (const RowWiseStep& step : llvm::reverse(steps)) {
    auto out_tensor_type =
        mlir::RankedTensorType::get(gathered_shape, step.dst_elem_type);
    if (step.kind == RowWiseStepKind::kWidenFloatConvert) {
      acc =
          builder.create<mlir::stablehlo::ConvertOp>(loc, out_tensor_type, acc)
              .getResult();
    } else if (step.kind == RowWiseStepKind::kRowScaleMul) {
      mlir::Value gathered_scale_1d = GatherOrUnwrapInversePerm1D(
          builder, loc, step.row_vec_1d, gather_op.getStartIndices(), m, n,
          index_vector_dim);
      mlir::Value scale_nd = builder
                                 .create<mlir::stablehlo::BroadcastInDimOp>(
                                     loc, out_tensor_type, gathered_scale_1d,
                                     builder.getDenseI64ArrayAttr({0}))
                                 .getResult();
      acc = step.prev_is_lhs ? builder
                                   .create<mlir::stablehlo::MulOp>(
                                       loc, out_tensor_type, acc, scale_nd)
                                   .getResult()
                             : builder
                                   .create<mlir::stablehlo::MulOp>(
                                       loc, out_tensor_type, scale_nd, acc)
                                   .getResult();
    } else {
      mlir::Value gathered_mask_1d = GatherOrUnwrapInversePerm1D(
          builder, loc, step.row_vec_1d, gather_op.getStartIndices(), m, n,
          index_vector_dim);
      auto pred_nd_type =
          mlir::RankedTensorType::get(gathered_shape, builder.getI1Type());
      mlir::Value mask_nd = builder
                                .create<mlir::stablehlo::BroadcastInDimOp>(
                                    loc, pred_nd_type, gathered_mask_1d,
                                    builder.getDenseI64ArrayAttr({0}))
                                .getResult();
      mlir::Value zero_nd =
          builder
              .create<mlir::stablehlo::ConstantOp>(
                  loc, mlir::DenseElementsAttr::get(out_tensor_type,
                                                    step.zero_scalar_attr))
              .getResult();
      acc = step.zero_on_true
                ? builder
                      .create<mlir::stablehlo::SelectOp>(loc, out_tensor_type,
                                                         mask_nd, zero_nd, acc)
                      .getResult()
                : builder
                      .create<mlir::stablehlo::SelectOp>(loc, out_tensor_type,
                                                         mask_nd, acc, zero_nd)
                      .getResult();
    }
  }
  return acc;
}

void TrySinkSingleGather(mlir::stablehlo::GatherOp gather_op,
                         mlir::Block& block) {
  int64_t m = 0;
  int64_t n = 0;
  llvm::SmallVector<int64_t, 6> slice_shape;
  if (!gather_op->getBlock() ||
      !IsRowGatherAlongDim0(gather_op, m, n, slice_shape)) {
    return;
  }

  llvm::SmallVector<RowWiseStep> steps;
  mlir::Value cur = gather_op.getOperand();
  bool saw_widen_convert = false;
  for (int depth = 0; depth < kMaxAnalysisRecursionDepth; ++depth) {
    RowWiseStep step;
    mlir::Value prev;
    if (!TryMatchSingleRowWiseStep(cur, m, slice_shape, block, gather_op, step,
                                   prev)) {
      break;
    }
    if (step.kind == RowWiseStepKind::kWidenFloatConvert) {
      saw_widen_convert = true;
    }
    steps.push_back(step);
    cur = prev;
  }
  if (!saw_widen_convert || !EnsureAvailableBefore(cur, gather_op)) {
    return;
  }

  mlir::OpBuilder builder(gather_op);
  mlir::Value old_operand = gather_op.getOperand();
  mlir::Value sunk = ReplayRowWiseStepsOnGathered(
      builder, gather_op.getLoc(), cur, gather_op, m, n, slice_shape, steps);
  gather_op.getResult().replaceAllUsesWith(sunk);
  gather_op.erase();
  EraseDeadOpsUpward({old_operand});
}

void SinkRowWiseOpsPastGather(mlir::Block& block) {
  llvm::SmallVector<mlir::stablehlo::GatherOp> gathers;
  for (mlir::Operation& op : block) {
    if (auto g = mlir::dyn_cast<mlir::stablehlo::GatherOp>(op)) {
      gathers.push_back(g);
    }
  }
  for (mlir::stablehlo::GatherOp gather_op : gathers) {
    TrySinkSingleGather(gather_op, block);
  }
}

/// Mechanism 3: Lower MoE Scatter to Gather.
bool IsRowAddScatter(mlir::stablehlo::ScatterOp scatter_op, int64_t& out_rows,
                     int64_t& in_rows,
                     llvm::SmallVectorImpl<int64_t>& slice_shape) {
  if (!scatter_op->getBlock() || scatter_op.getInputs().size() != 1 ||
      scatter_op.getUpdates().size() != 1 || scatter_op.getNumResults() != 1) {
    return false;
  }
  mlir::Value zeros = scatter_op.getInputs()[0];
  mlir::Value updates = scatter_op.getUpdates()[0];
  auto zeros_type = mlir::dyn_cast<mlir::RankedTensorType>(zeros.getType());
  auto updates_type = mlir::dyn_cast<mlir::RankedTensorType>(updates.getType());
  auto res_type =
      mlir::dyn_cast<mlir::RankedTensorType>(scatter_op.getResult(0).getType());
  if (!zeros_type || !updates_type || !res_type ||
      !zeros_type.hasStaticShape() || !updates_type.hasStaticShape() ||
      zeros_type.getRank() < 2 ||
      updates_type.getRank() != zeros_type.getRank() ||
      zeros_type != res_type ||
      zeros_type.getElementType() != updates_type.getElementType()) {
    return false;
  }
  const int64_t rank = zeros_type.getRank();
  const int64_t o_rows = zeros_type.getDimSize(0);
  const int64_t i_rows = updates_type.getDimSize(0);
  if (o_rows < 1 || i_rows < 1 || i_rows > kMaxSignedInt32 ||
      !HasStaticNumElements(scatter_op.getScatterIndices(), i_rows)) {
    return false;
  }
  slice_shape.clear();
  for (int64_t d = 1; d < rank; ++d) {
    if (updates_type.getDimSize(d) != zeros_type.getDimSize(d) ||
        zeros_type.getDimSize(d) < 1) {
      return false;
    }
    slice_shape.push_back(zeros_type.getDimSize(d));
  }
  mlir::stablehlo::ScatterDimensionNumbersAttr dnums =
      scatter_op.getScatterDimensionNumbers();
  llvm::SmallVector<int64_t, 6> expected_update_window_dims;
  for (int64_t d = 1; d < rank; ++d) {
    expected_update_window_dims.push_back(d);
  }
  if (dnums.getUpdateWindowDims() !=
          llvm::ArrayRef<int64_t>(expected_update_window_dims) ||
      dnums.getInsertedWindowDims() != llvm::ArrayRef<int64_t>{0} ||
      !dnums.getInputBatchingDims().empty() ||
      !dnums.getScatterIndicesBatchingDims().empty() ||
      dnums.getScatterDimsToOperandDims() != llvm::ArrayRef<int64_t>{0} ||
      dnums.getIndexVectorDim() != 1 || !IsZeroTensor(zeros) ||
      !IsScalarAddRegion(scatter_op.getUpdateComputation())) {
    return false;
  }
  out_rows = o_rows;
  in_rows = i_rows;
  return true;
}

mlir::Value CreateRowGather(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value src_nd, mlir::Value idx_2d,
                            int64_t out_rows,
                            llvm::ArrayRef<int64_t> slice_shape) {
  auto src_type = mlir::cast<mlir::RankedTensorType>(src_nd.getType());
  llvm::SmallVector<int64_t, 6> out_shape;
  out_shape.push_back(out_rows);
  out_shape.append(slice_shape.begin(), slice_shape.end());
  auto out_type =
      mlir::RankedTensorType::get(out_shape, src_type.getElementType());

  const int64_t rank = static_cast<int64_t>(out_shape.size());
  llvm::SmallVector<int64_t, 6> offset_dims;
  for (int64_t d = 1; d < rank; ++d) {
    offset_dims.push_back(d);
  }
  auto dnums = mlir::stablehlo::GatherDimensionNumbersAttr::get(
      builder.getContext(), offset_dims,
      /*collapsedSliceDims=*/{0},
      /*operandBatchingDims=*/{},
      /*startIndicesBatchingDims=*/{},
      /*startIndexMap=*/{0},
      /*indexVectorDim=*/1);

  llvm::SmallVector<int64_t, 6> slice_sizes;
  slice_sizes.push_back(1);
  slice_sizes.append(slice_shape.begin(), slice_shape.end());

  return builder
      .create<mlir::stablehlo::GatherOp>(
          loc, out_type, src_nd, idx_2d, dnums,
          builder.getDenseI64ArrayAttr(slice_sizes),
          /*indices_are_sorted=*/false)
      .getResult();
}

mlir::Value TryPeelWideningFloatConvert(mlir::Value val) {
  auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>();
  if (!conv) {
    return val;
  }
  auto dst_type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  auto src_type =
      mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
  auto dst_flt =
      dst_type ? mlir::dyn_cast<mlir::FloatType>(dst_type.getElementType())
               : nullptr;
  auto src_flt =
      src_type ? mlir::dyn_cast<mlir::FloatType>(src_type.getElementType())
               : nullptr;
  if (src_flt && dst_flt && src_type.hasStaticShape() &&
      src_type.getShape() == dst_type.getShape() &&
      src_flt.getWidth() < dst_flt.getWidth()) {
    return conv.getOperand();
  }
  return val;
}

mlir::Value ExtractBroadcastOperand(mlir::stablehlo::BroadcastInDimOp bcast,
                                    int64_t t,
                                    llvm::ArrayRef<int64_t> slice_shape) {
  mlir::Value bcast_in = bcast.getOperand();
  auto in_type = mlir::dyn_cast<mlir::RankedTensorType>(bcast_in.getType());
  const int64_t rank = static_cast<int64_t>(slice_shape.size()) + 1;
  llvm::SmallVector<int64_t, 6> expected_bcast_dims;
  expected_bcast_dims.push_back(0);
  for (int64_t d = 1; d < rank; ++d) {
    expected_bcast_dims.push_back(d + 1);
  }
  if (in_type && in_type.hasStaticShape() && in_type.getRank() == rank &&
      in_type.getDimSize(0) == t &&
      bcast.getBroadcastDimensions() ==
          llvm::ArrayRef<int64_t>(expected_bcast_dims)) {
    for (size_t i = 0; i < slice_shape.size(); ++i) {
      if (in_type.getDimSize(i + 1) != slice_shape[i]) {
        return nullptr;
      }
    }
    return bcast_in;
  }
  return nullptr;
}

bool TryUnwrapBroadcastedTokenSource(mlir::Value updates, int64_t n,
                                     llvm::ArrayRef<int64_t> slice_shape,
                                     mlir::Operation* target_op,
                                     mlir::Value& out_src, int64_t& out_k) {
  mlir::Value cur = TryPeelWideningFloatConvert(updates);
  if (auto reshape = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    auto bcast =
        reshape.getOperand().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    auto bcast_type =
        bcast ? mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType())
              : nullptr;
    if (bcast && bcast_type && bcast_type.hasStaticShape() &&
        bcast_type.getRank() == static_cast<int64_t>(slice_shape.size()) + 2) {
      const int64_t t = bcast_type.getDimSize(0);
      const int64_t k = bcast_type.getDimSize(1);
      if (t >= 1 && k >= 1 && t * k == n) {
        if (mlir::Value cand_td =
                ExtractBroadcastOperand(bcast, t, slice_shape)) {
          cand_td = TryPeelWideningFloatConvert(cand_td);
          if (EnsureAvailableBefore(cand_td, target_op)) {
            out_src = cand_td;
            out_k = k;
            return true;
          }
        }
      }
    }
  }
  out_src = EnsureAvailableBefore(cur, target_op) ? cur : updates;
  out_k = 1;
  return true;
}

bool TryLowerBijectiveInversePermScatter(mlir::stablehlo::ScatterOp scatter_op,
                                         int64_t n,
                                         llvm::ArrayRef<int64_t> slice_shape) {
  auto perm_1d = MatchInversePermutation1D(scatter_op.getScatterIndices(), n);
  if (!perm_1d.has_value() || !EnsureAvailableBefore(*perm_1d, scatter_op)) {
    return false;
  }

  mlir::Value gather_src;
  int64_t k = 1;
  TryUnwrapBroadcastedTokenSource(scatter_op.getUpdates()[0], n, slice_shape,
                                  scatter_op, gather_src, k);

  mlir::OpBuilder builder(scatter_op);
  mlir::Location loc = scatter_op.getLoc();
  mlir::Value perm_i32 = ConvertOrReshapeToI32(builder, loc, *perm_1d, {n});
  mlir::Value row_idx_1d = perm_i32;
  if (k > 1) {
    auto n_i32_type = mlir::RankedTensorType::get({n}, builder.getI32Type());
    auto k_const = builder.create<mlir::stablehlo::ConstantOp>(
        loc,
        mlir::DenseIntElementsAttr::get(n_i32_type, static_cast<int32_t>(k)));
    row_idx_1d = builder
                     .create<mlir::stablehlo::DivOp>(loc, n_i32_type, perm_i32,
                                                     k_const.getResult())
                     .getResult();
  }
  auto idx_2d_type = mlir::RankedTensorType::get({n, 1}, builder.getI32Type());
  mlir::Value idx_2d =
      builder.create<mlir::stablehlo::ReshapeOp>(loc, idx_2d_type, row_idx_1d)
          .getResult();

  mlir::Value gathered =
      CreateRowGather(builder, loc, gather_src, idx_2d, n, slice_shape);
  auto res_type =
      mlir::cast<mlir::RankedTensorType>(scatter_op.getResult(0).getType());
  if (gathered.getType() != res_type) {
    gathered =
        builder.create<mlir::stablehlo::ConvertOp>(loc, res_type, gathered)
            .getResult();
  }

  mlir::Value old_updates = scatter_op.getUpdates()[0];
  mlir::Value old_indices = scatter_op.getScatterIndices();
  scatter_op.getResult(0).replaceAllUsesWith(gathered);
  scatter_op.erase();
  EraseDeadOpsUpward({old_updates, old_indices});
  return true;
}

void LowerMoeScatterToGather(mlir::Block& block) {
  llvm::SmallVector<mlir::stablehlo::ScatterOp> scatters;
  for (mlir::Operation& op : block) {
    if (auto s = mlir::dyn_cast<mlir::stablehlo::ScatterOp>(op)) {
      scatters.push_back(s);
    }
  }
  for (mlir::stablehlo::ScatterOp scatter_op : scatters) {
    int64_t out_rows = 0;
    int64_t in_rows = 0;
    llvm::SmallVector<int64_t, 6> slice_shape;
    if (!IsRowAddScatter(scatter_op, out_rows, in_rows, slice_shape)) {
      continue;
    }
    if (out_rows == in_rows) {
      TryLowerBijectiveInversePermScatter(scatter_op, in_rows, slice_shape);
    }
  }
}

struct CanonicalizeMoEGatherScatterPass
    : public mlir::PassWrapper<CanonicalizeMoEGatherScatterPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeMoEGatherScatterPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-canonicalize-moe-gather-scatter";
  }

  llvm::StringRef getDescription() const override {
    return "Canonicalizes mixture-of-experts gather and scatter patterns into "
           "Gather.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(&getContext());

    llvm::SmallVector<mlir::stablehlo::GatherOp, 16> identity_gathers;
    func.walk([&identity_gathers](mlir::stablehlo::GatherOp op) {
      identity_gathers.push_back(op);
    });
    for (mlir::stablehlo::GatherOp gather_op : identity_gathers) {
      TryEliminateIdentityOrConstantGather(gather_op, rewriter);
    }

    for (mlir::Block& block : func) {
      SinkRowWiseOpsPastGather(block);
      LowerMoeScatterToGather(block);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createCanonicalizeMoEGatherScatterPass() {
  return std::make_unique<CanonicalizeMoEGatherScatterPass>();
}

}  // namespace torch_tpu
