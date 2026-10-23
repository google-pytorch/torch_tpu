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

#include "csrc/mlir/passes/canonicalize_broadcast_contractions_and_unit_reductions.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

bool IsStaticRankedTensor(mlir::Type type,
                          mlir::RankedTensorType* out = nullptr) {
  auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!rtt || !rtt.hasStaticShape()) {
    return false;
  }
  if (out != nullptr) {
    *out = rtt;
  }
  return true;
}

bool IsSplatConstantValue(mlir::Value val, double target) {
  while (val) {
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      val = bcast.getOperand();
    } else if (auto reshape = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      val = reshape.getOperand();
    } else if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      val = conv.getOperand();
    } else {
      break;
    }
  }
  auto const_op =
      val ? val.getDefiningOp<mlir::stablehlo::ConstantOp>() : nullptr;
  if (!const_op) {
    return false;
  }
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr || !attr.isSplat()) {
    return false;
  }
  mlir::Type elem_ty = attr.getElementType();
  if (mlir::isa<mlir::FloatType>(elem_ty)) {
    double v = attr.getSplatValue<llvm::APFloat>().convertToDouble();
    return v == target;
  }
  if (mlir::isa<mlir::IntegerType>(elem_ty)) {
    if (elem_ty.isInteger(1)) {
      double v = attr.getSplatValue<llvm::APInt>().getBoolValue() ? 1.0 : 0.0;
      return v == target;
    }
    int64_t v = attr.getSplatValue<llvm::APInt>().getSExtValue();
    return static_cast<double>(v) == target;
  }
  return false;
}

template <typename BinaryOp>
bool IsSingleBinaryReduceBody(mlir::stablehlo::ReduceOp reduce_op) {
  if (reduce_op.getInputs().size() != 1 ||
      reduce_op.getInitValues().size() != 1) {
    return false;
  }
  mlir::Region& body = reduce_op.getBody();
  if (!body.hasOneBlock()) {
    return false;
  }
  mlir::Block& block = body.front();
  // GENERIC: semantic Single-input StableHLO reduction block takes 2 arguments
  // (accumulator and element) and contains 2 ops (binary op and return).
  if (block.getNumArguments() != 2) {
    return false;
  }
  auto& ops = block.getOperations();
  // GENERIC: semantic Reduction block contains exactly 2 ops: binary op and
  // return terminator.
  if (ops.size() != 2) {
    return false;
  }
  auto bin_op = mlir::dyn_cast<BinaryOp>(&ops.front());
  auto ret_op = mlir::dyn_cast<mlir::stablehlo::ReturnOp>(&ops.back());
  if (!bin_op || !ret_op || ret_op.getResults().size() != 1 ||
      ret_op.getResults()[0] != bin_op.getResult()) {
    return false;
  }
  mlir::Value arg0 = block.getArgument(0);
  mlir::Value arg1 = block.getArgument(1);
  return (bin_op.getLhs() == arg0 && bin_op.getRhs() == arg1) ||
         (bin_op.getLhs() == arg1 && bin_op.getRhs() == arg0);
}

bool IsIdentityBroadcast(mlir::stablehlo::BroadcastInDimOp bcast) {
  mlir::RankedTensorType in_ty, out_ty;
  if (!IsStaticRankedTensor(bcast.getOperand().getType(), &in_ty) ||
      !IsStaticRankedTensor(bcast.getType(), &out_ty) || in_ty != out_ty) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = bcast.getBroadcastDimensions();
  if (static_cast<int64_t>(dims.size()) != out_ty.getRank()) {
    return false;
  }
  for (int64_t i = 0; i < out_ty.getRank(); ++i) {
    if (dims[i] != i) {
      return false;
    }
  }
  return true;
}

mlir::Value SkipIdentityBroadcasts(mlir::Value val) {
  if (!val) {
    return val;
  }
  while (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    if (!IsIdentityBroadcast(bcast)) {
      break;
    }
    val = bcast.getOperand();
  }
  return val;
}

bool IsUnitOnlyReshape(mlir::RankedTensorType in_ty,
                       mlir::RankedTensorType out_ty,
                       llvm::SmallVectorImpl<int64_t>* out_to_in_non_unit) {
  llvm::SmallVector<int64_t, 8> in_non_unit;
  llvm::SmallVector<int64_t, 8> out_non_unit;
  for (int64_t i = 0; i < in_ty.getRank(); ++i) {
    if (in_ty.getDimSize(i) != 1) {
      in_non_unit.push_back(i);
    }
  }
  for (int64_t i = 0; i < out_ty.getRank(); ++i) {
    if (out_ty.getDimSize(i) != 1) {
      out_non_unit.push_back(i);
    }
  }
  if (in_non_unit.size() != out_non_unit.size()) {
    return false;
  }
  for (size_t k = 0; k < in_non_unit.size(); ++k) {
    if (in_ty.getDimSize(in_non_unit[k]) !=
        out_ty.getDimSize(out_non_unit[k])) {
      return false;
    }
  }
  if (out_to_in_non_unit != nullptr) {
    out_to_in_non_unit->assign(out_ty.getRank(), -1);
    for (size_t k = 0; k < out_non_unit.size(); ++k) {
      (*out_to_in_non_unit)[out_non_unit[k]] = in_non_unit[k];
    }
  }
  return true;
}

bool IsUnitDimensionAddReduce(mlir::stablehlo::ReduceOp reduce_op) {
  if (!IsSingleBinaryReduceBody<mlir::stablehlo::AddOp>(reduce_op) ||
      !IsSplatConstantValue(reduce_op.getInitValues()[0], 0.0)) {
    return false;
  }
  mlir::RankedTensorType in_ty;
  if (!IsStaticRankedTensor(reduce_op.getInputs()[0].getType(), &in_ty)) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = reduce_op.getDimensions();
  if (dims.empty()) {
    return false;
  }
  for (int64_t d : dims) {
    if (d < 0 || d >= in_ty.getRank() || in_ty.getDimSize(d) != 1) {
      return false;
    }
  }
  return true;
}

mlir::Value UnwrapUnitReshapesAndUnitReduces(mlir::Value val) {
  while (val) {
    val = SkipIdentityBroadcasts(val);
    if (auto reshape = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      mlir::RankedTensorType in_ty, out_ty;
      if (IsStaticRankedTensor(reshape.getOperand().getType(), &in_ty) &&
          IsStaticRankedTensor(reshape.getType(), &out_ty) &&
          IsUnitOnlyReshape(in_ty, out_ty, nullptr)) {
        val = reshape.getOperand();
        continue;
      }
    }
    if (auto reduce = val.getDefiningOp<mlir::stablehlo::ReduceOp>()) {
      if (IsUnitDimensionAddReduce(reduce)) {
        val = reduce.getInputs()[0];
        continue;
      }
    }
    break;
  }
  return val;
}

struct GroupRepeatInfo {
  mlir::Value unrepeated_src;
  int64_t group_size = 1;
  int64_t repeat_size = 1;
};

std::optional<GroupRepeatInfo> MatchBatchDimGroupRepeat(mlir::Value operand,
                                                        int64_t batch_dim) {
  mlir::RankedTensorType op_ty;
  if (!IsStaticRankedTensor(operand.getType(), &op_ty) || batch_dim < 0 ||
      batch_dim >= op_ty.getRank()) {
    return std::nullopt;
  }
  int64_t batch_size = op_ty.getDimSize(batch_dim);
  if (batch_size <= 1) {
    return std::nullopt;
  }
  int64_t prefix_prod = 1;
  for (int64_t i = 0; i < batch_dim; ++i) {
    prefix_prod *= op_ty.getDimSize(i);
  }
  int64_t suffix_prod = 1;
  for (int64_t i = batch_dim + 1; i < op_ty.getRank(); ++i) {
    suffix_prod *= op_ty.getDimSize(i);
  }

  mlir::Value cur = operand;
  while (true) {
    cur = SkipIdentityBroadcasts(cur);
    if (auto reshape = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      cur = reshape.getOperand();
      continue;
    }
    break;
  }
  auto bcast = cur.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast || IsIdentityBroadcast(bcast)) {
    return std::nullopt;
  }
  mlir::RankedTensorType bcast_out_ty, bcast_in_ty;
  if (!IsStaticRankedTensor(bcast.getType(), &bcast_out_ty) ||
      !IsStaticRankedTensor(bcast.getOperand().getType(), &bcast_in_ty)) {
    return std::nullopt;
  }

  // Locate two adjacent dimensions [d_group, d_repeat] in bcast_out_ty whose
  // prefix product matches prefix_prod, whose product is batch_size, and whose
  // suffix product matches suffix_prod.
  int64_t running_prefix = 1;
  int64_t d_group = -1;
  for (int64_t d = 0; d + 1 < bcast_out_ty.getRank(); ++d) {
    int64_t g = bcast_out_ty.getDimSize(d);
    int64_t r = bcast_out_ty.getDimSize(d + 1);
    if (running_prefix == prefix_prod && g > 1 && r > 1 &&
        g * r == batch_size) {
      int64_t rem = 1;
      for (int64_t k = d + 2; k < bcast_out_ty.getRank(); ++k) {
        rem *= bcast_out_ty.getDimSize(k);
      }
      if (rem == suffix_prod) {
        d_group = d;
        break;
      }
    }
    running_prefix *= g;
  }
  if (d_group < 0) {
    return std::nullopt;
  }
  int64_t d_repeat = d_group + 1;

  llvm::SmallVector<bool, 8> in_bcast_dims(bcast_out_ty.getRank(), false);
  llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
  if (static_cast<int64_t>(bdims.size()) != bcast_in_ty.getRank()) {
    return std::nullopt;
  }
  for (int64_t i = 0; i < bcast_in_ty.getRank(); ++i) {
    int64_t out_d = bdims[i];
    if (out_d < 0 || out_d >= bcast_out_ty.getRank() ||
        bcast_in_ty.getDimSize(i) != bcast_out_ty.getDimSize(out_d)) {
      return std::nullopt;
    }
    if (i > 0 && bdims[i] <= bdims[i - 1]) {
      return std::nullopt;
    }
    in_bcast_dims[out_d] = true;
  }
  if (!in_bcast_dims[d_group] || in_bcast_dims[d_repeat]) {
    return std::nullopt;
  }
  for (int64_t d = 0; d < bcast_out_ty.getRank(); ++d) {
    if (d == d_repeat) {
      continue;
    }
    if (bcast_out_ty.getDimSize(d) > 1 && !in_bcast_dims[d]) {
      return std::nullopt;
    }
  }
  return GroupRepeatInfo{
      bcast.getOperand(),
      bcast_out_ty.getDimSize(d_group),
      bcast_out_ty.getDimSize(d_repeat),
  };
}

// Matches `val` (rank R) produced by `TransposeOp(ReshapeOp(bSrc), perm)`
// followed by dropping trailing unit dimensions (via `ReduceOp` or
// `ReshapeOp`).
bool MatchUnitPaddedTranspose(mlir::Value val, int64_t expected_rank,
                              mlir::Value* b_src_out,
                              llvm::SmallVectorImpl<int64_t>* val_to_b_dim) {
  val = SkipIdentityBroadcasts(val);
  auto tr = val.getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!tr) {
    return false;
  }
  mlir::Value tr_in = SkipIdentityBroadcasts(tr.getOperand());
  auto reshape = tr_in.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!reshape) {
    return false;
  }
  mlir::Value b_src = reshape.getOperand();
  mlir::RankedTensorType tr_out_ty, tr_in_ty, b_src_ty;
  if (!IsStaticRankedTensor(tr.getType(), &tr_out_ty) ||
      !IsStaticRankedTensor(tr_in.getType(), &tr_in_ty) ||
      !IsStaticRankedTensor(b_src.getType(), &b_src_ty) ||
      b_src_ty.getRank() != expected_rank ||
      tr_out_ty.getRank() < expected_rank) {
    return false;
  }
  for (int64_t d = expected_rank; d < tr_out_ty.getRank(); ++d) {
    if (tr_out_ty.getDimSize(d) != 1) {
      return false;
    }
  }
  llvm::SmallVector<int64_t, 8> r_out_to_in;
  if (!IsUnitOnlyReshape(b_src_ty, tr_in_ty, &r_out_to_in)) {
    return false;
  }
  llvm::ArrayRef<int64_t> perm = tr.getPermutation();
  val_to_b_dim->assign(expected_rank, -1);
  llvm::SmallVector<bool, 8> used_b(expected_rank, false);
  for (int64_t d = 0; d < expected_rank; ++d) {
    int64_t in_d = perm[d];
    if (in_d >= 0 && in_d < tr_in_ty.getRank() && r_out_to_in[in_d] >= 0) {
      (*val_to_b_dim)[d] = r_out_to_in[in_d];
      used_b[r_out_to_in[in_d]] = true;
    }
  }
  for (int64_t d = 0; d < expected_rank; ++d) {
    if ((*val_to_b_dim)[d] == -1) {
      for (int64_t b_d = 0; b_d < expected_rank; ++b_d) {
        if (!used_b[b_d] &&
            b_src_ty.getDimSize(b_d) == tr_out_ty.getDimSize(d)) {
          (*val_to_b_dim)[d] = b_d;
          used_b[b_d] = true;
          break;
        }
      }
    }
    if ((*val_to_b_dim)[d] < 0) {
      return false;
    }
  }
  *b_src_out = b_src;
  return true;
}

bool TryAbsorbBatchTransposeAcrossMul(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value* lhs_val_io,
    llvm::SmallVectorImpl<int64_t>* lhs_batch_dims,
    llvm::SmallVectorImpl<int64_t>* lhs_free_dims,
    llvm::SmallVectorImpl<int64_t>* lhs_contract_dims) {
  mlir::Value unwrapped = UnwrapUnitReshapesAndUnitReduces(*lhs_val_io);
  auto mul_op = unwrapped.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul_op) {
    return false;
  }
  mlir::RankedTensorType lhs_ty;
  if (!IsStaticRankedTensor(lhs_val_io->getType(), &lhs_ty)) {
    return false;
  }
  int64_t rank = lhs_ty.getRank();

  mlir::Value m_lhs = UnwrapUnitReshapesAndUnitReduces(mul_op.getLhs());
  mlir::Value m_rhs = SkipIdentityBroadcasts(mul_op.getRhs());
  auto tr_a = m_lhs.getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!tr_a) {
    return false;
  }
  mlir::Value a_src = tr_a.getOperand();
  mlir::RankedTensorType a_src_ty;
  if (!IsStaticRankedTensor(a_src.getType(), &a_src_ty) ||
      a_src_ty.getRank() != rank) {
    return false;
  }
  llvm::ArrayRef<int64_t> perm_a = tr_a.getPermutation();
  if (static_cast<int64_t>(perm_a.size()) != rank) {
    return false;
  }

  mlir::Value b_src;
  llvm::SmallVector<int64_t, 8> val_to_b;
  if (!MatchUnitPaddedTranspose(m_rhs, rank, &b_src, &val_to_b)) {
    return false;
  }
  mlir::RankedTensorType b_src_ty;
  if (!IsStaticRankedTensor(b_src.getType(), &b_src_ty) ||
      b_src_ty.getRank() != rank) {
    return false;
  }

  // Verify that the free and contracting dimensions of `Y_diag` are already at
  // the exact same positions in `a_src` and `b_src`, so aligning `a_src` to
  // `b_src` only permutes batch dimensions.
  for (int64_t d : *lhs_free_dims) {
    if (perm_a[d] != val_to_b[d]) {
      return false;
    }
  }
  for (int64_t d : *lhs_contract_dims) {
    if (perm_a[d] != val_to_b[d]) {
      return false;
    }
  }

  llvm::SmallVector<int64_t, 8> perm_a_to_b(rank, -1);
  for (int64_t d = 0; d < rank; ++d) {
    perm_a_to_b[val_to_b[d]] = perm_a[d];
  }
  for (int64_t p : perm_a_to_b) {
    if (p < 0) {
      return false;
    }
  }

  builder.setInsertionPoint(mul_op);
  mlir::Value a_aligned = builder.create<mlir::stablehlo::TransposeOp>(
      loc, b_src_ty, a_src, builder.getDenseI64ArrayAttr(perm_a_to_b));

  *lhs_val_io =
      builder.create<mlir::stablehlo::MulOp>(loc, b_src_ty, a_aligned, b_src);
  for (int64_t& d : *lhs_batch_dims) {
    d = val_to_b[d];
  }
  for (int64_t& d : *lhs_free_dims) {
    d = val_to_b[d];
  }
  for (int64_t& d : *lhs_contract_dims) {
    d = val_to_b[d];
  }
  return true;
}

bool TryRewriteBroadcastMulReduceToDot(mlir::stablehlo::ReduceOp reduce_op) {
  if (!IsSingleBinaryReduceBody<mlir::stablehlo::AddOp>(reduce_op) ||
      !IsSplatConstantValue(reduce_op.getInitValues()[0], 0.0)) {
    return false;
  }
  mlir::Value red_in = SkipIdentityBroadcasts(reduce_op.getInputs()[0]);
  auto mul_op = red_in.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul_op) {
    return false;
  }
  auto lhs_bcast =
      mul_op.getLhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  auto rhs_bcast =
      mul_op.getRhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!lhs_bcast || !rhs_bcast) {
    return false;
  }

  mlir::RankedTensorType mul_ty, lhs_in_ty, rhs_in_ty, out_ty;
  if (!IsStaticRankedTensor(mul_op.getType(), &mul_ty) ||
      !IsStaticRankedTensor(lhs_bcast.getOperand().getType(), &lhs_in_ty) ||
      !IsStaticRankedTensor(rhs_bcast.getOperand().getType(), &rhs_in_ty) ||
      !IsStaticRankedTensor(reduce_op.getResultTypes()[0], &out_ty)) {
    return false;
  }

  int64_t mul_rank = mul_ty.getRank();
  llvm::SmallVector<bool, 8> is_reduced(mul_rank, false);
  for (int64_t d : reduce_op.getDimensions()) {
    if (d < 0 || d >= mul_rank || mul_ty.getDimSize(d) <= 1) {
      return false;
    }
    is_reduced[d] = true;
  }

  auto build_dim_map = [mul_rank, mul_ty](
                           mlir::stablehlo::BroadcastInDimOp bcast,
                           mlir::RankedTensorType in_ty,
                           llvm::SmallVectorImpl<int64_t>* mul_to_in,
                           llvm::SmallVectorImpl<int64_t>* sq_shape,
                           bool* needs_squeeze) -> bool {
    mul_to_in->assign(mul_rank, -1);
    sq_shape->clear();
    *needs_squeeze = false;
    llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
    if (static_cast<int64_t>(bdims.size()) != in_ty.getRank()) {
      return false;
    }
    int64_t sq_idx = 0;
    for (int64_t i = 0; i < in_ty.getRank(); ++i) {
      int64_t md = bdims[i];
      if (md < 0 || md >= mul_rank) {
        return false;
      }
      if (in_ty.getDimSize(i) == mul_ty.getDimSize(md)) {
        (*mul_to_in)[md] = sq_idx++;
        sq_shape->push_back(in_ty.getDimSize(i));
      } else if (in_ty.getDimSize(i) == 1 && mul_ty.getDimSize(md) > 1) {
        *needs_squeeze = true;
      } else {
        return false;
      }
    }
    return true;
  };

  llvm::SmallVector<int64_t, 8> lhs_map, rhs_map;
  llvm::SmallVector<int64_t, 8> lhs_sq_shape, rhs_sq_shape;
  bool lhs_needs_squeeze = false;
  bool rhs_needs_squeeze = false;
  if (!build_dim_map(lhs_bcast, lhs_in_ty, &lhs_map, &lhs_sq_shape,
                     &lhs_needs_squeeze) ||
      !build_dim_map(rhs_bcast, rhs_in_ty, &rhs_map, &rhs_sq_shape,
                     &rhs_needs_squeeze)) {
    return false;
  }

  llvm::SmallVector<int64_t, 8> batch_mul_dims;
  llvm::SmallVector<int64_t, 8> lhs_free_mul_dims;
  llvm::SmallVector<int64_t, 8> rhs_free_mul_dims;
  llvm::SmallVector<int64_t, 8> lhs_batch_dims;
  llvm::SmallVector<int64_t, 8> rhs_batch_dims;
  llvm::SmallVector<int64_t, 8> lhs_free_dims;
  llvm::SmallVector<int64_t, 8> rhs_free_dims;
  llvm::SmallVector<int64_t, 8> lhs_contract_dims;
  llvm::SmallVector<int64_t, 8> rhs_contract_dims;

  for (int64_t d = 0; d < mul_rank; ++d) {
    int64_t ld = lhs_map[d];
    int64_t rd = rhs_map[d];
    if (is_reduced[d]) {
      if (ld < 0 || rd < 0) {
        return false;
      }
      lhs_contract_dims.push_back(ld);
      rhs_contract_dims.push_back(rd);
    } else {
      if (ld >= 0 && rd >= 0) {
        batch_mul_dims.push_back(d);
        lhs_batch_dims.push_back(ld);
        rhs_batch_dims.push_back(rd);
      } else if (ld >= 0 && rd < 0) {
        lhs_free_mul_dims.push_back(d);
        lhs_free_dims.push_back(ld);
      } else if (ld < 0 && rd >= 0) {
        rhs_free_mul_dims.push_back(d);
        rhs_free_dims.push_back(rd);
      } else {
        return false;
      }
    }
  }

  // GENERIC: semantic Outer-product contraction requires both operands to
  // contribute non-empty free dimensions so the materialized product expands
  // both inputs.
  if (lhs_contract_dims.empty() || lhs_free_dims.empty() ||
      rhs_free_dims.empty()) {
    return false;
  }

  mlir::OpBuilder builder(reduce_op);
  mlir::Location loc = reduce_op.getLoc();
  auto squeeze_bcast_operand =
      [&builder, loc](mlir::Value val, llvm::ArrayRef<int64_t> sq_shape,
                      mlir::Type elem_ty, bool needs_squeeze) -> mlir::Value {
    if (!needs_squeeze) {
      return val;
    }
    auto sq_ty = mlir::RankedTensorType::get(sq_shape, elem_ty);
    if (auto reshape = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      if (reshape.getOperand().getType() == sq_ty) {
        return reshape.getOperand();
      }
    }
    return builder.create<mlir::stablehlo::ReshapeOp>(loc, sq_ty, val);
  };
  mlir::Value lhs_val =
      squeeze_bcast_operand(lhs_bcast.getOperand(), lhs_sq_shape,
                            lhs_in_ty.getElementType(), lhs_needs_squeeze);
  mlir::Value rhs_val =
      squeeze_bcast_operand(rhs_bcast.getOperand(), rhs_sq_shape,
                            rhs_in_ty.getElementType(), rhs_needs_squeeze);

  // Check if `lhs_val` is `MulOp(Transpose(G), Transpose(L))` where both
  // transposes can be eliminated by multiplying in `L`'s native layout and
  // absorbing the batch permutation into `lhs_batch_dims`.
  TryAbsorbBatchTransposeAcrossMul(builder, loc, &lhs_val, &lhs_batch_dims,
                                   &lhs_free_dims, &lhs_contract_dims);

  // Check if any batch dimension has a shared group-repeat BroadcastInDimOp on
  // both `lhs_val` and `rhs_val` (e.g. GQA / Mamba2 group repeat).
  int64_t dedup_batch_idx = -1;
  GroupRepeatInfo lhs_repeat, rhs_repeat;
  for (size_t k = 0; k < lhs_batch_dims.size(); ++k) {
    auto lr = MatchBatchDimGroupRepeat(lhs_val, lhs_batch_dims[k]);
    auto rr = MatchBatchDimGroupRepeat(rhs_val, rhs_batch_dims[k]);
    if (lr.has_value() && rr.has_value() && lr->group_size == rr->group_size &&
        lr->repeat_size == rr->repeat_size && lr->repeat_size > 1) {
      dedup_batch_idx = static_cast<int64_t>(k);
      lhs_repeat = *lr;
      rhs_repeat = *rr;
      break;
    }
  }

  auto lhs_cur_ty = mlir::cast<mlir::RankedTensorType>(lhs_val.getType());
  auto rhs_cur_ty = mlir::cast<mlir::RankedTensorType>(rhs_val.getType());

  if (dedup_batch_idx >= 0) {
    llvm::SmallVector<int64_t, 8> lhs_unrep_shape(lhs_cur_ty.getShape().begin(),
                                                  lhs_cur_ty.getShape().end());
    lhs_unrep_shape[lhs_batch_dims[dedup_batch_idx]] = lhs_repeat.group_size;
    auto lhs_unrep_ty = mlir::RankedTensorType::get(
        lhs_unrep_shape, lhs_cur_ty.getElementType());
    lhs_val = builder.create<mlir::stablehlo::ReshapeOp>(
        loc, lhs_unrep_ty, lhs_repeat.unrepeated_src);

    llvm::SmallVector<int64_t, 8> rhs_unrep_shape(rhs_cur_ty.getShape().begin(),
                                                  rhs_cur_ty.getShape().end());
    rhs_unrep_shape[rhs_batch_dims[dedup_batch_idx]] = rhs_repeat.group_size;
    auto rhs_unrep_ty = mlir::RankedTensorType::get(
        rhs_unrep_shape, rhs_cur_ty.getElementType());
    rhs_val = builder.create<mlir::stablehlo::ReshapeOp>(
        loc, rhs_unrep_ty, rhs_repeat.unrepeated_src);

    lhs_cur_ty = lhs_unrep_ty;
    rhs_cur_ty = rhs_unrep_ty;
  }

  llvm::SmallVector<int64_t, 8> dot_out_shape;
  for (int64_t ld : lhs_batch_dims) {
    dot_out_shape.push_back(lhs_cur_ty.getDimSize(ld));
  }
  for (int64_t ld : lhs_free_dims) {
    dot_out_shape.push_back(lhs_cur_ty.getDimSize(ld));
  }
  for (int64_t rd : rhs_free_dims) {
    dot_out_shape.push_back(rhs_cur_ty.getDimSize(rd));
  }

  auto dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(), lhs_batch_dims, rhs_batch_dims, lhs_contract_dims,
      rhs_contract_dims);
  auto dot_ty =
      mlir::RankedTensorType::get(dot_out_shape, out_ty.getElementType());
  mlir::Value result = builder.create<mlir::stablehlo::DotGeneralOp>(
      loc, dot_ty, lhs_val, rhs_val, dot_dims,
      /*precision_config=*/mlir::ArrayAttr{},
      /*algorithm=*/mlir::stablehlo::DotAlgorithmAttr{});

  if (dedup_batch_idx >= 0) {
    llvm::SmallVector<int64_t, 8> bcast_shape;
    llvm::SmallVector<int64_t, 8> bcast_dims;
    for (int64_t i = 0; i < static_cast<int64_t>(dot_out_shape.size()); ++i) {
      bcast_dims.push_back(static_cast<int64_t>(bcast_shape.size()));
      bcast_shape.push_back(dot_out_shape[i]);
      if (i == dedup_batch_idx) {
        bcast_shape.push_back(lhs_repeat.repeat_size);
      }
    }
    auto bcast_ty =
        mlir::RankedTensorType::get(bcast_shape, out_ty.getElementType());
    mlir::Value repeated = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, bcast_ty, result, builder.getDenseI64ArrayAttr(bcast_dims));
    dot_out_shape[dedup_batch_idx] *= lhs_repeat.repeat_size;
    auto full_dot_ty =
        mlir::RankedTensorType::get(dot_out_shape, out_ty.getElementType());
    result =
        builder.create<mlir::stablehlo::ReshapeOp>(loc, full_dot_ty, repeated);
  }

  llvm::SmallVector<int64_t, 8> dot_mul_order;
  dot_mul_order.append(batch_mul_dims.begin(), batch_mul_dims.end());
  dot_mul_order.append(lhs_free_mul_dims.begin(), lhs_free_mul_dims.end());
  dot_mul_order.append(rhs_free_mul_dims.begin(), rhs_free_mul_dims.end());

  llvm::SmallVector<int64_t, 8> target_mul_order = dot_mul_order;
  llvm::sort(target_mul_order);
  if (dot_mul_order != target_mul_order) {
    llvm::SmallVector<int64_t, 8> perm;
    for (int64_t td : target_mul_order) {
      for (size_t i = 0; i < dot_mul_order.size(); ++i) {
        if (dot_mul_order[i] == td) {
          perm.push_back(static_cast<int64_t>(i));
          break;
        }
      }
    }
    result = builder.create<mlir::stablehlo::TransposeOp>(
        loc, out_ty, result, builder.getDenseI64ArrayAttr(perm));
  }

  reduce_op.getResult(0).replaceAllUsesWith(result);
  reduce_op.erase();
  return true;
}

// Folds unit-dimension softmax (`exp(x - max(x)) / sum(exp(x - max(x)))` or
// `exp(x - max(x)) * (1.0 / sum(exp(x - max(x))))` along a dimension of size 1)
// to a splat 1.0 constant, and folds `DotGeneralOp(splat(1.0), V)` (where
// `splat(1.0)` has a single size-1 contracting dimension and only unit free
// dimensions) to `ReshapeOp(V)`.
bool TryFoldUnitDimensionSoftmaxAndDot(mlir::func::FuncOp func) {
  bool changed = false;
  llvm::SmallVector<mlir::Operation*, 16> to_erase;

  func.walk([&changed, &to_erase](mlir::Operation* op) {
    mlir::Value exp_val;
    mlir::Value sum_unsqueezed;
    if (auto mul_op = mlir::dyn_cast<mlir::stablehlo::MulOp>(op)) {
      mlir::Value lhs = mul_op.getLhs();
      mlir::Value rhs = mul_op.getRhs();
      if (auto div_op = rhs.getDefiningOp<mlir::stablehlo::DivOp>()) {
        if (IsSplatConstantValue(div_op.getLhs(), 1.0)) {
          exp_val = lhs;
          sum_unsqueezed = div_op.getRhs();
        }
      } else if (auto div_op = lhs.getDefiningOp<mlir::stablehlo::DivOp>()) {
        if (IsSplatConstantValue(div_op.getLhs(), 1.0)) {
          exp_val = rhs;
          sum_unsqueezed = div_op.getRhs();
        }
      }
    } else if (auto div_op = mlir::dyn_cast<mlir::stablehlo::DivOp>(op)) {
      exp_val = div_op.getLhs();
      sum_unsqueezed = div_op.getRhs();
    }
    if (!exp_val || !sum_unsqueezed) {
      return;
    }

    auto exp_op = exp_val.getDefiningOp<mlir::stablehlo::ExpOp>();
    if (!exp_op) {
      return;
    }
    mlir::Value sum_src = SkipIdentityBroadcasts(sum_unsqueezed);
    if (auto bcast =
            sum_src.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      sum_src = bcast.getOperand();
    }
    if (auto reshape = sum_src.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      sum_src = reshape.getOperand();
    }
    auto sum_reduce = sum_src.getDefiningOp<mlir::stablehlo::ReduceOp>();
    if (!sum_reduce || !IsUnitDimensionAddReduce(sum_reduce) ||
        sum_reduce.getInputs()[0] != exp_val) {
      return;
    }

    mlir::RankedTensorType out_ty;
    if (!IsStaticRankedTensor(op->getResult(0).getType(), &out_ty)) {
      return;
    }
    auto float_ty = mlir::dyn_cast<mlir::FloatType>(out_ty.getElementType());
    if (!float_ty) {
      return;
    }
    mlir::OpBuilder builder(op);
    auto one_attr = mlir::DenseElementsAttr::get(
        out_ty, llvm::APFloat(float_ty.getFloatSemantics(), 1));
    mlir::Value one_const = builder.create<mlir::stablehlo::ConstantOp>(
        op->getLoc(), out_ty, one_attr);
    op->getResult(0).replaceAllUsesWith(one_const);
    to_erase.push_back(op);
    changed = true;
  });

  for (mlir::Operation* op : to_erase) {
    if (op->use_empty()) {
      op->erase();
    }
  }
  to_erase.clear();

  func.walk([&changed, &to_erase](mlir::stablehlo::DotGeneralOp dot_op) {
    if (!IsSplatConstantValue(dot_op.getLhs(), 1.0)) {
      return;
    }
    mlir::RankedTensorType lhs_ty, rhs_ty, out_ty;
    if (!IsStaticRankedTensor(dot_op.getLhs().getType(), &lhs_ty) ||
        !IsStaticRankedTensor(dot_op.getRhs().getType(), &rhs_ty) ||
        !IsStaticRankedTensor(dot_op.getType(), &out_ty) ||
        rhs_ty.getNumElements() != out_ty.getNumElements() ||
        rhs_ty.getElementType() != out_ty.getElementType()) {
      return;
    }
    auto dnums = dot_op.getDotDimensionNumbers();
    llvm::ArrayRef<int64_t> l_batch = dnums.getLhsBatchingDimensions();
    llvm::ArrayRef<int64_t> r_batch = dnums.getRhsBatchingDimensions();
    llvm::ArrayRef<int64_t> l_contract = dnums.getLhsContractingDimensions();
    llvm::ArrayRef<int64_t> r_contract = dnums.getRhsContractingDimensions();
    if (l_contract.size() != 1 || r_contract.size() != 1 ||
        lhs_ty.getDimSize(l_contract[0]) != 1 ||
        rhs_ty.getDimSize(r_contract[0]) != 1) {
      return;
    }
    // Require that every non-batch dimension of `lhs` has size 1, and that
    // `r_batch` is `[0, 1, ..., B - 1]` so all non-unit dimensions of `rhs`
    // keep their exact relative order in `out_ty`.
    for (int64_t i = 0; i < static_cast<int64_t>(r_batch.size()); ++i) {
      if (r_batch[i] != i) {
        return;
      }
    }
    llvm::SmallVector<bool, 8> lhs_is_batch(lhs_ty.getRank(), false);
    for (int64_t d : l_batch) {
      lhs_is_batch[d] = true;
    }
    for (int64_t d = 0; d < lhs_ty.getRank(); ++d) {
      if (!lhs_is_batch[d] && lhs_ty.getDimSize(d) != 1) {
        return;
      }
    }

    mlir::OpBuilder builder(dot_op);
    mlir::Value reshaped = builder.create<mlir::stablehlo::ReshapeOp>(
        dot_op.getLoc(), out_ty, dot_op.getRhs());
    dot_op.getResult().replaceAllUsesWith(reshaped);
    to_erase.push_back(dot_op);
    changed = true;
  });

  for (mlir::Operation* op : to_erase) {
    if (op->use_empty()) {
      op->erase();
    }
  }
  return changed;
}

bool IsTwoMomentAddReduce(mlir::stablehlo::ReduceOp reduce_op) {
  // GENERIC: semantic Dual-moment LayerNorm reduction takes 2 inputs (x and
  // x^2) and 2 initial values, with a 4-argument block returning 2 sums.
  if (reduce_op.getInputs().size() != 2 ||
      reduce_op.getInitValues().size() != 2 ||
      reduce_op.getDimensions().size() != 1) {
    return false;
  }
  if (!IsSplatConstantValue(reduce_op.getInitValues()[0], 0.0) ||
      !IsSplatConstantValue(reduce_op.getInitValues()[1], 0.0)) {
    return false;
  }
  mlir::Region& body = reduce_op.getBody();
  if (!body.hasOneBlock()) {
    return false;
  }
  mlir::Block& block = body.front();
  // GENERIC: semantic Dual-input reduction block has 4 scalar arguments.
  if (block.getNumArguments() != 4) {
    return false;
  }
  auto ret_op =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  // GENERIC: semantic Dual-input reduction block returns 2 scalar sums.
  if (!ret_op || ret_op.getResults().size() != 2) {
    return false;
  }
  auto add0 = ret_op.getResults()[0].getDefiningOp<mlir::stablehlo::AddOp>();
  auto add1 = ret_op.getResults()[1].getDefiningOp<mlir::stablehlo::AddOp>();
  return add0 != nullptr && add1 != nullptr;
}

// Squeezes interior unit batch dimensions (`0 < d < norm_axis` with
// `shape[d] == 1` and `shape[0] > 1`) in dual-output LayerNorm reductions and
// their downstream normalization chains.
bool TrySqueezeInteriorUnitDimLayerNorm(mlir::stablehlo::ReduceOp reduce_op) {
  if (!IsTwoMomentAddReduce(reduce_op)) {
    return false;
  }
  mlir::Value x_casted = reduce_op.getInputs()[0];
  mlir::Value x_sq = reduce_op.getInputs()[1];
  auto sq_mul = x_sq.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!sq_mul || sq_mul.getLhs() != x_casted || sq_mul.getRhs() != x_casted) {
    return false;
  }

  mlir::RankedTensorType in_ty, red_ty;
  if (!IsStaticRankedTensor(x_casted.getType(), &in_ty) ||
      !IsStaticRankedTensor(reduce_op.getResultTypes()[0], &red_ty)) {
    return false;
  }
  int64_t rank = in_ty.getRank();
  int64_t norm_axis = reduce_op.getDimensions()[0];
  // GENERIC: semantic An interior batch dimension 0 < d < norm_axis can only
  // exist when rank >= 3.
  if (rank < 3 || norm_axis != rank - 1 || in_ty.getDimSize(0) <= 1 ||
      in_ty.getDimSize(norm_axis) <= 1) {
    return false;
  }

  llvm::SmallVector<int64_t, 8> sq_in_shape;
  llvm::SmallVector<int64_t, 8> sq_red_shape;
  bool has_interior_unit = false;
  for (int64_t d = 0; d < norm_axis; ++d) {
    if (d > 0 && in_ty.getDimSize(d) == 1) {
      has_interior_unit = true;
      continue;
    }
    sq_in_shape.push_back(in_ty.getDimSize(d));
    sq_red_shape.push_back(in_ty.getDimSize(d));
  }
  if (!has_interior_unit) {
    return false;
  }
  sq_in_shape.push_back(in_ty.getDimSize(norm_axis));
  int64_t sq_norm_axis = static_cast<int64_t>(sq_in_shape.size()) - 1;

  // Locate the `SubtractOp(x_casted, mean_bcast)` consumer that centers
  // `x_casted`.
  mlir::stablehlo::SubtractOp center_sub;
  for (mlir::Operation* user : x_casted.getUsers()) {
    if (auto sub = mlir::dyn_cast<mlir::stablehlo::SubtractOp>(user)) {
      if (sub.getLhs() == x_casted) {
        center_sub = sub;
        break;
      }
    }
  }
  if (!center_sub) {
    return false;
  }
  auto mean_bcast =
      center_sub.getRhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!mean_bcast) {
    return false;
  }

  mlir::stablehlo::MulOp norm_mul;
  if (center_sub->hasOneUse()) {
    norm_mul =
        mlir::dyn_cast<mlir::stablehlo::MulOp>(*center_sub->user_begin());
  }
  if (!norm_mul) {
    return false;
  }
  mlir::Value rstd_bcast_val = (norm_mul.getLhs() == center_sub.getResult())
                                   ? norm_mul.getRhs()
                                   : norm_mul.getLhs();
  auto rstd_bcast =
      rstd_bcast_val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!rstd_bcast) {
    return false;
  }

  mlir::OpBuilder builder(reduce_op);
  mlir::Location loc = reduce_op.getLoc();
  auto sq_stats_ty =
      mlir::RankedTensorType::get(sq_in_shape, in_ty.getElementType());
  auto sq_red_ty =
      mlir::RankedTensorType::get(sq_red_shape, red_ty.getElementType());

  // Reshape the source of `x_casted` (before `ConvertOp` when present) so the
  // convert also operates on the squeezed shape.
  mlir::Value x_casted_sq;
  if (auto conv = x_casted.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    mlir::Value raw_in = conv.getOperand();
    auto raw_ty = mlir::cast<mlir::RankedTensorType>(raw_in.getType());
    auto sq_raw_ty =
        mlir::RankedTensorType::get(sq_in_shape, raw_ty.getElementType());
    builder.setInsertionPoint(conv);
    mlir::Value raw_sq;
    if (auto add_op = raw_in.getDefiningOp<mlir::stablehlo::AddOp>();
        add_op && add_op->hasOneUse()) {
      mlir::Value lhs_sq = builder.create<mlir::stablehlo::ReshapeOp>(
          loc, sq_raw_ty, add_op.getLhs());
      mlir::Value rhs_sq = builder.create<mlir::stablehlo::ReshapeOp>(
          loc, sq_raw_ty, add_op.getRhs());
      raw_sq = builder.create<mlir::stablehlo::AddOp>(loc, sq_raw_ty, lhs_sq,
                                                      rhs_sq);
    } else {
      raw_sq =
          builder.create<mlir::stablehlo::ReshapeOp>(loc, sq_raw_ty, raw_in);
    }
    x_casted_sq =
        builder.create<mlir::stablehlo::ConvertOp>(loc, sq_stats_ty, raw_sq);
  } else {
    builder.setInsertionPoint(sq_mul);
    x_casted_sq =
        builder.create<mlir::stablehlo::ReshapeOp>(loc, sq_stats_ty, x_casted);
  }

  builder.setInsertionPoint(reduce_op);
  mlir::Value x_sq_sq = builder.create<mlir::stablehlo::MulOp>(
      loc, sq_stats_ty, x_casted_sq, x_casted_sq);
  auto new_reduce = builder.create<mlir::stablehlo::ReduceOp>(
      loc, mlir::TypeRange{sq_red_ty, sq_red_ty},
      mlir::ValueRange{x_casted_sq, x_sq_sq}, reduce_op.getInitValues(),
      builder.getDenseI64ArrayAttr({sq_norm_axis}));
  new_reduce.getBody().takeBody(reduce_op.getBody());

  // Reshape the squeezed reduction results back to `red_ty` so MLIR's
  // canonicalizer and our broadcast rewrite below can fold them cleanly.
  mlir::Value sum_x_unsqueezed = builder.create<mlir::stablehlo::ReshapeOp>(
      loc, red_ty, new_reduce.getResult(0));
  mlir::Value sum_x2_unsqueezed = builder.create<mlir::stablehlo::ReshapeOp>(
      loc, red_ty, new_reduce.getResult(1));
  reduce_op.getResult(0).replaceAllUsesWith(sum_x_unsqueezed);
  reduce_op.getResult(1).replaceAllUsesWith(sum_x2_unsqueezed);

  // Sink the squeezed 1D/2D `sq_red_ty` shape through the elementwise
  // statistics DAG (`mean`, `mean_squared`, `variance`, `rstd`).
  llvm::DenseMap<mlir::Value, mlir::Value> sq_stats_map;
  sq_stats_map[sum_x_unsqueezed] = new_reduce.getResult(0);
  sq_stats_map[sum_x2_unsqueezed] = new_reduce.getResult(1);

  std::function<mlir::Value(mlir::Value)> get_squeezed_stat;  // STD_FUNCTION_OK
  get_squeezed_stat = [&get_squeezed_stat, &sq_stats_map, sq_red_ty,
                       red_ty](mlir::Value v) -> mlir::Value {
    if (auto it = sq_stats_map.find(v); it != sq_stats_map.end()) {
      return it->second;
    }
    mlir::Operation* def = v.getDefiningOp();
    if (!def) {
      return nullptr;
    }
    mlir::OpBuilder b(def);
    if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(def)) {
      if (bcast.getBroadcastDimensions().empty()) {
        mlir::Value res = b.create<mlir::stablehlo::BroadcastInDimOp>(
            def->getLoc(), sq_red_ty, bcast.getOperand(),
            b.getDenseI64ArrayAttr({}));
        sq_stats_map[v] = res;
        return res;
      }
      return nullptr;
    }
    if (auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(def)) {
      auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
      if (attr && attr.isSplat()) {
        auto new_attr = mlir::DenseElementsAttr::get(
            sq_red_ty, attr.getSplatValue<mlir::Attribute>());
        mlir::Value res = b.create<mlir::stablehlo::ConstantOp>(
            def->getLoc(), sq_red_ty, new_attr);
        sq_stats_map[v] = res;
        return res;
      }
      return nullptr;
    }
    if (def->getNumResults() != 1 || def->getResult(0).getType() != red_ty) {
      return nullptr;
    }
    if (!mlir::isa<mlir::stablehlo::MulOp, mlir::stablehlo::DivOp,
                   mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
                   mlir::stablehlo::MaxOp, mlir::stablehlo::RsqrtOp>(def)) {
      return nullptr;
    }
    llvm::SmallVector<mlir::Value, 2> sq_ops;
    for (mlir::Value opnd : def->getOperands()) {
      mlir::Value sq_opnd = get_squeezed_stat(opnd);
      if (!sq_opnd) {
        return nullptr;
      }
      sq_ops.push_back(sq_opnd);
    }
    mlir::OperationState state(def->getLoc(), def->getName());
    state.addOperands(sq_ops);
    state.addTypes(sq_red_ty);
    state.addAttributes(def->getAttrs());
    mlir::Operation* cloned = b.create(state);
    mlir::Value res = cloned->getResult(0);
    sq_stats_map[v] = res;
    return res;
  };

  mlir::Value mean_sq = get_squeezed_stat(mean_bcast.getOperand());
  mlir::Value rstd_sq = get_squeezed_stat(rstd_bcast.getOperand());
  if (mean_sq && rstd_sq) {
    llvm::SmallVector<int64_t, 8> unred_axes(sq_norm_axis);
    for (int64_t i = 0; i < sq_norm_axis; ++i) {
      unred_axes[i] = i;
    }
    builder.setInsertionPoint(norm_mul);
    mlir::Value mean_bcast_sq =
        builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, sq_stats_ty, mean_sq,
            builder.getDenseI64ArrayAttr(unred_axes));
    mlir::Value rstd_bcast_sq =
        builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, sq_stats_ty, rstd_sq,
            builder.getDenseI64ArrayAttr(unred_axes));
    mlir::Value sub_sq = builder.create<mlir::stablehlo::SubtractOp>(
        loc, sq_stats_ty, x_casted_sq, mean_bcast_sq);
    mlir::Value cur_sq = builder.create<mlir::stablehlo::MulOp>(
        loc, sq_stats_ty, sub_sq, rstd_bcast_sq);
    mlir::Operation* tail_op = norm_mul.getOperation();

    // Advance through optional ConvertOp, affine weight MulOp, and bias AddOp.
    while (tail_op->hasOneUse()) {
      mlir::Operation* next = *tail_op->user_begin();
      if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(next)) {
        auto next_ty = mlir::cast<mlir::RankedTensorType>(conv.getType());
        auto sq_next_ty =
            mlir::RankedTensorType::get(sq_in_shape, next_ty.getElementType());
        builder.setInsertionPoint(conv);
        cur_sq = builder.create<mlir::stablehlo::ConvertOp>(conv.getLoc(),
                                                            sq_next_ty, cur_sq);
        tail_op = conv.getOperation();
        continue;
      }
      if (auto w_mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(next)) {
        mlir::Value other = (w_mul.getLhs() == tail_op->getResult(0))
                                ? w_mul.getRhs()
                                : w_mul.getLhs();
        if (auto wb =
                other.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
          if (wb.getBroadcastDimensions().size() == 1 &&
              wb.getBroadcastDimensions()[0] == norm_axis) {
            builder.setInsertionPoint(w_mul);
            mlir::Value wb_sq =
                builder.create<mlir::stablehlo::BroadcastInDimOp>(
                    w_mul.getLoc(), cur_sq.getType(), wb.getOperand(),
                    builder.getDenseI64ArrayAttr({sq_norm_axis}));
            cur_sq = builder.create<mlir::stablehlo::MulOp>(
                w_mul.getLoc(), cur_sq.getType(), cur_sq, wb_sq);
            tail_op = w_mul.getOperation();
            continue;
          }
        }
      }
      if (auto b_add = mlir::dyn_cast<mlir::stablehlo::AddOp>(next)) {
        mlir::Value other = (b_add.getLhs() == tail_op->getResult(0))
                                ? b_add.getRhs()
                                : b_add.getLhs();
        if (auto bb =
                other.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
          if (bb.getBroadcastDimensions().size() == 1 &&
              bb.getBroadcastDimensions()[0] == norm_axis) {
            builder.setInsertionPoint(b_add);
            mlir::Value bb_sq =
                builder.create<mlir::stablehlo::BroadcastInDimOp>(
                    b_add.getLoc(), cur_sq.getType(), bb.getOperand(),
                    builder.getDenseI64ArrayAttr({sq_norm_axis}));
            cur_sq = builder.create<mlir::stablehlo::AddOp>(
                b_add.getLoc(), cur_sq.getType(), cur_sq, bb_sq);
            tail_op = b_add.getOperation();
            continue;
          }
        }
      }
      break;
    }

    builder.setInsertionPointAfter(tail_op);
    mlir::Value restored = builder.create<mlir::stablehlo::ReshapeOp>(
        tail_op->getLoc(), tail_op->getResult(0).getType(), cur_sq);
    tail_op->getResult(0).replaceAllUsesWith(restored);
  }

  reduce_op.erase();
  return true;
}

void EraseDeadPureOps(mlir::func::FuncOp func) {
  bool erased = true;
  while (erased) {
    erased = false;
    llvm::SmallVector<mlir::Operation*, 32> dead;
    func.walk([&dead](mlir::Operation* op) {
      if (!mlir::isa<mlir::func::FuncOp, mlir::func::ReturnOp,
                     mlir::stablehlo::ReturnOp>(op) &&
          op->use_empty() && mlir::isPure(op)) {
        dead.push_back(op);
      }
    });
    for (mlir::Operation* op : dead) {
      if (op->use_empty()) {
        op->erase();
        erased = true;
      }
    }
  }
}

class CanonicalizeBroadcastContractionsAndUnitReductionsPass
    : public mlir::PassWrapper<
          CanonicalizeBroadcastContractionsAndUnitReductionsPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CanonicalizeBroadcastContractionsAndUnitReductionsPass)

  llvm::StringRef getArgument() const final {
    return "canonicalize-broadcast-contractions-and-unit-reductions";
  }

  llvm::StringRef getDescription() const final {
    return "Canonicalizes outer-product broadcast-multiply-add reductions to "
           "stablehlo.dot_general, folds unit-dimension softmaxes, and "
           "squeezes interior unit batch dimensions in LayerNorm reductions.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    bool changed = false;

    llvm::SmallVector<mlir::stablehlo::ReduceOp, 16> reduces;
    func.walk(
        [&reduces](mlir::stablehlo::ReduceOp op) { reduces.push_back(op); });
    for (mlir::stablehlo::ReduceOp reduce_op : reduces) {
      if (!reduce_op || !reduce_op->getBlock()) {
        continue;
      }
      if (TryRewriteBroadcastMulReduceToDot(reduce_op)) {
        changed = true;
      }
    }

    if (TryFoldUnitDimensionSoftmaxAndDot(func)) {
      changed = true;
    }

    // Canonicalize unit-dimension additive ReduceOp (`dim_size == 1`) to
    // ReshapeOp so intermediate unit reductions disappear cleanly.
    llvm::SmallVector<mlir::stablehlo::ReduceOp, 8> unit_reduces;
    func.walk([&unit_reduces](mlir::stablehlo::ReduceOp op) {
      if (IsUnitDimensionAddReduce(op)) {
        unit_reduces.push_back(op);
      }
    });
    for (mlir::stablehlo::ReduceOp op : unit_reduces) {
      mlir::OpBuilder builder(op);
      mlir::Value reshaped = builder.create<mlir::stablehlo::ReshapeOp>(
          op.getLoc(), op.getResultTypes()[0], op.getInputs()[0]);
      op.getResult(0).replaceAllUsesWith(reshaped);
      op.erase();
      changed = true;
    }

    llvm::SmallVector<mlir::stablehlo::ReduceOp, 16> ln_reduces;
    func.walk([&ln_reduces](mlir::stablehlo::ReduceOp op) {
      if (IsTwoMomentAddReduce(op)) {
        ln_reduces.push_back(op);
      }
    });
    for (mlir::stablehlo::ReduceOp reduce_op : ln_reduces) {
      if (TrySqueezeInteriorUnitDimLayerNorm(reduce_op)) {
        changed = true;
      }
    }

    if (changed) {
      EraseDeadPureOps(func);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
createCanonicalizeBroadcastContractionsAndUnitReductionsPass() {
  return std::make_unique<
      CanonicalizeBroadcastContractionsAndUnitReductionsPass>();
}

}  // namespace torch_tpu
