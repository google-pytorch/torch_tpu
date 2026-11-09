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

#include "csrc/mlir/passes/fold_dot_transposes/fold_dot_transposes.h"  // NOLINT

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // NOLINT
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"  // NOLINT
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"  // NOLINT
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// Maximum number of fixed-point iterations for composite rewrites.
constexpr int kMaxPassIterations = 8;

// GENERIC: hardware 128 lane TPU vector register and MXU output tile width.
constexpr int64_t kTpuLaneWidth = 128;
// GENERIC: hardware 8 sublane TPU vector register and MXU sublane tile height.
constexpr int64_t kTpuSublaneCount = 8;

constexpr int kMaxProducerReshapeWalk = 8;

[[nodiscard]] bool IsStaticRankedTensor(
    mlir::Type type, mlir::RankedTensorType* out_type = nullptr) {
  auto r_type = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!r_type || !r_type.hasStaticShape()) return false;
  if (out_type != nullptr) {
    *out_type = r_type;
  }
  return true;
}

[[nodiscard]] mlir::Value PeelIdentityBroadcast(mlir::Value val) {
  while (true) {
    auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!bcast || !bcast.getResult().hasOneUse()) break;
    mlir::RankedTensorType in_type;
    mlir::RankedTensorType out_type;
    if (IsStaticRankedTensor(bcast.getOperand().getType(), &in_type) &&
        IsStaticRankedTensor(bcast.getType(), &out_type) &&
        in_type.getShape() == out_type.getShape()) {
      llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
      if (llvm::all_of(llvm::enumerate(bdims), [](const auto& item) {
            return item.value() == static_cast<int64_t>(item.index());
          })) {
        val = bcast.getOperand();
        continue;
      }
    }
    break;
  }
  return val;
}

[[nodiscard]] bool HasDimProduct(llvm::ArrayRef<int64_t> dims,
                                 const int64_t expected_product) {
  int64_t prod = 1;
  for (const int64_t d : dims) {
    if (d <= 0 || __builtin_mul_overflow(prod, d, &prod)) return false;
  }
  return prod == expected_product;
}

[[nodiscard]] bool IsStrictlyAscending(llvm::ArrayRef<int64_t> dims) {
  for (size_t i = 1; i < dims.size(); ++i) {
    if (dims[i] <= dims[i - 1]) return false;
  }
  return true;
}

// =============================================================================
// Mechanism 1 & 2: FoldBatchHeadMergedContractions
// =============================================================================

struct ProducerViewInfo {
  mlir::stablehlo::TransposeOp t_op;
  mlir::Value x;
  llvm::SmallVector<int64_t> sq_x_shape;
  llvm::SmallVector<int64_t> sq_g0_dims;
  llvm::SmallVector<int64_t> sq_g1_dims;
  llvm::SmallVector<int64_t> g0_sizes;
  llvm::SmallVector<int64_t> g1_sizes;
};

struct ContractionUse {
  mlir::Operation* dot_user = nullptr;
  mlir::stablehlo::TransposeOp via_tr = nullptr;
  bool is_lhs = false;
  int64_t r_c_dim = -1;
  int64_t other_c_dim = -1;
  mlir::ArrayAttr precision_attr;
  mlir::RankedTensorType out_type;
};

[[nodiscard]] bool IsReshapedContractionProducer(mlir::Value x) {
  auto first_reshape = x.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!first_reshape) return false;
  mlir::Value cur = first_reshape.getOperand();
  for (int step = 0; step < kMaxProducerReshapeWalk; ++step) {
    mlir::Operation* def_op = cur.getDefiningOp();
    if (!def_op) return false;
    if (llvm::isa<mlir::stablehlo::DotOp, mlir::stablehlo::DotGeneralOp>(
            def_op)) {
      return true;
    }
    auto resh = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(def_op);
    if (!resh) return false;
    cur = resh.getOperand();
  }
  return false;
}

[[nodiscard]] bool MatchesExpectedNonUnitDims(
    mlir::stablehlo::ReshapeOp resh,
    llvm::ArrayRef<int64_t> expected_non_unit_dims) {
  auto in_ty =
      llvm::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
  if (!in_ty || !in_ty.hasStaticShape()) return false;
  llvm::SmallVector<int64_t> in_non_unit;
  for (const int64_t d : in_ty.getShape()) {
    if (d > 1) {
      in_non_unit.push_back(d);
    }
  }
  return in_non_unit == expected_non_unit_dims;
}

[[nodiscard]] bool IsOpaqueCustomCallProducer(
    mlir::Value v, llvm::ArrayRef<int64_t> expected_non_unit_dims) {
  for (int depth = 0; depth < 8; ++depth) {
    if (v.getDefiningOp<mlir::stablehlo::CustomCallOp>() != nullptr) {
      return true;
    }
    if (auto gte = v.getDefiningOp<mlir::stablehlo::GetTupleElementOp>()) {
      v = gte.getOperand();
      continue;
    }
    if (auto resh = v.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      if (!MatchesExpectedNonUnitDims(resh, expected_non_unit_dims)) {
        return false;
      }
      v = resh.getOperand();
      continue;
    }
    return false;
  }
  return false;
}

[[nodiscard]] int64_t FindSplitPoint(llvm::ArrayRef<int64_t> t_shape,
                                     const int64_t m_size,
                                     const int64_t k_size) {
  const int64_t rank = static_cast<int64_t>(t_shape.size());
  for (int64_t s = 1; s < rank; ++s) {
    if (HasDimProduct(t_shape.slice(0, s), m_size) &&
        HasDimProduct(t_shape.drop_front(s), k_size)) {
      return s;
    }
  }
  return -1;
}

void SqueezeAndMapAxes(llvm::ArrayRef<int64_t> x_shape,
                       llvm::ArrayRef<int64_t> perm, const int64_t split,
                       ProducerViewInfo& info) {
  const int64_t rank = static_cast<int64_t>(x_shape.size());
  llvm::SmallVector<int64_t> old_to_sq(rank, -1);
  for (int64_t i = 0; i < rank; ++i) {
    if (x_shape[i] > 1) {
      old_to_sq[i] = static_cast<int64_t>(info.sq_x_shape.size());
      info.sq_x_shape.push_back(x_shape[i]);
    }
  }
  for (int64_t i = 0; i < split; ++i) {
    const int64_t axis = perm[i];
    if (x_shape[axis] > 1) {
      info.sq_g0_dims.push_back(old_to_sq[axis]);
    }
  }
  for (int64_t i = split; i < rank; ++i) {
    const int64_t axis = perm[i];
    if (x_shape[axis] > 1) {
      info.sq_g1_dims.push_back(old_to_sq[axis]);
    }
  }
}

[[nodiscard]] bool ValidateAxisInvariants(ProducerViewInfo& info) {
  if (info.sq_g0_dims.empty() || info.sq_g1_dims.size() < 2) return false;
  if (!llvm::is_sorted(info.sq_g0_dims) || !llvm::is_sorted(info.sq_g1_dims)) {
    return false;
  }
  llvm::SmallVector<int64_t> sq_perm = info.sq_g0_dims;
  sq_perm.append(info.sq_g1_dims.begin(), info.sq_g1_dims.end());
  if (llvm::is_sorted(sq_perm)) return false;
  if (info.sq_g1_dims.back() !=
          static_cast<int64_t>(info.sq_x_shape.size()) - 1 ||
      info.sq_x_shape.back() % kTpuLaneWidth != 0) {
    return false;
  }
  if (info.sq_x_shape[info.sq_g0_dims.back()] % kTpuSublaneCount != 0) {
    return false;
  }
  for (const int64_t d : info.sq_g0_dims) {
    info.g0_sizes.push_back(info.sq_x_shape[d]);
  }
  for (const int64_t d : info.sq_g1_dims) {
    info.g1_sizes.push_back(info.sq_x_shape[d]);
  }
  return true;
}

[[nodiscard]] bool PartitionAxes(llvm::ArrayRef<int64_t> x_shape,
                                 llvm::ArrayRef<int64_t> perm,
                                 const int64_t split, ProducerViewInfo& info) {
  SqueezeAndMapAxes(x_shape, perm, split, info);
  return ValidateAxisInvariants(info);
}

[[nodiscard]] bool ValidateReshapeOp(mlir::stablehlo::ReshapeOp r_op,
                                     int64_t& m_size, int64_t& k_size) {
  auto r_type = llvm::dyn_cast<mlir::RankedTensorType>(r_op.getType());
  if (!r_type || !r_type.hasStaticShape() || r_type.getRank() != 2 ||
      !llvm::isa<mlir::FloatType>(r_type.getElementType()) ||
      r_op.getResult().use_empty()) {
    return false;
  }
  m_size = r_type.getDimSize(0);
  k_size = r_type.getDimSize(1);
  if (m_size <= 0 || k_size <= 0) return false;
  return (m_size % kTpuSublaneCount == 0);
}

[[nodiscard]] bool ValidateTransposeOp(mlir::stablehlo::TransposeOp t_op,
                                       const mlir::Type elem_type,
                                       const int64_t m_size,
                                       const int64_t k_size, int64_t& split) {
  const mlir::Value x = t_op.getOperand();
  auto x_type = llvm::dyn_cast<mlir::RankedTensorType>(x.getType());
  if (!x_type || !x_type.hasStaticShape() || x_type.getRank() < 3 ||
      x_type.getElementType() != elem_type) {
    return false;
  }
  if (llvm::any_of(x_type.getShape(), [](int64_t d) { return d <= 0; })) {
    return false;
  }
  llvm::ArrayRef<int64_t> perm = t_op.getPermutation();
  if (perm.size() != static_cast<size_t>(x_type.getRank())) return false;
  if (llvm::any_of(perm, [&x_type](int64_t p) {
        return p < 0 || p >= x_type.getRank();
      })) {
    return false;
  }
  auto t_type = llvm::cast<mlir::RankedTensorType>(t_op.getType());
  split = FindSplitPoint(t_type.getShape(), m_size, k_size);
  return (split != -1);
}

[[nodiscard]] bool MatchProducerView(mlir::stablehlo::ReshapeOp r_op,
                                     ProducerViewInfo& info) {
  int64_t m_size = 0;
  int64_t k_size = 0;
  if (!ValidateReshapeOp(r_op, m_size, k_size)) return false;
  auto t_op = r_op.getOperand().getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!t_op || !t_op.getResult().hasOneUse() ||
      t_op->getBlock() != r_op->getBlock()) {
    return false;
  }
  const auto elem_type =
      llvm::cast<mlir::RankedTensorType>(r_op.getType()).getElementType();
  int64_t split = -1;
  if (!ValidateTransposeOp(t_op, elem_type, m_size, k_size, split)) {
    return false;
  }
  info.t_op = t_op;
  info.x = t_op.getOperand();
  auto x_type = llvm::cast<mlir::RankedTensorType>(info.x.getType());
  if (IsReshapedContractionProducer(info.x)) return false;
  return PartitionAxes(x_type.getShape(), t_op.getPermutation(), split, info);
}

[[nodiscard]] bool IsUnbatchedSingleAxisDotDims(
    mlir::stablehlo::DotDimensionNumbersAttr dnums) {
  return dnums.getLhsBatchingDimensions().empty() &&
         dnums.getRhsBatchingDimensions().empty() &&
         dnums.getLhsContractingDimensions().size() == 1 &&
         dnums.getRhsContractingDimensions().size() == 1;
}

[[nodiscard]] bool ExtractDotDimensions(mlir::Operation* dot_user,
                                        const unsigned operand_num,
                                        bool& is_lhs, int64_t& raw_c_dim,
                                        int64_t& other_c_dim,
                                        mlir::ArrayAttr& precision_attr) {
  is_lhs = (operand_num == 0);
  if (auto dot = llvm::dyn_cast<mlir::stablehlo::DotOp>(dot_user)) {
    if (dot.getLhs() == dot.getRhs()) return false;
    raw_c_dim = is_lhs ? 1 : 0;
    other_c_dim = 1 - raw_c_dim;
    precision_attr = dot.getPrecisionConfigAttr();
    return true;
  }
  auto dg = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(dot_user);
  if (!dg || dg.getLhs() == dg.getRhs() || dg.getAlgorithmAttr() != nullptr) {
    return false;
  }
  auto dnums = dg.getDotDimensionNumbers();
  if (!IsUnbatchedSingleAxisDotDims(dnums)) return false;
  const int64_t lhs_c = dnums.getLhsContractingDimensions()[0];
  const int64_t rhs_c = dnums.getRhsContractingDimensions()[0];
  raw_c_dim = is_lhs ? lhs_c : rhs_c;
  other_c_dim = is_lhs ? rhs_c : lhs_c;
  precision_attr = dg.getPrecisionConfigAttr();
  return true;
}

void PeelTransposeOp(
    mlir::Value other, const int64_t other_c_dim, mlir::Value& eff_other,
    int64_t& eff_other_c_dim,
    llvm::SmallVectorImpl<mlir::stablehlo::TransposeOp>* peeled_trs = nullptr) {
  eff_other = other;
  eff_other_c_dim = other_c_dim;
  if (auto other_tr = other.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    auto o_ty =
        llvm::dyn_cast<mlir::RankedTensorType>(other_tr.getOperand().getType());
    llvm::ArrayRef<int64_t> o_perm = other_tr.getPermutation();
    if (o_ty && o_ty.hasStaticShape() && o_ty.getRank() == 2 &&
        o_perm == llvm::ArrayRef<int64_t>{1, 0}) {
      eff_other = other_tr.getOperand();
      eff_other_c_dim = 1 - other_c_dim;
      if (peeled_trs != nullptr) {
        peeled_trs->push_back(other_tr);
      }
    }
  }
}

[[nodiscard]] bool ValidateDotTypesAndDims(
    mlir::Operation* dot_user, const bool is_lhs, const int64_t r_c_dim,
    const int64_t other_c_dim, const int64_t m_size, const int64_t k_size,
    const mlir::Type elem_type, mlir::RankedTensorType& out_type,
    int64_t& n_size, int64_t& eff_other_c_dim) {
  const mlir::Value other = dot_user->getOperand(is_lhs ? 1 : 0);
  auto other_type = llvm::dyn_cast<mlir::RankedTensorType>(other.getType());
  out_type =
      llvm::dyn_cast<mlir::RankedTensorType>(dot_user->getResult(0).getType());
  if (!other_type || !other_type.hasStaticShape() ||
      other_type.getRank() != 2 ||
      !llvm::isa<mlir::FloatType>(other_type.getElementType()) ||
      other_type.getElementType() != elem_type) {
    return false;
  }
  if (!out_type || !out_type.hasStaticShape() || out_type.getRank() != 2 ||
      !llvm::isa<mlir::FloatType>(out_type.getElementType())) {
    return false;
  }
  const int64_t exp_other_c = (r_c_dim == 1 ? k_size : m_size);
  if (other_type.getDimSize(other_c_dim) != exp_other_c) return false;
  n_size = other_type.getDimSize(1 - other_c_dim);
  if (n_size <= 0 || n_size % kTpuLaneWidth != 0) return false;
  const int64_t exp_out_r = (r_c_dim == 1 ? m_size : k_size);
  if (out_type.getDimSize(is_lhs ? 0 : 1) != exp_out_r ||
      out_type.getDimSize(is_lhs ? 1 : 0) != n_size) {
    return false;
  }
  mlir::Value dummy_eff_other;
  PeelTransposeOp(other, other_c_dim, dummy_eff_other, eff_other_c_dim);
  return true;
}

[[nodiscard]] bool ResolveDotUser(mlir::OpOperand& use, mlir::Block* block,
                                  mlir::Operation*& dot_user,
                                  mlir::stablehlo::TransposeOp& via_tr,
                                  unsigned& operand_num) {
  mlir::Operation* user = use.getOwner();
  if (llvm::isa<mlir::stablehlo::DotOp, mlir::stablehlo::DotGeneralOp>(user)) {
    dot_user = user;
    operand_num = use.getOperandNumber();
    return true;
  }
  auto tr_u = llvm::dyn_cast<mlir::stablehlo::TransposeOp>(user);
  if (!tr_u) return false;
  auto tr_ty = llvm::dyn_cast<mlir::RankedTensorType>(tr_u.getType());
  if (tr_u->getBlock() != block || !tr_u.getResult().hasOneUse() || !tr_ty ||
      tr_ty.getRank() != 2 ||
      tr_u.getPermutation() != llvm::ArrayRef<int64_t>{1, 0}) {
    return false;
  }
  mlir::OpOperand& tr_use = *tr_u.getResult().getUses().begin();
  dot_user = tr_use.getOwner();
  operand_num = tr_use.getOperandNumber();
  via_tr = tr_u;
  return true;
}

[[nodiscard]] bool ParseContractionUse(
    mlir::OpOperand& use, mlir::stablehlo::ReshapeOp r_op, const int64_t m_size,
    const int64_t k_size, llvm::DenseSet<mlir::Operation*>& seen_dot_users,
    ContractionUse& cu, int64_t& eff_other_c_dim, int64_t& n_size) {
  mlir::Operation* dot_user = nullptr;
  mlir::stablehlo::TransposeOp via_tr = nullptr;
  unsigned operand_num = 0;
  if (!ResolveDotUser(use, r_op->getBlock(), dot_user, via_tr, operand_num)) {
    return false;
  }
  if (dot_user->getBlock() != r_op->getBlock() ||
      !seen_dot_users.insert(dot_user).second) {
    return false;
  }
  bool is_lhs = false;
  int64_t raw_c_dim = -1;
  int64_t other_c_dim = -1;
  mlir::ArrayAttr precision_attr;
  if (!ExtractDotDimensions(dot_user, operand_num, is_lhs, raw_c_dim,
                            other_c_dim, precision_attr)) {
    return false;
  }
  if ((raw_c_dim != 0 && raw_c_dim != 1) ||
      (other_c_dim != 0 && other_c_dim != 1)) {
    return false;
  }
  const int64_t r_c_dim = via_tr ? (1 - raw_c_dim) : raw_c_dim;
  const auto elem_type =
      llvm::cast<mlir::RankedTensorType>(r_op.getType()).getElementType();
  mlir::RankedTensorType out_type;
  if (!ValidateDotTypesAndDims(dot_user, is_lhs, r_c_dim, other_c_dim, m_size,
                               k_size, elem_type, out_type, n_size,
                               eff_other_c_dim)) {
    return false;
  }
  cu = ContractionUse{dot_user,    via_tr,         is_lhs,  r_c_dim,
                      other_c_dim, precision_attr, out_type};
  return true;
}

[[nodiscard]] bool CollectContractionUses(
    mlir::stablehlo::ReshapeOp r_op, const int64_t m_size, const int64_t k_size,
    llvm::SmallVectorImpl<ContractionUse>& uses, bool& has_k_contraction,
    bool& has_m_contraction, bool& has_unfavorable_nk_weight,
    __int128& nk_weight_rows) {
  llvm::DenseSet<mlir::Operation*> seen_dot_users;
  for (mlir::OpOperand& use : r_op.getResult().getUses()) {
    ContractionUse cu;
    int64_t eff_other_c_dim = -1;
    int64_t n_size = -1;
    if (!ParseContractionUse(use, r_op, m_size, k_size, seen_dot_users, cu,
                             eff_other_c_dim, n_size)) {
      return false;
    }
    if (cu.r_c_dim == 1) {
      has_k_contraction = true;
      if (eff_other_c_dim == 1) {
        nk_weight_rows += static_cast<__int128>(n_size);
      }
      if (m_size < n_size) {
        has_unfavorable_nk_weight = true;
      }
    } else {
      has_m_contraction = true;
    }
    uses.push_back(cu);
  }
  return true;
}

[[nodiscard]] bool CheckEligibilityAndCostModel(
    const ProducerViewInfo& info, const int64_t m_size,
    const bool has_k_contraction, const bool has_m_contraction,
    const bool has_unfavorable_nk_weight, const __int128 nk_weight_rows) {
  const bool is_joint_fwd_bwd = has_k_contraction && has_m_contraction;
  const bool is_opaque_producer =
      IsOpaqueCustomCallProducer(info.x, info.sq_x_shape);
  if (!is_joint_fwd_bwd && !is_opaque_producer) return false;
  if (is_joint_fwd_bwd) {
    const __int128 saved_rows = static_cast<__int128>(2) * m_size;
    if (saved_rows < nk_weight_rows) return false;
  } else {
    // Single-direction opaque producer: fold only when M >= N.
    if (has_unfavorable_nk_weight) return false;
  }
  return true;
}

[[nodiscard]] bool IsConvertFromPred(mlir::Value val) {
  if (auto cvt = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    if (auto in_ty = llvm::dyn_cast<mlir::RankedTensorType>(
            cvt.getOperand().getType())) {
      return in_ty.hasStaticShape() && in_ty.getElementType().isInteger(1);
    }
  }
  return false;
}

[[nodiscard]] mlir::Value BuildUnflattenedOther(
    mlir::OpBuilder& builder, const mlir::Location loc, mlir::Value eff_other,
    const int64_t eff_other_c_dim, const mlir::Type elem_type,
    llvm::ArrayRef<int64_t> active_c_sizes, const int64_t n_size,
    llvm::SmallVectorImpl<int64_t>& other_c_dims) {
  llvm::SmallVector<int64_t> unflat_shape;
  if (eff_other_c_dim == 0) {
    unflat_shape.append(active_c_sizes.begin(), active_c_sizes.end());
    unflat_shape.push_back(n_size);
    for (int64_t i = 0; i < static_cast<int64_t>(active_c_sizes.size()); ++i) {
      other_c_dims.push_back(i);
    }
  } else {
    unflat_shape.push_back(n_size);
    unflat_shape.append(active_c_sizes.begin(), active_c_sizes.end());
    for (int64_t i = 0; i < static_cast<int64_t>(active_c_sizes.size()); ++i) {
      other_c_dims.push_back(i + 1);
    }
  }
  if (IsConvertFromPred(eff_other)) {
    auto cvt = eff_other.getDefiningOp<mlir::stablehlo::ConvertOp>();
    auto pred_in = cvt.getOperand();
    auto pred_in_ty = llvm::cast<mlir::RankedTensorType>(pred_in.getType());
    auto pred_unflat_ty =
        mlir::RankedTensorType::get(unflat_shape, pred_in_ty.getElementType());
    mlir::Value unflat_pred = builder.create<mlir::stablehlo::ReshapeOp>(
        loc, pred_unflat_ty, pred_in);
    return builder.create<mlir::stablehlo::ConvertOp>(
        loc, mlir::RankedTensorType::get(unflat_shape, elem_type), unflat_pred);
  }
  return builder.create<mlir::stablehlo::ReshapeOp>(
      loc, mlir::RankedTensorType::get(unflat_shape, elem_type), eff_other);
}

[[nodiscard]] mlir::Value EmitContractingDotGeneral(
    mlir::OpBuilder& builder, const mlir::Location loc, const bool is_lhs,
    mlir::Value x_sq, mlir::Value other_unflat,
    llvm::ArrayRef<int64_t> active_group_dims,
    llvm::ArrayRef<int64_t> other_c_dims, llvm::ArrayRef<int64_t> free_x_sizes,
    const int64_t n_size, const mlir::Type elem_type,
    mlir::ArrayAttr precision_attr) {
  llvm::SmallVector<int64_t> dg_out_shape;
  if (is_lhs) {
    dg_out_shape.append(free_x_sizes.begin(), free_x_sizes.end());
    dg_out_shape.push_back(n_size);
  } else {
    dg_out_shape.push_back(n_size);
    dg_out_shape.append(free_x_sizes.begin(), free_x_sizes.end());
  }
  mlir::Value lhs = is_lhs ? x_sq : other_unflat;
  mlir::Value rhs = is_lhs ? other_unflat : x_sq;
  llvm::ArrayRef<int64_t> lhs_c = is_lhs ? active_group_dims : other_c_dims;
  llvm::ArrayRef<int64_t> rhs_c = is_lhs ? other_c_dims : active_group_dims;
  auto dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(), /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{}, lhs_c, rhs_c);
  auto dg_out_type = mlir::RankedTensorType::get(dg_out_shape, elem_type);
  return builder.create<mlir::stablehlo::DotGeneralOp>(
      loc, dg_out_type, lhs, rhs, dot_dims, precision_attr,
      /*algorithm=*/nullptr);
}

void RewriteContractionUse(
    const ContractionUse& u, mlir::Value x_sq, const mlir::Type elem_type,
    llvm::ArrayRef<int64_t> sq_g0_dims, llvm::ArrayRef<int64_t> sq_g1_dims,
    llvm::ArrayRef<int64_t> g0_sizes, llvm::ArrayRef<int64_t> g1_sizes,
    llvm::SmallVectorImpl<mlir::stablehlo::TransposeOp>& peeled_trs_to_erase) {
  mlir::OpBuilder builder(u.dot_user);
  mlir::Value other = u.dot_user->getOperand(u.is_lhs ? 1 : 0);
  mlir::Value eff_other = other;
  int64_t eff_other_c_dim = u.other_c_dim;
  PeelTransposeOp(other, u.other_c_dim, eff_other, eff_other_c_dim,
                  &peeled_trs_to_erase);

  llvm::ArrayRef<int64_t> active_group_dims =
      (u.r_c_dim == 1) ? sq_g1_dims : sq_g0_dims;
  llvm::ArrayRef<int64_t> active_c_sizes =
      (u.r_c_dim == 1) ? g1_sizes : g0_sizes;
  llvm::ArrayRef<int64_t> free_x_sizes = (u.r_c_dim == 1) ? g0_sizes : g1_sizes;
  const int64_t n_size = u.out_type.getDimSize(u.is_lhs ? 1 : 0);

  llvm::SmallVector<int64_t> other_c_dims;
  mlir::Value other_unflat = BuildUnflattenedOther(
      builder, u.dot_user->getLoc(), eff_other, eff_other_c_dim, elem_type,
      active_c_sizes, n_size, other_c_dims);

  mlir::Value dg = EmitContractingDotGeneral(
      builder, u.dot_user->getLoc(), u.is_lhs, x_sq, other_unflat,
      active_group_dims, other_c_dims, free_x_sizes, n_size,
      u.out_type.getElementType(), u.precision_attr);

  mlir::Value replacement = dg;
  if (dg.getType() != u.out_type) {
    replacement = builder.create<mlir::stablehlo::ReshapeOp>(
        u.dot_user->getLoc(), u.out_type, dg);
  }
  u.dot_user->getResult(0).replaceAllUsesWith(replacement);
}

void EraseDeadOps(mlir::stablehlo::ReshapeOp r_op,
                  mlir::stablehlo::TransposeOp t_op,
                  llvm::ArrayRef<ContractionUse> uses,
                  llvm::ArrayRef<mlir::stablehlo::TransposeOp> peeled_trs,
                  llvm::DenseSet<mlir::Operation*>& erased) {
  for (ContractionUse u : uses) {
    erased.insert(u.dot_user);
    u.dot_user->erase();
    if (u.via_tr != nullptr && u.via_tr->use_empty()) {
      erased.insert(u.via_tr.getOperation());
      u.via_tr.erase();
    }
  }
  llvm::SmallDenseSet<mlir::Operation*, 8> seen;
  for (auto tr : peeled_trs) {
    if (!tr || !seen.insert(tr.getOperation()).second) {
      continue;
    }
    if (tr->use_empty()) {
      erased.insert(tr.getOperation());
      tr.erase();
    }
  }
  if (r_op->use_empty()) {
    erased.insert(r_op.getOperation());
    r_op.erase();
  }
  if (t_op->use_empty()) {
    erased.insert(t_op.getOperation());
    t_op.erase();
  }
}

bool TryFoldBatchHeadMergedContractions(
    mlir::stablehlo::ReshapeOp r_op, llvm::DenseSet<mlir::Operation*>& erased) {
  ProducerViewInfo info;
  if (!MatchProducerView(r_op, info)) return false;
  int64_t m_size = 0;
  int64_t k_size = 0;
  if (!ValidateReshapeOp(r_op, m_size, k_size)) return false;
  llvm::SmallVector<ContractionUse> uses;
  bool has_k_contraction = false;
  bool has_m_contraction = false;
  bool has_unfavorable_nk_weight = false;
  __int128 nk_weight_rows = 0;
  if (!CollectContractionUses(r_op, m_size, k_size, uses, has_k_contraction,
                              has_m_contraction, has_unfavorable_nk_weight,
                              nk_weight_rows)) {
    return false;
  }
  if (!CheckEligibilityAndCostModel(
          info, m_size, has_k_contraction, has_m_contraction,
          has_unfavorable_nk_weight, nk_weight_rows)) {
    return false;
  }

  mlir::OpBuilder builder(r_op);
  auto x_type = llvm::cast<mlir::RankedTensorType>(info.x.getType());
  mlir::Value x_sq = info.x;
  if (static_cast<size_t>(x_type.getRank()) != info.sq_x_shape.size()) {
    x_sq = builder.create<mlir::stablehlo::ReshapeOp>(
        r_op.getLoc(),
        mlir::RankedTensorType::get(info.sq_x_shape, x_type.getElementType()),
        info.x);
  }

  llvm::SmallVector<mlir::stablehlo::TransposeOp> peeled_trs_to_erase;
  for (const ContractionUse& u : uses) {
    RewriteContractionUse(u, x_sq, x_type.getElementType(), info.sq_g0_dims,
                          info.sq_g1_dims, info.g0_sizes, info.g1_sizes,
                          peeled_trs_to_erase);
  }
  EraseDeadOps(r_op, info.t_op, uses, peeled_trs_to_erase, erased);
  return true;
}

// =============================================================================
// Mechanism 3: FoldTransposeReshapeDot
// =============================================================================

struct SplitTransposeInfo {
  mlir::Value orig_input;
  mlir::stablehlo::TransposeOp transpose_op;
  mlir::stablehlo::ReshapeOp reshape_op;
  llvm::SmallVector<int64_t, 4> contract_dims;
  llvm::SmallVector<int64_t, 4> contract_sizes;
  llvm::SmallVector<int64_t, 4> dot_nc_dims;
  llvm::SmallVector<int64_t, 4> dot_nc_sizes;
  llvm::SmallVector<int64_t, 4> target_nc_sizes;
  llvm::SmallVector<int64_t, 4> post_dot_perm;
};

[[nodiscard]] llvm::SmallVector<int64_t, 4> GatherDimSizes(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> indices) {
  llvm::SmallVector<int64_t, 4> res;
  res.reserve(indices.size());
  for (const int64_t idx : indices) {
    res.push_back(shape[idx]);
  }
  return res;
}

[[nodiscard]] llvm::SmallVector<int64_t, 4> ExtractNonUnitDims(
    llvm::ArrayRef<int64_t> perm, llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t, 4> dims;
  for (const int64_t d : perm) {
    if (shape[d] > 1) {
      dims.push_back(d);
    }
  }
  return dims;
}

[[nodiscard]] bool HasCrossingNonContractDims(llvm::ArrayRef<int64_t> nc_perm,
                                              llvm::ArrayRef<int64_t> x_shape,
                                              int64_t first_c_dim) {
  for (const int64_t d : nc_perm) {
    if (x_shape[d] > 1 && d > first_c_dim) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool ComputePostDotPermutation(
    llvm::ArrayRef<int64_t> nc_perm, llvm::ArrayRef<int64_t> dot_nc_dims,
    llvm::SmallVectorImpl<int64_t>& post_dot_perm) {
  for (const int64_t target_d : nc_perm) {
    auto it = llvm::find(dot_nc_dims, target_d);
    if (it == dot_nc_dims.end()) return false;
    post_dot_perm.push_back(std::distance(dot_nc_dims.begin(), it));
  }
  post_dot_perm.push_back(static_cast<int64_t>(dot_nc_dims.size()));
  return true;
}

std::optional<SplitTransposeInfo> TryBuildSplitTransposeInfo(
    mlir::stablehlo::TransposeOp transpose_op,
    mlir::stablehlo::ReshapeOp reshape_op, mlir::Value orig_input,
    llvm::ArrayRef<int64_t> x_shape, llvm::ArrayRef<int64_t> perm,
    const int64_t split, const int64_t d0, const int64_t d1) {
  llvm::ArrayRef<int64_t> nc_perm = perm.slice(0, split);
  llvm::ArrayRef<int64_t> c_perm = perm.drop_front(split);
  if (!HasDimProduct(GatherDimSizes(x_shape, nc_perm), d0) ||
      !HasDimProduct(GatherDimSizes(x_shape, c_perm), d1)) {
    return std::nullopt;
  }
  llvm::SmallVector<int64_t, 4> c_dims = ExtractNonUnitDims(c_perm, x_shape);
  if (c_dims.empty() || !IsStrictlyAscending(c_dims)) return std::nullopt;
  llvm::SmallVector<int64_t, 4> c_sizes = GatherDimSizes(x_shape, c_dims);

  llvm::SmallVector<int64_t, 4> nc_non_unit_dims =
      ExtractNonUnitDims(nc_perm, x_shape);
  const bool crosses_groups =
      HasCrossingNonContractDims(nc_perm, x_shape, c_dims.front());
  const bool nc_permutes_axes = !IsStrictlyAscending(nc_non_unit_dims);
  const bool contracts_leading_axis =
      (c_dims.front() == 0 && c_sizes.back() % kTpuLaneWidth == 0 &&
       nc_non_unit_dims.size() == nc_perm.size());
  if (!crosses_groups || (!nc_permutes_axes && !contracts_leading_axis)) {
    return std::nullopt;
  }

  const int64_t rank = static_cast<int64_t>(x_shape.size());
  llvm::SmallVector<int64_t, 4> dot_nc_dims;
  llvm::SmallVector<int64_t, 4> dot_nc_sizes;
  for (int64_t d = 0; d < rank; ++d) {
    if (!llvm::is_contained(c_dims, d)) {
      dot_nc_dims.push_back(d);
      dot_nc_sizes.push_back(x_shape[d]);
    }
  }
  if (nc_perm.size() != dot_nc_dims.size()) return std::nullopt;

  SplitTransposeInfo info;
  info.orig_input = orig_input;
  info.transpose_op = transpose_op;
  info.reshape_op = reshape_op;
  info.contract_dims = c_dims;
  info.contract_sizes = c_sizes;
  info.dot_nc_dims = dot_nc_dims;
  info.dot_nc_sizes = dot_nc_sizes;
  info.target_nc_sizes = GatherDimSizes(x_shape, nc_perm);

  if (nc_permutes_axes &&
      !ComputePostDotPermutation(nc_perm, dot_nc_dims, info.post_dot_perm)) {
    return std::nullopt;
  }
  return info;
}

std::optional<SplitTransposeInfo> MatchTransposeReshape2D(mlir::Value val) {
  auto reshape_op = val.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!reshape_op) return std::nullopt;
  auto r_type = llvm::dyn_cast<mlir::RankedTensorType>(reshape_op.getType());
  if (!r_type || !r_type.hasStaticShape() || r_type.getRank() != 2) {
    return std::nullopt;
  }
  mlir::Value r_in = PeelIdentityBroadcast(reshape_op.getOperand());
  auto transpose_op = r_in.getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!transpose_op) return std::nullopt;
  mlir::Value orig_input = PeelIdentityBroadcast(transpose_op.getOperand());
  auto x_type = llvm::dyn_cast<mlir::RankedTensorType>(orig_input.getType());
  if (!x_type || !x_type.hasStaticShape() || x_type.getRank() < 3) {
    return std::nullopt;
  }
  const int64_t d0 = r_type.getDimSize(0);
  const int64_t d1 = r_type.getDimSize(1);
  if (d0 <= 0 || d1 <= 0) return std::nullopt;
  llvm::ArrayRef<int64_t> x_shape = x_type.getShape();
  llvm::ArrayRef<int64_t> perm = transpose_op.getPermutation();
  const int64_t rank = x_type.getRank();
  for (int64_t split = 1; split < rank; ++split) {
    if (auto info =
            TryBuildSplitTransposeInfo(transpose_op, reshape_op, orig_input,
                                       x_shape, perm, split, d0, d1)) {
      return info;
    }
  }
  return std::nullopt;
}

bool Extract2DDotParams(mlir::Operation* op, mlir::Value& lhs_out,
                        mlir::Value& rhs_out, int64_t& rhs_c_dim_out,
                        mlir::RankedTensorType& out_2d_type_out,
                        mlir::ArrayAttr& precision_config_out) {
  if (auto dot_op = llvm::dyn_cast<mlir::stablehlo::DotOp>(op)) {
    auto lhs_type =
        llvm::dyn_cast<mlir::RankedTensorType>(dot_op.getLhs().getType());
    auto rhs_type =
        llvm::dyn_cast<mlir::RankedTensorType>(dot_op.getRhs().getType());
    auto out_type = llvm::dyn_cast<mlir::RankedTensorType>(dot_op.getType());
    if (!lhs_type || !rhs_type || !out_type || !lhs_type.hasStaticShape() ||
        !rhs_type.hasStaticShape() || !out_type.hasStaticShape() ||
        lhs_type.getRank() != 2 || rhs_type.getRank() != 2 ||
        out_type.getRank() != 2 ||
        lhs_type.getElementType() != rhs_type.getElementType() ||
        lhs_type.getElementType() != out_type.getElementType()) {
      return false;
    }
    lhs_out = dot_op.getLhs();
    rhs_out = dot_op.getRhs();
    rhs_c_dim_out = 0;
    out_2d_type_out = out_type;
    precision_config_out = dot_op.getPrecisionConfigAttr();
    return true;
  }
  if (auto dg_op = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
    if (dg_op.getAlgorithmAttr() != nullptr) return false;
    auto lhs_type =
        llvm::dyn_cast<mlir::RankedTensorType>(dg_op.getLhs().getType());
    auto rhs_type =
        llvm::dyn_cast<mlir::RankedTensorType>(dg_op.getRhs().getType());
    auto out_type = llvm::dyn_cast<mlir::RankedTensorType>(dg_op.getType());
    if (!lhs_type || !rhs_type || !out_type || !lhs_type.hasStaticShape() ||
        !rhs_type.hasStaticShape() || !out_type.hasStaticShape() ||
        lhs_type.getRank() != 2 || rhs_type.getRank() != 2 ||
        out_type.getRank() != 2 ||
        lhs_type.getElementType() != rhs_type.getElementType() ||
        lhs_type.getElementType() != out_type.getElementType()) {
      return false;
    }
    auto dnums = dg_op.getDotDimensionNumbers();
    if (!dnums.getLhsBatchingDimensions().empty() ||
        !dnums.getRhsBatchingDimensions().empty() ||
        dnums.getLhsContractingDimensions() != llvm::ArrayRef<int64_t>({1}) ||
        dnums.getRhsContractingDimensions().size() != 1) {
      return false;
    }
    lhs_out = dg_op.getLhs();
    rhs_out = dg_op.getRhs();
    rhs_c_dim_out = dnums.getRhsContractingDimensions()[0];
    out_2d_type_out = out_type;
    precision_config_out = dg_op.getPrecisionConfigAttr();
    return true;
  }
  return false;
}

[[nodiscard]] mlir::Value BuildReshapedRhs(
    mlir::OpBuilder& builder, const mlir::Location loc, mlir::Value rhs,
    const int64_t rhs_c_dim, const int64_t out_features,
    llvm::ArrayRef<int64_t> contract_sizes,
    llvm::SmallVectorImpl<int64_t>& new_rhs_c_dims) {
  mlir::Value rhs_val = rhs;
  int64_t effective_rhs_c_dim = rhs_c_dim;
  if (auto wt = rhs_val.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    if (IsStaticRankedTensor(wt.getOperand().getType()) &&
        wt.getPermutation() == llvm::ArrayRef<int64_t>({1, 0})) {
      rhs_val = wt.getOperand();
      effective_rhs_c_dim = 1 - rhs_c_dim;
    }
  }

  const int64_t num_c = static_cast<int64_t>(contract_sizes.size());
  const int64_t c_offset = (effective_rhs_c_dim == 1) ? 1 : 0;
  llvm::SmallVector<int64_t, 4> new_rhs_shape;
  if (effective_rhs_c_dim == 1) {
    new_rhs_shape.push_back(out_features);
  }
  for (int64_t i = 0; i < num_c; ++i) {
    new_rhs_shape.push_back(contract_sizes[i]);
    new_rhs_c_dims.push_back(c_offset + i);
  }
  if (effective_rhs_c_dim != 1) {
    new_rhs_shape.push_back(out_features);
  }

  auto elem_type =
      llvm::cast<mlir::RankedTensorType>(rhs.getType()).getElementType();
  auto new_rhs_type = mlir::RankedTensorType::get(new_rhs_shape, elem_type);
  return builder.create<mlir::stablehlo::ReshapeOp>(loc, new_rhs_type, rhs_val);
}

void TryMatchSingleUseBiasAdd(mlir::Value dot_result, mlir::Operation* user,
                              mlir::stablehlo::AddOp& bias_add,
                              mlir::stablehlo::BroadcastInDimOp& bias_bcast) {
  auto add_user = llvm::dyn_cast<mlir::stablehlo::AddOp>(user);
  if (!add_user || !dot_result.hasOneUse()) return;
  if (dot_result.getDefiningOp() == nullptr ||
      add_user->getBlock() != dot_result.getDefiningOp()->getBlock()) {
    return;
  }
  mlir::Value other =
      add_user.getLhs() == dot_result ? add_user.getRhs() : add_user.getLhs();
  auto bcast = other.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast) return;
  auto bcast_type =
      llvm::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
  if (!bcast_type || !bcast_type.hasStaticShape()) return;
  llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
  if (bdims.empty() || bdims == llvm::ArrayRef<int64_t>({1})) {
    bias_add = add_user;
    bias_bcast = bcast;
  }
}

[[nodiscard]] bool IsLegalSplitFor2DDotFold(SplitTransposeInfo& split) {
  if (!split.reshape_op.getResult().hasOneUse() ||
      !split.transpose_op.getResult().hasOneUse()) {
    return false;
  }
  const int64_t minor_c_size = split.contract_sizes.back();
  const bool is_lane_aligned_leading_contract =
      (split.post_dot_perm.empty() && minor_c_size % kTpuLaneWidth == 0);
  const bool is_single_use_permuted_nc =
      (!split.post_dot_perm.empty() && minor_c_size % kTpuSublaneCount == 0);
  return is_lane_aligned_leading_contract || is_single_use_permuted_nc;
}

[[nodiscard]] mlir::Value ApplyBiasAddIfPresent(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value dot_result,
    mlir::Value nd_val, mlir::RankedTensorType dot_out_type,
    mlir::stablehlo::AddOp bias_add,
    mlir::stablehlo::BroadcastInDimOp bias_bcast) {
  if (!bias_add || !bias_bcast) return nd_val;
  llvm::SmallVector<int64_t, 1> new_bdims;
  if (!bias_bcast.getBroadcastDimensions().empty()) {
    new_bdims.push_back(dot_out_type.getRank() - 1);
  }
  mlir::Value new_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, dot_out_type, bias_bcast.getOperand(),
      builder.getDenseI64ArrayAttr(new_bdims));
  mlir::Value add_lhs = bias_add.getLhs() == dot_result ? nd_val : new_bcast;
  mlir::Value add_rhs = bias_add.getLhs() == dot_result ? new_bcast : nd_val;
  return builder.create<mlir::stablehlo::AddOp>(loc, dot_out_type, add_lhs,
                                                add_rhs);
}

void ReplaceFoldedDotUsers(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Operation* producer_to_replace,
                           mlir::Value target_nd_val,
                           mlir::RankedTensorType target_out_type,
                           mlir::RankedTensorType out_2d_type,
                           llvm::DenseSet<mlir::Operation*>& erased) {
  llvm::SmallVector<mlir::Operation*, 4> direct_target_reshapes;
  bool has_other_users = false;
  for (mlir::Operation* u : producer_to_replace->getResult(0).getUsers()) {
    auto ru = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(u);
    if (ru && ru.getType() == target_out_type) {
      direct_target_reshapes.push_back(ru);
    } else {
      has_other_users = true;
    }
  }
  for (mlir::Operation* ru : direct_target_reshapes) {
    ru->replaceAllUsesWith(mlir::ValueRange{target_nd_val});
    erased.insert(ru);
    ru->erase();
  }
  if (has_other_users) {
    mlir::Value fallback_2d = builder.create<mlir::stablehlo::ReshapeOp>(
        loc, out_2d_type, target_nd_val);
    producer_to_replace->getResult(0).replaceAllUsesWith(fallback_2d);
  }
}

bool TryFoldTransposeReshapeDot(mlir::Operation* op,
                                llvm::DenseSet<mlir::Operation*>& erased) {
  mlir::Value lhs;
  mlir::Value rhs;
  int64_t rhs_c_dim = 0;
  mlir::RankedTensorType out_2d_type;
  mlir::ArrayAttr precision_config;
  if (!Extract2DDotParams(op, lhs, rhs, rhs_c_dim, out_2d_type,
                          precision_config)) {
    return false;
  }

  auto split_opt = MatchTransposeReshape2D(lhs);
  if (!split_opt.has_value()) return false;
  SplitTransposeInfo& split = *split_opt;
  auto rhs_type = llvm::cast<mlir::RankedTensorType>(rhs.getType());
  const int64_t out_features = rhs_type.getDimSize(1 - rhs_c_dim);
  if (out_features <= 0 || out_features % kTpuLaneWidth != 0) return false;
  if (!IsLegalSplitFor2DDotFold(split)) return false;

  mlir::OpBuilder builder(op);
  const mlir::Location loc = op->getLoc();
  llvm::SmallVector<int64_t, 4> new_rhs_c_dims;
  mlir::Value new_rhs =
      BuildReshapedRhs(builder, loc, rhs, rhs_c_dim, out_features,
                       split.contract_sizes, new_rhs_c_dims);

  llvm::SmallVector<int64_t, 4> dot_out_shape(split.dot_nc_sizes.begin(),
                                              split.dot_nc_sizes.end());
  dot_out_shape.push_back(out_features);
  auto dot_out_type =
      mlir::RankedTensorType::get(dot_out_shape, out_2d_type.getElementType());

  llvm::SmallVector<int64_t, 4> target_out_shape(split.target_nc_sizes.begin(),
                                                 split.target_nc_sizes.end());
  target_out_shape.push_back(out_features);
  auto target_out_type = mlir::RankedTensorType::get(
      target_out_shape, out_2d_type.getElementType());

  auto new_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(), /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{}, split.contract_dims, new_rhs_c_dims);

  auto new_dot = builder.create<mlir::stablehlo::DotGeneralOp>(
      loc, dot_out_type, split.orig_input, new_rhs, new_dot_dims,
      precision_config,
      /*algorithm=*/nullptr);

  const mlir::Value dot_result = op->getResult(0);
  mlir::stablehlo::AddOp bias_add = nullptr;
  mlir::stablehlo::BroadcastInDimOp bias_bcast = nullptr;
  for (mlir::Operation* user : dot_result.getUsers()) {
    TryMatchSingleUseBiasAdd(dot_result, user, bias_add, bias_bcast);
  }

  mlir::Operation* producer_to_replace =
      bias_add ? bias_add.getOperation() : op;
  mlir::OpBuilder out_builder(producer_to_replace);
  const mlir::Location out_loc = producer_to_replace->getLoc();
  mlir::Value nd_val = ApplyBiasAddIfPresent(out_builder, out_loc, dot_result,
                                             new_dot.getResult(), dot_out_type,
                                             bias_add, bias_bcast);

  mlir::Value target_nd_val = nd_val;
  if (!split.post_dot_perm.empty()) {
    target_nd_val = out_builder.create<mlir::stablehlo::TransposeOp>(
        out_loc, target_out_type, nd_val,
        out_builder.getDenseI64ArrayAttr(split.post_dot_perm));
  }

  ReplaceFoldedDotUsers(out_builder, out_loc, producer_to_replace,
                        target_nd_val, target_out_type, out_2d_type, erased);
  if (bias_add) {
    erased.insert(bias_add.getOperation());
    bias_add.erase();
  }
  erased.insert(op);
  op->erase();
  return true;
}

// =============================================================================
// Mechanism 4: UnflattenBatchedDotGeneral
// =============================================================================

struct UnflattenedOperandInfo {
  mlir::Value tensor_nd;
  llvm::SmallVector<int64_t, 4> batch_dims;
  int64_t contracting_dim = -1;
  llvm::SmallVector<int64_t, 4> batch_sizes;
  bool absorbed_transpose = false;
};

[[nodiscard]] mlir::Value Peel3DBatchTranspose(
    mlir::Value val_3d, const int64_t contracting_axis_3d,
    int64_t& pre_tr_contracting_3d) {
  pre_tr_contracting_3d = contracting_axis_3d;
  if (auto tr_3d = val_3d.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    if (!tr_3d.getResult().hasOneUse()) return val_3d;
    llvm::ArrayRef<int64_t> perm_3d = tr_3d.getPermutation();
    if (perm_3d.size() == 3 && perm_3d[0] == 0 && perm_3d[1] == 2 &&
        perm_3d[2] == 1) {
      pre_tr_contracting_3d = (contracting_axis_3d == 1) ? 2 : 1;
      return tr_3d.getOperand();
    }
  }
  return val_3d;
}

[[nodiscard]] bool HasMatchingBatchAndMatrixDims(
    llvm::ArrayRef<int64_t> nd_shape, mlir::RankedTensorType resh_out_type,
    const int64_t rank_nd, const int64_t batch_rank) {
  llvm::ArrayRef<int64_t> b_sizes = nd_shape.slice(0, batch_rank);
  return HasDimProduct(b_sizes, resh_out_type.getDimSize(0)) &&
         nd_shape[rank_nd - 2] == resh_out_type.getDimSize(1) &&
         nd_shape[rank_nd - 1] == resh_out_type.getDimSize(2);
}

[[nodiscard]] bool TryAbsorbNdBatchTranspose(mlir::Value val_nd,
                                             const int64_t rank_nd,
                                             const int64_t batch_rank,
                                             const int64_t axis_in_val_nd,
                                             llvm::ArrayRef<int64_t> b_sizes,
                                             UnflattenedOperandInfo& out_info) {
  auto tr_nd = val_nd.getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!tr_nd || !tr_nd.getResult().hasOneUse()) return false;
  auto src_nd_type =
      llvm::dyn_cast<mlir::RankedTensorType>(tr_nd.getOperand().getType());
  llvm::ArrayRef<int64_t> perm = tr_nd.getPermutation();
  if (!src_nd_type || !src_nd_type.hasStaticShape() ||
      src_nd_type.getRank() != rank_nd ||
      static_cast<int64_t>(perm.size()) != rank_nd) {
    return false;
  }
  bool non_trivial_batch_perm = false;
  for (int64_t i = 0; i < batch_rank; ++i) {
    if (perm[i] != i) {
      non_trivial_batch_perm = true;
      break;
    }
  }
  if (!non_trivial_batch_perm) return false;

  out_info.tensor_nd = tr_nd.getOperand();
  out_info.batch_dims.clear();
  for (int64_t i = 0; i < batch_rank; ++i) {
    out_info.batch_dims.push_back(perm[i]);
  }
  out_info.contracting_dim = perm[axis_in_val_nd];
  out_info.batch_sizes.assign(b_sizes.begin(), b_sizes.end());
  out_info.absorbed_transpose = true;
  return true;
}

bool Match3DBatchedOperand(mlir::Value val_3d,
                           const int64_t contracting_axis_3d,
                           UnflattenedOperandInfo& out_info) {
  auto type_3d = llvm::dyn_cast<mlir::RankedTensorType>(val_3d.getType());
  if (!type_3d || !type_3d.hasStaticShape() || type_3d.getRank() != 3) {
    return false;
  }
  if (contracting_axis_3d != 1 && contracting_axis_3d != 2) return false;

  int64_t pre_tr_contracting_3d = contracting_axis_3d;
  mlir::Value cur =
      Peel3DBatchTranspose(val_3d, contracting_axis_3d, pre_tr_contracting_3d);

  auto resh_op = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!resh_op || !resh_op.getResult().hasOneUse()) return false;
  mlir::Value val_nd = resh_op.getOperand();
  auto type_nd = llvm::dyn_cast<mlir::RankedTensorType>(val_nd.getType());
  auto resh_out_type =
      llvm::dyn_cast<mlir::RankedTensorType>(resh_op.getType());
  if (!type_nd || !type_nd.hasStaticShape() || type_nd.getRank() < 4 ||
      !resh_out_type || !resh_out_type.hasStaticShape()) {
    return false;
  }

  const int64_t rank_nd = type_nd.getRank();
  const int64_t batch_rank = rank_nd - 2;
  llvm::ArrayRef<int64_t> nd_shape = type_nd.getShape();
  llvm::ArrayRef<int64_t> b_sizes = nd_shape.slice(0, batch_rank);
  if (!HasMatchingBatchAndMatrixDims(nd_shape, resh_out_type, rank_nd,
                                     batch_rank)) {
    return false;
  }

  const int64_t axis_in_val_nd = batch_rank - 1 + pre_tr_contracting_3d;
  if (val_nd.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    return TryAbsorbNdBatchTranspose(val_nd, rank_nd, batch_rank,
                                     axis_in_val_nd, b_sizes, out_info);
  }

  out_info.tensor_nd = val_nd;
  out_info.batch_dims.clear();
  for (int64_t i = 0; i < batch_rank; ++i) {
    out_info.batch_dims.push_back(i);
  }
  out_info.contracting_dim = axis_in_val_nd;
  out_info.batch_sizes.assign(b_sizes.begin(), b_sizes.end());
  out_info.absorbed_transpose = false;
  return true;
}

[[nodiscard]] bool HasStatic3DRankedShapes(mlir::RankedTensorType out_type,
                                           mlir::RankedTensorType lhs_type,
                                           mlir::RankedTensorType rhs_type) {
  auto is_static_3d = [](mlir::RankedTensorType t) {
    return t && t.hasStaticShape() && t.getRank() == 3;
  };
  return is_static_3d(out_type) && is_static_3d(lhs_type) &&
         is_static_3d(rhs_type);
}

[[nodiscard]] bool HasSingleBatchDotDimensions(
    mlir::stablehlo::DotDimensionNumbersAttr dnums) {
  llvm::ArrayRef<int64_t> lhs_batch = dnums.getLhsBatchingDimensions();
  llvm::ArrayRef<int64_t> rhs_batch = dnums.getRhsBatchingDimensions();
  llvm::ArrayRef<int64_t> lhs_contract = dnums.getLhsContractingDimensions();
  llvm::ArrayRef<int64_t> rhs_contract = dnums.getRhsContractingDimensions();
  return lhs_batch.size() == 1 && rhs_batch.size() == 1 && lhs_batch[0] == 0 &&
         rhs_batch[0] == 0 && lhs_contract.size() == 1 &&
         rhs_contract.size() == 1;
}

void ReplaceDirectUnflattenReshapes(mlir::stablehlo::DotGeneralOp dot_op,
                                    mlir::Value new_dot_result,
                                    mlir::RankedTensorType dot_nd_type,
                                    llvm::DenseSet<mlir::Operation*>& erased) {
  llvm::SmallVector<mlir::stablehlo::ReshapeOp> direct_unflatten_users;
  for (mlir::Operation* user : dot_op->getUsers()) {
    auto resh = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(user);
    if (resh && resh.getType() == dot_nd_type) {
      direct_unflatten_users.push_back(resh);
    }
  }
  for (mlir::stablehlo::ReshapeOp resh : direct_unflatten_users) {
    resh.replaceAllUsesWith(new_dot_result);
    erased.insert(resh.getOperation());
    resh.erase();
  }
}

bool TryUnflatten3DDotGeneral(mlir::stablehlo::DotGeneralOp dot_op,
                              llvm::DenseSet<mlir::Operation*>& erased) {
  auto out_type = llvm::dyn_cast<mlir::RankedTensorType>(dot_op.getType());
  auto lhs_type =
      llvm::dyn_cast<mlir::RankedTensorType>(dot_op.getLhs().getType());
  auto rhs_type =
      llvm::dyn_cast<mlir::RankedTensorType>(dot_op.getRhs().getType());
  if (!HasStatic3DRankedShapes(out_type, lhs_type, rhs_type)) {
    return false;
  }

  auto dnums = dot_op.getDotDimensionNumbers();
  if (!HasSingleBatchDotDimensions(dnums)) {
    return false;
  }
  llvm::ArrayRef<int64_t> lhs_contract = dnums.getLhsContractingDimensions();
  llvm::ArrayRef<int64_t> rhs_contract = dnums.getRhsContractingDimensions();

  UnflattenedOperandInfo lhs_info;
  UnflattenedOperandInfo rhs_info;
  if (!Match3DBatchedOperand(dot_op.getLhs(), lhs_contract[0], lhs_info) ||
      !Match3DBatchedOperand(dot_op.getRhs(), rhs_contract[0], rhs_info)) {
    return false;
  }
  if (lhs_info.batch_sizes != rhs_info.batch_sizes) return false;
  if (!lhs_info.absorbed_transpose && !rhs_info.absorbed_transpose) {
    return false;
  }

  mlir::OpBuilder builder(dot_op);
  auto new_dnums = mlir::stablehlo::DotDimensionNumbersAttr::get(
      dot_op.getContext(), lhs_info.batch_dims, rhs_info.batch_dims,
      llvm::ArrayRef<int64_t>{lhs_info.contracting_dim},
      llvm::ArrayRef<int64_t>{rhs_info.contracting_dim});

  llvm::SmallVector<int64_t, 6> dot_nd_shape(lhs_info.batch_sizes.begin(),
                                             lhs_info.batch_sizes.end());
  dot_nd_shape.push_back(out_type.getDimSize(1));
  dot_nd_shape.push_back(out_type.getDimSize(2));
  auto dot_nd_type =
      mlir::RankedTensorType::get(dot_nd_shape, out_type.getElementType());
  auto new_dot = builder.create<mlir::stablehlo::DotGeneralOp>(
      dot_op.getLoc(), dot_nd_type, lhs_info.tensor_nd, rhs_info.tensor_nd,
      new_dnums, dot_op.getPrecisionConfigAttr(), dot_op.getAlgorithmAttr());

  ReplaceDirectUnflattenReshapes(dot_op, new_dot.getResult(), dot_nd_type,
                                 erased);

  if (!dot_op->use_empty()) {
    auto reshaped_3d = builder.create<mlir::stablehlo::ReshapeOp>(
        dot_op.getLoc(), out_type, new_dot.getResult());
    dot_op.replaceAllUsesWith(reshaped_3d.getResult());
  }
  erased.insert(dot_op.getOperation());
  dot_op.erase();
  return true;
}

struct FoldDotTransposesPass
    : public mlir::PassWrapper<FoldDotTransposesPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldDotTransposesPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-fold-dot-transposes";
  }
  llvm::StringRef getDescription() const override {
    return "Folds transpose and reshape chains directly into dot contractions.";
  }

  static bool RunPhase1Unflatten(mlir::func::FuncOp func) {
    llvm::DenseSet<mlir::Operation*> erased;
    llvm::SmallVector<mlir::stablehlo::DotGeneralOp, 16> batched_dots;
    func.walk([&batched_dots](mlir::stablehlo::DotGeneralOp op) {
      batched_dots.push_back(op);
    });
    bool changed = false;
    for (mlir::stablehlo::DotGeneralOp op : batched_dots) {
      if (!erased.contains(op.getOperation()) && op->getBlock() != nullptr) {
        if (TryUnflatten3DDotGeneral(op, erased)) {
          changed = true;
        }
      }
    }
    if (changed) {
      EraseTriviallyDeadOps(func);
    }
    return changed;
  }

  static bool RunPhase2BatchHeadMerged(mlir::func::FuncOp func) {
    llvm::DenseSet<mlir::Operation*> erased;
    llvm::SmallVector<mlir::stablehlo::ReshapeOp, 16> reshapes;
    func.walk(
        [&reshapes](mlir::stablehlo::ReshapeOp op) { reshapes.push_back(op); });
    bool changed = false;
    for (mlir::stablehlo::ReshapeOp op : reshapes) {
      if (!erased.contains(op.getOperation()) && op->getBlock() != nullptr) {
        if (TryFoldBatchHeadMergedContractions(op, erased)) {
          changed = true;
        }
      }
    }
    if (changed) {
      EraseTriviallyDeadOps(func);
    }
    return changed;
  }

  static bool RunPhase3FoldTransposeReshape(mlir::func::FuncOp func) {
    llvm::DenseSet<mlir::Operation*> erased;
    llvm::SmallVector<mlir::Operation*, 16> dot_ops;
    func.walk([&dot_ops](mlir::Operation* op) {
      if (llvm::isa<mlir::stablehlo::DotOp, mlir::stablehlo::DotGeneralOp>(
              op)) {
        dot_ops.push_back(op);
      }
    });
    bool changed = false;
    for (mlir::Operation* op : dot_ops) {
      if (!erased.contains(op) && op->getBlock() != nullptr) {
        if (TryFoldTransposeReshapeDot(op, erased)) {
          changed = true;
        }
      }
    }
    if (changed) {
      EraseTriviallyDeadOps(func);
    }
    return changed;
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    for (int iter = 0; iter < kMaxPassIterations; ++iter) {
      bool changed = false;
      if (RunPhase1Unflatten(func)) changed = true;
      if (RunPhase2BatchHeadMerged(func)) changed = true;
      if (RunPhase3FoldTransposeReshape(func)) changed = true;
      if (!changed) break;
    }
    EraseTriviallyDeadOps(func);
  }

 private:
  static bool EraseTriviallyDeadOps(mlir::func::FuncOp func) {
    bool any_erased = false;
    bool changed = true;
    while (changed) {
      changed = false;
      func.walk([&changed, &any_erased](mlir::Block* block) {
        for (auto it = block->rbegin(); it != block->rend();) {
          mlir::Operation* op = &(*it++);
          if (mlir::isOpTriviallyDead(op)) {
            op->erase();
            changed = true;
            any_erased = true;
          }
        }
      });
    }
    return any_erased;
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createFoldDotTransposesPass() {
  return std::make_unique<FoldDotTransposesPass>();
}

}  // namespace torch_tpu
