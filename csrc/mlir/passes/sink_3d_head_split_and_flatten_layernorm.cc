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

#include "csrc/mlir/passes/sink_3d_head_split_and_flatten_layernorm.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

bool IsStaticRankedTensor(mlir::Type type,
                          mlir::RankedTensorType* out_type = nullptr) {
  auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!ranked || !ranked.hasStaticShape()) {
    return false;
  }
  if (out_type != nullptr) {
    *out_type = ranked;
  }
  return true;
}

bool IsDotOrDotGeneral(mlir::Operation* op) {
  return op && (llvm::isa<mlir::stablehlo::DotOp>(op) ||
                llvm::isa<mlir::stablehlo::DotGeneralOp>(op));
}

bool IsIdentityPerm(llvm::ArrayRef<int64_t> perm) {
  for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
    if (perm[i] != i) {
      return false;
    }
  }
  return true;
}

llvm::SmallVector<int64_t> GetNonUnitDimSizes(llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t> sizes;
  for (int64_t d : shape) {
    if (d > 1) {
      sizes.push_back(d);
    }
  }
  return sizes;
}

bool ComputeSqueezedPermutation(llvm::ArrayRef<int64_t> in_shape,
                                llvm::ArrayRef<int64_t> perm,
                                llvm::SmallVectorImpl<int64_t>& squeezed_perm) {
  if (in_shape.size() != perm.size()) {
    return false;
  }
  llvm::SmallVector<int64_t> axis_to_squeezed(in_shape.size(), -1);
  int64_t next_idx = 0;
  for (size_t i = 0; i < in_shape.size(); ++i) {
    if (in_shape[i] > 1) {
      axis_to_squeezed[i] = next_idx++;
    }
  }
  squeezed_perm.clear();
  squeezed_perm.reserve(next_idx);
  for (int64_t d : perm) {
    if (d < 0 || d >= static_cast<int64_t>(in_shape.size())) {
      return false;
    }
    if (axis_to_squeezed[d] != -1) {
      squeezed_perm.push_back(axis_to_squeezed[d]);
    }
  }
  return static_cast<int64_t>(squeezed_perm.size()) == next_idx;
}

mlir::Value PeelUnitDimPreservingReshapes(
    mlir::Value val, llvm::ArrayRef<int64_t> non_unit_sizes) {
  while (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    mlir::RankedTensorType in_type;
    if (!IsStaticRankedTensor(resh.getOperand().getType(), &in_type) ||
        GetNonUnitDimSizes(in_type.getShape()) != non_unit_sizes) {
      break;
    }
    val = resh.getOperand();
  }
  return val;
}

mlir::Value EmitReshapeIfNeeded(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value val,
                                mlir::RankedTensorType dst_type) {
  while (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    mlir::RankedTensorType op_type;
    if (IsStaticRankedTensor(resh.getOperand().getType(), &op_type) &&
        op_type.getNumElements() == dst_type.getNumElements() &&
        op_type.getElementType() == dst_type.getElementType()) {
      val = resh.getOperand();
    } else {
      break;
    }
  }
  if (val.getType() == dst_type) {
    return val;
  }
  return mlir::stablehlo::ReshapeOp::create(builder, loc, dst_type, val);
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  constexpr int kMaxIterations = 50;
  int iteration = 0;
  bool changed = true;
  while (changed && iteration++ < kMaxIterations) {
    changed = false;
    llvm::SmallVector<mlir::Operation*, 16> dead_ops;
    func.walk([&](mlir::Operation* op) {
      if (op != func.getOperation() && op->use_empty() &&
          mlir::isOpTriviallyDead(op)) {
        dead_ops.push_back(op);
      }
    });
    for (auto it = dead_ops.rbegin(); it != dead_ops.rend(); ++it) {
      if ((*it)->use_empty() && mlir::isOpTriviallyDead(*it)) {
        (*it)->erase();
        changed = true;
      }
    }
  }
}

bool IsScalarBroadcastOrSplat(mlir::Value val) {
  while (val) {
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      val = bcast.getOperand();
      continue;
    }
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastOp>()) {
      val = bcast.getOperand();
      continue;
    }
    if (auto reshape = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      val = reshape.getOperand();
      continue;
    }
    break;
  }
  if (auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
    auto attr = llvm::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    return attr && !attr.empty() && attr.isSplat();
  }
  mlir::RankedTensorType type;
  return IsStaticRankedTensor(val.getType(), &type) &&
         type.getNumElements() == 1;
}

mlir::Value RebuildScalarBroadcastOrSplat(mlir::OpBuilder& builder,
                                          mlir::Location loc, mlir::Value val,
                                          mlir::RankedTensorType target_type) {
  mlir::Value root = val;
  while (root) {
    if (auto bcast = root.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      root = bcast.getOperand();
      continue;
    }
    if (auto bcast = root.getDefiningOp<mlir::stablehlo::BroadcastOp>()) {
      root = bcast.getOperand();
      continue;
    }
    if (auto reshape = root.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      root = reshape.getOperand();
      continue;
    }
    break;
  }
  if (auto const_op = root.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
    auto attr = llvm::cast<mlir::DenseElementsAttr>(const_op.getValue());
    if (attr.isSplat() &&
        attr.getElementType() == target_type.getElementType()) {
      auto splat_attr = mlir::DenseElementsAttr::get(
          target_type, attr.getSplatValue<mlir::Attribute>());
      return mlir::stablehlo::ConstantOp::create(builder, loc, target_type,
                                                 splat_attr);
    }
  }
  auto src_type = llvm::cast<mlir::RankedTensorType>(root.getType());
  if (src_type.getRank() != 0) {
    auto scalar_type =
        mlir::RankedTensorType::get({}, src_type.getElementType());
    root = mlir::stablehlo::ReshapeOp::create(builder, loc, scalar_type, root);
  }
  if (src_type.getElementType() != target_type.getElementType()) {
    auto target_scalar_type =
        mlir::RankedTensorType::get({}, target_type.getElementType());
    root = mlir::stablehlo::ConvertOp::create(builder, loc, target_scalar_type,
                                              root);
  }
  return mlir::stablehlo::BroadcastInDimOp::create(
      builder, loc, target_type, root, builder.getDenseI64ArrayAttr({}));
}

std::optional<mlir::Value> MatchLastDimFeatureBroadcast(mlir::Value val,
                                                        int64_t tensor_rank,
                                                        int64_t feature_n) {
  if (tensor_rank < 2) {
    return std::nullopt;
  }
  if (auto bcast_op = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    mlir::Value src = bcast_op.getOperand();
    mlir::RankedTensorType src_type;
    if (!IsStaticRankedTensor(src.getType(), &src_type) ||
        src_type.getNumElements() != feature_n) {
      return std::nullopt;
    }
    llvm::ArrayRef<int64_t> dims = bcast_op.getBroadcastDimensions();
    if (static_cast<int64_t>(dims.size()) != src_type.getRank()) {
      return std::nullopt;
    }
    if (src_type.getRank() == 1) {
      if (dims[0] == tensor_rank - 1 && src_type.getDimSize(0) == feature_n) {
        return src;
      }
      return std::nullopt;
    }
    for (int64_t i = 0; i < src_type.getRank() - 1; ++i) {
      if (src_type.getDimSize(i) != 1 || dims[i] < 0 ||
          dims[i] >= tensor_rank - 1 || (i > 0 && dims[i] <= dims[i - 1])) {
        return std::nullopt;
      }
    }
    if (src_type.getDimSize(src_type.getRank() - 1) == feature_n &&
        dims.back() == tensor_rank - 1 &&
        dims.back() > dims[src_type.getRank() - 2]) {
      return src;
    }
  }
  return std::nullopt;
}

enum class PostDotStepKind {
  kBiasAdd,
  kScalarMul,
};

struct PostDotStep {
  PostDotStepKind kind;
  mlir::Value operand;
  bool lhs_was_main = true;
};

struct MatchedPostDotChain {
  mlir::Operation* dot_op = nullptr;
  mlir::RankedTensorType dot_type;
  llvm::SmallVector<PostDotStep, 4> steps;
};

std::optional<MatchedPostDotChain> MatchPostDotChain(
    mlir::Value start_val, int64_t expected_feature_n) {
  llvm::SmallVector<PostDotStep, 4> reverse_steps;
  mlir::Value curr = start_val;

  while (curr) {
    mlir::Operation* def_op = curr.getDefiningOp();
    if (!def_op || !def_op->hasOneUse()) {
      return std::nullopt;
    }
    mlir::RankedTensorType curr_type;
    if (!IsStaticRankedTensor(curr.getType(), &curr_type) ||
        curr_type.getRank() < 2 ||
        curr_type.getShape().back() != expected_feature_n) {
      return std::nullopt;
    }

    if (IsDotOrDotGeneral(def_op)) {
      if (reverse_steps.empty()) {
        return std::nullopt;
      }
      MatchedPostDotChain result;
      result.dot_op = def_op;
      result.dot_type = curr_type;
      result.steps.assign(reverse_steps.rbegin(), reverse_steps.rend());
      return result;
    }

    if (auto reshape_op = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(def_op)) {
      curr = reshape_op.getOperand();
      continue;
    }

    if (auto add_op = llvm::dyn_cast<mlir::stablehlo::AddOp>(def_op)) {
      const int64_t rank = curr_type.getRank();
      if (auto bias = MatchLastDimFeatureBroadcast(add_op.getRhs(), rank,
                                                   expected_feature_n)) {
        reverse_steps.push_back(PostDotStep{PostDotStepKind::kBiasAdd, *bias,
                                            /*lhs_was_main=*/true});
        curr = add_op.getLhs();
        continue;
      }
      if (auto bias = MatchLastDimFeatureBroadcast(add_op.getLhs(), rank,
                                                   expected_feature_n)) {
        reverse_steps.push_back(PostDotStep{PostDotStepKind::kBiasAdd, *bias,
                                            /*lhs_was_main=*/false});
        curr = add_op.getRhs();
        continue;
      }
      return std::nullopt;
    }

    if (auto mul_op = llvm::dyn_cast<mlir::stablehlo::MulOp>(def_op)) {
      if (IsScalarBroadcastOrSplat(mul_op.getRhs())) {
        reverse_steps.push_back(PostDotStep{PostDotStepKind::kScalarMul,
                                            mul_op.getRhs(),
                                            /*lhs_was_main=*/true});
        curr = mul_op.getLhs();
        continue;
      }
      if (IsScalarBroadcastOrSplat(mul_op.getLhs())) {
        reverse_steps.push_back(PostDotStep{PostDotStepKind::kScalarMul,
                                            mul_op.getLhs(),
                                            /*lhs_was_main=*/false});
        curr = mul_op.getRhs();
        continue;
      }
      return std::nullopt;
    }

    return std::nullopt;
  }

  return std::nullopt;
}

bool IsForwardActivation2DDot(mlir::Operation* dot_op, int64_t expected_rows) {
  if (dot_op->getNumOperands() < 2) {
    return false;
  }
  mlir::Value lhs = dot_op->getOperand(0);
  mlir::Value rhs = dot_op->getOperand(1);
  mlir::RankedTensorType lhs_type, rhs_type;
  if (!IsStaticRankedTensor(lhs.getType(), &lhs_type) ||
      !IsStaticRankedTensor(rhs.getType(), &rhs_type) ||
      lhs_type.getRank() != 2 || rhs_type.getRank() != 2 ||
      lhs_type.getDimSize(0) != expected_rows) {
    return false;
  }
  if (auto dg = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(dot_op)) {
    auto dnums = dg.getDotDimensionNumbers();
    if (!dnums.getLhsBatchingDimensions().empty() ||
        !dnums.getRhsBatchingDimensions().empty() ||
        dnums.getLhsContractingDimensions() != llvm::ArrayRef<int64_t>{1} ||
        dnums.getRhsContractingDimensions().size() != 1) {
      return false;
    }
  }
  for (mlir::Operation* user : lhs.getUsers()) {
    if (auto dg = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(user)) {
      auto dnums = dg.getDotDimensionNumbers();
      if (dg.getLhs() == lhs) {
        for (int64_t c : dnums.getLhsContractingDimensions()) {
          if (c == 0) return false;
        }
      }
      if (dg.getRhs() == lhs) {
        for (int64_t c : dnums.getRhsContractingDimensions()) {
          if (c == 0) return false;
        }
      }
    } else if (auto d = llvm::dyn_cast<mlir::stablehlo::DotOp>(user)) {
      if (d.getRhs() == lhs) return false;
    }
  }
  return true;
}

mlir::Value ReplayPostDotStepsHeadSplit(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value acc,
    llvm::ArrayRef<PostDotStep> steps, int64_t num_heads, int64_t head_dim,
    llvm::ArrayRef<int64_t> bias_bcast_dims) {
  auto acc_type = llvm::cast<mlir::RankedTensorType>(acc.getType());
  mlir::Type elem_type = acc_type.getElementType();
  auto bias_2d_type =
      mlir::RankedTensorType::get({num_heads, head_dim}, elem_type);

  for (const PostDotStep& step : steps) {
    if (step.kind == PostDotStepKind::kBiasAdd) {
      mlir::Value bias_2d = mlir::stablehlo::ReshapeOp::create(
          builder, loc, bias_2d_type, step.operand);
      mlir::Value bias_bcast = mlir::stablehlo::BroadcastInDimOp::create(
          builder, loc, acc_type, bias_2d,
          builder.getDenseI64ArrayAttr(bias_bcast_dims));
      acc = step.lhs_was_main
                ? mlir::stablehlo::AddOp::create(builder, loc, acc, bias_bcast)
                : mlir::stablehlo::AddOp::create(builder, loc, bias_bcast, acc);
    } else {
      mlir::Value scale_bcast =
          RebuildScalarBroadcastOrSplat(builder, loc, step.operand, acc_type);
      acc =
          step.lhs_was_main
              ? mlir::stablehlo::MulOp::create(builder, loc, acc, scale_bcast)
              : mlir::stablehlo::MulOp::create(builder, loc, scale_bcast, acc);
    }
  }
  return acc;
}

// Sinks 3D head-split `Reshape([S, H*D] -> [S, H, D]) -> Transpose({1, 0, 2} |
// {1, 2, 0})` and multi-use unit-batch 4D head-split `Reshape([S, H*D] -> [S,
// H, D, 1]) -> Transpose({3, 1, 0, 2})` past 1D feature bias `AddOp` and scalar
// `MulOp` directly onto 2D `DotOp` / `DotGeneralOp`.
void TrySinkHeadSplitReshapeTranspose(mlir::stablehlo::ReshapeOp reshape_op) {
  if (!reshape_op || !reshape_op->getBlock() || !reshape_op->hasOneUse()) {
    return;
  }
  auto trans_user = llvm::dyn_cast<mlir::stablehlo::TransposeOp>(
      *reshape_op->getUsers().begin());
  if (!trans_user || trans_user->use_empty()) {
    return;
  }
  mlir::RankedTensorType out_type;
  if (!IsStaticRankedTensor(reshape_op.getType(), &out_type)) {
    return;
  }
  llvm::ArrayRef<int64_t> perm = trans_user.getPermutation();

  // Case 1: 3D head-split `Reshape([S, H*D] -> [S, H, D]) -> Transpose({1, 0,
  // 2} | {1, 2, 0})`.
  if (out_type.getRank() == 3) {
    const bool is_102 = (perm == llvm::ArrayRef<int64_t>{1, 0, 2});
    const bool is_120 = (perm == llvm::ArrayRef<int64_t>{1, 2, 0});
    if (!is_102 && !is_120) {
      return;
    }
    const int64_t s = out_type.getDimSize(0);
    const int64_t h = out_type.getDimSize(1);
    const int64_t d = out_type.getDimSize(2);
    if (s <= 1 || h <= 1 || d <= 1) {
      return;
    }
    const int64_t feature_n = h * d;
    auto matched = MatchPostDotChain(reshape_op.getOperand(), feature_n);
    if (!matched || !IsForwardActivation2DDot(matched->dot_op, s)) {
      return;
    }

    mlir::OpBuilder builder(trans_user);
    mlir::Location loc = trans_user.getLoc();
    mlir::Value dot_3d = mlir::stablehlo::ReshapeOp::create(
        builder, loc, out_type, matched->dot_op->getResult(0));
    auto trans_type = llvm::cast<mlir::RankedTensorType>(trans_user.getType());
    mlir::Value trans_dot = mlir::stablehlo::TransposeOp::create(
        builder, loc, trans_type, dot_3d, trans_user.getPermutationAttr());
    llvm::SmallVector<int64_t, 2> bias_dims =
        is_102 ? llvm::SmallVector<int64_t, 2>{0, 2}
               : llvm::SmallVector<int64_t, 2>{0, 1};
    mlir::Value replaced = ReplayPostDotStepsHeadSplit(
        builder, loc, trans_dot, matched->steps, h, d, bias_dims);
    trans_user.replaceAllUsesWith(replaced);
    return;
  }

  // Case 2: Unit-batch 4D head-split `Reshape([S, H*D] -> [S, H, D, 1]) ->
  // Transpose({3, 1, 0, 2}) -> [1, H, S, D]` (including when `trans_user` has
  // multiple users such as relative positional bias branches).
  // GENERIC: semantic 4D tensor [S, H, D, 1] with trailing unit batch axis 3.
  if (out_type.getRank() == 4 && out_type.getDimSize(3) == 1 &&
      perm == llvm::ArrayRef<int64_t>{3, 1, 0, 2}) {
    const int64_t s = out_type.getDimSize(0);
    const int64_t h = out_type.getDimSize(1);
    const int64_t d = out_type.getDimSize(2);
    if (s <= 1 || h <= 1 || d <= 1) {
      return;
    }
    const int64_t feature_n = h * d;
    auto matched = MatchPostDotChain(reshape_op.getOperand(), feature_n);
    if (!matched || !IsForwardActivation2DDot(matched->dot_op, s)) {
      return;
    }

    mlir::OpBuilder builder(trans_user);
    mlir::Location loc = trans_user.getLoc();
    auto canonical_4d_type =
        mlir::RankedTensorType::get({1, s, h, d}, out_type.getElementType());
    mlir::Value dot_4d = mlir::stablehlo::ReshapeOp::create(
        builder, loc, canonical_4d_type, matched->dot_op->getResult(0));
    auto trans_type = llvm::cast<mlir::RankedTensorType>(trans_user.getType());
    mlir::Value trans_dot = mlir::stablehlo::TransposeOp::create(
        builder, loc, trans_type, dot_4d,
        builder.getDenseI64ArrayAttr({0, 2, 1, 3}));
    mlir::Value replaced = ReplayPostDotStepsHeadSplit(
        builder, loc, trans_dot, matched->steps, h, d,
        /*bias_bcast_dims=*/{1, 3});
    trans_user.replaceAllUsesWith(replaced);
  }
}

void CancelInverseViewTransposes(mlir::func::FuncOp func) {
  constexpr int kMaxIterations = 50;
  int iteration = 0;
  bool changed = true;
  while (changed && iteration++ < kMaxIterations) {
    changed = false;
    llvm::SmallVector<mlir::stablehlo::TransposeOp, 16> tr_ops;
    func.walk([&](mlir::stablehlo::TransposeOp op) { tr_ops.push_back(op); });

    for (mlir::stablehlo::TransposeOp tr2 : tr_ops) {
      if (!tr2 || !tr2->getBlock() || tr2->use_empty()) {
        continue;
      }
      mlir::RankedTensorType tr2_type, tr2_in_type;
      if (!IsStaticRankedTensor(tr2.getType(), &tr2_type) ||
          !IsStaticRankedTensor(tr2.getOperand().getType(), &tr2_in_type)) {
        continue;
      }

      // Rule 1: Fold `TransposeOp(BroadcastInDimOp(x, bdims), perm)` when
      // mapped broadcast dimensions remain strictly ascending.
      if (auto bcast_op =
              tr2.getOperand()
                  .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
        if (bcast_op->hasOneUse()) {
          llvm::ArrayRef<int64_t> perm = tr2.getPermutation();
          const int64_t rank = perm.size();
          llvm::SmallVector<int64_t> inv_perm(rank, -1);
          bool valid_perm = true;
          for (int64_t i = 0; i < rank; ++i) {
            if (perm[i] < 0 || perm[i] >= rank) {
              valid_perm = false;
              break;
            }
            inv_perm[perm[i]] = i;
          }
          if (valid_perm) {
            llvm::ArrayRef<int64_t> bdims = bcast_op.getBroadcastDimensions();
            llvm::SmallVector<int64_t> new_bdims;
            new_bdims.reserve(bdims.size());
            bool ascending = true;
            for (size_t k = 0; k < bdims.size(); ++k) {
              if (bdims[k] < 0 || bdims[k] >= rank) {
                ascending = false;
                break;
              }
              int64_t mapped = inv_perm[bdims[k]];
              if (k > 0 && mapped <= new_bdims.back()) {
                ascending = false;
                break;
              }
              new_bdims.push_back(mapped);
            }
            if (ascending) {
              mlir::OpBuilder builder(tr2);
              auto new_bcast = mlir::stablehlo::BroadcastInDimOp::create(
                  builder, tr2.getLoc(), tr2_type, bcast_op.getOperand(),
                  builder.getDenseI64ArrayAttr(new_bdims));
              tr2.replaceAllUsesWith(new_bcast.getResult());
              changed = true;
              continue;
            }
          }
        }
      }

      // Rule 2: Cancel `TransposeOp([unit-dim ReshapeOp]*(TransposeOp(x)))`
      // when the composed permutation on non-unit dimensions is the identity.
      llvm::SmallVector<int64_t> non_unit_before_tr2 =
          GetNonUnitDimSizes(tr2_in_type.getShape());
      llvm::SmallVector<int64_t> squeezed_perm2;
      if (!ComputeSqueezedPermutation(tr2_in_type.getShape(),
                                      tr2.getPermutation(), squeezed_perm2) ||
          IsIdentityPerm(squeezed_perm2)) {
        continue;
      }

      mlir::Value peeled =
          PeelUnitDimPreservingReshapes(tr2.getOperand(), non_unit_before_tr2);
      auto tr1 = peeled.getDefiningOp<mlir::stablehlo::TransposeOp>();
      if (!tr1) {
        continue;
      }
      mlir::RankedTensorType tr1_in_type;
      llvm::SmallVector<int64_t> squeezed_perm1;
      if (!IsStaticRankedTensor(tr1.getOperand().getType(), &tr1_in_type) ||
          !ComputeSqueezedPermutation(tr1_in_type.getShape(),
                                      tr1.getPermutation(), squeezed_perm1) ||
          squeezed_perm1.size() != squeezed_perm2.size() ||
          IsIdentityPerm(squeezed_perm1)) {
        continue;
      }
      llvm::SmallVector<int64_t> composed_perm(squeezed_perm2.size());
      for (size_t i = 0; i < squeezed_perm2.size(); ++i) {
        composed_perm[i] = squeezed_perm1[squeezed_perm2[i]];
      }
      if (IsIdentityPerm(composed_perm)) {
        llvm::SmallVector<int64_t> orig_non_unit =
            GetNonUnitDimSizes(tr1_in_type.getShape());
        mlir::Value root_val =
            PeelUnitDimPreservingReshapes(tr1.getOperand(), orig_non_unit);
        mlir::OpBuilder builder(tr2);
        mlir::Value rep =
            EmitReshapeIfNeeded(builder, tr2.getLoc(), root_val, tr2_type);
        tr2.replaceAllUsesWith(rep);
        changed = true;
      }
    }
    if (changed) {
      EraseTriviallyDeadOps(func);
    }
  }
}

bool IsLeadingAxesBroadcast(mlir::stablehlo::BroadcastInDimOp bcast_op,
                            int64_t rank) {
  llvm::ArrayRef<int64_t> dims = bcast_op.getBroadcastDimensions();
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

bool IsTrailingAxis1DBroadcast(mlir::Value val, int64_t rank, int64_t channels,
                               mlir::Value* out_1d_param) {
  auto bcast_op = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast_op) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = bcast_op.getBroadcastDimensions();
  if (dims.size() != 1 || dims[0] != rank - 1) {
    return false;
  }
  mlir::RankedTensorType param_type;
  if (!IsStaticRankedTensor(bcast_op.getOperand().getType(), &param_type) ||
      param_type.getRank() != 1 || param_type.getDimSize(0) != channels) {
    return false;
  }
  *out_1d_param = bcast_op.getOperand();
  return true;
}

bool IsMomentsLeafConstant(mlir::Value val, llvm::ArrayRef<int64_t> stats_shape,
                           mlir::Type expected_elem_type) {
  mlir::RankedTensorType type;
  if (!IsStaticRankedTensor(val.getType(), &type) ||
      type.getShape() != stats_shape ||
      type.getElementType() != expected_elem_type) {
    return false;
  }
  if (auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
    auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    return dense && dense.isSplat() &&
           dense.getElementType() == expected_elem_type;
  }
  if (auto bcast_op = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    mlir::RankedTensorType op_type;
    return bcast_op.getBroadcastDimensions().empty() &&
           IsStaticRankedTensor(bcast_op.getOperand().getType(), &op_type) &&
           op_type.getRank() == 0 &&
           op_type.getElementType() == expected_elem_type;
  }
  return false;
}

bool CollectMomentsSlice(mlir::Value val, mlir::Value sum_0, mlir::Value sum_1,
                         llvm::ArrayRef<int64_t> stats_shape,
                         mlir::Type stats_elem_type,
                         llvm::DenseSet<mlir::Operation*>& visited,
                         llvm::SmallVectorImpl<mlir::Operation*>& order) {
  if (val == sum_0 || val == sum_1 ||
      IsMomentsLeafConstant(val, stats_shape, stats_elem_type)) {
    return true;
  }
  mlir::RankedTensorType type;
  if (!IsStaticRankedTensor(val.getType(), &type) ||
      type.getShape() != stats_shape ||
      type.getElementType() != stats_elem_type) {
    return false;
  }
  mlir::Operation* op = val.getDefiningOp();
  if (op == nullptr || op->getNumResults() != 1) {
    return false;
  }
  if (visited.contains(op)) {
    return true;
  }
  if (!llvm::isa<mlir::stablehlo::MulOp, mlir::stablehlo::DivOp,
                 mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
                 mlir::stablehlo::MaxOp, mlir::stablehlo::MinOp,
                 mlir::stablehlo::RsqrtOp, mlir::stablehlo::SqrtOp,
                 mlir::stablehlo::ConvertOp, mlir::stablehlo::AbsOp,
                 mlir::stablehlo::NegOp>(op)) {
    return false;
  }
  for (mlir::Value operand : op->getOperands()) {
    if (!CollectMomentsSlice(operand, sum_0, sum_1, stats_shape,
                             stats_elem_type, visited, order)) {
      return false;
    }
  }
  visited.insert(op);
  order.push_back(op);
  return true;
}

mlir::Value RebuildMomentsLeaf1D(mlir::OpBuilder& builder, mlir::Location loc,
                                 mlir::Value val,
                                 mlir::RankedTensorType stats_1d_type) {
  if (auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
    auto dense = llvm::cast<mlir::DenseElementsAttr>(const_op.getValue());
    if (dense.getElementType() == stats_1d_type.getElementType()) {
      auto splat_val = dense.getSplatValue<mlir::Attribute>();
      auto new_attr = mlir::DenseElementsAttr::get(stats_1d_type, splat_val);
      return mlir::stablehlo::ConstantOp::create(builder, loc, stats_1d_type,
                                                 new_attr);
    }
  }
  auto bcast_op = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  mlir::Value src = bcast_op.getOperand();
  auto src_type = llvm::cast<mlir::RankedTensorType>(src.getType());
  if (src_type.getElementType() != stats_1d_type.getElementType()) {
    auto scalar_type =
        mlir::RankedTensorType::get({}, stats_1d_type.getElementType());
    src = mlir::stablehlo::ConvertOp::create(builder, loc, scalar_type, src);
  }
  return mlir::stablehlo::BroadcastInDimOp::create(
      builder, loc, stats_1d_type, src, builder.getDenseI64ArrayAttr({}));
}

// Flattens any LayerNorm subgraph on a static tensor `[d_0, ..., d_{R-2}, C]`
// (`R >= 3`) containing at least one unit unreduced axis (`d_i == 1`, such as
// `[1, S, C]` or `[S, 1, C]`) to 2D `[N, C]` with 1D `[N]` moments, regardless
// of whether `ln_input` or `ln_out` has multiple users (e.g. residual streams).
bool TryFlattenUnitUnreducedAxisLayerNorm(mlir::stablehlo::MulOp norm_mul) {
  if (!norm_mul || !norm_mul->getBlock() || norm_mul->use_empty()) {
    return false;
  }
  mlir::Block* block = norm_mul->getBlock();
  mlir::RankedTensorType stats_tensor_type;
  if (!IsStaticRankedTensor(norm_mul.getType(), &stats_tensor_type)) {
    return false;
  }
  const int64_t rank = stats_tensor_type.getRank();
  // GENERIC: semantic rank >= 3 tensor with at least one unit unreduced axis.
  if (rank < 3) {
    return false;
  }
  const int64_t channels = stats_tensor_type.getDimSize(rank - 1);
  if (channels <= 1) {
    return false;
  }
  int64_t num_rows = 1;
  bool has_unit_unreduced_dim = false;
  for (int64_t i = 0; i < rank - 1; ++i) {
    const int64_t d = stats_tensor_type.getDimSize(i);
    if (d <= 0) {
      return false;
    }
    if (d == 1) {
      has_unit_unreduced_dim = true;
    }
    num_rows *= d;
  }
  if (!has_unit_unreduced_dim || num_rows <= 1) {
    return false;
  }

  mlir::stablehlo::SubtractOp sub_op;
  mlir::stablehlo::BroadcastInDimOp rstd_bcast;
  if (auto s = norm_mul.getLhs().getDefiningOp<mlir::stablehlo::SubtractOp>()) {
    sub_op = s;
    rstd_bcast =
        norm_mul.getRhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  } else if (auto s = norm_mul.getRhs()
                          .getDefiningOp<mlir::stablehlo::SubtractOp>()) {
    sub_op = s;
    rstd_bcast =
        norm_mul.getLhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  }
  if (!sub_op || sub_op->getBlock() != block || !sub_op->hasOneUse() ||
      !rstd_bcast || rstd_bcast->getBlock() != block ||
      !rstd_bcast->hasOneUse() || !IsLeadingAxesBroadcast(rstd_bcast, rank)) {
    return false;
  }
  auto mean_bcast =
      sub_op.getRhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!mean_bcast || mean_bcast->getBlock() != block ||
      !mean_bcast->hasOneUse() || !IsLeadingAxesBroadcast(mean_bcast, rank)) {
    return false;
  }

  mlir::Value input_casted = sub_op.getLhs();
  mlir::stablehlo::ReduceOp reduce_op;
  mlir::stablehlo::MulOp x_sq_op;
  for (mlir::Operation* user : input_casted.getUsers()) {
    if (auto r = llvm::dyn_cast<mlir::stablehlo::ReduceOp>(user)) {
      if (r.getInputs().size() == 2 && r.getNumResults() == 2 &&
          r.getDimensions() == llvm::ArrayRef<int64_t>{rank - 1}) {
        reduce_op = r;
      }
    } else if (auto m = llvm::dyn_cast<mlir::stablehlo::MulOp>(user)) {
      if (m.getLhs() == input_casted && m.getRhs() == input_casted) {
        x_sq_op = m;
      }
    }
  }
  if (!reduce_op || reduce_op->getBlock() != block || !x_sq_op ||
      x_sq_op->getBlock() != block || !x_sq_op->hasOneUse()) {
    return false;
  }
  if (!((reduce_op.getInputs()[0] == input_casted &&
         reduce_op.getInputs()[1] == x_sq_op.getResult()) ||
        (reduce_op.getInputs()[1] == input_casted &&
         reduce_op.getInputs()[0] == x_sq_op.getResult()))) {
    return false;
  }

  mlir::Value ln_input = input_casted;
  if (auto up_conv = input_casted.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    mlir::RankedTensorType raw_type;
    if (up_conv->getBlock() == block && up_conv->hasOneUse() &&
        IsStaticRankedTensor(up_conv.getOperand().getType(), &raw_type) &&
        raw_type.getShape() == stats_tensor_type.getShape()) {
      ln_input = up_conv.getOperand();
    }
  }

  llvm::ArrayRef<int64_t> stats_shape =
      stats_tensor_type.getShape().drop_back(1);
  mlir::Type stats_elem_type = stats_tensor_type.getElementType();
  mlir::Value sum_0 = reduce_op.getResult(0);
  mlir::Value sum_1 = reduce_op.getResult(1);
  mlir::Value mean_val = mean_bcast.getOperand();
  mlir::Value rstd_val = rstd_bcast.getOperand();

  llvm::DenseSet<mlir::Operation*> moments_set;
  llvm::SmallVector<mlir::Operation*, 8> moments_order;
  if (!CollectMomentsSlice(mean_val, sum_0, sum_1, stats_shape, stats_elem_type,
                           moments_set, moments_order) ||
      !CollectMomentsSlice(rstd_val, sum_0, sum_1, stats_shape, stats_elem_type,
                           moments_set, moments_order)) {
    return false;
  }
  for (mlir::Operation* op : moments_order) {
    if (op->getBlock() != block) {
      return false;
    }
  }
  auto all_uses_internal = [&](mlir::Value v) {
    for (mlir::Operation* u : v.getUsers()) {
      if (!moments_set.contains(u) && u != mean_bcast.getOperation() &&
          u != rstd_bcast.getOperation()) {
        return false;
      }
    }
    return true;
  };
  if (!all_uses_internal(sum_0) || !all_uses_internal(sum_1)) {
    return false;
  }
  for (mlir::Operation* op : moments_order) {
    if (!all_uses_internal(op->getResult(0))) {
      return false;
    }
  }

  // Walk forward from `norm_mul` through optional downcast `ConvertOp`,
  // optional 1D `weight` `MulOp`, and optional 1D `bias` `AddOp`.
  mlir::Value ln_out = norm_mul.getResult();
  bool has_downcast = false;
  mlir::Type out_elem_type = stats_tensor_type.getElementType();
  if (ln_out.hasOneUse()) {
    if (auto cvt = llvm::dyn_cast<mlir::stablehlo::ConvertOp>(
            *ln_out.getUsers().begin())) {
      if (cvt->getBlock() == block) {
        has_downcast = true;
        ln_out = cvt.getResult();
        out_elem_type = llvm::cast<mlir::RankedTensorType>(ln_out.getType())
                            .getElementType();
      }
    }
  }

  mlir::Value weight_1d;
  if (ln_out.hasOneUse()) {
    if (auto mul_op = llvm::dyn_cast<mlir::stablehlo::MulOp>(
            *ln_out.getUsers().begin())) {
      if (mul_op->getBlock() == block) {
        mlir::Value other =
            (mul_op.getLhs() == ln_out) ? mul_op.getRhs() : mul_op.getLhs();
        if (IsTrailingAxis1DBroadcast(other, rank, channels, &weight_1d)) {
          ln_out = mul_op.getResult();
        }
      }
    }
  }

  mlir::Value bias_1d;
  if (ln_out.hasOneUse()) {
    if (auto add_op = llvm::dyn_cast<mlir::stablehlo::AddOp>(
            *ln_out.getUsers().begin())) {
      if (add_op->getBlock() == block) {
        mlir::Value other =
            (add_op.getLhs() == ln_out) ? add_op.getRhs() : add_op.getLhs();
        if (IsTrailingAxis1DBroadcast(other, rank, channels, &bias_1d)) {
          ln_out = add_op.getResult();
        }
      }
    }
  }

  mlir::OpBuilder builder(ln_out.getDefiningOp());
  mlir::Location loc = ln_out.getLoc();
  mlir::Type in_elem_type =
      llvm::cast<mlir::RankedTensorType>(ln_input.getType()).getElementType();
  auto in_2d_type =
      mlir::RankedTensorType::get({num_rows, channels}, in_elem_type);
  auto stats_2d_type =
      mlir::RankedTensorType::get({num_rows, channels}, stats_elem_type);

  mlir::Value in_2d = EmitReshapeIfNeeded(builder, loc, ln_input, in_2d_type);
  mlir::Value input_casted_2d = in_2d;
  if (stats_elem_type != in_elem_type) {
    input_casted_2d =
        mlir::stablehlo::ConvertOp::create(builder, loc, stats_2d_type, in_2d);
  }
  mlir::Value x_sq_2d = mlir::stablehlo::MulOp::create(
      builder, loc, input_casted_2d, input_casted_2d);

  auto stats_1d_type = mlir::RankedTensorType::get({num_rows}, stats_elem_type);
  llvm::SmallVector<mlir::Value, 2> red_inputs;
  if (reduce_op.getInputs()[0] == input_casted) {
    red_inputs = {input_casted_2d, x_sq_2d};
  } else {
    red_inputs = {x_sq_2d, input_casted_2d};
  }
  llvm::SmallVector<mlir::Type, 2> red_result_types = {stats_1d_type,
                                                       stats_1d_type};
  auto new_reduce = mlir::stablehlo::ReduceOp::create(
      builder, loc, red_result_types, red_inputs, reduce_op.getInitValues(),
      builder.getDenseI64ArrayAttr({1}));
  mlir::IRMapping body_map;
  reduce_op.getBody().cloneInto(&new_reduce.getBody(), body_map);

  mlir::IRMapping moments_map;
  moments_map.map(sum_0, new_reduce.getResult(0));
  moments_map.map(sum_1, new_reduce.getResult(1));
  for (mlir::Operation* op : moments_order) {
    for (mlir::Value operand : op->getOperands()) {
      if (!moments_map.contains(operand)) {
        mlir::Value leaf_1d =
            RebuildMomentsLeaf1D(builder, loc, operand, stats_1d_type);
        moments_map.map(operand, leaf_1d);
      }
    }
    mlir::Operation* cloned = builder.clone(*op, moments_map);
    auto old_res_type =
        llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto new_res_type =
        mlir::RankedTensorType::get({num_rows}, old_res_type.getElementType());
    cloned->getResult(0).setType(new_res_type);
    moments_map.map(op->getResult(0), cloned->getResult(0));
  }

  mlir::Value mean_1d = moments_map.lookup(mean_val);
  mlir::Value rstd_1d = moments_map.lookup(rstd_val);
  mlir::Value mean_bcast_2d = mlir::stablehlo::BroadcastInDimOp::create(
      builder, loc, stats_2d_type, mean_1d, builder.getDenseI64ArrayAttr({0}));
  mlir::Value sub_2d = mlir::stablehlo::SubtractOp::create(
      builder, loc, input_casted_2d, mean_bcast_2d);
  mlir::Value rstd_bcast_2d = mlir::stablehlo::BroadcastInDimOp::create(
      builder, loc, stats_2d_type, rstd_1d, builder.getDenseI64ArrayAttr({0}));
  mlir::Value norm_2d =
      mlir::stablehlo::MulOp::create(builder, loc, sub_2d, rstd_bcast_2d);

  auto out_2d_type =
      mlir::RankedTensorType::get({num_rows, channels}, out_elem_type);
  mlir::Value cur_out_2d = norm_2d;
  if (has_downcast && out_elem_type != stats_elem_type) {
    cur_out_2d = mlir::stablehlo::ConvertOp::create(builder, loc, out_2d_type,
                                                    cur_out_2d);
  }
  if (weight_1d) {
    mlir::Value w_bcast_2d = mlir::stablehlo::BroadcastInDimOp::create(
        builder, loc, out_2d_type, weight_1d,
        builder.getDenseI64ArrayAttr({1}));
    cur_out_2d =
        mlir::stablehlo::MulOp::create(builder, loc, cur_out_2d, w_bcast_2d);
  }
  if (bias_1d) {
    mlir::Value b_bcast_2d = mlir::stablehlo::BroadcastInDimOp::create(
        builder, loc, out_2d_type, bias_1d, builder.getDenseI64ArrayAttr({1}));
    cur_out_2d =
        mlir::stablehlo::AddOp::create(builder, loc, cur_out_2d, b_bcast_2d);
  }

  auto out_nd_type = llvm::cast<mlir::RankedTensorType>(ln_out.getType());
  mlir::Value out_nd =
      mlir::stablehlo::ReshapeOp::create(builder, loc, out_nd_type, cur_out_2d);
  ln_out.replaceAllUsesWith(out_nd);
  return true;
}

bool IsPushableElementwiseOp(mlir::Operation* op) {
  if (!op || op->getNumResults() != 1) {
    return false;
  }
  return llvm::isa<
      mlir::stablehlo::ConvertOp, mlir::stablehlo::NegOp,
      mlir::stablehlo::ExpOp, mlir::stablehlo::Expm1Op, mlir::stablehlo::LogOp,
      mlir::stablehlo::Log1pOp, mlir::chlo::ErfOp, mlir::stablehlo::TanhOp,
      mlir::stablehlo::SqrtOp, mlir::stablehlo::RsqrtOp, mlir::stablehlo::AbsOp,
      mlir::stablehlo::ClampOp, mlir::stablehlo::FloorOp,
      mlir::stablehlo::CeilOp, mlir::stablehlo::RoundOp,
      mlir::stablehlo::SineOp, mlir::stablehlo::CosineOp,
      mlir::stablehlo::LogisticOp, mlir::stablehlo::AddOp,
      mlir::stablehlo::SubtractOp, mlir::stablehlo::MulOp,
      mlir::stablehlo::DivOp, mlir::stablehlo::MaxOp, mlir::stablehlo::MinOp>(
      op);
}

// Pushes 2D-to-ND unit-axis `ReshapeOp` (`[N, C] -> [d_0, ..., d_{R-2}, C]`
// with at least one `d_i == 1`) forward through pure elementwise ops so
// activations (`ReLU`, `SiLU`, `GELU`) and residual streams (`* 0.5 + res`)
// stay in 2D `[N, C]`.
void PushUnitDim2DToNDReshapeThroughElementwise(mlir::func::FuncOp func) {
  constexpr int kMaxIterations = 50;
  int iteration = 0;
  bool changed = true;
  while (changed && iteration++ < kMaxIterations) {
    changed = false;
    // First fold any `ReshapeOp(ReshapeOp(x))` pairs.
    func.walk([&](mlir::stablehlo::ReshapeOp r2) {
      if (r2->use_empty()) {
        return;
      }
      if (auto r1 =
              r2.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
        if (r1.getOperand().getType() == r2.getType()) {
          r2.replaceAllUsesWith(r1.getOperand());
          changed = true;
        } else if (r1->hasOneUse() &&
                   r1.getOperand().getType() != r1.getType()) {
          r2.setOperand(r1.getOperand());
          changed = true;
        }
      }
    });
    if (changed) {
      EraseTriviallyDeadOps(func);
    }

    llvm::SmallVector<mlir::Operation*, 16> elem_ops;
    func.walk([&](mlir::Operation* op) {
      if (IsPushableElementwiseOp(op)) {
        elem_ops.push_back(op);
      }
    });

    for (mlir::Operation* op : elem_ops) {
      if (!op || !op->getBlock() || op->use_empty()) {
        continue;
      }
      mlir::RankedTensorType res_type;
      if (!IsStaticRankedTensor(op->getResult(0).getType(), &res_type)) {
        continue;
      }
      const int64_t rank = res_type.getRank();
      if (rank < 3) {
        continue;
      }
      const int64_t c = res_type.getDimSize(rank - 1);
      if (c <= 1) {
        continue;
      }
      int64_t n = 1;
      bool has_unit_leading = false;
      for (int64_t i = 0; i < rank - 1; ++i) {
        const int64_t d = res_type.getDimSize(i);
        if (d <= 0) {
          n = 0;
          break;
        }
        if (d == 1) {
          has_unit_leading = true;
        }
        n *= d;
      }
      if (!has_unit_leading || n <= 1) {
        continue;
      }

      bool has_2d_reshape_input = false;
      bool all_inputs_compatible = true;
      for (mlir::Value opnd : op->getOperands()) {
        mlir::RankedTensorType opnd_type;
        if (!IsStaticRankedTensor(opnd.getType(), &opnd_type) ||
            opnd_type.getShape() != res_type.getShape()) {
          all_inputs_compatible = false;
          break;
        }
        if (auto r = opnd.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
          mlir::RankedTensorType r_in_type;
          if (IsStaticRankedTensor(r.getOperand().getType(), &r_in_type) &&
              r_in_type.getShape() == llvm::ArrayRef<int64_t>{n, c}) {
            has_2d_reshape_input = true;
            continue;
          }
        }
        if (IsScalarBroadcastOrSplat(opnd)) {
          continue;
        }
        all_inputs_compatible = false;
        break;
      }
      if (!has_2d_reshape_input || !all_inputs_compatible) {
        continue;
      }

      mlir::OpBuilder builder(op);
      mlir::Location loc = op->getLoc();
      mlir::IRMapping mapping;
      for (mlir::Value opnd : op->getOperands()) {
        auto opnd_type = llvm::cast<mlir::RankedTensorType>(opnd.getType());
        auto opnd_2d_type =
            mlir::RankedTensorType::get({n, c}, opnd_type.getElementType());
        if (auto r = opnd.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
          mlir::RankedTensorType r_in_type;
          if (IsStaticRankedTensor(r.getOperand().getType(), &r_in_type) &&
              r_in_type.getShape() == llvm::ArrayRef<int64_t>{n, c}) {
            mapping.map(opnd, r.getOperand());
            continue;
          }
        }
        mlir::Value s_2d =
            RebuildScalarBroadcastOrSplat(builder, loc, opnd, opnd_2d_type);
        mapping.map(opnd, s_2d);
      }

      mlir::Operation* cloned = builder.clone(*op, mapping);
      auto res_2d_type =
          mlir::RankedTensorType::get({n, c}, res_type.getElementType());
      cloned->getResult(0).setType(res_2d_type);
      mlir::Value wrapped = mlir::stablehlo::ReshapeOp::create(
          builder, loc, res_type, cloned->getResult(0));
      op->getResult(0).replaceAllUsesWith(wrapped);
      op->erase();
      changed = true;
    }
    if (changed) {
      EraseTriviallyDeadOps(func);
    }
  }
}

class Sink3dHeadSplitAndFlattenLayerNormPass
    : public mlir::PassWrapper<Sink3dHeadSplitAndFlattenLayerNormPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      Sink3dHeadSplitAndFlattenLayerNormPass)

  llvm::StringRef getArgument() const final {
    return "sink-3d-head-split-and-flatten-layernorm";
  }

  llvm::StringRef getDescription() const final {
    return "Sinks 3D and multi-use unit-batch 4D head-split projection "
           "transposes past bias and scale onto dot contractions and flattens "
           "unit-unreduced-axis LayerNorm and elementwise subgraphs to 2D.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    CancelInverseViewTransposes(func);

    // Step 1: Flatten unit-unreduced-axis LayerNorm subgraphs (`[1, S, C]` or
    // `[S, 1, C]`) to 2D `[S, C]` with 1D `[S]` moments.
    llvm::SmallVector<mlir::stablehlo::MulOp, 16> mul_ops;
    func.walk([&](mlir::stablehlo::MulOp op) { mul_ops.push_back(op); });
    for (mlir::stablehlo::MulOp op : mul_ops) {
      if (op && op->getBlock() && !op->use_empty()) {
        TryFlattenUnitUnreducedAxisLayerNorm(op);
      }
    }
    EraseTriviallyDeadOps(func);

    // Step 2: Push 2D-to-ND unit-axis reshapes forward through elementwise
    // activations and residual streams.
    PushUnitDim2DToNDReshapeThroughElementwise(func);

    // Step 3: Sink 3D head-split and multi-use unit-batch 4D head-split
    // reshapes + transposes past bias AddOp and scalar MulOp onto
    // Dot/DotGeneral.
    llvm::SmallVector<mlir::stablehlo::ReshapeOp, 16> reshapes;
    func.walk([&](mlir::stablehlo::ReshapeOp op) { reshapes.push_back(op); });
    for (mlir::stablehlo::ReshapeOp op : reshapes) {
      if (op && op->getBlock() && !op->use_empty()) {
        TrySinkHeadSplitReshapeTranspose(op);
      }
    }
    EraseTriviallyDeadOps(func);

    CancelInverseViewTransposes(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createSink3dHeadSplitAndFlattenLayerNormPass() {
  return std::make_unique<Sink3dHeadSplitAndFlattenLayerNormPass>();
}

}  // namespace torch_tpu
