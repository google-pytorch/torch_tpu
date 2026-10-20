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

#include "csrc/mlir/passes/commute_pointwise_dags_across_channel_transposes.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "llvm/ADT/APFloat.h"
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

bool IsScalarSplatOrBroadcast(mlir::Value val) {
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
    auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    return dense && !dense.empty() && dense.isSplat();
  }
  mlir::RankedTensorType type;
  return IsStaticRankedTensor(val.getType(), &type) &&
         type.getNumElements() == 1;
}

std::optional<mlir::Attribute> GetSplatConstantAttr(mlir::Value val) {
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
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense || dense.empty() || !dense.isSplat()) {
    return std::nullopt;
  }
  return dense.getSplatValue<mlir::Attribute>();
}

bool IsOneConstant(mlir::Value val) {
  while (val) {
    if (auto cvt = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      val = cvt.getOperand();
      continue;
    }
    break;
  }
  auto attr = GetSplatConstantAttr(val);
  if (!attr.has_value()) {
    return false;
  }
  if (auto flt = llvm::dyn_cast<mlir::FloatAttr>(*attr)) {
    return flt.getValue().isExactlyValue(1.0);
  }
  if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(*attr)) {
    return integer.getValue().isOne();
  }
  return false;
}

mlir::Value RebuildScalarSplatOrBroadcast(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val,
    llvm::ArrayRef<int64_t> target_shape) {
  auto orig_type = llvm::cast<mlir::RankedTensorType>(val.getType());
  auto target_type =
      mlir::RankedTensorType::get(target_shape, orig_type.getElementType());
  if (auto attr = GetSplatConstantAttr(val)) {
    auto splat_attr = mlir::DenseElementsAttr::get(target_type, *attr);
    return mlir::stablehlo::ConstantOp::create(builder, loc, target_type,
                                               splat_attr);
  }
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
  auto root_type = llvm::cast<mlir::RankedTensorType>(root.getType());
  if (root_type.getRank() != 0) {
    auto scalar_type =
        mlir::RankedTensorType::get({}, target_type.getElementType());
    root = mlir::stablehlo::ReshapeOp::create(builder, loc, scalar_type, root);
  }
  return mlir::stablehlo::BroadcastInDimOp::create(
      builder, loc, target_type, root, builder.getDenseI64ArrayAttr({}));
}

bool IsSupportedPureElementwiseOp(mlir::Operation* op) {
  if (!op || op->getNumResults() != 1) {
    return false;
  }
  return llvm::isa<
      mlir::stablehlo::ConvertOp, mlir::stablehlo::NegOp,
      mlir::stablehlo::ExpOp, mlir::stablehlo::Expm1Op, mlir::stablehlo::LogOp,
      mlir::stablehlo::Log1pOp, mlir::chlo::ErfOp, mlir::stablehlo::TanhOp,
      mlir::stablehlo::SineOp, mlir::stablehlo::CosineOp,
      mlir::stablehlo::SqrtOp, mlir::stablehlo::RsqrtOp, mlir::stablehlo::AbsOp,
      mlir::stablehlo::LogisticOp, mlir::stablehlo::FloorOp,
      mlir::stablehlo::CeilOp, mlir::stablehlo::SignOp, mlir::stablehlo::AddOp,
      mlir::stablehlo::SubtractOp, mlir::stablehlo::MulOp,
      mlir::stablehlo::DivOp, mlir::stablehlo::MaxOp, mlir::stablehlo::MinOp,
      mlir::stablehlo::PowOp, mlir::stablehlo::CompareOp,
      mlir::stablehlo::SelectOp, mlir::stablehlo::ClampOp>(op);
}

struct PureElementwiseDagMatch {
  llvm::SmallVector<mlir::Operation*, 12> ops;
  mlir::Value exit_val;
  bool has_branch = false;
};

// Matches a single-entry (`entry_val`), single-exit (`exit_val`) pure
// elementwise DAG in the same block where every external operand is a scalar
// splat/broadcast and no intermediate value escapes outside the DAG.
std::optional<PureElementwiseDagMatch> MatchSingleEntrySingleExitElementwiseDag(
    mlir::Value entry_val) {
  mlir::RankedTensorType entry_type;
  if (!IsStaticRankedTensor(entry_val.getType(), &entry_type) ||
      entry_val.use_empty()) {
    return std::nullopt;
  }
  mlir::Block* block = entry_val.getParentBlock();
  if (!block) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> expected_shape = entry_type.getShape();

  llvm::DenseSet<mlir::Value> dag_values;
  dag_values.insert(entry_val);
  llvm::DenseSet<mlir::Operation*> dag_ops_set;
  llvm::SmallVector<mlir::Operation*, 12> dag_ops;
  bool has_branch = false;

  std::optional<PureElementwiseDagMatch> best_match;

  // Bounded forward closure: at each step, pick an unvisited user of any value
  // in `dag_values` whose non-scalar operands are all already in `dag_values`.
  while (true) {
    // Check if the current set of ops forms a valid single-exit DAG:
    // Every value in `dag_values` except exactly one `exit_cand` (with
    // `exit_cand != entry_val`) has all of its uses inside `dag_ops_set`.
    if (!dag_ops.empty()) {
      mlir::Value single_exit;
      bool valid_closure = true;
      for (mlir::Value v : dag_values) {
        bool has_external_use = false;
        int internal_uses = 0;
        for (mlir::Operation* u : v.getUsers()) {
          if (dag_ops_set.contains(u)) {
            ++internal_uses;
          } else {
            has_external_use = true;
          }
        }
        if (internal_uses > 1) {
          has_branch = true;
        }
        if (v == entry_val) {
          if (has_external_use) {
            valid_closure = false;
            break;
          }
        } else if (has_external_use) {
          if (single_exit) {
            valid_closure = false;
            break;
          }
          single_exit = v;
        }
      }
      if (valid_closure && single_exit) {
        best_match = PureElementwiseDagMatch{dag_ops, single_exit, has_branch};
      }
    }

    // Find the next ready operation to add to the closure.
    mlir::Operation* next_op = nullptr;
    for (mlir::Value v : dag_values) {
      for (mlir::Operation* u : v.getUsers()) {
        if (dag_ops_set.contains(u) || u->getBlock() != block ||
            !IsSupportedPureElementwiseOp(u)) {
          continue;
        }
        mlir::RankedTensorType res_type;
        if (!IsStaticRankedTensor(u->getResult(0).getType(), &res_type) ||
            res_type.getShape() != expected_shape) {
          continue;
        }
        bool operands_ready = true;
        int dag_operand_count = 0;
        for (mlir::Value opnd : u->getOperands()) {
          if (dag_values.contains(opnd)) {
            ++dag_operand_count;
          } else if (!IsScalarSplatOrBroadcast(opnd)) {
            operands_ready = false;
            break;
          }
        }
        if (operands_ready && dag_operand_count >= 1) {
          if (next_op == nullptr || u->isBeforeInBlock(next_op)) {
            next_op = u;
          }
        }
      }
    }
    if (!next_op) {
      break;
    }
    int dag_inputs = 0;
    for (mlir::Value opnd : next_op->getOperands()) {
      if (dag_values.contains(opnd)) {
        ++dag_inputs;
      }
    }
    if (dag_inputs > 1) {
      has_branch = true;
    }
    dag_ops_set.insert(next_op);
    dag_ops.push_back(next_op);
    dag_values.insert(next_op->getResult(0));
  }

  return best_match;
}

mlir::Value CloneElementwiseDagOnShape(mlir::OpBuilder& builder,
                                       mlir::Location loc,
                                       mlir::Value new_entry,
                                       llvm::ArrayRef<int64_t> target_shape,
                                       mlir::Value old_entry,
                                       const PureElementwiseDagMatch& match) {
  mlir::IRMapping mapping;
  mapping.map(old_entry, new_entry);

  for (mlir::Operation* op : match.ops) {
    for (mlir::Value opnd : op->getOperands()) {
      if (!mapping.contains(opnd)) {
        mlir::Value rebuilt =
            RebuildScalarSplatOrBroadcast(builder, loc, opnd, target_shape);
        mapping.map(opnd, rebuilt);
      }
    }
    mlir::Operation* cloned = builder.clone(*op, mapping);
    auto old_res_type =
        llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto new_res_type = mlir::RankedTensorType::get(
        target_shape, old_res_type.getElementType());
    cloned->getResult(0).setType(new_res_type);
    mapping.map(op->getResult(0), cloned->getResult(0));
  }
  return mapping.lookup(match.exit_val);
}

bool IsConvolutionOrBiasedConvolution(mlir::Value val) {
  auto val_type = llvm::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!val_type || !val_type.hasStaticShape() || val_type.getRank() < 3) {
    return false;
  }
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvolutionOp>()) {
    auto dnums = conv.getDimensionNumbers();
    return dnums.getOutputBatchDimension() == 0 &&
           dnums.getOutputFeatureDimension() == 1;
  }
  if (auto add = val.getDefiningOp<mlir::stablehlo::AddOp>()) {
    for (int side = 0; side < 2; ++side) {
      mlir::Value main_v = (side == 0) ? add.getLhs() : add.getRhs();
      mlir::Value bias_v = (side == 0) ? add.getRhs() : add.getLhs();
      auto conv = main_v.getDefiningOp<mlir::stablehlo::ConvolutionOp>();
      if (!conv || !main_v.hasOneUse()) {
        continue;
      }
      auto dnums = conv.getDimensionNumbers();
      if (dnums.getOutputBatchDimension() != 0 ||
          dnums.getOutputFeatureDimension() != 1) {
        continue;
      }
      auto bcast = bias_v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
      if (bcast && bcast.getBroadcastDimensions().size() == 1 &&
          bcast.getBroadcastDimensions()[0] == 1) {
        return true;
      }
    }
  }
  return false;
}

// Pattern 1: Commutes a multi-branch pure elementwise activation DAG F(y)
// (such as GELU or SiLU) upstream across a channel-first-to-channel-last
// ReshapeOp + TransposeOp view y = Pi(x) when x is produced by a ConvolutionOp
// (or ConvolutionOp + bias AddOp): F(Pi(x)) -> Pi(F(x)).
bool TryCommuteMultiBranchElementwiseBeforeConvTranspose(
    mlir::stablehlo::TransposeOp tr_op) {
  if (!tr_op || !tr_op->getBlock() || tr_op->use_empty()) {
    return false;
  }
  mlir::Value y = tr_op.getResult();
  mlir::RankedTensorType y_type;
  if (!IsStaticRankedTensor(y.getType(), &y_type)) {
    return false;
  }

  mlir::Value tr_in = tr_op.getOperand();
  if (!tr_in.hasOneUse()) {
    return false;
  }
  mlir::stablehlo::ReshapeOp mid_reshape;
  mlir::Value x = tr_in;
  if (auto r = tr_in.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    if (!r.getOperand().hasOneUse()) {
      return false;
    }
    mid_reshape = r;
    x = r.getOperand();
  }
  mlir::RankedTensorType x_type;
  if (!IsStaticRankedTensor(x.getType(), &x_type) ||
      !IsConvolutionOrBiasedConvolution(x)) {
    return false;
  }

  // Verify `tr_op` moves the channel dimension (axis 1 of `x`, or axis 0 of
  // `mid_reshape` when `x` has unit batch dim 0) to the trailing dimension of
  // `y`.
  llvm::ArrayRef<int64_t> perm = tr_op.getPermutation();
  if (perm.empty() || y_type.getShape().back() != x_type.getDimSize(1)) {
    return false;
  }
  if (mid_reshape) {
    auto mid_type = llvm::cast<mlir::RankedTensorType>(mid_reshape.getType());
    if (x_type.getDimSize(0) != 1 ||
        mid_type.getDimSize(0) != x_type.getDimSize(1) || perm.back() != 0) {
      return false;
    }
  } else {
    if (perm.back() != 1) {
      return false;
    }
  }

  auto dag_match = MatchSingleEntrySingleExitElementwiseDag(y);
  if (!dag_match.has_value() || !dag_match->has_branch) {
    return false;
  }
  auto exit_type =
      llvm::cast<mlir::RankedTensorType>(dag_match->exit_val.getType());

  mlir::OpBuilder builder(dag_match->exit_val.getDefiningOp());
  mlir::Location loc = tr_op.getLoc();
  mlir::Value fx = CloneElementwiseDagOnShape(builder, loc, x,
                                              x_type.getShape(), y, *dag_match);

  mlir::Value pi_in = fx;
  if (mid_reshape) {
    auto mid_type = llvm::cast<mlir::RankedTensorType>(mid_reshape.getType());
    auto new_mid_type = mlir::RankedTensorType::get(mid_type.getShape(),
                                                    exit_type.getElementType());
    pi_in =
        mlir::stablehlo::ReshapeOp::create(builder, loc, new_mid_type, pi_in);
  }
  mlir::Value new_tr = mlir::stablehlo::TransposeOp::create(
      builder, loc, exit_type, pi_in, tr_op.getPermutationAttr());
  dag_match->exit_val.replaceAllUsesWith(new_tr);
  return true;
}

bool IsUnitWindow1DConv(mlir::stablehlo::ConvolutionOp conv, int64_t& out_b,
                        int64_t& out_c_in, int64_t& out_c_out, int64_t& out_s) {
  if (!conv || conv.getFeatureGroupCount() != 1 ||
      conv.getBatchGroupCount() != 1) {
    return false;
  }
  mlir::RankedTensorType lhs_t, rhs_t, out_t;
  if (!IsStaticRankedTensor(conv.getLhs().getType(), &lhs_t) ||
      !IsStaticRankedTensor(conv.getRhs().getType(), &rhs_t) ||
      !IsStaticRankedTensor(conv.getType(), &out_t) || lhs_t.getRank() != 3 ||
      rhs_t.getRank() != 3 || out_t.getRank() != 3) {
    return false;
  }
  auto dnums = conv.getDimensionNumbers();
  // GENERIC: semantic 1D channel-first layout [B, C, S] x [C_out, C_in, 1].
  if (dnums.getInputBatchDimension() != 0 ||
      dnums.getInputFeatureDimension() != 1 ||
      dnums.getOutputBatchDimension() != 0 ||
      dnums.getOutputFeatureDimension() != 1 ||
      dnums.getKernelOutputFeatureDimension() != 0 ||
      dnums.getKernelInputFeatureDimension() != 1 ||
      dnums.getInputSpatialDimensions() != llvm::ArrayRef<int64_t>{2} ||
      dnums.getKernelSpatialDimensions() != llvm::ArrayRef<int64_t>{2} ||
      dnums.getOutputSpatialDimensions() != llvm::ArrayRef<int64_t>{2}) {
    return false;
  }
  if (rhs_t.getDimSize(2) != 1 || lhs_t.getDimSize(0) != out_t.getDimSize(0) ||
      lhs_t.getDimSize(2) != out_t.getDimSize(2)) {
    return false;
  }
  if (auto strides = conv.getWindowStrides()) {
    for (int64_t s : *strides) {
      if (s != 1) return false;
    }
  }
  if (auto pad = conv.getPadding()) {
    for (int64_t p : pad->getValues<int64_t>()) {
      if (p != 0) return false;
    }
  }
  if (auto ld = conv.getLhsDilation()) {
    for (int64_t d : *ld) {
      if (d != 1) return false;
    }
  }
  if (auto rd = conv.getRhsDilation()) {
    for (int64_t d : *rd) {
      if (d != 1) return false;
    }
  }
  if (auto wr = conv.getWindowReversal()) {
    for (bool r : *wr) {
      if (r) return false;
    }
  }
  out_b = lhs_t.getDimSize(0);
  out_c_in = lhs_t.getDimSize(1);
  out_c_out = out_t.getDimSize(1);
  out_s = lhs_t.getDimSize(2);
  return out_b > 0 && out_c_in > 0 && out_c_out > 0 && out_s > 0 &&
         rhs_t.getDimSize(0) == out_c_out && rhs_t.getDimSize(1) == out_c_in;
}

// Matches a channel-last-to-channel-first transpose producing `x_cf` of shape
// `[B, C, S]`:
//   - Direct `TransposeOp([B, S, C], {0, 2, 1})`
//   - Direct `TransposeOp([S, B, C], {1, 2, 0})`
//   - Unit-batch `ReshapeOp([1, S, C] -> [S, C, 1]) -> TransposeOp({2, 1, 0})`
// Emits a `[B * S, C]` 2D tensor when `builder` is provided, or returns true
// when probing (`builder == nullptr`).
bool MatchChannelLastToFirst3D(mlir::Value x_cf, int64_t b, int64_t c,
                               int64_t s, mlir::Value& out_base_cl,
                               bool& out_is_sbc) {
  auto tr = x_cf.getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!tr) {
    return false;
  }
  llvm::ArrayRef<int64_t> perm = tr.getPermutation();
  if (perm.size() != 3) {
    return false;
  }
  mlir::Value tr_in = tr.getOperand();
  mlir::RankedTensorType tr_in_type;
  if (!IsStaticRankedTensor(tr_in.getType(), &tr_in_type) ||
      tr_in_type.getRank() != 3) {
    return false;
  }
  if (perm == llvm::ArrayRef<int64_t>{0, 2, 1} &&
      tr_in_type.getShape() == llvm::ArrayRef<int64_t>{b, s, c}) {
    out_base_cl = tr_in;
    out_is_sbc = false;
    return true;
  }
  if (perm == llvm::ArrayRef<int64_t>{1, 2, 0} &&
      tr_in_type.getShape() == llvm::ArrayRef<int64_t>{s, b, c}) {
    out_base_cl = tr_in;
    out_is_sbc = true;
    return true;
  }
  if (b == 1 && perm == llvm::ArrayRef<int64_t>{2, 1, 0} &&
      tr_in_type.getShape() == llvm::ArrayRef<int64_t>{s, c, 1}) {
    if (auto r = tr_in.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      mlir::RankedTensorType r_in_type;
      if (IsStaticRankedTensor(r.getOperand().getType(), &r_in_type) &&
          r_in_type.getShape() == llvm::ArrayRef<int64_t>{1, s, c}) {
        out_base_cl = r.getOperand();
        out_is_sbc = false;
        return true;
      }
    }
  }
  return false;
}

bool Match1DFeatureBiasAdd(mlir::Value val, int64_t c_out,
                           mlir::Value& out_conv_val,
                           mlir::Value& out_bias_1d) {
  auto add = val.getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add) {
    return false;
  }
  for (int side = 0; side < 2; ++side) {
    mlir::Value main_v = (side == 0) ? add.getLhs() : add.getRhs();
    mlir::Value bias_v = (side == 0) ? add.getRhs() : add.getLhs();
    auto bcast = bias_v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!bcast ||
        bcast.getBroadcastDimensions() != llvm::ArrayRef<int64_t>{1}) {
      continue;
    }
    mlir::RankedTensorType b_type;
    if (!IsStaticRankedTensor(bcast.getOperand().getType(), &b_type) ||
        b_type.getRank() != 1 || b_type.getDimSize(0) != c_out) {
      continue;
    }
    out_conv_val = main_v;
    out_bias_1d = bcast.getOperand();
    return true;
  }
  return false;
}

// Matches `sigmoid(x)` implemented either as `stablehlo.logistic(x)` or
// `1 / (1 + exp(-x))`.
bool MatchSigmoidChain(mlir::Value sig_val, mlir::Value expected_in) {
  if (auto log_op = sig_val.getDefiningOp<mlir::stablehlo::LogisticOp>()) {
    return log_op.getOperand() == expected_in && log_op->hasOneUse();
  }
  auto div_op = sig_val.getDefiningOp<mlir::stablehlo::DivOp>();
  if (!div_op || !div_op->hasOneUse() || !IsOneConstant(div_op.getLhs())) {
    return false;
  }
  auto add_op = div_op.getRhs().getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add_op || !add_op->hasOneUse()) {
    return false;
  }
  mlir::Value exp_cand;
  if (IsOneConstant(add_op.getRhs())) {
    exp_cand = add_op.getLhs();
  } else if (IsOneConstant(add_op.getLhs())) {
    exp_cand = add_op.getRhs();
  } else {
    return false;
  }
  auto exp_op = exp_cand.getDefiningOp<mlir::stablehlo::ExpOp>();
  if (!exp_op || !exp_op->hasOneUse()) {
    return false;
  }
  auto neg_op = exp_op.getOperand().getDefiningOp<mlir::stablehlo::NegOp>();
  return neg_op && neg_op->hasOneUse() && neg_op.getOperand() == expected_in;
}

// Pattern 2a: Commutes channel-last-to-channel-first transpose `Pi_in` across
// 1D unit-window `pointwise_conv1` (`[B, C_in, S] -> [B, 2*C, S]`) and
// channel-axis `GLU` (`[B, 2*C, S] -> [B, C, S]`): lowers `pointwise_conv1` to
// a 2D channel-minor `DotGeneralOp` (`[B*S, C_in] @ [2*C, C_in]^T -> [B*S,
// 2*C]`), evaluates `GLU` on `[B*S, 2*C] -> [B*S, C]`, and transposes `[B, S,
// C] -> [B, C, S]`.
bool TryCommutePointwiseConv1AndGluToChannelMinor(mlir::Operation* glu_root) {
  if (!glu_root || !glu_root->getBlock() || glu_root->use_empty()) {
    return false;
  }
  if (auto mul = llvm::dyn_cast<mlir::stablehlo::MulOp>(glu_root)) {
    if (mul->hasOneUse() &&
        llvm::isa<mlir::stablehlo::ConvertOp>(*mul->user_begin())) {
      return false;
    }
  }
  mlir::Value glu_out = glu_root->getResult(0);
  mlir::RankedTensorType out_type;
  if (!IsStaticRankedTensor(glu_out.getType(), &out_type) ||
      out_type.getRank() != 3) {
    return false;
  }
  const int64_t b = out_type.getDimSize(0);
  const int64_t c = out_type.getDimSize(1);
  const int64_t s = out_type.getDimSize(2);
  if (b <= 0 || c <= 0 || s <= 0) {
    return false;
  }

  mlir::Value mul_cand = glu_out;
  if (auto cvt_out = glu_out.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    if (!cvt_out.getOperand().hasOneUse()) {
      return false;
    }
    mul_cand = cvt_out.getOperand();
  }
  auto mul_op = mul_cand.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul_op) {
    return false;
  }
  auto acc_type = llvm::dyn_cast<mlir::RankedTensorType>(mul_op.getType());
  if (!acc_type || !acc_type.hasStaticShape() ||
      acc_type.getShape() != llvm::ArrayRef<int64_t>{b, c, s}) {
    return false;
  }

  mlir::stablehlo::SliceOp slice0, slice1;
  for (int side = 0; side < 2; ++side) {
    mlir::Value s0_cand = (side == 0) ? mul_op.getLhs() : mul_op.getRhs();
    mlir::Value sig_cand = (side == 0) ? mul_op.getRhs() : mul_op.getLhs();
    auto sl0 = s0_cand.getDefiningOp<mlir::stablehlo::SliceOp>();
    if (!sl0 || !sl0->hasOneUse()) {
      continue;
    }
    // Find the second slice feeding `sig_cand`.
    mlir::Value s1_cand;
    if (auto log_op = sig_cand.getDefiningOp<mlir::stablehlo::LogisticOp>()) {
      s1_cand = log_op.getOperand();
    } else if (auto div_op = sig_cand.getDefiningOp<mlir::stablehlo::DivOp>()) {
      if (auto add_op =
              div_op.getRhs().getDefiningOp<mlir::stablehlo::AddOp>()) {
        mlir::Value exp_v =
            IsOneConstant(add_op.getRhs()) ? add_op.getLhs() : add_op.getRhs();
        if (auto exp_op = exp_v.getDefiningOp<mlir::stablehlo::ExpOp>()) {
          if (auto neg_op =
                  exp_op.getOperand().getDefiningOp<mlir::stablehlo::NegOp>()) {
            s1_cand = neg_op.getOperand();
          }
        }
      }
    }
    if (!s1_cand || !s1_cand.hasOneUse()) {
      continue;
    }
    auto sl1 = s1_cand.getDefiningOp<mlir::stablehlo::SliceOp>();
    if (sl1 && MatchSigmoidChain(sig_cand, s1_cand)) {
      slice0 = sl0;
      slice1 = sl1;
      break;
    }
  }
  if (!slice0 || !slice1 || slice0.getOperand() != slice1.getOperand()) {
    return false;
  }

  auto check_glu_slice = [&](mlir::stablehlo::SliceOp sl, int64_t c_start,
                             int64_t c_limit) {
    return sl.getStartIndices() == llvm::ArrayRef<int64_t>{0, c_start, 0} &&
           sl.getLimitIndices() == llvm::ArrayRef<int64_t>{b, c_limit, s} &&
           sl.getStrides() == llvm::ArrayRef<int64_t>{1, 1, 1};
  };
  if (!check_glu_slice(slice0, 0, c) || !check_glu_slice(slice1, c, 2 * c)) {
    return false;
  }

  mlir::Value glu_in = slice0.getOperand();
  for (mlir::Operation* u : glu_in.getUsers()) {
    if (u != slice0.getOperation() && u != slice1.getOperation()) {
      return false;
    }
  }

  mlir::Value conv_or_bias = glu_in;
  if (auto cvt_in = glu_in.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    if (!cvt_in.getOperand().hasOneUse()) {
      return false;
    }
    conv_or_bias = cvt_in.getOperand();
  }

  mlir::Value conv_val = conv_or_bias;
  mlir::Value bias_1d;
  if (Match1DFeatureBiasAdd(conv_or_bias, 2 * c, conv_val, bias_1d)) {
    if (!conv_val.hasOneUse()) {
      return false;
    }
  }
  auto conv_op = conv_val.getDefiningOp<mlir::stablehlo::ConvolutionOp>();
  int64_t conv_b = 0, c_in = 0, c_out = 0, conv_s = 0;
  if (!IsUnitWindow1DConv(conv_op, conv_b, c_in, c_out, conv_s) ||
      conv_b != b || c_out != 2 * c || conv_s != s ||
      !conv_op.getLhs().hasOneUse()) {
    return false;
  }

  mlir::Value base_cl;
  bool is_sbc = false;
  if (!MatchChannelLastToFirst3D(conv_op.getLhs(), b, c_in, s, base_cl,
                                 is_sbc)) {
    return false;
  }

  mlir::OpBuilder builder(glu_root);
  mlir::Location loc = glu_root->getLoc();
  mlir::Type io_elem_type = out_type.getElementType();
  mlir::Type acc_elem_type = acc_type.getElementType();
  const int64_t m = b * s;

  mlir::Value x_bsc = base_cl;
  if (is_sbc) {
    auto bsc_type = mlir::RankedTensorType::get({b, s, c_in}, io_elem_type);
    x_bsc = mlir::stablehlo::TransposeOp::create(
        builder, loc, bsc_type, base_cl,
        builder.getDenseI64ArrayAttr({1, 0, 2}));
  }
  auto x_2d_type = mlir::RankedTensorType::get({m, c_in}, io_elem_type);
  mlir::Value x_2d =
      mlir::stablehlo::ReshapeOp::create(builder, loc, x_2d_type, x_bsc);

  auto w_2d_type = mlir::RankedTensorType::get({2 * c, c_in}, io_elem_type);
  mlir::Value w_2d = mlir::stablehlo::ReshapeOp::create(builder, loc, w_2d_type,
                                                        conv_op.getRhs());

  auto dot_out_type = mlir::RankedTensorType::get({m, 2 * c}, io_elem_type);
  auto dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(),
      /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{},
      /*lhsContractingDimensions=*/{1},
      /*rhsContractingDimensions=*/{1});
  mlir::Value dot_val =
      mlir::stablehlo::DotGeneralOp::create(
          builder, loc, dot_out_type, x_2d, w_2d, dot_dims,
          conv_op.getPrecisionConfigAttr(), mlir::stablehlo::DotAlgorithmAttr{})
          .getResult();

  if (bias_1d) {
    mlir::Value b_bcast = mlir::stablehlo::BroadcastInDimOp::create(
        builder, loc, dot_out_type, bias_1d, builder.getDenseI64ArrayAttr({1}));
    dot_val = mlir::stablehlo::AddOp::create(builder, loc, dot_val, b_bcast);
  }

  mlir::Value glu_in_2d = dot_val;
  auto glu_in_2d_type = mlir::RankedTensorType::get({m, 2 * c}, acc_elem_type);
  if (acc_elem_type != io_elem_type) {
    glu_in_2d = mlir::stablehlo::ConvertOp::create(builder, loc, glu_in_2d_type,
                                                   dot_val);
  }

  auto half_2d_type = mlir::RankedTensorType::get({m, c}, acc_elem_type);
  mlir::Value half0_2d =
      mlir::stablehlo::SliceOp::create(builder, loc, half_2d_type, glu_in_2d,
                                       builder.getDenseI64ArrayAttr({0, 0}),
                                       builder.getDenseI64ArrayAttr({m, c}),
                                       builder.getDenseI64ArrayAttr({1, 1}));
  mlir::Value half1_2d =
      mlir::stablehlo::SliceOp::create(builder, loc, half_2d_type, glu_in_2d,
                                       builder.getDenseI64ArrayAttr({0, c}),
                                       builder.getDenseI64ArrayAttr({m, 2 * c}),
                                       builder.getDenseI64ArrayAttr({1, 1}));

  auto flt_type = llvm::cast<mlir::FloatType>(acc_elem_type);
  llvm::APFloat one_apf(flt_type.getFloatSemantics(), 1);
  mlir::Value one_2d = mlir::stablehlo::ConstantOp::create(
      builder, loc, half_2d_type,
      mlir::DenseElementsAttr::get(half_2d_type, one_apf));
  mlir::Value neg_2d = mlir::stablehlo::NegOp::create(builder, loc, half1_2d);
  mlir::Value exp_2d = mlir::stablehlo::ExpOp::create(builder, loc, neg_2d);
  mlir::Value denom_2d =
      mlir::stablehlo::AddOp::create(builder, loc, exp_2d, one_2d);
  mlir::Value sig_2d =
      mlir::stablehlo::DivOp::create(builder, loc, one_2d, denom_2d);
  mlir::Value prod_2d =
      mlir::stablehlo::MulOp::create(builder, loc, half0_2d, sig_2d);

  mlir::Value out_2d = prod_2d;
  auto out_2d_type = mlir::RankedTensorType::get({m, c}, io_elem_type);
  if (acc_elem_type != io_elem_type) {
    out_2d =
        mlir::stablehlo::ConvertOp::create(builder, loc, out_2d_type, prod_2d);
  }

  auto bsc_out_type = mlir::RankedTensorType::get({b, s, c}, io_elem_type);
  mlir::Value bsc_out =
      mlir::stablehlo::ReshapeOp::create(builder, loc, bsc_out_type, out_2d);
  mlir::Value bcs_out = mlir::stablehlo::TransposeOp::create(
      builder, loc, out_type, bsc_out, builder.getDenseI64ArrayAttr({0, 2, 1}));
  glu_out.replaceAllUsesWith(bcs_out);
  return true;
}

// Matches a single-input, single-output pure elementwise activation chain
// ending at `act_out` (walking backwards through unary/binary pure elementwise
// ops whose external inputs are scalar splats/broadcasts).
bool PeelBackwardElementwiseDag(
    mlir::Value act_out, mlir::Value& out_entry_val,
    llvm::SmallVectorImpl<mlir::Operation*>& out_topo_ops) {
  mlir::RankedTensorType act_type;
  if (!IsStaticRankedTensor(act_out.getType(), &act_type)) {
    return false;
  }
  mlir::Block* block = act_out.getParentBlock();
  llvm::ArrayRef<int64_t> shape = act_type.getShape();

  // Walk backwards from `act_out` collecting elementwise ops to find candidate
  // single-entry roots, then verify with forward `MatchSingleEntrySingleExit`.
  llvm::SmallVector<mlir::Value, 8> worklist{act_out};
  llvm::DenseSet<mlir::Value> visited;
  mlir::Value best_entry;
  std::optional<PureElementwiseDagMatch> best_dag;

  while (!worklist.empty()) {
    mlir::Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second) {
      continue;
    }
    mlir::Operation* def = cur.getDefiningOp();
    if (!def || def->getBlock() != block ||
        !IsSupportedPureElementwiseOp(def)) {
      continue;
    }
    for (mlir::Value opnd : def->getOperands()) {
      mlir::RankedTensorType opnd_type;
      if (IsStaticRankedTensor(opnd.getType(), &opnd_type) &&
          opnd_type.getShape() == shape && !IsScalarSplatOrBroadcast(opnd)) {
        worklist.push_back(opnd);
        if (auto match = MatchSingleEntrySingleExitElementwiseDag(opnd)) {
          if (match->exit_val == act_out &&
              (!best_dag.has_value() ||
               match->ops.size() > best_dag->ops.size())) {
            best_entry = opnd;
            best_dag = std::move(match);
          }
        }
      }
    }
  }
  if (!best_dag.has_value()) {
    return false;
  }
  out_entry_val = best_entry;
  out_topo_ops.assign(best_dag->ops.begin(), best_dag->ops.end());
  return true;
}

// Pattern 2b: Commutes channel-first-to-channel-last `Pi_out` upstream across
// 1D unit-window `pointwise_conv2` (+ optional bias `AddOp`), optional pure
// elementwise activation (`SiLU` / `GELU`), and optional
// `BatchNormInferenceOp(feature_index=1)`, canceling `Pi_out` against an
// upstream `Pi_mid` when present.
bool TryCommutePointwiseConv2ToChannelMinor(
    mlir::stablehlo::TransposeOp post_tr) {
  if (!post_tr || !post_tr->getBlock() || post_tr->use_empty()) {
    return false;
  }
  mlir::RankedTensorType out_type;
  if (!IsStaticRankedTensor(post_tr.getType(), &out_type) ||
      out_type.getRank() != 3) {
    return false;
  }
  llvm::ArrayRef<int64_t> perm = post_tr.getPermutation();
  if (perm.size() != 3) {
    return false;
  }

  mlir::Value tr_in = post_tr.getOperand();
  if (!tr_in.hasOneUse()) {
    return false;
  }
  mlir::Value conv2_or_bias;
  int64_t b = 0, s = 0, c_out = 0;
  bool out_is_sbc = false;

  if (perm == llvm::ArrayRef<int64_t>{0, 2, 1}) {
    b = out_type.getDimSize(0);
    s = out_type.getDimSize(1);
    c_out = out_type.getDimSize(2);
    conv2_or_bias = tr_in;
  } else if (perm == llvm::ArrayRef<int64_t>{2, 0, 1}) {
    s = out_type.getDimSize(0);
    b = out_type.getDimSize(1);
    c_out = out_type.getDimSize(2);
    out_is_sbc = true;
    conv2_or_bias = tr_in;
  } else if (out_type.getDimSize(0) == 1 &&
             perm == llvm::ArrayRef<int64_t>{2, 1, 0}) {
    auto r = tr_in.getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!r || !r.getOperand().hasOneUse()) {
      return false;
    }
    b = 1;
    s = out_type.getDimSize(1);
    c_out = out_type.getDimSize(2);
    mlir::RankedTensorType r_type;
    if (!IsStaticRankedTensor(r.getType(), &r_type) ||
        r_type.getShape() != llvm::ArrayRef<int64_t>{c_out, s, 1}) {
      return false;
    }
    conv2_or_bias = r.getOperand();
  } else {
    return false;
  }

  mlir::RankedTensorType conv2_type;
  if (!IsStaticRankedTensor(conv2_or_bias.getType(), &conv2_type) ||
      conv2_type.getShape() != llvm::ArrayRef<int64_t>{b, c_out, s}) {
    return false;
  }

  mlir::Value conv2_val = conv2_or_bias;
  mlir::Value bias2_1d;
  if (Match1DFeatureBiasAdd(conv2_or_bias, c_out, conv2_val, bias2_1d)) {
    if (!conv2_val.hasOneUse()) {
      return false;
    }
  }
  auto conv2_op = conv2_val.getDefiningOp<mlir::stablehlo::ConvolutionOp>();
  int64_t conv_b = 0, c_in = 0, conv_c_out = 0, conv_s = 0;
  if (!IsUnitWindow1DConv(conv2_op, conv_b, c_in, conv_c_out, conv_s) ||
      conv_b != b || conv_c_out != c_out || conv_s != s ||
      !conv2_op.getLhs().hasOneUse()) {
    return false;
  }

  mlir::Value cur_cf = conv2_op.getLhs();
  mlir::Value act_entry;
  llvm::SmallVector<mlir::Operation*, 12> act_ops;
  bool has_act = false;
  if (PeelBackwardElementwiseDag(cur_cf, act_entry, act_ops)) {
    has_act = true;
    cur_cf = act_entry;
  }

  mlir::stablehlo::BatchNormInferenceOp bn_op;
  bool bn_has_in_cvt = false;
  bool bn_has_out_cvt = false;
  mlir::Type bn_out_cvt_elem_type;
  if (has_act || cur_cf.hasOneUse()) {
    mlir::Value bn_cand = cur_cf;
    if (auto cvt_out = bn_cand.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      if (cvt_out.getOperand().hasOneUse() &&
          cvt_out.getOperand()
              .getDefiningOp<mlir::stablehlo::BatchNormInferenceOp>()) {
        bn_has_out_cvt = true;
        bn_out_cvt_elem_type =
            llvm::cast<mlir::RankedTensorType>(cvt_out.getType())
                .getElementType();
        bn_cand = cvt_out.getOperand();
      }
    }
    if (auto bn =
            bn_cand.getDefiningOp<mlir::stablehlo::BatchNormInferenceOp>()) {
      if (bn.getFeatureIndex() == 1 && bn.getOperand().hasOneUse()) {
        bn_op = bn;
        mlir::Value bn_in = bn.getOperand();
        if (auto cvt_in = bn_in.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
          mlir::RankedTensorType raw_t;
          if (cvt_in.getOperand().hasOneUse() &&
              IsStaticRankedTensor(cvt_in.getOperand().getType(), &raw_t) &&
              raw_t.getShape() == llvm::ArrayRef<int64_t>{b, c_in, s}) {
            bn_has_in_cvt = true;
            cur_cf = cvt_in.getOperand();
          } else {
            cur_cf = bn_in;
          }
        } else {
          cur_cf = bn_in;
        }
      }
    }
  }

  mlir::Value base_cl;
  bool base_is_sbc = false;
  const bool matched_mid_tr =
      MatchChannelLastToFirst3D(cur_cf, b, c_in, s, base_cl, base_is_sbc);

  // Require either an upstream activation/BatchNorm or an inverse transpose
  // cancellation so standalone 1D convolutions without surrounding ops are not
  // rewritten when unneeded.
  if (!has_act && !bn_op && !matched_mid_tr) {
    return false;
  }

  mlir::OpBuilder builder(post_tr);
  mlir::Location loc = post_tr.getLoc();
  mlir::Type io_elem_type = out_type.getElementType();
  mlir::Type cur_cf_elem_type =
      llvm::cast<mlir::RankedTensorType>(cur_cf.getType()).getElementType();
  const int64_t m = b * s;
  auto cl_2d_type = mlir::RankedTensorType::get({m, c_in}, cur_cf_elem_type);

  mlir::Value cur_2d;
  if (matched_mid_tr && base_is_sbc == out_is_sbc) {
    cur_2d =
        mlir::stablehlo::ReshapeOp::create(builder, loc, cl_2d_type, base_cl);
  } else {
    llvm::SmallVector<int64_t, 3> cl_3d_shape =
        out_is_sbc ? llvm::SmallVector<int64_t, 3>{s, b, c_in}
                   : llvm::SmallVector<int64_t, 3>{b, s, c_in};
    llvm::SmallVector<int64_t, 3> tr_perm =
        out_is_sbc ? llvm::SmallVector<int64_t, 3>{2, 0, 1}
                   : llvm::SmallVector<int64_t, 3>{0, 2, 1};
    auto cl_3d_type =
        mlir::RankedTensorType::get(cl_3d_shape, cur_cf_elem_type);
    mlir::Value cl_3d = mlir::stablehlo::TransposeOp::create(
        builder, loc, cl_3d_type, cur_cf,
        builder.getDenseI64ArrayAttr(tr_perm));
    cur_2d =
        mlir::stablehlo::ReshapeOp::create(builder, loc, cl_2d_type, cl_3d);
  }

  if (bn_op) {
    auto bn_elem_type =
        llvm::cast<mlir::RankedTensorType>(bn_op.getType()).getElementType();
    auto bn_2d_type = mlir::RankedTensorType::get({m, c_in}, bn_elem_type);
    mlir::Value bn_in_2d = cur_2d;
    if (bn_has_in_cvt && bn_elem_type != cur_cf_elem_type) {
      bn_in_2d =
          mlir::stablehlo::ConvertOp::create(builder, loc, bn_2d_type, cur_2d);
    }
    // GENERIC: semantic feature dimension is axis 1 on 2D [M, C_in].
    mlir::Value bn_out_2d = mlir::stablehlo::BatchNormInferenceOp::create(
        builder, loc, bn_2d_type, bn_in_2d, bn_op.getScale(), bn_op.getOffset(),
        bn_op.getMean(), bn_op.getVariance(), bn_op.getEpsilonAttr(),
        builder.getI64IntegerAttr(1));
    if (bn_has_out_cvt && bn_elem_type != bn_out_cvt_elem_type) {
      auto cvt_2d_type =
          mlir::RankedTensorType::get({m, c_in}, bn_out_cvt_elem_type);
      bn_out_2d = mlir::stablehlo::ConvertOp::create(builder, loc, cvt_2d_type,
                                                     bn_out_2d);
    }
    cur_2d = bn_out_2d;
  }

  if (has_act) {
    PureElementwiseDagMatch act_match{act_ops, conv2_op.getLhs(),
                                      /*has_branch=*/true};
    cur_2d = CloneElementwiseDagOnShape(builder, loc, cur_2d, {m, c_in},
                                        act_entry, act_match);
  }

  auto w2_2d_type = mlir::RankedTensorType::get({c_out, c_in}, io_elem_type);
  mlir::Value w2_2d = mlir::stablehlo::ReshapeOp::create(
      builder, loc, w2_2d_type, conv2_op.getRhs());

  auto dot2_out_type = mlir::RankedTensorType::get({m, c_out}, io_elem_type);
  auto dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(),
      /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{},
      /*lhsContractingDimensions=*/{1},
      /*rhsContractingDimensions=*/{1});
  mlir::Value dot2_val = mlir::stablehlo::DotGeneralOp::create(
                             builder, loc, dot2_out_type, cur_2d, w2_2d,
                             dot_dims, conv2_op.getPrecisionConfigAttr(),
                             mlir::stablehlo::DotAlgorithmAttr{})
                             .getResult();

  if (bias2_1d) {
    mlir::Value b2_bcast = mlir::stablehlo::BroadcastInDimOp::create(
        builder, loc, dot2_out_type, bias2_1d,
        builder.getDenseI64ArrayAttr({1}));
    dot2_val = mlir::stablehlo::AddOp::create(builder, loc, dot2_val, b2_bcast);
  }

  mlir::Value out_3d =
      mlir::stablehlo::ReshapeOp::create(builder, loc, out_type, dot2_val);
  post_tr.replaceAllUsesWith(out_3d);
  return true;
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
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

class CommutePointwiseDagsAcrossChannelTransposesPass
    : public mlir::PassWrapper<CommutePointwiseDagsAcrossChannelTransposesPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CommutePointwiseDagsAcrossChannelTransposesPass)

  llvm::StringRef getArgument() const final {
    return "commute-pointwise-dags-across-channel-transposes";
  }

  llvm::StringRef getDescription() const final {
    return "Commutes multi-branch elementwise activation DAGs and "
           "channel-sandwiched 1D unit-window convolutions across channel "
           "transposes.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    // Step 1: Commute multi-branch pure elementwise activation DAGs upstream
    // across channel-first-to-channel-last ReshapeOp + TransposeOp views
    // following ConvolutionOp.
    llvm::SmallVector<mlir::stablehlo::TransposeOp, 16> tr_ops;
    func.walk([&](mlir::stablehlo::TransposeOp op) { tr_ops.push_back(op); });
    for (mlir::stablehlo::TransposeOp op : tr_ops) {
      if (op && op->getBlock()) {
        TryCommuteMultiBranchElementwiseBeforeConvTranspose(op);
      }
    }
    EraseTriviallyDeadOps(func);

    // Step 2a: Commute channel-last-to-channel-first transposes across 1D
    // unit-window pointwise_conv1 + GLU.
    llvm::SmallVector<mlir::Operation*, 16> glu_cands;
    func.walk([&](mlir::Operation* op) {
      if (llvm::isa<mlir::stablehlo::ConvertOp, mlir::stablehlo::MulOp>(op)) {
        glu_cands.push_back(op);
      }
    });
    for (mlir::Operation* op : glu_cands) {
      if (op && op->getBlock()) {
        TryCommutePointwiseConv1AndGluToChannelMinor(op);
      }
    }
    EraseTriviallyDeadOps(func);

    // Step 2b: Commute channel-first-to-channel-last transposes upstream across
    // 1D unit-window pointwise_conv2, elementwise activation, and BatchNorm.
    tr_ops.clear();
    func.walk([&](mlir::stablehlo::TransposeOp op) { tr_ops.push_back(op); });
    for (mlir::stablehlo::TransposeOp op : tr_ops) {
      if (op && op->getBlock()) {
        TryCommutePointwiseConv2ToChannelMinor(op);
      }
    }
    EraseTriviallyDeadOps(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
createCommutePointwiseDagsAcrossChannelTransposesPass() {
  return std::make_unique<CommutePointwiseDagsAcrossChannelTransposesPass>();
}

}  // namespace torch_tpu
