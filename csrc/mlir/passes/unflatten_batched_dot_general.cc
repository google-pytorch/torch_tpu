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

#include "csrc/mlir/passes/unflatten_batched_dot_general.h"

#include <cstdint>
#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// Information about an unflattened 4D operand of a batched DotGeneralOp.
struct UnflattenedOperandInfo {
  mlir::Value tensor_4d;
  int64_t batch_dim0 = 0;
  int64_t batch_dim1 = 1;
  int64_t contracting_dim = -1;
  int64_t b0_size = 0;
  int64_t b1_size = 0;
  bool absorbed_transpose = false;
};

// Matches a 3D batched dot operand `val_3d` of shape `[B*H, D1, D2]` (where
// `contracting_axis_3d` is in {1, 2}) produced by reshaping a 4D tensor of
// shape `[B, H, ...]` (optionally followed by a 3D `{0, 2, 1}` transpose).
// When the 4D tensor is produced by a `TransposeOp` that moves nonleading
// batch axes into `(0, 1)`, absorbs that 4D `TransposeOp` directly into the
// returned `batch_dim0`, `batch_dim1`, and `contracting_dim`.
bool Match3DBatchedOperand(mlir::Value val_3d, int64_t contracting_axis_3d,
                           UnflattenedOperandInfo& out_info) {
  auto type_3d = mlir::dyn_cast<mlir::RankedTensorType>(val_3d.getType());
  if (!type_3d || !type_3d.hasStaticShape() || type_3d.getRank() != 3) {
    return false;
  }
  if (contracting_axis_3d != 1 && contracting_axis_3d != 2) {
    return false;
  }

  mlir::Value cur = val_3d;
  int64_t pre_tr_contracting_3d = contracting_axis_3d;
  if (auto tr_3d = cur.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    llvm::ArrayRef<int64_t> perm_3d = tr_3d.getPermutation();
    if (perm_3d.size() == 3 && perm_3d[0] == 0 && perm_3d[1] == 2 &&
        perm_3d[2] == 1) {
      cur = tr_3d.getOperand();
      pre_tr_contracting_3d = (contracting_axis_3d == 1) ? 2 : 1;
    }
  }

  auto resh_op = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!resh_op) {
    return false;
  }
  mlir::Value val_4d = resh_op.getOperand();
  auto type_4d = mlir::dyn_cast<mlir::RankedTensorType>(val_4d.getType());
  auto resh_out_type =
      mlir::dyn_cast<mlir::RankedTensorType>(resh_op.getType());
  if (!type_4d || !type_4d.hasStaticShape() || type_4d.getRank() != 4 ||
      !resh_out_type || !resh_out_type.hasStaticShape()) {
    return false;
  }

  const int64_t b0 = type_4d.getDimSize(0);
  const int64_t b1 = type_4d.getDimSize(1);
  if (b0 * b1 != resh_out_type.getDimSize(0) ||
      type_4d.getDimSize(2) != resh_out_type.getDimSize(1) ||
      type_4d.getDimSize(3) != resh_out_type.getDimSize(2)) {
    return false;
  }

  const int64_t axis_in_val_4d = pre_tr_contracting_3d + 1;
  if (auto tr_4d = val_4d.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    auto src_4d_type =
        mlir::dyn_cast<mlir::RankedTensorType>(tr_4d.getOperand().getType());
    llvm::ArrayRef<int64_t> perm = tr_4d.getPermutation();
    if (src_4d_type && src_4d_type.hasStaticShape() &&
        src_4d_type.getRank() == 4 && perm.size() == 4) {
      const bool non_trivial_batch_perm = !(perm[0] == 0 && perm[1] == 1);
      if (non_trivial_batch_perm) {
        out_info.tensor_4d = tr_4d.getOperand();
        out_info.batch_dim0 = perm[0];
        out_info.batch_dim1 = perm[1];
        out_info.contracting_dim = perm[axis_in_val_4d];
        out_info.b0_size = b0;
        out_info.b1_size = b1;
        out_info.absorbed_transpose = true;
        return true;
      }
    }
  }

  out_info.tensor_4d = val_4d;
  out_info.batch_dim0 = 0;
  out_info.batch_dim1 = 1;
  out_info.contracting_dim = axis_in_val_4d;
  out_info.b0_size = b0;
  out_info.b1_size = b1;
  out_info.absorbed_transpose = false;
  return true;
}

// Attempts to unflatten a 3D DotGeneralOp into 4D and fold 4D transposes.
// Returns true if the op was rewritten.
bool TryUnflatten3DDotGeneral(mlir::stablehlo::DotGeneralOp dot_op) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(dot_op.getType());
  auto lhs_type =
      mlir::dyn_cast<mlir::RankedTensorType>(dot_op.getLhs().getType());
  auto rhs_type =
      mlir::dyn_cast<mlir::RankedTensorType>(dot_op.getRhs().getType());
  if (!out_type || !out_type.hasStaticShape() || out_type.getRank() != 3 ||
      !lhs_type || !lhs_type.hasStaticShape() || lhs_type.getRank() != 3 ||
      !rhs_type || !rhs_type.hasStaticShape() || rhs_type.getRank() != 3) {
    return false;
  }

  auto dnums = dot_op.getDotDimensionNumbers();
  llvm::ArrayRef<int64_t> lhs_batch = dnums.getLhsBatchingDimensions();
  llvm::ArrayRef<int64_t> rhs_batch = dnums.getRhsBatchingDimensions();
  llvm::ArrayRef<int64_t> lhs_contract = dnums.getLhsContractingDimensions();
  llvm::ArrayRef<int64_t> rhs_contract = dnums.getRhsContractingDimensions();
  if (lhs_batch.size() != 1 || rhs_batch.size() != 1 || lhs_batch[0] != 0 ||
      rhs_batch[0] != 0 || lhs_contract.size() != 1 ||
      rhs_contract.size() != 1) {
    return false;
  }

  UnflattenedOperandInfo lhs_info;
  UnflattenedOperandInfo rhs_info;
  if (!Match3DBatchedOperand(dot_op.getLhs(), lhs_contract[0], lhs_info) ||
      !Match3DBatchedOperand(dot_op.getRhs(), rhs_contract[0], rhs_info)) {
    return false;
  }
  if (lhs_info.b0_size != rhs_info.b0_size ||
      lhs_info.b1_size != rhs_info.b1_size) {
    return false;
  }
  if (!lhs_info.absorbed_transpose && !rhs_info.absorbed_transpose) {
    return false;
  }

  mlir::OpBuilder builder(dot_op);
  auto new_dnums = mlir::stablehlo::DotDimensionNumbersAttr::get(
      dot_op.getContext(),
      /*lhsBatchingDimensions=*/{lhs_info.batch_dim0, lhs_info.batch_dim1},
      /*rhsBatchingDimensions=*/{rhs_info.batch_dim0, rhs_info.batch_dim1},
      /*lhsContractingDimensions=*/{lhs_info.contracting_dim},
      /*rhsContractingDimensions=*/{rhs_info.contracting_dim});

  auto dot_4d_type = mlir::RankedTensorType::get(
      {lhs_info.b0_size, lhs_info.b1_size, out_type.getDimSize(1),
       out_type.getDimSize(2)},
      out_type.getElementType());
  auto new_dot = mlir::stablehlo::DotGeneralOp::create(
      builder, dot_op.getLoc(), dot_4d_type, lhs_info.tensor_4d,
      rhs_info.tensor_4d, new_dnums, dot_op.getPrecisionConfigAttr(),
      dot_op.getAlgorithmAttr());

  // Forward any direct ReshapeOp users that unflatten [B*H, M, N] back to
  // [B, H, M, N] directly to `new_dot`.
  llvm::SmallVector<mlir::stablehlo::ReshapeOp> direct_unflatten_users;
  for (mlir::Operation* user : dot_op->getUsers()) {
    if (auto resh = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(user)) {
      if (resh.getType() == dot_4d_type) {
        direct_unflatten_users.push_back(resh);
      }
    }
  }
  for (mlir::stablehlo::ReshapeOp resh : direct_unflatten_users) {
    resh.replaceAllUsesWith(new_dot.getResult());
  }

  if (!dot_op->use_empty()) {
    auto reshaped_3d = mlir::stablehlo::ReshapeOp::create(
        builder, dot_op.getLoc(), out_type, new_dot.getResult());
    dot_op.replaceAllUsesWith(reshaped_3d.getResult());
  }
  return true;
}

struct UnflattenBatchedDotGeneralPass
    : public mlir::PassWrapper<UnflattenBatchedDotGeneralPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnflattenBatchedDotGeneralPass)

  llvm::StringRef getArgument() const final {
    return "unflatten-batched-dot-general";
  }

  llvm::StringRef getDescription() const final {
    return "Unflattens 3D collapsed batch DotGeneralOp contractions and folds "
           "upstream 4D head split transposes into DotGeneral dimension "
           "numbers.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::stablehlo::DotGeneralOp> dot_ops;
    func.walk([&dot_ops](mlir::stablehlo::DotGeneralOp op) {
      dot_ops.push_back(op);
    });

    for (mlir::stablehlo::DotGeneralOp dot_op : dot_ops) {
      if (!dot_op->use_empty()) {
        TryUnflatten3DDotGeneral(dot_op);
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createUnflattenBatchedDotGeneralPass() {
  return std::make_unique<UnflattenBatchedDotGeneralPass>();
}

}  // namespace torch_tpu
