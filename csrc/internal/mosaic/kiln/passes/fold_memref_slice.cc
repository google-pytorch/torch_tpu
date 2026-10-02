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

#include "csrc/internal/mosaic/kiln/passes/fold_memref_slice.h"

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Support/WalkResult.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/util.h"

namespace torch_tpu::kiln {
namespace {

// Walks up chains of `tpu.memref_squeeze` and `tpu.memref_slice` to fold their
// offsets into the indices of `op` (`memref.load` or `memref.store`). Dead
// slice/squeeze chains are left to `-dce`.
template <typename OpT>
mlir::LogicalResult FoldMemRefSliceAndSqueeze(OpT op) {
  mlir::ImplicitLocOpBuilder builder(op.getLoc(), op);
  mlir::Value memref = op.getMemRef();
  llvm::SmallVector<mlir::Value> indices(op.getIndices());

  // We use setInsertionPoint to ensure that the generated ops are in the same
  // block as the original squeeze / slice ops.
  while (true) {
    if (auto squeeze_op = memref.getDefiningOp<mlir::tpu::MemRefSqueezeOp>()) {
      FAILUREOR_ASSIGN_OR_RETURN(
          llvm::SmallVector<int> squeezed_dims,
          mlir::tpu::computeSqueezedDimsChecked(
              squeeze_op, squeeze_op.getInput().getType().getShape(),
              squeeze_op.getType().getShape()));
      builder.setInsertionPoint(squeeze_op);
      mlir::Value zero = mlir::arith::ConstantIndexOp::create(builder, 0);
      for (int dim : squeezed_dims) {
        indices.insert(indices.begin() + dim, zero);
      }
      memref = squeeze_op.getInput();
    } else if (auto slice_op =
                   memref.getDefiningOp<mlir::tpu::MemRefSliceOp>()) {
      for (auto [base, index] :
           llvm::zip_equal(slice_op.getBaseIdx(), indices)) {
        builder.setInsertionPointAfterValue(base);
        mlir::Value base_idx = builder.createOrFold<mlir::arith::IndexCastOp>(
            builder.getIndexType(), base);
        builder.setInsertionPoint(slice_op);
        index = builder.createOrFold<mlir::arith::AddIOp>(base_idx, index);
      }
      memref = slice_op.getMemRef();
    } else {
      break;
    }
  }

  if (memref != op.getMemRef()) {
    op.getMemrefMutable().assign(memref);
    op.getIndicesMutable().assign(indices);
  }
  return mlir::success();
}

class FoldMemRefSlicePass
    : public mlir::PassWrapper<FoldMemRefSlicePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldMemRefSlicePass)

  llvm::StringRef getArgument() const override {
    return "kiln-fold-memref-slice";
  }

  llvm::StringRef getDescription() const override {
    return "Folds tpu.memref_slice and tpu.memref_squeeze into scalar "
           "memref.load and memref.store operations.";
  }

  void runOnOperation() override {
    mlir::WalkResult result =
        getOperation().walk([](mlir::Operation* op) -> mlir::WalkResult {
          if (auto load_op = llvm::dyn_cast<mlir::memref::LoadOp>(op)) {
            return FoldMemRefSliceAndSqueeze(load_op);
          }
          if (auto store_op = llvm::dyn_cast<mlir::memref::StoreOp>(op)) {
            return FoldMemRefSliceAndSqueeze(store_op);
          }
          return mlir::WalkResult::advance();
        });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateFoldMemRefSlicePass() {
  return std::make_unique<FoldMemRefSlicePass>();
}

}  // namespace torch_tpu::kiln
