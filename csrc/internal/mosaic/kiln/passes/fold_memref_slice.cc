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

#include <cstdint>
#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
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

// Inserts zero indices into `indices` for the dimensions squeezed by
// `squeeze_op`. Returns the unsqueezed input memref on success, or failure if
// the squeezed dimensions cannot be computed.
mlir::FailureOr<mlir::Value> FoldMemRefSqueeze(
    mlir::tpu::MemRefSqueezeOp squeeze_op, mlir::ImplicitLocOpBuilder& builder,
    llvm::SmallVector<mlir::Value>& indices) {
  FAILUREOR_ASSIGN_OR_RETURN(
      llvm::SmallVector<int64_t> squeezed_dims,
      mlir::tpu::computeSqueezedDimsChecked(
          squeeze_op, squeeze_op.getInput().getType().getShape(),
          squeeze_op.getType().getShape()));
  builder.setInsertionPoint(squeeze_op);
  mlir::Value zero = mlir::arith::ConstantIndexOp::create(builder, 0);
  for (int64_t dim : squeezed_dims) {
    indices.insert(indices.begin() + dim, zero);
  }
  return squeeze_op.getInput();
}

// Folds `slice_op`'s dynamic offsets into `indices`, keeping each offset
// addition hoisted to `slice_op` when `index` dominates it and otherwise
// inserting after `index`. Returns the unsliced input memref of `slice_op`.
[[nodiscard]] mlir::Value FoldMemRefSlice(
    mlir::tpu::MemRefSliceOp slice_op, const mlir::DominanceInfo& dom,
    mlir::ImplicitLocOpBuilder& builder,
    llvm::SmallVector<mlir::Value>& indices) {
  for (auto [base, index] : llvm::zip_equal(slice_op.getBaseIdx(), indices)) {
    builder.setInsertionPointAfterValue(base);
    mlir::Value base_idx = builder.createOrFold<mlir::arith::IndexCastOp>(
        builder.getIndexType(), base);
    // Keep the offset addition hoisted to `slice_op` when `index` already
    // dominates it; otherwise insert after `index` so both operands dominate
    // the `arith.addi`.
    if (dom.properlyDominates(index, slice_op)) {
      builder.setInsertionPoint(slice_op);
    } else {
      builder.setInsertionPointAfterValue(index);
    }
    index = builder.createOrFold<mlir::arith::AddIOp>(base_idx, index);
  }
  return slice_op.getMemRef();
}

// Walks up chains of `tpu.memref_squeeze` and `tpu.memref_slice` to fold their
// offsets into the indices of `op` (`memref.load` or `memref.store`). Dead
// slice/squeeze chains are left to `-dce`. Returns failure if a squeeze
// operation has invalid squeezed dimensions.
template <typename OpT>
mlir::LogicalResult FoldMemRefSliceAndSqueeze(OpT op,
                                              const mlir::DominanceInfo& dom) {
  mlir::ImplicitLocOpBuilder builder(op.getLoc(), op);
  mlir::Value memref = op.getMemRef();
  llvm::SmallVector<mlir::Value> indices(op.getIndices());

  while (true) {
    if (auto squeeze_op = memref.getDefiningOp<mlir::tpu::MemRefSqueezeOp>()) {
      FAILUREOR_ASSIGN_OR_RETURN(
          memref, FoldMemRefSqueeze(squeeze_op, builder, indices));
    } else if (auto slice_op =
                   memref.getDefiningOp<mlir::tpu::MemRefSliceOp>()) {
      memref = FoldMemRefSlice(slice_op, dom, builder, indices);
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
    const mlir::DominanceInfo dom(getOperation());
    mlir::WalkResult result =
        getOperation().walk([&dom](mlir::Operation* op) -> mlir::WalkResult {
          if (auto load_op = llvm::dyn_cast<mlir::memref::LoadOp>(op)) {
            return FoldMemRefSliceAndSqueeze(load_op, dom);
          }
          if (auto store_op = llvm::dyn_cast<mlir::memref::StoreOp>(op)) {
            return FoldMemRefSliceAndSqueeze(store_op, dom);
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
