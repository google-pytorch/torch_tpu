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

#include "csrc/mlir/passes/fold_pad_add_to_concat.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

struct PadSliceLeaf {
  mlir::Value unpadded_input;
  int64_t pad_dim = -1;
  int64_t start_offset = 0;
  int64_t slice_size = 0;
};

bool IsZeroConstant(mlir::Value val) {
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr || !attr.isSplat()) {
    return false;
  }
  if (mlir::isa<mlir::FloatType>(attr.getElementType())) {
    return attr.getSplatValue<llvm::APFloat>().isZero();
  }
  if (mlir::isa<mlir::IntegerType>(attr.getElementType())) {
    return attr.getSplatValue<llvm::APInt>().isZero();
  }
  return false;
}

bool IsValidDimPadding(int64_t in_dim, int64_t out_dim, int64_t low,
                       int64_t high) {
  return in_dim > 0 && low >= 0 && high >= 0 && in_dim + low + high == out_dim;
}

int64_t FindSinglePaddedDim(mlir::RankedTensorType in_type,
                            mlir::RankedTensorType out_type,
                            llvm::ArrayRef<int64_t> low,
                            llvm::ArrayRef<int64_t> high) {
  int64_t pad_dim = -1;
  for (int64_t d = 0; d < in_type.getRank(); ++d) {
    if (!IsValidDimPadding(in_type.getDimSize(d), out_type.getDimSize(d),
                           low[d], high[d])) {
      return -1;
    }
    if (low[d] != 0 || high[d] != 0) {
      if (pad_dim != -1) {
        return -1;
      }
      pad_dim = d;
    }
  }
  return pad_dim;
}

bool ExtractPadSliceLeaf(mlir::Value val,
                         mlir::RankedTensorType expected_out_type,
                         PadSliceLeaf& out_leaf) {
  auto pad_op = val.getDefiningOp<mlir::stablehlo::PadOp>();
  if (!pad_op || pad_op.getType() != expected_out_type ||
      !IsZeroConstant(pad_op.getPaddingValue())) {
    return false;
  }
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(pad_op.getOperand().getType());
  if (!in_type || !in_type.hasStaticShape() ||
      in_type.getRank() != expected_out_type.getRank()) {
    return false;
  }
  const bool zero_interior = llvm::all_of(pad_op.getInteriorPadding(),
                                          [](int64_t v) { return v == 0; });
  if (!zero_interior) {
    return false;
  }
  const llvm::ArrayRef<int64_t> low = pad_op.getEdgePaddingLow();
  const llvm::ArrayRef<int64_t> high = pad_op.getEdgePaddingHigh();
  const int64_t pad_dim =
      FindSinglePaddedDim(in_type, expected_out_type, low, high);
  if (pad_dim < 0) {
    return false;
  }
  out_leaf.unpadded_input = pad_op.getOperand();
  out_leaf.pad_dim = pad_dim;
  out_leaf.start_offset = low[pad_dim];
  out_leaf.slice_size = in_type.getDimSize(pad_dim);
  return true;
}

bool CollectAddPadLeaves(mlir::stablehlo::AddOp root_add,
                         mlir::RankedTensorType out_type,
                         llvm::SmallVectorImpl<PadSliceLeaf>& out_leaves) {
  llvm::SmallVector<mlir::Value, 8> worklist{root_add.getRhs(),
                                             root_add.getLhs()};
  while (!worklist.empty()) {
    const mlir::Value cur = worklist.pop_back_val();
    if (auto nested_add = cur.getDefiningOp<mlir::stablehlo::AddOp>()) {
      if (nested_add->hasOneUse()) {
        worklist.push_back(nested_add.getRhs());
        worklist.push_back(nested_add.getLhs());
        continue;
      }
    }
    PadSliceLeaf leaf;
    if (!ExtractPadSliceLeaf(cur, out_type, leaf)) {
      return false;
    }
    out_leaves.push_back(leaf);
  }
  return out_leaves.size() >= 2;
}

bool ValidateAndSortPartition(int64_t total_dim_size,
                              llvm::SmallVectorImpl<PadSliceLeaf>& leaves,
                              int64_t& out_concat_dim) {
  const int64_t concat_dim = leaves.front().pad_dim;
  for (const PadSliceLeaf& leaf : leaves) {
    if (leaf.pad_dim != concat_dim) {
      return false;
    }
  }
  llvm::stable_sort(leaves, [](const PadSliceLeaf& a, const PadSliceLeaf& b) {
    return a.start_offset < b.start_offset;
  });
  int64_t expected_offset = 0;
  for (const PadSliceLeaf& leaf : leaves) {
    if (leaf.start_offset != expected_offset) {
      return false;
    }
    expected_offset += leaf.slice_size;
  }
  if (expected_offset != total_dim_size) {
    return false;
  }
  out_concat_dim = concat_dim;
  return true;
}

struct FoldComplementaryZeroPadAddPattern
    : public mlir::OpRewritePattern<mlir::stablehlo::AddOp> {
  using OpRewritePattern<mlir::stablehlo::AddOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::AddOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!out_type || !out_type.hasStaticShape() || out_type.getRank() < 1) {
      return mlir::failure();
    }
    llvm::SmallVector<PadSliceLeaf, 4> leaves;
    if (!CollectAddPadLeaves(op, out_type, leaves)) {
      return mlir::failure();
    }
    int64_t concat_dim = -1;
    if (!ValidateAndSortPartition(out_type.getDimSize(leaves.front().pad_dim),
                                  leaves, concat_dim)) {
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Value, 4> concat_inputs;
    concat_inputs.reserve(leaves.size());
    for (const PadSliceLeaf& leaf : leaves) {
      concat_inputs.push_back(leaf.unpadded_input);
    }
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConcatenateOp>(
        op, out_type, concat_inputs, rewriter.getI64IntegerAttr(concat_dim));
    return mlir::success();
  }
};

struct FoldPadAddToConcatPass
    : public mlir::PassWrapper<FoldPadAddToConcatPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldPadAddToConcatPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-fold-pad-add-to-concat";
  }
  llvm::StringRef getDescription() const override {
    return "Folds sums of complementary zero-padded tensors into "
           "stablehlo.concatenate.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<FoldComplementaryZeroPadAddPattern>(ctx);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createFoldPadAddToConcatPass() {
  return std::make_unique<FoldPadAddToConcatPass>();
}

}  // namespace torch_tpu
