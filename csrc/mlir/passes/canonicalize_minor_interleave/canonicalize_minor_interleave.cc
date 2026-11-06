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

#include "csrc/mlir/passes/canonicalize_minor_interleave/canonicalize_minor_interleave.h"  // NOLINT

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // NOLINT
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"  // NOLINT
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// COST_MODEL: DotGeneral with constant 0/1 permutation matrices replaces
// strided gathers/scatters when K <= 128 (hardware 128 lane TPU vector width).
constexpr int64_t kMaxInterleaveK = 128;

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation*> dead_ops;
    func.walk([&dead_ops](mlir::Operation* op) {
      if (mlir::isOpTriviallyDead(op)) {
        dead_ops.push_back(op);
      }
    });
    for (auto it = dead_ops.rbegin(); it != dead_ops.rend(); ++it) {
      if (mlir::isOpTriviallyDead(*it)) {
        (*it)->erase();
        changed = true;
      }
    }
  }
}

std::optional<int64_t> ExtractConstantScalarInt(mlir::Value v) {
  auto const_op = v.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  auto attr = llvm::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr || !attr.getElementType().isIntOrIndex()) {
    return std::nullopt;
  }
  if (attr.isSplat()) {
    return attr.getSplatValue<llvm::APInt>().getSExtValue();
  }
  if (attr.getNumElements() == 1) {
    return (*attr.getValues<llvm::APInt>().begin()).getSExtValue();
  }
  return std::nullopt;
}

bool IsZeroSplatTensor(mlir::Value val) {
  while (val) {
    if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      val = conv.getOperand();
      continue;
    }
    if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      val = bcast.getOperand();
      continue;
    }
    if (auto res = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      val = res.getOperand();
      continue;
    }
    break;
  }
  mlir::DenseElementsAttr attr;
  if (!mlir::matchPattern(val, mlir::m_Constant(&attr)) || !attr.isSplat()) {
    return false;
  }
  const auto elem_ty = attr.getElementType();
  if (llvm::isa<mlir::FloatType>(elem_ty)) {
    return attr.getSplatValue<llvm::APFloat>().isZero();
  }
  if (llvm::isa<mlir::IntegerType>(elem_ty)) {
    return attr.getSplatValue<llvm::APInt>().isZero();
  }
  return false;
}

mlir::Value CreateInterleaveMatrix(mlir::OpBuilder& builder, mlir::Location loc,
                                   int64_t rows, int64_t cols, int64_t parity,
                                   mlir::Type elem_ty) {
  auto mat_ty = mlir::RankedTensorType::get({rows, cols}, elem_ty);
  const int64_t total = rows * cols;
  const int64_t k = (rows < cols) ? rows : cols;
  if (auto flt_ty = llvm::dyn_cast<mlir::FloatType>(elem_ty)) {
    const llvm::fltSemantics& sem = flt_ty.getFloatSemantics();
    const llvm::APFloat zero(sem, 0);
    const llvm::APFloat one(sem, 1);
    std::vector<llvm::APFloat> vals(total, zero);
    for (int64_t i = 0; i < k; ++i) {
      const int64_t idx = (rows == k) ? (i * cols + 2 * i + parity)
                                      : ((2 * i + parity) * cols + i);
      vals[idx] = one;
    }
    auto attr = mlir::DenseElementsAttr::get(mat_ty, vals);
    return mlir::stablehlo::ConstantOp::create(builder, loc, attr);
  }
  if (auto int_ty = llvm::dyn_cast<mlir::IntegerType>(elem_ty)) {
    const llvm::APInt zero(int_ty.getWidth(), 0);
    const llvm::APInt one(int_ty.getWidth(), 1);
    std::vector<llvm::APInt> vals(total, zero);
    for (int64_t i = 0; i < k; ++i) {
      const int64_t idx = (rows == k) ? (i * cols + 2 * i + parity)
                                      : ((2 * i + parity) * cols + i);
      vals[idx] = one;
    }
    auto attr = mlir::DenseElementsAttr::get(mat_ty, vals);
    return mlir::stablehlo::ConstantOp::create(builder, loc, attr);
  }
  return nullptr;
}

mlir::ArrayAttr GetHighestPrecisionAttr(mlir::OpBuilder& builder) {
  namespace shlo = mlir::stablehlo;
  auto prec = shlo::PrecisionAttr::get(
      builder.getContext(),
      shlo::Precision::HIGHEST);  // EXPLICIT_PRECISION_OK=minor_interleave
  return builder.getArrayAttr({prec, prec});
}

mlir::Value CreateDotGeneral(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::Value lhs, mlir::Value rhs) {
  auto lhs_ty = llvm::cast<mlir::RankedTensorType>(lhs.getType());
  auto rhs_ty = llvm::cast<mlir::RankedTensorType>(rhs.getType());
  const int64_t rank = lhs_ty.getRank();
  llvm::SmallVector<int64_t> out_shape(lhs_ty.getShape().drop_back());
  out_shape.push_back(rhs_ty.getDimSize(1));
  auto out_ty = mlir::RankedTensorType::get(out_shape, lhs_ty.getElementType());
  auto dnums = mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(), /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{},
      /*lhsContractingDimensions=*/{rank - 1},
      /*rhsContractingDimensions=*/{0});
  // GENERIC_LOWERING_OK=minor_interleave
  return mlir::stablehlo::DotGeneralOp::create(
      builder, loc, out_ty, lhs, rhs, dnums,
      /*precision_config=*/GetHighestPrecisionAttr(builder),
      /*algorithm=*/nullptr);
}

mlir::Value FlattenMinorUnitDim(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value val, int64_t k) {
  if (auto res = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    auto in_ty =
        llvm::dyn_cast<mlir::RankedTensorType>(res.getOperand().getType());
    if (in_ty && in_ty.hasStaticShape() && in_ty.getRank() >= 1 &&
        in_ty.getDimSize(in_ty.getRank() - 1) == k) {
      return res.getOperand();
    }
  }
  auto val_ty = llvm::cast<mlir::RankedTensorType>(val.getType());
  llvm::SmallVector<int64_t> new_shape(val_ty.getShape().drop_back());
  auto out_ty = mlir::RankedTensorType::get(new_shape, val_ty.getElementType());
  return mlir::stablehlo::ReshapeOp::create(builder, loc, out_ty, val);
}

std::optional<int64_t> MatchMinor2DusSlot(
    mlir::stablehlo::DynamicUpdateSliceOp dus_op, int64_t k) {
  auto base_ty =
      llvm::dyn_cast<mlir::RankedTensorType>(dus_op.getOperand().getType());
  auto upd_ty =
      llvm::dyn_cast<mlir::RankedTensorType>(dus_op.getUpdate().getType());
  if (!base_ty || !upd_ty || !base_ty.hasStaticShape() ||
      !upd_ty.hasStaticShape()) {
    return std::nullopt;
  }
  const int64_t r = base_ty.getRank();
  if (r < 2 || base_ty.getDimSize(r - 2) != k ||
      base_ty.getDimSize(r - 1) != 2 || upd_ty.getRank() != r ||
      upd_ty.getDimSize(r - 2) != k || upd_ty.getDimSize(r - 1) != 1) {
    return std::nullopt;
  }
  for (int64_t i = 0; i < r - 2; ++i) {
    if (upd_ty.getDimSize(i) != base_ty.getDimSize(i)) {
      return std::nullopt;
    }
    auto idx = ExtractConstantScalarInt(dus_op.getStartIndices()[i]);
    if (!idx.has_value() || *idx != 0) {
      return std::nullopt;
    }
  }
  auto idx_penult = ExtractConstantScalarInt(dus_op.getStartIndices()[r - 2]);
  auto idx_last = ExtractConstantScalarInt(dus_op.getStartIndices()[r - 1]);
  if (!idx_penult.has_value() || *idx_penult != 0 || !idx_last.has_value() ||
      (*idx_last != 0 && *idx_last != 1)) {
    return std::nullopt;
  }
  return *idx_last;
}

// -----------------------------------------------------------------------------
// Rule A: Interleave [..., K, 2] -> [..., 2K] to DotGeneral sum
// -----------------------------------------------------------------------------
bool ApplyRuleA(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::ReshapeOp> reshapes;
  func.walk(
      [&reshapes](mlir::stablehlo::ReshapeOp op) { reshapes.push_back(op); });
  bool changed = false;
  for (mlir::stablehlo::ReshapeOp res_op : reshapes) {
    if (!res_op || res_op->use_empty()) {
      continue;
    }
    auto in_ty =
        llvm::dyn_cast<mlir::RankedTensorType>(res_op.getOperand().getType());
    auto out_ty = llvm::dyn_cast<mlir::RankedTensorType>(res_op.getType());
    if (!in_ty || !out_ty || !in_ty.hasStaticShape() ||
        !out_ty.hasStaticShape()) {
      continue;
    }
    const int64_t in_rank = in_ty.getRank();
    const int64_t out_rank = out_ty.getRank();
    if (in_rank < 2 || out_rank != in_rank - 1) {
      continue;
    }
    if (in_ty.getDimSize(in_rank - 1) != 2) {
      continue;
    }
    const int64_t k = in_ty.getDimSize(in_rank - 2);
    if (k > kMaxInterleaveK || out_ty.getDimSize(out_rank - 1) != 2 * k) {
      continue;
    }

    mlir::Value val_a;
    mlir::Value val_b;
    mlir::Operation* def_op = res_op.getOperand().getDefiningOp();
    if (!def_op) {
      continue;
    }

    if (auto cat = llvm::dyn_cast<mlir::stablehlo::ConcatenateOp>(def_op)) {
      if (static_cast<int64_t>(cat.getDimension()) != in_rank - 1 ||
          cat.getInputs().size() != 2) {
        continue;
      }
      mlir::OpBuilder builder(res_op);
      val_a =
          FlattenMinorUnitDim(builder, res_op.getLoc(), cat.getInputs()[0], k);
      val_b =
          FlattenMinorUnitDim(builder, res_op.getLoc(), cat.getInputs()[1], k);
    } else if (auto dus2 =
                   llvm::dyn_cast<mlir::stablehlo::DynamicUpdateSliceOp>(
                       def_op)) {
      auto slot2 = MatchMinor2DusSlot(dus2, k);
      if (!slot2.has_value()) {
        continue;
      }
      auto dus1 = dus2.getOperand()
                      .getDefiningOp<mlir::stablehlo::DynamicUpdateSliceOp>();
      if (!dus1) {
        continue;
      }
      auto slot1 = MatchMinor2DusSlot(dus1, k);
      if (!slot1.has_value() || *slot1 == *slot2) {
        continue;
      }
      mlir::OpBuilder builder(res_op);
      mlir::Value u1 =
          FlattenMinorUnitDim(builder, res_op.getLoc(), dus1.getUpdate(), k);
      mlir::Value u2 =
          FlattenMinorUnitDim(builder, res_op.getLoc(), dus2.getUpdate(), k);
      val_a = (*slot1 == 0) ? u1 : u2;
      val_b = (*slot1 == 0) ? u2 : u1;
    } else {
      continue;
    }

    mlir::OpBuilder builder(res_op);
    mlir::Value mat_even = CreateInterleaveMatrix(
        builder, res_op.getLoc(), k, 2 * k, 0, out_ty.getElementType());
    mlir::Value mat_odd = CreateInterleaveMatrix(
        builder, res_op.getLoc(), k, 2 * k, 1, out_ty.getElementType());
    if (!mat_even || !mat_odd) {
      continue;
    }
    mlir::Value dot_even =
        CreateDotGeneral(builder, res_op.getLoc(), val_a, mat_even);
    mlir::Value dot_odd =
        CreateDotGeneral(builder, res_op.getLoc(), val_b, mat_odd);
    auto sum = mlir::stablehlo::AddOp::create(builder, res_op.getLoc(), out_ty,
                                              dot_even, dot_odd);
    res_op.replaceAllUsesWith(sum.getResult());
    changed = true;
  }
  return changed;
}

// -----------------------------------------------------------------------------
// Rule B: Stride-2 Slices -> x @ I_even^T / x @ I_odd^T
// -----------------------------------------------------------------------------
struct ParitySlices {
  llvm::SmallVector<mlir::stablehlo::SliceOp, 2> even;
  llvm::SmallVector<mlir::stablehlo::SliceOp, 2> odd;
};

bool ApplyRuleB(mlir::func::FuncOp func) {
  llvm::DenseMap<mlir::Value, ParitySlices> candidate_slices;

  func.walk([&candidate_slices](mlir::stablehlo::SliceOp slice) {
    auto in_ty =
        llvm::dyn_cast<mlir::RankedTensorType>(slice.getOperand().getType());
    auto out_ty = llvm::dyn_cast<mlir::RankedTensorType>(slice.getType());
    if (!in_ty || !out_ty || !in_ty.hasStaticShape() ||
        !out_ty.hasStaticShape()) {
      return;
    }
    const int64_t rank = in_ty.getRank();
    if (rank < 1) {
      return;
    }
    const int64_t d = in_ty.getDimSize(rank - 1);
    if (d < 2 || d % 2 != 0) {
      return;
    }
    const int64_t k = d / 2;
    if (k > kMaxInterleaveK || out_ty.getDimSize(rank - 1) != k) {
      return;
    }
    llvm::ArrayRef<int64_t> starts = slice.getStartIndices();
    llvm::ArrayRef<int64_t> limits = slice.getLimitIndices();
    llvm::ArrayRef<int64_t> strides = slice.getStrides();
    for (int64_t i = 0; i < rank - 1; ++i) {
      if (starts[i] != 0 || limits[i] != in_ty.getDimSize(i) ||
          strides[i] != 1) {
        return;
      }
    }
    if (strides[rank - 1] != 2) {
      return;
    }
    if (starts[rank - 1] == 0 &&
        (limits[rank - 1] == d || limits[rank - 1] == d - 1)) {
      candidate_slices[slice.getOperand()].even.push_back(slice);
    } else if (starts[rank - 1] == 1 && limits[rank - 1] == d) {
      candidate_slices[slice.getOperand()].odd.push_back(slice);
    }
  });

  bool changed = false;
  for (auto& [x, slices] : candidate_slices) {
    auto& even_slices = slices.even;
    auto& odd_slices = slices.odd;
    if (even_slices.empty() || odd_slices.empty()) {
      continue;
    }
    auto in_ty = llvm::cast<mlir::RankedTensorType>(x.getType());
    const int64_t rank = in_ty.getRank();
    const int64_t d = in_ty.getDimSize(rank - 1);
    const int64_t k = d / 2;
    mlir::OpBuilder builder(even_slices.front());
    mlir::Value mat_even_t = CreateInterleaveMatrix(
        builder, even_slices.front().getLoc(), d, k, 0, in_ty.getElementType());
    mlir::Value mat_odd_t = CreateInterleaveMatrix(
        builder, odd_slices.front().getLoc(), d, k, 1, in_ty.getElementType());
    if (!mat_even_t || !mat_odd_t) {
      continue;
    }
    mlir::Value dot_even =
        CreateDotGeneral(builder, even_slices.front().getLoc(), x, mat_even_t);
    mlir::Value dot_odd =
        CreateDotGeneral(builder, odd_slices.front().getLoc(), x, mat_odd_t);
    for (mlir::stablehlo::SliceOp s : even_slices) {
      s.replaceAllUsesWith(dot_even);
    }
    for (mlir::stablehlo::SliceOp s : odd_slices) {
      s.replaceAllUsesWith(dot_odd);
    }
    changed = true;
  }
  return changed;
}

// -----------------------------------------------------------------------------
// Rule C: Concat(0, A) + Concat(B, 0) and Unary(Concat(u, u))
// -----------------------------------------------------------------------------
bool ApplyRuleC(mlir::func::FuncOp func) {
  bool changed = false;
  llvm::SmallVector<mlir::stablehlo::AddOp> add_ops;
  func.walk([&add_ops](mlir::stablehlo::AddOp op) { add_ops.push_back(op); });
  for (mlir::stablehlo::AddOp add_op : add_ops) {
    if (!add_op || add_op->use_empty()) {
      continue;
    }
    auto cat0 = add_op.getLhs().getDefiningOp<mlir::stablehlo::ConcatenateOp>();
    auto cat1 = add_op.getRhs().getDefiningOp<mlir::stablehlo::ConcatenateOp>();
    if (!cat0 || !cat1 || cat0.getInputs().size() != 2 ||
        cat1.getInputs().size() != 2 ||
        cat0.getDimension() != cat1.getDimension()) {
      continue;
    }
    mlir::Value a0 = cat0.getInputs()[0];
    mlir::Value a1 = cat0.getInputs()[1];
    mlir::Value b0 = cat1.getInputs()[0];
    mlir::Value b1 = cat1.getInputs()[1];
    if (a0.getType() != b0.getType() || a1.getType() != b1.getType()) {
      continue;
    }
    mlir::Value new_lo;
    mlir::Value new_hi;
    if (IsZeroSplatTensor(a0) && IsZeroSplatTensor(b1)) {
      new_lo = b0;
      new_hi = a1;
    } else if (IsZeroSplatTensor(a1) && IsZeroSplatTensor(b0)) {
      new_lo = a0;
      new_hi = b1;
    } else {
      continue;
    }
    mlir::OpBuilder builder(add_op);
    auto merged = mlir::stablehlo::ConcatenateOp::create(
        builder, add_op.getLoc(), add_op.getType(),
        mlir::ValueRange{new_lo, new_hi}, cat0.getDimension());
    add_op.replaceAllUsesWith(merged.getResult());
    changed = true;
  }

  llvm::SmallVector<mlir::Operation*> ops;
  func.walk([&ops](mlir::Operation* op) { ops.push_back(op); });
  for (mlir::Operation* op : ops) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      continue;
    }
    if (!mlir::isa<mlir::stablehlo::SineOp, mlir::stablehlo::CosineOp,
                   mlir::stablehlo::NegOp, mlir::stablehlo::ConvertOp,
                   mlir::stablehlo::AbsOp, mlir::stablehlo::ExpOp,
                   mlir::stablehlo::SqrtOp, mlir::stablehlo::RsqrtOp,
                   mlir::stablehlo::TanhOp, mlir::stablehlo::LogOp>(op)) {
      continue;
    }
    auto cat =
        op->getOperand(0).getDefiningOp<mlir::stablehlo::ConcatenateOp>();
    if (!cat || cat.getInputs().size() != 2 ||
        cat.getInputs()[0] != cat.getInputs()[1]) {
      continue;
    }
    mlir::Value u = cat.getInputs()[0];
    auto u_ty = llvm::dyn_cast<mlir::RankedTensorType>(u.getType());
    auto out_ty =
        llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!u_ty || !out_ty || !u_ty.hasStaticShape()) {
      continue;
    }
    auto u_out_ty =
        mlir::RankedTensorType::get(u_ty.getShape(), out_ty.getElementType());
    mlir::OpBuilder builder(op);
    mlir::IRMapping mapping;
    mapping.map(op->getOperand(0), u);
    mlir::Operation* u_op = builder.clone(*op, mapping);
    u_op->getResult(0).setType(u_out_ty);
    auto new_cat = mlir::stablehlo::ConcatenateOp::create(
        builder, op->getLoc(), out_ty,
        mlir::ValueRange{u_op->getResult(0), u_op->getResult(0)},
        cat.getDimension());
    op->replaceAllUsesWith(new_cat.getOperation());
    changed = true;
  }
  return changed;
}

class CanonicalizeMinorInterleavePass
    : public mlir::PassWrapper<CanonicalizeMinorInterleavePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeMinorInterleavePass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-canonicalize-minor-interleave";
  }

  llvm::StringRef getDescription() const override {
    return "Canonicalizes minor-dimension interleaving patterns";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    bool changed = false;
    for (int iter = 0; iter < 4; ++iter) {
      bool iter_changed = false;
      iter_changed |= ApplyRuleA(func);
      iter_changed |= ApplyRuleB(func);
      iter_changed |= ApplyRuleC(func);
      if (!iter_changed) {
        break;
      }
      changed = true;
    }
    if (changed) {
      EraseTriviallyDeadOps(func);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createCanonicalizeMinorInterleavePass() {
  return std::make_unique<CanonicalizeMinorInterleavePass>();
}

}  // namespace torch_tpu
