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

#include "csrc/mlir/passes/demote_f64_narrowed_ops.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
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

bool IsF64TensorType(mlir::Type type) {
  auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
  return ranked && ranked.hasStaticShape() && ranked.getElementType().isF64();
}

// GENERIC: semantic floating-point bitwidth check (<=32-bit float types).
bool IsAtMost32BitFloatElementType(mlir::Type elem_type) {
  return elem_type.isF32() || elem_type.isBF16() || elem_type.isF16();
}

// GENERIC: semantic bitwidth check for lossless/narrow source types (<=32-bit
// float or <=24-bit integer/boolean that converts exactly to IEEE-754 float32).
bool IsNarrowSourceElementType(mlir::Type elem_type) {
  if (IsAtMost32BitFloatElementType(elem_type)) {
    return true;
  }
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(elem_type)) {
    // GENERIC: capability IEEE-754 binary32 has 24 significand bits, so
    // <=16-bit integers and booleans are represented with zero rounding error
    // in f32.
    return int_type.getWidth() <= 16;
  }
  return false;
}

bool IsSupportedF64InteriorOp(mlir::Operation* op) {
  if (!op || op->getNumResults() != 1 ||
      !IsF64TensorType(op->getResult(0).getType())) {
    return false;
  }
  return mlir::isa<
      mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
      mlir::stablehlo::MulOp, mlir::stablehlo::DivOp, mlir::stablehlo::PowOp,
      mlir::stablehlo::MaxOp, mlir::stablehlo::MinOp, mlir::stablehlo::NegOp,
      mlir::stablehlo::AbsOp, mlir::stablehlo::SqrtOp, mlir::stablehlo::RsqrtOp,
      mlir::stablehlo::BroadcastInDimOp, mlir::stablehlo::ReshapeOp,
      mlir::stablehlo::TransposeOp, mlir::stablehlo::SliceOp>(op);
}

bool IsRepresentableInNormalF32(double d) {
  if (!std::isfinite(d)) {
    return false;
  }
  if (d == 0.0) {
    return true;
  }
  const double abs_d = std::abs(d);
  return abs_d >= static_cast<double>(std::numeric_limits<float>::min()) &&
         abs_d <= static_cast<double>(std::numeric_limits<float>::max());
}

bool CanNarrowF64ConstantToF32(mlir::stablehlo::ConstantOp const_op) {
  if (!IsF64TensorType(const_op.getType())) {
    return false;
  }
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr) {
    return false;
  }
  if (attr.isSplat()) {
    llvm::APFloat val = attr.getSplatValue<llvm::APFloat>();
    return IsRepresentableInNormalF32(val.convertToDouble());
  }
  // GENERIC: cost-model bound constant element scan to small constants.
  if (attr.getNumElements() > 64) {
    return false;
  }
  for (llvm::APFloat val : attr.getValues<llvm::APFloat>()) {
    if (!IsRepresentableInNormalF32(val.convertToDouble())) {
      return false;
    }
  }
  return true;
}

bool IsNarrowToF32Convert(mlir::Operation* op) {
  auto conv = mlir::dyn_cast_or_null<mlir::stablehlo::ConvertOp>(op);
  if (!conv) {
    return false;
  }
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
  auto out_type =
      mlir::dyn_cast<mlir::RankedTensorType>(conv.getResult().getType());
  return in_type && out_type && in_type.getElementType().isF64() &&
         IsAtMost32BitFloatElementType(out_type.getElementType());
}

bool IsUpcastFromNarrowConvert(mlir::Operation* op) {
  auto conv = mlir::dyn_cast_or_null<mlir::stablehlo::ConvertOp>(op);
  if (!conv) {
    return false;
  }
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
  auto out_type =
      mlir::dyn_cast<mlir::RankedTensorType>(conv.getResult().getType());
  return in_type && out_type && out_type.hasStaticShape() &&
         out_type.getElementType().isF64() &&
         IsNarrowSourceElementType(in_type.getElementType());
}

mlir::RankedTensorType ToF32TensorType(mlir::Type type) {
  auto ranked = mlir::cast<mlir::RankedTensorType>(type);
  return mlir::RankedTensorType::get(ranked.getShape(),
                                     mlir::Float32Type::get(type.getContext()));
}

mlir::Value MaterializeF32Constant(mlir::OpBuilder& builder,
                                   mlir::stablehlo::ConstantOp const_op) {
  auto f32_type = ToF32TensorType(const_op.getType());
  auto attr = mlir::cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (attr.isSplat()) {
    double d = attr.getSplatValue<llvm::APFloat>().convertToDouble();
    llvm::APFloat f32_val(static_cast<float>(d));
    auto splat_attr = mlir::DenseElementsAttr::get(f32_type, f32_val);
    return builder.create<mlir::stablehlo::ConstantOp>(const_op.getLoc(),
                                                       f32_type, splat_attr);
  }
  llvm::SmallVector<float> vals;
  vals.reserve(attr.getNumElements());
  for (llvm::APFloat val : attr.getValues<llvm::APFloat>()) {
    vals.push_back(static_cast<float>(val.convertToDouble()));
  }
  auto dense_attr =
      mlir::DenseElementsAttr::get(f32_type, llvm::ArrayRef<float>(vals));
  return builder.create<mlir::stablehlo::ConstantOp>(const_op.getLoc(),
                                                     f32_type, dense_attr);
}

mlir::Value CloneInteriorOpAsF32(mlir::OpBuilder& builder, mlir::Operation* op,
                                 llvm::ArrayRef<mlir::Value> f32_operands) {
  mlir::Location loc = op->getLoc();
  mlir::RankedTensorType out_type = ToF32TensorType(op->getResult(0).getType());

  if (mlir::isa<mlir::stablehlo::AddOp>(op)) {
    return builder.create<mlir::stablehlo::AddOp>(
        loc, out_type, f32_operands[0], f32_operands[1]);
  }
  if (mlir::isa<mlir::stablehlo::SubtractOp>(op)) {
    return builder.create<mlir::stablehlo::SubtractOp>(
        loc, out_type, f32_operands[0], f32_operands[1]);
  }
  if (mlir::isa<mlir::stablehlo::MulOp>(op)) {
    return builder.create<mlir::stablehlo::MulOp>(
        loc, out_type, f32_operands[0], f32_operands[1]);
  }
  if (mlir::isa<mlir::stablehlo::DivOp>(op)) {
    return builder.create<mlir::stablehlo::DivOp>(
        loc, out_type, f32_operands[0], f32_operands[1]);
  }
  if (mlir::isa<mlir::stablehlo::PowOp>(op)) {
    return builder.create<mlir::stablehlo::PowOp>(
        loc, out_type, f32_operands[0], f32_operands[1]);
  }
  if (mlir::isa<mlir::stablehlo::MaxOp>(op)) {
    return builder.create<mlir::stablehlo::MaxOp>(
        loc, out_type, f32_operands[0], f32_operands[1]);
  }
  if (mlir::isa<mlir::stablehlo::MinOp>(op)) {
    return builder.create<mlir::stablehlo::MinOp>(
        loc, out_type, f32_operands[0], f32_operands[1]);
  }
  if (mlir::isa<mlir::stablehlo::NegOp>(op)) {
    return builder.create<mlir::stablehlo::NegOp>(loc, out_type,
                                                  f32_operands[0]);
  }
  if (mlir::isa<mlir::stablehlo::AbsOp>(op)) {
    return builder.create<mlir::stablehlo::AbsOp>(loc, out_type,
                                                  f32_operands[0]);
  }
  if (mlir::isa<mlir::stablehlo::SqrtOp>(op)) {
    return builder.create<mlir::stablehlo::SqrtOp>(loc, out_type,
                                                   f32_operands[0]);
  }
  if (mlir::isa<mlir::stablehlo::RsqrtOp>(op)) {
    return builder.create<mlir::stablehlo::RsqrtOp>(loc, out_type,
                                                    f32_operands[0]);
  }
  if (auto bcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
    return builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, out_type, f32_operands[0], bcast.getBroadcastDimensionsAttr());
  }
  if (mlir::isa<mlir::stablehlo::ReshapeOp>(op)) {
    return builder.create<mlir::stablehlo::ReshapeOp>(loc, out_type,
                                                      f32_operands[0]);
  }
  if (auto trans = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op)) {
    return builder.create<mlir::stablehlo::TransposeOp>(
        loc, out_type, f32_operands[0], trans.getPermutationAttr());
  }
  if (auto slice = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op)) {
    return builder.create<mlir::stablehlo::SliceOp>(
        loc, out_type, f32_operands[0], slice.getStartIndicesAttr(),
        slice.getLimitIndicesAttr(), slice.getStridesAttr());
  }
  return nullptr;
}

void DemoteNarrowedF64Chains(mlir::func::FuncOp func) {
  llvm::DenseSet<mlir::Operation*> visited_interior;
  llvm::SmallVector<mlir::Operation*> all_ops;
  func.walk([&](mlir::Operation* op) {
    if (IsSupportedF64InteriorOp(op)) {
      all_ops.push_back(op);
    }
  });

  for (mlir::Operation* seed_op : all_ops) {
    if (!seed_op || seed_op->getParentOp() == nullptr ||
        visited_interior.contains(seed_op)) {
      continue;
    }

    // Collect the connected component of f64 interior ops reachable from
    // `seed_op` (both upward through f64 operands and downward through f64
    // users), and verify that:
    // 1. Every leaf operand entering the component is either an upcast
    //    `ConvertOp` from a narrow type or a representable `f64` `ConstantOp`.
    // 2. Every external consumer leaving the component is a narrowing
    //    `ConvertOp` to `f32`/`bf16`/`f16`.
    llvm::DenseSet<mlir::Operation*> component_interior;
    llvm::SmallVector<mlir::Operation*> worklist;
    worklist.push_back(seed_op);
    component_interior.insert(seed_op);

    bool valid = true;
    while (!worklist.empty() && valid) {
      mlir::Operation* curr = worklist.pop_back_val();

      // Check operands (upward).
      for (mlir::Value operand : curr->getOperands()) {
        mlir::Operation* def_op = operand.getDefiningOp();
        if (!def_op) {
          valid = false;
          break;
        }
        if (IsUpcastFromNarrowConvert(def_op)) {
          continue;
        }
        if (auto const_op =
                mlir::dyn_cast<mlir::stablehlo::ConstantOp>(def_op)) {
          if (CanNarrowF64ConstantToF32(const_op)) {
            continue;
          }
          valid = false;
          break;
        }
        if (IsSupportedF64InteriorOp(def_op)) {
          if (component_interior.insert(def_op).second) {
            worklist.push_back(def_op);
          }
          continue;
        }
        valid = false;
        break;
      }
      if (!valid) {
        break;
      }

      // Check users (downward).
      mlir::Value res = curr->getResult(0);
      if (res.use_empty()) {
        valid = false;
        break;
      }
      for (mlir::Operation* user : res.getUsers()) {
        if (IsNarrowToF32Convert(user)) {
          continue;
        }
        if (IsSupportedF64InteriorOp(user)) {
          if (component_interior.insert(user).second) {
            worklist.push_back(user);
          }
          continue;
        }
        valid = false;
        break;
      }
    }

    for (mlir::Operation* op : component_interior) {
      visited_interior.insert(op);
    }
    if (!valid) {
      continue;
    }

    // Process component ops in topological order (by walking the function in
    // forward order and filtering to `component_interior`).
    llvm::DenseMap<mlir::Value, mlir::Value> f32_map;
    auto get_or_materialize_f32 = [&](mlir::Value v) -> mlir::Value {
      if (auto it = f32_map.find(v); it != f32_map.end()) {
        return it->second;
      }
      mlir::Operation* def_op = v.getDefiningOp();
      mlir::OpBuilder builder(def_op);
      if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(def_op)) {
        mlir::Value src = conv.getOperand();
        auto src_type = mlir::cast<mlir::RankedTensorType>(src.getType());
        mlir::Value f32_val = src;
        if (!src_type.getElementType().isF32()) {
          f32_val = builder.create<mlir::stablehlo::ConvertOp>(
              conv.getLoc(), ToF32TensorType(conv.getType()), src);
        }
        f32_map[v] = f32_val;
        return f32_val;
      }
      auto const_op = mlir::cast<mlir::stablehlo::ConstantOp>(def_op);
      mlir::Value f32_val = MaterializeF32Constant(builder, const_op);
      f32_map[v] = f32_val;
      return f32_val;
    };

    llvm::SmallVector<mlir::Operation*> ordered_interior;
    ordered_interior.reserve(component_interior.size());
    func.walk([&](mlir::Operation* op) {
      if (component_interior.contains(op)) {
        ordered_interior.push_back(op);
      }
    });

    for (mlir::Operation* op : ordered_interior) {
      llvm::SmallVector<mlir::Value, 2> f32_operands;
      f32_operands.reserve(op->getNumOperands());
      for (mlir::Value operand : op->getOperands()) {
        f32_operands.push_back(get_or_materialize_f32(operand));
      }
      mlir::OpBuilder builder(op);
      mlir::Value f32_res = CloneInteriorOpAsF32(builder, op, f32_operands);
      f32_map[op->getResult(0)] = f32_res;
    }

    // Rewrite all terminal narrowing ConvertOps on the boundary of this
    // component.
    llvm::SmallVector<mlir::stablehlo::ConvertOp> terminal_converts;
    for (mlir::Operation* op : ordered_interior) {
      for (mlir::Operation* user : op->getResult(0).getUsers()) {
        if (IsNarrowToF32Convert(user)) {
          terminal_converts.push_back(
              mlir::cast<mlir::stablehlo::ConvertOp>(user));
        }
      }
    }
    for (mlir::stablehlo::ConvertOp conv : terminal_converts) {
      if (!conv || conv->use_empty()) {
        continue;
      }
      mlir::Value f32_val = f32_map.lookup(conv.getOperand());
      if (!f32_val) {
        continue;
      }
      auto dst_type = mlir::cast<mlir::RankedTensorType>(conv.getType());
      if (dst_type.getElementType().isF32()) {
        conv.replaceAllUsesWith(f32_val);
      } else {
        mlir::OpBuilder builder(conv);
        auto new_conv = builder.create<mlir::stablehlo::ConvertOp>(
            conv.getLoc(), dst_type, f32_val);
        conv.replaceAllUsesWith(new_conv.getResult());
      }
    }
  }
}

void SimplifyHalfExponentPowOps(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::PowOp> pow_ops;
  func.walk([&](mlir::stablehlo::PowOp op) { pow_ops.push_back(op); });

  for (mlir::stablehlo::PowOp pow_op : pow_ops) {
    if (!pow_op || pow_op->use_empty()) {
      continue;
    }
    auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(pow_op.getType());
    if (!out_type ||
        !IsAtMost32BitFloatElementType(out_type.getElementType())) {
      continue;
    }
    mlir::Value rhs = pow_op.getRhs();
    if (auto bcast = rhs.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      rhs = bcast.getOperand();
    }
    auto const_op = rhs.getDefiningOp<mlir::stablehlo::ConstantOp>();
    if (!const_op) {
      continue;
    }
    auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    if (!attr || !attr.isSplat() ||
        !mlir::isa<mlir::FloatType>(attr.getElementType())) {
      continue;
    }
    double exp_val = attr.getSplatValue<llvm::APFloat>().convertToDouble();
    mlir::OpBuilder builder(pow_op);
    if (exp_val == 0.5) {
      auto sqrt_op = builder.create<mlir::stablehlo::SqrtOp>(
          pow_op.getLoc(), out_type, pow_op.getLhs());
      pow_op.replaceAllUsesWith(sqrt_op.getResult());
    } else if (exp_val == -0.5) {
      auto rsqrt_op = builder.create<mlir::stablehlo::RsqrtOp>(
          pow_op.getLoc(), out_type, pow_op.getLhs());
      pow_op.replaceAllUsesWith(rsqrt_op.getResult());
    }
  }
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::Operation*> worklist;
  func.walk<mlir::WalkOrder::PostOrder>([&](mlir::Operation* op) {
    if (op != func.getOperation() && op->use_empty() &&
        mlir::isOpTriviallyDead(op)) {
      worklist.push_back(op);
    }
  });
  while (!worklist.empty()) {
    mlir::Operation* op = worklist.pop_back_val();
    if (!op->use_empty() || !mlir::isOpTriviallyDead(op)) {
      continue;
    }
    llvm::SmallVector<mlir::Value> operands(op->getOperands().begin(),
                                            op->getOperands().end());
    op->erase();
    for (mlir::Value operand : operands) {
      if (operand && operand.use_empty()) {
        if (mlir::Operation* def_op = operand.getDefiningOp()) {
          if (mlir::isOpTriviallyDead(def_op)) {
            worklist.push_back(def_op);
          }
        }
      }
    }
  }
}

struct DemoteF64NarrowedOpsPass
    : public mlir::PassWrapper<DemoteF64NarrowedOpsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DemoteF64NarrowedOpsPass)

  llvm::StringRef getArgument() const override {
    return "demote-f64-narrowed-ops";
  }
  llvm::StringRef getDescription() const override {
    return "Demotes f64 elementwise arithmetic chains back to f32 when fed by "
           "narrow types and consumed by narrowing conversions, and simplifies "
           "pow(x, +/-0.5) to sqrt/rsqrt.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    DemoteNarrowedF64Chains(func);
    SimplifyHalfExponentPowOps(func);
    EraseTriviallyDeadOps(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createDemoteF64NarrowedOpsPass() {
  return std::make_unique<DemoteF64NarrowedOpsPass>();
}

}  // namespace torch_tpu
