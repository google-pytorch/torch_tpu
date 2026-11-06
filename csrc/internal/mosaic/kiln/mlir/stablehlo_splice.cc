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

#include "csrc/internal/mosaic/kiln/mlir/stablehlo_splice.h"

#include <cstdint>
#include <string_view>

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/tile.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace torch_tpu::kiln {
namespace {

using TensorValue = mlir::TypedValue<mlir::RankedTensorType>;
using VectorValue = mlir::TypedValue<mlir::VectorType>;

// Kiln tiles (and Mosaic's native IR) are vectors; the spliced StableHLO ops
// take and produce tensors. These two bridges convert between them.
TensorValue VectorToTensor(mlir::ImplicitLocOpBuilder& b, VectorValue vec) {
  mlir::VectorType vty = vec.getType();
  mlir::Value empty =
      mlir::tensor::EmptyOp::create(b, vty.getShape(), vty.getElementType());

  mlir::Value c0 = mlir::arith::ConstantIndexOp::create(b, 0);
  llvm::SmallVector<mlir::Value> indices(vty.getRank(), c0);
  llvm::SmallVector<bool> in_bounds(vty.getRank(), true);

  return mlir::vector::TransferWriteOp::create(b, vec, empty, indices,
                                               llvm::ArrayRef<bool>(in_bounds))
      .getResult();
}

VectorValue TensorToVector(mlir::ImplicitLocOpBuilder& b, TensorValue tensor) {
  mlir::RankedTensorType tty = tensor.getType();
  auto vty = mlir::VectorType::get(tty.getShape(), tty.getElementType());

  mlir::Value c0 = mlir::arith::ConstantIndexOp::create(b, 0);
  llvm::SmallVector<mlir::Value> indices(tty.getRank(), c0);
  llvm::SmallVector<bool> in_bounds(tty.getRank(), true);
  // The whole tensor is read in bounds, so the padding is never used.
  mlir::Value padding = mlir::arith::ConstantOp::create(
      b, tty.getElementType(), b.getZeroAttr(tty.getElementType()));

  return mlir::vector::TransferReadOp::create(b, vty, tensor, indices, padding,
                                              llvm::ArrayRef<bool>(in_bounds))
      .getResult();
}

// Demotes every f64 result (and constant) of `op` to f32. Operands are already
// demoted because producers are visited first.
void DemoteF64(mlir::Operation* op) {
  auto f32 = mlir::Float32Type::get(op->getContext());
  for (mlir::OpResult result : op->getResults()) {
    auto shaped = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (shaped && shaped.getElementType().isF64()) {
      result.setType(mlir::RankedTensorType::get(shaped.getShape(), f32));
    }
  }
  if (auto constant = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
    auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue());
    if (attr && attr.getElementType().isF64()) {
      constant.setValueAttr(attr.mapValues(f32, [](const llvm::APFloat& v) {
        const float f = static_cast<float>(v.convertToDouble());
        return llvm::APInt(32, absl::bit_cast<uint32_t>(f));
      }));
    }
  }
}

// Clones `op` at the insertion point. Converts that become identities after
// f64 demotion are dropped.
void CloneOp(mlir::ImplicitLocOpBuilder& b, mlir::Operation& op,
             mlir::IRMapping& mapping) {
  if (auto convert = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
    mlir::Value in = mapping.lookup(convert.getOperand());
    mlir::Type out_elt =
        mlir::cast<mlir::RankedTensorType>(convert.getType()).getElementType();
    if (out_elt.isF64()) out_elt = b.getF32Type();
    if (out_elt ==
        mlir::cast<mlir::RankedTensorType>(in.getType()).getElementType()) {
      mapping.map(convert.getResult(), in);
      return;
    }
  }
  DemoteF64(b.clone(op, mapping));
}

// Kiln dtype of a spliced result; f64 results are demoted to f32.
absl::StatusOr<DType> DTypeOf(mlir::Type element_type) {
  if (element_type.isF32() || element_type.isF64()) return f32;
  if (element_type.isBF16()) return bf16;
  if (element_type.isF16()) return f16;
  if (element_type.isInteger(1)) return bool_;
  if (element_type.isInteger(8)) return i8;
  if (element_type.isInteger(16)) return i16;
  if (element_type.isInteger(32)) return i32;
  return TT_ERROR(error::kInvalidArgument)
         << "unsupported StableHLO result element type";
}

}  // namespace

absl::StatusOr<Tile> SpliceStablehlo(std::string_view mlir_text,
                                     llvm::ArrayRef<Tile> args) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(mlir_text.data(), mlir_text.size()), b.getContext());
  TT_RET_CHECK(module, error::kInvalidArgument)
      << "failed to parse the StableHLO module";
  // Add CHLO pass (needed for some ops like integer floor division)
  mlir::PassManager pm(b.getContext());
  pm.nest<mlir::func::FuncOp>().addPass(
      mlir::stablehlo::createChloLegalizeToStablehloPass());
  pm.addPass(mlir::createCanonicalizerPass());
  const bool legalized = llvm::succeeded(pm.run(*module));
  TT_RET_CHECK(legalized, error::kInternal)  // ERROR_COV_INFEASIBLE=not by mods
      << "failed to legalize the module to StableHLO";
  auto main = module->lookupSymbol<mlir::func::FuncOp>("main");
  TT_RET_CHECK(main, error::kInvalidArgument) << "module has no @main";
  TT_RET_CHECK(main.getNumArguments() == args.size(), error::kInvalidArgument)
      << "@main takes " << main.getNumArguments() << " arguments, got "
      << args.size();
  TT_RET_CHECK(main.getNumResults() == 1, error::kInvalidArgument)
      << "@main must return exactly one tensor";
  TT_RET_CHECK(main.getBody().hasOneBlock(), error::kInvalidArgument)
      << "@main must have exactly one block";
  auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(main.front().getTerminator());
  TT_RET_CHECK(ret, error::kInvalidArgument) << "@main must end in func.return";
  TT_RET_CHECK(!main.front().without_terminator().empty(),
               error::kInvalidArgument)
      << "@main computes nothing; skip the splice instead";

  // Validate against the signature before emitting anything, so a failed
  // splice leaves the kernel untouched.
  llvm::SmallVector<VectorValue> vectors;
  for (int64_t i = 0; i < args.size(); ++i) {
    vectors.push_back(mlir::cast<VectorValue>(internal::unwrap(args[i])));
    mlir::VectorType vty = vectors.back().getType();
    TT_RET_CHECK(
        mlir::RankedTensorType::get(vty.getShape(), vty.getElementType()) ==
            main.getArgument(i).getType(),
        error::kInvalidArgument)
        << "@main argument " << i << " does not match the tile's type";
  }
  auto result_type =
      mlir::dyn_cast<mlir::RankedTensorType>(main.getResultTypes().front());
  TT_RET_CHECK(result_type, error::kInvalidArgument)
      << "@main must return a ranked tensor";
  TT_ASSIGN_OR_RETURN(const DType dtype, DTypeOf(result_type.getElementType()));

  mlir::IRMapping mapping;
  for (int64_t i = 0; i < args.size(); ++i) {
    mapping.map(main.getArgument(i), VectorToTensor(b, vectors[i]));
  }
  for (mlir::Operation& op : main.front().without_terminator()) {
    CloneOp(b, op, mapping);
  }
  auto result = mlir::cast<TensorValue>(mapping.lookup(ret.getOperand(0)));
  return internal::wrap<Tile>(TensorToVector(b, result), dtype);
}

}  // namespace torch_tpu::kiln
