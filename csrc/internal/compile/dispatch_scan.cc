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

#include "csrc/internal/compile/dispatch_scan.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/ArrayRef.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/ops/scan_builder.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

// Prints `module` as MLIR text, keeping op locations so that re-parsing the
// text doesn't lose them. XLA derives HLO instruction names, and thus the op
// names shown in XProf, from those locations. Without them, e.g. every Pallas
// kernel in a scan body would show up under XLA's fallback name `name.N`.
std::string PrintModuleWithLocations(mlir::ModuleOp module) {
  std::string printed;
  llvm::raw_string_ostream os(printed);
  module.print(os, mlir::OpPrintingFlags().enableDebugInfo());
  return printed;
}

// Prints `module` as MLIR text without locations, for use as a cache key.
// Bodies that differ only in their source locations thus share a cache entry
// (and the op names of whichever body was compiled first).
std::string PrintModuleForCacheKey(mlir::ModuleOp module) {
  std::string printed;
  llvm::raw_string_ostream os(printed);
  module.print(os);
  return printed;
}

std::vector<DeviceBufferRef> GetInputBuffers(
    const std::vector<at::Tensor>& inits,
    const std::vector<at::Tensor>& inputs) {
  std::vector<DeviceBufferRef> input_buffers;
  input_buffers.reserve(inits.size() + inputs.size());

  auto get_buffer = [](const at::Tensor& t) -> DeviceBufferRef {
    TT_ASSIGN_OR_THROW(const DeviceBufferRef ref, GetBuffer(t));
    return ref;
  };

  absl::c_transform(inits, std::back_inserter(input_buffers), get_buffer);
  absl::c_transform(inputs, std::back_inserter(input_buffers), get_buffer);

  return input_buffers;
}

std::vector<Shape> GetOutputShapes(
    const std::vector<at::Tensor>& inits,
    const std::vector<at::Tensor>& dummy_outputs) {
  std::vector<Shape> output_shapes;
  output_shapes.reserve(inits.size() + dummy_outputs.size());

  auto get_init_shape = [](const at::Tensor& t) -> Shape {
    TT_ASSIGN_OR_THROW(const DeviceBufferRef ref, GetBuffer(t));
    return ref.shape();
  };

  auto get_outputs_shape = [](const at::Tensor& t) -> Shape {
    TT_ASSIGN_OR_THROW(const auto dtype,
                       ConvertTo<mlir::ElementType>(t.scalar_type()));
    return Shape(CopyIntVector(t.sizes()), dtype);
  };

  absl::c_transform(inits, std::back_inserter(output_shapes), get_init_shape);
  absl::c_transform(dummy_outputs, std::back_inserter(output_shapes),
                    get_outputs_shape);

  return output_shapes;
}

llvm::SmallVector<mlir::MlirOp> InitZeroOutputAccumulators(
    mlir::MlirBuilder& builder, mlir::func::FuncOp main_func,
    const int64_t num_carries, const int64_t seq_len) {
  const mlir::TypeRange output_types =
      main_func.getResultTypes().drop_front(num_carries);
  llvm::SmallVector<mlir::MlirOp> output_inits;
  output_inits.reserve(output_types.size());

  auto create_init = [&builder, seq_len](const mlir::Type t) -> mlir::MlirOp {
    const mlir::RankedTensorType output_type =
        llvm::cast<mlir::RankedTensorType>(t);
    llvm::SmallVector<int64_t> outputs_shape = {seq_len};
    llvm::append_range(outputs_shape, output_type.getShape());
    return MakeConstant(builder, 0, output_type.getElementType(),
                        outputs_shape);
  };

  absl::c_transform(output_types, std::back_inserter(output_inits),
                    create_init);
  return output_inits;
}

// Copies @main's sibling helper functions into the module the scan is built
// into.
//
// A step function may call private helpers declared beside @main.
// `BuildLoopBody` clones only @main's own operations into the while region, so
// the cloned call would reference a symbol that does not exist in the
// destination module.
//
// Each helper is renamed before any is cloned. renameToUnique appends a
// counter -- @foo becomes @foo_0 -- taking the first suffix free in both
// modules, and updates the calls to it. The order matters: if @outer calls
// @inner, cloning @outer early copies `call @inner` across, then @inner
// becomes @inner_0 and the copy calls a name that is gone.
absl::Status ImportStepFunctionCallees(mlir::MlirBuilder& builder,
                                       mlir::ModuleOp body_module,
                                       mlir::func::FuncOp main_func) {
  llvm::SmallVector<mlir::func::FuncOp> helpers;
  for (mlir::func::FuncOp func : body_module.getOps<mlir::func::FuncOp>()) {
    if (func != main_func) helpers.push_back(func);
  }
  if (helpers.empty()) return absl::OkStatus();

  // Both modules share an MLIRContext: the caller parses body_module into the
  // builder's own context, which is what makes the clone below legal.
  mlir::ModuleOp dst_module = GetModuleOp(builder);

  mlir::SymbolTableCollection symbol_tables;
  mlir::SymbolTable& dst_st = symbol_tables.getSymbolTable(dst_module);
  mlir::SymbolTable& src_st = symbol_tables.getSymbolTable(body_module);

  for (mlir::func::FuncOp func : helpers) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=a helper's only uses are func.calls
                   // in this module, which renameToUnique always rewrites.
        mlir::succeeded(src_st.renameToUnique(func, &dst_st)), error::kInternal)
        << "failed to rename step function callee " << func.getName().str();
  }

  mlir::IRRewriter rewriter(dst_module.getContext());
  rewriter.setInsertionPointToStart(dst_module.getBody());
  for (mlir::func::FuncOp func : helpers) {
    auto cloned = mlir::cast<mlir::func::FuncOp>(rewriter.clone(*func));
    // The copy is a helper of this module, not one of its entry points.
    cloned.setVisibility(mlir::SymbolTable::Visibility::Private);
  }
  return absl::OkStatus();
}

// Populates the StableHLO while-loop body by cloning the pre-lowered step
// function (main_func) operations. It maps the formal step arguments to the
// active loop carries and input slices, and returns the updated loop carries
// and iteration outputs (from the loop's terminator).
absl::StatusOr<ScanBodyResults> BuildLoopBody(mlir::OpBuilder& op_builder,
                                              mlir::func::FuncOp main_func,
                                              const int64_t num_carries,
                                              const int64_t total_inputs,
                                              mlir::ValueRange slices,
                                              mlir::ValueRange carries) {
  mlir::IRMapping mapper;
  auto map_args = [&mapper](const mlir::ValueRange args,
                            const mlir::ValueRange values) {
    for (const auto [arg, val] : llvm::zip(args, values)) {
      mapper.map(arg, val);
    }
  };

  const mlir::Block::BlockArgListType func_args = main_func.getArguments();
  map_args(func_args.take_front(num_carries), carries);
  map_args(func_args.slice(num_carries, total_inputs), slices);

  // Clone all operations except the terminator. We must skip the function's
  // return terminator because loop body regions in StableHLO require a loop-
  // specific yield terminator (stablehlo.return) instead of a function return.
  // The return values are instead manually extracted and mapped below.
  mlir::Block& body_block = main_func.front();
  absl::c_for_each(
      llvm::make_range(body_block.begin(), std::prev(body_block.end())),
      [&op_builder, &mapper](mlir::Operation& op) {
        op_builder.clone(op, mapper);
      });

  auto lookup_values = [&mapper](const mlir::ValueRange range) {
    llvm::SmallVector<mlir::Value> result;
    result.reserve(range.size());
    absl::c_transform(
        range, std::back_inserter(result),
        [&mapper](const mlir::Value v) { return mapper.lookup(v); });
    return result;
  };

  mlir::Operation* const terminator = body_block.getTerminator();
  const mlir::OperandRange operands = terminator->getOperands();
  return ScanBodyResults{lookup_values(operands.take_front(num_carries)),
                         lookup_values(operands.drop_front(num_carries))};
}

// Populates the associative scan body. The combine function maps
// `(*left_elems, *right_elems, *additional_inputs)` to `(*combined_elems)`, so
// with the running prefix bound to `left_elems` it has exactly the shape of a
// `scan` step function with no per-step outputs. Each combined value is both
// the next prefix carry and the scan output for this step.
absl::StatusOr<ScanBodyResults> BuildAssociativeLoopBody(
    mlir::OpBuilder& op_builder, mlir::func::FuncOp main_func,
    const int64_t num_carries, const int64_t total_inputs,
    mlir::ValueRange slices, mlir::ValueRange carries) {
  TT_ASSIGN_OR_RETURN(ScanBodyResults results,
                      BuildLoopBody(op_builder, main_func, num_carries,
                                    total_inputs, slices, carries));
  results.new_outputs = results.new_carries;
  return results;
}

// Checks whether an operation inside `main_func` is supported by XLA's TPU
// `ScanEmitter` (`VectorizingEmitter`) and `chlo.scan` decomposition.
//
// Why `stablehlo.constant` is excluded: `chlo.scan` does not carry the
// `IsolatedFromAbove` trait, so MLIR canonicalization hoists any
// `stablehlo.constant` inside `chlo.scan`'s body region out to the enclosing
// function before `ChloPreserveHighLevelOpsPass` runs. When
// `ChloPreserveHighLevelOpsPass` subsequently outlines the `chlo.scan` body
// into a standalone `IsolatedFromAbove` `func.func`, references to hoisted
// constants outside the region fail MLIR verification (`using value defined
// outside the region`). Restricting to elementwise StableHLO ops (which take at
// least one operand and exclude 0-operand `ConstantOp`) guarantees the body
// remains self-contained across canonicalization and outlining.
bool IsSupportedChloScanBodyOp(mlir::Operation& op) {
  return llvm::isa<mlir::stablehlo::StablehloDialect>(op.getDialect()) &&
         op.hasTrait<mlir::OpTrait::Elementwise>();
}

// Returns true if every type in `types` is a ranked tensor of shape `shape`.
bool AllRankedWithShape(mlir::TypeRange types,
                        const llvm::ArrayRef<int64_t> shape) {
  return llvm::all_of(types, [shape](const mlir::Type type) {
    const auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(type);
    return ranked != nullptr && ranked.getShape() == shape;
  });
}

// Returns true if every scanned input is a statically shaped, non-complex
// tensor with the same shape as `scanned_inputs[0]`, which has rank >= 1.
bool ScannedInputsAreUniformStatic(
    const llvm::ArrayRef<mlir::MlirOp> scanned_inputs) {
  const mlir::RankedTensorType first_type =
      GetTensorTypeOrDie(scanned_inputs[0]);
  if (!first_type.hasStaticShape() || first_type.getRank() < 1) {
    return false;
  }
  return llvm::all_of(scanned_inputs, [first_type](mlir::MlirOp input) {
    const mlir::RankedTensorType t = GetTensorTypeOrDie(input);
    return t.hasStaticShape() && t.getShape() == first_type.getShape() &&
           !IsComplexType(t);
  });
}

// Determines whether the associative scan can be lowered to `chlo.scan`
// (`is_associative = true`) for XLA's native TPU `ScanEmitter` instead of a
// sequential `stablehlo.while` loop.
//
// XLA's TPU `ScanEmitter` requires:
// 1. No external non-scanned inputs (`total_inputs == num_scan_inputs`),
//    because `ChloPreserveHighLevelOpsPass` outlines `chlo.scan` into a
//    composite function taking only `(inputs, inits)`.
// 2. Statically shaped non-complex tensors of identical shape across all
//    scanned inputs (`ScanEmitter` vectorizes across a uniform tile shape and
//    does not support complex element types).
// 3. Every argument and return value of `main_func` has the per-step
//    `slice_shape`, and every body op is an elementwise StableHLO op.
bool CanLowerToChloScan(mlir::func::FuncOp main_func,
                        llvm::ArrayRef<mlir::MlirOp> scanned_inputs,
                        const int64_t scan_dim, const int64_t num_scan_inputs,
                        const int64_t total_inputs) {
  if (total_inputs != num_scan_inputs || num_scan_inputs <= 0) {
    return false;
  }
  const llvm::ArrayRef<mlir::MlirOp> scanned =
      scanned_inputs.take_front(num_scan_inputs);
  if (!ScannedInputsAreUniformStatic(scanned)) {
    return false;
  }

  const llvm::ArrayRef<int64_t> full_shape =
      GetTensorTypeOrDie(scanned[0]).getShape();
  Dimensions slice_shape = CopyIntVector(full_shape);
  slice_shape.erase(slice_shape.begin() + scan_dim);
  if (!AllRankedWithShape(main_func.getArgumentTypes(), slice_shape) ||
      !AllRankedWithShape(main_func.getResultTypes(), slice_shape)) {
    return false;
  }

  // Elementwise ops require identical operand and result shapes, and constants
  // are excluded, so every intermediate value also has `slice_shape`.
  mlir::Block& body_block = main_func.front();
  return absl::c_all_of(
      llvm::make_range(body_block.begin(), std::prev(body_block.end())),
      IsSupportedChloScanBodyOp);
}

// Candidate per-leaf identity constants for symbolic identity inference:
// - `kZero`: additive / bitwise-OR / bitwise-XOR identity (`0` or `false`).
// - `kOne`: multiplicative identity (`1` or `true`).
// - `kMinValue`: `maximum` identity (`-inf` for floats, ` numeric_limits::min`
//   for integers).
// - `kMaxValue`: `minimum` identity (`+inf` for floats, `numeric_limits::max`
//   for integers).
enum class IdentityKind { kZero, kOne, kMinValue, kMaxValue };

// Lightweight symbolic value domain used to verify whether a candidate tuple of
// `IdentityKind` values is a true two-sided algebraic identity of `main_func`.
// Each input leaf is bound either to a symbolic variable `Var(i)` or to a
// candidate constant (`Const(0)`, `Const(1)`, `NegInf`, `PosInf`), and
// `EvalSymbolicOp` propagates symbolic values through the combine body using
// exact algebraic identities (e.g. `0 + x = x`, `1 * x = x`, `0 * x = 0`,
// `max(-inf, x) = x`, `min(+inf, x) = x`, `exp(-inf) = 0`, `log(1) = 0`).
struct SymVal {
  enum class Tag { kConst, kVar, kNegInf, kPosInf, kUnknown };
  Tag tag = Tag::kUnknown;
  double const_val = 0.0;
  int64_t var_id = -1;

  static SymVal Const(double v) {
    if (std::isnan(v)) return {Tag::kUnknown, 0.0, -1};
    if (std::isinf(v)) {
      return {v < 0 ? Tag::kNegInf : Tag::kPosInf, 0.0, -1};
    }
    return {Tag::kConst, v == 0.0 ? 0.0 : v, -1};
  }
  static SymVal Var(int64_t id) { return {Tag::kVar, 0.0, id}; }
  static SymVal NegInf() { return {Tag::kNegInf, 0.0, -1}; }
  static SymVal PosInf() { return {Tag::kPosInf, 0.0, -1}; }
  static SymVal Unknown() { return {Tag::kUnknown, 0.0, -1}; }

  bool IsZero() const { return tag == Tag::kConst && const_val == 0.0; }
  bool IsOne() const { return tag == Tag::kConst && const_val == 1.0; }
  bool IsConst() const { return tag == Tag::kConst; }
  bool IsNegInf() const { return tag == Tag::kNegInf; }
  bool IsPosInf() const { return tag == Tag::kPosInf; }
  bool IsUnknown() const { return tag == Tag::kUnknown; }
  // True for finite values (constants and variables, which are bound to finite
  // inputs for the purposes of identity inference).
  bool IsFinite() const { return tag == Tag::kConst || tag == Tag::kVar; }
  bool IsVar(const int64_t id) const {
    return tag == Tag::kVar && var_id == id;
  }
  bool IsSameVar(const SymVal& other) const {
    return other.tag == Tag::kVar && IsVar(other.var_id);
  }
};

SymVal IdentityKindToSymVal(const IdentityKind kind) {
  switch (kind) {
    case IdentityKind::kZero:
      return SymVal::Const(0.0);
    case IdentityKind::kOne:
      return SymVal::Const(1.0);
    case IdentityKind::kMinValue:
      return SymVal::NegInf();
    case IdentityKind::kMaxValue:
      return SymVal::PosInf();
  }
}

SymVal EvalNeg(const SymVal& a) {
  if (a.IsConst()) return SymVal::Const(-a.const_val);
  if (a.IsNegInf()) return SymVal::PosInf();
  if (a.IsPosInf()) return SymVal::NegInf();
  return SymVal::Unknown();
}

SymVal EvalAbs(const SymVal& a) {
  if (a.IsConst()) return SymVal::Const(std::abs(a.const_val));
  if (a.IsNegInf() || a.IsPosInf()) return SymVal::PosInf();
  return SymVal::Unknown();
}

SymVal EvalExp(const SymVal& a) {
  if (a.IsNegInf()) return SymVal::Const(0.0);
  if (a.IsZero()) return SymVal::Const(1.0);
  return SymVal::Unknown();
}

SymVal EvalLog(const SymVal& a) {
  if (a.IsZero()) return SymVal::NegInf();
  if (a.IsOne()) return SymVal::Const(0.0);
  return SymVal::Unknown();
}

SymVal EvalLog1p(const SymVal& a) {
  return a.IsZero() ? SymVal::Const(0.0) : SymVal::Unknown();
}

SymVal EvalAdd(const SymVal& lhs, const SymVal& rhs) {
  if (lhs.IsZero()) return rhs;
  if (rhs.IsZero()) return lhs;
  if (lhs.IsConst() && rhs.IsConst()) {
    return SymVal::Const(lhs.const_val + rhs.const_val);
  }
  // `-inf + y = -inf` unless `y = +inf` (which would be NaN).
  if ((lhs.IsNegInf() && !rhs.IsPosInf()) ||
      (rhs.IsNegInf() && !lhs.IsPosInf())) {
    return SymVal::NegInf();
  }
  return SymVal::Unknown();
}

SymVal EvalSubtract(const SymVal& lhs, const SymVal& rhs) {
  if (rhs.IsZero()) return lhs;
  if (lhs.IsConst() && rhs.IsConst()) {
    return SymVal::Const(lhs.const_val - rhs.const_val);
  }
  if (lhs.IsSameVar(rhs)) return SymVal::Const(0.0);
  // `-inf - y = -inf` unless `y = -inf`; `x - +inf = -inf` unless `x = +inf`.
  if ((lhs.IsNegInf() && !rhs.IsNegInf()) ||
      (rhs.IsPosInf() && !lhs.IsPosInf())) {
    return SymVal::NegInf();
  }
  return SymVal::Unknown();
}

SymVal EvalMul(const SymVal& lhs, const SymVal& rhs) {
  if (lhs.IsOne()) return rhs;
  if (rhs.IsOne()) return lhs;
  // `0 * x = 0` only for finite `x` (`0 * inf` is NaN).
  if ((lhs.IsZero() && rhs.IsFinite()) || (rhs.IsZero() && lhs.IsFinite())) {
    return SymVal::Const(0.0);
  }
  if (lhs.IsConst() && rhs.IsConst()) {
    return SymVal::Const(lhs.const_val * rhs.const_val);
  }
  return SymVal::Unknown();
}

SymVal EvalMax(const SymVal& lhs, const SymVal& rhs) {
  if (lhs.IsNegInf()) return rhs;
  if (rhs.IsNegInf()) return lhs;
  if (lhs.IsPosInf() || rhs.IsPosInf()) return SymVal::PosInf();
  if (lhs.IsSameVar(rhs)) return lhs;
  if (lhs.IsConst() && rhs.IsConst()) {
    return SymVal::Const(std::max(lhs.const_val, rhs.const_val));
  }
  return SymVal::Unknown();
}

SymVal EvalMin(const SymVal& lhs, const SymVal& rhs) {
  if (lhs.IsPosInf()) return rhs;
  if (rhs.IsPosInf()) return lhs;
  if (lhs.IsNegInf() || rhs.IsNegInf()) return SymVal::NegInf();
  if (lhs.IsSameVar(rhs)) return lhs;
  if (lhs.IsConst() && rhs.IsConst()) {
    return SymVal::Const(std::min(lhs.const_val, rhs.const_val));
  }
  return SymVal::Unknown();
}

SymVal EvalOrXor(const SymVal& lhs, const SymVal& rhs) {
  if (lhs.IsZero()) return rhs;
  if (rhs.IsZero()) return lhs;
  return SymVal::Unknown();
}

// Only called for bodies accepted by `CanLowerToChloScan`, so `op` is always an
// elementwise StableHLO op (never a constant, broadcast, or reshape).
SymVal EvalSymbolicOp(mlir::Operation& op,
                      const llvm::DenseMap<mlir::Value, SymVal>& env) {
  auto get_arg = [&env, &op](const int64_t i) -> SymVal {
    const auto it = env.find(op.getOperand(i));
    return it != env.end() ? it->second : SymVal::Unknown();
  };
  if (op.getNumOperands() == 1) {
    const SymVal a = get_arg(0);
    return llvm::TypeSwitch<mlir::Operation*, SymVal>(&op)
        .Case<mlir::stablehlo::ConvertOp>(
            [&](auto) {  // REF_CAPTURE_OK=called synchronously
              return a;
            })
        .Case<mlir::stablehlo::NegOp>(
            [&](auto) {  // REF_CAPTURE_OK=called synchronously
              return EvalNeg(a);
            })
        .Case<mlir::stablehlo::AbsOp>(
            [&](auto) {  // REF_CAPTURE_OK=called synchronously
              return EvalAbs(a);
            })
        .Case<mlir::stablehlo::ExpOp>(
            [&](auto) {  // REF_CAPTURE_OK=called synchronously
              return EvalExp(a);
            })
        .Case<mlir::stablehlo::LogOp>(
            [&](auto) {  // REF_CAPTURE_OK=called synchronously
              return EvalLog(a);
            })
        .Case<mlir::stablehlo::Log1pOp>(
            [&](auto) {  // REF_CAPTURE_OK=called synchronously
              return EvalLog1p(a);
            })
        .Default([](mlir::Operation*) { return SymVal::Unknown(); });
  }
  if (op.getNumOperands() != 2) {
    return SymVal::Unknown();
  }
  const SymVal lhs = get_arg(0);
  const SymVal rhs = get_arg(1);
  if (lhs.IsUnknown() || rhs.IsUnknown()) {
    return SymVal::Unknown();
  }
  // `AndOp` is deliberately absent: its identity is all-ones, which none of the
  // `IdentityKind`s materialize for signed integers (`kMaxValue` is INT_MAX).
  return llvm::TypeSwitch<mlir::Operation*, SymVal>(&op)
      .Case<mlir::stablehlo::AddOp>(
          [&](auto) {  // REF_CAPTURE_OK=called synchronously
            return EvalAdd(lhs, rhs);
          })
      .Case<mlir::stablehlo::SubtractOp>(
          [&](auto) {  // REF_CAPTURE_OK=called synchronously
            return EvalSubtract(lhs, rhs);
          })
      .Case<mlir::stablehlo::MulOp>(
          [&](auto) {  // REF_CAPTURE_OK=called synchronously
            return EvalMul(lhs, rhs);
          })
      .Case<mlir::stablehlo::MaxOp>(
          [&](auto) {  // REF_CAPTURE_OK=called synchronously
            return EvalMax(lhs, rhs);
          })
      .Case<mlir::stablehlo::MinOp>(
          [&](auto) {  // REF_CAPTURE_OK=called synchronously
            return EvalMin(lhs, rhs);
          })
      .Case<mlir::stablehlo::OrOp, mlir::stablehlo::XorOp>(
          [&](auto) {  // REF_CAPTURE_OK=called synchronously
            return EvalOrXor(lhs, rhs);
          })
      .Default([](mlir::Operation*) { return SymVal::Unknown(); });
}

// Which operand of the combine function an identity candidate is bound to.
enum class IdentitySide { kLeft, kRight };

// Evaluates `main_func` symbolically with `identity` bound to the `side`
// operand of the combine function and variables bound to the other, and returns
// true if the result is exactly the variables, i.e. `f(e, x) == x` for `kLeft`
// and `f(x, e) == x` for `kRight`.
bool IdentityHoldsOnSide(mlir::func::FuncOp main_func,
                         const int64_t num_scan_inputs,
                         const llvm::ArrayRef<IdentityKind> identity,
                         const IdentitySide side) {
  mlir::Block& body_block = main_func.front();
  const mlir::Block::BlockArgListType func_args = main_func.getArguments();
  const bool on_left = side == IdentitySide::kLeft;
  const int64_t identity_offset = on_left ? 0 : num_scan_inputs;
  const int64_t var_offset = on_left ? num_scan_inputs : 0;

  llvm::DenseMap<mlir::Value, SymVal> env;
  for (int64_t i = 0; i < num_scan_inputs; ++i) {
    env[func_args[identity_offset + i]] = IdentityKindToSymVal(identity[i]);
    env[func_args[var_offset + i]] = SymVal::Var(i);
  }
  for (mlir::Operation& op :
       llvm::make_range(body_block.begin(), std::prev(body_block.end()))) {
    const SymVal val =
        op.getNumResults() == 1 ? EvalSymbolicOp(op, env) : SymVal::Unknown();
    if (val.IsUnknown()) {
      return false;
    }
    env[op.getResult(0)] = val;
  }
  mlir::Operation* const terminator = body_block.getTerminator();
  for (int64_t i = 0; i < num_scan_inputs; ++i) {
    const auto it = env.find(terminator->getOperand(i));
    if (it == env.end() || !it->second.IsVar(i)) {
      return false;
    }
  }
  return true;
}

// Verifies that `candidate` is both a left identity (`f(e, x) == x`) and a
// right identity (`f(x, e) == x`) for `main_func`. Checking both directions is
// essential: non-associative or one-sided operations like `subtract(x, 0) = x`
// satisfy right-identity (`x - 0 = x`) but fail left-identity (`0 - x = -x`),
// whereas non-commutative associative operations like the Mamba/SSM recurrence
// `(a_l * a_r, a_r * b_l + b_r)` with `e = (1, 0)` satisfy both:
//   left:  `(1 * a_r, a_r * 0 + b_r) = (a_r, b_r)`
//   right: `(a_l * 1, 1 * b_l + 0)   = (a_l, b_l)`
bool IsSymbolicIdentityCandidate(mlir::func::FuncOp main_func,
                                 const int64_t num_scan_inputs,
                                 llvm::ArrayRef<IdentityKind> candidate) {
  return IdentityHoldsOnSide(main_func, num_scan_inputs, candidate,
                             IdentitySide::kLeft) &&
         IdentityHoldsOnSide(main_func, num_scan_inputs, candidate,
                             IdentitySide::kRight);
}

// Attempts to infer a two-sided algebraic identity element `(e_0, ...,
// e_{k-1})` for `main_func` such that `main_func(e, x) == x` and `main_func(x,
// e) == x`.
//
// Why this matters for performance: `chlo.scan` requires an explicit `inits`
// carry tensor. When we know the algebraic identity `e`, we can pass `e` as
// `inits` and run `chlo.scan` directly over the full `seq_len` input tensor in
// a single kernel (avoiding the Tier 2 `stablehlo.slice` + `(seq_len - 1)` scan
// + `stablehlo.concatenate` HBM copy overhead and keeping `seq_len` aligned to
// powers of two / TPU tile multiples).
//
// For `num_scan_inputs <= 4` (single tensors, 2-tensor Mamba/SSM tuples, and
// 3-/4-tensor linear state tuples), we exhaustively test all `4^k <= 256`
// combinations of `{kZero, kOne, kMinValue, kMaxValue}` so mixed tuples like
// `(kOne, kZero)` are discovered automatically. Wider tuples use Tier 2.
std::optional<llvm::SmallVector<IdentityKind, 4>> InferAssociativeIdentity(
    mlir::func::FuncOp main_func, const int64_t num_scan_inputs) {
  static constexpr IdentityKind kKinds[] = {
      IdentityKind::kZero, IdentityKind::kOne, IdentityKind::kMinValue,
      IdentityKind::kMaxValue};
  if (num_scan_inputs > 4) {
    return std::nullopt;
  }

  llvm::SmallVector<IdentityKind, 4> candidate(num_scan_inputs);
  const int64_t total_combinations = int64_t{1} << (2 * num_scan_inputs);
  for (int64_t mask = 0; mask < total_combinations; ++mask) {
    for (int64_t i = 0; i < num_scan_inputs; ++i) {
      candidate[i] = kKinds[(mask >> (2 * i)) & 3];
    }
    if (IsSymbolicIdentityCandidate(main_func, num_scan_inputs, candidate)) {
      return candidate;
    }
  }
  return std::nullopt;
}

// Materializes a constant tensor of shape `input.shape` with `scan_dim` removed
// filled with the algebraic identity value `kind` matching `input`'s dtype.
mlir::MlirOp CreateIdentityConstant(mlir::MlirBuilder& builder,
                                    mlir::MlirOp input, const int64_t scan_dim,
                                    const IdentityKind kind) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  Dimensions reduced_shape = CopyIntVector(input_type.getShape());
  reduced_shape.erase(reduced_shape.begin() + scan_dim);
  const mlir::Type element_type = input_type.getElementType();
  const mlir::RankedTensorType reduced_type =
      mlir::RankedTensorType::get(reduced_shape, element_type);

  switch (kind) {
    case IdentityKind::kZero:
      if (IsBooleanType(input_type)) {
        return MakeConstant(builder, false, reduced_type);
      }
      return MakeConstant(builder, 0, reduced_type);
    case IdentityKind::kOne:
      if (IsBooleanType(input_type)) {
        return MakeConstant(builder, true, reduced_type);
      }
      return MakeConstant(builder, 1, reduced_type);
    case IdentityKind::kMinValue: {
      if (const auto float_type =
              llvm::dyn_cast<mlir::FloatType>(element_type)) {
        const llvm::APFloat neg_inf = llvm::APFloat::getInf(
            float_type.getFloatSemantics(), /*Negative=*/true);
        return mlir::stablehlo::Constant(
            builder, mlir::DenseElementsAttr::get(reduced_type, neg_inf));
      }
      const mlir::Attribute attr =
          GetMinFiniteValueAttr(element_type, builder.getOpBuilder());
      return mlir::stablehlo::Constant(
          builder, mlir::DenseElementsAttr::get(reduced_type, attr));
    }
    case IdentityKind::kMaxValue: {
      if (const auto float_type =
              llvm::dyn_cast<mlir::FloatType>(element_type)) {
        const llvm::APFloat pos_inf = llvm::APFloat::getInf(
            float_type.getFloatSemantics(), /*Negative=*/false);
        return mlir::stablehlo::Constant(
            builder, mlir::DenseElementsAttr::get(reduced_type, pos_inf));
      }
      const mlir::Attribute attr =
          GetMaxFiniteValueAttr(element_type, builder.getOpBuilder());
      return mlir::stablehlo::Constant(
          builder, mlir::DenseElementsAttr::get(reduced_type, attr));
    }
  }
}

// Returns the row-major strides of a contiguous buffer with dimensions `dims`.
llvm::SmallVector<int64_t, 8> ContiguousStrides(
    const absl::Span<const int64_t> dims) {
  const int64_t rank = dims.size();
  llvm::SmallVector<int64_t, 8> strides(rank, 1);
  for (int64_t k = rank - 2; k >= 0; --k) {
    strides[k] = strides[k + 1] * std::max<int64_t>(1, dims[k + 1]);
  }
  return strides;
}

// Returns true if `t` is a `movedim(base, d, 0)` view of a contiguous base
// buffer with dimensions `base_dims` and strides `base_strides`. Such a view
// has:
// - view dimension 0 mapped to base dimension `d`
// - view dimension `k` (`1 <= k <= d`) mapped to base dimension `k - 1`
// - view dimension `k` (`k > d`) mapped to base dimension `k`
// Strides of size-1 dimensions are ignored since they are never dereferenced.
bool IsMovedimViewOfDim(const at::Tensor& t,
                        const absl::Span<const int64_t> base_dims,
                        const llvm::ArrayRef<int64_t> base_strides,
                        const int64_t d) {
  const int64_t rank = base_dims.size();
  for (int64_t k = 0; k < rank; ++k) {
    const int64_t src_dim = k == 0 ? d : (k <= d ? k - 1 : k);
    if (t.size(k) != base_dims[src_dim] ||
        (t.size(k) > 1 && t.stride(k) != base_strides[src_dim])) {
      return false;
    }
  }
  return true;
}

// Returns the base dimension `d > 0` such that `t` is a `movedim(base, d, 0)`
// view of its contiguous, statically shaped base buffer, or `std::nullopt` if
// there is none. If `required_dim` is set, only that `d` is considered so all
// scanned inputs agree on a common base dimension.
std::optional<int64_t> MatchMovedimToZeroView(
    const at::Tensor& t, const std::optional<int64_t> required_dim) {
  if (!t.has_storage() || t.storage_offset() != 0 || t.is_conj() ||
      t.is_neg() || t.dim() < 2) {
    return std::nullopt;
  }
  const absl::StatusOr<DeviceBufferRef> base_ref = GetBaseBuffer(t);
  if (!base_ref.ok() || !base_ref->shape().dynamic_dimensions().empty()) {
    return std::nullopt;
  }
  const absl::Span<const int64_t> base_dims = base_ref->dimensions();
  const int64_t rank = base_dims.size();
  // A view with the same sizes as its base is not a non-trivial `movedim`.
  if (t.dim() != rank ||
      t.sizes() == c10::IntArrayRef(base_dims.data(), base_dims.size())) {
    return std::nullopt;
  }

  const llvm::SmallVector<int64_t, 8> base_strides =
      ContiguousStrides(base_dims);
  for (int64_t d = 1; d < rank; ++d) {
    if ((!required_dim.has_value() || *required_dim == d) &&
        IsMovedimViewOfDim(t, base_dims, base_strides, d)) {
      return d;
    }
  }
  return std::nullopt;
}

// Detects whether all `inputs[0 .. num_scan_inputs - 1]` are pure
// `movedim(base, base_scan_dim, 0)` views of their contiguous base buffers
// for a common `base_scan_dim > 0`.
//
// Why this matters: PyTorch's `torch._higher_order_ops.associative_scan`
// frontend normalizes `dim` to `0` by wrapping every input with
// `torch.movedim(x, dim, 0)` before calling
// `torch.ops.higher_order.associative_scan` with `dim=0`, and then wrapping
// every output with `torch.movedim(out, 0, dim)`. If we materialize `inputs[i]`
// via `GetBuffer(inputs[i])`, TorchTPU emits a `ViewToContiguous`
// (`stablehlo.transpose`) before the scan and a second `stablehlo.transpose`
// after the scan. Instead, when all scanned inputs are `movedim(base,
// base_scan_dim, 0)` views over contiguous base buffers, we can:
// 1. Grab `GetBaseBuffer(inputs[i])` directly (zero-copy).
// 2. Execute `chlo.scan` along `base_scan_dim` on the contiguous base shape.
// 3. Return output tensors whose `TensorImpl` metadata has `inputs[i]`'s
//    `movedim` sizes and strides over the contiguous base output buffer, so the
//    caller's subsequent `torch.movedim(out, 0, dim)` restores standard
//    contiguous strides with zero `stablehlo.transpose` operations.
std::optional<int64_t> TryDetectMovedimToZeroScanDim(
    const std::vector<at::Tensor>& inputs, const int64_t dim,
    const int64_t num_scan_inputs) {
  if (dim != 0 || num_scan_inputs <= 0) {
    return std::nullopt;
  }
  std::optional<int64_t> common_base_dim;
  for (int64_t i = 0; i < num_scan_inputs; ++i) {
    common_base_dim = MatchMovedimToZeroView(inputs[i], common_base_dim);
    if (!common_base_dim.has_value()) {
      return std::nullopt;
    }
  }
  return common_base_dim;
}

// Device buffers passed to the deferred associative scan op, plus the shapes of
// its outputs (one per scanned input).
struct AssociativeScanBuffers {
  std::vector<DeviceBufferRef> inputs;
  std::vector<Shape> output_shapes;
};

// Collects the device buffers of `inputs`. When `unwrapped_base_dim` is set
// (see `TryDetectMovedimToZeroScanDim`), the scanned inputs use their
// contiguous base buffers instead of materializing the `movedim` views.
AssociativeScanBuffers CollectAssociativeScanBuffers(
    const std::vector<at::Tensor>& inputs, const int64_t num_scan_inputs,
    const std::optional<int64_t> unwrapped_base_dim) {
  const int64_t total_inputs = inputs.size();
  AssociativeScanBuffers buffers;
  buffers.inputs.reserve(total_inputs);
  buffers.output_shapes.reserve(num_scan_inputs);
  for (int64_t i = 0; i < total_inputs; ++i) {
    const bool is_scanned = i < num_scan_inputs;
    TT_ASSIGN_OR_THROW(const DeviceBufferRef ref,
                       is_scanned&& unwrapped_base_dim.has_value()
                           ? GetBaseBuffer(inputs[i])
                           : GetBuffer(inputs[i]));
    buffers.inputs.push_back(ref);
    if (is_scanned) {
      buffers.output_shapes.push_back(ref.shape());
    }
  }
  return buffers;
}

// Returns the `[start, start + size)` slice of `input` along `scan_dim`.
mlir::MlirOp SliceAlongDim(mlir::MlirOp input, const int64_t scan_dim,
                           const int64_t start, const int64_t size) {
  const llvm::ArrayRef<int64_t> shape = GetTensorTypeOrDie(input).getShape();
  Dimensions starts(shape.size(), 0);
  Dimensions limits = CopyIntVector(shape);
  const Dimensions strides(shape.size(), 1);
  starts[scan_dim] = start;
  limits[scan_dim] = start + size;
  return mlir::stablehlo::Slice(input, starts, limits, strides);
}

// Tier 1 (Algebraic Identity + `chlo.scan`): when the combine function admits a
// two-sided identity `e`, emits a single `chlo.scan` (`is_associative=true`)
// over the full `seq_len` input tensor with `carry_inits = e`.
absl::StatusOr<DynamicMlirOpResults> EmitIdentityInitChloScan(
    mlir::MlirBuilder& builder, const llvm::ArrayRef<mlir::MlirOp> scan_inputs,
    const int64_t scan_dim, const int64_t num_scan_inputs,
    const llvm::ArrayRef<IdentityKind> identity_kinds,
    const ScanDirection scan_direction,
    MultiInputScanBodyBuilder body_builder) {
  llvm::SmallVector<mlir::MlirOp, 4> carry_inits;
  carry_inits.reserve(num_scan_inputs);
  for (int64_t i = 0; i < num_scan_inputs; ++i) {
    carry_inits.push_back(CreateIdentityConstant(builder, scan_inputs[i],
                                                 scan_dim, identity_kinds[i]));
  }
  TT_ASSIGN_OR_RETURN(
      const DynamicMlirOpResults results,
      BuildScanShlo(builder, scan_inputs, scan_dim, num_scan_inputs,
                    carry_inits, scan_inputs, std::move(body_builder),
                    ScanOptions{.direction = scan_direction,
                                .should_squeeze = true,
                                .is_associative = true}));
  // `BuildScanShlo` returns `(*final_carries, *outputs)`.
  return DynamicMlirOpResults(results.begin() + num_scan_inputs, results.end());
}

// Tier 2 (Slice-Init): when no algebraic identity is known (or when
// `chlo.scan` is unsupported due to constants or additional inputs), uses the
// first element along `scan_dim` (`seq_len - 1` for reverse) as the initial
// carry, scans the remaining `seq_len - 1` elements via `chlo.scan` (if
// `is_associative`) or `stablehlo.while`, and concatenates the initial slice
// with the scanned outputs.
absl::StatusOr<DynamicMlirOpResults> EmitSliceInitScan(
    mlir::MlirBuilder& builder, const llvm::ArrayRef<mlir::MlirOp> scan_inputs,
    const int64_t scan_dim, const int64_t seq_len,
    const int64_t num_scan_inputs, const ScanOptions& options,
    MultiInputScanBodyBuilder body_builder) {
  const bool forward = options.direction == ScanDirection::kForward;
  const int64_t init_start = forward ? 0 : seq_len - 1;
  const int64_t rest_start = forward ? 1 : 0;

  llvm::SmallVector<mlir::MlirOp, 4> init_slices;
  llvm::SmallVector<mlir::MlirOp, 4> carry_inits;
  llvm::SmallVector<mlir::MlirOp, 4> rest_scan_inputs;
  llvm::SmallVector<mlir::MlirOp, 4> output_inits;
  init_slices.reserve(num_scan_inputs);
  carry_inits.reserve(num_scan_inputs);
  rest_scan_inputs.reserve(scan_inputs.size());
  output_inits.reserve(num_scan_inputs);
  for (int64_t i = 0; i < num_scan_inputs; ++i) {
    const mlir::MlirOp init_slice =
        SliceAlongDim(scan_inputs[i], scan_dim, init_start, 1);
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp carry_init,
                        Squeeze(init_slice, {scan_dim}));
    const mlir::MlirOp rest_input =
        SliceAlongDim(scan_inputs[i], scan_dim, rest_start, seq_len - 1);
    init_slices.push_back(init_slice);
    carry_inits.push_back(carry_init);
    rest_scan_inputs.push_back(rest_input);
    output_inits.push_back(MakeConstantLike(rest_input, 0));
  }
  llvm::append_range(rest_scan_inputs, scan_inputs.drop_front(num_scan_inputs));

  TT_ASSIGN_OR_RETURN(const DynamicMlirOpResults rest_results,
                      BuildScanShlo(builder, rest_scan_inputs, scan_dim,
                                    num_scan_inputs, carry_inits, output_inits,
                                    std::move(body_builder), options));

  DynamicMlirOpResults outputs;
  outputs.reserve(num_scan_inputs);
  for (int64_t i = 0; i < num_scan_inputs; ++i) {
    const mlir::MlirOp rest_out = rest_results[num_scan_inputs + i];
    const llvm::SmallVector<mlir::MlirOp, 2> parts =
        forward ? llvm::SmallVector<mlir::MlirOp, 2>{init_slices[i], rest_out}
                : llvm::SmallVector<mlir::MlirOp, 2>{rest_out, init_slices[i]};
    outputs.push_back(mlir::stablehlo::Concatenate(builder, parts, scan_dim));
  }
  return outputs;
}

// Lowers an associative scan of `op_inputs` (`num_scan_inputs` scanned inputs
// followed by additional inputs) with the combine function serialized in
// `serialized_mlir`.
absl::StatusOr<DynamicMlirOpResults> BuildAssociativeScan(
    mlir::MlirBuilder& builder, const absl::Span<mlir::MlirOp> op_inputs,
    const std::string& serialized_mlir, const int64_t total_inputs,
    const ScanDirection scan_direction, const int64_t effective_dim,
    const int64_t num_scan_inputs) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      op_inputs.size() == total_inputs, error::kInvalidArgument)
      << "expected " << total_inputs << " inputs, got " << op_inputs.size();

  auto body_module_op = mlir::parseSourceString<mlir::ModuleOp>(
      serialized_mlir, mlir::ParserConfig{&builder.getContext()});
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      body_module_op.get() != nullptr, error::kInternal)
      << "failed to parse body MLIR module";

  auto main_func = body_module_op->lookupSymbol<mlir::func::FuncOp>("main");
  TT_RETURN_IF_ERROR(
      ImportStepFunctionCallees(builder, *body_module_op, main_func));

  const llvm::ArrayRef<mlir::MlirOp> scan_inputs(op_inputs.data(),
                                                 op_inputs.size());
  const mlir::RankedTensorType scan_inputs0_type =
      GetTensorTypeOrDie(scan_inputs[0]);
  TT_ASSIGN_OR_RETURN(const int64_t scan_dim,
                      SafeWrapDim(effective_dim, scan_inputs0_type.getRank()));
  const int64_t seq_len = scan_inputs0_type.getDimSize(scan_dim);

  // An associative scan along a dimension of length 0 or 1 is an identity
  // operation on the scanned inputs.
  if (seq_len <= 1) {
    return DynamicMlirOpResults(scan_inputs.begin(),
                                scan_inputs.begin() + num_scan_inputs);
  }

  auto body_builder =
      [main_func, num_scan_inputs, total_inputs](
          mlir::OpBuilder& op_builder, mlir::Location loc,
          mlir::ValueRange slices, mlir::Value index,
          mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
    return BuildAssociativeLoopBody(op_builder, main_func, num_scan_inputs,
                                    total_inputs, slices, carries);
  };

  const bool can_use_chlo_scan = CanLowerToChloScan(
      main_func, scan_inputs, scan_dim, num_scan_inputs, total_inputs);
  if (can_use_chlo_scan) {
    if (const auto identity_kinds =
            InferAssociativeIdentity(main_func, num_scan_inputs);
        identity_kinds.has_value()) {
      return EmitIdentityInitChloScan(builder, scan_inputs, scan_dim,
                                      num_scan_inputs, *identity_kinds,
                                      scan_direction, std::move(body_builder));
    }
  }
  return EmitSliceInitScan(builder, scan_inputs, scan_dim, seq_len,
                           num_scan_inputs,
                           ScanOptions{.direction = scan_direction,
                                       .should_squeeze = true,
                                       .is_associative = can_use_chlo_scan},
                           std::move(body_builder));
}

}  // namespace

std::vector<at::Tensor> PyCreateScanOp(
    const std::vector<at::Tensor>& inits, const std::vector<at::Tensor>& inputs,
    const std::shared_ptr<ContextedModule>& body_module,
    ScanDirection scan_direction, const std::vector<at::Tensor>& dummy_outputs,
    int64_t num_scan_inputs) {
  ScopedPythonContextCapturer capturer(OpName::kCustomKernel);

  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      !inputs.empty(), error::kInvalidArgument)
      << "expected at least 1 input tensor";

  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      num_scan_inputs > 0, error::kInvalidArgument)
      << "expected num_scan_inputs to be greater than 0, got "
      << num_scan_inputs;
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      num_scan_inputs <= inputs.size(), error::kInvalidArgument)
      << "expected num_scan_inputs (" << num_scan_inputs
      << ") to be less than or equal to the number of inputs (" << inputs.size()
      << ")";

  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      body_module != nullptr && body_module->get(), error::kInvalidArgument)
      << "invalid body module";

  auto body_main_func =
      body_module->get().lookupSymbol<mlir::func::FuncOp>("main");
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      body_main_func != nullptr, error::kInternal)
      << "expected 'main' function in body MLIR module";

  const int64_t num_carries = inits.size();
  const int64_t num_results = body_main_func.getResultTypes().size();
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      num_results >= num_carries, error::kInvalidArgument)
      << "expected step function to have at least " << num_carries
      << " result types (corresponding to carries), but it only has "
      << num_results;

  const int64_t num_outputs = num_results - num_carries;
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      num_outputs == dummy_outputs.size(), error::kInvalidArgument)
      << "expected " << num_outputs << " dummy outputs, got "
      << dummy_outputs.size();

  // Serialize the body MLIR module to strings: one without locations to act as
  // a cache key, and one with locations to pass across different
  // `MLIRContext`s.
  const std::string cache_key_mlir = PrintModuleForCacheKey(body_module->get());
  const std::string serialized_mlir =
      PrintModuleWithLocations(body_module->get());

  const int64_t total_inputs = inputs.size();

  MlirOpBuilder op_builder = [serialized_mlir, num_carries, total_inputs,
                              scan_direction, num_scan_inputs](
                                 mlir::MlirBuilder& builder,
                                 absl::Span<mlir::MlirOp> op_inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
        op_inputs.size() == num_carries + total_inputs, error::kInvalidArgument)
        << "expected " << num_carries + total_inputs << " inputs, got "
        << op_inputs.size();

    // Parse the captured serialized MLIR module into the lowerer's context.
    auto body_module_op = mlir::parseSourceString<mlir::ModuleOp>(
        serialized_mlir, mlir::ParserConfig{&builder.getContext()});
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
        body_module_op.get() != nullptr, error::kInternal)
        << "failed to parse body MLIR module";

    auto main_func = body_module_op->lookupSymbol<mlir::func::FuncOp>("main");

    // Must run before the step function's operations are cloned into the loop
    // body: it is what makes any `func.call` among them resolve.
    TT_RETURN_IF_ERROR(
        ImportStepFunctionCallees(builder, *body_module_op, main_func));

    // Split inputs into carries and scanned inputs.
    const llvm::ArrayRef<mlir::MlirOp> inputs_ref(op_inputs.data(),
                                                  op_inputs.size());
    const auto carry_inits = inputs_ref.take_front(num_carries);
    const auto scanned_inputs = inputs_ref.drop_front(num_carries);

    const mlir::RankedTensorType scan_inputs0_type =
        GetTensorTypeOrDie(scanned_inputs[0]);
    const int64_t seq_len = scan_inputs0_type.getDimSize(0);
    const llvm::SmallVector<mlir::MlirOp> output_inits =
        InitZeroOutputAccumulators(builder, main_func, num_carries, seq_len);

    // Build the while-loop body.
    auto body_builder =
        [main_func, num_carries, total_inputs](
            mlir::OpBuilder& op_builder, mlir::Location loc,
            mlir::ValueRange slices, mlir::Value index,
            mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
      return BuildLoopBody(op_builder, main_func, num_carries, total_inputs,
                           slices, carries);
    };

    // PyTorch's frontend dispatcher always transposes the sequence dimension to
    // index 0. Hence, we unconditionally set dim=0 here.
    //
    // Further, should_squeeze=true because the slice returned from scanning
    // along the sequence dimension has shape [1, features...] but the step
    // function expects [features...]).
    return BuildScanShlo(
        builder, scanned_inputs, /*dim=*/0, num_scan_inputs, carry_inits,
        output_inits, std::move(body_builder),
        ScanOptions{.direction = scan_direction, .should_squeeze = true});
  };

  TT_ASSIGN_OR_THROW(auto cache_keys,
                     *(OpParamCacheKeysBuilder()
                           .SetParam("body_mlir", cache_key_mlir)
                           .SetParam("scan_direction", scan_direction)
                           .SetParam("num_scan_inputs", num_scan_inputs)
                           .SetParam("num_carries", num_carries)));

  std::vector<DeviceBufferRef> input_buffers = GetInputBuffers(inits, inputs);
  std::vector<Shape> output_shapes = GetOutputShapes(inits, dummy_outputs);
  TT_ASSIGN_OR_THROW(
      const auto result_refs,
      internal::CreateDeferredDeviceBufferList(internal::DeferredOpParams{
          .op_name = OpName::kCustomKernel,
          .op_builder = std::move(op_builder),
          .op_param_cache_keys = std::move(cache_keys),
          .inputs = std::move(input_buffers),
          .output_shapes = std::move(output_shapes),
          .device_assignment = body_module->device_assignment(),
      }));

  std::vector<at::Tensor> results;
  results.reserve(result_refs.size());
  absl::c_transform(result_refs, std::back_inserter(results),
                    [](const DeviceBufferRef& ref) { return MakeTensor(ref); });
  return results;
}

std::vector<at::Tensor> PyCreateAssociativeScanOp(
    const std::vector<at::Tensor>& inputs,
    const std::shared_ptr<ContextedModule>& body_module,
    ScanDirection scan_direction, int64_t dim, int64_t num_scan_inputs) {
  ScopedPythonContextCapturer capturer(OpName::kCustomKernel);

  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      !inputs.empty(), error::kInvalidArgument)
      << "expected at least 1 input tensor";
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      num_scan_inputs > 0, error::kInvalidArgument)
      << "expected num_scan_inputs to be greater than 0, got "
      << num_scan_inputs;
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      num_scan_inputs <= inputs.size(), error::kInvalidArgument)
      << "expected num_scan_inputs (" << num_scan_inputs
      << ") to be less than or equal to the number of inputs (" << inputs.size()
      << ")";
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      body_module != nullptr && body_module->get(), error::kInvalidArgument)
      << "invalid body module";

  auto body_main_func =
      body_module->get().lookupSymbol<mlir::func::FuncOp>("main");
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      body_main_func != nullptr, error::kInternal)
      << "expected 'main' function in body MLIR module";

  const int64_t num_results = body_main_func.getResultTypes().size();
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      num_results == num_scan_inputs, error::kInvalidArgument)
      << "expected associative combine function to have " << num_scan_inputs
      << " result types, but it has " << num_results;

  const int64_t total_inputs = inputs.size();
  const int64_t expected_num_args = num_scan_inputs + total_inputs;
  const int64_t actual_num_args = body_main_func.getNumArguments();
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      actual_num_args == expected_num_args, error::kInvalidArgument)
      << "expected associative combine function to have " << expected_num_args
      << " arguments, but it has " << actual_num_args;

  // If the caller normalized `dim` to 0 via `torch.movedim(base, d, 0)`,
  // unwrap the `movedim` view to scan the underlying contiguous base buffers
  // directly along `d` and avoid materializing `stablehlo.transpose` ops.
  const std::optional<int64_t> unwrapped_base_dim =
      TryDetectMovedimToZeroScanDim(inputs, dim, num_scan_inputs);
  const bool unwrap_movedim = unwrapped_base_dim.has_value();

  AssociativeScanBuffers buffers = CollectAssociativeScanBuffers(
      inputs, num_scan_inputs, unwrapped_base_dim);

  const int64_t first_rank = buffers.inputs[0].dimensions().size();
  TT_ASSIGN_OR_THROW(
      const int64_t effective_dim,
      SafeWrapDim(unwrap_movedim ? *unwrapped_base_dim : dim, first_rank));

  const std::string cache_key_mlir = PrintModuleForCacheKey(body_module->get());
  const std::string serialized_mlir =
      PrintModuleWithLocations(body_module->get());

  MlirOpBuilder op_builder = [serialized_mlir, total_inputs, scan_direction,
                              effective_dim, num_scan_inputs](
                                 mlir::MlirBuilder& builder,
                                 absl::Span<mlir::MlirOp> op_inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    return BuildAssociativeScan(builder, op_inputs, serialized_mlir,
                                total_inputs, scan_direction, effective_dim,
                                num_scan_inputs);
  };

  TT_ASSIGN_OR_THROW(auto cache_keys,
                     *(OpParamCacheKeysBuilder()
                           .SetParam("op_kind", "associative_scan")
                           .SetParam("body_mlir", cache_key_mlir)
                           .SetParam("scan_direction", scan_direction)
                           .SetParam("dim", effective_dim)
                           .SetParam("num_scan_inputs", num_scan_inputs)));

  TT_ASSIGN_OR_THROW(
      const auto result_refs,
      internal::CreateDeferredDeviceBufferList(internal::DeferredOpParams{
          .op_name = OpName::kCustomKernel,
          .op_builder = std::move(op_builder),
          .op_param_cache_keys = std::move(cache_keys),
          .inputs = std::move(buffers.inputs),
          .output_shapes = std::move(buffers.output_shapes),
      }));

  std::vector<at::Tensor> results;
  results.reserve(result_refs.size());
  for (size_t i = 0; i < result_refs.size(); ++i) {
    at::Tensor out = MakeTensor(result_refs[i]);
    if (unwrap_movedim) {
      out.unsafeGetTensorImpl()->set_sizes_and_strides(
          inputs[i].sizes(), inputs[i].strides(), /*storage_offset=*/0);
    }
    results.push_back(std::move(out));
  }
  return results;
}

}  // namespace torch_tpu
