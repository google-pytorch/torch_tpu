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

#include "csrc/internal/compile/dispatch_invoke_subgraph.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fingerprint_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_assignment.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/mlir/utils/error_util.h"

namespace torch_tpu {
namespace {

// Makes a valid prefix for an MLIR symbol name from an identifier.
std::string SanitizeIdentifier(const std::string_view identifier) {
  if (identifier.empty()) {
    return "invoke_subgraph";
  }
  std::string sanitized;
  sanitized.reserve(identifier.size() + 1);
  if (absl::ascii_isdigit(identifier.front())) {
    sanitized.push_back('_');
  }
  for (const char ch : identifier) {
    sanitized.push_back((absl::ascii_isalnum(ch) || ch == '_') ? ch : '_');
  }
  return sanitized;
}

// Converts an `@main` argument MLIR tensor type into a `Shape`, or returns an
// error for unranked or unbounded types.
absl::StatusOr<Shape> ArgumentShapeFromMlirType(const mlir::Type type) {
  const auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
  // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
      ranked != nullptr, error::kInvalidArgument)
      << "must be a ranked tensor type";
  TT_ASSIGN_OR_RETURN(const mlir::ElementType dtype,
                      ConvertTo<mlir::ElementType>(ranked.getElementType()));
  if (ranked.hasStaticShape()) {
    return Shape(CopyIntVector(ranked.getShape()), dtype);
  }
  const auto ext = mlir::dyn_cast_or_null<mlir::stablehlo::TypeExtensionsAttr>(
      ranked.getEncoding());
  // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
      ext != nullptr, error::kInvalidArgument)
      << "has unbounded dynamic dimensions, which are not supported";
  const mlir::ArrayRef<int64_t> bounds = ext.getBounds();
  ABSL_CHECK_EQ(bounds.size(), ranked.getRank());  // CRASH_OK=Verified by
                                                   // StableHLO parser.
  Dimensions dims;
  dims.reserve(ranked.getRank());
  absl::InlinedVector<BoundedDynamicDimension, 1> dynamic_dims;
  for (int64_t d = 0; d < ranked.getRank(); ++d) {
    if (!ranked.isDynamicDim(d)) {
      dims.push_back(ranked.getDimSize(d));
    } else {
      const int64_t bound = bounds[d];
      // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
          !mlir::ShapedType::isDynamic(bound), error::kInvalidArgument)
          << "has an unbounded dynamic dimension at axis " << d;
      // StableHLO type extensions only carry upper bounds, so dynamic
      // dimensions default to lower_bound = 2. TorchTPU requires
      // lower_bound >= 2 (and lower_bound <= upper_bound) because 0/1-sized
      // dimensions are treated specially (e.g., empty/no-data or
      // broadcastable dimensions).
      // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
          bound >= 2, error::kInvalidArgument)
          << "expected bounded dynamic dimension upper bound at axis " << d
          << " to be >= 2, got " << bound;
      dims.push_back(bound);
      dynamic_dims.push_back(BoundedDynamicDimension{
          .dimension = d, .lower_bound = 2, .upper_bound = bound});
    }
  }
  return Shape(std::move(dims), dtype, std::move(dynamic_dims));
}

// Converts an `@main` result MLIR tensor type into a static `Shape`, or returns
// an error for unranked or dynamic result types.
absl::StatusOr<Shape> OutputShapeFromMlirType(const mlir::Type type) {
  const auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
  // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
      ranked != nullptr, error::kInvalidArgument)
      << "must be a ranked tensor type";
  // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
      ranked.hasStaticShape(), error::kInvalidArgument)
      << "has dynamic dimensions, which are not supported";
  TT_ASSIGN_OR_RETURN(const mlir::ElementType dtype,
                      ConvertTo<mlir::ElementType>(ranked.getElementType()));
  return Shape(CopyIntVector(ranked.getShape()), dtype);
}

// Returns the upper bound of dimension `dim` if it is dynamic in `shape`, or
// `std::nullopt`.
std::optional<int64_t> FindDynamicUpperBound(const Shape& shape,
                                             const int64_t dim) {
  for (const BoundedDynamicDimension& dyn_dim : shape.dynamic_dimensions()) {
    if (dyn_dim.dimension == dim) {
      return dyn_dim.upper_bound;
    }
  }
  return std::nullopt;
}

// Checks that an input `Shape` matches the dtype, rank, and static or bounded
// dimensions of an MLIR type.
absl::Status ValidateInputShapeMatches(const Shape& actual,
                                       const mlir::Type type) {
  TT_ASSIGN_OR_RETURN(
      const Shape expected, ArgumentShapeFromMlirType(type),
      _.SetPrepend() << "failed to parse subgraph_module @main argument: ");
  // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
      actual.dtype() == expected.dtype(), error::kInvalidArgument)
      << "expected dtype to be " << ToString(expected.dtype()) << ", got "
      << ToString(actual.dtype());
  // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
      actual.dimensions().size() == expected.dimensions().size(),
      error::kInvalidArgument)
      << "expected rank to be " << expected.dimensions().size() << ", got "
      << actual.dimensions().size();

  if (expected.dynamic_dimensions().empty()) {
    // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
        actual.dynamic_dimensions().empty(), error::kInvalidArgument)
        << "expected shape to be static " << ToString(expected.dimensions())
        << ", got dynamic shape";
    // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
        actual.dimensions() == expected.dimensions(), error::kInvalidArgument)
        << "expected shape to be " << ToString(expected.dimensions())
        << ", got " << ToString(actual.dimensions());
    return absl::OkStatus();
  }

  for (int64_t d = 0; d < static_cast<int64_t>(actual.dimensions().size());
       ++d) {
    const std::optional<int64_t> actual_dynamic_ub =
        FindDynamicUpperBound(actual, d);
    const std::optional<int64_t> expected_dynamic_ub =
        FindDynamicUpperBound(expected, d);
    if (!expected_dynamic_ub.has_value()) {
      // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
          !actual_dynamic_ub.has_value(), error::kInvalidArgument)
          << "expected dimension " << d << " to be static "
          << expected.dimensions()[d]
          << ", got dynamic dimension with upper bound " << *actual_dynamic_ub;
      // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
          actual.dimensions()[d] == expected.dimensions()[d],
          error::kInvalidArgument)
          << "expected dimension " << d << " to be " << expected.dimensions()[d]
          << ", got " << actual.dimensions()[d];
    } else {
      const int64_t actual_bound =
          actual_dynamic_ub.value_or(actual.dimensions()[d]);
      const int64_t max_actual = std::max(actual.dimensions()[d], actual_bound);
      // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
          max_actual <= *expected_dynamic_ub, error::kInvalidArgument)
          << "expected dimension " << d << " to be at most "
          << *expected_dynamic_ub << ", got " << max_actual;
    }
  }
  return absl::OkStatus();
}

mlir::ArrayRef<int64_t> GetBoundsOrEmpty(const mlir::RankedTensorType type) {
  const auto ext = mlir::dyn_cast_or_null<mlir::stablehlo::TypeExtensionsAttr>(
      type.getEncoding());
  return ext != nullptr ? ext.getBounds() : mlir::ArrayRef<int64_t>{};
}

int64_t GetDimUpperBound(const mlir::RankedTensorType type,
                         const mlir::ArrayRef<int64_t> bounds,
                         const int64_t d) {
  return type.isDynamicDim(d) ? bounds[d] : type.getDimSize(d);
}

bool CanAdaptDimension(const mlir::RankedTensorType actual_ranked,
                       const mlir::ArrayRef<int64_t> actual_bounds,
                       const mlir::RankedTensorType expected_ranked,
                       const mlir::ArrayRef<int64_t> expected_bounds,
                       const int64_t d) {
  if (!expected_ranked.isDynamicDim(d)) {
    return !actual_ranked.isDynamicDim(d) &&
           actual_ranked.getDimSize(d) == expected_ranked.getDimSize(d);
  }
  const int64_t expected_ub = expected_bounds[d];
  const int64_t actual_ub = GetDimUpperBound(actual_ranked, actual_bounds, d);
  return !mlir::ShapedType::isDynamic(expected_ub) &&
         !mlir::ShapedType::isDynamic(actual_ub) && actual_ub <= expected_ub;
}

// Checks if `actual_type` can convert to `expected_type` with `stablehlo.pad`
// and `SetDimensionSize`.
bool CanAdaptInputType(const mlir::Type actual_type,
                       const mlir::Type expected_type) {
  if (actual_type == expected_type) {
    return true;
  }
  const auto actual_ranked =
      mlir::dyn_cast<mlir::RankedTensorType>(actual_type);
  const auto expected_ranked =
      mlir::dyn_cast<mlir::RankedTensorType>(expected_type);
  if (actual_ranked == nullptr || expected_ranked == nullptr ||
      actual_ranked.getElementType() != expected_ranked.getElementType() ||
      actual_ranked.getRank() != expected_ranked.getRank() ||
      expected_ranked.hasStaticShape()) {
    return false;
  }
  const mlir::ArrayRef<int64_t> expected_bounds =
      GetBoundsOrEmpty(expected_ranked);
  const mlir::ArrayRef<int64_t> actual_bounds = GetBoundsOrEmpty(actual_ranked);
  if (expected_bounds.size() != expected_ranked.getRank() ||
      (!actual_ranked.hasStaticShape() &&
       actual_bounds.size() != actual_ranked.getRank())) {
    return false;
  }
  for (int64_t d = 0; d < expected_ranked.getRank(); ++d) {
    if (!CanAdaptDimension(actual_ranked, actual_bounds, expected_ranked,
                           expected_bounds, d)) {
      return false;
    }
  }
  return true;
}

// Strips dynamic bounds from `input` and pads it up to `expected_bounds` along
// any dynamic dimension whose upper bound is smaller than `expected_bounds`.
// Returns `std::nullopt` if no dimension requires padding.
std::optional<mlir::MlirOp> PadInputToExpectedBounds(
    mlir::MlirBuilder& builder, mlir::MlirOp input,
    const mlir::RankedTensorType actual_ranked,
    const mlir::ArrayRef<int64_t> actual_bounds,
    const mlir::RankedTensorType expected_ranked,
    const mlir::ArrayRef<int64_t> expected_bounds) {
  const int64_t rank = expected_ranked.getRank();
  llvm::SmallVector<int64_t> high_padding(rank, 0);
  bool needs_pad = false;
  for (int64_t d = 0; d < rank; ++d) {
    if (expected_ranked.isDynamicDim(d)) {
      const int64_t pad_amount =
          expected_bounds[d] -
          GetDimUpperBound(actual_ranked, actual_bounds, d);
      if (pad_amount > 0) {
        high_padding[d] = pad_amount;
        needs_pad = true;
      }
    }
  }
  if (!needs_pad) {
    return std::nullopt;
  }
  // Strip existing dynamic bounds before padding so `stablehlo.pad` operates
  // on a static upper-bounded tensor.
  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  mlir::MlirOp stripped = input;
  for (int64_t d = 0; d < rank; ++d) {
    if (actual_ranked.isDynamicDim(d)) {
      mlir::MlirOp bound_const =
          MakeScalarConstant(builder, actual_bounds[d], i32_type);
      stripped = mlir::stablehlo::SetDimensionSize(stripped, bound_const, d);
    }
  }
  const llvm::SmallVector<int64_t> zero_padding(rank, 0);
  mlir::MlirOp zero_pad_value =
      MakeScalarConstant(builder, 0, actual_ranked.getElementType());
  return mlir::stablehlo::Pad(stripped, zero_pad_value, zero_padding,
                              high_padding, zero_padding);
}

// Pads an input value and sets dynamic dimension sizes so its MLIR type
// matches `expected_type` (for subgraphs exported with bounded dynamic shapes,
// e.g. via `export.exported_to_mlir` or `fx_to_mlir(...,
// use_stablehlo_bounds=True)`).
absl::StatusOr<mlir::MlirOp> AdaptInvokeSubgraphInput(
    mlir::MlirBuilder& builder, mlir::MlirOp input,
    const mlir::Type expected_type) {
  if (input.getType() == expected_type) {
    return input;
  }
  // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
      CanAdaptInputType(input.getType(), expected_type),
      error::kInvalidArgument)
      << "MLIR type " << mlir::debugString(input.getType())
      << " is incompatible with subgraph_module @main argument type "
      << mlir::debugString(expected_type)
      << "; the subgraph was exported with a static or smaller-bounded shape, "
         "but this input is dynamic";

  // CanAdaptInputType above guarantees both types are RankedTensorType.
  const auto actual_ranked =
      mlir::cast<mlir::RankedTensorType>(input.getType());
  const auto expected_ranked =
      mlir::cast<mlir::RankedTensorType>(expected_type);
  const mlir::ArrayRef<int64_t> expected_bounds =
      GetBoundsOrEmpty(expected_ranked);
  const mlir::ArrayRef<int64_t> actual_bounds = GetBoundsOrEmpty(actual_ranked);

  const std::optional<mlir::MlirOp> padded =
      PadInputToExpectedBounds(builder, input, actual_ranked, actual_bounds,
                               expected_ranked, expected_bounds);
  mlir::MlirOp adapted = padded.value_or(input);
  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  for (int64_t d = 0; d < expected_ranked.getRank(); ++d) {
    if (!expected_ranked.isDynamicDim(d)) {
      continue;
    }
    if (!actual_ranked.isDynamicDim(d)) {
      mlir::MlirOp size_const =
          MakeScalarConstant(builder, actual_ranked.getDimSize(d), i32_type);
      // When `size_const` equals the dimension's static upper bound,
      // `SetDimensionSizeOp::inferReturnTypeComponents` folds the result back
      // to a static dimension unless the size operand is hidden behind an
      // `optimization_barrier`.
      auto barrier_ops =
          mlir::stablehlo::OptimizationBarrier(builder, {size_const});
      adapted = mlir::stablehlo::SetDimensionSize(adapted, barrier_ops[0], d);
    } else if (padded.has_value()) {
      mlir::MlirOp runtime_size = mlir::stablehlo::GetDimensionSize(input, d);
      adapted = mlir::stablehlo::SetDimensionSize(adapted, runtime_size, d);
    }
  }

  ABSL_CHECK_EQ(adapted.getType(), expected_type);  // CRASH_OK=Guaranteed by
                                                    // CanAdaptInputType.
  return adapted;
}

// Converts all input values so their MLIR types match the argument types of
// `target_func`.
absl::StatusOr<llvm::SmallVector<mlir::MlirOp>> AdaptInvokeSubgraphInputs(
    mlir::MlirBuilder& builder, const absl::Span<const mlir::MlirOp> op_inputs,
    mlir::func::FuncOp target_func) {
  ABSL_CHECK_EQ(  // CRASH_OK=Validated at dispatch time.
      target_func.getNumArguments(), op_inputs.size());
  const mlir::ArrayRef<mlir::Type> arg_types = target_func.getArgumentTypes();
  llvm::SmallVector<mlir::MlirOp> adapted_inputs;
  adapted_inputs.reserve(op_inputs.size());
  for (size_t i = 0; i < op_inputs.size(); ++i) {
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp adapted,
        AdaptInvokeSubgraphInput(builder, op_inputs[i], arg_types[i]),
        _.SetPrepend() << "input " << i << " ");
    adapted_inputs.push_back(adapted);
  }
  return adapted_inputs;
}

// Checks that all result types of `func` match `expected_output_shapes`.
bool FuncOpResultsMatch(mlir::func::FuncOp func,
                        const absl::Span<const Shape> expected_output_shapes) {
  if (func.getNumResults() != expected_output_shapes.size()) {
    return false;
  }
  const mlir::ArrayRef<mlir::Type> result_types = func.getResultTypes();
  for (size_t i = 0; i < expected_output_shapes.size(); ++i) {
    const absl::StatusOr<Shape> result_shape =
        OutputShapeFromMlirType(result_types[i]);
    if (!result_shape.ok() || *result_shape != expected_output_shapes[i]) {
      return false;
    }
  }
  return true;
}

// Checks if `func` in the destination module is a private definition that
// matches the current inputs and expected outputs.
bool CanReuseExistingFunc(
    mlir::func::FuncOp func, const absl::Span<const mlir::MlirOp> op_inputs,
    const absl::Span<const Shape> expected_output_shapes) {
  if (!func.isPrivate() || func.isDeclaration() ||
      func.getNumArguments() != op_inputs.size() ||
      !FuncOpResultsMatch(func, expected_output_shapes)) {
    return false;
  }
  const mlir::ArrayRef<mlir::Type> arg_types = func.getArgumentTypes();
  for (size_t i = 0; i < op_inputs.size(); ++i) {
    if (!CanAdaptInputType(op_inputs[i].getType(), arg_types[i])) {
      return false;
    }
  }
  return true;
}

// Renames colliding symbols in `parsed_module`, renames `parsed_main` to
// `entry_name` (or a unique fallback name if `entry_name` is already taken in
// `dst_module`), and clones all symbols into `dst_module`.
// TODO: b/556849778 - Consolidate symbol-table cloning and renaming with
// MoveCustomMlirKernel, ImportStepFunctionCallees, and
// InjectClonedBodyFunction.
mlir::func::FuncOp ImportSubgraphSymbols(mlir::ModuleOp dst_module,
                                         mlir::ModuleOp parsed_module,
                                         mlir::func::FuncOp parsed_main,
                                         const std::string_view entry_name) {
  mlir::SymbolTableCollection symbol_tables;
  mlir::SymbolTable& dst_st = symbol_tables.getSymbolTable(dst_module);
  mlir::SymbolTable& src_st = symbol_tables.getSymbolTable(parsed_module);

  const llvm::SmallVector<mlir::SymbolOpInterface> src_syms(
      parsed_module.getOps<mlir::SymbolOpInterface>());

  // Rename non-entry symbols before `@main` if any symbol happened to collide
  // with `entry_name` or with symbols already in `dst_module`.
  for (mlir::SymbolOpInterface sym : src_syms) {
    if (sym.getOperation() == parsed_main.getOperation()) {
      continue;
    }
    if (sym.getName() == mlir::StringRef(entry_name) ||
        dst_st.lookup(sym.getName()) != nullptr) {
      ABSL_CHECK(  // CRASH_OK=Helper symbol references stay local to
                   // parsed_module.
          mlir::succeeded(src_st.renameToUnique(sym.getOperation(), &dst_st)));
    }
  }

  ABSL_CHECK(  // CRASH_OK=entry_name is absent from helpers.
      mlir::succeeded(src_st.rename(parsed_main, entry_name)));
  if (dst_st.lookup(entry_name) != nullptr) {
    ABSL_CHECK(  // CRASH_OK=Symbol uniquing on parsed ModuleOp cannot fail.
        mlir::succeeded(src_st.renameToUnique(parsed_main, &dst_st)));
  }

  mlir::func::FuncOp cloned_entry = nullptr;
  for (mlir::SymbolOpInterface sym : src_syms) {
    mlir::Operation* const cloned = sym->clone();
    if (auto cloned_func = mlir::dyn_cast<mlir::func::FuncOp>(cloned)) {
      cloned_func.setVisibility(mlir::SymbolTable::Visibility::Private);
    }
    dst_st.insert(cloned);
    if (sym.getOperation() == parsed_main.getOperation()) {
      cloned_entry = mlir::cast<mlir::func::FuncOp>(cloned);
    }
  }
  ABSL_CHECK(cloned_entry != nullptr);  // CRASH_OK=parsed_main is in src_syms.
  return cloned_entry;
}

// Inserts the subgraph into the destination module and adds a `func.call`
// operation. Reuses a private function in the module that matches the signature
// when available.
absl::StatusOr<DynamicMlirOpResults> BuildInvokeSubgraph(
    mlir::MlirBuilder& builder, const absl::Span<const mlir::MlirOp> op_inputs,
    const std::string_view serialized_mlir, const std::string_view entry_name,
    const absl::Span<const Shape> expected_output_shapes) {
  mlir::ModuleOp dst_module = GetModuleOp(builder);

  // Reuse the already-injected private function if this exact subgraph was
  // previously lowered into `dst_module`.
  mlir::Operation* const existing_sym = dst_module.lookupSymbol(entry_name);
  if (auto existing_func =
          mlir::dyn_cast_or_null<mlir::func::FuncOp>(existing_sym);
      existing_func != nullptr &&
      CanReuseExistingFunc(existing_func, op_inputs, expected_output_shapes)) {
    TT_ASSIGN_OR_RETURN(
        const llvm::SmallVector<mlir::MlirOp> adapted_inputs,
        AdaptInvokeSubgraphInputs(builder, op_inputs, existing_func));
    return mlir::func::Call(builder, existing_func, adapted_inputs);
  }

  // Parse into `builder.getContext()` so imported ops and types belong to the
  // destination MLIRContext while preserving debug locations.
  mlir::BaseScopedDiagnosticHandler diag_handler(&builder.getContext());
  mlir::OwningOpRef<mlir::ModuleOp> parsed_module =
      mlir::parseSourceString<mlir::ModuleOp>(
          serialized_mlir, mlir::ParserConfig{&builder.getContext()});
  ABSL_CHECK(parsed_module.get() != nullptr)  // CRASH_OK=serialized_mlir was
                                              // serialized from a valid module.
      << "failed to parse serialized subgraph module: "
      << ConsumeMlirDiagnostics(diag_handler);

  mlir::func::FuncOp parsed_main =
      parsed_module->lookupSymbol<mlir::func::FuncOp>("main");
  ABSL_CHECK(parsed_main != nullptr);  // CRASH_OK=Verified at dispatch time.
  ABSL_CHECK(  // CRASH_OK=Validated from @main at dispatch time.
      FuncOpResultsMatch(parsed_main, expected_output_shapes));

  TT_ASSIGN_OR_RETURN(
      const llvm::SmallVector<mlir::MlirOp> adapted_inputs,
      AdaptInvokeSubgraphInputs(builder, op_inputs, parsed_main));

  mlir::func::FuncOp cloned_entry = ImportSubgraphSymbols(
      dst_module, *parsed_module, parsed_main, entry_name);

  return mlir::func::Call(builder, cloned_entry, adapted_inputs);
}

}  // namespace

// Creates a deferred invoke_subgraph operation and returns its output tensors.
std::vector<at::Tensor> PyCreateInvokeSubgraphOp(
    const std::vector<at::Tensor>& inputs,
    const std::shared_ptr<ContextedModule>& subgraph_module,
    const std::string_view identifier) {
  ScopedPythonContextCapturer capturer(OpName::kTorchTpuInternalInvokeSubgraph);

  // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
      subgraph_module != nullptr && subgraph_module->get() != nullptr,
      error::kInvalidArgument)
      << "subgraph_module must not be null";

  mlir::ModuleOp src_module = subgraph_module->get();
  mlir::func::FuncOp main_func =
      src_module.lookupSymbol<mlir::func::FuncOp>("main");
  // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
      main_func != nullptr && !main_func.isDeclaration(),
      error::kInvalidArgument)
      << "subgraph_module must define a @main function";

  // TODO: b/556849778 - Remove ERROR_COV_INFEASIBLE once bound to Python.
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Not yet bound to Python.
      inputs.size() == main_func.getNumArguments(), error::kInvalidArgument)
      << "expected " << main_func.getNumArguments() << " inputs, got "
      << inputs.size();

  std::vector<DeviceBufferRef> input_refs;
  input_refs.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    TT_ASSIGN_OR_THROW(DeviceBufferRef input_ref, GetBuffer(inputs[i]));
    TT_THROW_IF_ERROR(ValidateInputShapeMatches(
                          input_ref.shape(), main_func.getArgumentTypes()[i]))
            .SetPrepend()
        << "failed to validate input " << i << ": ";
    input_refs.push_back(std::move(input_ref));
  }

  std::vector<Shape> output_shapes;
  output_shapes.reserve(main_func.getNumResults());
  for (size_t i = 0; i < main_func.getNumResults(); ++i) {
    TT_ASSIGN_OR_THROW(
        Shape shape, OutputShapeFromMlirType(main_func.getResultTypes()[i]),
        _.SetPrepend() << "subgraph_module @main result " << i << " ");
    output_shapes.push_back(std::move(shape));
  }

  if (output_shapes.empty()) {
    return {};
  }

  // TODO: b/556849778 - Cache the MLIR fingerprint and serialized bytecode on
  // ContextedModule so repeated calls do not re-serialize the module.
  const FingerprintType cache_key_fp = Fingerprint(
      DebugString(src_module, DebugStringOptions::kDisableDebugInfo));
  TT_ASSIGN_OR_THROW(std::string serialized_mlir,
                     SerializeBytecode(src_module));

  // Include the device assignment in the entry symbol fingerprint so subgraphs
  // lowered under different mesh topologies within the same parent module
  // receive distinct private function symbols.
  const DeviceAssignment& da = subgraph_module->device_assignment();
  const FingerprintType entry_name_fp =
      FingerprintCat(cache_key_fp, da.local, da.global,
                     static_cast<int>(da.contains_pallas_kernel));
  std::string entry_name =
      absl::StrCat(SanitizeIdentifier(identifier), "_",
                   absl::Hex(entry_name_fp, absl::kZeroPad16));

  OpParamCacheKeysBuilder cache_keys_builder;
  cache_keys_builder.SetParam("subgraph_mlir", cache_key_fp)
      .SetParam("contains_pallas_kernel", da.contains_pallas_kernel);
  TT_ASSIGN_OR_THROW(OpParamCacheKeys op_param_cache_keys,
                     *std::move(cache_keys_builder));

  auto op_builder = [serialized_mlir = std::move(serialized_mlir),
                     entry_name = std::move(entry_name),
                     expected_output_shapes = output_shapes](
                        mlir::MlirBuilder& builder,
                        absl::Span<mlir::MlirOp> op_inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    return BuildInvokeSubgraph(builder, op_inputs, serialized_mlir, entry_name,
                               expected_output_shapes);
  };

  TT_ASSIGN_OR_THROW(
      std::vector<DeviceBufferRef> outputs,
      internal::CreateDeferredDeviceBufferList(internal::DeferredOpParams{
          .op_name = OpName::kTorchTpuInternalInvokeSubgraph,
          .op_builder = std::move(op_builder),
          .op_param_cache_keys = std::move(op_param_cache_keys),
          .inputs = std::move(input_refs),
          .output_shapes = std::move(output_shapes),
          .device_assignment = da,
      }));

  std::vector<at::Tensor> result_tensors;
  result_tensors.reserve(outputs.size());
  for (DeviceBufferRef& output : outputs) {
    result_tensors.push_back(MakeTensor(std::move(output)));
  }
  return result_tensors;
}

}  // namespace torch_tpu
