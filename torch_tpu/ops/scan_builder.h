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

#ifndef TORCH_TPU_OPS_SCAN_BUILDER_H_
#define TORCH_TPU_OPS_SCAN_BUILDER_H_

#include <cstdint>
#include <string>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

enum class ScanDirection { kForward, kReverse };

[[nodiscard]] inline std::string FormatParamCacheKey(
    const ScanDirection value) {
  return value == ScanDirection::kForward ? "forward" : "reverse";
}

struct ScanOptions {
  ScanDirection direction = ScanDirection::kForward;
  bool should_squeeze = false;

  // When true, the scan body is asserted to be associative and the scan is
  // lowered to chlo.ScanOp(is_associative=true), which the TPU compiler's
  // native scan emitter compiles directly. When false (the default), the scan
  // is lowered to a StableHLO while loop. Only set this for genuinely
  // associative reductions (e.g. cumsum/cumprod/cummax/cummin); a
  // non-associative body under is_associative=true would be miscompiled by the
  // associative emitter.
  bool is_associative = false;
};
// (op_builder, loc, input_slice, index_0d, carries) -> new_carries.
using ScanBodyBuilder =
    absl::AnyInvocable<absl::StatusOr<llvm::SmallVector<mlir::Value>>(
        mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value,
        mlir::ValueRange) const>;

// Lowers a prefix scan along 'dim'. By default uses a StableHLO while loop; if
// 'options.is_associative' is true the scan is lowered to chlo.ScanOp for the
// native scan emitter instead (see ScanOptions).
absl::StatusOr<DynamicMlirOpResults> BuildScanShlo(
    mlir::MlirBuilder& builder, mlir::MlirOp input, int64_t dim,
    llvm::ArrayRef<mlir::MlirOp> carry_inits,
    llvm::ArrayRef<mlir::MlirOp> output_inits, ScanBodyBuilder body_builder,
    const ScanOptions& options = {});

struct ScanBodyResults {
  llvm::SmallVector<mlir::Value> new_carries;
  llvm::SmallVector<mlir::Value> new_outputs;
};

// (op_builder, loc, input_slices, index_0d, carries) -> (new_carries,
// new_outputs).
using MultiInputScanBodyBuilder =
    absl::AnyInvocable<absl::StatusOr<ScanBodyResults>(
        mlir::OpBuilder&, mlir::Location, mlir::ValueRange, mlir::Value,
        mlir::ValueRange) const>;

// Generalized version of BuildScanShlo supporting multiple inputs, separate
// carries and outputs, and reverse scanning.
//
// 'scan_inputs' contains 'num_scan_inputs' scannable tensors followed by
// additional tensors that are passed as-is to the body.
//
// If 'options.direction' is kReverse, scans the sequence dimension in reverse
// order. If 'options.should_squeeze' is true, slices along the sequence
// dimension are rank-reduced by squeezing the sequence dimension before passing
// to the body builder.
//
// If 'options.is_associative' is true, the scan is lowered to chlo.ScanOp for
// the native scan emitter instead of a while loop (see ScanOptions).
absl::StatusOr<DynamicMlirOpResults> BuildScanShlo(
    mlir::MlirBuilder& builder, llvm::ArrayRef<mlir::MlirOp> scan_inputs,
    int64_t dim, int64_t num_scan_inputs,
    llvm::ArrayRef<mlir::MlirOp> carry_inits,
    llvm::ArrayRef<mlir::MlirOp> output_inits,
    MultiInputScanBodyBuilder body_builder, const ScanOptions& options = {});

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_SCAN_BUILDER_H_
