/*
 * Copyright 2025 Google LLC
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

#ifndef TORCH_TPU_OPS_BINARY_H_
#define TORCH_TPU_OPS_BINARY_H_

#include "absl/status/statusor.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/macro_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

// This simple macro is used to construct functions that call the
// proper StableHLO or CHLO builder function for a given MlirOp code.
#define TT_BINARY_BUILDER_(func_name, func_call)                          \
  inline absl::StatusOr<mlir::MlirOp> func_name(mlir::MlirOp input1_op,   \
                                                mlir::MlirOp input2_op) { \
    TT_ASSIGN_OR_RETURN(                                                  \
        (auto [op1, op2]),                                                \
        ::torch_tpu::ApplyBroadcastIfNeeded(input1_op, input2_op));       \
    return func_call(op1, op2);                                           \
  }                                                                       \
  TT_REQUIRE_SEMICOLON_

// This macro is used to construct comparison functions.
#define TT_COMPARISON_BUILDER_(func_name, comparison)                     \
  inline absl::StatusOr<mlir::MlirOp> func_name(mlir::MlirOp input1_op,   \
                                                mlir::MlirOp input2_op) { \
    TT_ASSIGN_OR_RETURN(                                                  \
        (auto [op1, op2]),                                                \
        ::torch_tpu::ApplyBroadcastIfNeeded(input1_op, input2_op));       \
    return mlir::stablehlo::Compare(op1, op2, comparison);                \
  }                                                                       \
  TT_REQUIRE_SEMICOLON_

// Keep the following functions definitions in sync with the entries
// in binary_aten_kernels.cc.
// go/keep-sorted start
TT_BINARY_BUILDER_(BuildAtan2Shlo, mlir::stablehlo::Atan2);
TT_BINARY_BUILDER_(BuildBitwiseAndShlo, mlir::stablehlo::And);
TT_BINARY_BUILDER_(BuildBitwiseLeftShiftShlo, mlir::stablehlo::ShiftLeft);
TT_BINARY_BUILDER_(BuildBitwiseOrShlo, mlir::stablehlo::Or);
TT_BINARY_BUILDER_(BuildBitwiseXorShlo, mlir::stablehlo::Xor);
TT_BINARY_BUILDER_(BuildDivShlo, mlir::stablehlo::Div);
TT_BINARY_BUILDER_(BuildFmodTensorShlo, mlir::stablehlo::Rem);
TT_BINARY_BUILDER_(BuildMaximumShlo, mlir::stablehlo::Max);
TT_BINARY_BUILDER_(BuildMinimumShlo, mlir::stablehlo::Min);
TT_BINARY_BUILDER_(BuildMulShlo, mlir::stablehlo::Mul);
TT_BINARY_BUILDER_(BuildPowShlo, mlir::stablehlo::Pow);
TT_BINARY_BUILDER_(BuildSubShlo, mlir::stablehlo::Subtract);
TT_COMPARISON_BUILDER_(BuildEqShlo, mlir::stablehlo::ComparisonDirection::EQ);
TT_COMPARISON_BUILDER_(BuildGeShlo, mlir::stablehlo::ComparisonDirection::GE);
TT_COMPARISON_BUILDER_(BuildGtShlo, mlir::stablehlo::ComparisonDirection::GT);
TT_COMPARISON_BUILDER_(BuildLeShlo, mlir::stablehlo::ComparisonDirection::LE);
TT_COMPARISON_BUILDER_(BuildLtShlo, mlir::stablehlo::ComparisonDirection::LT);
TT_COMPARISON_BUILDER_(BuildNeShlo, mlir::stablehlo::ComparisonDirection::NE);
// go/keep-sorted end

// This macro is not needed anywhere else.
#undef TT_BINARY_BUILDER_
#undef TT_COMPARISON_BUILDER_

// For functions that require more than a single-line StableHlo builder
// call sequence, their prototypes will reside here.

// Special case `Add` lowering for boolean addition. XLA has ambiguity between
// `i1` integer addition that can overflow, and `pred` boolean addition where
// True+True=True. StableHLO always means the latter, so we lower these to
// `stablehlo.or` to force the proper semantics.
absl::StatusOr<mlir::MlirOp> BuildAddShlo(mlir::MlirOp lhs_op,
                                          mlir::MlirOp rhs_op);

absl::StatusOr<mlir::MlirOp> BuildPolarShlo(mlir::MlirOp abs_op,
                                            mlir::MlirOp angle_op);

absl::StatusOr<mlir::MlirOp> BuildComplexShlo(mlir::MlirOp real_op,
                                              mlir::MlirOp imag_op);

// PyTorch expects that signed integers will use an arithmetic right shift
// (preserving the sign bit) but unsigned integers use a logical right shift.
absl::StatusOr<mlir::MlirOp> BuildBitwiseRightShiftShlo(mlir::MlirOp lhs_op,
                                                        mlir::MlirOp rhs_op);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_BINARY_H_
