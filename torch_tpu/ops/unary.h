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

#ifndef TORCH_TPU_OPS_UNARY_H_
#define TORCH_TPU_OPS_UNARY_H_

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/macro_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

// This simple macro is used to construct functions that call the
// proper StableHLO or CHLO builder function for a given MlirOp code.
#define TT_UNARY_BUILDER_(FctName, FctCall)                            \
  inline absl::StatusOr<mlir::MlirOp> FctName(mlir::MlirOp input_op) { \
    return FctCall(input_op);                                          \
  }                                                                    \
  TT_REQUIRE_SEMICOLON_

// This simple macro is used to construct functions that call the
// proper StableHLO or CHLO builder function for a given MlirOp code
// for ops that require floating-point outputs.
#define TT_UNARY_BUILDER_FP_ONLY_(FctName, FctCall)                       \
  inline absl::StatusOr<mlir::MlirOp> FctName(                            \
      mlir::MlirOp input_op, mlir::ElementType out_mlir_type) {           \
    TT_ASSIGN_OR_RETURN(/* ERROR_COV_INFEASIBLE=in a macro definition. */ \
                        mlir::MlirOp op, ::torch_tpu::ConvertIfInteger(   \
                                             input_op, out_mlir_type));   \
    return FctCall(op);                                                   \
  }                                                                       \
  TT_REQUIRE_SEMICOLON_

// Keep the following functions definitions in sync with the entries
// in unary_aten_kernels.cc and make sure all of these functions are
// being registered in tpu_aten_kernels.cc.
// go/keep-sorted start
TT_UNARY_BUILDER_(BuildCeilShlo, mlir::stablehlo::Ceil);
TT_UNARY_BUILDER_(BuildFloorShlo, mlir::stablehlo::Floor);
TT_UNARY_BUILDER_(BuildNegShlo, mlir::stablehlo::Neg);
TT_UNARY_BUILDER_(BuildNotShlo, mlir::stablehlo::Not);
TT_UNARY_BUILDER_FP_ONLY_(BuildAcosShlo, mlir::chlo::Acos);
TT_UNARY_BUILDER_FP_ONLY_(BuildAcoshShlo, mlir::chlo::Acosh);
TT_UNARY_BUILDER_FP_ONLY_(BuildAsinShlo, mlir::chlo::Asin);
TT_UNARY_BUILDER_FP_ONLY_(BuildAsinhShlo, mlir::chlo::Asinh);
TT_UNARY_BUILDER_FP_ONLY_(BuildAtanShlo, mlir::chlo::Atan);
TT_UNARY_BUILDER_FP_ONLY_(BuildAtanhShlo, mlir::chlo::Atanh);
TT_UNARY_BUILDER_FP_ONLY_(BuildCosShlo, mlir::stablehlo::Cosine);
TT_UNARY_BUILDER_FP_ONLY_(BuildCoshShlo, mlir::chlo::Cosh);
TT_UNARY_BUILDER_FP_ONLY_(BuildErfInvShlo, mlir::chlo::ErfInv);
TT_UNARY_BUILDER_FP_ONLY_(BuildErfShlo, mlir::chlo::Erf);
TT_UNARY_BUILDER_FP_ONLY_(BuildExpShlo, mlir::stablehlo::Exp);
TT_UNARY_BUILDER_FP_ONLY_(BuildExpm1Shlo, mlir::stablehlo::Expm1);
TT_UNARY_BUILDER_FP_ONLY_(BuildLgammaShlo, mlir::chlo::Lgamma);
TT_UNARY_BUILDER_FP_ONLY_(BuildRsqrtShlo, mlir::stablehlo::Rsqrt);
TT_UNARY_BUILDER_FP_ONLY_(BuildSinShlo, mlir::stablehlo::Sine);
TT_UNARY_BUILDER_FP_ONLY_(BuildSinhShlo, mlir::chlo::Sinh);
TT_UNARY_BUILDER_FP_ONLY_(BuildSqrtShlo, mlir::stablehlo::Sqrt);
TT_UNARY_BUILDER_FP_ONLY_(BuildTanShlo, mlir::stablehlo::Tan);
TT_UNARY_BUILDER_FP_ONLY_(BuildTanhShlo, mlir::stablehlo::Tanh);
// go/keep-sorted end

// This macro is not needed anywhere else.
#undef TT_UNARY_BUILDER_

// For functions that require more than a single-line StableHlo builder
// call sequence, their prototypes will reside here.
absl::StatusOr<mlir::MlirOp> BuildAbsShlo(mlir::MlirOp input);
absl::StatusOr<mlir::MlirOp> BuildConjPhysicalShlo(mlir::MlirOp input_op);
absl::StatusOr<mlir::MlirOp> BuildReciprocalShlo(
    mlir::MlirOp input_op, mlir::ElementType default_mlir_type);
absl::StatusOr<mlir::MlirOp> BuildReluShlo(mlir::MlirOp input_op);
absl::StatusOr<mlir::MlirOp> BuildSgnShlo(mlir::MlirOp input);
absl::StatusOr<mlir::MlirOp> BuildSignShlo(mlir::MlirOp input_op);
absl::StatusOr<mlir::MlirOp> BuildSiluShlo(mlir::MlirOp input_op);
absl::StatusOr<mlir::MlirOp> BuildTruncShlo(mlir::MlirOp input_op);
absl::StatusOr<mlir::MlirOp> BuildLiftFreshShlo(mlir::MlirOp input_op);
absl::StatusOr<mlir::MlirOp> BuildExp2Shlo(mlir::MlirOp input_op,
                                           mlir::ElementType default_mlir_type);
absl::StatusOr<mlir::MlirOp> BuildLogShlo(mlir::MlirOp input_op,
                                          mlir::ElementType default_dtype);
absl::StatusOr<mlir::MlirOp> BuildLog1pShlo(mlir::MlirOp input_op,
                                            mlir::ElementType default_dtype);
absl::StatusOr<mlir::MlirOp> BuildLog2Shlo(mlir::MlirOp input_op,
                                           mlir::ElementType default_mlir_type);
absl::StatusOr<mlir::MlirOp> BuildLog10Shlo(
    mlir::MlirOp input_op, mlir::ElementType default_mlir_type);
absl::StatusOr<mlir::MlirOp> BuildErfcShlo(mlir::MlirOp input,
                                           mlir::ElementType out_mlir_type);
absl::StatusOr<mlir::MlirOp> BuildFracShlo(mlir::MlirOp input);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_UNARY_H_
