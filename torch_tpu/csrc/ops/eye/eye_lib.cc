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

#include "torch_tpu/csrc/ops/eye/eye_lib.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildEyeShlo(mlir::MlirBuilder& builder,
                                          mlir::ElementType element_type,
                                          int64_t m, int64_t n) {
  const auto iota_type =
      mlir::RankedTensorType::get({m, n}, builder.getOpBuilder().getI32Type());

  // Row indices:    0 0 0 ...
  //                 1 1 1 ...
  //                 2 2 2 ...
  //                 ...
  mlir::MlirOp rows =
      mlir::stablehlo::Iota(builder, iota_type, /*iota_dimension=*/0);

  // Column indices: 0 1 2 ...
  //                 0 1 2 ...
  //                 0 1 2 ...
  //                 ...
  mlir::MlirOp cols =
      mlir::stablehlo::Iota(builder, iota_type, /*iota_dimension=*/1);

  const auto is_diagonal = mlir::stablehlo::Compare(
      rows, cols, mlir::stablehlo::ComparisonDirection::EQ);

  return mlir::stablehlo::ConvertElementType(is_diagonal, element_type);
}

}  // namespace torch_tpu
