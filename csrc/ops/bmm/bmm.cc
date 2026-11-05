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

#include "csrc/ops/bmm/bmm.h"

#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/matmul/matmul.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildBmmShlo(
    mlir::MlirOp self_op, mlir::MlirOp mat2_op, mlir::ElementType out_dtype,
    mlir::stablehlo::Precision precision) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType mat2_type = GetTensorTypeOrDie(mat2_op);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Rank constraint is checked before Mlir
                 // lowering.
      self_type.getRank() == 3 && mat2_type.getRank() == 3,
      error::kInvalidArgument)
      << "BuildBmmShlo: expected 3D tensors, got ranks " << self_type.getRank()
      << " and " << mat2_type.getRank();
  return BuildMatmulShlo(self_op, mat2_op, precision,
                         MatmulShloOptions{.target_element_type = out_dtype});
}

}  // namespace torch_tpu
