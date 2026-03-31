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

#include "torch_tpu/ops/addmv/addmv.h"

#include <cstdint>
#include <utility>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildAddmvShlo(
    mlir::MlirOp self, mlir::MlirOp mat, mlir::MlirOp vec, mlir::MlirOp beta,
    mlir::MlirOp alpha, mlir::ElementType computation_dtype) {
  TT_ASSIGN_OR_RETURN(
      (auto [self_bcast, mat_bcast, vec_bcast, beta_bcast, alpha_bcast]),
      ApplyBroadcastIfNeeded(self, mat, vec, beta, alpha));

  // mat @ vec
  auto dot_dimension_numbers = stablehlo::DotDimensionNumbersAttr::get(
      &mat_bcast.getContext(),
      /*lhs_batch_dimensions=*/{},
      /*rhs_batch_dimensions=*/{},
      /*lhs_contracting_dimensions=*/{1},
      /*rhs_contracting_dimensions=*/{0});

  mlir::MlirOp mv_result = stablehlo::DotGeneral(
      mat_bcast, vec_bcast, dot_dimension_numbers, nullptr);

  // alpha * (mat @ vec)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp alpha_mv,
                      BuildMulShlo(alpha_bcast, mv_result));

  // beta * self
  TT_ASSIGN_OR_RETURN(mlir::MlirOp beta_self,
                      BuildMulShlo(beta_bcast, self_bcast));

  // (beta * self) + (alpha * (mat @ vec))
  return BuildAddShlo(beta_self, alpha_mv);
}

}  // namespace torch_tpu
