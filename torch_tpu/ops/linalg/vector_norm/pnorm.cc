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

#include "torch_tpu/ops/linalg/vector_norm/pnorm.h"

#include <cmath>
#include <cstdint>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildPNormShlo(
    mlir::MlirOp input_op, double ord, absl::Span<const int64_t> reduce_dims,
    ReductionMode reduction_mode, mlir::ElementType out_type) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  mlir::MlirOp ord_op = MakeScalarConstant(builder, ord, out_type);
  return BuildPNormShlo(input_op, ord_op, ord, reduce_dims, reduction_mode,
                        out_type);
}

absl::StatusOr<mlir::MlirOp> BuildPNormShlo(
    mlir::MlirOp input_op, mlir::MlirOp ord_op, double ord,
    absl::Span<const int64_t> reduce_dims, ReductionMode reduction_mode,
    mlir::ElementType out_type) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  TT_ASSIGN_OR_RETURN(mlir::ElementType compute_dtype,
                      InferComputationDtype(out_type));
  auto compute_mlir_type =
      mlir::getElementType(builder.getContext(), compute_dtype);

  input_op = mlir::stablehlo::Abs(input_op);
  TT_ASSIGN_OR_RETURN(input_op, CastIfNeeded(input_op, compute_dtype));
  TT_ASSIGN_OR_RETURN(ord_op, CastIfNeeded(ord_op, compute_dtype));

  mlir::MlirOp init_val = MakeScalarConstant(builder, 0.0, compute_dtype);

  // 0-norm is taken care outside of this function
  ABSL_CHECK(ord != 0);  // CRASH_OK

  if (ord == INFINITY) {
    // Inf-norm is max(abs(x)).
    auto reduce_fn = [compute_mlir_type](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::MaxOp>(
          compute_mlir_type, rb.getRegion(), rb.getOpBuilder());
    };
    mlir::MlirOp reduction_res =
        BuildReductionShlo(input_op, reduce_dims, compute_mlir_type, init_val,
                           reduce_fn, reduction_mode);
    return CastIfNeeded(reduction_res, out_type);
  } else if (ord == -INFINITY) {
    // -Inf-norm min(abs(x))
    mlir::MlirOp init_val_inf =
        MakeScalarConstant(builder, INFINITY, compute_dtype);
    auto reduce_fn = [compute_mlir_type](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::MinOp>(
          compute_mlir_type, rb.getRegion(), rb.getOpBuilder());
    };
    mlir::MlirOp reduction_res =
        BuildReductionShlo(input_op, reduce_dims, compute_mlir_type,
                           init_val_inf, reduce_fn, reduction_mode);
    return CastIfNeeded(reduction_res, out_type);
  } else if (ord == 1.0) {
    // 1-norm is sum(abs(x)).
    auto reduce_fn = [compute_mlir_type](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
          compute_mlir_type, rb.getRegion(), rb.getOpBuilder());
    };
    mlir::MlirOp reduction_res =
        BuildReductionShlo(input_op, reduce_dims, compute_mlir_type, init_val,
                           reduce_fn, reduction_mode);
    return CastIfNeeded(reduction_res, out_type);
  } else if (ord == 2.0) {
    // 2-norm is sqrt(sum(abs(x)^2)).
    input_op = mlir::stablehlo::Mul(input_op, input_op);
    auto reduce_fn = [compute_mlir_type](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
          compute_mlir_type, rb.getRegion(), rb.getOpBuilder());
    };
    mlir::MlirOp reduction_res =
        BuildReductionShlo(input_op, reduce_dims, compute_mlir_type, init_val,
                           reduce_fn, reduction_mode);
    mlir::MlirOp sqrt_res = mlir::stablehlo::Sqrt(reduction_res);
    return CastIfNeeded(sqrt_res, out_type);
  } else {
    // The general p-norm is sum(abs(x)^ord)^(1/ord).
    TT_ASSIGN_OR_RETURN(input_op, BuildPowShlo(input_op, ord_op));
    auto reduce_fn = [compute_mlir_type](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
          compute_mlir_type, rb.getRegion(), rb.getOpBuilder());
    };
    mlir::MlirOp reduction_res =
        BuildReductionShlo(input_op, reduce_dims, compute_mlir_type, init_val,
                           reduce_fn, reduction_mode);

    auto one = MakeScalarConstant(builder, 1.0, compute_dtype);
    auto one_bcast =
        mlir::stablehlo::BroadcastInDim(GetTensorTypeOrDie(ord_op), one, {});
    TT_ASSIGN_OR_RETURN(auto root, BuildDivShlo(one_bcast, ord_op));
    TT_ASSIGN_OR_RETURN(auto res_root, BuildPowShlo(reduction_res, root));

    return CastIfNeeded(res_root, out_type);
  }
}

}  // namespace torch_tpu
