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
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildPNormShlo(
    mlir::MlirOp input_op, double ord, absl::Span<const int64_t> reduce_dims,
    ReductionMode reduction_mode, mlir::ElementType out_type) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  auto mlir_type = mlir::getElementType(builder.getContext(), out_type);

  input_op = mlir::stablehlo::Abs(input_op);
  mlir::MlirOp init_val = MakeScalarConstant(builder, 0.0, out_type);

  // 0-norm is taken care outside of this function
  ABSL_CHECK(ord != 0);  // CRASH_OK

  if (ord == INFINITY) {
    // Inf-norm is max(abs(x)).
    auto reduce_fn = [mlir_type](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::MaxOp>(
          mlir_type, rb.getRegion(), rb.getOpBuilder());
    };
    return BuildReductionShlo(input_op, reduce_dims, mlir_type, init_val,
                              reduce_fn, reduction_mode);
  } else if (ord == -INFINITY) {
    // -Inf-norm min(abs(x))
    mlir::MlirOp init_val = MakeScalarConstant(builder, INFINITY, out_type);
    auto reduce_fn = [mlir_type](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::MinOp>(
          mlir_type, rb.getRegion(), rb.getOpBuilder());
    };
    return BuildReductionShlo(input_op, reduce_dims, mlir_type, init_val,
                              reduce_fn, reduction_mode);
  } else if (ord == 1.0) {
    // 1-norm is sum(abs(x)).
    auto reduce_fn = [mlir_type](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
          mlir_type, rb.getRegion(), rb.getOpBuilder());
    };
    return BuildReductionShlo(input_op, reduce_dims, mlir_type, init_val,
                              reduce_fn, reduction_mode);
  } else if (ord == 2.0) {
    // 2-norm is sqrt(sum(abs(x)^2)).
    input_op = mlir::stablehlo::Mul(input_op, input_op);
    auto reduce_fn = [mlir_type](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
          mlir_type, rb.getRegion(), rb.getOpBuilder());
    };
    mlir::MlirOp reduction_res = BuildReductionShlo(
        input_op, reduce_dims, mlir_type, init_val, reduce_fn, reduction_mode);
    return mlir::stablehlo::Sqrt(reduction_res);
  } else {
    // The general p-norm is sum(abs(x)^ord)^(1/ord).
    // If the input is float16 or bfloat16, pow(x, ord) can overflow, cast it to
    // float32 and avoid this.
    // TODO: Is there a better way to do it without upcasting?
    mlir::ElementType maybe_upcasted_out_type = out_type;
    mlir::Type maybe_upcasted_mlir_type = mlir_type;

    if (out_type == mlir::ElementType::F16 ||
        out_type == mlir::ElementType::BF16) {
      maybe_upcasted_out_type = mlir::ElementType::F32;
      maybe_upcasted_mlir_type =
          mlir::getElementType(builder.getContext(), maybe_upcasted_out_type);
      init_val = mlir::stablehlo::ConvertElementType(init_val,
                                                     maybe_upcasted_out_type);
      input_op = mlir::stablehlo::ConvertElementType(input_op,
                                                     maybe_upcasted_out_type);
    }

    mlir::MlirOp pow_ord =
        MakeConstantLike(input_op, ord, maybe_upcasted_out_type);
    input_op = mlir::stablehlo::Pow(input_op, pow_ord);
    auto reduce_fn = [maybe_upcasted_mlir_type](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
          maybe_upcasted_mlir_type, rb.getRegion(), rb.getOpBuilder());
    };
    mlir::MlirOp reduction_res =
        BuildReductionShlo(input_op, reduce_dims, maybe_upcasted_mlir_type,
                           init_val, reduce_fn, reduction_mode);

    mlir::MlirOp root =
        MakeConstantLike(reduction_res, 1.0 / ord, maybe_upcasted_out_type);
    auto res_root = mlir::stablehlo::Pow(reduction_res, root);

    if (maybe_upcasted_out_type != out_type) {
      return mlir::stablehlo::ConvertElementType(res_root, out_type);
    }
    return res_root;
  }
}

}  // namespace torch_tpu
