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

#include "torch_tpu/ops/dynamic/dynamic_arange/dynamic_arange.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildDynamicArange(
    mlir::MlirBuilder& builder, mlir::MlirOp start_op, mlir::MlirOp end_op,
    mlir::MlirOp step_op, mlir::ElementType element_type, int64_t max_length) {
  TT_ASSIGN_OR_RETURN(const mlir::Type mlir_type,
                      GetMlirType(builder.getContext(), element_type));

  // Cast inputs to the requested target element type
  TT_ASSIGN_OR_RETURN(mlir::MlirOp start_cast,
                      CastIfNeeded(start_op, element_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp end_cast,
                      CastIfNeeded(end_op, element_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp step_cast,
                      CastIfNeeded(step_op, element_type));

  mlir::RankedTensorType output_type =
      mlir::RankedTensorType::get({max_length}, mlir_type);

  // Broadcast start and step to 1D [max_length]
  mlir::MlirOp broadcasted_start =
      mlir::stablehlo::BroadcastInDim(output_type, start_cast, {});
  mlir::MlirOp broadcasted_step =
      mlir::stablehlo::BroadcastInDim(output_type, step_cast, {});

  // Generate full static iota sequence
  mlir::MlirOp iota = mlir::stablehlo::Iota(builder, output_type, 0);

  // Compute values: iota * step + start
  mlir::MlirOp iota_step = mlir::stablehlo::Mul(iota, broadcasted_step);
  mlir::MlirOp padded_arange =
      mlir::stablehlo::Add(iota_step, broadcasted_start);

  // Compute dynamic length
  mlir::MlirOp dynamic_length;
  if (mlir_type.isInteger()) {
    // Formula: (end - start + step - sgn) / step
    // sgn = (step > 0) - (step < 0)
    mlir::MlirOp diff = mlir::stablehlo::Subtract(end_cast, start_cast);

    mlir::MlirOp zero_i64 = mlir::stablehlo::Constant(builder, (int64_t)0);
    mlir::MlirOp zero =
        mlir::stablehlo::ConvertElementType(zero_i64, mlir_type);

    mlir::MlirOp gt = mlir::stablehlo::Compare(
        step_cast, zero, mlir::stablehlo::ComparisonDirection::GT);
    mlir::MlirOp lt = mlir::stablehlo::Compare(
        step_cast, zero, mlir::stablehlo::ComparisonDirection::LT);

    mlir::MlirOp gt_int = mlir::stablehlo::ConvertElementType(gt, mlir_type);
    mlir::MlirOp lt_int = mlir::stablehlo::ConvertElementType(lt, mlir_type);
    mlir::MlirOp sgn = mlir::stablehlo::Subtract(gt_int, lt_int);

    mlir::MlirOp term1 = mlir::stablehlo::Add(diff, step_cast);
    mlir::MlirOp term2 = mlir::stablehlo::Subtract(term1, sgn);
    dynamic_length = mlir::stablehlo::Div(term2, step_cast);
  } else {
    // Formula: ceil((end - start) / step)
    mlir::MlirOp diff = mlir::stablehlo::Subtract(end_cast, start_cast);
    mlir::MlirOp div = mlir::stablehlo::Div(diff, step_cast);
    dynamic_length = mlir::stablehlo::Ceil(div);
  }

  dynamic_length = mlir::stablehlo::ConvertElementType(
      dynamic_length, builder.getOpBuilder().getI32Type());

  // Ensure that the dynamic length is at least 0 ( to handle case when end <
  // start and step > 0)
  mlir::MlirOp zero_i32 =
      MakeScalarConstant(builder, 0, builder.getOpBuilder().getI32Type());
  dynamic_length = mlir::stablehlo::Max(dynamic_length, zero_i32);

  return mlir::stablehlo::SetDimensionSize(padded_arange, dynamic_length,
                                           0 /* dimension */);
}

at::Tensor DynamicArange(const at::Tensor& start, const at::Tensor& end,
                         const at::Tensor& step, int64_t max_length,
                         at::ScalarType dtype) {
  TT_KERNEL(
      OpName::kDynamicArange, param_keys, (start, end, step, max_length, dtype),
      {
        TT_CHECK_THROW(start.dim() == 0, error::kInvalidArgument)
            << "expected a 0-dimensional tensor for start, got " << start.dim()
            << "-dimensional tensor";
        TT_CHECK_THROW(end.dim() == 0, error::kInvalidArgument)
            << "expected a 0-dimensional tensor for end, got " << end.dim()
            << "-dimensional tensor";
        TT_CHECK_THROW(step.dim() == 0, error::kInvalidArgument)
            << "expected a 0-dimensional tensor for step, got " << step.dim()
            << "-dimensional tensor";

        TT_CHECK_THROW(c10::isIntegralType(dtype, /*includeBool=*/false) ||
                           c10::isFloatingType(dtype),
                       error::kInvalidArgument)
            << "expected float or int dtype, got " << ToString(dtype);

        TT_ASSIGN_OR_THROW(const mlir::ElementType element_type,
                           ConvertTo<mlir::ElementType>(dtype));

        Dimensions out_dims = {max_length};

        auto builder_fn = [max_length,
                           element_type](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildDynamicArange(inputs[0].getBuilder(), inputs[0],
                                    inputs[1], inputs[2], element_type,
                                    max_length);
        };

        TT_ASSIGN_OR_THROW(
            auto result_buf,
            DispatchOp<3>(std::move(builder_fn), {start, end, step},
                          {.out_dtype = element_type,
                           .out_dims = out_dims,
                           .op_param_cache_keys = std::move(param_keys)}));
        return MakeTensor(std::move(result_buf));
      });
}

}  // namespace torch_tpu
