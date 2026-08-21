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

#include "torch_tpu/csrc/ops/angle/angle_aten_kernels.h"

#include <numbers>
#include <optional>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/common/aten_utils.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/ops/binary.h"
#include "torch_tpu/csrc/ops/is/is.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/resize/resize_aten_kernels.h"
#include "torch_tpu/csrc/ops/unary_aten_kernels.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildAngleShlo(mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);

  if (IsComplexType(input_type)) {
    auto real = mlir::stablehlo::Real(input);
    auto imag = mlir::stablehlo::Imag(input);
    return BuildAtan2Shlo(imag, real);
  }

  auto zero = MakeConstantLike(input, 0.0);
  auto pi = MakeConstantLike(input, std::numbers::pi);

  auto is_negative = mlir::stablehlo::Compare(
      input, zero, mlir::stablehlo::ComparisonDirection::LT);
  auto angle = mlir::stablehlo::Select(is_negative, pi, zero);

  TT_ASSIGN_OR_RETURN(auto is_nan, BuildIsNanShlo(input));
  return mlir::stablehlo::Select(is_nan, input, angle);
}

}  // namespace

at::Tensor AtenAngle(const at::Tensor& self) {
  TT_KERNEL(OpName::kAngle, _, (self), {
    const c10::ScalarType out_dtype =
        c10::toRealValueType(InferOutputDtype(self));
    TT_ASSIGN_OR_THROW(const auto out_element_type,
                       ConvertTo<mlir::ElementType>(out_dtype));

    std::optional<mlir::ElementType> computation_dtype;
    if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
      TT_ASSIGN_OR_THROW(computation_dtype, ConvertTo<mlir::ElementType>(
                                                ToAccumulateType(out_dtype)));
    }

    TT_ASSIGN_OR_THROW(
        auto result, UnaryOp(self, BuildAngleShlo,
                             {.op_param_cache_keys = OpParamCacheKeys::Empty(),
                              .out_dtype = out_element_type,
                              .computation_dtype = computation_dtype}));
    return result;
  });
}

at::Tensor& AtenAngleOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kAngleOut, _, (self, out), {
    TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));

    const c10::ScalarType expected_dtype =
        c10::toRealValueType(InferOutputDtype(self));

    TT_CHECK_THROW(at::canCast(expected_dtype, out.scalar_type()),
                   error::kInvalidArgument)
        << "expected the output dtype to be " << ToString(expected_dtype)
        << ", got " << ToString(out.scalar_type());

    TT_ASSIGN_OR_THROW(const auto out_element_type,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));

    std::optional<mlir::ElementType> computation_dtype;
    if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
      TT_ASSIGN_OR_THROW(
          computation_dtype,
          ConvertTo<mlir::ElementType>(ToAccumulateType(expected_dtype)));
    }

    auto op_builder_with_cast =
        [out_element_type](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      TT_ASSIGN_OR_RETURN(mlir::MlirOp result, BuildAngleShlo(input));
      if (GetElementTypeOrDie(result) != out_element_type) {
        result = mlir::stablehlo::ConvertElementType(result, out_element_type);
      }
      return result;
    };

    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, std::move(op_builder_with_cast),
                   {.op_param_cache_keys = OpParamCacheKeys::Empty(),
                    .out_dtype = out_element_type,
                    .computation_dtype = computation_dtype}));
    return out;
  });
}

}  // namespace torch_tpu
