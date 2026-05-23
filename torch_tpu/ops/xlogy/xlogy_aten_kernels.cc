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

#include "torch_tpu/ops/xlogy/xlogy_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/result_type.h"
#include "absl/status/statusor.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildXlogyShlo(
    mlir::MlirOp x, mlir::MlirOp y, mlir::ElementType computation_dtype) {
  x = mlir::stablehlo::ConvertElementType(x, computation_dtype);
  y = mlir::stablehlo::ConvertElementType(y, computation_dtype);

  TT_ASSIGN_OR_RETURN((auto [x_bcst, y_bcst]), ApplyBroadcastIfNeeded(x, y));

  auto zero = MakeConstantLike(x_bcst, 0.0, computation_dtype);
  auto one = MakeConstantLike(x_bcst, 1.0, computation_dtype);

  // If x is zero, use 1.0 for y to avoid log(0)
  mlir::MlirOp is_zero_x = mlir::stablehlo::Compare(
      x_bcst, zero, mlir::stablehlo::ComparisonDirection::EQ);
  auto safe_y = mlir::stablehlo::Select(is_zero_x, one, y_bcst);

  TT_ASSIGN_OR_RETURN(auto log_y, BuildLogShlo(safe_y, computation_dtype));
  TT_ASSIGN_OR_RETURN(auto x_log_y, BuildMulShlo(x_bcst, log_y));

  // If y is NaN, always return NaN regardless of x
  mlir::MlirOp is_nan_y = mlir::stablehlo::Compare(
      y_bcst, y_bcst, mlir::stablehlo::ComparisonDirection::NE);

  auto x_or_zero = mlir::stablehlo::Select(is_zero_x, zero, x_log_y);
  return mlir::stablehlo::Select(is_nan_y, y_bcst, x_or_zero);
}

}  // namespace

at::Tensor& AtenXlogyOutTensor(const at::Tensor& self, const at::Tensor& other,
                               at::Tensor& out) {
  TT_KERNEL(OpName::kXlogyOutTensor, _, (self, other, out), {
    TT_CHECK_THROW(!self.is_complex() && !other.is_complex(),
                   error::kInvalidArgument)
        << "complex dtypes are not supported, got x dtype "
        << ToString(self.scalar_type()) << " and y dtype "
        << ToString(other.scalar_type());

    at::ScalarType promoted_scalar_type = at::result_type(self, other);
    // If promoted_scalar_type is integral, use the default float scalar type
    if (c10::isIntegralType(promoted_scalar_type, /*includeBool=*/true)) {
      promoted_scalar_type = c10::get_default_dtype_as_scalartype();
    }
    TT_ASSIGN_OR_THROW(auto computation_dtype,
                       ConvertTo<mlir::ElementType>(promoted_scalar_type));

    auto op_builder = [computation_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [x_op, y_op] = inputs;
      return BuildXlogyShlo(x_op, y_op, computation_dtype);
    };

    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));

    TT_ASSIGN_OR_THROW(
        auto result_buffer,
        DispatchOp<2>(std::move(op_builder), {self, other},
                      {.out_dtype = out_dtype,
                       .out_dims = CopyIntVector(out.sizes()),
                       .op_param_cache_keys = OpParamCacheKeys::Empty()}));

    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buffer), out));
    return out;
  });
}

}  // namespace torch_tpu
