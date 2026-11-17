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

#include "csrc/ops/xlogy/xlogy_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/binary_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildXlogyShlo(mlir::MlirOp x, mlir::MlirOp y) {
  const mlir::ElementType input_dtype = GetElementTypeOrDie(x);
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      InferComputationDtype(input_dtype));

  TT_ASSIGN_OR_RETURN((auto [x_bcst, y_bcst]), ApplyBroadcastIfNeeded(x, y));

  auto comp_x = mlir::stablehlo::ConvertElementType(x_bcst, computation_dtype);
  auto comp_y = mlir::stablehlo::ConvertElementType(y_bcst, computation_dtype);

  auto zero = MakeConstantLike(comp_x, 0.0);
  auto one = MakeConstantLike(comp_x, 1.0);

  // If x is zero, use 1.0 for y to avoid log(0)
  mlir::MlirOp is_zero_x = mlir::stablehlo::Compare(
      comp_x, zero, mlir::stablehlo::ComparisonDirection::EQ);
  auto safe_y = mlir::stablehlo::Select(is_zero_x, one, comp_y);

  TT_ASSIGN_OR_RETURN(auto log_y, BuildLogShlo(safe_y, computation_dtype));
  TT_ASSIGN_OR_RETURN(auto x_log_y, BuildMulShlo(comp_x, log_y));

  // If y is NaN, always return NaN regardless of x
  mlir::MlirOp is_nan_y = mlir::stablehlo::Compare(
      comp_y, comp_y, mlir::stablehlo::ComparisonDirection::NE);

  auto x_or_zero = mlir::stablehlo::Select(is_zero_x, zero, x_log_y);
  auto res = mlir::stablehlo::Select(is_nan_y, comp_y, x_or_zero);
  return mlir::stablehlo::ConvertElementType(res, input_dtype);
}

}  // namespace

at::Tensor& AtenXlogyOutTensor(const at::Tensor& self, const at::Tensor& other,
                               at::Tensor& out) {
  TT_KERNEL(OpName::kXlogyOutTensor, _, (self, other, out), {
    TT_CHECK_THROW(IsPrivateUse1Device(out), error::kInvalidArgument)
        << "expected out tensor to be on " << GetPrivateUse1DeviceDebugName()
        << ", got " << out.device();

    TT_CHECK_THROW(!self.is_complex() && !other.is_complex(),
                   error::kInvalidArgument)
        << "expected non-complex dtypes, got x dtype "
        << ToString(self.scalar_type()) << " and y dtype "
        << ToString(other.scalar_type());

    TT_ASSIGN_OR_THROW(const auto out_dims,
                       InferSize(self.sizes(), other.sizes()));
    if (out.is_alias_of(self) || out.is_alias_of(other)) {
      TT_CHECK_THROW(absl::MakeConstSpan(out.sizes()) == out_dims,
                     error::kInvalidArgument)
          << "output with shape " << ToString(out.sizes())
          << " doesn't match the broadcast shape of the tensor being "
             "operated on in-place, which has shape "
          << ToString(out_dims);
    }

    BinaryOpOptions opts{
        .force_float_inputs = true,
        .op_param_cache_keys = OpParamCacheKeys::Empty(),
    };
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildXlogyShlo, std::move(opts)));
    return out;
  });
}

}  // namespace torch_tpu
