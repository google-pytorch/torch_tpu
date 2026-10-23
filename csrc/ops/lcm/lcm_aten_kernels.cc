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

#include "csrc/ops/lcm/lcm_aten_kernels.h"

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/result_type.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/binary_aten_kernels.h"
#include "csrc/ops/gcd/gcd_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

absl::Status ValidateLcmInputs(const at::Tensor& self, const at::Tensor& other,
                               const at::Tensor& out) {
  const at::ScalarType common_dtype = at::result_type(self, other);
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, common_dtype));
  TT_RET_CHECK(c10::isIntegralType(common_dtype, /*includeBool=*/false),
               error::kInvalidArgument)
      << "expected the promoted input dtype to be an integer type, got "
      << ToString(common_dtype);
  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildLcmShlo(mlir::MlirOp self,
                                          mlir::MlirOp other) {
  TT_ASSIGN_OR_RETURN((auto [lhs, rhs]), ApplyBroadcastIfNeeded(self, other));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType elem_type, GetElementType(lhs));

  // Promote 8-bit and 16-bit integers (i8, ui8, i16) to i32 to match C++
  // standard integral promotion in `std::abs(a / g * b)` and avoid unsigned
  // `stablehlo::Abs` or narrow overflow before `abs`.
  const bool promote_to_i32 = (elem_type == mlir::ElementType::I8 ||
                               elem_type == mlir::ElementType::UI8 ||
                               elem_type == mlir::ElementType::I16);
  if (promote_to_i32) {
    lhs = mlir::stablehlo::ConvertElementType(lhs, mlir::ElementType::I32);
    rhs = mlir::stablehlo::ConvertElementType(rhs, mlir::ElementType::I32);
  }

  TT_ASSIGN_OR_RETURN(auto g, BuildGcdShlo(lhs, rhs));

  auto zero = MakeConstantLike(lhs, 0);
  auto one = MakeConstantLike(lhs, 1);
  auto is_g_zero = mlir::stablehlo::Compare(
      g, zero, mlir::stablehlo::ComparisonDirection::EQ);
  auto safe_g = mlir::stablehlo::Select(is_g_zero, one, g);
  auto quotient = mlir::stablehlo::Div(lhs, safe_g);
  auto product = mlir::stablehlo::Mul(quotient, rhs);
  auto abs_product = mlir::stablehlo::Abs(product);
  auto result = mlir::stablehlo::Select(is_g_zero, zero, abs_product);

  if (promote_to_i32) {
    result = mlir::stablehlo::ConvertElementType(result, elem_type);
  }
  return result;
}

}  // namespace

at::Tensor& AtenLcmOut(const at::Tensor& self, const at::Tensor& other,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kLcmOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(ValidateLcmInputs(self, other, out));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildLcmShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

}  // namespace torch_tpu
