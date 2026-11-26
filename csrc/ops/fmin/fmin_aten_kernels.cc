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

#include "csrc/ops/fmin/fmin_aten_kernels.h"

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/binary_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

absl::Status ValidateInputsNotComplex(const at::Tensor& self,
                                      const at::Tensor& other) {
  TT_RET_CHECK(!self.is_complex(), error::kInvalidArgument)
      << "expected the dtype of the first argument not to be complex, got "
      << ToString(self.scalar_type());
  TT_RET_CHECK(!other.is_complex(), error::kInvalidArgument)
      << "expected the dtype of the second argument not to be complex, got "
      << ToString(other.scalar_type());
  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildFminShlo(mlir::MlirOp self,
                                           mlir::MlirOp other) {
  TT_ASSIGN_OR_RETURN((auto [self_, other_]),
                      ApplyBroadcastIfNeeded(self, other));
  if (!IsFloatType(GetTensorTypeOrDie(self_))) {
    return mlir::stablehlo::Min(self_, other_);
  }

  // `x == x` is false only for NaN.
  mlir::MlirOp self_not_nan = mlir::stablehlo::Compare(
      self_, self_, mlir::stablehlo::ComparisonDirection::EQ);
  mlir::MlirOp other_nan = mlir::stablehlo::Compare(
      other_, other_, mlir::stablehlo::ComparisonDirection::NE);
  // Use IEEE-754 TOTALORDER so -0.0 < +0.0; guard with self_not_nan because
  // TOTALORDER places -NaN below -inf.
  mlir::MlirOp self_less = mlir::stablehlo::Compare(
      self_, other_, mlir::stablehlo::ComparisonDirection::LT,
      mlir::stablehlo::ComparisonTypeAttr::get(
          &self_.getContext(), mlir::stablehlo::ComparisonType::TOTALORDER));
  mlir::MlirOp self_valid_less = mlir::stablehlo::And(self_not_nan, self_less);
  mlir::MlirOp choose_self = mlir::stablehlo::Or(other_nan, self_valid_less);
  return mlir::stablehlo::Select(choose_self, self_, other_);
}
}  // namespace

at::Tensor& AtenFminOut(const at::Tensor& self, const at::Tensor& other,
                        at::Tensor& out) {
  TT_KERNEL(OpName::kFminOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(ValidateInputsNotComplex(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildFminShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}
}  // namespace torch_tpu
