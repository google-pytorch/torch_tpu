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

#include "torch_tpu/ops/experimental/scaled_add_aten_kernels.h"

#include <utility>

#include "mlir/IR/Types.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/result_type.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/binary_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildScaledAddShlo(mlir::MlirOp self_op,
                                                mlir::MlirOp other_op,
                                                mlir::MlirOp alpha_op) {
  const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other_op);
  auto other_element_type = other_type.getElementType();
  auto casted_alpha_op =
      mlir::stablehlo::ConvertElementType(alpha_op, other_element_type);
  TT_ASSIGN_OR_RETURN((auto [broadcasted_other, broadcasted_alpha]),
                      ApplyBroadcastIfNeeded(other_op, casted_alpha_op));
  TT_ASSIGN_OR_RETURN(auto mul_op,
                      BuildMulShlo(broadcasted_other, broadcasted_alpha));
  return BuildAddShlo(self_op, mul_op);
}

}  // namespace

at::Tensor& AtenScaledAddOut(const at::Tensor& self, const at::Tensor& other,
                             const at::Scalar& alpha, at::Tensor& out) {
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(
      OpName::kScaledAddOut, param_keys, (self, other, promoted_alpha, out), {
        if (promoted_alpha.IsOne()) {
          TT_THROW_IF_ERROR(
              BinaryOpOut(self, other, out, BuildAddShlo,
                          {.op_param_cache_keys = std::move(param_keys)}));
          return out;
        }

        TT_ASSIGN_OR_THROW(const at::Tensor alpha_tensor,
                           promoted_alpha.GetTensor(out.scalar_type()));

        TT_THROW_IF_ERROR(
            TernaryOpOut(self, other, alpha_tensor, out, BuildScaledAddShlo,
                         {.op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

at::Tensor AtenScaledAdd(const at::Tensor& self, const at::Tensor& other,
                         const at::Scalar& alpha) {
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(OpName::kScaledAdd, param_keys, (self, other, promoted_alpha), {
    const at::ScalarType promoted_scalar_type = at::result_type(self, other);
    TT_ASSIGN_OR_THROW(const Dimensions output_dims,
                       InferSize(self.sizes(), other.sizes()));
    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        MakeEmptyTensor(output_dims, promoted_scalar_type, self.device()));

    if (promoted_alpha.IsOne()) {
      TT_THROW_IF_ERROR(
          BinaryOpOut(self, other, out, BuildAddShlo,
                      {.op_param_cache_keys = std::move(param_keys)}));
      return out;
    }

    TT_ASSIGN_OR_THROW(const at::Tensor alpha_tensor,
                       promoted_alpha.GetTensor(promoted_scalar_type));

    TT_THROW_IF_ERROR(
        TernaryOpOut(self, other, alpha_tensor, out, BuildScaledAddShlo,
                     {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

}  // namespace torch_tpu
