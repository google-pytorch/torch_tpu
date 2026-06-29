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

#include "torch_tpu/ops/logit/logit_aten_kernels.h"

#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/clamp/clamp.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"
#include "torch_tpu/ops/unary.h"

namespace torch_tpu {
namespace {

absl::Status ValidateLogitInputs(
    const at::Tensor& self,
    const std::optional<at::Tensor>& out = std::nullopt) {
  TT_RET_CHECK(!self.is_complex(), error::kInvalidArgument)
      << "expected floating point tensor, got complex tensor with dtype "
      << ToString(self.scalar_type());
  if (out.has_value()) {
    TT_RET_CHECK(!out->is_complex() && out->is_floating_point(),
                 error::kInvalidArgument)
        << "expected floating point output tensor, got tensor with dtype "
        << ToString(out->scalar_type());
  }
  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildLogitShlo(
    mlir::MlirOp input, std::optional<mlir::MlirOp> eps,
    mlir::ElementType computation_dtype) {
  input = mlir::stablehlo::ConvertElementType(input, computation_dtype);
  mlir::MlirOp z = input;
  const auto one = MakeConstantLike(input, 1.0, computation_dtype);

  if (eps.has_value()) {
    const mlir::MlirOp eps_op =
        mlir::stablehlo::ConvertElementType(*eps, computation_dtype);
    TT_ASSIGN_OR_RETURN(const auto max_val, BuildSubShlo(one, eps_op));
    TT_ASSIGN_OR_RETURN(z, BuildClampShlo(input, eps_op, max_val));
  }

  TT_ASSIGN_OR_RETURN(const auto one_minus_z, BuildSubShlo(one, z));
  TT_ASSIGN_OR_RETURN(const auto div_result, BuildDivShlo(z, one_minus_z));
  return BuildLogShlo(div_result, computation_dtype);
}

absl::StatusOr<DeviceBufferRef> AtenLogitHelper(
    const at::Tensor& self, std::optional<double> eps,
    at::ScalarType output_scalar_type, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(output_scalar_type));
  const Dimensions output_dims = CopyIntVector(self.sizes());

  at::ScalarType computation_scalar_type = self.scalar_type();
  if (c10::isIntegralType(computation_scalar_type, /*includeBool=*/true)) {
    computation_scalar_type = c10::get_default_dtype_as_scalartype();
  }
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(computation_scalar_type));

  auto op_builder = [eps, computation_dtype](
                        mlir::MlirOp self_op) -> absl::StatusOr<mlir::MlirOp> {
    std::optional<mlir::MlirOp> eps_op;
    if (eps.has_value()) {
      eps_op = MakeConstantLike(self_op, *eps, computation_dtype);
    }
    return BuildLogitShlo(self_op, eps_op, computation_dtype);
  };
  return DispatchOp<1>(std::move(op_builder), self,
                       {.out_dtype = output_dtype,
                        .out_dims = output_dims,
                        .computation_dtype = computation_dtype,
                        .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenLogit(const at::Tensor& self, std::optional<double> eps) {
  TT_KERNEL(OpName::kLogit, param_keys, (self, eps), {
    TT_THROW_IF_ERROR(ValidateLogitInputs(self));
    at::ScalarType output_scalar_type = self.scalar_type();
    if (c10::isIntegralType(output_scalar_type, /*includeBool=*/true)) {
      output_scalar_type = c10::get_default_dtype_as_scalartype();
    }
    TT_ASSIGN_OR_THROW(
        auto result_buf,
        AtenLogitHelper(self, eps, output_scalar_type, std::move(param_keys)));
    return MakeTensor(std::move(result_buf));
  });
}

at::Tensor& AtenLogit_(at::Tensor& self, std::optional<double> eps) {
  TT_KERNEL(OpName::kLogit_, param_keys, (self, eps), {
    TT_THROW_IF_ERROR(ValidateLogitInputs(self, self));
    TT_ASSIGN_OR_THROW(
        auto result_buf,
        AtenLogitHelper(self, eps, self.scalar_type(), std::move(param_keys)));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), self));
    return self;
  });
}

at::Tensor& AtenLogitOut(const at::Tensor& self, std::optional<double> eps,
                         at::Tensor& out) {
  TT_KERNEL(OpName::kLogitOut, param_keys, (self, eps, out), {
    TT_THROW_IF_ERROR(ValidateLogitInputs(self, out));
    AtenResize_(out, self.sizes(), std::nullopt);
    TT_ASSIGN_OR_THROW(
        auto result_buf,
        AtenLogitHelper(self, eps, out.scalar_type(), std::move(param_keys)));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

}  // namespace torch_tpu
