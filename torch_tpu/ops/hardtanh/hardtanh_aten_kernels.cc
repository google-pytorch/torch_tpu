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

#include "torch_tpu/ops/hardtanh/hardtanh_aten_kernels.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/hardtanh/hardtanh.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

absl::Status VerifyCommonPreconditions(const at::Tensor& self) {
  TT_RET_CHECK(!c10::isComplexType(self.scalar_type()), error::kInvalidArgument)
      << "hardtanh: complex types are not supported.";
  TT_RET_CHECK(self.scalar_type() != c10::ScalarType::Bool,
               error::kInvalidArgument)
      << "hardtanh: bool type is not supported.";
  return absl::OkStatus();
}

// Helper for building hardtanh with Tensors.
absl::StatusOr<DeviceBufferRef> AtenHardtanhTensorHelper(
    const at::Tensor& self, const at::Tensor& min_val,
    const at::Tensor& max_val, at::ScalarType output_scalar_type,
    OpParamCacheKeys param_keys) {
  // Our StableHLO implementation supports these types but PT CPU kernels do
  // not, and we want to have a consistent behavior across the backends.
  TT_RETURN_IF_ERROR(VerifyCommonPreconditions(self));

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(output_scalar_type));
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  // Since bounds are tensors, we use InferSize to compute broadcasted shapes.
  TT_ASSIGN_OR_RETURN(
      const Dimensions output_dims,
      InferSize(self.sizes(), min_val.sizes(), max_val.sizes()));

  auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
    auto& [self_op, min_op, max_op] = inputs;
    return BuildHardtanhShlo(self_op, min_op, max_op);
  };

  return DispatchOp<3>(std::move(op_builder), {self, min_val, max_val},
                       {.out_dtype = output_dtype,
                        .out_dims = output_dims,
                        .computation_dtype = computation_dtype,
                        .op_param_cache_keys = std::move(param_keys)});
}

// Helper for building hardtanh with promoted scalars.
absl::StatusOr<DeviceBufferRef> HardtanhScalarHelper(
    const at::Tensor& self, PromotedScalar min_val, PromotedScalar max_val,
    at::ScalarType output_scalar_type, OpParamCacheKeys param_keys) {
  const at::Scalar& min_scalar = min_val.scalar();
  const at::Scalar& max_scalar = max_val.scalar();

  TT_RETURN_IF_ERROR(VerifyCommonPreconditions(self));

  TT_RET_CHECK(c10::isSignedType(self.scalar_type()) ||
                   (min_scalar.toInt() >= 0 && max_scalar.toInt() >= 0),
               error::kInvalidArgument)
      << "hardtanh: cannot do hardtanh on an unsigned type with negative "
         "limits.";

  TT_ASSIGN_OR_RETURN(const at::Tensor min_tensor,
                      min_val.GetTensor(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(const at::Tensor max_tensor,
                      max_val.GetTensor(self.scalar_type()));

  return AtenHardtanhTensorHelper(self, min_tensor, max_tensor,
                                  output_scalar_type, std::move(param_keys));
}

}  // namespace

at::Tensor AtenHardtanh(const at::Tensor& self, const at::Scalar& min_val,
                        const at::Scalar& max_val) {
  PromotedScalar promoted_min_val = PromoteScalar(min_val);
  PromotedScalar promoted_max_val = PromoteScalar(max_val);
  TT_KERNEL(
      OpName::kHardtanh, param_keys, (self, promoted_min_val, promoted_max_val),
      {
        TT_ASSIGN_OR_THROW(
            auto result_buf,
            HardtanhScalarHelper(self, std::move(promoted_min_val),
                                 std::move(promoted_max_val),
                                 self.scalar_type(), std::move(param_keys)));
        return MakeTensor(std::move(result_buf));
      });
}

at::Tensor& AtenHardtanh_(at::Tensor& self, const at::Scalar& min_val,
                          const at::Scalar& max_val) {
  PromotedScalar promoted_min_val = PromoteScalar(min_val);
  PromotedScalar promoted_max_val = PromoteScalar(max_val);
  TT_KERNEL(
      OpName::kHardtanh_, param_keys,
      (self, promoted_min_val, promoted_max_val), {
        TT_ASSIGN_OR_THROW(
            auto result_buf,
            HardtanhScalarHelper(self, std::move(promoted_min_val),
                                 std::move(promoted_max_val),
                                 self.scalar_type(), std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), self));
        return self;
      });
}

at::Tensor& AtenHardtanhOut(const at::Tensor& self, const at::Scalar& min_val,
                            const at::Scalar& max_val, at::Tensor& out) {
  PromotedScalar promoted_min_val = PromoteScalar(min_val);
  PromotedScalar promoted_max_val = PromoteScalar(max_val);
  TT_KERNEL(
      OpName::kHardtanhOut, param_keys,
      (self, promoted_min_val, promoted_max_val, out), {
        TT_ASSIGN_OR_THROW(
            auto result_buf,
            HardtanhScalarHelper(self, std::move(promoted_min_val),
                                 std::move(promoted_max_val), out.scalar_type(),
                                 std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
