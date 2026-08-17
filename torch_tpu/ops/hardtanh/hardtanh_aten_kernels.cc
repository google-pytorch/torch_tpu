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

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/aten_utils.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {
absl::StatusOr<mlir::MlirOp> BuildHardtanhShlo(mlir::MlirOp input,
                                               mlir::MlirOp min_val,
                                               mlir::MlirOp max_val) {
  return mlir::stablehlo::Clamp(min_val, input, max_val);
}

absl::StatusOr<mlir::MlirOp> BuildHardtanhBackwardShlo(mlir::MlirOp grad_output,
                                                       mlir::MlirOp self,
                                                       mlir::MlirOp min_val,
                                                       mlir::MlirOp max_val) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp min_val_bcast,
                      BroadcastIfNeeded(min_val, self));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp max_val_bcast,
                      BroadcastIfNeeded(max_val, self));

  mlir::MlirOp gt_min = mlir::stablehlo::Compare(
      self, min_val_bcast, mlir::stablehlo::ComparisonDirection::GT);
  mlir::MlirOp lt_max = mlir::stablehlo::Compare(
      self, max_val_bcast, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp in_range = mlir::stablehlo::And(gt_min, lt_max);
  mlir::MlirOp zero = MakeConstantLike(grad_output, 0.0);
  return mlir::stablehlo::Select(in_range, grad_output, zero);
}

absl::Status CheckHardtanhInputs(const at::Tensor& self,
                                 PromotedScalar& promoted_min,
                                 PromotedScalar& promoted_max) {
  auto scalar_type = self.scalar_type();

  TT_RET_CHECK(!IsComplex(self), error::kInvalidArgument)
      << "expected the input dtype to be non-complex, got "
      << ToString(scalar_type);
  TT_RET_CHECK(!IsBool(self), error::kInvalidArgument)
      << "expected the input dtype to be non-boolean, got "
      << ToString(scalar_type);
  TT_RET_CHECK(c10::isSignedType(scalar_type)  // MLIR_SIGNED_INT_OK=Checking
                                               // ATen scalar type signedness
                   || (promoted_min.scalar().toInt() >= 0 &&
                       promoted_max.scalar().toInt() >= 0),
               error::kInvalidArgument)
      << "expected positive limit values when executing on an unsigned tensor, "
      << "got min_val=" << promoted_min.scalar().toInt()
      << " and max_val=" << promoted_max.scalar().toInt();
  return absl::OkStatus();
}

// Helper to dispatch the hardtanh computation on the device.
absl::StatusOr<DeviceBufferRef> AtenHardtanhImpl(const at::Tensor& self,
                                                 PromotedScalar& promoted_min,
                                                 PromotedScalar& promoted_max,
                                                 OpParamCacheKeys param_keys) {
  auto scalar_type = self.scalar_type();
  TT_RETURN_IF_ERROR(CheckHardtanhInputs(self, promoted_min, promoted_max));

  TT_ASSIGN_OR_RETURN(at::Tensor min_tensor,
                      promoted_min.GetTensor(scalar_type));
  TT_ASSIGN_OR_RETURN(at::Tensor max_tensor,
                      promoted_max.GetTensor(scalar_type));

  auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, min_op, max_op] = inputs;
    return BuildHardtanhShlo(self_op, min_op, max_op);
  };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(scalar_type));

  return DispatchOp<3>(op_builder, {self, min_tensor, max_tensor},
                       {.out_dtype = output_dtype,
                        .out_dims = CopyIntVector(self.sizes()),
                        .op_param_cache_keys = std::move(param_keys)});
}

// Helper to dispatch the hardtanh_backward computation on the device.
absl::StatusOr<DeviceBufferRef> AtenHardtanhBackwardImpl(
    const at::Tensor& grad_output, const at::Tensor& self,
    PromotedScalar& promoted_min, PromotedScalar& promoted_max,
    OpParamCacheKeys param_keys) {
  auto scalar_type = self.scalar_type();
  TT_RET_CHECK(c10::isFloatingType(scalar_type), error::kInvalidArgument)
      << "expected the input dtype to be floating point, got "
      << ToString(scalar_type);

  TT_ASSIGN_OR_RETURN(at::Tensor min_tensor,
                      promoted_min.GetTensor(scalar_type));
  TT_ASSIGN_OR_RETURN(at::Tensor max_tensor,
                      promoted_max.GetTensor(scalar_type));

  auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 4> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [grad_output_op, self_op, min_op, max_op] = inputs;
    return BuildHardtanhBackwardShlo(grad_output_op, self_op, min_op, max_op);
  };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(scalar_type));

  return DispatchOp<4>(op_builder, {grad_output, self, min_tensor, max_tensor},
                       {.out_dtype = output_dtype,
                        .out_dims = CopyIntVector(self.sizes()),
                        .op_param_cache_keys = std::move(param_keys)});
}
}  // namespace

at::Tensor AtenHardtanh(const at::Tensor& self, const at::Scalar& min_val,
                        const at::Scalar& max_val) {
  PromotedScalar promoted_min = PromoteScalar(min_val);
  PromotedScalar promoted_max = PromoteScalar(max_val);
  TT_KERNEL(OpName::kHardtanh, param_keys, (self, promoted_min, promoted_max), {
    TT_ASSIGN_OR_THROW(auto result_buf,
                       AtenHardtanhImpl(self, promoted_min, promoted_max,
                                        std::move(param_keys)));
    return MakeTensor(std::move(result_buf));
  });
}

at::Tensor& AtenHardtanh_(at::Tensor& self, const at::Scalar& min_val,
                          const at::Scalar& max_val) {
  PromotedScalar promoted_min = PromoteScalar(min_val);
  PromotedScalar promoted_max = PromoteScalar(max_val);
  TT_KERNEL(
      OpName::kHardtanh_, param_keys, (self, promoted_min, promoted_max), {
        TT_ASSIGN_OR_THROW(auto result_buf,
                           AtenHardtanhImpl(self, promoted_min, promoted_max,
                                            std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), self));
        return self;
      });
}

at::Tensor& AtenHardtanhOut(const at::Tensor& self, const at::Scalar& min_val,
                            const at::Scalar& max_val, at::Tensor& out) {
  PromotedScalar promoted_min = PromoteScalar(min_val);
  PromotedScalar promoted_max = PromoteScalar(max_val);
  TT_KERNEL(
      OpName::kHardtanhOut, param_keys, (self, promoted_min, promoted_max, out),
      {
        TT_ASSIGN_OR_THROW(auto result_buf,
                           AtenHardtanhImpl(self, promoted_min, promoted_max,
                                            std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

at::Tensor AtenHardtanhBackward(const at::Tensor& grad_output,
                                const at::Tensor& self,
                                const at::Scalar& min_val,
                                const at::Scalar& max_val) {
  PromotedScalar promoted_min = PromoteScalar(min_val);
  PromotedScalar promoted_max = PromoteScalar(max_val);
  TT_KERNEL(OpName::kHardtanhBackward, param_keys,
            (grad_output, self, promoted_min, promoted_max), {
              TT_ASSIGN_OR_THROW(auto result_buf,
                                 AtenHardtanhBackwardImpl(
                                     grad_output, self, promoted_min,
                                     promoted_max, std::move(param_keys)));
              return MakeTensor(std::move(result_buf));
            });
}

at::Tensor& AtenHardtanhBackwardGradInput(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          const at::Scalar& min_val,
                                          const at::Scalar& max_val,
                                          at::Tensor& grad_input) {
  PromotedScalar promoted_min = PromoteScalar(min_val);
  PromotedScalar promoted_max = PromoteScalar(max_val);
  TT_KERNEL(OpName::kHardtanhBackwardGradInput, param_keys,
            (grad_output, self, promoted_min, promoted_max, grad_input), {
              TT_ASSIGN_OR_THROW(auto result_buf,
                                 AtenHardtanhBackwardImpl(
                                     grad_output, self, promoted_min,
                                     promoted_max, std::move(param_keys)));
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buf), grad_input));
              return grad_input;
            });
}

}  // namespace torch_tpu
