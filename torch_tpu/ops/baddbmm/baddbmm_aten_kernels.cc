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

#include "torch_tpu/ops/baddbmm/baddbmm_aten_kernels.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ATen/core/ATen_fwd.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/bmm/bmm.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/precision_context.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

absl::Status CheckBaddbmmOut(const at::Tensor& out) {
  TT_RET_CHECK(!IsBool(out), error::kInvalidArgument)
      << "expected out tensor to have a non-boolean dtype, got bool";
  return absl::OkStatus();
}

absl::Status CheckBaddbmmInputs(const at::Tensor& self,
                                const at::Tensor& batch1,
                                const at::Tensor& batch2) {
  TT_RET_CHECK(batch1.dim() == 3, error::kInvalidArgument)
      << "expected batch1 to be a 3D tensor (batch of matrices), got "
      << batch1.dim() << "D";
  TT_RET_CHECK(batch2.dim() == 3, error::kInvalidArgument)
      << "expected batch2 to be a 3D tensor (batch of matrices), got "
      << batch2.dim() << "D";

  TT_RET_CHECK(batch1.size(0) == batch2.size(0), error::kInvalidArgument)
      << "expected the batch dimension of the first argument (of shape "
      << ToString(batch1.sizes())
      << ") to match the batch dimension of the second argument (of shape "
      << ToString(batch2.sizes()) << "), got " << batch1.size(0) << " vs "
      << batch2.size(0);
  TT_RET_CHECK(batch1.size(2) == batch2.size(1), error::kInvalidArgument)
      << "expected the last dimension of the first argument (of shape "
      << ToString(batch1.sizes())
      << ") to match the second dimension of the second argument (of shape "
      << ToString(batch2.sizes()) << "), got " << batch1.size(2) << " vs "
      << batch2.size(1);

  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildBaddbmmShlo(
    mlir::MlirOp self_op, mlir::MlirOp batch1_op, mlir::MlirOp batch2_op,
    mlir::MlirOp beta_op, mlir::MlirOp alpha_op, ScalingMode beta_mode,
    ScalingMode alpha_mode, at::ScalarType out_dtype) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_dtype_mlir,
                      ConvertTo<mlir::ElementType>(out_dtype));
  if (GetElementTypeOrDie(self_op) != out_dtype_mlir) {
    self_op = mlir::stablehlo::ConvertElementType(self_op, out_dtype_mlir);
  }

  const auto current_precision = GetPrecision();
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp bmm_res,
      BuildBmmShlo(batch1_op, batch2_op, out_dtype_mlir, current_precision));

  if (GetElementTypeOrDie(bmm_res) != out_dtype_mlir) {
    bmm_res = mlir::stablehlo::ConvertElementType(bmm_res, out_dtype_mlir);
  }

  if (alpha_mode != ScalingMode::kOne) {
    if (GetElementTypeOrDie(alpha_op) != out_dtype_mlir) {
      alpha_op = mlir::stablehlo::ConvertElementType(alpha_op, out_dtype_mlir);
    }
    TT_ASSIGN_OR_RETURN(auto alpha_tensor,
                        BroadcastIfNeeded(alpha_op, bmm_res));
    bmm_res = mlir::stablehlo::Mul(bmm_res, alpha_tensor);
  }

  if (beta_mode == ScalingMode::kZero) {
    return bmm_res;
  }

  mlir::MlirOp scaled_self = self_op;
  if (beta_mode != ScalingMode::kOne) {
    if (GetElementTypeOrDie(beta_op) != out_dtype_mlir) {
      beta_op = mlir::stablehlo::ConvertElementType(beta_op, out_dtype_mlir);
    }
    TT_ASSIGN_OR_RETURN(auto beta_tensor, BroadcastIfNeeded(beta_op, self_op));
    scaled_self = mlir::stablehlo::Mul(self_op, beta_tensor);
  }

  TT_ASSIGN_OR_RETURN(auto self_bcst, BroadcastIfNeeded(scaled_self, bmm_res));
  return mlir::stablehlo::Add(self_bcst, bmm_res);
}

absl::StatusOr<DeviceBufferRef> Baddbmm(
    const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2,
    PromotedScalar beta, PromotedScalar alpha, at::ScalarType out_dtype,
    OpParamCacheKeys& param_keys) {
  TT_RETURN_IF_ERROR(CheckBaddbmmInputs(self, batch1, batch2));

  TT_ASSIGN_OR_RETURN(mlir::ElementType out_dtype_mlir,
                      ConvertTo<mlir::ElementType>(out_dtype));
  Dimensions out_dims = {batch1.size(0), batch1.size(1), batch2.size(2)};

  ScalingMode beta_mode = GetScalingMode(beta.scalar());
  ScalingMode alpha_mode = GetScalingMode(alpha.scalar());

  const auto current_precision = GetPrecision();
  TT_RETURN_IF_ERROR(param_keys.SetParam("precision", current_precision));
  TT_RETURN_IF_ERROR(param_keys.SetParam("beta_mode", beta_mode));
  TT_RETURN_IF_ERROR(param_keys.SetParam("alpha_mode", alpha_mode));

  auto op_builder = [beta_mode, alpha_mode,
                     out_dtype](FixedSizeSpan<mlir::MlirOp, 5> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, batch1_op, batch2_op, beta_op, alpha_op] = inputs;
    return BuildBaddbmmShlo(self_op, batch1_op, batch2_op, beta_op, alpha_op,
                            beta_mode, alpha_mode, out_dtype);
  };

  TT_ASSIGN_OR_RETURN(at::Tensor beta_tensor, beta.GetTensor());
  TT_ASSIGN_OR_RETURN(at::Tensor alpha_tensor, alpha.GetTensor());

  TT_ASSIGN_OR_RETURN(
      auto result_buffer,
      DispatchOp<5>(
          std::move(op_builder),
          {self, batch1, batch2, beta_tensor, alpha_tensor},
          // It's OK for both calls to Baddbmm() to use the same OpName for
          // dispatching, as the logic is the same for both.
          {.op_name = OpName::kBaddbmmOut,
           .out_dtype = out_dtype_mlir,
           .out_dims = out_dims,
           .op_param_cache_keys = std::move(param_keys)}));
  return result_buffer;
}

}  // namespace

at::Tensor AtenBaddbmmDtype(const at::Tensor& self, const at::Tensor& batch1,
                            const at::Tensor& batch2, at::ScalarType out_dtype,
                            const at::Scalar& beta, const at::Scalar& alpha) {
  auto promoted_beta = PromoteScalar(beta);
  auto promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(OpName::kBaddbmmDtype, param_keys,
            (self, batch1, batch2, out_dtype, promoted_beta, promoted_alpha), {
              TT_ASSIGN_OR_THROW(
                  auto result_buffer,
                  Baddbmm(self, batch1, batch2, std::move(promoted_beta),
                          std::move(promoted_alpha), out_dtype, param_keys));
              return MakeTensor(result_buffer);
            });
}

at::Tensor& AtenBaddbmmDtypeOut(const at::Tensor& self,
                                const at::Tensor& batch1,
                                const at::Tensor& batch2,
                                at::ScalarType out_dtype,
                                const at::Scalar& beta, const at::Scalar& alpha,
                                at::Tensor& out) {
  auto promoted_beta = PromoteScalar(beta);
  auto promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(
      OpName::kBaddbmmDtypeOut, param_keys,
      (self, batch1, batch2, out_dtype, promoted_beta, promoted_alpha, out), {
        TT_THROW_IF_ERROR(CheckBaddbmmOut(out));
        TT_CHECK_THROW(out.scalar_type() == out_dtype, error::kInvalidArgument)
            << "out dtype should match out_dtype, got out dtype "
            << torch_tpu::ToString(out.scalar_type()) << " and out_dtype "
            << torch_tpu::ToString(out_dtype);
        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            Baddbmm(self, batch1, batch2, std::move(promoted_beta),
                    std::move(promoted_alpha), out_dtype, param_keys));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), out));
        return out;
      });
}

at::Tensor& AtenBaddbmmOut(const at::Tensor& self, const at::Tensor& batch1,
                           const at::Tensor& batch2, const at::Scalar& beta,
                           const at::Scalar& alpha, at::Tensor& out) {
  auto promoted_beta = PromoteScalar(beta);
  auto promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(
      OpName::kBaddbmmOut, param_keys,
      (self, batch1, batch2, promoted_beta, promoted_alpha, out), {
        TT_THROW_IF_ERROR(CheckBaddbmmOut(out));
        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            Baddbmm(self, batch1, batch2, std::move(promoted_beta),
                    std::move(promoted_alpha), out.scalar_type(), param_keys));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), out));
        return out;
      });
}

}  // namespace torch_tpu
