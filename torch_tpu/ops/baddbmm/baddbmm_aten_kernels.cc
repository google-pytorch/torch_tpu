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

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/bmm/bmm.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/precision_context.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {
namespace {

absl::Status CheckBaddbmmOut(const at::Tensor& out) {
  TT_RET_CHECK(!IsBool(out), error::kInvalidArgument)
      << "expected out tensor to have dtype float32, got bool";
  return absl::OkStatus();
}

absl::Status CheckBaddbmmInputs(const at::Tensor& self,
                                const at::Tensor& batch1,
                                const at::Tensor& batch2) {
  TT_RET_CHECK(batch1.numel() == 0 || batch2.numel() == 0 ||
                   (batch1.scalar_type() != at::kInt &&
                    batch1.scalar_type() != at::kLong),
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(batch1.scalar_type());

  TT_RET_CHECK(self.scalar_type() == batch1.scalar_type() &&
                   batch1.scalar_type() == batch2.scalar_type(),
               error::kInvalidArgument)
      << "expected input dtypes to be the same, got: self="
      << ToString(self.scalar_type())
      << ", batch1=" << ToString(batch1.scalar_type())
      << ", batch2=" << ToString(batch2.scalar_type());

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
    std::optional<mlir::MlirOp> self_op, mlir::MlirOp batch1_op,
    mlir::MlirOp batch2_op, std::optional<mlir::MlirOp> beta_op,
    std::optional<mlir::MlirOp> alpha_op, const MaybePromotedScalar& beta,
    const MaybePromotedScalar& alpha, mlir::stablehlo::Precision precision,
    mlir::ElementType out_dtype) {
  auto batch1_shape = GetTensorTypeOrDie(batch1_op).getShape();
  auto batch2_shape = GetTensorTypeOrDie(batch2_op).getShape();
  Dimensions out_shape = {batch1_shape[0], batch1_shape[1], batch2_shape[2]};
  auto& builder = batch1_op.getBuilder();

  // Case 1: beta is zero. Baddbmm result is just alpha * (batch1 @ batch2).
  if (beta.IsZero()) {
    if (alpha.IsZero()) {
      return MakeConstant(builder, at::Scalar(0.0), out_dtype, out_shape);
    }
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp bmm_res,
        BuildBmmShlo(batch1_op, batch2_op, out_dtype, precision));
    if (alpha.IsOne()) {
      return bmm_res;
    }
    // alpha is neither 0 nor 1, so alpha_op is guaranteed to have a value.
    TT_ASSIGN_OR_RETURN(auto alpha_tensor,
                        BroadcastIfNeeded(*alpha_op, bmm_res));
    return mlir::stablehlo::Mul(bmm_res, alpha_tensor);
  }

  // Case 2: beta is not zero. self_op is guaranteed to have a value.

  // Case 2a: alpha is zero. Baddbmm result is just beta * self.
  if (alpha.IsZero()) {
    TT_ASSIGN_OR_RETURN(auto self_bcst, BroadcastIfNeeded(*self_op, out_shape));
    if (!beta.IsOne()) {
      // beta is neither 0 nor 1, so beta_op is guaranteed to have a value.
      TT_ASSIGN_OR_RETURN(auto beta_tensor,
                          BroadcastIfNeeded(*beta_op, self_bcst));
      self_bcst = mlir::stablehlo::Mul(self_bcst, beta_tensor);
    }
    return self_bcst;
  }

  // Case 2b: alpha is not zero. Compute bmm_res = alpha * (batch1 @ batch2).
  TT_ASSIGN_OR_RETURN(mlir::MlirOp bmm_res,
                      BuildBmmShlo(batch1_op, batch2_op, out_dtype, precision));
  if (!alpha.IsOne()) {
    // alpha is neither 0 nor 1, so alpha_op is guaranteed to have a value.
    TT_ASSIGN_OR_RETURN(auto alpha_tensor,
                        BroadcastIfNeeded(*alpha_op, bmm_res));
    bmm_res = mlir::stablehlo::Mul(bmm_res, alpha_tensor);
  }
  // Compute beta * self and add.
  TT_ASSIGN_OR_RETURN(auto self_bcst, BroadcastIfNeeded(*self_op, bmm_res));
  if (!beta.IsOne()) {
    // beta is neither 0 nor 1, so beta_op is guaranteed to have a value.
    TT_ASSIGN_OR_RETURN(auto beta_tensor,
                        BroadcastIfNeeded(*beta_op, self_bcst));
    self_bcst = mlir::stablehlo::Mul(self_bcst, beta_tensor);
  }

  return mlir::stablehlo::Add(self_bcst, bmm_res);
}

absl::StatusOr<DeviceBufferRef> Baddbmm(
    const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2,
    MaybePromotedScalar beta, std::optional<at::Tensor> beta_tensor,
    MaybePromotedScalar alpha, std::optional<at::Tensor> alpha_tensor,
    at::ScalarType out_dtype, OpParamCacheKeys& param_keys,
    std::optional<OpName> op_name_override = std::nullopt) {
  TT_RETURN_IF_ERROR(CheckBaddbmmInputs(self, batch1, batch2));

  TT_ASSIGN_OR_RETURN(mlir::ElementType out_dtype_mlir,
                      ConvertTo<mlir::ElementType>(out_dtype));
  Dimensions out_dims = {batch1.size(0), batch1.size(1), batch2.size(2)};

  // Dynamically packs only the necessary tensors to optimize dispatch and
  // cache keys. The boolean flags are used to match this packing structure
  // during unpacking in the lambda.
  std::vector<at::Tensor> inputs;
  size_t expected_size = 2;  // batch1 and batch2 are always added.
  if (!beta.IsZero()) {
    expected_size++;
  }
  if (beta_tensor.has_value()) {
    expected_size++;
  }
  if (alpha_tensor.has_value()) {
    expected_size++;
  }
  inputs.reserve(expected_size);

  if (!beta.IsZero()) {
    inputs.push_back(self);
  }
  inputs.push_back(batch1);
  inputs.push_back(batch2);
  if (beta_tensor.has_value()) {
    inputs.push_back(std::move(*beta_tensor));
  }
  if (alpha_tensor.has_value()) {
    inputs.push_back(std::move(*alpha_tensor));
  }

  bool beta_is_zero = beta.IsZero();
  bool has_beta_tensor = beta_tensor.has_value();
  bool has_alpha_tensor = alpha_tensor.has_value();

  const auto precision = GetAndAddPrecisionTo(param_keys);

  auto op_builder =
      [beta = std::move(beta), alpha = std::move(alpha), beta_is_zero,
       has_beta_tensor, has_alpha_tensor, out_dtype_mlir,
       precision](absl::Span<mlir::MlirOp> inputs_op,
                  mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    // Unpacks inputs from the flat span. Packing order in Baddbmm():
    // [self (if beta != 0), batch1, batch2, beta_tensor, alpha_tensor]
    size_t idx = 0;
    std::optional<mlir::MlirOp> self_op;
    if (!beta_is_zero) {
      self_op = inputs_op[idx++];
    }
    mlir::MlirOp batch1_op = inputs_op[idx++];
    mlir::MlirOp batch2_op = inputs_op[idx++];
    std::optional<mlir::MlirOp> beta_op;
    if (has_beta_tensor) {
      beta_op = inputs_op[idx++];
    }
    std::optional<mlir::MlirOp> alpha_op;
    if (has_alpha_tensor) {
      alpha_op = inputs_op[idx++];
    }

    return BuildBaddbmmShlo(self_op, batch1_op, batch2_op, beta_op, alpha_op,
                            beta, alpha, precision, out_dtype_mlir);
  };

  TT_ASSIGN_OR_RETURN(
      auto result_buffer,
      DispatchOp<kDynamicSize>(
          std::move(op_builder), inputs,
          // It's OK for all three calls to Baddbmm() to use the same OpName
          // for dispatching, as the logic is the same for all three.
          {.op_name = op_name_override,
           .out_dtype = out_dtype_mlir,
           .out_dims = out_dims,
           .op_param_cache_keys = std::move(param_keys)}));
  return result_buffer;
}

absl::StatusOr<DeviceBufferRef> ResolveAndRunBaddbmm(
    const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2,
    MaybePromotedScalar beta, MaybePromotedScalar alpha,
    at::ScalarType out_dtype, OpParamCacheKeys& param_keys,
    std::optional<OpName> op_name_override = std::nullopt) {
  std::optional<at::Tensor> beta_tensor;
  if (!beta.ValueMatchesExclude()) {
    TT_ASSIGN_OR_RETURN(beta_tensor, beta.GetTensor(out_dtype));
  }
  std::optional<at::Tensor> alpha_tensor;
  if (!alpha.ValueMatchesExclude()) {
    TT_ASSIGN_OR_RETURN(alpha_tensor, alpha.GetTensor(out_dtype));
  }
  return Baddbmm(self, batch1, batch2, std::move(beta), std::move(beta_tensor),
                 std::move(alpha), std::move(alpha_tensor), out_dtype,
                 param_keys, op_name_override);
}

}  // namespace

at::Tensor AtenBaddbmmDtype(const at::Tensor& self, const at::Tensor& batch1,
                            const at::Tensor& batch2, at::ScalarType out_dtype,
                            const at::Scalar& beta, const at::Scalar& alpha) {
  auto promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero, ScalarValue::kOne);
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kZero,
                                                            ScalarValue::kOne);
  TT_KERNEL(
      OpName::kBaddbmmDtype, param_keys,
      (self, batch1, batch2, out_dtype, promoted_beta, promoted_alpha), {
        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            ResolveAndRunBaddbmm(self, batch1, batch2, std::move(promoted_beta),
                                 std::move(promoted_alpha), out_dtype,
                                 param_keys, OpName::kBaddbmmOut));
        return MakeTensor(result_buffer);
      });
}

at::Tensor& AtenBaddbmmDtypeOut(const at::Tensor& self,
                                const at::Tensor& batch1,
                                const at::Tensor& batch2,
                                at::ScalarType out_dtype,
                                const at::Scalar& beta, const at::Scalar& alpha,
                                at::Tensor& out) {
  auto promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero, ScalarValue::kOne);
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kZero,
                                                            ScalarValue::kOne);
  TT_KERNEL(
      OpName::kBaddbmmDtypeOut, param_keys,
      (self, batch1, batch2, out_dtype, promoted_beta, promoted_alpha, out), {
        TT_THROW_IF_ERROR(CheckBaddbmmOut(out));
        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            ResolveAndRunBaddbmm(self, batch1, batch2, std::move(promoted_beta),
                                 std::move(promoted_alpha), out_dtype,
                                 param_keys, OpName::kBaddbmmOut));
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(out, result_buffer.dimensions()));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), out));
        return out;
      });
}

at::Tensor& AtenBaddbmmOut(const at::Tensor& self, const at::Tensor& batch1,
                           const at::Tensor& batch2, const at::Scalar& beta,
                           const at::Scalar& alpha, at::Tensor& out) {
  auto promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero, ScalarValue::kOne);
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kZero,
                                                            ScalarValue::kOne);
  TT_KERNEL(
      OpName::kBaddbmmOut, param_keys,
      (self, batch1, batch2, promoted_beta, promoted_alpha, out), {
        TT_THROW_IF_ERROR(CheckBaddbmmOut(out));
        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            ResolveAndRunBaddbmm(self, batch1, batch2, std::move(promoted_beta),
                                 std::move(promoted_alpha), out.scalar_type(),
                                 param_keys));
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(out, result_buffer.dimensions()));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), out));
        return out;
      });
}

}  // namespace torch_tpu
