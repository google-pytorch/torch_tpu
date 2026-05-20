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

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ATen/core/ATen_fwd.h"
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
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

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
    std::optional<mlir::MlirOp> beta_op, std::optional<mlir::MlirOp> alpha_op,
    const MaybePromotedScalar& beta, const MaybePromotedScalar& alpha,
    at::ScalarType out_dtype) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_dtype_mlir,
                      ConvertTo<mlir::ElementType>(out_dtype));

  const auto current_precision = GetPrecision();
  const mlir::RankedTensorType batch1_type = GetTensorTypeOrDie(batch1_op);
  const mlir::RankedTensorType batch2_type = GetTensorTypeOrDie(batch2_op);
  const int64_t out_dim0 = batch1_type.getShape()[0];
  const int64_t out_dim1 = batch1_type.getShape()[1];
  const int64_t out_dim2 = batch2_type.getShape()[2];

  mlir::MlirOp bmm_res;
  if (!alpha_op.has_value() && alpha.IsZero()) {
    llvm::SmallVector<int64_t, 3> out_shape = {out_dim0, out_dim1, out_dim2};
    TT_ASSIGN_OR_RETURN(bmm_res,
                        MakeConstant(self_op.getBuilder(), at::Scalar(0.0),
                                     out_dtype_mlir, out_shape));
  } else {
    TT_ASSIGN_OR_RETURN(
        bmm_res,
        BuildBmmShlo(batch1_op, batch2_op, out_dtype_mlir, current_precision));
  }

  // If alpha is not 1, multiply bmm_res by alpha.
  if (alpha_op.has_value()) {
    TT_ASSIGN_OR_RETURN(auto alpha_tensor,
                        BroadcastIfNeeded(*alpha_op, bmm_res));
    bmm_res = mlir::stablehlo::Mul(bmm_res, alpha_tensor);
  } else {
    TT_RET_CHECK(alpha.IsOne() || alpha.IsZero(), error::kInvalidArgument)
        << "alpha must be 0 or 1 when not using dynamic alpha, got "
        << alpha.scalar().toDouble();
  }

  // If beta is 0, content of input is ignored (no NaN / infinity propagation)
  if (beta.IsZero()) {
    return bmm_res;
  }

  // Broadcast self to match bmm_res shape
  TT_ASSIGN_OR_RETURN(auto self_bcst, BroadcastIfNeeded(self_op, bmm_res));

  // If beta is not 1, multiply self_bcst by beta.
  if (beta_op.has_value()) {
    TT_ASSIGN_OR_RETURN(auto beta_tensor,
                        BroadcastIfNeeded(*beta_op, self_bcst));
    self_bcst = mlir::stablehlo::Mul(self_bcst, beta_tensor);
  } else {
    TT_RET_CHECK(beta.IsOne(), error::kInvalidArgument)
        << "beta must be 1 when not using dynamic beta";
  }

  return mlir::stablehlo::Add(self_bcst, bmm_res);
}

absl::StatusOr<DeviceBufferRef> Baddbmm(
    const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2,
    MaybePromotedScalar beta, MaybePromotedScalar alpha,
    at::ScalarType out_dtype, OpParamCacheKeys& param_keys) {
  TT_RETURN_IF_ERROR(CheckBaddbmmInputs(self, batch1, batch2));

  TT_ASSIGN_OR_RETURN(mlir::ElementType out_dtype_mlir,
                      ConvertTo<mlir::ElementType>(out_dtype));
  Dimensions out_dims = {batch1.size(0), batch1.size(1), batch2.size(2)};

  std::vector<at::Tensor> inputs = {self, batch1, batch2};
  if (!beta.ValueMatchesExclude()) {
    TT_ASSIGN_OR_RETURN(const at::Tensor beta_tensor,
                        beta.GetTensor(out_dtype));
    inputs.push_back(beta_tensor);
  }
  if (!alpha.ValueMatchesExclude()) {
    TT_ASSIGN_OR_RETURN(const at::Tensor alpha_tensor,
                        alpha.GetTensor(out_dtype));
    inputs.push_back(alpha_tensor);
  }

  auto op_builder =
      [beta = std::move(beta), alpha = std::move(alpha), out_dtype](
          absl::Span<mlir::MlirOp> inputs_op,
          mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    const mlir::MlirOp self_op = inputs_op[0];
    const mlir::MlirOp batch1_op = inputs_op[1];
    const mlir::MlirOp batch2_op = inputs_op[2];
    int next_idx = 3;
    std::optional<mlir::MlirOp> beta_op;
    if (!beta.ValueMatchesExclude()) {
      beta_op = inputs_op[next_idx++];
    }
    std::optional<mlir::MlirOp> alpha_op;
    if (!alpha.ValueMatchesExclude()) {
      alpha_op = inputs_op[next_idx++];
    }
    return BuildBaddbmmShlo(self_op, batch1_op, batch2_op, beta_op, alpha_op,
                            beta, alpha, out_dtype);
  };

  TT_ASSIGN_OR_RETURN(
      auto result_buffer,
      DispatchOp<kDynamicSize>(
          std::move(op_builder), inputs,
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
  auto promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero, ScalarValue::kOne);
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kZero,
                                                            ScalarValue::kOne);
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
  auto promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero, ScalarValue::kOne);
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kZero,
                                                            ScalarValue::kOne);
  TT_KERNEL(OpName::kBaddbmmDtypeOut, param_keys,
            (self, batch1, batch2, out_dtype, promoted_beta, promoted_alpha,
             out),
            {
              TT_THROW_IF_ERROR(CheckBaddbmmOut(out));
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
  auto promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero, ScalarValue::kOne);
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kZero,
                                                            ScalarValue::kOne);
  TT_KERNEL(
      OpName::kBaddbmmOut, _,
      (self, batch1, batch2,
       IgnoreInCacheKey(promoted_beta, "Delegated to AtenBaddbmmDtypeOut"),
       IgnoreInCacheKey(promoted_alpha, "Delegated to AtenBaddbmmDtypeOut"),
       out),
      {
        return AtenBaddbmmDtypeOut(self, batch1, batch2, out.scalar_type(),
                                   beta, alpha, out);
      });
}

}  // namespace torch_tpu
