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

#include "csrc/ops/addmm/addmm_aten_kernels.h"

#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/gelu/gelu.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "csrc/ops/unary.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildAddmmShlo(
    mlir::MlirOp self_op, mlir::MlirOp mat1_op, mlir::MlirOp mat2_op,
    std::optional<mlir::MlirOp> beta_op, mlir::MlirOp alpha_op,
    mlir::ElementType out_dtype, mlir::stablehlo::Precision precision) {
  const mlir::RankedTensorType mat1_type = GetTensorTypeOrDie(mat1_op);
  const mlir::RankedTensorType mat2_type = GetTensorTypeOrDie(mat2_op);

  // TODO: XLA doesn't support matmuls with i64, so we convert them to f64.
  const bool is_any_i64 = mat1_type.getElementType().isInteger(64) ||
                          mat2_type.getElementType().isInteger(64);
  if (is_any_i64) {
    mat1_op = stablehlo::ConvertElementType(mat1_op, mlir::ElementType::F64);
    mat2_op = stablehlo::ConvertElementType(mat2_op, mlir::ElementType::F64);
  }

  const auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
      &self_op.getContext(), {precision, precision});

  mlir::MlirOp dot_result = stablehlo::Dot(mat1_op, mat2_op, precision_config);
  if (is_any_i64) {
    dot_result = stablehlo::ConvertElementType(dot_result, out_dtype);
  }

  TT_ASSIGN_OR_RETURN(mlir::MlirOp alpha_tensor,
                      BroadcastIfNeeded(alpha_op, dot_result));
  mlir::MlirOp scaled_dot_result = stablehlo::Mul(dot_result, alpha_tensor);

  if (!beta_op.has_value()) {
    return scaled_dot_result;
  }

  TT_ASSIGN_OR_RETURN(mlir::MlirOp beta_tensor,
                      BroadcastIfNeeded(*beta_op, self_op));
  mlir::MlirOp scaled_input_result = stablehlo::Mul(self_op, beta_tensor);

  TT_ASSIGN_OR_RETURN(
      (auto [bcast_scaled_input_result, bcast_scaled_dot_result]),
      ApplyBroadcastIfNeeded(scaled_input_result, scaled_dot_result));

  return stablehlo::Add(bcast_scaled_input_result, bcast_scaled_dot_result);
}

absl::Status ValidateInputBroadcast(const at::Tensor& self,
                                    const Dimensions& output_dims_vec) {
  if (self.dim() == 1 && self.size(0) == 1) {
    return absl::OkStatus();
  }
  TT_RET_CHECK(self.dim() <= output_dims_vec.size(), error::kInvalidArgument)
      << "expected input tensor to have at most as many dimensions as the"
      << " product of mat1 @ mat2, got " << self.dim() << "-D input and "
      << output_dims_vec.size() << "-D product of mat1 @ mat2";
  absl::StatusOr<Dimensions> broadcast_shape =
      InferSize(self.sizes(), output_dims_vec);
  TT_RET_CHECK(broadcast_shape.ok() && *broadcast_shape == output_dims_vec,
               error::kInvalidArgument)
      << "input tensor shape [" << absl::StrJoin(self.sizes(), ", ")
      << "] cannot be broadcasted to matmul result shape ["
      << absl::StrJoin(output_dims_vec, ", ") << "]";
  return absl::OkStatus();
}

// Validates input dtypes, matrix dimensions, and broadcastability for addmm
// variants, returning the corresponding MLIR element type for the output.
absl::StatusOr<mlir::ElementType> ValidateAddmmInputsAndGetOutputDtype(
    const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2,
    const MaybePromotedScalar& beta, const PromotedScalar& alpha,
    at::ScalarType out_scalar_type) {
  TT_RET_CHECK(self.scalar_type() != at::kBool &&
                   mat1.scalar_type() != at::kBool &&
                   mat2.scalar_type() != at::kBool &&
                   beta.scalar().type() != at::kBool &&
                   alpha.scalar().type() != at::kBool,
               error::kInvalidArgument)
      << "boolean dtypes are not supported";
  TT_RET_CHECK(
      mat1.numel() == 0 || mat2.numel() == 0 ||
          (mat1.scalar_type() != at::kInt && mat1.scalar_type() != at::kLong),
      error::kPythonNotImplementedError)
      << "not implemented for " << ToString(mat1.scalar_type());
  TT_RET_CHECK(!self.is_complex() && !mat1.is_complex() && !mat2.is_complex() &&
                   beta.scalar().type() != at::kComplexFloat &&
                   beta.scalar().type() != at::kComplexDouble &&
                   alpha.scalar().type() != at::kComplexFloat &&
                   alpha.scalar().type() != at::kComplexDouble,
               error::kPythonNotImplementedError)
      << "complex dtypes are not yet supported";
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch prevents non-matrix mat1.
      mat1.dim() == 2, error::kInvalidArgument)
      << "expected mat1 to be a matrix, got " << mat1.dim() << "-D tensor";
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch prevents non-matrix mat2.
      mat2.dim() == 2, error::kInvalidArgument)
      << "expected mat2 to be a matrix, got " << mat2.dim() << "-D tensor";
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch prevents mismatch.
      mat1.size(1) == mat2.size(0), error::kInvalidArgument)
      << "expected size 1 of mat1 to be same as size 0 of mat2, got "
      << mat1.size(1) << " and " << mat2.size(0) << " respectively";

  Dimensions output_dims_vec = {mat1.size(0), mat2.size(1)};
  TT_RETURN_IF_ERROR(ValidateInputBroadcast(self, output_dims_vec));
  TT_ASSIGN_OR_RETURN(mlir::ElementType output_dtype_mlir,
                      ConvertTo<mlir::ElementType>(out_scalar_type),
                      _.SetOverride()
                          << "TorchTPU does not yet support the output dtype "
                          << ToString(out_scalar_type));
  return output_dtype_mlir;
}

template <typename ReturnType, typename Dispatch4, typename Dispatch5>
ReturnType DispatchAddmm(const at::Tensor& self, const at::Tensor& mat1,
                         const at::Tensor& mat2, MaybePromotedScalar beta,
                         PromotedScalar alpha, at::ScalarType out_scalar_type,
                         OpParamCacheKeys& param_keys,
                         std::optional<bool> use_gelu, Dispatch4&& dispatch4,
                         Dispatch5&& dispatch5) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType output_dtype_mlir,
                      ValidateAddmmInputsAndGetOutputDtype(
                          self, mat1, mat2, beta, alpha, out_scalar_type));
  Dimensions output_dims_vec = {mat1.size(0), mat2.size(1)};
  const auto current_precision = GetAndAddPrecisionTo(param_keys);

  auto apply_activation =
      [use_gelu,
       output_dtype_mlir](mlir::MlirOp res) -> absl::StatusOr<mlir::MlirOp> {
    if (!use_gelu.has_value()) {
      return res;
    }
    if (*use_gelu) {
      return BuildGeluShlo(res, "none", output_dtype_mlir);
    }
    return BuildReluShlo(res);
  };

  if (beta.ValueMatchesExclude()) {
    TT_ASSIGN_OR_RETURN(const at::Tensor alpha_tensor,
                        alpha.GetTensor(out_scalar_type));
    auto op_builder = [output_dtype_mlir, current_precision, apply_activation](
                          FixedSizeSpan<mlir::MlirOp, 4> inputs_op)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [self_op, mat1_op, mat2_op, alpha_op] = inputs_op;
      TT_ASSIGN_OR_RETURN(
          mlir::MlirOp res,
          BuildAddmmShlo(self_op, mat1_op, mat2_op, std::nullopt, alpha_op,
                         output_dtype_mlir, current_precision));
      return apply_activation(res);
    };
    return dispatch4(
        std::move(op_builder), alpha_tensor,
        DispatchOpOptions<1>{.out_dtype = output_dtype_mlir,
                             .out_dims = output_dims_vec,
                             .op_param_cache_keys = std::move(param_keys)});
  }

  TT_ASSIGN_OR_RETURN(const at::Tensor beta_tensor,
                      beta.GetTensor(out_scalar_type));
  TT_ASSIGN_OR_RETURN(const at::Tensor alpha_tensor,
                      alpha.GetTensor(out_scalar_type));
  auto op_builder = [output_dtype_mlir, current_precision,
                     apply_activation](FixedSizeSpan<mlir::MlirOp, 5> inputs_op)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, mat1_op, mat2_op, beta_op, alpha_op] = inputs_op;
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp res,
        BuildAddmmShlo(self_op, mat1_op, mat2_op, beta_op, alpha_op,
                       output_dtype_mlir, current_precision));
    return apply_activation(res);
  };
  return dispatch5(
      std::move(op_builder), beta_tensor, alpha_tensor,
      DispatchOpOptions<1>{.out_dtype = output_dtype_mlir,
                           .out_dims = output_dims_vec,
                           .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRef> AddMm(
    const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2,
    MaybePromotedScalar beta, PromotedScalar alpha,
    at::ScalarType out_scalar_type, OpParamCacheKeys& param_keys) {
  return DispatchAddmm<absl::StatusOr<DeviceBufferRef>>(
      self, mat1, mat2, std::move(beta), std::move(alpha), out_scalar_type,
      param_keys, /*use_gelu=*/std::nullopt,
      [&](auto op_builder, const at::Tensor& alpha_tensor, auto options) {
        return DispatchOp<4>(std::move(op_builder),
                             {self, mat1, mat2, alpha_tensor},
                             std::move(options));
      },
      [&](auto op_builder, const at::Tensor& beta_tensor,
          const at::Tensor& alpha_tensor, auto options) {
        return DispatchOp<5>(std::move(op_builder),
                             {self, mat1, mat2, beta_tensor, alpha_tensor},
                             std::move(options));
      });
}

absl::Status AddMmOut(const at::Tensor& self, const at::Tensor& mat1,
                      const at::Tensor& mat2, MaybePromotedScalar beta,
                      PromotedScalar alpha, at::ScalarType out_scalar_type,
                      OpParamCacheKeys& param_keys, at::Tensor& out) {
  return DispatchAddmm<absl::Status>(
      self, mat1, mat2, std::move(beta), std::move(alpha), out_scalar_type,
      param_keys, /*use_gelu=*/std::nullopt,
      [&](auto op_builder, const at::Tensor& alpha_tensor, auto options) {
        return DispatchOpOut<4>(std::move(op_builder),
                                {self, mat1, mat2, alpha_tensor}, out,
                                std::move(options));
      },
      [&](auto op_builder, const at::Tensor& beta_tensor,
          const at::Tensor& alpha_tensor, auto options) {
        return DispatchOpOut<5>(std::move(op_builder),
                                {self, mat1, mat2, beta_tensor, alpha_tensor},
                                out, std::move(options));
      });
}

absl::Status AddMmActivationOut(const at::Tensor& self, const at::Tensor& mat1,
                                const at::Tensor& mat2,
                                MaybePromotedScalar beta, PromotedScalar alpha,
                                bool use_gelu, at::ScalarType out_scalar_type,
                                OpParamCacheKeys& param_keys, at::Tensor& out) {
  return DispatchAddmm<absl::Status>(
      self, mat1, mat2, std::move(beta), std::move(alpha), out_scalar_type,
      param_keys, use_gelu,
      [&](auto op_builder, const at::Tensor& alpha_tensor, auto options) {
        return DispatchOpOut<4>(std::move(op_builder),
                                {self, mat1, mat2, alpha_tensor}, out,
                                std::move(options));
      },
      [&](auto op_builder, const at::Tensor& beta_tensor,
          const at::Tensor& alpha_tensor, auto options) {
        return DispatchOpOut<5>(std::move(op_builder),
                                {self, mat1, mat2, beta_tensor, alpha_tensor},
                                out, std::move(options));
      });
}

}  // namespace

at::Tensor& AtenAddmmOut(const at::Tensor& self, const at::Tensor& mat1,
                         const at::Tensor& mat2, const at::Scalar& beta,
                         const at::Scalar& alpha, at::Tensor& out) {
  MaybePromotedScalar promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero);
  PromotedScalar promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(
      OpName::kAddmmOut, param_keys,
      (self, mat1, mat2, promoted_beta, promoted_alpha, out), {
        TT_CHECK_THROW(out.scalar_type() == self.scalar_type(),
                       error::kInvalidArgument)
            << "expected input and out tensors to have the same dtype, got "
            << torch_tpu::ToString(self.scalar_type()) << " vs "
            << torch_tpu::ToString(out.scalar_type());
        TT_THROW_IF_ERROR(AddMmOut(self, mat1, mat2, std::move(promoted_beta),
                                   std::move(promoted_alpha), out.scalar_type(),
                                   param_keys, out));
        return out;
      });
}

at::Tensor& AtenAddmmActivationOut(const at::Tensor& self,
                                   const at::Tensor& mat1,
                                   const at::Tensor& mat2,
                                   const at::Scalar& beta,
                                   const at::Scalar& alpha, bool use_gelu,
                                   at::Tensor& out) {
  MaybePromotedScalar promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero);
  PromotedScalar promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(
      OpName::kAddmmActivationOut, param_keys,
      (self, mat1, mat2, promoted_beta, promoted_alpha, use_gelu, out), {
        TT_CHECK_THROW(out.scalar_type() == self.scalar_type(),
                       error::kInvalidArgument)
            << "expected input and out tensors to have the same dtype, got "
            << torch_tpu::ToString(self.scalar_type()) << " vs "
            << torch_tpu::ToString(out.scalar_type());
        TT_THROW_IF_ERROR(
            AddMmActivationOut(self, mat1, mat2, std::move(promoted_beta),
                               std::move(promoted_alpha), use_gelu,
                               out.scalar_type(), param_keys, out));
        return out;
      });
}

at::Tensor AtenAddmmDtype(const at::Tensor& self, const at::Tensor& mat1,
                          const at::Tensor& mat2, at::ScalarType out_dtype,
                          const at::Scalar& beta, const at::Scalar& alpha) {
  MaybePromotedScalar promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero);
  PromotedScalar promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(OpName::kAddmmDtype, param_keys,
            (self, mat1, mat2, out_dtype, promoted_beta, promoted_alpha), {
              TT_ASSIGN_OR_THROW(
                  auto result_buffer,
                  AddMm(self, mat1, mat2, std::move(promoted_beta),
                        std::move(promoted_alpha), out_dtype, param_keys));
              return MakeTensor(std::move(result_buffer));
            });
}

at::Tensor& AtenAddmmDtypeOut(const at::Tensor& self, const at::Tensor& mat1,
                              const at::Tensor& mat2, at::ScalarType out_dtype,
                              const at::Scalar& beta, const at::Scalar& alpha,
                              at::Tensor& out) {
  MaybePromotedScalar promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero);
  PromotedScalar promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(
      OpName::kAddmmDtypeOut, param_keys,
      (self, mat1, mat2, out_dtype, promoted_beta, promoted_alpha, out), {
        TT_CHECK_THROW(out.scalar_type() == out_dtype, error::kInvalidArgument)
            << "expected out dtype to match out_dtype, got out dtype "
            << torch_tpu::ToString(out.scalar_type()) << " and out_dtype "
            << torch_tpu::ToString(out_dtype);
        TT_THROW_IF_ERROR(AddMmOut(self, mat1, mat2, std::move(promoted_beta),
                                   std::move(promoted_alpha), out_dtype,
                                   param_keys, out));
        return out;
      });
}

}  // namespace torch_tpu
