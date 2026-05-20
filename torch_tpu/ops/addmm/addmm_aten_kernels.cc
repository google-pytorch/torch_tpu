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

#include "torch_tpu/ops/addmm/addmm_aten_kernels.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/Resize.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/precision_context.h"

namespace torch_tpu {

namespace {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildAddmmShlo(
    mlir::MlirOp self_op, mlir::MlirOp mat1_op, mlir::MlirOp mat2_op,
    std::optional<mlir::MlirOp> beta_op, std::optional<mlir::MlirOp> alpha_op,
    const MaybePromotedScalar& beta, const MaybePromotedScalar& alpha,
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

  const int64_t out_dim0 = mat1_type.getShape()[0];
  const int64_t out_dim1 = mat2_type.getShape()[1];

  mlir::MlirOp dot_result;
  if (!alpha_op.has_value() && alpha.IsZero()) {
    llvm::SmallVector<int64_t, 2> out_shape = {out_dim0, out_dim1};
    TT_ASSIGN_OR_RETURN(dot_result,
                        MakeConstant(self_op.getBuilder(), at::Scalar(0.0),
                                     out_dtype, out_shape));
  } else {
    dot_result = stablehlo::Dot(mat1_op, mat2_op, precision_config);
    if (is_any_i64) {
      dot_result = stablehlo::ConvertElementType(dot_result, out_dtype);
    }
  }

  mlir::MlirOp scaled_dot_result = dot_result;
  if (alpha_op.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp alpha_tensor,
                        BroadcastIfNeeded(*alpha_op, dot_result));
    scaled_dot_result = stablehlo::Mul(scaled_dot_result, alpha_tensor);
  } else {
    TT_RET_CHECK(alpha.IsOne() || alpha.IsZero(), error::kInvalidArgument)
        << "expected alpha to be 0 or 1 when not using dynamic alpha, got "
        << alpha.scalar().toDouble();
  }

  if (beta.IsZero()) {
    return scaled_dot_result;
  }

  mlir::MlirOp scaled_input_result = self_op;
  if (beta_op.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp beta_tensor,
                        BroadcastIfNeeded(*beta_op, self_op));
    scaled_input_result = stablehlo::Mul(scaled_input_result, beta_tensor);
  } else {
    TT_RET_CHECK(beta.IsOne(), error::kInvalidArgument)
        << "expected beta to be 1 when not using dynamic beta, got "
        << beta.scalar().toDouble();
  }

  TT_ASSIGN_OR_RETURN(
      (auto [bcast_scaled_input_result, bcast_scaled_dot_result]),
      ApplyBroadcastIfNeeded(scaled_input_result, scaled_dot_result));

  return stablehlo::Add(bcast_scaled_input_result, bcast_scaled_dot_result);
}

absl::Status CheckInputBroadcast(const at::Tensor& self,
                                 const Dimensions& output_dims_vec) {
  if (self.dim() == 1 && self.size(0) == 1) {
    return absl::OkStatus();
  }
  TT_RET_CHECK(self.dim() <= output_dims_vec.size(), error::kInvalidArgument)
      << "input tensor should not have more dimensions than the"
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

absl::StatusOr<DeviceBufferRef> AddMm(
    const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2,
    MaybePromotedScalar beta, MaybePromotedScalar alpha,
    at::ScalarType out_scalar_type, OpParamCacheKeys& param_keys) {
  TT_RET_CHECK(self.scalar_type() != at::kBool &&
                   mat1.scalar_type() != at::kBool &&
                   mat2.scalar_type() != at::kBool &&
                   beta.scalar().type() != at::kBool &&
                   alpha.scalar().type() != at::kBool,
               error::kInvalidArgument)
      << "boolean dtypes are not supported";
  TT_RET_CHECK(!self.is_complex() && !mat1.is_complex() && !mat2.is_complex() &&
                   beta.scalar().type() != at::kComplexFloat &&
                   beta.scalar().type() != at::kComplexDouble &&
                   alpha.scalar().type() != at::kComplexFloat &&
                   alpha.scalar().type() != at::kComplexDouble,
               error::kUnimplemented)
      << "complex dtypes are not yet supported";
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch prevents non-matrix mat1.
      mat1.dim() == 2, error::kInvalidArgument)
      << "mat1 must be a matrix, got " << mat1.dim() << "-D tensor";
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch prevents non-matrix mat2.
      mat2.dim() == 2, error::kInvalidArgument)
      << "mat2 must be a matrix, got " << mat2.dim() << "-D tensor";
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch prevents mismatch.
      mat1.size(1) == mat2.size(0), error::kInvalidArgument)
      << "size 1 of mat1 must be same as size 0 of mat2, got " << mat1.size(1)
      << " and " << mat2.size(0) << " respectively";

  Dimensions output_dims_vec = {mat1.size(0), mat2.size(1)};
  TT_RETURN_IF_ERROR(CheckInputBroadcast(self, output_dims_vec));
  TT_ASSIGN_OR_RETURN(mlir::ElementType output_dtype_mlir,
                      ConvertTo<mlir::ElementType>(out_scalar_type),
                      _.SetOverride()
                          << "TorchTPU does not yet support the output dtype "
                          << ToString(out_scalar_type));
  const auto current_precision = GetPrecision();
  TT_ASSIGN_OR_RETURN(param_keys,
                      *OpParamCacheKeys::Builder(std::move(param_keys))
                           .SetParam("precision", current_precision));

  std::vector<at::Tensor> inputs = {self, mat1, mat2};
  if (!beta.ValueMatchesExclude()) {
    TT_ASSIGN_OR_RETURN(const at::Tensor beta_tensor,
                        beta.GetTensor(out_scalar_type));
    inputs.push_back(beta_tensor);
  }
  if (!alpha.ValueMatchesExclude()) {
    TT_ASSIGN_OR_RETURN(const at::Tensor alpha_tensor,
                        alpha.GetTensor(out_scalar_type));
    inputs.push_back(alpha_tensor);
  }

  auto op_builder = [beta = std::move(beta), alpha = std::move(alpha),
                     output_dtype_mlir, output_dims_vec, current_precision](
                        absl::Span<mlir::MlirOp> inputs_op,
                        mlir::MlirBuilder&) -> absl::StatusOr<mlir::MlirOp> {
    const size_t expected_size = 3 + (!beta.ValueMatchesExclude() ? 1 : 0) +
                                 (!alpha.ValueMatchesExclude() ? 1 : 0);
    TT_RET_CHECK(inputs_op.size() == expected_size, error::kInternal)
        << "expected " << expected_size << " inputs, got " << inputs_op.size();

    const mlir::MlirOp self_op = inputs_op[0];
    const mlir::MlirOp mat1_op = inputs_op[1];
    const mlir::MlirOp mat2_op = inputs_op[2];
    int next_idx = 3;

    std::optional<mlir::MlirOp> beta_op;
    if (!beta.ValueMatchesExclude()) {
      beta_op = inputs_op[next_idx++];
    }
    std::optional<mlir::MlirOp> alpha_op;
    if (!alpha.ValueMatchesExclude()) {
      alpha_op = inputs_op[next_idx++];
    }

    return BuildAddmmShlo(self_op, mat1_op, mat2_op, beta_op, alpha_op, beta,
                          alpha, output_dtype_mlir, current_precision);
  };

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=No user triggerable errors.
      auto result_buffer,
      DispatchOp<kDynamicSize>(
          std::move(op_builder), inputs,
          // It's OK for all 3 variants of Addmm to use the same OpName for
          // dispatching, as the logic is the same for all 3.
          {.op_name = OpName::kAddmm,
           .out_dtype = output_dtype_mlir,
           .out_dims = output_dims_vec,
           .op_param_cache_keys = std::move(param_keys)}));

  return result_buffer;
}

}  // namespace

at::Tensor& AtenAddmmOut(const at::Tensor& self, const at::Tensor& mat1,
                         const at::Tensor& mat2, const at::Scalar& beta,
                         const at::Scalar& alpha, at::Tensor& out) {
  MaybePromotedScalar promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero, ScalarValue::kOne);
  MaybePromotedScalar promoted_alpha = PromoteScalar(alpha).AvoidPromoting(
      ScalarValue::kZero, ScalarValue::kOne);
  TT_KERNEL(
      OpName::kAddmmOut, param_keys,
      (self, mat1, mat2, promoted_beta, promoted_alpha, out), {
        TT_CHECK_THROW(out.scalar_type() == self.scalar_type(),
                       error::kInvalidArgument)
            << "input and out tensors expected to have the same dtype, got "
               "input dtype "
            << torch_tpu::ToString(self.scalar_type()) << " and out dtype "
            << torch_tpu::ToString(out.scalar_type());
        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            AddMm(self, mat1, mat2, std::move(promoted_beta),
                  std::move(promoted_alpha), out.scalar_type(), param_keys));
        at::IntArrayRef output_sizes_array_ref(result_buffer.dimensions());
        ABSL_VLOG(3) << "calling at::native::resize_output for out tensor with "
                        "target shape: ["
                     << absl::StrJoin(output_sizes_array_ref, ", ") << "]";
        at::native::resize_output(out, output_sizes_array_ref);
        TT_THROW_IF_ERROR(  // ERROR_COV_INFEASIBLE=No user triggerable errors.
            AssignBufferToAtTensor(std::move(result_buffer), out));
        return out;
      });
}

at::Tensor AtenAddmmDtype(const at::Tensor& self, const at::Tensor& mat1,
                          const at::Tensor& mat2, at::ScalarType out_dtype,
                          const at::Scalar& beta, const at::Scalar& alpha) {
  MaybePromotedScalar promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero, ScalarValue::kOne);
  MaybePromotedScalar promoted_alpha = PromoteScalar(alpha).AvoidPromoting(
      ScalarValue::kZero, ScalarValue::kOne);
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
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero, ScalarValue::kOne);
  MaybePromotedScalar promoted_alpha = PromoteScalar(alpha).AvoidPromoting(
      ScalarValue::kZero, ScalarValue::kOne);
  TT_KERNEL(
      OpName::kAddmmDtypeOut, param_keys,
      (self, mat1, mat2, out_dtype, promoted_beta, promoted_alpha, out), {
        TT_CHECK_THROW(out.scalar_type() == out_dtype, error::kInvalidArgument)
            << "out dtype should match out_dtype, got out dtype "
            << torch_tpu::ToString(out.scalar_type()) << " and out_dtype "
            << torch_tpu::ToString(out_dtype);
        TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=errors from addmm
                             // tested individually.
            auto result_buffer,
            AddMm(self, mat1, mat2, std::move(promoted_beta),
                  std::move(promoted_alpha), out_dtype, param_keys));
        TT_THROW_IF_ERROR(  // ERROR_COV_INFEASIBLE=No user triggerable errors.
            AssignBufferToAtTensor(std::move(result_buffer), out));
        return out;
      });
}

}  // namespace torch_tpu
