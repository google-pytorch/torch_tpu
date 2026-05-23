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

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/Resize.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "c10/core/ScalarType.h"
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
#include "torch_tpu/common/fixed_size_span.h"
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
    const at::Scalar& beta, const at::Scalar& alpha,
    mlir::ElementType out_dtype, mlir::stablehlo::Precision precision) {
  const mlir::RankedTensorType mat1_type = GetTensorTypeOrDie(mat1_op);
  const mlir::RankedTensorType mat2_type = GetTensorTypeOrDie(mat2_op);

  // TODO: XLA doesn't support matmuls with i64, so we convert them to f64.
  bool is_any_i64 = mat1_type.getElementType().isInteger(64) ||
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
  auto alpha_op = MakeConstantLike(dot_result, alpha.toDouble(), out_dtype);
  auto scaled_dot_result = stablehlo::Mul(alpha_op, dot_result);

  const bool is_beta_zero = !beta.toBool();
  if (is_beta_zero) {
    return scaled_dot_result;
  }
  auto beta_op = MakeConstantLike(self_op, beta.toDouble(), out_dtype);
  auto scaled_input_result = stablehlo::Mul(self_op, beta_op);
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
    const at::Scalar& beta, const at::Scalar& alpha,
    at::ScalarType out_scalar_type, OpParamCacheKeys& param_keys) {
  TT_RET_CHECK(self.scalar_type() != at::kBool &&
                   mat1.scalar_type() != at::kBool &&
                   mat2.scalar_type() != at::kBool &&
                   beta.type() != at::kBool && alpha.type() != at::kBool,
               error::kInvalidArgument)
      << "boolean dtypes are not supported";
  TT_RET_CHECK(!self.is_complex() && !mat1.is_complex() && !mat2.is_complex() &&
                   beta.type() != at::kComplexFloat &&
                   beta.type() != at::kComplexDouble &&
                   alpha.type() != at::kComplexFloat &&
                   alpha.type() != at::kComplexDouble,
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

  auto op_builder = [beta, alpha, output_dtype_mlir, output_dims_vec,
                     current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, mat1_op, mat2_op] = inputs;
    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=No user triggerable errors.
        auto result, BuildAddmmShlo(self_op, mat1_op, mat2_op, beta, alpha,
                                    output_dtype_mlir, current_precision));
    return result;
  };

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=No user triggerable errors.
      auto result_buffer,
      DispatchOp<3>(
          std::move(op_builder), {self, mat1, mat2},
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
  TT_KERNEL(
      OpName::kAddmmOut, param_keys, (self, mat1, mat2, beta, alpha, out), {
        TT_CHECK_THROW(out.scalar_type() == self.scalar_type(),
                       error::kInvalidArgument)
            << "input and out tensors expected to have the same dtype, got "
               "input dtype "
            << torch_tpu::ToString(self.scalar_type()) << " and out dtype "
            << torch_tpu::ToString(out.scalar_type());
        TT_ASSIGN_OR_THROW(auto result_buffer,
                           AddMm(self, mat1, mat2, beta, alpha,
                                 out.scalar_type(), param_keys));
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
  TT_KERNEL(OpName::kAddmmDtype, param_keys,
            (self, mat1, mat2, out_dtype, beta, alpha), {
              TT_ASSIGN_OR_THROW(
                  auto result_buffer,
                  AddMm(self, mat1, mat2, beta, alpha, out_dtype, param_keys));
              return MakeTensor(std::move(result_buffer));
            });
}

at::Tensor& AtenAddmmDtypeOut(const at::Tensor& self, const at::Tensor& mat1,
                              const at::Tensor& mat2, at::ScalarType out_dtype,
                              const at::Scalar& beta, const at::Scalar& alpha,
                              at::Tensor& out) {
  TT_KERNEL(
      OpName::kAddmmDtypeOut, param_keys,
      (self, mat1, mat2, out_dtype, beta, alpha, out), {
        TT_CHECK_THROW(out.scalar_type() == out_dtype, error::kInvalidArgument)
            << "out dtype should match out_dtype, got out dtype "
            << torch_tpu::ToString(out.scalar_type()) << " and out_dtype "
            << torch_tpu::ToString(out_dtype);
        TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=errors from addmm
                             // tested individually.
            auto result_buffer,
            AddMm(self, mat1, mat2, beta, alpha, out_dtype, param_keys));
        TT_THROW_IF_ERROR(  // ERROR_COV_INFEASIBLE=No user triggerable errors.
            AssignBufferToAtTensor(std::move(result_buffer), out));
        return out;
      });
}

}  // namespace torch_tpu
