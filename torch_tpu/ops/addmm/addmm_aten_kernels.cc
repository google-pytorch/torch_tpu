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

#include <cstdint>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/Resize.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/addmm/addmm.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<DeviceBufferRef> Addmm(
    const OpName op_name, const at::Tensor& self, const at::Tensor& mat1,
    const at::Tensor& mat2, const at::Tensor& beta, const at::Tensor& alpha,
    at::ScalarType out_dtype, OpParamCacheKeys param_keys) {
  ABSL_VLOG(1) << "Addmm self: " << ToString(self.sizes())
               << " mat1: " << ToString(mat1.sizes())
               << " mat2: " << ToString(mat2.sizes());

  at::ScalarType mat1_dtype = mat1.scalar_type();
  at::ScalarType mat2_dtype = mat2.scalar_type();

  at::ScalarType computation_scalar_type =
      c10::promoteTypes(self.scalar_type(), mat1_dtype);
  computation_scalar_type =
      c10::promoteTypes(computation_scalar_type, mat2_dtype);

  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(computation_scalar_type));

  auto op_builder = [computation_dtype](FixedSizeSpan<mlir::MlirOp, 5> inputs) {
    auto& [self, mat1, mat2, beta, alpha] = inputs;
    return BuildAddmmShlo(self, mat1, mat2, beta, alpha, computation_dtype);
  };

  TT_ASSIGN_OR_RETURN(const auto out_mlir_dtype,
                      ConvertTo<mlir::ElementType>(out_dtype));
  Dimensions output_dims = {mat1.size(0), mat2.size(1)};

  return DispatchOp<5>(op_name, std::move(op_builder),
                       {self, mat1, mat2, beta, alpha},
                       {.out_dtype = out_mlir_dtype,
                        .out_dims = output_dims,
                        .computation_dtype = computation_dtype,
                        .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor& AtenAddmmOut(const at::Tensor& self_bias, const at::Tensor& mat1,
                         const at::Tensor& mat2, const at::Scalar& beta,
                         const at::Scalar& alpha, at::Tensor& out) {
  TT_KERNEL(
      OpName::kAddmmOut, param_keys,
      (self_bias, mat1, mat2, IgnoreInCacheKey(beta), IgnoreInCacheKey(alpha),
       out),
      {
        TT_ASSIGN_OR_THROW(at::Tensor beta_tensor,
                           MakeTensor(beta, self_bias.scalar_type()));
        TT_ASSIGN_OR_THROW(at::Tensor alpha_tensor,
                           MakeTensor(alpha, self_bias.scalar_type()));
        TT_ASSIGN_OR_THROW(
            auto result_buf,
            Addmm(OpName::kAddmmOut, self_bias, mat1, mat2, beta_tensor,
                  alpha_tensor, out.scalar_type(), std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

at::Tensor AtenAddmmDtype(const at::Tensor& self_bias, const at::Tensor& mat1,
                          const at::Tensor& mat2, at::ScalarType out_dtype,
                          const at::Scalar& beta, const at::Scalar& alpha) {
  TT_KERNEL(OpName::kAddmmDtype, param_keys,
            (self_bias, mat1, mat2, IgnoreInCacheKey(out_dtype),
             IgnoreInCacheKey(beta), IgnoreInCacheKey(alpha)),
            {
              TT_ASSIGN_OR_THROW(at::Tensor beta_tensor,
                                 MakeTensor(beta, self_bias.scalar_type()));
              TT_ASSIGN_OR_THROW(at::Tensor alpha_tensor,
                                 MakeTensor(alpha, self_bias.scalar_type()));
              TT_ASSIGN_OR_THROW(
                  auto result_buf,
                  Addmm(OpName::kAddmmDtype, self_bias, mat1, mat2, beta_tensor,
                        alpha_tensor, out_dtype, std::move(param_keys)));
              return MakeTensor(std::move(result_buf));
            });
}

at::Tensor& AtenAddmmDtypeOut(const at::Tensor& self_bias,
                              const at::Tensor& mat1, const at::Tensor& mat2,
                              at::ScalarType out_dtype, const at::Scalar& beta,
                              const at::Scalar& alpha, at::Tensor& out) {
  TT_KERNEL(
      OpName::kAddmmDtypeOut, param_keys,
      (self_bias, mat1, mat2, IgnoreInCacheKey(out_dtype),
       IgnoreInCacheKey(beta), IgnoreInCacheKey(alpha), out),
      {
        TT_ASSIGN_OR_THROW(at::Tensor beta_tensor,
                           MakeTensor(beta, self_bias.scalar_type()));
        TT_ASSIGN_OR_THROW(at::Tensor alpha_tensor,
                           MakeTensor(alpha, self_bias.scalar_type()));
        TT_ASSIGN_OR_THROW(
            auto result_buf,
            Addmm(OpName::kAddmmDtypeOut, self_bias, mat1, mat2, beta_tensor,
                  alpha_tensor, out_dtype, std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
