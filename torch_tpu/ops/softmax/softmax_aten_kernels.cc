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

#include "torch_tpu/ops/softmax/softmax_aten_kernels.h"

#include <cstdint>
#include <functional>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
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
#include "torch_tpu/ops/softmax/softmax.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {
namespace {

MlirUnaryOpBuilder GetSoftmaxFunctional(int64_t dim, SoftmaxMode softmax_mode) {
  return std::bind(&BuildSoftmaxShlo, std::placeholders::_1, dim, softmax_mode);
}

NAryMlirOpBuilder<2> GetSoftmaxBackwardDataFunctional(
    int64_t dim, SoftmaxMode softmax_mode,
    mlir::stablehlo::Precision precision) {
  return [dim, softmax_mode, precision](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    auto& [grad_output_op, output_op] = inputs;
    return BuildSoftmaxBackwardDataShlo(grad_output_op, output_op, dim,
                                        precision, softmax_mode);
  };
}

absl::StatusOr<mlir::ElementType> GetComputationDType(const at::Tensor& self,
                                                      bool half_to_float) {
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=all dtypes are supported.
      auto computation_element_type,
      ConvertTo<mlir::ElementType>(self.scalar_type()));
  if (half_to_float && (computation_element_type == mlir::ElementType::F16 ||
                        computation_element_type == mlir::ElementType::BF16)) {
    computation_element_type = mlir::ElementType::F32;
  }
  return computation_element_type;
}

absl::Status SoftmaxInternalOut(const at::Tensor& self, int64_t dim,
                                bool half_to_float, SoftmaxMode softmax_mode,
                                at::Tensor& out, OpParamCacheKeys param_keys) {
  if (self.numel() == 0) {
    out = self;
    return absl::OkStatus();
  }

  // As Pytorch's input and output must be the same type, and softmax does not
  // output non-floating point types. Instead of silent type casting, Pytorch
  // errors out. We should do the same here. Sample Error on CPU:
  // "log_softmax_lastdim_kernel_impl" not implemented for {Type}
  TT_RET_CHECK(self.is_floating_point(), error::kUnimplemented)
      << "not implemented for input type " << ToString(self.scalar_type());

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=all dtypes are supported.
      const mlir::ElementType computation_dtype,
      GetComputationDType(self, half_to_float));
  return UnaryOpOut(self, out, GetSoftmaxFunctional(dim, softmax_mode),
                    {.op_param_cache_keys = std::move(param_keys),
                     .computation_dtype = computation_dtype});
}

absl::StatusOr<DeviceBufferRef> SoftmaxBackwardDataInternalOut(
    const at::Tensor& grad_output, const at::Tensor& output, int64_t dim,
    at::ScalarType input_dtype, SoftmaxMode softmax_mode,
    OpParamCacheKeys param_keys) {
  const auto precision = GetPrecision();
  TT_ASSIGN_OR_RETURN(param_keys,
                      *OpParamCacheKeys::Builder(std::move(param_keys))
                           .SetParam("precision", precision));

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=input checked during forward
                 // pass. It is guaranteed to be floating point in the
                 // backward pass.
      c10::isFloatingType(input_dtype), error::kUnimplemented)
      << "not implemented for input type " << ToString(input_dtype);
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=all dtypes are supported.
      const auto input_mlir_type, ConvertTo<mlir::ElementType>(input_dtype));

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=errors should be covered
                        // inside.
      auto result,
      (DispatchOp<2>(
          GetSoftmaxBackwardDataFunctional(dim, softmax_mode, precision),
          {grad_output, output},
          {.out_dtype = input_mlir_type,
           .out_dims = output.sizes(),
           .op_param_cache_keys = std::move(param_keys)})));
  return result;
}

}  // namespace

at::Tensor& AtenSoftmaxOut(const at::Tensor& self, int64_t dim,
                           bool half_to_float, at::Tensor& out) {
  TT_KERNEL(OpName::kSoftmaxOut, param_keys, (self, dim, half_to_float, out), {
    TT_THROW_IF_ERROR(SoftmaxInternalOut(self, dim, half_to_float,
                                         SoftmaxMode::kSoftmax, out,
                                         std::move(param_keys)));
    return out;
  });
}

at::Tensor& AtenLogSoftmaxOut(const at::Tensor& self, int64_t dim,
                              bool half_to_float, at::Tensor& out) {
  TT_KERNEL(OpName::kLogSoftmaxOut, param_keys, (self, dim, half_to_float, out),
            {
              TT_THROW_IF_ERROR(SoftmaxInternalOut(self, dim, half_to_float,
                                                   SoftmaxMode::kLogSoftmax,
                                                   out, std::move(param_keys)));
              return out;
            });
}

at::Tensor& AtenSoftmaxBackwardDataOut(const at::Tensor& grad_output,
                                       const at::Tensor& output, int64_t dim,
                                       at::ScalarType input_dtype,
                                       at::Tensor& grad_input) {
  TT_KERNEL(OpName::kSoftmaxBackwardDataOut, param_keys,
            (grad_output, output, dim, input_dtype, grad_input), {
              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result,
                  SoftmaxBackwardDataInternalOut(
                      grad_output, output, dim, input_dtype,
                      SoftmaxMode::kSoftmax, std::move(param_keys)));
              TT_THROW_IF_ERROR(  // ERROR_COV_INFEASIBLE=errors should be
                                  // covered inside.
                  AssignBufferToAtTensor(std::move(result), grad_input));
              return grad_input;
            });
}

at::Tensor& AtenLogSoftmaxBackwardDataOut(const at::Tensor& grad_output,
                                          const at::Tensor& output, int64_t dim,
                                          at::ScalarType input_dtype,
                                          at::Tensor& grad_input) {
  TT_KERNEL(OpName::kLogSoftmaxBackwardDataOut, param_keys,
            (grad_output, output, dim, input_dtype, grad_input), {
              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result,
                  SoftmaxBackwardDataInternalOut(
                      grad_output, output, dim, input_dtype,
                      SoftmaxMode::kLogSoftmax, std::move(param_keys)));
              TT_THROW_IF_ERROR(  // ERROR_COV_INFEASIBLE=errors should be
                                  // covered inside.
                  AssignBufferToAtTensor(std::move(result), grad_input));
              return grad_input;
            });
}

}  // namespace torch_tpu
