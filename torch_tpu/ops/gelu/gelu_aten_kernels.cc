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

#include "torch_tpu/ops/gelu/gelu_aten_kernels.h"

#include <functional>
#include <string>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/util/string_view.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/gelu/gelu.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {
namespace {

MlirUnaryOpBuilder GetGeluFunctional(c10::string_view approximate,
                                     mlir::ElementType output_dtype) {
  return [approximate = static_cast<std::string>(approximate),
          output_dtype](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
    return BuildGeluShlo(input, approximate, output_dtype);
  };
}

}  // namespace

at::Tensor AtenGelu(const at::Tensor& self, c10::string_view approximate) {
  TT_KERNEL(OpName::kGelu, param_keys, (self, approximate), {
    TT_CHECK_THROW(approximate == "none" || approximate == "tanh",
                   error::kInvalidArgument)
        << "unsupported approximate argument: " << approximate;
    const c10::ScalarType self_dtype = self.scalar_type();
    TT_CHECK_THROW(!c10::isComplexType(self_dtype), error::kUnimplemented)
        << "unsupported input dtype: '" << ToString(self_dtype) << "'";
    TT_CHECK_THROW(!c10::isIntegralType(self_dtype, /*includeBool=*/true),
                   error::kUnimplemented)
        << "unsupported input dtype: '" << ToString(self_dtype) << "'";
    const c10::ScalarType out_type = InferOutputDtype(self);
    TT_ASSIGN_OR_THROW(const auto out_dtype,
                       ConvertTo<mlir::ElementType>(out_type));
    TT_ASSIGN_OR_THROW(auto out,
                       UnaryOp(self, GetGeluFunctional(approximate, out_dtype),
                               {.op_param_cache_keys = std::move(param_keys),
                                .out_dtype = out_dtype}));
    return out;
  });
}

at::Tensor& AtenGeluOut(const at::Tensor& self, c10::string_view approximate,
                        at::Tensor& out) {
  TT_KERNEL(OpName::kGeluOut, param_keys, (self, approximate, out), {
    TT_CHECK_THROW(approximate == "none" || approximate == "tanh",
                   error::kInvalidArgument)
        << "unsupported approximate argument: " << approximate;
    const c10::ScalarType self_dtype = self.scalar_type();
    TT_CHECK_THROW(!c10::isComplexType(self_dtype), error::kUnimplemented)
        << "unsupported input dtype: '" << ToString(self_dtype) << "'";
    TT_CHECK_THROW(!c10::isIntegralType(self_dtype, /*includeBool=*/true),
                   error::kUnimplemented)
        << "unsupported input dtype: '" << ToString(self_dtype) << "'";
    const c10::ScalarType out_tensor_type = out.scalar_type();
    TT_CHECK_THROW(!c10::isIntegralType(out_tensor_type, /*includeBool=*/true),
                   error::kUnimplemented)
        << "unsupported output dtype: '" << ToString(out_tensor_type) << "'";
    TT_ASSIGN_OR_THROW(const auto out_dtype,
                       ConvertTo<mlir::ElementType>(out_tensor_type));
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, GetGeluFunctional(approximate, out_dtype),
                   {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

at::Tensor& AtenGeluBackwardGradInput(const at::Tensor& grad_output,
                                      const at::Tensor& self,
                                      c10::string_view approximate,
                                      at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kGeluBackwardGradInput, param_keys,
      (grad_output, self, approximate, grad_input), {
        TT_CHECK_THROW(approximate == "none" || approximate == "tanh",
                       error::kInvalidArgument)
            << "unsupported approximate argument: " << approximate;
        const c10::ScalarType self_dtype = self.scalar_type();
        TT_CHECK_THROW(!c10::isComplexType(self_dtype), error::kUnimplemented)
            << "unsupported input dtype: '" << ToString(self_dtype) << "'";
        TT_CHECK_THROW(!c10::isIntegralType(self_dtype, /*includeBool=*/true),
                       error::kUnimplemented)
            << "unsupported input dtype: '" << ToString(self_dtype) << "'";
        const c10::ScalarType output_tensor_type = InferOutputDtype(self);
        TT_ASSIGN_OR_THROW(const auto out_dtype,
                           ConvertTo<mlir::ElementType>(output_tensor_type));

        auto op_builder = [approximate = static_cast<std::string>(approximate),
                           out_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output_op, input_op] = inputs;
          return BuildGeluBackwardGradInputShlo(grad_output_op, input_op,
                                                approximate, out_dtype);
        };
        auto output_shape = CopyIntVector(self.sizes());

        TT_ASSIGN_OR_THROW(
            auto result,
            DispatchOp<2>(std::move(op_builder), {grad_output, self},
                          /*options=*/
                          {.out_dtype = out_dtype,
                           .out_dims = output_shape,
                           .op_param_cache_keys = std::move(param_keys)}));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result), grad_input));
        return grad_input;
      });
}

}  // namespace torch_tpu
