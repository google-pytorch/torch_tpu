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

#include "torch_tpu/ops/digamma/digamma_aten_kernels.h"

#include <limits>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {
absl::StatusOr<mlir::MlirOp> BuildDigammaShlo(mlir::MlirOp input_op,
                                              mlir::ElementType out_mlir_type) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp op,
                      ::torch_tpu::ConvertIfInteger(input_op, out_mlir_type));
  mlir::MlirOp digamma_op = mlir::chlo::Digamma(op);

  auto zero_cst = MakeConstantLike(op, 0.0);
  auto is_zero = mlir::stablehlo::Compare(
      op, zero_cst, mlir::stablehlo::ComparisonDirection::EQ);

  auto neginf_cst =
      MakeConstantLike(op, -std::numeric_limits<double>::infinity());

  // Return -inf if input is 0, else return digamma(input)
  return mlir::stablehlo::Select(is_zero, neginf_cst, digamma_op);
}

at::Tensor& AtenDigammaOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kDigammaOut, param_keys, (self, out), {
    TT_CHECK_THROW(!IsComplex(self), error::kInvalidArgument)
        << "expected the input dtype not to be complex, got "
        << ToString(self.scalar_type());

    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    auto op_builder =
        [out_dtype](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildDigammaShlo(input, out_dtype);
    };
    TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));
    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<1>(std::move(op_builder), self,
                      {.out_dtype = out_dtype,
                       .out_dims = out.sizes(),
                       .op_param_cache_keys = std::move(param_keys)}));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

}  // namespace torch_tpu
