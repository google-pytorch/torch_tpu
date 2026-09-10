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

#include "csrc/ops/digamma/digamma_aten_kernels.h"

#include <limits>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

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
    TT_THROW_IF_ERROR(
        DispatchOpOut<1>(std::move(op_builder), self, out,
                         {.out_dtype = out_dtype,
                          .out_dims = self.sizes(),
                          .op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

}  // namespace torch_tpu
