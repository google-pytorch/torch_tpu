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

// In csrc/ops/{op_name}/{op_name}_aten_kernels.cc
// (This is a reference file for agents)

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildCustomScaledAddShlo(mlir::MlirOp x_op,
                                                      mlir::MlirOp y_op,
                                                      mlir::MlirOp alpha_op) {
  auto scaled_y = mlir::stablehlo::Mul(alpha_op, y_op);
  return mlir::stablehlo::Add(x_op, scaled_y);
}

}  // namespace

at::Tensor AtenCustomScaledAdd(const at::Tensor& x, const at::Tensor& y,
                               at::Scalar alpha) {
  auto promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(OpName::kCustomKernel, param_keys, (x, y, promoted_alpha), {
    const auto x_dtype = x.scalar_type();
    const Dimensions out_dims = CopyIntVector(x.sizes());

    TT_ASSIGN_OR_THROW(const auto mlir_dtype,
                       ConvertTo<mlir::ElementType>(x_dtype));

    TT_ASSIGN_OR_THROW(at::Tensor alpha_tensor,
                       promoted_alpha.GetTensor(x_dtype));

    auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto [x, y, alpha] = inputs;
      return BuildCustomScaledAddShlo(x, y, alpha);
    };

    TT_ASSIGN_OR_THROW(
        auto result,
        DispatchOp<3>(std::move(op_builder), {x, y, alpha_tensor},
                      {.out_dtype = mlir_dtype,
                       .out_dims = out_dims,
                       .op_param_cache_keys = std::move(param_keys)}));
    return MakeTensor(std::move(result));
  });
}

// Out-variant kernel: always use DispatchOpOut to automatically configure
// buffer donation and return 'out'.
at::Tensor& AtenCustomScaledAddOut(const at::Tensor& x, const at::Tensor& y,
                                   at::Scalar alpha, at::Tensor& out) {
  auto promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(OpName::kCustomKernel, param_keys, (x, y, promoted_alpha, out), {
    const auto x_dtype = x.scalar_type();
    const Dimensions out_dims = CopyIntVector(x.sizes());

    TT_ASSIGN_OR_THROW(const auto mlir_dtype,
                       ConvertTo<mlir::ElementType>(x_dtype));

    TT_ASSIGN_OR_THROW(at::Tensor alpha_tensor,
                       promoted_alpha.GetTensor(x_dtype));

    auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto [x, y, alpha] = inputs;
      return BuildCustomScaledAddShlo(x, y, alpha);
    };

    TT_THROW_IF_ERROR(
        DispatchOpOut<3>(std::move(op_builder), {x, y, alpha_tensor}, out,
                         {.out_dtype = mlir_dtype,
                          .out_dims = out_dims,
                          .op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

}  // namespace torch_tpu
