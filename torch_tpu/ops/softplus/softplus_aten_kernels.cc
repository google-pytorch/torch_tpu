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

#include "torch_tpu/ops/softplus/softplus_aten_kernels.h"

#include <utility>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ATen/core/ATen_fwd.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/softplus/softplus.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> SoftplusShlo(mlir::MlirOp self_op,
                                          mlir::MlirOp beta_op,
                                          mlir::MlirOp threshold_op) {
  TT_ASSIGN_OR_RETURN((auto [self, beta, threshold]),
                      ApplyBroadcastIfNeeded(self_op, beta_op, threshold_op));
  return BuildSoftplusShlo(self, beta, threshold);
}

absl::StatusOr<mlir::MlirOp> SoftplusBackwardShlo(mlir::MlirOp grad_output_op,
                                                  mlir::MlirOp input_op,
                                                  mlir::MlirOp beta_op,
                                                  mlir::MlirOp threshold_op) {
  TT_ASSIGN_OR_RETURN(
      (auto [grad_output, input, beta, threshold]),
      ApplyBroadcastIfNeeded(grad_output_op, input_op, beta_op, threshold_op));
  return BuildSoftplusBackwardShlo(grad_output, input, beta, threshold);
}

}  // namespace

at::Tensor& AtenSoftplusOut(const at::Tensor& self, const at::Scalar& beta,
                            const at::Scalar& threshold, at::Tensor& out) {
  TT_KERNEL(
      OpName::kSoftplusOut, param_keys,
      (self, IgnoreInCacheKey(beta), IgnoreInCacheKey(threshold), out), {
        TT_ASSIGN_OR_THROW(at::Tensor beta_tensor,
                           MakeTensor(beta, self.scalar_type()));
        TT_ASSIGN_OR_THROW(at::Tensor threshold_tensor,
                           MakeTensor(threshold, self.scalar_type()));

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [self_op, beta_op, threshold_op] = inputs;
          return SoftplusShlo(self_op, beta_op, threshold_op);
        };

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));
        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            DispatchOp<3>(OpName::kSoftplusOut, std::move(op_builder),
                          {self, beta_tensor, threshold_tensor},
                          {.out_dtype = out_dtype,
                           .out_dims = CopyIntVector(out.sizes()),
                           .op_param_cache_keys = std::move(param_keys)}));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), out));
        return out;
      });
}

at::Tensor& AtenSoftplusBackwardGradInput(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          const at::Scalar& beta,
                                          const at::Scalar& threshold,
                                          at::Tensor& grad_input) {
  TT_KERNEL(OpName::kSoftplusBackwardGradInput, param_keys,
            (grad_output, self, IgnoreInCacheKey(beta),
             IgnoreInCacheKey(threshold), grad_input),
            {
              TT_ASSIGN_OR_THROW(at::Tensor beta_tensor,
                                 MakeTensor(beta, self.scalar_type()));
              TT_ASSIGN_OR_THROW(at::Tensor threshold_tensor,
                                 MakeTensor(threshold, self.scalar_type()));

              auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 4> inputs)
                  -> absl::StatusOr<mlir::MlirOp> {
                auto& [grad_output_op, input_op, beta_op, threshold_op] =
                    inputs;
                return SoftplusBackwardShlo(grad_output_op, input_op, beta_op,
                                            threshold_op);
              };

              TT_ASSIGN_OR_THROW(
                  mlir::ElementType out_dtype,
                  ConvertTo<mlir::ElementType>(grad_input.scalar_type()));
              TT_ASSIGN_OR_THROW(
                  auto result_buffer,
                  DispatchOp<4>(
                      OpName::kSoftplusBackwardGradInput, std::move(op_builder),
                      {grad_output, self, beta_tensor, threshold_tensor},
                      {.out_dtype = out_dtype,
                       .out_dims = CopyIntVector(grad_input.sizes()),
                       .op_param_cache_keys = std::move(param_keys)}));
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buffer), grad_input));
              return grad_input;
            });
}

}  // namespace torch_tpu
