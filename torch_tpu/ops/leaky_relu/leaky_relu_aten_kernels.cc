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

#include "torch_tpu/ops/leaky_relu/leaky_relu_aten_kernels.h"

#include <utility>

#include "absl/status/statusor.h"
#include "ATen/core/ATen_fwd.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/leaky_relu/leaky_relu.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> LeakyReluShlo(mlir::MlirOp self_op,
                                           mlir::MlirOp negative_slope_op) {
  TT_ASSIGN_OR_RETURN((auto [self, negative_slope]),
                      ApplyBroadcastIfNeeded(self_op, negative_slope_op));
  return BuildLeakyReluShlo(self, negative_slope);
}

absl::StatusOr<mlir::MlirOp> LeakyReluBackwardShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_op,
    mlir::MlirOp negative_slope_op) {
  TT_ASSIGN_OR_RETURN(
      (auto [grad_output, self, negative_slope]),
      ApplyBroadcastIfNeeded(grad_output_op, self_op, negative_slope_op));
  return BuildLeakyReluBackwardShlo(grad_output, self, negative_slope);
}

}  // namespace

at::Tensor AtenLeakyRelu(const at::Tensor& self,
                         const at::Scalar& negative_slope) {
  TT_KERNEL(
      OpName::kLeakyReluOut, param_keys,
      (self, IgnoreInCacheKey(negative_slope)), {
        TT_ASSIGN_OR_THROW(at::Tensor negative_slope_tensor,
                           MakeTensor(negative_slope, self.scalar_type()));

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [self_op, negative_slope_op] = inputs;
          return LeakyReluShlo(self_op, negative_slope_op);
        };

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            DispatchOp<2>(OpName::kLeakyReluOut, std::move(op_builder),
                          {self, negative_slope_tensor},
                          {.out_dtype = out_dtype,
                           .out_dims = CopyIntVector(self.sizes()),
                           .op_param_cache_keys = std::move(param_keys)}));
        return MakeTensor(std::move(result_buffer));
      });
}

at::Tensor& AtenLeakyRelu_(at::Tensor& self, const at::Scalar& negative_slope) {
  TT_KERNEL(
      OpName::kLeakyReluOut, param_keys,
      (self, IgnoreInCacheKey(negative_slope)), {
        TT_ASSIGN_OR_THROW(at::Tensor negative_slope_tensor,
                           MakeTensor(negative_slope, self.scalar_type()));

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [self_op, negative_slope_op] = inputs;
          return LeakyReluShlo(self_op, negative_slope_op);
        };

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            DispatchOp<2>(OpName::kLeakyReluOut, std::move(op_builder),
                          {self, negative_slope_tensor},
                          {.out_dtype = out_dtype,
                           .out_dims = CopyIntVector(self.sizes()),
                           .op_param_cache_keys = std::move(param_keys)}));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), self));
        return self;
      });
}

at::Tensor& AtenLeakyReluOut(const at::Tensor& self,
                             const at::Scalar& negative_slope,
                             at::Tensor& out) {
  TT_KERNEL(
      OpName::kLeakyReluOut, param_keys,
      (self, IgnoreInCacheKey(negative_slope), out), {
        TT_ASSIGN_OR_THROW(at::Tensor negative_slope_tensor,
                           MakeTensor(negative_slope, self.scalar_type()));

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [self_op, negative_slope_op] = inputs;
          return LeakyReluShlo(self_op, negative_slope_op);
        };

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            DispatchOp<2>(OpName::kLeakyReluOut, std::move(op_builder),
                          {self, negative_slope_tensor},
                          {.out_dtype = out_dtype,
                           .out_dims = CopyIntVector(out.sizes()),
                           .op_param_cache_keys = std::move(param_keys)}));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), out));
        return out;
      });
}

at::Tensor& AtenLeakyReluBackwardOut(const at::Tensor& grad_output,
                                     const at::Tensor& self,
                                     const at::Scalar& negative_slope,
                                     bool self_is_result,
                                     at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kLeakyReluOut, param_keys,
      (grad_output, self, IgnoreInCacheKey(negative_slope),
       IgnoreInCacheKey(self_is_result), grad_input),
      {
        TT_ASSIGN_OR_THROW(at::Tensor negative_slope_tensor,
                           MakeTensor(negative_slope, self.scalar_type()));

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output_op, self_op, negative_slope_op] = inputs;
          return LeakyReluBackwardShlo(grad_output_op, self_op,
                                       negative_slope_op);
        };

        TT_ASSIGN_OR_THROW(
            mlir::ElementType out_dtype,
            ConvertTo<mlir::ElementType>(grad_input.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            DispatchOp<3>(OpName::kLeakyReluOut, std::move(op_builder),
                          {grad_output, self, negative_slope_tensor},
                          {.out_dtype = out_dtype,
                           .out_dims = CopyIntVector(grad_input.sizes()),
                           .op_param_cache_keys = std::move(param_keys)}));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), grad_input));
        return grad_input;
      });
}

}  // namespace torch_tpu
