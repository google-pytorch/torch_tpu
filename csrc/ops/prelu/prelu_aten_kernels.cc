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

#include "csrc/ops/prelu/prelu_aten_kernels.h"

#include <array>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/ArrayRef.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

absl::Status ValidateIsFloatingPoint(const at::Tensor& tensor,
                                     const std::string_view name) {
  TT_RET_CHECK(IsFloatingPoint(tensor), error::kInvalidArgument)
      << "expected the " << name << " dtype to be floating point, got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildPreluKernelShlo(mlir::MlirOp self_op,
                                                  mlir::MlirOp weight_op) {
  TT_ASSIGN_OR_RETURN(const auto broadcasted,
                      ApplyBroadcastIfNeeded(self_op, weight_op));
  auto self_bcast = broadcasted[0];
  auto weight_bcast = broadcasted[1];

  auto zero = MakeConstantLike(self_bcast, 0.0);
  auto pred = mlir::stablehlo::Compare(
      self_bcast, zero, mlir::stablehlo::ComparisonDirection::GT);
  auto pos_branch = self_bcast;
  auto neg_branch = mlir::stablehlo::Mul(self_bcast, weight_bcast);

  return mlir::stablehlo::Select(pred, pos_branch, neg_branch);
}

absl::StatusOr<MlirOpResults<2>> BuildPreluKernelBackwardShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_op, mlir::MlirOp weight_op,
    c10::IntArrayRef self_shape, c10::IntArrayRef weight_shape) {
  TT_ASSIGN_OR_RETURN(const auto broadcasted,
                      ApplyBroadcastIfNeeded(self_op, weight_op));
  auto self_bcast = broadcasted[0];
  auto weight_bcast = broadcasted[1];

  auto zero = MakeConstantLike(self_bcast, 0.0);

  // grad_self = where(self > 0, grad_output, grad_output * weight)
  auto cond_gt_zero = mlir::stablehlo::Compare(
      self_bcast, zero, mlir::stablehlo::ComparisonDirection::GT);
  auto neg_grad_self = mlir::stablehlo::Mul(grad_output_op, weight_bcast);
  auto grad_self =
      mlir::stablehlo::Select(cond_gt_zero, grad_output_op, neg_grad_self);

  // grad_weight_elementwise = where(self > 0, zero, grad_output * self)
  auto neg_grad_weight = mlir::stablehlo::Mul(grad_output_op, self_bcast);
  auto grad_weight_elem =
      mlir::stablehlo::Select(cond_gt_zero, zero, neg_grad_weight);

  // Reduce grad_weight_elem to weight_shape
  int64_t self_rank = self_shape.size();
  int64_t weight_rank = weight_shape.size();

  llvm::SmallVector<int64_t> reduce_dims;
  int64_t rank_diff = self_rank - weight_rank;
  for (int64_t i = 0; i < rank_diff; ++i) {
    reduce_dims.push_back(i);
  }
  for (int64_t i = 0; i < weight_rank; ++i) {
    if (weight_shape[i] == 1 && self_shape[rank_diff + i] != 1) {
      reduce_dims.push_back(rank_diff + i);
    }
  }

  mlir::MlirOp grad_weight;
  if (!reduce_dims.empty()) {
    TT_ASSIGN_OR_RETURN(grad_weight, BuildSumShlo(grad_weight_elem, reduce_dims,
                                                  ReductionMode::kKeepDims));
    if (GetTensorTypeOrDie(grad_weight).getShape() != weight_shape) {
      const auto target_type = mlir::RankedTensorType::get(
          weight_shape, GetTensorTypeOrDie(grad_weight).getElementType());
      grad_weight = mlir::stablehlo::Reshape(target_type, grad_weight);
    }
  } else {
    if (GetTensorTypeOrDie(grad_weight_elem).getShape() != weight_shape) {
      const auto target_type = mlir::RankedTensorType::get(
          weight_shape, GetTensorTypeOrDie(grad_weight_elem).getElementType());
      grad_weight = mlir::stablehlo::Reshape(target_type, grad_weight_elem);
    } else {
      grad_weight = grad_weight_elem;
    }
  }

  return MlirOpResults<2>{grad_self, grad_weight};
}

}  // namespace

at::Tensor AtenPreluKernel(const at::Tensor& self, const at::Tensor& weight) {
  TT_KERNEL(OpName::kPreluKernel, _, (self, weight), {
    TT_THROW_IF_ERROR(ValidateIsFloatingPoint(self, "self"));
    TT_THROW_IF_ERROR(ValidateIsFloatingPoint(weight, "weight"));

    TT_CHECK_THROW(self.scalar_type() == weight.scalar_type(),
                   error::kInvalidArgument)
        << "expected self and weight to have the same dtype, got "
        << ToString(self.scalar_type()) << " and "
        << ToString(weight.scalar_type());

    TT_ASSIGN_OR_THROW(const auto expected_size,
                       InferSize(self.sizes(), weight.sizes()));
    TT_CHECK_THROW(expected_size == self.sizes(), error::kInvalidArgument)
        << "expected weight tensor shape to be broadcastable to self shape "
        << ToString(self.sizes()) << ", got " << ToString(weight.sizes());

    TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));

    const auto self_shape = self.sizes();

    auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto [self_op, weight_op] = inputs;
      return BuildPreluKernelShlo(self_op, weight_op);
    };

    TT_ASSIGN_OR_THROW(
        auto output_buf,
        (DispatchOp<2, 1>(std::move(op_builder), {self, weight},
                          {.out_dtype = out_dtype,
                           .out_dims = self_shape,
                           .op_param_cache_keys = OpParamCacheKeys::Empty()})));

    return MakeTensor(std::move(output_buf));
  });
}

std::tuple<at::Tensor, at::Tensor> AtenPreluKernelBackward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& weight) {
  TT_KERNEL(OpName::kPreluKernelBackward, _, (grad_output, self, weight), {
    TT_THROW_IF_ERROR(ValidateIsFloatingPoint(grad_output, "grad_output"));
    TT_THROW_IF_ERROR(ValidateIsFloatingPoint(self, "self"));
    TT_THROW_IF_ERROR(ValidateIsFloatingPoint(weight, "weight"));

    TT_CHECK_THROW(grad_output.scalar_type() == self.scalar_type() &&
                       self.scalar_type() == weight.scalar_type(),
                   error::kInvalidArgument)
        << "expected grad_output, self, and weight to have the same "
        << "dtype, got " << ToString(grad_output.scalar_type()) << ", "
        << ToString(self.scalar_type()) << ", and "
        << ToString(weight.scalar_type());

    TT_CHECK_THROW(grad_output.sizes() == self.sizes(), error::kInvalidArgument)
        << "expected grad_output shape to match self shape "
        << ToString(self.sizes()) << ", got " << ToString(grad_output.sizes());

    TT_ASSIGN_OR_THROW(const auto expected_size,
                       InferSize(self.sizes(), weight.sizes()));
    TT_CHECK_THROW(expected_size == self.sizes(), error::kInvalidArgument)
        << "expected weight tensor shape to be broadcastable to self shape "
        << ToString(self.sizes()) << ", got " << ToString(weight.sizes());

    TT_ASSIGN_OR_THROW(const mlir::ElementType self_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(const mlir::ElementType weight_dtype,
                       ConvertTo<mlir::ElementType>(weight.scalar_type()));

    const auto self_shape = CopyIntVector(self.sizes());
    const auto weight_shape = CopyIntVector(weight.sizes());

    auto op_builder = [self_shape,
                       weight_shape](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<MlirOpResults<2>> {
      auto [grad_output_op, self_op, weight_op] = inputs;
      return BuildPreluKernelBackwardShlo(grad_output_op, self_op, weight_op,
                                          self_shape, weight_shape);
    };

    const std::array<mlir::ElementType, 2> out_dtypes = {self_dtype,
                                                         weight_dtype};
    const std::array<absl::Span<const int64_t>, 2> out_dims_list = {
        self_shape, weight_shape};

    TT_ASSIGN_OR_THROW(
        auto output_bufs,
        (DispatchOp<3, 2>(std::move(op_builder), {grad_output, self, weight},
                          {.out_dtypes = out_dtypes,
                           .out_dims_list = out_dims_list,
                           .op_param_cache_keys = OpParamCacheKeys::Empty()})));

    return {MakeTensor(std::move(output_bufs[0])),
            MakeTensor(std::move(output_bufs[1]))};
  });
}

}  // namespace torch_tpu
