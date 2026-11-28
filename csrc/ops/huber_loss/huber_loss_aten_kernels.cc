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

#include "csrc/ops/huber_loss/huber_loss_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Reduction.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/result_type.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/sum.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildHuberLossShlo(mlir::MlirOp self_op,
                                                mlir::MlirOp target_op,
                                                int64_t reduction,
                                                double delta) {
  TT_ASSIGN_OR_RETURN((auto [self_b, target_b]),
                      ApplyBroadcastIfNeeded(self_op, target_op));
  const mlir::RankedTensorType tensor_type = GetTensorTypeOrDie(self_b);
  const mlir::Type elem_type = tensor_type.getElementType();

  mlir::MlirOp diff = mlir::stablehlo::Subtract(self_b, target_b);
  mlir::MlirOp abs_diff = mlir::stablehlo::Abs(diff);

  mlir::MlirOp half = MakeConstantLike(abs_diff, 0.5);
  mlir::MlirOp delta_op = MakeConstantLike(abs_diff, delta);
  mlir::MlirOp half_delta = mlir::stablehlo::Mul(half, delta_op);

  // l_n = 0.5 * |x_n - y_n|^2                if |x_n - y_n| < delta
  //       delta * (|x_n - y_n| - 0.5 * delta) otherwise
  mlir::MlirOp sq_diff = mlir::stablehlo::Mul(abs_diff, abs_diff);
  mlir::MlirOp quad_loss = mlir::stablehlo::Mul(half, sq_diff);
  mlir::MlirOp shifted_diff = mlir::stablehlo::Subtract(abs_diff, half_delta);
  mlir::MlirOp linear_loss = mlir::stablehlo::Mul(delta_op, shifted_diff);

  mlir::MlirOp is_small = mlir::stablehlo::Compare(
      abs_diff, delta_op, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp loss = mlir::stablehlo::Select(is_small, quad_loss, linear_loss);

  if (reduction == at::Reduction::None) {
    return loss;
  }

  TT_ASSIGN_OR_RETURN(mlir::MlirOp reduced_loss,
                      BuildSumShlo(loss, GetAllDimensions(loss)));

  if (reduction == at::Reduction::Sum) {
    return reduced_loss;
  }

  mlir::MlirOp numel = GetNumElements(loss, elem_type);
  TT_ASSIGN_OR_RETURN((auto [reduced_b, numel_b]),
                      ApplyBroadcastIfNeeded(reduced_loss, numel));
  return mlir::stablehlo::Div(reduced_b, numel_b);
}

struct HuberLossPrepared {
  mlir::ElementType out_dtype;
  Dimensions out_dims;
  mlir::ElementType computation_dtype;
  NAryMlirOpBuilder<2, 1> op_builder;
};

absl::StatusOr<HuberLossPrepared> PrepareHuberLoss(
    const at::Tensor& self, const at::Tensor& target, int64_t reduction,
    double delta, at::ScalarType out_scalar_type) {
  TT_RET_CHECK(delta > 0.0, error::kInvalidArgument)
      << "expected delta to be positive, got " << delta;
  TT_RET_CHECK(reduction == at::Reduction::None ||
                   reduction == at::Reduction::Mean ||
                   reduction == at::Reduction::Sum,
               error::kInvalidArgument)
      << "unrecognized reduction mode " << reduction;

  TT_ASSIGN_OR_RETURN(Dimensions bcast_shape, InferSize(self, target));

  // Match PyTorch's TensorIterator::borrowing_binary_op + AT_DISPATCH order:
  // output castability is checked before verifying the promoted dtype is
  // floating-point.
  const at::ScalarType result_dtype = at::result_type(self, target);
  TT_RET_CHECK(c10::canCast(result_dtype, out_scalar_type),
               error::kInvalidArgument)
      << "result type " << ToString(result_dtype)
      << " can't be cast to the desired output type "
      << ToString(out_scalar_type);

  TT_RET_CHECK(c10::isFloatingType(result_dtype),
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(result_dtype);
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(result_dtype));
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(out_scalar_type));

  Dimensions out_dims = (reduction == at::Reduction::None)
                            ? std::move(bcast_shape)
                            : Dimensions();

  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(out_scalar_type));
  const at::ScalarType comp_type = ToAccumulateType(result_dtype);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType computation_dtype,
                      ConvertTo<mlir::ElementType>(comp_type));

  auto op_builder = [reduction, delta](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, target_op] = inputs;
    return BuildHuberLossShlo(self_op, target_op, reduction, delta);
  };

  return HuberLossPrepared{
      .out_dtype = out_dtype,
      .out_dims = std::move(out_dims),
      .computation_dtype = computation_dtype,
      .op_builder = std::move(op_builder),
  };
}

}  // namespace

at::Tensor AtenHuberLoss(const at::Tensor& self, const at::Tensor& target,
                         int64_t reduction, double delta) {
  TT_KERNEL(
      OpName::kHuberLoss, param_keys, (self, target, reduction, delta), {
        // Upstream `aten::huber_loss` allocates `loss = at::empty_like(self)`
        // before running `TensorIterator::borrowing_binary_op`, so the output
        // dtype is `self.scalar_type()` rather than the promoted `result_dtype`.
        TT_ASSIGN_OR_THROW(HuberLossPrepared prepared,
                           PrepareHuberLoss(self, target, reduction, delta,
                                            self.scalar_type()));

        DispatchOpOptions<1> options = {
            .out_dtype = prepared.out_dtype,
            .out_dims = prepared.out_dims,
            .computation_dtype = prepared.computation_dtype,
            .op_param_cache_keys = std::move(param_keys),
        };
        TT_ASSIGN_OR_THROW(DeviceBufferRef result_buf,
                           DispatchOp<2>(std::move(prepared.op_builder),
                                         {self, target}, std::move(options)));
        return MakeTensor(std::move(result_buf));
      });
}

at::Tensor& AtenHuberLossOut(const at::Tensor& self, const at::Tensor& target,
                             int64_t reduction, double delta, at::Tensor& out) {
  TT_KERNEL(
      OpName::kHuberLossOut, param_keys,
      (self, target, reduction, delta, out), {
        TT_ASSIGN_OR_THROW(
            HuberLossPrepared prepared,
            PrepareHuberLoss(self, target, reduction, delta, out.scalar_type()));

        DispatchOpOptions<1> options = {
            .out_dtype = prepared.out_dtype,
            .out_dims = prepared.out_dims,
            .computation_dtype = prepared.computation_dtype,
            .op_param_cache_keys = std::move(param_keys),
        };
        TT_THROW_IF_ERROR(DispatchOpOut<2>(std::move(prepared.op_builder),
                                           {self, target}, out,
                                           std::move(options)));
        return out;
      });
}

}  // namespace torch_tpu
