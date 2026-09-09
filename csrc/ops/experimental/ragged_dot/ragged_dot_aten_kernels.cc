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

#include "csrc/ops/experimental/ragged_dot/ragged_dot_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/ops/result_type.h"
#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/autograd/custom_function.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

static absl::StatusOr<mlir::MlirOp> BuildRaggedDotShlo(
    mlir::MlirOp lhs, mlir::MlirOp rhs, mlir::MlirOp group_sizes,
    mlir::ElementType output_element_type,
    mlir::stablehlo::Precision precision) {
  const mlir::RankedTensorType lhs_type = GetTensorTypeOrDie(lhs);
  const mlir::RankedTensorType rhs_type = GetTensorTypeOrDie(rhs);
  // Assuming mk, gkn, g -> mn
  auto dimension_numbers = mlir::chlo::RaggedDotDimensionNumbersAttr::get(
      &lhs.getContext(), /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{},
      /*lhsContractingDimensions=*/{1},
      /*rhsContractingDimensions=*/{1},
      /*lhsRaggedDimensions=*/{0},
      /*rhsGroupDimensions=*/{0});
  mlir::Type out_type = mlir::makeTensorType(
      lhs.getContext(), {lhs_type.getShape()[0], rhs_type.getShape()[2]},
      output_element_type);

  mlir::chlo::Precision chlo_precision =
      static_cast<mlir::chlo::Precision>(precision);
  auto precision_attr =
      mlir::chlo::PrecisionAttr::get(&lhs.getContext(), chlo_precision);
  auto precision_config =
      mlir::ArrayAttr::get(&lhs.getContext(), {precision_attr, precision_attr});

  return mlir::chlo::RaggedDot(out_type, lhs, rhs, group_sizes,
                               dimension_numbers, precision_config);
}

static absl::StatusOr<DeviceBufferRef> RaggedDotCommon(
    const at::Tensor& lhs, const at::Tensor& rhs, const at::Tensor& group_sizes,
    OpParamCacheKeys& param_keys) {
  // ragged_dot(mk, gkn, g) -> mn
  TT_RET_CHECK(lhs.dim() == 2, error::kInvalidArgument)
      << "expected lhs to be 2D, got dim: " << lhs.dim();
  TT_RET_CHECK(rhs.dim() == 3, error::kInvalidArgument)
      << "expected rhs to be 3D, got dim: " << rhs.dim();
  TT_RET_CHECK(group_sizes.dim() == 1, error::kInvalidArgument)
      << "expected group_sizes to be 1D, got dim: " << group_sizes.dim();
  TT_RET_CHECK(lhs.size(1) == rhs.size(1), error::kInvalidArgument)
      << "expected contracting dimension to be the same, got " << lhs.size(1)
      << " vs " << rhs.size(1);
  TT_RET_CHECK(rhs.size(0) == group_sizes.size(0), error::kInvalidArgument)
      << "expected lhs and group_sizes to have the same number of groups, got "
      << rhs.size(0) << " vs " << group_sizes.size(0);

  at::ScalarType out_scalar_type = at::result_type(lhs, rhs);
  TT_ASSIGN_OR_RETURN(auto out_dtype,
                      ConvertTo<mlir::ElementType>(out_scalar_type));
  const auto current_precision = GetAndAddPrecisionTo(param_keys);
  auto op_builder = [out_dtype,
                     current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
    auto& [lhs, rhs, group_sizes] = inputs;
    return BuildRaggedDotShlo(lhs, rhs, group_sizes, out_dtype,
                              current_precision);
  };
  return DispatchOp<3>(std::move(op_builder), {lhs, rhs, group_sizes},
                       {.out_dtype = out_dtype,
                        .out_dims = {lhs.size(0), rhs.size(2)},
                        .op_param_cache_keys = std::move(param_keys)});
}

static absl::StatusOr<mlir::MlirOp> BuildRaggedDotWeightGradShlo(
    mlir::MlirOp lhs, mlir::MlirOp grad_output, mlir::MlirOp group_sizes,
    mlir::ElementType output_element_type,
    mlir::stablehlo::Precision precision) {
  mlir::MlirOp lhs_t = mlir::stablehlo::Transpose(lhs, {1, 0});
  const mlir::RankedTensorType lhs_t_type = GetTensorTypeOrDie(lhs_t);
  const mlir::RankedTensorType grad_output_type =
      GetTensorTypeOrDie(grad_output);
  const mlir::RankedTensorType group_sizes_type =
      GetTensorTypeOrDie(group_sizes);

  auto dimension_numbers = mlir::chlo::RaggedDotDimensionNumbersAttr::get(
      &lhs.getContext(),
      /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{},
      /*lhsContractingDimensions=*/{1},
      /*rhsContractingDimensions=*/{0},
      /*lhsRaggedDimensions=*/{1},
      /*rhsGroupDimensions=*/{});
  mlir::Type out_type = mlir::makeTensorType(
      lhs.getContext(),
      {group_sizes_type.getShape()[0], lhs_t_type.getShape()[0],
       grad_output_type.getShape()[1]},
      output_element_type);

  mlir::chlo::Precision chlo_precision =
      static_cast<mlir::chlo::Precision>(precision);
  auto precision_attr =
      mlir::chlo::PrecisionAttr::get(&lhs.getContext(), chlo_precision);
  auto precision_config =
      mlir::ArrayAttr::get(&lhs.getContext(), {precision_attr, precision_attr});

  return mlir::chlo::RaggedDot(out_type, lhs_t, grad_output, group_sizes,
                               dimension_numbers, precision_config);
}

static absl::StatusOr<DeviceBufferRef> RaggedDotWeightGradCommon(
    const at::Tensor& lhs, const at::Tensor& grad_output,
    const at::Tensor& group_sizes, OpParamCacheKeys& param_keys) {
  TT_RET_CHECK(lhs.dim() == 2, error::kInvalidArgument)
      << "expected lhs to be 2D, got dim: " << lhs.dim();
  TT_RET_CHECK(grad_output.dim() == 2, error::kInvalidArgument)
      << "expected grad_output to be 2D, got dim: " << grad_output.dim();
  TT_RET_CHECK(group_sizes.dim() == 1, error::kInvalidArgument)
      << "expected group_sizes to be 1D, got dim: " << group_sizes.dim();
  TT_RET_CHECK(lhs.size(0) == grad_output.size(0), error::kInvalidArgument)
      << "expected lhs and grad_output to have the same batch dimension, got "
      << lhs.size(0) << " vs " << grad_output.size(0);

  at::ScalarType out_scalar_type = at::result_type(lhs, grad_output);
  TT_ASSIGN_OR_RETURN(auto out_dtype,
                      ConvertTo<mlir::ElementType>(out_scalar_type));
  const auto current_precision = GetAndAddPrecisionTo(param_keys);
  auto op_builder = [out_dtype,
                     current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
    auto& [lhs, grad_output, group_sizes] = inputs;
    return BuildRaggedDotWeightGradShlo(lhs, grad_output, group_sizes,
                                        out_dtype, current_precision);
  };
  const Dimensions out_dims = {group_sizes.size(0), lhs.size(1),
                               grad_output.size(1)};
  return DispatchOp<3>(std::move(op_builder), {lhs, grad_output, group_sizes},
                       {.out_dtype = out_dtype,
                        .out_dims = out_dims,
                        .op_param_cache_keys = std::move(param_keys)});
}
}  // namespace

at::Tensor AtenRaggedDot(const at::Tensor& lhs, const at::Tensor& rhs,
                         const at::Tensor& group_sizes) {
  TT_KERNEL(OpName::kRaggedDot, param_keys, (lhs, rhs, group_sizes), {
    TT_ASSIGN_OR_THROW(auto result,
                       RaggedDotCommon(lhs, rhs, group_sizes, param_keys));
    return MakeTensor(std::move(result));
  });
}

at::Tensor& AtenRaggedDotOut(const at::Tensor& lhs, const at::Tensor& rhs,
                             const at::Tensor& group_sizes, at::Tensor& out) {
  TT_KERNEL(OpName::kRaggedDot, param_keys, (lhs, rhs, group_sizes, out), {
    TT_ASSIGN_OR_THROW(auto result,
                       RaggedDotCommon(lhs, rhs, group_sizes, param_keys));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
    return out;
  });
}

at::Tensor AtenRaggedDotWeightGrad(const at::Tensor& lhs,
                                   const at::Tensor& grad_output,
                                   const at::Tensor& group_sizes) {
  TT_KERNEL(OpName::kRaggedDotWeightGrad, param_keys,
            (lhs, grad_output, group_sizes), {
              TT_ASSIGN_OR_THROW(
                  auto result, RaggedDotWeightGradCommon(
                                   lhs, grad_output, group_sizes, param_keys));
              return MakeTensor(std::move(result));
            });
}

at::Tensor AtenRaggedDotAutograd::forward(torch::autograd::AutogradContext* ctx,
                                          const at::Tensor& lhs,
                                          const at::Tensor& rhs,
                                          const at::Tensor& group_sizes) {
  ctx->save_for_backward({lhs, rhs, group_sizes});

  // Cache the operator handle to avoid string schema lookup on every call.
  static const absl::NoDestructor op(
      at::Dispatcher::singleton()
          .findSchemaOrThrow("tpu::ragged_dot", "")
          .typed<at::Tensor(const at::Tensor&, const at::Tensor&,
                            const at::Tensor&)>());

  at::AutoDispatchBelowADInplaceOrView guard;
  return op->call(lhs, rhs, group_sizes);
}

torch::autograd::variable_list AtenRaggedDotAutograd::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
  const auto saved = ctx->get_saved_variables();
  const at::Tensor& lhs = saved[0];
  const at::Tensor& rhs = saved[1];
  const at::Tensor& group_sizes = saved[2];
  const at::Tensor& grad_out = grad_outputs[0];

  at::Tensor grad_lhs;  // UNINITIALIZED_TENSOR_OK
  if (ctx->needs_input_grad(0)) {
    // rhs: [g, k, n] -> rhs_t: [g, n, k]
    // grad_out: [m, n], rhs_t: [g, n, k], group_sizes: [g] -> grad_lhs: [m, k]
    at::Tensor rhs_t = rhs.transpose(1, 2).contiguous();

    // Cache the operator handle to avoid string schema lookup on every call.
    static const absl::NoDestructor op(
        at::Dispatcher::singleton()
            .findSchemaOrThrow("tpu::ragged_dot", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&,
                              const at::Tensor&)>());

    at::AutoDispatchBelowADInplaceOrView guard;
    grad_lhs = op->call(grad_out, rhs_t, group_sizes);
  }

  at::Tensor grad_rhs;  // UNINITIALIZED_TENSOR_OK
  if (ctx->needs_input_grad(1)) {
    // lhs: [m, k], grad_out: [m, n], group_sizes: [g] -> grad_rhs: [g, k, n]
    // Cache the operator handle to avoid string schema lookup on every call.
    static const absl::NoDestructor op(
        at::Dispatcher::singleton()
            .findSchemaOrThrow("tpu::ragged_dot_weight_grad", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&,
                              const at::Tensor&)>());

    at::AutoDispatchBelowADInplaceOrView guard;
    grad_rhs = op->call(lhs, grad_out, group_sizes);
  }

  return {grad_lhs, grad_rhs, at::Tensor()};
}

}  // namespace torch_tpu
