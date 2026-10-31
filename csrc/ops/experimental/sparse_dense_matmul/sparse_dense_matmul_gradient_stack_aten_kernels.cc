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

#include "csrc/ops/experimental/sparse_dense_matmul/sparse_dense_matmul_gradient_stack_aten_kernels.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "csrc/common/const_tensor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/distributed/process_group_tpu.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

auto SparseDenseMatmulGradientStackBuilder(int64_t stacked_batch_size,
                                           int64_t stacked_feature_dim) {
  return [stacked_batch_size, stacked_feature_dim](
             absl::Span<mlir::MlirOp> inputs,
             mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    mlir::OpBuilder& op_builder = builder.getOpBuilder();

    const size_t num_inputs = inputs.size();
    std::vector<mlir::Value> operands;
    operands.reserve(num_inputs);
    std::vector<mlir::Attribute> operand_layout_attrs;
    operand_layout_attrs.reserve(num_inputs);

    for (const auto& input : inputs) {
      operands.push_back(input.getValue());
      operand_layout_attrs.push_back(op_builder.getIndexTensorAttr({0, 1}));
    }

    const std::string call_target = "SparseGradientsStackInterleaved";
    mlir::NamedAttribute call_target_attr = op_builder.getNamedAttr(
        "call_target_name", op_builder.getStringAttr(call_target));
    mlir::NamedAttribute has_side_effect_attr = op_builder.getNamedAttr(
        "has_side_effect", op_builder.getBoolAttr(false));
    auto api_version_attr = op_builder.getNamedAttr(
        "api_version",
        mlir::stablehlo::CustomCallApiVersionAttr::get(
            &builder.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL));

    auto operand_layouts_attr = op_builder.getNamedAttr(
        "operand_layouts", op_builder.getArrayAttr(operand_layout_attrs));
    auto result_layouts_attr = op_builder.getNamedAttr(
        "result_layouts",
        op_builder.getArrayAttr({op_builder.getIndexTensorAttr({1, 0})}));

    auto out_type = mlir::RankedTensorType::get(
        {stacked_batch_size, stacked_feature_dim}, op_builder.getF32Type());

    auto op = mlir::stablehlo::CustomCallOp::create(
        op_builder, builder.getLoc(),
        /*resultTypes=*/{out_type},
        /*operands=*/operands,
        {call_target_attr, has_side_effect_attr, api_version_attr,
         operand_layouts_attr, result_layouts_attr});

    return mlir::MlirOp(builder, op.getResult(0));
  };
}

}  // namespace

absl::Status ValidateSparseDenseMatmulGradientStackInputs(
    at::TensorList unstacked_gradients, const c10::SymInt& stacked_batch_size,
    const c10::SymInt& stacked_feature_dim) {
  // For symbolic sizes, the comparisons below add relational guards but don't
  // specialize.
  TT_RET_CHECK(stacked_batch_size.sym_gt(0).guard_bool(__FILE__, __LINE__),
               error::kInvalidArgument)
      << "expected stacked_batch_size to be positive, got "
      << stacked_batch_size;

  TT_RET_CHECK(stacked_feature_dim.sym_gt(0).guard_bool(__FILE__, __LINE__),
               error::kInvalidArgument)
      << "expected stacked_feature_dim to be positive, got "
      << stacked_feature_dim;

  c10::SymInt total_batch_size = 0;

  for (size_t i = 0; i < unstacked_gradients.size(); ++i) {
    const ConstTensor grad(unstacked_gradients[i]);
    TT_ASSIGN_OR_RETURN(const int64_t grad_dim, grad.dim());
    TT_ASSIGN_OR_RETURN(const c10::SymIntArrayRef grad_sizes, grad.sym_sizes());
    TT_RET_CHECK(grad_dim == 2, error::kInvalidArgument)
        << "expected unstacked_gradients[" << i << "] to be a 2D tensor, got a "
        << grad_dim << "D tensor of shape " << ToString(grad_sizes);

    TT_RET_CHECK(grad.scalar_type() == at::kFloat, error::kInvalidArgument)
        << "expected unstacked_gradients[" << i
        << "] to have float32 dtype, got " << ToString(grad.scalar_type());

    TT_RET_CHECK(grad_sizes[1]
                     .sym_le(stacked_feature_dim)
                     .guard_bool(__FILE__, __LINE__),
                 error::kInvalidArgument)
        << "expected unstacked_gradients[" << i << "] dim 1 (" << grad_sizes[1]
        << ") to be at most stacked_feature_dim (" << stacked_feature_dim
        << ")";

    total_batch_size += grad_sizes[0];
  }

  TT_RET_CHECK(total_batch_size.sym_eq(stacked_batch_size)
                   .guard_bool(__FILE__, __LINE__),
               error::kInvalidArgument)
      << "expected sum of unstacked gradient batch sizes (" << total_batch_size
      << ") to match stacked_batch_size (" << stacked_batch_size << ")";

  return absl::OkStatus();
}

at::Tensor AtenSparseDenseMatmulGradientStack(
    at::TensorList unstacked_gradients, int64_t stacked_batch_size,
    int64_t stacked_feature_dim) {
  TT_KERNEL(
      OpName::kSparseDenseMatmulGradientStack, param_keys,
      (unstacked_gradients, stacked_batch_size, stacked_feature_dim), {
        TT_THROW_IF_ERROR(ValidateSparseDenseMatmulGradientStackInputs(
            unstacked_gradients, stacked_batch_size, stacked_feature_dim));

        std::vector<at::Tensor> inputs_vec(unstacked_gradients.begin(),
                                           unstacked_gradients.end());

        Dimensions out_dims = {stacked_batch_size, stacked_feature_dim};

        auto builder_fn = SparseDenseMatmulGradientStackBuilder(
            stacked_batch_size, stacked_feature_dim);

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(at::kFloat));

        TT_ASSIGN_OR_THROW(
            std::vector<int64_t> device_assignment,  // INT_VEC_OK
            GetWorldDeviceAssignment());

        DispatchOpOptions<1> options = {
            .out_dtype = out_dtype,
            .out_dims = out_dims,
            .op_param_cache_keys = std::move(param_keys),
            .device_assignment = std::move(device_assignment),
        };

        TT_ASSIGN_OR_THROW(
            DeviceBufferRef result,
            (DispatchOp<kDynamicSize, 1>(std::move(builder_fn), inputs_vec,
                                         std::move(options))));

        return torch_tpu::MakeTensor(result);
      });
}

}  // namespace torch_tpu
