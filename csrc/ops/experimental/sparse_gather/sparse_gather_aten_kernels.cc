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

#include "csrc/ops/experimental/sparse_gather/sparse_gather_aten_kernels.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/ops/zeros.h"
#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/pjrt/pjrt_utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/autograd/custom_function.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

auto SparseGatherBuilder(int64_t max_non_zeroes_per_row) {
  return [max_non_zeroes_per_row](
             torch_tpu::FixedSizeSpan<mlir::MlirOp, 3> inputs)
             -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirOp row_pointers = inputs[0];
    mlir::MlirOp indices = inputs[1];
    mlir::MlirOp operand = inputs[2];
    mlir::MlirBuilder& builder = row_pointers.getBuilder();
    mlir::OpBuilder& op_builder = builder.getOpBuilder();

    // 1. Pack CSR components (row_pointers, indices) into a tuple.
    auto tuple_op = mlir::stablehlo::TupleOp::create(
        op_builder, builder.getLoc(),
        mlir::ValueRange{row_pointers.getValue(), indices.getValue()});

    // 2. Build Backend Config JSON for SparseCore.
    std::string backend_config = absl::StrFormat(
        R"({"device_type":"DEVICE_TYPE_SPARSECORE","csr_config":{"max_non_zeroes_per_row":%d,"max_pad_per_row":8},"sparse_map_row_config":{"wait_threshold":0.5}})",
        max_non_zeroes_per_row);

    mlir::NamedAttribute backend_config_attr = op_builder.getNamedAttr(
        "backend_config", op_builder.getStringAttr(backend_config));
    mlir::NamedAttribute call_target_attr = op_builder.getNamedAttr(
        "call_target_name", op_builder.getStringAttr("SparseGather"));
    mlir::NamedAttribute has_side_effect_attr = op_builder.getNamedAttr(
        "has_side_effect", op_builder.getBoolAttr(false));
    auto api_version_attr = op_builder.getNamedAttr(
        "api_version",
        mlir::stablehlo::CustomCallApiVersionAttr::get(
            &builder.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL));

    auto operand_type = torch_tpu::GetTensorTypeOrDie(operand);
    auto indices_type = torch_tpu::GetTensorTypeOrDie(indices);
    int64_t num_gathered_elements = indices_type.getShape()[0];
    int64_t embedding_dim = operand_type.getShape()[1];

    auto out_type = mlir::RankedTensorType::get(
        {num_gathered_elements, embedding_dim}, operand_type.getElementType());

    auto op = mlir::stablehlo::CustomCallOp::create(
        op_builder, builder.getLoc(),
        /*resultTypes=*/{out_type},
        /*operands=*/
        mlir::ValueRange{operand.getValue(), tuple_op.getResult()},
        {call_target_attr, has_side_effect_attr, api_version_attr,
         backend_config_attr});

    return mlir::MlirOp(builder, op.getResult(0));
  };
}

auto SparseGatherBackwardBuilder() {
  return [](torch_tpu::FixedSizeSpan<mlir::MlirOp, 3> inputs)
             -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirOp grad_output = inputs[0];
    mlir::MlirOp indices = inputs[1];
    mlir::MlirOp grad_operand = inputs[2];
    mlir::MlirBuilder& builder = grad_output.getBuilder();

    auto grad_operand_type = torch_tpu::GetTensorTypeOrDie(grad_operand);
    auto indices_type = torch_tpu::GetTensorTypeOrDie(indices);

    int64_t v = grad_operand_type.getShape()[0];
    mlir::Type index_elem_type = indices_type.getElementType();
    mlir::Type data_elem_type = grad_operand_type.getElementType();

    // 1. Create validity mask for indices:
    // A valid embedding index satisfies: 0 <= idx < V.
    // Padding elements (INT_MAX = 2147483647 or negative) are invalid.
    auto zero_indices = MakeConstantLike(indices, 0);
    auto is_ge_zero = mlir::stablehlo::Compare(
        indices, zero_indices, mlir::stablehlo::ComparisonDirection::GE);

    auto v_scalar = MakeScalarConstant(builder, v, index_elem_type);
    TT_ASSIGN_OR_RETURN(auto v_bcst, BroadcastIfNeeded(v_scalar, indices));
    auto is_lt_v = mlir::stablehlo::Compare(
        indices, v_bcst, mlir::stablehlo::ComparisonDirection::LT);

    auto is_valid = mlir::stablehlo::And(is_ge_zero, is_lt_v);

    // 2. Clamp invalid indices to 0 so scatter index stays within [0, V-1].
    auto safe_indices =
        mlir::stablehlo::Select(is_valid, indices, zero_indices);

    // 3. Mask out invalid updates in grad_output to 0.0 so they contribute
    // nothing.
    TT_ASSIGN_OR_RETURN(auto is_valid_unsq, Unsqueeze(is_valid, 1));
    TT_ASSIGN_OR_RETURN(auto is_valid_bcst,
                        BroadcastIfNeeded(is_valid_unsq, grad_output));
    auto zero_grad_output = MakeConstantLike(grad_output, 0.0);
    auto safe_grad_output =
        mlir::stablehlo::Select(is_valid_bcst, grad_output, zero_grad_output);

    // 4. Configure Scatter:
    // Operand: grad_operand of shape [V, D]
    // Scatter indices: safe_indices unsqueezed to [N, 1]
    // Updates: safe_grad_output of shape [N, D]
    TT_ASSIGN_OR_RETURN(auto safe_indices_2d, Unsqueeze(safe_indices, 1));

    auto scatter_dims = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
        &builder.getContext(),
        /*update_window_dims=*/{1},
        /*inserted_window_dims=*/{0},
        /*input_batching_dims=*/{},
        /*scatter_indices_batching_dims=*/{},
        /*scatter_dims_to_operand_dims=*/{0},
        /*index_vector_dim=*/1);

    auto body = [data_elem_type](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
          data_elem_type, rb.getRegion(), rb.getOpBuilder());
    };

    auto scatter_res =
        mlir::stablehlo::Scatter({grad_operand}, safe_indices_2d,
                                 {safe_grad_output}, body, scatter_dims)[0];

    return scatter_res;
  };
}

absl::Status ValidateSparseGatherBackwardInputs(
    const at::Tensor& grad_output, const at::Tensor& indices,
    const at::Tensor& grad_operand) {
  TT_RET_CHECK(grad_output.dim() == 2, error::kInvalidArgument)
      << "expected grad_output to be a 2D tensor, got a " << grad_output.dim()
      << "D tensor of shape " << ToString(grad_output.sizes());

  TT_RET_CHECK(indices.dim() == 1, error::kInvalidArgument)
      << "expected indices to be a 1D tensor, got a " << indices.dim()
      << "D tensor of shape " << ToString(indices.sizes());

  TT_RET_CHECK(grad_operand.dim() == 2, error::kInvalidArgument)
      << "expected grad_operand to be a 2D tensor, got a " << grad_operand.dim()
      << "D tensor of shape " << ToString(grad_operand.sizes());

  TT_RET_CHECK(indices.size(0) == grad_output.size(0), error::kInvalidArgument)
      << "expected indices length (" << indices.size(0)
      << ") to match grad_output batch size, got " << grad_output.size(0);

  TT_RET_CHECK(grad_operand.size(1) == grad_output.size(1),
               error::kInvalidArgument)
      << "expected embedding dimension to match (" << grad_operand.size(1)
      << "), got " << grad_output.size(1);

  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateSparseGatherInputs(const at::Tensor& row_pointers,
                                        const at::Tensor& indices,
                                        const at::Tensor& operand,
                                        int64_t max_non_zeroes_per_row) {
  TT_RET_CHECK(row_pointers.dim() == 1, error::kInvalidArgument)
      << "expected row_pointers to be a 1D tensor, got a " << row_pointers.dim()
      << "D tensor of shape " << ToString(row_pointers.sizes());

  TT_RET_CHECK(indices.dim() == 1, error::kInvalidArgument)
      << "expected indices to be a 1D tensor, got a " << indices.dim()
      << "D tensor of shape " << ToString(indices.sizes());

  TT_RET_CHECK(operand.dim() == 2, error::kInvalidArgument)
      << "expected operand to be a 2D tensor, got a " << operand.dim()
      << "D tensor of shape " << ToString(operand.sizes());

  TT_RET_CHECK(indices.size(0) == row_pointers.size(0) * max_non_zeroes_per_row,
               error::kInvalidArgument)
      << "expected indices length to match the maximum number of "
         "non-zeroes, i.e. row_pointers length * maximum number of "
         "non-zeroes per row ("
      << row_pointers.size(0) << " * " << max_non_zeroes_per_row << " = "
      << row_pointers.size(0) * max_non_zeroes_per_row << "), got "
      << indices.size(0);

  // SparseGather relies on SparseCore hardware to execute. If not tracing for
  // meta shape inference, ensure that the active TPU device supports
  // SparseCore.
  if (!operand.is_meta()) {
    TT_RET_CHECK(TpuDeviceSupportsSparseCore(), error::kFailedPrecondition)
        << "sparse_gather requires a TPU device with SparseCore support";
  }

  return absl::OkStatus();
}

at::Tensor AtenSparseGather(const at::Tensor& row_pointers,
                            const at::Tensor& indices,
                            const at::Tensor& operand,
                            int64_t max_non_zeroes_per_row) {
  TT_KERNEL(
      torch_tpu::OpName::kSparseGather, param_keys,
      (row_pointers, indices, operand, max_non_zeroes_per_row), {
        TT_THROW_IF_ERROR(ValidateSparseGatherInputs(
            row_pointers, indices, operand, max_non_zeroes_per_row));

        std::array<torch_tpu::TensorHolder, 3> inputs = {row_pointers, indices,
                                                         operand};

        torch_tpu::Dimensions out_dims = {indices.size(0), operand.size(1)};

        auto builder_fn = SparseGatherBuilder(max_non_zeroes_per_row);

        TT_ASSIGN_OR_THROW(
            mlir::ElementType out_dtype,
            torch_tpu::ConvertTo<mlir::ElementType>(operand.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto results, (torch_tpu::DispatchOp<3, 1>(
                              builder_fn, inputs,
                              {.out_dtype = out_dtype,
                               .out_dims = out_dims,
                               .op_param_cache_keys = std::move(param_keys)})));
        return torch_tpu::MakeTensor(results);
      });
}

at::Tensor AtenSparseGatherBackward(const at::Tensor& grad_output,
                                    const at::Tensor& indices,
                                    const at::Tensor& grad_operand) {
  TT_KERNEL(torch_tpu::OpName::kSparseGatherBackward, param_keys,
            (grad_output, indices, grad_operand), {
              TT_THROW_IF_ERROR(ValidateSparseGatherBackwardInputs(
                  grad_output, indices, grad_operand));

              std::array<torch_tpu::TensorHolder, 3> inputs = {
                  grad_output, indices, grad_operand};

              torch_tpu::Dimensions out_dims = {grad_operand.size(0),
                                                grad_operand.size(1)};

              auto builder_fn = SparseGatherBackwardBuilder();

              TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                                 torch_tpu::ConvertTo<mlir::ElementType>(
                                     grad_operand.scalar_type()));

              TT_ASSIGN_OR_THROW(
                  auto results,
                  (torch_tpu::DispatchOp<3, 1>(
                      builder_fn, inputs,
                      {.out_dtype = out_dtype,
                       .out_dims = out_dims,
                       .op_param_cache_keys = std::move(param_keys)})));
              return torch_tpu::MakeTensor(results);
            });
}

// Autograd implementation for torch.ops.tpu.sparse_gather.
//
// In forward, we only save `indices` and the shape/dtype of `operand`. Saving
// the entire embedding table is avoided to significantly reduce peak memory
// consumption during training.
//
// In backward, we allocate a zero buffer of shape [V, D] and invoke
// torch.ops.tpu.sparse_gather_backward to scatter-add the incoming gradients.
at::Tensor AtenSparseGatherAutograd::forward(
    torch::autograd::AutogradContext* ctx, const at::Tensor& row_pointers,
    const at::Tensor& indices, const at::Tensor& operand,
    int64_t max_non_zeroes_per_row) {
  // Save indices and operand metadata for backward.
  ctx->save_for_backward({indices});
  ctx->saved_data["operand_sizes"] = operand.sizes().vec();  // VEC_OK
  ctx->saved_data["operand_scalar_type"] =
      static_cast<int64_t>(operand.scalar_type());

  static const absl::NoDestructor sparse_gather_op(
      at::Dispatcher::singleton()
          .findSchemaOrThrow("tpu::sparse_gather", "")
          .typed<at::Tensor(const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, int64_t)>());

  at::AutoDispatchBelowADInplaceOrView guard;
  return sparse_gather_op->call(row_pointers, indices, operand,
                                max_non_zeroes_per_row);
}

torch::autograd::variable_list AtenSparseGatherAutograd::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
  const auto saved = ctx->get_saved_variables();
  const at::Tensor& indices = saved[0];
  const auto operand_sizes = ctx->saved_data["operand_sizes"].toIntVector();
  const auto operand_scalar_type = static_cast<at::ScalarType>(
      ctx->saved_data["operand_scalar_type"].toInt());

  const at::Tensor grad_output = grad_outputs[0].contiguous();

  at::Tensor grad_operand;  // UNINITIALIZED_TENSOR_OK
  // Check if operand (input index 2) requires gradient.
  if (ctx->needs_input_grad(2)) {
    static const absl::NoDestructor sparse_gather_backward_op(
        at::Dispatcher::singleton()
            .findSchemaOrThrow("tpu::sparse_gather_backward", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&,
                              const at::Tensor&)>());

    at::Tensor grad_operand_template = at::zeros(
        operand_sizes, grad_output.options().dtype(operand_scalar_type));
    at::AutoDispatchBelowADInplaceOrView guard;
    grad_operand = sparse_gather_backward_op->call(grad_output, indices,
                                                   grad_operand_template);
  }

  // Inputs are: (row_pointers, indices, operand, max_non_zeroes_per_row).
  // Only operand receives gradient.
  return {at::Tensor(), at::Tensor(), grad_operand, at::Tensor()};
}

}  // namespace torch_tpu
