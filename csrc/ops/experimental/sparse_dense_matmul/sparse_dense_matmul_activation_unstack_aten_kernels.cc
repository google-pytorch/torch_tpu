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

#include "csrc/ops/experimental/sparse_dense_matmul/sparse_dense_matmul_activation_unstack_aten_kernels.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/ops/zeros.h"
#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
// NOTE: Do not add torch/csrc/autograd/node.h here; see the note on
// AtenSparseDenseMatmulActivationUnstackAutograd in the header.
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
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/autograd/custom_function.h"
#include "torch/headeronly/core/DeviceType.h"

namespace torch_tpu {

namespace {

auto SparseDenseMatmulActivationUnstackBuilder(
    at::IntArrayRef per_feature_batch_sizes, at::IntArrayRef per_feature_dims) {
  using IntVector = std::vector<int64_t>;  // INT_VEC_OK
  return [batch_sizes = IntVector(per_feature_batch_sizes.begin(),
                                  per_feature_batch_sizes.end()),
          dims = IntVector(per_feature_dims.begin(), per_feature_dims.end())](
             mlir::MlirOp stacked_activations)
             -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    mlir::MlirBuilder& builder = stacked_activations.getBuilder();
    mlir::OpBuilder& op_builder = builder.getOpBuilder();

    const size_t num_features = batch_sizes.size();
    std::vector<mlir::Type> result_types;
    result_types.reserve(num_features);
    std::vector<mlir::Attribute> result_layout_attrs;
    result_layout_attrs.reserve(num_features);

    for (size_t i = 0; i < num_features; ++i) {
      result_types.push_back(mlir::RankedTensorType::get(
          {batch_sizes[i], dims[i]}, op_builder.getF32Type()));
      result_layout_attrs.push_back(op_builder.getIndexTensorAttr({0, 1}));
    }

    const std::string call_target = "SparseActivationsUnstackInterleaved";
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
        "operand_layouts",
        op_builder.getArrayAttr({op_builder.getIndexTensorAttr({1, 0})}));
    auto result_layouts_attr = op_builder.getNamedAttr(
        "result_layouts", op_builder.getArrayAttr(result_layout_attrs));

    auto op = mlir::stablehlo::CustomCallOp::create(
        op_builder, builder.getLoc(),
        /*resultTypes=*/result_types,
        /*operands=*/mlir::ValueRange{stacked_activations.getValue()},
        {call_target_attr, has_side_effect_attr, api_version_attr,
         operand_layouts_attr, result_layouts_attr});

    mlir::SmallVector<mlir::MlirOp> results;
    results.reserve(op.getNumResults());
    for (size_t i = 0; i < op.getNumResults(); ++i) {
      results.push_back(mlir::MlirOp(builder, op.getResult(i)));
    }
    return results;
  };
}

}  // namespace

absl::Status ValidateSparseDenseMatmulActivationUnstackInputs(
    const ConstTensor& stacked_activations,
    at::IntArrayRef per_feature_batch_sizes, at::IntArrayRef per_feature_dims) {
  TT_ASSIGN_OR_RETURN(const int64_t stacked_activations_dim,
                      stacked_activations.dim());
  TT_ASSIGN_OR_RETURN(const c10::SymIntArrayRef stacked_activations_sizes,
                      stacked_activations.sym_sizes());
  TT_RET_CHECK(stacked_activations_dim == 2, error::kInvalidArgument)
      << "expected stacked_activations to be a 2D tensor, got a "
      << stacked_activations_dim << "D tensor of shape "
      << ToString(stacked_activations_sizes);

  TT_RET_CHECK(stacked_activations.scalar_type() == at::kFloat,
               error::kInvalidArgument)
      << "expected stacked_activations to have float32 dtype, got "
      << ToString(stacked_activations.scalar_type());

  TT_RET_CHECK(!per_feature_batch_sizes.empty(), error::kInvalidArgument)
      << "expected at least one feature, but per_feature_batch_sizes is empty";

  TT_RET_CHECK(per_feature_batch_sizes.size() == per_feature_dims.size(),
               error::kInvalidArgument)
      << "expected per_feature_batch_sizes and per_feature_dims to have the "
         "same "
         "size, got "
      << per_feature_batch_sizes.size() << " and " << per_feature_dims.size();

  // For a symbolic stacked_embedding_dim, the comparison below adds a
  // relational guard but doesn't specialize.
  const c10::SymInt& stacked_embedding_dim = stacked_activations_sizes[1];
  int64_t total_batch_size = 0;
  for (size_t i = 0; i < per_feature_batch_sizes.size(); ++i) {
    TT_RET_CHECK(per_feature_batch_sizes[i] > 0, error::kInvalidArgument)
        << "expected per_feature_batch_sizes[" << i << "] to be positive, got "
        << per_feature_batch_sizes[i];
    TT_RET_CHECK(per_feature_dims[i] > 0, error::kInvalidArgument)
        << "expected per_feature_dims[" << i << "] to be positive, got "
        << per_feature_dims[i];
    TT_RET_CHECK(stacked_embedding_dim.sym_ge(per_feature_dims[i])
                     .guard_bool(__FILE__, __LINE__),
                 error::kInvalidArgument)
        << "expected per_feature_dims[" << i << "] (" << per_feature_dims[i]
        << ") to be at most stacked_embedding_dim (" << stacked_embedding_dim
        << ")";
    total_batch_size += per_feature_batch_sizes[i];
  }

  const c10::SymInt& stacked_batch_size = stacked_activations_sizes[0];
  TT_RET_CHECK(stacked_batch_size.sym_eq(total_batch_size)
                   .guard_bool(__FILE__, __LINE__),
               error::kInvalidArgument)
      << "expected stacked_activations dim 0 (" << stacked_batch_size
      << ") to match total stacked batch size (" << total_batch_size << ")";

  return absl::OkStatus();
}

std::vector<at::Tensor> AtenSparseDenseMatmulActivationUnstack(
    const at::Tensor& stacked_activations,
    at::IntArrayRef per_feature_batch_sizes, at::IntArrayRef per_feature_dims) {
  TT_KERNEL(
      torch_tpu::OpName::kSparseDenseMatmulActivationUnstack, param_keys,
      (stacked_activations, per_feature_batch_sizes, per_feature_dims), {
        TT_THROW_IF_ERROR(ValidateSparseDenseMatmulActivationUnstackInputs(
            stacked_activations, per_feature_batch_sizes, per_feature_dims));

        const size_t num_features = per_feature_batch_sizes.size();
        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           torch_tpu::ConvertTo<mlir::ElementType>(
                               stacked_activations.scalar_type()));

        std::vector<mlir::ElementType> out_dtypes(num_features, out_dtype);
        std::vector<torch_tpu::Dimensions> out_dims_storage;
        out_dims_storage.reserve(num_features);
        std::vector<absl::Span<const int64_t>> out_dims_list;
        out_dims_list.reserve(num_features);
        for (size_t i = 0; i < num_features; ++i) {
          out_dims_storage.push_back(
              {per_feature_batch_sizes[i], per_feature_dims[i]});
          out_dims_list.push_back(out_dims_storage.back());
        }

        auto builder_fn = SparseDenseMatmulActivationUnstackBuilder(
            per_feature_batch_sizes, per_feature_dims);

        TT_ASSIGN_OR_THROW(
            std::vector<int64_t> device_assignment,  // INT_VEC_OK
            GetWorldDeviceAssignment());

        DispatchOpOptions<torch_tpu::kDynamicSize> options = {
            .out_dtypes = out_dtypes,
            .out_dims_list = out_dims_list,
            .op_param_cache_keys = std::move(param_keys),
            .device_assignment = std::move(device_assignment),
        };

        TT_ASSIGN_OR_THROW(std::vector<DeviceBufferRef> result_buffers,
                           (torch_tpu::DispatchOp<1, torch_tpu::kDynamicSize>(
                               std::move(builder_fn), stacked_activations,
                               std::move(options))));

        std::vector<at::Tensor> results;
        results.reserve(result_buffers.size());
        for (auto& buf : result_buffers) {
          results.push_back(torch_tpu::MakeTensor(std::move(buf)));
        }
        return results;
      });
}

// Autograd implementation for
// torch.ops.tpu._sparse_dense_matmul_activation_unstack.
//
// In forward, we save the stacked dimensions and per-feature sizes in
// `ctx->saved_data`. Saving `stacked_activations` is explicitly avoided to
// minimize peak HBM memory.
//
// In backward, we invoke torch.ops.tpu._sparse_dense_matmul_gradient_stack to
// stack and interleave the incoming gradients across SparseCore cores.
std::vector<at::Tensor> AtenSparseDenseMatmulActivationUnstackAutograd::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& stacked_activations,
    at::IntArrayRef per_feature_batch_sizes, at::IntArrayRef per_feature_dims) {
  // Saved as SymInts so that tracing with symbolic shapes doesn't specialize.
  ctx->saved_data["stacked_batch_size"] = stacked_activations.sym_size(0);
  ctx->saved_data["stacked_feature_dim"] = stacked_activations.sym_size(1);
  ctx->saved_data["per_feature_batch_sizes"] =
      per_feature_batch_sizes.vec();                             // VEC_OK
  ctx->saved_data["per_feature_dims"] = per_feature_dims.vec();  // VEC_OK

  static const absl::NoDestructor unstack_op(
      at::Dispatcher::singleton()
          .findSchemaOrThrow("tpu::_sparse_dense_matmul_activation_unstack", "")
          .typed<std::vector<at::Tensor>(const at::Tensor&, at::IntArrayRef,
                                         at::IntArrayRef)>());

  at::AutoDispatchBelowADInplaceOrView guard;
  return unstack_op->call(stacked_activations, per_feature_batch_sizes,
                          per_feature_dims);
}

std::vector<at::Tensor>
AtenSparseDenseMatmulActivationUnstackAutograd::backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<at::Tensor> grad_outputs) {
  at::Tensor grad_stacked;  // UNINITIALIZED_TENSOR_OK
  if (ctx->needs_input_grad(0)) {
    // The `int` arguments of `tpu::_sparse_dense_matmul_gradient_stack` force
    // symbolic sizes to be specialized here.
    const int64_t stacked_batch_size =
        ctx->saved_data["stacked_batch_size"].toSymInt().guard_int(__FILE__,
                                                                   __LINE__);
    const int64_t stacked_feature_dim =
        ctx->saved_data["stacked_feature_dim"].toSymInt().guard_int(__FILE__,
                                                                    __LINE__);
    const auto per_feature_batch_sizes =
        ctx->saved_data["per_feature_batch_sizes"].toIntVector();
    const auto per_feature_dims =
        ctx->saved_data["per_feature_dims"].toIntVector();

    // Ensure all gradients are defined and contiguous. If downstream loss does
    // not use certain feature outputs, materialize a zero tensor of matching
    // shape so gradient_stack receives valid 2D tensors.
    std::vector<at::Tensor> unstacked_gradients;
    unstacked_gradients.reserve(grad_outputs.size());
    for (size_t i = 0; i < grad_outputs.size(); ++i) {
      if (grad_outputs[i].defined()) {
        unstacked_gradients.push_back(grad_outputs[i].contiguous());
      } else {
        unstacked_gradients.push_back(at::zeros(
            {per_feature_batch_sizes[i], per_feature_dims[i]},
            at::TensorOptions().device(at::kPrivateUse1).dtype(at::kFloat)));
      }
    }

    static const absl::NoDestructor grad_stack_op(
        at::Dispatcher::singleton()
            .findSchemaOrThrow("tpu::_sparse_dense_matmul_gradient_stack", "")
            .typed<at::Tensor(at::TensorList, int64_t, int64_t)>());

    at::AutoDispatchBelowADInplaceOrView guard;
    grad_stacked = grad_stack_op->call(unstacked_gradients, stacked_batch_size,
                                       stacked_feature_dim);
  }

  // Forward inputs: (stacked_activations, per_feature_batch_sizes,
  // per_feature_dims). Only stacked_activations receives gradient.
  return {grad_stacked, at::Tensor(), at::Tensor()};
}

}  // namespace torch_tpu
