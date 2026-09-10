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

#include "csrc/ops/experimental/ragged_all_to_all/ragged_all_to_all_aten_kernels.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/ops/empty_like.h"
#include "ATen/ops/zeros.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/utils.h"
#include "csrc/distributed/process_group_tpu.h"
#include "csrc/distributed/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/experimental/ragged_all_to_all/ragged_all_to_all_builder.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/autograd/custom_function.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/GroupRegistry.hpp"
#include "torch/csrc/distributed/c10d/ProcessGroup.hpp"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

absl::StatusOr<ProcessGroupTpu* absl_nonnull> GetProcessGroupTpu(
    std::string_view process_group_name) {
  std::string process_group_name_str(process_group_name);
  c10::intrusive_ptr<c10d::ProcessGroup> pg =
      c10d::resolve_process_group(process_group_name_str);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=process group resolution
      pg != nullptr, error::kInternal)
      << "failed to resolve given process group";
  c10::intrusive_ptr<c10d::Backend> backend =
      pg->getBackend(c10::DeviceType::PrivateUse1);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=backend resolution
      backend != nullptr, error::kInternal)
      << "failed to get backend for tpu device";

  auto* process_group_tpu = dynamic_cast<ProcessGroupTpu*>(backend.get());
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=backend dynamic cast
      process_group_tpu != nullptr, error::kInternal)
      << "failed to cast c10d::Backend to ProcessGroupTpu";
  return process_group_tpu;
}

static absl::StatusOr<DeviceBufferRef> RaggedAllToAllCommon(
    const at::Tensor& operand, const at::Tensor& output,
    const at::Tensor& input_offsets, const at::Tensor& send_sizes,
    const at::Tensor& output_offsets, const at::Tensor& recv_sizes,
    std::string_view process_group_name, OpParamCacheKeys& param_keys) {
  at::ScalarType out_scalar_type = output.scalar_type();
  TT_ASSIGN_OR_RETURN(auto out_dtype,
                      ConvertTo<mlir::ElementType>(out_scalar_type));

  std::string process_group_name_str(process_group_name);

  auto op_builder =
      [process_group_name_str](FixedSizeSpan<mlir::MlirOp, 6> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [operand, output, input_offsets, send_sizes, output_offsets,
           recv_sizes] = inputs;
    auto& builder = operand.getBuilder();
    TT_ASSIGN_OR_RETURN(ProcessGroupTpu * pg,
                        GetProcessGroupTpu(process_group_name_str));
    mlir::DenseIntElementsAttr replica_groups_attr =
        BuildReplicaGroupsAttr(builder, pg->GetSubgroupDeviceIds());

    return BuildRaggedAllToAllShlo(operand, output, input_offsets, send_sizes,
                                   output_offsets, recv_sizes,
                                   replica_groups_attr);
  };

  return DispatchOp<6>(
      std::move(op_builder),
      {operand, output, input_offsets, send_sizes, output_offsets, recv_sizes},
      {.out_dtype = out_dtype,
       .out_dims = CopyIntVector(output.sizes()),
       .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenRaggedAllToAll(const at::Tensor& operand,
                              const at::Tensor& output,
                              const at::Tensor& input_offsets,
                              const at::Tensor& send_sizes,
                              const at::Tensor& output_offsets,
                              const at::Tensor& recv_sizes,
                              std::string_view process_group_name) {
  TT_KERNEL(OpName::kRaggedAllToAll, param_keys,
            (operand, output, input_offsets, send_sizes, output_offsets,
             recv_sizes, process_group_name),
            {
              TT_ASSIGN_OR_THROW(
                  auto result,
                  RaggedAllToAllCommon(operand, output, input_offsets,
                                       send_sizes, output_offsets, recv_sizes,
                                       process_group_name, param_keys));
              return MakeTensor(std::move(result));
            });
}

at::Tensor& AtenRaggedAllToAllOut(
    const at::Tensor& operand, const at::Tensor& output,
    const at::Tensor& input_offsets, const at::Tensor& send_sizes,
    const at::Tensor& output_offsets, const at::Tensor& recv_sizes,
    std::string_view process_group_name, at::Tensor& out) {
  TT_KERNEL(OpName::kRaggedAllToAll, param_keys,
            (operand, output, input_offsets, send_sizes, output_offsets,
             recv_sizes, process_group_name, out),
            {
              TT_ASSIGN_OR_THROW(
                  auto result,
                  RaggedAllToAllCommon(operand, output, input_offsets,
                                       send_sizes, output_offsets, recv_sizes,
                                       process_group_name, param_keys));
              TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
              return out;
            });
}

at::Tensor AtenRaggedAllToAllAutograd::forward(
    torch::autograd::AutogradContext* ctx, const at::Tensor& operand,
    const at::Tensor& output, const at::Tensor& input_offsets,
    const at::Tensor& send_sizes, const at::Tensor& output_offsets,
    const at::Tensor& recv_sizes, std::string_view process_group_name) {
  // Cache the operator handle to avoid string schema lookup on every call.
  static const absl::NoDestructor op(
      at::Dispatcher::singleton()
          .findSchemaOrThrow("tpu::ragged_all_to_all", "")
          .typed<at::Tensor(const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, const at::Tensor&,
                            std::string_view)>());

  if (operand.requires_grad()) {
    TT_ASSIGN_OR_THROW(ProcessGroupTpu* const pg,
                       GetProcessGroupTpu(process_group_name));

    // Note: dev_output_offsets and dev_input_offsets cannot be marked const
    // because c10d::ProcessGroup::alltoall_base takes mutable at::Tensor&
    // references per PyTorch Backend API.
    at::Tensor dev_output_offsets = output_offsets.device() == operand.device()
                                        ? output_offsets
                                        : output_offsets.to(operand.device());
    at::Tensor dev_input_offsets = input_offsets.device() == operand.device()
                                       ? input_offsets
                                       : input_offsets.to(operand.device());

    at::Tensor bwd_input_offsets = at::empty_like(dev_output_offsets);
    at::Tensor bwd_output_offsets = at::empty_like(dev_input_offsets);

    if (pg->getSize() <= 1) {
      bwd_input_offsets = dev_output_offsets.clone();
      bwd_output_offsets = dev_input_offsets.clone();
    } else {
      std::vector<int64_t> empty_split_sizes;  // INT_VEC_OK
      const c10d::AllToAllOptions opts;

      // In the backward pass, gradients flow in the reverse direction of the
      // forward pass. If rank i sent tokens to rank j in forward:
      //   - Forward send:   rank i reads operand at forward input_offsets[j].
      //   - Forward recv:   rank j writes output at forward output_offsets[i]
      //                     (known on rank i as output_offsets[j]).
      // In backward, rank j sends gradients of those tokens back to rank i:
      //   - Backward send:  rank j reads grad_output at its forward receive
      //                     offset from rank i, which is rank i's forward
      //                     output_offsets[j].
      //   - Backward recv:  rank i writes grad_operand at its forward send
      //                     offset to rank j, which is rank i's forward
      //                     input_offsets[j].
      //
      // Because each rank i originally knows only its own forward offsets:
      //   output_offsets[j] = destination offset on rank j for data from rank i
      //   input_offsets[j]  = source offset on rank i for data sent to rank j
      //
      // The offsets must be transposed across ranks:
      //   bwd_input_offsets[j] on rank i  = forward output_offsets[i] on rank j
      //   bwd_output_offsets[j] on rank i = forward input_offsets[i] on rank j
      //
      // We achieve this transposition directly on TPU using an equal-splits
      // alltoall_base (1 int32 per peer rank):
      //   - In alltoall_base, rank i sends element j of its tensor to rank j,
      //     and receives element i from rank j into position j of the output.
      //   - Transposing dev_output_offsets across ranks produces
      //   bwd_input_offsets.
      //   - Transposing dev_input_offsets across ranks produces
      //   bwd_output_offsets.
      pg->alltoall_base(bwd_input_offsets, dev_output_offsets,
                        empty_split_sizes, empty_split_sizes, opts);
      pg->alltoall_base(bwd_output_offsets, dev_input_offsets,
                        empty_split_sizes, empty_split_sizes, opts);
    }

    ctx->save_for_backward(
        {bwd_input_offsets, send_sizes, bwd_output_offsets, recv_sizes});
    ctx->saved_data["operand_sizes"] = operand.sizes().vec();  // VEC_OK
    ctx->saved_data["operand_scalar_type"] =
        static_cast<int64_t>(operand.scalar_type());
    ctx->saved_data["process_group_name"] = std::string(process_group_name);
  }

  at::AutoDispatchBelowADInplaceOrView guard;
  return op->call(operand, output, input_offsets, send_sizes, output_offsets,
                  recv_sizes, process_group_name);
}

torch::autograd::variable_list AtenRaggedAllToAllAutograd::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
  // Cache the operator handle to avoid string schema lookup on every call.
  static const absl::NoDestructor op(
      at::Dispatcher::singleton()
          .findSchemaOrThrow("tpu::ragged_all_to_all", "")
          .typed<at::Tensor(const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, const at::Tensor&,
                            std::string_view)>());

  const auto saved = ctx->get_saved_variables();
  if (saved.empty() || !ctx->needs_input_grad(0)) {
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor()};
  }

  const at::Tensor& bwd_input_offsets = saved[0];
  const at::Tensor& send_sizes = saved[1];
  const at::Tensor& bwd_output_offsets = saved[2];
  const at::Tensor& recv_sizes = saved[3];

  const auto operand_sizes = ctx->saved_data["operand_sizes"].toIntVector();
  const auto operand_scalar_type = static_cast<at::ScalarType>(
      ctx->saved_data["operand_scalar_type"].toInt());
  const std::string& process_group_name =
      ctx->saved_data["process_group_name"].toStringRef();

  const at::Tensor& grad_output = grad_outputs[0];

  at::Tensor grad_operand_template = at::zeros(
      operand_sizes, grad_output.options().dtype(operand_scalar_type));

  // The backward pass performs reverse ragged all-to-all communication:
  //
  // Forward pass:
  //   Rank i -> Rank j:
  //     Source:      operand at input_offsets[j], size = send_sizes[j]
  //     Destination: output at output_offsets[j] on rank j, size =
  //     send_sizes[j]
  //
  // Backward pass (routing gradients back to source operands):
  //   Rank i -> Rank j (sending gradients for tokens received from j in
  //   forward):
  //     Source:      grad_output at bwd_input_offsets[j], size = recv_sizes[j]
  //                  (where bwd_input_offsets[j] is rank j's forward
  //                  output_offsets[i],
  //                   the exact slice where j's tokens landed in i's output
  //                   buffer).
  //     Destination: grad_operand at bwd_output_offsets[j] on rank j, size =
  //     recv_sizes[j]
  //                  (where bwd_output_offsets[j] is rank j's forward
  //                  input_offsets[i],
  //                   the exact slice where j originally read its tokens from
  //                   its operand buffer).
  //
  // Therefore, the ragged_all_to_all operator is invoked with:
  //   - operand:        grad_output
  //   - output:         grad_operand_template (initialized to zeros)
  //   - input_offsets:  bwd_input_offsets (transposed forward output_offsets)
  //   - send_sizes:     recv_sizes (tokens received in forward = gradients sent
  //   in backward)
  //   - output_offsets: bwd_output_offsets (transposed forward input_offsets)
  //   - recv_sizes:     send_sizes (tokens sent in forward = gradients received
  //   in backward)
  at::AutoDispatchBelowADInplaceOrView guard;
  at::Tensor grad_operand =
      op->call(grad_output, grad_operand_template, bwd_input_offsets,
               recv_sizes, bwd_output_offsets, send_sizes, process_group_name);

  return {grad_operand, at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor()};
}

}  // namespace torch_tpu
