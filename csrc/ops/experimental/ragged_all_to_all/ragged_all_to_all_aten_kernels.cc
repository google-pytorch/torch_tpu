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

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/core/function_schema.h"
#include "ATen/core/ivalue.h"
#include "ATen/core/stack.h"
#include "ATen/ops/zeros.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "c10/core/GradMode.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/api_error_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
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
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/GroupRegistry.hpp"
#include "torch/csrc/distributed/c10d/ProcessGroup.hpp"
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
  TT_RET_CHECK(operand.dim() == output.dim(), error::kInvalidArgument)
      << "expected operand and output to have the same number of dimensions, "
      << "got " << operand.dim() << " and " << output.dim();
  TT_RET_CHECK(operand.sizes().slice(1) == output.sizes().slice(1),
               error::kInvalidArgument)
      << "expected trailing dimensions of operand and output to match, got "
      << operand.sizes() << " and " << output.sizes();
  TT_RET_CHECK(operand.scalar_type() == output.scalar_type(),
               error::kInvalidArgument)
      << "expected operand and output to have the same dtype, got "
      << ToString(operand.scalar_type()) << " and "
      << ToString(output.scalar_type());

  at::ScalarType out_scalar_type = output.scalar_type();
  TT_ASSIGN_OR_RETURN(auto out_dtype,
                      ConvertTo<mlir::ElementType>(out_scalar_type));

  TT_ASSIGN_OR_RETURN(ProcessGroupTpu * pg,
                      GetProcessGroupTpu(process_group_name));

  auto op_builder = [device_groups = pg->GetReplicaGroups()](
                        FixedSizeSpan<mlir::MlirOp, 6> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [operand, output, input_offsets, send_sizes, output_offsets,
           recv_sizes] = inputs;
    auto& builder = operand.getBuilder();
    mlir::DenseIntElementsAttr replica_groups_attr =
        BuildReplicaGroupsAttr(builder, device_groups);

    return BuildRaggedAllToAllShlo(operand, output, input_offsets, send_sizes,
                                   output_offsets, recv_sizes,
                                   replica_groups_attr);
  };

  return DispatchOp<6>(
      std::move(op_builder),
      {operand, output, input_offsets, send_sizes, output_offsets, recv_sizes},
      {.out_dtype = out_dtype,
       .out_dims = CopyIntVector(output.sizes()),
       .op_param_cache_keys = std::move(param_keys),
       .device_assignment = pg->device_assignment()});
}

// Transposes a `[pg_size * K]` offset vector across `group_name` via
// `_c10d_functional::all_to_all_single` + `wait_tensor` (`callBoxed` handles
// both 4-arg and 5-arg PyTorch schemas).
absl::StatusOr<at::Tensor> TransposeOffsetsAllToAll(
    const at::Tensor& offsets, int64_t pg_size, std::string_view group_name) {
  static const absl::NoDestructor all_to_all_single_op(
      at::Dispatcher::singleton().findSchemaOrThrow(
          "_c10d_functional::all_to_all_single", ""));
  static const absl::NoDestructor wait_tensor_op(
      at::Dispatcher::singleton()
          .findSchemaOrThrow("_c10d_functional::wait_tensor", "")
          .typed<at::Tensor(const at::Tensor&)>());

  TT_RET_CHECK(offsets.dim() == 1, error::kInvalidArgument)
      << "expected 1D offsets, got " << offsets.dim() << "D";
  const c10::SymInt num_offsets = offsets.sym_size(0);
  TT_RET_CHECK(num_offsets % pg_size == 0, error::kInvalidArgument)
      << "expected the size of offsets to be divisible by process group size "
      << pg_size << ", got " << num_offsets;

  const std::vector<c10::SymInt> equal_splits(pg_size, num_offsets / pg_size);
  const std::vector<c10::Argument>& schema_args =
      all_to_all_single_op->schema().arguments();
  torch::jit::Stack stack;
  stack.reserve(schema_args.size());
  stack.emplace_back(offsets);
  stack.emplace_back(equal_splits);
  stack.emplace_back(equal_splits);
  stack.emplace_back(group_name);
  for (size_t i = stack.size(); i < schema_args.size(); ++i) {
    stack.push_back(schema_args[i].default_value().value_or(c10::IValue()));
  }
  all_to_all_single_op->callBoxed(&stack);
  return wait_tensor_op->call(torch::jit::pop(stack).toTensor());
}

[[nodiscard]] at::Tensor CallRaggedAllToAll(
    const at::Tensor& operand, const at::Tensor& output,
    const at::Tensor& input_offsets, const at::Tensor& send_sizes,
    const at::Tensor& output_offsets, const at::Tensor& recv_sizes,
    std::string_view process_group_name) {
  static const absl::NoDestructor op(
      at::Dispatcher::singleton()
          .findSchemaOrThrow("tpu::_ragged_all_to_all", "")
          .typed<at::Tensor(const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, const at::Tensor&,
                            std::string_view)>());
  at::AutoDispatchBelowADInplaceOrView guard;
  return op->call(operand, output, input_offsets, send_sizes, output_offsets,
                  recv_sizes, process_group_name);
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

namespace {

struct RaggedAllToAllAutogradNode
    : public torch::autograd::Function<RaggedAllToAllAutogradNode> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            const at::Tensor& operand, const at::Tensor& output,
                            const at::Tensor& input_offsets,
                            const at::Tensor& send_sizes,
                            const at::Tensor& output_offsets,
                            const at::Tensor& recv_sizes,
                            std::string_view process_group_name);

  static at::Tensor ForwardImpl(
      torch::autograd::AutogradContext* ctx, const at::Tensor& operand,
      const at::Tensor& output, const at::Tensor& input_offsets,
      const at::Tensor& send_sizes, const at::Tensor& output_offsets,
      const at::Tensor& recv_sizes, std::string_view process_group_name);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs);
};

at::Tensor RaggedAllToAllAutogradNode::forward(
    torch::autograd::AutogradContext* ctx, const at::Tensor& operand,
    const at::Tensor& output, const at::Tensor& input_offsets,
    const at::Tensor& send_sizes, const at::Tensor& output_offsets,
    const at::Tensor& recv_sizes, std::string_view process_group_name) {
  return RunWithApiErrorHandling(
      "ragged_all_to_all", &RaggedAllToAllAutogradNode::ForwardImpl, ctx,
      operand, output, input_offsets, send_sizes, output_offsets, recv_sizes,
      process_group_name);
}

at::Tensor RaggedAllToAllAutogradNode::ForwardImpl(
    torch::autograd::AutogradContext* ctx, const at::Tensor& operand,
    const at::Tensor& output, const at::Tensor& input_offsets,
    const at::Tensor& send_sizes, const at::Tensor& output_offsets,
    const at::Tensor& recv_sizes, std::string_view process_group_name) {
  const std::string group_name(process_group_name);
  const c10::intrusive_ptr<c10d::ProcessGroup> pg =
      c10d::resolve_process_group(group_name);
  const int64_t pg_size = pg->getSize();

  const at::Tensor dev_output_offsets =
      output_offsets.device() == operand.device()
          ? output_offsets
          : output_offsets.to(operand.device());
  const at::Tensor dev_input_offsets =
      input_offsets.device() == operand.device()
          ? input_offsets
          : input_offsets.to(operand.device());

  // Each rank knows only its own forward offsets (`[pg_size * K]` for
  // `K >= 1` updates per peer). Multi-rank groups transpose these `K`-element
  // blocks across ranks via `_c10d_functional::all_to_all_single` +
  // `wait_tensor` so `torch.compile` traces and orders the collectives.
  auto get_bwd_offsets =
      [pg_size,
       &group_name](const at::Tensor& offsets) -> absl::StatusOr<at::Tensor> {
    if (pg_size <= 1) {
      return offsets.clone();
    }
    return TransposeOffsetsAllToAll(offsets, pg_size, group_name);
  };
  TT_ASSIGN_OR_THROW(at::Tensor bwd_input_offsets,
                     get_bwd_offsets(dev_output_offsets));
  TT_ASSIGN_OR_THROW(at::Tensor bwd_output_offsets,
                     get_bwd_offsets(dev_input_offsets));

  ctx->save_for_backward(
      {bwd_input_offsets, send_sizes, bwd_output_offsets, recv_sizes});
  ctx->saved_data["operand_sizes"] = operand.sym_sizes();
  ctx->saved_data["operand_scalar_type"] =
      static_cast<int64_t>(operand.scalar_type());
  ctx->saved_data["process_group_name"] = group_name;

  return CallRaggedAllToAll(operand, output, input_offsets, send_sizes,
                            output_offsets, recv_sizes, process_group_name);
}

torch::autograd::variable_list RaggedAllToAllAutogradNode::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
  const auto saved = ctx->get_saved_variables();
  if (saved.empty() || !ctx->needs_input_grad(0)) {
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor()};
  }

  const at::Tensor& bwd_input_offsets = saved[0];
  const at::Tensor& send_sizes = saved[1];
  const at::Tensor& bwd_output_offsets = saved[2];
  const at::Tensor& recv_sizes = saved[3];

  const std::vector<c10::SymInt> operand_sizes =
      ctx->saved_data["operand_sizes"].toSymIntVector();
  const auto operand_scalar_type = static_cast<at::ScalarType>(
      ctx->saved_data["operand_scalar_type"].toInt());
  const std::string& process_group_name =
      ctx->saved_data["process_group_name"].toStringRef();

  const at::Tensor& grad_output = grad_outputs[0];

  const at::Tensor grad_operand_template = at::zeros_symint(
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
  const at::Tensor grad_operand = CallRaggedAllToAll(
      grad_output, grad_operand_template, bwd_input_offsets, recv_sizes,
      bwd_output_offsets, send_sizes, process_group_name);

  return {grad_operand, at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor()};
}

}  // namespace

at::Tensor AtenRaggedAllToAllAutograd(const at::Tensor& operand,
                                      const at::Tensor& output,
                                      const at::Tensor& input_offsets,
                                      const at::Tensor& send_sizes,
                                      const at::Tensor& output_offsets,
                                      const at::Tensor& recv_sizes,
                                      std::string_view process_group_name) {
  if (!c10::GradMode::is_enabled() || !operand.requires_grad()) {
    return CallRaggedAllToAll(operand, output, input_offsets, send_sizes,
                              output_offsets, recv_sizes, process_group_name);
  }
  return RaggedAllToAllAutogradNode::apply(operand, output, input_offsets,
                                           send_sizes, output_offsets,
                                           recv_sizes, process_group_name);
}

}  // namespace torch_tpu
