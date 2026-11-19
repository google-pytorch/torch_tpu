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

#include "csrc/ops/optimization_barrier/optimization_barrier_kernels.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/dispatch/Dispatcher.h"
#include "absl/base/no_destructor.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/autograd_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/view_decomposition/contiguous_to_view.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/dynamo/compiled_autograd.h"

namespace torch_tpu {

namespace {

// Calls torch.ops.tpu._optimization_barrier through the dispatcher.
std::vector<at::Tensor> CallOptimizationBarrier(at::TensorList tensors) {
  static const absl::NoDestructor op(
      at::Dispatcher::singleton()
          .findSchemaOrThrow("tpu::_optimization_barrier", "")
          .typed<std::vector<at::Tensor>(at::TensorList)>());
  return op->call(tensors);
}

// Passes the defined tensors together through one barrier.
[[nodiscard]] std::vector<at::Tensor> PassThroughOneBarrier(
    std::vector<at::Tensor> tensors) {
  std::vector<at::Tensor> defined;
  for (const at::Tensor& tensor : tensors) {
    if (tensor.defined()) {
      defined.push_back(tensor);
    }
  }
  if (!defined.empty()) {
    // Dispatched above autograd, which keeps higher-order derivatives working.
    std::vector<at::Tensor> fenced = CallOptimizationBarrier(defined);
    size_t next = 0;
    for (at::Tensor& tensor : tensors) {
      if (tensor.defined()) {
        tensor = std::move(fenced[next++]);
      }
    }
  }
  return tensors;
}

// Backward node of the barrier. Its k-th input and its k-th edge belong to the
// k-th tensor of the barrier that requires gradients.
class OptimizationBarrierBackward : public torch::autograd::Node {
 public:
  std::string name() const override { return "OptimizationBarrierBackward0"; }

  // Compiled autograd traces `apply`; the node has no state to swap.
  void compiled_args(
      torch::dynamo::autograd::CompiledNodeArgs& /*args*/) const override {}
  std::vector<at::Tensor> apply_with_saved(
      const std::vector<at::Tensor>& inputs,
      torch::dynamo::autograd::SwapSavedVariables& /*saved*/) override {
    return apply(std::vector<at::Tensor>(inputs));
  }

 protected:
  // The gradients pass together through one barrier, as in JAX, so that the
  // backward computation is fenced like the forward one.
  std::vector<at::Tensor> apply(std::vector<at::Tensor>&& grads) override {
    for (size_t i = 0; i < grads.size(); ++i) {
      if (!task_should_compute_output(i)) {
        grads[i].reset();
      }
    }
    return PassThroughOneBarrier(std::move(grads));
  }
};

}  // namespace

std::vector<at::Tensor> TorchTpuOptimizationBarrier(at::TensorList self) {
  TT_KERNEL(OpName::kTorchTpuOptimizationBarrier, _, (self), {
    auto op_builder = [](absl::Span<mlir::MlirOp> inputs,
                         mlir::MlirBuilder& builder)
        -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
      return mlir::stablehlo::OptimizationBarrier(builder, inputs);
    };

    std::vector<at::Tensor> inputs(self.begin(), self.end());
    std::vector<mlir::ElementType> out_dtypes;
    std::vector<absl::Span<const int64_t>> out_dims_list;
    out_dtypes.reserve(inputs.size());
    out_dims_list.reserve(inputs.size());

    for (const auto& input : inputs) {
      TT_ASSIGN_OR_THROW(const auto output_dtype,
                         ConvertTo<mlir::ElementType>(input.scalar_type()));
      out_dtypes.push_back(output_dtype);
      out_dims_list.push_back(input.sizes());
    }

    DispatchOpOptions<kDynamicSize> options = {
        .out_dtypes = out_dtypes,
        .out_dims_list = out_dims_list,
        .op_param_cache_keys = OpParamCacheKeys::Empty()};

    TT_ASSIGN_OR_THROW(std::vector<DeviceBufferRef> result_buffers,
                       (DispatchOp<kDynamicSize, kDynamicSize>(
                           std::move(op_builder), inputs, std::move(options))));

    std::vector<at::Tensor> result;
    result.reserve(result_buffers.size());
    for (auto i = 0; i < result_buffers.size(); ++i) {
      TT_ASSIGN_OR_THROW(
          result.emplace_back(),
          ContiguousToView(std::move(result_buffers[i]), inputs[i].strides(),
                           inputs[i].storage_offset()));
    }

    return result;
  });
}

std::vector<at::Tensor> TorchTpuOptimizationBarrierAutograd(
    at::TensorList self) {
  std::vector<at::Tensor> outputs;
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    outputs = CallOptimizationBarrier(self);
  }
  if (torch::autograd::compute_requires_grad(self)) {
    // An output has history only if its own input requires gradients.
    std::vector<at::Tensor> differentiable_inputs;
    std::vector<at::Tensor> differentiable_outputs;
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (torch::autograd::compute_requires_grad(self[i])) {
        differentiable_inputs.push_back(self[i]);
        differentiable_outputs.push_back(outputs[i]);
      }
    }
    const auto grad_fn = MakeAutogradNode<OptimizationBarrierBackward>();
    grad_fn->set_next_edges(
        torch::autograd::collect_next_edges(differentiable_inputs));
    torch::autograd::set_history(differentiable_outputs, grad_fn);
    FireNodeCreationHooks(grad_fn);
  }
  if (torch::autograd::isFwGradDefinedTensorList(self)) {
    // Forward mode: the tangents pass together through one barrier, as in JAX.
    std::vector<at::Tensor> tangents;
    tangents.reserve(self.size());
    for (const at::Tensor& tensor : self) {
      tangents.push_back(tensor._fw_grad(/*level=*/0));
    }
    tangents = PassThroughOneBarrier(std::move(tangents));
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (tangents[i].defined()) {
        outputs[i]._set_fw_grad(tangents[i], /*level=*/0,
                                /*is_inplace_op=*/false);
      }
    }
  }
  return outputs;
}

}  // namespace torch_tpu
