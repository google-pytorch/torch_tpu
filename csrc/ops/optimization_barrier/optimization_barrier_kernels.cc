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
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/dispatch/Dispatcher.h"
#include "absl/base/no_destructor.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
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
#include "torch/csrc/autograd/custom_function.h"
#include "torch/csrc/autograd/functions/utils.h"

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

// Autograd function of the barrier, which is an identity on every tensor.
class OptimizationBarrierFunction
    : public torch::autograd::Function<OptimizationBarrierFunction> {
 public:
  static constexpr bool is_traceable = true;

  static std::vector<at::Tensor> forward(torch::autograd::AutogradContext* ctx,
                                         at::TensorList self) {
    // An output without a gradient must not send zeros back: the derivative
    // of what produced it may be infinite.
    ctx->set_materialize_grads(false);
    at::AutoDispatchBelowADInplaceOrView guard;
    return CallOptimizationBarrier(self);
  }

  // The gradients pass together through one barrier, as in JAX, so that the
  // backward computation is fenced like the forward one.
  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      std::vector<at::Tensor> grad_outputs) {
    std::vector<at::Tensor> gradients;
    for (size_t i = 0; i < grad_outputs.size(); ++i) {
      if (!ctx->needs_input_grad(i)) {
        grad_outputs[i].reset();
      } else if (grad_outputs[i].defined()) {
        gradients.push_back(grad_outputs[i]);
      }
    }
    if (!gradients.empty()) {
      // Dispatched above autograd, which keeps double backward working.
      std::vector<at::Tensor> fenced = CallOptimizationBarrier(gradients);
      size_t next = 0;
      for (at::Tensor& gradient : grad_outputs) {
        if (gradient.defined()) {
          gradient = std::move(fenced[next++]);
        }
      }
    }
    return grad_outputs;
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
  // Nothing to differentiate. Forward-mode tangents are not dropped here: the
  // autograd function below rejects them.
  if (!torch::autograd::compute_requires_grad(self) &&
      !torch::autograd::isFwGradDefinedTensorList(self)) {
    at::AutoDispatchBelowADInplaceOrView guard;
    return CallOptimizationBarrier(self);
  }
  std::vector<at::Tensor> outputs = OptimizationBarrierFunction::apply(self);
  // An output requires gradients only if its own input does. It is detached
  // rather than marked non-differentiable, which compiled autograd rejects.
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].requires_grad() && !self[i].requires_grad()) {
      outputs[i].detach_();
    }
  }
  return outputs;
}

}  // namespace torch_tpu
