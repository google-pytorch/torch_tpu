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

#include "torch_tpu/ops/optimization_barrier/optimization_barrier_kernels.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/view_decomposition/contiguous_to_view.h"

namespace torch_tpu {

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

    DispatchOpOptions<torch_tpu::kDynamicSize> options = {
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

}  // namespace torch_tpu
