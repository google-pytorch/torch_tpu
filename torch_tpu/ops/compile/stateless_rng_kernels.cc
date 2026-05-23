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

#include "torch_tpu/ops/compile/stateless_rng_kernels.h"

#include <optional>
#include <tuple>
#include <utility>

#include "ATen/Context.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/full_like.h"
#include "ATen/ops/ones.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Optional.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_gen_impl.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/dropout/dropout.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

std::tuple<at::Tensor, at::Tensor, at::Tensor> TorchTpuStatelessDropout(
    const at::Tensor& rng_state, const at::Tensor& input, double p,
    c10::optional<bool> train) {
  TT_KERNEL(
      OpName::kTorchTpuStatelessDropout, param_keys,
      (rng_state, input, p, train), {
        at::Tensor rng_state_u64 = rng_state.view(at::kUInt64);
        auto gen_impl = at::get_generator_or_default<DeviceGeneratorImpl>(
            std::nullopt, GetDefaultDeviceGenerator(input.get_device()));
        TT_THROW_IF_ERROR(gen_impl->CheckDeviceStateTensor(rng_state_u64));

        if (!train.has_value() || p <= 0.0) {
          return {rng_state, input,
                  at::ones(
                      input.sizes(),
                      at::TensorOptions(input.scalar_type()).dtype(at::kBool))};
        } else if (p >= 1.0) {
          // If p is 1.0, we drop all values.
          return {rng_state,
                  at::full_like(input, 0.0, std::nullopt, std::nullopt,
                                std::nullopt, std::nullopt, std::nullopt),
                  at::full_like(input, 0, c10::ScalarType::Bool, std::nullopt,
                                std::nullopt, std::nullopt, std::nullopt)};
        }

        TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));

        auto op_builder = [p](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
          auto& [rng_state_u64, input] = inputs;
          return BuildDropoutTrainShlo(rng_state_u64, input, p);
        };

        TT_ASSIGN_OR_THROW(
            (auto [rng_output_state_buf, output_buf, mask_buf]),
            (DispatchOp<2, 3>(
                std::move(op_builder), {rng_state_u64, input},
                {.out_dtypes = {mlir::ElementType::UI64, output_dtype,
                                mlir::ElementType::PRED},
                 .out_dims_list = {{2}, input.sizes(), input.sizes()},
                 .op_param_cache_keys = std::move(param_keys)})));

        return {
            MakeTensor(std::move(rng_output_state_buf)).view(at::kByte),
            MakeTensor(std::move(output_buf)),
            MakeTensor(std::move(mask_buf)),
        };
      });
}

}  // namespace torch_tpu
