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

#include "torch_tpu/ops/dropout/dropout_aten_kernels.h"

#include <mutex>
#include <optional>
#include <tuple>
#include <utility>

#include "absl/status/statusor.h"
#include "ATen/Context.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/ops/full_like.h"
#include "ATen/ops/ones.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Optional.h"
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
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

NAryMlirOpBuilder<2, 3> GetDropoutFunctional(const at::Tensor& input,
                                             double p) {
  return [p](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<MlirOpResults<3>> {
    auto& [rng_state, input] = inputs;
    return BuildDropoutTrainShlo(rng_state, input, p);
  };
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> AtenDropout(const at::Tensor& input,
                                               double p,
                                               c10::optional<bool> train) {
  TT_KERNEL(OpName::kDropout, param_keys, (input, p, train), {
    TT_CHECK_THROW(p > 0 && p < 1.0, error::kInvalidArgument)
        << "expected p to be in the exclusive range (0, 1), got " << p;

    // https://docs.pytorch.org/docs/stable/generated/torch.nn.functionelu_aten_kernelsal.dropout.html#torch-nn-functional-dropout
    // at::dropout returns a scaled tensor, and a boolean mask of retained
    // values. In inference mode or with nonpositive drop rate p, we don't scale
    // the tensor, and return an all-true mask as a no-op.
    if (!train.has_value() || p <= 0.0) {
      return {
          input,
          at::ones(input.sizes(),
                   at::TensorOptions(input.scalar_type()).dtype(at::kBool))};
    } else if (p >= 1.0) {
      // If p is 1.0, we drop all values.
      return {at::full_like(input, 0.0, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, std::nullopt),
              at::full_like(input, 0, c10::ScalarType::Bool, std::nullopt,
                            std::nullopt, std::nullopt, std::nullopt)};
    }

    auto gen = at::get_generator_or_default<DeviceGeneratorImpl>(
        std::nullopt, GetDefaultDeviceGenerator());

    // See Note [Acquire lock when using random generators]
    // NOLINTNEXTLINE
    std::scoped_lock<std::mutex> lock(gen->mutex_);

    // Retrieve the rng_state tensor from the default device generator.
    at::Tensor rng_input_state = gen->DeviceStateTensor();

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(input.scalar_type()));

    TT_ASSIGN_OR_THROW(
        (auto [rng_output_state_buf, output_buf, mask_buf]),
        (DispatchOp<2, 3>(GetDropoutFunctional(input, p),
                          {rng_input_state, input},
                          {.out_dtypes = {mlir::ElementType::UI64, output_dtype,
                                          mlir::ElementType::PRED},
                           .out_dims_list = {{2}, input.sizes(), input.sizes()},
                           .op_param_cache_keys = std::move(param_keys)})));

    auto rng_output_state = MakeTensor(std::move(rng_output_state_buf));
    TT_THROW_IF_ERROR(gen->SetDeviceStateTensor(rng_output_state));

    return {MakeTensor(std::move(output_buf)), MakeTensor(std::move(mask_buf))};
  });
}

namespace {

NAryMlirOpBuilder<2, 1> GetDropoutBackwardFunctional(double scale) {
  return [scale](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    auto& [grad_output, mask] = inputs;
    return BuildDropoutBackwardShlo(grad_output, mask, scale);
  };
}

}  // namespace

at::Tensor AtenNativeDropoutBackward(const at::Tensor& grad_output,
                                     const at::Tensor& mask, double scale) {
  TT_KERNEL(
      OpName::kNativeDropoutBackward, param_keys, (grad_output, mask, scale), {
        at::ScalarType output_scalar_type = grad_output.scalar_type();
        if (!c10::isFloatingType(output_scalar_type) &&
            !c10::isComplexType(output_scalar_type)) {
          output_scalar_type = at::kFloat;
        }
        TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(output_scalar_type));
        TT_ASSIGN_OR_THROW(
            auto output_buf,
            (DispatchOp<2, 1>(GetDropoutBackwardFunctional(scale),
                              {grad_output, mask},
                              {.out_dtype = output_dtype,
                               .out_dims = grad_output.sizes(),
                               .op_param_cache_keys = std::move(param_keys)})));
        return MakeTensor(std::move(output_buf));
      });
}

}  // namespace torch_tpu
