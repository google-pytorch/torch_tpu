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

#include "torch_tpu/csrc/ops/dropout/dropout_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "ATen/Context.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/ops/ones_like.h"
#include "ATen/ops/zeros_like.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Optional.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/dropout/dropout.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/rng_utils.h"

namespace torch_tpu {

namespace {

// Builds the SHLO for dropout in the train mode.
//
// Parameters:
//   input: The input tensor.
//   p: The dropout rate.
// Returns:
//   A pair of MlirOpResults, the first for the output tensor and the second for
//   the mask tensor.
NAryMlirOpBuilder<2, 2> GetDropoutFunctional(const at::Tensor& input,
                                             double p) {
  return [p](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<MlirOpResults<2>> {
    auto& [rng_state, input] = inputs;
    return BuildDropoutTrainShlo(rng_state, input, p);
  };
}

// Common helper function for forward dropout operations (AtenDropout and
// AtenFusedDropout). Validates input types and probability range, handles
// edge-case short circuits for p <= 0 and p >= 1, and dispatches the SHLO
// RNG dropout kernel.
std::tuple<at::Tensor, at::Tensor> AtenDropoutCommonImpl(
    const at::Tensor& input, double p, bool train,
    c10::optional<at::Generator> generator, OpParamCacheKeys param_keys) {
  TT_CHECK_THROW(p >= 0.0 && p <= 1.0, error::kInvalidArgument)
      << "expected p to be in the range [0, 1], got " << p;
  TT_CHECK_THROW(input.is_floating_point() || input.is_complex(),
                 error::kInvalidArgument)
      << "expected input to be floating point or complex, got "
      << input.scalar_type();

  if (p <= 0.0 || !train) {
    return {input, at::ones_like(input, input.options().dtype(at::kBool))};
  }
  if (p >= 1.0) {
    return {at::zeros_like(input),
            at::zeros_like(input, input.options().dtype(at::kBool))};
  }

  TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                     ConvertTo<mlir::ElementType>(input.scalar_type()));

  const int64_t num_elements = input.numel();
  const int64_t bit_width = TorchEquivalentBitwidth(output_dtype);

  TT_ASSIGN_OR_THROW(
      auto results,
      DispatchRngOpGeneral(
          generator,
          [&](at::Tensor rng_input_state) {
            return DispatchOp<2, 2>(
                GetDropoutFunctional(input, p), {rng_input_state, input},
                {.out_dtypes = {output_dtype, mlir::ElementType::PRED},
                 .out_dims_list = {input.sizes(), input.sizes()},
                 .op_param_cache_keys = std::move(param_keys)});
          },
          RngUsage{num_elements, bit_width}));

  // [output, mask]
  return {MakeTensor(std::move(results[0])), MakeTensor(std::move(results[1]))};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> AtenDropout(const at::Tensor& input,
                                               double p,
                                               c10::optional<bool> train) {
  TT_KERNEL(OpName::kDropout, param_keys, (input, p, train), {
    return AtenDropoutCommonImpl(input, p, train.value_or(true),
                                 /*generator=*/std::nullopt,
                                 std::move(param_keys));
  });
}

std::tuple<at::Tensor, at::Tensor> AtenFusedDropout(
    const at::Tensor& self, double p, c10::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kFusedDropout, param_keys, (self, p, generator), {
    return AtenDropoutCommonImpl(self, p, /*train=*/true, generator,
                                 std::move(param_keys));
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
        TT_CHECK_THROW(grad_output.scalar_type() != at::kBool &&
                           grad_output.scalar_type() != at::kInt &&
                           grad_output.scalar_type() != at::kLong,
                       error::kPythonNotImplementedError)
            << "not implemented for " << ToString(grad_output.scalar_type());
        TT_CHECK_THROW(mask.scalar_type() == at::kBool, error::kInvalidArgument)
            << "expected mask to be Bool scalar type, got "
            << mask.scalar_type();
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
