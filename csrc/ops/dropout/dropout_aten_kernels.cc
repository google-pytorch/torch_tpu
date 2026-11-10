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

#include "csrc/ops/dropout/dropout_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "ATen/Context.h"
#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Optional.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/dropout/dropout.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/rng_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

// Common helper function for forward dropout operations (AtenDropout and
// AtenFusedDropout). Validates input types and probability range, handles
// edge-case short circuits for p <= 0 and p >= 1, and dispatches the SHLO
// RNG dropout kernel.
std::tuple<at::Tensor, at::Tensor> AtenDropoutCommonImpl(
    const at::Tensor& input, double p, bool train,
    c10::optional<at::Generator> generator, DropoutProbabilityType prob_type,
    DropoutMaskType mask_type, OpParamCacheKeys param_keys) {
  TT_CHECK_THROW(p >= 0.0 && p <= 1.0, error::kInvalidArgument)
      << "expected p to be in the range [0, 1], got " << p;
  TT_CHECK_THROW(input.is_floating_point() || input.is_complex(),
                 error::kInvalidArgument)
      << "expected input to be floating point or complex, got "
      << ToString(input.scalar_type());

  const bool keep_all =
      !train ||
      ((prob_type == DropoutProbabilityType::kDropProbability) ? (p <= 0.0)
                                                               : (p >= 1.0));
  const bool drop_all =
      train &&
      ((prob_type == DropoutProbabilityType::kDropProbability) ? (p >= 1.0)
                                                               : (p <= 0.0));

  const mlir::ElementType mask_dtype = (mask_type == DropoutMaskType::kUInt8)
                                           ? mlir::ElementType::UI8
                                           : mlir::ElementType::PRED;

  TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                     ConvertTo<mlir::ElementType>(input.scalar_type()));

  if (keep_all) {
    auto op_builder =
        [mask_dtype](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<2>> {
      auto mask_op = MakeConstantLike(input, true, mask_dtype);
      return {{input, mask_op}};
    };
    TT_ASSIGN_OR_THROW(
        auto results,
        (DispatchOp<1, 2>(std::move(op_builder), input,
                          {.out_dtypes = {output_dtype, mask_dtype},
                           .out_dims_list = {input.sizes(), input.sizes()},
                           .op_param_cache_keys = std::move(param_keys)})));
    return {MakeTensor(std::move(results[0])),
            MakeTensor(std::move(results[1]))};
  }
  if (drop_all) {
    auto op_builder =
        [mask_dtype](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<2>> {
      auto zero_const = MakeConstantLike(input, 0.0);
      auto mask_op = MakeConstantLike(input, false, mask_dtype);
      return {{zero_const, mask_op}};
    };
    TT_ASSIGN_OR_THROW(
        auto results,
        (DispatchOp<1, 2>(std::move(op_builder), input,
                          {.out_dtypes = {output_dtype, mask_dtype},
                           .out_dims_list = {input.sizes(), input.sizes()},
                           .op_param_cache_keys = std::move(param_keys)})));
    return {MakeTensor(std::move(results[0])),
            MakeTensor(std::move(results[1]))};
  }

  if (input.numel() == 0) {
    auto op_builder =
        [mask_dtype](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<2>> {
      auto mask_op = MakeConstantLike(input, false, mask_dtype);
      return {{input, mask_op}};
    };
    TT_ASSIGN_OR_THROW(
        auto results,
        (DispatchOp<1, 2>(std::move(op_builder), input,
                          {.out_dtypes = {output_dtype, mask_dtype},
                           .out_dims_list = {input.sizes(), input.sizes()},
                           .op_param_cache_keys = std::move(param_keys)})));
    return {MakeTensor(std::move(results[0])),
            MakeTensor(std::move(results[1]))};
  }

  const int64_t num_elements = input.numel();
  TT_ASSIGN_OR_THROW(const mlir::ElementType compute_elem_type,
                     InferComputationDtype(output_dtype));
  const mlir::ElementType rand_elem_type = RealComponentOf(compute_elem_type);
  const int64_t bit_width = TorchEquivalentBitwidth(rand_elem_type);

  auto dropout_builder = [p, prob_type,
                          mask_type](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<MlirOpResults<2>> {
    auto& [rng_state, input] = inputs;
    return BuildDropoutTrainShlo(rng_state, input, p, prob_type, mask_type);
  };

  TT_ASSIGN_OR_THROW(auto results,
                     DispatchRngOpGeneral(
                         generator,
                         [&]  // REF_CAPTURE_OK=called synchronously
                         (at::Tensor rng_input_state) {
                           return DispatchOp<2, 2>(
                               dropout_builder, {rng_input_state, input},
                               {.out_dtypes = {output_dtype, mask_dtype},
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
                                 DropoutProbabilityType::kDropProbability,
                                 DropoutMaskType::kBool, std::move(param_keys));
  });
}

std::tuple<at::Tensor, at::Tensor> AtenFusedDropout(
    const at::Tensor& self, double p, c10::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kFusedDropout, param_keys, (self, p, generator), {
    return AtenDropoutCommonImpl(self, p, /*train=*/true, generator,
                                 DropoutProbabilityType::kKeepProbability,
                                 DropoutMaskType::kUInt8,
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

at::Tensor AtenDropoutBackwardCommonImpl(const at::Tensor& grad_output,
                                         const at::Tensor& mask, double scale,
                                         at::ScalarType expected_mask_dtype,
                                         OpParamCacheKeys param_keys) {
  TT_CHECK_THROW(mask.scalar_type() == expected_mask_dtype,
                 error::kInvalidArgument)
      << "expected mask dtype to be " << ToString(expected_mask_dtype)
      << ", got " << ToString(mask.scalar_type());
  TT_CHECK_THROW(at::is_expandable_to(mask.sizes(), grad_output.sizes()),
                 error::kInvalidArgument)
      << "expected mask to be broadcastable to output shape, got "
      << mask.sizes() << " vs " << grad_output.sizes();
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
        return AtenDropoutBackwardCommonImpl(grad_output, mask, scale,
                                             at::kBool, std::move(param_keys));
      });
}

at::Tensor AtenMaskedScale(const at::Tensor& self, const at::Tensor& mask,
                           double scale) {
  TT_KERNEL(OpName::kMaskedScale, param_keys, (self, mask, scale), {
    TT_CHECK_THROW(self.is_floating_point() || self.is_complex(),
                   error::kPythonNotImplementedError)
        << "not implemented for " << ToString(self.scalar_type());
    return AtenDropoutBackwardCommonImpl(self, mask, scale, at::kByte,
                                         std::move(param_keys));
  });
}

}  // namespace torch_tpu
