// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/ops/multinomial/multinomial_aten_kernels.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/util/Optional.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_gen_impl.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/min_max/min_max.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/rng_utils.h"
#include "csrc/ops/topk/topk.h"
#include "csrc/ops/uniform/uniform.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

namespace stablehlo = mlir::stablehlo;

// Generates Gumbel noise matching `target_op`'s type and shape using the
// provided RNG input state.
absl::StatusOr<mlir::MlirOp> BuildGumbelNoise(mlir::MlirOp rng_input_state,
                                              mlir::MlirOp target_op) {
  // Gumbel noise is -log(-log(U)), where U ~ Uniform(0, 1).
  // Use the smallest positive normal float as the lower bound so U is strictly
  // in (0, 1). This prevents U == 0 from producing log(0) = -inf (which would
  // tie with zero-weight categories where log(weight) = -inf).
  const mlir::RankedTensorType noise_type = GetTensorTypeOrDie(target_op);
  const auto float_type =
      mlir::cast<mlir::FloatType>(noise_type.getElementType());
  const double from =
      llvm::APFloat::getSmallestNormalized(float_type.getFloatSemantics())
          .convertToDouble();
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp uniform,
      BuildUniformShlo(rng_input_state, from, 1.0, noise_type, target_op));

  mlir::MlirOp log_uniform = stablehlo::Log(uniform);
  mlir::MlirOp neg_log_uniform = stablehlo::Neg(log_uniform);
  mlir::MlirOp log_neg_log_uniform = stablehlo::Log(neg_log_uniform);
  mlir::MlirOp gumbel = stablehlo::Neg(log_neg_log_uniform);
  return gumbel;
}

// This uses the Gumbel-Max trick to sample from the multinomial distribution.
// This is equivalent to drawing samples from the distribution with
// probabilities proportional to the input weights.
// See https://en.wikipedia.org/wiki/Gumbel_distribution.
absl::StatusOr<mlir::MlirOp> BuildMultinomialShlo(mlir::MlirOp input_op,
                                                  mlir::MlirOp rng_input_state,
                                                  int64_t num_samples,
                                                  bool replacement) {
  const mlir::RankedTensorType orig_input_type = GetTensorTypeOrDie(input_op);
  if (orig_input_type.getElementType().getIntOrFloatBitWidth() < 32) {
    // Upcast low-precision floats (bf16, f16) to f32 so Gumbel noise and
    // log-probabilities have 23-bit mantissa precision rather than 7/10-bit
    // quantization.
    TT_ASSIGN_OR_RETURN(input_op,
                        CastIfNeeded(input_op, mlir::ElementType::F32));
  }
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  auto input_shape = input_type.getShape();
  const int64_t rank = input_type.getRank();
  const int64_t feature_dim = rank - 1;

  mlir::MlirOp logits = stablehlo::Log(input_op);

  if (replacement) {
    // Sample with replacement.
    Dimensions expanded_shape = CopyIntVector(input_shape);
    expanded_shape.insert(expanded_shape.begin() + feature_dim, 1);
    mlir::MlirOp expanded_logits = stablehlo::Reshape(logits, expanded_shape);

    Dimensions broadcast_shape = expanded_shape;
    broadcast_shape[feature_dim] = num_samples;
    TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_logits,
                        BroadcastIfNeeded(expanded_logits, broadcast_shape));

    TT_ASSIGN_OR_RETURN(mlir::MlirOp gumbel_noise,
                        BuildGumbelNoise(rng_input_state, broadcasted_logits));

    mlir::MlirOp perturbed_logits =
        stablehlo::Add(broadcasted_logits, gumbel_noise);

    ReductionMode mode = ReductionMode::kDropDims;
    TT_ASSIGN_OR_RETURN(mlir::MlirOp result,
                        BuildMinMaxShlo(feature_dim + 1, MinMaxOp::kMax, mode,
                                        perturbed_logits))
        .indices;
    return result;
  } else {
    // Sample without replacement.
    TT_ASSIGN_OR_RETURN(mlir::MlirOp gumbel_noise,
                        BuildGumbelNoise(rng_input_state, logits));

    mlir::MlirOp perturbed_logits = stablehlo::Add(logits, gumbel_noise);

    TT_ASSIGN_OR_RETURN(TopKOutputs topk_results,
                        BuildTopKShlo(perturbed_logits, num_samples,
                                      feature_dim, TopKMode::kLargest));
    return topk_results.indices;
  }
}

Dimensions GetOutputDimensions(const at::Tensor& self, int64_t num_samples) {
  return self.dim() == 1 ? Dimensions{num_samples}
                         : Dimensions{self.size(0), num_samples};
}

absl::Status ValidateInputs(const at::Tensor& self, int64_t num_samples,
                            bool replacement) {
  TT_RET_CHECK(IsFloatingPoint(self), error::kInvalidArgument)
      << "expected the input dtype to be floating-point, got "
      << ToString(self.scalar_type());
  TT_RET_CHECK(self.dim() >= 1 && self.dim() <= 2, error::kInvalidArgument)
      << "expected the input to have either 1 or 2 dimensions, got "
      << self.dim() << " of shape " << ToString(self.sizes());

  TT_RET_CHECK(num_samples > 0, error::kInvalidArgument)
      << "expected the number of samples to be > 0, got " << num_samples;

  TT_RET_CHECK(replacement || num_samples <= self.size(-1),
               error::kInvalidArgument)
      << "expected the number of samples to be <= " << self.size(-1)
      << " (population size) when replacement is disabled, got " << num_samples;

  TT_RET_CHECK(!IsLowPrecisionFloat(self), error::kPythonNotImplementedError)
      << "not implemented for " << ToString(self.scalar_type());

  return absl::OkStatus();
}

absl::StatusOr<RngUsage> GetRngUsage(const at::Tensor& self,
                                     int64_t num_samples, bool replacement) {
  const int64_t num_rng_elements =
      replacement ? (self.numel() * num_samples) : self.numel();
  TT_ASSIGN_OR_RETURN(mlir::ElementType in_elem_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  const int64_t bit_width =
      std::max<int64_t>(32, TorchEquivalentBitwidth(in_elem_type));
  return RngUsage{num_rng_elements, bit_width};
}

NAryMlirOpBuilder<2, 1> GetMultinomialFunctional(int64_t num_samples,
                                                 bool replacement) {
  return [num_samples, replacement](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    auto [input_op, rng_input_state] = inputs;
    return BuildMultinomialShlo(input_op, rng_input_state, num_samples,
                                replacement);
  };
}

}  // namespace

at::Tensor AtenMultinomial(const at::Tensor& self, int64_t num_samples,
                           bool replacement,
                           c10::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kMultinomial, param_keys,
            (self, num_samples, replacement, generator), {
              TT_THROW_IF_ERROR(ValidateInputs(self, num_samples, replacement));
              TT_ASSIGN_OR_THROW(const RngUsage usage,
                                 GetRngUsage(self, num_samples, replacement));

              auto* gen = at::get_generator_or_default<DeviceGeneratorImplBase>(
                  generator, GetDefaultDeviceGenerator());
              // NOLINTNEXTLINE(build/c++11) - std::mutex required by PyTorch
              std::scoped_lock<std::mutex> lock(gen->mutex_);
              const at::Tensor rng_input_state = gen->DeviceStateTensor();

              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result_buf,
                  (DispatchOp<2, 1>(
                      GetMultinomialFunctional(num_samples, replacement),
                      {self, rng_input_state},
                      {.out_dtype = mlir::ElementType::I64,
                       .out_dims = GetOutputDimensions(self, num_samples),
                       .op_param_cache_keys = std::move(param_keys)})));

              TT_THROW_IF_ERROR(gen->AdvanceDeviceStateTensor(
                  usage.num_elements, usage.bit_width));
              return MakeTensor(std::move(result_buf));
            });
}

at::Tensor& AtenMultinomialOut(const at::Tensor& self, int64_t num_samples,
                               bool replacement,
                               c10::optional<at::Generator> generator,
                               at::Tensor& out) {
  TT_KERNEL(
      OpName::kMultinomialOut, param_keys,
      (self, num_samples, replacement, generator, out), {
        TT_THROW_IF_ERROR(ValidateInputs(self, num_samples, replacement));
        TT_THROW_IF_ERROR(ValidateOutDtype(out, at::ScalarType::Long,
                                           /*allow_out_dtype_cast=*/false));
        TT_ASSIGN_OR_THROW(const RngUsage usage,
                           GetRngUsage(self, num_samples, replacement));

        auto* gen = at::get_generator_or_default<DeviceGeneratorImplBase>(
            generator, GetDefaultDeviceGenerator());
        // NOLINTNEXTLINE(build/c++11) - std::mutex required by PyTorch
        std::scoped_lock<std::mutex> lock(gen->mutex_);
        const at::Tensor rng_input_state = gen->DeviceStateTensor();

        TT_THROW_IF_ERROR((DispatchOpOut<2, 1>(
            GetMultinomialFunctional(num_samples, replacement),
            {self, rng_input_state}, out,
            {.out_dtype = mlir::ElementType::I64,
             .out_dims = GetOutputDimensions(self, num_samples),
             .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            gen->AdvanceDeviceStateTensor(usage.num_elements, usage.bit_width));
        return out;
      });
}

}  // namespace torch_tpu
