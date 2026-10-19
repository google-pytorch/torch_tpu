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
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_gen_impl.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/multinomial/multinomial.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/rng_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

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
              return MakeTensor(result_buf);
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
