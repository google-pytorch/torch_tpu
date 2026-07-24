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

#include "torch_tpu/ops/multinomial/multinomial_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/native/Resize.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/util/Optional.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/multinomial/multinomial.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {
namespace {

Dimensions GetOutputDimensions(const at::Tensor& self, int64_t num_samples) {
  return self.dim() == 1 ? Dimensions{num_samples}
                         : Dimensions{self.size(0), num_samples};
}

absl::Status CheckInputs(const at::Tensor& self, int64_t num_samples,
                         bool replacement,
                         c10::optional<at::Generator> generator) {
  TT_RET_CHECK(IsFloatingPoint(self), error::kInvalidArgument)
      << "expected the input dtype to be floating-point, got "
      << ToString(self.scalar_type());
  TT_RET_CHECK(self.dim() >= 1 && self.dim() <= 2, error::kInvalidArgument)
      << "expected the input to have either 1 or 2 dimensions, got "
      << self.dim() << " of shape " << ToString(self.sizes());

  TT_RET_CHECK(num_samples > 0, error::kInvalidArgument)
      << "expected the number of samples to be > 0, got " << num_samples;

  TT_RET_CHECK(!generator.has_value(), error::kPythonNotImplementedError)
      << "generator is not yet supported";

  TT_RET_CHECK(replacement || num_samples <= self.size(-1),
               error::kInvalidArgument)
      << "expected the number of samples to be <= " << self.size(-1)
      << " (population size) when replacement is disabled, got " << num_samples;

  return absl::OkStatus();
}

absl::StatusOr<DeviceBufferRefArray<1>> Multinomial(
    const at::Tensor& self, int64_t num_samples, bool replacement,
    c10::optional<at::Generator> generator, OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(CheckInputs(self, num_samples, replacement, generator));

  auto op_builder = [num_samples, replacement](
                        mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
    return BuildMultinomialShlo(input, num_samples, replacement);
  };

  TT_ASSIGN_OR_RETURN(
      DeviceBufferRefArray<1> result_buf,
      DispatchOp<1>(std::move(op_builder), self,
                    {.out_dtype = mlir::ElementType::I64,
                     .out_dims = GetOutputDimensions(self, num_samples),
                     .op_param_cache_keys = std::move(param_keys),
                     .split_mode = OpSplitMode::kSplitAfter}));
  return result_buf;
}

}  // namespace

at::Tensor AtenMultinomial(const at::Tensor& self, int64_t num_samples,
                           bool replacement,
                           c10::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kMultinomial, param_keys,
            (self, num_samples, replacement, generator), {
              TT_ASSIGN_OR_THROW(DeviceBufferRefArray<1> result_buf,
                                 Multinomial(self, num_samples, replacement,
                                             generator, std::move(param_keys)));
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
        TT_ASSIGN_OR_THROW(DeviceBufferRefArray<1> result_buf,
                           Multinomial(self, num_samples, replacement,
                                       generator, std::move(param_keys)));

        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(out, result_buf.dimensions()));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
