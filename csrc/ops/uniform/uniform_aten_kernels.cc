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

#include "csrc/ops/uniform/uniform_aten_kernels.h"

#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/rng_utils.h"
#include "csrc/ops/uniform/uniform.h"
#include "csrc/ops/view/view_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch/headeronly/util/BFloat16.h"
#include "torch/headeronly/util/Half.h"

namespace torch_tpu {

namespace {

NAryMlirOpBuilder<1, 1> GetUniformFunctional(Dimensions dims,
                                             mlir::ElementType output_dtype,
                                             double from, double to) {
  return [dims, output_dtype, from, to](mlir::MlirOp rng_input_state) {
    return BuildUniformShlo(rng_input_state, from, to, dims, output_dtype);
  };
}

struct FloatBounds {
  double min;
  double max;
};

absl::StatusOr<FloatBounds> GetFloatBounds(const at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Half:
      return FloatBounds{
          .min = static_cast<double>(std::numeric_limits<c10::Half>::lowest()),
          .max = static_cast<double>(std::numeric_limits<c10::Half>::max()),
      };
    case at::ScalarType::BFloat16:
      return FloatBounds{
          .min =
              static_cast<double>(std::numeric_limits<c10::BFloat16>::lowest()),
          .max = static_cast<double>(std::numeric_limits<c10::BFloat16>::max()),
      };
    case at::ScalarType::Float:
      return FloatBounds{
          .min = static_cast<double>(std::numeric_limits<float>::lowest()),
          .max = static_cast<double>(std::numeric_limits<float>::max()),
      };
    case at::ScalarType::Double:
      return FloatBounds{
          .min = std::numeric_limits<double>::lowest(),
          .max = std::numeric_limits<double>::max(),
      };
    default:
      return TT_ERROR(  // ERROR_COV_INFEASIBLE=Caller validates standard
                        // floating-point dtypes before querying bounds.
                 error::kInvalidArgument)
             << "expected the input dtype to be floating point or complex, got "
             << ToString(dtype);
  }
}

absl::Status ValidateUniformPreconditions(const at::Tensor& self) {
  TT_RET_CHECK(IsFloatingPoint(self) || IsComplex(self),
               error::kInvalidArgument)
      << "expected the input dtype to be floating point or complex, got "
      << ToString(self.scalar_type());
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(self));

  return absl::OkStatus();
}

absl::Status ValidateUniformBounds(const at::ScalarType real_dtype,
                                   const double from, const double to) {
  TT_ASSIGN_OR_RETURN(const FloatBounds bounds, GetFloatBounds(real_dtype));
  TT_RET_CHECK(from >= bounds.min && from <= bounds.max,
               error::kInvalidArgument)
      << "expected from to be in [" << ToString(bounds.min) << ", "
      << ToString(bounds.max) << "] for " << ToString(real_dtype) << ", got "
      << ToString(from);
  TT_RET_CHECK(to >= bounds.min && to <= bounds.max, error::kInvalidArgument)
      << "expected to to be in [" << ToString(bounds.min) << ", "
      << ToString(bounds.max) << "] for " << ToString(real_dtype) << ", got "
      << ToString(to);
  TT_RET_CHECK(from <= to, error::kInvalidArgument)
      << "expected from <= to, got from=" << ToString(from)
      << " and to=" << ToString(to);
  TT_RET_CHECK((to - from) <= bounds.max, error::kInvalidArgument)
      << "expected to - from <= " << ToString(bounds.max) << " for "
      << ToString(real_dtype) << ", got to=" << ToString(to)
      << " and from=" << ToString(from);
  return absl::OkStatus();
}

}  // namespace

at::Tensor& AtenUniform_(at::Tensor& self, double from, double to,
                         std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kUniform_, param_keys, (self, from, to, generator), {
    // Match upstream PyTorch/CUDA, which skips bounds validation on empty
    // complex tensors while still validating bounds on non-complex empty ones.
    if (self.is_complex() && self.numel() == 0) {
      return self;
    }
    TT_THROW_IF_ERROR(ValidateUniformPreconditions(self));
    const at::ScalarType real_dtype = c10::toRealValueType(self.scalar_type());
    TT_THROW_IF_ERROR(ValidateUniformBounds(real_dtype, from, to));
    if (self.numel() == 0) {
      return self;
    }
    at::Tensor self_real = self.is_complex() ? AtenViewAsReal(self) : self;

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self_real.scalar_type()));
    auto dims = CopyIntVector(self_real.sizes());

    TT_THROW_IF_ERROR(DispatchRngOp(
        self_real, generator,
        [&](at::Tensor rng_input_state)  // REF_CAPTURE_OK=called synchronously
        -> absl::StatusOr<std::vector<DeviceBufferRef>> {
          TT_ASSIGN_OR_RETURN(
              auto buf, (DispatchOp<1, 1>(
                            GetUniformFunctional(dims, output_dtype, from, to),
                            {rng_input_state},
                            {.out_dtype = output_dtype,
                             .out_dims = self_real.sizes(),
                             .op_param_cache_keys = std::move(param_keys)})));
          return std::vector<DeviceBufferRef>{std::move(buf)};
        }));
    return self;
  });
}

}  // namespace torch_tpu
