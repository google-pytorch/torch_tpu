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

#include "torch_tpu/ops/uniform/uniform_aten_kernels.h"

#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "absl/status/status.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/rng_utils.h"
#include "torch_tpu/ops/uniform/uniform.h"
#include "torch_tpu/ops/view/view_aten_kernels.h"

namespace torch_tpu {

namespace {

NAryMlirOpBuilder<1, 2> GetUniformFunctional(Dimensions dims,
                                             mlir::ElementType output_dtype,
                                             double from, double to) {
  return [dims, output_dtype, from, to](mlir::MlirOp rng_input_state) {
    return BuildUniformShlo(rng_input_state, from, to, dims, output_dtype);
  };
}

absl::Status CheckUniformPreconditions(const at::Tensor& self) {
  TT_RET_CHECK(IsFloatingPoint(self) || IsComplex(self),
               error::kInvalidArgument)
      << "expected the input dtype to be floating point or complex, got "
      << ToString(self.scalar_type());

  return absl::OkStatus();
}

}  // namespace

at::Tensor& AtenUniform_(at::Tensor& self, double from, double to,
                         std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kUniform_, param_keys, (self, from, to, generator), {
    if (self.numel() == 0) {
      return self;
    }
    TT_THROW_IF_ERROR(CheckUniformPreconditions(self));
    at::Tensor self_real = self.is_complex() ? AtenViewAsReal(self) : self;

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self_real.scalar_type()));
    auto dims = CopyIntVector(self_real.sizes());
    TT_THROW_IF_ERROR(
        DispatchRngOp(self_real, generator, [&](at::Tensor rng_input_state) {
          return DispatchOp<1, 2>(
              GetUniformFunctional(dims, output_dtype, from, to),
              {rng_input_state},
              {.out_dtypes = {mlir::ElementType::UI64, output_dtype},
               .out_dims_list = {{2}, self_real.sizes()},
               .op_param_cache_keys = std::move(param_keys),
               .split_mode = OpSplitMode::kSplitAfter});
        }));
    return self;
  });
}

}  // namespace torch_tpu
