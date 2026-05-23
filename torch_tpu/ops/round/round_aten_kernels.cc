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

#include "torch_tpu/ops/round/round_aten_kernels.h"

#include <cstdint>
#include <functional>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/round/round.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {
namespace {

MlirUnaryOpBuilder GetRoundFunctional(int64_t decimals) {
  return std::bind(&BuildRoundShlo, std::placeholders::_1, decimals);
}

// Check that the provided dtype is appropriate for rounding.
absl::Status CheckNumericDtype(at::ScalarType scalar_type) {
  TT_ASSIGN_OR_RETURN(const auto dtype,
                      ConvertTo<mlir::ElementType>(scalar_type));
  TT_RET_CHECK(c10::isFloatingType(scalar_type) ||
                   c10::isIntegralType(scalar_type, /*include_bool=*/false),
               error::kInvalidArgument)
      << "dtype " << ToString(dtype) << " is not supported";
  return absl::OkStatus();
}

}  // namespace

at::Tensor& AtenRoundOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kRoundOut, _, (self, out), {
    TT_THROW_IF_ERROR(CheckNumericDtype(self.scalar_type()));
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, GetRoundFunctional(/*decimals=*/0),
                   {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenRoundDecimalsOut(const at::Tensor& self, int64_t decimals,
                                 at::Tensor& out) {
  TT_KERNEL(OpName::kRoundDecimalsOut, param_keys, (self, decimals, out), {
    TT_THROW_IF_ERROR(CheckNumericDtype(self.scalar_type()));
    TT_CHECK_THROW(!IsInteger(self), error::kInvalidArgument)
        << "expected the input dtype not to be integer when the decimals "
           "argument is specified ("
        << decimals << "), got " << ToString(self.scalar_type());
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, GetRoundFunctional(decimals),
                   {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

}  // namespace torch_tpu
