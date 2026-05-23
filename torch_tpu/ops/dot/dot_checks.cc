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

#include "torch_tpu/ops/dot/dot_checks.h"

#include <string_view>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/result_type.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"

namespace torch_tpu {

absl::Status CheckIsVector(const at::Tensor& tensor,
                           std::string_view arg_name) {
  TT_RET_CHECK(tensor.dim() == 1, error::kInvalidArgument)
      << "expected the " << arg_name << " argument to be a 1D tensor, got "
      << tensor.dim() << "D of shape " << ToString(tensor.sizes());
  return absl::OkStatus();
}

absl::StatusOr<mlir::ElementType> CheckedGetDotOutputType(
    const at::Tensor& lhs, const at::Tensor& rhs) {
  at::ScalarType output_scalar_type = at::result_type(lhs, rhs);

  TT_RET_CHECK(!IsBool(output_scalar_type), error::kInvalidArgument)
      << "the input dtypes cannot be bool";

  TT_ASSIGN_OR_RETURN(auto output_type,
                      ConvertTo<mlir::ElementType>(at::result_type(lhs, rhs)));

  return output_type;
}

}  // namespace torch_tpu
