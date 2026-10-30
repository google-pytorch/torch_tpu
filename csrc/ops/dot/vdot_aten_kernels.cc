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

#include "csrc/ops/dot/vdot_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/dot/dot.h"
#include "csrc/ops/dot/dot_checks.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "csrc/ops/unary.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildVdotShlo(
    mlir::MlirOp lhs, mlir::MlirOp rhs, mlir::stablehlo::Precision precision) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp conjugated_lhs, BuildConjPhysicalShlo(lhs));
  return BuildDotShlo(conjugated_lhs, rhs, precision);
}

absl::Status ValidateInputs(const at::Tensor& lhs, const at::Tensor& rhs) {
  TT_RETURN_IF_ERROR(ValidateIsVector(lhs, "first"));
  TT_RETURN_IF_ERROR(ValidateIsVector(rhs, "second"));

  TT_RET_CHECK(lhs.size(0) == rhs.size(0), error::kInvalidArgument)
      << "expected inputs to have the same shape, got " << ToString(lhs.sizes())
      << " vs " << ToString(rhs.sizes());

  return absl::OkStatus();
}

absl::StatusOr<DeviceBufferRef> Vdot(const at::Tensor& lhs,
                                     const at::Tensor& rhs,
                                     OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateInputs(lhs, rhs));

  TT_ASSIGN_OR_RETURN(auto result_scalar_type,
                      ValidateAndGetDotOutputType(lhs, rhs));

  const auto current_precision = GetAndAddPrecisionTo(param_keys);

  auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [lhs_op, rhs_op] = inputs;
    return BuildVdotShlo(lhs_op, rhs_op, current_precision);
  };

  TT_ASSIGN_OR_RETURN(
      auto result,
      DispatchOp<2>(std::move(op_builder), {lhs, rhs},
                    {.out_dtype = result_scalar_type,
                     .out_dims = {},
                     .op_param_cache_keys = std::move(param_keys)}));
  return result;
}

}  // namespace

at::Tensor AtenVdot(const at::Tensor& lhs, const at::Tensor& rhs) {
  TT_KERNEL(OpName::kVdot, param_keys, (lhs, rhs), {
    TT_ASSIGN_OR_THROW(auto result, Vdot(lhs, rhs, std::move(param_keys)));
    return MakeTensor(std::move(result));
  });
}

}  // namespace torch_tpu
