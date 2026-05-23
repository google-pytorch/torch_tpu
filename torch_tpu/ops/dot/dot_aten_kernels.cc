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

#include "torch_tpu/ops/dot/dot_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/dot/dot.h"
#include "torch_tpu/ops/dot/dot_checks.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/precision_context.h"

namespace torch_tpu {

namespace {

absl::StatusOr<DeviceBufferRef> Dot(const at::Tensor& lhs,
                                    const at::Tensor& rhs,
                                    OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(CheckIsVector(lhs, "first"));
  TT_RETURN_IF_ERROR(CheckIsVector(rhs, "second"));

  TT_ASSIGN_OR_RETURN(auto result_scalar_type,
                      CheckedGetDotOutputType(lhs, rhs));

  const auto current_precision = GetPrecision();
  TT_ASSIGN_OR_RETURN(param_keys,
                      *OpParamCacheKeys::Builder(std::move(param_keys))
                           .SetParam("precision", current_precision));

  // TODO: XLA doesn't support matmuls with i64, so we convert them to f64.
  auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [lhs_op, rhs_op] = inputs;
    return BuildDotShlo(lhs_op, rhs_op, current_precision);
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

at::Tensor AtenDot(const at::Tensor& lhs, const at::Tensor& rhs) {
  TT_KERNEL(OpName::kDot, param_keys, (lhs, rhs), {
    TT_ASSIGN_OR_THROW(auto result, Dot(lhs, rhs, std::move(param_keys)));
    return MakeTensor(std::move(result));
  });
}

}  // namespace torch_tpu
