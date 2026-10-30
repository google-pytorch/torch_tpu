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

#include "csrc/ops/dynamic/set_dimension_logical_size/set_dimension_logical_size.h"

#include <cstdint>
#include <utility>

#include "ATen/WrapDimUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

at::Tensor SetDimensionLogicalSize(const at::Tensor& input, int64_t dim,
                                   const at::Tensor& size) {
  TT_KERNEL(OpName::kSetDimensionLogicalSize, param_keys, (input, dim, size), {
    TT_ASSIGN_OR_THROW(dim, SafeWrapDim(dim, input.dim()));
    TT_CHECK_THROW(size.dim() == 0, error::kInvalidArgument)
        << "expected a 0-dimensional tensor for size, got " << size.dim()
        << "-dimensional tensor";
    TT_CHECK_THROW(size.scalar_type() == at::kInt, error::kInvalidArgument)
        << "expected an " << ToString(at::kInt) << " tensor for size, got "
        << ToString(size.scalar_type());
    TT_ASSIGN_OR_THROW(const mlir::ElementType mlir_dtype,
                       ConvertTo<mlir::ElementType>(input.scalar_type()));
    const auto out_dims = CopyIntVector(input.sizes());

    auto builder = [dim](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      return mlir::stablehlo::SetDimensionSize(inputs[0], inputs[1], dim);
    };

    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<2>(std::move(builder), {input, size},
                      {.out_dtype = mlir_dtype,
                       .out_dims = out_dims,
                       .op_param_cache_keys = std::move(param_keys)}));
    return MakeTensor(std::move(result_buf));
  });
}

}  // namespace torch_tpu
