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

#include "csrc/ops/cummin/cummin_aten_kernels.h"

#include <cstdint>
#include <limits>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/cummin/cummin.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

void AtenCumminHelper(const at::Tensor& self, at::Tensor& values,
                      at::Tensor& indices, int64_t dim) {
  TT_KERNEL(OpName::kCumminHelper, param_keys, (self, values, indices, dim), {
    if (self.dim() == 0) {
      values.copy_(self);
      indices.fill_(0);
      return;
    }

    TT_CHECK_THROW(!self.is_complex(), error::kInvalidArgument)
        << "expected supported element type, got " << self.scalar_type();

    TT_ASSIGN_OR_THROW(const int64_t normalized_dim,
                       SafeWrapDim(dim, self.dim()));

    const int64_t current_dim_size = self.sizes()[normalized_dim];
    TT_CHECK_THROW(current_dim_size <= std::numeric_limits<int32_t>::max(),
                   error::kPythonNotImplementedError)
        << "expected dimension size to be less than or equal to "
        << std::numeric_limits<int32_t>::max() << ", got " << current_dim_size;

    Dimensions output_dims = CopyIntVector(self.sizes());

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));

    auto op_builder =
        [normalized_dim](
            mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<2>> {
      TT_ASSIGN_OR_RETURN(auto cummin_outputs,
                          BuildCumminShlo(normalized_dim, input));
      return MlirOpResults<2>{cummin_outputs.values, cummin_outputs.indices};
    };

    TT_THROW_IF_ERROR((DispatchOpOut<1, 2>(
        std::move(op_builder), self, {values, indices},
        {
            .out_dtypes = {output_dtype, mlir::ElementType::I64},
            .out_dims_list = {output_dims, output_dims},
            .op_param_cache_keys = std::move(param_keys),
        })));
  });
}

}  // namespace torch_tpu
