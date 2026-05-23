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

#include "torch_tpu/ops/index_select/index_select_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/index_select/index_select.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

absl::StatusOr<DeviceBufferRef> IndexSelect(const at::Tensor& self, int64_t dim,
                                            const at::Tensor& index,
                                            OpParamCacheKeys param_keys) {
  TT_RET_CHECK(index.dim() == 1, error::kInvalidArgument)
      << "index must be 1D, got shape " << index.sizes();

  Dimensions output_dims;
  if (self.dim() == 0) {
    TT_RET_CHECK(dim == 0, error::kInvalidArgument)
        << "dim must be 0 for scalar input, got " << dim;
    TT_RET_CHECK(index.size(0) == 1, error::kInvalidArgument)
        << "index must be 1D of size 1 for scalar input, got shape "
        << index.sizes();
    output_dims = {};
  } else {
    TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self.dim()));
    output_dims = CopyIntVector(self.sizes());
    output_dims[dim] = index.size(0);
  }

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=there is no way to pass self
                        // tensor with unsupported dtype to this point. It
                        // errors out earlier.
      const auto computation_dtype,
      ConvertTo<mlir::ElementType>(self.scalar_type()));
  auto index_select_op_builder = [dim](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [self, index] = inputs;
    return BuildIndexSelectShlo(self, dim, index);
  };

  return DispatchOp<2>(std::move(index_select_op_builder), {self, index},
                       {.out_dtype = computation_dtype,
                        .out_dims = output_dims,
                        .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor TpuAtenIndexSelect(const at::Tensor& self, int64_t dim,
                              const at::Tensor& index) {
  TT_KERNEL(OpName::kIndexSelect, param_keys, (self, dim, index), {
    TT_ASSIGN_OR_THROW(DeviceBufferRef result_buf,
                       IndexSelect(self, dim, index, std::move(param_keys)));
    return MakeTensor(std::move(result_buf));
  });
}

}  // namespace torch_tpu
