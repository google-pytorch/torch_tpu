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

#include "torch_tpu/ops/gather/gather_aten_kernels.h"

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
#include "torch_tpu/ops/gather/gather.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

absl::StatusOr<DeviceBufferRef> Gather(const at::Tensor& self, int64_t dim,
                                       const at::Tensor& index,
                                       bool sparse_grad,
                                       OpParamCacheKeys&& param_keys) {
  TT_RET_CHECK(sparse_grad == false, error::kUnimplemented)
      << "sparse_grad is not yet supported";

  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self.dim()));
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  Dimensions output_dims = CopyIntVector(index.sizes());

  if (index.numel() == 0) {
    // Gathering with 0 indexes results in a zero-sized tensor.
    return DeviceBufferList::CreateZeroSize(output_dims, output_dtype);
  }

  auto gather_op_builder =
      [dim, sparse_grad, output_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
        auto& [self, index] = inputs;
        return BuildGatherShlo(self, dim, index, sparse_grad, output_dtype);
      };

  return DispatchOp<2>(std::move(gather_op_builder), {self, index},
                       {.out_dtype = output_dtype,
                        .out_dims = output_dims,
                        .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenGather(const at::Tensor& self, int64_t dim,
                      const at::Tensor& index, bool sparse_grad) {
  TT_KERNEL(OpName::kGather, param_keys, (self, dim, index, sparse_grad), {
    TT_ASSIGN_OR_THROW(auto result, Gather(self, dim, index, sparse_grad,
                                           std::move(param_keys)));
    return MakeTensor(std::move(result));
  });
}

at::Tensor& AtenGatherOut(const at::Tensor& self, int64_t dim,
                          const at::Tensor& index, bool sparse_grad,
                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kGatherOut, param_keys, (self, dim, index, sparse_grad, out), {
        TT_ASSIGN_OR_THROW(auto result, Gather(self, dim, index, sparse_grad,
                                               std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
        return out;
      });
}

}  // namespace torch_tpu
