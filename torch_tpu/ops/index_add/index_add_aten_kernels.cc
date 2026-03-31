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

#include "torch_tpu/ops/index_add/index_add_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/Resize.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/index_add/index_add.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<DeviceBufferRef> IndexAdd(OpName op_name, const at::Tensor& self,
                                         int64_t dim, const at::Tensor& index,
                                         const at::Tensor& source,
                                         const at::Tensor& alpha,
                                         at::Tensor& out,
                                         OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self.dim()));

  at::ScalarType promoted_scalar_type =
      c10::promoteTypes(self.scalar_type(), source.scalar_type());
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(promoted_scalar_type));

  auto index_add_op_builder =
      [dim, computation_dtype](FixedSizeSpan<mlir::MlirOp, 4> inputs) {
        auto& [self, index, source, alpha] = inputs;
        return BuildIndexAddShlo(self, dim, index, source, alpha,
                                 computation_dtype);
      };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));
  absl::Span<const int64_t> output_dims = out.sizes();

  return DispatchOp<4>(op_name, std::move(index_add_op_builder),
                       {self, index, source, alpha},
                       {.out_dtype = output_dtype,
                        .out_dims = output_dims,
                        .computation_dtype = computation_dtype,
                        .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor& TpuAtenIndexAddOut(const at::Tensor& self, int64_t dim,
                               const at::Tensor& index,
                               const at::Tensor& source,
                               const at::Scalar& alpha, at::Tensor& out) {
  TT_KERNEL(
      OpName::kIndexAddOut, param_keys, (self, dim, index, source, alpha, out),
      {
        TT_ASSIGN_OR_THROW(at::Tensor alpha_tensor,
                           MakeTensor(alpha, self.scalar_type()));
        TT_ASSIGN_OR_THROW(
            DeviceBufferRef result_buf,
            IndexAdd(OpName::kIndexAddOut, self, dim, index, source,
                     alpha_tensor, out, std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
