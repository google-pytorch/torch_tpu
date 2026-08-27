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

#include "torch_tpu/csrc/ops/index_add/index_add_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/index_add/index_add.h"
#include "torch_tpu/csrc/ops/index_utils.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {

namespace {

absl::StatusOr<DeviceBufferRef> IndexAdd(const at::Tensor& self, int64_t dim,
                                         const at::Tensor& index,
                                         const at::Tensor& source,
                                         PromotedScalar& promoted_alpha,
                                         const at::ScalarType& out_scalar_type,
                                         OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(dim,
                      ValidateIndexInputsAndGetDim(self, dim, index, source));

  at::ScalarType promoted_scalar_type =
      c10::promoteTypes(self.scalar_type(), out_scalar_type);
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(promoted_scalar_type));

  TT_ASSIGN_OR_RETURN(const at::Tensor alpha_tensor,
                      promoted_alpha.GetTensor(promoted_scalar_type));

  auto index_add_op_builder =
      [dim, computation_dtype](FixedSizeSpan<mlir::MlirOp, 4> inputs) {
        auto& [self, index, source, alpha_op] = inputs;
        return BuildIndexAddShlo(self, dim, index, source, alpha_op,
                                 computation_dtype);
      };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(out_scalar_type));
  return DispatchOp<4>(std::move(index_add_op_builder),
                       {self, index, source, alpha_tensor},
                       {.out_dtype = output_dtype,
                        .out_dims = CopyIntVector(self.sizes()),
                        .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor& TpuAtenIndexAddOut(const at::Tensor& self, int64_t dim,
                               const at::Tensor& index,
                               const at::Tensor& source,
                               const at::Scalar& alpha, at::Tensor& out) {
  PromotedScalar promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(
      OpName::kIndexAddOut, param_keys,
      (self, dim, index, source, promoted_alpha, out), {
        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));
        TT_ASSIGN_OR_THROW(DeviceBufferRef result_buf,
                           IndexAdd(self, dim, index, source, promoted_alpha,
                                    out.scalar_type(), std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
