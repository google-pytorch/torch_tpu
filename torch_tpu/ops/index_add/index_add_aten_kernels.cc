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

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/index_add/index_add.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

absl::StatusOr<int64_t> ValidateIndexAddInputsAndGetDim(
    const at::Tensor& self, int64_t dim, const at::Tensor& index,
    const at::Tensor& source) {
  TT_RET_CHECK(index.dim() == 1, error::kInvalidArgument)
      << "index must be 1D, got shape " << index.sizes();
  if (self.dim() == 0) {
    TT_RET_CHECK(dim == 0, error::kInvalidArgument)
        << "dim must be 0 for scalar input, got " << dim;
    TT_RET_CHECK(source.dim() == 0, error::kInvalidArgument)
        << "source shape must match self shape, excluding the specified "
           "dimension, got source shape "
        << source.sizes() << " and self shape " << self.sizes();
    TT_RET_CHECK(index.size(0) == 1, error::kInvalidArgument)
        << "index must be 1D of size 1 for scalar input, got shape "
        << index.sizes();
    return dim;
  }

  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self.dim()));

  TT_RET_CHECK(source.dim() == self.dim(), error::kInvalidArgument)
      << "self and source must have the same number of dimensions, got "
      << self.dim() << " and " << source.dim();
  TT_RET_CHECK(source.size(dim) == index.size(0), error::kInvalidArgument)
      << "source must have the same number of elements as the index along "
         "dimension "
      << dim << ", got " << source.size(dim) << " and " << index.size(0);
  for (int i = 0; i < self.dim(); ++i) {
    if (i != dim) {
      TT_RET_CHECK(self.size(i) == source.size(i), error::kInvalidArgument)
          << "self and source must have the same size along dimension " << i
          << ", got " << self.size(i) << " and " << source.size(i);
    }
  }
  return dim;
}

absl::StatusOr<DeviceBufferRef> IndexAdd(const at::Tensor& self, int64_t dim,
                                         const at::Tensor& index,
                                         const at::Tensor& source,
                                         const at::Scalar& alpha,
                                         const at::ScalarType& out_scalar_type,
                                         OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(
      dim, ValidateIndexAddInputsAndGetDim(self, dim, index, source));

  at::ScalarType promoted_scalar_type =
      c10::promoteTypes(self.scalar_type(), out_scalar_type);
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(promoted_scalar_type));

  auto index_add_op_builder =
      [dim, alpha, computation_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
        auto& [self, index, source] = inputs;
        mlir::MlirOp alpha_op = MakeScalarConstant(
            self.getBuilder(), alpha.toDouble(), computation_dtype);
        return BuildIndexAddShlo(self, dim, index, source, alpha_op,
                                 computation_dtype);
      };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(out_scalar_type));
  return DispatchOp<3>(std::move(index_add_op_builder), {self, index, source},
                       {.out_dtype = output_dtype,
                        .out_dims = CopyIntVector(self.sizes()),
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
        TT_ASSIGN_OR_THROW(DeviceBufferRef result_buf,
                           IndexAdd(self, dim, index, source, alpha,
                                    out.scalar_type(), std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
