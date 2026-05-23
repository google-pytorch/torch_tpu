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

#include "torch_tpu/ops/masked_scatter/masked_scatter_aten_kernels.h"

#include <utility>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/broadcast_tensors.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/copy_from/copy_from_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/masked_scatter/masked_scatter.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

at::Tensor& AtenMaskedScatter_(at::Tensor& self, const at::Tensor& mask,
                               const at::Tensor& source) {
  TT_KERNEL(OpName::kMaskedScatter_, _, (self, mask, source), {
    TT_ASSIGN_OR_THROW(const auto mask_dtype,
                       ConvertTo<mlir::ElementType>(mask.scalar_type()));
    TT_CHECK_THROW(mask.scalar_type() == at::kBool, error::kInvalidArgument)
        << "expected Boolean tensor for mask, got " << ToString(mask_dtype);

    TT_ASSIGN_OR_THROW(const auto self_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(const auto source_dtype,
                       ConvertTo<mlir::ElementType>(source.scalar_type()));
    TT_CHECK_THROW(self_dtype == source_dtype, error::kInvalidArgument)
        << "expected same dtype for self and source,"
        << " got self dtype " << ToString(self_dtype) << " and source dtype "
        << ToString(source_dtype);

    auto broadcasted = at::broadcast_tensors({mask, self});
    at::Tensor mask_broadcasted = broadcasted[0];
    at::Tensor self_broadcasted = broadcasted[1];

    TT_ASSIGN_OR_THROW(auto param_keys, *OpParamCacheKeysBuilder());
    Dimensions out_dims = CopyIntVector(self.sizes());

    auto shlo_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      // inputs[0]: self, inputs[1]: mask, inputs[2]: source
      return BuildMaskedScatterShlo(inputs[0], inputs[1], inputs[2]);
    };

    TT_ASSIGN_OR_THROW(
        auto result,
        (DispatchOp<3>(std::move(shlo_builder),
                       {self_broadcasted, mask_broadcasted, source},
                       {.out_dtype = self_dtype,
                        .out_dims = out_dims,
                        .op_param_cache_keys = std::move(param_keys)})));

    AtenCopyFrom(MakeTensor(result), self, /*non_blocking=*/true);
    return self;
  });
}

}  // namespace torch_tpu
