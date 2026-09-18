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

#include "csrc/ops/masked_scatter/masked_scatter_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/masked_scatter/masked_scatter.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

absl::Status ValidateInferredOutputShapeMatchesSelf(const at::Tensor& self,
                                                    const at::Tensor& mask) {
  const absl::Span<const int64_t> self_dims = self.sizes();

  TT_ASSIGN_OR_RETURN(const Dimensions out_dims,
                      InferSize(self_dims, mask.sizes()));

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch dispatcher catches shape
                 // mismatch first.
      out_dims == self_dims, error::kInvalidArgument)
      << "expected the output shape (broadcast of self and mask) to match "
      << ToString(self_dims) << " (shape of self), got " << ToString(out_dims);

  return absl::OkStatus();
}

}  // namespace

at::Tensor AtenMaskedScatter(const at::Tensor& self, const at::Tensor& mask,
                             const at::Tensor& source) {
  TT_KERNEL(OpName::kMaskedScatter, param_keys, (self, mask, source), {
    TT_ASSIGN_OR_THROW(const auto mask_dtype,
                       ConvertTo<mlir::ElementType>(mask.scalar_type()));
    TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch catches mask dtype first.
        mask.scalar_type() == at::kBool, error::kInvalidArgument)
        << "expected Boolean tensor for mask, got " << ToString(mask_dtype);

    TT_ASSIGN_OR_THROW(const auto self_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(const auto source_dtype,
                       ConvertTo<mlir::ElementType>(source.scalar_type()));
    TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch catches dtype mismatch
                     // first.
        self_dtype == source_dtype, error::kInvalidArgument)
        << "expected same dtype for self and source,"
        << " got self dtype " << ToString(self_dtype) << " and source dtype "
        << ToString(source_dtype);

    TT_ASSIGN_OR_THROW(const Dimensions out_dims,
                       InferSize(self.sizes(), mask.sizes()));

    auto shlo_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      // inputs[0]: self, inputs[1]: mask, inputs[2]: source
      return BuildMaskedScatterShlo(inputs[0], inputs[1], inputs[2]);
    };

    TT_ASSIGN_OR_THROW(
        auto result,
        DispatchOp<3>(std::move(shlo_builder), {self, mask, source},
                      {.out_dtype = self_dtype,
                       .out_dims = out_dims,
                       .op_param_cache_keys = std::move(param_keys)}));

    return MakeTensor(std::move(result));
  });
}

at::Tensor& AtenMaskedScatter_(at::Tensor& self, const at::Tensor& mask,
                               const at::Tensor& source) {
  TT_KERNEL(OpName::kMaskedScatter_, param_keys, (self, mask, source), {
    TT_ASSIGN_OR_THROW(const auto mask_dtype,
                       ConvertTo<mlir::ElementType>(mask.scalar_type()));
    TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch catches mask dtype first.
        mask.scalar_type() == at::kBool, error::kInvalidArgument)
        << "expected Boolean tensor for mask, got " << ToString(mask_dtype);

    TT_ASSIGN_OR_THROW(const auto self_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(const auto source_dtype,
                       ConvertTo<mlir::ElementType>(source.scalar_type()));
    TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch catches dtype mismatch
                     // first.
        self_dtype == source_dtype, error::kInvalidArgument)
        << "expected same dtype for self and source,"
        << " got self dtype " << ToString(self_dtype) << " and source dtype "
        << ToString(source_dtype);

    TT_THROW_IF_ERROR(ValidateInferredOutputShapeMatchesSelf(self, mask));

    Dimensions out_dims = CopyIntVector(self.sizes());

    auto shlo_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      // inputs[0]: self, inputs[1]: mask, inputs[2]: source
      return BuildMaskedScatterShlo(inputs[0], inputs[1], inputs[2]);
    };

    TT_THROW_IF_ERROR(
        DispatchOpOut<3>(std::move(shlo_builder), {self, mask, source}, self,
                         {.out_dtype = self_dtype,
                          .out_dims = out_dims,
                          .op_param_cache_keys = std::move(param_keys)}));

    return self;
  });
}

}  // namespace torch_tpu
