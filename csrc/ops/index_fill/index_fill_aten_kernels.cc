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

#include "torch_tpu/csrc/ops/index_fill/index_fill_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/aten_utils.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildIndexFillShlo(mlir::MlirOp self, int64_t dim,
                                                mlir::MlirOp index,
                                                mlir::MlirOp value) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType index_type = GetTensorTypeOrDie(index);
  const mlir::Type computation_type = self_type.getElementType();

  // If self is 0D (scalar), the output is just the value.
  if (self_type.getRank() == 0) {
    return value;
  }

  // Create updates tensor by broadcasting scalar value.
  Dimensions updates_shape = CopyIntVector(self_type.getShape());
  updates_shape[dim] = index_type.getDimSize(0);
  mlir::RankedTensorType updates_type =
      mlir::RankedTensorType::get(updates_shape, computation_type);
  mlir::MlirOp updates =
      mlir::stablehlo::BroadcastInDim(updates_type, value, {});

  Dimensions all_other_dims;
  all_other_dims.reserve(self_type.getRank() - 1);
  for (int i = 0; i < self_type.getRank(); ++i) {
    if (i != dim) {
      all_other_dims.push_back(i);
    }
  }

  index = mlir::stablehlo::Reshape(index, {index_type.getDimSize(0), 1});
  mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers =
      mlir::stablehlo::ScatterDimensionNumbersAttr::get(
          &self.getContext(),
          /*update_window_dims=*/all_other_dims,
          /*inserted_window_dims=*/{dim},
          /*input_batching_dims=*/{},
          /*scatter_indices_batching_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{dim},
          /*index_vector_dim=*/1);

  // Create a region builder callback.
  auto block_type = mlir::RankedTensorType::get({}, computation_type);
  auto region_builder = [block_type](mlir::RegionBuilder& builder) {
    mlir::Argument(builder, block_type);
    mlir::MlirOp update_value = mlir::Argument(builder, block_type);
    mlir::stablehlo::Return(builder, {update_value});
  };

  return mlir::stablehlo::Scatter(self, index, updates, region_builder,
                                  scatter_dimension_numbers)[0];
}

absl::Status CheckIndexFillInputs(const at::Tensor& self, int64_t dim,
                                  const at::Tensor& index) {
  TT_RET_CHECK(index.dim() <= 1, error::kInvalidArgument)
      << "expected index to be at most 1-D, got " << index.dim() << "-D";
  TT_RET_CHECK(IsLong(index), error::kInvalidArgument)
      << "expected index dtype to be Long, got "
      << ToString(index.scalar_type());
  return absl::OkStatus();
}
}  // namespace

at::Tensor& AtenIndexFillIntScalar_(at::Tensor& self, int64_t dim,
                                    const at::Tensor& index,
                                    const at::Scalar& value) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(
      OpName::kIndexFillIntScalar, param_keys,
      (self, dim, index, promoted_value), {
        TT_THROW_IF_ERROR(CheckIndexFillInputs(self, dim, index));

        TT_ASSIGN_OR_THROW(const int64_t wrapped_dim,
                           SafeWrapDim(dim, self.dim()));
        TT_ASSIGN_OR_THROW(const at::Tensor value_tensor,
                           promoted_value.GetTensor(self.scalar_type()));

        auto op_builder = [wrapped_dim](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [self, index, value] = inputs;
          return BuildIndexFillShlo(self, wrapped_dim, index, value);
        };

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        TT_ASSIGN_OR_THROW(
            DeviceBufferRef result_buf,
            DispatchOp<3>(std::move(op_builder), {self, index, value_tensor},
                          {.out_dtype = output_dtype,
                           .out_dims = self.sizes(),
                           .op_param_cache_keys = std::move(param_keys)}));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), self));
        return self;
      });
}

at::Tensor& AtenIndexFillIntTensor_(at::Tensor& self, int64_t dim,
                                    const at::Tensor& index,
                                    const at::Tensor& value) {
  TT_KERNEL(
      OpName::kIndexFillIntTensor, param_keys, (self, dim, index, value), {
        TT_THROW_IF_ERROR(CheckIndexFillInputs(self, dim, index));
        TT_CHECK_THROW(value.dim() == 0, error::kInvalidArgument)
            << "expected value to be a 0-D tensor, got " << value.dim()
            << "-D tensor";

        TT_ASSIGN_OR_THROW(const int64_t wrapped_dim,
                           SafeWrapDim(dim, self.dim()));

        auto op_builder = [wrapped_dim](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [self, index, value] = inputs;
          return BuildIndexFillShlo(self, wrapped_dim, index, value);
        };

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        TT_ASSIGN_OR_THROW(
            DeviceBufferRef result_buf,
            DispatchOp<3>(std::move(op_builder), {self, index, value},
                          {.out_dtype = output_dtype,
                           .out_dims = self.sizes(),
                           .op_param_cache_keys = std::move(param_keys)}));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), self));
        return self;
      });
}

}  // namespace torch_tpu
