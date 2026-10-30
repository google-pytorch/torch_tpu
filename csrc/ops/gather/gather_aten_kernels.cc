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

#include "csrc/ops/gather/gather_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/Storage.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/gather/gather.h"
#include "csrc/ops/index_utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

absl::StatusOr<DeviceBufferRef> Gather(const at::Tensor& self, int64_t dim,
                                       const at::Tensor& index,
                                       bool sparse_grad,
                                       OpParamCacheKeys&& param_keys) {
  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self.dim()));
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  Dimensions output_dims = CopyIntVector(index.sizes());

  if (index.numel() == 0) {
    // Gathering with 0 indexes results in a zero-sized tensor.
    return CreateZeroSizeDeviceBufferRef(output_dims, output_dtype);
  }

  TT_RETURN_IF_ERROR(
      ValidateGatherInputs(self.sizes(), dim, index.sizes(), sparse_grad));

  if (auto unexpanded = TryUnexpandZeroStrideIndex(index)) {
    Dimensions squeezed_dims = std::move(unexpanded->squeezed_dims);
    Dimensions bcast_dims = std::move(unexpanded->bcast_dims);
    Dimensions logical_dims = std::move(unexpanded->logical_dims);
    TT_ASSIGN_OR_RETURN(
        auto full_param_keys,
        TT_MAKE_OP_PARAM_CACHE_KEYS(dim, sparse_grad, squeezed_dims, bcast_dims,
                                    logical_dims));
    auto gather_op_builder = [dim, sparse_grad, output_dtype, squeezed_dims,
                              bcast_dims, logical_dims](
                                 FixedSizeSpan<mlir::MlirOp, 2> inputs) {
      auto& [self_op, index_op] = inputs;
      const auto index_type = GetTensorTypeOrDie(index_op);
      if (index_type.getRank() != static_cast<int64_t>(squeezed_dims.size())) {
        index_op = mlir::stablehlo::Reshape(index_op, squeezed_dims);
      }
      const auto logical_type = mlir::RankedTensorType::get(
          logical_dims, index_type.getElementType());
      mlir::MlirOp expanded_index =
          mlir::stablehlo::BroadcastInDim(logical_type, index_op, bcast_dims);
      return BuildGatherShlo(self_op, dim, expanded_index, sparse_grad,
                             output_dtype);
    };
    return DispatchOp<2>(std::move(gather_op_builder),
                         {self, unexpanded->tensor},
                         {.out_dtype = output_dtype,
                          .out_dims = output_dims,
                          .op_param_cache_keys = std::move(full_param_keys)});
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

absl::Status GatherOut(const at::Tensor& self, int64_t dim,
                       const at::Tensor& index, bool sparse_grad,
                       OpParamCacheKeys&& param_keys, at::Tensor& out) {
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, self.scalar_type(),
                                      /*allow_out_dtype_cast=*/false));
  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self.dim()));
  TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out, index.sizes()));

  if (index.numel() == 0) {
    return absl::OkStatus();
  }

  TT_RETURN_IF_ERROR(
      ValidateGatherInputs(self.sizes(), dim, index.sizes(), sparse_grad));

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  Dimensions output_dims = CopyIntVector(index.sizes());

  if (auto unexpanded = TryUnexpandZeroStrideIndex(index)) {
    Dimensions squeezed_dims = std::move(unexpanded->squeezed_dims);
    Dimensions bcast_dims = std::move(unexpanded->bcast_dims);
    Dimensions logical_dims = std::move(unexpanded->logical_dims);
    TT_ASSIGN_OR_RETURN(
        auto full_param_keys,
        TT_MAKE_OP_PARAM_CACHE_KEYS(dim, sparse_grad, squeezed_dims, bcast_dims,
                                    logical_dims));
    auto gather_op_builder = [dim, sparse_grad, output_dtype, squeezed_dims,
                              bcast_dims, logical_dims](
                                 FixedSizeSpan<mlir::MlirOp, 2> inputs) {
      auto& [self_op, index_op] = inputs;
      const auto index_type = GetTensorTypeOrDie(index_op);
      if (index_type.getRank() != static_cast<int64_t>(squeezed_dims.size())) {
        index_op = mlir::stablehlo::Reshape(index_op, squeezed_dims);
      }
      const auto logical_type = mlir::RankedTensorType::get(
          logical_dims, index_type.getElementType());
      mlir::MlirOp expanded_index =
          mlir::stablehlo::BroadcastInDim(logical_type, index_op, bcast_dims);
      return BuildGatherShlo(self_op, dim, expanded_index, sparse_grad,
                             output_dtype);
    };
    return DispatchOpOut<2, 1>(
        std::move(gather_op_builder), {self, unexpanded->tensor}, out,
        {.out_dtype = output_dtype,
         .out_dims = output_dims,
         .op_param_cache_keys = std::move(full_param_keys)});
  }

  auto gather_op_builder =
      [dim, sparse_grad, output_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
        auto& [self, index] = inputs;
        return BuildGatherShlo(self, dim, index, sparse_grad, output_dtype);
      };

  return DispatchOpOut<2, 1>(std::move(gather_op_builder), {self, index}, out,
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
  TT_KERNEL(OpName::kGatherOut, param_keys,
            (self, dim, index, sparse_grad, out), {
              TT_THROW_IF_ERROR(GatherOut(self, dim, index, sparse_grad,
                                          std::move(param_keys), out));
              return out;
            });
}

}  // namespace torch_tpu
