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

#include "csrc/ops/pooling/max_unpool_aten_kernels.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/SymIntArrayRef.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Lowers max_unpool{2,3}d to a StableHLO ScatterOp.
//
// `self` and `indices` have shape [batch..., in_spatial...] and `indices`
// holds, for every input element, its flattened position within the output's
// spatial volume (as produced by max_pool*d_with_indices). Both are reshaped to
// [B, S_in] where B is the product of the batch dimensions, scattered into a
// zero tensor of shape [B, S_out] using B as a batching dimension, and the
// result is reshaped to [batch..., output_size...].
absl::StatusOr<mlir::MlirOp> BuildMaxUnpoolShlo(mlir::MlirOp self,
                                                mlir::MlirOp indices,
                                                int64_t spatial_dim_count,
                                                const Dimensions& out_dims) {
  mlir::MlirBuilder& builder = self.getBuilder();
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::Type computation_type = self_type.getElementType();

  const int64_t rank = self_type.getRank();
  const int64_t batch_rank = rank - spatial_dim_count;
  int64_t batch_size = 1;
  for (int64_t i = 0; i < batch_rank; ++i) {
    batch_size *= self_type.getDimSize(i);
  }
  int64_t in_spatial_size = 1;
  int64_t out_spatial_size = 1;
  for (int64_t i = batch_rank; i < rank; ++i) {
    in_spatial_size *= self_type.getDimSize(i);
    out_spatial_size *= out_dims[i];
  }

  mlir::MlirOp updates =
      mlir::stablehlo::Reshape(self, {batch_size, in_spatial_size});
  mlir::MlirOp scatter_indices =
      mlir::stablehlo::Reshape(indices, {batch_size, in_spatial_size, 1});
  mlir::MlirOp zeros = MakeConstant(builder, 0, computation_type,
                                    Dimensions{batch_size, out_spatial_size});

  const mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers =
      mlir::stablehlo::ScatterDimensionNumbersAttr::get(
          &builder.getContext(),
          /*update_window_dims=*/{},
          /*inserted_window_dims=*/{1},
          /*input_batching_dims=*/{0},
          /*scatter_indices_batching_dims=*/{0},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/2);

  // The update replaces whatever is at the target position. Positions that no
  // index points to keep their zero.
  const auto block_type = mlir::RankedTensorType::get({}, computation_type);
  auto region_builder = [block_type](mlir::RegionBuilder& region) {
    mlir::Argument(region, block_type);
    mlir::MlirOp update_value = mlir::Argument(region, block_type);
    mlir::stablehlo::Return(region, {update_value});
  };

  mlir::MlirOp scattered =
      mlir::stablehlo::Scatter(zeros, scatter_indices, updates, region_builder,
                               scatter_dimension_numbers)[0];
  return mlir::stablehlo::Reshape(scattered, out_dims);
}

// Mirrors the checks in ATen's max_unpooling{2,3}d_forward_out_cpu.
absl::StatusOr<Dimensions> ValidateMaxUnpoolInputs(
    const at::Tensor& self, const at::Tensor& indices,
    c10::SymIntArrayRef output_size, int64_t spatial_dim_count) {
  TT_RET_CHECK(IsLong(indices), error::kInvalidArgument)
      << "expected indices dtype to be Long, got "
      << ToString(indices.scalar_type());
  TT_RET_CHECK(output_size.size() == spatial_dim_count, error::kInvalidArgument)
      << "expected output_size to have " << spatial_dim_count
      << " elements, got " << output_size.size();
  const int64_t num_dims = self.dim();
  TT_RET_CHECK(
      num_dims == spatial_dim_count + 1 || num_dims == spatial_dim_count + 2,
      error::kInvalidArgument)
      << "expected input to be a " << spatial_dim_count + 1 << "-D or "
      << spatial_dim_count + 2 << "-D tensor, got " << num_dims << "-D tensor";
  TT_RET_CHECK(self.sizes() == indices.sizes(), error::kInvalidArgument)
      << "expected indices to have the same shape as input " << self.sizes()
      << ", got " << indices.sizes();
  // Non-batch dimensions must be non-empty; an empty batch dimension is fine.
  for (int64_t i = num_dims - spatial_dim_count - 1; i < num_dims; ++i) {
    TT_RET_CHECK(self.size(i) != 0, error::kInvalidArgument)
        << "expected input to have non-zero size for non-batch dimensions, got "
        << self.sizes() << " with dimension " << i << " being empty";
  }

  Dimensions out_dims = CopyIntVector(self.sizes());
  for (int64_t i = 0; i < spatial_dim_count; ++i) {
    const int64_t size = output_size[i].expect_int();
    TT_RET_CHECK(size >= 0, error::kInvalidArgument)
        << "expected output_size elements to be non-negative, got "
        << output_size;
    out_dims[num_dims - spatial_dim_count + i] = size;
  }
  return out_dims;
}

absl::StatusOr<at::Tensor> MaxUnpoolHelper(
    const at::Tensor& self, const at::Tensor& indices,
    c10::SymIntArrayRef output_size, int64_t spatial_dim_count,
    OpParamCacheKeys param_keys,
    std::optional<std::reference_wrapper<at::Tensor>> out = std::nullopt) {
  TT_ASSIGN_OR_RETURN(
      Dimensions out_dims,
      ValidateMaxUnpoolInputs(self, indices, output_size, spatial_dim_count));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  if (out.has_value()) {
    TT_RETURN_IF_ERROR(ValidateOutDtype(out->get(), self.scalar_type(),
                                        /*allow_out_dtype_cast=*/false));
    TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out->get(), out_dims));
  }

  // An empty batch or an empty output volume yields an empty output.
  int64_t out_numel = 1;
  for (const int64_t size : out_dims) {
    out_numel *= size;
  }
  if (self.numel() == 0 || out_numel == 0) {
    TT_ASSIGN_OR_RETURN(DeviceBufferRef zero_buf,
                        CreateZeroSizeDeviceBufferRef(out_dims, output_dtype));
    if (out.has_value()) {
      TT_RETURN_IF_ERROR(
          AssignBufferToAtTensor(std::move(zero_buf), out->get()));
      return out->get();
    }
    return MakeTensor(zero_buf);
  }

  auto op_builder = [spatial_dim_count,
                     out_dims](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto [self_op, indices_op] = inputs;
    return BuildMaxUnpoolShlo(self_op, indices_op, spatial_dim_count, out_dims);
  };

  if (out.has_value()) {
    TT_RETURN_IF_ERROR(
        DispatchOpOut<2>(std::move(op_builder), {self, indices}, out->get(),
                         {.out_dtype = output_dtype,
                          .out_dims = out_dims,
                          .op_param_cache_keys = std::move(param_keys)}));
    return out->get();
  }
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef buffer,
      DispatchOp<2>(std::move(op_builder), {self, indices},
                    {.out_dtype = output_dtype,
                     .out_dims = out_dims,
                     .op_param_cache_keys = std::move(param_keys)}));
  return MakeTensor(buffer);
}

}  // namespace

at::Tensor AtenMaxUnpool2d(const at::Tensor& self, const at::Tensor& indices,
                           c10::SymIntArrayRef output_size) {
  TT_KERNEL(OpName::kMaxUnpool2d, param_keys, (self, indices, output_size), {
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        MaxUnpoolHelper(self, indices, output_size, /*spatial_dim_count=*/2,
                        std::move(param_keys)));
    return result;
  });
}

at::Tensor& AtenMaxUnpool2dOut(const at::Tensor& self,
                               const at::Tensor& indices,
                               c10::SymIntArrayRef output_size,
                               at::Tensor& out) {
  TT_KERNEL(OpName::kMaxUnpool2dOut, param_keys,
            (self, indices, output_size, out), {
              TT_THROW_IF_ERROR(MaxUnpoolHelper(self, indices, output_size,
                                                /*spatial_dim_count=*/2,
                                                std::move(param_keys), out)
                                    .status());
              return out;
            });
}

at::Tensor AtenMaxUnpool3d(const at::Tensor& self, const at::Tensor& indices,
                           c10::SymIntArrayRef output_size,
                           at::IntArrayRef stride, at::IntArrayRef padding) {
  TT_KERNEL(OpName::kMaxUnpool3d, param_keys,
            (self, indices, output_size,
             IgnoreInCacheKey(stride, "does not affect the computation"),
             IgnoreInCacheKey(padding, "does not affect the computation")),
            {
              TT_CHECK_THROW(stride.size() == 3 && padding.size() == 3,
                             error::kInvalidArgument)
                  << "expected stride and padding to have 3 elements, got "
                  << stride.size() << " and " << padding.size();
              TT_ASSIGN_OR_THROW(at::Tensor result,
                                 MaxUnpoolHelper(self, indices, output_size,
                                                 /*spatial_dim_count=*/3,
                                                 std::move(param_keys)));
              return result;
            });
}

at::Tensor& AtenMaxUnpool3dOut(const at::Tensor& self,
                               const at::Tensor& indices,
                               c10::SymIntArrayRef output_size,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               at::Tensor& out) {
  TT_KERNEL(OpName::kMaxUnpool3dOut, param_keys,
            (self, indices, output_size,
             IgnoreInCacheKey(stride, "does not affect the computation"),
             IgnoreInCacheKey(padding, "does not affect the computation"), out),
            {
              TT_CHECK_THROW(stride.size() == 3 && padding.size() == 3,
                             error::kInvalidArgument)
                  << "expected stride and padding to have 3 elements, got "
                  << stride.size() << " and " << padding.size();
              TT_THROW_IF_ERROR(MaxUnpoolHelper(self, indices, output_size,
                                                /*spatial_dim_count=*/3,
                                                std::move(param_keys), out)
                                    .status());
              return out;
            });
}

}  // namespace torch_tpu
