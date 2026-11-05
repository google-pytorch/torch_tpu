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

#include <algorithm>
#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
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
#include "csrc/ops/scatter/scatter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Lowers max_unpool{2,3}d to a scatter along the flattened spatial dimension.
//
// `self` and `indices` have shape [batch..., in_spatial...] and `indices`
// holds, for every input element, its flattened position within the output's
// spatial volume (as produced by max_pool*d_with_indices). Both are reshaped to
// [B, S_in], where B is the product of the batch dimensions, and `self` is
// scattered along dimension 1 into a zero tensor of shape [B, S_out]. Positions
// that no index points to keep their zero. The result is reshaped to
// [batch..., output_size...].
absl::StatusOr<mlir::MlirOp> BuildMaxUnpoolShlo(mlir::MlirOp self,
                                                mlir::MlirOp indices,
                                                int64_t spatial_dim_count,
                                                const Dimensions& out_dims,
                                                mlir::ElementType element_type) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const int64_t batch_rank = self_type.getRank() - spatial_dim_count;
  const absl::Span<const int64_t> self_dims = self_type.getShape();
  TT_ASSIGN_OR_RETURN(const int64_t batch_size,
                      NumElements(self_dims.subspan(0, batch_rank)));
  TT_ASSIGN_OR_RETURN(const int64_t in_spatial_size,
                      NumElements(self_dims.subspan(batch_rank)));
  TT_ASSIGN_OR_RETURN(
      const int64_t out_spatial_size,
      NumElements(absl::MakeConstSpan(out_dims).subspan(batch_rank)));

  mlir::MlirOp src =
      mlir::stablehlo::Reshape(self, {batch_size, in_spatial_size});
  mlir::MlirOp index =
      mlir::stablehlo::Reshape(indices, {batch_size, in_spatial_size});
  mlir::MlirOp zeros =
      MakeConstant(self.getBuilder(), 0, self_type.getElementType(),
                   Dimensions{batch_size, out_spatial_size});
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp scattered,
      BuildScatterShlo(zeros, /*dim=*/1, index, src, ScatterOp::kReplace,
                       element_type));
  return mlir::stablehlo::Reshape(scattered, out_dims);
}

// Mirrors the checks in ATen's max_unpooling{2,3}d_forward_out_cpu, which is
// only implemented for floating-point inputs.
absl::Status ValidateMaxUnpoolInputs(const at::Tensor& self,
                                     const at::Tensor& indices,
                                     c10::SymIntArrayRef output_size,
                                     int64_t spatial_dim_count) {
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(self));
  TT_RET_CHECK(IsFloatingPoint(self), error::kPythonNotImplementedError)
      << "not implemented for " << ToString(self.scalar_type());
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
  return absl::OkStatus();
}

// Returns the output shape: the batch dimensions of `self` followed by
// `output_size`.
absl::StatusOr<Dimensions> ComputeOutputDimensions(
    const at::Tensor& self, c10::SymIntArrayRef output_size) {
  TT_ASSIGN_OR_RETURN(const Dimensions spatial_dims,
                      ConcreteSizeToDimensions(output_size));
  for (const int64_t size : spatial_dims) {
    TT_RET_CHECK(size >= 0, error::kInvalidArgument)
        << "expected output_size elements to be non-negative, got "
        << output_size;
  }
  Dimensions out_dims = CopyIntVector(self.sizes());
  std::copy(spatial_dims.begin(), spatial_dims.end(),
            out_dims.end() - spatial_dims.size());
  return out_dims;
}

absl::Status ValidateMaxUnpool3dStrideAndPadding(at::IntArrayRef stride,
                                                 at::IntArrayRef padding) {
  TT_RET_CHECK(stride.size() == 3 && padding.size() == 3,
               error::kInvalidArgument)
      << "expected stride and padding to have 3 elements, got "
      << stride.size() << " and " << padding.size();
  return absl::OkStatus();
}

// Writes max_unpool{2,3}d(self, indices, output_size) to `out`, resizing it to
// the output shape if needed.
absl::Status MaxUnpoolOut(const at::Tensor& self, const at::Tensor& indices,
                          c10::SymIntArrayRef output_size,
                          int64_t spatial_dim_count,
                          OpParamCacheKeys param_keys, at::Tensor& out) {
  TT_RETURN_IF_ERROR(
      ValidateMaxUnpoolInputs(self, indices, output_size, spatial_dim_count));
  TT_ASSIGN_OR_RETURN(const Dimensions out_dims,
                      ComputeOutputDimensions(self, output_size));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, self.scalar_type(),
                                      /*allow_out_dtype_cast=*/false));

  // An empty batch or an empty output volume yields an empty output.
  TT_ASSIGN_OR_RETURN(const int64_t out_numel, NumElements(out_dims));
  if (out_numel == 0) {
    TT_ASSIGN_OR_RETURN(DeviceBufferRef zero_buf,
                        CreateZeroSizeDeviceBufferRef(out_dims, output_dtype));
    TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out, out_dims));
    return AssignBufferToAtTensor(std::move(zero_buf), out);
  }

  auto op_builder = [spatial_dim_count, out_dims, output_dtype](
                        FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto [self_op, indices_op] = inputs;
    return BuildMaxUnpoolShlo(self_op, indices_op, spatial_dim_count, out_dims,
                              output_dtype);
  };
  return DispatchOpOut<2>(std::move(op_builder), {self, indices}, out,
                          {.out_dtype = output_dtype,
                           .out_dims = out_dims,
                           .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenMaxUnpool2d(const at::Tensor& self, const at::Tensor& indices,
                           c10::SymIntArrayRef output_size) {
  TT_KERNEL(OpName::kMaxUnpool2d, param_keys, (self, indices, output_size), {
    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        MakeEmptyTensor(/*size=*/{0}, self.scalar_type(), self.device()));
    TT_THROW_IF_ERROR(MaxUnpoolOut(self, indices, output_size,
                                   /*spatial_dim_count=*/2,
                                   std::move(param_keys), out));
    return out;
  });
}

at::Tensor& AtenMaxUnpool2dOut(const at::Tensor& self,
                               const at::Tensor& indices,
                               c10::SymIntArrayRef output_size,
                               at::Tensor& out) {
  TT_KERNEL(OpName::kMaxUnpool2dOut, param_keys,
            (self, indices, output_size, out), {
              TT_THROW_IF_ERROR(MaxUnpoolOut(self, indices, output_size,
                                             /*spatial_dim_count=*/2,
                                             std::move(param_keys), out));
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
              TT_THROW_IF_ERROR(
                  ValidateMaxUnpool3dStrideAndPadding(stride, padding));
              TT_ASSIGN_OR_THROW(at::Tensor out,
                                 MakeEmptyTensor(/*size=*/{0},
                                                 self.scalar_type(),
                                                 self.device()));
              TT_THROW_IF_ERROR(MaxUnpoolOut(self, indices, output_size,
                                             /*spatial_dim_count=*/3,
                                             std::move(param_keys), out));
              return out;
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
              TT_THROW_IF_ERROR(
                  ValidateMaxUnpool3dStrideAndPadding(stride, padding));
              TT_THROW_IF_ERROR(MaxUnpoolOut(self, indices, output_size,
                                             /*spatial_dim_count=*/3,
                                             std::move(param_keys), out));
              return out;
            });
}

}  // namespace torch_tpu
