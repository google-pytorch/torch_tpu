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

#include "csrc/ops/select_scatter/select_scatter_aten_kernels.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/device_type.h"
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
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

struct SelectScatterParams {
  int64_t wrapped_dim;
  int64_t wrapped_index;
  Dimensions input_sizes;
  mlir::ElementType element_type;
};

absl::StatusOr<SelectScatterParams> ValidateSelectScatterInputs(
    const at::Tensor& self, const at::Tensor& src, int64_t dim, int64_t index) {
  TT_RET_CHECK(IsPrivateUse1Device(self), error::kInvalidArgument)
      << "expected self to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << self.device();
  TT_RET_CHECK(IsPrivateUse1Device(src), error::kInvalidArgument)
      << "expected src to be on " << GetPrivateUse1DeviceDebugName() << ", got "
      << src.device();
  const int64_t ndim = self.dim();
  TT_RET_CHECK(ndim > 0, error::kPythonIndexError)
      << "expected tensor to be at least 1D, got 0D";
  TT_ASSIGN_OR_RETURN(const int64_t wrapped_dim, SafeWrapDim(dim, ndim));
  const int64_t total_len = self.size(wrapped_dim);
  TT_RET_CHECK(index >= -total_len && index < total_len,
               error::kPythonIndexError)
      << "expected index to be in range [" << -total_len << ", "
      << total_len - 1 << "], got " << index;
  const int64_t wrapped_index = (index < 0) ? index + total_len : index;

  Dimensions expected_slice_shape;
  expected_slice_shape.reserve(ndim - 1);
  for (int64_t i = 0; i < ndim; ++i) {
    if (i != wrapped_dim) {
      expected_slice_shape.push_back(self.size(i));
    }
  }
  TT_RET_CHECK(src.sizes() == at::IntArrayRef(expected_slice_shape),
               error::kInvalidArgument)
      << "expected src to have a size equal to the slice of self, got "
      << src.sizes() << " vs " << at::IntArrayRef(expected_slice_shape);

  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  return SelectScatterParams{
      .wrapped_dim = wrapped_dim,
      .wrapped_index = wrapped_index,
      .input_sizes = CopyIntVector(self.sizes()),
      .element_type = element_type,
  };
}

// Builds the StableHLO graph for `aten::select_scatter`.
//
// Unsqueezes `src_op` to rank of `self_op` with size 1 along `dim`.
// If `dim` has static size 1 and `reshaped_src` matches `self_op`'s type,
// returns `reshaped_src` directly.
// Otherwise emits `stablehlo.dynamic_update_slice`.
absl::StatusOr<mlir::MlirOp> BuildSelectScatterShlo(
    mlir::MlirOp self_op, mlir::MlirOp src_op, const int64_t dim,
    const int64_t raw_index, const int64_t wrapped_index,
    const mlir::ElementType element_type) {
  TT_ASSIGN_OR_RETURN(src_op,
                      ComplexSafeConvertElementType(src_op, element_type));
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp reshaped_src, Unsqueeze(src_op, dim));
  reshaped_src = ClampToTargetBounds(reshaped_src, self_op);

  // If the target dimension has static size 1 and both tensors share the same
  // shape and bounds, the unsqueezed src already spans the entire output
  // tensor.
  if (!self_type.isDynamicDim(dim) && self_type.getDimSize(dim) == 1 &&
      GetTensorTypeOrDie(reshaped_src) == self_type) {
    return reshaped_src;
  }

  mlir::MlirBuilder& builder = self_op.getBuilder();
  const int64_t rank = self_type.getRank();
  std::vector<mlir::MlirOp> start_indices;
  start_indices.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    if (i == dim) {
      start_indices.push_back(
          ResolveDynamicIndex(self_op, dim, raw_index, wrapped_index));
    } else {
      start_indices.push_back(
          MakeScalarConstant(builder, 0, mlir::ElementType::I64));
    }
  }
  return mlir::stablehlo::DynamicUpdateSlice(self_op, reshaped_src,
                                             start_indices);
}

}  // namespace

at::Tensor AtenSelectScatter(const at::Tensor& self, const at::Tensor& src,
                             int64_t dim, int64_t index) {
  TT_KERNEL(OpName::kSelectScatter, param_keys, (self, src, dim, index), {
    TT_ASSIGN_OR_THROW(const SelectScatterParams params,
                       ValidateSelectScatterInputs(self, src, dim, index));

    if (self.numel() == 0) {
      TT_ASSIGN_OR_THROW(auto zero_buf,
                         CreateZeroSizeDeviceBufferRef(params.input_sizes,
                                                       params.element_type));
      return MakeTensor(std::move(zero_buf));
    }

    auto op_builder = [wrapped_dim = params.wrapped_dim, raw_index = index,
                       wrapped_index = params.wrapped_index,
                       element_type = params.element_type](
                          FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      return BuildSelectScatterShlo(inputs[0], inputs[1], wrapped_dim,
                                    raw_index, wrapped_index, element_type);
    };

    TT_ASSIGN_OR_THROW(
        auto out_buf,
        (DispatchOp<2>(std::move(op_builder), {self, src},
                       {.out_dtype = params.element_type,
                        .out_dims = params.input_sizes,
                        .op_param_cache_keys = std::move(param_keys)})));
    return MakeTensor(std::move(out_buf));
  });
}

at::Tensor& AtenSelectScatterOut(const at::Tensor& self, const at::Tensor& src,
                                 int64_t dim, int64_t index, at::Tensor& out) {
  TT_KERNEL(
      OpName::kSelectScatterOut, param_keys, (self, src, dim, index, out), {
        TT_CHECK_THROW(IsPrivateUse1Device(out), error::kInvalidArgument)
            << "expected out to be on " << GetPrivateUse1DeviceDebugName()
            << ", got " << out.device();
        TT_ASSIGN_OR_THROW(const SelectScatterParams params,
                           ValidateSelectScatterInputs(self, src, dim, index));
        TT_CHECK_THROW(out.scalar_type() == self.scalar_type(),
                       error::kInvalidArgument)
            << "expected out dtype " << ToString(self.scalar_type()) << ", got "
            << ToString(out.scalar_type());

        if (self.numel() == 0) {
          TT_ASSIGN_OR_THROW(auto zero_buf,
                             CreateZeroSizeDeviceBufferRef(
                                 params.input_sizes, params.element_type));
          TT_THROW_IF_ERROR(
              ResizeTensorIfShapeDiffers(out, params.input_sizes));
          TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(zero_buf), out));
          return out;
        }

        auto op_builder = [wrapped_dim = params.wrapped_dim, raw_index = index,
                           wrapped_index = params.wrapped_index,
                           element_type = params.element_type](
                              FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildSelectScatterShlo(inputs[0], inputs[1], wrapped_dim,
                                        raw_index, wrapped_index, element_type);
        };

        TT_THROW_IF_ERROR(
            (DispatchOpOut<2>(std::move(op_builder), {self, src}, out,
                              {.out_dtype = params.element_type,
                               .out_dims = params.input_sizes,
                               .op_param_cache_keys = std::move(param_keys)})));
        return out;
      });
}

}  // namespace torch_tpu
