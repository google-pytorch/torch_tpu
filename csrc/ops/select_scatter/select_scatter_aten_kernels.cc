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

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
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

// Builds the StableHLO graph for `aten::select_scatter`.
//
// Unsqueezes `src_op` to rank of `self_op` with size 1 along `dim`.
// If `total_len == 1`, returns `reshaped_src` directly.
// Otherwise emits `stablehlo.dynamic_update_slice`.
absl::StatusOr<mlir::MlirOp> BuildSelectScatterShlo(
    mlir::MlirOp self_op, mlir::MlirOp src_op, const Dimensions& input_sizes,
    const int64_t dim, const int64_t index) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType src_type = GetTensorTypeOrDie(src_op);
  if (src_type.getElementType() != self_type.getElementType()) {
    src_op =
        mlir::stablehlo::ConvertElementType(src_op, self_type.getElementType());
  }
  mlir::MlirBuilder& builder = self_op.getBuilder();

  // Unsqueeze `src_op` from (rank - 1) to rank with size 1 along `dim`.
  Dimensions unsqueezed_shape = input_sizes;
  unsqueezed_shape[dim] = 1;
  const auto unsqueezed_type =
      mlir::RankedTensorType::get(unsqueezed_shape, self_type.getElementType());
  mlir::MlirOp reshaped_src = mlir::stablehlo::Reshape(unsqueezed_type, src_op);

  // If the target dimension has size 1, the unsqueezed src already
  // spans the entire output tensor.
  if (input_sizes[dim] == 1) {
    return reshaped_src;
  }

  std::vector<mlir::MlirOp> start_indices;
  start_indices.reserve(input_sizes.size());
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    const int64_t start_idx = (static_cast<int64_t>(i) == dim) ? index : 0;
    start_indices.push_back(
        MakeScalarConstant(builder, start_idx, mlir::ElementType::I64));
  }
  return mlir::stablehlo::DynamicUpdateSlice(self_op, reshaped_src,
                                             start_indices);
}

}  // namespace

at::Tensor AtenSelectScatter(const at::Tensor& self, const at::Tensor& src,
                             int64_t dim, int64_t index) {
  TT_KERNEL(OpName::kSelectScatter, param_keys, (self, src, dim, index), {
    const int64_t ndim = self.dim();
    TT_CHECK_THROW(ndim > 0, error::kPythonIndexError)
        << "expected tensor to be at least 1-dimensional, got 0-dimensional";
    TT_ASSIGN_OR_THROW(const int64_t wrapped_dim, SafeWrapDim(dim, ndim));
    const int64_t total_len = self.size(wrapped_dim);
    TT_CHECK_THROW(index >= -total_len && index < total_len,
                   error::kPythonIndexError)
        << "expected index to be in range [" << -total_len << ", "
        << total_len - 1 << "], got " << index;
    int64_t wrapped_index = index;
    if (wrapped_index < 0) {
      wrapped_index += total_len;
    }

    Dimensions expected_slice_shape;
    expected_slice_shape.reserve(ndim - 1);
    for (int64_t i = 0; i < ndim; ++i) {
      if (i != wrapped_dim) {
        expected_slice_shape.push_back(self.size(i));
      }
    }
    TT_CHECK_THROW(src.sizes() == at::IntArrayRef(expected_slice_shape),
                   error::kInvalidArgument)
        << "expected src to have a size equal to the slice of self, got "
        << src.sizes() << " vs " << at::IntArrayRef(expected_slice_shape);

    const Dimensions input_sizes_vec = CopyIntVector(self.sizes());
    TT_ASSIGN_OR_THROW(const mlir::ElementType element_type,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));

    if (self.numel() == 0) {
      TT_ASSIGN_OR_THROW(auto zero_buf, CreateZeroSizeDeviceBufferRef(
                                            input_sizes_vec, element_type));
      return MakeTensor(std::move(zero_buf));
    }

    auto op_builder = [input_sizes_vec, wrapped_dim,
                       wrapped_index](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      return BuildSelectScatterShlo(inputs[0], inputs[1], input_sizes_vec,
                                    wrapped_dim, wrapped_index);
    };

    TT_ASSIGN_OR_THROW(
        auto out_buf,
        (DispatchOp<2>(std::move(op_builder), {self, src},
                       {.out_dtype = element_type,
                        .out_dims = input_sizes_vec,
                        .op_param_cache_keys = std::move(param_keys)})));
    return MakeTensor(std::move(out_buf));
  });
}

at::Tensor& AtenSelectScatterOut(const at::Tensor& self, const at::Tensor& src,
                                 int64_t dim, int64_t index, at::Tensor& out) {
  TT_KERNEL(
      OpName::kSelectScatterOut, param_keys, (self, src, dim, index, out), {
        TT_CHECK_THROW(out.scalar_type() == self.scalar_type(),
                       error::kInvalidArgument)
            << "expected out dtype " << ToString(self.scalar_type()) << ", got "
            << ToString(out.scalar_type());
        const int64_t ndim = self.dim();
        TT_CHECK_THROW(ndim > 0, error::kPythonIndexError)
            << "expected tensor to be at least 1-dimensional, got "
               "0-dimensional";
        TT_ASSIGN_OR_THROW(const int64_t wrapped_dim, SafeWrapDim(dim, ndim));
        const int64_t total_len = self.size(wrapped_dim);
        TT_CHECK_THROW(index >= -total_len && index < total_len,
                       error::kPythonIndexError)
            << "expected index to be in range [" << -total_len << ", "
            << total_len - 1 << "], got " << index;
        int64_t wrapped_index = index;
        if (wrapped_index < 0) {
          wrapped_index += total_len;
        }

        Dimensions expected_slice_shape;
        expected_slice_shape.reserve(ndim - 1);
        for (int64_t i = 0; i < ndim; ++i) {
          if (i != wrapped_dim) {
            expected_slice_shape.push_back(self.size(i));
          }
        }
        TT_CHECK_THROW(src.sizes() == at::IntArrayRef(expected_slice_shape),
                       error::kInvalidArgument)
            << "expected src to have a size equal to the slice of self, got "
            << src.sizes() << " vs " << at::IntArrayRef(expected_slice_shape);

        const Dimensions input_sizes_vec = CopyIntVector(self.sizes());
        TT_ASSIGN_OR_THROW(const mlir::ElementType element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        if (self.numel() == 0) {
          TT_ASSIGN_OR_THROW(auto zero_buf, CreateZeroSizeDeviceBufferRef(
                                                input_sizes_vec, element_type));
          TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, input_sizes_vec));
          TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(zero_buf), out));
          return out;
        }

        auto op_builder = [input_sizes_vec, wrapped_dim,
                           wrapped_index](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildSelectScatterShlo(inputs[0], inputs[1], input_sizes_vec,
                                        wrapped_dim, wrapped_index);
        };

        TT_THROW_IF_ERROR(
            (DispatchOpOut<2>(std::move(op_builder), {self, src}, out,
                              {.out_dtype = element_type,
                               .out_dims = input_sizes_vec,
                               .op_param_cache_keys = std::move(param_keys)})));
        return out;
      });
}

}  // namespace torch_tpu
