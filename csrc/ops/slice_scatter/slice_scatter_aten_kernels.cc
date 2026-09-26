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

#include "csrc/ops/slice_scatter/slice_scatter_aten_kernels.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Builds the StableHLO graph for `aten::slice_scatter` when step == 1.
mlir::MlirOp BuildSliceScatterShlo(mlir::MlirOp self_op, mlir::MlirOp src_op,
                                   const Dimensions& self_sizes,
                                   const int64_t dim, const int64_t start) {
  mlir::MlirBuilder& builder = self_op.getBuilder();
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType src_type = GetTensorTypeOrDie(src_op);
  mlir::MlirOp converted_src = src_op;
  if (src_type.getElementType() != self_type.getElementType()) {
    converted_src =
        mlir::stablehlo::ConvertElementType(src_op, self_type.getElementType());
  }

  std::vector<mlir::MlirOp> start_indices;
  start_indices.reserve(self_sizes.size());
  for (size_t i = 0; i < self_sizes.size(); ++i) {
    const int64_t start_idx = (static_cast<int64_t>(i) == dim) ? start : 0;
    start_indices.push_back(
        MakeScalarConstant(builder, start_idx, mlir::ElementType::I64));
  }
  return mlir::stablehlo::DynamicUpdateSlice(self_op, converted_src,
                                             start_indices);
}

// Builds the StableHLO graph for `aten::select_scatter`.
mlir::MlirOp BuildSelectScatterShlo(mlir::MlirOp self_op, mlir::MlirOp src_op,
                                    const Dimensions& self_sizes,
                                    const int64_t dim, const int64_t index) {
  mlir::MlirBuilder& builder = self_op.getBuilder();
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType src_type = GetTensorTypeOrDie(src_op);
  mlir::MlirOp converted_src = src_op;
  if (src_type.getElementType() != self_type.getElementType()) {
    converted_src =
        mlir::stablehlo::ConvertElementType(src_op, self_type.getElementType());
  }

  // Unsqueeze `converted_src` to rank of `self_op` with size 1 along `dim`.
  Dimensions unsqueezed_shape = self_sizes;
  unsqueezed_shape[dim] = 1;
  mlir::MlirOp reshaped_src =
      mlir::stablehlo::Reshape(converted_src, unsqueezed_shape);

  std::vector<mlir::MlirOp> start_indices;
  start_indices.reserve(self_sizes.size());
  for (size_t i = 0; i < self_sizes.size(); ++i) {
    const int64_t start_idx = (static_cast<int64_t>(i) == dim) ? index : 0;
    start_indices.push_back(
        MakeScalarConstant(builder, start_idx, mlir::ElementType::I64));
  }
  return mlir::stablehlo::DynamicUpdateSlice(self_op, reshaped_src,
                                             start_indices);
}

}  // namespace

at::Tensor AtenSliceScatter(const at::Tensor& self, const at::Tensor& src,
                            int64_t dim, std::optional<int64_t> start,
                            std::optional<int64_t> end, int64_t step) {
  const int64_t ndim = self.dim();
  TT_ASSIGN_OR_THROW(const int64_t wrapped_dim, SafeWrapDim(dim, ndim));
  const int64_t dim_size = self.size(wrapped_dim);

  TT_CHECK_THROW(step > 0, error::kInvalidArgument)
      << "slice step must be positive";

  int64_t start_val = start.has_value() ? *start : 0;
  int64_t end_val = end.has_value() ? *end : dim_size;

  if (start_val < 0) {
    start_val += dim_size;
  }
  if (end_val < 0) {
    end_val += dim_size;
  }
  if (start_val < 0) {
    start_val = 0;
  } else if (start_val >= dim_size) {
    start_val = dim_size;
  }
  if (end_val < start_val) {
    end_val = start_val;
  } else if (end_val >= dim_size) {
    end_val = dim_size;
  }
  const int64_t slice_len = (end_val - start_val + step - 1) / step;

  Dimensions expected_slice_sizes = CopyIntVector(self.sizes());
  expected_slice_sizes[wrapped_dim] = slice_len;
  TT_CHECK_THROW(src.sizes() == c10::IntArrayRef(expected_slice_sizes),
                 error::kInvalidArgument)
      << "expected src to have a size equal to the slice of self. src size = "
      << src.sizes()
      << ", slice size = " << c10::IntArrayRef(expected_slice_sizes);

  TT_CHECK_THROW(step == 1, error::kPythonNotImplementedError)
      << "slice_scatter with step != 1 is not supported yet";

  const at::Tensor src_tensor =
      src.device() == self.device() ? src : src.to(self.device());
  const Dimensions self_sizes = CopyIntVector(self.sizes());

  TT_KERNEL(
      OpName::kSliceScatter, param_keys, (self, src, dim, start, end, step), {
        TT_ASSIGN_OR_THROW(const mlir::ElementType element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto op_builder = [self_sizes, wrapped_dim,
                           start_val](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [self_op, src_op] = inputs;
          return BuildSliceScatterShlo(self_op, src_op, self_sizes, wrapped_dim,
                                       start_val);
        };

        TT_ASSIGN_OR_THROW(
            DeviceBufferRef out_buf,
            (DispatchOp<2>(std::move(op_builder), {self, src_tensor},
                           {.out_dtype = element_type,
                            .out_dims = self_sizes,
                            .op_param_cache_keys = std::move(param_keys)})));
        return MakeTensor(std::move(out_buf));
      });
}

at::Tensor AtenSelectScatter(const at::Tensor& self, const at::Tensor& src,
                             int64_t dim, int64_t index) {
  const int64_t ndim = self.dim();
  TT_ASSIGN_OR_THROW(const int64_t wrapped_dim, SafeWrapDim(dim, ndim));
  const int64_t dim_size = self.size(wrapped_dim);
  if (index < 0) {
    index += dim_size;
  }
  TT_CHECK_THROW(index >= 0 && index < dim_size, error::kPythonIndexError)
      << "index out of range";

  Dimensions expected_slice_sizes;
  expected_slice_sizes.reserve(self.dim() - 1);
  for (int64_t i = 0; i < self.dim(); ++i) {
    if (i != wrapped_dim) {
      expected_slice_sizes.push_back(self.size(i));
    }
  }
  TT_CHECK_THROW(src.sizes() == c10::IntArrayRef(expected_slice_sizes),
                 error::kInvalidArgument)
      << "expected src to have a size equal to the slice of self. src size = "
      << src.sizes()
      << ", slice size = " << c10::IntArrayRef(expected_slice_sizes);

  const at::Tensor src_tensor =
      src.device() == self.device() ? src : src.to(self.device());
  const Dimensions self_sizes = CopyIntVector(self.sizes());

  TT_KERNEL(OpName::kSelectScatter, param_keys, (self, src, dim, index), {
    TT_ASSIGN_OR_THROW(const mlir::ElementType element_type,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));

    auto op_builder = [self_sizes, wrapped_dim,
                       index](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [self_op, src_op] = inputs;
      return BuildSelectScatterShlo(self_op, src_op, self_sizes, wrapped_dim,
                                    index);
    };

    TT_ASSIGN_OR_THROW(
        DeviceBufferRef out_buf,
        (DispatchOp<2>(std::move(op_builder), {self, src_tensor},
                       {.out_dtype = element_type,
                        .out_dims = self_sizes,
                        .op_param_cache_keys = std::move(param_keys)})));
    return MakeTensor(std::move(out_buf));
  });
}

}  // namespace torch_tpu
