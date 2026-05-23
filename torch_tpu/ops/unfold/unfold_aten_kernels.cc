// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/ops/unfold/unfold_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/status/statusor.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/Optional.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/as_strided/as_strided_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildUnfoldBackwardShlo(
    mlir::MlirOp grad_in, SmallInt64Vector input_sizes, int64_t dimension,
    int64_t size, int64_t step) {
  auto& builder = grad_in.getBuilder();
  auto& op_builder = builder.getOpBuilder();

  const auto grad_in_type = GetTensorTypeOrDie(grad_in);
  const auto grad_in_shape = grad_in_type.getShape();
  const int64_t rank = input_sizes.size();

  TT_ASSIGN_OR_RETURN(const int64_t normalized_dim,
                      SafeWrapDim(dimension, rank));

  SmallInt64Vector permutation;
  int64_t batch = 1;
  for (int i = 0; i < rank; ++i) {
    if (i != normalized_dim) {
      permutation.push_back(i);
      batch *= input_sizes[i];
    }
  }
  permutation.push_back(rank);
  permutation.push_back(normalized_dim);

  const int64_t num_windows = grad_in_shape[normalized_dim];

  auto transposed_grad_in = mlir::stablehlo::Transpose(grad_in, permutation);
  auto reshaped_grad_in =
      mlir::stablehlo::Reshape(transposed_grad_in, {batch, size, num_windows});

  const auto iota = mlir::stablehlo::Iota(
      builder, mlir::RankedTensorType::get({size}, op_builder.getI32Type()), 0);
  const auto matrix_type =
      mlir::RankedTensorType::get({size, size}, op_builder.getI32Type());
  auto iota_row_reshaped = mlir::stablehlo::Reshape(iota, {size, 1});
  auto iota_row =
      mlir::stablehlo::BroadcastInDim(matrix_type, iota_row_reshaped, {0, 1});
  auto iota_col_reshaped = mlir::stablehlo::Reshape(iota, {1, size});
  auto iota_col =
      mlir::stablehlo::BroadcastInDim(matrix_type, iota_col_reshaped, {0, 1});
  const auto identity_bool = mlir::stablehlo::Compare(
      iota_row, iota_col, mlir::stablehlo::ComparisonDirection::EQ);
  const auto element_type = grad_in_type.getElementType();
  const auto identity =
      mlir::stablehlo::ConvertElementType(identity_bool, element_type);

  auto weights = mlir::stablehlo::Reshape(identity, {1, size, size});

  const int64_t pad_left = size - 1;
  const int64_t normalized_dim_size = input_sizes[normalized_dim];
  const int64_t pad_right = normalized_dim_size - 1 - (num_windows - 1) * step;

  const auto padding_type =
      mlir::RankedTensorType::get({1, 2}, op_builder.getI64Type());
  const auto padding_attr =
      mlir::DenseIntElementsAttr::get(padding_type, {pad_left, pad_right});

  const auto conv_output_type = mlir::RankedTensorType::get(
      {batch, 1, normalized_dim_size}, element_type);

  auto conv = mlir::stablehlo::Convolution(
      conv_output_type, reshaped_grad_in, weights,
      mlir::stablehlo::ConvDimensionNumbersAttr::get(
          &builder.getContext(),
          /*inputBatchDimension=*/0,
          /*inputFeatureDimension=*/1,
          /*inputSpatialDimensions=*/{2},
          /*kernelInputFeatureDimension=*/1,
          /*kernelOutputFeatureDimension=*/0,
          /*kernelSpatialDimensions=*/{2},
          /*outputBatchDimension=*/0,
          /*outputFeatureDimension=*/1,
          /*outputSpatialDimensions=*/{2}),
      /*feature_group_count=*/1,
      /*batch_group_count=*/1,
      /*window_strides=*/op_builder.getDenseI64ArrayAttr({1}),
      /*padding=*/padding_attr,
      /*lhs_dilation=*/op_builder.getDenseI64ArrayAttr({step}),
      /*rhs_dilation=*/op_builder.getDenseI64ArrayAttr({1}),
      /*window_reversal=*/op_builder.getDenseBoolArrayAttr({true}),
      /*precision_config=*/{});

  SmallInt64Vector reshaped_conv_shape;
  for (int i = 0; i < rank; ++i) {
    if (i != normalized_dim) {
      reshaped_conv_shape.push_back(input_sizes[i]);
    }
  }
  reshaped_conv_shape.push_back(normalized_dim_size);

  auto reshaped_conv = mlir::stablehlo::Reshape(conv, reshaped_conv_shape);

  SmallInt64Vector inv_permutation;
  for (int i = 0; i < normalized_dim; ++i) {
    inv_permutation.push_back(i);
  }
  inv_permutation.push_back(rank - 1);
  for (int i = 0; i < rank - 1 - normalized_dim; ++i) {
    inv_permutation.push_back(normalized_dim + i);
  }

  return mlir::stablehlo::Transpose(reshaped_conv, inv_permutation);
}

}  // namespace

at::Tensor AtenUnfold(const at::Tensor& self, int64_t dimension, int64_t size,
                      int64_t step) {
  TT_KERNEL(
      OpName::kUnfold, _,
      (self, IgnoreInCacheKey(dimension, "Delegates to AtenAsStrided"),
       IgnoreInCacheKey(size, "Delegates to AtenAsStrided"),
       IgnoreInCacheKey(step, "Delegates to AtenAsStrided")),
      {
        const bool self_is_scalar = self.dim() == 0;
        at::Tensor self_reshaped = self_is_scalar ? self.unsqueeze(0) : self;
        int64_t dims = self_reshaped.dim();

        TT_CHECK_THROW(step > 0, error::kInvalidArgument)
            << "expected step > 0, got " << step;

        TT_CHECK_THROW(dimension >= -dims && dimension < dims,
                       error::kOutOfRange)
            << "expected dimension to be in range of [" << -dims << ", "
            << dims - 1 << "] for shape " << self.sizes() << ", got "
            << dimension;

        TT_ASSIGN_OR_THROW(const int64_t normalized_dim,
                           SafeWrapDim(dimension, dims));
        TT_CHECK_THROW(size <= self_reshaped.size(normalized_dim),
                       error::kInvalidArgument)
            << "expected size <= dimension size (shape[" << dimension
            << "]: " << self_reshaped.size(normalized_dim)
            << "), got size: " << size;

        if (self_is_scalar) {
          return size == 0 ? at::empty({0}, self.options()) : self_reshaped;
        }

        // Compute the new layout.
        const int64_t num_windows =
            (self.size(normalized_dim) - size) / step + 1;
        Dimensions new_size = CopyIntVector(self.sizes());
        Strides new_stride = CopyIntVector(self.strides());
        // The unfolded dimension goes from old_size, old_stride
        // to num_windows, old_stride * step.
        new_size[normalized_dim] = num_windows;
        new_stride[normalized_dim] *= step;
        // A new dimension is added with the given size and the same stride as
        // the pre-unfolded dimension.
        new_size.push_back(size);
        new_stride.push_back(self.stride(normalized_dim));

        c10::SymIntArrayRef new_size_sym =
            c10::fromIntArrayRefKnownNonNegative(new_size);
        c10::SymIntArrayRef new_stride_sym =
            c10::fromIntArrayRefKnownNonNegative(new_stride);
        c10::optional<c10::SymInt> storage_offset_sym_opt = c10::nullopt;

        return AtenAsStrided(self, new_size_sym, new_stride_sym,
                             storage_offset_sym_opt);
      });
}

at::Tensor AtenUnfoldBackward(const at::Tensor& grad_in,
                              at::IntArrayRef input_sizes, int64_t dimension,
                              int64_t size, int64_t step) {
  TT_KERNEL(
      OpName::kUnfoldBackward, param_keys,
      (grad_in, input_sizes, dimension, size, step), {
        if (input_sizes.empty()) {
          // If the original input was a scalar (shape []), we are in a scalar
          // corner case.
          if (size == 0) {
            // Case A: No gradient was propagated during forward pass because
            // window size is 0. We return a constant scalar zero tensor [].
            return AtenEfficientZeroTensor(
                {}, grad_in.scalar_type(), /*layout_opt=*/c10::nullopt,
                grad_in.device(), /*pin_memory_opt=*/c10::nullopt);
          }
          // Case B: The forward pass unsqueezed the scalar to [1] to extract
          // windows of size > 0. grad_in has shape [1] (after autograd
          // squeeze). To return a valid scalar gradient [], we reshape/squeeze
          // the dummy dimension [1] -> [] using direct stablehlo::Reshape to
          // completely avoid delegating to autograd squeeze(0) / as_strided.
          auto squeeze_builder =
              [](mlir::MlirOp grad_in) -> absl::StatusOr<mlir::MlirOp> {
            return mlir::stablehlo::Reshape(grad_in, {});
          };
          TT_ASSIGN_OR_THROW(const auto elem_type, ConvertTo<mlir::ElementType>(
                                                       grad_in.scalar_type()));
          TT_ASSIGN_OR_THROW(
              DeviceBufferRef result,
              DispatchOp<1>(std::move(squeeze_builder), {grad_in},
                            {.out_dtype = elem_type,
                             .out_dims = {},
                             .op_param_cache_keys = std::move(param_keys)}));
          return MakeTensor(std::move(result));
        }

        if (grad_in.numel() == 0) {
          return AtenEfficientZeroTensor(
              input_sizes, grad_in.scalar_type(), /*layout_opt=*/c10::nullopt,
              grad_in.device(), /*pin_memory_opt=*/c10::nullopt);
        }

        Dimensions out_dims = CopyIntVector(input_sizes);

        auto op_builder = [input_sizes = CopyIntVector(input_sizes), dimension,
                           size, step](mlir::MlirOp grad_in) {
          return BuildUnfoldBackwardShlo(grad_in, input_sizes, dimension, size,
                                         step);
        };

        TT_ASSIGN_OR_THROW(const auto elem_type,
                           ConvertTo<mlir::ElementType>(grad_in.scalar_type()));

        TT_ASSIGN_OR_THROW(
            DeviceBufferRef result,
            DispatchOp<1>(std::move(op_builder), {grad_in},
                          {.out_dtype = elem_type,
                           .out_dims = out_dims,
                           .op_param_cache_keys = std::move(param_keys)}));

        return MakeTensor(std::move(result));
      });
}

}  // namespace torch_tpu
