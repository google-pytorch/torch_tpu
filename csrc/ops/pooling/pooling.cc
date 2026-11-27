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

#include "csrc/ops/pooling/pooling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/log/absl_check.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

// Adds a batch dimension of size 1 if the input tensor doesn't have it.
// Returns the batch input and the original dimension size.
absl::StatusOr<BatchInput> CreateBatchInput(mlir::MlirOp input,
                                            const int64_t spatial_dim_count) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  auto num_dims = input_type.getRank();

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Rank is already checked by
                 // GetPoolingOutputSize.
      num_dims == spatial_dim_count + 1 || num_dims == spatial_dim_count + 2,
      error::kInvalidArgument)
      << "expected non-empty " << spatial_dim_count + 1 << "D or "
      << spatial_dim_count + 2 << "D (batch mode) tensor for input, got "
      << num_dims << "D tensor";

  // If the input doesn't have a batch dimension, add a size 1 batch dimension
  if (num_dims == spatial_dim_count + 1) {
    Dimensions new_shape = {1};
    new_shape.insert(new_shape.end(), input_type.getShape().begin(),
                     input_type.getShape().end());
    auto batch_input = mlir::stablehlo::Reshape(input, new_shape);
    return BatchInput{batch_input, num_dims};
  }
  return BatchInput{input, num_dims};
}

namespace {

enum class EmptyAttrPolicy {
  kReject,
  kAllow,
};

std::string FormatExpectedAttrLength(const int64_t spatial_dim_count,
                                     const EmptyAttrPolicy empty_policy) {
  if (empty_policy == EmptyAttrPolicy::kAllow) {
    return spatial_dim_count == 1
               ? "0 or 1 elements"
               : absl::StrCat("0, 1, or ", spatial_dim_count, " elements");
  }
  return spatial_dim_count == 1
             ? "1 element"
             : absl::StrCat("1 or ", spatial_dim_count, " elements");
}

absl::Status ValidatePoolingAttrLength(const at::IntArrayRef attr,
                                       const std::string_view name,
                                       const int64_t spatial_dim_count,
                                       const EmptyAttrPolicy empty_policy) {
  const bool valid =
      (empty_policy == EmptyAttrPolicy::kAllow && attr.empty()) ||
      attr.size() == 1 || attr.size() == static_cast<size_t>(spatial_dim_count);
  TT_RET_CHECK(valid, error::kInvalidArgument)
      << "expected " << name << " to have "
      << FormatExpectedAttrLength(spatial_dim_count, empty_policy) << ", got "
      << attr.size();
  return absl::OkStatus();
}

}  // namespace

// Expands the attribute to match the dim size.
absl::StatusOr<Dimensions> ExpandAttribute(const Dimensions& values,
                                           const int64_t dim) {
  if (values.size() == 1 && dim > 1) {
    return Dimensions(dim, values[0]);
  }

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Validated in GetPoolingOutputSize.
      values.size() == static_cast<size_t>(dim), error::kInvalidArgument)
      << "expected pooling attribute to have "
      << FormatExpectedAttrLength(dim, EmptyAttrPolicy::kReject) << ", got "
      << values.size();
  return values;
}

// Removes the batch dimension if it doesn't have it at the beginning.
mlir::MlirOp RemoveTrivialBatch(mlir::MlirOp batch_op,
                                int64_t original_dim_size,
                                int64_t spatial_dim_count) {
  if (original_dim_size == spatial_dim_count + 2) {
    return batch_op;
  }

  const mlir::RankedTensorType current_type = GetTensorTypeOrDie(batch_op);
  llvm::SmallVector<int64_t> squeezed_shape;
  squeezed_shape.append(current_type.getShape().begin() + 1,
                        current_type.getShape().end());
  return mlir::stablehlo::Reshape(batch_op, squeezed_shape);
}

namespace {

// Returns the extra right-side padding elements needed so `ceil_mode` includes
// any partial trailing window, or 0 if no extra padding is needed or if that
// trailing window would start entirely in right-padding (`>= padding +
// input_size`, matching PyTorch's sliding-window rule).
int64_t ComputeExtraCeilModePadding(const int64_t input_size,
                                    const int64_t kernel_size,
                                    const int64_t stride, const int64_t padding,
                                    const int64_t dilation) {
  if (stride <= 1) {
    return 0;
  }
  const int64_t k_eff = (kernel_size - 1) * dilation + 1;
  const int64_t output_size_num = input_size + 2 * padding - k_eff;
  int64_t output_size_rem = output_size_num % stride;
  if (output_size_rem < 0) {
    output_size_rem += stride;
  }
  const int64_t extra_padding = stride - output_size_rem;
  return (output_size_rem != 0 && extra_padding < k_eff - padding)
             ? extra_padding
             : 0;
}

// Computes the scalar i32 runtime extra right padding for `ceil_mode` on a
// dynamic spatial dimension of size `input_size_i32`, mirroring
// `ComputeExtraCeilModePadding`.
mlir::MlirOp ComputeDynamicExtraCeilModePadding(mlir::MlirOp input_size_i32,
                                                const int64_t kernel_size,
                                                const int64_t stride,
                                                const int64_t padding,
                                                const int64_t dilation) {
  mlir::MlirBuilder& builder = input_size_i32.getBuilder();
  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  const int64_t k_eff = (kernel_size - 1) * dilation + 1;

  mlir::MlirOp zero = MakeScalarConstant(builder, 0, i32_type);
  mlir::MlirOp stride_op = MakeScalarConstant(builder, stride, i32_type);
  mlir::MlirOp max_extra_op =
      MakeScalarConstant(builder, k_eff - padding, i32_type);
  mlir::MlirOp offset_op =
      MakeScalarConstant(builder, 2 * padding - k_eff, i32_type);

  mlir::MlirOp num = mlir::stablehlo::Add(input_size_i32, offset_op);
  mlir::MlirOp rem = mlir::stablehlo::Rem(num, stride_op);
  mlir::MlirOp rem_neg = mlir::stablehlo::Compare(
      rem, zero, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp rem_plus_stride = mlir::stablehlo::Add(rem, stride_op);
  rem = mlir::stablehlo::Select(rem_neg, rem_plus_stride, rem);

  mlir::MlirOp rem_nz = mlir::stablehlo::Compare(
      rem, zero, mlir::stablehlo::ComparisonDirection::NE);
  mlir::MlirOp cand_extra = mlir::stablehlo::Subtract(stride_op, rem);
  mlir::MlirOp in_image = mlir::stablehlo::Compare(
      cand_extra, max_extra_op, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp use_extra = mlir::stablehlo::And(rem_nz, in_image);
  return mlir::stablehlo::Select(use_extra, cand_extra, zero);
}

}  // namespace

// Calculates the adjusted (left_padding, right_padding) pairs given the padding
// (which is symmetric) and ceil mode.
// Additional high padding could be required when ceil mode is set.
// This ensures that the last pooling is fully inside the image.
std::vector<PaddingParams> CeilModePadding(
    const mlir::RankedTensorType& input_shape,  // (N, C, H, W)/(N, C, D, H, W)
    const mlir::ArrayRef<int64_t> kernel_size,  // [K_h, K_w]
    const mlir::ArrayRef<int64_t> stride,       // [S_h, S_w]
    const mlir::ArrayRef<int64_t> padding,      // [P_h, P_w]
    const mlir::ArrayRef<int64_t> dilation,     // [D_h, D_w]
    const bool ceil_mode) {
  std::vector<PaddingParams> ceil_mode_padding;
  ceil_mode_padding.reserve(padding.size());

  for (size_t i = 0; i < padding.size(); ++i) {
    const int64_t left_padding = padding[i];
    int64_t right_padding = padding[i];
    const int64_t spatial_dim = 2 + static_cast<int64_t>(i);

    if (ceil_mode && stride[i] > 1 && !input_shape.isDynamicDim(spatial_dim)) {
      right_padding += ComputeExtraCeilModePadding(
          input_shape.getDimSize(spatial_dim), kernel_size[i], stride[i],
          padding[i], dilation[i]);
    }
    ceil_mode_padding.push_back({left_padding, right_padding});
  }

  return ceil_mode_padding;
}

std::vector<PaddingParams> CeilModePadding(
    const mlir::RankedTensorType& input_shape,
    const PoolingWindowParams& params) {
  return CeilModePadding(input_shape, params.kernel_size, params.stride,
                         params.padding, params.dilation, params.ceil_mode);
}

bool HasDynamicCeilModePadding(const mlir::RankedTensorType& input_shape,
                               const PoolingWindowParams& params) {
  if (!params.ceil_mode) {
    return false;
  }
  for (size_t i = 0; i < params.stride.size(); ++i) {
    const int64_t spatial_dim = 2 + static_cast<int64_t>(i);
    if (params.stride[i] > 1 && input_shape.isDynamicDim(spatial_dim)) {
      return true;
    }
  }
  return false;
}

mlir::MlirOp PadDynamicCeilModeSpatialDims(mlir::MlirOp operand,
                                           mlir::MlirOp pad_value,
                                           mlir::MlirOp batch_input,
                                           const PoolingWindowParams& params) {
  const mlir::RankedTensorType batch_type = GetTensorTypeOrDie(batch_input);
  if (!HasDynamicCeilModePadding(batch_type, params)) {
    return operand;
  }

  const int64_t rank = GetTensorTypeOrDie(operand).getRank();
  ABSL_CHECK_EQ(rank, batch_type.getRank())  // CRASH_OK
      << "Expected operand rank to match batch_input rank. This is a TorchTPU "
         "bug.";
  const Dimensions batch_bounds = GetStaticShape(batch_input);

  StaticAtBounds static_operand = MakeStaticAtBounds(operand);
  std::vector<std::optional<mlir::MlirOp>> mask_sizes(rank);

  Dimensions high_pad(rank, 0);
  bool needs_static_pad = false;

  for (size_t i = 0; i < params.stride.size(); ++i) {
    const int64_t dim = 2 + static_cast<int64_t>(i);
    if (params.stride[i] <= 1 || !batch_type.isDynamicDim(dim)) {
      continue;
    }
    ABSL_CHECK(static_operand.runtime_sizes[dim].has_value())  // CRASH_OK
        << "Expected operand dimension " << dim
        << " to be dynamic when batch_input dimension " << dim
        << " is dynamic. This is a TorchTPU bug.";
    mlir::MlirOp cur_size = *static_operand.runtime_sizes[dim];
    mask_sizes[dim] = cur_size;

    const int64_t bound_extra = ComputeExtraCeilModePadding(
        batch_bounds[dim], params.kernel_size[i], params.stride[i],
        params.padding[i], params.dilation[i]);
    if (bound_extra > 0) {
      high_pad[dim] = bound_extra;
      needs_static_pad = true;
    }

    mlir::MlirOp batch_dim_size =
        mlir::stablehlo::GetDimensionSize(batch_input, dim);
    mlir::MlirOp runtime_extra = ComputeDynamicExtraCeilModePadding(
        batch_dim_size, params.kernel_size[i], params.stride[i],
        params.padding[i], params.dilation[i]);
    static_operand.runtime_sizes[dim] =
        mlir::stablehlo::Add(cur_size, runtime_extra);
  }

  mlir::MlirOp result =
      FillPastRuntimeSizes(static_operand.op, pad_value, mask_sizes);

  if (needs_static_pad) {
    const Dimensions zero_pad(rank, 0);
    result =
        mlir::stablehlo::Pad(result, pad_value, zero_pad, high_pad, zero_pad);
  }

  return RestoreDynamicDimSizes(result, static_operand.runtime_sizes);
}

// Creates the padding config to be used to pad the input tensor and iota tensor
// consistently, which is also required by StableHLO's ReduceWindowOp.
ReduceWindowAttributes GetReduceWindowAttributes(
    mlir::MlirBuilder& builder, Dimensions kernel_size_attr,
    Dimensions stride_attr, Dimensions dilation_attr,
    std::vector<PaddingParams> padding_pairs, const int64_t spatial_dim_count,
    const int64_t total_num_dims) {
  Dimensions kernel_size(2, 1);
  kernel_size.insert(kernel_size.end(), kernel_size_attr.begin(),
                     kernel_size_attr.end());

  Dimensions stride;
  stride.resize(2, 1);
  stride.insert(stride.end(), stride_attr.begin(), stride_attr.end());

  Dimensions dilation(2, 1);
  dilation.insert(dilation.end(), dilation_attr.begin(), dilation_attr.end());

  Dimensions base_dilations(total_num_dims, 1);

  // Create a flat list of padding values for StableHLO's ReduceWindowOp
  llvm::SmallVector<int64_t> flat_padding_list;
  flat_padding_list.reserve(total_num_dims * 2);

  for (int i = 0; i < total_num_dims - spatial_dim_count; ++i) {
    flat_padding_list.push_back(0);
    flat_padding_list.push_back(0);
  }
  for (const auto& dim_padding : padding_pairs) {
    flat_padding_list.push_back(dim_padding.left);   // low
    flat_padding_list.push_back(dim_padding.right);  // high
  }

  auto reduce_padding_attr = mlir::DenseIntElementsAttr::get(
      mlir::makeTensorType(builder.getContext(), {total_num_dims, 2},
                           builder.getOpBuilder().getI64Type()),
      flat_padding_list);

  return ReduceWindowAttributes{
      .window_dimensions =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), kernel_size),
      .window_strides =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), stride),
      .base_dilations =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), base_dilations),
      .window_dilations =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), dilation),
      .padding = reduce_padding_attr};
}

namespace {

absl::int128 GetSingleDim(const int64_t in, const int64_t k_eff,
                          const int64_t s, const int64_t p,
                          const bool ceil_mode) {
  // Caller guarantees 0 <= 2 * p <= k_eff <= INT64_MAX and s > 0, so
  // `numerator` is in [-INT64_MAX, INT64_MAX] and cannot overflow int64_t.
  const int64_t numerator = in + (2 * p - k_eff);
  const int64_t quotient = ceil_mode ? llvm::divideCeilSigned(numerator, s)
                                     : llvm::divideFloorSigned(numerator, s);
  absl::int128 out = absl::int128(quotient) + 1;
  if (ceil_mode && (out - 1) * s >= absl::int128(in) + p) {
    --out;
  }
  return out;
}

absl::StatusOr<int64_t> ValidateAndComputeSpatialOutputDim(
    const int64_t in, const int64_t dim_index, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    const bool ceil_mode) {
  const int64_t k =
      kernel_size.size() == 1 ? kernel_size[0] : kernel_size[dim_index];
  const int64_t s =
      stride.empty() ? k : (stride.size() == 1 ? stride[0] : stride[dim_index]);
  const int64_t p = padding.size() == 1 ? padding[0] : padding[dim_index];
  const int64_t d = dilation.size() == 1 ? dilation[0] : dilation[dim_index];
  TT_RET_CHECK(k > 0, error::kInvalidArgument)
      << "expected kernel_size to be greater than zero, got " << k;
  TT_RET_CHECK(s > 0, error::kInvalidArgument)
      << "expected stride to be greater than zero, got " << s;
  TT_RET_CHECK(d > 0, error::kInvalidArgument)
      << "expected dilation to be greater than zero, got " << d;
  TT_RET_CHECK(p >= 0, error::kInvalidArgument)
      << "expected padding to be >= 0, got " << p;
  TT_RET_CHECK(p <= k / 2, error::kInvalidArgument)
      << "expected padding to be at most half of kernel_size, got padding=" << p
      << " and kernel_size=" << k;

  const absl::int128 k_eff_i128 = absl::int128(d) * (k - 1) + 1;
  TT_RET_CHECK(k_eff_i128 <= std::numeric_limits<int64_t>::max(),
               error::kInvalidArgument)
      << "expected effective kernel size to fit in int64, got kernel_size=" << k
      << " and dilation=" << d;
  const int64_t k_eff = static_cast<int64_t>(k_eff_i128);

  const absl::int128 out = GetSingleDim(in, k_eff, s, p, ceil_mode);
  TT_RET_CHECK(out >= 1 && out <= std::numeric_limits<int64_t>::max(),
               error::kInvalidArgument)
      << "calculated output size (" << out << ") is "
      << (out < 1 ? "too small" : "too large") << " for input size " << in
      << " at spatial dim " << dim_index;
  return static_cast<int64_t>(out);
}

}  // namespace

absl::StatusOr<Dimensions> GetPoolingOutputSize(
    at::IntArrayRef input_size, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    const bool ceil_mode, const int64_t spatial_dim_count) {
  const int64_t dims_num = input_size.size();
  TT_RET_CHECK(
      dims_num == spatial_dim_count + 1 || dims_num == spatial_dim_count + 2,
      error::kInvalidArgument)
      << "expected non-empty " << spatial_dim_count + 1 << "D or "
      << spatial_dim_count + 2 << "D (batch mode) tensor for input, got "
      << dims_num << "D tensor";

  TT_RETURN_IF_ERROR(ValidatePoolingAttrLength(
      kernel_size, "kernel_size", spatial_dim_count, EmptyAttrPolicy::kReject));
  TT_RETURN_IF_ERROR(ValidatePoolingAttrLength(
      stride, "stride", spatial_dim_count, EmptyAttrPolicy::kAllow));
  TT_RETURN_IF_ERROR(ValidatePoolingAttrLength(
      padding, "padding", spatial_dim_count, EmptyAttrPolicy::kReject));
  TT_RETURN_IF_ERROR(ValidatePoolingAttrLength(
      dilation, "dilation", spatial_dim_count, EmptyAttrPolicy::kReject));

  // If (N, C, ...), spatial starts at 2. If (C, ...), starts at 1.
  const int64_t spatial_start_index =
      (dims_num == spatial_dim_count + 2) ? 2 : 1;

  Dimensions output_size(dims_num);
  std::copy(input_size.begin(), input_size.begin() + spatial_start_index,
            output_size.begin());

  // Calculates Spatial Dims (D, H, W).
  for (int i = 0; i < spatial_dim_count; ++i) {
    TT_ASSIGN_OR_RETURN(output_size[spatial_start_index + i],
                        ValidateAndComputeSpatialOutputDim(
                            input_size[spatial_start_index + i], i, kernel_size,
                            stride, padding, dilation, ceil_mode));
  }
  return output_size;
}

}  // namespace torch_tpu
