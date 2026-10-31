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

#include "csrc/ops/upsample/upsample_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/UpSample.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/util/string_view.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/einsum/einsum_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

// Upsample nearest interpolation for 1D, 2D and 3D tensors.
// This implementation is based on creating a gather tensor from scaled iota
// tensors that are broadcast in each dimension. These iota tensors are
// concatenated into a single gather tensor with an index in the last dimension.
// The gather operation maps the input to the output.

namespace torch_tpu {

namespace {

enum class UpsampleMode {
  // Nearest mode uses the floor of the scaled coordinate.
  kNearest,
  // Nearest exact mode aligns the centers of the pixels by adding 0.5 before
  // scaling and applying the floor function.
  kNearestExact,
};

using MlirOpVector = absl::InlinedVector<mlir::MlirOp, 6>;
using TensorVector = absl::InlinedVector<at::Tensor, 6>;

// Configuration for the gather operation used in upsampling.
// Contains the dimension numbers for the gather op and the slice sizes
// for each dimension of the input tensor.
struct UpsampleGatherConfig {
  mlir::stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers;
  SmallInt64Vector slice_sizes;
};

struct UpsampleScatterConfig {
  mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers;
};

UpsampleScatterConfig GetUpsampleScatterConfig(mlir::MlirOp grad_output,
                                               int64_t spatial_dim_size) {
  int64_t num_output_dimensions =
      GetTensorTypeOrDie(grad_output).getShape().size();
  Dimensions scatter_dims_to_operand_dims;
  scatter_dims_to_operand_dims.reserve(spatial_dim_size);
  for (int i = 0; i < spatial_dim_size; ++i) {
    scatter_dims_to_operand_dims.push_back(i + num_output_dimensions -
                                           spatial_dim_size);
  }
  Dimensions batching_dims;
  batching_dims.reserve(num_output_dimensions - spatial_dim_size);
  for (int i = 0; i < num_output_dimensions - spatial_dim_size; ++i) {
    batching_dims.push_back(i);
  }
  Dimensions inserted_window_dims;
  inserted_window_dims.reserve(spatial_dim_size);
  for (int i = 0; i < spatial_dim_size; ++i) {
    inserted_window_dims.push_back(i + num_output_dimensions -
                                   spatial_dim_size);
  }

  mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers =
      mlir::stablehlo::ScatterDimensionNumbersAttr::get(
          &grad_output.getContext(),
          /*update_window_dims=*/{},
          /*inserted_window_dims=*/
          AsArrayRef<int64_t>(inserted_window_dims),
          /*input_batching_dims=*/AsArrayRef<int64_t>(batching_dims),
          /*scatter_indices_batching_dims=*/AsArrayRef<int64_t>(batching_dims),
          /*scatter_dims_to_operand_dims=*/
          AsArrayRef<int64_t>(scatter_dims_to_operand_dims),
          /*index_vector_dim=*/num_output_dimensions);
  return {scatter_dimension_numbers};
}

UpsampleGatherConfig GetUpsampleGatherConfig(mlir::MlirOp input,
                                             int64_t spatial_dim_size) {
  Dimensions input_shape = CopyIntVector(GetTensorTypeOrDie(input).getShape());
  int64_t offset_dim_size = input_shape.size() - spatial_dim_size;

  // We gather on the spatial dimensions.
  // slice_sizes will be 1 for spatial dims, and unchanged for offset dims.
  auto slice_sizes = input_shape;
  for (int i = 0; i < spatial_dim_size; ++i) {
    slice_sizes[offset_dim_size + i] = 1;
  }

  SmallInt64Vector collapsed_slice_dims;
  SmallInt64Vector start_index_map;
  for (int i = 0; i < spatial_dim_size; ++i) {
    collapsed_slice_dims.push_back(offset_dim_size + i);
    start_index_map.push_back(offset_dim_size + i);
  }

  SmallInt64Vector offset_dims;
  offset_dims.reserve(offset_dim_size);
  for (int i = 0; i < offset_dim_size; ++i) {
    offset_dims.push_back(i);
  }

  mlir::stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers =
      mlir::stablehlo::GatherDimensionNumbersAttr::get(
          &input.getContext(),
          /*offset_dims=*/AsArrayRef<int64_t>(offset_dims),
          /*collapsed_slice_dims=*/AsArrayRef<int64_t>(collapsed_slice_dims),
          /*operand_batching_dims=*/{},
          /*start_indices_batching_dims=*/{},
          /*start_index_map=*/AsArrayRef<int64_t>(start_index_map),
          /*index_vector_dim=*/spatial_dim_size);

  return {gather_dimension_numbers, slice_sizes};
}

constexpr int64_t kBilinearHeightDim = 2;
constexpr int64_t kBilinearWidthDim = 3;

// Inspired by `jax.image.scale` (`jax/_src/image/scale.py`),
// builds the [out_size, in_size] 1D linear interpolation weight matrix
//   A[out_idx, in_idx] =
//       lambda0[out_idx] * (in_idx == index0[out_idx]) +
//       lambda1[out_idx] * (in_idx == index1[out_idx])
// directly in StableHLO from `Iota` and scalar splat constants.
mlir::MlirOp BuildInterpolationMatrix(mlir::MlirBuilder& builder,
                                      int64_t in_size, int64_t out_size,
                                      bool align_corners,
                                      std::optional<double> scale,
                                      mlir::Type calc_type) {
  const double ratio =
      calc_type.isF64()
          ? at::native::area_pixel_compute_scale<double>(in_size, out_size,
                                                         align_corners, scale)
          : static_cast<double>(at::native::area_pixel_compute_scale<float>(
                in_size, out_size, align_corners, scale));

  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  const mlir::RankedTensorType weight_vec_type =
      mlir::RankedTensorType::get({out_size}, calc_type);
  const mlir::RankedTensorType index_mat_type =
      mlir::RankedTensorType::get({out_size, in_size}, i32_type);
  const mlir::RankedTensorType weight_mat_type =
      mlir::RankedTensorType::get({out_size, in_size}, calc_type);

  mlir::MlirOp dst_idx =
      mlir::stablehlo::Iota(builder, weight_vec_type, /*iotaDimension=*/0);
  mlir::MlirOp ratio_vec = MakeConstant(builder, ratio, calc_type, out_size);
  mlir::MlirOp zero_vec = MakeConstant(builder, 0.0, calc_type, out_size);
  mlir::MlirOp one_vec = MakeConstant(builder, 1.0, calc_type, out_size);
  mlir::MlirOp src_idx;
  if (align_corners) {
    src_idx = mlir::stablehlo::Mul(dst_idx, ratio_vec);
  } else {
    mlir::MlirOp half_vec = MakeConstant(builder, 0.5, calc_type, out_size);
    mlir::MlirOp centered = mlir::stablehlo::Add(dst_idx, half_vec);
    mlir::MlirOp scaled = mlir::stablehlo::Mul(centered, ratio_vec);
    mlir::MlirOp unshifted = mlir::stablehlo::Subtract(scaled, half_vec);
    src_idx = mlir::stablehlo::Max(unshifted, zero_vec);
  }

  const int64_t max_in_idx = in_size - 1;
  mlir::MlirOp max_in_float = MakeConstant(
      builder, static_cast<double>(max_in_idx), calc_type, out_size);
  mlir::MlirOp floor_src = mlir::stablehlo::Floor(src_idx);
  mlir::MlirOp idx0_float = mlir::stablehlo::Min(floor_src, max_in_float);
  mlir::MlirOp frac = mlir::stablehlo::Subtract(src_idx, idx0_float);
  mlir::MlirOp lambda1 = mlir::stablehlo::Clamp(zero_vec, frac, one_vec);
  mlir::MlirOp lambda0 = mlir::stablehlo::Subtract(one_vec, lambda1);

  mlir::MlirOp index0 =
      mlir::stablehlo::ConvertElementType(idx0_float, i32_type);
  mlir::MlirOp one_i32 = MakeConstant(builder, 1, i32_type, out_size);
  mlir::MlirOp max_in_i32 =
      MakeConstant(builder, max_in_idx, i32_type, out_size);
  mlir::MlirOp index0_plus_one = mlir::stablehlo::Add(index0, one_i32);
  mlir::MlirOp index1 = mlir::stablehlo::Min(index0_plus_one, max_in_i32);

  mlir::MlirOp col_indices =
      mlir::stablehlo::Iota(builder, index_mat_type, /*iotaDimension=*/1);
  mlir::MlirOp zeros =
      MakeConstant(builder, 0.0, calc_type, {out_size, in_size});

  auto build_term = [col_indices, zeros, index_mat_type, weight_mat_type](
                        mlir::MlirOp index_vec,
                        mlir::MlirOp weight_vec) mutable {
    mlir::MlirOp index_mat =
        mlir::stablehlo::BroadcastInDim(index_mat_type, index_vec, {0});
    mlir::MlirOp weight_mat =
        mlir::stablehlo::BroadcastInDim(weight_mat_type, weight_vec, {0});
    mlir::MlirOp is_tap = mlir::stablehlo::Compare(
        col_indices, index_mat, mlir::stablehlo::ComparisonDirection::EQ);
    return mlir::stablehlo::Select(is_tap, weight_mat, zeros);
  };

  mlir::MlirOp term0 = build_term(index0, lambda0);
  mlir::MlirOp term1 = build_term(index1, lambda1);
  return mlir::stablehlo::Add(term0, term1);
}

// Interpolates `operand` along `spatial_dim` (H=2 or W=3) by contracting it
// with the [out_size, in_size] 1D weight matrix (or its transpose when
// `is_backward` is true).
absl::StatusOr<mlir::MlirOp> InterpolateSpatialDim(
    mlir::MlirOp operand, int64_t in_size, int64_t out_size, bool align_corners,
    std::optional<double> scale, int64_t spatial_dim,
    mlir::stablehlo::Precision precision, bool is_backward) {
  ABSL_DCHECK(spatial_dim == kBilinearHeightDim ||
              spatial_dim == kBilinearWidthDim);
  ABSL_DCHECK_EQ(GetTensorTypeOrDie(operand).getRank(), 4);
  if (in_size == out_size) {
    return operand;
  }
  mlir::MlirBuilder& builder = operand.getBuilder();
  const mlir::Type calc_type = GetTensorTypeOrDie(operand).getElementType();
  mlir::MlirOp matrix = BuildInterpolationMatrix(
      builder, in_size, out_size, align_corners, scale, calc_type);
  const c10::string_view equation =
      is_backward ? (spatial_dim == kBilinearHeightDim ? "ncow,oh->nchw"
                                                       : "ncho,ow->nchw")
                  : (spatial_dim == kBilinearHeightDim ? "nchw,oh->ncow"
                                                       : "nchw,ow->ncho");
  return BuildEinsumShlo({operand, matrix}, equation, precision);
}

// Builds the 2D bilinear interpolation (or its adjoint when `is_backward` is
// true) of 4D NCHW `operand`. `in_h`/`in_w` and `out_h`/`out_w` are always the
// forward input and output spatial sizes.
absl::StatusOr<mlir::MlirOp> BuildBilinear2d(
    mlir::MlirOp operand, int64_t in_h, int64_t in_w, int64_t out_h,
    int64_t out_w, bool align_corners, std::optional<double> scale_h,
    std::optional<double> scale_w, mlir::stablehlo::Precision precision,
    bool is_backward) {
  if (in_h == out_h && in_w == out_w) {
    return operand;
  }

  mlir::MlirBuilder& builder = operand.getBuilder();
  const mlir::Type elem_type = GetTensorTypeOrDie(operand).getElementType();
  // Matches ATen's `at::opmath_type<scalar_t>` (`ATen/OpMathType.h`) and
  // `UpSampleKernel.cpp`, which promote narrow floating-point types (f16,
  // bf16, f8) and integer types (uint8) to f32 (and keep f64 as f64) for
  // intermediate interpolation and backward gradient accumulation.
  const mlir::Type calc_type = elem_type.isF64()
                                   ? builder.getOpBuilder().getF64Type()
                                   : builder.getOpBuilder().getF32Type();
  mlir::MlirOp result =
      elem_type == calc_type
          ? operand
          : mlir::stablehlo::ConvertElementType(operand, calc_type);

  TT_ASSIGN_OR_RETURN(result, InterpolateSpatialDim(
                                  result, in_h, out_h, align_corners, scale_h,
                                  kBilinearHeightDim, precision, is_backward));
  TT_ASSIGN_OR_RETURN(
      result, InterpolateSpatialDim(result, in_w, out_w, align_corners, scale_w,
                                    kBilinearWidthDim, precision, is_backward));

  return elem_type == calc_type
             ? result
             : mlir::stablehlo::ConvertElementType(result, elem_type);
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleNearestSingleDimIndexTensor(
    mlir::MlirBuilder& builder, int num_dimensions, int dimension,
    mlir::MlirOp grad_output, at::IntArrayRef grad_output_shape,
    const MlirOpVector& inverse_scale_factor_array,
    UpsampleMode upsample_mode) {
  auto inverse_scale_factor_float32 =
      mlir::stablehlo::ConvertElementType(inverse_scale_factor_array[dimension],
                                          builder.getOpBuilder().getF32Type());
  auto iota_type = mlir::RankedTensorType::get(
      grad_output_shape[dimension + 2], builder.getOpBuilder().getF32Type());
  auto iota = mlir::stablehlo::Iota(builder, iota_type, 0);
  mlir::MlirOp offset_iota = [&]() {  // REF_CAPTURE_OK=called locally
    if (upsample_mode != UpsampleMode::kNearestExact) {
      return iota;
    }
    float center_offset_val = 0.5f;
    mlir::MlirOp center_offset_op =
        MakeConstant(builder, center_offset_val, mlir::ElementType::F32,
                     grad_output_shape[dimension + 2]);
    return mlir::stablehlo::Add(iota, center_offset_op);
  }();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp scaled_iota,
                      BuildMulShlo(offset_iota, inverse_scale_factor_float32));
  return mlir::stablehlo::ConvertElementType(
      mlir::stablehlo::Floor(scaled_iota), builder.getOpBuilder().getI32Type());
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleNearestIndexTensor(
    mlir::MlirBuilder& builder, int num_dimensions, mlir::MlirOp input,
    at::IntArrayRef output_shape,
    const MlirOpVector& inverse_scale_factor_array,
    UpsampleMode upsample_mode) {
  Dimensions broadcast_shape_array(num_dimensions + 1);
  broadcast_shape_array[num_dimensions] = 1;
  for (int i = 0; i < num_dimensions; ++i) {
    broadcast_shape_array[i] = output_shape[i + 2];
  }
  auto broadcast_type = mlir::RankedTensorType::get(
      broadcast_shape_array, builder.getOpBuilder().getI32Type());

  // Generate the set of reshaped index tensors to form the gather tensor.
  MlirOpVector index_tensor_reshape_array(num_dimensions);
  for (int i = 0; i < num_dimensions; ++i) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp index_tensor,
                        BuildUpsampleNearestSingleDimIndexTensor(
                            builder, num_dimensions, i, input, output_shape,
                            inverse_scale_factor_array, upsample_mode));

    // Reshape and project the index tensors into the same shape for the gather
    // dimensions
    index_tensor_reshape_array[i] =
        mlir::stablehlo::BroadcastInDim(broadcast_type, index_tensor, {i});
  }

  // Concatenate the index tensors into a single index tensor with gather
  // dimension pairs in the third dimension. This creates a tensor where the
  // final dimension is of 1, 2 or 3-element tuples that address the input
  // tensor in its gather dimensions.
  return mlir::stablehlo::Concatenate(
      input.getBuilder(), index_tensor_reshape_array, output_shape.size() - 2);
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleNearestShlo(
    mlir::MlirOp input, at::IntArrayRef output_shape,
    const MlirOpVector& inverse_scale_factor_array,
    UpsampleMode upsample_mode) {
  mlir::MlirBuilder& builder = input.getBuilder();
  const int num_dimensions = inverse_scale_factor_array.size();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp index_tensor,
                      BuildUpsampleNearestIndexTensor(
                          builder, num_dimensions, input, output_shape,
                          inverse_scale_factor_array, upsample_mode));

  UpsampleGatherConfig gather_config =
      GetUpsampleGatherConfig(input, num_dimensions);
  auto& slice_sizes = gather_config.slice_sizes;
  auto& gather_dimension_numbers = gather_config.gather_dimension_numbers;

  return mlir::stablehlo::Gather(input, index_tensor, gather_dimension_numbers,
                                 slice_sizes,
                                 /*indices_are_sorted=*/false);
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleNearestBackwardIndexTensor(
    mlir::MlirBuilder& builder, int num_dimensions, mlir::MlirOp grad_output,
    at::IntArrayRef grad_input_shape,
    const MlirOpVector& inverse_scale_factor_array,
    UpsampleMode upsample_mode) {
  Dimensions grad_output_shape =
      CopyIntVector(GetTensorTypeOrDie(grad_output).getShape());
  int64_t rank = grad_output_shape.size();
  auto broadcast_to_indices_type = mlir::RankedTensorType::get(
      grad_output_shape, builder.getOpBuilder().getI32Type());

  // 1. Compute scatter indices
  MlirOpVector index_tensors_to_concat(num_dimensions);
  for (int i = 0; i < num_dimensions; ++i) {
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp index_tensor,
        BuildUpsampleNearestSingleDimIndexTensor(
            builder, num_dimensions, i, grad_output, grad_output_shape,
            inverse_scale_factor_array, upsample_mode));

    auto broadcast_indices = mlir::stablehlo::BroadcastInDim(
        broadcast_to_indices_type, index_tensor, {i + 2});

    SmallInt64Vector reshape_dims = grad_output_shape;
    reshape_dims.push_back(1);
    auto reshape_type = mlir::RankedTensorType::get(
        reshape_dims, builder.getOpBuilder().getI32Type());
    index_tensors_to_concat[i] =
        mlir::stablehlo::Reshape(reshape_type, broadcast_indices);
  }
  return mlir::stablehlo::Concatenate(builder, index_tensors_to_concat, rank);
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleNearestBackwardShlo(
    mlir::MlirOp grad_output, at::IntArrayRef grad_input_shape,
    const MlirOpVector& inverse_scale_factor_array,
    UpsampleMode upsample_mode) {
  mlir::MlirBuilder& builder = grad_output.getBuilder();
  const int num_dimensions = inverse_scale_factor_array.size();
  TT_ASSIGN_OR_RETURN(
      auto scatter_indices,
      BuildUpsampleNearestBackwardIndexTensor(
          builder, num_dimensions, grad_output, grad_input_shape,
          inverse_scale_factor_array, upsample_mode));

  // 2. Create zero tensor to accumulate the grad_input into
  auto grad_input_type = mlir::RankedTensorType::get(
      grad_input_shape, GetTensorTypeOrDie(grad_output).getElementType());
  auto element_type = grad_input_type.getElementType();
  auto grad_input_zeros =
      MakeConstant(builder, 0.0, element_type, grad_input_shape);

  // 3. Call scatter
  auto scatter_config = GetUpsampleScatterConfig(grad_output, num_dimensions);

  auto scatter_body = [element_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        element_type, rb.getRegion(), rb.getOpBuilder());
  };

  return mlir::stablehlo::Scatter(grad_input_zeros, scatter_indices,
                                  grad_output, scatter_body,
                                  scatter_config.scatter_dimension_numbers,
                                  /*indices_are_sorted=*/false)[0];
}

absl::InlinedVector<double, 6> StandardizeScaleFactors(
    at::IntArrayRef input_shape, at::IntArrayRef upsample_shape,
    absl::InlinedVector<std::optional<double>, 6> scale_array, bool inverse) {
  // Generate a scale factor from the output shape when scale factor not
  // provided.
  absl::InlinedVector<double, 6> scale_factor_result(upsample_shape.size());
  for (int i = 0; i < upsample_shape.size(); ++i) {
    double scale_factor = scale_array[i].value_or(
        static_cast<double>(upsample_shape[i]) / input_shape[i + 2]);

    float maybe_inverse_scale_factor =
        inverse ? (1.0 / scale_factor) : scale_factor;
    scale_factor_result[i] = maybe_inverse_scale_factor;
  }

  return scale_factor_result;
}

absl::StatusOr<TensorVector> ConstructScaleFactorArray(
    absl::InlinedVector<double, 6> scale_array) {
  // Generate a scale factor from the output shape when scale factor not
  // provided.

  TensorVector scale_factor_result(scale_array.size());
  for (int i = 0; i < scale_factor_result.size(); ++i) {
    at::Scalar maybe_inverse_scale_factor_scalar(scale_array[i]);
    TT_ASSIGN_OR_RETURN(scale_factor_result[i],
                        MakeTensor(maybe_inverse_scale_factor_scalar));
  }

  return scale_factor_result;
}

void CheckUpsampleDtypes(const at::Tensor& tensor) {
  TT_CHECK_THROW(tensor.scalar_type() != at::kBool &&
                     tensor.scalar_type() != at::kInt &&
                     tensor.scalar_type() != at::kLong,
                 error::kPythonNotImplementedError)
      << "not implemented for " << ToString(tensor.scalar_type());
}

}  // namespace

at::Tensor& AtenUpsampleNearest1dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_shape_in,
    at::IntArrayRef input_shape, std::optional<double> scale,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleNearest1dBackwardGradInput, param_keys,
      (grad_output, output_shape_in, input_shape, scale, grad_input), {
        auto output_shape = CopyIntVector(grad_output.sizes());

        // TODO(lwh): Add check for output_shape and scaled version from
        // input_shape, and check that grad_input and grad_output shapes match
        // those.
        auto preprocessed_scale_factors = StandardizeScaleFactors(
            input_shape, output_shape_in, {scale}, /*inverse=*/true);
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        auto op_builder =
            [output_shape, grad_input_shape = CopyIntVector(input_shape),
             preprocessed_scale_factors](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output, inverse_scale_factor_tensor] = inputs;
          TT_ASSIGN_OR_RETURN(
              auto grad_input,
              BuildUpsampleNearestBackwardShlo(grad_output, grad_input_shape,
                                               {inverse_scale_factor_tensor},
                                               UpsampleMode::kNearest));
          return grad_input;
        };
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<2>(std::move(op_builder),
                           {grad_output, inverse_scale_factors[0]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_shape),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

at::Tensor& AtenUpsampleNearest2dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_shape_in,
    at::IntArrayRef input_shape, std::optional<double> scale_h,
    std::optional<double> scale_w, at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleNearest2dBackwardGradInput, param_keys,
      (grad_output, output_shape_in, input_shape, scale_h, scale_w, grad_input),
      {
        auto output_shape = CopyIntVector(grad_output.sizes());
        // TODO(lwh): Add check for output_shape and scaled version from
        // input_shape, and check that grad_input and grad_output shapes match
        // those.
        auto preprocessed_scale_factors = StandardizeScaleFactors(
            input_shape, output_shape_in, {scale_h, scale_w}, /*inverse=*/true);
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        auto op_builder =
            [output_shape, grad_input_shape = CopyIntVector(input_shape),
             preprocessed_scale_factors](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w] = inputs;
          TT_ASSIGN_OR_RETURN(
              auto grad_input,
              BuildUpsampleNearestBackwardShlo(grad_output, grad_input_shape,
                                               {inverse_scale_factor_tensor_h,
                                                inverse_scale_factor_tensor_w},
                                               UpsampleMode::kNearest));
          return grad_input;
        };
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<3>(std::move(op_builder),
                           {grad_output, inverse_scale_factors[0],
                            inverse_scale_factors[1]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_shape),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

at::Tensor& AtenUpsampleNearest3dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_shape_in,
    at::IntArrayRef input_shape, std::optional<double> scale_h,
    std::optional<double> scale_w, std::optional<double> scale_d,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleNearest3dBackwardGradInput, param_keys,
      (grad_output, output_shape_in, input_shape, scale_h, scale_w, scale_d,
       grad_input),
      {
        auto output_shape = CopyIntVector(grad_output.sizes());
        // TODO(lwh): Add check for output_shape and scaled version from
        // input_shape, and check that grad_input and grad_output shapes match
        // those.
        auto preprocessed_scale_factors = StandardizeScaleFactors(
            input_shape, output_shape_in, {scale_h, scale_w, scale_d},
            /*inverse=*/true);
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        auto op_builder =
            [output_shape, grad_input_shape = CopyIntVector(input_shape),
             preprocessed_scale_factors](FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w, inverse_scale_factor_tensor_d] =
              inputs;
          TT_ASSIGN_OR_RETURN(
              auto grad_input,
              BuildUpsampleNearestBackwardShlo(
                  grad_output, grad_input_shape,
                  {inverse_scale_factor_tensor_h, inverse_scale_factor_tensor_w,
                   inverse_scale_factor_tensor_d},
                  UpsampleMode::kNearest));
          return grad_input;
        };
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<4>(std::move(op_builder),
                           {grad_output, inverse_scale_factors[0],
                            inverse_scale_factors[1], inverse_scale_factors[2]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_shape),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

at::Tensor& AtenUpsampleNearestExact1dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_shape_in,
    at::IntArrayRef input_shape, std::optional<double> scale,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact1dBackwardGradInput, param_keys,
      (grad_output, output_shape_in, input_shape, scale, grad_input), {
        auto output_shape = CopyIntVector(grad_output.sizes());
        // TODO(lwh): Add check for output_shape and scaled version from
        // input_shape, and check that grad_input and grad_output shapes match
        // those.
        auto preprocessed_scale_factors = StandardizeScaleFactors(
            input_shape, output_shape_in, {scale}, /*inverse=*/true);
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        auto op_builder =
            [output_shape, grad_input_shape = CopyIntVector(input_shape),
             preprocessed_scale_factors](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output, inverse_scale_factor_tensor] = inputs;
          TT_ASSIGN_OR_RETURN(
              auto grad_input,
              BuildUpsampleNearestBackwardShlo(grad_output, grad_input_shape,
                                               {inverse_scale_factor_tensor},
                                               UpsampleMode::kNearestExact));
          return grad_input;
        };
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<2>(std::move(op_builder),
                           {grad_output, inverse_scale_factors[0]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_shape),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

at::Tensor& AtenUpsampleNearestExact2dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_shape_in,
    at::IntArrayRef input_shape, std::optional<double> scale_h,
    std::optional<double> scale_w, at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact2dBackwardGradInput, param_keys,
      (grad_output, output_shape_in, input_shape, scale_h, scale_w, grad_input),
      {
        auto output_shape = CopyIntVector(grad_output.sizes());
        // TODO(lwh): Add check for output_shape and scaled version from
        // input_shape, and check that grad_input and grad_output shapes match
        // those.
        auto preprocessed_scale_factors = StandardizeScaleFactors(
            input_shape, output_shape_in, {scale_h, scale_w}, /*inverse=*/true);
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        auto op_builder =
            [output_shape, grad_input_shape = CopyIntVector(input_shape),
             preprocessed_scale_factors](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w] = inputs;
          TT_ASSIGN_OR_RETURN(
              auto grad_input,
              BuildUpsampleNearestBackwardShlo(grad_output, grad_input_shape,
                                               {inverse_scale_factor_tensor_h,
                                                inverse_scale_factor_tensor_w},
                                               UpsampleMode::kNearestExact));
          return grad_input;
        };
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<3>(std::move(op_builder),
                           {grad_output, inverse_scale_factors[0],
                            inverse_scale_factors[1]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_shape),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

at::Tensor& AtenUpsampleNearestExact3dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_shape_in,
    at::IntArrayRef input_shape, std::optional<double> scale_h,
    std::optional<double> scale_w, std::optional<double> scale_d,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact3dBackwardGradInput, param_keys,
      (grad_output, output_shape_in, input_shape, scale_h, scale_w, scale_d,
       grad_input),
      {
        auto output_shape = CopyIntVector(grad_output.sizes());
        // TODO(lwh): Add check for output_shape and scaled version from
        // input_shape, and check that grad_input and grad_output shapes match
        // those.
        auto preprocessed_scale_factors = StandardizeScaleFactors(
            input_shape, output_shape_in, {scale_h, scale_w, scale_d},
            /*inverse=*/true);
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        auto op_builder =
            [output_shape, grad_input_shape = CopyIntVector(input_shape),
             preprocessed_scale_factors](FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w, inverse_scale_factor_tensor_d] =
              inputs;
          TT_ASSIGN_OR_RETURN(
              auto grad_input,
              BuildUpsampleNearestBackwardShlo(
                  grad_output, grad_input_shape,
                  {inverse_scale_factor_tensor_h, inverse_scale_factor_tensor_w,
                   inverse_scale_factor_tensor_d},
                  UpsampleMode::kNearestExact));
          return grad_input;
        };
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<4>(std::move(op_builder),
                           {grad_output, inverse_scale_factors[0],
                            inverse_scale_factors[1], inverse_scale_factors[2]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_shape),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

// Lowers 2D bilinear interpolation as separable contractions along H and W:
// Y = A_h * X * A_w^T.
at::Tensor& AtenUpsampleBilinear2dOut(const at::Tensor& self,
                                      at::IntArrayRef upsample_shape,
                                      bool align_corners,
                                      std::optional<double> scale_h,
                                      std::optional<double> scale_w,
                                      at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleBilinear2dOut, param_keys,
      (self, upsample_shape, align_corners, scale_h, scale_w, out), {
        CheckUpsampleDtypes(self);
        TT_CHECK_THROW(self.dim() == 4 && out.dim() == 4,
                       error::kInvalidArgument)
            << "expected 4D input and out tensors, got input: "
            << ToString(self.dim()) << "D, out: " << ToString(out.dim()) << "D";
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=bilinear

        auto op_builder =
            [in_h = self.size(kBilinearHeightDim),
             in_w = self.size(kBilinearWidthDim),
             out_h = out.size(kBilinearHeightDim),
             out_w = out.size(kBilinearWidthDim), align_corners, scale_h,
             scale_w,
             precision](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildBilinear2d(input, in_h, in_w, out_h, out_w, align_corners,
                                 scale_h, scale_w, precision,
                                 /*is_backward=*/false);
        };

        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<1>(std::move(op_builder), {self},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(out.sizes()),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

// Lowers 2D bilinear interpolation backward as the adjoint separable
// contractions along H and W: dX = A_h^T * dY * A_w.
at::Tensor& AtenUpsampleBilinear2dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleBilinear2dBackwardGradInput, param_keys,
      (grad_output, output_size, input_size, align_corners, scales_h, scales_w,
       grad_input),
      {
        TT_CHECK_THROW(grad_output.dim() == 4 && input_size.size() == 4,
                       error::kInvalidArgument)
            << "expected 4D grad_output tensor and 4-element input_size, got "
               "grad_output: "
            << ToString(grad_output.dim())
            << "D, input_size: " << ToString(input_size.size()) << " elements";
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=bilinear
        auto op_builder =
            [in_h = input_size[kBilinearHeightDim],
             in_w = input_size[kBilinearWidthDim],
             out_h = grad_output.size(kBilinearHeightDim),
             out_w = grad_output.size(kBilinearWidthDim), align_corners,
             scales_h, scales_w,
             precision](mlir::MlirOp grad_out) -> absl::StatusOr<mlir::MlirOp> {
          return BuildBilinear2d(grad_out, in_h, in_w, out_h, out_w,
                                 align_corners, scales_h, scales_w, precision,
                                 /*is_backward=*/true);
        };

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<1>(std::move(op_builder), {grad_output},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_size),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

at::Tensor& AtenUpsampleNearest1dOut(const at::Tensor& self,
                                     at::IntArrayRef upsample_shape,
                                     std::optional<double> scale,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearest1dOut, param_keys,
      (self, upsample_shape, scale, out), {
        CheckUpsampleDtypes(self);
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        auto op_builder = [output_shape = CopyIntVector(out.sizes())](
                              FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [input, inverse_scale_factor_tensor] = inputs;
          TT_ASSIGN_OR_RETURN(auto output, BuildUpsampleNearestShlo(
                                               input, output_shape,
                                               {inverse_scale_factor_tensor},
                                               UpsampleMode::kNearest));
          return output;
        };
        auto preprocessed_inverse_scale_factors = StandardizeScaleFactors(
            self.sizes(), upsample_shape, {scale}, /*inverse=*/true);
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_inverse_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<2>(std::move(op_builder),
                           {self, inverse_scale_factors[0]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(out.sizes()),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

at::Tensor& AtenUpsampleNearest2dOut(const at::Tensor& self,
                                     at::IntArrayRef upsample_shape,
                                     std::optional<double> scale_h,
                                     std::optional<double> scale_w,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearest2dOut, param_keys,
      (self, upsample_shape, scale_h, scale_w, out), {
        CheckUpsampleDtypes(self);
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto op_builder = [output_shape = CopyIntVector(out.sizes())](
                              FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [input, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w] = inputs;
          TT_ASSIGN_OR_RETURN(auto output, BuildUpsampleNearestShlo(
                                               input, output_shape,
                                               {inverse_scale_factor_tensor_h,
                                                inverse_scale_factor_tensor_w},
                                               UpsampleMode::kNearest));
          return output;
        };

        auto preprocessed_inverse_scale_factors = StandardizeScaleFactors(
            self.sizes(), upsample_shape, {scale_h, scale_w},
            /*inverse=*/true);
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_inverse_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<3>(
                std::move(op_builder),
                {self, inverse_scale_factors[0], inverse_scale_factors[1]},
                {.out_dtype = element_type,
                 .out_dims = CopyIntVector(out.sizes()),
                 .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

at::Tensor& AtenUpsampleNearest3dOut(const at::Tensor& self,
                                     at::IntArrayRef upsample_shape,
                                     std::optional<double> scale_h,
                                     std::optional<double> scale_w,
                                     std::optional<double> scale_d,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearest3dOut, param_keys,
      (self, upsample_shape, scale_h, scale_w, scale_d, out), {
        CheckUpsampleDtypes(self);
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        auto op_builder = [output_shape = CopyIntVector(out.sizes())](
                              FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [input, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w, inverse_scale_factor_tensor_d] =
              inputs;
          TT_ASSIGN_OR_RETURN(auto output, BuildUpsampleNearestShlo(
                                               input, output_shape,
                                               {inverse_scale_factor_tensor_h,
                                                inverse_scale_factor_tensor_w,
                                                inverse_scale_factor_tensor_d},
                                               UpsampleMode::kNearest));
          return output;
        };
        auto preprocessed_inverse_scale_factors = StandardizeScaleFactors(
            self.sizes(), upsample_shape, {scale_h, scale_w, scale_d},
            /*inverse=*/true);
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_inverse_scale_factors));
        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<4>(std::move(op_builder),
                           {self, inverse_scale_factors[0],
                            inverse_scale_factors[1], inverse_scale_factors[2]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(out.sizes()),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));

        return out;
      });
}

at::Tensor& AtenUpsampleNearestExact1dOut(const at::Tensor& self,
                                          at::IntArrayRef upsample_shape,
                                          std::optional<double> scale,
                                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact1dOut, param_keys,
      (self, upsample_shape, scale, out), {
        CheckUpsampleDtypes(self);
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        auto op_builder = [output_shape = CopyIntVector(out.sizes())](
                              FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [input, inverse_scale_factor_tensor] = inputs;
          TT_ASSIGN_OR_RETURN(auto output, BuildUpsampleNearestShlo(
                                               input, output_shape,
                                               {inverse_scale_factor_tensor},
                                               UpsampleMode::kNearestExact));
          return output;
        };

        auto preprocessed_inverse_scale_factors = StandardizeScaleFactors(
            self.sizes(), upsample_shape, {scale}, /*inverse=*/true);
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_inverse_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<2>(std::move(op_builder),
                           {self, inverse_scale_factors[0]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(out.sizes()),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

at::Tensor& AtenUpsampleNearestExact2dOut(const at::Tensor& self,
                                          at::IntArrayRef upsample_shape,
                                          std::optional<double> scale_h,
                                          std::optional<double> scale_w,
                                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact2dOut, param_keys,
      (self, upsample_shape, scale_h, scale_w, out), {
        CheckUpsampleDtypes(self);
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto op_builder = [output_shape = CopyIntVector(out.sizes())](
                              FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [input, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w] = inputs;
          TT_ASSIGN_OR_RETURN(auto output, BuildUpsampleNearestShlo(
                                               input, output_shape,
                                               {inverse_scale_factor_tensor_h,
                                                inverse_scale_factor_tensor_w},
                                               UpsampleMode::kNearestExact));
          return output;
        };
        auto preprocessed_inverse_scale_factors = StandardizeScaleFactors(
            self.sizes(), upsample_shape, {scale_h, scale_w}, /*inverse=*/true);
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_inverse_scale_factors));
        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<3>(
                std::move(op_builder),
                {self, inverse_scale_factors[0], inverse_scale_factors[1]},
                {.out_dtype = element_type,
                 .out_dims = CopyIntVector(out.sizes()),
                 .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

at::Tensor& AtenUpsampleNearestExact3dOut(const at::Tensor& self,
                                          at::IntArrayRef upsample_shape,
                                          std::optional<double> scale_h,
                                          std::optional<double> scale_w,
                                          std::optional<double> scale_d,
                                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact3dOut, param_keys,
      (self, upsample_shape, scale_h, scale_w, scale_d, out), {
        CheckUpsampleDtypes(self);
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        auto op_builder = [output_shape = CopyIntVector(out.sizes())](
                              FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [input, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w, inverse_scale_factor_tensor_d] =
              inputs;
          TT_ASSIGN_OR_RETURN(auto output, BuildUpsampleNearestShlo(
                                               input, output_shape,
                                               {inverse_scale_factor_tensor_h,
                                                inverse_scale_factor_tensor_w,
                                                inverse_scale_factor_tensor_d},
                                               UpsampleMode::kNearestExact));
          return output;
        };
        auto preprocessed_inverse_scale_factors = StandardizeScaleFactors(
            self.sizes(), upsample_shape, {scale_h, scale_w, scale_d},
            /*inverse=*/true);
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_inverse_scale_factors));
        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<4>(std::move(op_builder),
                           {self, inverse_scale_factors[0],
                            inverse_scale_factors[1], inverse_scale_factors[2]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(out.sizes()),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));

        return out;
      });
}

}  // namespace torch_tpu
