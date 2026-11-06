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

#include "csrc/ops/scaled_mm/scaled_mm_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/BlasBackend.h"
#include "ATen/ceil_div.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/macro_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

enum class Operand { kLhs, kRhs };

absl::StatusOr<mlir::MlirOp> BuildStablehloDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs, mlir::stablehlo::Precision precision) {
  mlir::MlirBuilder& builder = lhs.getBuilder();
  mlir::MLIRContext& ctx = builder.getContext();
  auto dot_dimension_numbers = mlir::stablehlo::DotDimensionNumbersAttr::get(
      &ctx, /*lhs_batching_dimensions=*/{}, /*rhs_batching_dimensions=*/{},
      /*lhs_contracting_dimensions=*/{1}, /*rhs_contracting_dimensions=*/{0});

  auto precision_config_attr =
      mlir::stablehlo::PrecisionConfigAttr::get(&ctx, {precision, precision});

  auto lhs_type = GetTensorTypeOrDie(lhs);
  auto rhs_type = GetTensorTypeOrDie(rhs);
  Dimensions result_shape = {lhs_type.getShape()[0], rhs_type.getShape()[1]};

  TT_ASSIGN_OR_RETURN(mlir::ElementType lhs_elem_type,
                      ConvertTo<mlir::ElementType>(lhs_type.getElementType()));
  TT_ASSIGN_OR_RETURN(auto comp_dtype, InferComputationDtype(lhs_elem_type));

  auto result_type = mlir::RankedTensorType::get(
      result_shape, mlir::getElementType(ctx, comp_dtype));

  return mlir::stablehlo::DotGeneral(
      result_type, lhs, rhs, dot_dimension_numbers, precision_config_attr);
}

absl::StatusOr<mlir::MlirOp> ReshapeScaleIfNeeded(mlir::MlirOp scale,
                                                  const Dimensions& target) {
  auto type = GetTensorTypeOrDie(scale);
  if (type.getNumElements() == 1) {
    // Scalar (tensorwise) scale: broadcasts as-is, no reshape needed.
    return scale;
  }
  Dimensions before = CopyIntVector(type.getShape());
  if (before == target) {
    return scale;
  }
  return ReshapeFromStaticDimensions(scale, before, target);
}

// Builds the StableHLO operations for scaled matrix multiplication.
// It performs matrix multiplication, applies scales, adds bias, and applies
// result scale.
//
// The high-level mathematical formula is:
//   result = (lhs @ rhs) * (scale_a * scale_b) + bias
// If scale_result is present and not folded:
//   result = ((lhs @ rhs) * (scale_a * scale_b) + bias) * scale_result
//
absl::StatusOr<mlir::MlirOp> BuildScaledMmShlo(
    mlir::MlirOp lhs, mlir::MlirOp rhs, mlir::MlirOp scale_a,
    mlir::MlirOp scale_b, std::optional<mlir::MlirOp> bias,
    std::optional<mlir::MlirOp> scale_result,
    std::optional<mlir::Type> out_dtype, mlir::stablehlo::Precision precision) {
  // 1. Perform matrix multiplication with inferred computation dtype.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mm_result,
                      BuildStablehloDotGeneral(lhs, rhs, precision));

  // Cast mm_result to F32 to avoid type mismatch with scales (which are F32).
  TT_ASSIGN_OR_RETURN(mm_result,
                      CastIfNeeded(mm_result, mlir::ElementType::F32));

  // Normalize row-wise / per-channel scales to [M, 1] / [1, N] so they
  // broadcast along the correct axis of the [M, N] result. Scalar (tensorwise)
  // scales are left unchanged. See
  // https://github.com/google-pytorch/torch_tpu/issues/1820.
  auto mm_type = GetTensorTypeOrDie(mm_result);
  const int64_t m_dim = mm_type.getShape()[0];
  const int64_t n_dim = mm_type.getShape()[1];
  TT_ASSIGN_OR_RETURN(scale_a,
                      ReshapeScaleIfNeeded(scale_a, Dimensions{m_dim, 1}));
  TT_ASSIGN_OR_RETURN(scale_b,
                      ReshapeScaleIfNeeded(scale_b, Dimensions{1, n_dim}));

  // Combine scale_a and scale_b into a single scale tensor to reduce
  // multiplications.
  TT_ASSIGN_OR_RETURN(auto broadcasted_scales,
                      ApplyBroadcastIfNeeded<2>({scale_a, scale_b}));

  mlir::MlirOp combined_scale =
      mlir::stablehlo::Mul(broadcasted_scales[0], broadcasted_scales[1]);

  // If there is no bias, we can fold scale_result into the combined scale to
  // further reduce multiplications.
  bool folded_scale_result = false;
  if (!bias.has_value() && scale_result.has_value()) {
    TT_ASSIGN_OR_RETURN(
        auto bcast_scale_result,
        ApplyBroadcastIfNeeded<2>({combined_scale, *scale_result}));

    combined_scale =
        mlir::stablehlo::Mul(bcast_scale_result[0], bcast_scale_result[1]);
    folded_scale_result = true;
  }

  // Scale the matrix multiplication result.
  TT_ASSIGN_OR_RETURN(auto broadcasted_mm_scale,
                      ApplyBroadcastIfNeeded<2>({mm_result, combined_scale}));

  mlir::MlirOp scaled_result =
      mlir::stablehlo::Mul(broadcasted_mm_scale[0], broadcasted_mm_scale[1]);

  if (bias.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp bias_f32,
                        CastIfNeeded(*bias, mlir::ElementType::F32));
    TT_ASSIGN_OR_RETURN(auto broadcasted_bias,
                        ApplyBroadcastIfNeeded<2>({scaled_result, bias_f32}));

    scaled_result =
        mlir::stablehlo::Add(broadcasted_bias[0], broadcasted_bias[1]);
  }

  if (scale_result.has_value() && !folded_scale_result) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp scale_result_f32,
                        CastIfNeeded(*scale_result, mlir::ElementType::F32));
    TT_ASSIGN_OR_RETURN(
        auto broadcasted_scale_result,
        ApplyBroadcastIfNeeded<2>({scaled_result, scale_result_f32}));

    scaled_result = mlir::stablehlo::Mul(broadcasted_scale_result[0],
                                         broadcasted_scale_result[1]);
  }

  // 5. Cast to out_dtype if present.
  if (out_dtype.has_value()) {
    scaled_result =
        mlir::stablehlo::ConvertElementType(scaled_result, *out_dtype);
  }

  return scaled_result;
}

// SWIZZLE_32_4_4 requires fixed 128x4 hardware tiles (interleaving
// 4 groups of 32 rows). Get the expected element count for the scaler.
int64_t SwizzledScaleNumel(int64_t outer, int64_t k_logical) {
  return at::ceil_div(outer, int64_t{128}) * 128 *
         at::ceil_div(k_logical, int64_t{4}) * 4;
}

#if TT_TORCH_VERSION_GE(2, 14)  // See scaled_mm_aten_kernels.h.
absl::Status ValidateScaledMmV2Inputs(
    const at::Tensor& self, const at::Tensor& mat2,
    const at::ITensorListRef& scale_a, at::IntArrayRef recipe_a,
    at::IntArrayRef swizzle_a, const at::ITensorListRef& scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim,
    bool use_fast_accum) {
#else
absl::Status ValidateScaledMmV2Inputs(
    const at::Tensor& self, const at::Tensor& mat2, at::TensorList scale_a,
    at::IntArrayRef recipe_a, at::IntArrayRef swizzle_a, at::TensorList scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim,
    bool use_fast_accum) {
#endif
  if (!contraction_dim.empty()) {
    TT_RET_CHECK(contraction_dim.size() == 2, error::kInvalidArgument)
        << "expected contraction_dim list to contain exactly 2 elements, got "
        << contraction_dim.size();
    TT_RET_CHECK(contraction_dim[0] == 1 && contraction_dim[1] == 0,
                 error::kPythonNotImplementedError)
        << "expected contraction_dim to be [1, 0], got [" << contraction_dim[0]
        << ", " << contraction_dim[1] << "]";
  }

  TT_RETURN_IF_ERROR(ValidateIsMatrix(self, "self"));
  TT_RETURN_IF_ERROR(ValidateIsMatrix(mat2, "mat2"));
  TT_RET_CHECK(self.size(1) == mat2.size(0), error::kInvalidArgument)
      << "expected column size of first matrix to match row size of second "
         "matrix, got shapes "
      << ToString(self.sizes()) << " and " << ToString(mat2.sizes());

  TT_RET_CHECK(scale_a.size() == 1, error::kInvalidArgument)
      << "expected scale_a list to contain exactly 1 tensor, got "
      << scale_a.size();
  TT_RET_CHECK(scale_b.size() == 1, error::kInvalidArgument)
      << "expected scale_b list to contain exactly 1 tensor, got "
      << scale_b.size();

  TT_RET_CHECK(recipe_a.size() == 1, error::kInvalidArgument)
      << "expected recipe_a list to contain exactly 1 recipe, got "
      << recipe_a.size();
  TT_RET_CHECK(recipe_b.size() == 1, error::kInvalidArgument)
      << "expected recipe_b list to contain exactly 1 recipe, got "
      << recipe_b.size();

  TT_RET_CHECK(swizzle_a.size() == 1, error::kInvalidArgument)
      << "expected swizzle_a list to contain exactly 1 swizzle mode, got "
      << swizzle_a.size();
  TT_RET_CHECK(swizzle_b.size() == 1, error::kInvalidArgument)
      << "expected swizzle_b list to contain exactly 1 swizzle mode, got "
      << swizzle_b.size();

  const auto scaling_a = static_cast<at::blas::ScalingType>(recipe_a[0]);
  const auto scaling_b = static_cast<at::blas::ScalingType>(recipe_b[0]);

  const auto swiz_a = static_cast<at::blas::SwizzleType>(swizzle_a[0]);
  const auto swiz_b = static_cast<at::blas::SwizzleType>(swizzle_b[0]);

  const auto check_recipe = [](at::blas::ScalingType recipe,
                               std::string_view name) -> absl::Status {
    switch (recipe) {
      case at::blas::ScalingType::TensorWise:
      case at::blas::ScalingType::RowWise:
      case at::blas::ScalingType::BlockWise1x32:
        return absl::OkStatus();
      default:
        return TT_ERROR(error::kPythonNotImplementedError)
               << "expected scaling recipe for " << name
               << " to be TensorWise, RowWise, or BlockWise1x32, got "
               << static_cast<int>(recipe);
    }
  };
  TT_RETURN_IF_ERROR(check_recipe(scaling_a, "scale_a"));
  TT_RETURN_IF_ERROR(check_recipe(scaling_b, "scale_b"));

  const auto check_swizzle = [](at::blas::SwizzleType swizzle,
                                std::string_view name) -> absl::Status {
    switch (swizzle) {
      case at::blas::SwizzleType::NO_SWIZZLE:
      case at::blas::SwizzleType::SWIZZLE_32_4_4:
        return absl::OkStatus();
      default:
        return TT_ERROR(error::kPythonNotImplementedError)
               << "expected swizzle type for " << name
               << " to be NO_SWIZZLE or SWIZZLE_32_4_4, got "
               << static_cast<int>(swizzle);
    }
  };
  TT_RETURN_IF_ERROR(check_swizzle(swiz_a, "scale_a"));
  TT_RETURN_IF_ERROR(check_swizzle(swiz_b, "scale_b"));

  // A swizzled scale buffer is flat. Checking the element count of the scale
  // satisfy the need for swizzle and corresponding mat.
  const auto check_swizzled_scale = [](const at::Tensor& scale, int64_t outer,
                                       int64_t k_logical) -> absl::Status {
    const int64_t expected_numel = SwizzledScaleNumel(outer, k_logical);
    TT_RET_CHECK(scale.numel() == expected_numel, error::kInvalidArgument)
        << "expected scale buffer to have " << expected_numel
        << " elements for outer=" << outer << " and k_logical=" << k_logical
        << ", got " << scale.numel() << " (" << ToString(scale.sizes()) << ")";
    return absl::OkStatus();
  };
  if (swiz_a == at::blas::SwizzleType::SWIZZLE_32_4_4) {
    TT_RETURN_IF_ERROR(
        check_swizzled_scale(scale_a.front(), self.size(0), self.size(1) / 32));
  }
  if (swiz_b == at::blas::SwizzleType::SWIZZLE_32_4_4) {
    TT_RETURN_IF_ERROR(
        check_swizzled_scale(scale_b.front(), mat2.size(1), mat2.size(0) / 32));
  }

  if (bias.has_value()) {
    TT_RET_CHECK(bias->dim() == 1 || bias->dim() == 2, error::kInvalidArgument)
        << "expected bias to be 1D or 2D tensor, got " << bias->dim() << "D";
    TT_RET_CHECK(bias->scalar_type() == at::kFloat ||
                     bias->scalar_type() == at::kBFloat16,
                 error::kInvalidArgument)
        << "expected bias dtype to be " << ToString(at::kFloat) << " or "
        << ToString(at::kBFloat16) << ", got " << ToString(bias->scalar_type());
  }

  if (out_dtype.has_value()) {
    TT_RET_CHECK(*out_dtype == at::kFloat || *out_dtype == at::kBFloat16 ||
                     *out_dtype == at::kHalf,
                 error::kInvalidArgument)
        << "expected out dtype to be " << ToString(at::kFloat) << ", "
        << ToString(at::kBFloat16) << ", or " << ToString(at::kHalf) << ", got "
        << ToString(*out_dtype);
  }

  return absl::OkStatus();
}

// Unswizzles a SWIZZLE_32_4_4 block-scale buffer back to its logical shape.
//
// SWIZZLE_32_4_4 is NVIDIA's blocked scale-factor layout. The logical [outer,
// K_logical] scale matrix is cut into 128x4 tiles visited in row-block-major
// order, and each tile is flattened to 512 contiguous elements by dealing its
// 128 rows into 32 groups of 4, like dealing cards: group 0 takes rows 0, 32,
// 64, 96; group 1 takes rows 1, 33, 65, 97; and so on up to group 31. Every row
// brings its 4 column values, so a group occupies 16 contiguous elements:
//
//   group 0   elem   0  1  2  3 | 4  5  6  7 | 8  9 10 11 | 12 13 14 15
//             row    0  0  0  0 |32 32 32 32 |64 64 64 64 | 96 96 96 96
//             col    0  1  2  3 | 0  1  2  3 | 0  1  2  3 |  0  1  2  3
//
// Reading the whole tile end to end, its 128 rows therefore come out in this
// order, and we want them back in plain order:
//
//   stored order:  0, 32, 64, 96, 1, 33, 65, 97, 2, 34, 66, 98, ... 31, 63, ...
//   wanted order:  0,  1,  2,  3, 4,  5,  6,  7, 8,  9, 10, 11, ...
//
// A row's 4 column values stay glued together throughout, so this is a
// permutation of 128 four-element chunks.
//
// `outer` is the non-contraction extent the 128-tiling
// applies to: M for the LHS (with shape [M, K]), N for the RHS (with shape
// [N, K]). The RHS result is transposed back to the [K_logical, N] orientation
// that BuildDequantizeBlockWise32 expects.
absl::StatusOr<mlir::MlirOp> BuildUnswizzle(mlir::MlirOp scale, Operand operand,
                                            int64_t outer, int64_t k_logical) {
  // PyTorch to_blocked pads non-multiples to ceil_div multiples of (128, 4).
  const int64_t outer_blocks = at::ceil_div(outer, int64_t{128});
  const int64_t k_blocks = at::ceil_div(k_logical, int64_t{4});
  const int64_t padded_outer = outer_blocks * 128;
  const int64_t padded_k = k_blocks * 4;
  const auto scale_type = GetTensorTypeOrDie(scale);

  ABSL_CHECK_EQ(  // CRASH_OK=checked in ValidateScaledMmV2Inputs
      scale_type.getNumElements(), padded_outer * padded_k)
      << "unexpected scale buffer shape for outer=" << outer
      << " and k_logical=" << k_logical << ": "
      << ToString(scale_type.getShape());

  // Worked example: outer = 256, K_logical = 8, i.e. a 256x8 logical scale
  // matrix stored as a 2x2 grid of 128x4 swizzled tiles, arriving flat.

  // 1. Reshape the flat scale from [padded_outer * padded_k] to
  //    [outer_blocks, k_blocks, 32, 4, 4], isolating the 32x4x4 swizzled tiles,
  //    each identified by [row_block, col_block]:
  //    [row_block, col_block, group, position, column]
  const Dimensions shape1 = {outer_blocks, k_blocks, 32, 4, 4};
  TT_ASSIGN_OR_RETURN(mlir::MlirOp op1,
                      ReshapeFromStaticDimensions(
                          scale, CopyIntVector(scale_type.getShape()), shape1));

  // 2. Transpose each [32, 4, 4] tile to [4, 32, 4], which undoes the deal:
  //    its rows now read 0, 1, 2, ... 127 instead of 0, 32, 64, 96, 1, 33, ...
  const Dimensions perm1 = {0, 1, 3, 2, 4};
  mlir::MlirOp op2 = mlir::stablehlo::Transpose(op1, perm1);

  // 3. Reshape to [outer_blocks, k_blocks, 128, 4], collapsing each tile into a
  //    plain row-major 128x4 block.
  const Dimensions shape2 = {outer_blocks, k_blocks, 128, 4};
  TT_ASSIGN_OR_RETURN(mlir::MlirOp op3,
                      ReshapeFromStaticDimensions(
                          op2, {outer_blocks, k_blocks, 4, 32, 4}, shape2));

  // 4. Transpose to [outer_blocks, 128, k_blocks, 4], i.e.
  //    [row_block, row, col_block, col], putting each grid index next to the
  //    extent it indexes so the two row axes and the two column axes pair up.
  const Dimensions perm2 = {0, 2, 1, 3};
  mlir::MlirOp op4 = mlir::stablehlo::Transpose(op3, perm2);

  // 5. Reshape to [padded_outer, padded_k], the padded logical scale matrix.
  const Dimensions shape3 = {padded_outer, padded_k};
  TT_ASSIGN_OR_RETURN(mlir::MlirOp unswizzled,
                      ReshapeFromStaticDimensions(
                          op4, {outer_blocks, 128, k_blocks, 4}, shape3));

  // 6. Slice to [outer, k_logical] if the input buffer was padded.
  if (padded_outer != outer || padded_k != k_logical) {
    unswizzled =
        mlir::stablehlo::Slice(unswizzled, {0, 0}, {outer, k_logical}, {1, 1});
  }

  // The RHS is blocked over [N, K_logical]; flip it to the [K_logical, N]
  // orientation the dequantize step expects.
  if (operand == Operand::kRhs) {
    const Dimensions perm = {1, 0};
    unswizzled = mlir::stablehlo::Transpose(unswizzled, perm);
  }

  return unswizzled;
}

absl::StatusOr<mlir::MlirOp> BuildDequantizeBlockWise32(mlir::MlirOp input,
                                                        mlir::MlirOp scale,
                                                        Operand operand) {
  const auto input_type = GetTensorTypeOrDie(input);
  const auto scale_type = GetTensorTypeOrDie(scale);

  const bool is_lhs = (operand == Operand::kLhs);
  const int64_t m_or_n =
      is_lhs ? input_type.getShape()[0] : input_type.getShape()[1];
  const int64_t k =
      is_lhs ? input_type.getShape()[1] : input_type.getShape()[0];

  TT_ASSIGN_OR_RETURN(mlir::MlirOp input_f32,
                      CastIfNeeded(input, mlir::ElementType::F32));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp scale_f32,
                      CastIfNeeded(scale, mlir::ElementType::F32));

  Dimensions reshape_in_dims;
  Dimensions reshape_scale_dims;
  if (is_lhs) {
    reshape_in_dims = {m_or_n, k / 32, 32};
    reshape_scale_dims = {m_or_n, k / 32, 1};
  } else {
    reshape_in_dims = {k / 32, 32, m_or_n};
    reshape_scale_dims = {k / 32, 1, m_or_n};
  }

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp input_reshaped,
      ReshapeFromStaticDimensions(
          input_f32, CopyIntVector(input_type.getShape()), reshape_in_dims));

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp scale_reshaped,
      ReshapeFromStaticDimensions(
          scale_f32, CopyIntVector(scale_type.getShape()), reshape_scale_dims));

  TT_ASSIGN_OR_RETURN(
      auto bcast, ApplyBroadcastIfNeeded<2>({input_reshaped, scale_reshaped}));
  mlir::MlirOp dequantized = mlir::stablehlo::Mul(bcast[0], bcast[1]);

  return ReshapeFromStaticDimensions(dequantized, reshape_in_dims,
                                     CopyIntVector(input_type.getShape()));
}

absl::StatusOr<mlir::MlirOp> ApplyScalesAndBias(
    mlir::MlirOp mm_result, mlir::MlirOp scale_a, bool apply_scale_a,
    mlir::MlirOp scale_b, bool apply_scale_b, std::optional<mlir::MlirOp> bias,
    std::optional<mlir::Type> out_dtype) {
  mlir::MlirOp scaled_result = mm_result;
  if (apply_scale_a) {
    const auto mm_type = GetTensorTypeOrDie(scaled_result);
    const int64_t m_dim = mm_type.getShape()[0];
    TT_ASSIGN_OR_RETURN(mlir::MlirOp scale_a_f32,
                        CastIfNeeded(scale_a, mlir::ElementType::F32));
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp scale_a_reshaped,
        ReshapeScaleIfNeeded(scale_a_f32, Dimensions{m_dim, 1}));
    TT_ASSIGN_OR_RETURN(auto bcast, ApplyBroadcastIfNeeded<2>(
                                        {scaled_result, scale_a_reshaped}));
    scaled_result = mlir::stablehlo::Mul(bcast[0], bcast[1]);
  }
  if (apply_scale_b) {
    const auto mm_type = GetTensorTypeOrDie(scaled_result);
    const int64_t n_dim = mm_type.getShape()[1];
    TT_ASSIGN_OR_RETURN(mlir::MlirOp scale_b_f32,
                        CastIfNeeded(scale_b, mlir::ElementType::F32));
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp scale_b_reshaped,
        ReshapeScaleIfNeeded(scale_b_f32, Dimensions{1, n_dim}));
    TT_ASSIGN_OR_RETURN(auto bcast, ApplyBroadcastIfNeeded<2>(
                                        {scaled_result, scale_b_reshaped}));
    scaled_result = mlir::stablehlo::Mul(bcast[0], bcast[1]);
  }

  if (bias.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp bias_f32,
                        CastIfNeeded(*bias, mlir::ElementType::F32));
    TT_ASSIGN_OR_RETURN(auto broadcasted_bias,
                        ApplyBroadcastIfNeeded<2>({scaled_result, bias_f32}));
    scaled_result =
        mlir::stablehlo::Add(broadcasted_bias[0], broadcasted_bias[1]);
  }

  if (out_dtype.has_value()) {
    scaled_result =
        mlir::stablehlo::ConvertElementType(scaled_result, *out_dtype);
  }
  return scaled_result;
}

absl::StatusOr<mlir::MlirOp> BuildScaledMmV2Shlo(
    mlir::MlirOp lhs, mlir::MlirOp rhs, mlir::MlirOp scale_a,
    mlir::MlirOp scale_b, std::optional<mlir::MlirOp> bias,
    std::optional<mlir::Type> out_dtype, mlir::stablehlo::Precision precision,
    at::blas::ScalingType scaling_a, at::blas::ScalingType scaling_b,
    at::blas::SwizzleType swiz_a, at::blas::SwizzleType swiz_b) {
  // The blocked scale buffer carries no shape of its own, so its logical
  // extents come from the operand it scales: [M, K / 32] and [K / 32, N].
  const auto lhs_shape = GetTensorTypeOrDie(lhs).getShape();
  const auto rhs_shape = GetTensorTypeOrDie(rhs).getShape();
  if (swiz_a == at::blas::SwizzleType::SWIZZLE_32_4_4) {
    TT_ASSIGN_OR_RETURN(scale_a,
                        BuildUnswizzle(scale_a, Operand::kLhs, lhs_shape[0],
                                       lhs_shape[1] / 32));
  }
  if (swiz_b == at::blas::SwizzleType::SWIZZLE_32_4_4) {
    TT_ASSIGN_OR_RETURN(scale_b,
                        BuildUnswizzle(scale_b, Operand::kRhs, rhs_shape[1],
                                       rhs_shape[0] / 32));
  }

  mlir::MlirOp lhs_deq = lhs;
  mlir::MlirOp rhs_deq = rhs;
  const bool is_lhs_block = (scaling_a == at::blas::ScalingType::BlockWise1x32);
  const bool is_rhs_block = (scaling_b == at::blas::ScalingType::BlockWise1x32);

  if (is_lhs_block) {
    TT_ASSIGN_OR_RETURN(
        lhs_deq, BuildDequantizeBlockWise32(lhs, scale_a, Operand::kLhs));
  }
  if (is_rhs_block) {
    TT_ASSIGN_OR_RETURN(
        rhs_deq, BuildDequantizeBlockWise32(rhs, scale_b, Operand::kRhs));
  }

  if (is_lhs_block || is_rhs_block) {
    if (!is_lhs_block) {
      TT_ASSIGN_OR_RETURN(lhs_deq, CastIfNeeded(lhs, mlir::ElementType::F32));
    }
    if (!is_rhs_block) {
      TT_ASSIGN_OR_RETURN(rhs_deq, CastIfNeeded(rhs, mlir::ElementType::F32));
    }

    TT_ASSIGN_OR_RETURN(mlir::MlirOp mm_result,
                        BuildStablehloDotGeneral(lhs_deq, rhs_deq, precision));

    return ApplyScalesAndBias(mm_result, scale_a, !is_lhs_block, scale_b,
                              !is_rhs_block, bias, out_dtype);
  } else {
    return BuildScaledMmShlo(lhs, rhs, scale_a, scale_b, bias, std::nullopt,
                             out_dtype, precision);
  }
}

absl::Status ValidateScaledMmInputs(
    const at::Tensor& self, const at::Tensor& mat2, const at::Tensor& scale_a,
    const at::Tensor& scale_b, const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& scale_result) {
  TT_RETURN_IF_ERROR(ValidateIsMatrix(self, "self"));
  TT_RETURN_IF_ERROR(ValidateIsMatrix(mat2, "mat2"));

  const int64_t k_a = self.size(1);
  const int64_t k_b = mat2.size(0);
  TT_RET_CHECK(k_a == k_b, error::kInvalidArgument)
      << "expected column size of first matrix to match row size of second "
         "matrix, got shapes "
      << ToString(self.sizes()) << " and " << ToString(mat2.sizes());

  const int64_t m_dim = self.size(0);
  const int64_t n_dim = mat2.size(1);
  TT_RET_CHECK(scale_a.numel() == 1 || scale_a.numel() == m_dim,
               error::kPythonNotImplementedError)
      << "expected scale_a to have numel 1 (tensorwise) or " << m_dim
      << " (row-wise), got numel " << scale_a.numel();
  TT_RET_CHECK(scale_b.numel() == 1 || scale_b.numel() == n_dim,
               error::kPythonNotImplementedError)
      << "expected scale_b to have numel 1 (tensorwise) or " << n_dim
      << " (per-channel), got numel " << scale_b.numel();

  // A 2-D row-wise scale_a must be [M, 1] and a 2-D per-channel scale_b must be
  // [1, N]. Reject other orientations (e.g. [1, M]) so a transposed scale is a
  // clean error rather than a silent wrong-axis broadcast. 1-D scales ([M]/[N])
  // are unambiguous and allowed.
  if (scale_a.numel() == m_dim && m_dim != 1 && scale_a.dim() == 2) {
    TT_RET_CHECK(scale_a.size(0) == m_dim && scale_a.size(1) == 1,
                 error::kInvalidArgument)
        << "expected row-wise scale_a to have shape [" << m_dim << ", 1], got "
        << ToString(scale_a.sizes());
  }
  if (scale_b.numel() == n_dim && n_dim != 1 && scale_b.dim() == 2) {
    TT_RET_CHECK(scale_b.size(0) == 1 && scale_b.size(1) == n_dim,
                 error::kInvalidArgument)
        << "expected per-channel scale_b to have shape [1, " << n_dim
        << "], got " << ToString(scale_b.sizes());
  }

  if (scale_result.has_value() && scale_result->defined()) {
    TT_RET_CHECK(scale_result->numel() == 1, error::kPythonNotImplementedError)
        << "expected scale_result to have numel 1, got numel "
        << scale_result->numel();
  }

  return absl::OkStatus();
}

auto GetScaledMmOpBuilder(bool has_bias, bool has_scale_result,
                          mlir::ElementType target_elem_dtype,
                          mlir::stablehlo::Precision current_precision) {
  return [has_bias, has_scale_result, target_elem_dtype, current_precision](
             absl::Span<mlir::MlirOp> builder_inputs,
             mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirOp lhs_op = builder_inputs[0];
    mlir::MlirOp rhs_op = builder_inputs[1];
    mlir::MlirOp scale_a_op = builder_inputs[2];
    mlir::MlirOp scale_b_op = builder_inputs[3];

    std::optional<mlir::MlirOp> bias_op;
    std::optional<mlir::MlirOp> scale_result_op;

    if (has_bias) {
      bias_op = builder_inputs[4];
    }
    if (has_scale_result) {
      scale_result_op = builder_inputs[has_bias ? 5 : 4];
    }

    TT_ASSIGN_OR_RETURN(std::optional<mlir::Type> out_dtype_mlir,
                        GetMlirType(builder.getContext(), target_elem_dtype));

    return BuildScaledMmShlo(lhs_op, rhs_op, scale_a_op, scale_b_op, bias_op,
                             scale_result_op, out_dtype_mlir,
                             current_precision);
  };
}

std::vector<at::Tensor> BuildScaledMmInputs(
    const at::Tensor& self, const at::Tensor& mat2, const at::Tensor& scale_a,
    const at::Tensor& scale_b, const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& scale_result, bool has_bias,
    bool has_scale_result) {
  std::vector<at::Tensor> inputs = {self, mat2, scale_a, scale_b};
  if (has_bias) {
    inputs.push_back(*bias);
  }
  if (has_scale_result) {
    inputs.push_back(*scale_result);
  }
  return inputs;
}

absl::Status ScaledMmOut(const at::Tensor& self, const at::Tensor& mat2,
                         const at::Tensor& scale_a, const at::Tensor& scale_b,
                         std::optional<at::Tensor> bias,
                         std::optional<at::Tensor> scale_result,
                         std::optional<at::ScalarType> out_dtype,
                         OpParamCacheKeys param_keys, at::Tensor& out) {
  TT_RETURN_IF_ERROR(
      ValidateScaledMmInputs(self, mat2, scale_a, scale_b, bias, scale_result));

  Dimensions output_dims = {self.size(0), mat2.size(1)};
  at::ScalarType target_scalar_type =
      out_dtype.has_value() ? *out_dtype : self.scalar_type();
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, target_scalar_type,
                                      /*allow_out_dtype_cast=*/false));
  TT_ASSIGN_OR_RETURN(mlir::ElementType target_elem_dtype,
                      ConvertTo<mlir::ElementType>(target_scalar_type));

  TT_ASSIGN_OR_RETURN(mlir::ElementType self_elem_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(mlir::ElementType comp_dtype,
                      InferComputationDtype(self_elem_dtype));

  const mlir::stablehlo::Precision current_precision =
      GetAndAddPrecisionTo(param_keys);

  bool has_bias = bias.has_value() && bias->defined();
  bool has_scale_result = scale_result.has_value() && scale_result->defined();

  std::vector<at::Tensor> inputs =
      BuildScaledMmInputs(self, mat2, scale_a, scale_b, bias, scale_result,
                          has_bias, has_scale_result);
  auto op_builder = GetScaledMmOpBuilder(has_bias, has_scale_result,
                                         target_elem_dtype, current_precision);

  return DispatchOpOut<kDynamicSize>(
      std::move(op_builder), inputs, out,
      {.out_dtype = target_elem_dtype,
       .out_dims = output_dims,
       .computation_dtype = comp_dtype,
       .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenScaledMm(const at::Tensor& self, const at::Tensor& mat2,
                        const at::Tensor& scale_a, const at::Tensor& scale_b,
                        const std::optional<at::Tensor>& bias,
                        const std::optional<at::Tensor>& scale_result,
                        std::optional<at::ScalarType> out_dtype,
                        bool use_fast_accum) {
  TT_KERNEL(OpName::kScaledMm, param_keys,
            (self, mat2, scale_a, scale_b, bias, scale_result, out_dtype,
             IgnoreInCacheKey(use_fast_accum, "Doesn't affect SHLO")),
            {
              at::ScalarType target_scalar_type =
                  out_dtype.has_value() ? *out_dtype : self.scalar_type();
              const int64_t n = mat2.size(1);
              TT_ASSIGN_OR_THROW(
                  at::Tensor out,
                  MakeEmptyTensor({self.size(0), n}, target_scalar_type,
                                  self.device()));
              // use_fast_accum is accepted and ignored: the MXU always
              // accumulates in F32.
              TT_THROW_IF_ERROR(ScaledMmOut(self, mat2, scale_a, scale_b, bias,
                                            scale_result, out_dtype,
                                            std::move(param_keys), out));
              return out;
            });
}

at::Tensor& AtenScaledMmOut(const at::Tensor& self, const at::Tensor& mat2,
                            const at::Tensor& scale_a,
                            const at::Tensor& scale_b,
                            const std::optional<at::Tensor>& bias,
                            const std::optional<at::Tensor>& scale_result,
                            std::optional<at::ScalarType> out_dtype,
                            bool use_fast_accum, at::Tensor& out) {
  TT_KERNEL(OpName::kScaledMmOut, param_keys,
            (self, mat2, scale_a, scale_b, bias, scale_result, out_dtype,
             IgnoreInCacheKey(use_fast_accum, "Doesn't affect SHLO"), out),
            {
              // use_fast_accum is accepted and ignored: the MXU always
              // accumulates in F32.
              TT_THROW_IF_ERROR(ScaledMmOut(self, mat2, scale_a, scale_b, bias,
                                            scale_result, out_dtype,
                                            std::move(param_keys), out));
              return out;
            });
}

#if TT_TORCH_VERSION_GE(2, 14)  // See scaled_mm_aten_kernels.h.
absl::Status ScaledMmV2Out(const at::Tensor& self, const at::Tensor& mat2,
                           const at::ITensorListRef& scale_a,
                           at::IntArrayRef recipe_a, at::IntArrayRef swizzle_a,
                           const at::ITensorListRef& scale_b,
                           at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
                           std::optional<at::Tensor> bias,
                           std::optional<at::ScalarType> out_dtype,
                           at::IntArrayRef contraction_dim, bool use_fast_accum,
                           OpParamCacheKeys param_keys, at::Tensor& out) {
#else
absl::Status ScaledMmV2Out(const at::Tensor& self, const at::Tensor& mat2,
                           at::TensorList scale_a, at::IntArrayRef recipe_a,
                           at::IntArrayRef swizzle_a, at::TensorList scale_b,
                           at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
                           std::optional<at::Tensor> bias,
                           std::optional<at::ScalarType> out_dtype,
                           at::IntArrayRef contraction_dim, bool use_fast_accum,
                           OpParamCacheKeys param_keys, at::Tensor& out) {
#endif
  TT_RETURN_IF_ERROR(ValidateScaledMmV2Inputs(
      self, mat2, scale_a, recipe_a, swizzle_a, scale_b, recipe_b, swizzle_b,
      bias, out_dtype, contraction_dim, use_fast_accum));

  const auto scaling_a = static_cast<at::blas::ScalingType>(recipe_a[0]);
  const auto scaling_b = static_cast<at::blas::ScalingType>(recipe_b[0]);
  const auto swiz_a = static_cast<at::blas::SwizzleType>(swizzle_a[0]);
  const auto swiz_b = static_cast<at::blas::SwizzleType>(swizzle_b[0]);

  const at::Tensor& s_a = scale_a.front();
  const at::Tensor& s_b = scale_b.front();

  Dimensions output_dims = {self.size(0), mat2.size(1)};
  at::ScalarType target_scalar_type =
      out_dtype.has_value() ? *out_dtype : self.scalar_type();
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, target_scalar_type,
                                      /*allow_out_dtype_cast=*/false));
  TT_ASSIGN_OR_RETURN(mlir::ElementType target_elem_dtype,
                      ConvertTo<mlir::ElementType>(target_scalar_type));

  TT_ASSIGN_OR_RETURN(mlir::ElementType self_elem_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(mlir::ElementType comp_dtype,
                      InferComputationDtype(self_elem_dtype));

  mlir::stablehlo::Precision current_precision =
      use_fast_accum ? mlir::stablehlo::Precision::
                           DEFAULT  // EXPLICIT_PRECISION_OK=use_fast_accum
                                    // explicitly requests fast
                                    // accumulator precision on TPU
                     : GetAndAddPrecisionTo(param_keys);

  const bool has_bias = bias.has_value() && bias->defined();

  std::vector<at::Tensor> inputs = {self, mat2, s_a, s_b};
  inputs.reserve(5);
  if (has_bias) {
    inputs.push_back(*bias);
  }

  auto op_builder =
      [has_bias, target_elem_dtype, current_precision, scaling_a, scaling_b,
       swiz_a,
       swiz_b](absl::Span<mlir::MlirOp> builder_inputs,
               mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirOp lhs_op = builder_inputs[0];
    mlir::MlirOp rhs_op = builder_inputs[1];
    mlir::MlirOp scale_a_op = builder_inputs[2];
    mlir::MlirOp scale_b_op = builder_inputs[3];

    std::optional<mlir::MlirOp> bias_op;
    if (has_bias) {
      bias_op = builder_inputs[4];
    }

    TT_ASSIGN_OR_RETURN(std::optional<mlir::Type> out_dtype_mlir,
                        GetMlirType(builder.getContext(), target_elem_dtype));

    return BuildScaledMmV2Shlo(lhs_op, rhs_op, scale_a_op, scale_b_op, bias_op,
                               out_dtype_mlir, current_precision, scaling_a,
                               scaling_b, swiz_a, swiz_b);
  };

  return DispatchOpOut<kDynamicSize>(
      std::move(op_builder), inputs, out,
      {.out_dtype = target_elem_dtype,
       .out_dims = output_dims,
       .computation_dtype = comp_dtype,
       .op_param_cache_keys = std::move(param_keys)});
}

#if TT_TORCH_VERSION_GE(2, 14)  // See scaled_mm_aten_kernels.h.
at::Tensor AtenScaledMmV2(const at::Tensor& self, const at::Tensor& mat2,
                          const at::ITensorListRef& scale_a,
                          at::IntArrayRef recipe_a, at::IntArrayRef swizzle_a,
                          const at::ITensorListRef& scale_b,
                          at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
                          const std::optional<at::Tensor>& bias,
                          std::optional<at::ScalarType> out_dtype,
                          at::IntArrayRef contraction_dim,
                          bool use_fast_accum) {
#else
at::Tensor AtenScaledMmV2(const at::Tensor& self, const at::Tensor& mat2,
                          at::TensorList scale_a, at::IntArrayRef recipe_a,
                          at::IntArrayRef swizzle_a, at::TensorList scale_b,
                          at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
                          const std::optional<at::Tensor>& bias,
                          std::optional<at::ScalarType> out_dtype,
                          at::IntArrayRef contraction_dim,
                          bool use_fast_accum) {
#endif
  TT_KERNEL(
      OpName::kScaledMmV2, param_keys,
      (self, mat2, scale_a, recipe_a, swizzle_a, scale_b, recipe_b, swizzle_b,
       bias, out_dtype, contraction_dim, use_fast_accum),
      {
        at::ScalarType target_scalar_type =
            out_dtype.has_value() ? *out_dtype : self.scalar_type();
        TT_ASSIGN_OR_THROW(at::Tensor out,
                           MakeEmptyTensor({self.size(0), mat2.size(1)},
                                           target_scalar_type, self.device()));
        TT_THROW_IF_ERROR(
            ScaledMmV2Out(self, mat2, scale_a, recipe_a, swizzle_a, scale_b,
                          recipe_b, swizzle_b, bias, out_dtype, contraction_dim,
                          use_fast_accum, std::move(param_keys), out));
        return out;
      });
}

#if TT_TORCH_VERSION_GE(2, 14)  // See scaled_mm_aten_kernels.h.
at::Tensor& AtenScaledMmV2Out(
    const at::Tensor& self, const at::Tensor& mat2,
    const at::ITensorListRef& scale_a, at::IntArrayRef recipe_a,
    at::IntArrayRef swizzle_a, const at::ITensorListRef& scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim,
    bool use_fast_accum, at::Tensor& out) {
#else
at::Tensor& AtenScaledMmV2Out(const at::Tensor& self, const at::Tensor& mat2,
                              at::TensorList scale_a, at::IntArrayRef recipe_a,
                              at::IntArrayRef swizzle_a, at::TensorList scale_b,
                              at::IntArrayRef recipe_b,
                              at::IntArrayRef swizzle_b,
                              const std::optional<at::Tensor>& bias,
                              std::optional<at::ScalarType> out_dtype,
                              at::IntArrayRef contraction_dim,
                              bool use_fast_accum, at::Tensor& out) {
#endif
  TT_KERNEL(OpName::kScaledMmV2Out, param_keys,
            (self, mat2, scale_a, recipe_a, swizzle_a, scale_b, recipe_b,
             swizzle_b, bias, out_dtype, contraction_dim, use_fast_accum, out),
            {
              TT_THROW_IF_ERROR(ScaledMmV2Out(
                  self, mat2, scale_a, recipe_a, swizzle_a, scale_b, recipe_b,
                  swizzle_b, bias, out_dtype, contraction_dim, use_fast_accum,
                  std::move(param_keys), out));
              return out;
            });
}

}  // namespace torch_tpu
