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

#include "torch_tpu/csrc/ops/distance/dist_aten_kernels.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/common/aten_utils.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/binary.h"
#include "torch_tpu/csrc/ops/index_select/index_select.h"
#include "torch_tpu/csrc/ops/linalg/vector_norm/pnorm.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/nullary_aten_kernels.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/reductions/reductions.h"
#include "torch_tpu/csrc/ops/unary.h"

namespace torch_tpu {

namespace {

// When p=0, compute the 0-norm (Hamming distance equivalent)
// which is defined as the count of non-zero elements in the vector:
//
// 1. Compare input elements with 0.0 to get a boolean tensor.
// 2. Cast the boolean tensor to the input floating-point type (0.0 or 1.0).
// 3. Sum the values along the reduction dimension.
absl::StatusOr<mlir::MlirOp> BuildZeroNorm(mlir::MlirOp input,
                                           int64_t reduce_dim,
                                           mlir::Type element_type) {
  // Create a scalar zero and broadcast it for comparison
  mlir::MlirBuilder& builder = input.getBuilder();

  auto zero = MakeScalarConstant(builder, 0.0, element_type);
  auto input_type = GetTensorTypeOrDie(input);
  auto zero_bcast = mlir::stablehlo::BroadcastInDim(input_type, zero, {});

  auto is_non_zero = mlir::stablehlo::Compare(
      input, zero_bcast, mlir::stablehlo::ComparisonDirection::NE);

  // Convert bool to float (0.0 or 1.0)
  auto is_non_zero_casted = mlir::stablehlo::Convert(input_type, is_non_zero);

  auto reduce_op = mlir::stablehlo::Reduce(
      builder,
      /*inputs=*/{is_non_zero_casted},
      /*init_values=*/{zero},
      /*body_builder=*/
      [&](mlir::RegionBuilder& rb) {
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            element_type, rb.getRegion(), rb.getOpBuilder());
      },
      /*dimensions=*/{reduce_dim});

  return reduce_op[0];
}

struct FeatureDimMapping {
  int64_t first_dim;
  int64_t second_dim;
};

// Helper to broadcast cdist inputs (x1, x2, grad, cdist) to target shape
// based on their feature dimension mappings.
mlir::MlirOp BroadcastCdistInput(mlir::MlirOp op,
                                 mlir::RankedTensorType target_bcast_type,
                                 const int64_t common_batch_rank,
                                 FeatureDimMapping feature_dim_mappings) {
  const mlir::RankedTensorType op_type = GetTensorTypeOrDie(op);
  const int64_t op_rank = op_type.getRank();
  const int64_t op_batch_dims_count = op_rank - 2;
  const int64_t offset = common_batch_rank - op_batch_dims_count;

  Dimensions bcast_dims;
  bcast_dims.reserve(op_rank);
  for (int i = 0; i < op_batch_dims_count; ++i) {
    bcast_dims.push_back(i + offset);
  }
  bcast_dims.push_back(feature_dim_mappings.first_dim);
  bcast_dims.push_back(feature_dim_mappings.second_dim);

  return mlir::stablehlo::BroadcastInDim(target_bcast_type, op, bcast_dims);
}

// Computes the p-norm distance of the difference between each pair of row
// vectors in x1 and x2.
// x1 shape: (B..., R1, C)
// x2 shape: (B..., R2, C)
// Output shape: (B..., R1, R2)
absl::StatusOr<mlir::MlirOp> BuildCdistForwardShlo(
    mlir::MlirOp x1_op, mlir::MlirOp x2_op, double p, int64_t compute_mode,
    const Dimensions common_batch_shape, int64_t r1, int64_t r2, int64_t c) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_type, GetElementType(x1_op));
  TT_ASSIGN_OR_RETURN(mlir::ElementType compute_dtype,
                      InferComputationDtype(out_type));

  TT_ASSIGN_OR_RETURN(x1_op, CastIfNeeded(x1_op, compute_dtype));
  TT_ASSIGN_OR_RETURN(x2_op, CastIfNeeded(x2_op, compute_dtype));

  const mlir::RankedTensorType x1_type = GetTensorTypeOrDie(x1_op);
  const mlir::Type element_type = x1_type.getElementType();

  // Define the target shape for the difference tensor: (B_common..., R1, R2, C)
  Dimensions target_diff_shape(common_batch_shape.begin(),
                               common_batch_shape.end());
  target_diff_shape.insert(target_diff_shape.end(), {r1, r2, c});
  const int64_t common_batch_rank = common_batch_shape.size();
  mlir::RankedTensorType bcast_type =
      mlir::RankedTensorType::get(target_diff_shape, element_type);

  // Broadcast X1 to (B_common..., R1, 1, C)
  mlir::MlirOp x1_bcast =
      BroadcastCdistInput(x1_op, bcast_type, common_batch_rank,
                          {common_batch_rank, common_batch_rank + 2});

  // Broadcast X2 to (B_common..., 1, R2, C)
  mlir::MlirOp x2_bcast =
      BroadcastCdistInput(x2_op, bcast_type, common_batch_rank,
                          {common_batch_rank + 1, common_batch_rank + 2});

  // Compute diff = X1 - X2
  mlir::MlirOp diff_op = mlir::stablehlo::Subtract(x1_bcast, x2_bcast);

  // Reduce over the C dimension.
  int64_t reduce_dim_idx = common_batch_rank + 2;

  if (p == 0.0) {
    // Special handling for 0-norm (hamming distance equivalent)
    TT_ASSIGN_OR_RETURN(mlir::MlirOp res,
                        BuildZeroNorm(diff_op, reduce_dim_idx, element_type));
    return CastIfNeeded(res, out_type);
  } else {
    // General case for p-norm
    return BuildPNormShlo(diff_op, p, {reduce_dim_idx},
                          ReductionMode::kDropDims, out_type);
  };
}

// Computes the p-norm distance of the difference between each pair of elements
// in input.
// Input shape: (B..., R, C)
// Output shape: (B..., R(R-1)/2)
absl::StatusOr<mlir::MlirOp> BuildPdistForwardHlo(mlir::MlirOp input_op,
                                                  double p) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_type, GetElementType(input_op));
  TT_ASSIGN_OR_RETURN(mlir::ElementType compute_dtype,
                      InferComputationDtype(out_type));
  TT_ASSIGN_OR_RETURN(input_op, CastIfNeeded(input_op, compute_dtype));

  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  const mlir::Type element_type = input_type.getElementType();

  const int64_t n = input_type.getDimSize(0);
  const int64_t num_pairs = n * (n - 1) / 2;

  // Create tensors of indices for pairs (i, j) such that i < j.
  Dimensions idx_left;
  Dimensions idx_right;
  idx_left.reserve(num_pairs);
  idx_right.reserve(num_pairs);

  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = i + 1; j < n; ++j) {
      idx_left.push_back(i);
      idx_right.push_back(j);
    }
  }

  auto indices_type = mlir::RankedTensorType::get(
      {num_pairs}, builder.getOpBuilder().getI64Type());
  mlir::MlirOp idx_left_op = mlir::stablehlo::Constant(
      builder, mlir::cast<mlir::DenseElementsAttr>(mlir::makeConstant(
                   llvm::ArrayRef<int64_t>(idx_left), indices_type)));
  mlir::MlirOp idx_right_op = mlir::stablehlo::Constant(
      builder, mlir::cast<mlir::DenseElementsAttr>(mlir::makeConstant(
                   llvm::ArrayRef<int64_t>(idx_right), indices_type)));

  mlir::MlirOp lhs = BuildIndexSelectShlo(input_op, /*dim=*/0, idx_left_op);
  mlir::MlirOp rhs = BuildIndexSelectShlo(input_op, /*dim=*/0, idx_right_op);

  mlir::MlirOp diff_op = mlir::stablehlo::Subtract(lhs, rhs);

  if (p == 0.0) {
    // Special handling for 0-norm (hamming distance equivalent)
    TT_ASSIGN_OR_RETURN(mlir::MlirOp res,
                        BuildZeroNorm(diff_op, /*reduce_dim=*/1, element_type));
    return CastIfNeeded(res, out_type);
  } else {
    // General case for p-norm
    return BuildPNormShlo(diff_op, p, {1}, ReductionMode::kDropDims, out_type);
  }
}

absl::StatusOr<mlir::MlirOp> BuildPdistBackwardShlo(mlir::MlirOp grad_op,
                                                    mlir::MlirOp self_op,
                                                    double p,
                                                    mlir::MlirOp pdist_op) {
  mlir::MlirBuilder& builder = self_op.getBuilder();
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::Type element_type = self_type.getElementType();
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_type, GetElementType(self_op));

  const int64_t n = self_type.getDimSize(0);
  const int64_t m = self_type.getDimSize(1);
  const int64_t num_pairs = n * (n - 1) / 2;

  // Create tensors of indices for pairs (i, j) such that i < j.
  Dimensions idx_left;
  Dimensions idx_right;
  idx_left.reserve(num_pairs);
  idx_right.reserve(num_pairs);

  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = i + 1; j < n; ++j) {
      idx_left.push_back(i);
      idx_right.push_back(j);
    }
  }

  auto indices_type = mlir::RankedTensorType::get(
      {num_pairs}, builder.getOpBuilder().getI64Type());
  mlir::MlirOp idx_left_op = mlir::stablehlo::Constant(
      builder, mlir::cast<mlir::DenseElementsAttr>(mlir::makeConstant(
                   llvm::ArrayRef<int64_t>(idx_left), indices_type)));
  mlir::MlirOp idx_right_op = mlir::stablehlo::Constant(
      builder, mlir::cast<mlir::DenseElementsAttr>(mlir::makeConstant(
                   llvm::ArrayRef<int64_t>(idx_right), indices_type)));

  mlir::MlirOp lhs = BuildIndexSelectShlo(self_op, /*dim=*/0, idx_left_op);
  mlir::MlirOp rhs = BuildIndexSelectShlo(self_op, /*dim=*/0, idx_right_op);

  mlir::MlirOp diff = mlir::stablehlo::Subtract(lhs, rhs);

  // Reshape grad and pdist from (num_pairs) to (num_pairs, 1)
  mlir::RankedTensorType reshaped_type =
      mlir::RankedTensorType::get({num_pairs, 1}, element_type);
  mlir::MlirOp grad_reshaped = mlir::stablehlo::Reshape(reshaped_type, grad_op);
  mlir::MlirOp pdist_reshaped =
      mlir::stablehlo::Reshape(reshaped_type, pdist_op);

  // Broadcast grad and pdist to (num_pairs, m)
  mlir::RankedTensorType bcast_type =
      mlir::RankedTensorType::get({num_pairs, m}, element_type);
  mlir::MlirOp grad_bcast =
      mlir::stablehlo::BroadcastInDim(bcast_type, grad_reshaped, {0, 1});
  mlir::MlirOp pdist_bcast =
      mlir::stablehlo::BroadcastInDim(bcast_type, pdist_reshaped, {0, 1});

  mlir::MlirOp unreduced_grad;
  if (p == 0.0) {
    // Case 1: p = 0.0
    auto zero = MakeScalarConstant(builder, 0.0, out_type);
    unreduced_grad = mlir::stablehlo::BroadcastInDim(bcast_type, zero, {});
  } else if (p == 1.0) {
    // Case 2: p = 1.0
    TT_ASSIGN_OR_RETURN(auto sign_diff, BuildSignShlo(diff));
    TT_ASSIGN_OR_RETURN(unreduced_grad, BuildMulShlo(grad_bcast, sign_diff));
  } else if (p == 2.0) {
    // Case 3: p = 2.0
    auto zero = MakeScalarConstant(builder, 0.0, out_type);
    auto zero_bcast = mlir::stablehlo::BroadcastInDim(bcast_type, zero, {});
    auto one = MakeScalarConstant(builder, 1.0, out_type);
    auto one_bcast = mlir::stablehlo::BroadcastInDim(bcast_type, one, {});
    auto cdist_is_zero = mlir::stablehlo::Compare(
        pdist_bcast, zero_bcast, mlir::stablehlo::ComparisonDirection::EQ);
    auto safe_cdist =
        mlir::stablehlo::Select(cdist_is_zero, one_bcast, pdist_bcast);
    TT_ASSIGN_OR_RETURN(auto grad_mul_diff, BuildMulShlo(grad_bcast, diff));
    TT_ASSIGN_OR_RETURN(auto div, BuildDivShlo(grad_mul_diff, safe_cdist));
    unreduced_grad = mlir::stablehlo::Select(cdist_is_zero, zero_bcast, div);
  } else if (std::isinf(p)) {
    // Case 4: p = infinity
    TT_ASSIGN_OR_RETURN(auto sign_diff, BuildSignShlo(diff));
    TT_ASSIGN_OR_RETURN(auto abs_diff, BuildAbsShlo(diff));
    auto is_max = mlir::stablehlo::Compare(
        abs_diff, pdist_bcast, mlir::stablehlo::ComparisonDirection::EQ);
    auto is_max_float = mlir::stablehlo::Convert(bcast_type, is_max);
    TT_ASSIGN_OR_RETURN(auto grad_mul_sign,
                        BuildMulShlo(grad_bcast, sign_diff));
    TT_ASSIGN_OR_RETURN(unreduced_grad,
                        BuildMulShlo(grad_mul_sign, is_max_float));
  } else {
    // Case 5: General p
    auto zero = MakeScalarConstant(builder, 0.0, out_type);
    auto zero_bcast = mlir::stablehlo::BroadcastInDim(bcast_type, zero, {});
    auto one = MakeScalarConstant(builder, 1.0, out_type);
    auto one_bcast = mlir::stablehlo::BroadcastInDim(bcast_type, one, {});

    auto cdist_is_zero = mlir::stablehlo::Compare(
        pdist_bcast, zero_bcast, mlir::stablehlo::ComparisonDirection::EQ);

    auto diff_is_zero = mlir::stablehlo::Compare(
        diff, zero_bcast, mlir::stablehlo::ComparisonDirection::EQ);

    mlir::MlirOp cond = cdist_is_zero;
    if (p < 1.0) {
      cond = mlir::stablehlo::Or(cdist_is_zero, diff_is_zero);
    }

    auto p_minus_1 = MakeScalarConstant(builder, p - 1.0, out_type);
    auto p_minus_1_bcast =
        mlir::stablehlo::BroadcastInDim(bcast_type, p_minus_1, {});

    TT_ASSIGN_OR_RETURN(auto abs_diff, BuildAbsShlo(diff));
    auto safe_abs_diff = mlir::stablehlo::Select(cond, one_bcast, abs_diff);
    auto safe_cdist = mlir::stablehlo::Select(cond, one_bcast, pdist_bcast);

    TT_ASSIGN_OR_RETURN(auto pow_diff,
                        BuildPowShlo(safe_abs_diff, p_minus_1_bcast));
    TT_ASSIGN_OR_RETURN(auto pow_cdist,
                        BuildPowShlo(safe_cdist, p_minus_1_bcast));

    TT_ASSIGN_OR_RETURN(auto sign_diff, BuildSignShlo(diff));
    TT_ASSIGN_OR_RETURN(auto term1, BuildMulShlo(sign_diff, pow_diff));
    TT_ASSIGN_OR_RETURN(auto term2, BuildMulShlo(term1, grad_bcast));
    TT_ASSIGN_OR_RETURN(auto div, BuildDivShlo(term2, pow_cdist));

    unreduced_grad = mlir::stablehlo::Select(cond, zero_bcast, div);
  }

  // Create self_grad filled with zeros
  auto zero_scalar = MakeScalarConstant(builder, 0.0, out_type);
  mlir::MlirOp self_grad =
      mlir::stablehlo::BroadcastInDim(self_type, zero_scalar, {});

  // Reshape indices to (num_pairs, 1) for Scatter
  auto indices_scatter_type = mlir::RankedTensorType::get(
      {num_pairs, 1}, builder.getOpBuilder().getI64Type());
  mlir::MlirOp idx_left_scatter =
      mlir::stablehlo::Reshape(indices_scatter_type, idx_left_op);
  mlir::MlirOp idx_right_scatter =
      mlir::stablehlo::Reshape(indices_scatter_type, idx_right_op);

  // Setup scatter dimensions numbers (scatter to dim 0 of self_grad)
  mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers =
      mlir::stablehlo::ScatterDimensionNumbersAttr::get(
          &self_op.getContext(),
          /*update_window_dims=*/{1},
          /*inserted_window_dims=*/{0},
          /*input_batching_dims=*/{},
          /*scatter_indices_batching_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0},
          /*index_vector_dim=*/1);

  // Setup add region builder
  auto block_type = self_type.clone({}, element_type);
  auto add_region_builder = [block_type](mlir::RegionBuilder& r_builder) {
    auto arg0 = mlir::Argument(r_builder, block_type);
    auto arg1 = mlir::Argument(r_builder, block_type);
    mlir::MlirOp result = mlir::stablehlo::Add(arg0, arg1);
    mlir::stablehlo::Return(r_builder, {result});
  };

  // Setup sub region builder (for rhs contribution)
  auto sub_region_builder = [block_type](mlir::RegionBuilder& r_builder) {
    auto arg0 = mlir::Argument(r_builder, block_type);
    auto arg1 = mlir::Argument(r_builder, block_type);
    mlir::MlirOp result = mlir::stablehlo::Subtract(arg0, arg1);
    mlir::stablehlo::Return(r_builder, {result});
  };

  // Scatter add unreduced_grad to left indices
  mlir::MlirOp grad_left = mlir::stablehlo::Scatter(
      {self_grad}, idx_left_scatter, {unreduced_grad}, add_region_builder,
      scatter_dimension_numbers)[0];

  // Scatter subtract unreduced_grad from right indices
  mlir::MlirOp grad_final = mlir::stablehlo::Scatter(
      {grad_left}, idx_right_scatter, {unreduced_grad}, sub_region_builder,
      scatter_dimension_numbers)[0];

  return grad_final;
}

absl::StatusOr<mlir::MlirOp> BuildCdistBackwardShlo(
    mlir::MlirOp grad_op, mlir::MlirOp x1_op, mlir::MlirOp x2_op, double p,
    mlir::MlirOp cdist_op, const Dimensions common_batch_shape, int64_t r1,
    int64_t r2, int64_t c) {
  const mlir::RankedTensorType x1_type = GetTensorTypeOrDie(x1_op);
  const mlir::Type element_type = x1_type.getElementType();
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_type, GetElementType(x1_op));
  mlir::MlirBuilder& builder = x1_op.getBuilder();

  const int64_t common_batch_rank = common_batch_shape.size();
  Dimensions target_diff_shape(common_batch_shape.begin(),
                               common_batch_shape.end());
  target_diff_shape.insert(target_diff_shape.end(), {r1, r2, c});
  mlir::RankedTensorType bcast_type =
      mlir::RankedTensorType::get(target_diff_shape, element_type);

  // Broadcast X1 to (B_common..., R1, 1, C)
  mlir::MlirOp x1_bcast =
      BroadcastCdistInput(x1_op, bcast_type, common_batch_rank,
                          {common_batch_rank, common_batch_rank + 2});

  // Broadcast X2 to (B_common..., 1, R2, C)
  mlir::MlirOp x2_bcast =
      BroadcastCdistInput(x2_op, bcast_type, common_batch_rank,
                          {common_batch_rank + 1, common_batch_rank + 2});

  // Broadcast grad to (B_common..., R1, R2, C)
  mlir::MlirOp grad_bcast =
      BroadcastCdistInput(grad_op, bcast_type, common_batch_rank,
                          {common_batch_rank, common_batch_rank + 1});

  // Broadcast cdist to (B_common..., R1, R2, C)
  mlir::MlirOp cdist_bcast =
      BroadcastCdistInput(cdist_op, bcast_type, common_batch_rank,
                          {common_batch_rank, common_batch_rank + 1});

  // diff = x1_bcast - x2_bcast
  TT_ASSIGN_OR_RETURN(mlir::MlirOp diff, BuildSubShlo(x1_bcast, x2_bcast));

  mlir::MlirOp unreduced_grad;
  if (p == 0.0) {
    // Case 1: p = 0.0 (Hamming distance gradient is 0.0)
    auto zero = MakeScalarConstant(builder, 0.0, out_type);
    unreduced_grad = mlir::stablehlo::BroadcastInDim(bcast_type, zero, {});
  } else if (p == 1.0) {
    // Case 2: p = 1.0 (Manhattan distance, gradient: grad * sign(diff))
    TT_ASSIGN_OR_RETURN(auto sign_diff, BuildSignShlo(diff));
    TT_ASSIGN_OR_RETURN(unreduced_grad, BuildMulShlo(grad_bcast, sign_diff));
  } else if (p == 2.0) {
    // Case 3: p = 2.0 (Euclidean distance, gradient: grad * diff / cdist)
    // Replace cdist with 1.0 where cdist is 0.0 to avoid division-by-zero,
    // then mask those positions to 0.0.
    auto zero = MakeScalarConstant(builder, 0.0, out_type);
    auto zero_bcast = mlir::stablehlo::BroadcastInDim(bcast_type, zero, {});
    auto one = MakeScalarConstant(builder, 1.0, out_type);
    auto one_bcast = mlir::stablehlo::BroadcastInDim(bcast_type, one, {});
    auto cdist_is_zero = mlir::stablehlo::Compare(
        cdist_bcast, zero_bcast, mlir::stablehlo::ComparisonDirection::EQ);
    auto safe_cdist =
        mlir::stablehlo::Select(cdist_is_zero, one_bcast, cdist_bcast);
    TT_ASSIGN_OR_RETURN(auto grad_mul_diff, BuildMulShlo(grad_bcast, diff));
    TT_ASSIGN_OR_RETURN(auto div, BuildDivShlo(grad_mul_diff, safe_cdist));
    unreduced_grad = mlir::stablehlo::Select(cdist_is_zero, zero_bcast, div);
  } else if (std::isinf(p)) {
    // Case 4: p = infinity, gradient: grad * sign(diff) * [abs(diff) == cdist])
    TT_ASSIGN_OR_RETURN(auto sign_diff, BuildSignShlo(diff));
    TT_ASSIGN_OR_RETURN(auto abs_diff, BuildAbsShlo(diff));
    auto is_max = mlir::stablehlo::Compare(
        abs_diff, cdist_bcast, mlir::stablehlo::ComparisonDirection::EQ);
    auto is_max_float = mlir::stablehlo::Convert(bcast_type, is_max);
    TT_ASSIGN_OR_RETURN(auto grad_mul_sign,
                        BuildMulShlo(grad_bcast, sign_diff));
    TT_ASSIGN_OR_RETURN(unreduced_grad,
                        BuildMulShlo(grad_mul_sign, is_max_float));
  } else {
    // Case 5: General p (Minkowski distance)
    // Gradient: sign(diff) * |diff|^(p-1) * grad / cdist^(p-1)
    // Use safety condition to mask division-by-zero positions to 0.0.
    auto zero = MakeScalarConstant(builder, 0.0, out_type);
    auto zero_bcast = mlir::stablehlo::BroadcastInDim(bcast_type, zero, {});
    auto one = MakeScalarConstant(builder, 1.0, out_type);
    auto one_bcast = mlir::stablehlo::BroadcastInDim(bcast_type, one, {});

    auto cdist_is_zero = mlir::stablehlo::Compare(
        cdist_bcast, zero_bcast, mlir::stablehlo::ComparisonDirection::EQ);

    auto diff_is_zero = mlir::stablehlo::Compare(
        diff, zero_bcast, mlir::stablehlo::ComparisonDirection::EQ);

    mlir::MlirOp cond = cdist_is_zero;
    if (p < 1.0) {
      cond = mlir::stablehlo::Or(cdist_is_zero, diff_is_zero);
    }

    auto p_minus_1 = MakeScalarConstant(builder, p - 1.0, out_type);
    auto p_minus_1_bcast =
        mlir::stablehlo::BroadcastInDim(bcast_type, p_minus_1, {});

    TT_ASSIGN_OR_RETURN(auto abs_diff, BuildAbsShlo(diff));
    auto safe_abs_diff = mlir::stablehlo::Select(cond, one_bcast, abs_diff);
    auto safe_cdist = mlir::stablehlo::Select(cond, one_bcast, cdist_bcast);

    TT_ASSIGN_OR_RETURN(auto pow_diff,
                        BuildPowShlo(safe_abs_diff, p_minus_1_bcast));
    TT_ASSIGN_OR_RETURN(auto pow_cdist,
                        BuildPowShlo(safe_cdist, p_minus_1_bcast));

    TT_ASSIGN_OR_RETURN(auto sign_diff, BuildSignShlo(diff));
    TT_ASSIGN_OR_RETURN(auto term1, BuildMulShlo(sign_diff, pow_diff));
    TT_ASSIGN_OR_RETURN(auto term2, BuildMulShlo(term1, grad_bcast));
    TT_ASSIGN_OR_RETURN(auto div, BuildDivShlo(term2, pow_cdist));

    unreduced_grad = mlir::stablehlo::Select(cond, zero_bcast, div);
  }

  // Reduce sum over R2 (dimension common_batch_rank + 1)
  auto zero = MakeScalarConstant(builder, 0.0, out_type);
  auto reduce_op = mlir::stablehlo::Reduce(
      builder,
      /*inputs=*/{unreduced_grad},
      /*init_values=*/{zero},
      /*body_builder=*/
      [&](mlir::RegionBuilder& rb) {
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            element_type, rb.getRegion(), rb.getOpBuilder());
      },

      /*dimensions=*/{common_batch_rank + 1});

  return reduce_op[0];
}

bool IsBFloatOrHalf(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::ScalarType::BFloat16 ||
         tensor.scalar_type() == at::ScalarType::Half;
}

absl::Status CheckIsFloatingPoint(const at::Tensor& tensor,
                                  const std::string_view name) {
  TT_RET_CHECK(IsFloatingPoint(tensor), error::kInvalidArgument)
      << "expected the " << name << " dtype to be floating point, got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

absl::Status CheckNotBFloatOrHalf(const at::Tensor& tensor,
                                  const std::string_view name) {
  TT_RET_CHECK(!IsBFloatOrHalf(tensor), error::kInvalidArgument)
      << "expected the " << name << " dtype not to be bfloat16 or float16, got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

}  // namespace

at::Tensor AtenCdistForward(const at::Tensor& x1, const at::Tensor& x2,
                            double p, std::optional<int64_t> compute_mode) {
  TT_KERNEL(OpName::kCdistForward, param_keys, (x1, x2, p, compute_mode), {
    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(x1.scalar_type()));

    TT_THROW_IF_ERROR(CheckIsFloatingPoint(x1, "first argument's"));
    TT_THROW_IF_ERROR(CheckIsFloatingPoint(x2, "second argument's"));

    TT_CHECK_THROW(p >= 0, error::kInvalidArgument)
        << "expected the p value to be >= 0, got " << p;

    const int64_t mode = compute_mode.value_or(0);
    const int64_t c = x1.size(-1);
    const int64_t r1 = x1.size(-2);
    const int64_t r2 = x2.size(-2);

    Dimensions batch_tensor1(x1.sizes().begin(), x1.sizes().end() - 2);
    Dimensions batch_tensor2(x2.sizes().begin(), x2.sizes().end() - 2);
    TT_ASSIGN_OR_THROW(Dimensions common_batch_shape,
                       InferSize(batch_tensor1, batch_tensor2));

    // Output shape: (B_common..., R1, R2)
    Dimensions output_shape(common_batch_shape);
    output_shape.insert(output_shape.end(), {r1, r2});

    // If the output shape contains a 0 dimension, return an empty tensor
    for (int64_t dim : output_shape) {
      if (dim == 0) {
        TT_ASSIGN_OR_THROW(
            at::Tensor out,
            MakeEmptyTensor(output_shape, x1.scalar_type(), x1.device()));
        return out;
      }
    }

    // Add check for bf16 and float16 after the empty dimension check,
    // because they are supported for empty tensors but not for non-empty ones.
    TT_THROW_IF_ERROR(CheckNotBFloatOrHalf(x1, "first argument's"));
    TT_THROW_IF_ERROR(CheckNotBFloatOrHalf(x2, "second argument's"));

    auto op_builder = [p, mode, common_batch_shape, r1, r2,
                       c](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [x1_op, x2_op] = inputs;
      return BuildCdistForwardShlo(x1_op, x2_op, p, mode, common_batch_shape,
                                   r1, r2, c);
    };

    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<2>(std::move(op_builder), {x1, x2},
                      {.out_dtype = out_dtype,
                       .out_dims = output_shape,
                       .op_param_cache_keys = std::move(param_keys)}));

    return MakeTensor(std::move(result_buf));
  });
}

at::Tensor AtenCdistBackward(const at::Tensor& grad, const at::Tensor& x1,
                             const at::Tensor& x2, double p,
                             const at::Tensor& cdist) {
  TT_KERNEL(OpName::kCdistBackward, param_keys, (grad, x1, x2, p, cdist), {
    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(x1.scalar_type()));

    TT_THROW_IF_ERROR(CheckIsFloatingPoint(x1, "first argument's"));
    TT_THROW_IF_ERROR(CheckIsFloatingPoint(x2, "second argument's"));
    TT_THROW_IF_ERROR(CheckIsFloatingPoint(grad, "gradient"));
    TT_THROW_IF_ERROR(CheckIsFloatingPoint(cdist, "cdist"));

    TT_CHECK_THROW(p >= 0, error::kInvalidArgument)
        << "expected the p value to be >= 0, got " << p;

    const int64_t r1 = x1.size(-2);
    const int64_t c = x1.size(-1);
    const int64_t r2 = x2.size(-2);
    Dimensions batch_tensor1(x1.sizes().begin(), x1.sizes().end() - 2);
    Dimensions batch_tensor2(x2.sizes().begin(), x2.sizes().end() - 2);
    TT_ASSIGN_OR_THROW(Dimensions common_batch_shape,
                       InferSize(batch_tensor1, batch_tensor2));

    Dimensions out_shape(common_batch_shape);
    out_shape.insert(out_shape.end(), {r1, c});

    for (int64_t dim : out_shape) {
      if (dim == 0) {
        TT_ASSIGN_OR_THROW(
            at::Tensor out,
            MakeEmptyTensor(out_shape, x1.scalar_type(), x1.device()));
        return out;
      }
    }

    TT_THROW_IF_ERROR(CheckNotBFloatOrHalf(x1, "first argument's"));
    TT_THROW_IF_ERROR(CheckNotBFloatOrHalf(x2, "second argument's"));

    auto op_builder = [p, common_batch_shape, r1, r2,
                       c](FixedSizeSpan<mlir::MlirOp, 4> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [grad_op, x1_op, x2_op, cdist_op] = inputs;
      return BuildCdistBackwardShlo(grad_op, x1_op, x2_op, p, cdist_op,
                                    common_batch_shape, r1, r2, c);
    };

    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<4>(std::move(op_builder), {grad, x1, x2, cdist},
                      {.out_dtype = out_dtype,
                       .out_dims = out_shape,
                       .op_param_cache_keys = std::move(param_keys)}));

    return MakeTensor(std::move(result_buf));
  });
}

at::Tensor AtenPdistForward(const at::Tensor& self, double p) {
  TT_KERNEL(OpName::kPdistForward, param_keys, (self, p), {
    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));

    // If input has shape (n, m), output will have shape n(n-1)/2.
    const int64_t n = self.size(0);
    const int64_t num_pairs = n * (n - 1) / 2;
    Dimensions output_shape = {num_pairs};

    // If there are no pairs to compute the distance for, return an empty tensor
    if (num_pairs == 0) {
      TT_ASSIGN_OR_THROW(
          at::Tensor out,
          MakeEmptyTensor(output_shape, self.scalar_type(), self.device()));
      return out;
    }

    // Add check for bfloat16 and float16 after the empty dimension check,
    // because empty tensors are supported for these dtypes
    TT_THROW_IF_ERROR(CheckNotBFloatOrHalf(self, /* name= */ "input"));

    auto op_builder = [p](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildPdistForwardHlo(input, p);
    };

    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<1>(std::move(op_builder), self,
                      {.out_dtype = out_dtype,
                       .out_dims = output_shape,
                       .op_param_cache_keys = std::move(param_keys)}));

    return MakeTensor(std::move(result_buf));
  });
}

at::Tensor AtenPdistBackward(const at::Tensor& grad, const at::Tensor& self,
                             double p, const at::Tensor& pdist) {
  TT_KERNEL(OpName::kPdistBackward, param_keys, (grad, self, p, pdist), {
    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));

    TT_THROW_IF_ERROR(CheckIsFloatingPoint(self, "input"));
    TT_THROW_IF_ERROR(CheckIsFloatingPoint(grad, "gradient"));
    TT_THROW_IF_ERROR(CheckIsFloatingPoint(pdist, "pdist"));

    TT_CHECK_THROW(p >= 0, error::kInvalidArgument)
        << "expected the p value to be >= 0, got " << p;

    const int64_t n = self.size(0);
    const int64_t m = self.size(1);
    Dimensions output_shape = {n, m};

    // Handle empty inputs
    if (n <= 1 || m == 0) {
      TT_ASSIGN_OR_THROW(
          at::Tensor out,
          MakeEmptyTensor(output_shape, self.scalar_type(), self.device()));
      return out;
    }

    TT_THROW_IF_ERROR(CheckNotBFloatOrHalf(self, "input"));
    TT_THROW_IF_ERROR(CheckNotBFloatOrHalf(grad, "gradient"));
    TT_THROW_IF_ERROR(CheckNotBFloatOrHalf(pdist, "pdist"));

    auto op_builder = [p](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [grad_op, self_op, pdist_op] = inputs;
      return BuildPdistBackwardShlo(grad_op, self_op, p, pdist_op);
    };

    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<3>(std::move(op_builder), {grad, self, pdist},
                      {.out_dtype = out_dtype,
                       .out_dims = output_shape,
                       .op_param_cache_keys = std::move(param_keys)}));

    return MakeTensor(std::move(result_buf));
  });
}

}  // namespace torch_tpu
