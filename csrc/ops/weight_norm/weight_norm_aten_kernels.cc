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

#include "csrc/ops/weight_norm/weight_norm_aten_kernels.h"

#include <cstdint>
#include <tuple>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/linalg/vector_norm/pnorm.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {
// Supports the following configurations for the weight magnitude `g`:
// - Scalar: Reduces all dimensions of `v` to compute a single global norm.
// - 1D Tensor: Reduces all dimensions of `v` except `dim`.
// - Same-rank Tensor with Singletons: Keeps the reduced dimensions as 1s,
//   matching `g`'s shape (e.g., `g` is [C, 1, 1] for `v` [C, H, W] with dim=0).
//
// Intermediate norm calculations (sum of squares) are performed in F32
// precision to prevent numerical overflow in FP16/BF16.
absl::StatusOr<MlirOpResults<2>> BuildWeightNormShlo(mlir::MlirOp v_op,
                                                     mlir::MlirOp g_op,
                                                     int64_t dim) {
  // Promote to F32 for calculations if needed
  TT_ASSIGN_OR_RETURN(v_op, PromoteFloatDtype(v_op));
  TT_ASSIGN_OR_RETURN(g_op, PromoteFloatDtype(g_op));

  auto v_type = GetTensorTypeOrDie(v_op);
  auto v_rank = v_type.getRank();
  auto g_rank = GetTensorTypeOrDie(g_op).getRank();
  TT_ASSIGN_OR_RETURN(auto element_type,
                      ConvertTo<mlir::ElementType>(v_type.getElementType()));

  Dimensions reduce_dims;
  reduce_dims.reserve(v_rank);
  if (g_rank == 0) {
    // If g is a scalar, we can drop all dimensions
    for (int i = 0; i < v_rank; ++i) {
      reduce_dims.push_back(i);
    }
  } else {
    // Otherwise, reduce all dimensions except the one specified by dim
    for (int i = 0; i < v_rank; ++i) {
      if (i != dim) {
        reduce_dims.push_back(i);
      }
    }
  }

  ReductionMode reduction_mode =
      (g_rank == v_rank) ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
  TT_ASSIGN_OR_RETURN(
      auto v_norm,
      BuildPNormShlo(v_op, /*ord=*/2.0, /*reduce_dims=*/reduce_dims,
                     /*reduction_mode=*/reduction_mode, element_type));

  // Align v_norm shape with g_op (e.g., if g is [C, 1, 1] but v_norm is [C])
  TT_ASSIGN_OR_RETURN(auto v_norm_aligned, BroadcastIfNeeded(v_norm, g_op));
  auto inv_norm = mlir::stablehlo::Div(g_op, v_norm_aligned);

  Dimensions broadcast_dims;
  broadcast_dims.reserve(v_rank);
  auto inv_norm_rank = GetTensorTypeOrDie(inv_norm).getRank();
  if (inv_norm_rank == v_rank) {
    for (int i = 0; i < v_rank; ++i) broadcast_dims.push_back(i);
  } else if (inv_norm_rank > 0) {
    broadcast_dims.push_back(dim);
  }

  auto inv_norm_bcst =
      mlir::stablehlo::BroadcastInDim(v_type, inv_norm, broadcast_dims);

  auto weight_norm = mlir::stablehlo::Mul(v_op, inv_norm_bcst);

  return MlirOpResults<2>{weight_norm, v_norm_aligned};
}

// Builds the StableHLO graph for weight_norm backward, following PyTorch ATen's
// native analytical backward formulation (see WeightNorm.cpp and
// WeightNormKernel.cpp).
//
// Forward formula:
//   w = g * (v / ||v||)
//
// Analytical gradients:
//   grad_g = sum(grad_w * v) / ||v||
//   grad_v = (g / ||v||) * (grad_w - v * (grad_g / ||v||))
//
// where:
//   - sum(...) reduces over all dimensions of v (if g is a scalar) or all
//     dimensions except `dim`.
//   - ||v|| is `norms_op`.
absl::StatusOr<MlirOpResults<2>> BuildWeightNormBackwardShlo(
    mlir::MlirOp grad_w_op, mlir::MlirOp v_op, mlir::MlirOp g_op,
    mlir::MlirOp norms_op, int64_t dim) {
  auto v_orig_type = GetTensorTypeOrDie(v_op);
  auto g_orig_type = GetTensorTypeOrDie(g_op);
  auto v_rank = v_orig_type.getRank();
  auto g_rank = g_orig_type.getRank();

  // Like PyTorch's native CPU and CUDA kernels (which dispatch with
  // accscalar_t = at::opmath_type<scalar_t> and evaluate in float32 for
  // BFloat16/Half to maintain numerical stability during reductions and
  // divisions), promote lower-precision inputs (FP16/BF16) to F32. F32 and F64
  // inputs remain unchanged.
  TT_ASSIGN_OR_RETURN(grad_w_op, PromoteFloatDtype(grad_w_op));
  TT_ASSIGN_OR_RETURN(v_op, PromoteFloatDtype(v_op));
  TT_ASSIGN_OR_RETURN(g_op, PromoteFloatDtype(g_op));
  TT_ASSIGN_OR_RETURN(norms_op, PromoteFloatDtype(norms_op));

  auto v_calc_type = GetTensorTypeOrDie(v_op);
  TT_ASSIGN_OR_RETURN(auto element_type, ConvertTo<mlir::ElementType>(
                                             v_calc_type.getElementType()));

  // Determine reduction dimensions:
  // - If g is a scalar (rank 0), reduce all dimensions of v.
  // - If g is per-slice (rank > 0), reduce all dimensions except `dim`.
  // - If g has the same rank as v (singleton dims), preserve dimensions
  //   (ReductionMode::kKeepDims); otherwise drop them
  //   (ReductionMode::kDropDims).
  Dimensions reduce_dims;
  reduce_dims.reserve(v_rank);
  for (int i = 0; i < v_rank; ++i) {
    if (g_rank == 0 || i != dim) {
      reduce_dims.push_back(i);
    }
  }

  ReductionMode reduction_mode =
      (g_rank == v_rank) ? ReductionMode::kKeepDims : ReductionMode::kDropDims;

  // Step 1: Compute inner product per_dim_sums = sum(grad_w * v) over the
  // reduction dimensions.
  auto gw_v = mlir::stablehlo::Mul(grad_w_op, v_op);
  TT_ASSIGN_OR_RETURN(
      auto per_dim_sums,
      BuildSumShlo(gw_v, reduce_dims, reduction_mode, element_type));

  // Align shapes with g_op (e.g., if g is [C, 1, 1] but per_dim_sums is [C]).
  TT_ASSIGN_OR_RETURN(auto per_dim_sums_aligned,
                      BroadcastIfNeeded(per_dim_sums, g_op));
  TT_ASSIGN_OR_RETURN(auto norms_aligned, BroadcastIfNeeded(norms_op, g_op));

  // Step 2: Compute grad_g = per_dim_sums / ||v||
  auto grad_g = mlir::stablehlo::Div(per_dim_sums_aligned, norms_aligned);

  // Step 3: Compute grad_v = (g / ||v||) * (grad_w - v * (grad_g / ||v||))
  //
  // First compute the projection scaling factor:
  //   v_scale = grad_g / ||v|| = per_dim_sums / (||v||^2)
  // and broadcast it to match v's shape.
  Dimensions broadcast_dims;
  broadcast_dims.reserve(v_rank);
  auto g_calc_rank = GetTensorTypeOrDie(g_op).getRank();
  if (g_calc_rank == v_rank) {
    for (int i = 0; i < v_rank; ++i) broadcast_dims.push_back(i);
  } else if (g_calc_rank > 0) {
    broadcast_dims.push_back(dim);
  }

  auto v_scale = mlir::stablehlo::Div(grad_g, norms_aligned);
  auto v_scale_bcst =
      mlir::stablehlo::BroadcastInDim(v_calc_type, v_scale, broadcast_dims);

  // diff = grad_w - v * v_scale
  auto v_scaled = mlir::stablehlo::Mul(v_op, v_scale_bcst);
  auto diff = mlir::stablehlo::Subtract(grad_w_op, v_scaled);

  // inv_norm = g / ||v||
  auto inv_norm = mlir::stablehlo::Div(g_op, norms_aligned);
  auto inv_norm_bcst =
      mlir::stablehlo::BroadcastInDim(v_calc_type, inv_norm, broadcast_dims);

  // grad_v = diff * inv_norm
  auto grad_v = mlir::stablehlo::Mul(diff, inv_norm_bcst);

  // Convert outputs back to original dtypes if they were promoted.
  auto maybe_convert = [](mlir::MlirOp op, mlir::Type target_type) {
    return GetTensorTypeOrDie(op).getElementType() != target_type
               ? mlir::stablehlo::ConvertElementType(op, target_type)
               : op;
  };
  grad_v = maybe_convert(grad_v, v_orig_type.getElementType());
  grad_g = maybe_convert(grad_g, g_orig_type.getElementType());

  return MlirOpResults<2>{grad_v, grad_g};
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> AtenWeightNormInterface(const at::Tensor& v,
                                                           const at::Tensor& g,
                                                           const int64_t dim) {
  TT_KERNEL(OpName::kWeightNormInterface, param_keys, (v, g, dim), {
    const int64_t v_rank = v.dim();
    TT_ASSIGN_OR_THROW(auto wrapped_dim, SafeWrapDim(dim, v_rank));

    TT_CHECK_THROW(IsFloatingPoint(v), error::kInvalidArgument)
        << "expected the input dtype to be floating point, got "
        << ToString(v.scalar_type());
    TT_CHECK_THROW(v_rank > 0, error::kIndexError)
        << "expected v to have at least 1 dimension, got " << v_rank;
    TT_CHECK_THROW(wrapped_dim == 0 || wrapped_dim == v.dim() - 1,
                   error::kInvalidArgument)
        << "expected dim to be 0 or the last dimension of v, got " << dim;

    // Note: CUDA/CPU does not explicitly check on the shape of `g`. It may
    // perform unsafe out-of-bounds reads if `g` is smaller than expected,
    // or silently ignore extra elements if `g` is larger.
    //
    // However, on TPU, StableHLO enforces strict shape compatibility for
    // elementwise division and broadcasting at compile time. Since
    // `torch._weight_norm()` is an internal API called via public wrappers like
    // `torch.nn.utils.weight_norm()` (which guarantee valid `g` shapes), we
    // enforce strict validations here to ensure StableHLO compilation succeeds.
    const int64_t g_rank = g.dim();
    TT_CHECK_THROW(g_rank == 0 || g_rank == 1 || g_rank == v_rank,
                   error::kInvalidArgument)
        << "expected the weight magnitude (g) to be a scalar, a 1D tensor, "
           "or have the same rank as v, got a tensor of shape "
        << ToString(g.sizes());

    if (g_rank == 1) {
      TT_CHECK_THROW(g.size(0) == v.size(wrapped_dim), error::kInvalidArgument)
          << "expected weight magnitude (g) size 0 to match weight size "
          << v.size(wrapped_dim) << " at dimension " << wrapped_dim << ", got "
          << g.size(0);
    } else if (g_rank == v_rank) {
      for (int64_t i = 0; i < v_rank; ++i) {
        if (i == wrapped_dim) {
          TT_CHECK_THROW(g.size(i) == v.size(wrapped_dim),
                         error::kInvalidArgument)
              << "expected the size of the weight magnitude (g) at dimension "
              << i << " to be " << v.size(wrapped_dim) << ", got " << g.size(i);
        } else {
          TT_CHECK_THROW(g.size(i) == 1, error::kInvalidArgument)
              << "expected the size of the weight magnitude (g) at dimension "
              << i << " to be 1, got " << g.size(i);
        }
      }
    }

    TT_ASSIGN_OR_THROW(const auto out_dtype,
                       ConvertTo<mlir::ElementType>(v.scalar_type()));

    // the output dtype is expected to be F32 when the input dtype precision
    // is lower than F32
    auto norm_dtype = (v.scalar_type() == at::ScalarType::Half ||
                       v.scalar_type() == at::ScalarType::BFloat16)
                          ? mlir::ElementType::F32
                          : out_dtype;

    auto op_builder = [wrapped_dim](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<MlirOpResults<2>> {
      return BuildWeightNormShlo(inputs[0], inputs[1], wrapped_dim);
    };

    TT_ASSIGN_OR_THROW(
        (auto [weight_norm_buf, v_norm_buf]),
        (DispatchOp<2, 2>(std::move(op_builder), {v, g},
                          {.out_dtypes = {out_dtype, norm_dtype},
                           .out_dims_list = {CopyIntVector(v.sizes()),
                                             CopyIntVector(g.sizes())},
                           .op_param_cache_keys = std::move(param_keys)})));

    return std::make_tuple(MakeTensor(std::move(weight_norm_buf)),
                           MakeTensor(std::move(v_norm_buf)));
  });
}

std::tuple<at::Tensor, at::Tensor> AtenWeightNormInterfaceBackward(
    const at::Tensor& grad_w, const at::Tensor& saved_v,
    const at::Tensor& saved_g, const at::Tensor& saved_norms,
    const int64_t dim) {
  TT_KERNEL(
      OpName::kWeightNormInterfaceBackward, param_keys,
      (grad_w, saved_v, saved_g, saved_norms, dim), {
        const int64_t v_rank = saved_v.dim();
        TT_ASSIGN_OR_THROW(auto wrapped_dim, SafeWrapDim(dim, v_rank));

        TT_ASSIGN_OR_THROW(const auto grad_v_dtype,
                           ConvertTo<mlir::ElementType>(saved_v.scalar_type()));
        TT_ASSIGN_OR_THROW(const auto grad_g_dtype,
                           ConvertTo<mlir::ElementType>(saved_g.scalar_type()));

        auto op_builder = [wrapped_dim](FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<MlirOpResults<2>> {
          return BuildWeightNormBackwardShlo(inputs[0], inputs[1], inputs[2],
                                             inputs[3], wrapped_dim);
        };

        TT_ASSIGN_OR_THROW(
            (auto [grad_v_buf, grad_g_buf]),
            (DispatchOp<4, 2>(
                std::move(op_builder), {grad_w, saved_v, saved_g, saved_norms},
                {.out_dtypes = {grad_v_dtype, grad_g_dtype},
                 .out_dims_list = {CopyIntVector(saved_v.sizes()),
                                   CopyIntVector(saved_g.sizes())},
                 .op_param_cache_keys = std::move(param_keys)})));

        return std::make_tuple(MakeTensor(std::move(grad_v_buf)),
                               MakeTensor(std::move(grad_g_buf)));
      });
}

}  // namespace torch_tpu
