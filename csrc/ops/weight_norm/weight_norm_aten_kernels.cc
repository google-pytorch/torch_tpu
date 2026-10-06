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

absl::StatusOr<MlirOpResults<2>> BuildWeightNormBackwardShlo(
    mlir::MlirOp grad_w_op, mlir::MlirOp v_op, mlir::MlirOp g_op,
    mlir::MlirOp norms_op, int64_t dim) {
  auto v_orig_type = GetTensorTypeOrDie(v_op);
  auto g_orig_type = GetTensorTypeOrDie(g_op);
  auto v_rank = v_orig_type.getRank();
  auto g_rank = g_orig_type.getRank();

  // Promote to F32 for calculations if needed
  TT_ASSIGN_OR_RETURN(grad_w_op, PromoteFloatDtype(grad_w_op));
  TT_ASSIGN_OR_RETURN(v_op, PromoteFloatDtype(v_op));
  TT_ASSIGN_OR_RETURN(g_op, PromoteFloatDtype(g_op));
  TT_ASSIGN_OR_RETURN(norms_op, PromoteFloatDtype(norms_op));

  auto v_calc_type = GetTensorTypeOrDie(v_op);
  TT_ASSIGN_OR_RETURN(auto element_type, ConvertTo<mlir::ElementType>(
                                             v_calc_type.getElementType()));

  Dimensions reduce_dims;
  reduce_dims.reserve(v_rank);
  if (g_rank == 0) {
    for (int i = 0; i < v_rank; ++i) {
      reduce_dims.push_back(i);
    }
  } else {
    for (int i = 0; i < v_rank; ++i) {
      if (i != dim) {
        reduce_dims.push_back(i);
      }
    }
  }

  ReductionMode reduction_mode =
      (g_rank == v_rank) ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
  auto gw_v = mlir::stablehlo::Mul(grad_w_op, v_op);
  TT_ASSIGN_OR_RETURN(
      auto per_dim_sums,
      BuildSumShlo(gw_v, reduce_dims, reduction_mode, element_type));

  // Align per_dim_sums shape with g_op (e.g., if g is [C, 1, 1] but
  // per_dim_sums is [C])
  TT_ASSIGN_OR_RETURN(auto per_dim_sums_aligned,
                      BroadcastIfNeeded(per_dim_sums, g_op));
  TT_ASSIGN_OR_RETURN(auto norms_aligned, BroadcastIfNeeded(norms_op, g_op));

  // grad_g = per_dim_sums / norms
  auto grad_g = mlir::stablehlo::Div(per_dim_sums_aligned, norms_aligned);

  Dimensions broadcast_dims;
  broadcast_dims.reserve(v_rank);
  auto g_calc_rank = GetTensorTypeOrDie(g_op).getRank();
  if (g_calc_rank == v_rank) {
    for (int i = 0; i < v_rank; ++i) broadcast_dims.push_back(i);
  } else if (g_calc_rank > 0) {
    broadcast_dims.push_back(dim);
  }

  // v_scale = per_dim_sums / (norms * norms)
  auto norms_sq = mlir::stablehlo::Mul(norms_aligned, norms_aligned);
  auto v_scale = mlir::stablehlo::Div(per_dim_sums_aligned, norms_sq);
  auto v_scale_bcst =
      mlir::stablehlo::BroadcastInDim(v_calc_type, v_scale, broadcast_dims);

  auto v_scaled = mlir::stablehlo::Mul(v_op, v_scale_bcst);
  auto diff = mlir::stablehlo::Subtract(grad_w_op, v_scaled);

  // inv_norm = g / norms
  auto inv_norm = mlir::stablehlo::Div(g_op, norms_aligned);
  auto inv_norm_bcst =
      mlir::stablehlo::BroadcastInDim(v_calc_type, inv_norm, broadcast_dims);

  auto grad_v = mlir::stablehlo::Mul(diff, inv_norm_bcst);

  auto v_elem_type = v_orig_type.getElementType();
  auto g_elem_type = g_orig_type.getElementType();
  if (GetTensorTypeOrDie(grad_v).getElementType() != v_elem_type) {
    grad_v = mlir::stablehlo::ConvertElementType(grad_v, v_elem_type);
  }
  if (GetTensorTypeOrDie(grad_g).getElementType() != g_elem_type) {
    grad_g = mlir::stablehlo::ConvertElementType(grad_g, g_elem_type);
  }

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

        TT_CHECK_THROW(IsFloatingPoint(saved_v), error::kInvalidArgument)
            << "expected the input dtype to be floating point, got "
            << ToString(saved_v.scalar_type());
        TT_CHECK_THROW(saved_v.scalar_type() == grad_w.scalar_type(),
                       error::kInvalidArgument)
            << "expected grad_w to have the same dtype as saved_v, got "
            << ToString(grad_w.scalar_type()) << " and "
            << ToString(saved_v.scalar_type());
        TT_CHECK_THROW(v_rank > 0, error::kIndexError)
            << "expected v to have at least 1 dimension, got " << v_rank;
        TT_CHECK_THROW(wrapped_dim == 0 || wrapped_dim == saved_v.dim() - 1,
                       error::kInvalidArgument)
            << "expected dim to be 0 or the last dimension of v, got " << dim;
        TT_CHECK_THROW(grad_w.sizes() == saved_v.sizes(),
                       error::kInvalidArgument)
            << "expected grad_w to have the same shape as saved_v, got "
            << ToString(grad_w.sizes()) << " and " << ToString(saved_v.sizes());
        TT_CHECK_THROW(saved_norms.sizes() == saved_g.sizes(),
                       error::kInvalidArgument)
            << "expected saved_norms to have the same shape as saved_g, got "
            << ToString(saved_norms.sizes()) << " and "
            << ToString(saved_g.sizes());

        const int64_t g_rank = saved_g.dim();
        TT_CHECK_THROW(g_rank == 0 || g_rank == 1 || g_rank == v_rank,
                       error::kInvalidArgument)
            << "expected the weight magnitude (g) to be a scalar, a 1D tensor, "
               "or have the same rank as v, got a tensor of shape "
            << ToString(saved_g.sizes());

        if (g_rank == 1) {
          TT_CHECK_THROW(saved_g.size(0) == saved_v.size(wrapped_dim),
                         error::kInvalidArgument)
              << "expected weight magnitude (g) size 0 to match weight size "
              << saved_v.size(wrapped_dim) << " at dimension " << wrapped_dim
              << ", got " << saved_g.size(0);
        } else if (g_rank == v_rank) {
          for (int64_t i = 0; i < v_rank; ++i) {
            if (i == wrapped_dim) {
              TT_CHECK_THROW(saved_g.size(i) == saved_v.size(wrapped_dim),
                             error::kInvalidArgument)
                  << "expected the size of the weight magnitude (g) at "
                     "dimension "
                  << i << " to be " << saved_v.size(wrapped_dim) << ", got "
                  << saved_g.size(i);
            } else {
              TT_CHECK_THROW(saved_g.size(i) == 1, error::kInvalidArgument)
                  << "expected the size of the weight magnitude (g) at "
                     "dimension "
                  << i << " to be 1, got " << saved_g.size(i);
            }
          }
        }

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
