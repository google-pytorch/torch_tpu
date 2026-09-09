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

}  // namespace torch_tpu
