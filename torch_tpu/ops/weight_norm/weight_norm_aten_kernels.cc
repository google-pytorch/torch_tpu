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

#include "torch_tpu/ops/weight_norm/weight_norm_aten_kernels.h"

#include <cstdint>
#include <tuple>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/linalg/vector_norm/pnorm.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {
namespace {
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
  Dimensions broadcast_dims;
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
    broadcast_dims.push_back(dim);
  }

  TT_ASSIGN_OR_RETURN(
      auto v_norm,
      BuildPNormShlo(v_op, /*ord=*/2.0, /*reduce_dims=*/reduce_dims,
                     /*reduction_mode=*/ReductionMode::kDropDims,
                     element_type));

  auto inv_norm = mlir::stablehlo::Div(g_op, v_norm);

  auto inv_norm_bcst =
      mlir::stablehlo::BroadcastInDim(v_type, inv_norm, broadcast_dims);

  auto weight_norm = mlir::stablehlo::Mul(v_op, inv_norm_bcst);

  return MlirOpResults<2>{weight_norm, v_norm};
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> AtenWeightNormInterface(const at::Tensor& v,
                                                           const at::Tensor& g,
                                                           const int64_t dim) {
  TT_KERNEL(OpName::kWeightNormInterface, param_keys, (v, g, dim), {
    TT_CHECK_THROW(IsFloatingPoint(v), error::kInvalidArgument)
        << "expected the input dtype to be floating point, got "
        << v.scalar_type();
    TT_CHECK_THROW(v.dim() > 0, error::kIndexError)
        << "expected v to have at least 1 dimension, got " << v.dim();
    TT_CHECK_THROW(dim == 0 || dim == v.dim() - 1, error::kInvalidArgument)
        << "expected dim to be 0 or the last dimension of v, got " << dim;

    TT_CHECK_THROW(g.dim() <= 1, error::kInvalidArgument)
        << "expected the weight magnitude (g) to be a scalar or a 1D tensor, "
           "got a tensor of shape "
        << ToString(g.sizes());

    if (g.dim() == 1) {
      TT_CHECK_THROW(g.size(0) == v.size(dim), error::kInvalidArgument)
          << "expected the size of the weight magnitude (g) to be "
          << v.size(dim) << ", which is the size of the weight at dimension "
          << dim << ", got " << g.size(0);
    }

    TT_ASSIGN_OR_THROW(const auto out_dtype,
                       ConvertTo<mlir::ElementType>(v.scalar_type()));

    // the output dtype is expected to be F32 when the input dtype precision
    // is lower than F32
    auto norm_dtype = (v.scalar_type() == at::ScalarType::Half ||
                       v.scalar_type() == at::ScalarType::BFloat16)
                          ? mlir::ElementType::F32
                          : out_dtype;

    auto op_builder = [dim](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<MlirOpResults<2>> {
      return BuildWeightNormShlo(inputs[0], inputs[1], dim);
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
