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

#include "torch_tpu/ops/experimental/ragged_dot_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/result_type.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

static absl::StatusOr<mlir::MlirOp> BuildRaggedDotShlo(
    mlir::MlirOp lhs, mlir::MlirOp rhs, mlir::MlirOp group_sizes,
    mlir::ElementType output_element_type) {
  const mlir::RankedTensorType lhs_type = GetTensorTypeOrDie(lhs);
  const mlir::RankedTensorType rhs_type = GetTensorTypeOrDie(rhs);
  // TODO(pfilipiuk): Add precision to the API.
  // Assuming mk, gkn, g -> mn
  auto dimension_numbers = mlir::chlo::RaggedDotDimensionNumbersAttr::get(
      &lhs.getContext(), /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{},
      /*lhsContractingDimensions=*/{1},
      /*rhsContractingDimensions=*/{1},
      /*lhsRaggedDimensions=*/{0},
      /*rhsGroupDimensions=*/{0});
  mlir::Type out_type = mlir::makeTensorType(
      lhs.getContext(), {lhs_type.getShape()[0], rhs_type.getShape()[2]},
      output_element_type);
  return mlir::chlo::RaggedDot(out_type, lhs, rhs, group_sizes,
                               dimension_numbers);
}

static absl::StatusOr<DeviceBufferRef> RaggedDotCommon(
    const at::Tensor& lhs, const at::Tensor& rhs, const at::Tensor& group_sizes,
    OpParamCacheKeys& param_keys) {
  // ragged_dot(mk, gkn, g) -> mn
  TT_RET_CHECK(lhs.dim() == 2, error::kInvalidArgument)
      << "lhs must be 2D, got dim: " << lhs.dim();
  TT_RET_CHECK(rhs.dim() == 3, error::kInvalidArgument)
      << "rhs must be 3D, got dim: " << rhs.dim();
  TT_RET_CHECK(group_sizes.dim() == 1, error::kInvalidArgument)
      << "group_sizes must be 1D, got dim: " << group_sizes.dim();
  TT_RET_CHECK(lhs.size(1) == rhs.size(1), error::kInvalidArgument)
      << "contracting dimension should be the same, got: " << lhs.size(1)
      << " vs " << rhs.size(1);
  TT_RET_CHECK(rhs.size(0) == group_sizes.size(0), error::kInvalidArgument)
      << "lhs and group_sizes should have the same number of groups, got: "
      << rhs.size(0) << " vs " << group_sizes.size(0);

  at::ScalarType out_scalar_type = at::result_type(lhs, rhs);
  TT_ASSIGN_OR_RETURN(auto out_dtype,
                      ConvertTo<mlir::ElementType>(out_scalar_type));
  auto op_builder = [out_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
    auto& [lhs, rhs, group_sizes] = inputs;
    return BuildRaggedDotShlo(lhs, rhs, group_sizes, out_dtype);
  };
  return DispatchOp<3>(std::move(op_builder), {lhs, rhs, group_sizes},
                       {.out_dtype = out_dtype,
                        .out_dims = {lhs.size(0), rhs.size(2)},
                        .op_param_cache_keys = std::move(param_keys)});
}
}  // namespace

at::Tensor AtenRaggedDot(const at::Tensor& lhs, const at::Tensor& rhs,
                         const at::Tensor& group_sizes) {
  TT_KERNEL(OpName::kRaggedDot, param_keys, (lhs, rhs, group_sizes), {
    TT_ASSIGN_OR_THROW(auto result,
                       RaggedDotCommon(lhs, rhs, group_sizes, param_keys));
    return MakeTensor(std::move(result));
  });
}

at::Tensor& AtenRaggedDotOut(const at::Tensor& lhs, const at::Tensor& rhs,
                             const at::Tensor& group_sizes, at::Tensor& out) {
  TT_KERNEL(OpName::kRaggedDot, param_keys, (lhs, rhs, group_sizes, out), {
    TT_ASSIGN_OR_THROW(auto result,
                       RaggedDotCommon(lhs, rhs, group_sizes, param_keys));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
    return out;
  });
}

}  // namespace torch_tpu
