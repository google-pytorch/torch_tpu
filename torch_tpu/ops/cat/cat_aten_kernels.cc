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

#include "torch_tpu/ops/cat/cat_aten_kernels.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/IListRef.h"
#include "ATen/core/TensorBase.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorImpl.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/cat/cat.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

struct CatComputationResult {
  DeviceBufferRef result_buf;
  at::ScalarType promoted_dtype;
  int64_t num_dims;
  Dimensions output_dims;
};

struct CatShapeInfo {
  int64_t num_dims;
  int64_t wrapped_dim;
  Dimensions output_dims;
};

absl::StatusOr<CatShapeInfo> ValidateCatTensors(
    const at::ITensorListRef& tensors, const int64_t dim) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches this error first.
      !tensors.empty(), error::kInvalidArgument)
      << "expect a non-empty list of Tensors.";

  // Reject scalars.
  int i = 0;
  for (const at::Tensor& tensor : tensors) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches this error first.
        tensor.dim() > 0, error::kInvalidArgument)
        << "zero-dimensional tensor (at position " << i
        << ") cannot be concatenated";
    ++i;
  }

  // Find the tensor with the most dimensions.
  const at::Tensor* tensor_with_most_dims = &(*tensors.begin());
  for (const at::Tensor& tensor : tensors) {
    if (tensor.dim() > tensor_with_most_dims->dim()) {
      tensor_with_most_dims = &tensor;
    }
  }
  ABSL_CHECK_NE(tensor_with_most_dims, nullptr);  // CRASH_OK
  const int64_t max_num_dims = tensor_with_most_dims->dim();
  TT_ASSIGN_OR_RETURN(const int64_t wrapped_dim,
                      SafeWrapDim(dim, max_num_dims));

  Dimensions output_dims = CopyIntVector(tensor_with_most_dims->sizes());
  output_dims[wrapped_dim] = 0;

  for (const at::Tensor& tensor : tensors) {
    if (tensor.dim() == 1 && tensor.size(0) == 0) continue;
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches this error first.
        tensor.dim() == max_num_dims, error::kInvalidArgument)
        << "tensors must have same number of dimensions or be 1D with "
           "size (0,), got "
        << max_num_dims << " and " << tensor.dim();
    for (int i = 0; i < max_num_dims; ++i) {
      if (i == wrapped_dim) continue;
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches this error first.
          tensor.size(i) == tensor_with_most_dims->size(i),
          error::kInvalidArgument)
          << "tensor sizes must match except at dimension " << wrapped_dim
          << ", got " << tensor.size(i) << " and "
          << tensor_with_most_dims->size(i) << " at dimension " << i;
    }
    output_dims[wrapped_dim] += tensor.size(wrapped_dim);
  }

  return CatShapeInfo{max_num_dims, wrapped_dim, std::move(output_dims)};
}

absl::StatusOr<CatComputationResult> CatHelper(
    const at::ITensorListRef& tensors, const int64_t dim,
    OpParamCacheKeys param_keys,
    std::optional<at::ScalarType> output_dtype_override = std::nullopt) {
  ABSL_VLOG(1) << "====== [C++ KERNEL AtenCat] ======";
  TT_ASSIGN_OR_RETURN(CatShapeInfo shape_info,
                      ValidateCatTensors(tensors, dim));
  const int64_t num_dims = shape_info.num_dims;
  const int64_t wrapped_dim = shape_info.wrapped_dim;
  Dimensions output_dims = std::move(shape_info.output_dims);

  // Get the first tensor to initialize promotion.
  const at::Tensor& first_tensor = *tensors.begin();

  // Iterative type promotion using a range-based for loop.
  at::ScalarType target_dtype = first_tensor.scalar_type();
  for (const at::Tensor& tensor : tensors) {
    target_dtype = c10::promoteTypes(target_dtype, tensor.scalar_type());
  }
  if (output_dtype_override.has_value()) {
    TT_RET_CHECK(at::canCast(target_dtype, *output_dtype_override),
                 error::kInvalidArgument)
        << "expected the input to be castable to the desired dtype "
        << ToString(*output_dtype_override) << ", got "
        << ToString(target_dtype);
    target_dtype = *output_dtype_override;
  }
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(target_dtype));

  if (std::any_of(output_dims.begin(), output_dims.end(),
                  [](int64_t size) { return size == 0; })) {
    // If the concatenation is exclusively 0-element tensors, then the result is
    // also a 0-element tensor, and we can skip stablehlo::ConcatenateOp().
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef result_buf,
        DeviceBufferList::CreateZeroSize(output_dims, output_dtype));
    return CatComputationResult{
        .result_buf = std::move(result_buf),
        .promoted_dtype = target_dtype,
        .num_dims = num_dims,
        .output_dims = std::move(output_dims),
    };
  }

  std::vector<at::Tensor> promoted_tensors;
  promoted_tensors.reserve(tensors.size());
  for (const at::Tensor& tensor : tensors) {
    // Skip 1D tensors with size (0,). They don't contribute to the output.
    // They will cause mlir::stablehlo::ConcatenateOp() to crash.
    if (tensor.dim() == 1 && tensor.size(0) == 0) continue;
    promoted_tensors.push_back(tensor.toType(target_dtype));
  }

  ABSL_VLOG(1) << "[AtenCat] Inferred output shape: ["
               << absl::StrJoin(output_dims, ",") << "]";

  auto cat_op_builder = [wrapped_dim](absl::Span<mlir::MlirOp> inputs,
                                      mlir::MlirBuilder& builder) {
    return BuildCatShlo(inputs, wrapped_dim);
  };

  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<kDynamicSize>(std::move(cat_op_builder), promoted_tensors,
                               {.out_dtype = output_dtype,
                                .out_dims = output_dims,
                                .op_param_cache_keys = std::move(param_keys)}));
  return CatComputationResult{
      .result_buf = std::move(result_buf),
      .promoted_dtype = target_dtype,
      .num_dims = num_dims,
      .output_dims = std::move(output_dims),
  };
}

}  // namespace

at::Tensor& AtenCatOut(const at::ITensorListRef& tensors, int64_t dim,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kCatOut, param_keys, (tensors, dim, out), {
    TT_ASSIGN_OR_THROW(
        CatComputationResult cat_result,
        CatHelper(tensors, dim, std::move(param_keys), out.scalar_type()));
    TT_THROW_IF_ERROR(
        AssignBufferToAtTensor(std::move(cat_result.result_buf), out));
    ABSL_VLOG(1) << "[C++ KERNEL tpu_aten_cat_out] out(final): "
                 << ToString(out, "out");
    return out;
  });
}

}  // namespace torch_tpu
