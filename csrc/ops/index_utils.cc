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

#include "csrc/ops/index_utils.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/copy_from/cpu_to_tpu.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/where/where.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

// PyTorch's dispatcher bypasses backend device guards for index tensors
// during indexing out-variant and composite execution, allowing them to arrive
// on CPU. We use CopyCpuToTpuBuffer instead of ATen's .to() to transfer CPU
// tensors to TPU without triggering dispatcher re-entry or CompositeOpCheck
// failures during tracing.
absl::StatusOr<at::Tensor> ResolveAuxiliaryTensorDevice(
    const at::Tensor& tensor) {
  if (!tensor.defined() ||
      tensor.device().type() == GetPrivateUse1DeviceType()) {
    return tensor;
  }
  TT_ASSIGN_OR_RETURN(DeviceBufferRef tpu_buffer,
                      CopyCpuToTpuBuffer(tensor, /*non_blocking=*/false));
  return MakeTensor(std::move(tpu_buffer));
}

}  // namespace

absl::Status ResolveNegativeIndices(std::vector<at::Tensor>& indices,
                                    const at::IntArrayRef sizes,
                                    const Indices& dimensions) {
  std::vector<at::Tensor> indices_to_resolve;
  std::vector<at::Tensor> dim_size_tensors;
  Indices original_indices;

  for (size_t i = 0; i < indices.size(); ++i) {
    at::Tensor& index = indices[i];
    if (index.scalar_type() == c10::ScalarType::Long ||
        index.scalar_type() == c10::ScalarType::Int) {
      TT_ASSIGN_OR_RETURN(index, ResolveAuxiliaryTensorDevice(index));
      indices_to_resolve.push_back(index);

      const int64_t dim_size = sizes[dimensions[i]];
      auto promoted_dim_size = PromoteScalar(at::Scalar(dim_size));
      TT_ASSIGN_OR_RETURN(at::Tensor dim_size_tensor,
                          promoted_dim_size.GetTensor(index.scalar_type()));
      dim_size_tensors.push_back(dim_size_tensor);

      original_indices.push_back(i);
    }
  }

  if (indices_to_resolve.empty()) {
    return absl::OkStatus();
  }

  size_t num_resolutions = indices_to_resolve.size();
  std::vector<at::Tensor> inputs;
  inputs.reserve(num_resolutions * 2);
  inputs.insert(inputs.end(), indices_to_resolve.begin(),
                indices_to_resolve.end());
  inputs.insert(inputs.end(), dim_size_tensors.begin(), dim_size_tensors.end());

  std::vector<mlir::ElementType> out_dtypes;
  std::vector<Dimensions> out_dims_storage;
  std::vector<absl::Span<const int64_t>> out_dims_list;

  out_dtypes.reserve(num_resolutions);
  out_dims_storage.reserve(num_resolutions);
  out_dims_list.reserve(num_resolutions);

  for (const auto& index : indices_to_resolve) {
    TT_ASSIGN_OR_RETURN(auto dtype,
                        ConvertTo<mlir::ElementType>(index.scalar_type()));
    out_dtypes.push_back(dtype);

    out_dims_storage.push_back(
        Dimensions(index.sizes().begin(), index.sizes().end()));
  }

  for (const auto& dims : out_dims_storage) {
    out_dims_list.push_back(dims);
  }

  auto op_builder =
      [num_resolutions](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<DynamicMlirOpResults> {
    DynamicMlirOpResults results;
    results.reserve(num_resolutions);

    for (size_t i = 0; i < num_resolutions; ++i) {
      mlir::MlirOp index_op = inputs[i];
      mlir::MlirOp dim_size_op = inputs[i + num_resolutions];

      auto index_type = GetTensorTypeOrDie(index_op);
      auto element_type = index_type.getElementType();
      auto zero = MakeScalarConstant(builder, 0, element_type);

      TT_ASSIGN_OR_RETURN(auto lt_cond, BuildLtShlo(index_op, zero));
      TT_ASSIGN_OR_RETURN(auto added_index,
                          BuildAddShlo(index_op, dim_size_op));
      TT_ASSIGN_OR_RETURN(auto output_element_type, GetElementType(index_op));
      // NOTE: This logic resolves negative indices but doesn't check for
      // out-of-bounds (OOB) results. For example, if `index` is `-6` and
      // `dim_size` is `5`, the result is `-1`. PyTorch typically throws an
      // `IndexError` in this case.
      TT_ASSIGN_OR_RETURN(
          auto resolved_index,
          BuildWhereShlo(lt_cond, added_index, index_op, output_element_type));
      results.push_back(resolved_index);
    }
    return results;
  };

  DispatchOpOptions<kDynamicSize> options = {
      .op_name = OpName::kResolveNegativeIndices,
      .out_dtypes = out_dtypes,
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
  };

  TT_ASSIGN_OR_RETURN(std::vector<DeviceBufferRef> result_bufs,
                      (DispatchOp<kDynamicSize, kDynamicSize>(
                          std::move(op_builder), inputs, std::move(options))));

  for (size_t i = 0; i < num_resolutions; ++i) {
    size_t original_idx = original_indices[i];
    indices[original_idx] = MakeTensor(std::move(result_bufs[i]));
  }

  return absl::OkStatus();
}

absl::StatusOr<int64_t> ValidateIndexInputsAndGetDim(const at::Tensor& self,
                                                     int64_t dim,
                                                     const at::Tensor& index,
                                                     const at::Tensor& source) {
  TT_RET_CHECK(index.dim() == 1, error::kInvalidArgument)
      << "expected index to be 1D, got shape " << index.sizes();
  if (self.dim() == 0) {
    TT_RET_CHECK(dim == 0, error::kInvalidArgument)
        << "expected dim to be 0 for scalar input, got " << dim;
    TT_RET_CHECK(source.dim() == 0, error::kInvalidArgument)
        << "expected source shape to match self shape, excluding the specified "
           "dimension, got source shape "
        << source.sizes() << " and self shape " << self.sizes();
    TT_RET_CHECK(index.size(0) == 1, error::kInvalidArgument)
        << "expected index to be 1D of size 1 for scalar input, got shape "
        << index.sizes();
    return dim;
  }

  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self.dim()));

  TT_RET_CHECK(source.dim() == self.dim(), error::kPythonIndexError)
      << "expected self and source to have the same number of dimensions, got "
      << self.dim() << " and " << source.dim();
  TT_RET_CHECK(source.size(dim) == index.size(0), error::kInvalidArgument)
      << "expected source to have the same number of elements as the index "
         "along "
         "dimension "
      << dim << ", got " << source.size(dim) << " and " << index.size(0);
  for (int i = 0; i < self.dim(); ++i) {
    if (i != dim) {
      TT_RET_CHECK(self.size(i) == source.size(i), error::kInvalidArgument)
          << "expected self and source to have the same size along dimension "
          << i << ", got " << self.size(i) << " and " << source.size(i);
    }
  }
  return dim;
}

}  // namespace torch_tpu
