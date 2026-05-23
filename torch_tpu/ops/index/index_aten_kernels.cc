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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Optional.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/index/index.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/masked_select/masked_select_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

struct IndicesInfo {
  // Defined tensor indices.
  //
  // In contrast to the `indices_list_opt`, this is a list of tensors, where all
  // tensors are defined. However, they might not correspond to consecutive
  // dimensions.
  std::vector<at::Tensor> indices;

  // List of indexed dimensions.
  //
  // For each tensor `indices[i]`, `dimensions[i]` represents which dimension
  // `indices[i]` indexes.
  Indices dimensions;
};

// Preprocess and check the `indices_list_opt` input into `IndicesInfo`.
absl::StatusOr<IndicesInfo> CheckedGetIndicesInfo(
    const c10::List<c10::optional<at::Tensor>>& indices_list_opt) {
  IndicesInfo info;

  for (int64_t i = 0; i < indices_list_opt.size(); ++i) {
    const auto& tensor = indices_list_opt[i];

    if (tensor.has_value() && tensor->defined()) {
      // TODO(unda): Add support for more than one bool index tensor.
      TT_RET_CHECK(!IsBool(*tensor) || info.indices.empty(),
                   error::kUnimplemented)
          << "indexing with more than one bool tensor is not yet supported";

      info.dimensions.push_back(i);
      info.indices.push_back(*tensor);
    }
  }

  TT_RET_CHECK(!info.indices.empty(), error::kInvalidArgument)
      << "at least one index tensor must be defined";

  return info;
}

absl::StatusOr<Dimensions> GetOutputDims(const at::Tensor& self,
                                         const IndicesInfo& info) {
  // The shape of the output tensor is the combination of the shape of the
  // (broadcasted) index tensors and the shape of the unindexed dimensions of
  // the input tensor. Unless the indexed dimensions are consecutive, they will
  // be mapped to the first dimension of the output. See
  // https://numpy.org/devdocs/user/basics.indexing.html#combining-advanced-and-basic-indexing.
  Dimensions broadcasted_index_shape;
  Dimensions unindexed_dims_shape;

  bool indexed_dimensions_consecutive = true;

  for (int i = 0, j = 0; i < self.dim(); ++i) {
    if (j < info.dimensions.size() && i == info.dimensions[j]) {
      if (j > 0 && indexed_dimensions_consecutive &&
          info.dimensions[j - 1] != i - 1) {
        indexed_dimensions_consecutive = false;
      }

      TT_ASSIGN_OR_RETURN(
          broadcasted_index_shape,
          InferSize(broadcasted_index_shape, info.indices[j].sizes()));

      j++;
    } else {
      unindexed_dims_shape.push_back(self.size(i));
    }
  }

  Dimensions output_dims;
  output_dims.reserve(broadcasted_index_shape.size() +
                      unindexed_dims_shape.size());

  size_t insertion_index =
      indexed_dimensions_consecutive ? info.dimensions[0] : 0;

  output_dims.insert(output_dims.end(), unindexed_dims_shape.begin(),
                     unindexed_dims_shape.begin() + insertion_index);
  output_dims.insert(output_dims.end(), broadcasted_index_shape.begin(),
                     broadcasted_index_shape.end());
  output_dims.insert(output_dims.end(),
                     unindexed_dims_shape.begin() + insertion_index,
                     unindexed_dims_shape.end());

  return output_dims;
}

}  // namespace

at::Tensor& AtenIndexTensorOut(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices_list_opt,
    at::Tensor& out) {
  TT_KERNEL(
      OpName::kIndexTensorOut, _,
      (self, IgnoreInCacheKey(indices_list_opt, "Legacy usage"), out), {
        TT_CHECK_THROW(indices_list_opt.size() <= self.dim(),
                       error::kIndexError)
            << "expected the size of the indices to be <= " << self.dim()
            << " (number of input dimensions), got " << indices_list_opt.size();

        TT_ASSIGN_OR_THROW(IndicesInfo info,
                           CheckedGetIndicesInfo(indices_list_opt));

        // If the indices are a single boolean tensor, use masked_select.
        if (info.indices.size() == 1 &&
            info.indices[0].scalar_type() == at::kBool) {
          // We need to align the indexing dimensions to the left first.
          auto indexing_tensor = info.indices[0];

          TT_CHECK_THROW(indexing_tensor.dim() <= self.dim(),
                         error::kIndexError)
              << "expected the size of the indices to be <= " << self.dim()
              << " (number of input dimensions), got " << indexing_tensor.dim();

          Dimensions trailing_dims(self.sizes().begin() + indexing_tensor.dim(),
                                   self.sizes().end());

          if (!trailing_dims.empty()) {
            Dimensions unsqueezed_index_shape =
                CopyIntVector(indexing_tensor.sizes());
            unsqueezed_index_shape.insert(unsqueezed_index_shape.end(),
                                          self.dim() - indexing_tensor.dim(),
                                          1);
            indexing_tensor = indexing_tensor.reshape(unsqueezed_index_shape);
          }

          AtenMaskedSelectOut(self, indexing_tensor, out);
          if (!trailing_dims.empty()) {
            trailing_dims.insert(trailing_dims.begin(), -1);
            out = out.reshape(trailing_dims);
            return out;
          }
          return out;
        }

        TT_ASSIGN_OR_THROW(Dimensions output_dims, GetOutputDims(self, info));
        // The indices_list_opt gets ignored in the cache key, but we still
        // want to record which dimensions are indexed.
        const auto& dimensions = info.dimensions;
        TT_ASSIGN_OR_THROW(auto param_keys,
                           TT_MAKE_OP_PARAM_CACHE_KEYS(dimensions));

        ABSL_VLOG(2) << "[AtenIndexTensorOut] self: " << ToString(self);
        for (const auto& t : info.indices) {
          ABSL_VLOG(2) << "[AtenIndexTensorOut] indices_list: " << ToString(t);
        }
        ABSL_VLOG(2) << "[AtenIndexTensorOut] indexed_dims: "
                     << ToString(info.dimensions);
        ABSL_VLOG(2) << "[AtenIndexTensorOut] output_dims: "
                     << ToString(output_dims);

        if (self.dim() == 0) {
          output_dims = {};
        }

        std::vector<at::Tensor> self_and_indices(std::move(info.indices));
        self_and_indices.insert(self_and_indices.begin(), self);

        TT_ASSIGN_OR_THROW(const auto computation_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto index_op_builder =
            [indexed_dimensions = std::move(info.dimensions)](
                absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder) {
              return BuildIndexShlo(inputs, indexed_dimensions);
            };

        TT_ASSIGN_OR_THROW(
            auto result_buf,
            DispatchOp<kDynamicSize>(
                std::move(index_op_builder), self_and_indices,
                {.out_dtype = computation_dtype,
                 .out_dims = absl::Span<const int64_t>(output_dims),
                 .op_param_cache_keys = std::move(param_keys)}));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
