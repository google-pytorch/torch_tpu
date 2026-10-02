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
#include "absl/functional/bind_front.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/Optional.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/index/index.h"
#include "csrc/ops/index_utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/nonzero/nonzero.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

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

  // Number of non-zero (`True`) elements when indexing with a boolean mask.
  std::optional<int64_t> num_non_zero_elements;
};

// Computes the number of non-zero (`True`) elements in `mask` on TPU via
// `BuildNonzeroGetOutputSizeShlo` and materializes the scalar count on CPU.
// This count is required ahead of time to determine the data-dependent output
// shape of boolean mask indexing.
absl::StatusOr<int64_t> GetNumNonZeroElements(const at::Tensor& mask) {
  if (mask.numel() == 0) {
    return 0;
  }
  TT_ASSIGN_OR_RETURN(at::Tensor num_non_zero_elements,
                      UnaryOp(mask,
                              absl::bind_front(BuildNonzeroGetOutputSizeShlo,
                                               mlir::ElementType::I64),
                              {.op_name = OpName::kNonzeroSize,
                               .op_param_cache_keys = OpParamCacheKeys::Empty(),
                               .out_dtype = mlir::ElementType::I64,
                               .out_dims = Dimensions()}));
  return num_non_zero_elements.cpu().item<int64_t>();
}

// Preprocess and check the `indices_list_opt` input into `IndicesInfo`.
absl::StatusOr<IndicesInfo> ValidateAndGetIndicesInfo(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices_list_opt) {
  IndicesInfo info;
  int64_t dim_indexed = 0;
  bool has_bool_mask = false;

  for (int64_t i = 0; i < indices_list_opt.size(); ++i) {
    const auto& tensor = indices_list_opt[i];

    // An undefined/nullopt entry corresponds to a full slice (`:`), which
    // advances the current dimension of `self` by 1.
    if (!tensor.has_value() || !tensor->defined()) {
      TT_RET_CHECK(dim_indexed < self.dim(), error::kIndexError)
          << "too many indices for tensor of dimension " << self.dim();
      dim_indexed++;
      continue;
    }

    const at::Tensor& index_tensor = *tensor;
    if (IsBool(index_tensor)) {
      // TODO(b/567574040): Add support for more than one bool index tensor.
      TT_RET_CHECK(!has_bool_mask && info.indices.empty(),
                   error::kPythonNotImplementedError)
          << "indexing with more than one bool tensor is not yet supported";
      has_bool_mask = true;

      // Validate that each dimension of the boolean mask matches the
      // corresponding dimension of `self` starting at `dim_indexed` (accounting
      // for any preceding `:` slices).
      for (int64_t mask_dim = 0; mask_dim < index_tensor.dim(); ++mask_dim) {
        const int64_t self_dim = dim_indexed + mask_dim;
        TT_RET_CHECK(self_dim < self.dim() &&
                         index_tensor.size(mask_dim) == self.size(self_dim),
                     error::kIndexError)
            << "expected the shape of the mask at index " << mask_dim
            << " to match the shape of the indexed tensor at index " << self_dim
            << ", got mask shape " << index_tensor.sizes()
            << " and indexed tensor shape " << self.sizes();
      }

      // Compute the number of `True` elements for the output shape, and store
      // the unflattened mask along with `dim_indexed` so that all reshapes,
      // nonzero extraction, and gathering happen inside a single StableHLO
      // `op_builder` lowering (`BuildBooleanMaskIndexShlo`).
      // An M-D mask consumes M dimensions of `self` (a 0D mask consumes 0
      // dimensions and inserts a new dimension at `dim_indexed`).
      TT_ASSIGN_OR_RETURN(info.num_non_zero_elements,
                          GetNumNonZeroElements(index_tensor));
      info.indices.push_back(index_tensor);
      info.dimensions.push_back(dim_indexed);
      dim_indexed += index_tensor.dim();
    } else {
      // TODO(b/567574040): Add support for more than one bool index tensor.
      TT_RET_CHECK(!has_bool_mask, error::kPythonNotImplementedError)
          << "indexing with more than one bool tensor is not yet supported";
      TT_RET_CHECK(dim_indexed < self.dim(), error::kIndexError)
          << "too many indices for tensor of dimension " << self.dim();
      info.indices.push_back(index_tensor);
      info.dimensions.push_back(dim_indexed++);
    }
  }

  TT_RET_CHECK(!info.indices.empty(), error::kInvalidArgument)
      << "expected at least one index tensor to be defined, got 0";

  // Negative-index resolution copies indices to CPU for bounds validation and
  // wrapping. For boolean masks, indices are generated on-device by
  // `BuildNonzeroShlo` and are guaranteed to be non-negative `[0, dim_size)`.
  if (!has_bool_mask) {
    TT_RETURN_IF_ERROR(
        ResolveNegativeIndices(info.indices, self.sizes(), info.dimensions));
  }

  return info;
}

absl::StatusOr<Dimensions> GetOutputDims(const at::Tensor& self,
                                         const IndicesInfo& info) {
  // For boolean mask indexing, the `dims_consumed` (`mask.dim()`) consecutive
  // dimensions of `self` starting at `dim_indexed` are replaced by a single 1D
  // dimension of size `*info.num_non_zero_elements` (for a 0D mask,
  // `dims_consumed == 0`, so a new dimension is inserted at `dim_indexed`),
  // while leading and trailing unindexed dimensions are preserved.
  if (info.num_non_zero_elements.has_value()) {
    const int64_t dim_indexed = info.dimensions[0];
    const int64_t dims_consumed = info.indices[0].dim();
    Dimensions output_dims;
    output_dims.reserve(self.dim() - dims_consumed + 1);
    for (int64_t i = 0; i < dim_indexed; ++i) {
      output_dims.push_back(self.size(i));
    }
    output_dims.push_back(*info.num_non_zero_elements);
    for (int64_t i = dim_indexed + dims_consumed; i < self.dim(); ++i) {
      output_dims.push_back(self.size(i));
    }
    return output_dims;
  }

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
      (self,
       IgnoreInCacheKey(indices_list_opt,
                        "This is being tracked by the dimensions variable"),
       out),
      {
        TT_ASSIGN_OR_THROW(IndicesInfo info,
                           ValidateAndGetIndicesInfo(self, indices_list_opt));

        TT_ASSIGN_OR_THROW(Dimensions output_dims, GetOutputDims(self, info));
        // The indices_list_opt gets ignored in the cache key, but we still
        // want to record which dimensions are indexed and, for boolean masks,
        // the data-dependent output size.
        const auto& dimensions = info.dimensions;
        const int64_t num_non_zero_elements =
            info.num_non_zero_elements.value_or(-1);
        TT_ASSIGN_OR_THROW(
            auto param_keys,
            TT_MAKE_OP_PARAM_CACHE_KEYS(dimensions, num_non_zero_elements));

        ABSL_VLOG(2) << "[AtenIndexTensorOut] self: " << ToString(self);
        for (const auto& t : info.indices) {
          ABSL_VLOG(2) << "[AtenIndexTensorOut] indices_list: " << ToString(t);
        }
        ABSL_VLOG(2) << "[AtenIndexTensorOut] indexed_dims: "
                     << ToString(info.dimensions);
        ABSL_VLOG(2) << "[AtenIndexTensorOut] output_dims: "
                     << ToString(output_dims);

        if (self.dim() == 0 && !info.num_non_zero_elements.has_value()) {
          output_dims = {};
        }

        std::vector<at::Tensor> self_and_indices(std::move(info.indices));
        self_and_indices.insert(self_and_indices.begin(), self);

        TT_ASSIGN_OR_THROW(const auto computation_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        // Dispatch either boolean mask indexing (which fuses `self` dimension
        // collapsing, mask flattening, nonzero index computation, and gather)
        // or standard integer tensor indexing in a single `op_builder` call.
        auto index_op_builder =
            [indexed_dimensions = std::move(info.dimensions),
             num_non_zero_elements = info.num_non_zero_elements](
                absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder) {
              if (num_non_zero_elements.has_value()) {
                return BuildBooleanMaskIndexShlo(inputs[0], inputs[1],
                                                 indexed_dimensions[0],
                                                 *num_non_zero_elements);
              }
              return BuildIndexShlo(inputs, indexed_dimensions);
            };

        TT_THROW_IF_ERROR(DispatchOpOut<kDynamicSize>(
            std::move(index_op_builder), self_and_indices, out,
            {.out_dtype = computation_dtype,
             .out_dims = absl::Span<const int64_t>(output_dims),
             .op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

}  // namespace torch_tpu
