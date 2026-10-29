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
#include "ATen/ops/count_nonzero.h"
#include "absl/algorithm/container.h"
#include "absl/functional/bind_front.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/Optional.h"
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
  // The defined index tensors, on the device of `self`, with the dimensions of
  // `self` they index. They might not index consecutive dimensions.
  std::vector<IndexEntry> entries;

  // Number of non-zero (`True`) elements when indexing with a boolean mask.
  std::optional<int64_t> num_non_zero_elements;
};

// Computes the number of non-zero (`True`) elements in `mask`. This count is
// required ahead of time to determine the data-dependent output shape of
// boolean mask indexing.
//
// A CPU `mask` is counted on the host with `at::count_nonzero`, which only runs
// the CPU kernel and never dispatches a TPU op. A TPU `mask` is counted on TPU
// via `BuildNonzeroGetOutputSizeShlo` and the scalar count is materialized on
// CPU.
absl::StatusOr<int64_t> GetNumNonZeroElements(const at::Tensor& mask) {
  if (mask.numel() == 0) {
    return 0;
  }
  if (mask.is_cpu()) {
    return at::count_nonzero(mask).item<int64_t>();
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

// Converts the parsed `entries` of `self` (see `ParseIndexEntries`) into
// `IndicesInfo`, with all index tensors moved to the device of `self`. Only
// checks the `index`-specific rule that a boolean mask is the only defined
// index.
absl::StatusOr<IndicesInfo> GetIndicesInfo(const at::Tensor& self,
                                           std::vector<IndexEntry> entries) {
  const bool has_bool_mask = absl::c_any_of(
      entries, [](const IndexEntry& entry) { return entry.is_mask; });
  // TODO(b/567574040): Add support for more than one bool index tensor.
  TT_RET_CHECK(!has_bool_mask || entries.size() == 1,
               error::kPythonNotImplementedError)
      << "indexing with more than one bool tensor is not yet supported";

  IndicesInfo info;
  for (IndexEntry& entry : entries) {
    if (entry.is_mask) {
      // Compute the number of `True` elements for the output shape, and store
      // the unflattened mask along with its first indexed dimension so that all
      // reshapes, nonzero extraction, and gathering happen inside a single
      // StableHLO `op_builder` lowering (`BuildBooleanMaskIndexShlo`).
      // An M-D mask consumes M dimensions of `self` (a 0D mask consumes 0
      // dimensions and inserts a new dimension at its first indexed dimension).
      // The count is computed before moving a CPU mask to TPU so that it can
      // be done on the host.
      TT_ASSIGN_OR_RETURN(info.num_non_zero_elements,
                          GetNumNonZeroElements(entry.tensor));
    }
    TT_ASSIGN_OR_RETURN(entry.tensor,
                        MoveIndexToDevice(entry.tensor, self.device()));
  }
  info.entries = std::move(entries);

  // Negative integer indices are wrapped on TPU by adding the size of the
  // indexed dimension (no bounds checking is done; see
  // `ResolveNegativeIndices`). This is skipped for boolean masks, whose
  // positions are generated on-device by `BuildNonzeroShlo` and are guaranteed
  // to be non-negative `[0, dim_size)`.
  if (!has_bool_mask) {
    TT_RETURN_IF_ERROR(
        ResolveNegativeIndices(absl::MakeSpan(info.entries), self.sizes()));
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
    const int64_t dim_indexed = info.entries[0].first_dim;
    const int64_t dims_consumed = info.entries[0].tensor.dim();
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

  const std::vector<IndexEntry>& entries = info.entries;
  for (int i = 0, j = 0; i < self.dim(); ++i) {
    if (j < entries.size() && i == entries[j].first_dim) {
      if (j > 0 && indexed_dimensions_consecutive &&
          entries[j - 1].first_dim != i - 1) {
        indexed_dimensions_consecutive = false;
      }

      TT_ASSIGN_OR_RETURN(
          broadcasted_index_shape,
          InferSize(broadcasted_index_shape, entries[j].tensor.sizes()));

      j++;
    } else {
      unindexed_dims_shape.push_back(self.size(i));
    }
  }

  Dimensions output_dims;
  output_dims.reserve(broadcasted_index_shape.size() +
                      unindexed_dims_shape.size());

  size_t insertion_index =
      indexed_dimensions_consecutive ? entries[0].first_dim : 0;

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
        TT_ASSIGN_OR_THROW(std::vector<IndexEntry> entries,
                           ParseIndexEntries(self, indices_list_opt));
        TT_ASSIGN_OR_THROW(IndicesInfo info,
                           GetIndicesInfo(self, std::move(entries)));

        TT_ASSIGN_OR_THROW(Dimensions output_dims, GetOutputDims(self, info));
        // The indices_list_opt gets ignored in the cache key, but we still
        // want to record which dimensions are indexed and, for boolean masks,
        // the data-dependent output size.
        Indices dimensions = GetFirstDims(info.entries);
        const int64_t num_non_zero_elements =
            info.num_non_zero_elements.value_or(-1);
        TT_ASSIGN_OR_THROW(
            auto param_keys,
            TT_MAKE_OP_PARAM_CACHE_KEYS(dimensions, num_non_zero_elements));

        ABSL_VLOG(2) << "[AtenIndexTensorOut] self: " << ToString(self);
        for (const IndexEntry& entry : info.entries) {
          ABSL_VLOG(2) << "[AtenIndexTensorOut] indices_list: "
                       << ToString(entry.tensor);
        }
        ABSL_VLOG(2) << "[AtenIndexTensorOut] indexed_dims: "
                     << ToString(dimensions);
        ABSL_VLOG(2) << "[AtenIndexTensorOut] output_dims: "
                     << ToString(output_dims);

        if (self.dim() == 0 && !info.num_non_zero_elements.has_value()) {
          output_dims = {};
        }

        std::vector<at::Tensor> self_and_indices;
        self_and_indices.reserve(info.entries.size() + 1);
        self_and_indices.push_back(self);
        for (IndexEntry& entry : info.entries) {
          self_and_indices.push_back(std::move(entry.tensor));
        }

        TT_ASSIGN_OR_THROW(const auto computation_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        // Dispatch either boolean mask indexing (which fuses `self` dimension
        // collapsing, mask flattening, nonzero index computation, and gather)
        // or standard integer tensor indexing in a single `op_builder` call.
        auto index_op_builder =
            [indexed_dimensions = std::move(dimensions),
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
