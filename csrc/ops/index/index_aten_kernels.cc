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
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "c10/util/Optional.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/index_utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/nonzero/nonzero.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

namespace torch_tpu {

namespace {

struct DynamicUnindexedDim {
  mlir::stablehlo::DimensionInfo dim_info;
  int64_t out_dim;
};

bool AreConsecutive(const Indices& indexed_dims) {
  for (size_t i = 1; i < indexed_dims.size(); ++i) {
    if (indexed_dims[i] != indexed_dims[i - 1] + 1) {
      return false;
    }
  }
  return true;
}

namespace stablehlo = mlir::stablehlo;

// Builds StableHLO for tensor indexing with integer index tensors.
// `input_ops[0]` is `self`, and `input_ops[1..]` are the index tensors
// corresponding to `indexed_dims`.
absl::StatusOr<mlir::MlirOp> BuildIndexShlo(
    mlir::ArrayRef<mlir::MlirOp> input_ops, Indices indexed_dims) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=AtenIndexTensorOut (caller) makes sure
                 // `input_ops` has the `self` input and, at least, one index.
      input_ops.size() >= 2, error::kInvalidArgument)
      << "[BuildIndexShlo]: requires at least two input ops: an operand and an "
         "indexing tensor";
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=AtenIndexTensorOut (caller) creates
                 // `indexed_dims`, making sure there's always a corresponding
                 // index.
      input_ops.size() == indexed_dims.size() + 1, error::kInvalidArgument)
      << "[BuildIndexShlo]: requires exactly one indexing tensor per indexed "
         "dimension, plus one for the operand";

  mlir::MlirOp self = input_ops[0];
  mlir::ArrayRef<mlir::MlirOp> indices = input_ops.drop_front();
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);

  if (self_type.getRank() == 0) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=AtenIndexTensorOut (caller) errors
                   // before calling this function, when rank == 0.
        indexed_dims.size() == 1 && indexed_dims[0] == 0,
        error::kInvalidArgument)
        << "[BuildIndexShlo]: if the operand is a scalar, the only indexed "
           "dimension must be 0";
    return self;
  }

  // Indexing tensors need to be broadcastable together.
  TT_ASSIGN_OR_RETURN(std::vector<mlir::MlirOp> broadcasted_indices,
                      ApplyBroadcastIfNeeded(indices));

  const int64_t broadcasted_rank =
      GetTensorTypeOrDie(broadcasted_indices[0]).getRank();

  std::vector<mlir::MlirOp> reshaped_indices;
  reshaped_indices.reserve(broadcasted_indices.size());
  for (const mlir::MlirOp& broadcasted_index : broadcasted_indices) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp reshaped_index,
                        Unsqueeze(broadcasted_index, broadcasted_rank));
    reshaped_indices.push_back(reshaped_index);
  }
  mlir::MlirOp index = mlir::stablehlo::Concatenate(
      self.getBuilder(), reshaped_indices, broadcasted_rank);

  // Dynamism in Gather:
  // - `gather_slice_sizes` is a static attribute in StableHLO and must contain
  //   compile-time non-negative integers. Therefore, we use `self_dims[i].size`
  //   (the upper bound / physical dimension size) rather than
  //   `self_type.getShape()[i]` which evaluates to `kDynamic` (-1) for dynamic
  //   dimensions.
  // - For indexed dimensions, the slice size is 1. If `index` has dynamic
  //   dimensions, StableHLO Gather automatically preserves and propagates those
  //   dynamic batch dimensions from `index` (via `start_indices`) into the
  //   result type.
  // - For unindexed dimensions, the slice size extracts the full upper-bound
  //   extent from `self`. If an unindexed dimension is dynamic, we re-attach
  //   its dynamic runtime bound onto the Gather output via `SetDimensionSize`.
  mlir::stablehlo::Dimensions self_dims = GetDimensions(self);
  Dimensions gather_slice_sizes;
  gather_slice_sizes.reserve(self_dims.size());
  Indices offset_dims;

  bool index_dims_consecutive = AreConsecutive(indexed_dims);

  std::vector<DynamicUnindexedDim> dynamic_unindexed_dims;

  for (size_t i = 0, j = 0; i < self_dims.size(); ++i) {
    if (j < indexed_dims.size() && indexed_dims[j] == i) {
      gather_slice_sizes.push_back(1);
      ++j;
    } else {
      gather_slice_sizes.push_back(self_dims[i].size);
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=AtenIndexTensorOut (caller) creates
                     // `indexed_dims`, making sure they are always increasing,
                     // non-negative, and distinct.
          j >= indexed_dims.size() || i < indexed_dims[j],
          error::kInvalidArgument)
          << "[BuildIndexShlo]: indexed dimensions must be >= 0, increasing, "
             "and distinct";
      // If the indexed dimensions are consecutive, the indexed dimensions will
      // be inserted at the position of the first indexed dimension. Otherwise,
      // they will be inserted at the beginning of the output tensor shape.
      // offset_dims contains the positions of the non-indexed dimensions in the
      // output tensor shape.
      // See
      // https://numpy.org/devdocs/user/basics.indexing.html#combining-advanced-and-basic-indexing
      int64_t out_dim =
          index_dims_consecutive && j == 0 ? i : broadcasted_rank + i - j;
      offset_dims.push_back(out_dim);
      if (self_dims[i].boundOp.has_value()) {
        dynamic_unindexed_dims.push_back({self_dims[i], out_dim});
      }
    }
  }

  ABSL_VLOG(2) << "[BuildIndexShlo]: gather_slice_sizes = "
               << absl::StrJoin(gather_slice_sizes, ",");

  stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers =
      stablehlo::GatherDimensionNumbersAttr::get(
          &self.getContext(),
          /*offset_dims=*/offset_dims,
          /*collapsed_slice_dims=*/indexed_dims,
          /*operand_batching_dims=*/{},
          /*start_indices_batching_dims=*/{},
          /*start_index_map=*/indexed_dims,
          /*index_vector_dim=*/broadcasted_rank);
  ABSL_VLOG(2) << "[BuildIndexShlo]: GatherDimensionNumbers = "
               << mlir::debugString(gather_dimension_numbers);

  auto result = stablehlo::Gather(self, index, gather_dimension_numbers,
                                  gather_slice_sizes,
                                  /*indices_are_sorted=*/false);
  for (const auto& [dim_info, out_dim] : dynamic_unindexed_dims) {
    mlir::MlirOp bound_op_mlir(self.getBuilder(), *dim_info.boundOp);
    mlir::MlirOp dim_size =
        mlir::stablehlo::GetDimensionSize(bound_op_mlir, dim_info.boundOpDim);
    result = mlir::stablehlo::SetDimensionSize(result, dim_size, out_dim);
  }
  return stablehlo::ConvertElementType(result, self_type.getElementType());
}

// Builds StableHLO for tensor indexing with a single boolean mask at
// `dim_indexed`, given the precomputed `num_non_zero_elements` (`True` count).
// Performs dimension collapsing on `self`, mask flattening, nonzero index
// computation, and gathering within a single StableHLO lowering.
absl::StatusOr<mlir::MlirOp> BuildBooleanMaskIndexShlo(
    mlir::MlirOp self, mlir::MlirOp mask, int64_t dim_indexed,
    int64_t num_non_zero_elements) {
  // In PyTorch, an M-dimensional boolean mask with K `True` entries replaces
  // the M indexed dimensions `[dim_indexed, dim_indexed + M - 1]` of `self`
  // with a single 1D dimension of size K in row-major order.
  //
  // Performing all reshapes, nonzero index extraction, and gathering inside
  // this single StableHLO lowering avoids multiple ATen tensor-level dispatches
  // (e.g. `.flatten()` and `AtenNonzero`) outside `op_builder`, keeping the
  // computation fused in one graph for XLA optimization.
  const int64_t mask_rank = GetTensorTypeOrDie(mask).getRank();

  // Step 1: Adjust `self` and `mask` so that the masked dimensions of `self`
  // correspond to a single dimension at `dim_indexed` and `mask` is 1D:
  // - When `mask_rank == 0` (0D scalar boolean mask), no dimensions of `self`
  //   are consumed; instead, a new dimension of size 1 is inserted at
  //   `dim_indexed` via `Unsqueeze`, and `mask` is flattened to shape `(1,)`.
  // - When `mask_rank > 1` (multi-dimensional boolean mask), the M masked
  //   dimensions `[dim_indexed, dim_indexed + M - 1]` of `self` are collapsed
  //   into a single dimension at `dim_indexed`, and `mask` is flattened to 1D.
  if (mask_rank == 0) {
    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=dim_indexed is guaranteed to
                          // be in [0, self.dim()].
        self, Unsqueeze(self, dim_indexed));
    mask = Flatten(mask);
  } else if (mask_rank > 1) {
    mlir::stablehlo::Dimensions self_dims = GetDimensions(self);
    Dimensions static_shape_before;
    static_shape_before.reserve(self_dims.size());
    for (const auto& dim : self_dims) {
      static_shape_before.push_back(dim.size);
    }

    Dimensions static_shape_after;
    static_shape_after.reserve(self_dims.size() - mask_rank + 1);
    for (int64_t i = 0; i < dim_indexed; ++i) {
      static_shape_after.push_back(static_shape_before[i]);
    }
    int64_t collapsed_dim_size = 1;
    for (int64_t i = dim_indexed; i < dim_indexed + mask_rank; ++i) {
      collapsed_dim_size *= static_shape_before[i];
    }
    static_shape_after.push_back(collapsed_dim_size);
    for (int64_t i = dim_indexed + mask_rank; i < static_shape_before.size();
         ++i) {
      static_shape_after.push_back(static_shape_before[i]);
    }

    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=collapsing contiguous valid
                          // dimensions always succeeds.
        self, ReshapeFromStaticDimensions(self, static_shape_before,
                                          static_shape_after));
    mask = Flatten(mask);
  }

  // If the indexed dimension of `self` has size 0 (i.e. `mask` has 0 elements
  // and `num_non_zero_elements == 0`), `self` after Step 1 already has the
  // exact output shape, whereas `stablehlo.gather` requires `slice_sizes[d] ==
  // 1 <= operand.dim(d)` on collapsed slice dimensions.
  if (GetTensorTypeOrDie(self).getDimSize(dim_indexed) == 0) {
    return self;
  }

  // Step 2: Compute the 1D linear indices of the `True` elements in `mask`.
  // `BuildNonzeroShlo` on a 1D mask returns shape `(num_non_zero_elements, 1)`,
  // which we flatten to `(num_non_zero_elements,)`.
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=1D mask nonzero always succeeds.
      mlir::MlirOp nonzero_indices,
      BuildNonzeroShlo(num_non_zero_elements, mask));
  mlir::MlirOp flat_indices = Flatten(nonzero_indices);

  // Step 3: Gather from `self` along `dim_indexed` using the 1D flat indices.
  return BuildIndexShlo({self, flat_indices}, {dim_indexed});
}

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
    const int64_t dim_indexed = info.entries[0].indexed_dim;
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
    if (j < entries.size() && i == entries[j].indexed_dim) {
      if (j > 0 && indexed_dimensions_consecutive &&
          entries[j - 1].indexed_dim != i - 1) {
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
      indexed_dimensions_consecutive ? entries[0].indexed_dim : 0;

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
        Indices dimensions = GetIndexedDims(info.entries);
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
