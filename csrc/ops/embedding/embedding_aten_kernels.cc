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

#include "csrc/ops/embedding/embedding_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/empty.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/static_shape_check.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/copy_from/tpu_to_cpu.h"
#include "csrc/ops/embedding/embedding.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/sparse_coo/sparse_coo.h"
#include "csrc/ops/sparse_coo/sparse_coo_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {
absl::Status ValidateWeightType(const at::Tensor& weight) {
  auto scalar_type = weight.scalar_type();
  TT_RET_CHECK(scalar_type == at::kHalf || scalar_type == at::kBFloat16 ||
                   scalar_type == at::kFloat || scalar_type == at::kDouble,
               error::kInvalidArgument)
      << "expected weight dtype to be float16, bfloat16, float32, or float64, "
         "got "
      << ToString(scalar_type);
  return absl::OkStatus();
}
absl::Status ValidateStaticShape(const at::Tensor& tensor,
                                 const std::string_view arg_name) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref, GetBaseBuffer(tensor));
  return ValidateStaticShape(tensor, buffer_ref, arg_name);
}

absl::Status ValidateEmbeddingArgs(const at::Tensor& weight,
                                   const at::Tensor& indices) {
  TT_RET_CHECK(weight.dim() == 2, error::kInvalidArgument)
      << "expected weight to be 2D, got tensor of shape "
      << ToString(weight.sizes());
  TT_RET_CHECK(
      indices.scalar_type() == at::kLong || indices.scalar_type() == at::kInt,
      error::kInvalidArgument)
      << "expected indices to be int64 or int32, got "
      << ToString(indices.scalar_type());
  TT_RETURN_IF_ERROR(ValidateStaticShape(weight, "weight"));
  TT_RETURN_IF_ERROR(ValidateStaticShape(indices, "indices"));
  return absl::OkStatus();
}

Dimensions GetEmbeddingOutputDims(const at::Tensor& weight,
                                  const at::Tensor& indices) {
  if (indices.dim() == 0) {
    return {weight.size(1)};
  }
  Dimensions out_dims = CopyIntVector(indices.sizes());
  out_dims.push_back(weight.size(1));
  return out_dims;
}

absl::StatusOr<DeviceBufferRef> BuildAndDispatchEmbedding(
    const at::Tensor& weight, const at::Tensor& indices,
    const Dimensions& out_dims, OpParamCacheKeys&& pk) {
  TT_ASSIGN_OR_RETURN(const auto result_dtype,
                      ConvertTo<mlir::ElementType>(weight.scalar_type()));

  auto builder_fn = [](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
    return BuildEmbeddingShlo(inputs[0], inputs[1]);
  };

  return DispatchOp<2, 1>(builder_fn, {weight, indices},
                          {.out_dtype = result_dtype,
                           .out_dims = out_dims,
                           .op_param_cache_keys = std::move(pk)});
}

absl::StatusOr<int64_t> ExtractValidCount(const at::Tensor& count_tensor) {
  at::Tensor count_cpu = at::empty(
      {}, at::TensorOptions().dtype(c10::ScalarType::Long).device(at::kCPU));
  TT_RETURN_IF_ERROR(CopyTpuToCpu(count_tensor, count_cpu,
                                  GetTransferSyncMode(/*non_blocking=*/false)));
  return count_cpu.item<int64_t>();
}

at::Tensor MakeEmptySparseEmbeddingGrad(const at::Tensor& grad,
                                        const int64_t num_weights,
                                        const int64_t num_features) {
  TT_ASSIGN_OR_THROW(
      at::Tensor empty_indices,
      MakeEmptyTensor({1, 0}, c10::ScalarType::Long, grad.device()));
  TT_ASSIGN_OR_THROW(
      at::Tensor empty_values,
      MakeEmptyTensor({0, num_features}, grad.scalar_type(), grad.device()));
  return MakeSparseCooTensor(/*sparse_dim=*/1, /*dense_dim=*/1,
                             {num_weights, num_features}, empty_indices,
                             empty_values, /*is_coalesced=*/true);
}

absl::Status ValidateEmbeddingBagSparseBackwardArgs(const at::Tensor& indices,
                                                    bool scale_grad_by_freq,
                                                    int64_t mode,
                                                    bool has_psw) {
  TT_RET_CHECK(!scale_grad_by_freq, error::kInvalidArgument)
      << "scale_grad_by_freq not supported with sparse gradients";
  TT_RET_CHECK(
      indices.scalar_type() == at::kLong || indices.scalar_type() == at::kInt,
      error::kInvalidArgument)
      << "expected indices to be int64 or int32, got "
      << ToString(indices.scalar_type());
  TT_RET_CHECK(mode == static_cast<int64_t>(EmbeddingBagMode::kSum) ||
                   mode == static_cast<int64_t>(EmbeddingBagMode::kMean),
               error::kInvalidArgument)
      << "expected mode to be sum (0) or mean (1) for sparse "
         "embedding_bag_backward, got "
      << mode;
  if (has_psw) {
    TT_RET_CHECK(mode == static_cast<int64_t>(EmbeddingBagMode::kSum),
                 error::kInvalidArgument)
        << "per_sample_weights only supported for mode='sum'";
  }
  return absl::OkStatus();
}

// Matches CUDA, whose max-mode backward kernel
// (embedding_bag_backward_cuda_max) only supports float16, float32, and
// float64.
absl::Status ValidateEmbeddingBagMaxBackwardDtype(const at::Tensor& grad,
                                                  int64_t mode) {
  if (mode != static_cast<int64_t>(EmbeddingBagMode::kMax)) {
    return absl::OkStatus();
  }
  const c10::ScalarType scalar_type = grad.scalar_type();
  TT_RET_CHECK(scalar_type == at::kHalf || scalar_type == at::kFloat ||
                   scalar_type == at::kDouble,
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(scalar_type)
      << " when mode is 'max'";
  return absl::OkStatus();
}

at::Tensor DispatchEmbeddingSparseBackward(const at::Tensor& grad,
                                           const at::Tensor& indices,
                                           const int64_t num_weights,
                                           const int64_t padding_idx,
                                           const bool scale_grad_by_freq,
                                           OpParamCacheKeys&& pk) {
  const int64_t num_features = grad.size(-1);
  const int64_t total_indices = indices.numel();
  if (total_indices == 0 || grad.numel() == 0) {
    return MakeEmptySparseEmbeddingGrad(grad, num_weights, num_features);
  }

  TT_ASSIGN_OR_THROW(const auto val_dtype,
                     ConvertTo<mlir::ElementType>(grad.scalar_type()));
  TT_ASSIGN_OR_THROW(const auto i64_dtype,
                     ConvertTo<mlir::ElementType>(at::kLong));
  const Dimensions ind_dims = {1, total_indices};
  const Dimensions val_dims = {total_indices, num_features};

  if (padding_idx < 0) {
    const std::array<mlir::ElementType, 2> out_dtypes = {i64_dtype, val_dtype};
    const std::array<absl::Span<const int64_t>, 2> out_dims_list = {ind_dims,
                                                                    val_dims};
    const auto builder_fn = [](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
      return BuildEmbeddingSparseBackwardNoPadShlo(inputs[0], inputs[1]);
    };
    TT_ASSIGN_OR_THROW(
        auto results,
        (DispatchOp<2, 2>(builder_fn, {grad, indices},
                          {.out_dtypes = out_dtypes,
                           .out_dims_list = out_dims_list,
                           .op_param_cache_keys = std::move(pk)})));
    return MakeSparseCooTensor(
        /*sparse_dim=*/1, /*dense_dim=*/1, {num_weights, num_features},
        MakeTensor(results[0]), MakeTensor(results[1]),
        /*is_coalesced=*/false);
  }

  const Dimensions count_dims = {};
  const std::array<mlir::ElementType, 3> out_dtypes = {i64_dtype, val_dtype,
                                                       i64_dtype};
  const std::array<absl::Span<const int64_t>, 3> out_dims_list = {
      ind_dims, val_dims, count_dims};
  const auto pad_builder_fn =
      [padding_idx](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
        return BuildEmbeddingSparseBackwardPadShlo(inputs[0], inputs[1],
                                                   padding_idx);
      };
  TT_ASSIGN_OR_THROW(
      auto pad_results,
      (DispatchOp<2, 3>(pad_builder_fn, {grad, indices},
                        {.out_dtypes = out_dtypes,
                         .out_dims_list = out_dims_list,
                         .op_param_cache_keys = std::move(pk)})));
  const at::Tensor compacted_indices = MakeTensor(pad_results[0]);
  const at::Tensor compacted_values = MakeTensor(pad_results[1]);
  const at::Tensor count_tensor = MakeTensor(pad_results[2]);
  TT_ASSIGN_OR_THROW(const int64_t valid_count,
                     ExtractValidCount(count_tensor));
  if (valid_count == 0) {
    return MakeEmptySparseEmbeddingGrad(grad, num_weights, num_features);
  }
  if (valid_count == total_indices) {
    return MakeSparseCooTensor(/*sparse_dim=*/1, /*dense_dim=*/1,
                               {num_weights, num_features}, compacted_indices,
                               compacted_values, /*is_coalesced=*/false);
  }

  const Dimensions sliced_ind_dims = {1, valid_count};
  const Dimensions sliced_val_dims = {valid_count, num_features};
  const std::array<mlir::ElementType, 2> slice_dtypes = {i64_dtype, val_dtype};
  const std::array<absl::Span<const int64_t>, 2> slice_dims_list = {
      sliced_ind_dims, sliced_val_dims};
  TT_ASSIGN_OR_THROW(auto slice_pk, TT_MAKE_OP_PARAM_CACHE_KEYS(valid_count));
  const auto slice_builder_fn =
      [valid_count](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
        return BuildSliceCoalescedShlo(inputs[0], inputs[1], valid_count);
      };
  TT_ASSIGN_OR_THROW(
      auto slice_results,
      (DispatchOp<2, 2>(slice_builder_fn, {compacted_indices, compacted_values},
                        {.out_dtypes = slice_dtypes,
                         .out_dims_list = slice_dims_list,
                         .op_param_cache_keys = std::move(slice_pk)})));
  return MakeSparseCooTensor(
      /*sparse_dim=*/1, /*dense_dim=*/1, {num_weights, num_features},
      MakeTensor(slice_results[0]), MakeTensor(slice_results[1]),
      /*is_coalesced=*/false);
}

}  // namespace

// Note: `padding_idx`, `scale_grad_by_freq`, and `sparse` are wrapped in
// `IgnoreInCacheKey` because they are backward/autograd directives that do not
// affect forward pass execution or the generated StableHLO graph:
//   - `padding_idx`: In the forward pass, PyTorch looks up the row at
//     `padding_idx` normally. It only affects backward by zeroing out the
//     gradient for that entry in `embedding_dense_backward`.
//   - `scale_grad_by_freq`: Gradients do not exist in the forward pass; this
//     flag is only used during backward to scale gradients by inverse
//     frequency.
//   - `sparse`: Directs autograd to generate a sparse COO gradient tensor
//     during backward; the forward pass always produces a dense tensor.
// Excluding these arguments from the cache key prevents redundant graph
// recompilations when the same embedding table is queried with different
// backward settings.
at::Tensor AtenEmbedding(const at::Tensor& weight, const at::Tensor& indices,
                         at::SymInt padding_idx, bool scale_grad_by_freq,
                         bool sparse) {
  TT_KERNEL(
      OpName::kEmbedding, pk,
      (weight, indices, IgnoreInCacheKey(padding_idx, "Unused in forward"),
       IgnoreInCacheKey(scale_grad_by_freq, "Unused in forward"),
       IgnoreInCacheKey(sparse, "Unused in forward")),
      {
        TT_THROW_IF_ERROR(ValidateEmbeddingArgs(weight, indices));
        Dimensions out_dims = GetEmbeddingOutputDims(weight, indices);
        TT_ASSIGN_OR_THROW(auto result,
                           BuildAndDispatchEmbedding(weight, indices, out_dims,
                                                     std::move(pk)));
        return MakeTensor(result);
      });
}

at::Tensor& AtenEmbeddingOut(const at::Tensor& weight,
                             const at::Tensor& indices, at::SymInt padding_idx,
                             bool scale_grad_by_freq, bool sparse,
                             at::Tensor& out) {
  TT_KERNEL(
      OpName::kEmbeddingOut, pk,
      (weight, indices, IgnoreInCacheKey(padding_idx, "Unused in forward"),
       IgnoreInCacheKey(scale_grad_by_freq, "Unused in forward"),
       IgnoreInCacheKey(sparse, "Unused in forward"), out),
      {
        TT_THROW_IF_ERROR(ValidateEmbeddingArgs(weight, indices));
        TT_CHECK_THROW(out.scalar_type() == weight.scalar_type(),
                       error::kInvalidArgument)
            << "expected out tensor to have dtype "
            << ToString(weight.scalar_type()) << ", got "
            << ToString(out.scalar_type());
        Dimensions out_dims = GetEmbeddingOutputDims(weight, indices);
        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, out_dims));
        TT_ASSIGN_OR_THROW(auto result,
                           BuildAndDispatchEmbedding(weight, indices, out_dims,
                                                     std::move(pk)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
        return out;
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> AtenEmbeddingBag(
    const at::Tensor& weight, const at::Tensor& indices,
    const at::Tensor& offsets, bool scale_grad_by_freq, int64_t mode,
    bool sparse, const std::optional<at::Tensor>& per_sample_weights,
    bool include_last_offset, int64_t padding_idx) {
  TT_KERNEL(
      OpName::kEmbeddingBag, pk,
      (weight, indices, offsets, scale_grad_by_freq, mode, sparse,
       per_sample_weights, include_last_offset, padding_idx),
      {
        TT_THROW_IF_ERROR(ValidateWeightType(weight));
        TT_THROW_IF_ERROR(ValidateStaticShape(weight, "weight"));
        TT_THROW_IF_ERROR(ValidateStaticShape(indices, "indices"));
        TT_THROW_IF_ERROR(ValidateStaticShape(offsets, "offsets"));

        TT_ASSIGN_OR_THROW(auto weight_dtype,
                           ConvertTo<mlir::ElementType>(weight.scalar_type()));
        TT_ASSIGN_OR_THROW(auto i64_dtype,
                           ConvertTo<mlir::ElementType>(at::kLong));

        int64_t batch_size = offsets.numel();
        if (include_last_offset) {
          TT_CHECK_THROW(batch_size >= 1, error::kInvalidArgument)
              << "expected offsets to have at least 1 element when "
                 "include_last_offset is true, got "
              << batch_size;
          batch_size -= 1;
        }
        int64_t emb_dim = weight.size(1);

        bool has_psw =
            per_sample_weights.has_value() && per_sample_weights->defined();

        auto builder_fn = [scale_grad_by_freq, mode, sparse,
                           include_last_offset, padding_idx,
                           has_psw](absl::Span<mlir::MlirOp> inputs,
                                    mlir::MlirBuilder& builder) {
          std::optional<mlir::MlirOp> psw;
          if (has_psw) psw = inputs[3];
          return BuildEmbeddingBagShlo(inputs[0], inputs[1], inputs[2],
                                       scale_grad_by_freq, mode, sparse, psw,
                                       include_last_offset, padding_idx);
        };

        std::vector<at::Tensor> inputs = {weight, indices, offsets};
        if (has_psw) inputs.push_back(*per_sample_weights);

        std::array<mlir::ElementType, 4> out_dtypes = {weight_dtype, i64_dtype,
                                                       i64_dtype, i64_dtype};
        Dimensions d0 = {batch_size, emb_dim};
        Dimensions d1 = {indices.numel()};
        Dimensions d2 = {batch_size};
        Dimensions d3 = (mode == static_cast<int64_t>(EmbeddingBagMode::kMax))
                            ? Dimensions{batch_size, emb_dim}
                            : Dimensions{batch_size};
        std::array<absl::Span<const int64_t>, 4> out_dims_list = {d0, d1, d2,
                                                                  d3};

        TT_ASSIGN_OR_THROW(auto results,
                           (DispatchOp<kDynamicSize, 4>(
                               builder_fn, inputs,
                               {.out_dtypes = out_dtypes,
                                .out_dims_list = out_dims_list,
                                .op_param_cache_keys = std::move(pk)})));

        return std::make_tuple(MakeTensor(results[0]), MakeTensor(results[1]),
                               MakeTensor(results[2]), MakeTensor(results[3]));
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenEmbeddingBagForwardOnly(const at::Tensor& weight, const at::Tensor& indices,
                            const at::Tensor& offsets, bool scale_grad_by_freq,
                            int64_t mode, bool sparse,
                            const std::optional<at::Tensor>& per_sample_weights,
                            bool include_last_offset, int64_t padding_idx) {
  TT_KERNEL(
      OpName::kEmbeddingBagForwardOnly, _,
      (weight, indices, offsets,
       IgnoreInCacheKey(scale_grad_by_freq, "Delegates to AtenEmbeddingBag"),
       IgnoreInCacheKey(mode, "Delegates to AtenEmbeddingBag"),
       IgnoreInCacheKey(sparse, "Delegates to AtenEmbeddingBag"),
       IgnoreInCacheKey(per_sample_weights, "Delegates to AtenEmbeddingBag"),
       IgnoreInCacheKey(include_last_offset, "Delegates to AtenEmbeddingBag"),
       IgnoreInCacheKey(padding_idx, "Delegates to AtenEmbeddingBag")),
      {
        return AtenEmbeddingBag(weight, indices, offsets, scale_grad_by_freq,
                                mode, sparse, per_sample_weights,
                                include_last_offset, padding_idx);
      });
}

at::Tensor AtenEmbeddingBagBackward(
    const at::Tensor& grad, const at::Tensor& indices,
    const at::Tensor& offsets, const at::Tensor& offset2bag,
    const at::Tensor& bag_size, const at::Tensor& max_indices,
    at::SymInt num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse,
    const std::optional<at::Tensor>& per_sample_weights, int64_t padding_idx) {
  TT_KERNEL(
      OpName::kEmbeddingBagBackward, pk,
      (grad, indices, offsets, offset2bag, bag_size, max_indices, num_weights,
       scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx),
      {
        // CUDA only raises the max-mode dtype error in its dense backward, so
        // the sparse backward rejects max mode altogether
        if (!sparse) {
          TT_THROW_IF_ERROR(ValidateEmbeddingBagMaxBackwardDtype(grad, mode));
        }

        bool has_psw =
            per_sample_weights.has_value() && per_sample_weights->defined();
        int64_t nw_val = num_weights.expect_int();

        if (sparse) {
          TT_THROW_IF_ERROR(ValidateEmbeddingBagSparseBackwardArgs(
              indices, scale_grad_by_freq, mode, has_psw));
          if (indices.numel() == 0 || grad.numel() == 0) {
            return MakeEmptySparseEmbeddingGrad(grad, nw_val, grad.size(-1));
          }
          auto index_grad_builder_fn = [mode, padding_idx, has_psw](
                                           absl::Span<mlir::MlirOp> inputs,
                                           mlir::MlirBuilder& builder) {
            std::optional<mlir::MlirOp> psw;
            if (has_psw) psw = inputs[5];
            return BuildEmbeddingBagSparseBackwardIndexGradShlo(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], mode,
                psw, padding_idx);
          };
          std::vector<at::Tensor> inputs = {grad, indices, offsets, offset2bag,
                                            bag_size};
          if (has_psw) inputs.push_back(*per_sample_weights);
          TT_ASSIGN_OR_THROW(auto result_dtype,
                             ConvertTo<mlir::ElementType>(grad.scalar_type()));
          TT_ASSIGN_OR_THROW(auto index_grad_buf,
                             (DispatchOp<kDynamicSize, 1>(
                                 index_grad_builder_fn, inputs,
                                 {.out_dtype = result_dtype,
                                  .out_dims = {indices.numel(), grad.size(-1)},
                                  .op_param_cache_keys = std::move(pk)})));
          at::Tensor index_grad = MakeTensor(index_grad_buf);
          TT_ASSIGN_OR_THROW(auto sparse_pk,
                             TT_MAKE_OP_PARAM_CACHE_KEYS(nw_val, padding_idx,
                                                         scale_grad_by_freq));
          return DispatchEmbeddingSparseBackward(
              index_grad, indices, nw_val, padding_idx, scale_grad_by_freq,
              std::move(sparse_pk));
        }

        auto builder_fn =
            [nw_val, scale_grad_by_freq, mode, sparse, padding_idx, has_psw](
                absl::Span<mlir::MlirOp> inputs, mlir::MlirBuilder& builder) {
              std::optional<mlir::MlirOp> psw;
              if (has_psw) psw = inputs[6];
              return BuildEmbeddingBagBackwardShlo(
                  inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                  inputs[5], at::SymInt(nw_val), scale_grad_by_freq, mode,
                  sparse, psw, padding_idx);
            };

        std::vector<at::Tensor> inputs = {grad,       indices,  offsets,
                                          offset2bag, bag_size, max_indices};
        if (has_psw) inputs.push_back(*per_sample_weights);

        TT_ASSIGN_OR_THROW(auto result_dtype,
                           ConvertTo<mlir::ElementType>(grad.scalar_type()));

        TT_ASSIGN_OR_THROW(auto results,
                           (DispatchOp<kDynamicSize, 1>(
                               builder_fn, inputs,
                               {.out_dtype = result_dtype,
                                .out_dims = {nw_val, grad.size(-1)},
                                .op_param_cache_keys = std::move(pk)})));

        return MakeTensor(results);
      });
}

at::Tensor AtenEmbeddingDenseBackward(const at::Tensor& grad_output,
                                      const at::Tensor& indices,
                                      at::SymInt num_weights,
                                      at::SymInt padding_idx,
                                      bool scale_grad_by_freq) {
  TT_KERNEL(
      OpName::kEmbeddingDenseBackward, pk,
      (grad_output, indices, num_weights, padding_idx, scale_grad_by_freq), {
        int64_t nw_val = num_weights.expect_int();
        int64_t pi_val = padding_idx.expect_int();

        auto builder_fn = [nw_val, pi_val, scale_grad_by_freq](
                              FixedSizeSpan<mlir::MlirOp, 2> inputs) {
          return BuildEmbeddingDenseBackwardShlo(
              inputs[0], inputs[1], at::SymInt(nw_val), at::SymInt(pi_val),
              scale_grad_by_freq);
        };

        TT_ASSIGN_OR_THROW(auto result_dtype, ConvertTo<mlir::ElementType>(
                                                  grad_output.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto result,
            (DispatchOp<2, 1>(builder_fn, {grad_output, indices},
                              {.out_dtype = result_dtype,
                               .out_dims = {nw_val, grad_output.size(-1)},
                               .op_param_cache_keys = std::move(pk)})));

        return MakeTensor(result);
      });
}

at::Tensor& AtenEmbeddingRenorm_(at::Tensor& self, const at::Tensor& indices,
                                 double max_norm, double norm_type) {
  TT_KERNEL(
      OpName::kEmbeddingRenorm_, pk, (self, indices, max_norm, norm_type), {
        TT_CHECK_THROW(IsFloatingPoint(self) || IsComplex(self),
                       error::kInvalidArgument)
            << "expected floating point or complex, got "
            << ToString(self.scalar_type());

        Dimensions out_dims = CopyIntVector(indices.sizes());
        out_dims.push_back(self.size(1));

        TT_ASSIGN_OR_THROW(auto elem_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto builder_fn = [max_norm,
                           norm_type](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
          return BuildEmbeddingRenormShlo(inputs[0], inputs[1], max_norm,
                                          norm_type);
        };

        TT_ASSIGN_OR_THROW(
            auto renorm_rows,
            (DispatchOp<2, 1>(builder_fn, {self, indices},
                              {.out_dtype = elem_type,
                               .out_dims = out_dims,
                               .op_param_cache_keys = std::move(pk)})));

        self.index_put_({indices}, MakeTensor(renorm_rows));
        return self;
      });
}

at::Tensor AtenEmbeddingSparseBackward(const at::Tensor& grad,
                                       const at::Tensor& indices,
                                       const int64_t num_weights,
                                       const int64_t padding_idx,
                                       const bool scale_grad_by_freq) {
  TT_KERNEL(OpName::kEmbeddingSparseBackward, pk,
            (grad, indices, num_weights, padding_idx, scale_grad_by_freq), {
              TT_CHECK_THROW(!scale_grad_by_freq, error::kInvalidArgument)
                  << "scale_grad_by_freq not supported with sparse gradients";
              TT_CHECK_THROW(indices.scalar_type() == at::kLong ||
                                 indices.scalar_type() == at::kInt,
                             error::kInvalidArgument)
                  << "expected indices to be int64 or int32, got "
                  << ToString(indices.scalar_type());
              return DispatchEmbeddingSparseBackward(
                  grad, indices, num_weights, padding_idx, scale_grad_by_freq,
                  std::move(pk));
            });
}

at::Tensor AtenEmbeddingBackward(const at::Tensor& grad,
                                 const at::Tensor& indices,
                                 const at::SymInt num_weights,
                                 const at::SymInt padding_idx,
                                 const bool scale_grad_by_freq,
                                 const bool sparse) {
  if (sparse) {
    return AtenEmbeddingSparseBackward(grad, indices, num_weights.expect_int(),
                                       padding_idx.expect_int(),
                                       scale_grad_by_freq);
  }
  return AtenEmbeddingDenseBackward(grad, indices, num_weights, padding_idx,
                                    scale_grad_by_freq);
}

}  // namespace torch_tpu
