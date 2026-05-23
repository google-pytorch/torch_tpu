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

#include "torch_tpu/ops/embedding/embedding_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/static_shape_check.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/embedding/embedding.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {
absl::Status CheckWeightType(const at::Tensor& weight) {
  auto scalar_type = weight.scalar_type();
  TT_RET_CHECK(scalar_type == at::kHalf || scalar_type == at::kBFloat16 ||
                   scalar_type == at::kFloat || scalar_type == at::kDouble,
               error::kInvalidArgument)
      << "expected weight dtype to be float16, bfloat16, float32, or float64, "
         "got "
      << ToString(scalar_type);
  return absl::OkStatus();
}
absl::Status CheckStaticShape(const at::Tensor& tensor,
                              const std::string_view arg_name) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref, GetBaseBuffer(tensor));
  return CheckStaticShape(tensor, buffer_ref, arg_name);
}

}  // namespace

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
        TT_THROW_IF_ERROR(CheckWeightType(weight));
        TT_THROW_IF_ERROR(CheckStaticShape(weight, "weight"));
        TT_THROW_IF_ERROR(CheckStaticShape(indices, "indices"));
        TT_THROW_IF_ERROR(CheckStaticShape(offsets, "offsets"));

        TT_ASSIGN_OR_THROW(auto weight_dtype,
                           ConvertTo<mlir::ElementType>(weight.scalar_type()));
        TT_ASSIGN_OR_THROW(auto i64_dtype,
                           ConvertTo<mlir::ElementType>(at::kLong));

        int64_t batch_size = offsets.numel();
        if (include_last_offset) batch_size -= 1;
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
        Dimensions d3 = {batch_size, emb_dim};
        std::array<absl::Span<const int64_t>, 4> out_dims_list = {d0, d1, d2,
                                                                  d3};

        TT_ASSIGN_OR_THROW(auto results,
                           (DispatchOp<kDynamicSize, 4>(
                               builder_fn, inputs,
                               {.out_dtypes = out_dtypes,
                                .out_dims_list = out_dims_list,
                                .op_param_cache_keys = std::move(pk)})));

        return std::make_tuple(MakeTensor(std::move(results[0])),
                               MakeTensor(std::move(results[1])),
                               MakeTensor(std::move(results[2])),
                               MakeTensor(std::move(results[3])));
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenEmbeddingBagForwardOnly(const at::Tensor& weight, const at::Tensor& indices,
                            const at::Tensor& offsets, bool scale_grad_by_freq,
                            int64_t mode, bool sparse,
                            const std::optional<at::Tensor>& per_sample_weights,
                            bool include_last_offset, int64_t padding_idx) {
  TT_KERNEL(OpName::kEmbeddingBagForwardOnly, _,
            (weight, indices, offsets,
             IgnoreInCacheKey(scale_grad_by_freq, "Legacy usage"),
             IgnoreInCacheKey(mode, "Legacy usage"),
             IgnoreInCacheKey(sparse, "Legacy usage"),
             IgnoreInCacheKey(per_sample_weights, "Legacy usage"),
             IgnoreInCacheKey(include_last_offset, "Legacy usage"),
             IgnoreInCacheKey(padding_idx, "Legacy usage")),
            {
              return AtenEmbeddingBag(
                  weight, indices, offsets, scale_grad_by_freq, mode, sparse,
                  per_sample_weights, include_last_offset, padding_idx);
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
        bool has_psw =
            per_sample_weights.has_value() && per_sample_weights->defined();
        int64_t nw_val = num_weights.expect_int();

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

        return MakeTensor(std::move(results));
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

        return MakeTensor(std::move(result));
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

        Dimensions out_dims(indices.sizes().begin(), indices.sizes().end());
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

        self.index_put_({indices}, MakeTensor(std::move(renorm_rows)));
        return self;
      });
}

}  // namespace torch_tpu
