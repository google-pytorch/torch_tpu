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

#include "torch_tpu/ops/all_any/all_any_aten_kernels.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/all_any/all_any.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reduction_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {

namespace {

MlirUnaryOpBuilder GetAllBuilder(Dimensions reduce_dims,
                                 ReductionMode reduction_mode) {
  return std::bind(&BuildAllShlo, std::placeholders::_1, reduce_dims,
                   reduction_mode);
}

MlirUnaryOpBuilder GetAnyBuilder(Dimensions reduce_dims,
                                 ReductionMode reduction_mode) {
  return std::bind(&BuildAnyShlo, std::placeholders::_1, reduce_dims,
                   reduction_mode);
}

struct AllAnyReductionParams {
  Dimensions dims_to_reduce;
  ReductionMode reduction_mode;
  Dimensions reduced_shape;
};

absl::StatusOr<AllAnyReductionParams> GetAllAnyReductionParams(
    const at::Tensor& self, c10::OptionalArrayRef<int64_t> dims,
    bool keep_dims) {
  Dimensions dims_to_reduce;
  if (!dims.has_value()) {
    dims_to_reduce.resize(self.dim());
    absl::c_iota(dims_to_reduce, 0);
  } else {
    TT_ASSIGN_OR_RETURN(dims_to_reduce, CanonicalizeDims(self, dims));
  }

  ReductionMode reduction_mode =
      keep_dims ? ReductionMode::kKeepDims : ReductionMode::kDropDims;

  Dimensions reduced_shape =
      GetSizesAfterReduction(self.sizes(), reduction_mode, dims_to_reduce);

  return AllAnyReductionParams{
      .dims_to_reduce = std::move(dims_to_reduce),
      .reduction_mode = reduction_mode,
      .reduced_shape = std::move(reduced_shape),
  };
}

// Conditionally casts the result tensor to at::kByte if the original self
// tensor has scalar type at::kByte.
// See https://docs.pytorch.org/docs/stable/generated/torch.all.html#torch.all
// for why this cast is needed.
at::Tensor& MaybeCastToByte(const at::Tensor& self, at::Tensor& result) {
  if (self.scalar_type() == at::kByte) {
    result = result.to(at::kByte);
  }
  return result;
}

}  // namespace

at::Tensor& AtenAllOut(const at::Tensor& self, int64_t dim, bool keep_dim,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kAllOut, param_keys, (self, dim, keep_dim, out), {
    TT_ASSIGN_OR_THROW(auto params,
                       GetAllAnyReductionParams(self, {dim}, keep_dim));
    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out,
        GetAllBuilder(std::move(params.dims_to_reduce), params.reduction_mode),
        {.op_param_cache_keys = std::move(param_keys),
         .out_dims = std::move(params.reduced_shape)}));
    return MaybeCastToByte(self, out);
  });
}

// This is all.all_out, NOT all.out.
at::Tensor& AtenAllAllOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kAllAllOut, _, (self, out), {
    TT_ASSIGN_OR_THROW(
        auto params,
        GetAllAnyReductionParams(self, std::nullopt, /*keep_dims=*/false));
    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out,
        GetAllBuilder(std::move(params.dims_to_reduce), params.reduction_mode),
        {.op_param_cache_keys = OpParamCacheKeys::Empty(),
         .out_dims = std::move(params.reduced_shape)}));
    return MaybeCastToByte(self, out);
  });
}

at::Tensor& AtenAnyOut(const at::Tensor& self, int64_t dim, bool keep_dim,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kAnyOut, param_keys, (self, dim, keep_dim, out), {
    TT_ASSIGN_OR_THROW(auto params,
                       GetAllAnyReductionParams(self, {dim}, keep_dim));
    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out,
        GetAnyBuilder(std::move(params.dims_to_reduce), params.reduction_mode),
        {.op_param_cache_keys = std::move(param_keys),
         .out_dims = std::move(params.reduced_shape)}));
    return MaybeCastToByte(self, out);
  });
}

at::Tensor& AtenAnyAllOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kAnyAllOut, _, (self, out), {
    TT_ASSIGN_OR_THROW(
        auto params,
        GetAllAnyReductionParams(self, std::nullopt, /*keep_dims=*/false));
    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out,
        GetAnyBuilder(std::move(params.dims_to_reduce), params.reduction_mode),
        {.op_param_cache_keys = OpParamCacheKeys::Empty(),
         .out_dims = std::move(params.reduced_shape)}));
    return MaybeCastToByte(self, out);
  });
}

}  // namespace torch_tpu
