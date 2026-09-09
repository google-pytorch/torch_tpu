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

#include "csrc/ops/all_any/all_any_aten_kernels.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/all_any/all_any.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reduction_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

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

// Verifies that the out tensor has a valid dtype for `all` and `any`.
// PyTorch only allows `bool` and `uint8` (`kByte`) for out (refer to
// check_result_is_bytebool in PyTorch's ReduceOps.cpp).
absl::Status ValidateAllAnyOut(const at::Tensor& out) {
  TT_RET_CHECK(out.scalar_type() == at::kBool || out.scalar_type() == at::kByte,
               error::kInvalidArgument)
      << "expected the output dtype to be bool or uint8, got "
      << ToString(out.scalar_type());
  return absl::OkStatus();
}

// Returns the output element type for `all` and `any` out-variants.
// If the out tensor is uint8, produces UI8 (refer to
// get_result_or_bytebool_dtype in PyTorch's ReduceOps.cpp), otherwise PRED.
mlir::ElementType GetAllAnyOutDtype(const at::Tensor& out) {
  return out.scalar_type() == at::kByte ? mlir::ElementType::UI8
                                        : mlir::ElementType::PRED;
}

}  // namespace

at::Tensor& AtenAllOut(const at::Tensor& self, int64_t dim, bool keep_dim,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kAllOut, param_keys, (self, dim, keep_dim, out), {
    TT_THROW_IF_ERROR(ValidateAllAnyOut(out));
    TT_ASSIGN_OR_THROW(auto params,
                       GetAllAnyReductionParams(self, {dim}, keep_dim));
    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out,
        GetAllBuilder(std::move(params.dims_to_reduce), params.reduction_mode),
        {.op_param_cache_keys = std::move(param_keys),
         .out_dtype = GetAllAnyOutDtype(out),
         .out_dims = std::move(params.reduced_shape),
         .allow_out_dtype_cast = false}));
    return out;
  });
}

// This is all.all_out, NOT all.out.
at::Tensor& AtenAllAllOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kAllAllOut, _, (self, out), {
    TT_THROW_IF_ERROR(ValidateAllAnyOut(out));
    TT_ASSIGN_OR_THROW(
        auto params,
        GetAllAnyReductionParams(self, std::nullopt, /*keep_dims=*/false));
    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out,
        GetAllBuilder(std::move(params.dims_to_reduce), params.reduction_mode),
        {.op_param_cache_keys = OpParamCacheKeys::Empty(),
         .out_dtype = GetAllAnyOutDtype(out),
         .out_dims = std::move(params.reduced_shape),
         .allow_out_dtype_cast = false}));
    return out;
  });
}

at::Tensor& AtenAnyOut(const at::Tensor& self, int64_t dim, bool keep_dim,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kAnyOut, param_keys, (self, dim, keep_dim, out), {
    TT_THROW_IF_ERROR(ValidateAllAnyOut(out));
    TT_ASSIGN_OR_THROW(auto params,
                       GetAllAnyReductionParams(self, {dim}, keep_dim));
    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out,
        GetAnyBuilder(std::move(params.dims_to_reduce), params.reduction_mode),
        {.op_param_cache_keys = std::move(param_keys),
         .out_dtype = GetAllAnyOutDtype(out),
         .out_dims = std::move(params.reduced_shape),
         .allow_out_dtype_cast = false}));
    return out;
  });
}

at::Tensor& AtenAnyAllOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kAnyAllOut, _, (self, out), {
    TT_THROW_IF_ERROR(ValidateAllAnyOut(out));
    TT_ASSIGN_OR_THROW(
        auto params,
        GetAllAnyReductionParams(self, std::nullopt, /*keep_dims=*/false));
    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out,
        GetAnyBuilder(std::move(params.dims_to_reduce), params.reduction_mode),
        {.op_param_cache_keys = OpParamCacheKeys::Empty(),
         .out_dtype = GetAllAnyOutDtype(out),
         .out_dims = std::move(params.reduced_shape),
         .allow_out_dtype_cast = false}));
    return out;
  });
}

}  // namespace torch_tpu
