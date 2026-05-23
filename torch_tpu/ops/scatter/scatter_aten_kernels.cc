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

#include "torch_tpu/ops/scatter/scatter_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/full_like.h"
#include "ATen/ops/result_type.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/string_view.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/scatter/scatter.h"

namespace torch_tpu {
namespace {

absl::StatusOr<DeviceBufferRef> Scatter(
    const at::Tensor& self, int64_t dim, const at::Tensor& index,
    const at::Tensor& src, ScatterOp reduction_op = ScatterOp::kReplace,
    ScatterIncludeSelf include_self = ScatterIncludeSelf::kYes) {
  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self.dim()));

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(const auto promoted_dtype,
                      ConvertTo<mlir::ElementType>(at::result_type(self, src)));
  Dimensions output_dims = CopyIntVector(self.sizes());
  auto scatter_op_builder = [dim, reduction_op, promoted_dtype, include_self](
                                FixedSizeSpan<mlir::MlirOp, 3> inputs) {
    auto& [self, index, src] = inputs;
    return BuildScatterShlo(self, dim, index, src, reduction_op, promoted_dtype,
                            include_self);
  };

  TT_ASSIGN_OR_RETURN(auto param_keys, TT_MAKE_OP_PARAM_CACHE_KEYS(
                                           dim, reduction_op, include_self));
  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<3>(std::move(scatter_op_builder), {self, index, src},
                    {.out_dtype = output_dtype,
                     .out_dims = output_dims,
                     .op_param_cache_keys = std::move(param_keys)}));
  return result_buf;
}

absl::StatusOr<ScatterOp> ParseScatterOp(
    std::string_view reduction_op,
    ScatterVersion version = ScatterVersion::kV1) {
  if (version == ScatterVersion::kV2) {
    // scatter_reduce.two op only supports sum, prod, mean, amax, and amin
    if (reduction_op == "sum") {
      return ScatterOp::kSum;
    } else if (reduction_op == "prod") {
      return ScatterOp::kProd;
    } else if (reduction_op == "mean") {
      return ScatterOp::kMean;
    } else if (reduction_op == "amax") {
      return ScatterOp::kAmax;
    } else if (reduction_op == "amin") {
      return ScatterOp::kAmin;
    } else {
      return TT_ERROR(error::kInvalidArgument)
             << "expected the reduction_op to be 'sum', 'prod', 'mean', "
                "'amax', or 'amin', got '"
             << reduction_op << "'";
    }
  } else {
    // scatter supports add and multiply.
    if (reduction_op == "add") {
      return ScatterOp::kAdd;
    } else if (reduction_op == "multiply") {
      return ScatterOp::kMul;
    } else {
      return TT_ERROR(error::kInvalidArgument)
             << "expected the reduction_op to be 'add' or 'multiply', got '"
             << reduction_op << "'";
    }
  }
}

at::Tensor TensorFromValue(const at::Tensor& self, const at::Scalar& value) {
  auto promoted_type = at::result_type(self, value);
  return at::full_like(self, value, promoted_type, std::nullopt, self.device(),
                       std::nullopt, std::nullopt);
}

}  // namespace

at::Tensor& AtenScatterSrcOut(const at::Tensor& self, int64_t dim,
                              const at::Tensor& index, const at::Tensor& src,
                              at::Tensor& out) {
  TT_KERNEL(OpName::kScatterSrcOut, _,
            (self, IgnoreInCacheKey(dim, "Legacy usage"), index, src, out), {
              TT_ASSIGN_OR_THROW(DeviceBufferRef result,
                                 Scatter(self, dim, index, src));
              TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
              return out;
            });
}

at::Tensor& AtenScatterValueOut(const at::Tensor& self, int64_t dim,
                                const at::Tensor& index,
                                const at::Scalar& value, at::Tensor& out) {
  TT_KERNEL(OpName::kScatterValueOut, _,
            (self, IgnoreInCacheKey(dim, "Legacy usage"), index,
             IgnoreInCacheKey(value, "Legacy usage"), out),
            {
              auto value_tensor = TensorFromValue(self, value);
              TT_ASSIGN_OR_THROW(DeviceBufferRef result,
                                 Scatter(self, dim, index, value_tensor));
              TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
              return out;
            });
}

at::Tensor& AtenScatterReduceOut(const at::Tensor& self, int64_t dim,
                                 const at::Tensor& index, const at::Tensor& src,
                                 std::string_view reduction_op,
                                 at::Tensor& out) {
  TT_KERNEL(OpName::kScatterReduceOut, _,
            (self, IgnoreInCacheKey(dim, "Legacy usage"), index, src,
             IgnoreInCacheKey(reduction_op, "Legacy usage"), out),
            {
              TT_ASSIGN_OR_THROW(ScatterOp scatter_op,
                                 ParseScatterOp(reduction_op));
              TT_ASSIGN_OR_THROW(DeviceBufferRef result,
                                 Scatter(self, dim, index, src, scatter_op));
              TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
              return out;
            });
}

at::Tensor& AtenScatterReduceTwoOut(const at::Tensor& self, int64_t dim,
                                    const at::Tensor& index,
                                    const at::Tensor& src,
                                    c10::string_view reduction,
                                    bool include_self, at::Tensor& out) {
  TT_KERNEL(
      OpName::kScatterReduceTwoOut, _,
      (self, IgnoreInCacheKey(dim, "delegates to Scatter()"), index, src,
       IgnoreInCacheKey(reduction, "delegates to Scatter()"),
       IgnoreInCacheKey(include_self, "delegates to Scatter()"), out),
      {
        TT_ASSIGN_OR_THROW(ScatterOp scatter_op,
                           ParseScatterOp(reduction, ScatterVersion::kV2));
        TT_ASSIGN_OR_THROW(DeviceBufferRef result,
                           Scatter(self, dim, index, src, scatter_op,
                                   include_self ? ScatterIncludeSelf::kYes
                                                : ScatterIncludeSelf::kNo));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
        return out;
      });
}

at::Tensor& AtenScatterValueReduceOut(const at::Tensor& self, int64_t dim,
                                      const at::Tensor& index,
                                      const at::Scalar& value,
                                      std::string_view reduction_op,
                                      at::Tensor& out) {
  TT_KERNEL(
      OpName::kScatterValueReduceOut, _,
      (self, IgnoreInCacheKey(dim, "Legacy usage"), index,
       IgnoreInCacheKey(value, "Legacy usage"),
       IgnoreInCacheKey(reduction_op, "Legacy usage"), out),
      {
        auto value_tensor = TensorFromValue(self, value);
        TT_ASSIGN_OR_THROW(ScatterOp scatter_op, ParseScatterOp(reduction_op));
        TT_ASSIGN_OR_THROW(DeviceBufferRef result,
                           Scatter(self, dim, index, value_tensor, scatter_op));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
        return out;
      });
}

at::Tensor& AtenScatterAddOut(const at::Tensor& self, int64_t dim,
                              const at::Tensor& index, const at::Tensor& src,
                              at::Tensor& out) {
  TT_KERNEL(OpName::kScatterAddOut, _,
            (self, IgnoreInCacheKey(dim, "Legacy usage"), index, src, out), {
              return AtenScatterReduceOut(self, dim, index, src, "add", out);
            });
}

}  // namespace torch_tpu
