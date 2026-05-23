// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/ops/min_max/min_max_aten_kernels.h"

#include <cstdint>
#include <tuple>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/Fill.h"
#include "ATen/native/ReduceOpsUtils.h"
#include "ATen/native/Resize.h"
#include "ATen/ops/empty.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/util/DimVector.h"
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
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/min_max/min_max.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

namespace {

absl::StatusOr<std::tuple<at::Tensor, at::Tensor>> AtenMinMaxDim(
    const at::Tensor& self, int64_t dim, bool keep_dim, MinMaxOp min_max_op,
    OpParamCacheKeys& op_param_cache_keys, at::Tensor& value,
    at::Tensor& indices) {
  ABSL_VLOG(3) << "[AtenMinMaxDim] start";

  TT_ASSIGN_OR_RETURN(const int64_t normalized_dim,
                      SafeWrapDim(dim, self.dim()));

  Dimensions output_dims;
  at::Tensor input = self;
  auto tensor_shape = at::meta::get_reduction_shape(self, {dim}, keep_dim,
                                                    /*allow_empty_dims=*/false);
  output_dims = CopyIntVector(tensor_shape);

  TT_ASSIGN_OR_RETURN(mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  ReductionMode mode =
      keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
  auto op_builder =
      [normalized_dim, min_max_op,
       mode](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<2>> {
    TT_ASSIGN_OR_RETURN(
        auto min_max_shlo_outputs,
        BuildMinMaxShlo(normalized_dim, min_max_op, mode, input));
    return {{min_max_shlo_outputs.values, min_max_shlo_outputs.indices}};
  };

  TT_ASSIGN_OR_RETURN(
      (auto [values_buf, indices_buf]),
      (DispatchOp<1, 2>(
          std::move(op_builder), input,
          {
              .out_dtypes = {output_dtype, mlir::ElementType::I64},
              .out_dims_list = {output_dims, output_dims},
              .op_param_cache_keys = std::move(op_param_cache_keys),
          })));

  TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(values_buf), value));
  TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(indices_buf), indices));
  return {{value, indices}};
}

absl::Status CheckNotZeroElementTensor(const at::Tensor& tensor) {
  TT_RET_CHECK(tensor.numel() > 0, error::kInvalidArgument)
      << "expected the dim argument to be specified when the input tensor has "
         "0 elements";

  return absl::OkStatus();
}

}  // namespace

absl::Status ArgMinMax(const at::Tensor& self, c10::optional<int64_t> dim,
                       bool keep_dim, MinMaxOp op, at::Tensor& out,
                       OpParamCacheKeys param_keys) {
  TT_RET_CHECK(IsPrivateUse1Device(out), error::kInvalidArgument)
      << "expected output tensor to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << out.device();
  TT_RET_CHECK(IsLong(out), error::kInvalidArgument)
      << "expected the output dtype to be int64, got "
      << ToString(out.scalar_type());

  at::Tensor
      input_tensor;  // UNINITIALIZED_TENSOR_OK=initialized in the if-else
  int64_t wrapped_dim;
  if (dim) {
    input_tensor = self;
    TT_ASSIGN_OR_RETURN(wrapped_dim, SafeWrapDim(dim.value(), self.dim()));
    auto sizes = input_tensor.sizes();
    if (sizes[wrapped_dim] == 1) {
      out.fill_(0);
      return absl::OkStatus();
    }
  } else {
    input_tensor = self.flatten();
    wrapped_dim = 0;
  }
  // native PyTorch works for complex and bool dtypes only when the input
  // dimension corresponding to the wrapped dimension is 1. All other cases are
  // not supported. Hence, this check is after the above logic to handle the
  // special case where the input dimension corresponding to the wrapped
  // dimension is 1.
  TT_RET_CHECK(!IsComplex(self) && !IsBool(self), error::kInvalidArgument)
      << "expected the input dtype to be neither complex nor bool, got "
      << ToString(self.scalar_type());

  // If the input tensor is a scalar, then argmax should just return 0.
  if (self.dim() == 0) {
    out.fill_(0);
    return absl::OkStatus();
  }

  c10::DimVector out_shape = at::meta::get_reduction_shape(
      input_tensor, {wrapped_dim}, keep_dim, /*allow_empty_dims=*/false);
  Dimensions out_dims = CopyIntVector(out_shape);

  ReductionMode mode =
      keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;

  auto op_builder =
      [wrapped_dim, op,
       mode](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<1>> {
    TT_ASSIGN_OR_RETURN(auto min_max_outputs,
                        BuildMinMaxShlo(wrapped_dim, op, mode, input));
    return {min_max_outputs.indices};
  };

  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<1>(std::move(op_builder), input_tensor,
                    {.out_dtype = mlir::ElementType::I64,
                     .out_dims = std::move(out_dims),
                     .op_param_cache_keys = std::move(param_keys)}));

  at::native::resize_output(out, out_shape);
  return AssignBufferToAtTensor(std::move(result_buf), out);
}

at::Tensor& AtenArgmaxOut(const at::Tensor& self, c10::optional<int64_t> dim,
                          bool keep_dim, at::Tensor& out) {
  TT_KERNEL(OpName::kArgMaxOut, _,
            (self, IgnoreInCacheKey(dim, "Legacy usage"),
             IgnoreInCacheKey(keep_dim, "Legacy usage"), out),
            {
              // keep_dim does not affect the SHLO and therefore does not need
              // to be included in the cache key.
              TT_ASSIGN_OR_THROW(
                  auto param_keys,
                  *OpParamCacheKeysBuilder().SetParam("dim", dim));

              TT_THROW_IF_ERROR(ArgMinMax(self, dim, keep_dim, MinMaxOp::kMax,
                                          out, std::move(param_keys)));
              return out;
            });
}

at::Tensor& AtenArgminOut(const at::Tensor& self, c10::optional<int64_t> dim,
                          bool keep_dim, at::Tensor& out) {
  TT_KERNEL(OpName::kArgMinOut, _,
            (self, IgnoreInCacheKey(dim, "Legacy usage"),
             IgnoreInCacheKey(keep_dim, "Legacy usage"), out),
            {
              // keep_dim does not affect the SHLO and therefore does not need
              // to be included in the cache key.
              TT_ASSIGN_OR_THROW(
                  auto param_keys,
                  *OpParamCacheKeysBuilder().SetParam("dim", dim));

              TT_THROW_IF_ERROR(ArgMinMax(self, dim, keep_dim, MinMaxOp::kMin,
                                          out, std::move(param_keys)));
              return out;
            });
}

at::Tensor AtenMax(const at::Tensor& self) {
  TT_KERNEL(OpName::kMax, _, (self), {
    TT_THROW_IF_ERROR(CheckNotZeroElementTensor(self));
    at::Tensor self_flat = self.reshape(-1);
    at::Tensor max = at::empty({}, self.options());
    at::Tensor max_indices = at::empty({}, self.options().dtype(at::kLong));
    AtenMaxDimMax(self_flat, /*dim=*/0, /*keep_dim=*/false, /*max=*/max,
                  /*max_indices=*/max_indices);
    return max;
  });
}

at::Tensor& AtenMaxUnaryOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kMaxUnaryOut, _, (self, out), {
    out = AtenMax(self);
    return out;
  });
}

std::tuple<at::Tensor&, at::Tensor&> AtenMaxDimMax(const at::Tensor& self,
                                                   int64_t dim, bool keep_dim,
                                                   at::Tensor& max,
                                                   at::Tensor& max_indices) {
  TT_KERNEL(
      OpName::kMaxDimMax, param_keys, (self, dim, keep_dim, max, max_indices), {
        TT_THROW_IF_ERROR(AtenMinMaxDim(self, dim, keep_dim, MinMaxOp::kMax,
                                        param_keys, max, max_indices));
        return {max, max_indices};
      });
}

at::Tensor AtenMin(const at::Tensor& self) {
  TT_KERNEL(OpName::kMin, _, (self), {
    TT_THROW_IF_ERROR(CheckNotZeroElementTensor(self));
    at::Tensor self_flat = self.reshape(-1);
    at::Tensor min = at::empty({}, self.options());
    at::Tensor min_indices = at::empty({}, self.options().dtype(at::kLong));
    AtenMinDimMin(self_flat, /*dim=*/0, /*keep_dim=*/false, /*min=*/min,
                  /*min_indices=*/min_indices);
    return min;
  });
}

at::Tensor& AtenMinUnaryOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kMinUnaryOut, _, (self, out), {
    out = AtenMin(self);
    return out;
  });
}

std::tuple<at::Tensor&, at::Tensor&> AtenMinDimMin(const at::Tensor& self,
                                                   int64_t dim, bool keep_dim,
                                                   at::Tensor& min,
                                                   at::Tensor& min_indices) {
  TT_KERNEL(
      OpName::kMinDimMin, param_keys, (self, dim, keep_dim, min, min_indices), {
        TT_THROW_IF_ERROR(AtenMinMaxDim(self, dim, keep_dim, MinMaxOp::kMin,
                                        param_keys, min, min_indices));
        return {min, min_indices};
      });
}

}  // namespace torch_tpu
