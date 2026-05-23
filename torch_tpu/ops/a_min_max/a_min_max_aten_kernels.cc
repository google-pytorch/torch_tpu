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

#include "torch_tpu/ops/a_min_max/a_min_max_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "ATen/WrapDimUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/Fill.h"
#include "ATen/native/ReduceOpsUtils.h"
#include "ATen/native/Resize.h"
#include "absl/functional/bind_front.h"
#include "absl/status/status.h"
#include "c10/util/DimVector.h"
#include "c10/util/Optional.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
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
#include "torch_tpu/ops/a_min_max/a_min_max.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

namespace {

OpName GetOpName(AMinMaxOp op) {
  return op == AMinMaxOp::kAmax ? OpName::kAmax : OpName::kAmin;
}

absl::Status CheckAMinMaxInputs(const at::Tensor& self, at::Tensor& out) {
  TT_RET_CHECK(IsPrivateUse1Device(out), error::kInvalidArgument)
      << "expected output tensor to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << out.device();

  TT_RET_CHECK(!IsComplex(self), error::kInvalidArgument)
      << "expected the dtype of the input not to be complex, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(out.scalar_type() == self.scalar_type(), error::kInvalidArgument)
      << "expected output tensor dtype to match the dtype of the first "
         "argument ("
      << ToString(self.scalar_type()) << "), got "
      << ToString(out.scalar_type());

  return absl::OkStatus();
}

absl::Status AMinMax(const at::Tensor& self, const at::IntArrayRef dims,
                     const bool keep_dim, const AMinMaxOp a_min_max_op,
                     at::Tensor& out) {
  TT_RETURN_IF_ERROR(CheckAMinMaxInputs(self, out));
  TT_ASSIGN_OR_RETURN(auto param_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(dims, keep_dim));
  if (self.dim() == 0) {
    out.copy_(self);
    return absl::OkStatus();
  }

  c10::DimVector dim_vec = at::native::make_dim_vector(
      dims.empty() ? std::nullopt : std::make_optional(dims), self.dim());
  at::maybe_wrap_dims(dim_vec, self.dim());
  auto out_shape = at::meta::get_reduction_shape(self, dim_vec, keep_dim,
                                                 /*allow_empty_dims=*/false);
  Dimensions out_dims = CopyIntVector(out_shape);

  TT_ASSIGN_OR_RETURN(auto out_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  ReductionMode mode =
      keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;

  Dimensions reduction_dims = CopyIntVector(dim_vec);
  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<1>(absl::bind_front(BuildAMinMaxShlo, reduction_dims, mode,
                                     a_min_max_op),
                    self,
                    // Override the op name as opposed to using the one passed
                    // to TT_KERNEL, as this is a shared utility function.
                    {.op_name = GetOpName(a_min_max_op),
                     .out_dtype = out_dtype,
                     .out_dims = std::move(out_dims),
                     .op_param_cache_keys = std::move(param_keys)}));

  at::native::resize_output(out, out_shape);
  return AssignBufferToAtTensor(std::move(result_buf), out);
}

}  // namespace

at::Tensor& AtenAmaxOut(const at::Tensor& self, const at::IntArrayRef dims,
                        const bool keep_dim, at::Tensor& out) {
  TT_KERNEL(
      OpName::kAmaxOut, _,
      (self, IgnoreInCacheKey(dims, "delegates to AMinMax"),
       IgnoreInCacheKey(keep_dim, "delegates to AMinMax"), out),
      {
        TT_THROW_IF_ERROR(AMinMax(self, dims, keep_dim, AMinMaxOp::kAmax, out));
        return out;
      });
}

at::Tensor& AtenAminOut(const at::Tensor& self, const at::IntArrayRef dims,
                        const bool keep_dim, at::Tensor& out) {
  TT_KERNEL(
      OpName::kAminOut, _,
      (self, IgnoreInCacheKey(dims, "delegates to AMinMax"),
       IgnoreInCacheKey(keep_dim, "delegates to AMinMax"), out),
      {
        TT_THROW_IF_ERROR(AMinMax(self, dims, keep_dim, AMinMaxOp::kAmin, out));
        return out;
      });
}

std::tuple<at::Tensor&, at::Tensor&> AtenAminmaxOut(
    const at::Tensor& self, const c10::optional<int64_t> dim,
    const bool keep_dim, at::Tensor& min, at::Tensor& max) {
  TT_KERNEL(
      OpName::kAminmaxOut, _,
      (self, IgnoreInCacheKey(dim, "delegates to AMinMax"),
       IgnoreInCacheKey(keep_dim, "delegates to AMinMax"), min, max),
      {
        // aminmax only supports one optional dimension, but the builder expects
        // a vector of dimensions, so we build a vector here for the optional
        // dim.
        c10::DimVector dims_vec;
        if (dim.has_value()) {
          dims_vec.push_back(*dim);
        }
        at::IntArrayRef dims(dims_vec);
        TT_THROW_IF_ERROR(AMinMax(self, dims, keep_dim, AMinMaxOp::kAmin, min));
        TT_THROW_IF_ERROR(AMinMax(self, dims, keep_dim, AMinMaxOp::kAmax, max));
        return std::forward_as_tuple(min, max);
      });
}

}  // namespace torch_tpu
