/*
 * Copyright 2026 Google LLC
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

#include "csrc/ops/linalg/cross/linalg_cross_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/linalg/cross/linalg_cross.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

absl::Status ValidateLinalgCrossInputs(const at::Tensor& self,
                                       const at::Tensor& other, int64_t dim,
                                       const at::Tensor& out,
                                       int64_t& wrapped_dim) {
  TT_RET_CHECK(self.dim() == other.dim(), error::kInvalidArgument)
      << "expected inputs to have the same number of dimensions, got "
      << self.dim() << " and " << other.dim();
  TT_RET_CHECK(self.dim() > 0, error::kIndexError)
      << "expected input tensor to have at least 1 dimension, got 0";
  TT_ASSIGN_OR_RETURN(wrapped_dim, SafeWrapDim(dim, self.dim()));
  TT_RET_CHECK(self.size(wrapped_dim) == 3 && other.size(wrapped_dim) == 3,
               error::kInvalidArgument)
      << "expected inputs dimension " << dim << " to have length 3, got "
      << self.size(wrapped_dim) << " and " << other.size(wrapped_dim);
  TT_RET_CHECK(self.scalar_type() == other.scalar_type(),
               error::kInvalidArgument)
      << "expected self and other to have the same dtype, got "
      << ToString(self.scalar_type()) << " and "
      << ToString(other.scalar_type());
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, self.scalar_type(),
                                      /*allow_out_dtype_cast=*/false));
  TT_RET_CHECK(!IsBool(self), error::kPythonNotImplementedError)
      << "not implemented for " << ToString(self.scalar_type());
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(self));
  return absl::OkStatus();
}

absl::Status LinalgCrossOut(const at::Tensor& self, const at::Tensor& other,
                            int64_t dim, at::Tensor& out,
                            OpParamCacheKeys param_keys) {
  int64_t wrapped_dim = 0;
  TT_RETURN_IF_ERROR(
      ValidateLinalgCrossInputs(self, other, dim, out, wrapped_dim));
  TT_ASSIGN_OR_RETURN(const Dimensions output_dims,
                      InferSize(self.sizes(), other.sizes()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  auto op_builder = [wrapped_dim](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [lhs_op, rhs_op] = inputs;
    return BuildLinalgCrossShlo(lhs_op, rhs_op, wrapped_dim);
  };

  return DispatchOpOut<2>(std::move(op_builder), {self, other}, out,
                          {.out_dtype = out_dtype,
                           .out_dims = output_dims,
                           .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor& AtenLinalgCrossOut(const at::Tensor& self, const at::Tensor& other,
                               int64_t dim, at::Tensor& out) {
  TT_KERNEL(OpName::kLinalgCrossOut, param_keys, (self, other, dim, out), {
    TT_THROW_IF_ERROR(
        LinalgCrossOut(self, other, dim, out, std::move(param_keys)));
    return out;
  });
}

}  // namespace torch_tpu
