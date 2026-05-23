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

#include "torch_tpu/ops/bmm/bmm_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/bmm/bmm.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/precision_context.h"

namespace torch_tpu {

namespace {

absl::Status CheckBmmOut(const at::Tensor& out) {
  TT_RET_CHECK(!IsBool(out), error::kInvalidArgument)
      << "the dtype of the output tensor cannot be bool";
  return absl::OkStatus();
}

absl::Status CheckBmmInputs(const at::Tensor& self, const at::Tensor& mat2) {
  TT_RET_CHECK(!IsBool(self), error::kInvalidArgument)
      << "the dtype of the first argument cannot be bool";
  TT_RET_CHECK(self.dim() == 3, error::kInvalidArgument)
      << "expected the first argument to be a 3D tensor (batch of matrices), "
         "got "
      << self.dim() << "D";

  TT_RET_CHECK(!IsBool(mat2), error::kInvalidArgument)
      << "the dtype of the second argument cannot be bool";
  TT_RET_CHECK(mat2.dim() == 3, error::kInvalidArgument)
      << "expected the second argument to be a 3D tensor (batch of matrices), "
         "got "
      << mat2.dim() << "D";

  TT_RET_CHECK(self.size(0) == mat2.size(0), error::kInvalidArgument)
      << "expected the batch dimension of the first argument "
      << ToString(self.sizes())
      << " to match the batch dimension of the second argument "
      << ToString(mat2.sizes()) << ", got " << self.size(0) << " vs "
      << mat2.size(0);
  TT_RET_CHECK(self.size(2) == mat2.size(1), error::kInvalidArgument)
      << "expected the last dimension of the first argument "
      << ToString(self.sizes())
      << " to match the second dimension of the second argument "
      << ToString(mat2.sizes()) << ", got " << self.size(2) << " vs "
      << mat2.size(1);

  return absl::OkStatus();
}

absl::StatusOr<DeviceBufferRef> Bmm(const at::Tensor& self,
                                    const at::Tensor& mat2,
                                    at::ScalarType out_dtype,
                                    OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(CheckBmmInputs(self, mat2));
  TT_ASSIGN_OR_RETURN(mlir::ElementType output_dtype_mlir,
                      ConvertTo<mlir::ElementType>(out_dtype));
  Dimensions output_dims_vec = {self.size(0), self.size(1), mat2.size(2)};

  const auto current_precision = GetPrecision();
  TT_ASSIGN_OR_RETURN(param_keys,
                      *OpParamCacheKeys::Builder(std::move(param_keys))
                           .SetParam("precision", current_precision));

  auto op_builder = [output_dtype_mlir,
                     current_precision](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, mat2_op] = inputs;
    TT_ASSIGN_OR_RETURN(
        auto result,
        BuildBmmShlo(self_op, mat2_op, output_dtype_mlir, current_precision));
    return result;
  };
  TT_ASSIGN_OR_RETURN(
      auto result_buffer,
      DispatchOp<2>(std::move(op_builder), {self, mat2},
                    {.out_dtype = output_dtype_mlir,
                     .out_dims = output_dims_vec,
                     .op_param_cache_keys = std::move(param_keys)}));
  return result_buffer;
}

}  // namespace

at::Tensor AtenBmmDtype(const at::Tensor& self, const at::Tensor& mat2,
                        at::ScalarType out_dtype) {
  TT_KERNEL(OpName::kBmm, param_keys, (self, mat2, out_dtype), {
    TT_ASSIGN_OR_THROW(auto result_buffer,
                       Bmm(self, mat2, out_dtype, std::move(param_keys)));
    return MakeTensor(result_buffer);
  });
}

at::Tensor& AtenBmmDtypeOut(const at::Tensor& self, const at::Tensor& mat2,
                            at::ScalarType out_dtype, at::Tensor& out) {
  TT_KERNEL(OpName::kBmm, param_keys, (self, mat2, out_dtype, out), {
    TT_THROW_IF_ERROR(CheckBmmOut(out));
    TT_ASSIGN_OR_THROW(auto result_buffer,
                       Bmm(self, mat2, out_dtype, std::move(param_keys)));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buffer), out));
    return out;
  });
}

at::Tensor& AtenBmmOut(const at::Tensor& self, const at::Tensor& mat2,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kBmm, _, (self, mat2, out),
            { return AtenBmmDtypeOut(self, mat2, out.scalar_type(), out); });
}

}  // namespace torch_tpu
