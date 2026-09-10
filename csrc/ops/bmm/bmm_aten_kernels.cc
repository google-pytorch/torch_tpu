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

#include "csrc/ops/bmm/bmm_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/bmm/bmm.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

absl::Status ValidateBmmOut(const at::Tensor& out, at::ScalarType out_dtype) {
  TT_RET_CHECK(out.scalar_type() == out_dtype, error::kInvalidArgument)
      << "expected out tensor to have dtype " << ToString(out_dtype) << ", got "
      << ToString(out.scalar_type());
  return absl::OkStatus();
}

absl::Status ValidateBmmInputs(const at::Tensor& self, const at::Tensor& mat2) {
  TT_RET_CHECK(self.dim() == 3, error::kInvalidArgument)
      << "expected the first argument to be a 3D tensor (batch of matrices), "
         "got "
      << self.dim() << "D";

  TT_RET_CHECK(mat2.dim() == 3, error::kInvalidArgument)
      << "expected the second argument to be a 3D tensor (batch of matrices), "
         "got "
      << mat2.dim() << "D";

  TT_RET_CHECK(self.scalar_type() == mat2.scalar_type(),
               error::kInvalidArgument)
      << "expected self and mat2 to have the same dtype, got "
      << ToString(self.scalar_type()) << " vs " << ToString(mat2.scalar_type());

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

  // Must come after the dtype and shape checks. CUDA validates those criteria
  // before reaching the dispatch that reports the dtype as unimplemented.
  TT_RET_CHECK(
      self.numel() == 0 || mat2.numel() == 0 ||
          (self.scalar_type() != at::kBool && self.scalar_type() != at::kInt &&
           self.scalar_type() != at::kLong),
      error::kPythonNotImplementedError)
      << "not implemented for " << ToString(self.scalar_type());

  return absl::OkStatus();
}

template <typename ReturnType, typename DispatchFn>
ReturnType DispatchBmm(const at::Tensor& self, const at::Tensor& mat2,
                       at::ScalarType out_dtype, OpParamCacheKeys param_keys,
                       DispatchFn&& dispatch_fn) {
  TT_RETURN_IF_ERROR(ValidateBmmInputs(self, mat2));
  TT_ASSIGN_OR_RETURN(mlir::ElementType output_dtype_mlir,
                      ConvertTo<mlir::ElementType>(out_dtype));
  Dimensions output_dims_vec = {self.size(0), self.size(1), mat2.size(2)};

  const auto current_precision = GetAndAddPrecisionTo(param_keys);
  auto op_builder = [output_dtype_mlir,
                     current_precision](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, mat2_op] = inputs;
    return BuildBmmShlo(self_op, mat2_op, output_dtype_mlir, current_precision);
  };

  return dispatch_fn(
      std::move(op_builder),
      DispatchOpOptions<1>{.out_dtype = output_dtype_mlir,
                           .out_dims = std::move(output_dims_vec),
                           .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRef> Bmm(const at::Tensor& self,
                                    const at::Tensor& mat2,
                                    at::ScalarType out_dtype,
                                    OpParamCacheKeys param_keys) {
  return DispatchBmm<absl::StatusOr<DeviceBufferRef>>(
      self, mat2, out_dtype, std::move(param_keys),
      [&](auto op_builder, auto config) {
        return DispatchOp<2>(std::move(op_builder), {self, mat2},
                             std::move(config));
      });
}

absl::Status BmmOut(const at::Tensor& self, const at::Tensor& mat2,
                    at::ScalarType out_dtype, at::Tensor& out,
                    OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateBmmInputs(self, mat2));
  TT_RETURN_IF_ERROR(ValidateBmmOut(out, out_dtype));
  return DispatchBmm<absl::Status>(self, mat2, out_dtype, std::move(param_keys),
                                   [&](auto op_builder, auto config) {
                                     return DispatchOpOut<2>(
                                         std::move(op_builder), {self, mat2},
                                         out, std::move(config));
                                   });
}

}  // namespace

at::Tensor AtenBmmDtype(const at::Tensor& self, const at::Tensor& mat2,
                        at::ScalarType out_dtype) {
  TT_KERNEL(OpName::kBmmDtype, param_keys, (self, mat2, out_dtype), {
    TT_ASSIGN_OR_THROW(auto result_buffer,
                       Bmm(self, mat2, out_dtype, std::move(param_keys)));
    return MakeTensor(result_buffer);
  });
}

at::Tensor& AtenBmmDtypeOut(const at::Tensor& self, const at::Tensor& mat2,
                            at::ScalarType out_dtype, at::Tensor& out) {
  TT_KERNEL(OpName::kBmmDtypeOut, param_keys, (self, mat2, out_dtype, out), {
    TT_THROW_IF_ERROR(
        BmmOut(self, mat2, out_dtype, out, std::move(param_keys)));
    return out;
  });
}

at::Tensor& AtenBmmOut(const at::Tensor& self, const at::Tensor& mat2,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kBmmOut, param_keys, (self, mat2, out), {
    TT_THROW_IF_ERROR(
        BmmOut(self, mat2, self.scalar_type(), out, std::move(param_keys)));
    return out;
  });
}

}  // namespace torch_tpu
