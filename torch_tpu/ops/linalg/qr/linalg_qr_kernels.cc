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

#include "torch_tpu/ops/linalg/qr/linalg_qr_kernels.h"

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/util/string_view.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/linalg/qr/qr_lib.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

constexpr mlir::ElementType kCastDtypeForIntegerInput = mlir::ElementType::F64;

absl::StatusOr<mlir::ElementType> GetOutDtype(const at::Tensor& self) {
  const bool input_is_integer = IsInteger(self);
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  if (input_is_integer) {
    out_dtype = kCastDtypeForIntegerInput;
  }
  return out_dtype;
}

mlir::MlirOp MaybeCastToFloat(mlir::MlirOp self_op, bool input_is_integer) {
  return input_is_integer ? mlir::stablehlo::ConvertElementType(
                                self_op, kCastDtypeForIntegerInput)
                          : self_op;
}

absl::StatusOr<DeviceBufferRefArray<2>> Geqrf(const at::Tensor& self,
                                              OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype, GetOutDtype(self));

  // Capture metadata properties (like 'input_is_integer') by copy instead of
  // capturing the 'at::Tensor self' handle. This prevents locking 'self' and
  // its entire autograd graph in memory, avoiding high peak HBM consumption.
  const bool input_is_integer = IsInteger(self);

  TT_RET_CHECK(self.dim() >= 2, error::kInvalidArgument)
      << "expected input to have at least 2 dimensions, got " << self.dim();
  const int64_t m = self.size(self.dim() - 2);
  const int64_t n = self.size(self.dim() - 1);
  Dimensions tau_dims(self.sizes().begin(), self.sizes().end() - 2);
  tau_dims.push_back(std::min(m, n));

  auto op_builder =
      [input_is_integer](
          mlir::MlirOp self_op) -> absl::StatusOr<MlirOpResults<2>> {
    const mlir::MlirOp cast_self_op =
        MaybeCastToFloat(self_op, input_is_integer);
    TT_ASSIGN_OR_RETURN(const MlirOpResults<2> result_ops,
                        BuildGeqrfShlo(cast_self_op));
    return result_ops;
  };

  return DispatchOp<1, 2>(std::move(op_builder), {self},
                          {.out_dtypes = {out_dtype, out_dtype},
                           .out_dims_list = {self.sizes(), tau_dims},
                           .op_param_cache_keys = std::move(param_keys)});
}

struct QrDims {
  Dimensions q_dims;
  Dimensions r_dims;
};

absl::StatusOr<QrDims> ComputeQrDims(const at::Tensor& self,
                                     c10::string_view mode) {
  TT_RET_CHECK(self.dim() >= 2, error::kInvalidArgument)
      << "expected input tensor to have at least 2 dimensions, got "
      << self.dim();

  const int64_t m = self.size(self.dim() - 2);
  const int64_t n = self.size(self.dim() - 1);
  const int64_t k = std::min(m, n);

  Dimensions q_dims(self.sizes().begin(), self.sizes().end() - 2);
  Dimensions r_dims(self.sizes().begin(), self.sizes().end() - 2);
  if (mode == "reduced") {
    q_dims.push_back(m);
    q_dims.push_back(k);
    r_dims.push_back(k);
    r_dims.push_back(n);
  } else if (mode == "complete") {
    q_dims.push_back(m);
    q_dims.push_back(m);
    r_dims.push_back(m);
    r_dims.push_back(n);
  } else if (mode == "r") {
    q_dims.push_back(0);
    q_dims.push_back(0);
    r_dims.push_back(k);
    r_dims.push_back(n);
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "expected mode to be one of 'reduced', 'complete', or 'r', got "
           << mode;
  }

  return QrDims{std::move(q_dims), std::move(r_dims)};
}

absl::StatusOr<DeviceBufferRefArray<2>> Qr(const at::Tensor& self,
                                           c10::string_view mode,
                                           OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype, GetOutDtype(self));

  auto op_builder =
      [mode = std::string(mode)](
          mlir::MlirOp self_op) -> absl::StatusOr<MlirOpResults<2>> {
    TT_ASSIGN_OR_RETURN(const MlirOpResults<2> result_ops,
                        BuildQrShlo(self_op, mode));
    return result_ops;
  };

  TT_ASSIGN_OR_RETURN(const QrDims qr_dims, ComputeQrDims(self, mode));
  return DispatchOp<1, 2>(std::move(op_builder), {self},
                          {.out_dtypes = {out_dtype, out_dtype},
                           .out_dims_list = {qr_dims.q_dims, qr_dims.r_dims},
                           .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> AtenGeqrf(const at::Tensor& self) {
  TT_KERNEL(OpName::kGeqrf, param_keys, (self), {
    TT_ASSIGN_OR_THROW(const DeviceBufferRefArray<2> result_buffers,
                       Geqrf(self, std::move(param_keys)));
    return {MakeTensor(result_buffers[0]), MakeTensor(result_buffers[1])};
  });
}

std::tuple<at::Tensor&, at::Tensor&> AtenGeqrfA(const at::Tensor& self,
                                                at::Tensor& a,
                                                at::Tensor& tau) {
  TT_KERNEL(OpName::kGeqrfA, param_keys, (self, a, tau), {
    TT_ASSIGN_OR_THROW(const DeviceBufferRefArray<2> result_buffers,
                       Geqrf(self, std::move(param_keys)));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(result_buffers[0], a));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(result_buffers[1], tau));
    return {a, tau};
  });
}

std::tuple<at::Tensor&, at::Tensor&> AtenLinalgQrOut(const at::Tensor& self,
                                                     c10::string_view mode,
                                                     at::Tensor& q,
                                                     at::Tensor& r) {
  TT_KERNEL(OpName::kLinalgQrOut, param_keys, (self, mode, q, r), {
    TT_ASSIGN_OR_THROW(const DeviceBufferRefArray<2> result_buffers,
                       Qr(self, mode, std::move(param_keys)));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(result_buffers[0], q));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(result_buffers[1], r));
    return {q, r};
  });
}

}  // namespace torch_tpu
