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

#include "csrc/ops/linalg/qr/linalg_qr_kernels.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/linalg/qr/qr_lib.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

absl::Status ValidateDevices(const at::Tensor& self, const at::Tensor& out1,
                             const at::Tensor& out2) {
  TT_RET_CHECK(IsPrivateUse1Device(self) && IsPrivateUse1Device(out1) &&
                   IsPrivateUse1Device(out2),
               error::kInvalidArgument)
      << "expected all tensors to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << ToString(self.device()) << ", " << ToString(out1.device())
      << ", and " << ToString(out2.device());
  return absl::OkStatus();
}

absl::Status ValidateSupportedDtype(const at::Tensor& self) {
  const at::ScalarType dtype = self.scalar_type();
  const bool is_supported = dtype == at::kFloat || dtype == at::kDouble ||
                            dtype == at::kComplexFloat ||
                            dtype == at::kComplexDouble;
  TT_RET_CHECK(is_supported, error::kPythonNotImplementedError)
      << "expected the input dtype to be float32, float64, complex64, or "
         "complex128, got "
      << ToString(dtype);
  return absl::OkStatus();
}

absl::Status ValidateGeqrfInputs(const at::Tensor& self) {
  TT_RET_CHECK(self.dim() >= 2, error::kInvalidArgument)
      << "expected input to have at least 2 dimensions, got " << self.dim();
  return ValidateSupportedDtype(self);
}

template <typename ReturnType, typename DispatchFn>
ReturnType DispatchGeqrf(const at::Tensor& self,
                         std::array<mlir::ElementType, 2> out_dtypes,
                         OpParamCacheKeys param_keys,
                         DispatchFn&& dispatch_fn) {
  const int64_t m = self.size(self.dim() - 2);
  const int64_t n = self.size(self.dim() - 1);
  Dimensions tau_dims = CopyIntVector(self.sizes().slice(0, self.dim() - 2));
  tau_dims.push_back(std::min(m, n));

  auto op_builder =
      [](mlir::MlirOp self_op) -> absl::StatusOr<MlirOpResults<2>> {
    return BuildGeqrfShlo(self_op);
  };

  return dispatch_fn(
      std::move(op_builder),
      DispatchOpOptions<2>{.out_dtypes = out_dtypes,
                           .out_dims_list = {self.sizes(), std::move(tau_dims)},
                           .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRefArray<2>> Geqrf(const at::Tensor& self,
                                              OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateGeqrfInputs(self));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  return DispatchGeqrf<absl::StatusOr<DeviceBufferRefArray<2>>>(
      self, {out_dtype, out_dtype}, std::move(param_keys),
      [&self](auto op_builder, DispatchOpOptions<2> options) {
        return DispatchOp<1, 2>(std::move(op_builder), {self},
                                std::move(options));
      });
}

absl::Status GeqrfOut(const at::Tensor& self, OpParamCacheKeys param_keys,
                      at::Tensor& a, at::Tensor& tau) {
  TT_RETURN_IF_ERROR(ValidateDevices(self, a, tau));
  TT_RETURN_IF_ERROR(ValidateGeqrfInputs(self));
  TT_RETURN_IF_ERROR(
      ValidateOutDtype(a, self.scalar_type(), /*allow_out_dtype_cast=*/true));
  TT_RETURN_IF_ERROR(
      ValidateOutDtype(tau, self.scalar_type(), /*allow_out_dtype_cast=*/true));

  TT_ASSIGN_OR_RETURN(const mlir::ElementType a_out_dtype,
                      ConvertTo<mlir::ElementType>(a.scalar_type()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType tau_out_dtype,
                      ConvertTo<mlir::ElementType>(tau.scalar_type()));

  return DispatchGeqrf<absl::Status>(
      self, {a_out_dtype, tau_out_dtype}, std::move(param_keys),
      [&self, &a, &tau](auto op_builder, DispatchOpOptions<2> options) {
        return DispatchOpOut<1, 2>(std::move(op_builder), self, {a, tau},
                                   std::move(options));
      });
}

struct QrDims {
  Dimensions q_dims;
  Dimensions r_dims;
};

absl::StatusOr<QrDims> ComputeQrDims(const at::Tensor& self,
                                     std::string_view mode) {
  const int64_t m = self.size(self.dim() - 2);
  const int64_t n = self.size(self.dim() - 1);
  const int64_t k = std::min(m, n);

  Dimensions q_dims = CopyIntVector(self.sizes().slice(0, self.dim() - 2));
  Dimensions r_dims = CopyIntVector(self.sizes().slice(0, self.dim() - 2));
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
    q_dims = {0};
    r_dims.push_back(k);
    r_dims.push_back(n);
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "expected mode to be one of 'reduced', 'complete', or 'r', got '"
           << mode << "'";
  }

  return QrDims{std::move(q_dims), std::move(r_dims)};
}

absl::Status QrOut(const at::Tensor& self, std::string_view mode,
                   OpParamCacheKeys param_keys, at::Tensor& q, at::Tensor& r) {
  TT_RETURN_IF_ERROR(ValidateDevices(self, q, r));
  TT_RET_CHECK(self.dim() >= 2, error::kInvalidArgument)
      << "expected input tensor to have at least 2 dimensions, got "
      << self.dim();
  TT_RETURN_IF_ERROR(ValidateSupportedDtype(self));
  TT_ASSIGN_OR_RETURN(const QrDims qr_dims, ComputeQrDims(self, mode));
  TT_RETURN_IF_ERROR(
      ValidateOutDtype(q, self.scalar_type(), /*allow_out_dtype_cast=*/false));
  TT_RETURN_IF_ERROR(
      ValidateOutDtype(r, self.scalar_type(), /*allow_out_dtype_cast=*/false));

  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  auto op_builder =
      [mode = std::string(mode)](
          mlir::MlirOp self_op) -> absl::StatusOr<MlirOpResults<2>> {
    return BuildQrShlo(self_op, mode);
  };

  return DispatchOpOut<1, 2>(std::move(op_builder), self, {q, r},
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
    TT_THROW_IF_ERROR(GeqrfOut(self, std::move(param_keys), a, tau));
    return {a, tau};
  });
}

std::tuple<at::Tensor&, at::Tensor&> AtenLinalgQrOut(const at::Tensor& self,
                                                     std::string_view mode,
                                                     at::Tensor& q,
                                                     at::Tensor& r) {
  TT_KERNEL(OpName::kLinalgQrOut, param_keys, (self, mode, q, r), {
    TT_THROW_IF_ERROR(QrOut(self, mode, std::move(param_keys), q, r));
    return {q, r};
  });
}

}  // namespace torch_tpu
