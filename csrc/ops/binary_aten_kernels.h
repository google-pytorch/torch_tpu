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

#ifndef TORCH_TPU_CSRC_OPS_BINARY_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_BINARY_ATEN_KERNELS_H_

#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/DeprecatedTypeProperties.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {

struct BinaryOpOptions {
  // The op name for dispatching. If omitted, use the op name from the active
  // TT_KERNEL() context.
  std::optional<OpName> op_name = std::nullopt;
  bool reverse_operands = false;
  bool force_float_inputs =
      false;  // Force inputs to be floats, by casting if necessary.
  std::optional<mlir::ElementType> output_dtype_override = std::nullopt;
  OpParamCacheKeys op_param_cache_keys;
  OpSplitMode split_mode = OpSplitMode::kNone;
  Indices donated_indices = {};
};

struct TernaryOpOptions {
  // The op name for dispatching. If omitted, use the op name from the active
  // TT_KERNEL() context.
  std::optional<OpName> op_name = std::nullopt;
  bool force_float_inputs =
      false;  // Force inputs to be floats, by casting if necessary.
  std::optional<mlir::ElementType> output_dtype_override = std::nullopt;
  OpParamCacheKeys op_param_cache_keys;
  OpSplitMode split_mode = OpSplitMode::kNone;
  Indices donated_indices = {};
};

namespace internal {

absl::StatusOr<DeviceBufferRef> DispatchBinaryOp(const at::Tensor& self,
                                                 const at::Scalar& other,
                                                 MlirBinaryOpBuilder op_builder,
                                                 BinaryOpOptions opts);

absl::StatusOr<DeviceBufferRef> DispatchBinaryOp(
    const at::Tensor& self, const at::Tensor& other,
    MlirBinaryOpBuilder bin_op_builder, BinaryOpOptions opts);

absl::StatusOr<DeviceBufferRef> DispatchTernaryOp(
    const at::Tensor& self, const at::Tensor& other, const at::Tensor& third,
    MlirTernaryOpBuilder ternary_op_builder, TernaryOpOptions opts);

absl::Status DispatchBinaryOpOut(const at::Tensor& self,
                                 const at::Tensor& other, at::Tensor& out,
                                 MlirBinaryOpBuilder bin_op_builder,
                                 BinaryOpOptions opts);

absl::Status DispatchBinaryOpOut(const at::Tensor& self,
                                 const at::Scalar& other, at::Tensor& out,
                                 MlirBinaryOpBuilder bin_op_builder,
                                 BinaryOpOptions opts);

}  // namespace internal

template <typename OtherType>
absl::StatusOr<at::Tensor> BinaryOp(const at::Tensor& tensor,
                                    const OtherType& other,
                                    MlirBinaryOpBuilder op_builder,
                                    BinaryOpOptions opts) {
  TT_ASSIGN_OR_RETURN(auto result_buf, internal::DispatchBinaryOp(
                                           tensor, other, std::move(op_builder),
                                           std::move(opts)));
  return MakeTensor(std::move(result_buf));
}

template <typename OtherType>
absl::Status BinaryOpOut(const at::Tensor& tensor, const OtherType& other,
                         at::Tensor& out, MlirBinaryOpBuilder op_builder,
                         BinaryOpOptions opts) {
  return internal::DispatchBinaryOpOut(tensor, other, out,
                                       std::move(op_builder), std::move(opts));
}

// Performs output tensor processing for Ternary operations on TPU.
// Validates that `out` is on the TPU, enforces shape constraints for in-place
// aliasing, and dispatches via DispatchOpOut.
absl::Status TernaryOpOut(const at::Tensor& self, const at::Tensor& other,
                          const at::Tensor& third, at::Tensor& out,
                          MlirTernaryOpBuilder op_builder,
                          TernaryOpOptions opts);

// NOLINTBEGIN
// clang-format off
// go/keep-sorted start ignore_prefixes=at::Tensor,at::Tensor& newline_separated=yes
// clang-format on
// NOLINTEND
at::Tensor& AtenAddOut(const at::Tensor& self, const at::Tensor& other,
                       const at::Scalar& alpha, at::Tensor& out);

at::Tensor& AtenAddReluOut(const at::Tensor& self, const at::Tensor& other,
                           const at::Scalar& alpha, at::Tensor& out);

at::Tensor AtenAddReluScalar(const at::Tensor& self, const at::Scalar& other,
                             const at::Scalar& alpha);

at::Tensor AtenAddReluTensor(const at::Tensor& self, const at::Tensor& other,
                             const at::Scalar& alpha);

at::Tensor& AtenAddRelu_Scalar(at::Tensor& self, const at::Scalar& other,
                               const at::Scalar& alpha);

at::Tensor& AtenAddRelu_Tensor(at::Tensor& self, const at::Tensor& other,
                               const at::Scalar& alpha);

at::Tensor& AtenAtan2Out(const at::Tensor& x, const at::Tensor& y,
                         at::Tensor& out);

at::Tensor& AtenBitwiseAndTensorOut(const at::Tensor& self,
                                    const at::Tensor& other, at::Tensor& out);

at::Tensor& AtenBitwiseLeftShiftTensorOut(const at::Tensor& self,
                                          const at::Tensor& other,
                                          at::Tensor& out);

at::Tensor& AtenBitwiseOrTensorOut(const at::Tensor& self,
                                   const at::Tensor& other, at::Tensor& out);

at::Tensor& AtenBitwiseRightShiftTensorOut(const at::Tensor& self,
                                           const at::Tensor& other,
                                           at::Tensor& out);

at::Tensor& AtenBitwiseXorTensorOut(const at::Tensor& self,
                                    const at::Tensor& other, at::Tensor& out);

at::Tensor& AtenComplexOut(const at::Tensor& real, const at::Tensor& imag,
                           at::Tensor& out);

at::Tensor& AtenDivOut(const at::Tensor& self, const at::Tensor& other,
                       at::Tensor& out);

at::Tensor& AtenDivOutMode(const at::Tensor& self, const at::Tensor& other,
                           std::optional<std::string_view> mode,
                           at::Tensor& out);

at::Tensor& AtenEqScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out);

at::Tensor& AtenEqTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out);

at::Tensor AtenFloorDivide(const at::Tensor& self, const at::Tensor& other);

at::Tensor& AtenFloorDivideOut(const at::Tensor& self, const at::Tensor& other,
                               at::Tensor& out);

at::Tensor& AtenFloorDivide_Tensor(at::Tensor& self, const at::Tensor& other);

at::Tensor& AtenFmodTensorOut(const at::Tensor& self, const at::Tensor& other,
                              at::Tensor& out);

at::Tensor& AtenGeScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out);

at::Tensor& AtenGeTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out);

at::Tensor& AtenGtScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out);

at::Tensor& AtenGtTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out);

at::Tensor& AtenIlshiftScalar(at::Tensor& self, const at::Scalar& other);

at::Tensor& AtenIlshiftTensor(at::Tensor& self, const at::Tensor& other);

at::Tensor& AtenIrshiftScalar(at::Tensor& self, const at::Scalar& other);

at::Tensor& AtenIrshiftTensor(at::Tensor& self, const at::Tensor& other);

at::Tensor& AtenLdexpOut(const at::Tensor& self, const at::Tensor& other,
                         at::Tensor& out);

at::Tensor AtenLdexpTensor(const at::Tensor& self, const at::Tensor& other);

at::Tensor& AtenLdexp_(at::Tensor& self, const at::Tensor& other);

at::Tensor& AtenLeScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out);

at::Tensor& AtenLeTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out);

at::Tensor AtenLshiftScalar(const at::Tensor& self, const at::Scalar& other);

at::Tensor AtenLshiftTensor(const at::Tensor& self, const at::Tensor& other);

at::Tensor& AtenLtScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out);

at::Tensor& AtenLtTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out);

at::Tensor& AtenMaximumOut(const at::Tensor& self, const at::Tensor& other,
                           at::Tensor& out);

at::Tensor& AtenMinimumOut(const at::Tensor& self, const at::Tensor& other,
                           at::Tensor& out);

at::Tensor& AtenMulOut(const at::Tensor& self, const at::Tensor& other,
                       at::Tensor& out);

at::Tensor& AtenNeScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out);

at::Tensor& AtenNeTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out);

at::Tensor& AtenPolarOut(const at::Tensor& abs, const at::Tensor& angle,
                         at::Tensor& out);

at::Tensor& AtenPowScalarOut(const at::Scalar& self, const at::Tensor& exponent,
                             at::Tensor& out);

at::Tensor& AtenPowTensorScalarOut(const at::Tensor& self,
                                   const at::Scalar& exponent, at::Tensor& out);

at::Tensor& AtenPowTensorTensorOut(const at::Tensor& self,
                                   const at::Tensor& exponent, at::Tensor& out);

at::Tensor AtenRemainderScalarTensor(const at::Scalar& self,
                                     const at::Tensor& other);

at::Tensor& AtenRemainderTensorOut(const at::Tensor& self,
                                   const at::Tensor& other, at::Tensor& out);

at::Tensor AtenRshiftScalar(const at::Tensor& self, const at::Scalar& other);

at::Tensor AtenRshiftTensor(const at::Tensor& self, const at::Tensor& other);

at::Tensor AtenRsubTensor(const at::Tensor& self, const at::Tensor& other,
                          const at::Scalar& alpha);

at::Tensor& AtenSubOut(const at::Tensor& self, const at::Tensor& other,
                       const at::Scalar& alpha, at::Tensor& out);
// go/keep-sorted end

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_BINARY_ATEN_KERNELS_H_
