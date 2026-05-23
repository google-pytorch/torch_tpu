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

#ifndef TORCH_TPU_OPS_BINARY_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_BINARY_ATEN_KERNELS_H_

#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/DeprecatedTypeProperties.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

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
};

enum class OutputOpMode {
  kOutOfPlace,
  kInPlace,
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

}  // namespace internal

// Handles output tensor post-processing for Binary and Ternary operations on
// TPU. Enforces that outputs for in-place aliases match the result buffer's
// shape, and resizes the output tensor for out-of-place execution.
absl::Status FinalizeOpOutput(at::Tensor& out, DeviceBufferRef result_buf,
                              OutputOpMode mode);

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
  TT_RET_CHECK(out.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "the out tensor is expected to be on tpu, got " << out.device().str();
  if (!opts.output_dtype_override) {
    TT_ASSIGN_OR_RETURN(auto output_dtype,
                        ConvertTo<mlir::ElementType>(out.scalar_type()));
    opts.output_dtype_override = output_dtype;
  }
  TT_ASSIGN_OR_RETURN(auto result_buf, internal::DispatchBinaryOp(
                                           tensor, other, std::move(op_builder),
                                           std::move(opts)));

  bool is_inplace = out.is_alias_of(tensor);
  if constexpr (std::is_same_v<OtherType, at::Tensor>) {
    is_inplace = is_inplace || out.is_alias_of(other);
  }

  OutputOpMode mode =
      is_inplace ? OutputOpMode::kInPlace : OutputOpMode::kOutOfPlace;
  return FinalizeOpOutput(out, std::move(result_buf), mode);
}

// Performs output tensor post-processing for Ternary operations on TPU.
// Validates that `out` is on the TPU, delegates dispatching to
// `DispatchTernaryOp`, checks for in-place aliasing, and finalizes the output
// tensor using `FinalizeOpOutput`.
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

#endif  // TORCH_TPU_OPS_BINARY_ATEN_KERNELS_H_
