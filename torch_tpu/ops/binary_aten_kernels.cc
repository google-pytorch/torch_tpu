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

#include "torch_tpu/ops/binary_aten_kernels.h"

#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/BinaryOps.h"
#include "ATen/ops/promote_types.h"
#include "ATen/ops/result_type.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace internal {

absl::StatusOr<DeviceBufferRef> DispatchBinaryOp(const at::Tensor& self,
                                                 const at::Scalar& other,
                                                 OpName op_name,
                                                 MlirBinaryOpBuilder op_builder,
                                                 BinaryOpOptions opts) {
  at::ScalarType result_type = at::result_type(self, other);
  TT_ASSIGN_OR_RETURN(at::Tensor other_tensor, MakeTensor(other, result_type));
  return DispatchBinaryOp(self, other_tensor, op_name, std::move(op_builder),
                          std::move(opts));
}

absl::StatusOr<DeviceBufferRef> DispatchBinaryOp(
    const at::Tensor& self, const at::Tensor& other, OpName op_name,
    MlirBinaryOpBuilder bin_op_builder, BinaryOpOptions opts) {
  if (opts.reverse_operands) {
    opts.reverse_operands = false;
    return DispatchBinaryOp(other, self, op_name, std::move(bin_op_builder),
                            std::move(opts));
  }

  TT_ASSIGN_OR_RETURN(Dimensions output_dims, InferSize(self, other));

  at::ScalarType promoted_scalar_type = at::result_type(self, other);
  if (opts.force_float_inputs) {
    if (c10::isIntegralType(promoted_scalar_type, /*includeBool=*/true)) {
      promoted_scalar_type = c10::get_default_dtype_as_scalartype();
    }
  }
  TT_ASSIGN_OR_RETURN(mlir::ElementType computation_dtype,
                      ToElementType(promoted_scalar_type));
  mlir::ElementType output_dtype = opts.output_dtype_override
                                       ? *opts.output_dtype_override
                                       : computation_dtype;

  auto op_builder = [bin_op_builder = std::move(bin_op_builder)](
                        FixedSizeSpan<mlir::MlirOp, 2> inputs) {
    auto& [self, other] = inputs;
    return bin_op_builder(self, other);
  };
  return DispatchOp<2>(
      op_name, std::move(op_builder), {self, other},
      {.out_dtype = output_dtype,
       .out_dims = output_dims,
       .computation_dtype = computation_dtype,
       .op_param_cache_keys = std::move(opts.op_param_cache_keys),
       .split_mode = opts.split_mode});
}

}  // namespace internal

namespace {

absl::StatusOr<MlirBinaryOpBuilder> AtenAddHelper(const at::Tensor& self,
                                                  const at::Tensor& other,
                                                  const at::Scalar& alpha) {
  bool alpha_is_one = false;
  if (alpha.isIntegral(/*include_bool=*/false)) {
    alpha_is_one = (alpha.toLong() == 1);
  } else if (alpha.isFloatingPoint()) {
    alpha_is_one = (alpha.toDouble() == 1.0);
  } else if (alpha.isBoolean()) {
    alpha_is_one = (alpha.toBool() == true);
  } else {
    // TODO: add support to complex alpha on TPU.
    TT_ASSIGN_OR_RETURN(const auto alpha_element_type,
                        internal::ToElementType(alpha.type()));
    return TT_ERROR(error::kUnimplemented)
           << ToString(alpha_element_type)
           << " alpha value is not yet supported";
  }

  if (alpha_is_one) {
    return BuildAddShlo;
  }

  auto op_builder = [alpha](
                        mlir::MlirOp self_op,
                        mlir::MlirOp other_op) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(auto alpha_op, MakeConstantLike(other_op, alpha));
    const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other_op);
    auto other_element_type = other_type.getElementType();
    alpha_op = stablehlo::ConvertElementType(alpha_op, other_element_type);
    TT_ASSIGN_OR_RETURN(auto mul_op, BuildMulShlo(other_op, alpha_op));

    return BuildAddShlo(self_op, mul_op);
  };

  return std::move(op_builder);
}

struct DivOpOptions {
  MlirBinaryOpBuilder op_builder;
  OpParamCacheKeys op_param_cache_keys;
  std::optional<mlir::ElementType> output_dtype_override;
};

template <typename OtherType>
absl::StatusOr<DivOpOptions> GetDivOpOptionsNoMode(const at::Tensor& self,
                                                   const OtherType& other) {
  ABSL_VLOG(1) << "[GetDivOpOptionsNoMode]";
  // Capture the default dtype at **dispatch** time, not **compile** time,
  // and convert to XLA and MLIR types.
  at::ScalarType default_aten_type = c10::get_default_dtype_as_scalartype();
  TT_ASSIGN_OR_RETURN(auto default_dtype,
                      ConvertTo<mlir::ElementType>(default_aten_type));

  auto op_builder = [default_dtype](
                        mlir::MlirOp self_op,
                        mlir::MlirOp other_op) -> absl::StatusOr<mlir::MlirOp> {
    // Convert integer and booleans to torch's default scalar type.
    TT_ASSIGN_OR_RETURN((auto [promoted_self_op, promoted_other_op]),
                        ConvertIfIntegers(self_op, other_op, default_dtype));
    return BuildDivShlo(promoted_self_op, promoted_other_op);
  };

  // Integer and booleans will be converted to torch's default scalar type.
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true) &&
      c10::isIntegralType(GetScalarType(other), /*includeBool=*/true)) {
    return DivOpOptions{.op_builder = std::move(op_builder),
                        .op_param_cache_keys = OpParamCacheKeys::Empty(),
                        .output_dtype_override = default_dtype};
  }

  return DivOpOptions{.op_builder = std::move(op_builder),
                      .op_param_cache_keys = OpParamCacheKeys::Empty(),
                      .output_dtype_override = std::nullopt};
}

absl::StatusOr<DivOpOptions> GetDivOpOptionsTruncMode() {
  ABSL_VLOG(1) << "[GetDivOpOptionsTruncMode]";
  TT_ASSIGN_OR_RETURN(auto param_keys,
                      *OpParamCacheKeysBuilder().SetParam("mode", "trunc"));
  auto op_builder = [](mlir::MlirOp self_op,
                       mlir::MlirOp other_op) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp div_op, BuildDivShlo(self_op, other_op));

    // Rounds the results of the division towards zero. Equivalent to
    // C-style integer division.
    const mlir::RankedTensorType div_type = GetTensorTypeOrDie(div_op);
    // For integer division, the result is already an integer.
    if (div_type.getElementType().isInteger()) {
      return div_op;
    }

    // For floating-point division, we floor the absolute value of the result
    // and multiply by the sign of the original result to achieve truncation
    // towards zero.
    auto abs = stablehlo::Abs(div_op);
    auto floor = stablehlo::Floor(abs);
    auto sign = stablehlo::Sign(div_op);
    return stablehlo::Mul(floor, sign);
  };
  return DivOpOptions{std::move(op_builder), std::move(param_keys),
                      std::nullopt};
}

absl::StatusOr<DivOpOptions> GetDivOpOptionsFloorMode() {
  ABSL_VLOG(1) << "[GetDivOpOptionsFloorMode]";
  TT_ASSIGN_OR_RETURN(auto param_keys,
                      *OpParamCacheKeysBuilder().SetParam("mode", "floor"));
  auto op_builder = [](mlir::MlirOp self_op,
                       mlir::MlirOp other_op) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp div_op, BuildDivShlo(self_op, other_op));

    // Rounds the results of the division down. Equivalent to floor division
    // in Python (the // operator) and NumPy’s np.floor_divide.
    const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
    const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other_op);
    if (!self_type.getElementType().isInteger() &&
        !other_type.getElementType().isInteger()) {
      TT_ASSIGN_OR_RETURN(auto floor_div_op, BuildFloorShlo(div_op));
      return floor_div_op;
    }

    // Performing negative numerator / denominator is tricky.
    // For integer types, BuildFloorDivShlo handles this correctly.
    TT_ASSIGN_OR_RETURN(mlir::MlirOp floor_div_op,
                        BuildFloorDivShlo(self_op, other_op));
    return floor_div_op;
  };
  return DivOpOptions{std::move(op_builder), std::move(param_keys),
                      std::nullopt};
}

absl::Status CheckBitwiseOpsInputs(const at::Tensor& self,
                                   const at::Tensor& other) {
  TT_RET_CHECK(IsIntegral(self) || IsBool(self), error::kInvalidArgument)
      << "bitwise_* only supports boolean and integral types, "
      << "got self: " << ToString(self.scalar_type());

  TT_RET_CHECK(IsIntegral(other) || IsBool(other), error::kInvalidArgument)
      << "bitwise_* only supports boolean and integral types, "
      << "got other: " << ToString(other.scalar_type());

  return absl::OkStatus();
}

absl::Status CheckBitwiseShiftInputs(const at::Tensor& self,
                                     const at::Tensor& other) {
  TT_RET_CHECK(IsIntegral(self), error::kInvalidArgument)
      << "bitwise_shift only supports integral types, "
      << "got self: " << ToString(self.scalar_type());

  TT_RET_CHECK(IsIntegral(other), error::kInvalidArgument)
      << "bitwise_shift only supports integral types, "
      << "got other: " << ToString(other.scalar_type());

  return absl::OkStatus();
}

absl::Status CheckAtan2Inputs(const at::Tensor& x, const at::Tensor& y) {
  TT_RET_CHECK(!IsComplex(x), error::kInvalidArgument)
      << "atan2 is not implemented for complex types, got x: "
      << ToString(x.scalar_type());

  TT_RET_CHECK(!IsComplex(y), error::kInvalidArgument)
      << "atan2 is not implemented for complex types, got y: "
      << ToString(y.scalar_type());

  return absl::OkStatus();
}

// Checks the dtypes of the inputs of the `remainder()` op.
//
// In this context, the template parameter `T` should be one of:
//
//   - at::Tensor
//   - at::Scalar
template <typename T>
absl::Status CheckRemainderInputs(const T& self, const at::Tensor& other) {
  at::ScalarType result_type = at::result_type(self, other);

  TT_RET_CHECK(!IsBool(result_type) && !IsComplex(result_type),
               error::kInvalidArgument)
      << "expected the dtype of the output (promoted inputs dtype) to be "
         "neither bool nor complex, got "
      << ToString(result_type);

  return absl::OkStatus();
}

absl::Status CheckPolarInputs(const at::Tensor& abs, const at::Tensor& angle) {
  TT_RET_CHECK(IsFloatOrDouble(abs), error::kInvalidArgument)
      << "expected the dtype of the first argument to be float32 or float64, "
         "got "
      << ToString(abs.scalar_type());

  TT_RET_CHECK(IsFloatOrDouble(angle), error::kInvalidArgument)
      << "expected the dtype of the second argument to be float32 or float64, "
         "got "
      << ToString(angle.scalar_type());

  return absl::OkStatus();
}

absl::Status CheckSubInputs(const at::Tensor& self, const at::Tensor& other) {
  TT_RET_CHECK(!IsBool(self), error::kInvalidArgument)
      << "the dtype of the first argument cannot be bool";

  TT_RET_CHECK(!IsBool(other), error::kInvalidArgument)
      << "the dtype of the second argument cannot be bool";

  return absl::OkStatus();
}

}  // namespace

// NOLINTBEGIN
// clang-format off
// go/keep-sorted start ignore_prefixes=at::Tensor,at::Tensor& newline_separated=yes
// clang-format on
// NOLINTEND
at::Tensor& AtenAddOut(const at::Tensor& self, const at::Tensor& other,
                       const at::Scalar& alpha, at::Tensor& out) {
  TT_KERNEL(OpName::kAddOut, param_keys, (self, other, alpha, out), {
    TT_ASSIGN_OR_THROW(auto op_builder, AtenAddHelper(self, other, alpha));
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kAddOut, self, other, out, std::move(op_builder),
                    {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

at::Tensor& AtenAtan2Out(const at::Tensor& x, const at::Tensor& y,
                         at::Tensor& out) {
  TT_KERNEL(OpName::kAtan2Out, _, (x, y, out), {
    TT_THROW_IF_ERROR(CheckAtan2Inputs(x, y));
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kAtan2Out, x, y, out, BuildAtan2Shlo,
                    {.force_float_inputs = true,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenBitwiseAndTensorOut(const at::Tensor& self,
                                    const at::Tensor& other, at::Tensor& out) {
  TT_KERNEL(OpName::kBitwiseAnd, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckBitwiseOpsInputs(self, other));
    TT_THROW_IF_ERROR(BinaryOpOut(
        OpName::kBitwiseAndOut, self, other, out, BuildBitwiseAndShlo,
        {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenBitwiseLeftShiftTensorOut(const at::Tensor& self,
                                          const at::Tensor& other,
                                          at::Tensor& out) {
  TT_KERNEL(OpName::kBitwiseLeftShiftTensorOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckBitwiseShiftInputs(self, other));
    at::Tensor new_self = self.to(out.scalar_type());
    at::Tensor new_other = other.to(out.scalar_type());
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kBitwiseLeftShiftTensorOut, new_self, new_other,
                    out, BuildBitwiseLeftShiftShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenBitwiseOrTensorOut(const at::Tensor& self,
                                   const at::Tensor& other, at::Tensor& out) {
  TT_KERNEL(OpName::kBitwiseOr, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckBitwiseOpsInputs(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kBitwiseOrOut, self, other, out, BuildBitwiseOrShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenBitwiseRightShiftTensorOut(const at::Tensor& self,
                                           const at::Tensor& other,
                                           at::Tensor& out) {
  TT_KERNEL(OpName::kBitwiseRightShiftTensorOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckBitwiseShiftInputs(self, other));
    at::Tensor new_self = self.to(out.scalar_type());
    at::Tensor new_other = other.to(out.scalar_type());
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kBitwiseRightShiftTensorOut, new_self, new_other,
                    out, BuildBitwiseRightShiftShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenBitwiseXorTensorOut(const at::Tensor& self,
                                    const at::Tensor& other, at::Tensor& out) {
  TT_KERNEL(OpName::kBitwiseXor, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckBitwiseOpsInputs(self, other));
    TT_THROW_IF_ERROR(BinaryOpOut(
        OpName::kBitwiseXorOut, self, other, out, BuildBitwiseXorShlo,
        {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenComplexOut(const at::Tensor& real, const at::Tensor& imag,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kComplexOut, _, (real, imag, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kComplexOut, real, imag, out, BuildComplexShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenDivOut(const at::Tensor& self, const at::Tensor& other,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kDivOut, _, (self, other, out), {
    TT_ASSIGN_OR_THROW(auto opts, GetDivOpOptionsNoMode(self, other));
    TT_THROW_IF_ERROR(BinaryOpOut(
        OpName::kDivOut, self, other, out, std::move(opts.op_builder),
        {.output_dtype_override = opts.output_dtype_override,
         .op_param_cache_keys = std::move(opts.op_param_cache_keys)}));
    return out;
  });
}

at::Tensor& AtenDivOutMode(const at::Tensor& self, const at::Tensor& other,
                           std::optional<std::string_view> mode,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kDivOut, _, (self, other, IgnoreInCacheKey(mode), out), {
    DivOpOptions opts{.op_builder = nullptr,
                      .op_param_cache_keys = OpParamCacheKeys::Empty(),
                      .output_dtype_override = std::nullopt};
    if (!mode.has_value()) {
      TT_ASSIGN_OR_THROW(opts, GetDivOpOptionsNoMode(self, other));
    } else if (*mode == "trunc") {
      TT_ASSIGN_OR_THROW(opts, GetDivOpOptionsTruncMode());
    } else if (*mode == "floor") {
      TT_ASSIGN_OR_THROW(opts, GetDivOpOptionsFloorMode());
    } else {
      TT_THROW_IF_ERROR(TT_ERROR(error::kInvalidArgument)
                        << "invalid mode for div: " << *mode);
    }
    TT_THROW_IF_ERROR(BinaryOpOut(
        OpName::kDivOut, self, other, out, std::move(opts.op_builder),
        {.output_dtype_override = opts.output_dtype_override,
         .op_param_cache_keys = std::move(opts.op_param_cache_keys)}));
    return out;
  });
}

at::Tensor& AtenEqScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kEq, _, (self, IgnoreInCacheKey(other), out), {
    TT_ASSIGN_OR_THROW(at::Tensor other_tensor,
                       MakeTensor(other, self.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kEqOut, self, other_tensor, out, BuildEqShlo,
                    {.output_dtype_override = mlir::ElementType::PRED,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenEqTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kEq, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kEqOut, self, other, out, BuildEqShlo,
                    {.output_dtype_override = mlir::ElementType::PRED,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenFloorDivide(const at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kFloorDivide, _, (self, other), {
    at::ScalarType promoted_scalar_type = at::result_type(self, other);
    at::Tensor out =
        MakeEmptyTensor(self.sizes(), promoted_scalar_type, self.device());
    AtenFloorDivideOut(self, other, out);
    return out;
  });
}

at::Tensor& AtenFloorDivideOut(const at::Tensor& self, const at::Tensor& other,
                               at::Tensor& out) {
  TT_KERNEL(OpName::kFloorDivide, _, (self, other, out), {
    TT_ASSIGN_OR_THROW(auto opts, GetDivOpOptionsFloorMode());
    TT_THROW_IF_ERROR(BinaryOpOut(
        OpName::kFloorDivide, self, other, out, std::move(opts.op_builder),
        {.output_dtype_override = opts.output_dtype_override,
         .op_param_cache_keys = std::move(opts.op_param_cache_keys)}));
    return out;
  });
}

at::Tensor& AtenFloorDivide_Tensor(at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kFloorDivide, _, (self, other), {
    AtenFloorDivideOut(self, other, self);
    return self;
  });
}

at::Tensor& AtenFmodTensorOut(const at::Tensor& self, const at::Tensor& other,
                              at::Tensor& out) {
  TT_KERNEL(OpName::kFmodTensorOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(BinaryOpOut(
        OpName::kFmodTensorOut, self, other, out, BuildFmodTensorShlo,
        {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenGeScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kGe, _, (self, IgnoreInCacheKey(other), out), {
    TT_ASSIGN_OR_THROW(at::Tensor other_tensor,
                       MakeTensor(other, self.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kGeOut, self, other_tensor, out, BuildGeShlo,
                    {.output_dtype_override = mlir::ElementType::PRED,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenGeTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kGe, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kGeOut, self, other, out, BuildGeShlo,
                    {.output_dtype_override = mlir::ElementType::PRED,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenGtScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kGt, _, (self, IgnoreInCacheKey(other), out), {
    TT_ASSIGN_OR_THROW(at::Tensor other_tensor,
                       MakeTensor(other, self.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kGtOut, self, other_tensor, out, BuildGtShlo,
                    {.output_dtype_override = mlir::ElementType::PRED,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenGtTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kGt, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kGtOut, self, other, out, BuildGtShlo,
                    {.output_dtype_override = mlir::ElementType::PRED,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenIlshiftScalar(at::Tensor& self, const at::Scalar& other) {
  TT_KERNEL(OpName::kIlshiftScalar, _, (self, IgnoreInCacheKey(other)), {
    TT_ASSIGN_OR_THROW(at::Tensor wrapped_scalar, MakeTensor(other));
    AtenBitwiseLeftShiftTensorOut(self, wrapped_scalar, self);
    return self;
  });
}

at::Tensor& AtenIlshiftTensor(at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kIlshiftTensor, _, (self, other), {
    AtenBitwiseLeftShiftTensorOut(self, other, self);
    return self;
  });
}

at::Tensor& AtenIrshiftScalar(at::Tensor& self, const at::Scalar& other) {
  TT_KERNEL(OpName::kIrshiftScalar, _, (self, IgnoreInCacheKey(other)), {
    TT_ASSIGN_OR_THROW(at::Tensor wrapped_scalar, MakeTensor(other));
    AtenBitwiseRightShiftTensorOut(self, wrapped_scalar, self);
    return self;
  });
}

at::Tensor& AtenIrshiftTensor(at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kIrshiftTensor, _, (self, other), {
    AtenBitwiseRightShiftTensorOut(self, other, self);
    return self;
  });
}

at::Tensor& AtenLeScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kLe, _, (self, IgnoreInCacheKey(other), out), {
    TT_ASSIGN_OR_THROW(at::Tensor other_tensor,
                       MakeTensor(other, self.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kLeOut, self, other_tensor, out, BuildLeShlo,
                    {.output_dtype_override = mlir::ElementType::PRED,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenLeTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kLe, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kLeOut, self, other, out, BuildLeShlo,
                    {.output_dtype_override = mlir::ElementType::PRED,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenLshiftScalar(const at::Tensor& self, const at::Scalar& other) {
  TT_KERNEL(OpName::kLshiftScalar, _, (self, IgnoreInCacheKey(other)), {
    at::Tensor out =
        MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device());
    TT_ASSIGN_OR_THROW(at::Tensor wrapped_scalar, MakeTensor(other));
    AtenBitwiseLeftShiftTensorOut(self, wrapped_scalar, out);
    return out;
  });
}

at::Tensor AtenLshiftTensor(const at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kLshiftTensor, _, (self, other), {
    at::ScalarType promoted_scalar_type = at::result_type(self, other);
    at::Tensor out =
        MakeEmptyTensor(self.sizes(), promoted_scalar_type, self.device());
    AtenBitwiseLeftShiftTensorOut(self, other, out);
    return out;
  });
}

at::Tensor& AtenLtScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kLt, _, (self, IgnoreInCacheKey(other), out), {
    TT_ASSIGN_OR_THROW(at::Tensor other_tensor,
                       MakeTensor(other, self.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kLtOut, self, other_tensor, out, BuildLtShlo,
                    {.output_dtype_override = mlir::ElementType::PRED,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenLtTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kLt, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kLtOut, self, other, out, BuildLtShlo,
                    {.output_dtype_override = mlir::ElementType::PRED,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenMaximumOut(const at::Tensor& self, const at::Tensor& other,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kMaximum, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kMaximumOut, self, other, out, BuildMaximumShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenMinimumOut(const at::Tensor& self, const at::Tensor& other,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kMinimum, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kMinimumOut, self, other, out, BuildMinimumShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenMulOut(const at::Tensor& self, const at::Tensor& other,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kMulOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kMulOut, self, other, out, BuildMulShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenNeScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kNe, _, (self, IgnoreInCacheKey(other), out), {
    TT_ASSIGN_OR_THROW(at::Tensor other_tensor,
                       MakeTensor(other, self.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kNeOut, self, other_tensor, out, BuildNeShlo,
                    {.output_dtype_override = mlir::ElementType::PRED,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenNeTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kNe, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kNeOut, self, other, out, BuildNeShlo,
                    {.output_dtype_override = mlir::ElementType::PRED,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenPolarOut(const at::Tensor& abs, const at::Tensor& angle,
                         at::Tensor& out) {
  TT_KERNEL(OpName::kPolarOut, _, (abs, angle, out), {
    TT_THROW_IF_ERROR(CheckPolarInputs(abs, angle));
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kPolarOut, abs, angle, out, BuildPolarShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenPowScalarOut(const at::Scalar& self, const at::Tensor& exponent,
                             at::Tensor& out) {
  TT_KERNEL(OpName::kPow, _, (IgnoreInCacheKey(self), exponent, out), {
    TT_ASSIGN_OR_THROW(at::Tensor self_tensor,
                       MakeTensor(self, exponent.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kPowOut, self_tensor, exponent, out, BuildPowShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenPowTensorScalarOut(const at::Tensor& self,
                                   const at::Scalar& exponent,
                                   at::Tensor& out) {
  TT_KERNEL(OpName::kPow, _, (self, IgnoreInCacheKey(exponent), out), {
    TT_ASSIGN_OR_THROW(at::Tensor exponent_tensor,
                       MakeTensor(exponent, self.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kPowOut, self, exponent_tensor, out, BuildPowShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenPowTensorTensorOut(const at::Tensor& self,
                                   const at::Tensor& exponent,
                                   at::Tensor& out) {
  TT_KERNEL(OpName::kPow, _, (self, exponent, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kPowOut, self, exponent, out, BuildPowShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenRemainderScalarTensor(const at::Scalar& self,
                                     const at::Tensor& other) {
  TT_KERNEL(OpName::kRemainder, _, (IgnoreInCacheKey(self), other), {
    TT_THROW_IF_ERROR(CheckRemainderInputs(self, other));
    at::ScalarType promoted_scalar_type = at::result_type(self, other);
    at::Tensor out =
        MakeEmptyTensor(other.sizes(), promoted_scalar_type, other.device());
    TT_ASSIGN_OR_THROW(at::Tensor self_tensor,
                       MakeTensor(self, promoted_scalar_type));
    AtenRemainderTensorOut(self_tensor, other, out);
    return out;
  });
}

at::Tensor& AtenRemainderTensorOut(const at::Tensor& self,
                                   const at::Tensor& other, at::Tensor& out) {
  TT_KERNEL(OpName::kRemainder, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckRemainderInputs(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kRemainderOut, self, other, out, BuildRemainderShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenRshiftScalar(const at::Tensor& self, const at::Scalar& other) {
  TT_KERNEL(OpName::kRshiftScalar, _, (self, IgnoreInCacheKey(other)), {
    at::Tensor out =
        MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device());
    TT_ASSIGN_OR_THROW(at::Tensor wrapped_scalar, MakeTensor(other));
    AtenBitwiseRightShiftTensorOut(self, wrapped_scalar, out);
    return out;
  });
}

at::Tensor AtenRshiftTensor(const at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kRshiftTensor, _, (self, other), {
    at::ScalarType promoted_scalar_type = at::result_type(self, other);
    at::Tensor out =
        MakeEmptyTensor(self.sizes(), promoted_scalar_type, self.device());
    AtenBitwiseRightShiftTensorOut(self, other, out);
    return out;
  });
}

at::Tensor AtenRsubTensor(const at::Tensor& self, const at::Tensor& other,
                          const at::Scalar& alpha) {
  TT_KERNEL(OpName::kRsub, _, (self, other, IgnoreInCacheKey(alpha)),
            { return at::sub(other, self, alpha); });
}

at::Tensor& AtenSubOut(const at::Tensor& self, const at::Tensor& other,
                       const at::Scalar& alpha, at::Tensor& out) {
  TT_KERNEL(OpName::kSubOut, param_keys, (self, other, alpha, out), {
    TT_THROW_IF_ERROR(CheckSubInputs(self, other));
    TT_ASSIGN_OR_THROW(auto op_builder, AtenAddHelper(self, other, -alpha));
    TT_THROW_IF_ERROR(
        BinaryOpOut(OpName::kSubOut, self, other, out, std::move(op_builder),
                    {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

// go/keep-sorted end

}  // namespace torch_tpu
