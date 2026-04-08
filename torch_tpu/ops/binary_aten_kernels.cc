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

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "ATen/ScalarOps.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/div.h"
#include "ATen/ops/result_type.h"
#include "ATen/ops/sub.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/DeviceType.h"
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
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;
namespace chlo = mlir::chlo;

namespace internal {

absl::StatusOr<DeviceBufferRef> DispatchBinaryOp(const at::Tensor& self,
                                                 const at::Scalar& other,
                                                 MlirBinaryOpBuilder op_builder,
                                                 BinaryOpOptions opts) {
  // Lift the "other" scalar to a tensor, in either constant or variable mode.

  at::ScalarType result_type = at::result_type(self, other);
  TT_ASSIGN_OR_RETURN(at::Tensor other_tensor, MakeTensor(other, result_type));
  return DispatchBinaryOp(self, other_tensor, std::move(op_builder),
                          std::move(opts));
}

absl::StatusOr<DeviceBufferRef> DispatchBinaryOp(
    const at::Tensor& self, const at::Tensor& other,
    MlirBinaryOpBuilder bin_op_builder, BinaryOpOptions opts) {
  // Special case in which the LHS is a scalar allocated on the CPU.
  if (self.device().type() == c10::DeviceType::CPU && self.numel() == 1) {
    // MakeTensor uses a cache to deduplicate scalar tensors.
    TT_ASSIGN_OR_RETURN(at::Tensor self_tensor, MakeTensor(self.item()));
    return DispatchBinaryOp(self_tensor, other, std::move(bin_op_builder),
                            std::move(opts));
  }
  // Special case in which the RHS is a scalar allocated on the CPU.
  if (other.device().type() == c10::DeviceType::CPU && other.numel() == 1) {
    // MakeTensor uses a cache to deduplicate scalar tensors.
    TT_ASSIGN_OR_RETURN(at::Tensor other_tensor, MakeTensor(other.item()));
    return DispatchBinaryOp(self, other_tensor, std::move(bin_op_builder),
                            std::move(opts));
  }

  TT_ASSIGN_OR_RETURN(const Dimensions output_dims,
                      InferSize(self.sizes(), other.sizes()));

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
      std::move(op_builder), {self, other},
      {.op_name = opts.op_name,
       .out_dtype = output_dtype,
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
                      .op_param_cache_keys = OpParamCacheKeys::Empty()};
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
    // Inspired by
    // [ref](https://github.com/tensorflow/tensorflow/blob/a0d9adf50eb4dbc3b8de346b0c95acce288959c0/tensorflow/compiler/mlir/tf2xla/transforms/legalize_tf_patterns.td#L151-L186)
    // Here's an equivalent pseudo-code:
    // T z = x / y
    // return (z * y != x && (x < 0) != (y < 0)) ? z - 1 : z
    auto div = chlo::BroadcastDiv(self_op, other_op);
    auto mul = chlo::BroadcastMul(div, other_op);
    auto l_and =
        chlo::BroadcastCompare(mul, self_op, chlo::ComparisonDirection::NE);

    auto l_zeros =
        MakeScalarConstant(self_op.getBuilder(), 0, self_type.getElementType());

    auto l_cmp =
        chlo::BroadcastCompare(self_op, l_zeros, chlo::ComparisonDirection::LT);

    auto r_cmp = chlo::BroadcastCompare(other_op, l_zeros,
                                        chlo::ComparisonDirection::LT);

    auto r_and =
        chlo::BroadcastCompare(l_cmp, r_cmp, chlo::ComparisonDirection::NE);

    auto pred_select = chlo::BroadcastAnd(l_and, r_and);

    const mlir::RankedTensorType div_type = GetTensorTypeOrDie(div_op);
    auto ones =
        MakeScalarConstant(self_op.getBuilder(), 1, div_type.getElementType());
    auto true_select = chlo::BroadcastSub(div, ones);

    return stablehlo::Select(pred_select, true_select, div);
  };
  return DivOpOptions{std::move(op_builder), std::move(param_keys),
                      std::nullopt};
}

// Returns the appropriate division operator builder based on the provided mode.
//
// See torch.div documentation for the mode options:
// https://docs.pytorch.org/docs/stable/generated/torch.div.html
//
// Args:
//  self: The dividend tensor.
//  other: The divisor tensor or scalar.
//  mode: The division mode. Can be "trunc", "floor", or std::nullopt.
//
// Returns:
//  A DivOpOptions struct containing the operator builder, cache keys, and
//  output dtype override.
template <typename OtherType>
absl::StatusOr<DivOpOptions> GetDivOpBuilder(
    const at::Tensor& self, const OtherType& other,
    std::optional<std::string_view> mode = {}) {
  if (!mode.has_value()) {
    return GetDivOpOptionsNoMode(self, other);
  }
  if (mode.value() == "trunc") {
    return GetDivOpOptionsTruncMode();
  }
  if (mode.value() == "floor") {
    return GetDivOpOptionsFloorMode();
  }
  return TT_ERROR(error::kInvalidArgument)
         << "Unsupported mode for div: " << mode.value();
}

absl::Status CheckFloorDivideInputs(const at::Tensor& self,
                                    const at::Tensor& other) {
  TT_RET_CHECK(!IsComplex(self) && !IsBool(self), error::kInvalidArgument)
      << "expected dtype of the first argument to be neither complex nor "
         "bool, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(!IsComplex(other) && !IsBool(other), error::kInvalidArgument)
      << "expected dtype of the second argument to be neither complex nor "
         "bool, got "
      << ToString(other.scalar_type());

  return absl::OkStatus();
}

absl::Status CheckAtan2Inputs(const at::Tensor& x, const at::Tensor& y) {
  TT_RET_CHECK(!IsComplex(x), error::kInvalidArgument)
      << "expected the dtype of the first argument not to be complex, got "
      << ToString(x.scalar_type());

  TT_RET_CHECK(!IsComplex(y), error::kInvalidArgument)
      << "expected the dtype of the second argument not to be complex, got "
      << ToString(y.scalar_type());

  return absl::OkStatus();
}

absl::Status CheckBitwiseOpsInputs(const at::Tensor& self,
                                   const at::Tensor& other) {
  TT_RET_CHECK(!IsFloatingPoint(self) && !IsComplex(self),
               error::kInvalidArgument)
      << "expected the dtype of the first argument to be neither "
         "floating-point nor complex, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(!IsFloatingPoint(other) && !IsComplex(other),
               error::kInvalidArgument)
      << "expected the dtype of the second argument to be neither "
         "floating-point nor complex, got "
      << ToString(other.scalar_type());

  return absl::OkStatus();
}

absl::Status CheckBitwiseShiftInputs(const at::Tensor& self,
                                     const at::Tensor& other) {
  TT_RET_CHECK(IsInteger(self), error::kInvalidArgument)
      << "expected the dtype of the first argument to be integer, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(IsInteger(other), error::kInvalidArgument)
      << "expected the dtype of the second argument to be integer, got "
      << ToString(other.scalar_type());

  return absl::OkStatus();
}

absl::Status CheckComplexOutInputs(const at::Tensor& real,
                                   const at::Tensor& imag,
                                   const at::Tensor& out) {
  TT_RET_CHECK(IsFloatOrDouble(real), error::kInvalidArgument)
      << "expected the dtype of the first argument to be float32 or float64, "
         "got "
      << ToString(real.scalar_type());

  TT_RET_CHECK(IsFloatOrDouble(imag), error::kInvalidArgument)
      << "expected the dtype of the second argument to be float32 or float64, "
         "got "
      << ToString(imag.scalar_type());

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Errors when creating the complex32
                 // output tensor before.
      IsXlaSupportedComplex(out), error::kInvalidArgument)
      << "expected the dtype of the output to be one of the OpenXLA supported "
         "complex dtypes (either complex64 or complex128), got "
      << ToString(out.scalar_type());

  return absl::OkStatus();
}

// Checks the dtypes of the inputs of the following comparison ops:
//
//   - GE
//   - GT
//   - LE
//   - LT
//
// In this context, the template parameter `T` should be one of:
//
//   - at::Tensor
//   - at::Scalar
template <typename T>
absl::Status CheckComparisonOpsInputs(const at::Tensor& self, const T& other) {
  TT_RET_CHECK(!IsComplex(self), error::kInvalidArgument)
      << "expected the dtype of the first argument not to be complex, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(!IsComplex(other), error::kInvalidArgument)
      << "expected the dtype of the second argument not to be complex, got "
      << ToString(GetScalarType(other));

  return absl::OkStatus();
}

// The template parameters `T` and `U` should be one of:
//
//   - at::Tensor
//   - at::Scalar
template <typename T, typename U>
absl::Status CheckPowInputs(const T& self, const U& exponent) {
  // TODO: b/481396743 remove these checks once we start supporting bool dtype.

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch native devices supports boolean
                 // dtype.
      !IsBool(self), error::kInvalidArgument)
      << "expected the dtype of the first argument not to be boolean, got "
      << ToString(GetScalarType(self));

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch native devices supports boolean
                 // dtype.
      !IsBool(exponent), error::kInvalidArgument)
      << "expected the dtype of the second argument not to be boolean, got "
      << ToString(GetScalarType(exponent));

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
  bool alpha_is_one = false;
  if (alpha.isIntegral(/*include_bool=*/false)) {
    alpha_is_one = (alpha.toLong() == 1);
  } else if (alpha.isFloatingPoint()) {
    alpha_is_one = (alpha.toDouble() == 1.0);
  } else if (alpha.isBoolean()) {
    alpha_is_one = (alpha.toBool() == true);
  }

  auto promoted_alpha = PromoteScalar(alpha);

  auto op_builder = [alpha_is_one](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<MlirOpResults<1>> {
    auto [self_op, other_op, alpha_op] = inputs;
    if (alpha_is_one) {
      TT_ASSIGN_OR_RETURN(auto result, BuildAddShlo(self_op, other_op));
      return MlirOpResults<1>{result};
    }
    const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other_op);
    auto other_element_type = other_type.getElementType();
    auto converted_alpha =
        mlir::stablehlo::ConvertElementType(alpha_op, other_element_type);

    TT_ASSIGN_OR_RETURN(auto mul_op, BuildMulShlo(other_op, converted_alpha));
    TT_ASSIGN_OR_RETURN(auto result, BuildAddShlo(self_op, mul_op));
    return MlirOpResults<1>{result};
  };

  TT_KERNEL(OpName::kAddOut, param_keys, (self, other, promoted_alpha, out), {
    TT_THROW_IF_ERROR(param_keys.SetParam("alpha_is_one", alpha_is_one));
    TT_ASSIGN_OR_THROW(auto alpha_tensor, promoted_alpha.GetTensor());
    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));

    TT_ASSIGN_OR_THROW(
        auto result,
        (DispatchOp<3>(op_builder,
                       /*inputs=*/{self, other, alpha_tensor},
                       /*options=*/
                       {.out_dtype = output_dtype,
                        .out_dims = out.sizes(),
                        .op_param_cache_keys = std::move(param_keys)})));

    TT_THROW_IF_ERROR(AssignBufferToAtTensor(result, out));
    return out;
  });
}

at::Tensor& AtenAddReluOut(const at::Tensor& self, const at::Tensor& other,
                           const at::Scalar& alpha, at::Tensor& out) {
  const bool alpha_is_one = [&]() {
    if (alpha.isIntegral(/*include_bool=*/false)) {
      return alpha.toLong() == 1;
    } else if (alpha.isFloatingPoint()) {
      return alpha.toDouble() == 1.0;
    } else if (alpha.isBoolean()) {
      return alpha.toBool() == true;
    }
    return false;
  }();

  auto promoted_alpha = PromoteScalar(alpha);

  auto op_builder = [alpha_is_one](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<MlirOpResults<1>> {
    auto [self_op, other_op, alpha_op] = inputs;
    mlir::MlirOp add_op;
    if (alpha_is_one) {
      TT_ASSIGN_OR_RETURN(add_op, BuildAddShlo(self_op, other_op));
    } else {
      const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other_op);
      auto other_element_type = other_type.getElementType();
      auto alpha_converted =
          mlir::stablehlo::ConvertElementType(alpha_op, other_element_type);
      TT_ASSIGN_OR_RETURN(auto mul_op, BuildMulShlo(other_op, alpha_converted));
      TT_ASSIGN_OR_RETURN(add_op, BuildAddShlo(self_op, mul_op));
    }

    // Apply Relu: Max(0, add_op)
    TT_ASSIGN_OR_RETURN(auto zero_op, MakeConstantLike(add_op, at::Scalar(0)));
    auto relu_op = mlir::stablehlo::Max(add_op, zero_op);

    return MlirOpResults<1>{relu_op};
  };

  TT_KERNEL(
      OpName::kAddReluOut, param_keys, (self, other, promoted_alpha, out), {
        TT_THROW_IF_ERROR(param_keys.SetParam("alpha_is_one", alpha_is_one));
        TT_ASSIGN_OR_THROW(auto alpha_tensor, promoted_alpha.GetTensor());
        TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto result,
            (DispatchOp<3>(op_builder,
                           /*inputs=*/{self, other, alpha_tensor},
                           /*options=*/
                           {.out_dtype = output_dtype,
                            .out_dims = out.sizes(),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(result, out));
        return out;
      });
}

at::Tensor AtenAddReluScalar(const at::Tensor& self, const at::Scalar& other,
                             const at::Scalar& alpha) {
  TT_KERNEL(OpName::kAddReluScalar, _,
            (self, IgnoreInCacheKey(other, "Legacy usage"),
             IgnoreInCacheKey(alpha, "Legacy usage")),
            {
              at::ScalarType promoted_scalar_type =
                  at::result_type(self, other);
              at::Tensor out = MakeEmptyTensor(
                  self.sizes(), promoted_scalar_type, self.device());
              TT_ASSIGN_OR_THROW(auto wrapped_other, MakeTensor(other));
              AtenAddReluOut(self, wrapped_other, alpha, out);
              return out;
            });
}

at::Tensor AtenAddReluTensor(const at::Tensor& self, const at::Tensor& other,
                             const at::Scalar& alpha) {
  TT_KERNEL(OpName::kAddReluTensor, _,
            (self, other, IgnoreInCacheKey(alpha, "Legacy usage")), {
              at::ScalarType promoted_scalar_type =
                  at::result_type(self, other);
              TT_ASSIGN_OR_THROW(const Dimensions output_dims,
                                 InferSize(self.sizes(), other.sizes()));
              at::Tensor out = MakeEmptyTensor(
                  output_dims, promoted_scalar_type, self.device());
              AtenAddReluOut(self, other, alpha, out);
              return out;
            });
}

at::Tensor& AtenAddRelu_Scalar(at::Tensor& self, const at::Scalar& other,
                               const at::Scalar& alpha) {
  TT_KERNEL(OpName::kAddRelu_Scalar, _,
            (self, IgnoreInCacheKey(other, "Legacy usage"),
             IgnoreInCacheKey(alpha, "Legacy usage")),
            {
              TT_ASSIGN_OR_THROW(auto wrapped_other, MakeTensor(other));
              AtenAddReluOut(self, wrapped_other, alpha, self);
              return self;
            });
}

at::Tensor& AtenAddRelu_Tensor(at::Tensor& self, const at::Tensor& other,
                               const at::Scalar& alpha) {
  TT_KERNEL(OpName::kAddRelu_Tensor, _,
            (self, other, IgnoreInCacheKey(alpha, "Legacy usage")), {
              AtenAddReluOut(self, other, alpha, self);
              return self;
            });
}

at::Tensor& AtenAtan2Out(const at::Tensor& x, const at::Tensor& y,
                         at::Tensor& out) {
  TT_KERNEL(OpName::kAtan2Out, _, (x, y, out), {
    TT_THROW_IF_ERROR(CheckAtan2Inputs(x, y));
    TT_THROW_IF_ERROR(
        BinaryOpOut(x, y, out, BuildAtan2Shlo,
                    {.force_float_inputs = true,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenBitwiseAndTensorOut(const at::Tensor& self,
                                    const at::Tensor& other, at::Tensor& out) {
  TT_KERNEL(OpName::kBitwiseAndOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckBitwiseOpsInputs(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildBitwiseAndShlo,
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
        BinaryOpOut(new_self, new_other, out, BuildBitwiseLeftShiftShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenBitwiseOrTensorOut(const at::Tensor& self,
                                   const at::Tensor& other, at::Tensor& out) {
  TT_KERNEL(OpName::kBitwiseOrOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckBitwiseOpsInputs(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildBitwiseOrShlo,
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
        BinaryOpOut(new_self, new_other, out, BuildBitwiseRightShiftShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenBitwiseXorTensorOut(const at::Tensor& self,
                                    const at::Tensor& other, at::Tensor& out) {
  TT_KERNEL(OpName::kBitwiseXorOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckBitwiseOpsInputs(self, other));
    TT_ASSIGN_OR_THROW(auto output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildBitwiseXorShlo,
                    {.output_dtype_override = output_dtype,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenComplexOut(const at::Tensor& real, const at::Tensor& imag,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kComplexOut, _, (real, imag, out), {
    TT_THROW_IF_ERROR(CheckComplexOutInputs(real, imag, out));
    TT_THROW_IF_ERROR(
        BinaryOpOut(real, imag, out, BuildComplexShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenDivOut(const at::Tensor& self, const at::Tensor& other,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kDivOut, _, (self, other, out), {
    TT_ASSIGN_OR_THROW(auto div_op_options, GetDivOpBuilder(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, std::move(div_op_options.op_builder),
                    {.op_param_cache_keys =
                         std::move(div_op_options.op_param_cache_keys)}));
    return out;
  });
}

at::Tensor& AtenDivOutMode(const at::Tensor& self, const at::Tensor& other,
                           std::optional<std::string_view> mode,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kDivOut, _,
            (self, other, IgnoreInCacheKey(mode, "Legacy usage"), out), {
              TT_ASSIGN_OR_THROW((auto [op_builder, op_param_cache_keys, _]),
                                 GetDivOpBuilder(self, other, mode));
              TT_THROW_IF_ERROR(BinaryOpOut(
                  self, other, out, std::move(op_builder),
                  {.op_param_cache_keys = std::move(op_param_cache_keys)}));
              return out;
            });
}

at::Tensor& AtenEqScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kEqScalarOut, _,
            (self, IgnoreInCacheKey(other, "Legacy usage"), out), {
              TT_THROW_IF_ERROR(BinaryOpOut(
                  self, other, out, BuildEqShlo,
                  {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
              return out;
            });
}

at::Tensor& AtenEqTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kEqOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildEqShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenFloorDivide(const at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kFloorDivide, _, (self, other), {
    TT_THROW_IF_ERROR(CheckFloorDivideInputs(self, other));
    return at::div(self, other, "floor");
  });
}

at::Tensor& AtenFloorDivideOut(const at::Tensor& self, const at::Tensor& other,
                               at::Tensor& out) {
  TT_KERNEL(OpName::kFloorDivide, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckFloorDivideInputs(self, other));
    return AtenDivOutMode(self, other, "floor", out);
  });
}

at::Tensor& AtenFloorDivide_Tensor(at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kFloorDivide, _, (self, other), {
    TT_THROW_IF_ERROR(CheckFloorDivideInputs(self, other));
    return self.div_(other, "floor");
  });
}

at::Tensor& AtenFmodTensorOut(const at::Tensor& self, const at::Tensor& other,
                              at::Tensor& out) {
  TT_KERNEL(OpName::kFmodTensorOut, _, (self, other, out), {
    TT_CHECK_THROW(!self.is_complex() && !other.is_complex(),
                   error::kInvalidArgument)
        << "complex dtypes are not supported";
    TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Bool &&
                       other.scalar_type() != at::ScalarType::Bool,
                   error::kInvalidArgument)
        << "boolean dtypes are not supported";

    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildFmodTensorShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenGeScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kGeScalarOut, _,
            (self, IgnoreInCacheKey(other, "Legacy usage"), out), {
              TT_THROW_IF_ERROR(CheckComparisonOpsInputs(self, other));
              TT_THROW_IF_ERROR(BinaryOpOut(
                  self, other, out, BuildGeShlo,
                  {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
              return out;
            });
}

at::Tensor& AtenGeTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kGeOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckComparisonOpsInputs(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildGeShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenGtScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kGtScalarOut, _,
            (self, IgnoreInCacheKey(other, "Legacy usage"), out), {
              TT_THROW_IF_ERROR(CheckComparisonOpsInputs(self, other));
              TT_THROW_IF_ERROR(BinaryOpOut(
                  self, other, out, BuildGtShlo,
                  {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
              return out;
            });
}

at::Tensor& AtenGtTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kGtOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckComparisonOpsInputs(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildGtShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenIlshiftScalar(at::Tensor& self, const at::Scalar& other) {
  TT_KERNEL(OpName::kIlshiftScalar, _,
            (self, IgnoreInCacheKey(other, "Legacy usage")), {
              at::Tensor wrapped_scalar =
                  at::native::wrapped_scalar_tensor(other);
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
  TT_KERNEL(OpName::kIrshiftScalar, _,
            (self, IgnoreInCacheKey(other, "Legacy usage")), {
              at::Tensor wrapped_scalar =
                  at::native::wrapped_scalar_tensor(other);
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
  TT_KERNEL(OpName::kLeScalarOut, _,
            (self, IgnoreInCacheKey(other, "Legacy usage"), out), {
              TT_THROW_IF_ERROR(CheckComparisonOpsInputs(self, other));
              TT_THROW_IF_ERROR(BinaryOpOut(
                  self, other, out, BuildLeShlo,
                  {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
              return out;
            });
}

at::Tensor& AtenLeTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kLeOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckComparisonOpsInputs(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildLeShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenLshiftScalar(const at::Tensor& self, const at::Scalar& other) {
  TT_KERNEL(OpName::kLshiftScalar, _,
            (self, IgnoreInCacheKey(other, "Legacy usage")), {
              at::Tensor out = MakeEmptyTensor(self.sizes(), self.scalar_type(),
                                               self.device());
              at::Tensor wrapped_scalar =
                  at::native::wrapped_scalar_tensor(other);
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
  TT_KERNEL(OpName::kLtScalarOut, _,
            (self, IgnoreInCacheKey(other, "Legacy usage"), out), {
              TT_THROW_IF_ERROR(CheckComparisonOpsInputs(self, other));
              TT_THROW_IF_ERROR(BinaryOpOut(
                  self, other, out, BuildLtShlo,
                  {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
              return out;
            });
}

at::Tensor& AtenLtTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kLtOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckComparisonOpsInputs(self, other));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildLtShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenMaximumOut(const at::Tensor& self, const at::Tensor& other,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kMaximumOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildMaximumShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenMinimumOut(const at::Tensor& self, const at::Tensor& other,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kMinimumOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildMinimumShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenMulOut(const at::Tensor& self, const at::Tensor& other,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kMulOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildMulShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenNeScalarOut(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  TT_KERNEL(
      OpName::kNeScalarOut, _,
      (self, IgnoreInCacheKey(other, "Legacy usage"), out), {
        TT_ASSIGN_OR_THROW(auto output_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));
        TT_THROW_IF_ERROR(
            BinaryOpOut(self, other, out, BuildNeShlo,
                        {.output_dtype_override = output_dtype,
                         .op_param_cache_keys = OpParamCacheKeys::Empty()}));
        return out;
      });
}

at::Tensor& AtenNeTensorOut(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  TT_KERNEL(OpName::kNeOut, _, (self, other, out), {
    TT_ASSIGN_OR_THROW(auto output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildNeShlo,
                    {.output_dtype_override = output_dtype,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenPolarOut(const at::Tensor& abs, const at::Tensor& angle,
                         at::Tensor& out) {
  TT_KERNEL(OpName::kPolarOut, _, (abs, angle, out), {
    TT_THROW_IF_ERROR(CheckPolarInputs(abs, angle));
    TT_THROW_IF_ERROR(
        BinaryOpOut(abs, angle, out, BuildPolarShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenPowScalarOut(const at::Scalar& self, const at::Tensor& exponent,
                             at::Tensor& out) {
  TT_KERNEL(OpName::kPowScalarOut, _,
            (IgnoreInCacheKey(self, "Legacy usage"), exponent, out), {
              TT_THROW_IF_ERROR(CheckPowInputs(self, exponent));
              // Can't use reverse_operands here because a^b != b^a.
              // Cast to out tensor dtype to be consistent with PyTorch.
              TT_ASSIGN_OR_THROW(at::Tensor self_tensor,
                                 MakeTensor(self, out.scalar_type()));
              TT_THROW_IF_ERROR(BinaryOpOut(
                  self_tensor, exponent, out, BuildPowShlo,
                  // Use kPowOut for cache key as the builder logic is the same
                  // as for AtenPowTensorTensorOut.
                  {.op_name = OpName::kPowOut,
                   .op_param_cache_keys = OpParamCacheKeys::Empty()}));
              return out;
            });
}

at::Tensor& AtenPowTensorScalarOut(const at::Tensor& self,
                                   const at::Scalar& exponent,
                                   at::Tensor& out) {
  TT_KERNEL(OpName::kPowTensorScalarOut, _,
            (self, IgnoreInCacheKey(exponent, "Legacy usage"), out), {
              TT_THROW_IF_ERROR(CheckPowInputs(self, exponent));
              // Cast to self dtype to be consistent with PyTorch.
              TT_ASSIGN_OR_THROW(at::Tensor exponent_tensor,
                                 MakeTensor(exponent, self.scalar_type()));
              TT_THROW_IF_ERROR(BinaryOpOut(
                  self, exponent, out, BuildPowShlo,
                  // Use kPowOut for cache key as the builder logic is the same
                  // as for AtenPowTensorTensorOut.
                  {.op_name = OpName::kPowOut,
                   .op_param_cache_keys = OpParamCacheKeys::Empty()}));
              return out;
            });
}

at::Tensor& AtenPowTensorTensorOut(const at::Tensor& self,
                                   const at::Tensor& exponent,
                                   at::Tensor& out) {
  TT_KERNEL(OpName::kPowOut, _, (self, exponent, out), {
    TT_THROW_IF_ERROR(CheckPowInputs(self, exponent));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, exponent, out, BuildPowShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenRemainderScalarTensor(const at::Scalar& self,
                                     const at::Tensor& other) {
  TT_KERNEL(
      OpName::kRemainder, _, (IgnoreInCacheKey(self, "Legacy usage"), other), {
        TT_THROW_IF_ERROR(CheckRemainderInputs(self, other));
        TT_ASSIGN_OR_THROW(auto div_opts, GetDivOpOptionsFloorMode());

        auto remainder_builder =
            [div_opts = std::move(div_opts)](
                mlir::MlirOp self_op,
                mlir::MlirOp other_op) -> absl::StatusOr<mlir::MlirOp> {
          TT_ASSIGN_OR_RETURN((auto [cast_self_op, cast_other_op]),
                              ApplyBroadcastIfNeeded(self_op, other_op));
          TT_ASSIGN_OR_RETURN(auto div_op,
                              div_opts.op_builder(cast_self_op, cast_other_op));
          auto mul_op = stablehlo::Mul(div_op, cast_other_op);
          return stablehlo::Subtract(cast_self_op, mul_op);
        };

        TT_ASSIGN_OR_THROW(
            auto result,
            BinaryOp(other, self, std::move(remainder_builder),
                     {.reverse_operands = true,
                      .op_param_cache_keys = OpParamCacheKeys::Empty()}));
        return result;
      });
}

at::Tensor& AtenRemainderTensorOut(const at::Tensor& self,
                                   const at::Tensor& other, at::Tensor& out) {
  TT_KERNEL(OpName::kRemainderOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(CheckRemainderInputs(self, other));
    TT_ASSIGN_OR_THROW(auto div_opts, GetDivOpOptionsFloorMode());

    auto remainder_builder =
        [div_opts = std::move(div_opts)](
            mlir::MlirOp self_op,
            mlir::MlirOp other_op) -> absl::StatusOr<mlir::MlirOp> {
      TT_ASSIGN_OR_RETURN((auto [cast_self_op, cast_other_op]),
                          ApplyBroadcastIfNeeded(self_op, other_op));
      TT_ASSIGN_OR_RETURN(auto div_op,
                          div_opts.op_builder(cast_self_op, cast_other_op));
      auto mul_op = stablehlo::Mul(div_op, cast_other_op);
      return stablehlo::Subtract(cast_self_op, mul_op);
    };
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, std::move(remainder_builder),
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenRshiftScalar(const at::Tensor& self, const at::Scalar& other) {
  TT_KERNEL(OpName::kRshiftScalar, _,
            (self, IgnoreInCacheKey(other, "Legacy usage")), {
              at::Tensor out = MakeEmptyTensor(self.sizes(), self.scalar_type(),
                                               self.device());
              at::Tensor wrapped_scalar =
                  at::native::wrapped_scalar_tensor(other);
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
  TT_KERNEL(OpName::kRsub, _,
            (self, other, IgnoreInCacheKey(alpha, "Legacy usage")),
            { return at::sub(other, self, alpha); });
}

at::Tensor& AtenSubOut(const at::Tensor& self, const at::Tensor& other,
                       const at::Scalar& alpha, at::Tensor& out) {
  bool alpha_is_one = false;
  if (alpha.isIntegral(/*include_bool=*/false)) {
    alpha_is_one = (alpha.toLong() == 1);
  } else if (alpha.isFloatingPoint()) {
    alpha_is_one = (alpha.toDouble() == 1.0);
  } else if (alpha.isBoolean()) {
    alpha_is_one = (alpha.toBool() == true);
  }

  auto promoted_alpha = PromoteScalar(alpha);

  auto op_builder = [alpha_is_one](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<MlirOpResults<1>> {
    auto [self_op, other_op, alpha_op] = inputs;
    if (alpha_is_one) {
      TT_ASSIGN_OR_RETURN(auto result, BuildSubShlo(self_op, other_op));
      return MlirOpResults<1>{result};
    }
    const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other_op);
    auto other_element_type = other_type.getElementType();
    auto converted_alpha =
        mlir::stablehlo::ConvertElementType(alpha_op, other_element_type);
    TT_ASSIGN_OR_RETURN(auto mul_op, BuildMulShlo(other_op, converted_alpha));
    TT_ASSIGN_OR_RETURN(auto result, BuildSubShlo(self_op, mul_op));
    return MlirOpResults<1>{result};
  };

  TT_KERNEL(OpName::kSubOut, param_keys, (self, other, promoted_alpha, out), {
    TT_THROW_IF_ERROR(CheckSubInputs(self, other));
    TT_THROW_IF_ERROR(param_keys.SetParam("alpha_is_one", alpha_is_one));
    TT_ASSIGN_OR_THROW(auto alpha_tensor, promoted_alpha.GetTensor());
    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));

    TT_ASSIGN_OR_THROW(
        auto result,
        (DispatchOp<3>(op_builder,
                       /*inputs=*/{self, other, alpha_tensor},
                       /*options=*/
                       {.out_dtype = output_dtype,
                        .out_dims = out.sizes(),
                        .op_param_cache_keys = std::move(param_keys)})));

    TT_THROW_IF_ERROR(AssignBufferToAtTensor(result, out));
    return out;
  });
}

// go/keep-sorted end

}  // namespace torch_tpu
