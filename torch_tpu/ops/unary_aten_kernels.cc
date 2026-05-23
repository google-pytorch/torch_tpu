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

#include "torch_tpu/ops/unary_aten_kernels.h"

#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/native/Resize.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"  // IWYU pragma: keep
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/macro_utils.h"  // IWYU pragma: keep
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary.h"

// A "Simple Unary" op is an Aten op that:
// - takes 1 input and produces 1 output
// - the dtype and shape of the output is the same as the input
// - doesn't have any parameters
// TODO(b/442660800): minimize the number of kernels implemented by macros
#define TT_DEFINE_ATEN_UNARY_OUT(op_name, func_name, op_builder)        \
  at::Tensor& func_name##Out(const at::Tensor& self, at::Tensor& out) { \
    TT_KERNEL(op_name, _, (self, out), {                                \
      TT_THROW_IF_ERROR(::torch_tpu::UnaryOpOut(                        \
          self, out, op_builder,                                        \
          {.op_param_cache_keys = OpParamCacheKeys::Empty()}));         \
      return out;                                                       \
    });                                                                 \
  }                                                                     \
  TT_REQUIRE_SEMICOLON_

// A floating-point (FP) only op is a simple Aten op (see above), with the
// additional restriction that it will cast its integer arguments to the default
// PyTorch data type. This captures the default dtype at **dispatch** time, not
// **compile** time, to mirror eager CUDA behavior.
// This means the default dtype also needs to be included as a parameter key.
// TODO(b/440585584): parameterize by explicit output dtype, not by default
// dtype, and remove the cache to avoid unnecessary cache misses when unused.
// TODO(b/442660800): minimize the number of kernels implemented by macros
#define TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(op_name, func_name, op_builder)     \
  at::Tensor& func_name##Out(const at::Tensor& self, at::Tensor& out) {      \
    TT_KERNEL(op_name, param_keys, (self, out), {                            \
      /* Find the desired output dtype. */                                   \
      const c10::ScalarType out_dtype = ::torch_tpu::InferOutputDtype(self); \
      TT_ASSIGN_OR_THROW(                                                    \
          const auto out_mlir_type,                                          \
          ::torch_tpu::ConvertTo<mlir::ElementType>(out_dtype));             \
      ::torch_tpu::MlirUnaryOpBuilder op_builder_with_out_dtype =            \
          [out_mlir_type](                                                   \
              mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {          \
        return op_builder(input, out_mlir_type);                             \
      };                                                                     \
      TT_THROW_IF_ERROR(::torch_tpu::UnaryOpOut(                             \
          self, out, std::move(op_builder_with_out_dtype),                   \
          {.op_param_cache_keys = std::move(param_keys),                     \
           .out_dtype = out_mlir_type}));                                    \
      return out;                                                            \
    });                                                                      \
  }                                                                          \
  TT_REQUIRE_SEMICOLON_

namespace torch_tpu {

absl::StatusOr<at::Tensor> UnaryOp(const at::Tensor& self,
                                   MlirUnaryOpBuilder op_builder,
                                   UnaryOpOptions options) {
  mlir::ElementType output_dtype;
  if (options.out_dtype.has_value()) {
    output_dtype = *options.out_dtype;
  } else {
    TT_ASSIGN_OR_RETURN(output_dtype,
                        ConvertTo<mlir::ElementType>(self.scalar_type()));
  }
  const at::IntArrayRef output_dims = options.out_dims.has_value()
                                          ? options.out_dims.value()
                                          : at::IntArrayRef(self.sizes());
  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<1>(
          std::move(op_builder), self,
          {.op_name = options.op_name,
           .out_dtype = output_dtype,
           .out_dims = output_dims,
           .computation_dtype = options.computation_dtype,
           .op_param_cache_keys = std::move(options.op_param_cache_keys)}));
  return MakeTensor(std::move(result_buf));
}
absl::StatusOr<at::Tensor> UnaryOp(const at::Tensor& self, OpName op_name,
                                   MlirUnaryOpBuilder op_builder,
                                   UnaryOpOptions options) {
  ABSL_CHECK(!options.op_name.has_value())  // CRASH_OK
      << "Cannot set the op name in options when calling UnaryOp with an "
         "explicit OpName parameter.";
  options.op_name = op_name;
  return UnaryOp(self, std::move(op_builder), std::move(options));
}

absl::Status UnaryOpInPlace(at::Tensor& self, MlirUnaryOpBuilder op_builder,
                            UnaryOpOptions options) {
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<1>(
          std::move(op_builder), self,
          {.op_name = options.op_name,
           .out_dtype = output_dtype,
           .out_dims = self.sizes(),
           .computation_dtype = options.computation_dtype,
           .op_param_cache_keys = std::move(options.op_param_cache_keys)}));
  return AssignBufferToAtTensor(std::move(result_buf), self);
}

absl::Status UnaryOpInPlace(at::Tensor& self, OpName op_name,
                            MlirUnaryOpBuilder op_builder,
                            UnaryOpOptions options) {
  ABSL_CHECK(!options.op_name.has_value())  // CRASH_OK
      << "Cannot set the op name in options when calling UnaryOpInPlace with "
         "an explicit OpName parameter.";
  options.op_name = op_name;
  return UnaryOpInPlace(self, std::move(op_builder), std::move(options));
}

absl::Status UnaryOpOut(const at::Tensor& self, at::Tensor& out,
                        MlirUnaryOpBuilder op_builder, UnaryOpOptions options) {
  TT_RET_CHECK(IsPrivateUse1Device(out), error::kInvalidArgument)
      << "expected the output tensor to be on "
      << GetPrivateUse1DeviceDebugName() << ", got " << out.device();

  const at::IntArrayRef shape =
      options.out_dims.has_value() ? *options.out_dims : self.sizes();

  mlir::ElementType output_dtype;
  if (options.out_dtype.has_value()) {
    output_dtype = *options.out_dtype;
  } else {
    TT_ASSIGN_OR_RETURN(output_dtype,
                        ConvertTo<mlir::ElementType>(out.scalar_type()));
  }

  const at::ScalarType expected_dtype = ConvertTo<at::ScalarType>(output_dtype);
  TT_RET_CHECK(out.scalar_type() == expected_dtype, error::kInvalidArgument)
      << "expected the output dtype to be " << ToString(expected_dtype)
      << ", got " << ToString(out.scalar_type());

  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<1>(
          std::move(op_builder), self,
          {.op_name = options.op_name,
           .out_dtype = output_dtype,
           .out_dims = shape,
           .computation_dtype = options.computation_dtype,
           .op_param_cache_keys = std::move(options.op_param_cache_keys)}));
  at::native::resize_output(out, shape);
  return AssignBufferToAtTensor(std::move(result_buf), out);
}

absl::Status UnaryOpOut(const at::Tensor& self, at::Tensor& out, OpName op_name,
                        MlirUnaryOpBuilder op_builder, UnaryOpOptions options) {
  ABSL_CHECK(!options.op_name.has_value())  // CRASH_OK
      << "Cannot set the op name in options when calling UnaryOpOut with an "
         "explicit OpName parameter.";
  options.op_name = op_name;
  return UnaryOpOut(self, out, std::move(op_builder), std::move(options));
}

// go/keep-sorted start
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kAcosOut, AtenAcos, BuildAcosShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kAcoshOut, AtenAcosh, BuildAcoshShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kAsinOut, AtenAsin, BuildAsinShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kAsinhOut, AtenAsinh, BuildAsinhShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kAtanOut, AtenAtan, BuildAtanShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kAtanhOut, AtenAtanh, BuildAtanhShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kCosOut, AtenCos, BuildCosShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kCoshOut, AtenCosh, BuildCoshShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kErfInvOut, AtenErfInv,
                                 BuildErfInvShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kErfOut, AtenErf, BuildErfShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kExp2Out, AtenExp2, BuildExp2Shlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kExpM1Out, AtenExpm1, BuildExpm1Shlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kExpOut, AtenExp, BuildExpShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kLgammaOut, AtenLgamma,
                                 BuildLgammaShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kLog10Out, AtenLog10, BuildLog10Shlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kLog1pOut, AtenLog1p, BuildLog1pShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kLog2Out, AtenLog2, BuildLog2Shlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kLogOut, AtenLog, BuildLogShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kReciprocalOut, AtenReciprocal,
                                 BuildReciprocalShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kRsqrtOut, AtenRsqrt, BuildRsqrtShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kSinOut, AtenSin, BuildSinShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kSinhOut, AtenSinh, BuildSinhShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kSqrtOut, AtenSqrt, BuildSqrtShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kTanOut, AtenTan, BuildTanShlo);
TT_DEFINE_FP_ONLY_ATEN_UNARY_OUT(OpName::kTanhOut, AtenTanh, BuildTanhShlo);
// go/keep-sorted end

at::Tensor& AtenAbsOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kAbsOut, _, (self, out), {
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, BuildAbsShlo,
                   {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenNegOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kNegOut, _, (self, out), {
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, BuildNegShlo,
                   {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor AtenRelu(const at::Tensor& self) {
  TT_KERNEL(OpName::kRelu, _, (self), {
    TT_ASSIGN_OR_THROW(at::Tensor result,
                       ::torch_tpu::UnaryOp(
                           self, BuildReluShlo,
                           {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return result;
  });
}

at::Tensor& AtenRelu_(at::Tensor& self) {
  TT_KERNEL(OpName::kRelu_, _, (self), {
    TT_THROW_IF_ERROR(
        UnaryOpInPlace(self, BuildReluShlo,
                       {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return self;
  });
}

at::Tensor& AtenSignOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kSignOut, _, (self, out), {
    TT_CHECK_THROW(!IsComplex(self), error::kInvalidArgument)
        << "expected the input dtype not to be complex, got "
        << ToString(self.scalar_type())
        << "; use torch.sgn() instead if you intend to normalize a complex "
           "tensor to each complex element having magnitude 1";
    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<1>(BuildSignShlo, self,
                      /*options=*/
                      {.out_dtype = out_dtype,
                       .out_dims = self.sizes(),
                       .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

at::Tensor& AtenSignbitOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kSignbitOut, _, (self, out), {
    const auto output_dtype = mlir::ElementType::PRED;
    auto op_builder =
        [output_dtype](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);

      if (IsBooleanType(input_type) || IsUnsignedType(input_type)) {
        return MakeConstantLike(input, false, output_dtype);
      } else if (IsFloatType(input_type)) {
        auto zero_float = MakeConstantLike(input, +0.0f);
        return mlir::stablehlo::Compare(
            input, zero_float, mlir::stablehlo::ComparisonDirection::LT,
            mlir::stablehlo::ComparisonTypeAttr::get(
                &input.getContext(),
                mlir::stablehlo::ComparisonType::TOTALORDER));
      } else {
        auto zero = MakeConstantLike(input, 0);
        return mlir::stablehlo::Compare(
            input, zero, mlir::stablehlo::ComparisonDirection::LT);
      }
    };

    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<1>(std::move(op_builder), self,
                      /*options=*/
                      {.out_dtype = output_dtype,
                       .out_dims = out.sizes(),
                       .op_param_cache_keys = OpParamCacheKeys::Empty()}));

    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

at::Tensor& AtenTruncOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kTruncOut, _, (self, out), {
    TT_CHECK_THROW(self.scalar_type() != c10::ScalarType::Bool,
                   error::kInvalidArgument)
        << "does not support boolean types";
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, BuildTruncShlo,
                   {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

// go/keep-sorted start
TT_DEFINE_ATEN_UNARY_OUT(OpName::kBitwiseNotOut, AtenBitwiseNot, BuildNotShlo);
TT_DEFINE_ATEN_UNARY_OUT(OpName::kCeilOut, AtenCeil, BuildCeilShlo);
TT_DEFINE_ATEN_UNARY_OUT(OpName::kConjPhysicalOut, AtenConjPhysical,
                         BuildConjPhysicalShlo);
TT_DEFINE_ATEN_UNARY_OUT(OpName::kFloorOut, AtenFloor, BuildFloorShlo);
TT_DEFINE_ATEN_UNARY_OUT(OpName::kLiftFreshOut, AtenLiftFresh,
                         BuildLiftFreshShlo);
TT_DEFINE_ATEN_UNARY_OUT(OpName::kSgnOut, AtenSgn, BuildSgnShlo);
TT_DEFINE_ATEN_UNARY_OUT(OpName::kSiluOut, AtenSilu, BuildSiluShlo);
// go/keep-sorted end

}  // namespace torch_tpu
