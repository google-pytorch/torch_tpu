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

#include "torch_tpu/ops/normal/normal_aten_kernels.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/native/Resize.h"
#include "ATen/ops/broadcast_tensors.h"
#include "ATen/ops/scalar_tensor.h"
#include "ATen/ops/stack.h"
#include "ATen/ops/zeros_like.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/rng_utils.h"
#include "torch_tpu/ops/uniform/uniform.h"
#include "torch_tpu/ops/view/view_aten_kernels.h"

namespace torch_tpu {
namespace {

// This function generates a vector of standard normal random numbers using the
// chlo::ErfInv function. We control the source of our randomness by keeping
// our random generator state in eager/device_gen_impl.h and pass it to
// shlo::RngBitGenerator to generate random bits. We then use those to sample
// from distributions. Note: shlo::Rng doesn't support a custom seed/algorithm.
absl::StatusOr<MlirOpResults<1>> BuildStandardNormalShlo(
    mlir::MlirOp rng_state, llvm::ArrayRef<int64_t> shape,
    mlir::ElementType element_type) {
  // Use chlo::erfcinv to generate a standard normal distribution from a uniform
  // distribution.
  // Compute nextafter(-1, 0) in the target type to prevent erfinv(-1) = -inf.
  mlir::Type mlir_type =
      mlir::getElementType(rng_state.getContext(), element_type);
  auto float_type = mlir::cast<mlir::FloatType>(mlir_type);
  llvm::APFloat from_ap(float_type.getFloatSemantics(), "-1.0");
  from_ap.next(/*nextDown=*/false);
  double from = from_ap.convertToDouble();
  TT_ASSIGN_OR_RETURN(
      auto uniform_results,
      BuildUniformShlo(rng_state, from, 1.0, shape, element_type));
  auto uniform_op = uniform_results;
  auto erf_inv_op = mlir::chlo::ErfInv(uniform_op);
  auto two = MakeConstantLike(erf_inv_op, 2.0, element_type);
  auto sqrt_two = mlir::stablehlo::Sqrt(two);
  auto gaussian_op = mlir::stablehlo::Mul(erf_inv_op, sqrt_two);
  return gaussian_op;
}

/// Generates a tensor of normal random numbers with the given shape, mean,
// standard deviation, and standard deviation scale (std_scale).
// The `std_scale` parameter ensures the variance of complex outputs is
// correct. Because PyTorch handles complex random numbers by independently
// generating normal variants for the real and imaginary components,
// `std_scale` is set to 1/sqrt(2) for complex types so that the resulting
// joint variance equals the requested standard deviation squared.
// For real types, `std_scale` is simply 1.0.
absl::StatusOr<mlir::MlirOp> MakeConstantScalar(
    mlir::MlirBuilder& builder, double val,
    mlir::ElementType element_type_enum) {
  auto& ctx = builder.getContext();
  mlir::Type mlir_type = mlir::getElementType(ctx, element_type_enum);
  if (mlir::isa<mlir::ComplexType>(mlir_type)) {
    auto complex_type = mlir::cast<mlir::ComplexType>(mlir_type);
    auto real_element_type = complex_type.getElementType();
    TT_ASSIGN_OR_RETURN(auto real_element_type_enum,
                        ConvertTo<mlir::ElementType>(real_element_type));
    auto real_part = MakeScalarConstant(builder, val, real_element_type_enum);
    auto imag_part = MakeScalarConstant(builder, 0.0, real_element_type_enum);
    return mlir::stablehlo::Complex(real_part, imag_part);
  } else {
    return MakeScalarConstant(builder, val, element_type_enum);
  }
}

absl::StatusOr<mlir::MlirOp> MakeConstantRealScalar(
    mlir::MlirBuilder& builder, double val,
    mlir::ElementType element_type_enum) {
  auto& ctx = builder.getContext();
  mlir::Type mlir_type = mlir::getElementType(ctx, element_type_enum);
  mlir::ElementType real_element_type_enum;
  if (mlir::isa<mlir::ComplexType>(mlir_type)) {
    auto complex_type = mlir::cast<mlir::ComplexType>(mlir_type);
    TT_ASSIGN_OR_RETURN(
        real_element_type_enum,
        ConvertTo<mlir::ElementType>(complex_type.getElementType()));
  } else {
    real_element_type_enum = element_type_enum;
  }
  return MakeScalarConstant(builder, val, real_element_type_enum);
}

// Generates a tensor of normal random numbers with the given shape, mean,
// standard deviation, and standard deviation scale (std_scale).
absl::StatusOr<MlirOpResults<1>> BuildNormalShlo(
    mlir::MlirOp rng_state, llvm::ArrayRef<int64_t> shape,
    mlir::ElementType element_type_enum, mlir::MlirOp mean, mlir::MlirOp std) {
  mlir::Type mlir_type =
      mlir::getElementType(rng_state.getContext(), element_type_enum);

  if (mlir::isa<mlir::ComplexType>(mlir_type)) {
    auto complex_type = mlir::cast<mlir::ComplexType>(mlir_type);
    auto real_element_type = complex_type.getElementType();
    TT_ASSIGN_OR_RETURN(auto real_element_type_enum,
                        ConvertTo<mlir::ElementType>(real_element_type));

    // Construct a real shape by appending 2 to shape
    Dimensions real_shape(shape.begin(), shape.end());
    real_shape.push_back(2);

    // Generate real normal distribution
    TT_ASSIGN_OR_RETURN(
        auto std_normal_results,
        BuildStandardNormalShlo(rng_state, real_shape, real_element_type_enum));
    auto std_normal = std_normal_results;

    // Slice real and imaginary parts from the standard normal distribution
    int64_t rank = real_shape.size();
    Indices start_indices(rank, 0);
    Dimensions limit_indices(real_shape.begin(), real_shape.end());
    Strides strides(rank, 1);

    limit_indices.back() = 1;
    auto real_part_sliced = mlir::stablehlo::Slice(std_normal, start_indices,
                                                   limit_indices, strides);

    start_indices.back() = 1;
    limit_indices.back() = 2;
    auto imag_part_sliced = mlir::stablehlo::Slice(std_normal, start_indices,
                                                   limit_indices, strides);

    auto real_part = mlir::stablehlo::Reshape(real_part_sliced, shape);
    auto imag_part = mlir::stablehlo::Reshape(imag_part_sliced, shape);

    // Scale standard deviation std by 1/sqrt(2.0)
    auto std_scale_const = MakeConstantLike(real_part, 1.0 / std::sqrt(2.0),
                                            real_element_type_enum);
    TT_ASSIGN_OR_RETURN(auto b_std, BroadcastIfNeeded(std, shape));
    TT_ASSIGN_OR_RETURN(auto b_std_scale,
                        BroadcastIfNeeded(std_scale_const, shape));
    TT_ASSIGN_OR_RETURN(b_std, CastIfNeeded(b_std, real_element_type_enum));
    TT_ASSIGN_OR_RETURN(b_std_scale,
                        CastIfNeeded(b_std_scale, real_element_type_enum));
    auto std_scaled = mlir::stablehlo::Mul(b_std, b_std_scale);

    // Scale real and imag parts
    auto real_part_scaled = mlir::stablehlo::Mul(real_part, std_scaled);
    auto imag_part_scaled = mlir::stablehlo::Mul(imag_part, std_scaled);

    // Extract/Broadcast real and imaginary parts of mean
    mlir::MlirOp mean_real;
    mlir::MlirOp mean_imag;

    auto mean_type = GetTensorTypeOrDie(mean);
    auto mean_element_type = mean_type.getElementType();

    if (mlir::isa<mlir::ComplexType>(mean_element_type)) {
      TT_ASSIGN_OR_RETURN(auto b_mean, BroadcastIfNeeded(mean, shape));
      mean_real = mlir::stablehlo::Real(b_mean);
      mean_imag = mlir::stablehlo::Imag(b_mean);
    } else {
      TT_ASSIGN_OR_RETURN(auto cast_mean,
                          CastIfNeeded(mean, real_element_type_enum));
      TT_ASSIGN_OR_RETURN(mean_real, BroadcastIfNeeded(cast_mean, shape));
      mean_imag = MakeConstantLike(mean_real, 0.0, real_element_type_enum);
    }

    auto real_part_final = mlir::stablehlo::Add(real_part_scaled, mean_real);
    auto imag_part_final = mlir::stablehlo::Add(imag_part_scaled, mean_imag);

    return mlir::stablehlo::Complex(real_part_final, imag_part_final);
  } else {
    // If element type is real
    TT_ASSIGN_OR_RETURN(
        auto std_normal_results,
        BuildStandardNormalShlo(rng_state, shape, element_type_enum));
    auto std_normal = std_normal_results;

    TT_ASSIGN_OR_RETURN(auto b_mean, BroadcastIfNeeded(mean, shape));
    TT_ASSIGN_OR_RETURN(auto b_std, BroadcastIfNeeded(std, shape));
    TT_ASSIGN_OR_RETURN(b_mean, CastIfNeeded(b_mean, element_type_enum));
    TT_ASSIGN_OR_RETURN(b_std, CastIfNeeded(b_std, element_type_enum));

    auto normal_with_variance = mlir::stablehlo::Mul(std_normal, b_std);
    auto normal = mlir::stablehlo::Add(normal_with_variance, b_mean);
    return normal;
  }
}

absl::Status CheckNormalPreconditions(const at::Tensor& tensor,
                                      std::string_view arg_name) {
  TT_RET_CHECK(IsFloatingPoint(tensor) || IsComplex(tensor),
               error::kInvalidArgument)
      << "expected the " << arg_name
      << " tensor to be floating point or complex type, got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

absl::Status CheckNormalStdPreconditions(double std) {
  TT_RET_CHECK(std >= 0.0, error::kInvalidArgument)
      << "expected std >= 0.0, but found std " << std;
  return absl::OkStatus();
}

absl::Status CheckNormalStdPreconditions(const at::Tensor& std,
                                         bool allow_integer = true) {
  TT_RET_CHECK(IsFloatingPoint(std) || (allow_integer && IsInteger(std)),
               error::kInvalidArgument)
      << "expected the std tensor to be "
      << (allow_integer ? "non-complex" : "floating point") << ", got "
      << ToString(std.scalar_type());
  // Value-based non-negative checks on TPU tensors are bypassed to avoid
  // performance-killing CPU-TPU synchronization stalls and composite dispatch
  // issues.
  return absl::OkStatus();
}

absl::StatusOr<DeviceBufferRef> NormalTensorTensor(
    llvm::ArrayRef<int64_t> shape, at::ScalarType scalar_type,
    const at::Tensor& mean, const at::Tensor& std,
    std::optional<at::Generator> generator) {
  TT_ASSIGN_OR_RETURN(auto mlir_type,
                      ConvertTo<mlir::ElementType>(scalar_type));
  auto builder = [shape = Dimensions(shape.begin(), shape.end()),
                  mlir_type](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<MlirOpResults<1>> {
    auto [rng_state, mean, std] = inputs;
    return BuildNormalShlo(rng_state, shape, mlir_type, mean, std);
  };
  return DispatchRngOpAndReturnBuffer(
      generator,
      [builder = std::move(builder),
       shape = Dimensions(shape.begin(), shape.end()), mean, std,
       mlir_type](at::Tensor rng_input_state) mutable
          -> absl::StatusOr<std::vector<DeviceBufferRef>> {
        TT_ASSIGN_OR_RETURN(
            auto buf,
            (DispatchOp<3, 1>(std::move(builder), {rng_input_state, mean, std},
                              {.out_dtype = mlir_type,
                               .out_dims = shape,
                               .op_param_cache_keys = OpParamCacheKeys::Empty(),
                               .split_mode = OpSplitMode::kSplitAfter})));
        return std::vector<DeviceBufferRef>{std::move(buf)};
      });
}

absl::StatusOr<DeviceBufferRef> NormalScalarScalar(
    llvm::ArrayRef<int64_t> shape, at::ScalarType scalar_type, double mean,
    double std, std::optional<at::Generator> generator) {
  TT_ASSIGN_OR_RETURN(auto mlir_type,
                      ConvertTo<mlir::ElementType>(scalar_type));
  auto builder =
      [mean, std, shape = Dimensions(shape.begin(), shape.end()),
       mlir_type](mlir::MlirOp rng_state) -> absl::StatusOr<MlirOpResults<1>> {
    TT_ASSIGN_OR_RETURN(auto mean_op, MakeConstantScalar(rng_state.getBuilder(),
                                                         mean, mlir_type));
    TT_ASSIGN_OR_RETURN(
        auto std_op,
        MakeConstantRealScalar(rng_state.getBuilder(), std, mlir_type));
    return BuildNormalShlo(rng_state, shape, mlir_type, mean_op, std_op);
  };
  return DispatchRngOpAndReturnBuffer(
      generator,
      [builder = std::move(builder),
       shape = Dimensions(shape.begin(), shape.end()),
       mlir_type](at::Tensor rng_input_state) mutable
          -> absl::StatusOr<std::vector<DeviceBufferRef>> {
        TT_ASSIGN_OR_RETURN(
            auto buf,
            (DispatchOp<1, 1>(std::move(builder), {rng_input_state},
                              {.out_dtype = mlir_type,
                               .out_dims = shape,
                               .op_param_cache_keys = OpParamCacheKeys::Empty(),
                               .split_mode = OpSplitMode::kSplitAfter})));
        return std::vector<DeviceBufferRef>{std::move(buf)};
      });
}

absl::StatusOr<DeviceBufferRef> NormalScalarTensor(
    llvm::ArrayRef<int64_t> shape, at::ScalarType scalar_type, double mean,
    const at::Tensor& std, std::optional<at::Generator> generator) {
  TT_ASSIGN_OR_RETURN(auto mlir_type,
                      ConvertTo<mlir::ElementType>(scalar_type));
  auto builder = [mean, shape = Dimensions(shape.begin(), shape.end()),
                  mlir_type](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<MlirOpResults<1>> {
    auto [rng_state, std] = inputs;
    TT_ASSIGN_OR_RETURN(auto mean_op,
                        MakeConstantScalar(std.getBuilder(), mean, mlir_type));
    return BuildNormalShlo(rng_state, shape, mlir_type, mean_op, std);
  };
  return DispatchRngOpAndReturnBuffer(
      generator,
      [builder = std::move(builder),
       shape = Dimensions(shape.begin(), shape.end()), std,
       mlir_type](at::Tensor rng_input_state) mutable
          -> absl::StatusOr<std::vector<DeviceBufferRef>> {
        TT_ASSIGN_OR_RETURN(
            auto buf,
            (DispatchOp<2, 1>(std::move(builder), {rng_input_state, std},
                              {.out_dtype = mlir_type,
                               .out_dims = shape,
                               .op_param_cache_keys = OpParamCacheKeys::Empty(),
                               .split_mode = OpSplitMode::kSplitAfter})));
        return std::vector<DeviceBufferRef>{std::move(buf)};
      });
}

absl::StatusOr<DeviceBufferRef> NormalTensorScalar(
    llvm::ArrayRef<int64_t> shape, at::ScalarType scalar_type,
    const at::Tensor& mean, double std,
    std::optional<at::Generator> generator) {
  TT_ASSIGN_OR_RETURN(auto mlir_type,
                      ConvertTo<mlir::ElementType>(scalar_type));
  auto builder = [std, shape = Dimensions(shape.begin(), shape.end()),
                  mlir_type](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<MlirOpResults<1>> {
    auto [rng_state, mean] = inputs;
    TT_ASSIGN_OR_RETURN(
        auto std_op, MakeConstantRealScalar(mean.getBuilder(), std, mlir_type));
    return BuildNormalShlo(rng_state, shape, mlir_type, mean, std_op);
  };
  return DispatchRngOpAndReturnBuffer(
      generator,
      [builder = std::move(builder),
       shape = Dimensions(shape.begin(), shape.end()), mean,
       mlir_type](at::Tensor rng_input_state) mutable
          -> absl::StatusOr<std::vector<DeviceBufferRef>> {
        TT_ASSIGN_OR_RETURN(
            auto buf,
            (DispatchOp<2, 1>(std::move(builder), {rng_input_state, mean},
                              {.out_dtype = mlir_type,
                               .out_dims = shape,
                               .op_param_cache_keys = OpParamCacheKeys::Empty(),
                               .split_mode = OpSplitMode::kSplitAfter})));
        return std::vector<DeviceBufferRef>{std::move(buf)};
      });
}

}  // namespace

at::Tensor& AtenNormal_(at::Tensor& self, double mean, double std,
                        std::optional<at::Generator> generator) {
  TT_KERNEL(
      OpName::kNormal_, _,
      (self, IgnoreInCacheKey(mean, "Legacy usage"),
       IgnoreInCacheKey(std, "Legacy usage"),
       IgnoreInCacheKey(generator, "Legacy usage")),
      {
        TT_THROW_IF_ERROR(CheckNormalPreconditions(self, /*arg_name=*/"self"));
        TT_THROW_IF_ERROR(CheckNormalStdPreconditions(std));
        TT_ASSIGN_OR_THROW(auto output_buf,
                           NormalScalarScalar(self.sizes(), self.scalar_type(),
                                              mean, std, generator));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), self));
        return self;
      });
}

at::Tensor AtenNormalFloatTensor(double mean, const at::Tensor& std,
                                 std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kNormalFloatTensor, _,
            (IgnoreInCacheKey(mean, "Legacy usage"), std,
             IgnoreInCacheKey(generator, "Legacy usage")),
            {
              TT_THROW_IF_ERROR(
                  CheckNormalStdPreconditions(std, /*allow_integer=*/false));
              TT_ASSIGN_OR_THROW(
                  auto output_buf,
                  NormalScalarTensor(std.sizes(), std.scalar_type(), mean, std,
                                     generator));
              return MakeTensor(std::move(output_buf));
            });
}

at::Tensor& AtenNormalFloatTensorOut(double mean, const at::Tensor& std,
                                     std::optional<at::Generator> generator,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kNormalFloatTensorOut, _,
      (IgnoreInCacheKey(mean, "Legacy usage"), std,
       IgnoreInCacheKey(generator, "Legacy usage"), out),
      {
        TT_THROW_IF_ERROR(
            CheckNormalStdPreconditions(std, /*allow_integer=*/false));
        at::native::resize_output(out, std.sizes());
        TT_THROW_IF_ERROR(CheckNormalPreconditions(out, /*arg_name=*/"out"));
        TT_ASSIGN_OR_THROW(auto output_buf,
                           NormalScalarTensor(out.sizes(), out.scalar_type(),
                                              mean, std, generator));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), out));
        return out;
      });
}

at::Tensor AtenNormalTensorFloat(const at::Tensor& mean, double std,
                                 std::optional<at::Generator> generator) {
  TT_KERNEL(
      OpName::kNormalTensorFloat, _,
      (mean, IgnoreInCacheKey(std, "Legacy usage"),
       IgnoreInCacheKey(generator, "Legacy usage")),
      {
        TT_THROW_IF_ERROR(CheckNormalPreconditions(mean, /*arg_name=*/"mean"));
        TT_THROW_IF_ERROR(CheckNormalStdPreconditions(std));
        TT_ASSIGN_OR_THROW(auto output_buf,
                           NormalTensorScalar(mean.sizes(), mean.scalar_type(),
                                              mean, std, generator));
        return MakeTensor(std::move(output_buf));
      });
}

at::Tensor& AtenNormalTensorFloatOut(const at::Tensor& mean, double std,
                                     std::optional<at::Generator> generator,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kNormalTensorFloatOut, _,
      (mean, IgnoreInCacheKey(std, "Legacy usage"),
       IgnoreInCacheKey(generator, "Legacy usage"), out),
      {
        TT_THROW_IF_ERROR(CheckNormalPreconditions(mean, /*arg_name=*/"mean"));
        TT_THROW_IF_ERROR(CheckNormalStdPreconditions(std));
        at::native::resize_output(out, mean.sizes());
        TT_THROW_IF_ERROR(CheckNormalPreconditions(out, /*arg_name=*/"out"));
        TT_ASSIGN_OR_THROW(auto output_buf,
                           NormalTensorScalar(out.sizes(), out.scalar_type(),
                                              mean, std, generator));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), out));
        return out;
      });
}

at::Tensor AtenNormalTensorTensor(const at::Tensor& mean, const at::Tensor& std,
                                  std::optional<at::Generator> generator) {
  TT_KERNEL(
      OpName::kNormalTensorTensor, _,
      (mean, std, IgnoreInCacheKey(generator, "Legacy usage")), {
        TT_THROW_IF_ERROR(CheckNormalPreconditions(mean, /*arg_name=*/"mean"));
        TT_THROW_IF_ERROR(CheckNormalStdPreconditions(std));
        TT_ASSIGN_OR_THROW(auto shape, InferSize(mean, std));
        TT_ASSIGN_OR_THROW(auto output_buf,
                           NormalTensorTensor(shape, mean.scalar_type(), mean,
                                              std, generator));
        return MakeTensor(std::move(output_buf));
      });
}

at::Tensor& AtenNormalTensorTensorOut(const at::Tensor& mean,
                                      const at::Tensor& std,
                                      std::optional<at::Generator> generator,
                                      at::Tensor& out) {
  TT_KERNEL(
      OpName::kNormalTensorTensorOut, _,
      (mean, std, IgnoreInCacheKey(generator, "Legacy usage"), out), {
        TT_THROW_IF_ERROR(CheckNormalPreconditions(mean, /*arg_name=*/"mean"));
        TT_THROW_IF_ERROR(CheckNormalStdPreconditions(std));
        TT_ASSIGN_OR_THROW(auto shape, InferSize(mean, std));
        at::native::resize_output(out, shape);
        TT_THROW_IF_ERROR(CheckNormalPreconditions(out, /*arg_name=*/"out"));
        TT_ASSIGN_OR_THROW(
            auto output_buf,
            NormalTensorTensor(shape, out.scalar_type(), mean, std, generator));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
