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

#include "csrc/ops/normal/normal_aten_kernels.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/ops/result_type.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/util/Optional.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_gen_impl.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/min_max/min_max.h"
#include "csrc/ops/min_max/min_max_aten_kernels.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/rng_utils.h"
#include "csrc/ops/uniform/uniform.h"
#include "csrc/pjrt/pjrt_utils.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

// Generates a standard normal distribution N(0, 1) using the Inverse Transform
// Sampling method (via the inverse error function).
//
// We generate a uniform random variable in [from, 1.0) and apply chlo::ErfInv
// to map it to a normal distribution.
absl::StatusOr<MlirOpResults<1>> BuildStandardNormalShlo(
    mlir::MlirOp rng_state, llvm::ArrayRef<int64_t> shape,
    mlir::ElementType type) {
  // Use chlo::erfcinv to generate a standard normal distribution from a uniform
  // distribution.
  // Compute nextafter(-1, 0) in the target type to prevent erfinv(-1) = -inf.
  auto mlir_type = mlir::getElementType(rng_state.getContext(), type);
  auto float_type = mlir::cast<mlir::FloatType>(mlir_type);
  llvm::APFloat from_ap(float_type.getFloatSemantics(), "-1.0");
  from_ap.next(/*nextDown=*/false);
  double from = from_ap.convertToDouble();
  TT_ASSIGN_OR_RETURN(auto uniform_results,
                      BuildUniformShlo(rng_state, from, 1.0, shape, type));
  auto uniform_op = uniform_results;
  auto erf_inv_op = mlir::chlo::ErfInv(uniform_op);
  auto two = MakeConstantLike(erf_inv_op, 2.0, type);
  auto sqrt_two = mlir::stablehlo::Sqrt(two);
  auto gaussian_op = mlir::stablehlo::Mul(erf_inv_op, sqrt_two);
  return gaussian_op;
}

// Scales and shifts the standard normal distribution to match the requested
// mean and standard deviation for real-valued outputs:
// Y = std * X_std + mean.
absl::StatusOr<MlirOpResults<1>> BuildNormalRealShlo(
    mlir::MlirOp rng_state, std::optional<mlir::MlirOp> mean_op,
    std::optional<mlir::MlirOp> std_op, double mean_val, double std_val,
    llvm::ArrayRef<int64_t> out_dims, mlir::ElementType out_dtype) {
  TT_ASSIGN_OR_RETURN(auto std_normal_res,
                      BuildStandardNormalShlo(rng_state, out_dims, out_dtype));
  auto std_normal = std_normal_res;

  mlir::MlirOp mean;
  if (mean_op.has_value()) {
    mean = *mean_op;
    TT_ASSIGN_OR_RETURN(mean, BroadcastIfNeeded(mean, out_dims));
    TT_ASSIGN_OR_RETURN(mean, CastIfNeeded(mean, out_dtype));
  } else {
    mean = MakeConstantLike(std_normal, mean_val, out_dtype);
  }

  mlir::MlirOp std;
  if (std_op.has_value()) {
    std = *std_op;
    TT_ASSIGN_OR_RETURN(std, BroadcastIfNeeded(std, out_dims));
    TT_ASSIGN_OR_RETURN(std, CastIfNeeded(std, out_dtype));
  } else {
    std = MakeConstantLike(std_normal, std_val, out_dtype);
  }

  auto normal_with_variance = mlir::stablehlo::Mul(std_normal, std);
  auto normal = mlir::stablehlo::Add(normal_with_variance, mean);
  return normal;
}

// Generates a complex-valued normal distribution.
//
// In PyTorch, a complex normal variable Z = X + iY has independent real (X) and
// imaginary (Y) parts, each with half the variance of the target distribution
// (variance = std^2 / 2). This means the standard deviation of each component
// is std / sqrt(2).
//
// We generate a standard normal of shape [S, 2] (where S is the output shape)
// to get independent random numbers for both components, slice them, apply the
// scaling and shifting, and combine them using stablehlo::Complex.
absl::StatusOr<MlirOpResults<1>> BuildNormalComplexShlo(
    mlir::MlirOp rng_state, std::optional<mlir::MlirOp> mean_op,
    std::optional<mlir::MlirOp> std_op, double mean_val, double std_val,
    llvm::ArrayRef<int64_t> out_dims, mlir::ElementType out_dtype) {
  auto real_dtype = RealComponentOf(out_dtype);

  // 1. Generate standard normal [S, 2]
  llvm::SmallVector<int64_t> rng_shape(out_dims.begin(), out_dims.end());
  rng_shape.push_back(2);

  TT_ASSIGN_OR_RETURN(
      auto std_normal_raw_res,
      BuildStandardNormalShlo(rng_state, rng_shape, real_dtype));
  auto std_normal_raw = std_normal_raw_res;

  // 2. Slice and Squeeze to get real and imag components [S]
  const int64_t rank = out_dims.size();

  llvm::SmallVector<int64_t> start_indices_real(rank + 1, 0);
  llvm::SmallVector<int64_t> limit_indices_real(rng_shape.begin(),
                                                rng_shape.end());
  limit_indices_real[rank] = 1;
  const llvm::SmallVector<int64_t> strides(rank + 1, 1);

  auto slice_real = mlir::stablehlo::Slice(std_normal_raw, start_indices_real,
                                           limit_indices_real, strides);

  llvm::SmallVector<int64_t> start_indices_imag(rank + 1, 0);
  start_indices_imag[rank] = 1;
  llvm::SmallVector<int64_t> limit_indices_imag(rng_shape.begin(),
                                                rng_shape.end());
  limit_indices_imag[rank] = 2;

  auto slice_imag = mlir::stablehlo::Slice(std_normal_raw, start_indices_imag,
                                           limit_indices_imag, strides);

  const llvm::SmallVector<int64_t> squeezed_shape(out_dims.begin(),
                                                  out_dims.end());
  auto std_normal_real = mlir::stablehlo::Reshape(slice_real, squeezed_shape);
  auto std_normal_imag = mlir::stablehlo::Reshape(slice_imag, squeezed_shape);

  // 3. Prepare mean components [S]
  mlir::MlirOp mean_real, mean_imag;
  if (mean_op.has_value()) {
    auto mean_type = GetTensorTypeOrDie(*mean_op);
    if (mlir::isa<mlir::ComplexType>(mean_type.getElementType())) {
      mean_real = mlir::stablehlo::Real(*mean_op);
      mean_imag = mlir::stablehlo::Imag(*mean_op);
    } else {
      mean_real = *mean_op;
      mean_imag = MakeConstantLike(std_normal_real, 0.0, real_dtype);
    }
    TT_ASSIGN_OR_RETURN(mean_real,
                        BroadcastIfNeeded(mean_real, squeezed_shape));
    TT_ASSIGN_OR_RETURN(mean_imag,
                        BroadcastIfNeeded(mean_imag, squeezed_shape));
    TT_ASSIGN_OR_RETURN(mean_real, CastIfNeeded(mean_real, real_dtype));
    TT_ASSIGN_OR_RETURN(mean_imag, CastIfNeeded(mean_imag, real_dtype));
  } else {
    mean_real = MakeConstantLike(std_normal_real, mean_val, real_dtype);
    mean_imag = MakeConstantLike(std_normal_real, 0.0, real_dtype);
  }

  // 4. Prepare std [S]
  mlir::MlirOp std_real;
  if (std_op.has_value()) {
    auto std_type = GetTensorTypeOrDie(*std_op);
    if (mlir::isa<mlir::ComplexType>(std_type.getElementType())) {
      std_real = mlir::stablehlo::Real(*std_op);
    } else {
      std_real = *std_op;
    }
    TT_ASSIGN_OR_RETURN(std_real, BroadcastIfNeeded(std_real, squeezed_shape));
    TT_ASSIGN_OR_RETURN(std_real, CastIfNeeded(std_real, real_dtype));
  } else {
    std_real = MakeConstantLike(std_normal_real, std_val, real_dtype);
  }

  // 5. Scale std by 1/sqrt(2)
  auto std_scale =
      MakeConstantLike(std_normal_real, 1.0 / std::sqrt(2.0), real_dtype);
  auto scaled_std = mlir::stablehlo::Mul(std_real, std_scale);

  // 6. Compute real and imag parts
  auto mul_real = mlir::stablehlo::Mul(std_normal_real, scaled_std);
  auto real_part = mlir::stablehlo::Add(mul_real, mean_real);
  auto mul_imag = mlir::stablehlo::Mul(std_normal_imag, scaled_std);
  auto imag_part = mlir::stablehlo::Add(mul_imag, mean_imag);

  // 7. Combine
  return mlir::stablehlo::Complex(real_part, imag_part);
}

// Unified entry point for building normal distribution graphs. Dispatches to
// the real or complex implementation based on the output data type.
absl::StatusOr<MlirOpResults<1>> BuildNormalShlo(
    mlir::MlirOp rng_state, std::optional<mlir::MlirOp> mean_op,
    std::optional<mlir::MlirOp> std_op, double mean_val, double std_val,
    llvm::ArrayRef<int64_t> out_dims, mlir::ElementType out_dtype) {
  if (mlir::IsComplex(out_dtype)) {
    return BuildNormalComplexShlo(rng_state, mean_op, std_op, mean_val, std_val,
                                  out_dims, out_dtype);
  }
  return BuildNormalRealShlo(rng_state, mean_op, std_op, mean_val, std_val,
                             out_dims, out_dtype);
}

absl::Status ValidateNormalPreconditions(const at::Tensor& tensor,
                                         std::string_view arg_name) {
  TT_RET_CHECK(IsFloatingPoint(tensor) || IsComplex(tensor),
               error::kInvalidArgument)
      << "expected the " << arg_name
      << " tensor to be floating point or complex type, got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

absl::Status ValidateNormalStdPreconditions(double std) {
  TT_RET_CHECK(std >= 0.0, error::kInvalidArgument)
      << "expected std >= 0.0, got " << std;
  return absl::OkStatus();
}

absl::Status ValidateNormalStdPreconditions(const at::Tensor& std,
                                            bool allow_integer = true) {
  TT_RET_CHECK(IsFloatingPoint(std) || (allow_integer && IsInteger(std)),
               error::kInvalidArgument)
      << "expected the std tensor to be "
      << (allow_integer ? "non-complex" : "floating point") << ", got "
      << ToString(std.scalar_type());
  if (std.numel() == 0) {
    return absl::OkStatus();
  }

  // Run min natively on TPU.
  TT_ASSIGN_OR_RETURN(DeviceBufferRef min_buf,
                      DispatchUnaryMinMax(std, MinMaxOp::kMin, OpName::kMin));
  TT_ASSIGN_OR_RETURN(at::Tensor cpu_min, TpuMemcpyDtoH(min_buf));
  TT_RET_CHECK(cpu_min.item<double>() >= 0.0, error::kInvalidArgument)
      << "expected all elements of std >= 0.0, got min element: "
      << cpu_min.item();

  return absl::OkStatus();
}

NAryMlirOpBuilder<1, 1> GetNormalScalarScalarBuilder(
    double mean, double std, llvm::ArrayRef<int64_t> out_dims,
    at::ScalarType out_dtype) {
  return [mean, std, out_dims = CopyIntVector(out_dims), out_dtype](
             mlir::MlirOp rng_state) -> absl::StatusOr<MlirOpResults<1>> {
    TT_ASSIGN_OR_RETURN(auto mlir_dtype,
                        ConvertTo<mlir::ElementType>(out_dtype));
    return BuildNormalShlo(rng_state,
                           /*mean_op=*/std::nullopt,
                           /*std_op=*/std::nullopt,
                           /*mean_val=*/mean,
                           /*std_val=*/std, out_dims, mlir_dtype);
  };
}

absl::StatusOr<DeviceBufferRef> DispatchNormal1(
    c10::optional<at::Generator> generator, NAryMlirOpBuilder<1, 1> builder,
    mlir::ElementType mlir_type, llvm::ArrayRef<int64_t> out_dims,
    OpParamCacheKeys param_keys) {
  return DispatchRngOpAndReturnBuffer(
      generator,
      [builder = std::move(builder), mlir_type, out_dims,
       param_keys = std::move(param_keys)](at::Tensor rng_input_state) mutable
          -> absl::StatusOr<std::vector<DeviceBufferRef>> {
        TT_ASSIGN_OR_RETURN(
            auto buf,
            (DispatchOp<1, 1>(std::move(builder), {rng_input_state},
                              {.out_dtype = mlir_type,
                               .out_dims = out_dims,
                               .op_param_cache_keys = std::move(param_keys)})));
        return std::vector<DeviceBufferRef>{std::move(buf)};
      });
}

absl::StatusOr<DeviceBufferRef> DispatchNormal2(
    c10::optional<at::Generator> generator, NAryMlirOpBuilder<2, 1> builder,
    const at::Tensor& input_tensor, mlir::ElementType mlir_type,
    llvm::ArrayRef<int64_t> out_dims, OpParamCacheKeys param_keys,
    Indices donated_indices = {}) {
  return DispatchRngOpAndReturnBuffer(
      generator,
      [builder = std::move(builder), input_tensor, mlir_type, out_dims,
       param_keys = std::move(param_keys),
       donated_indices =
           std::move(donated_indices)](at::Tensor rng_input_state) mutable
          -> absl::StatusOr<std::vector<DeviceBufferRef>> {
        TT_ASSIGN_OR_RETURN(
            auto buf, (DispatchOp<2, 1>(
                          std::move(builder), {input_tensor, rng_input_state},
                          {.out_dtype = mlir_type,
                           .out_dims = out_dims,
                           .op_param_cache_keys = std::move(param_keys),
                           .donated_indices = std::move(donated_indices)})));
        return std::vector<DeviceBufferRef>{std::move(buf)};
      });
}

absl::Status DispatchNormal2Out(c10::optional<at::Generator> generator,
                                NAryMlirOpBuilder<2, 1> builder,
                                const at::Tensor& input_tensor, at::Tensor& out,
                                mlir::ElementType mlir_type,
                                llvm::ArrayRef<int64_t> out_dims,
                                OpParamCacheKeys param_keys) {
  return DispatchRngOpOut(out, generator, mlir_type,
                          [&](at::Tensor rng_input_state) -> absl::Status {
                            return DispatchOpOut<2, 1>(
                                std::move(builder),
                                {input_tensor, rng_input_state}, out,
                                {.out_dtype = mlir_type,
                                 .out_dims = CopyIntVector(out_dims),
                                 .op_param_cache_keys = std::move(param_keys)});
                          });
}

NAryMlirOpBuilder<2, 1> GetNormalFloatTensorBuilder(
    double mean, llvm::ArrayRef<int64_t> out_dims, at::ScalarType out_dtype) {
  return [mean, out_dims = CopyIntVector(out_dims),
          out_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    auto [std_op, rng_state] = inputs;
    TT_ASSIGN_OR_RETURN(auto mlir_dtype,
                        ConvertTo<mlir::ElementType>(out_dtype));
    return BuildNormalShlo(rng_state,
                           /*mean_op=*/std::nullopt,
                           /*std_op=*/std_op,
                           /*mean_val=*/mean,
                           /*std_val=*/0.0, out_dims, mlir_dtype);
  };
}

NAryMlirOpBuilder<2, 1> GetNormalTensorFloatBuilder(
    double std, llvm::ArrayRef<int64_t> out_dims, at::ScalarType out_dtype) {
  return [std, out_dims = CopyIntVector(out_dims),
          out_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    auto [mean_op, rng_state] = inputs;
    TT_ASSIGN_OR_RETURN(auto mlir_dtype,
                        ConvertTo<mlir::ElementType>(out_dtype));
    return BuildNormalShlo(rng_state,
                           /*mean_op=*/mean_op,
                           /*std_op=*/std::nullopt,
                           /*mean_val=*/0.0,
                           /*std_val=*/std, out_dims, mlir_dtype);
  };
}

absl::StatusOr<DeviceBufferRef> DispatchNormal3(
    c10::optional<at::Generator> generator, NAryMlirOpBuilder<3, 1> builder,
    const at::Tensor& input_tensor1, const at::Tensor& input_tensor2,
    mlir::ElementType mlir_type, llvm::ArrayRef<int64_t> out_dims,
    OpParamCacheKeys param_keys, Indices donated_indices = {}) {
  return DispatchRngOpAndReturnBuffer(
      generator,
      [builder = std::move(builder), input_tensor1, input_tensor2, mlir_type,
       out_dims, param_keys = std::move(param_keys),
       donated_indices =
           std::move(donated_indices)](at::Tensor rng_input_state) mutable
          -> absl::StatusOr<std::vector<DeviceBufferRef>> {
        TT_ASSIGN_OR_RETURN(
            auto buf, (DispatchOp<3, 1>(
                          std::move(builder),
                          {input_tensor1, input_tensor2, rng_input_state},
                          {.out_dtype = mlir_type,
                           .out_dims = out_dims,
                           .op_param_cache_keys = std::move(param_keys),
                           .donated_indices = std::move(donated_indices)})));
        return std::vector<DeviceBufferRef>{std::move(buf)};
      });
}

absl::Status DispatchNormal3Out(c10::optional<at::Generator> generator,
                                NAryMlirOpBuilder<3, 1> builder,
                                const at::Tensor& input_tensor1,
                                const at::Tensor& input_tensor2,
                                at::Tensor& out, mlir::ElementType mlir_type,
                                llvm::ArrayRef<int64_t> out_dims,
                                OpParamCacheKeys param_keys) {
  return DispatchRngOpOut(out, generator, mlir_type,
                          [&](at::Tensor rng_input_state) -> absl::Status {
                            return DispatchOpOut<3, 1>(
                                std::move(builder),
                                {input_tensor1, input_tensor2, rng_input_state},
                                out,
                                {.out_dtype = mlir_type,
                                 .out_dims = CopyIntVector(out_dims),
                                 .op_param_cache_keys = std::move(param_keys)});
                          });
}

NAryMlirOpBuilder<3, 1> GetNormalTensorTensorBuilder(
    llvm::ArrayRef<int64_t> out_dims, at::ScalarType out_dtype) {
  return [out_dims = CopyIntVector(out_dims),
          out_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    auto [mean_op, std_op, rng_state] = inputs;
    TT_ASSIGN_OR_RETURN(auto mlir_dtype,
                        ConvertTo<mlir::ElementType>(out_dtype));
    return BuildNormalShlo(rng_state,
                           /*mean_op=*/mean_op,
                           /*std_op=*/std_op,
                           /*mean_val=*/0.0,
                           /*std_val=*/0.0, out_dims, mlir_dtype);
  };
}

}  // namespace

at::Tensor& AtenNormal_(at::Tensor& self, double mean, double std,
                        std::optional<at::Generator> generator) {
  TT_KERNEL(
      OpName::kNormal_, param_keys,
      (self, mean, std, IgnoreInCacheKey(generator, "Doesn't affect SHLO")), {
        TT_THROW_IF_ERROR(
            ValidateNormalPreconditions(self, /*arg_name=*/"self"));
        TT_THROW_IF_ERROR(ValidateNormalStdPreconditions(std));

        auto gen =
            generator.has_value() ? *generator : GetDefaultDeviceGenerator();

        auto out_dims = CopyIntVector(self.sizes());
        auto out_dtype = self.scalar_type();

        auto builder =
            GetNormalScalarScalarBuilder(mean, std, out_dims, out_dtype);

        TT_ASSIGN_OR_THROW(auto mlir_type,
                           ConvertTo<mlir::ElementType>(out_dtype));

        TT_ASSIGN_OR_THROW(auto output_buf,
                           DispatchNormal1(gen, std::move(builder), mlir_type,
                                           out_dims, std::move(param_keys)));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), self));
        return self;
      });
}

at::Tensor AtenNormalFloatTensor(double mean, const at::Tensor& std,
                                 std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kNormalFloatTensor, param_keys,
            (mean, std, IgnoreInCacheKey(generator, "Doesn't affect SHLO")), {
              TT_THROW_IF_ERROR(
                  // This variant (scalar mean, tensor std) does not allow
                  // integer std.
                  ValidateNormalStdPreconditions(std, /*allow_integer=*/false));

              auto gen = generator.has_value() ? *generator
                                               : GetDefaultDeviceGenerator();
              auto out_dims = CopyIntVector(std.sizes());
              auto out_dtype = std.scalar_type();

              auto builder =
                  GetNormalFloatTensorBuilder(mean, out_dims, out_dtype);

              TT_ASSIGN_OR_THROW(auto mlir_type,
                                 ConvertTo<mlir::ElementType>(out_dtype));

              TT_ASSIGN_OR_THROW(
                  auto output_buf,
                  DispatchNormal2(gen, std::move(builder), std, mlir_type,
                                  out_dims, std::move(param_keys)));

              return MakeTensor(std::move(output_buf));
            });
}

at::Tensor& AtenNormalFloatTensorOut(double mean, const at::Tensor& std,
                                     std::optional<at::Generator> generator,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kNormalFloatTensorOut, param_keys,
      (mean, std, IgnoreInCacheKey(generator, "Doesn't affect SHLO"), out), {
        TT_THROW_IF_ERROR(
            // This variant (scalar mean, tensor std) does not allow integer
            // std.
            ValidateNormalStdPreconditions(std, /*allow_integer=*/false));
        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, std.sizes()));
        TT_THROW_IF_ERROR(ValidateNormalPreconditions(out, /*arg_name=*/"out"));

        auto gen =
            generator.has_value() ? *generator : GetDefaultDeviceGenerator();
        auto out_dims = CopyIntVector(std.sizes());
        auto out_dtype = out.scalar_type();

        auto builder = GetNormalFloatTensorBuilder(mean, out_dims, out_dtype);

        TT_ASSIGN_OR_THROW(auto mlir_type,
                           ConvertTo<mlir::ElementType>(out_dtype));

        TT_THROW_IF_ERROR(DispatchNormal2Out(gen, std::move(builder), std, out,
                                             mlir_type, out_dims,
                                             std::move(param_keys)));
        return out;
      });
}

at::Tensor AtenNormalTensorFloat(const at::Tensor& mean, double std,
                                 std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kNormalTensorFloat, param_keys,
            (mean, std, IgnoreInCacheKey(generator, "Doesn't affect SHLO")), {
              TT_THROW_IF_ERROR(
                  ValidateNormalPreconditions(mean, /*arg_name=*/"mean"));
              TT_THROW_IF_ERROR(ValidateNormalStdPreconditions(std));

              auto gen = generator.has_value() ? *generator
                                               : GetDefaultDeviceGenerator();

              auto out_dims = CopyIntVector(mean.sizes());
              auto out_dtype = mean.scalar_type();

              auto builder =
                  GetNormalTensorFloatBuilder(std, out_dims, out_dtype);

              TT_ASSIGN_OR_THROW(auto mlir_type,
                                 ConvertTo<mlir::ElementType>(out_dtype));

              TT_ASSIGN_OR_THROW(
                  auto output_buf,
                  DispatchNormal2(gen, std::move(builder), mean, mlir_type,
                                  out_dims, std::move(param_keys)));

              return MakeTensor(std::move(output_buf));
            });
}

at::Tensor& AtenNormalTensorFloatOut(const at::Tensor& mean, double std,
                                     std::optional<at::Generator> generator,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kNormalTensorFloatOut, param_keys,
      (mean, std, IgnoreInCacheKey(generator, "Doesn't affect SHLO"), out), {
        TT_THROW_IF_ERROR(
            ValidateNormalPreconditions(mean, /*arg_name=*/"mean"));
        TT_THROW_IF_ERROR(ValidateNormalStdPreconditions(std));
        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, mean.sizes()));
        TT_THROW_IF_ERROR(ValidateNormalPreconditions(out, /*arg_name=*/"out"));

        auto gen =
            generator.has_value() ? *generator : GetDefaultDeviceGenerator();

        auto out_dims = CopyIntVector(mean.sizes());
        auto out_dtype = out.scalar_type();

        auto builder = GetNormalTensorFloatBuilder(std, out_dims, out_dtype);

        TT_ASSIGN_OR_THROW(auto mlir_type,
                           ConvertTo<mlir::ElementType>(out_dtype));

        TT_THROW_IF_ERROR(DispatchNormal2Out(gen, std::move(builder), mean, out,
                                             mlir_type, out_dims,
                                             std::move(param_keys)));
        return out;
      });
}

at::Tensor AtenNormalTensorTensor(const at::Tensor& mean, const at::Tensor& std,
                                  std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kNormalTensorTensor, _,
            (mean, std, IgnoreInCacheKey(generator, "Doesn't affect SHLO")), {
              TT_THROW_IF_ERROR(
                  ValidateNormalPreconditions(mean, /*arg_name=*/"mean"));
              TT_THROW_IF_ERROR(ValidateNormalStdPreconditions(std));

              auto gen = generator.has_value() ? *generator
                                               : GetDefaultDeviceGenerator();
              TT_ASSIGN_OR_THROW(auto out_dims, InferSize(mean, std));
              auto out_dtype = at::result_type(mean, std);

              auto builder = GetNormalTensorTensorBuilder(out_dims, out_dtype);

              TT_ASSIGN_OR_THROW(auto mlir_type,
                                 ConvertTo<mlir::ElementType>(out_dtype));

              TT_ASSIGN_OR_THROW(
                  auto output_buf,
                  DispatchNormal3(gen, std::move(builder), mean, std, mlir_type,
                                  out_dims, OpParamCacheKeys::Empty()));

              return MakeTensor(std::move(output_buf));
            });
}

at::Tensor& AtenNormalTensorTensorOut(const at::Tensor& mean,
                                      const at::Tensor& std,
                                      std::optional<at::Generator> generator,
                                      at::Tensor& out) {
  TT_KERNEL(
      OpName::kNormalTensorTensorOut, _,
      (mean, std, IgnoreInCacheKey(generator, "Doesn't affect SHLO"), out), {
        TT_THROW_IF_ERROR(
            ValidateNormalPreconditions(mean, /*arg_name=*/"mean"));
        TT_THROW_IF_ERROR(ValidateNormalStdPreconditions(std));
        TT_ASSIGN_OR_THROW(auto shape, InferSize(mean, std));
        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, shape));
        TT_THROW_IF_ERROR(ValidateNormalPreconditions(out, /*arg_name=*/"out"));

        auto gen =
            generator.has_value() ? *generator : GetDefaultDeviceGenerator();
        auto out_dims = CopyIntVector(out.sizes());
        auto out_dtype = out.scalar_type();

        auto builder = GetNormalTensorTensorBuilder(out_dims, out_dtype);

        TT_ASSIGN_OR_THROW(auto mlir_type,
                           ConvertTo<mlir::ElementType>(out_dtype));

        TT_THROW_IF_ERROR(DispatchNormal3Out(gen, std::move(builder), mean, std,
                                             out, mlir_type, out_dims,
                                             OpParamCacheKeys::Empty()));
        return out;
      });
}

}  // namespace torch_tpu
