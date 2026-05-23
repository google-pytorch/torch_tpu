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
#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/native/Resize.h"
#include "ATen/ops/broadcast_tensors.h"
#include "ATen/ops/scalar_tensor.h"
#include "ATen/ops/stack.h"
#include "ATen/ops/zeros_like.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
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
absl::StatusOr<MlirOpResults<2>> BuildStandardNormalShloLike(
    mlir::MlirOp self, mlir::MlirOp rng_state) {
  auto self_type = GetTensorTypeOrDie(self);
  TT_ASSIGN_OR_RETURN(auto mlir_type,
                      ConvertTo<mlir::ElementType>(self_type.getElementType()));

  // Use chlo::erfcinv to generate a standard normal distribution from a uniform
  // distribution.
  // Compute nextafter(-1, 0) in the target type to prevent erfinv(-1) = -inf.
  auto float_type = mlir::cast<mlir::FloatType>(self_type.getElementType());
  llvm::APFloat from_ap(float_type.getFloatSemantics(), "-1.0");
  from_ap.next(/*nextDown=*/false);
  double from = from_ap.convertToDouble();
  TT_ASSIGN_OR_RETURN(
      (auto [rng_output_state, uniform_op]),
      BuildUniformShlo(rng_state, from, 1.0, self_type.getShape(), mlir_type));
  auto erf_inv_op = mlir::chlo::ErfInv(uniform_op);
  auto two = MakeConstantLike(erf_inv_op, 2.0, mlir_type);
  auto sqrt_two = mlir::stablehlo::Sqrt(two);
  auto gaussian_op = mlir::stablehlo::Mul(erf_inv_op, sqrt_two);
  return {{rng_output_state, gaussian_op}};
}

// Generates a tensor of normal random numbers with the given shape, mean,
// standard deviation, and standard deviation scale (std_scale).
// The `std_scale` parameter ensures the variance of complex outputs is
// correct. Because PyTorch handles complex random numbers by independently
// generating normal variants for the real and imaginary components,
// `std_scale` is set to 1/sqrt(2) for complex types so that the resulting
// joint variance equals the requested standard deviation squared.
// For real types, `std_scale` is simply 1.0.
absl::StatusOr<MlirOpResults<2>> BuildNormalShloLike(mlir::MlirOp self,
                                                     mlir::MlirOp rng_state,
                                                     mlir::MlirOp mean,
                                                     mlir::MlirOp std,
                                                     mlir::MlirOp std_scale) {
  TT_ASSIGN_OR_RETURN((auto [rng_output_state, std_normal]),
                      BuildStandardNormalShloLike(self, rng_state));
  TT_ASSIGN_OR_RETURN(mean, BroadcastIfNeeded(mean, self));
  TT_ASSIGN_OR_RETURN(std, BroadcastIfNeeded(std, self));
  TT_ASSIGN_OR_RETURN(std_scale, BroadcastIfNeeded(std_scale, self));
  auto self_type = GetTensorTypeOrDie(self);
  TT_ASSIGN_OR_RETURN(auto mlir_type,
                      ConvertTo<mlir::ElementType>(self_type.getElementType()));
  TT_ASSIGN_OR_RETURN(mean, CastIfNeeded(mean, mlir_type));
  TT_ASSIGN_OR_RETURN(std, CastIfNeeded(std, mlir_type));
  TT_ASSIGN_OR_RETURN(std_scale, CastIfNeeded(std_scale, mlir_type));
  auto std_scaled = mlir::stablehlo::Mul(std, std_scale);
  auto normal_with_variance = mlir::stablehlo::Mul(std_normal, std_scaled);
  auto normal = mlir::stablehlo::Add(normal_with_variance, mean);
  return {{rng_output_state, normal}};
}

absl::StatusOr<NAryMlirOpBuilder<5, 2>> GetNormalFunctional() {
  return [](FixedSizeSpan<mlir::MlirOp, 5> inputs)
             -> absl::StatusOr<MlirOpResults<2>> {
    auto [self, rng_state, mean, std, std_scale] = inputs;
    return BuildNormalShloLike(self, rng_state, mean, std, std_scale);
  };
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
  if (std.numel() == 0) {
    return absl::OkStatus();
  }
  // Note: checking that all elements are >= 0.0 requires accessing the tensor
  // elements via .item(), which causes a synchronous sync between TPU and CPU.
  at::Tensor min = std.min();
  TT_RET_CHECK(min.ge(0).item<bool>(), error::kInvalidArgument)
      << "expected all elements of std >= 0.0, got min element: " << min.item();
  return absl::OkStatus();
}

// Retrieves the rng_state tensor from the generator, dispatches the normal op,
// updates the generator with the new rng_state, and returns the output tensor.
absl::StatusOr<DeviceBufferRef> NormalLike(
    const at::Tensor& self, const at::Tensor& mean, const at::Tensor& std,
    std::optional<at::Generator> generator) {
  at::Tensor self_real = self.is_complex() ? AtenViewAsReal(self) : self;
  at::Tensor mean_real =
      mean.is_complex()
          ? AtenViewAsReal(mean)
          : (self.is_complex() ? at::stack({mean, at::zeros_like(mean)}, -1)
                               : mean);
  // std should never be complex naturally for normal, but it might have been
  // promoted by broadcasting. If so, we only want its real part (imaginary
  // part is 0.0), and we then unsqueeze it so it can be broadcasted to both
  // real and imaginary parts of the output.
  at::Tensor std_real =
      std.is_complex() ? AtenViewAsReal(std).select(-1, 0) : std;
  if (self.is_complex()) {
    std_real = std_real.unsqueeze(-1);
  }
  at::Tensor std_scale = at::scalar_tensor(
      self.is_complex() ? 1.0 / std::sqrt(2.0) : 1.0, self_real.options());

  TT_ASSIGN_OR_RETURN(auto mlir_type,
                      ConvertTo<mlir::ElementType>(self_real.scalar_type()));

  TT_ASSIGN_OR_RETURN(auto builder, GetNormalFunctional());
  return DispatchRngOpAndReturnBuffer(
      generator, [builder = std::move(builder), self_real, mean_real, std_real,
                  std_scale, mlir_type](at::Tensor rng_input_state) mutable {
        return DispatchOp<5, 2>(
            std::move(builder),
            {self_real, rng_input_state, mean_real, std_real, std_scale},
            {.out_dtypes = {mlir::ElementType::UI64, mlir_type},
             .out_dims_list = {{2}, self_real.sizes()},
             .op_param_cache_keys = OpParamCacheKeys::Empty(),
             .split_mode = OpSplitMode::kSplitAfter});
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
        at::Tensor mean_tensor = at::scalar_tensor(mean, self.options());
        at::Tensor std_tensor = at::scalar_tensor(std, self.options());
        TT_ASSIGN_OR_THROW(auto output_buf, NormalLike(self, mean_tensor,
                                                       std_tensor, generator));
        if (self.is_complex()) {
          at::Tensor real_imag = AtenViewAsReal(self);
          TT_THROW_IF_ERROR(
              AssignBufferToAtTensor(std::move(output_buf), real_imag));
        } else {
          TT_THROW_IF_ERROR(
              AssignBufferToAtTensor(std::move(output_buf), self));
        }
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
                  // This variant (scalar mean, tensor std) does not allow
                  // integer std.
                  CheckNormalStdPreconditions(std, /*allow_integer=*/false));
              at::Tensor mean_tensor = at::scalar_tensor(mean, std.options());
              TT_ASSIGN_OR_THROW(auto output_buf,
                                 NormalLike(std, mean_tensor, std, generator));
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
            // This variant (scalar mean, tensor std) does not allow integer
            // std.
            CheckNormalStdPreconditions(std, /*allow_integer=*/false));
        at::native::resize_output(out, std.sizes());
        TT_THROW_IF_ERROR(CheckNormalPreconditions(out, /*arg_name=*/"out"));
        at::Tensor mean_tensor = at::scalar_tensor(mean, out.options());
        TT_ASSIGN_OR_THROW(auto output_buf,
                           NormalLike(out, mean_tensor, std, generator));
        if (out.is_complex()) {
          at::Tensor out_real_imag = AtenViewAsReal(out);
          TT_THROW_IF_ERROR(
              AssignBufferToAtTensor(std::move(output_buf), out_real_imag));
        } else {
          TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), out));
        }
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
        at::Tensor std_tensor = at::scalar_tensor(std, mean.options());
        TT_ASSIGN_OR_THROW(auto output_buf,
                           NormalLike(mean, mean, std_tensor, generator));
        at::Tensor res = MakeTensor(std::move(output_buf));
        return mean.is_complex() ? AtenViewAsComplex(res) : res;
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
        at::Tensor std_tensor = at::scalar_tensor(std, out.options());
        TT_ASSIGN_OR_THROW(auto output_buf,
                           NormalLike(out, mean, std_tensor, generator));
        if (out.is_complex()) {
          at::Tensor out_real_imag = AtenViewAsReal(out);
          TT_THROW_IF_ERROR(
              AssignBufferToAtTensor(std::move(output_buf), out_real_imag));
        } else {
          TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), out));
        }
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
        // ATen's normal_impl_ uses standard broadcasting via
        // TensorIterator or infer_size, despite documentation stating
        // that inputs are not broadcasted. We follow ATen's functional
        // behavior here.
        auto broadcasted = at::broadcast_tensors({mean, std});
        const at::Tensor& b_mean = broadcasted[0];
        const at::Tensor& b_std = broadcasted[1];
        TT_ASSIGN_OR_THROW(auto output_buf,
                           NormalLike(b_mean, b_mean, b_std, generator));
        at::Tensor res = MakeTensor(std::move(output_buf));
        return b_mean.is_complex() ? AtenViewAsComplex(res) : res;
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
        // ATen's normal_impl_ uses standard broadcasting via TensorIterator or
        // infer_size, despite documentation stating that inputs are not
        // broadcasted. We follow ATen's functional behavior here.
        auto broadcasted = at::broadcast_tensors({mean, std});
        const at::Tensor& b_mean = broadcasted[0];
        const at::Tensor& b_std = broadcasted[1];
        TT_ASSIGN_OR_THROW(auto output_buf,
                           NormalLike(out, b_mean, b_std, generator));
        if (out.is_complex()) {
          at::Tensor out_real_imag = AtenViewAsReal(out);
          TT_THROW_IF_ERROR(
              AssignBufferToAtTensor(std::move(output_buf), out_real_imag));
        } else {
          TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), out));
        }
        return out;
      });
}

}  // namespace torch_tpu
