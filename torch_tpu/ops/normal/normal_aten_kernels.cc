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

#include <optional>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_gen_impl.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/uniform/uniform.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// This function generates a vector of standard normal random numbers using the
// chlo::ErfInv function.// We control the source of our randomness by keeping
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

// Generate a tensor of normal random numbers with the given shape, mean, and
// standard deviation.
absl::StatusOr<MlirOpResults<2>> BuildNormalShloLike(mlir::MlirOp self,
                                                     mlir::MlirOp rng_state,
                                                     mlir::MlirOp mean,
                                                     mlir::MlirOp std) {
  TT_ASSIGN_OR_RETURN((auto [rng_output_state, std_normal]),
                      BuildStandardNormalShloLike(self, rng_state));
  TT_ASSIGN_OR_RETURN(mean, BroadcastIfNeeded(mean, self));
  TT_ASSIGN_OR_RETURN(std, BroadcastIfNeeded(std, self));
  auto self_type = GetTensorTypeOrDie(self);
  TT_ASSIGN_OR_RETURN(auto mlir_type,
                      ConvertTo<mlir::ElementType>(self_type.getElementType()));
  TT_ASSIGN_OR_RETURN(mean, CastIfNeeded(mean, mlir_type));
  TT_ASSIGN_OR_RETURN(std, CastIfNeeded(std, mlir_type));
  auto normal_with_variance = mlir::stablehlo::Mul(std_normal, std);
  auto normal = mlir::stablehlo::Add(normal_with_variance, mean);
  return {{rng_output_state, normal}};
}

absl::StatusOr<NAryMlirOpBuilder<4, 2>> GetNormalFunctional() {
  return [](FixedSizeSpan<mlir::MlirOp, 4> inputs)
             -> absl::StatusOr<MlirOpResults<2>> {
    auto [self, rng_state, mean, std] = inputs;
    return BuildNormalShloLike(self, rng_state, mean, std);
  };
}

absl::Status CheckInputIsFloatingPoint(const at::Tensor& tensor,
                                       const std::string_view arg_name) {
  TT_RET_CHECK(IsFloatingPoint(tensor), error::kInvalidArgument)
      << "expected the " << arg_name << " tensor to be floating point, got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

// Retrieves the rng_state tensor from the generator, dispatches the normal op,
// updates the generator with the new rng_state, and returns the output tensor.
absl::StatusOr<DeviceBufferRef> NormalLike(
    const at::Tensor& self, OpName op_name, const at::Tensor& mean,
    const at::Tensor& std, std::optional<at::Generator> generator) {
  TT_ASSIGN_OR_RETURN(auto mlir_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  // Retrieve the rng_state tensor from the generator.
  TT_ASSIGN_OR_RETURN(auto rng_input_state, GetRngState(generator));
  TT_ASSIGN_OR_RETURN(auto builder, GetNormalFunctional());
  TT_ASSIGN_OR_RETURN(
      (auto [rng_output_state_buf, output_buf]),
      (DispatchOp<4, 2>(OpName::kNormal_, std::move(builder),
                        {self, rng_input_state, mean, std},
                        {.out_dtypes = {mlir::ElementType::UI64, mlir_type},
                         .out_dims_list = {{2}, self.sizes()},
                         .op_param_cache_keys = OpParamCacheKeys::Empty(),
                         .split_mode = OpSplitMode::kSplitAfter})));
  // After the state has been used (and updated) to generate random bits, we
  // give it back to the generator, so that it can be used by other ops in the
  // same graph.
  auto rng_output_state = MakeTensor(std::move(rng_output_state_buf));
  TT_RETURN_IF_ERROR(UpdateRngState(generator, rng_output_state));
  return output_buf;
}

}  // namespace

at::Tensor& AtenNormal_(at::Tensor& self, double mean, double std,
                        std::optional<at::Generator> generator) {
  TT_KERNEL(
      OpName::kNormal_, _,
      (self, IgnoreInCacheKey(mean), IgnoreInCacheKey(std),
       IgnoreInCacheKey(generator)),
      {
        TT_THROW_IF_ERROR(
            CheckInputIsFloatingPoint(self, /* arg_name= */ "self"));
        TT_ASSIGN_OR_THROW(at::Tensor mean_tensor,
                           MakeTensor(mean, self.scalar_type()));
        TT_ASSIGN_OR_THROW(at::Tensor std_tensor,
                           MakeTensor(std, self.scalar_type()));
        TT_ASSIGN_OR_THROW(auto output_buf,
                           NormalLike(self, OpName::kNormal_, mean_tensor,
                                      std_tensor, generator));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), self));
        return self;
      });
}

at::Tensor AtenNormalFloatTensor(double mean, const at::Tensor& std,
                                 std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kNormalFloatTensor, _,
            (IgnoreInCacheKey(mean), std, IgnoreInCacheKey(generator)), {
              TT_THROW_IF_ERROR(
                  CheckInputIsFloatingPoint(std, /* arg_name= */ "std"));
              TT_ASSIGN_OR_THROW(at::Tensor mean_tensor,
                                 MakeTensor(mean, std.scalar_type()));
              TT_ASSIGN_OR_THROW(auto output_buf,
                                 NormalLike(std, OpName::kNormalFloatTensor,
                                            mean_tensor, std, generator));
              return MakeTensor(std::move(output_buf));
            });
}

at::Tensor& AtenNormalFloatTensorOut(double mean, const at::Tensor& std,
                                     std::optional<at::Generator> generator,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kNormalFloatTensorOut, _,
      (IgnoreInCacheKey(mean), std, IgnoreInCacheKey(generator), out), {
        TT_THROW_IF_ERROR(
            CheckInputIsFloatingPoint(out, /* arg_name= */ "out"));
        TT_ASSIGN_OR_THROW(at::Tensor mean_tensor,
                           MakeTensor(mean, out.scalar_type()));
        TT_ASSIGN_OR_THROW(auto output_buf,
                           NormalLike(out, OpName::kNormalFloatTensorOut,
                                      mean_tensor, std, generator));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), out));
        return out;
      });
}

at::Tensor AtenNormalTensorFloat(const at::Tensor& mean, double std,
                                 std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kNormalTensorFloat, _,
            (mean, IgnoreInCacheKey(std), IgnoreInCacheKey(generator)), {
              TT_THROW_IF_ERROR(
                  CheckInputIsFloatingPoint(mean, /* arg_name= */ "mean"));
              TT_ASSIGN_OR_THROW(at::Tensor std_tensor,
                                 MakeTensor(std, mean.scalar_type()));
              TT_ASSIGN_OR_THROW(auto output_buf,
                                 NormalLike(mean, OpName::kNormalTensorFloat,
                                            mean, std_tensor, generator));
              return MakeTensor(std::move(output_buf));
            });
}

at::Tensor& AtenNormalTensorFloatOut(const at::Tensor& mean, double std,
                                     std::optional<at::Generator> generator,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kNormalTensorFloatOut, _,
      (mean, IgnoreInCacheKey(std), IgnoreInCacheKey(generator), out), {
        TT_THROW_IF_ERROR(
            CheckInputIsFloatingPoint(out, /* arg_name= */ "out"));
        TT_ASSIGN_OR_THROW(at::Tensor std_tensor,
                           MakeTensor(std, out.scalar_type()));
        TT_ASSIGN_OR_THROW(auto output_buf,
                           NormalLike(out, OpName::kNormalTensorFloatOut, mean,
                                      std_tensor, generator));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), out));
        return out;
      });
}

at::Tensor AtenNormalTensorTensor(const at::Tensor& mean, const at::Tensor& std,
                                  std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kNormalTensorTensor, _,
            (mean, std, IgnoreInCacheKey(generator)), {
              TT_THROW_IF_ERROR(
                  CheckInputIsFloatingPoint(mean, /* arg_name= */ "mean"));
              TT_THROW_IF_ERROR(
                  CheckInputIsFloatingPoint(std, /* arg_name= */ "std"));
              TT_ASSIGN_OR_THROW(auto output_buf,
                                 NormalLike(mean, OpName::kNormalTensorTensor,
                                            mean, std, generator));
              return MakeTensor(std::move(output_buf));
            });
}

at::Tensor& AtenNormalTensorTensorOut(const at::Tensor& mean,
                                      const at::Tensor& std,
                                      std::optional<at::Generator> generator,
                                      at::Tensor& out) {
  TT_KERNEL(
      OpName::kNormalTensorTensorOut, _,
      (mean, std, IgnoreInCacheKey(generator), out), {
        TT_THROW_IF_ERROR(
            CheckInputIsFloatingPoint(out, /* arg_name= */ "out"));
        TT_ASSIGN_OR_THROW(auto output_buf,
                           NormalLike(out, OpName::kNormalTensorTensorOut, mean,
                                      std, generator));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
