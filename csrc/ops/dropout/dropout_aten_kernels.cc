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

#include "csrc/ops/dropout/dropout_aten_kernels.h"

#include <cstdint>
#include <memory>  // IWYU pragma: keep
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "ATen/Context.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty_like.h"
#include "ATen/ops/native_dropout.h"
#include "ATen/ops/zeros.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/GradMode.h"
#include "c10/core/ScalarType.h"
#include "c10/core/impl/LocalDispatchKeySet.h"
#include "c10/util/Optional.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/dropout/dropout.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/rng_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/autograd/VariableTypeUtils.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/generated/Functions.h"  // IWYU pragma: keep
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

void ValidateDropoutInputs(const at::Tensor& input, double p) {
  TT_CHECK_THROW(p >= 0.0 && p <= 1.0, error::kInvalidArgument)
      << "expected p to be in the range [0, 1], got " << p;
  TT_CHECK_THROW(IsFloatingPoint(input) || IsComplex(input),
                 error::kInvalidArgument)
      << "expected input to be floating point or complex, got "
      << ToString(input.scalar_type());
}

// Builds the SHLO for dropout in the train mode.
//
// Parameters:
//   input: The input tensor.
//   p: The dropout rate.
// Returns:
//   A pair of MlirOpResults, the first for the output tensor and the second for
//   the mask tensor.
NAryMlirOpBuilder<2, 2> GetDropoutFunctional(const at::Tensor& input,
                                             double p) {
  return [p](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<MlirOpResults<2>> {
    auto& [rng_state, input] = inputs;
    return BuildDropoutTrainShlo(rng_state, input, p);
  };
}

// Builds the 1-output SHLO for dropout in train mode (discarding the boolean
// mask in MLIR so no mask buffer is materialized on device).
NAryMlirOpBuilder<2, 1> GetDropoutOutputOnlyFunctional(double p) {
  return [p](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<mlir::MlirOp> {
    auto& [rng_state, input_op] = inputs;
    TT_ASSIGN_OR_RETURN(MlirOpResults<2> results,
                        BuildDropoutTrainShlo(rng_state, input_op, p));
    return results[0];
  };
}

bool IsFakeOrFunctorchTensor(const at::Tensor& tensor) {
  const auto include_set = c10::impl::tls_local_dispatch_key_set().included_;
  if (include_set.has(c10::DispatchKey::FuncTorchDynamicLayerFrontMode) ||
      include_set.has(c10::DispatchKey::FuncTorchDynamicLayerBackMode) ||
      !tensor.has_storage()) {
    return true;
  }
  return tensor.storage().data_ptr() == nullptr;
}

bool IsActiveValidTrainDropout(const at::Tensor& input, double p, bool train) {
  return p > 0.0 && p <= 1.0 && train &&
         (IsFloatingPoint(input) || IsComplex(input)) && input.sym_numel() != 0;
}

// Composite fallback for FakeTensor (torch.compile), functorch transforms, and
// forward-mode AD when dropout is active (0 < p <= 1, train=true, non-empty).
at::Tensor DropoutCompositeFallback(const at::Tensor& input, double p) {
  if (p >= 1.0) {
    return input.mul(at::zeros({}, input.options()));
  }
  if (!IsComplex(input) && !input._fw_grad(/*level=*/0).defined()) {
    return std::get<0>(at::native_dropout(input, p, /*train=*/true));
  }
  const c10::ScalarType noise_dtype =
      IsComplex(input) ? c10::toRealValueType(input.scalar_type())
                       : input.scalar_type();
  at::Tensor noise = at::empty_like(input, input.options().dtype(noise_dtype));
  noise.bernoulli_(1.0 - p);
  noise.div_(1.0 - p);
  return input.mul(noise);
}

// Creates an autograd `Node` smart pointer (`c10::intrusive_ptr<T>` in PyTorch
// 2.12+ where `Node` inherits from `c10::intrusive_ptr_target`, or
// `std::shared_ptr<T>` with `torch::autograd::deleteNode` in earlier PyTorch
// versions used by OSS builds).
template <typename T>
auto MakeAutogradNode() {
  if constexpr (std::is_base_of_v<c10::intrusive_ptr_target, T>) {
    return c10::make_intrusive<T>();
  } else {
    return std::shared_ptr<T>(new T(), [](T* ptr) { deleteNode(ptr); });
  }
}

// Common helper function for 2-output forward dropout operations
// (AtenNativeDropout and AtenFusedDropout). Validates input types and
// probability range, handles edge-case short circuits for p <= 0 and p >= 1,
// and dispatches the SHLO RNG dropout kernel.
std::tuple<at::Tensor, at::Tensor> AtenDropoutCommonImpl(
    const at::Tensor& input, double p, bool train,
    c10::optional<at::Generator> generator, OpParamCacheKeys param_keys) {
  ValidateDropoutInputs(input, p);

  TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                     ConvertTo<mlir::ElementType>(input.scalar_type()));

  if (p <= 0.0 || !train || input.numel() == 0) {
    TT_ASSIGN_OR_THROW(DeviceBufferRef input_buf, GetBuffer(input));
    auto op_builder =
        [](mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
      auto& builder = input_op.getBuilder();
      return MakeConstantLike(input_op, true,
                              builder.getOpBuilder().getI1Type());
    };
    TT_ASSIGN_OR_THROW(
        DeviceBufferRef mask_buf,
        (DispatchOp<1, 1>(std::move(op_builder), input,
                          {.out_dtype = mlir::ElementType::PRED,
                           .out_dims = input.sizes(),
                           .op_param_cache_keys = std::move(param_keys)})));
    return {MakeTensor(std::move(input_buf)), MakeTensor(std::move(mask_buf))};
  }
  if (p >= 1.0) {
    auto op_builder =
        [](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<2>> {
      auto& builder = input.getBuilder();
      auto zero_const = MakeConstantLike(input, 0.0);
      auto mask_op =
          MakeConstantLike(input, false, builder.getOpBuilder().getI1Type());
      return {{zero_const, mask_op}};
    };
    TT_ASSIGN_OR_THROW(
        auto results,
        (DispatchOp<1, 2>(
            std::move(op_builder), input,
            {.out_dtypes = {output_dtype, mlir::ElementType::PRED},
             .out_dims_list = {input.sizes(), input.sizes()},
             .op_param_cache_keys = std::move(param_keys)})));
    return {MakeTensor(std::move(results[0])),
            MakeTensor(std::move(results[1]))};
  }

  const int64_t num_elements = input.numel();
  const int64_t bit_width = TorchEquivalentBitwidth(output_dtype);

  TT_ASSIGN_OR_THROW(
      auto results,
      DispatchRngOpGeneral(
          generator,
          [&](at::Tensor rng_input_state) {
            return DispatchOp<2, 2>(
                GetDropoutFunctional(input, p), {rng_input_state, input},
                {.out_dtypes = {output_dtype, mlir::ElementType::PRED},
                 .out_dims_list = {input.sizes(), input.sizes()},
                 .op_param_cache_keys = std::move(param_keys)});
          },
          RngUsage{num_elements, bit_width}));

  // [output, mask]
  return {MakeTensor(std::move(results[0])), MakeTensor(std::move(results[1]))};
}

}  // namespace

at::Tensor AtenDropout(const at::Tensor& input, double p, bool train) {
  if (IsActiveValidTrainDropout(input, p, train) &&
      IsFakeOrFunctorchTensor(input)) {
    return DropoutCompositeFallback(input, p);
  }
  TT_KERNEL(
      OpName::kDropout, param_keys,
      (input, p,
       IgnoreInCacheKey(train,
                        "train=false returns early without dispatching SHLO")),
      {
        ValidateDropoutInputs(input, p);
        if (p <= 0.0 || !train || input.sym_numel() == 0) {
          return input;
        }
        TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));
        if (p >= 1.0) {
          auto op_builder =
              [](mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
            return MakeConstantLike(input_op, 0.0);
          };
          TT_ASSIGN_OR_THROW(DeviceBufferRef output_buf,
                             (DispatchOp<1, 1>(std::move(op_builder), input,
                                               {.out_dtype = output_dtype,
                                                .out_dims = input.sizes(),
                                                .op_param_cache_keys =
                                                    std::move(param_keys)})));
          return MakeTensor(std::move(output_buf));
        }
        const int64_t num_elements = input.numel();
        const int64_t bit_width = TorchEquivalentBitwidth(output_dtype);
        TT_ASSIGN_OR_THROW(
            DeviceBufferRef output_buf,
            DispatchRngOpAndReturnBuffer(
                /*generator=*/std::nullopt,
                [&](at::Tensor rng_input_state)
                    -> absl::StatusOr<std::vector<DeviceBufferRef>> {
                  TT_ASSIGN_OR_RETURN(
                      DeviceBufferRef buf,
                      (DispatchOp<2, 1>(
                          GetDropoutOutputOnlyFunctional(p),
                          {rng_input_state, input},
                          {.out_dtype = output_dtype,
                           .out_dims = input.sizes(),
                           .op_param_cache_keys = std::move(param_keys)})));
                  return std::vector<DeviceBufferRef>{std::move(buf)};
                },
                RngUsage{num_elements, bit_width}));
        return MakeTensor(std::move(output_buf));
      });
}

at::Tensor AtenDropoutAutograd(const at::Tensor& input, double p, bool train) {
  if (!IsActiveValidTrainDropout(input, p, train)) {
    c10::impl::ExcludeDispatchKeyGuard no_autograd(
        c10::autograd_dispatch_keyset);
    return AtenDropout(input, p, train);
  }
  if (IsFakeOrFunctorchTensor(input) || input._fw_grad(/*level=*/0).defined()) {
    return DropoutCompositeFallback(input, p);
  }
  if (!c10::GradMode::is_enabled() || !input.requires_grad()) {
    c10::impl::ExcludeDispatchKeyGuard no_autograd(
        c10::autograd_dispatch_keyset);
    return AtenDropout(input, p, train);
  }
  if (IsComplex(input)) {
    using NativeDropoutBackwardNode =
        torch::autograd::generated::NativeDropoutBackward0;
    auto grad_fn = MakeAutogradNode<NativeDropoutBackwardNode>();
    grad_fn->set_next_edges(torch::autograd::collect_next_edges(input));
    grad_fn->p = p;
    grad_fn->train = train;
    auto [output, mask] = ([&]() {
      at::AutoDispatchBelowADInplaceOrView guard;
      return at::native_dropout(input, p, train);
    })();
    torch::autograd::set_history(torch::autograd::flatten_tensor_args(output),
                                 grad_fn);
    grad_fn->result1_ =
        torch::autograd::SavedVariable(mask, /*is_output=*/true);
    return output;
  }
  return std::get<0>(at::native_dropout(input, p, train));
}

at::Tensor& AtenDropout_(at::Tensor& self, double p, bool train) {
  if (IsActiveValidTrainDropout(self, p, train) &&
      IsFakeOrFunctorchTensor(self)) {
    return self.copy_(DropoutCompositeFallback(self, p));
  }
  TT_KERNEL(OpName::kDropout_, param_keys,
            (self, p,
             IgnoreInCacheKey(
                 train, "train=false returns early without dispatching SHLO")),
            {
              ValidateDropoutInputs(self, p);
              if (p <= 0.0 || !train || self.sym_numel() == 0) {
                return self;
              }
              TT_ASSIGN_OR_THROW(
                  mlir::ElementType output_dtype,
                  ConvertTo<mlir::ElementType>(self.scalar_type()));
              if (p >= 1.0) {
                auto op_builder =
                    [](mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
                  return MakeConstantLike(input_op, 0.0);
                };
                TT_THROW_IF_ERROR((DispatchOpOut<1, 1>(
                    std::move(op_builder), self, self,
                    {.out_dtype = output_dtype,
                     .out_dims = self.sizes(),
                     .op_param_cache_keys = std::move(param_keys)})));
                return self;
              }
              const int64_t num_elements = self.numel();
              const int64_t bit_width = TorchEquivalentBitwidth(output_dtype);
              TT_THROW_IF_ERROR(DispatchRngOpOut(
                  self, /*generator=*/std::nullopt, output_dtype,
                  [&](at::Tensor rng_input_state) -> absl::Status {
                    return DispatchOpOut<2, 1>(
                        GetDropoutOutputOnlyFunctional(p),
                        {rng_input_state, self}, self,
                        {.out_dtype = output_dtype,
                         .out_dims = self.sizes(),
                         .op_param_cache_keys = std::move(param_keys)});
                  },
                  RngUsage{num_elements, bit_width}));
              return self;
            });
}

at::Tensor& AtenDropout_Autograd(at::Tensor& self, double p, bool train) {
  if (!IsActiveValidTrainDropout(self, p, train)) {
    c10::impl::ExcludeDispatchKeyGuard no_autograd(
        c10::autograd_dispatch_keyset);
    return AtenDropout_(self, p, train);
  }
  if (IsFakeOrFunctorchTensor(self) || self._fw_grad(/*level=*/0).defined()) {
    return self.copy_(DropoutCompositeFallback(self, p));
  }
  if (!c10::GradMode::is_enabled() || !self.requires_grad()) {
    c10::impl::ExcludeDispatchKeyGuard no_autograd(
        c10::autograd_dispatch_keyset);
    return AtenDropout_(self, p, train);
  }
  return self.copy_(AtenDropoutAutograd(self, p, train));
}

std::tuple<at::Tensor, at::Tensor> AtenNativeDropout(
    const at::Tensor& input, double p, c10::optional<bool> train) {
  TT_KERNEL(
      OpName::kNativeDropout, param_keys,
      (input, p,
       IgnoreInCacheKey(
           train,
           "train=false dispatches 1-output mask SHLO while train=true with "
           "p>0 dispatches 2-output SHLO")),
      {
        return AtenDropoutCommonImpl(input, p, train.value_or(true),
                                     /*generator=*/std::nullopt,
                                     std::move(param_keys));
      });
}

std::tuple<at::Tensor, at::Tensor> AtenFusedDropout(
    const at::Tensor& self, double p, c10::optional<at::Generator> generator) {
  TT_KERNEL(
      OpName::kFusedDropout, param_keys,
      (self, p,
       IgnoreInCacheKey(
           generator,
           "Generator state is passed as an input tensor and does not affect "
           "SHLO")),
      {
        return AtenDropoutCommonImpl(self, p, /*train=*/true, generator,
                                     std::move(param_keys));
      });
}

namespace {

NAryMlirOpBuilder<2, 1> GetDropoutBackwardFunctional(double scale) {
  return [scale](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    auto& [grad_output, mask] = inputs;
    return BuildDropoutBackwardShlo(grad_output, mask, scale);
  };
}

}  // namespace

at::Tensor AtenNativeDropoutBackward(const at::Tensor& grad_output,
                                     const at::Tensor& mask, double scale) {
  TT_KERNEL(
      OpName::kNativeDropoutBackward, param_keys, (grad_output, mask, scale), {
        TT_CHECK_THROW(grad_output.scalar_type() != at::kBool &&
                           grad_output.scalar_type() != at::kInt &&
                           grad_output.scalar_type() != at::kLong,
                       error::kPythonNotImplementedError)
            << "expected grad_output dtype not to be bool, int32, or int64, "
               "got "
            << ToString(grad_output.scalar_type());
        TT_CHECK_THROW(mask.scalar_type() == at::kBool, error::kInvalidArgument)
            << "expected mask to be bool, got " << ToString(mask.scalar_type());
        at::ScalarType output_scalar_type = grad_output.scalar_type();
        if (!c10::isFloatingType(output_scalar_type) &&
            !c10::isComplexType(output_scalar_type)) {
          output_scalar_type = at::kFloat;
        }
        TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(output_scalar_type));
        TT_ASSIGN_OR_THROW(
            auto output_buf,
            (DispatchOp<2, 1>(GetDropoutBackwardFunctional(scale),
                              {grad_output, mask},
                              {.out_dtype = output_dtype,
                               .out_dims = grad_output.sizes(),
                               .op_param_cache_keys = std::move(param_keys)})));
        return MakeTensor(std::move(output_buf));
      });
}

}  // namespace torch_tpu
