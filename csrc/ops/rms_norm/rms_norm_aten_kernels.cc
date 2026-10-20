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

#include "csrc/ops/rms_norm/rms_norm_aten_kernels.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/OpMathType.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBase.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/_fused_rms_norm.h"
#include "ATen/ops/pow.h"
#include "ATen/ops/rsqrt.h"
#include "ATen/ops/zeros.h"
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/GradMode.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/core/impl/LocalDispatchKeySet.h"
#include "csrc/common/const_tensor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fake_tensor_utils.h"
#include "csrc/common/op_name_stack.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/layer_norm/layer_norm.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/rms_norm/rms_norm.h"
#include "csrc/ops/to_copy/to_copy_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

double GetRmsNormEpsilon(std::optional<double> eps,
                         c10::ScalarType input_dtype) {
  if (eps.has_value()) {
    return *eps;
  }
  if (ToAccumulateType(input_dtype) == at::ScalarType::Double) {
    return std::numeric_limits<double>::epsilon();
  }
  return static_cast<double>(std::numeric_limits<float>::epsilon());
}

// Takes ConstTensor so that the validation neither throws on nor specializes
// the symbolic shapes of the FakeTensors that torch.compile traces with.
absl::Status ValidateRmsNormInputs(const ConstTensor& input,
                                   at::IntArrayRef normalized_shape,
                                   const std::optional<ConstTensor>& weight) {
  const bool input_is_empty = input.sym_numel() == 0;
  TT_RET_CHECK(input_is_empty || input.is_floating_point(),
               error::kInvalidArgument)
      << "expected the input dtype to be floating point, got "
      << ToString(input.scalar_type());

  TT_RET_CHECK(!normalized_shape.empty(), error::kInvalidArgument)
      << "the normalized shape must have >= 1 dimensions";

  const int64_t normalized_ndim = normalized_shape.size();
  TT_ASSIGN_OR_RETURN(const int64_t input_ndim, input.dim());
  TT_RET_CHECK(normalized_ndim <= input_ndim, error::kInvalidArgument)
      << "expected the normalized shape to have <= " << input_ndim
      << " dimensions, got " << normalized_ndim;

  TT_ASSIGN_OR_RETURN(const c10::SymIntArrayRef input_sizes, input.sym_sizes());
  // Like upstream, this guards only on the normalized (trailing) dimensions.
  TT_RET_CHECK(absl::c_equal(input_sizes.slice(input_ndim - normalized_ndim),
                             normalized_shape),
               error::kInvalidArgument)
      << "expected input trailing dimensions to match normalized_shape "
      << ToString(normalized_shape) << ", got input of shape "
      << ToString(input_sizes);

  if (weight.has_value() && weight->defined()) {
    // Upstream _fused_rms_norm_cuda skips RmsNormKernelImpl (and its dtype
    // dispatch check) when M == 0 (input.numel() == 0) and dtypes match, which
    // ops_test exercises for bool/int32/int64 in subtest4. However, never allow
    // mixed-dtype non-floating-point weight (e.g. empty float input with int32
    // weight).
    TT_RET_CHECK(
        weight->is_floating_point() ||
            (input_is_empty && weight->scalar_type() == input.scalar_type()),
        error::kInvalidArgument)
        << "expected the weight dtype to be floating point, got "
        << ToString(weight->scalar_type());
    TT_ASSIGN_OR_RETURN(const c10::SymIntArrayRef weight_sizes,
                        weight->sym_sizes());
    TT_RET_CHECK(absl::c_equal(weight_sizes, normalized_shape),
                 error::kInvalidArgument)
        << "expected weight to be of same shape as normalized_shape "
        << ToString(normalized_shape) << ", got weight of shape "
        << ToString(weight_sizes);
  }

  return absl::OkStatus();
}

void ValidateRmsNormInputsOrThrow(const ConstTensor& input,
                                  at::IntArrayRef normalized_shape,
                                  const std::optional<ConstTensor>& weight) {
  internal::ScopedOpName scoped_op_name(OpName::kRmsNorm);
  try {
    TT_THROW_IF_ERROR(ValidateRmsNormInputs(input, normalized_shape, weight));
  } catch (const TtError& e) {
    TranslateToC10ErrorAndThrow(e);
  }
}

// Composite fallback using TORCH_API-exported ATen ops for FakeTensor
// (torch.compile), functorch transforms, and forward-mode AD when
// _fused_rms_norm cannot be used directly (e.g. mixed-dtype input/weight or
// forward AD). Note: at::native::rms_norm_composite is not exported with
// TORCH_API from libtorch_cpu.so and has hidden visibility in open-source
// builds.
at::Tensor RmsNormComposite(const at::Tensor& input,
                            at::IntArrayRef normalized_shape,
                            const std::optional<at::Tensor>& weight,
                            std::optional<double> eps) {
  const int64_t input_ndim = input.dim();
  const size_t normalized_ndim = normalized_shape.size();
  Dimensions dims_to_reduce;
  dims_to_reduce.reserve(normalized_ndim);
  for (size_t i = 0; i < normalized_ndim; ++i) {
    dims_to_reduce.push_back(input_ndim - static_cast<int64_t>(i) - 1);
  }

  const at::ScalarType opmath_type = at::toOpMathType(input.scalar_type());
  const at::Tensor upcasted_input = input.to(opmath_type);
  const double eps_val = GetRmsNormEpsilon(eps, input.scalar_type());

  at::Tensor mean_sq =
      at::pow(upcasted_input, 2).mean(dims_to_reduce, /*keepdim=*/true);
  at::Tensor rstd = (opmath_type == at::ScalarType::Float ||
                     opmath_type == at::ScalarType::ComplexFloat)
                        ? at::rsqrt(mean_sq.add(static_cast<float>(eps_val)))
                        : at::rsqrt(mean_sq.add(eps_val));
  at::Tensor upcasted_result = upcasted_input.mul(rstd);

  const bool has_weight = weight.has_value() && weight->defined();
  if (has_weight) {
    upcasted_result = upcasted_result.mul(*weight);
  }
  if (!input.is_nested() && !(has_weight && weight->is_nested()) &&
      input.suggest_memory_format() != c10::MemoryFormat::ChannelsLast &&
      input.suggest_memory_format() != c10::MemoryFormat::ChannelsLast3d) {
    upcasted_result = upcasted_result.contiguous();
  }
  return upcasted_result.type_as(input);
}

at::Tensor DelegateRmsNormFakeOrFunctorch(
    const at::Tensor& input, at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight, std::optional<double> eps,
    bool any_has_fw_grad) {
  ValidateRmsNormInputsOrThrow(input, normalized_shape, weight);
  const bool has_weight = weight.has_value() && weight->defined();
  if (any_has_fw_grad ||
      (has_weight && weight->scalar_type() != input.scalar_type())) {
    return RmsNormComposite(input, normalized_shape, weight, eps);
  }
  return std::get<0>(at::_fused_rms_norm(input, normalized_shape, weight, eps));
}

}  // namespace

at::Tensor AtenRmsNorm(const at::Tensor& input,
                       at::IntArrayRef normalized_shape,
                       const std::optional<at::Tensor>& weight,
                       std::optional<double> eps) {
  const bool has_weight = weight.has_value() && weight->defined();
  if (IsFakeOrFunctorchTensor(input) ||
      (has_weight && IsFakeOrFunctorchTensor(*weight))) {
    return DelegateRmsNormFakeOrFunctorch(input, normalized_shape, weight, eps,
                                          /*any_has_fw_grad=*/false);
  }
  const double epsilon = GetRmsNormEpsilon(eps, input.scalar_type());
  TT_KERNEL(
      OpName::kRmsNorm, param_keys, (input, normalized_shape, weight, epsilon),
      {
        TT_THROW_IF_ERROR(
            ValidateRmsNormInputs(input, normalized_shape, weight));

        const auto input_sizes = input.sizes();
        if (input.numel() == 0) {
          TT_ASSIGN_OR_THROW(at::Tensor out,
                             MakeEmptyTensor(input_sizes, input.scalar_type(),
                                             input.device()));
          return out;
        }

        const size_t normalized_shape_dims = normalized_shape.size();
        auto op_builder = [normalized_shape_dims, epsilon](
                              absl::Span<const mlir::MlirOp> inputs,
                              mlir::MlirBuilder& builder)
            -> absl::StatusOr<MlirOpResults<1>> {
          TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=The inputs vector is always
                         // constructed with this op input, possibly followed by
                         // the weights.
              !inputs.empty() && inputs.size() <= 2, error::kInternal)
              << "expected 1 or 2 inputs, got " << inputs.size();

          mlir::MlirOp input_op = inputs[0];
          std::optional<mlir::MlirOp> weight_op;
          if (inputs.size() > 1) {
            weight_op = inputs[1];
          }

          TT_ASSIGN_OR_RETURN(LayerNormShloResults results,
                              BuildRmsNormShlo(input_op, weight_op,
                                               normalized_shape_dims, epsilon));
          return results.normalized_values;
        };

        TT_ASSIGN_OR_THROW(const auto elem_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));

        std::vector<at::Tensor> inputs = {input};
        if (has_weight) {
          inputs.push_back(*weight);
        }

        TT_ASSIGN_OR_THROW(
            DeviceBufferRef output_buf,
            (DispatchOp<kDynamicSize, 1>(
                std::move(op_builder), inputs,
                {.out_dtype = elem_dtype,
                 .out_dims = input_sizes,
                 .op_param_cache_keys = std::move(param_keys)})));

        return MakeTensor(output_buf);
      });
}

at::Tensor AtenRmsNormAutograd(const at::Tensor& input,
                               at::IntArrayRef normalized_shape,
                               const std::optional<at::Tensor>& weight,
                               std::optional<double> eps) {
  const bool has_weight = weight.has_value() && weight->defined();
  const bool any_has_fw_grad =
      input._fw_grad(/*level=*/0).defined() ||
      (has_weight && weight->_fw_grad(/*level=*/0).defined());
  const bool is_fake_or_functorch =
      IsFakeOrFunctorchTensor(input) ||
      (has_weight && IsFakeOrFunctorchTensor(*weight));

  // 1. FakeTensor (torch.compile), Functorch, or Forward-mode AD:
  // Must be checked BEFORE GradMode / requires_grad so FakeTensors in
  // torch.no_grad() inference mode never enter AtenRmsNorm's DeviceBuffer
  // path directly.
  if (is_fake_or_functorch || any_has_fw_grad) {
    return DelegateRmsNormFakeOrFunctorch(input, normalized_shape, weight, eps,
                                          any_has_fw_grad);
  }

  // 2. Eager Inference fast path: dispatch directly to 1-output AtenRmsNorm
  // without allocating an unused rstd DeviceBuffer.
  const bool any_requires_grad =
      input.requires_grad() || (has_weight && weight->requires_grad());
  if (!c10::GradMode::is_enabled() || !any_requires_grad) {
    c10::impl::ExcludeDispatchKeyGuard no_autograd(
        c10::autograd_dispatch_keyset);
    return AtenRmsNorm(input, normalized_shape, weight, eps);
  }

  // 3. Eager Training path: delegate to at::_fused_rms_norm so PyTorch's
  // autograd engine (derivatives.yaml:1301) attaches _FusedRmsNormBackward0
  // and invokes AtenFusedRmsNorm at OpNameStack depth 1 (where
  // ValidateRmsNormInputs is executed inside TT_KERNEL).
  return std::get<0>(at::_fused_rms_norm(input, normalized_shape, weight, eps));
}

std::tuple<at::Tensor, at::Tensor> AtenFusedRmsNorm(
    const at::Tensor& input, at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight, std::optional<double> eps) {
  // _fused_rms_norm(Tensor input, int[] normalized_shape, Tensor? weight,
  // float? eps) -> (Tensor, Tensor)
  const double epsilon = GetRmsNormEpsilon(eps, input.scalar_type());
  TT_KERNEL(
      OpName::kFusedRmsNorm, param_keys,
      (input, normalized_shape, weight, epsilon), {
        TT_THROW_IF_ERROR(
            ValidateRmsNormInputs(input, normalized_shape, weight));

        const size_t normalized_shape_dims = normalized_shape.size();
        const auto input_sizes = input.sizes();
        const int64_t input_dims = input.dim();

        // The StableHLO reduction op will remove all normalized dimensions, but
        // PyTorch expects the dimensions to be retained as size 1. Return rstd
        // with these singleton dimensions.
        Dimensions rstd_buffer_shape;
        for (int i = 0; i < input_dims - normalized_shape_dims; ++i) {
          rstd_buffer_shape.push_back(input_sizes[i]);
        }
        for (int i = input_dims - normalized_shape_dims; i < input_dims; ++i) {
          rstd_buffer_shape.push_back(1);
        }

        if (input.numel() == 0) {
          TT_ASSIGN_OR_THROW(at::Tensor out,
                             MakeEmptyTensor(input_sizes, input.scalar_type(),
                                             input.device()));
          TT_ASSIGN_OR_THROW(
              at::Tensor rstd,
              MakeEmptyTensor(rstd_buffer_shape,
                              ToAccumulateType(input.scalar_type()),
                              input.device()));
          return std::make_tuple(std::move(out), std::move(rstd));
        }

        auto op_builder = [normalized_shape_dims, epsilon](
                              absl::Span<const mlir::MlirOp> inputs,
                              mlir::MlirBuilder& builder)
            -> absl::StatusOr<MlirOpResults<2>> {
          TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=The inputs vector is always
                         // constructed with this op input, possibly followed by
                         // the weights.
              !inputs.empty() && inputs.size() <= 2, error::kInternal)
              << "expected 1 or 2 inputs, got " << inputs.size();

          mlir::MlirOp input_op = inputs[0];
          std::optional<mlir::MlirOp> weight_op;
          if (inputs.size() > 1) {
            weight_op = inputs[1];
          }

          TT_ASSIGN_OR_RETURN(LayerNormShloResults results,
                              BuildRmsNormShlo(input_op, weight_op,
                                               normalized_shape_dims, epsilon));
          return MlirOpResults<2>{results.normalized_values,
                                  results.reciprocal_std};
        };

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));

        std::vector<at::Tensor> inputs = {input};
        if (weight.has_value() && weight->defined()) {
          inputs.push_back(*weight);
        }

        mlir::ElementType rstd_dtype = output_dtype;
        if (output_dtype != mlir::ElementType::F32 &&
            output_dtype != mlir::ElementType::F64) {
          rstd_dtype = mlir::ElementType::F32;
        }

        const auto elem_dtype = output_dtype;
        const auto rstd_elem_dtype = rstd_dtype;

        TT_ASSIGN_OR_THROW(
            auto output_bufs,
            (DispatchOp<kDynamicSize, 2>(
                std::move(op_builder), inputs,
                {.out_dtypes = {elem_dtype, rstd_elem_dtype},
                 .out_dims_list = {input_sizes, rstd_buffer_shape},
                 .op_param_cache_keys = std::move(param_keys)})));

        return std::make_tuple(MakeTensor(output_bufs[0]),
                               MakeTensor(output_bufs[1]));
      });
}

std::tuple<at::Tensor, at::Tensor> AtenFusedRmsNormBackward(
    const at::Tensor& grad_out, const at::Tensor& input,
    at::IntArrayRef normalized_shape, const at::Tensor& rstd,
    const std::optional<at::Tensor>& weight, std::array<bool, 2> output_mask) {
  // _fused_rms_norm_backward(Tensor grad_out, Tensor input, int[]
  // normalized_shape, Tensor rstd, Tensor? weight, bool[2] output_mask) ->
  // (Tensor, Tensor)
  TT_KERNEL(
      OpName::kFusedRmsNormBackward, param_keys,
      (grad_out, input, normalized_shape, rstd, weight, output_mask), {
        TT_THROW_IF_ERROR(
            ValidateRmsNormInputs(input, normalized_shape, weight));

        // PyTorch's autograd engine passes an output_mask where output_mask[0]
        // indicates if grad_input is required and output_mask[1] indicates if
        // grad_weight is required. If neither gradient is requested,
        // short-circuit early and return undefined tensors without performing
        // any TPU allocation or kernel dispatch.
        if (!output_mask[0] && !output_mask[1]) {
          return {at::Tensor(), at::Tensor()};
        }

        // When an optional Tensor (like weight) is None (or nullopt) in the
        // forward pass, PyTorch's autograd engine saves an undefined Tensor for
        // the backward pass. It does not strip the argument from the backward
        // function call; it passes a Tensor object for which .defined() returns
        // false. Thus, it's insufficient to only rely on weight.has_value() to
        // determine presence of the weight Tensor, we must also call
        // weight->defined().
        const bool has_weight = weight.has_value() && weight->defined();

        const auto input_sizes = input.sizes();
        Dimensions input_dims = CopyIntVector(input_sizes);

        const Dimensions weight_dims =
            has_weight ? CopyIntVector(weight.value().sizes())
                       : CopyIntVector(normalized_shape);

        const at::ScalarType w_dtype =
            has_weight ? weight.value().scalar_type() : input.scalar_type();

        // For zero-element inputs (e.g. empty batch dimension), grad_input is
        // an empty tensor with 0 elements. However, weight is a non-empty
        // parameter vector (normalized_shape), and its gradient is
        // mathematically defined as the sum over batch dimensions. The sum over
        // zero elements evaluates to zeros. We populate grad_w with zeros on
        // CPU and transfer it to TPU via DMA to fulfill PyTorch autograd
        // semantics without triggering XLA compilation for degenerate shapes.
        if (input.numel() == 0) {
          at::Tensor grad_in;  // UNINITIALIZED_TENSOR_OK
          if (output_mask[0]) {
            TT_ASSIGN_OR_THROW(grad_in,
                               MakeEmptyTensor(input_sizes, input.scalar_type(),
                                               input.device()));
          }
          at::Tensor grad_w;  // UNINITIALIZED_TENSOR_OK
          if (output_mask[1]) {
            // Allocate the zeros on CPU and DMA transfer to TPU rather than
            // calling at::zeros(..., device=kXLA) or grad_w.zero_(), which
            // would build and execute an XLA constant kernel on TPU. For small
            // 1D parameter shapes, host allocation and DMA take
            // sub-microseconds.
            at::Tensor cpu_zeros =
                at::zeros(weight_dims,
                          at::TensorOptions().dtype(w_dtype).device(at::kCPU));
            TT_ASSIGN_OR_THROW(
                grad_w, CopyToTpuTensor(cpu_zeros, /*dtype=*/std::nullopt,
                                        /*non_blocking=*/false,
                                        /*memory_format=*/
                                        c10::MemoryFormat::Contiguous));
          }
          return std::make_tuple(std::move(grad_in), std::move(grad_w));
        }

        const Dimensions empty_dims = {0};
        const Dimensions grad_in_dims =
            output_mask[0] ? input_dims : empty_dims;
        const Dimensions grad_w_dims =
            output_mask[1] ? weight_dims : empty_dims;

        Dimensions normalized_shape_vec = CopyIntVector(normalized_shape);
        auto op_builder = [normalized_shape_vec =
                               std::move(normalized_shape_vec),
                           output_mask](absl::Span<const mlir::MlirOp> inputs,
                                        mlir::MlirBuilder& builder)
            -> absl::StatusOr<MlirOpResults<2>> {
          TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=The inputs sequence is always
                         // constructed with the output grad, this op's input,
                         // the rstd tensor, and possibly followed by the
                         // weights.
              inputs.size() >= 3 && inputs.size() <= 4, error::kInternal)
              << "expected 3 or 4 inputs, got " << inputs.size();

          mlir::MlirOp grad_out_op = inputs[0];
          mlir::MlirOp input_op = inputs[1];
          mlir::MlirOp rstd_op = inputs[2];
          std::optional<mlir::MlirOp> weight_op;
          if (inputs.size() > 3) {
            weight_op = inputs[3];
          }

          TT_ASSIGN_OR_RETURN(RmsNormBackwardShloResults results,
                              BuildRmsNormBackwardShlo(
                                  grad_out_op, input_op, rstd_op, weight_op,
                                  normalized_shape_vec, output_mask));
          return MlirOpResults<2>{results.grad_input, results.grad_weight};
        };

        std::vector<at::Tensor> inputs = {grad_out, input, rstd};
        if (has_weight) {
          inputs.push_back(weight.value());
        }

        TT_ASSIGN_OR_THROW(const auto elem_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));
        TT_ASSIGN_OR_THROW(const auto weight_dtype,
                           ConvertTo<mlir::ElementType>(w_dtype));

        TT_ASSIGN_OR_THROW(
            auto output_bufs,
            (DispatchOp<kDynamicSize, 2>(
                std::move(op_builder), inputs,
                {.out_dtypes = {elem_dtype, weight_dtype},
                 .out_dims_list = {grad_in_dims, grad_w_dims},
                 .op_param_cache_keys = std::move(param_keys)})));

        return {output_mask[0] ? MakeTensor(output_bufs[0]) : at::Tensor(),
                output_mask[1] ? MakeTensor(output_bufs[1]) : at::Tensor()};
      });
}

}  // namespace torch_tpu
