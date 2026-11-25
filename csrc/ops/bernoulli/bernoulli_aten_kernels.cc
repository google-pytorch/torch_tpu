/*
 * Copyright 2026 Google LLC
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

#include "csrc/ops/bernoulli/bernoulli_aten_kernels.h"

#include <cstdint>
#include <numeric>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/common/warnings.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/all_any/all_any.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/copy_from/tpu_to_cpu.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/rng_utils.h"
#include "csrc/ops/uniform/uniform.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Determines the floating-point element type used for uniform sampling in
// [0, 1). Low-precision floats (bfloat16, float16) are promoted to float32 via
// `InferComputationDtype` so small probabilities are not distorted by 7/10-bit
// mantissa quantization.
absl::StatusOr<mlir::ElementType> InferUniformElemType(
    const mlir::ElementType p_dtype, const mlir::ElementType output_dtype) {
  if (mlir::IsFloat(p_dtype)) {
    return InferComputationDtype(p_dtype);
  }
  if (mlir::IsFloat(output_dtype)) {
    return InferComputationDtype(output_dtype);
  }
  return mlir::ElementType::F32;
}

absl::Status ValidateScalarProbability(const double p) {
  TT_RET_CHECK(p >= 0.0 && p <= 1.0, error::kInvalidArgument)
      << "expected p to be in the range [0, 1], got " << p;
  return absl::OkStatus();
}

absl::Status ValidateTargetDevice(const at::Tensor& tensor,
                                  const std::string_view arg_label) {
  TT_RET_CHECK(IsPrivateUse1Device(tensor), error::kInvalidArgument)
      << "expected the " << arg_label << " tensor to be on "
      << GetPrivateUse1DeviceDebugName() << ", got " << tensor.device();
  return absl::OkStatus();
}

absl::Status ValidateTargetDtype(const at::Tensor& tensor,
                                 const std::string_view arg_label) {
  TT_RET_CHECK(!IsComplex(tensor), error::kInvalidArgument)
      << "expected the " << arg_label << " dtype not to be complex, got "
      << ToString(tensor.scalar_type());
  return ValidateNotLowPrecisionFloat(tensor);
}

absl::Status ValidateProbabilitiesDevice(const at::Tensor& p) {
  TT_RET_CHECK(IsPrivateUse1Device(p), error::kInvalidArgument)
      << "expected probabilities tensor to be on "
      << GetPrivateUse1DeviceDebugName() << ", got " << p.device();
  return absl::OkStatus();
}

absl::Status ValidateProbabilitiesTensor(const at::Tensor& p) {
  TT_RETURN_IF_ERROR(ValidateProbabilitiesDevice(p));
  TT_RET_CHECK(at::isFloatingType(p.scalar_type()), error::kInvalidArgument)
      << "expected probabilities tensor to have floating type, got "
      << ToString(p.scalar_type());
  return ValidateNotLowPrecisionFloat(p);
}

absl::StatusOr<MlirOpResults<1>> BuildProbabilitiesInRangeShlo(
    mlir::MlirOp p_op) {
  mlir::MlirBuilder& builder = p_op.getBuilder();
  const mlir::ElementType p_dtype = GetElementTypeOrDie(p_op);
  mlir::MlirOp zero = MakeScalarConstant(builder, 0.0, p_dtype);
  mlir::MlirOp one = MakeScalarConstant(builder, 1.0, p_dtype);
  TT_ASSIGN_OR_RETURN(zero, BroadcastIfNeeded(zero, p_op));
  TT_ASSIGN_OR_RETURN(one, BroadcastIfNeeded(one, p_op));
  mlir::MlirOp ge_zero = mlir::stablehlo::Compare(
      p_op, zero, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp le_one = mlir::stablehlo::Compare(
      p_op, one, mlir::stablehlo::ComparisonDirection::LE);
  mlir::MlirOp in_range = mlir::stablehlo::And(ge_zero, le_one);
  const int64_t rank = GetTensorTypeOrDie(p_op).getRank();
  if (rank == 0) {
    return in_range;
  }
  Dimensions reduce_dims(rank);
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp all_valid,
      BuildAllAnyReductionShlo(AllAnyOp::kAll, in_range, reduce_dims));
  return all_valid;
}

absl::Status ValidateProbabilitiesRange(const at::Tensor& p) {
  if (p.numel() == 0) {
    return absl::OkStatus();
  }
  // Checking the values needs a blocking transfer to CPU, which stalls the
  // host thread and breaks graph fusion, so this is only done when debug checks
  // are enabled.
  if (!GetEnableDebugChecks()) {
    TT_WARN_ONCE(
        WarningCategory::kCorrectness,
        "Unlike PyTorch on CUDA or CPU, TorchTPU doesn't by default check "
        "that the probabilities tensor for bernoulli() is in the range "
        "[0, 1]. Set the TORCH_TPU_ENABLE_DEBUG_CHECKS environment variable "
        "to \"1\" to enable this check, which incurs a performance penalty.");
    return absl::OkStatus();
  }
  // Skip synchronous device-to-host transfer during compilation/tracing or
  // manual-sync deferral.
  const EagerMode mode = GetEagerMode();
  if (mode == EagerMode::kInternalCompileFxGraph ||
      mode == EagerMode::kInternalDeferAll) {
    return absl::OkStatus();
  }
  // Look at the base buffer: the buffer of a view is a deferred node that is
  // never a placeholder, even if its base is.
  if (absl::StatusOr<DeviceBufferRef> buffer = GetBaseBuffer(p);
      !buffer.ok() || buffer->is_placeholder()) {
    return absl::OkStatus();
  }

  const std::string_view stage = "validate_probabilities_range";
  TT_ASSIGN_OR_RETURN(OpParamCacheKeys validate_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(stage));
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef all_valid_buf,
      (DispatchOp<1, 1>(BuildProbabilitiesInRangeShlo, {p},
                        {.out_dtype = mlir::ElementType::PRED,
                         .out_dims = {},
                         .op_param_cache_keys = std::move(validate_keys)})));
  TT_ASSIGN_OR_RETURN(
      at::Tensor cpu_all_valid,
      CopyDeviceBufferRefToCpuTensor(
          all_valid_buf, GetTransferSyncMode(/*non_blocking=*/false)));
  TT_RET_CHECK(cpu_all_valid.item<bool>(), error::kInvalidArgument)
      << "all probabilities in p must be in the range [0, 1]";
  return absl::OkStatus();
}

// Draws samples from a Bernoulli distribution with parameter `p`.
// This is done by drawing samples `u` from a uniform distribution in [0, 1)
// and setting the result to 1 if `u < p` and 0 otherwise. The result takes the
// shape and bounded dynamic dimensions of `shape_ref_op`, and `p_op` must be
// broadcastable to it.
absl::StatusOr<MlirOpResults<1>> BuildBernoulliShlo(
    mlir::MlirOp rng_input_state, mlir::MlirOp p_op, mlir::MlirOp shape_ref_op,
    const mlir::ElementType mlir_type) {
  // Determine floating point element type for uniform sampling.
  const mlir::ElementType p_dtype = GetElementTypeOrDie(p_op);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType uniform_elem_type,
                      InferUniformElemType(p_dtype, mlir_type));

  const mlir::Type uniform_mlir_type =
      mlir::getElementType(shape_ref_op.getContext(), uniform_elem_type);
  const mlir::RankedTensorType uniform_tensor_type =
      GetTensorTypeOrDie(shape_ref_op).clone(uniform_mlir_type);

  // Generate uniform distribution in [0, 1).
  TT_ASSIGN_OR_RETURN(mlir::MlirOp u,
                      BuildUniformShlo(rng_input_state, 0.0, 1.0,
                                       uniform_tensor_type, shape_ref_op));

  // bernoulli = u < p
  const mlir::RankedTensorType u_type = GetTensorTypeOrDie(u);
  TT_ASSIGN_OR_RETURN(p_op, BroadcastIfNeeded(p_op, u));

  // If u has a different dtype than p, cast p to the dtype of u.
  if (u_type.getElementType() != GetTensorTypeOrDie(p_op).getElementType()) {
    p_op = mlir::stablehlo::ConvertElementType(p_op, u_type.getElementType());
  }

  mlir::MlirOp comp = mlir::stablehlo::Compare(
      u, p_op, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp result = mlir::stablehlo::ConvertElementType(comp, mlir_type);

  return result;
}

NAryMlirOpBuilder<2, 1> GetBernoulliFloatFunctional(
    const mlir::ElementType output_dtype,
    const mlir::ElementType uniform_elem_type, const double p) {
  return [output_dtype, uniform_elem_type,
          p](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    auto [rng_input_state, self_op] = inputs;
    mlir::MlirBuilder& builder = rng_input_state.getBuilder();
    mlir::MlirOp p_op = MakeScalarConstant(builder, p, uniform_elem_type);
    return BuildBernoulliShlo(rng_input_state, p_op, self_op, output_dtype);
  };
}

NAryMlirOpBuilder<2, 1> GetBernoulliOutFunctional(
    const mlir::ElementType output_dtype) {
  return [output_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    auto [rng_input_state, p_op] = inputs;
    return BuildBernoulliShlo(rng_input_state, p_op, p_op, output_dtype);
  };
}

NAryMlirOpBuilder<3, 1> GetBernoulliTensorFunctional(
    const mlir::ElementType output_dtype) {
  return [output_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    auto [rng_input_state, self_op, p_op] = inputs;
    return BuildBernoulliShlo(rng_input_state, p_op, self_op, output_dtype);
  };
}

// Samples a new tensor with the shape and bounded dynamic dimensions of `self`
// by dispatching `builder` on the RNG state and `self`.
//
// Unlike the composite `at::empty_like(self).bernoulli_(...)`, which samples
// into a newly allocated static tensor, sampling with `self` as an operand
// keeps the bounded dynamic dimensions of `self` on the result.
absl::StatusOr<at::Tensor> SampleBernoulliLike(
    const at::Tensor& self, std::optional<at::Generator> generator,
    const mlir::ElementType output_dtype,
    const mlir::ElementType uniform_elem_type, NAryMlirOpBuilder<2, 1> builder,
    OpParamCacheKeys param_keys) {
  const Dimensions dims = CopyIntVector(self.sizes());
  const RngUsage usage{
      .num_elements = self.numel(),
      .bit_width = TorchEquivalentBitwidth(uniform_elem_type),
  };
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef output_buf,
      DispatchRngOpAndReturnBuffer(
          generator,
          [&](at::Tensor rng_input_state)  // REF_CAPTURE_OK=called
                                           // synchronously
          -> absl::StatusOr<std::vector<DeviceBufferRef>> {
            TT_ASSIGN_OR_RETURN(
                DeviceBufferRef buf,
                (DispatchOp<2, 1>(
                    std::move(builder), {rng_input_state, self},
                    {.out_dtype = output_dtype,
                     .out_dims = dims,
                     .op_param_cache_keys = std::move(param_keys)})));
            return std::vector<DeviceBufferRef>{std::move(buf)};
          },
          usage));
  return MakeTensor(std::move(output_buf));
}

}  // namespace

at::Tensor AtenBernoulli(const at::Tensor& self,
                         std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kBernoulli, param_keys, (self, generator), {
    if (self.numel() == 0) {
      // `at::empty_like` would dispatch a nested kernel from this one.
      TT_ASSIGN_OR_THROW(
          at::Tensor result,
          MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
      return result;
    }
    TT_THROW_IF_ERROR(ValidateProbabilitiesTensor(self));
    TT_THROW_IF_ERROR(ValidateProbabilitiesRange(self));

    TT_ASSIGN_OR_THROW(const mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(const mlir::ElementType uniform_elem_type,
                       InferUniformElemType(output_dtype, output_dtype));
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        SampleBernoulliLike(self, generator, output_dtype, uniform_elem_type,
                            GetBernoulliOutFunctional(output_dtype),
                            std::move(param_keys)));
    return result;
  });
}

at::Tensor& AtenBernoulliOut(const at::Tensor& self,
                             std::optional<at::Generator> generator,
                             at::Tensor& out) {
  TT_KERNEL(OpName::kBernoulliOut, param_keys, (self, generator, out), {
    TT_THROW_IF_ERROR(ValidateTargetDevice(out, "output"));
    TT_THROW_IF_ERROR(ValidateProbabilitiesDevice(self));
    // Like CUDA, resize `out` before validating dtypes and values.
    TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));
    if (self.numel() == 0) {
      return out;
    }
    TT_THROW_IF_ERROR(ValidateProbabilitiesTensor(self));
    TT_THROW_IF_ERROR(ValidateTargetDtype(out, "output"));
    TT_THROW_IF_ERROR(ValidateProbabilitiesRange(self));

    TT_ASSIGN_OR_THROW(const mlir::ElementType p_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(const mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    TT_ASSIGN_OR_THROW(const mlir::ElementType uniform_elem_type,
                       InferUniformElemType(p_dtype, output_dtype));
    const Dimensions dims = CopyIntVector(self.sizes());
    const RngUsage usage{
        .num_elements = self.numel(),
        .bit_width = TorchEquivalentBitwidth(uniform_elem_type),
    };

    TT_THROW_IF_ERROR(DispatchRngOpOut(
        out, generator, output_dtype,
        [&](at::Tensor rng_input_state)  // REF_CAPTURE_OK=called synchronously
        -> absl::Status {
          return DispatchOpOut<2, 1>(
              GetBernoulliOutFunctional(output_dtype), {rng_input_state, self},
              out,
              {.out_dtype = output_dtype,
               .out_dims = dims,
               .op_param_cache_keys = std::move(param_keys)});
        },
        usage));
    return out;
  });
}

at::Tensor AtenBernoulliP(const at::Tensor& self, double p,
                          std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kBernoulliP, param_keys, (self, p, generator), {
    TT_THROW_IF_ERROR(ValidateScalarProbability(p));
    if (self.numel() == 0) {
      // `at::empty_like` would dispatch a nested kernel from this one.
      TT_ASSIGN_OR_THROW(
          at::Tensor result,
          MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
      return result;
    }
    TT_THROW_IF_ERROR(ValidateTargetDtype(self, "input"));

    TT_ASSIGN_OR_THROW(const mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(const mlir::ElementType uniform_elem_type,
                       InferUniformElemType(output_dtype, output_dtype));
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        SampleBernoulliLike(
            self, generator, output_dtype, uniform_elem_type,
            GetBernoulliFloatFunctional(output_dtype, uniform_elem_type, p),
            std::move(param_keys)));
    return result;
  });
}

at::Tensor& AtenBernoulli_Float(at::Tensor& self, double p,
                                std::optional<at::Generator> generator) {
  // We use param_keys here because 'p' is a scalar embedded in the graph,
  // unlike in the _Tensor variant where 'p' is a tensor input.
  TT_KERNEL(OpName::kBernoulli_Float, param_keys, (self, p, generator), {
    TT_THROW_IF_ERROR(ValidateScalarProbability(p));
    if (self.numel() == 0) {
      return self;
    }
    TT_THROW_IF_ERROR(ValidateTargetDtype(self, "input"));

    TT_ASSIGN_OR_THROW(const mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(const mlir::ElementType uniform_elem_type,
                       InferUniformElemType(output_dtype, output_dtype));
    const Dimensions dims = CopyIntVector(self.sizes());
    const RngUsage usage{
        .num_elements = self.numel(),
        .bit_width = TorchEquivalentBitwidth(uniform_elem_type),
    };

    TT_THROW_IF_ERROR(DispatchRngOpOut(
        self, generator, output_dtype,
        [&](at::Tensor rng_input_state)  // REF_CAPTURE_OK=called synchronously
        -> absl::Status {
          return DispatchOpOut<2, 1>(
              GetBernoulliFloatFunctional(output_dtype, uniform_elem_type, p),
              {rng_input_state, self}, self,
              {.out_dtype = output_dtype,
               .out_dims = dims,
               .op_param_cache_keys = std::move(param_keys)});
        },
        usage));
    return self;
  });
}

at::Tensor& AtenBernoulli_Tensor(at::Tensor& self, const at::Tensor& p,
                                 std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kBernoulli_Tensor, param_keys, (self, p, generator), {
    TT_THROW_IF_ERROR(ValidateTargetDevice(self, "input"));
    TT_THROW_IF_ERROR(ValidateProbabilitiesDevice(p));
    if (self.numel() == 0) {
      return self;
    }
    TT_THROW_IF_ERROR(ValidateProbabilitiesTensor(p));
    TT_THROW_IF_ERROR(ValidateTargetDtype(self, "input"));
    // `bernoulli_` writes into `self` in place, so `p` has to fit `self` rather
    // than the two being broadcast against each other.
    TT_CHECK_THROW(at::is_expandable_to(p.sizes(), self.sizes()),
                   error::kInvalidArgument)
        << "expected p to be broadcastable to self shape, got "
        << ToString(p.sizes()) << " vs " << ToString(self.sizes());
    TT_THROW_IF_ERROR(ValidateProbabilitiesRange(p));

    TT_ASSIGN_OR_THROW(const mlir::ElementType p_dtype,
                       ConvertTo<mlir::ElementType>(p.scalar_type()));
    TT_ASSIGN_OR_THROW(const mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(const mlir::ElementType uniform_elem_type,
                       InferUniformElemType(p_dtype, output_dtype));
    const Dimensions dims = CopyIntVector(self.sizes());
    const RngUsage usage{
        .num_elements = self.numel(),
        .bit_width = TorchEquivalentBitwidth(uniform_elem_type),
    };

    TT_THROW_IF_ERROR(DispatchRngOpOut(
        self, generator, output_dtype,
        [&](at::Tensor rng_input_state)  // REF_CAPTURE_OK=called synchronously
        -> absl::Status {
          return DispatchOpOut<3, 1>(
              GetBernoulliTensorFunctional(output_dtype),
              {rng_input_state, self, p}, self,
              {.out_dtype = output_dtype,
               .out_dims = dims,
               .op_param_cache_keys = std::move(param_keys)});
        },
        usage));
    return self;
  });
}

}  // namespace torch_tpu
