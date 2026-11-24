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

#include <optional>
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
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
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
  const mlir::ElementType uniform_elem_type =
      mlir::IsFloat(p_dtype)
          ? p_dtype
          : (mlir::IsFloat(mlir_type) ? mlir_type : mlir::ElementType::F32);

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

  auto comp = mlir::stablehlo::Compare(
      u, p_op, mlir::stablehlo::ComparisonDirection::LT);
  auto result = mlir::stablehlo::ConvertElementType(comp, mlir_type);

  return result;
}

NAryMlirOpBuilder<2, 1> GetBernoulliFloatFunctional(
    mlir::ElementType output_dtype, double p) {
  return [output_dtype, p](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    auto [rng_input_state, self_op] = inputs;
    auto& builder = rng_input_state.getBuilder();
    const mlir::ElementType p_dtype =
        mlir::IsFloat(output_dtype) ? output_dtype : mlir::ElementType::F32;
    mlir::MlirOp p_op = MakeScalarConstant(builder, p, p_dtype);
    return BuildBernoulliShlo(rng_input_state, p_op, self_op, output_dtype);
  };
}

NAryMlirOpBuilder<2, 1> GetBernoulliOutFunctional(
    mlir::ElementType output_dtype) {
  return [output_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    auto [rng_input_state, p_op] = inputs;
    return BuildBernoulliShlo(rng_input_state, p_op, p_op, output_dtype);
  };
}

NAryMlirOpBuilder<3, 1> GetBernoulliTensorFunctional(
    mlir::ElementType output_dtype) {
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
    mlir::ElementType output_dtype, NAryMlirOpBuilder<2, 1> builder,
    OpParamCacheKeys param_keys) {
  const auto dims = CopyIntVector(self.sizes());
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
          }));
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
    TT_CHECK_THROW(at::isFloatingType(self.scalar_type()),
                   error::kInvalidArgument)
        << "expected probabilities tensor to have floating type, got "
        << ToString(self.scalar_type());

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        SampleBernoulliLike(self, generator, output_dtype,
                            GetBernoulliOutFunctional(output_dtype),
                            std::move(param_keys)));
    return result;
  });
}

at::Tensor& AtenBernoulliOut(const at::Tensor& self,
                             std::optional<at::Generator> generator,
                             at::Tensor& out) {
  TT_KERNEL(OpName::kBernoulliOut, param_keys, (self, generator, out), {
    TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));
    if (self.numel() == 0) {
      return out;
    }
    TT_CHECK_THROW(at::isFloatingType(self.scalar_type()),
                   error::kInvalidArgument)
        << "expected probabilities tensor to have floating type, got "
        << ToString(self.scalar_type());
    TT_CHECK_THROW(!IsComplex(out), error::kInvalidArgument)
        << "expected the output dtype not to be complex, got "
        << ToString(out.scalar_type());

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    const auto dims = CopyIntVector(self.sizes());

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
        }));
    return out;
  });
}

at::Tensor AtenBernoulliP(const at::Tensor& self, double p,
                          std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kBernoulliP, param_keys, (self, p, generator), {
    TT_CHECK_THROW(p >= 0.0 && p <= 1.0, error::kInvalidArgument)
        << "expected p to be in the range [0, 1], got " << p;
    if (self.numel() == 0) {
      // `at::empty_like` would dispatch a nested kernel from this one.
      TT_ASSIGN_OR_THROW(
          at::Tensor result,
          MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
      return result;
    }

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        SampleBernoulliLike(self, generator, output_dtype,
                            GetBernoulliFloatFunctional(output_dtype, p),
                            std::move(param_keys)));
    return result;
  });
}

at::Tensor& AtenBernoulli_Float(at::Tensor& self, double p,
                                std::optional<at::Generator> generator) {
  // We use param_keys here because 'p' is a scalar embedded in the graph,
  // unlike in the _Tensor variant where 'p' is a tensor input.
  TT_KERNEL(OpName::kBernoulli_Float, param_keys, (self, p, generator), {
    if (self.numel() == 0) {
      return self;
    }
    TT_CHECK_THROW(p >= 0.0 && p <= 1.0, error::kInvalidArgument)
        << "expected p to be in the range [0, 1], got " << p;

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    const auto dims = CopyIntVector(self.sizes());

    TT_THROW_IF_ERROR(DispatchRngOpOut(
        self, generator, output_dtype,
        [&](at::Tensor rng_input_state)  // REF_CAPTURE_OK=called synchronously
        -> absl::Status {
          return DispatchOpOut<2, 1>(
              GetBernoulliFloatFunctional(output_dtype, p),
              {rng_input_state, self}, self,
              {.out_dtype = output_dtype,
               .out_dims = dims,
               .op_param_cache_keys = std::move(param_keys)});
        }));
    return self;
  });
}

at::Tensor& AtenBernoulli_Tensor(at::Tensor& self, const at::Tensor& p,
                                 std::optional<at::Generator> generator) {
  TT_KERNEL(OpName::kBernoulli_Tensor, param_keys, (self, p, generator), {
    if (self.numel() == 0) {
      return self;
    }
    TT_CHECK_THROW(at::isFloatingType(p.scalar_type()), error::kInvalidArgument)
        << "expected probabilities tensor to have floating type, got "
        << ToString(p.scalar_type());
    // `bernoulli_` writes into `self` in place, so `p` has to fit `self` rather
    // than the two being broadcast against each other.
    TT_CHECK_THROW(at::is_expandable_to(p.sizes(), self.sizes()),
                   error::kInvalidArgument)
        << "expected p to be broadcastable to self shape, got "
        << ToString(p.sizes()) << " vs " << ToString(self.sizes());

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    const auto dims = CopyIntVector(self.sizes());

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
        }));
    return self;
  });
}

}  // namespace torch_tpu
