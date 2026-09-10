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
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/rng_utils.h"
#include "csrc/ops/uniform/uniform.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Draws samples from a Bernoulli distribution with parameter `p`.
// This is done by drawing samples `u` from a uniform distribution in [0, 1)
// and setting the result to 1 if `u < p` and 0 otherwise.
absl::StatusOr<MlirOpResults<1>> BuildBernoulliShlo(
    mlir::MlirOp rng_input_state, mlir::MlirOp p_op,
    const llvm::ArrayRef<int64_t> sizes, const mlir::ElementType mlir_type) {
  // Generate uniform distribution in [0, 1).
  TT_ASSIGN_OR_RETURN(
      auto uniform_results,
      BuildUniformShlo(rng_input_state, 0.0, 1.0, sizes, mlir_type));

  mlir::MlirOp u = uniform_results;

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

NAryMlirOpBuilder<1, 1> GetBernoulliFloatFunctional(
    Dimensions dims, mlir::ElementType output_dtype, double p) {
  return [dims, output_dtype,
          p](mlir::MlirOp rng_input_state) -> absl::StatusOr<MlirOpResults<1>> {
    auto& builder = rng_input_state.getBuilder();
    mlir::MlirOp p_op = MakeScalarConstant(builder, p, output_dtype);
    return BuildBernoulliShlo(rng_input_state, p_op, dims, output_dtype);
  };
}

NAryMlirOpBuilder<2, 1> GetBernoulliTensorFunctional(
    Dimensions dims, mlir::ElementType output_dtype) {
  return [dims, output_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    return BuildBernoulliShlo(inputs[0], inputs[1], dims, output_dtype);
  };
}

absl::Status DispatchBernoulliTensorRng(at::Tensor& dest,
                                        const at::Tensor& p_tensor,
                                        std::optional<at::Generator> generator,
                                        Dimensions dims,
                                        mlir::ElementType output_dtype,
                                        OpParamCacheKeys param_keys) {
  return DispatchRngOpOut(
      dest, generator, output_dtype,
      [&](at::Tensor rng_input_state) -> absl::Status {
        return DispatchOpOut<2, 1>(
            GetBernoulliTensorFunctional(dims, output_dtype),
            {rng_input_state, p_tensor}, dest,
            {.out_dtype = output_dtype,
             .out_dims = dims,
             .op_param_cache_keys = std::move(param_keys)});
      });
}

}  // namespace

at::Tensor& AtenBernoulliOut(const at::Tensor& self,
                             std::optional<at::Generator> generator,
                             at::Tensor& out) {
  TT_KERNEL(OpName::kBernoulliOut, param_keys, (self, generator, out), {
    TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));
    if (self.numel() == 0) {
      return out;
    }

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    const auto dims = CopyIntVector(self.sizes());

    TT_THROW_IF_ERROR(DispatchBernoulliTensorRng(
        out, self, generator, dims, output_dtype, std::move(param_keys)));
    return out;
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

    TT_THROW_IF_ERROR(DispatchRngOp(
        self, generator,
        [&](at::Tensor rng_input_state)
            -> absl::StatusOr<std::vector<DeviceBufferRef>> {
          TT_ASSIGN_OR_RETURN(
              auto buf, (DispatchOp<1, 1>(
                            GetBernoulliFloatFunctional(dims, output_dtype, p),
                            {rng_input_state},
                            {.out_dtype = output_dtype,
                             .out_dims = dims,
                             .op_param_cache_keys = std::move(param_keys)})));
          return std::vector<DeviceBufferRef>{std::move(buf)};
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

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    const auto dims = CopyIntVector(self.sizes());

    TT_THROW_IF_ERROR(DispatchBernoulliTensorRng(
        self, p, generator, dims, output_dtype, std::move(param_keys)));
    return self;
  });
}

}  // namespace torch_tpu
