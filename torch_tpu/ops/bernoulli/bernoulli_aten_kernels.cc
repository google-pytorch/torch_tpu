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

#include "torch_tpu/ops/bernoulli/bernoulli_aten_kernels.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_gen_impl.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/uniform/uniform.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Draws samples from a Bernoulli distribution with parameter `p`.
// This is done by drawing samples `u` from a uniform distribution in [0, 1)
// and setting the result to 1 if `u < p` and 0 otherwise.
absl::StatusOr<MlirOpResults<2>> BuildBernoulliShlo(
    mlir::MlirOp rng_input_state, mlir::MlirOp p_op,
    const llvm::ArrayRef<int64_t> sizes, const mlir::ElementType mlir_type) {
  // Generate uniform distribution in [0, 1).
  TT_ASSIGN_OR_RETURN(
      auto uniform_results,
      BuildUniformShlo(rng_input_state, 0.0, 1.0, sizes, mlir_type));

  mlir::MlirOp rng_output_state = uniform_results[0];
  mlir::MlirOp u = uniform_results[1];

  // bernoulli = u < p
  const mlir::RankedTensorType u_type = GetTensorTypeOrDie(u);
  p_op = mlir::stablehlo::BroadcastInDim(u_type, p_op, {});

  // If u has a different dtype than p, cast p to the dtype of u.
  if (u_type.getElementType() != GetTensorTypeOrDie(p_op).getElementType()) {
    p_op = mlir::stablehlo::ConvertElementType(p_op, u_type.getElementType());
  }

  auto comp = mlir::stablehlo::Compare(
      u, p_op, mlir::stablehlo::ComparisonDirection::LT);
  auto result = mlir::stablehlo::ConvertElementType(comp, mlir_type);

  return {{rng_output_state, result}};
}

NAryMlirOpBuilder<1, 2> GetBernoulliFloatFunctional(
    Dimensions dims, mlir::ElementType output_dtype, double p) {
  return [dims, output_dtype,
          p](mlir::MlirOp rng_input_state) -> absl::StatusOr<MlirOpResults<2>> {
    auto& builder = rng_input_state.getBuilder();
    mlir::MlirOp p_op = MakeScalarConstant(builder, p, output_dtype);
    return BuildBernoulliShlo(rng_input_state, p_op, dims, output_dtype);
  };
}

}  // namespace

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

    auto gen = at::get_generator_or_default<DeviceGeneratorImpl>(
        generator, GetDefaultDeviceGenerator());

    // See Note [Acquire lock when using random generators]
    // NOLINTNEXTLINE
    std::scoped_lock<std::mutex> lock(gen->mutex_);

    // Since we need to generate random bits, we query for the rng state tensor.
    at::Tensor rng_input_state = gen->DeviceStateTensor();

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    const auto dims = CopyIntVector(self.sizes());
    TT_ASSIGN_OR_THROW(
        (auto [rng_output_state_buf, output_buf]),
        (DispatchOp<1, 2>(
            GetBernoulliFloatFunctional(dims, output_dtype, p),
            {rng_input_state},
            {.out_dtypes = {mlir::ElementType::UI64, output_dtype},
             .out_dims_list = {{2}, dims},
             .op_param_cache_keys = std::move(param_keys)})));
    // After the state has been used (and updated) after generating random bits,
    // we give it back to the generator, so that it can be used by other ops in
    // the same graph.
    auto rng_output_state = MakeTensor(std::move(rng_output_state_buf));
    TT_THROW_IF_ERROR(gen->SetDeviceStateTensor(rng_output_state));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(output_buf), self));
    return self;
  });
}

}  // namespace torch_tpu
