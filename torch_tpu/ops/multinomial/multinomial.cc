// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/ops/multinomial/multinomial.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/min_max/min_max.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/topk/topk.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

// Generates Gumbel noise of the given shape.
absl::StatusOr<mlir::MlirOp> BuildGumbelNoise(
    mlir::MlirBuilder& builder, mlir::RankedTensorType noise_type) {
  // Gumbel noise is -log(-log(U)), where U ~ Uniform(0, 1).
  auto noise_element_type = noise_type.getElementType();
  auto zero = MakeScalarConstant(builder, 0.0, noise_element_type);
  auto one = MakeScalarConstant(builder, 1.0, noise_element_type);
  auto shape_op = stablehlo::Constant(
      builder,
      makeConstant(noise_type.getShape(),
                   makeTensorType(builder.getContext(), {noise_type.getRank()},
                                  mlir::ElementType::I64)));
  mlir::MlirOp uniform =
      stablehlo::Rng(zero, one, shape_op, stablehlo::RngDistribution::UNIFORM);

  mlir::MlirOp log_uniform = stablehlo::Log(uniform);
  mlir::MlirOp neg_log_uniform = stablehlo::Neg(log_uniform);
  mlir::MlirOp log_neg_log_uniform = stablehlo::Log(neg_log_uniform);
  mlir::MlirOp gumbel = stablehlo::Neg(log_neg_log_uniform);
  return gumbel;
}

}  // namespace

// This uses the Gumbel-Max trick to sample from the multinomial distribution.
// This is equivalent to drawing samples from the distribution with
// probabilities proportional to the input weights.
// See https://en.wikipedia.org/wiki/Gumbel_distribution.
absl::StatusOr<mlir::MlirOp> BuildMultinomialShlo(mlir::MlirOp input_op,
                                                  int64_t num_samples,
                                                  bool replacement) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  mlir::MlirBuilder& builder = input_op.getBuilder();
  auto input_shape = input_type.getShape();
  const int64_t rank = input_type.getRank();
  const int64_t feature_dim = rank - 1;

  mlir::MlirOp logits = stablehlo::Log(input_op);

  if (replacement) {
    // Sample with replacement.
    Dimensions expanded_shape = CopyIntVector(input_shape);
    expanded_shape.insert(expanded_shape.begin() + feature_dim, 1);
    mlir::MlirOp expanded_logits = stablehlo::Reshape(logits, expanded_shape);

    Dimensions broadcast_shape = expanded_shape;
    broadcast_shape[feature_dim] = num_samples;
    TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_logits,
                        BroadcastIfNeeded(expanded_logits, broadcast_shape));

    auto noise_type = mlir::RankedTensorType::get(broadcast_shape,
                                                  input_type.getElementType());
    TT_ASSIGN_OR_RETURN(mlir::MlirOp gumbel_noise,
                        BuildGumbelNoise(builder, noise_type));

    mlir::MlirOp perturbed_logits =
        stablehlo::Add(broadcasted_logits, gumbel_noise);

    ReductionMode mode = ReductionMode::kDropDims;
    TT_ASSIGN_OR_RETURN(mlir::MlirOp result,
                        BuildMinMaxShlo(feature_dim + 1, MinMaxOp::kMax, mode,
                                        perturbed_logits))
        .indices;
    return result;
  } else {
    // Sample without replacement.
    TT_ASSIGN_OR_RETURN(mlir::MlirOp gumbel_noise,
                        BuildGumbelNoise(builder, input_type));

    mlir::MlirOp perturbed_logits = stablehlo::Add(logits, gumbel_noise);

    TT_ASSIGN_OR_RETURN(
        TopKOutputs topk_results,
        BuildTopKShlo(perturbed_logits, num_samples, feature_dim,
                      TopKMode::kLargest, TopKStableMode::kUnstable));
    return topk_results.indices;
  }
}

}  // namespace torch_tpu
