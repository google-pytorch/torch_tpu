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

#include "torch_tpu/ops/precision_context.h"

#include <optional>
#include <utility>

#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/error_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

struct PrecisionAndDotAlgorithm {
  mlir::stablehlo::Precision precision;
  std::optional<DotAlgorithmConfig> dot_algorithm;
};

PrecisionAndDotAlgorithm GetPrecisionAndDotAlgorithm() {
  auto precision = GetContextState<mlir::stablehlo::Precision>(
      mlir::stablehlo::Precision::DEFAULT);  // EXPLICIT_PRECISION_OK=root usage
  auto dot_algorithm = GetContextState<DotAlgorithmContextState>(std::nullopt);

  if (dot_algorithm.has_value()) {
    TT_CHECK_THROW(
        precision ==
            mlir::stablehlo::Precision::DEFAULT,  // EXPLICIT_PRECISION_OK=root
                                                  // usage
        error::kFailedPrecondition)
        << "precision must be DEFAULT when dot_algorithm is set";
  }
  return {precision, dot_algorithm};
}

}  // namespace

mlir::stablehlo::Precision GetPrecision() {
  return GetPrecisionAndDotAlgorithm().precision;
}

std::optional<DotAlgorithmConfig> GetDotAlgorithm() {
  return GetPrecisionAndDotAlgorithm().dot_algorithm;
}

}  // namespace torch_tpu
