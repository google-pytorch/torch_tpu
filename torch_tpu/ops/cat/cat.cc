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

#include "torch_tpu/ops/cat/cat.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildCatShlo(absl::Span<mlir::MlirOp> input_ops,
                                          int64_t dimension) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches this error first.
      !input_ops.empty(), error::kInvalidArgument)
      << "expected a non-empty list of Tensors";
  return stablehlo::Concatenate(input_ops[0].getBuilder(), input_ops,
                                dimension);
}

}  // namespace torch_tpu
