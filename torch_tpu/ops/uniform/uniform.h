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

#ifndef TORCH_TPU_OPS_UNIFORM_UNIFORM_H_
#define TORCH_TPU_OPS_UNIFORM_UNIFORM_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

// Uses the philox algorithm and `rng_input_state` to generate random u64s and
// convert them to uniformly distributed random number in [to, from) of the
// desired type and shape.
absl::StatusOr<MlirOpResults<2>> BuildUniformShlo(mlir::MlirOp rng_input_state,
                                                  double from, double to,
                                                  llvm::ArrayRef<int64_t> sizes,
                                                  mlir::ElementType mlir_type);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_UNIFORM_UNIFORM_H_
