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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_COMMUTE_POINTWISE_DAGS_ACROSS_CHANNEL_TRANSPOSES_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_COMMUTE_POINTWISE_DAGS_ACROSS_CHANNEL_TRANSPOSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that commutes multi-branch pure elementwise activation
// DAGs and 1D unit-window convolutions across channel transposes.
std::unique_ptr<mlir::Pass>
createCommutePointwiseDagsAcrossChannelTransposesPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_COMMUTE_POINTWISE_DAGS_ACROSS_CHANNEL_TRANSPOSES_H_
