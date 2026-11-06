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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_GATHER_SCATTER_REDUCTIONS_CANONICALIZE_GATHER_SCATTER_REDUCTIONS_H_  // NOLINT(whitespace/line_length)
#define TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_GATHER_SCATTER_REDUCTIONS_CANONICALIZE_GATHER_SCATTER_REDUCTIONS_H_  // NOLINT(whitespace/line_length)

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that hoists gather operations across invariant-broadcast
// elementwise DAGs and folds additive reductions over single-element-per-slice
// zero-base scatter operations into in-bounds selections.
std::unique_ptr<mlir::Pass> createCanonicalizeGatherScatterReductionsPass();

}  // namespace torch_tpu

// clang-format off
#endif  // TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_GATHER_SCATTER_REDUCTIONS_CANONICALIZE_GATHER_SCATTER_REDUCTIONS_H_  // NOLINT
// clang-format on
