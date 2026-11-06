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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_LAYOUTS_AND_RESHAPES_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_LAYOUTS_AND_RESHAPES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that commutes reshapes, transposes, and permutations
// across elementwise DAGs and linear contractions so they cancel or fold.
std::unique_ptr<mlir::Pass> createCanonicalizeLayoutsAndReshapesPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_LAYOUTS_AND_RESHAPES_H_
