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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_INDEX_ARITHMETIC_CANONICALIZE_INDEX_ARITHMETIC_H_  // NOLINT
#define TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_INDEX_ARITHMETIC_CANONICALIZE_INDEX_ARITHMETIC_H_  // NOLINT

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that canonicalizes index arithmetic in StableHLO:
// 1. Zero-init full-row axis-0 single-row additive scatters to masked
//    reductions.
// 2. Trailing unit gather index_vector_dim squeeze and negative index wrap
//    narrowing to i32.
// 3. Interval-bounded i64 index subgraphs and iota-convert hoisting narrowed to
//    i32.
std::unique_ptr<mlir::Pass> createCanonicalizeIndexArithmeticPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_INDEX_ARITHMETIC_CANONICALIZE_INDEX_ARITHMETIC_H_
        // // NOLINT
