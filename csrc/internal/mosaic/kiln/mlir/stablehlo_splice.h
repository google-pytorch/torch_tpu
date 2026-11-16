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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_MLIR_STABLEHLO_SPLICE_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_MLIR_STABLEHLO_SPLICE_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "csrc/internal/mosaic/kiln/core/tile.h"
#include "llvm/ADT/ArrayRef.h"

namespace torch_tpu::kiln {

// Inlines the StableHLO/CHLO module `mlir_text` into the kernel at the current
// point and returns its result as a Tile.
//
// `@main` must take one tensor argument per entry of `args`, matching its
// shape and element type, and return a single ranked tensor; the module may
// not broadcast.
//
// The kernel must be wrapped with `CreateKilnCustomCallOp(...,
// needs_hlo_passes=true)` so that Mosaic legalizes the spliced StableHLO.
absl::StatusOr<Tile> SpliceStablehlo(std::string_view mlir_text,
                                     llvm::ArrayRef<Tile> args);

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_MLIR_STABLEHLO_SPLICE_H_
