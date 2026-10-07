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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_KILN_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_KILN_H_

// KILN: A small, Pallas-shaped kernel language for authoring Mosaic TPU kernels
// in C++.
//
// The goal is that a framework kernel reads like the Pallas kernel it would
// otherwise be, while everything (grid, tile sizes, optional operands, which
// branches exist) remains a runtime decision:
//
//   auto kernel = CreateKernel(
//       GridSpec{.name = "add", .grid = {m / tm}},
//       InSpec{{tm, n}, f32, row_map},
//       InSpec{{tm, n}, f32, row_map},
//       OutSpec{{tm, n}, f32, row_map},
//       [&](Ref x, Ref y, Ref out) {
//         out = Tile(x) + Tile(y);
//       });
//
// Concepts map 1:1 onto Pallas TPU:
//   * GridSpec / InSpec / OutSpec / ScratchSpec <-> pallas_call specs
//   * std::optional<InSpec / OutSpec>           <-> optional operand ->
//   std::optional<Ref>
//   * ProgramId / NumPrograms                   <-> pl.program_id /
//   pl.num_programs
//   * ref(0, 0) / ref(0, Slice(c, kChunk), all) <-> ref slicing; emits
//   tpu::MemRefSlice / Squeeze
//   * Tile(ref) / ref = tile / Scalar(ref)      <-> load/store on ref arguments
//   * Tile                                      <-> a jax.Array value inside
//   the kernel
//   * Scalar / Cond                             <-> scalar (SMEM / grid index)
//   values
//   * If / For                                  <-> pl.when / lax.fori_loop
//
#include "csrc/internal/mosaic/kiln/core/cond.h"          // IWYU pragma: export
#include "csrc/internal/mosaic/kiln/core/control_flow.h"  // IWYU pragma: export
#include "csrc/internal/mosaic/kiln/core/dtype.h"         // IWYU pragma: export
#include "csrc/internal/mosaic/kiln/core/index_arg.h"     // IWYU pragma: export
#include "csrc/internal/mosaic/kiln/core/index_map.h"     // IWYU pragma: export
#include "csrc/internal/mosaic/kiln/core/kernel.h"        // IWYU pragma: export
#include "csrc/internal/mosaic/kiln/core/literal.h"       // IWYU pragma: export
#include "csrc/internal/mosaic/kiln/core/ref.h"           // IWYU pragma: export
#include "csrc/internal/mosaic/kiln/core/scalar.h"        // IWYU pragma: export
#include "csrc/internal/mosaic/kiln/core/specs.h"         // IWYU pragma: export
#include "csrc/internal/mosaic/kiln/core/tile.h"          // IWYU pragma: export
#include "csrc/internal/mosaic/kiln/core/value_base.h"    // IWYU pragma: export

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_KILN_H_
