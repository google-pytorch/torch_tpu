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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SPECS_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SPECS_H_

#include <string>
#include <utility>

#include "csrc/internal/mosaic/kiln/core/index_map.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "llvm/ADT/SmallVector.h"

namespace torch_tpu::kiln {

enum class MemorySpace {
  kVmem,
  kSmem,
  kHbm,
};

// Iteration order guarantees for a grid dimension (Pallas
// `dimension_semantics`).
enum class DimensionSemantics {
  // Iterations along this axis are independent and may be reordered or executed
  // concurrently by the compiler.
  kParallel,
  // Iterations along this axis have data dependencies and must execute
  // sequentially in increasing index order.
  kArbitrary,
};

enum class BlockRole { kNone, kIn, kOut };

// Block specification for a kernel operand, optionally tagged with a role
// (`kIn` / `kOut`) for `CreateKernel` overload resolution.
template <BlockRole Role = BlockRole::kNone>
struct BasicBlockSpec {
  Shape shape;
  IndexMap map;
  MemorySpace space = MemorySpace::kVmem;

  static BasicBlockSpec Vmem(Shape shape, IndexMap map = IndexMap()) {
    return {std::move(shape), std::move(map), MemorySpace::kVmem};
  }
  static BasicBlockSpec Smem(Shape shape, IndexMap map = IndexMap()) {
    return {std::move(shape), std::move(map), MemorySpace::kSmem};
  }
  static BasicBlockSpec Hbm(Shape shape) {
    return {std::move(shape), IndexMap(), MemorySpace::kHbm};
  }
};

using BlockSpec = BasicBlockSpec<>;
using InSpec = BasicBlockSpec<BlockRole::kIn>;
using OutSpec = BasicBlockSpec<BlockRole::kOut>;

struct ScratchSpec {
  Shape shape;
  MemorySpace space = MemorySpace::kVmem;

  static ScratchSpec Vmem(Shape shape) {
    return {std::move(shape), MemorySpace::kVmem};
  }
  static ScratchSpec Smem(Shape shape) {
    return {std::move(shape), MemorySpace::kSmem};
  }
};

struct ScalarPrefetchSpec {
  Shape shape;
};

struct GridSpec {
  std::string name = "kernel";
  Dimensions grid;
  // One entry per grid axis. Defaults to `kParallel` for every axis.
  llvm::SmallVector<DimensionSemantics, 4> dimension_semantics;
};

struct KernelSpec {
  GridSpec grid;
  llvm::SmallVector<ScalarPrefetchSpec, 2> scalar_prefetch;
  llvm::SmallVector<BlockSpec, 8> in_specs;
  llvm::SmallVector<BlockSpec, 4> out_specs;
  llvm::SmallVector<ScratchSpec, 4> scratch;
};

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SPECS_H_
