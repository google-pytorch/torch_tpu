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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_MEMORY_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_MEMORY_H_

#include <utility>

#include "csrc/internal/mosaic/kiln/core/ref.h"
#include "csrc/internal/mosaic/kiln/core/semaphore.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "csrc/internal/mosaic/kiln/core/specs.h"

namespace torch_tpu::kiln {

// Allocates a scratch `Ref` in VMEM or SMEM inside the current region/function
// (`memref.alloca`).
Ref Alloca(const ScratchSpec& spec);

inline Ref AllocaVmem(Shape shape) {
  return Alloca(ScratchSpec::Vmem(std::move(shape)));
}

inline Ref AllocaSmem(Shape shape) {
  return Alloca(ScratchSpec::Smem(std::move(shape)));
}

// Allocates a 0-D DMA semaphore inside the current region/function
// (`tpu.sem_alloc` with `!tpu.dma_semaphore`).
Semaphore AllocaDmaSemaphore();

// Allocates a multi-dimensional array of DMA semaphores inside the current
// region/function (`tpu.sem_alloc` with `!tpu.dma_semaphore`).
SemaphoreArray AllocaDmaSemaphore(Dimensions shape);

// Allocates a 0-D regular semaphore inside the current region/function
// (`tpu.sem_alloc` with `!tpu.semaphore`).
Semaphore AllocaSemaphore();

// Allocates a multi-dimensional array of regular semaphores inside the current
// region/function (`tpu.sem_alloc` with `!tpu.semaphore`).
SemaphoreArray AllocaSemaphore(Dimensions shape);

// Returns the barrier semaphore for collective kernels (`tpu.sem_barrier`).
Semaphore GetBarrierSemaphore();

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_MEMORY_H_
