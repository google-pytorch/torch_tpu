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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SEMAPHORE_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SEMAPHORE_H_

#include <array>
#include <cstdint>
#include <optional>
#include <utility>

#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/value_handle.h"
#include "llvm/ADT/ArrayRef.h"

namespace torch_tpu::kiln {

struct SemaphoreSignalOptions {
  std::optional<IndexArg> device_id = std::nullopt;
  std::optional<IndexArg> core_id = std::nullopt;
  // Note: subcore_id is omitted until SparseCore support is added.
};

// A 0-D TPU regular or DMA semaphore (`memref<!tpu.semaphore,
// #tpu.memory_space<semaphore_mem>>` or `memref<!tpu.dma_semaphore,
// #tpu.memory_space<semaphore_mem>>`), allocated inside a kernel via
// `AllocaSemaphore()` / `AllocaDmaSemaphore()`, obtained via
// `GetBarrierSemaphore()`, or indexed from a `SemaphoreArray`.
//
// Unlike `Ref`, 0-D semaphores (`Semaphore`) and multi-dimensional semaphore
// arrays (`SemaphoreArray`) are distinct types: all TPU semaphore and DMA ops
// (`tpu.sem_read`, `tpu.sem_signal`, `tpu.sem_wait`, `tpu.enqueue_dma`,
// `tpu.wait_dma`) strictly require a 0-D semaphore memref and semaphore arrays
// cannot be sliced into sub-arrays, so separating the types enforces full
// indexing to a 0-D `Semaphore` at compile time.
class Semaphore final {
 public:
  explicit Semaphore(internal::ValueHandle handle);

  internal::ValueHandle handle() const { return handle_; }

  bool is_dma() const;

  // Reads the current value of this semaphore as an i32 Scalar
  // (`tpu.sem_read`).
  Scalar Read() const;

  // Signals (increments) this regular semaphore by `amount` (`tpu.sem_signal`).
  void Signal(IndexArg amount = 1, SemaphoreSignalOptions options = {}) const;

  // Waits until this regular semaphore reaches `amount` and decrements it
  // (`tpu.sem_wait`).
  void Wait(IndexArg amount = 1) const;

 private:
  internal::ValueHandle handle_;
};

// A multi-dimensional array of TPU regular or DMA semaphores
// (`memref<...x!tpu.semaphore, #tpu.memory_space<semaphore_mem>>` or
// `memref<...x!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>`),
// allocated via `AllocaSemaphore(shape)` / `AllocaDmaSemaphore(shape)`.
// Indexing with `sems(i, ...)` extracts a single 0-D `Semaphore`.
class SemaphoreArray final {
 public:
  explicit SemaphoreArray(internal::ValueHandle handle);

  internal::ValueHandle handle() const { return handle_; }

  llvm::ArrayRef<int64_t> shape() const;
  int64_t rank() const;
  bool is_dma() const;

  // Extracts a single 0-D `Semaphore` at the given element indices (one
  // `IndexArg` per dimension), e.g. `sems(slot)`.
  template <typename... Indices>
  Semaphore operator()(Indices&&... indices) const {
    static_assert(sizeof...(Indices) >= 1,
                  "SemaphoreArray indexing requires at least one index");
    std::array<IndexArg, sizeof...(Indices)> idx = {
        IndexArg(std::forward<Indices>(indices))...};
    return AtImpl(idx);
  }

 private:
  Semaphore AtImpl(llvm::ArrayRef<IndexArg> indices) const;

  internal::ValueHandle handle_;
};

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SEMAPHORE_H_
