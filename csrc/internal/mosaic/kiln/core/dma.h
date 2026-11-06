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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_DMA_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_DMA_H_

#include <optional>

#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/ref.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/semaphore.h"

namespace torch_tpu::kiln {

struct AsyncCopyOptions {
  // DMA priority (0 is highest and default; 0 or 1 supported).
  int priority = 0;
};

struct AsyncRemoteCopyOptions {
  // Destination device / core IDs for remote or cross-core DMAs.
  // At least one of `device_id` or `core_id` must be specified.
  std::optional<IndexArg> device_id = std::nullopt;
  std::optional<IndexArg> core_id = std::nullopt;
  // Note: subcore_id and strict_ordering are omitted until SparseCore support
  // is added.
};

// Describes a local asynchronous DMA transfer between two `Ref`s (e.g.
// HBM <-> VMEM or HBM <-> SMEM), modeled on Pallas `pltpu.make_async_copy`.
class AsyncCopyDescriptor {
 public:
  AsyncCopyDescriptor(Ref src_ref, Ref dst_ref, Semaphore sem,
                      AsyncCopyOptions options = {});

  // Enqueues the DMA copy (`tpu.enqueue_dma`).
  void Start() const;

  // Waits until writing to `dst_ref` has completed (`tpu.wait_dma`).
  void Wait() const;

  const Ref& src_ref() const { return src_ref_; }
  const Ref& dst_ref() const { return dst_ref_; }
  const Semaphore& sem() const { return sem_; }
  const AsyncCopyOptions& options() const { return options_; }

 private:
  Ref src_ref_;
  Ref dst_ref_;
  Semaphore sem_;
  AsyncCopyOptions options_;
};

// Describes a remote or cross-core asynchronous DMA transfer between two `Ref`s
// (e.g. VMEM <-> VMEM across devices/cores), modeled on Pallas
// `pltpu.make_async_remote_copy`.
class AsyncRemoteCopyDescriptor {
 public:
  AsyncRemoteCopyDescriptor(Ref src_ref, Ref dst_ref, Semaphore send_sem,
                            Semaphore recv_sem, AsyncRemoteCopyOptions options);

  // Enqueues the remote DMA copy (`tpu.enqueue_dma`).
  void Start() const;

  // Waits for both `send_sem` (read from `src_ref`) and `recv_sem` (write to
  // `dst_ref`) to complete (`tpu.wait_dma`).
  void Wait() const;

  // Waits until sending from `src_ref` has completed (`wait_target = false`).
  void WaitSend() const;

  // Waits until receiving into `dst_ref` has completed (`wait_target = true`).
  void WaitRecv() const;

  const Ref& src_ref() const { return src_ref_; }
  const Ref& dst_ref() const { return dst_ref_; }
  const Semaphore& send_sem() const { return send_sem_; }
  const Semaphore& recv_sem() const { return recv_sem_; }
  const std::optional<Scalar>& device_id() const { return device_id_; }
  const std::optional<Scalar>& core_id() const { return core_id_; }

 private:
  Ref src_ref_;
  Ref dst_ref_;
  Semaphore send_sem_;
  Semaphore recv_sem_;
  std::optional<Scalar> device_id_;
  std::optional<Scalar> core_id_;
};

// Creates a descriptor for an asynchronous DMA copy without starting it
// (Pallas `pltpu.make_async_copy`).
AsyncCopyDescriptor MakeAsyncCopy(Ref src_ref, Ref dst_ref, Semaphore sem,
                                  AsyncCopyOptions options = {});

// Creates and immediately starts an asynchronous DMA copy from `src_ref` to
// `dst_ref` (Pallas `pltpu.async_copy`).
AsyncCopyDescriptor AsyncCopy(Ref src_ref, Ref dst_ref, Semaphore sem,
                              AsyncCopyOptions options = {});

// Copies `src_ref` to `dst_ref` and waits for completion, using a DMA semaphore
// scoped to the copy (Pallas `pltpu.sync_copy`). Both refs are safe to reuse
// once it returns.
void SyncCopy(Ref src_ref, Ref dst_ref);

// Creates a descriptor for a remote asynchronous DMA copy without starting it
// (Pallas `pltpu.make_async_remote_copy`).
AsyncRemoteCopyDescriptor MakeAsyncRemoteCopy(Ref src_ref, Ref dst_ref,
                                              Semaphore send_sem,
                                              Semaphore recv_sem,
                                              AsyncRemoteCopyOptions options);

// Creates and immediately starts a remote asynchronous DMA copy
// (Pallas `pltpu.async_remote_copy`).
AsyncRemoteCopyDescriptor AsyncRemoteCopy(Ref src_ref, Ref dst_ref,
                                          Semaphore send_sem,
                                          Semaphore recv_sem,
                                          AsyncRemoteCopyOptions options);

// Queries the runtime device ID as an i32 Scalar (`tpu.device_id`).
Scalar DeviceId();

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_DMA_H_
