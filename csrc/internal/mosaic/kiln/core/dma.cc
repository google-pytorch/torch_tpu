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

#include "csrc/internal/mosaic/kiln/core/dma.h"

#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/control_flow.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/memory.h"
#include "csrc/internal/mosaic/kiln/core/mlir_utils.h"
#include "csrc/internal/mosaic/kiln/core/ref.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/semaphore.h"
#include "csrc/internal/mosaic/kiln/core/specs.h"
#include "mlir/IR/Builders.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace torch_tpu::kiln {
namespace {

namespace tpu = mlir::tpu;

void ValidateCopyRefs(const Ref& src_ref, const Ref& dst_ref,
                      const Semaphore& sem) {
  ABSL_CHECK_EQ(src_ref.shape(), dst_ref.shape())  // CRASH_OK
      << "DMA source and destination Refs must have matching shapes";
  ABSL_CHECK(  // CRASH_OK
      sem.is_dma() || (src_ref.memory_space() != MemorySpace::kSmem &&
                       dst_ref.memory_space() != MemorySpace::kSmem))
      << "Non-DMA semaphores are not supported for DMAs involving SMEM";
}

enum class RemoteWaitDirection {
  kSend,
  kRecv,
};

void EmitRemoteWait(const AsyncRemoteCopyDescriptor& desc,
                    RemoteWaitDirection direction) {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  const bool wait_target = direction == RemoteWaitDirection::kRecv;
  tpu::WaitDMAOp::create(
      b, internal::unwrap(desc.src_ref()), internal::unwrap(desc.send_sem()),
      internal::unwrap(desc.dst_ref()), internal::unwrap(desc.recv_sem()),
      internal::UnwrapOptionalScalar(desc.device_id()),
      internal::UnwrapOptionalScalar(desc.core_id()),
      /*subcore_id=*/nullptr, /*priority=*/0, /*strict_ordering=*/false,
      wait_target);
}

}  // namespace

AsyncCopyDescriptor::AsyncCopyDescriptor(Ref src_ref, Ref dst_ref,
                                         Semaphore sem,
                                         AsyncCopyOptions options)
    : src_ref_(src_ref), dst_ref_(dst_ref), sem_(sem), options_(options) {
  ValidateCopyRefs(src_ref_, dst_ref_, sem_);
  ABSL_CHECK(options_.priority >= 0 && options_.priority <= 1)  // CRASH_OK
      << "DMA priority must be 0 or 1";
}

void AsyncCopyDescriptor::Start() const {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  tpu::EnqueueDMAOp::create(b, internal::unwrap(src_ref_),
                            /*source_semaphore=*/nullptr,
                            internal::unwrap(dst_ref_), internal::unwrap(sem_),
                            /*device_id=*/nullptr, /*core_id=*/nullptr,
                            options_.priority);
}

void AsyncCopyDescriptor::Wait() const {
  ABSL_CHECK(sem_.is_dma())  // CRASH_OK
      << "Wait() requires a DMA semaphore (use Semaphore::Wait() for regular "
         "semaphores)";
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  tpu::WaitDMAOp::create(b, internal::unwrap(src_ref_),
                         /*source_semaphore=*/nullptr,
                         internal::unwrap(dst_ref_), internal::unwrap(sem_),
                         /*device_id=*/nullptr, /*core_id=*/nullptr,
                         /*subcore_id=*/nullptr, options_.priority,
                         /*strict_ordering=*/false, /*wait_target=*/true);
}

AsyncRemoteCopyDescriptor::AsyncRemoteCopyDescriptor(
    Ref src_ref, Ref dst_ref, Semaphore send_sem, Semaphore recv_sem,
    AsyncRemoteCopyOptions options)
    : src_ref_(src_ref),
      dst_ref_(dst_ref),
      send_sem_(send_sem),
      recv_sem_(recv_sem) {
  ABSL_CHECK(options.device_id.has_value() ||  // CRASH_OK
             options.core_id.has_value())
      << "Remote DMA requires device_id or core_id to be specified";
  ABSL_CHECK_EQ(send_sem_.is_dma(), recv_sem_.is_dma())  // CRASH_OK
      << "DMA source and target semaphores must have the same kind";
  ValidateCopyRefs(src_ref_, dst_ref_, recv_sem_);
  device_id_ = internal::OptionalI32Scalar(options.device_id,
                                           "AsyncRemoteCopyOptions::device_id");
  core_id_ = internal::OptionalI32Scalar(options.core_id,
                                         "AsyncRemoteCopyOptions::core_id");
}

void AsyncRemoteCopyDescriptor::Start() const {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  tpu::EnqueueDMAOp::create(
      b, internal::unwrap(src_ref_), internal::unwrap(send_sem_),
      internal::unwrap(dst_ref_), internal::unwrap(recv_sem_),
      internal::UnwrapOptionalScalar(device_id_),
      internal::UnwrapOptionalScalar(core_id_));
}

void AsyncRemoteCopyDescriptor::Wait() const {
  WaitSend();
  WaitRecv();
}

void AsyncRemoteCopyDescriptor::WaitSend() const {
  ABSL_CHECK(send_sem_.is_dma())  // CRASH_OK
      << "WaitSend() requires a DMA semaphore";
  EmitRemoteWait(*this, RemoteWaitDirection::kSend);
}

void AsyncRemoteCopyDescriptor::WaitRecv() const {
  ABSL_CHECK(recv_sem_.is_dma())  // CRASH_OK
      << "WaitRecv() requires a DMA semaphore (use Semaphore::Wait() for "
         "regular semaphores)";
  EmitRemoteWait(*this, RemoteWaitDirection::kRecv);
}

AsyncCopyDescriptor MakeAsyncCopy(Ref src_ref, Ref dst_ref, Semaphore sem,
                                  AsyncCopyOptions options) {
  return AsyncCopyDescriptor(src_ref, dst_ref, sem, options);
}

AsyncCopyDescriptor AsyncCopy(Ref src_ref, Ref dst_ref, Semaphore sem,
                              AsyncCopyOptions options) {
  AsyncCopyDescriptor desc = MakeAsyncCopy(src_ref, dst_ref, sem, options);
  desc.Start();
  return desc;
}

void SyncCopy(Ref src_ref, Ref dst_ref) {
  RunScoped([&src_ref, &dst_ref] {
    AsyncCopy(src_ref, dst_ref, AllocaDmaSemaphore()).Wait();
  });
}

AsyncRemoteCopyDescriptor MakeAsyncRemoteCopy(Ref src_ref, Ref dst_ref,
                                              Semaphore send_sem,
                                              Semaphore recv_sem,
                                              AsyncRemoteCopyOptions options) {
  return AsyncRemoteCopyDescriptor(src_ref, dst_ref, send_sem, recv_sem,
                                   options);
}

AsyncRemoteCopyDescriptor AsyncRemoteCopy(Ref src_ref, Ref dst_ref,
                                          Semaphore send_sem,
                                          Semaphore recv_sem,
                                          AsyncRemoteCopyOptions options) {
  AsyncRemoteCopyDescriptor desc =
      MakeAsyncRemoteCopy(src_ref, dst_ref, send_sem, recv_sem, options);
  desc.Start();
  return desc;
}

Scalar DeviceId() {
  mlir::ImplicitLocOpBuilder& b = internal::CurrentBuilder();
  return internal::wrap<Scalar>(tpu::DeviceIdOp::create(b, b.getI32Type()),
                                i32);
}

}  // namespace torch_tpu::kiln
