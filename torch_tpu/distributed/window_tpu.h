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

#ifndef TORCH_TPU_DISTRIBUTED_WINDOW_TPU_H_
#define TORCH_TPU_DISTRIBUTED_WINDOW_TPU_H_

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch_tpu/csrc/common/macro_utils.h"
#if TT_TORCH_VERSION_GE(2, 14)
#include "torch/csrc/distributed/c10d/Window.hpp"
#endif
#include "torch/csrc/distributed/c10d/Work.hpp"
#include "torch_tpu/csrc/common/error_utils.h"
#include "xla/future.h"

#if TT_TORCH_VERSION_GE(2, 14)
namespace torch_tpu {

// Work handle representing asynchronous TPU zero-copy DMA or RMA transfers.
class WorkTPU : public c10d::Work {
 public:
  explicit WorkTPU(xla::Future<> future, std::shared_ptr<void> hold = nullptr)
      : future_(std::move(future)), hold_(std::move(hold)) {}

  bool isCompleted() override { return future_.IsReady(); }

  bool isSuccess() const override {
    return future_.IsReady() && future_.Await().ok();
  }

  bool wait(std::chrono::milliseconds timeout) override {
    TT_THROW_IF_ERROR(future_.Await()) << "TPU Window DMA Work failed: ";
    return true;
  }

  void synchronize() override { wait(kNoTimeout); }

 private:
  xla::Future<> future_;
  std::shared_ptr<void> hold_;
};

// WindowTPU: One-sided RMA communication and zero-copy DMA window for TPUs
// conforming to PyTorch's native c10d::Window interface.
//
// Communication Modalities:
// 1. Local PCIe Transfers (Host <-> Device): Implemented as true one-sided
//    asynchronous hardware DMA (via PJRT RawBuffer extension).
// 2. Peer P2P Transfers (ICI Cross-Chip): Currently emulated via two-sided
//    ProcessGroup send/recv rendezvous.
//
// Downsides of Emulating One-Way DMA with Two-Way Send/Recv:
// - Active Receiver CPU Overhead: The destination rank's CPU must actively
//   participate by calling wait_signal() (which calls CrossHostReceiveBuffers)
//   to allocate placeholder buffers and publish descriptors to c10d::Store.
// - Coupled Synchronization & Data Movement: put() only initiates a send; data
//   is received only when wait_signal() executes, preventing pure decoupled
//   atomic signaling.
// - Offset Handling & Lack of P2P Offset Support: This implementation does
//   not support non-zero targetOffsetNelems for remote peer P2P transfers (and
//   throws an error if requested). Supporting arbitrary offsets in two-sided
//   emulation requires allocating temporary staging buffers in HBM and
//   performing a secondary Device-to-Device slice copy on the receiver,
//   whereas true one-sided RMA writes directly to the target HBM offset in a
//   single hardware pass with zero staging memory.
// - Rendezvous Deadlock Risk: Because CrossHostSendBuffers blocks on descriptor
//   availability in c10d::Store, circular or un-synchronized P2P patterns can
//   deadlock if send and recv calls are not strictly sequenced.
// - Increased Latency & Coordination Overhead: Incurs host-level c10d::Store
//   TCP round-trips to exchange transfer descriptors on the critical path
//   before hardware ICI DMA can initiate, as well as receiver thread
//   synchronization delays and dynamic buffer allocation/binding overhead
//   (compared to true hardware DMA which programs DMA registers in < 1 us
//   directly against pre-registered static memory).
//
// TODO(cbasile): Upgrade peer P2P transfers to true one-sided hardware RDMA
// over ICI once the PJRT C API exports a native RMA / direct cross-chip DMA
// extension, allowing direct HBM writes without receiver CPU involvement.
class WindowTPU : public c10d::Window {
 public:
  explicit WindowTPU(c10::intrusive_ptr<c10d::Backend> backend,
                     int64_t rank = 0);
  ~WindowTPU() override = default;

  WindowTPU(const WindowTPU&) = delete;
  WindowTPU(WindowTPU&&) = delete;
  WindowTPU& operator=(const WindowTPU&) = delete;
  WindowTPU& operator=(WindowTPU&&) = delete;

  // Collective buffer registration with low-level hardware memory hooks.
  void tensor_register(const at::Tensor& tensor, bool owning) override;
  void tensor_deregister() override;

  // One-sided zero-copy DMA write (Put) into destination rank at an element
  // offset.
  c10::intrusive_ptr<c10d::Work> put(const at::Tensor& tensor, int64_t dstRank,
                                     int64_t targetOffsetNelems, bool asyncOp,
                                     const c10d::PutOptions& opts) override;

  // Point-to-point signaling synchronization.
  c10::intrusive_ptr<c10d::Work> signal(
      int64_t peerRank, bool asyncOp, const c10d::SignalOptions& opts) override;

  c10::intrusive_ptr<c10d::Work> wait_signal(
      int64_t peerRank, bool asyncOp,
      const c10d::WaitSignalOptions& opts) override;

  at::Tensor map_remote_tensor(int64_t rank) override;
  c10d::WindowAttr get_attr(int64_t peerRank) override;

 private:
  // Internal helper methods for the 3 distinct Put transfer modalities.
  c10::intrusive_ptr<c10d::Work> PutDeviceToHostDma(const at::Tensor& tensor,
                                                    int64_t targetOffsetNelems,
                                                    bool asyncOp);
  c10::intrusive_ptr<c10d::Work> PutHostToDeviceDma(const at::Tensor& tensor,
                                                    int64_t targetOffsetNelems,
                                                    bool asyncOp);
  c10::intrusive_ptr<c10d::Work> PutPeerP2P(const at::Tensor& tensor,
                                            int64_t dstRank, bool asyncOp);

  c10::intrusive_ptr<c10d::Backend> backend_;
  int64_t local_rank_{0};
  std::optional<at::Tensor> registered_tensor_;
};

}  // namespace torch_tpu

#endif  // TT_TORCH_VERSION_GE(2, 14)

#endif  // TORCH_TPU_DISTRIBUTED_WINDOW_TPU_H_
