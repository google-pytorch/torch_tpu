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

#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch_tpu/common/macro_utils.h"
#if TT_TORCH_VERSION_GE(2, 14)
#include "torch/csrc/distributed/c10d/Window.hpp"
#endif
#include "torch/csrc/distributed/c10d/Work.hpp"
#include "torch_tpu/common/error_utils.h"
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
// Transfer Modalities:
// 1. Local PCIe DMA (Host <-> Device): One-sided asynchronous DMA transfers
//    between Host RAM and TPU HBM using the PJRT RawBuffer C API extension.
// 2. Cross-Chip Inter-Chip Interconnect (ICI) One-Sided RMA: True one-sided
//    direct hardware RDMA transfers into destination HBM at arbitrary element
//    offsets and decoupled signaling using the PJRT RMA C API extension,
//    bypassing remote CPU involvement.
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

  // Exchanges and caches peer RMA window descriptors for cross-rank transfers.
  void set_peer_window_descriptors(
      const absl::flat_hash_map<int64_t, std::string>& peer_window_descriptors);
  const std::optional<std::string>& local_window_descriptor() const {
    return local_window_descriptor_;
  }

 private:
  // Internal helper methods for the Put transfer modalities.
  c10::intrusive_ptr<c10d::Work> PutDeviceToHostDma(const at::Tensor& tensor,
                                                    int64_t targetOffsetNelems,
                                                    bool asyncOp);
  c10::intrusive_ptr<c10d::Work> PutHostToDeviceDma(const at::Tensor& tensor,
                                                    int64_t targetOffsetNelems,
                                                    bool asyncOp);
  c10::intrusive_ptr<c10d::Work> PutPeerRma(const at::Tensor& tensor,
                                            int64_t dstRank,
                                            int64_t targetOffsetNelems,
                                            bool asyncOp);
  c10::intrusive_ptr<c10d::Work> PutPeerP2P(const at::Tensor& tensor,
                                            int64_t dstRank, bool asyncOp);

  c10::intrusive_ptr<c10d::Backend> backend_;
  int64_t local_rank_{0};
  std::optional<at::Tensor> registered_tensor_;
  std::optional<std::string> local_window_descriptor_;
  absl::flat_hash_map<int64_t, std::string> peer_window_descriptors_;
  absl::flat_hash_map<int64_t, std::shared_ptr<void>> peer_remote_windows_;
  absl::flat_hash_map<int64_t, uint64_t> peer_signal_seq_;
  absl::flat_hash_map<int64_t, uint64_t> peer_wait_seq_;
};

}  // namespace torch_tpu

#endif  // TT_TORCH_VERSION_GE(2, 14)

#endif  // TORCH_TPU_DISTRIBUTED_WINDOW_TPU_H_
