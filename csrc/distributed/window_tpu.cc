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

#include "csrc/distributed/window_tpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/macro_utils.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/Types.hpp"
#if TT_TORCH_VERSION_GE(2, 14)
#include "torch/csrc/distributed/c10d/Window.hpp"
#endif
#include "csrc/common/device_type.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/pjrt/pjrt_utils.h"
#include "torch/csrc/distributed/c10d/Work.hpp"
namespace torch_tpu {

static std::atomic<int> g_window_num_stripes{4};
static std::atomic<int64_t> g_window_stripe_chunk_mb{8};

void PySetWindowNumStripes(int num_stripes) {
  TT_CHECK_THROW(num_stripes > 0, error::kInvalidArgument)
      << "expected window_num_stripes to be > 0, got " << num_stripes;
  g_window_num_stripes.store(num_stripes);
}

int PyGetWindowNumStripes() { return g_window_num_stripes.load(); }

void PySetWindowStripeChunkMb(int64_t chunk_mb) {
  TT_CHECK_THROW(chunk_mb > 0, error::kInvalidArgument)
      << "expected window_stripe_chunk_mb to be > 0, got " << chunk_mb;
  g_window_stripe_chunk_mb.store(chunk_mb);
}

int64_t PyGetWindowStripeChunkMb() { return g_window_stripe_chunk_mb.load(); }

namespace {

// Retrieves configured stripe chunk size in bytes (minimum bytes per DMA
// stripe). Default is 8 MB per stripe chunk.
int64_t GetStripeChunkSizeBytes() {
  return PyGetWindowStripeChunkMb() * 1024 * 1024;
}

// Retrieves maximum parallel DMA streams/stripes. Default is 4 (for TPU
// v5e/v6e).
int GetMaxStripes() { return PyGetWindowNumStripes(); }

}  // namespace

// Computes the optimal stripe count for a tensor of given byte size.
// Dynamically scales (assuming default 8 MB chunk size and max_stripes = 4):
//   - < 16 MB: 1 stripe (single DMA, minimizes dispatch latency)
//   - 16 MB - 23 MB: 2 stripes
//   - 24 MB - 31 MB: 3 stripes
//   - >= 32 MB: 4 stripes (saturates physical HBM/DMA channels)
int PyComputeWindowStripeCount(int64_t total_bytes) {
  if (total_bytes <= 0) return 1;
  const int64_t chunk_bytes = GetStripeChunkSizeBytes();
  const int max_stripes = GetMaxStripes();
  const int desired_stripes = total_bytes / chunk_bytes;
  return std::clamp(desired_stripes, 1, max_stripes);
}

#if TT_TORCH_VERSION_GE(2, 14)

WindowTpu::WindowTpu(c10::intrusive_ptr<c10d::Backend> backend, int64_t rank)
    : backend_(std::move(backend)), local_rank_(rank) {}

void WindowTpu::tensor_register(const at::Tensor& tensor, bool owning) {
  TT_CHECK_THROW(tensor.is_contiguous(), error::kInvalidArgument)
      << "Window tensor must be contiguous.";
  registered_tensor_ = tensor;
}

void WindowTpu::tensor_deregister() { registered_tensor_.reset(); }

c10::intrusive_ptr<c10d::Work> WindowTpu::PutDeviceToHostDma(
    const at::Tensor& tensor, int64_t targetOffsetNelems, bool asyncOp) {
  TT_CHECK_THROW(tensor.is_contiguous(), error::kInvalidArgument)
      << "Source device tensor must be contiguous for zero-copy DMA.";
  const int64_t element_size = tensor.element_size();
  const int64_t dst_byte_offset = targetOffsetNelems * element_size;
  const int64_t copy_bytes = tensor.nbytes();
  const int64_t registered_bytes = registered_tensor_->nbytes();

  TT_CHECK_THROW(dst_byte_offset + copy_bytes <= registered_bytes,
                 error::kInvalidArgument)
      << "Target offset and size exceeds registered window buffer size: "
      << dst_byte_offset + copy_bytes << " > " << registered_bytes;

  uint8_t* dst_ptr =
      reinterpret_cast<uint8_t*>(registered_tensor_->data_ptr()) +
      dst_byte_offset;

  TT_ASSIGN_OR_THROW(
      const DeviceBufferRef buffer_ref,
      MaterializeAndReturn(tensor, MaterializationReason::kCpuTransfer));

  TT_ASSIGN_OR_THROW(auto result,
                     TpuAsyncDmaCopyDtoH(buffer_ref, dst_ptr, copy_bytes));

  auto work = c10::make_intrusive<WorkTpu>(std::move(result.future),
                                           std::move(result.buffer_hold));
  if (!asyncOp) {
    work->wait(kNoTimeout);
  }
  return work;
}

c10::intrusive_ptr<c10d::Work> WindowTpu::PutHostToDeviceDma(
    const at::Tensor& tensor, int64_t targetOffsetNelems, bool asyncOp) {
  TT_CHECK_THROW(tensor.is_contiguous(), error::kInvalidArgument)
      << "Source host tensor must be contiguous for zero-copy DMA.";
  const int64_t element_size = tensor.element_size();
  const int64_t dst_byte_offset = targetOffsetNelems * element_size;
  const int64_t copy_bytes = tensor.nbytes();
  const int64_t registered_bytes = registered_tensor_->nbytes();

  TT_CHECK_THROW(dst_byte_offset + copy_bytes <= registered_bytes,
                 error::kInvalidArgument)
      << "Target offset and size exceeds registered window buffer size: "
      << dst_byte_offset + copy_bytes << " > " << registered_bytes;

  const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(tensor.data_ptr());

  TT_ASSIGN_OR_THROW(const DeviceBufferRef buffer_ref,
                     MaterializeAndReturn(*registered_tensor_,
                                          MaterializationReason::kCpuTransfer));

  TT_ASSIGN_OR_THROW(
      auto result,
      TpuAsyncDmaCopyHtoD(src_ptr, buffer_ref, dst_byte_offset, copy_bytes));

  auto work = c10::make_intrusive<WorkTpu>(std::move(result.future),
                                           std::move(result.buffer_hold));
  if (!asyncOp) {
    work->wait(kNoTimeout);
  }
  return work;
}

bool CompositeWorkTpu::isCompleted() {
  for (const auto& w : sub_works_) {
    if (w && !w->isCompleted()) return false;
  }
  return true;
}

bool CompositeWorkTpu::wait(std::chrono::milliseconds timeout) {
  if (timeout == kNoTimeout) {
    for (auto& w : sub_works_) {
      if (w && !w->wait(kNoTimeout)) {
        return false;
      }
    }
    return true;
  }

  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (auto& w : sub_works_) {
    if (!w) continue;
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      return false;
    }
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    if (!w->wait(remaining)) {
      return false;
    }
  }
  return true;
}

namespace {

// Flattens the tensor into a 1D contiguous view so PyTorch chunking evenly
// divides elements across stripes regardless of tensor shape, returning chunk
// views.
inline std::vector<at::Tensor> ChunkTensorForStriping(const at::Tensor& tensor,
                                                      int num_stripes) {
  // Dimension -1 infers total elements for flattening into a 1D view.
  return tensor.contiguous().view(-1).chunk(num_stripes, /*dim=*/0);
}

}  // namespace

// TODO(cbasile): Replace two-sided backend_->send() with true one-sided
// hardware RDMA over ICI once the PJRT C API exports direct cross-chip DMA.
c10::intrusive_ptr<c10d::Work> WindowTpu::PutPeerP2P(const at::Tensor& tensor,
                                                     int64_t dstRank,
                                                     bool asyncOp) {
  at::Tensor contiguous_tensor = tensor.contiguous();
  const int num_stripes =
      PyComputeWindowStripeCount(contiguous_tensor.nbytes());
  if (num_stripes <= 1) {
    std::vector<at::Tensor> tensors = {contiguous_tensor};
    auto work = backend_->send(tensors, static_cast<int>(dstRank), /*tag=*/0);
    if (!asyncOp) {
      work->wait(kNoTimeout);
    }
    return work;
  }

  // Multi-stream striping across parallel DMA channels.
  auto chunks = ChunkTensorForStriping(contiguous_tensor, num_stripes);
  std::vector<c10::intrusive_ptr<c10d::Work>> sub_works;
  sub_works.reserve(chunks.size());
  for (size_t i = 0; i < chunks.size(); ++i) {
    std::vector<at::Tensor> chunk_tensors = {chunks[i].contiguous()};
    sub_works.push_back(backend_->send(chunk_tensors, static_cast<int>(dstRank),
                                       static_cast<int>(i)));
  }

  auto composite_work =
      c10::make_intrusive<CompositeWorkTpu>(std::move(sub_works));
  if (!asyncOp) {
    composite_work->wait(kNoTimeout);
  }
  return composite_work;
}

c10::intrusive_ptr<c10d::Work> WindowTpu::put(const at::Tensor& tensor,
                                              int64_t dstRank,
                                              int64_t targetOffsetNelems,
                                              bool asyncOp,
                                              const c10d::PutOptions& opts) {
  TT_CHECK_THROW(registered_tensor_.has_value(), error::kFailedPrecondition)
      << "Window tensor must be registered before calling put.";

  const bool is_device_window =
      registered_tensor_->device().type() == GetPrivateUse1DeviceType();

  // Case 1: Local Device-to-Host (D2H) DMA transfer
  if (dstRank == local_rank_ && !is_device_window &&
      tensor.device().type() == GetPrivateUse1DeviceType()) {
    return PutDeviceToHostDma(tensor, targetOffsetNelems, asyncOp);
  }

  // Case 2: Local Host-to-Device (H2D) DMA transfer
  if (dstRank == local_rank_ && is_device_window && tensor.is_cpu()) {
    return PutHostToDeviceDma(tensor, targetOffsetNelems, asyncOp);
  }

  // Case 3: Peer Rank P2P transfer (currently emulated via ProcessGroup P2P)
  TT_CHECK_THROW(targetOffsetNelems == 0, error::kInvalidArgument)
      << "Non-zero targetOffsetNelems (" << targetOffsetNelems
      << ") is not supported for remote peer P2P transfers.";
  return PutPeerP2P(tensor, dstRank, asyncOp);
}

// TODO(cbasile): Transition signal and wait_signal to lightweight atomic
// counters once peer P2P transfers are true one-sided RDMA.
c10::intrusive_ptr<c10d::Work> WindowTpu::signal(
    int64_t peerRank, bool asyncOp, const c10d::SignalOptions& opts) {
  TT_CHECK_THROW(registered_tensor_.has_value(), error::kFailedPrecondition)
      << "Window tensor must be registered before calling signal.";
  const int num_stripes =
      PyComputeWindowStripeCount(registered_tensor_->nbytes());
  if (num_stripes <= 1) {
    std::vector<at::Tensor> tensors = {*registered_tensor_};
    auto work = backend_->send(tensors, static_cast<int>(peerRank), /*tag=*/0);
    if (!asyncOp) {
      work->wait(kNoTimeout);
    }
    return work;
  }

  auto chunks = ChunkTensorForStriping(*registered_tensor_, num_stripes);
  std::vector<c10::intrusive_ptr<c10d::Work>> sub_works;
  sub_works.reserve(chunks.size());
  for (size_t i = 0; i < chunks.size(); ++i) {
    std::vector<at::Tensor> chunk_tensors = {chunks[i].contiguous()};
    sub_works.push_back(backend_->send(
        chunk_tensors, static_cast<int>(peerRank), static_cast<int>(i)));
  }

  auto composite_work =
      c10::make_intrusive<CompositeWorkTpu>(std::move(sub_works));
  if (!asyncOp) {
    composite_work->wait(kNoTimeout);
  }
  return composite_work;
}

c10::intrusive_ptr<c10d::Work> WindowTpu::wait_signal(
    int64_t peerRank, bool asyncOp, const c10d::WaitSignalOptions& opts) {
  TT_CHECK_THROW(registered_tensor_.has_value(), error::kFailedPrecondition)
      << "Window tensor must be registered before calling wait_signal.";
  const int num_stripes =
      PyComputeWindowStripeCount(registered_tensor_->nbytes());
  if (num_stripes <= 1) {
    std::vector<at::Tensor> tensors = {*registered_tensor_};
    auto work = backend_->recv(tensors, static_cast<int>(peerRank), /*tag=*/0);
    if (!asyncOp) {
      work->wait(kNoTimeout);
    }
    return work;
  }

  auto chunks = ChunkTensorForStriping(*registered_tensor_, num_stripes);
  std::vector<c10::intrusive_ptr<c10d::Work>> sub_works;
  sub_works.reserve(chunks.size());
  for (size_t i = 0; i < chunks.size(); ++i) {
    std::vector<at::Tensor> chunk_tensors = {chunks[i]};
    sub_works.push_back(backend_->recv(
        chunk_tensors, static_cast<int>(peerRank), static_cast<int>(i)));
  }

  auto composite_work =
      c10::make_intrusive<CompositeWorkTpu>(std::move(sub_works));
  if (!asyncOp) {
    composite_work->wait(kNoTimeout);
  }
  return composite_work;
}

at::Tensor WindowTpu::map_remote_tensor(int64_t rank) {
  TT_CHECK_THROW(registered_tensor_.has_value(), error::kFailedPrecondition)
      << "Window tensor must be registered.";
  return *registered_tensor_;
}

c10d::WindowAttr WindowTpu::get_attr(int64_t peerRank) {
  c10d::WindowAttr attr;
  attr.access_type = c10d::WindowAccessType::UNIFIED;
  return attr;
}

#endif  // TT_TORCH_VERSION_GE(2, 14)

}  // namespace torch_tpu
