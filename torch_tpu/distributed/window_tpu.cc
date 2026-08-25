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

#include "torch_tpu/distributed/window_tpu.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/container/flat_hash_map.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch_tpu/common/macro_utils.h"
#if TT_TORCH_VERSION_GE(2, 14)
#include "torch/csrc/distributed/c10d/Window.hpp"
#endif
#include "torch/csrc/distributed/c10d/Work.hpp"
#include "torch_tpu/common/device_type.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/pjrt/pjrt_utils.h"

#if TT_TORCH_VERSION_GE(2, 14)
namespace torch_tpu {
namespace {

// Synchronizes the work handle if the operation was called synchronously.
c10::intrusive_ptr<c10d::Work> CompleteWork(c10::intrusive_ptr<c10d::Work> work,
                                            bool asyncOp) {
  if (!asyncOp) {
    work->wait(kNoTimeout);
  }
  return work;
}

}  // namespace

WindowTPU::WindowTPU(c10::intrusive_ptr<c10d::Backend> backend, int64_t rank)
    : backend_(std::move(backend)), local_rank_(rank) {}

void WindowTPU::tensor_register(const at::Tensor& tensor, bool owning) {
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=contiguous check
      tensor.is_contiguous(), error::kInvalidArgument)
      << "Window tensor must be contiguous.";
  registered_tensor_ = tensor;

  // Export local RMA window descriptor if the active PJRT plugin supports RMA.
  if (TpuSupportsRma() &&
      tensor.device().type() == GetPrivateUse1DeviceType()) {
    auto buf_ref =
        MaterializeAndReturn(tensor, MaterializationReason::kCpuTransfer);
    if (buf_ref.ok()) {
      auto desc = TpuRmaExportWindow(*buf_ref);
      if (desc.ok()) {
        local_window_descriptor_ = *desc;
      }
    }
  }
}

void WindowTPU::tensor_deregister() {
  registered_tensor_.reset();
  local_window_descriptor_.reset();
  peer_window_descriptors_.clear();
  peer_remote_windows_.clear();
}

void WindowTPU::set_peer_window_descriptors(
    const absl::flat_hash_map<int64_t, std::string>& peer_window_descriptors) {
  peer_window_descriptors_ = peer_window_descriptors;
  peer_remote_windows_.clear();
  if (TpuSupportsRma()) {
    for (const auto& [rank, desc] : peer_window_descriptors_) {
      auto handle = TpuRmaImportRemoteWindow(desc);
      if (handle.ok()) {
        peer_remote_windows_[rank] = *std::move(handle);
      }
    }
  }
}

c10::intrusive_ptr<c10d::Work> WindowTPU::PutDeviceToHostDma(
    const at::Tensor& tensor, int64_t targetOffsetNelems, bool asyncOp) {
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=contiguous check
      tensor.is_contiguous(), error::kInvalidArgument)
      << "Source device tensor must be contiguous for zero-copy DMA.";
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=offset check
      targetOffsetNelems >= 0, error::kInvalidArgument)
      << "targetOffsetNelems must be non-negative, got " << targetOffsetNelems;
  int64_t dst_byte_offset = 0;
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=overflow check
      !__builtin_mul_overflow(targetOffsetNelems, tensor.element_size(),
                              &dst_byte_offset),
      error::kInvalidArgument)
      << "targetOffsetNelems * element_size overflowed";
  const int64_t copy_bytes = tensor.nbytes();
  const int64_t registered_bytes = registered_tensor_->nbytes();

  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=bounds check
      dst_byte_offset >= 0 && dst_byte_offset + copy_bytes <= registered_bytes,
      error::kInvalidArgument)
      << "Target offset and size exceeds registered window buffer size: "
      << dst_byte_offset + copy_bytes << " > " << registered_bytes;

  uint8_t* dst_ptr =
      reinterpret_cast<uint8_t*>(registered_tensor_->data_ptr()) +
      dst_byte_offset;

  TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=MaterializeAndReturn error
      const DeviceBufferRef buffer_ref,
      MaterializeAndReturn(tensor, MaterializationReason::kCpuTransfer));

  TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=D2H error
      auto result, TpuAsyncDmaCopyDtoH(buffer_ref, dst_ptr, copy_bytes));

  return CompleteWork(
      c10::make_intrusive<WorkTPU>(std::move(result.future),
                                   std::move(result.buffer_hold)),
      asyncOp);
}

c10::intrusive_ptr<c10d::Work> WindowTPU::PutHostToDeviceDma(
    const at::Tensor& tensor, int64_t targetOffsetNelems, bool asyncOp) {
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=contiguous check
      tensor.is_contiguous(), error::kInvalidArgument)
      << "Source host tensor must be contiguous for zero-copy DMA.";
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=offset check
      targetOffsetNelems >= 0, error::kInvalidArgument)
      << "targetOffsetNelems must be non-negative, got " << targetOffsetNelems;
  int64_t dst_byte_offset = 0;
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=overflow check
      !__builtin_mul_overflow(targetOffsetNelems, tensor.element_size(),
                              &dst_byte_offset),
      error::kInvalidArgument)
      << "targetOffsetNelems * element_size overflowed";
  const int64_t copy_bytes = tensor.nbytes();
  const int64_t registered_bytes = registered_tensor_->nbytes();

  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=bounds check
      dst_byte_offset >= 0 && dst_byte_offset + copy_bytes <= registered_bytes,
      error::kInvalidArgument)
      << "Target offset and size exceeds registered window buffer size: "
      << dst_byte_offset + copy_bytes << " > " << registered_bytes;

  const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(tensor.data_ptr());

  TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=MaterializeAndReturn error
      const DeviceBufferRef buffer_ref,
      MaterializeAndReturn(*registered_tensor_,
                           MaterializationReason::kCpuTransfer));

  TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=H2D error
      auto result,
      TpuAsyncDmaCopyHtoD(src_ptr, buffer_ref, dst_byte_offset, copy_bytes));

  return CompleteWork(
      c10::make_intrusive<WorkTPU>(std::move(result.future),
                                   std::move(result.buffer_hold)),
      asyncOp);
}

c10::intrusive_ptr<c10d::Work> WindowTPU::PutPeerRma(const at::Tensor& tensor,
                                                     int64_t dstRank,
                                                     int64_t targetOffsetNelems,
                                                     bool asyncOp) {
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=contiguous check
      tensor.is_contiguous(), error::kInvalidArgument)
      << "Source device tensor must be contiguous for zero-copy RMA Put.";
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=registered check
      registered_tensor_.has_value(), error::kFailedPrecondition)
      << "Window tensor must be registered before PutPeerRma.";
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=dtype match check
      tensor.dtype() == registered_tensor_->dtype(), error::kInvalidArgument)
      << "Source tensor dtype (" << tensor.dtype().name()
      << ") does not match registered window tensor dtype ("
      << registered_tensor_->dtype().name() << ").";
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=offset check
      targetOffsetNelems >= 0, error::kInvalidArgument)
      << "targetOffsetNelems must be non-negative, got " << targetOffsetNelems;
  int64_t dst_byte_offset = 0;
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=overflow check
      !__builtin_mul_overflow(targetOffsetNelems, tensor.element_size(),
                              &dst_byte_offset),
      error::kInvalidArgument)
      << "targetOffsetNelems * element_size overflowed";
  const int64_t copy_bytes = tensor.nbytes();
  const int64_t registered_bytes = registered_tensor_->nbytes();

  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=bounds check
      dst_byte_offset >= 0 && dst_byte_offset + copy_bytes <= registered_bytes,
      error::kInvalidArgument)
      << "Target offset and size exceeds registered window buffer size: "
      << dst_byte_offset + copy_bytes << " > " << registered_bytes;

  TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=MaterializeAndReturn error
      const DeviceBufferRef buffer_ref,
      MaterializeAndReturn(tensor, MaterializationReason::kCpuTransfer));

  auto& remote_window = peer_remote_windows_[dstRank];
  if (remote_window == nullptr) {
    auto desc_it = peer_window_descriptors_.find(dstRank);
    TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=desc exists check
        desc_it != peer_window_descriptors_.end(), error::kInvalidArgument)
        << "No RMA window descriptor found for destination rank " << dstRank;
    TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=import error
        auto imported_handle, TpuRmaImportRemoteWindow(desc_it->second));
    remote_window = std::move(imported_handle);
  }

  TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=Put error
      auto result, TpuAsyncRmaPut(buffer_ref, /*src_byte_offset=*/0,
                                  remote_window, dst_byte_offset, copy_bytes));
  return CompleteWork(
      c10::make_intrusive<WorkTPU>(std::move(result.future),
                                   std::move(result.buffer_hold)),
      asyncOp);
}

// Fallback: Two-sided backend_->send() when RMA descriptors are not mapped.
c10::intrusive_ptr<c10d::Work> WindowTPU::PutPeerP2P(const at::Tensor& tensor,
                                                     int64_t dstRank,
                                                     bool asyncOp) {
  std::vector<at::Tensor> tensors = {tensor.contiguous()};
  return CompleteWork(
      backend_->send(tensors, static_cast<int>(dstRank), /*tag=*/0), asyncOp);
}

c10::intrusive_ptr<c10d::Work> WindowTPU::put(const at::Tensor& tensor,
                                              int64_t dstRank,
                                              int64_t targetOffsetNelems,
                                              bool asyncOp,
                                              const c10d::PutOptions& opts) {
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=registered check
      registered_tensor_.has_value(), error::kFailedPrecondition)
      << "Window tensor must be registered before calling put.";
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=offset check
      targetOffsetNelems >= 0, error::kInvalidArgument)
      << "targetOffsetNelems must be non-negative, got " << targetOffsetNelems;

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

  // Case 3: Peer Rank One-Sided RMA transfer
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=rma support check
      TpuSupportsRma(), error::kUnavailable)
      << "Peer-to-peer Window put requires a PJRT plugin with "
         "PJRT_Rma_Extension support.";

  if (peer_window_descriptors_.contains(dstRank)) {
    return PutPeerRma(tensor, dstRank, targetOffsetNelems, asyncOp);
  }

  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=fallback offset check
      targetOffsetNelems == 0, error::kInvalidArgument)
      << "Non-zero targetOffsetNelems (" << targetOffsetNelems
      << ") is not supported for remote peer P2P transfers without RMA "
         "descriptors.";
  return PutPeerP2P(tensor, dstRank, asyncOp);
}

c10::intrusive_ptr<c10d::Work> WindowTPU::signal(
    int64_t peerRank, bool asyncOp, const c10d::SignalOptions& opts) {
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=registered check
      registered_tensor_.has_value(), error::kFailedPrecondition)
      << "Window tensor must be registered before calling signal.";
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=rma support check
      TpuSupportsRma(), error::kUnavailable)
      << "Window signal requires a PJRT plugin with PJRT_Rma_Extension "
         "support.";

  uint64_t signal_id = peer_signal_seq_[peerRank]++;
  auto desc_it = peer_window_descriptors_.find(peerRank);
  if (desc_it != peer_window_descriptors_.end()) {
    auto& remote_window = peer_remote_windows_[peerRank];
    if (remote_window == nullptr) {
      auto imported_handle = TpuRmaImportRemoteWindow(desc_it->second);
      if (imported_handle.ok()) {
        remote_window = *std::move(imported_handle);
      }
    }
    if (remote_window != nullptr) {
      TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=signal error
          auto result, TpuAsyncRmaSignal(remote_window, signal_id));
      return CompleteWork(
          c10::make_intrusive<WorkTPU>(std::move(result.future),
                                       std::move(result.buffer_hold)),
          asyncOp);
    }
    TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=signal error
        auto result, TpuAsyncRmaSignal(desc_it->second, signal_id));
    return CompleteWork(
        c10::make_intrusive<WorkTPU>(std::move(result.future),
                                     std::move(result.buffer_hold)),
        asyncOp);
  }

  std::vector<at::Tensor> tensors = {*registered_tensor_};
  return CompleteWork(
      backend_->send(tensors, static_cast<int>(peerRank), /*tag=*/0), asyncOp);
}

c10::intrusive_ptr<c10d::Work> WindowTPU::wait_signal(
    int64_t peerRank, bool asyncOp, const c10d::WaitSignalOptions& opts) {
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=registered check
      registered_tensor_.has_value(), error::kFailedPrecondition)
      << "Window tensor must be registered before calling wait_signal.";
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=rma support check
      TpuSupportsRma(), error::kUnavailable)
      << "Window wait_signal requires a PJRT plugin with PJRT_Rma_Extension "
         "support.";

  uint64_t signal_id = peer_wait_seq_[peerRank]++;
  if (local_window_descriptor_.has_value() &&
      peer_window_descriptors_.contains(peerRank)) {
    TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=wait_signal error
        auto result,
        TpuAsyncRmaWaitSignal(*local_window_descriptor_, signal_id));
    return CompleteWork(
        c10::make_intrusive<WorkTPU>(std::move(result.future),
                                     std::move(result.buffer_hold)),
        asyncOp);
  }

  std::vector<at::Tensor> tensors = {*registered_tensor_};
  return CompleteWork(
      backend_->recv(tensors, static_cast<int>(peerRank), /*tag=*/0), asyncOp);
}

at::Tensor WindowTPU::map_remote_tensor(int64_t rank) {
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=registered check
      registered_tensor_.has_value(), error::kFailedPrecondition)
      << "Window tensor must be registered.";
  return *registered_tensor_;
}

c10d::WindowAttr WindowTPU::get_attr(int64_t peerRank) {
  c10d::WindowAttr attr;
  attr.access_type = c10d::WindowAccessType::UNIFIED;
  return attr;
}

}  // namespace torch_tpu
#endif  // TT_TORCH_VERSION_GE(2, 14)
