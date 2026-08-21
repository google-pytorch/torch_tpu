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

#include "torch_tpu/csrc/distributed/window_tpu.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch_tpu/csrc/common/macro_utils.h"
#if TT_TORCH_VERSION_GE(2, 14)
#include "torch/csrc/distributed/c10d/Window.hpp"
#endif
#include "torch/csrc/distributed/c10d/Work.hpp"
#include "torch_tpu/csrc/common/device_type.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/structured_log_buffer.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/pjrt/pjrt_utils.h"

#if TT_TORCH_VERSION_GE(2, 14)
namespace torch_tpu {

WindowTPU::WindowTPU(c10::intrusive_ptr<c10d::Backend> backend, int64_t rank)
    : backend_(std::move(backend)), local_rank_(rank) {}

void WindowTPU::tensor_register(const at::Tensor& tensor, bool owning) {
  TT_CHECK_THROW(tensor.is_contiguous(), error::kInvalidArgument)
      << "Window tensor must be contiguous.";
  registered_tensor_ = tensor;
}

void WindowTPU::tensor_deregister() { registered_tensor_.reset(); }

c10::intrusive_ptr<c10d::Work> WindowTPU::PutDeviceToHostDma(
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

  auto work = c10::make_intrusive<WorkTPU>(std::move(result.future),
                                           std::move(result.buffer_hold));
  if (!asyncOp) {
    work->wait(kNoTimeout);
  }
  return work;
}

c10::intrusive_ptr<c10d::Work> WindowTPU::PutHostToDeviceDma(
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

  auto work = c10::make_intrusive<WorkTPU>(std::move(result.future),
                                           std::move(result.buffer_hold));
  if (!asyncOp) {
    work->wait(kNoTimeout);
  }
  return work;
}

// TODO(cbasile): Replace two-sided backend_->send() with true one-sided
// hardware RDMA over ICI once the PJRT C API exports direct cross-chip DMA.
c10::intrusive_ptr<c10d::Work> WindowTPU::PutPeerP2P(const at::Tensor& tensor,
                                                     int64_t dstRank,
                                                     bool asyncOp) {
  std::vector<at::Tensor> tensors = {tensor.contiguous()};
  auto work = backend_->send(tensors, static_cast<int>(dstRank), /*tag=*/0);
  if (!asyncOp) {
    work->wait(kNoTimeout);
  }
  return work;
}

c10::intrusive_ptr<c10d::Work> WindowTPU::put(const at::Tensor& tensor,
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
c10::intrusive_ptr<c10d::Work> WindowTPU::signal(
    int64_t peerRank, bool asyncOp, const c10d::SignalOptions& opts) {
  TT_CHECK_THROW(registered_tensor_.has_value(), error::kFailedPrecondition)
      << "Window tensor must be registered before calling signal.";
  std::vector<at::Tensor> tensors = {*registered_tensor_};
  auto work = backend_->send(tensors, static_cast<int>(peerRank), /*tag=*/0);
  if (!asyncOp) {
    work->wait(kNoTimeout);
  }
  return work;
}

c10::intrusive_ptr<c10d::Work> WindowTPU::wait_signal(
    int64_t peerRank, bool asyncOp, const c10d::WaitSignalOptions& opts) {
  TT_CHECK_THROW(registered_tensor_.has_value(), error::kFailedPrecondition)
      << "Window tensor must be registered before calling wait_signal.";
  std::vector<at::Tensor> tensors = {*registered_tensor_};
  auto work = backend_->recv(tensors, static_cast<int>(peerRank), /*tag=*/0);
  if (!asyncOp) {
    work->wait(kNoTimeout);
  }
  return work;
}

at::Tensor WindowTPU::map_remote_tensor(int64_t rank) {
  TT_CHECK_THROW(registered_tensor_.has_value(), error::kFailedPrecondition)
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
