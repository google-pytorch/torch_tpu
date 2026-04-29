#ifndef TORCH_TPU_PJRT_RAW_TRANSFER_H_
#define TORCH_TPU_PJRT_RAW_TRANSFER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "ATen/core/TensorBody.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

struct RawBufferHolder;

class PjRtCopyFuture {
 public:
  PjRtCopyFuture() = default;

  explicit PjRtCopyFuture(std::vector<xla::Future<>> futures,
                          std::vector<at::Tensor> tensors_to_keep_alive)
      : futures_(std::move(futures)),
        tensors_(std::move(tensors_to_keep_alive)) {}

  explicit PjRtCopyFuture(
      std::vector<xla::Future<>> futures,
      std::vector<std::shared_ptr<RawBufferHolder>> c_api_holds,
      std::vector<std::shared_ptr<void>> holds, // Type-erased hold to avoid internal dependencies
      std::vector<at::Tensor> tensors_to_keep_alive)
      : futures_(std::move(futures)),
        c_api_holds_(std::move(c_api_holds)),
        holds_(std::move(holds)),
        tensors_(std::move(tensors_to_keep_alive)) {}

  void Await();
  void Append(PjRtCopyFuture other);
  void Append(std::vector<xla::Future<>> other_futures,
              std::vector<at::Tensor> other_tensors);

  void Append(
      std::vector<xla::Future<>> other_futures,
      std::vector<std::shared_ptr<RawBufferHolder>> other_c_api_holds,
      std::vector<std::shared_ptr<void>> other_holds);

  void Append(
      std::vector<xla::Future<>> other_futures,
      std::shared_ptr<RawBufferHolder> other_c_api_hold,
      std::shared_ptr<void> other_hold);

  void SetTag(std::string tag) { tag_ = std::move(tag); }
  const std::string& GetTag() const { return tag_; }

 private:
  std::vector<xla::Future<>> futures_;
  std::vector<std::shared_ptr<RawBufferHolder>> c_api_holds_;
  std::vector<std::shared_ptr<void>> holds_;
  std::vector<at::Tensor> tensors_;
  std::string tag_;
};

// Asynchronous raw transfers for single tensors
PjRtCopyFuture transfer_d2h_async(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim = {},
    const std::vector<int64_t>& dst_offsets_major_dim = {},
    const std::vector<int64_t>& copy_sizes_major_dim = {});

PjRtCopyFuture transfer_h2d_async(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim = {},
    const std::vector<int64_t>& dst_offsets_major_dim = {},
    const std::vector<int64_t>& copy_sizes_major_dim = {});

// Synchronous raw transfers for single tensors
void transfer_d2h(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim = {},
    const std::vector<int64_t>& dst_offsets_major_dim = {},
    const std::vector<int64_t>& copy_sizes_major_dim = {});

void transfer_h2d(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim = {},
    const std::vector<int64_t>& dst_offsets_major_dim = {},
    const std::vector<int64_t>& copy_sizes_major_dim = {});

// Batch transfers
PjRtCopyFuture transfer_d2h_batch_async(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim = {},
    const std::vector<int64_t>& dst_offsets_major_dim = {},
    const std::vector<int64_t>& copy_sizes_major_dim = {});

PjRtCopyFuture transfer_h2d_batch_async(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim = {},
    const std::vector<int64_t>& dst_offsets_major_dim = {},
    const std::vector<int64_t>& copy_sizes_major_dim = {});

// Synchronous batch transfers
void transfer_d2h_batch(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim = {},
    const std::vector<int64_t>& dst_offsets_major_dim = {},
    const std::vector<int64_t>& copy_sizes_major_dim = {});

void transfer_h2d_batch(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim = {},
    const std::vector<int64_t>& dst_offsets_major_dim = {},
    const std::vector<int64_t>& copy_sizes_major_dim = {});

}  // namespace torch_tpu

#endif  // TORCH_TPU_PJRT_RAW_TRANSFER_H_
