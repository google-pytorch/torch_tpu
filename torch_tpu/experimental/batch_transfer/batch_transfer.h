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

#ifndef THIRD_PARTY_PY_TORCH_TPU_EXPERIMENTAL_BATCH_TRANSFER_BATCH_TRANSFER_H_
#define THIRD_PARTY_PY_TORCH_TPU_EXPERIMENTAL_BATCH_TRANSFER_BATCH_TRANSFER_H_ /* NON_TT_MACRO_OK=header guard */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_tpu/common/dimension_types.h"
#include "xla/future.h"

namespace torch_tpu {

struct RawBufferHolder;

class TransferFuture {
 public:
  TransferFuture() = default;

  void Await();
  void AndThen(TransferFuture other);
  // Chains other futures and tensors to this TransferFuture. The tensors in
  // other_tensors will be kept alive until this TransferFuture is destroyed.
  void AndThen(std::vector<xla::Future<>> other_futures,
               std::vector<at::Tensor> other_tensors);

  void AndThen(std::vector<xla::Future<>> other_futures,
               std::vector<std::shared_ptr<RawBufferHolder>> other_c_api_holds,
               std::vector<std::shared_ptr<void>> other_holds);

  void AndThen(std::vector<xla::Future<>> other_futures,
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

// Batch transfers
absl::StatusOr<TransferFuture> BatchTransferD2H(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const Dimensions& src_offsets_major_dim = {},
    const Dimensions& dst_offsets_major_dim = {},
    const Dimensions& copy_sizes_major_dim = {});

// Asynchronous Host to Device batch transfer.
// NOTE: This API does not register the raw-copy future with TorchTPU stream
// state, and therefore does not synchronize with TPU streams. Callers must
// explicitly wait for the returned TransferFuture to complete before launching
// TPU computations that consume the destination buffer.
absl::StatusOr<TransferFuture> BatchTransferH2D(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const Dimensions& src_offsets_major_dim = {},
    const Dimensions& dst_offsets_major_dim = {},
    const Dimensions& copy_sizes_major_dim = {});

// Synchronous batch transfers
absl::Status BatchTransferD2HSync(const std::vector<at::Tensor>& src_arrs,
                                  const std::vector<at::Tensor>& dst_arrs,
                                  const Dimensions& src_offsets_major_dim = {},
                                  const Dimensions& dst_offsets_major_dim = {},
                                  const Dimensions& copy_sizes_major_dim = {});

absl::Status BatchTransferH2DSync(const std::vector<at::Tensor>& src_arrs,
                                  const std::vector<at::Tensor>& dst_arrs,
                                  const Dimensions& src_offsets_major_dim = {},
                                  const Dimensions& dst_offsets_major_dim = {},
                                  const Dimensions& copy_sizes_major_dim = {});

}  // namespace torch_tpu

#endif  // THIRD_PARTY_PY_TORCH_TPU_EXPERIMENTAL_BATCH_TRANSFER_BATCH_TRANSFER_H_
