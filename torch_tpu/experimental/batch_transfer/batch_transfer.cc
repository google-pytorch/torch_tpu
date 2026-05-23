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

#include "torch_tpu/experimental/batch_transfer/batch_transfer.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <ratio>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_log.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ATen/core/TensorBody.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_external.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace torch_tpu {

struct RawBufferHolder {
  const PJRT_Api* c_api;
  const PJRT_RawBuffer_Extension* extension;
  PJRT_RawBuffer* buffer;

  RawBufferHolder(const PJRT_Api* api, const PJRT_RawBuffer_Extension* ext,
                  PJRT_RawBuffer* buf)
      : c_api(api), extension(ext), buffer(buf) {}

  ~RawBufferHolder() {
    if (buffer) {
      pjrt::PjRtCApiRawBuffer_Destroy(c_api, extension, buffer);
    }
  }
};

void TransferFuture::Await() {
  // Ensure resources are always cleared, even if an exception is thrown.
  auto cleanup = absl::MakeCleanup([this] {
    futures_.clear();
    tensors_.clear();
    c_api_holds_.clear();
    holds_.clear();
  });

  // Measure the latency spent waiting for asynchronous device-to-host transfers
  // to finish.
  auto start_wait = std::chrono::high_resolution_clock::now();
  absl::Status first_error;
  for (auto& f : futures_) {
    absl::Status status = f.Await();
    if (!status.ok() && first_error.ok()) {
      first_error = status;
    }
  }
  TT_CHECK_THROW(first_error.ok(), error::kInternal)
      << "Async copy failed: " << first_error.message();
  auto end_wait = std::chrono::high_resolution_clock::now();

  // If a tag is provided, log the wait latency as raw_copy_wait_ms.
  if (!tag_.empty()) {
    double wait_ms =
        std::chrono::duration<double, std::milli>(end_wait - start_wait)
            .count();
    ABSL_LOG(INFO) << "[" << tag_ << "] raw_copy_wait_ms: " << wait_ms;
  }
}

void TransferFuture::AndThen(TransferFuture other) {
  futures_.insert(futures_.end(),
                  std::make_move_iterator(other.futures_.begin()),
                  std::make_move_iterator(other.futures_.end()));
  tensors_.insert(tensors_.end(),
                  std::make_move_iterator(other.tensors_.begin()),
                  std::make_move_iterator(other.tensors_.end()));
  c_api_holds_.insert(c_api_holds_.end(),
                      std::make_move_iterator(other.c_api_holds_.begin()),
                      std::make_move_iterator(other.c_api_holds_.end()));
  holds_.insert(holds_.end(), std::make_move_iterator(other.holds_.begin()),
                std::make_move_iterator(other.holds_.end()));
}

void TransferFuture::AndThen(std::vector<xla::Future<>> other_futures,
                             std::vector<at::Tensor> other_tensors) {
  futures_.insert(futures_.end(),
                  std::make_move_iterator(other_futures.begin()),
                  std::make_move_iterator(other_futures.end()));
  tensors_.insert(tensors_.end(),
                  std::make_move_iterator(other_tensors.begin()),
                  std::make_move_iterator(other_tensors.end()));
}

void TransferFuture::AndThen(
    std::vector<xla::Future<>> other_futures,
    std::vector<std::shared_ptr<RawBufferHolder>> other_c_api_holds,
    std::vector<std::shared_ptr<void>> other_holds) {
  futures_.insert(futures_.end(),
                  std::make_move_iterator(other_futures.begin()),
                  std::make_move_iterator(other_futures.end()));
  c_api_holds_.insert(c_api_holds_.end(),
                      std::make_move_iterator(other_c_api_holds.begin()),
                      std::make_move_iterator(other_c_api_holds.end()));
  holds_.insert(holds_.end(), std::make_move_iterator(other_holds.begin()),
                std::make_move_iterator(other_holds.end()));
}

void TransferFuture::AndThen(std::vector<xla::Future<>> other_futures,
                             std::shared_ptr<RawBufferHolder> other_c_api_hold,
                             std::shared_ptr<void> other_hold) {
  futures_.insert(futures_.end(),
                  std::make_move_iterator(other_futures.begin()),
                  std::make_move_iterator(other_futures.end()));
  if (other_c_api_hold) c_api_holds_.push_back(std::move(other_c_api_hold));
  if (other_hold) holds_.push_back(std::move(other_hold));
}

namespace {

absl::Status ValidatePartialSpec(const Dimensions& src_offsets_major_dim,
                                 const Dimensions& dst_offsets_major_dim,
                                 const Dimensions& copy_sizes_major_dim) {
  bool partial_spec_present = !src_offsets_major_dim.empty() ||
                              !dst_offsets_major_dim.empty() ||
                              !copy_sizes_major_dim.empty();
  if (partial_spec_present) {
    TT_RET_CHECK(
        src_offsets_major_dim.size() == dst_offsets_major_dim.size() &&
            dst_offsets_major_dim.size() == copy_sizes_major_dim.size(),
        error::kInvalidArgument)
        << "src_offsets_major_dim, dst_offsets_major_dim, and "
           "copy_sizes_major_dim must have the same size";
  }
  return absl::OkStatus();
}

absl::StatusOr<int64_t> GetSizePerMajorDim(const xla::Shape& shape,
                                           const xla::Layout* xla_layout,
                                           int64_t itemsize) {
  if (xla_layout && !xla_layout->tiles().empty()) {
    const xla::Tile& tile = xla_layout->tiles()[0];
    auto tile_dims = tile.dimensions();
    TT_RET_CHECK(tile_dims.size() == 2, error::kInternal)
        << "Only 2D tiling supported for now";
    int64_t tH = tile_dims[0];
    int64_t tW = tile_dims[1];

    int64_t rank = shape.dimensions().size();
    int64_t H = shape.dimensions(rank - 2);
    int64_t W = shape.dimensions(rank - 1);

    int64_t num_tiles_H = (H + tH - 1) / tH;
    int64_t num_tiles_W = (W + tW - 1) / tW;

    int64_t size_per_major_dim = num_tiles_H * num_tiles_W * tH * tW * itemsize;
    for (int i = 1; i < rank - 2; ++i) {
      size_per_major_dim *= shape.dimensions(i);
    }
    return size_per_major_dim;
  }
  return 0;
}

absl::StatusOr<DeviceBufferRef> GetPjrtBuffer(const at::Tensor& tpu_tensor,
                                              xla::PjRtBuffer*& out_buffer) {
  TT_RET_CHECK(tpu_tensor.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "Expected TPU tensor.";
  TT_ASSIGN_OR_RETURN(
      auto buffer_ref,
      GetMaterialized(tpu_tensor, MaterializationReason::kCpuTransfer));
  TT_ASSIGN_OR_RETURN(out_buffer, buffer_ref.AwaitBuffer());
  return buffer_ref;
}

}  // namespace

absl::StatusOr<TransferFuture>
BatchTransferD2H(  // NOLINT(readability-function-cognitive-complexity)
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const Dimensions& src_offsets_major_dim,
    const Dimensions& dst_offsets_major_dim,
    const Dimensions& copy_sizes_major_dim) {
  // Latency metrics for the stages.
  double materialize_ms = 0.0;
  double raw_copy_issue_ms = 0.0;

  // Step 1: Validate input sizes.
  TT_RET_CHECK(src_arrs.size() == dst_arrs.size(), error::kInvalidArgument)
      << "Lengths of src_arrs and dst_arrs must match";
  TT_RETURN_IF_ERROR(ValidatePartialSpec(
      src_offsets_major_dim, dst_offsets_major_dim, copy_sizes_major_dim));
  size_t n = src_arrs.size();
  TransferFuture acc;
  if (src_arrs.empty()) return acc;

  // Step 2: Extract metadata from the first tensor.
  // We assume all tensors in the batch have the same shape and layout.
  // Measure time spent in GetPjrtBuffer, which relates to materializing
  // PJRT buffers.
  auto start_mat = std::chrono::high_resolution_clock::now();
  xla::PjRtBuffer* first_buffer = nullptr;
  TT_ASSIGN_OR_RETURN(DeviceBufferRef first_buffer_ref,
                      GetPjrtBuffer(src_arrs[0], first_buffer));
  materialize_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - start_mat)
                        .count();

  const xla::Shape& shape = first_buffer->on_device_shape();

  // Step 3: Determine if the copy requests a partial slice of the major
  // dimension.
  bool is_partial = !src_offsets_major_dim.empty() ||
                    !dst_offsets_major_dim.empty() ||
                    !copy_sizes_major_dim.empty();

  // Step 4: Validate constraints for partial copies.
  TT_RET_CHECK(!is_partial || shape.dimensions().size() >= 3,
               error::kInvalidArgument)
      << "Only support arrays with rank >= 3 for partial copies";

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  int64_t stride = 1;
  if (shape.dimensions().size() > 1) {
    for (int i = 1; i < shape.dimensions().size(); ++i) {
      stride *= shape.dimensions(i);
    }
  }

  // Partial copy constraints: product of non-major dimensions must align with
  // TPU 4KB tile boundaries.
  TT_RET_CHECK(!is_partial || (stride * itemsize) % 4096 == 0,
               error::kInvalidArgument)
      << "Unsupported shape: product of non-major dimensions must be a "
         "multiple of tile size (4KB) on device for partial copies";

  TT_ASSIGN_OR_RETURN(int64_t physical_size,
                      first_buffer->GetOnDeviceSizeInBytes());

  // Step 5: Retrieve PJRT C API and Raw Buffer Extension.
  const PJRT_Api* c_api = nullptr;
  const PJRT_RawBuffer_Extension* extension = nullptr;
  xla::PjRtCApiBuffer* first_capi_buffer =
      dynamic_cast<xla::PjRtCApiBuffer*>(first_buffer);

  TT_RET_CHECK(first_capi_buffer != nullptr, error::kInvalidArgument)
      << "first_buffer must be a PjRtCApiBuffer";
  c_api = ABSL_DIE_IF_NULL(first_capi_buffer)->pjrt_c_api();
  TT_RET_CHECK(c_api != nullptr, error::kInternal) << "PJRT C API not found";
  xla::PjRtCApiClient* capi_client = dynamic_cast<xla::PjRtCApiClient*>(
      ABSL_DIE_IF_NULL(first_capi_buffer)->client());
  TT_RET_CHECK(capi_client != nullptr, error::kInternal)
      << "Failed to cast client to PjRtCApiClient";
  extension = ABSL_DIE_IF_NULL(capi_client)
                  ->FindExtension<PJRT_RawBuffer_Extension>(
                      PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
  TT_RET_CHECK(extension != nullptr, error::kInternal)
      << "RawBuffer extension not found in PjRtCApiClient";

  // Step 6: Handle 2D tiling if applicable to calculate bytes per major
  // dimension.
  auto pjrt_layout = first_buffer->layout();
  const xla::Layout* xla_layout = nullptr;
  if (pjrt_layout) {
    xla_layout = &pjrt_layout->xla_layout();
  }

  int64_t size_per_major_dim = 0;
  if (is_partial) {
    TT_ASSIGN_OR_RETURN(size_per_major_dim,
                        GetSizePerMajorDim(shape, xla_layout, itemsize));
  }

  std::vector<xla::Future<>> batch_futures;
  batch_futures.reserve(n);
  std::vector<std::shared_ptr<RawBufferHolder>> batch_c_api_holds;
  std::vector<std::shared_ptr<void>> batch_holds;
  batch_c_api_holds.reserve(n);

  // Step 7: Loop over all tensors in the batch and issue copies.
  for (size_t layer_idx = 0; layer_idx < n; ++layer_idx) {
    const at::Tensor& src = src_arrs[layer_idx];
    const at::Tensor& dst = dst_arrs[layer_idx];

    TT_RET_CHECK(dst.device().type() == at::DeviceType::CPU,
                 error::kInvalidArgument)
        << "Destination must be on CPU.";
    size_t dst_size = dst.nbytes();
    uint8_t* dst_data = reinterpret_cast<uint8_t*>(dst.data_ptr());

    // Step 7a: Materialize current tensor.
    auto start_mat_loop = std::chrono::high_resolution_clock::now();
    xla::PjRtBuffer* src_buffer = nullptr;
    TT_ASSIGN_OR_RETURN(DeviceBufferRef src_buffer_ref,
                        GetPjrtBuffer(src, src_buffer));
    TT_RET_CHECK(src_buffer->on_device_shape() == shape,
                 error::kInvalidArgument)
        << "Tensor at index " << layer_idx << " has shape "
        << src_buffer->on_device_shape().ToString()
        << " which is different from expected shape " << shape.ToString();
    // The raw PJRT copy API does not chain on buffer readiness, so we must wait
    // here for the producing computation to commit its result before issuing
    // the DMA.
    TT_RETURN_IF_ERROR(src_buffer->GetReadyFuture().Await());
    materialize_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start_mat_loop)
            .count();

    // Step 7b: Create raw alias of the buffer using PJRT C API.
    auto start_issue = std::chrono::high_resolution_clock::now();
    PJRT_RawBuffer* c_raw_buffer = nullptr;

    xla::PjRtCApiBuffer* capi_buffer =
        static_cast<xla::PjRtCApiBuffer*>(src_buffer);
    TT_ASSIGN_OR_RETURN(c_raw_buffer,
                        pjrt::PjRtCApiBuffer_CreateRawAliasOfBuffer(
                            c_api, extension, capi_buffer->c_buffer()));
    batch_c_api_holds.push_back(
        std::make_shared<RawBufferHolder>(c_api, extension, c_raw_buffer));

    // Step 7c: Issue DMA copy (Full or Partial).
    if (!is_partial) {
      TT_RET_CHECK(dst_size >= physical_size, error::kInvalidArgument)
          << "Destination buffer too small for raw copy";
      xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawDeviceToHost(
          c_api, extension, c_raw_buffer, dst_data, 0,
          physical_size);  // NOLINT
      batch_futures.push_back(std::move(future));
    } else {
      for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
        int64_t src_major_dim_offset = src_offsets_major_dim[j];
        int64_t dst_major_dim_offset = dst_offsets_major_dim[j];
        int64_t major_dim_size = copy_sizes_major_dim[j];

        int64_t src_offset, size_to_copy, dst_offset;
        if (xla_layout && !xla_layout->tiles().empty()) {
          src_offset = src_major_dim_offset * size_per_major_dim;
          size_to_copy = major_dim_size * size_per_major_dim;
          dst_offset = dst_major_dim_offset * size_per_major_dim;
        } else {
          src_offset = src_major_dim_offset * stride * itemsize;
          dst_offset = dst_major_dim_offset * stride * itemsize;
          size_to_copy = major_dim_size * stride * itemsize;
        }

        TT_RET_CHECK(src_offset + size_to_copy <= physical_size,
                     error::kInvalidArgument)
            << "Copy range exceeds source buffer size";
        TT_RET_CHECK(dst_offset + size_to_copy <= dst_size,
                     error::kInvalidArgument)
            << "Copy range exceeds destination buffer size";

        uint8_t* dst_ptr = dst_data + dst_offset;
        xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawDeviceToHost(
            c_api, extension, c_raw_buffer, dst_ptr, src_offset,
            size_to_copy);  // NOLINT
        batch_futures.push_back(std::move(future));
      }
    }
    raw_copy_issue_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start_issue)
            .count();
    // Keep the DeviceBufferRef alive until the transfer completes.
    batch_holds.push_back(
        std::make_shared<DeviceBufferRef>(std::move(src_buffer_ref)));
  }

  // Step 8: Aggregate futures and keep tensors alive.
  std::vector<at::Tensor> all_tensors;
  all_tensors.reserve(src_arrs.size() + dst_arrs.size());
  all_tensors.insert(all_tensors.end(), src_arrs.begin(), src_arrs.end());
  all_tensors.insert(all_tensors.end(), dst_arrs.begin(), dst_arrs.end());

  acc.AndThen(std::move(batch_futures), std::move(batch_c_api_holds),
              std::move(batch_holds));
  acc.AndThen(std::vector<xla::Future<>>{}, std::move(all_tensors));

  // Tag the TransferFuture to measure raw_copy_wait_ms later during the Await()
  // call.
  acc.SetTag("BatchTransferD2H");
  ABSL_LOG(INFO) << "[BatchTransferD2H] materialize_ms: " << materialize_ms
                 << ", raw_copy_issue_ms: " << raw_copy_issue_ms;

  return acc;
}

absl::StatusOr<TransferFuture>
BatchTransferH2D(  // NOLINT(readability-function-cognitive-complexity)
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const Dimensions& src_offsets_major_dim,
    const Dimensions& dst_offsets_major_dim,
    const Dimensions& copy_sizes_major_dim) {
  TT_RET_CHECK(src_arrs.size() == dst_arrs.size(), error::kInvalidArgument)
      << "Lengths of src_arrs and dst_arrs must match";
  TT_RETURN_IF_ERROR(ValidatePartialSpec(
      src_offsets_major_dim, dst_offsets_major_dim, copy_sizes_major_dim));
  size_t n = src_arrs.size();
  TransferFuture acc;
  if (src_arrs.empty()) return acc;

  // Extract metadata from the first tensor.
  xla::PjRtBuffer* first_buffer = nullptr;
  TT_ASSIGN_OR_RETURN(DeviceBufferRef first_buffer_ref,
                      GetPjrtBuffer(dst_arrs[0], first_buffer));

  const xla::Shape& shape = first_buffer->on_device_shape();

  bool is_partial = !src_offsets_major_dim.empty() ||
                    !dst_offsets_major_dim.empty() ||
                    !copy_sizes_major_dim.empty();

  TT_RET_CHECK(!is_partial || shape.dimensions().size() >= 3,
               error::kInvalidArgument)
      << "Only support arrays with rank >= 3 for partial copies";

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  int64_t stride = 1;
  if (shape.dimensions().size() > 1) {
    for (int i = 1; i < shape.dimensions().size(); ++i) {
      stride *= shape.dimensions(i);
    }
  }

  TT_RET_CHECK(!is_partial || (stride * itemsize) % 4096 == 0,
               error::kInvalidArgument)
      << "Unsupported shape: product of non-major dimensions must be a "
         "multiple of tile size (4KB) on device for partial copies";

  TT_ASSIGN_OR_RETURN(int64_t physical_size,
                      first_buffer->GetOnDeviceSizeInBytes());

  const PJRT_Api* c_api = nullptr;
  const PJRT_RawBuffer_Extension* extension = nullptr;
  xla::PjRtCApiBuffer* first_capi_buffer =
      dynamic_cast<xla::PjRtCApiBuffer*>(first_buffer);

  TT_RET_CHECK(first_capi_buffer != nullptr, error::kInvalidArgument)
      << "first_buffer must be a PjRtCApiBuffer";
  c_api = ABSL_DIE_IF_NULL(first_capi_buffer)->pjrt_c_api();
  TT_RET_CHECK(c_api != nullptr, error::kInternal) << "PJRT C API not found";
  xla::PjRtCApiClient* capi_client = dynamic_cast<xla::PjRtCApiClient*>(
      ABSL_DIE_IF_NULL(first_capi_buffer)->client());
  TT_RET_CHECK(capi_client != nullptr, error::kInternal)
      << "Failed to cast client to PjRtCApiClient";
  extension = ABSL_DIE_IF_NULL(capi_client)
                  ->FindExtension<PJRT_RawBuffer_Extension>(
                      PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
  TT_RET_CHECK(extension != nullptr, error::kInternal)
      << "RawBuffer extension not found in PjRtCApiClient";

  auto pjrt_layout = first_buffer->layout();
  const xla::Layout* xla_layout = nullptr;
  if (pjrt_layout) {
    xla_layout = &pjrt_layout->xla_layout();
  }

  int64_t size_per_major_dim = 0;
  if (is_partial) {
    TT_ASSIGN_OR_RETURN(size_per_major_dim,
                        GetSizePerMajorDim(shape, xla_layout, itemsize));
  }

  std::vector<xla::Future<>> batch_futures;
  batch_futures.reserve(n);
  std::vector<std::shared_ptr<RawBufferHolder>> batch_c_api_holds;
  std::vector<std::shared_ptr<void>> batch_holds;
  batch_c_api_holds.reserve(n);

  for (size_t layer_idx = 0; layer_idx < n; ++layer_idx) {
    const at::Tensor& src = src_arrs[layer_idx];
    const at::Tensor& dst = dst_arrs[layer_idx];

    TT_RET_CHECK(src.device().type() == at::DeviceType::CPU,
                 error::kInvalidArgument)
        << "Source must be on CPU";
    size_t src_size = src.nbytes();
    const uint8_t* src_data = reinterpret_cast<const uint8_t*>(src.data_ptr());

    xla::PjRtBuffer* dst_buffer = nullptr;
    TT_ASSIGN_OR_RETURN(DeviceBufferRef dst_buffer_ref,
                        GetPjrtBuffer(dst, dst_buffer));
    TT_RET_CHECK(dst_buffer->on_device_shape() == shape,
                 error::kInvalidArgument)
        << "Tensor at index " << layer_idx << " has shape "
        << dst_buffer->on_device_shape().ToString()
        << " which is different from expected shape " << shape.ToString();
    // The raw PJRT copy API does not chain on buffer readiness; the destination
    // buffer's ready future signals that it is safe to write into.
    TT_RETURN_IF_ERROR(dst_buffer->GetReadyFuture().Await());

    PJRT_RawBuffer* c_raw_buffer = nullptr;

    xla::PjRtCApiBuffer* capi_buffer =
        static_cast<xla::PjRtCApiBuffer*>(dst_buffer);
    TT_ASSIGN_OR_RETURN(c_raw_buffer,
                        pjrt::PjRtCApiBuffer_CreateRawAliasOfBuffer(
                            c_api, extension, capi_buffer->c_buffer()));
    batch_c_api_holds.push_back(
        std::make_shared<RawBufferHolder>(c_api, extension, c_raw_buffer));

    if (!is_partial) {
      TT_RET_CHECK(src_size >= physical_size, error::kInvalidArgument)
          << "Source buffer too small for raw copy";
      xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(
          c_api, extension, c_raw_buffer, src_data, 0,
          physical_size);  // NOLINT
      batch_futures.push_back(std::move(future));
    } else {
      for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
        int64_t src_major_dim_offset = src_offsets_major_dim[j];
        int64_t dst_major_dim_offset = dst_offsets_major_dim[j];
        int64_t major_dim_size = copy_sizes_major_dim[j];

        int64_t src_offset, size_to_copy, dst_offset;
        if (xla_layout && !xla_layout->tiles().empty()) {
          src_offset = src_major_dim_offset * size_per_major_dim;
          size_to_copy = major_dim_size * size_per_major_dim;
          dst_offset = dst_major_dim_offset * size_per_major_dim;
        } else {
          src_offset = src_major_dim_offset * stride * itemsize;
          dst_offset = dst_major_dim_offset * stride * itemsize;
          size_to_copy = major_dim_size * stride * itemsize;
        }

        TT_RET_CHECK(src_offset + size_to_copy <= src_size,
                     error::kInvalidArgument)
            << "Copy range exceeds source buffer size";
        TT_RET_CHECK(dst_offset + size_to_copy <= physical_size,
                     error::kInvalidArgument)
            << "Copy range exceeds destination buffer size";

        const uint8_t* src_ptr = src_data + src_offset;
        xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(
            c_api, extension, c_raw_buffer, src_ptr, dst_offset,
            size_to_copy);  // NOLINT
        batch_futures.push_back(std::move(future));
      }
    }
  }

  // Combine the tensor arrays for keeping them alive.
  std::vector<at::Tensor> all_tensors;
  all_tensors.reserve(src_arrs.size() + dst_arrs.size());
  all_tensors.insert(all_tensors.end(), src_arrs.begin(), src_arrs.end());
  all_tensors.insert(all_tensors.end(), dst_arrs.begin(), dst_arrs.end());

  acc.AndThen(std::move(batch_futures), std::move(batch_c_api_holds),
              std::move(batch_holds));
  acc.AndThen(std::vector<xla::Future<>>{}, std::move(all_tensors));

  return acc;
}

absl::Status BatchTransferD2HSync(const std::vector<at::Tensor>& src_arrs,
                                  const std::vector<at::Tensor>& dst_arrs,
                                  const Dimensions& src_offsets_major_dim,
                                  const Dimensions& dst_offsets_major_dim,
                                  const Dimensions& copy_sizes_major_dim) {
  TT_ASSIGN_OR_RETURN(
      auto res, BatchTransferD2H(src_arrs, dst_arrs, src_offsets_major_dim,
                                 dst_offsets_major_dim, copy_sizes_major_dim));
  res.Await();
  return absl::OkStatus();
}

absl::Status BatchTransferH2DSync(const std::vector<at::Tensor>& src_arrs,
                                  const std::vector<at::Tensor>& dst_arrs,
                                  const Dimensions& src_offsets_major_dim,
                                  const Dimensions& dst_offsets_major_dim,
                                  const Dimensions& copy_sizes_major_dim) {
  TT_ASSIGN_OR_RETURN(
      auto res, BatchTransferH2D(src_arrs, dst_arrs, src_offsets_major_dim,
                                 dst_offsets_major_dim, copy_sizes_major_dim));
  res.Await();
  return absl::OkStatus();
}

}  // namespace torch_tpu
