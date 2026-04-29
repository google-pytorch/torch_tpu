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

#include "torch_tpu/pjrt/raw_transfer.h"

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_external.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
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

void PjRtCopyFuture::Await() {
  // Measure the latency spent waiting for asynchronous device-to-host transfers to finish.
  auto start_wait = std::chrono::high_resolution_clock::now();
  for (auto& f : futures_) {
    absl::Status status = f.Await();
    if (!status.ok()) {
      throw std::runtime_error("Async copy failed: " + std::string(status.message()));
    }
  }
  auto end_wait = std::chrono::high_resolution_clock::now();
  
  // If a tag is provided, log the wait latency as raw_copy_wait_ms.
  if (!tag_.empty()) {
    double wait_ms = std::chrono::duration<double, std::milli>(end_wait - start_wait).count();
    ABSL_LOG(INFO) << "[" << tag_ << "] raw_copy_wait_ms: " << wait_ms;
  }
  futures_.clear();
  tensors_.clear();
}

void PjRtCopyFuture::Append(PjRtCopyFuture other) {
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

void PjRtCopyFuture::Append(std::vector<xla::Future<>> other_futures,
                            std::vector<at::Tensor> other_tensors) {
  futures_.insert(futures_.end(),
                  std::make_move_iterator(other_futures.begin()),
                  std::make_move_iterator(other_futures.end()));
  tensors_.insert(tensors_.end(),
                  std::make_move_iterator(other_tensors.begin()),
                  std::make_move_iterator(other_tensors.end()));
}

void PjRtCopyFuture::Append(
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

void PjRtCopyFuture::Append(
    std::vector<xla::Future<>> other_futures,
    std::shared_ptr<RawBufferHolder> other_c_api_hold,
    std::shared_ptr<void> other_hold) {
  futures_.insert(futures_.end(),
                  std::make_move_iterator(other_futures.begin()),
                  std::make_move_iterator(other_futures.end()));
  if (other_c_api_hold) c_api_holds_.push_back(std::move(other_c_api_hold));
  if (other_hold) holds_.push_back(std::move(other_hold));
}

namespace {

std::optional<DeviceBufferRef> GetPjrtBufferOrThrow(const at::Tensor& tpu_tensor,
                                                    xla::PjRtBuffer*& out_buffer) {
  if (tpu_tensor.device().type() != at::DeviceType::PrivateUse1) {
    throw std::runtime_error("Expected TPU tensor.");
  }
  auto buffer_ref_or = GetMaterialized(tpu_tensor);
  if (!buffer_ref_or.ok()) {
    throw std::runtime_error("Failed to materialize TPU tensor.");
  }
  auto out_buffer_ref = std::move(buffer_ref_or.value());
  auto pjrt_buffer_or = out_buffer_ref.GetOrMaterializeBuffer();
  if (!pjrt_buffer_or.ok()) {
    throw std::runtime_error("Failed to get PjRtBuffer.");
  }
  out_buffer = pjrt_buffer_or.value();
  return out_buffer_ref;
}

void transfer_d2h_internal(const at::Tensor& src_arr, const at::Tensor& dst_arr,
                           const std::vector<int64_t>& src_offsets_major_dim,
                           const std::vector<int64_t>& dst_offsets_major_dim,
                           const std::vector<int64_t>& copy_sizes_major_dim,
                           PjRtCopyFuture& acc) {
  // 1. Ensure all offset and copy size vectors are aligned in size.
  if (src_offsets_major_dim.size() != dst_offsets_major_dim.size() ||
      src_offsets_major_dim.size() != copy_sizes_major_dim.size()) {
    throw std::runtime_error("Lengths of offset and size lists must match");
  }

  // 2. Extract the materialized xla::PjRtBuffer from the source ATen tensor.
  xla::PjRtBuffer* src_buffer = nullptr;
  std::optional<DeviceBufferRef> src_buffer_ref = GetPjrtBufferOrThrow(src_arr, src_buffer);

  const xla::Shape& shape = src_buffer->on_device_shape();

  // 3. Determine if the copy requests a partial slice of the major dimension.
  bool is_partial = false;
  int64_t full_major_dim_size = shape.dimensions().empty() ? 1 : shape.dimensions(0);
  for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
    if (src_offsets_major_dim[i] != 0 || dst_offsets_major_dim[i] != 0 ||
        copy_sizes_major_dim[i] != full_major_dim_size) {
      is_partial = true;
      break;
    }
  }

  // 4. Partial copies are only supported on rank >= 3 buffers.
  if (is_partial && shape.dimensions_size() < 3) {
    throw std::runtime_error("Only support arrays with rank >= 3 for partial copies");
  }

  // 5. Determine element byte sizes and non-major product strides.
  int64_t itemsize = xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  int64_t stride = 1;
  if (shape.dimensions_size() > 1) {
    for (int i = 1; i < shape.dimensions_size(); ++i) {
      stride *= shape.dimensions(i);
    }
  }

  int64_t non_major_product = stride;

  // 6. Partial copy constraints: product of non-major dimensions must align with TPU 4KB tile boundaries.
  if (is_partial && (non_major_product * itemsize) % 4096 != 0) {
    throw std::runtime_error(
        "Unsupported shape: product of non-major dimensions must be a "
        "multiple of tile size (4KB) on device for partial copies");
  }

  auto status_or_src_size = src_buffer->GetOnDeviceSizeInBytes();
  if (!status_or_src_size.ok()) {
    throw std::runtime_error("Failed to get source buffer size");
  }
  int64_t physical_size = status_or_src_size.value();

  // 7. Retrieve layout metrics, specifically tracking 2D tiling constraints on device.
  auto pjrt_layout = src_buffer->layout();
  const xla::Layout* xla_layout = nullptr;
  if (pjrt_layout) {
    xla_layout = &pjrt_layout->xla_layout();
  }

  // 8. Compute bytes per major dimension when 2D tiling is present on a partial copy.
  int64_t size_per_major_dim = 0;
  if (is_partial) {
    if (xla_layout && !xla_layout->tiles().empty()) {
      const xla::Tile& tile = xla_layout->tiles()[0];
      auto tile_dims = tile.dimensions();
      if (tile_dims.size() != 2) {
        throw std::runtime_error("Only 2D tiling supported for now");
      }
      int64_t tH = tile_dims[0];
      int64_t tW = tile_dims[1];

      int64_t rank = shape.dimensions_size();
      int64_t H = shape.dimensions(rank - 2);
      int64_t W = shape.dimensions(rank - 1);

      int64_t num_tiles_H = (H + tH - 1) / tH;
      int64_t num_tiles_W = (W + tW - 1) / tW;

      size_per_major_dim = num_tiles_H * num_tiles_W * tH * tW * itemsize;
      for (int i = 1; i < rank - 2; ++i) {
        size_per_major_dim *= shape.dimensions(i);
      }
    }
  }

  // 9. Resolve destination CPU host memory pointer.
  if (dst_arr.device().type() != at::DeviceType::CPU) {
    throw std::runtime_error("Destination must be on CPU.");
  }
  size_t dst_size = dst_arr.nbytes();
  uint8_t* dst_data = reinterpret_cast<uint8_t*>(dst_arr.data_ptr());

  std::vector<xla::Future<>> shard_futures;

  // 10. Execute full copy.
  if (!is_partial) {
    if (dst_size < physical_size) {
      throw std::runtime_error("Destination buffer too small for raw copy");
    }
    xla::Future<> future = src_buffer->CopyRawToHost(dst_data, 0, physical_size);
    shard_futures.push_back(std::move(future));
  } else {
    // 11. Execute partial copy shard-by-shard.
    for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
      int64_t src_major_dim_offset = src_offsets_major_dim[j];
      int64_t dst_major_dim_offset = dst_offsets_major_dim[j];
      int64_t major_dim_size = copy_sizes_major_dim[j];

      if (xla_layout && !xla_layout->tiles().empty()) {
        // Compute partial offsets for tiled 2D layout.
        int64_t physical_offset = src_major_dim_offset * size_per_major_dim;
        int64_t size_to_copy = major_dim_size * size_per_major_dim;
        int64_t dst_offset = dst_major_dim_offset * size_per_major_dim;

        if (physical_offset + size_to_copy > physical_size) {
          throw std::runtime_error("Copy range exceeds source buffer size");
        }
        if (dst_offset + size_to_copy > dst_size) {
          throw std::runtime_error("Copy range exceeds destination buffer size");
        }

        uint8_t* dst_ptr = dst_data + dst_offset;
        xla::Future<> future = src_buffer->CopyRawToHost(dst_ptr, physical_offset, size_to_copy);
        shard_futures.push_back(std::move(future));
      } else {
        // Compute partial offsets for contiguous/non-tiled major dimension.
        int64_t src_offset = src_major_dim_offset * stride * itemsize;
        int64_t dst_offset = dst_major_dim_offset * stride * itemsize;
        int64_t size = major_dim_size * stride * itemsize;

        if (src_offset + size > physical_size) {
          throw std::runtime_error("Copy range exceeds source buffer size");
        }
        if (dst_offset + size > dst_size) {
          throw std::runtime_error("Copy range exceeds destination buffer size");
        }

        uint8_t* dst_ptr = dst_data + dst_offset;
        xla::Future<> future = src_buffer->CopyRawToHost(dst_ptr, src_offset, size);
        shard_futures.push_back(std::move(future));
      }
    }
  }

  // Keep both ATen tensors alive until the DMA finishes.
  acc.Append(std::move(shard_futures), {src_arr, dst_arr});
}

void transfer_h2d_internal(const at::Tensor& src_arr, const at::Tensor& dst_arr,
                           const std::vector<int64_t>& src_offsets_major_dim,
                           const std::vector<int64_t>& dst_offsets_major_dim,
                           const std::vector<int64_t>& copy_sizes_major_dim,
                           PjRtCopyFuture& acc) {
  // 1. Ensure all offset and copy size vectors are aligned in size.
  if (src_offsets_major_dim.size() != dst_offsets_major_dim.size() ||
      src_offsets_major_dim.size() != copy_sizes_major_dim.size()) {
    throw std::runtime_error("Lengths of offset and size lists must match");
  }

  // 2. Extract the materialized xla::PjRtBuffer from the destination ATen tensor.
  xla::PjRtBuffer* dst_buffer = nullptr;
  std::optional<DeviceBufferRef> dst_buffer_ref = GetPjrtBufferOrThrow(dst_arr, dst_buffer);

  const xla::Shape& shape = dst_buffer->on_device_shape();

  // 3. Determine if the copy requests a partial slice of the major dimension.
  bool is_partial = false;
  int64_t full_major_dim_size = shape.dimensions().empty() ? 1 : shape.dimensions(0);
  for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
    if (src_offsets_major_dim[i] != 0 || dst_offsets_major_dim[i] != 0 ||
        copy_sizes_major_dim[i] != full_major_dim_size) {
      is_partial = true;
      break;
    }
  }

  // 4. Partial copies are only supported on rank >= 3 buffers.
  if (is_partial && shape.dimensions_size() < 3) {
    throw std::runtime_error("Only support arrays with rank >= 3 for partial copies");
  }

  // 5. Determine element byte sizes and non-major product strides.
  int64_t itemsize = xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  int64_t stride = 1;
  if (shape.dimensions_size() > 1) {
    for (int i = 1; i < shape.dimensions_size(); ++i) {
      stride *= shape.dimensions(i);
    }
  }

  int64_t non_major_product = stride;

  // 6. Partial copy constraints: product of non-major dimensions must align with TPU 4KB tile boundaries.
  if (is_partial && (non_major_product * itemsize) % 4096 != 0) {
    throw std::runtime_error(
        "Unsupported shape: product of non-major dimensions must be a "
        "multiple of tile size (4KB) on device for partial copies");
  }

  auto status_or_dst_size = dst_buffer->GetOnDeviceSizeInBytes();
  if (!status_or_dst_size.ok()) {
    throw std::runtime_error("Failed to get destination buffer size");
  }
  int64_t physical_size = status_or_dst_size.value();

  // 7. Retrieve layout metrics, specifically tracking 2D tiling constraints on device.
  auto pjrt_layout = dst_buffer->layout();
  const xla::Layout* xla_layout = nullptr;
  if (pjrt_layout) {
    xla_layout = &pjrt_layout->xla_layout();
  }

  // 8. Compute bytes per major dimension when 2D tiling is present on a partial copy.
  int64_t size_per_major_dim = 0;
  if (is_partial) {
    if (xla_layout && !xla_layout->tiles().empty()) {
      const xla::Tile& tile = xla_layout->tiles()[0];
      auto tile_dims = tile.dimensions();
      if (tile_dims.size() != 2) {
        throw std::runtime_error("Only 2D tiling supported for now");
      }
      int64_t tH = tile_dims[0];
      int64_t tW = tile_dims[1];

      int64_t rank = shape.dimensions_size();
      int64_t H = shape.dimensions(rank - 2);
      int64_t W = shape.dimensions(rank - 1);

      int64_t num_tiles_H = (H + tH - 1) / tH;
      int64_t num_tiles_W = (W + tW - 1) / tW;

      size_per_major_dim = num_tiles_H * num_tiles_W * tH * tW * itemsize;
      for (int i = 1; i < rank - 2; ++i) {
        size_per_major_dim *= shape.dimensions(i);
      }
    }
  }

  // 9. Resolve source CPU host memory pointer.
  if (src_arr.device().type() != at::DeviceType::CPU) {
    throw std::runtime_error("Source must be on CPU.");
  }
  size_t src_size = src_arr.nbytes();
  const uint8_t* src_data = reinterpret_cast<const uint8_t*>(src_arr.data_ptr());

  std::vector<xla::Future<>> shard_futures;

  const PJRT_Api* c_api = nullptr;
  const PJRT_RawBuffer_Extension* extension = nullptr;
  PJRT_RawBuffer* c_raw_buffer = nullptr;
  std::shared_ptr<RawBufferHolder> c_api_hold;

  // 10. Obtain C API pointers from the main PJRT client and extract raw buffer extensions.
  xla::PjRtCApiBuffer* capi_buffer = static_cast<xla::PjRtCApiBuffer*>(dst_buffer);
  c_api = capi_buffer->pjrt_c_api();
  xla::PjRtCApiClient* capi_client = dynamic_cast<xla::PjRtCApiClient*>(capi_buffer->client());
  extension = capi_client->FindExtension<PJRT_RawBuffer_Extension>(PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
  
  if (!extension) {
    throw std::runtime_error("RawBuffer extension not found in PjRtCApiClient");
  }

  // 11. Create a raw alias of the destination buffer.
  auto status_or_raw = pjrt::PjRtCApiBuffer_CreateRawAliasOfBuffer(c_api, extension, capi_buffer->c_buffer());
  if (!status_or_raw.ok()) {
    throw std::runtime_error("Failed to create raw alias of buffer");
  }
  c_raw_buffer = status_or_raw.value();
  c_api_hold = std::make_shared<RawBufferHolder>(c_api, extension, c_raw_buffer);

  // 12. Execute full host-to-device transfer.
  if (!is_partial) {
    if (src_size < physical_size) {
      throw std::runtime_error("Source buffer too small for raw copy");
    }
    xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(c_api, extension, c_raw_buffer, src_data, 0, physical_size);
    shard_futures.push_back(std::move(future));
  } else {
    // 13. Execute partial host-to-device transfer shard-by-shard.
    for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
      int64_t src_major_dim_offset = src_offsets_major_dim[j];
      int64_t dst_major_dim_offset = dst_offsets_major_dim[j];
      int64_t major_dim_size = copy_sizes_major_dim[j];

      if (xla_layout && !xla_layout->tiles().empty()) {
        // Compute partial offsets for tiled 2D layout.
        int64_t src_offset = src_major_dim_offset * size_per_major_dim;
        int64_t size_to_copy = major_dim_size * size_per_major_dim;
        int64_t dst_physical_offset = dst_major_dim_offset * size_per_major_dim;

        if (src_offset + size_to_copy > src_size) {
          throw std::runtime_error("Copy range exceeds source buffer size");
        }
        if (dst_physical_offset + size_to_copy > physical_size) {
          throw std::runtime_error("Copy range exceeds destination buffer size");
        }

        const uint8_t* src_ptr = src_data + src_offset;
        xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(c_api, extension, c_raw_buffer, src_ptr, dst_physical_offset, size_to_copy);
        shard_futures.push_back(std::move(future));
      } else {
        // Compute partial offsets for contiguous/non-tiled major dimension.
        int64_t src_offset = src_major_dim_offset * stride * itemsize;
        int64_t dst_offset = dst_major_dim_offset * stride * itemsize;
        int64_t size = major_dim_size * stride * itemsize;

        if (src_offset + size > src_size) {
          throw std::runtime_error("Copy range exceeds source buffer size");
        }
        if (dst_offset + size > physical_size) {
          throw std::runtime_error("Copy range exceeds destination buffer size");
        }

        const uint8_t* src_ptr = src_data + src_offset;
        xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(c_api, extension, c_raw_buffer, src_ptr, dst_offset, size);
        shard_futures.push_back(std::move(future));
      }
    }
  }

  // Keep both ATen tensors alive until the DMA finishes.
  acc.Append(std::move(shard_futures), c_api_hold, nullptr);
  acc.Append(std::vector<xla::Future<>>{}, {src_arr, dst_arr});
}

}  // namespace

PjRtCopyFuture transfer_d2h_async(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  PjRtCopyFuture acc;
  transfer_d2h_internal(src_arr, dst_arr, src_offsets_major_dim,
                        dst_offsets_major_dim, copy_sizes_major_dim, acc);
  return acc;
}

PjRtCopyFuture transfer_h2d_async(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  PjRtCopyFuture acc;
  transfer_h2d_internal(src_arr, dst_arr, src_offsets_major_dim,
                        dst_offsets_major_dim, copy_sizes_major_dim, acc);
  return acc;
}

PjRtCopyFuture transfer_d2h_batch_async(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  // Latency metrics for the stages.
  double materialize_ms = 0.0;
  double raw_copy_issue_ms = 0.0;

  if (src_arrs.size() != dst_arrs.size()) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = src_arrs.size();
  PjRtCopyFuture acc;
  if (n == 0) return acc;

  // Extract metadata from the first tensor.
  // Measure time spent in GetPjrtBufferOrThrow, which relates to materializing PJRT buffers.
  auto start_mat = std::chrono::high_resolution_clock::now();
  xla::PjRtBuffer* first_buffer = nullptr;
  std::optional<DeviceBufferRef> first_buffer_ref = GetPjrtBufferOrThrow(src_arrs[0], first_buffer);
  materialize_ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_mat).count();

  const xla::Shape& shape = first_buffer->on_device_shape();

  bool is_partial = false;
  int64_t full_major_dim_size = shape.dimensions().empty() ? 1 : shape.dimensions(0);
  for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
    if (src_offsets_major_dim[i] != 0 || dst_offsets_major_dim[i] != 0 ||
        copy_sizes_major_dim[i] != full_major_dim_size) {
      is_partial = true;
      break;
    }
  }

  if (is_partial && shape.dimensions_size() < 3) {
    throw std::runtime_error("Only support arrays with rank >= 3 for partial copies");
  }

  int64_t itemsize = xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  int64_t stride = 1;
  if (shape.dimensions_size() > 1) {
    for (int i = 1; i < shape.dimensions_size(); ++i) {
      stride *= shape.dimensions(i);
    }
  }

  if (is_partial && (stride * itemsize) % 4096 != 0) {
    throw std::runtime_error(
        "Unsupported shape: product of non-major dimensions must be a "
        "multiple of tile size (4KB) on device for partial copies");
  }

  auto status_or_src_size = first_buffer->GetOnDeviceSizeInBytes();
  if (!status_or_src_size.ok()) {
    throw std::runtime_error("Failed to get source buffer size");
  }
  int64_t physical_size = status_or_src_size.value();

  const PJRT_Api* c_api = nullptr;
  const PJRT_RawBuffer_Extension* extension = nullptr;
  xla::PjRtCApiBuffer* first_capi_buffer = dynamic_cast<xla::PjRtCApiBuffer*>(first_buffer);

  if (first_capi_buffer) {
    c_api = first_capi_buffer->pjrt_c_api();
    xla::PjRtCApiClient* capi_client = dynamic_cast<xla::PjRtCApiClient*>(first_capi_buffer->client());
    extension = capi_client->FindExtension<PJRT_RawBuffer_Extension>(
        PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
    if (!extension) {
      throw std::runtime_error("RawBuffer extension not found in PjRtCApiClient");
    }
  }

  auto pjrt_layout = first_buffer->layout();
  const xla::Layout* xla_layout = nullptr;
  if (pjrt_layout) {
    xla_layout = &pjrt_layout->xla_layout();
  }

  int64_t size_per_major_dim = 0;
  if (is_partial) {
    if (xla_layout && !xla_layout->tiles().empty()) {
      const xla::Tile& tile = xla_layout->tiles()[0];
      auto tile_dims = tile.dimensions();
      if (tile_dims.size() != 2) {
        throw std::runtime_error("Only 2D tiling supported for now");
      }
      int64_t tH = tile_dims[0];
      int64_t tW = tile_dims[1];

      int64_t rank = shape.dimensions_size();
      int64_t H = shape.dimensions(rank - 2);
      int64_t W = shape.dimensions(rank - 1);

      int64_t num_tiles_H = (H + tH - 1) / tH;
      int64_t num_tiles_W = (W + tW - 1) / tW;

      size_per_major_dim = num_tiles_H * num_tiles_W * tH * tW * itemsize;
      for (int i = 1; i < rank - 2; ++i) {
        size_per_major_dim *= shape.dimensions(i);
      }
    }
  }

  std::vector<xla::Future<>> batch_futures;
  batch_futures.reserve(n);
  std::vector<std::shared_ptr<RawBufferHolder>> batch_c_api_holds;
  std::vector<std::shared_ptr<void>> batch_holds;
  batch_c_api_holds.reserve(n);

  for (size_t layer_idx = 0; layer_idx < n; ++layer_idx) {
    const at::Tensor& src = src_arrs[layer_idx];
    const at::Tensor& dst = dst_arrs[layer_idx];

    if (dst.device().type() != at::DeviceType::CPU) {
      throw std::runtime_error("Destination must be on CPU.");
    }
    size_t dst_size = dst.nbytes();
    uint8_t* dst_data = reinterpret_cast<uint8_t*>(dst.data_ptr());

    // Measure materialization time for the current layer's source tensor.
    auto start_mat_loop = std::chrono::high_resolution_clock::now();
    xla::PjRtBuffer* src_buffer = nullptr;
    std::optional<DeviceBufferRef> src_buffer_ref = GetPjrtBufferOrThrow(src, src_buffer);
    materialize_ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_mat_loop).count();

    // Measure raw copy issuing time (C API raw alias creation & starting host-to-device DMA/transfer commands).
    auto start_issue = std::chrono::high_resolution_clock::now();
    PJRT_RawBuffer* c_raw_buffer = nullptr;

    xla::PjRtCApiBuffer* capi_buffer = static_cast<xla::PjRtCApiBuffer*>(src_buffer);
    auto status_or_raw = pjrt::PjRtCApiBuffer_CreateRawAliasOfBuffer(c_api, extension, capi_buffer->c_buffer());
    if (!status_or_raw.ok()) {
      throw std::runtime_error("Failed to create raw alias of buffer");
    }
    c_raw_buffer = status_or_raw.value();
    batch_c_api_holds.push_back(std::make_shared<RawBufferHolder>(c_api, extension, c_raw_buffer));

    if (!is_partial) {
      if (dst_size < physical_size) {
        throw std::runtime_error("Destination buffer too small for raw copy");
      }
      xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawDeviceToHost(c_api, extension, c_raw_buffer, dst_data, 0, physical_size);
      batch_futures.push_back(std::move(future));
    } else {
      for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
        int64_t src_major_dim_offset = src_offsets_major_dim[j];
        int64_t dst_major_dim_offset = dst_offsets_major_dim[j];
        int64_t major_dim_size = copy_sizes_major_dim[j];

        if (xla_layout && !xla_layout->tiles().empty()) {
          int64_t physical_offset = src_major_dim_offset * size_per_major_dim;
          int64_t size_to_copy = major_dim_size * size_per_major_dim;
          int64_t dst_offset = dst_major_dim_offset * size_per_major_dim;

          if (physical_offset + size_to_copy > physical_size) {
            throw std::runtime_error("Copy range exceeds source buffer size");
          }
          if (dst_offset + size_to_copy > dst_size) {
            throw std::runtime_error("Copy range exceeds destination buffer size");
          }

          uint8_t* dst_ptr = dst_data + dst_offset;
          xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawDeviceToHost(c_api, extension, c_raw_buffer, dst_ptr, physical_offset, size_to_copy);
          batch_futures.push_back(std::move(future));
        } else {
          int64_t src_offset = src_major_dim_offset * stride * itemsize;
          int64_t dst_offset = dst_major_dim_offset * stride * itemsize;
          int64_t size = major_dim_size * stride * itemsize;

          if (src_offset + size > physical_size) {
            throw std::runtime_error("Copy range exceeds source buffer size");
          }
          if (dst_offset + size > dst_size) {
            throw std::runtime_error("Copy range exceeds destination buffer size");
          }

          uint8_t* dst_ptr = dst_data + dst_offset;
          xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawDeviceToHost(c_api, extension, c_raw_buffer, dst_ptr, src_offset, size);
          batch_futures.push_back(std::move(future));
        }
      }
    }
    raw_copy_issue_ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_issue).count();
  }

  // Combine the tensor arrays for keeping them alive.
  std::vector<at::Tensor> all_tensors;
  all_tensors.reserve(src_arrs.size() + dst_arrs.size());
  all_tensors.insert(all_tensors.end(), src_arrs.begin(), src_arrs.end());
  all_tensors.insert(all_tensors.end(), dst_arrs.begin(), dst_arrs.end());

  acc.Append(std::move(batch_futures), std::move(batch_c_api_holds), std::move(batch_holds));
  acc.Append(std::vector<xla::Future<>>{}, std::move(all_tensors));

  // Tag the PjRtCopyFuture to measure raw_copy_wait_ms later during the Await() call.
  acc.SetTag("transfer_d2h_batch_async");
  ABSL_LOG(INFO) << "[transfer_d2h_batch_async] materialize_ms: " << materialize_ms
                 << ", raw_copy_issue_ms: " << raw_copy_issue_ms;

  return acc;
}

PjRtCopyFuture transfer_h2d_batch_async(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  if (src_arrs.size() != dst_arrs.size()) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = src_arrs.size();
  PjRtCopyFuture acc;
  if (n == 0) return acc;

  // Extract metadata from the first tensor.
  xla::PjRtBuffer* first_buffer = nullptr;
  std::optional<DeviceBufferRef> first_buffer_ref = GetPjrtBufferOrThrow(dst_arrs[0], first_buffer);

  const xla::Shape& shape = first_buffer->on_device_shape();

  bool is_partial = false;
  int64_t full_major_dim_size = shape.dimensions().empty() ? 1 : shape.dimensions(0);
  for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
    if (src_offsets_major_dim[i] != 0 || dst_offsets_major_dim[i] != 0 ||
        copy_sizes_major_dim[i] != full_major_dim_size) {
      is_partial = true;
      break;
    }
  }

  if (is_partial && shape.dimensions_size() < 3) {
    throw std::runtime_error("Only support arrays with rank >= 3 for partial copies");
  }

  int64_t itemsize = xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  int64_t stride = 1;
  if (shape.dimensions_size() > 1) {
    for (int i = 1; i < shape.dimensions_size(); ++i) {
      stride *= shape.dimensions(i);
    }
  }

  if (is_partial && (stride * itemsize) % 4096 != 0) {
    throw std::runtime_error(
        "Unsupported shape: product of non-major dimensions must be a "
        "multiple of tile size (4KB) on device for partial copies");
  }

  auto status_or_dst_size = first_buffer->GetOnDeviceSizeInBytes();
  if (!status_or_dst_size.ok()) {
    throw std::runtime_error("Failed to get destination buffer size");
  }
  int64_t physical_size = status_or_dst_size.value();

  const PJRT_Api* c_api = nullptr;
  const PJRT_RawBuffer_Extension* extension = nullptr;
  xla::PjRtCApiBuffer* first_capi_buffer = dynamic_cast<xla::PjRtCApiBuffer*>(first_buffer);

  if (first_capi_buffer) {
    c_api = first_capi_buffer->pjrt_c_api();
    xla::PjRtCApiClient* capi_client = dynamic_cast<xla::PjRtCApiClient*>(first_capi_buffer->client());
    extension = capi_client->FindExtension<PJRT_RawBuffer_Extension>(
        PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
    if (!extension) {
      throw std::runtime_error("RawBuffer extension not found in PjRtCApiClient");
    }
  }

  auto pjrt_layout = first_buffer->layout();
  const xla::Layout* xla_layout = nullptr;
  if (pjrt_layout) {
    xla_layout = &pjrt_layout->xla_layout();
  }

  int64_t size_per_major_dim = 0;
  if (is_partial) {
    if (xla_layout && !xla_layout->tiles().empty()) {
      const xla::Tile& tile = xla_layout->tiles()[0];
      auto tile_dims = tile.dimensions();
      if (tile_dims.size() != 2) {
        throw std::runtime_error("Only 2D tiling supported for now");
      }
      int64_t tH = tile_dims[0];
      int64_t tW = tile_dims[1];

      int64_t rank = shape.dimensions_size();
      int64_t H = shape.dimensions(rank - 2);
      int64_t W = shape.dimensions(rank - 1);

      int64_t num_tiles_H = (H + tH - 1) / tH;
      int64_t num_tiles_W = (W + tW - 1) / tW;

      size_per_major_dim = num_tiles_H * num_tiles_W * tH * tW * itemsize;
      for (int i = 1; i < rank - 2; ++i) {
        size_per_major_dim *= shape.dimensions(i);
      }
    }
  }

  std::vector<xla::Future<>> batch_futures;
  batch_futures.reserve(n);
  std::vector<std::shared_ptr<RawBufferHolder>> batch_c_api_holds;
  std::vector<std::shared_ptr<void>> batch_holds;
  batch_c_api_holds.reserve(n);

  for (size_t layer_idx = 0; layer_idx < n; ++layer_idx) {
    const at::Tensor& src = src_arrs[layer_idx];
    const at::Tensor& dst = dst_arrs[layer_idx];

    if (src.device().type() != at::DeviceType::CPU) {
      throw std::runtime_error("Source must be on CPU.");
    }
    size_t src_size = src.nbytes();
    const uint8_t* src_data = reinterpret_cast<const uint8_t*>(src.data_ptr());

    xla::PjRtBuffer* dst_buffer = nullptr;
    std::optional<DeviceBufferRef> dst_buffer_ref = GetPjrtBufferOrThrow(dst, dst_buffer);

    PJRT_RawBuffer* c_raw_buffer = nullptr;

    xla::PjRtCApiBuffer* capi_buffer = static_cast<xla::PjRtCApiBuffer*>(dst_buffer);
    auto status_or_raw = pjrt::PjRtCApiBuffer_CreateRawAliasOfBuffer(c_api, extension, capi_buffer->c_buffer());
    if (!status_or_raw.ok()) {
      throw std::runtime_error("Failed to create raw alias of buffer");
    }
    c_raw_buffer = status_or_raw.value();
    batch_c_api_holds.push_back(std::make_shared<RawBufferHolder>(c_api, extension, c_raw_buffer));

    if (!is_partial) {
      if (src_size < physical_size) {
        throw std::runtime_error("Source buffer too small for raw copy");
      }
      xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(c_api, extension, c_raw_buffer, src_data, 0, physical_size);
      batch_futures.push_back(std::move(future));
    } else {
      for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
        int64_t src_major_dim_offset = src_offsets_major_dim[j];
        int64_t dst_major_dim_offset = dst_offsets_major_dim[j];
        int64_t major_dim_size = copy_sizes_major_dim[j];

        if (xla_layout && !xla_layout->tiles().empty()) {
          int64_t src_offset = src_major_dim_offset * size_per_major_dim;
          int64_t size_to_copy = major_dim_size * size_per_major_dim;
          int64_t dst_physical_offset = dst_major_dim_offset * size_per_major_dim;

          if (src_offset + size_to_copy > src_size) {
            throw std::runtime_error("Copy range exceeds source buffer size");
          }
          if (dst_physical_offset + size_to_copy > physical_size) {
            throw std::runtime_error("Copy range exceeds destination buffer size");
          }

          const uint8_t* src_ptr = src_data + src_offset;
          xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(c_api, extension, c_raw_buffer, src_ptr, dst_physical_offset, size_to_copy);
          batch_futures.push_back(std::move(future));
        } else {
          int64_t src_offset = src_major_dim_offset * stride * itemsize;
          int64_t dst_offset = dst_major_dim_offset * stride * itemsize;
          int64_t size = major_dim_size * stride * itemsize;

          if (src_offset + size > src_size) {
            throw std::runtime_error("Copy range exceeds source buffer size");
          }
          if (dst_offset + size > physical_size) {
            throw std::runtime_error("Copy range exceeds destination buffer size");
          }

          const uint8_t* src_ptr = src_data + src_offset;
          xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(c_api, extension, c_raw_buffer, src_ptr, dst_offset, size);
          batch_futures.push_back(std::move(future));
        }
      }
    }
  }

  // Combine the tensor arrays for keeping them alive.
  std::vector<at::Tensor> all_tensors;
  all_tensors.reserve(src_arrs.size() + dst_arrs.size());
  all_tensors.insert(all_tensors.end(), src_arrs.begin(), src_arrs.end());
  all_tensors.insert(all_tensors.end(), dst_arrs.begin(), dst_arrs.end());

  acc.Append(std::move(batch_futures), std::move(batch_c_api_holds), std::move(batch_holds));
  acc.Append(std::vector<xla::Future<>>{}, std::move(all_tensors));

  return acc;
}

void transfer_d2h(const at::Tensor& src_arr, const at::Tensor& dst_arr,
                  const std::vector<int64_t>& src_offsets_major_dim,
                  const std::vector<int64_t>& dst_offsets_major_dim,
                  const std::vector<int64_t>& copy_sizes_major_dim) {
  transfer_d2h_async(src_arr, dst_arr, src_offsets_major_dim,
                     dst_offsets_major_dim, copy_sizes_major_dim).Await();
}

void transfer_h2d(const at::Tensor& src_arr, const at::Tensor& dst_arr,
                  const std::vector<int64_t>& src_offsets_major_dim,
                  const std::vector<int64_t>& dst_offsets_major_dim,
                  const std::vector<int64_t>& copy_sizes_major_dim) {
  transfer_h2d_async(src_arr, dst_arr, src_offsets_major_dim,
                     dst_offsets_major_dim, copy_sizes_major_dim).Await();
}

void transfer_d2h_batch(const std::vector<at::Tensor>& src_arrs,
                        const std::vector<at::Tensor>& dst_arrs,
                        const std::vector<int64_t>& src_offsets_major_dim,
                        const std::vector<int64_t>& dst_offsets_major_dim,
                        const std::vector<int64_t>& copy_sizes_major_dim) {
  transfer_d2h_batch_async(src_arrs, dst_arrs, src_offsets_major_dim,
                           dst_offsets_major_dim, copy_sizes_major_dim).Await();
}

void transfer_h2d_batch(const std::vector<at::Tensor>& src_arrs,
                        const std::vector<at::Tensor>& dst_arrs,
                        const std::vector<int64_t>& src_offsets_major_dim,
                        const std::vector<int64_t>& dst_offsets_major_dim,
                        const std::vector<int64_t>& copy_sizes_major_dim) {
  transfer_h2d_batch_async(src_arrs, dst_arrs, src_offsets_major_dim,
                           dst_offsets_major_dim, copy_sizes_major_dim).Await();
}

}  // namespace torch_tpu
