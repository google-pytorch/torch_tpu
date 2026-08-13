// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/runtime/eager_bridge.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/ScalarTypeToTypeMeta.h"
#include "c10/core/Storage.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/SmallVector.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/device_buffer.h"

namespace torch_tpu {
namespace {

void TtrDeviceBufferDeleter(void* ctx) {
  if (ctx != nullptr) {
    delete static_cast<DeviceBuffer*>(ctx);
  }
}

}  // namespace

at::Tensor CreateAtTensorFromTtrBuffer(std::unique_ptr<DeviceBuffer> buffer) {
  if (buffer == nullptr) {
    return at::Tensor();
  }

  at::ScalarType dtype = buffer->dtype();
  auto shape_span = buffer->shape();
  size_t size_in_bytes = buffer->GetOnDeviceSizeInBytes();

  c10::SmallVector<int64_t, 5> sizes(shape_span.begin(), shape_span.end());
  c10::SmallVector<int64_t, 5> strides(sizes.size());

  int64_t stride = 1;
  for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= sizes[i];
  }

  int device_id = (buffer->device() != nullptr) ? buffer->device()->id() : 0;
  DeviceBuffer* raw_buf = buffer.release();
  c10::DataPtr data_ptr(
      /*data=*/raw_buf,
      /*ctx=*/raw_buf,
      /*ctx_deleter=*/&TtrDeviceBufferDeleter,
      /*device=*/c10::Device(c10::DeviceType::XLA, device_id));

  c10::Storage storage(c10::Storage::use_byte_size_t{}, size_in_bytes,
                       std::move(data_ptr),
                       /*allocator=*/nullptr,
                       /*resizable=*/false);

  auto tensor_impl = c10::make_intrusive<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::XLA),
      c10::scalarTypeToTypeMeta(dtype));

  tensor_impl->set_sizes_and_strides(sizes, strides);
  return at::Tensor(std::move(tensor_impl));
}

absl::Status AssignTtrBufferToAtTensor(at::Tensor& tensor,
                                       std::unique_ptr<DeviceBuffer> buffer) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Defense check for defined tensor
      tensor.defined(), error::kInvalidArgument)
      << "undefined tensor passed to AssignTtrBufferToAtTensor";
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Defense check for non-null buffer
      buffer != nullptr, error::kInvalidArgument)
      << "null buffer passed to AssignTtrBufferToAtTensor";

  auto shape_span = buffer->shape();
  size_t size_in_bytes = buffer->GetOnDeviceSizeInBytes();

  c10::SmallVector<int64_t, 5> sizes(shape_span.begin(), shape_span.end());
  c10::SmallVector<int64_t, 5> strides(sizes.size());

  int64_t stride = 1;
  for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= sizes[i];
  }

  int device_id = (buffer->device() != nullptr) ? buffer->device()->id() : 0;
  DeviceBuffer* raw_buf = buffer.release();
  c10::DataPtr data_ptr(
      /*data=*/raw_buf,
      /*ctx=*/raw_buf,
      /*ctx_deleter=*/&TtrDeviceBufferDeleter,
      /*device=*/c10::Device(c10::DeviceType::XLA, device_id));

  c10::Storage storage(c10::Storage::use_byte_size_t{}, size_in_bytes,
                       std::move(data_ptr),
                       /*allocator=*/nullptr,
                       /*resizable=*/false);

  tensor.storage().set_data_ptr_noswap(std::move(storage.mutable_data_ptr()));
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);

  return absl::OkStatus();
}

DeviceBuffer* GetTtrBufferFromAtTensor(const at::Tensor& tensor) {
  if (!tensor.defined() || !tensor.has_storage()) {
    return nullptr;
  }
  return const_cast<DeviceBuffer*>(
      static_cast<const DeviceBuffer*>(tensor.storage().data()));
}

std::unique_ptr<DeviceBuffer> ExtractTtrBufferFromAtTensor(at::Tensor& tensor) {
  if (!tensor.defined() || !tensor.has_storage()) {
    return nullptr;
  }
  DeviceBuffer* raw_buf = GetTtrBufferFromAtTensor(tensor);
  if (raw_buf == nullptr) {
    return nullptr;
  }

  void* ctx = tensor.storage().mutable_data_ptr().release_context();
  tensor.storage().mutable_data_ptr().clear();
  tensor.reset();
  return std::unique_ptr<DeviceBuffer>(static_cast<DeviceBuffer*>(ctx));
}

}  // namespace torch_tpu
