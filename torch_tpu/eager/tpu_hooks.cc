/*
 * Copyright 2025 Google LLC
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

#include "torch_tpu/eager/tpu_hooks.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/detail/PrivateUse1HooksInterface.h"
#include "ATen/ops/empty.h"
#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/core/ScalarType.h"
#include "c10/core/Storage.h"
#include "c10/core/Stream.h"
#include "c10/core/TensorOptions.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "c10/macros/Export.h"
#include "c10/util/Exception.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch/headeronly/macros/Export.h"
#include "torch_tpu/_internal/sync/sync.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_gen_impl.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/pjrt/pjrt_state.h"

namespace torch_tpu {
namespace {

constexpr int kMaxDevices = 8;

class TpuDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
 public:
  TpuDeviceGuardImpl() = default;
  explicit TpuDeviceGuardImpl(c10::DeviceType t);

  c10::DeviceType type() const override;
  c10::Device exchangeDevice(c10::Device d) const override;
  c10::Device getDevice() const override;
  void setDevice(c10::Device d) const override;
  void uncheckedSetDevice(c10::Device d) const noexcept override;
  c10::Stream getStream(c10::Device d) const override;
  c10::Stream getDefaultStream(c10::Device d) const override;
  c10::Stream exchangeStream(c10::Stream s) const override;
  void destroyEvent(void* event,
                    c10::DeviceIndex device_index) const noexcept override;
  void record(void** event, const c10::Stream& stream,
              c10::DeviceIndex device_index,
              c10::EventFlag flag) const override;
  void block(void* event, const c10::Stream& stream) const override;
  bool queryEvent(void* event) const override;
  c10::DeviceIndex deviceCount() const noexcept override;
  bool queryStream(const c10::Stream& stream) const override;
  void synchronizeStream(const c10::Stream& stream) const override;
  void synchronizeDevice(c10::DeviceIndex device_index) const override;
  void synchronizeEvent(void* event) const override;

 private:
  // TODO: make this per-python-thread.
  // clang-format off
  // NOLINTNEXTLINE(whitespace/line_length)
  static thread_local c10::DeviceIndex current_device_index_;  // CPP_THREAD_LOCAL_OK=State is per-C++-thread for device guard
  // NOLINTNEXTLINE(whitespace/line_length)
  static thread_local std::array<c10::StreamId, kMaxDevices> current_stream_ids_;  // CPP_THREAD_LOCAL_OK=State is per-C++-thread for device guard
  // clang-format on
};

}  // namespace

// TODO: make this per-python-thread.
// clang-format off
// NOLINTNEXTLINE(whitespace/line_length)
thread_local c10::DeviceIndex TpuDeviceGuardImpl::current_device_index_ = 0;  // CPP_THREAD_LOCAL_OK=State is per-C++-thread for device guard
// NOLINTNEXTLINE(whitespace/line_length)
thread_local std::array<c10::StreamId, kMaxDevices> TpuDeviceGuardImpl::current_stream_ids_ = {0};  // CPP_THREAD_LOCAL_OK=State is per-C++-thread for device guard
// clang-format on

c10::DeviceType TpuDeviceGuardImpl::type() const {
  return GetPrivateUse1DeviceType();
}

c10::Device TpuDeviceGuardImpl::exchangeDevice(c10::Device d) const {
  TT_CHECK_THROW(d.type() == type(), error::kInvalidArgument)
      << "TpuDeviceGuardImpl: invalid device type " << d.type();
  c10::Device old_device(type(), current_device_index_);
  auto* pjrt_client = PjrtBackend::GetInstance().GetClient();
  if (pjrt_client != nullptr) {
    const int addressable_device_count =
        pjrt_client->addressable_device_count();
    TT_CHECK_THROW(d.index() >= 0 && d.index() < static_cast<c10::DeviceIndex>(
                                                     addressable_device_count),
                   error::kInvalidArgument)
        << "Device index " << static_cast<int>(d.index())
        << " out of bounds. Available: " << addressable_device_count;
  } else if (d.index() != 0) {
    TORCH_WARN("PJRT client not available, but setting TPU device index to ",
               static_cast<int>(d.index()));
  }
  current_device_index_ = d.index();
  return old_device;
}

c10::Device TpuDeviceGuardImpl::getDevice() const {
  return c10::Device(type(), current_device_index_);
}

void TpuDeviceGuardImpl::setDevice(c10::Device d) const {
  TT_CHECK_THROW(d.type() == type(), error::kInvalidArgument)
      << "TpuDeviceGuardImpl: invalid device type " << d.type();
  auto* pjrt_client = PjrtBackend::GetInstance().GetClient();
  if (pjrt_client != nullptr) {
    const int addressable_device_count =
        pjrt_client->addressable_device_count();
    TT_CHECK_THROW(d.index() >= 0 && d.index() < static_cast<c10::DeviceIndex>(
                                                     addressable_device_count),
                   error::kInvalidArgument)
        << "Device index " << static_cast<int>(d.index())
        << " out of bounds. Available: " << addressable_device_count;
  } else if (d.index() != 0) {
    TORCH_WARN("PJRT client not available, but setting TPU device index to ",
               static_cast<int>(d.index()));
  }
  current_device_index_ = d.index();
}

void TpuDeviceGuardImpl::uncheckedSetDevice(c10::Device d) const noexcept {
  if (d.type() == type()) {
    current_device_index_ = d.index();
  }
}

c10::Stream TpuDeviceGuardImpl::getStream(c10::Device d) const {
  TT_CHECK_THROW(d.type() == type(), error::kInvalidArgument)
      << "TpuDeviceGuardImpl: invalid device type for getStream " << d.type();
  const c10::DeviceIndex index = d.index();
  TT_CHECK_THROW(index >= 0 && index < kMaxDevices, error::kInvalidArgument)
      << "device index out of bounds: " << index;
  const c10::StreamId stream_id = current_stream_ids_[index];
  return c10::Stream(c10::Stream::UNSAFE, d, stream_id);
}

c10::Stream TpuDeviceGuardImpl::exchangeStream(c10::Stream s) const {
  TT_CHECK_THROW(s.device_type() == type(), error::kInvalidArgument)
      << "TpuDeviceGuardImpl: invalid stream device type for exchangeStream "
      << s.device_type();
  c10::DeviceIndex device_index = s.device_index();
  TT_CHECK_THROW(device_index >= 0 && device_index < kMaxDevices,
                 error::kInvalidArgument)
      << "device index out of bounds: " << device_index;
  c10::StreamId old_stream_id = current_stream_ids_[device_index];
  current_stream_ids_[device_index] = s.id();
  return c10::Stream(c10::Stream::UNSAFE, s.device(), old_stream_id);
}

c10::DeviceIndex TpuDeviceGuardImpl::deviceCount() const noexcept {
  auto* pjrt_client = PjrtBackend::GetInstance().GetClient();
  if (pjrt_client != nullptr) {
    return static_cast<c10::DeviceIndex>(
        pjrt_client->addressable_device_count());
  }
  return 0;
}

c10::Stream TpuDeviceGuardImpl::getDefaultStream(c10::Device d) const {
  TT_CHECK_THROW(d.type() == type(), error::kInvalidArgument)
      << "TpuDeviceGuardImpl: invalid device type for getDefaultStream "
      << d.type();
  return c10::Stream(c10::Stream::DEFAULT, d);
}

void TpuDeviceGuardImpl::record(void** event, const c10::Stream& stream,
                                const c10::DeviceIndex device_index,
                                const c10::EventFlag flag) const {
  TORCH_WARN_ONCE("events (record) not yet implemented.");
}
void TpuDeviceGuardImpl::block(void* event, const c10::Stream& stream) const {
  TORCH_WARN_ONCE("events (block) not yet implemented.");
}
bool TpuDeviceGuardImpl::queryEvent(void* event) const {
  TORCH_WARN_ONCE("events (queryEvent) not yet implemented.");
  return true;
}
bool TpuDeviceGuardImpl::queryStream(const c10::Stream& stream) const {
  TORCH_WARN_ONCE("queryStream not yet implemented.");
  return true;
}
void TpuDeviceGuardImpl::synchronizeStream(const c10::Stream& stream) const {
  PjrtBackend::GetInstance().SynchronizeStream(stream.device_index(),
                                               stream.id());
}
void TpuDeviceGuardImpl::synchronizeDevice(
    c10::DeviceIndex device_index) const {
  TT_THROW_IF_ERROR(SynchronizeAll(WaitOnExecution::kYes));
  PjrtBackend::GetInstance().SynchronizeDevice(device_index);
}
void TpuDeviceGuardImpl::destroyEvent(
    void* event, const c10::DeviceIndex device_index) const noexcept {
  TORCH_WARN_ONCE("destroyEvent not yet implemented.");
}

void TpuDeviceGuardImpl::synchronizeEvent(void* event) const {
  TORCH_WARN_ONCE("TpuDeviceGuardImpl::synchronizeEvent not implemented.");
}

C10_REGISTER_GUARD_IMPL(PrivateUse1, TpuDeviceGuardImpl);

struct TORCH_API TpuHooksInterface : public at::PrivateUse1HooksInterface {
  ~TpuHooksInterface() override = default;

  const at::Generator& getDefaultGenerator(
      c10::DeviceIndex device_index) const override {
    return GetDefaultDeviceGenerator(device_index);
  }

  void init() const override {}

  at::Generator getNewGenerator(c10::DeviceIndex device_index) const override {
    return MakeDeviceGenerator(device_index);
  }

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    return PjrtBackend::GetInstance().GetClient() != nullptr;
  }

  bool isPinnedPtr(const void* data) const override {
    return IsTpuPinnedPtr(data);
  }

  at::Allocator* getPinnedMemoryAllocator() const override {
    return GetTpuPinnedAllocator();
  }

  // TODO: b/449801230 - once TPU is upstreamed, invoke this logic from
  //  resize_bytes_nocuda() in Resize.cpp.
  void resizePrivateUse1Bytes(const c10::Storage& storage,
                              size_t new_bytes) const override {
    TT_KERNEL(
        OpName::kUntypedStorageResize_, _,
        (IgnoreInCacheKey(storage, "Legacy usage"),
         IgnoreInCacheKey(new_bytes, "Legacy usage")),
        {
          const size_t current_bytes = storage.nbytes();
          TT_ASSIGN_OR_THROW(const DeviceBufferRef old_buffer_ref,
                             GetBaseBuffer(storage));
          const mlir::ElementType element_type = old_buffer_ref.element_type();
          const at::ScalarType dtype = ConvertTo<at::ScalarType>(element_type);
          const int64_t itemsize = static_cast<int64_t>(at::elementSize(dtype));
          // Calculates the new number of elements, rounding down if new_bytes
          // is not a multiple of itemsize. Attempting to access this partial
          // item would throw an error so there is no need to save the partial
          // bytes.
          const int64_t new_numel = static_cast<int64_t>(new_bytes) / itemsize;

          if (new_bytes == 0 || current_bytes == 0) {
            // Resizes to or from 0 bytes. Creates an empty buffer with the new
            // number of elements and assigns the storage to it.
            TT_ASSIGN_OR_THROW(
                DeviceBufferRef buffer_ref,
                DeviceBufferList::CreateEmpty({new_numel}, element_type));
            c10::DataPtr data_ptr =
                MakeDataPtr(std::move(buffer_ref), storage.device().index());
            storage.set_data_ptr(std::move(data_ptr));
            storage.set_nbytes(new_bytes);
            return;
          }
          if (new_bytes > current_bytes) {
            // Resizes from non-zero to a larger number of bytes. Creates a
            // dummy tensor that shares the storage and uses aten::resize_,
            // which inserts the pad operation.
            const auto options =
                at::TensorOptions().dtype(dtype).device(storage.device());
            at::Tensor dummy_tensor = at::empty({0}, options).set_(storage);
            dummy_tensor.resize_({new_numel});
            // After resizing the storage has (new_numel * itemsize) bytes,
            // which is not necessarily equal to new_bytes.
            storage.set_nbytes(new_bytes);
            return;
          }
          // Resizes to a smaller yet non-zero number of bytes. Creates an empty
          // tensor of the new size and writes the relevant slice of the
          // original data into it (accessed via a dummy tensor that shares the
          // storage). This has the effect of erasing the rest of the original
          // data, whereas a simple slice would preserve all of the data in the
          // buffer.
          const auto options =
              at::TensorOptions().dtype(dtype).device(storage.device());
          TT_ASSIGN_OR_THROW(
              DeviceBufferRef new_buffer_ref,
              DeviceBufferList::CreateEmpty({new_numel}, element_type));
          at::Tensor new_tensor = MakeTensor(new_buffer_ref);
          at::Tensor dummy_tensor = at::empty({0}, options).set_(storage);
          new_tensor.copy_(
              dummy_tensor.slice(/*dim=*/0, /*start=*/0, /*end=*/new_numel));
          // Transfers the new tensor's storage pointer to the original storage.
          storage.set_data_ptr(new_tensor.storage().set_data_ptr({}));
          storage.set_nbytes(new_bytes);
        });
  }

  bool isAvailable() const override {
    return PjrtBackend::GetInstance().GetClient() != nullptr;
  }
};

struct TORCH_API TpuHooksArgs : public at::PrivateUse1HooksArgs {};

namespace {

// register to PrivateUse1HooksInterface
at::PrivateUse1HooksInterface* GetTpuHooks() {
  static absl::NoDestructor<TpuHooksInterface> tpu_hooks;
  return tpu_hooks.get();
}

}  // namespace

absl::Status AddTpuHooks() {
  at::RegisterPrivateUse1HooksInterface(GetTpuHooks());

  return absl::OkStatus();
}

}  // namespace torch_tpu
