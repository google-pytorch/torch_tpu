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

#include "csrc/eager/tpu_hooks.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/detail/PrivateUse1HooksInterface.h"
#include "ATen/ops/empty.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/core/ScalarType.h"
#include "c10/core/Storage.h"
#include "c10/core/Stream.h"
#include "c10/core/TensorOptions.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "c10/macros/Export.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dtype.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/environment.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/macro_utils.h"
#include "csrc/common/tpu_hardware.h"
#include "csrc/common/warnings.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/device_gen_impl.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/host_allocator.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "csrc/pjrt/pjrt_state.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch/headeronly/macros/Export.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {
namespace {

constexpr int kMaxDevices = 8;

// Returns whether `device_index` is below `torch.tpu.device_count()`.
[[nodiscard]] bool IsVisibleDeviceIndex(c10::DeviceIndex device_index) {
  const absl::StatusOr<int> visible_device_count =
      PjrtBackend::GetInstance().GetVisibleDeviceCount();
  return visible_device_count.ok() && device_index < *visible_device_count;
}

void ValidateDeviceIndex(c10::DeviceIndex device_index) {
  // This error message should not be hit; it should be impossible for users to
  // construct a torch.device with a negative device index, which will fail
  // before this point.
  TT_CHECK_THROW(device_index >= 0, error::kInvalidArgument)
      << "expected device index to be >= 0, got "
      << static_cast<int>(device_index);

  TT_CHECK_THROW(device_index < kMaxDevices, error::kInvalidArgument)
      << "device index " << static_cast<int>(device_index)
      << " is out of bounds of the maximum device count " << kMaxDevices;

  const absl::StatusOr<xla::PjRtClient* absl_nonnull> pjrt_client_or =
      PjrtBackend::GetInstance().GetClient();
  if (!pjrt_client_or.ok()) {
    TT_WARN(WarningCategory::kTorchTpu,
            "PJRT client not available, using TPU device index ",
            static_cast<int>(device_index),
            " unchecked: ", pjrt_client_or.status().ToString());
    return;
  }

  const int addressable_device_count =
      (*pjrt_client_or)->addressable_device_count();
  TT_CHECK_THROW(
      device_index < static_cast<c10::DeviceIndex>(addressable_device_count),
      error::kInvalidArgument)
      << "device index " << static_cast<int>(device_index)
      << " is out of bounds of the number of addressable devices "
      << addressable_device_count
      << (IsVisibleDeviceIndex(device_index)
              ? ". Using a visible TPU device other than the ones this process "
                "uses is not supported yet"
              : "");
}

// Translates TtError into c10::Error across every non-noexcept guard hook.
class TpuDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
 public:
  c10::DeviceType type() const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::Type, this);
  }
  c10::Device exchangeDevice(c10::Device d) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::ExchangeDevice, this,
                                      d);
  }
  c10::Device getDevice() const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::GetDevice, this);
  }
  void setDevice(c10::Device d) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::SetDevice, this, d);
  }
  void uncheckedSetDevice(c10::Device d) const noexcept override;
  c10::Stream getStream(c10::Device d) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::GetStream, this, d);
  }
  c10::Stream getNewStream(c10::Device d, int priority) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::GetNewStream, this,
                                      d, priority);
  }
  c10::Stream getDefaultStream(c10::Device d) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::GetDefaultStream,
                                      this, d);
  }
  c10::Stream exchangeStream(c10::Stream s) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::ExchangeStream, this,
                                      s);
  }
  // All `void* event` values are actually `EventSnapshot*`.
  void destroyEvent(void* event,
                    c10::DeviceIndex device_index) const noexcept override;
  void record(void** event, const c10::Stream& stream,
              c10::DeviceIndex device_index,
              c10::EventFlag flag) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::Record, this, event,
                                      stream, device_index, flag);
  }
  void block(void* event, const c10::Stream& stream) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::Block, this, event,
                                      stream);
  }
  bool queryEvent(void* event) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::QueryEvent, this,
                                      event);
  }
  // Returns how many addressable PJRT TPU devices this process's runtime owns,
  // initializing the runtime first if needed (0 if initialization fails). If
  // the runtime isn't initialized and no TPU device is reachable (no TPU device
  // nodes and no custom device paths), returns 0 without initializing it.
  //
  // Unlike `torch.tpu.device_count()`, counts only addressable devices, since
  // PyTorch sizes per-device state with it, e.g. the autograd engine sets each
  // device index below it on a thread of its own.
  c10::DeviceIndex deviceCount() const noexcept override;
  bool queryStream(const c10::Stream& stream) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::QueryStream, this,
                                      stream);
  }
  void synchronizeStream(const c10::Stream& stream) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::SynchronizeStream,
                                      this, stream);
  }
  void synchronizeDevice(c10::DeviceIndex device_index) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::SynchronizeDevice,
                                      this, device_index);
  }
  void synchronizeEvent(void* event) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::SynchronizeEvent,
                                      this, event);
  }
  double elapsedTime(void* event1, void* event2,
                     c10::DeviceIndex device_index) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::ElapsedTime, this,
                                      event1, event2, device_index);
  }
#if TT_TORCH_VERSION_GE(2, 12)  // PyTorch 2.12 added isStreamCapturing.
  bool isStreamCapturing(const c10::Stream& stream) const override {
    return TranslateTtErrorToC10Error(&TpuDeviceGuardImpl::IsStreamCapturing,
                                      this, stream);
  }
#endif
  // Not implemented: getStreamFromGlobalPool
  // Not implemented: getDeviceCapability
  // Not implemented: getStreamNativeHandle
  // Not implemented: recordDataPtrOnStream
  // Not implemented: getStreamCaptureId, which identifies the graph capture in
  //   progress on a stream. It intentionally returns 0, the ID of no capture:
  //   copy-on-write storages record the ID of the capture that created them
  //   only on CUDA, and reject the reads in a capture whose ID differs (see
  //   c10/core/impl/COW.cpp), so a nonzero ID would reject every read of a
  //   copy-on-write TPU storage during capture.

 private:
  c10::DeviceType Type() const;
  c10::Device ExchangeDevice(c10::Device d) const;
  c10::Device GetDevice() const;
  void SetDevice(c10::Device d) const;
  c10::Stream GetStream(c10::Device d) const;
  c10::Stream GetNewStream(c10::Device d, int priority) const;
  c10::Stream GetDefaultStream(c10::Device d) const;
  c10::Stream ExchangeStream(c10::Stream s) const;
  void Record(void** event, const c10::Stream& stream,
              c10::DeviceIndex device_index, c10::EventFlag flag) const;
  void Block(void* event, const c10::Stream& stream) const;
  bool QueryEvent(void* event) const;
  bool QueryStream(const c10::Stream& stream) const;
  void SynchronizeStream(const c10::Stream& stream) const;
  void SynchronizeDevice(c10::DeviceIndex device_index) const;
  void SynchronizeEvent(void* event) const;
  double ElapsedTime(void* event1, void* event2,
                     c10::DeviceIndex device_index) const;
  bool IsStreamCapturing(const c10::Stream& stream) const;
  void ValidateDevice(c10::Device d) const;
};

// Returns whether this process may reach a TPU through a device node: TPU
// device paths are configured explicitly, or a TPU device node exists at the
// default paths (see HasTpuDeviceNodes()). If not, initializing the TPU runtime
// is assumed to fail. Does not initialize the TPU runtime.
[[nodiscard]] bool MayReachTpuDevice() {
  return HasCustomTpuDevicePaths() || HasTpuDeviceNodes();
}

}  // namespace

c10::DeviceType TpuDeviceGuardImpl::Type() const {
  return GetPrivateUse1DeviceType();
}

void TpuDeviceGuardImpl::ValidateDevice(c10::Device d) const {
  // This error message should not be hit; if the device is not tpu, then it
  // should fail before calling this DeviceGuardImpl (such as in the
  // torch.accelerator submodule).
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=see comment above
      d.type() == Type(), error::kInvalidArgument)
      << "TpuDeviceGuardImpl: invalid device type " << d.type();
  ValidateDeviceIndex(d.index());
}

c10::Device TpuDeviceGuardImpl::ExchangeDevice(c10::Device d) const {
  ValidateDevice(d);
  c10::DeviceIndex old_device_index = ExchangeCurrentDeviceIndex(d.index());
  return c10::Device(Type(), old_device_index);
}

c10::Device TpuDeviceGuardImpl::GetDevice() const {
  return c10::Device(Type(), GetCurrentDeviceIndex());
}

void TpuDeviceGuardImpl::SetDevice(c10::Device d) const {
  ValidateDevice(d);
  ExchangeCurrentDeviceIndex(d.index());
}

void TpuDeviceGuardImpl::uncheckedSetDevice(c10::Device d) const noexcept {
  if (d.type() == Type()) {
    ExchangeCurrentDeviceIndex(d.index());
  }
}

c10::Stream TpuDeviceGuardImpl::GetStream(c10::Device d) const {
  ValidateDevice(d);
  const c10::StreamId stream_id = GetCurrentStreamId(d.index());
  return c10::Stream(c10::Stream::UNSAFE, d, stream_id);
}

c10::Stream TpuDeviceGuardImpl::GetNewStream(c10::Device d,
                                             int priority) const {
  ValidateDevice(d);
  const c10::StreamId stream_id = NextStreamId(d.index());
  return c10::Stream(c10::Stream::UNSAFE, d, stream_id);
}

c10::Stream TpuDeviceGuardImpl::ExchangeStream(c10::Stream s) const {
  ValidateDevice(s.device());
  c10::StreamId old_stream_id =
      ExchangeCurrentStreamId(s.device_index(), s.id());
  if (s.id() == old_stream_id) {
    // No-op, the stream is already active.
    return s;
  }

  if (GetBooleanEnvOnce<kTorchTpuInternalBlockOnStreamExitEnvVar>().value_or(
          true)) {
    // Defensively block the calling thread until all deferred ops on the old
    // stream have started execution. This ensures that any distributed
    // collectives issued by the calling thread will be executed in the exact
    // order as they were issued, even if they are on different streams.
    //
    // Normally this is not necessary in eager mode; collectives have waits
    // inserted automatically (see GlobalCollectivesOrder in events_queue.cc).
    // But it can be enabled for safety at the cost of performance for users
    // with both compiled collectives and multiple uncoordinated streams.
    TT_ASSIGN_OR_THROW(EventSnapshot snapshot,
                       MaterializeStream(s.device_index(), old_stream_id,
                                         MaterializationReason::kExplicitSync));
    TT_THROW_IF_ERROR(snapshot.BlockUntilStarted());
    // Because we're dropping this snapshot without returning it to the user,
    // any errors would be lost if we only chained the OkCallback. To ensure all
    // errors are raised correctly we inject the full completion future into
    // the stream so errors are picked up by the next snapshot.
    internal::MarkStreamActive(s.device_index(), old_stream_id,
                               snapshot.completion_future());
  } else {
    // Exiting a stream context acts as a natural graph break; this ensures that
    // the work created on the old stream will eventually be executed.
    TT_THROW_IF_ERROR(EagerlyExecuteStream(
        s.device_index(), old_stream_id, MaterializationReason::kExplicitSync));
  }

  return c10::Stream(c10::Stream::UNSAFE, s.device(), old_stream_id);
}

c10::DeviceIndex TpuDeviceGuardImpl::deviceCount() const noexcept {
  // The autograd engine calls this on the first backward() in every process,
  // even for CPU-only graphs, and sizes its device threads once from the
  // result, so a wrong 0 breaks a later TPU backward(). Initializing the
  // runtime on a host without TPUs can be very slow before it fails, though. A
  // process without TPU device nodes cannot use a TPU, so initialization would
  // fail and this would return 0 anyway; return 0 without trying. If device
  // paths are configured explicitly, they may point elsewhere, so initialize as
  // usual. CheckTpuDeviceNodesAfterInitialization() logs an error if the
  // runtime initializes on a host where this would return 0.
  //
  // Future work:
  // - Device nodes at non-default paths other than those configured explicitly
  //   are not detected.
  // - TPU device-plugin setups (VBAR_CONTROL_SERVICE_URL) are assumed to still
  //   expose /dev/vfio; this is unverified.
  // - CPU hosts with non-TPU VFIO or accel devices still pay the initialization
  //   cost (slow but correct).
  // - TpuHooksInterface::isAvailable() still initializes the runtime, and is
  //   reached via torch._C._get_accelerator() when torch.distributed or FSDP
  //   pick a default device. The same check could guard it.
  // - A CPU-only backward() on a TPU host still initializes the runtime and
  //   holds the TPUs, so forking afterwards is unsafe.
  // - Before initialization, Python's device_count() counts TPUs by PCI device
  //   ID, so it can disagree with this for TPUs whose IDs are not yet listed.
  if (!PjrtBackend::GetInstance().IsInitialized() && !MayReachTpuDevice()) {
    return 0;
  }
  // This method is noexcept, so initialization errors cannot be thrown.
  const absl::StatusOr<xla::PjRtClient* absl_nonnull> pjrt_client_or =
      PjrtBackend::GetInstance().GetClient();
  if (!pjrt_client_or.ok()) {
    ABSL_LOG_FIRST_N(WARNING, 1)
        << "TpuDeviceGuardImpl::deviceCount: PjRt backend unavailable: "
        << pjrt_client_or.status();
    return 0;
  }
  return static_cast<c10::DeviceIndex>(
      (*pjrt_client_or)->addressable_device_count());
}

c10::Stream TpuDeviceGuardImpl::GetDefaultStream(c10::Device d) const {
  ValidateDevice(d);
  return c10::Stream(c10::Stream::DEFAULT, d);
}

void TpuDeviceGuardImpl::Record(void** event, const c10::Stream& stream,
                                const c10::DeviceIndex device_index,
                                const c10::EventFlag flag) const {
  if (*event != nullptr) {
    this->destroyEvent(*event, device_index);
  }
  TT_ASSIGN_OR_THROW(EventSnapshot snapshot,
                     MaterializeStream(stream.device_index(), stream.id(),
                                       MaterializationReason::kExplicitSync));
  *event = new EventSnapshot(std::move(snapshot));
}
void TpuDeviceGuardImpl::Block(void* event, const c10::Stream& stream) const {
  EventSnapshot* snapshot = reinterpret_cast<EventSnapshot*>(event);
  c10::DeviceIndex device_index = stream.device_index();
  c10::StreamId stream_id = stream.id();
  InsertStreamWait(device_index, stream_id, *snapshot);
}

bool TpuDeviceGuardImpl::QueryEvent(void* event) const {
  EventSnapshot* snapshot = reinterpret_cast<EventSnapshot*>(event);
  TT_ASSIGN_OR_THROW(bool ready, snapshot->Query());
  return ready;
}
bool TpuDeviceGuardImpl::QueryStream(const c10::Stream& stream) const {
  TT_WARN_ONCE(WarningCategory::kTorchTpu, "queryStream not yet implemented.");
  return true;
}
void TpuDeviceGuardImpl::SynchronizeStream(const c10::Stream& stream) const {
  TT_ASSIGN_OR_THROW(EventSnapshot snapshot,
                     MaterializeStream(stream.device_index(), stream.id(),
                                       MaterializationReason::kExplicitSync));
  TT_THROW_IF_ERROR(snapshot.Synchronize());
}
void TpuDeviceGuardImpl::SynchronizeDevice(
    c10::DeviceIndex device_index) const {
  TT_ASSIGN_OR_THROW(
      std::vector<EventSnapshot> snapshots,
      MaterializeDevice(device_index, MaterializationReason::kExplicitSync));
  for (const auto& snapshot : snapshots) {
    TT_THROW_IF_ERROR(snapshot.Synchronize());
  }
}
void TpuDeviceGuardImpl::destroyEvent(
    void* event, const c10::DeviceIndex device_index) const noexcept {
  EventSnapshot* snapshot = reinterpret_cast<EventSnapshot*>(event);
  delete snapshot;
}

void TpuDeviceGuardImpl::SynchronizeEvent(void* event) const {
  EventSnapshot* snapshot = reinterpret_cast<EventSnapshot*>(event);
  TT_THROW_IF_ERROR(snapshot->Synchronize());
}

double TpuDeviceGuardImpl::ElapsedTime(
    void* event1, void* event2, const c10::DeviceIndex device_index) const {
  EventSnapshot* start = reinterpret_cast<EventSnapshot*>(event1);
  EventSnapshot* end = reinterpret_cast<EventSnapshot*>(event2);
  TT_ASSIGN_OR_THROW(bool start_ready, start->Query());
  TT_ASSIGN_OR_THROW(bool end_ready, end->Query());
  TT_CHECK_THROW(start_ready && end_ready, error::kFailedPrecondition)
      << "both events must be completed before calculating elapsed time";
  TT_ASSIGN_OR_THROW(std::chrono::steady_clock::time_point start_time,
                     start->CompletionTime());
  TT_ASSIGN_OR_THROW(std::chrono::steady_clock::time_point end_time,
                     end->CompletionTime());
  absl::Duration duration = absl::FromChrono(end_time - start_time);
  return absl::ToDoubleMilliseconds(duration);
}

bool TpuDeviceGuardImpl::IsStreamCapturing(const c10::Stream& stream) const {
  // A pure query: unlike the other methods, this does not validate the device,
  // so that querying any stream neither raises nor initializes the runtime.
  return torch_tpu::IsStreamCapturing(stream.device_index(), stream.id());
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
    return PjrtBackend::GetInstance().IsInitialized();
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
        (IgnoreInCacheKey(storage, "Doesn't affect SHLO"),
         IgnoreInCacheKey(new_bytes, "Doesn't affect SHLO")),
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
                CreateEmptyDeviceBufferRef({new_numel}, element_type));
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
              CreateEmptyDeviceBufferRef({new_numel}, element_type));
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
    TT_THROW_IF_ERROR(PjrtBackend::GetInstance().EnsureInitialized());
    return true;
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

void CheckTpuDeviceNodesAfterInitialization() {
  // Without a reachable TPU device, TpuDeviceGuardImpl::deviceCount() returns 0
  // without initializing the runtime, on the assumption that initialization
  // would fail. The runtime just initialized successfully, so check that
  // assumption.
  if (MayReachTpuDevice()) {
    return;
  }
  ABSL_LOG_FIRST_N(ERROR, 1)
      << "The TPU runtime initialized successfully, but no TPU device node "
         "was found (checked /dev/vfio/<group>, /dev/vfio/devices/*, "
         "/dev/accel*, /dev/accel/*, /dev_borg/accel* and /dev_borg/accel/*, "
         "counting accel nodes only if their major is registered in "
         "/proc/devices) and no TPU device paths are configured explicitly. "
         "On this host, torch_tpu reports 0 TPU devices to PyTorch's device "
         "guard (deviceCount()) before the runtime is initialized, e.g. on "
         "the first backward() of a process. Please file a bug at "
         "https://github.com/google-pytorch/torch_tpu/issues with the TPU "
         "device paths of this host.";
}

absl::Status AddTpuHooks() {
  at::RegisterPrivateUse1HooksInterface(GetTpuHooks());

  return absl::OkStatus();
}

}  // namespace torch_tpu
