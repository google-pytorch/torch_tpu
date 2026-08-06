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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "ATen/core/Generator.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/time/time.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "pybind11/chrono.h"
#include "pybind11/gil.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch/csrc/utils/pybind.h"  // IWYU pragma: keep, needed for at::Tensor mapping
#include "torch_tpu/common/compilation_cache.h"
#include "torch_tpu/common/device_type.h"
#include "torch_tpu/common/discovery.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/excess_precision.h"
#include "torch_tpu/eager/current_stream.h"
#include "torch_tpu/eager/device_gen_impl.h"
#include "torch_tpu/eager/events_queue.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/eager/tpu_hooks.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

namespace py = pybind11;

namespace {

// A wrapper around a TPU event snapshot to be returned to Python.
// This mostly exists so that the methods can throw errors, rather than return
// a status.
class PyTpuEventBase {
 public:
  explicit PyTpuEventBase(
      absl_nonnull std::shared_ptr<EventSnapshot> event_snapshot)
      : event_snapshot_(std::move(event_snapshot)) {}

  // Wait for the event snapshot to complete.
  void Wait() { TT_THROW_IF_ERROR(event_snapshot_->Wait()); }

  // Query whether the event snapshot has completed.
  bool Query() {
    TT_ASSIGN_OR_THROW(bool is_ready, event_snapshot_->Query());
    return is_ready;
  }

 private:
  // The wrapped event snapshot.
  absl_nonnull std::shared_ptr<EventSnapshot> event_snapshot_;
};

void PySynchronizeDevice(int device_index) {
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl(GetPrivateUse1DeviceType());
  TT_CHECK_THROW(impl != nullptr, error::kInternal)
      << "TPU DeviceGuardImpl not found";

  impl->synchronizeDevice(device_index);
}

PyTpuEventBase PyRecordEvent(std::optional<int> device_index,
                             std::optional<int64_t> stream_id) {
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl(GetPrivateUse1DeviceType());
  TT_CHECK_THROW(impl != nullptr, error::kInternal)
      << "TPU DeviceGuardImpl not found";

  c10::DeviceIndex index = device_index.has_value()
                               ? static_cast<c10::DeviceIndex>(*device_index)
                               : impl->getDevice().index();

  c10::StreamId id;
  if (stream_id.has_value()) {
    id = *stream_id;
  } else {
    c10::Stream current_stream =
        impl->getStream(c10::Device(GetPrivateUse1DeviceType(), index));
    id = current_stream.id();
  }
  return PyTpuEventBase(EventSnapshot::Record(index, id));
}

void InitRuntimeOptions(const std::string& device_type) {
  if (device_type == "tpu" || device_type == "xla_cuda") {
    TT_THROW_IF_ERROR(AddTpuHooks()) << "Failed to add TPU hooks.";
    RegisterTpuAllocator();
  }
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = device_type,
       .premapped_buffer_size_bytes =
           GetPremappedBufferSizeFromEnvOnce().value_or(0)});
  ABSL_LOG(INFO) << "PjRt runtime initialization deferred for " << device_type;
  CompilationCache::GetInstance().SetOptions({});
}

int64_t PyGetCurrentStreamId(std::optional<int> device_index) {
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl(GetPrivateUse1DeviceType());
  TT_CHECK_THROW(impl != nullptr, error::kInternal)
      << "TPU DeviceGuardImpl not found";

  c10::DeviceIndex index = device_index.has_value()
                               ? static_cast<c10::DeviceIndex>(*device_index)
                               : impl->getDevice().index();
  c10::Stream current_stream =
      impl->getStream(c10::Device(GetPrivateUse1DeviceType(), index));
  return current_stream.id();
}

c10::Stream MakeStream(const c10::impl::DeviceGuardImplInterface& impl,
                       int64_t stream_id, std::optional<int> device_index) {
  c10::DeviceIndex index = device_index.has_value()
                               ? static_cast<c10::DeviceIndex>(*device_index)
                               : impl.getDevice().index();
  c10::Stream s =
      c10::Stream(c10::Stream::UNSAFE,
                  c10::Device(GetPrivateUse1DeviceType(), index), stream_id);
  return s;
}

void PySetCurrentStreamId(int64_t stream_id, std::optional<int> device_index) {
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl(GetPrivateUse1DeviceType());
  TT_CHECK_THROW(impl != nullptr, error::kInternal)
      << "TPU DeviceGuardImpl not found";
  impl->exchangeStream(MakeStream(*impl, stream_id, device_index));
}

void PySynchronizeStream(int64_t stream_id, std::optional<int> device_index) {
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl(GetPrivateUse1DeviceType());
  TT_CHECK_THROW(impl != nullptr, error::kInternal)
      << "TPU DeviceGuardImpl not found";
  impl->synchronizeStream(MakeStream(*impl, stream_id, device_index));
}

int64_t PyNextStreamId(int64_t device_index) {
  TT_CHECK_THROW(device_index >= 0 && device_index < 8, error::kInvalidArgument)
      << "device index must be in the range [0, 8), but got: " << device_index;
  return NextStreamId(device_index);
}

py::dict PyGetLocalDeviceAttributes() {
  absl::StatusOr<PjRtDeviceAttributes> attrs_or =
      PjrtBackend::GetInstance().GetDeviceAttributes();
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PjrtBackend is always initialized in
                   // Python test environment.
      attrs_or.ok(), error::kInternal)
      << attrs_or.status().message();

  py::dict attrs;
  attrs["id"] = attrs_or->id;
  attrs["process_index"] = attrs_or->process_index;
  attrs["device_kind"] = attrs_or->device_kind;
  if (attrs_or->coords.has_value()) {
    attrs["coords"] = *attrs_or->coords;
  }
  if (attrs_or->core_on_chip.has_value()) {
    attrs["core_on_chip"] = *attrs_or->core_on_chip;
  }
  if (attrs_or->slice_index.has_value()) {
    attrs["slice_index"] = *attrs_or->slice_index;
  }

  return attrs;
}

}  // namespace

PYBIND11_MODULE(_device_ops_backend, m) {
  py::module::import("torch");
  m.doc() =
      "PjRt backend utilities for PyTorch PrivateUse1 integration. "
      "Core ops are registered via C++ TORCH_LIBRARY_IMPL.";

  m.def("_init_runtime_options", &InitRuntimeOptions,
        py::arg("device_type") = "tpu",
        "Initializes the PjRt runtime options for the specified device type. "
        "This function configures backend options and hooks, but actual "
        "hardware initialization is deferred until the first device request.");

  m.def(
      "_is_initialized",
      []() -> bool { return PjrtBackend::GetInstance().IsInitialized(); },
      "Returns whether the PjRt runtime is initialized.");

  m.def(
      "_shutdown_runtime",
      []() {
        ShutDownMaterializationState();
        CompilationCache::ShutDown();
        PjrtBackend::GetInstance().Shutdown();
      },
      "Shuts down the PjRt runtime.");

  m.def("_get_current_stream_id", &PyGetCurrentStreamId,
        py::arg("device_index") = py::none(),
        "Returns the current stream ID for the specified device.");

  m.def("_set_current_stream_id", &PySetCurrentStreamId, py::arg("stream_id"),
        py::arg("device_index") = py::none(),
        "Sets the current stream ID for the specified device.");

  m.def("_synchronize_stream", &PySynchronizeStream, py::arg("stream_id"),
        py::arg("device_index") = py::none(),
        py::call_guard<py::gil_scoped_release>(),
        "Blocks until all operations on the specified stream have completed.");

  m.def("_synchronize_device", &PySynchronizeDevice, py::arg("device_index"),
        py::call_guard<py::gil_scoped_release>(),
        "Blocks until all async d2h copies and deferred operations on the "
        "specified device have completed.");

  py::class_<PyTpuEventBase, py::smart_holder>(m, "TpuEventBase")
      .def("wait", &PyTpuEventBase::Wait,
           py::call_guard<py::gil_scoped_release>(),
           "Blocks until the recorded event snapshot has completed.")
      .def("query", &PyTpuEventBase::Query,
           "Returns whether the recorded event snapshot has completed.");

  m.def("_record_event", &PyRecordEvent, py::arg("device_index") = py::none(),
        py::arg("stream_id") = py::none(),
        "Records a fence over async d2h copies already enqueued on the "
        "specified device and stream.");

  m.def("_get_next_stream_id", &PyNextStreamId, py::arg("device_index"),
        "Returns the next available stream ID for the given device.");

  m.def(
      "_get_current_device_id",
      []() -> int {
        xla::PjRtClient* client = PjrtBackend::GetInstance().GetClient();
        ABSL_CHECK(client != nullptr)  // CRASH_OK
            << "PjRtClient is null after initialization.";
        xla::PjRtDevice* device = PjrtBackend::GetInstance().GetDevice();
        ABSL_CHECK(device != nullptr)  // CRASH_OK
            << "PjRtDevice is null after initialization.";

        const auto& addressable_devices = client->addressable_devices();
        for (size_t i = 0; i < addressable_devices.size(); ++i) {
          if (addressable_devices[i] == device) {
            return i;
          }
        }
        ABSL_LOG(FATAL)  // CRASH_OK
            << "Current device is not addressable: global_device_id="
            << device->global_device_id();
      },
      "Returns the local addressable ID of the current PJRT device.");

  m.def(
      "_get_device_count",
      []() -> int {
        xla::PjRtClient* client = PjrtBackend::GetInstance().GetClient();
        TT_CHECK_THROW(client != nullptr, error::kInternal)
            << "PjRtClient is null after initialization.";
        return client->addressable_device_count();
      },
      "Returns the number of devices visible to the PJRT client. This count "
      "is equivalent to the addressable device count.");

  m.def(
      "_get_local_device_attributes", &PyGetLocalDeviceAttributes,
      "Returns explicitly needed attributes and properties of the current PJRT "
      "device.");

  m.def(
      "_set_allow_cache",
      [](bool allow) {
        CompilationCache::GetInstance().SetAllowCacheMode(allow);
      },
      py::arg("allow"),
      "If True (default), the compilation cache will be enabled. If False, "
      "the compilation cache will be disabled and every computation graph "
      "will be compiled from scratch (this is useful for debugging and perf "
      "analysis).");

  m.def(
      "_set_cache_only",
      [](bool cache_only) {
        CompilationCache::GetInstance().SetCacheOnlyMode(cache_only);
      },
      py::arg("cache_only"),
      "If True, the compilation cache will only lookup and not compile. Any "
      "cache miss will result in an error.");

  m.def(
      "_set_dump_on_cache_miss",
      [](bool enable) {
        CompilationCache::GetInstance().SetDumpOnCacheMissMode(enable);
      },
      py::arg("enable"),
      "If True, the compilation cache will dump the input StableHLO module for "
      "any cache misses.");
  m.def(
      "_get_dump_on_cache_miss",
      []() { return CompilationCache::GetInstance().GetDumpOnCacheMissMode(); },
      "Check if compilation cache dumping is enabled.");
  m.def(
      "_get_cache_requests",
      []() { return CompilationCache::GetInstance().GetCacheRequests(); },
      "Get compilation cache requests.");
  m.def(
      "_get_cache_hits",
      []() { return CompilationCache::GetInstance().GetCacheHits(); },
      "Get compilation cache hits.");
  m.def(
      "_get_cache_misses",
      []() { return CompilationCache::GetInstance().GetCacheMisses(); },
      "Get compilation cache misses.");
  m.def(
      "_clear_cache", []() { CompilationCache::GetInstance().EvictAll(); },
      "Evict all existing entries in the compilation cache. The function "
      "waits for all in-flight compilations to complete.");
  m.def("_set_allow_excess_precision", &SetAllowExcessPrecision,
        py::arg("allow"),
        "Sets whether XLA is allowed to use excess precision for all "
        "compilations.");
  m.def("_get_allow_excess_precision", &GetAllowExcessPrecision,
        "Returns whether XLA is allowed to use excess precision for all "
        "compilations.");
  m.def(
      "_hbm_usage_summary",
      []() { return CompilationCache::GetInstance().HbmUsageSummary(); },
      "Return the total HBM usage of the compilation cache.");
  m.def(
      "manual_seed", [](uint64_t seed) { SetManualSeed(seed); },
      py::arg("seed"),
      "Manually set the seed for the generator of the current device.");
  m.def("manual_seed_all", &SetManualSeedAll, py::arg("seed"),
        "Manually set the seed for all generators (one per device).");

  m.def(
      "get_rng_state",
      [](int device_index) -> at::Tensor {
        // Gets the current internal state of the generator for a given device
        // index. The internal state is returned as a CPU byte tensor.
        const at::Generator& gen = GetDefaultDeviceGenerator(device_index);
        return gen.get_state();
      },
      py::arg("device_index"), "Get RNG state for the given device index.");

  m.def(
      "set_rng_state",
      [](at::Tensor state, int device_index) {
        // Sets the internal state of the generator for a given device index.
        // The new internal state must be a CPU byte tensor.
        at::Generator& gen = GetDefaultDeviceGenerator(device_index);
        gen.set_state(state);
      },
      py::arg("state"), py::arg("device_index"),
      "Set RNG state for the given device index.");

  m.def(
      "get_default_generator",
      [](int device_index) -> at::Generator {
        return GetDefaultDeviceGenerator(device_index);
      },
      py::arg("device_index"),
      "Get the default generator for the given device index.");

  py::class_<CacheEntryStats>(m, "CacheEntryStats")
      .def_property_readonly(
          "compilation_duration",
          [](const CacheEntryStats& stats) {
            // Pybind doesn't support absl::Duration, so we convert to
            // microseconds.
            return absl::ToChronoMicroseconds(stats.compilation_duration);
          })
      .def_property_readonly("last_read",
                             [](const CacheEntryStats& stats) {
                               // Pybind doesn't support absl::Time, so we
                               // convert to chrono time.
                               return absl::ToChronoTime(stats.last_read);
                             })
      .def_readonly("read_count", &CacheEntryStats::read_count);

  // Suppress the clang-tidy warning that the variable is destroyed immediately
  // after the definition. We are just using the ctor's side effect to register
  // the class with pybind11.
  py::class_<PerfStats::EntryStats, CacheEntryStats>(  // NOLINT
      m, "EntryStats");

  py::class_<PerfStats>(m, "PerfStats")
      .def_readonly("num_cache_reqs", &PerfStats::num_cache_reqs)
      .def_readonly("num_cache_hits", &PerfStats::num_cache_hits)
      // Gets the peak memory used by the host during compilation. Resetting
      // value during eviction is not supported, limitation with cgroups v2.
      .def_readonly("peak_compilation_host_memory_bytes",
                    &PerfStats::peak_compilation_memory_bytes)
      .def_readonly("per_entry_stats", &PerfStats::per_entry_stats);

  m.def(
      "_get_cache_stats",
      []() { return CompilationCache::GetInstance().GetCacheStats(); },
      "Get compilation cache statistics.");

  m.def(
      "_is_optimized_build",
      []() -> bool {
#ifdef NDEBUG
        return true;
#else
        return false;
#endif
      },
      "Returns True if this is an optimized build (compiled with -c opt / "
      "NDEBUG defined).");
}

}  // namespace torch_tpu
