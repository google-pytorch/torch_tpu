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

#include <cstdint>
#include <optional>
#include <string>

#include "absl/log/absl_log.h"
#include "absl/time/time.h"
#include "ATen/core/Generator.h"
#include "c10/core/Device.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "torch/csrc/utils/pybind.h"  // IWYU pragma: keep, needed for at::Tensor mapping
#include "torch_tpu/common/compilation_cache.h"
#include "torch_tpu/common/discovery.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_gen_impl.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/eager/tpu_hooks.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "pybind11/chrono.h"
#include "pybind11/gil.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

namespace py = pybind11;

namespace {
void SetRngStatePy(at::Tensor state, int device_index) {
  const at::Generator& gen = GetDefaultDeviceGenerator(device_index);
  if (state.dtype() == at::kByte) {
    state = state.view(at::kUInt64);
  }
  if (state.device() != gen.device()) {
    state = state.to(gen.device());
  }
  TT_THROW_IF_ERROR(UpdateRngState(gen, state)).SetPrepend()
      << "failed to set RNG state: ";
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

void WaitEventSnapshotPy(int64_t event_id) {
  TT_THROW_IF_ERROR(WaitEventSnapshot(event_id)).SetPrepend()
      << "failed waiting for event snapshot: ";
}

bool QueryEventSnapshotPy(int64_t event_id) {
  TT_ASSIGN_OR_THROW(bool is_ready, QueryEventSnapshot(event_id),
                     _.SetPrepend() << "failed querying event snapshot: ");
  return is_ready;
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
        CompilationCache::ShutDown();
        PjrtBackend::GetInstance().Shutdown();
      },
      "Shuts down the PjRt runtime.");

  m.def(
      "_synchronize",
      [](std::optional<int> device_index) {
        c10::DeviceIndex index;
        if (device_index.has_value()) {
          index = static_cast<c10::DeviceIndex>(*device_index);
        } else {
          index = c10::impl::getDeviceGuardImpl(GetPrivateUse1DeviceType())
                      ->getDevice()
                      .index();
        }
        PjrtBackend::GetInstance().SynchronizeStream(index);
      },
      py::arg("device_index") = py::none(),
      py::call_guard<py::gil_scoped_release>(),
      "Blocks until all async d2h copies on the specified device have "
      "completed.");

  m.def(
      "_record_event",
      [](std::optional<int> device_index) {
        c10::DeviceIndex index;
        if (device_index.has_value()) {
          index = static_cast<c10::DeviceIndex>(*device_index);
        } else {
          index = c10::impl::getDeviceGuardImpl(GetPrivateUse1DeviceType())
                      ->getDevice()
                      .index();
        }
        return RecordEventSnapshot(index);
      },
      py::arg("device_index") = py::none(),
      "Records a fence over async d2h copies already enqueued on the "
      "specified device.");

  m.def("_wait_event", &WaitEventSnapshotPy, py::arg("event_id"),
        py::call_guard<py::gil_scoped_release>(),
        "Blocks until the recorded event snapshot has completed.");

  m.def("_query_event", &QueryEventSnapshotPy, py::arg("event_id"),
        "Returns whether the recorded event snapshot has completed.");

  m.def(
      "_release_event",
      [](int64_t event_id) { ReleaseEventSnapshot(event_id); },
      py::arg("event_id"), "Releases the recorded event snapshot.");

  m.def(
      "_get_current_device_id",
      []() -> int {
        TT_ASSIGN_OR_THROW(int device_id,
                           PjrtBackend::GetInstance().GetGlobalDeviceId(),
                           _ << "failed to get current device ID");
        return device_id;
      },
      "Returns the global ID of the current PJRT device.");

  m.def(
      "_get_device_count",
      []() -> int {
        xla::PjRtClient* client = PjrtBackend::GetInstance().GetClient();
        TT_CHECK_THROW(client != nullptr, error::kInternal)
            << "PjRtClient is null after initialization.";
        return client->device_count();
      },
      "Returns the number of devices visible to the PJRT client. This count "
      "is equivalent to the global device count.");

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

  m.def(
      "_hbm_usage_summary",
      []() { return CompilationCache::GetInstance().HbmUsageSummary(); },
      "Return the total HBM usage of the compilation cache.");
  m.def(
      "manual_seed", [](uint64_t seed) { SetManualSeed(seed); },
      py::arg("seed"),
      "Manually set the seed for the generator of the current device.");
  m.def(
      "manual_seed_all", [](uint64_t seed) { SetManualSeedAll(seed); },
      py::arg("seed"),
      "Manually set the seed for all generators (one per device).");
  m.def(
      "get_rng_state",
      [](int device_index) -> at::Tensor {
        TT_ASSIGN_OR_THROW(at::Tensor state,
                           GetRngState(GetDefaultDeviceGenerator(device_index)),
                           _.SetPrepend() << "failed to get RNG state: ");
        return state.view(at::kByte);
      },
      py::arg("device_index"), "Get RNG state for the given device index.");
  m.def("set_rng_state", &SetRngStatePy, py::arg("state"),
        py::arg("device_index"), "Set RNG state for the given device index.");

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
      .def_readonly("per_entry_stats", &PerfStats::per_entry_stats);

  m.def(
      "_get_cache_stats",
      []() { return CompilationCache::GetInstance().GetCacheStats(); },
      "Get compilation cache statistics.");
}

}  // namespace torch_tpu
