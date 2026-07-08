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

#include "torch_tpu/pjrt/pjrt_state.h"

#include <memory>
#include <string>
#include <string_view>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "torch_tpu/common/contain.h"
#include "torch_tpu/common/device_type.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/environment.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/distributed/slicebuilder/discovery.h"
#include "torch_tpu/pjrt/pjrt_client.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "xla/backends/profiler/plugin/plugin_tracer.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/plugin_names.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {
namespace {

bool IsRunningInTest() {
  return GetEnvOnce<kTestWorkspaceEnvVar>().has_value() ||
         GetEnvOnce<kTestTargetEnvVar>().has_value();
}

// Creates a PluginTracer instance to collect TPU profile metrics.
// Returns nullptr in statically-linked test environments to prevent TSL global
// mutex deadlocks.
std::unique_ptr<tsl::profiler::ProfilerInterface> absl_nullable
CreateTpuProfiler(const PLUGIN_Profiler_Api* absl_nonnull profiler_api,
                  const tensorflow::ProfileOptions& opts) {
  // In test environments (e.g. running on simulators with shared symbols),
  // the mock libtpu plugin is often linked statically into the framework
  // binary.
  //
  // When the framework starts tracing, it calls
  // `tsl::profiler::CreateProfilers`, which acquires a global non-recursive
  // mutex in TSL.
  //
  // 1. Framework acquires Global Mutex A.
  // 2. Calls this factory lambda to create `PluginTracer`.
  // 3. `PluginTracer` constructor calls the plugin (mock libtpu).
  // 4. The mock plugin attempts to initialize its own profilers by calling
  //    `tsl::profiler::CreateProfilers` again (nested call).
  // 5. The nested call tries to acquire Global Mutex A and deadlocks!
  //
  // Since we cannot modify TSL (to make it re-entrant safe or using
  // recursive mutexes) and we cannot modify the mock plugin (to stop it
  // from calling out), the only safe solution in this statically-linked
  // environment is to SKIP creating the `PluginTracer` in tests.
  if (IsRunningInTest()) {
    ABSL_LOG(WARNING)
        << "Skipping PluginTracer in test environment to avoid shared "
           "TSL mutex deadlock.";
    return nullptr;
  }
  return std::make_unique<xla::profiler::PluginTracer>(profiler_api, opts);
}

// Dynamically discovers the PJRT Profiler Extension within the given plugin
// and registers the factory to collect performance metrics
void MaybeRegisterProfiler(std::string_view plugin_name) {
  // Retrieve the core PJRT C-API interface for this plugin (e.g. libtpu).
  absl::StatusOr<const PJRT_Api*> api = pjrt::PjrtApi(plugin_name);
  if (!api.ok() || *api == nullptr) {
    return;
  }

  // Attempt to discover the standard PJRT Profiler Extension within the plugin.
  auto* profiler_ext = pjrt::FindExtension<PJRT_Profiler_Extension>(
      *api, PJRT_Extension_Type::PJRT_Extension_Type_Profiler);
  if (profiler_ext == nullptr || profiler_ext->profiler_api == nullptr) {
    ABSL_VLOG(1) << "PJRT Profiler Extension not found for plugin: "
                 << plugin_name;
    return;
  }

  // Ensure that the profiler factory is registered exactly once per process,
  // even if multiple client initializations occur concurrently.
  static const bool factory_registered = [profiler_ext]() {
    const PLUGIN_Profiler_Api* profiler_api = profiler_ext->profiler_api;
    tsl::profiler::RegisterProfilerFactory(
        [profiler_api](const tensorflow::ProfileOptions& opts) {
          return CreateTpuProfiler(profiler_api, opts);
        });
    return true;
  }();
  static_cast<void>(factory_registered);  // VOID_CAST_OK=dummy result.
}

}  // namespace

PjrtBackend& PjrtBackend::GetInstance() {
  static absl::NoDestructor<PjrtBackend> instance;
  return *instance;
}

void PjrtBackend::SetPjRtInitializationOptions(
    const PjRtInitializationOptions& options) {
  absl::MutexLock lock(mutex_);
  options_ = options;
}

absl::Status PjrtBackend::EnsureInitialized() {
  {
    absl::ReaderMutexLock lock(mutex_);
    if (init_attempted_) {
      return init_status_;
    }
  }

  absl::MutexLock lock(mutex_);
  if (init_attempted_) {
    return init_status_;
  }

  init_status_ = InitializeInternal();
  init_attempted_ = true;
  return init_status_;
}

absl::Status PjrtBackend::InitializeInternal() {
  PjRtDeviceType device_type = PjRtDeviceType::kUnknown;
  std::string plugin_name;
  if (options_.device_type == "tpu") {
    device_type = PjRtDeviceType::kTpu;
    plugin_name = kTpuPjrtName;

    if (auto world_size_or = GetWorldSizeFromEnvOnce(); world_size_or.ok()) {
      TT_ASSIGN_OR_RETURN(auto config, GetDistributedWorkerConfiguration());
      TT_RETURN_IF_ERROR(InitializeDistributedEnvironment(config)).SetPrepend()
          << "InitializeDistributedEnvironment failed: ";
    }
  } else if (options_.device_type == "xla_cuda") {
    plugin_name = kGpuPjrtName;
    device_type = PjRtDeviceType::kCuda;
  } else if (options_.device_type == "xla_cpu") {
    plugin_name = kCpuPjrtName;
    device_type = PjRtDeviceType::kCpu;
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "unsupported device type: " << options_.device_type;
  }
  // Move active thread into container for memory tracking, if the flag is
  // enabled.
  torch_tpu::ScopedMemMeasuringContainer container;

  TT_ASSIGN_OR_RETURN(bool is_initialized,
                      pjrt::IsPjrtPluginInitialized(plugin_name));
  if (!is_initialized) {
    TT_RETURN_IF_ERROR(pjrt::InitializePjrtPlugin(plugin_name)).SetPrepend()
        << "initializing the PjRt plugin failed with: ";
  }

  // Mem tracking note: Background threads in XLA spun up during GetPjRtClient.
  TT_ASSIGN_OR_RETURN(
      client_,
      torch_tpu::GetPjRtClient(options_.device_type,
                               options_.premapped_buffer_size_bytes),
      _.SetPrepend() << "retrieving a PjRtClient failed with: ");

  ABSL_CHECK_NE(client_, nullptr)  // CRASH_OK
      << "PjRtClient should not be null after initialization.";
  ABSL_CHECK(!client_->devices().empty())  // CRASH_OK
      << "No PjRt devices found after client initialization.";

  if (options_.device_type == "tpu") {
    MaybeRegisterProfiler(plugin_name);
  }

  const auto& addressable_devices = client_->addressable_devices();
  ABSL_CHECK(!addressable_devices.empty())  // CRASH_OK
      << "No addressable PjRt devices found.";

  int world_size = 1;
  if (auto world_size_or = GetWorldSizeFromEnvOnce(); world_size_or.ok()) {
    world_size = *world_size_or;
  }

  const int device_count = client_->device_count();

  // Initialize the global device count. This is used by the CompilationCache
  // to determine the number of replicas of the XLA computation.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=We can hit this error check, but we
                 // currently do not propagate its status.
      world_size == 1 || world_size == device_count, error::kInvalidArgument)
      << "invalid world_size configuration: expected 1 or " << device_count
      << ", but got " << world_size
      << ". Please check your environment variables";

  if (addressable_devices.size() > 1 && world_size == 1) {
    ABSL_LOG(WARNING) << "Only using 1 device out of all the "
                      << addressable_devices.size() << " addressable devices.";
  }

  device_ = addressable_devices[0];
  device_type_ = device_type;
  global_device_count_ = world_size;
  global_device_id_ = device_->global_device_id().value();
  return absl::OkStatus();
}

bool PjrtBackend::IsInitialized() const {
  absl::ReaderMutexLock lock(mutex_);
  return client_ != nullptr;
}

xla::PjRtClient* absl_nullable PjrtBackend::GetClient() {
  if (const absl::Status s = EnsureInitialized(); !s.ok()) {
    ABSL_LOG(ERROR) << "PjrtBackend::GetClient failed to initialize: " << s;
    return nullptr;
  }
  absl::ReaderMutexLock lock(mutex_);
  return client_.get();
}

absl::StatusOr<absl_nonnull std::shared_ptr<xla::PjRtClient>>
PjrtBackend::GetSharedClient() {
  TT_RETURN_IF_ERROR(EnsureInitialized());
  absl::ReaderMutexLock lock(mutex_);
  return client_;
}

xla::PjRtDevice* absl_nullable PjrtBackend::GetDevice() {
  if (const absl::Status s = EnsureInitialized(); !s.ok()) {
    ABSL_LOG(ERROR) << "PjrtBackend::GetDevice failed to initialize: " << s;
    return nullptr;
  }
  absl::ReaderMutexLock lock(mutex_);
  return device_;
}

absl::StatusOr<int> PjrtBackend::GetGlobalDeviceCount() {
  TT_RETURN_IF_ERROR(EnsureInitialized());
  absl::ReaderMutexLock lock(mutex_);
  if (client_ == nullptr) {
    return TT_ERROR(error::kInternal) << "PjRt is not initialized";
  }
  return global_device_count_;
}

absl::StatusOr<int> PjrtBackend::GetGlobalDeviceId() {
  TT_RETURN_IF_ERROR(EnsureInitialized());
  absl::ReaderMutexLock lock(mutex_);
  if (client_ == nullptr) {
    return TT_ERROR(error::kInternal) << "PjRt is not initialized";
  }
  return global_device_id_;
}

PjRtDeviceType PjrtBackend::GetDeviceType() {
  absl::ReaderMutexLock lock(mutex_);
  return device_type_;
}

absl::StatusOr<tsl::AllocatorStats> PjrtBackend::GetAllocatorStats() {
  xla::PjRtDevice* device = GetDevice();
  if (device == nullptr) {
    return TT_ERROR(error::kInternal) << "PjRt device is not initialized";
  }
  return device->GetAllocatorStats();
}

absl::Status PjrtBackend::ClearAllocatorStats() {
  xla::PjRtDevice* device = GetDevice();
  if (device == nullptr) {
    return TT_ERROR(error::kInternal) << "PjRt device is not initialized";
  }
  return device->ClearMemoryStats();
}

absl::StatusOr<xla::HostMemoryAllocator*> PjrtBackend::GetHostAllocator() {
  xla::PjRtClient* client = GetClient();
  if (client == nullptr) {
    return TT_ERROR(error::kInternal) << "PjRt client is not initialized";
  }
  if (auto host_allocator = client->GetHostMemoryAllocator();
      host_allocator != nullptr) {
    return host_allocator;
  }
  return TT_ERROR(error::kPythonNotImplementedError)
         << "host memory allocator is not implemented for this client";
}

void PjrtBackend::Shutdown() {
  absl::MutexLock lock(mutex_);
  if (!init_attempted_) {
    return;
  }
  ABSL_VLOG(1) << "ShutdownPjRt";

  client_.reset();
  device_ = nullptr;
  device_type_ = PjRtDeviceType::kUnknown;
  global_device_count_ = 0;
  init_attempted_ = false;
  init_status_ = absl::OkStatus();
  ABSL_VLOG(1) << "PjRt shut down.";
}

}  // namespace torch_tpu
