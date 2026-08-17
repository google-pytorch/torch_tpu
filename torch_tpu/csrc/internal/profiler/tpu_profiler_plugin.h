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

#ifndef TORCH_TPU_INTERNAL_PROFILER_TPU_PROFILER_H_
#define TORCH_TPU_INTERNAL_PROFILER_TPU_PROFILER_H_

#include <kineto/ActivityType.h>
#include <kineto/IActivityProfiler.h>

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/internal/profiler/xprof_callback_handler.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace torch_tpu {

// TPU implementation of libkineto::IActivityProfilerSession.
// It manages a single profiling session, interacting with tensorflow::profiler.
class TpuKinetoProfilerSession : public libkineto::IActivityProfilerSession {
 public:
  TpuKinetoProfilerSession(const libkineto::Config& config,
                           std::set<libkineto::ActivityType> activity_types);

  void start() override ABSL_LOCKS_EXCLUDED(mutex_);
  void stop() override ABSL_LOCKS_EXCLUDED(mutex_);
  std::vector<std::string> errors() override ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(mutex_);
    return errors_;
  }
  void processTrace(libkineto::ActivityLogger& logger) override;
  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override {
    // DeviceInfo parameters:
    // 0: Process ID (using 0 as default/placeholder)
    // 0: Sort index in the trace view
    // "TPU": Device/Process name
    // "TPU Device": Human-readable label
    return std::make_unique<libkineto::DeviceInfo>(0, 0, "TPU", "TPU Device");
  }
  std::vector<libkineto::ResourceInfo> getResourceInfos() override {
    return resource_infos_;
  }
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override {
    return nullptr;
  }

 private:
  mutable absl::Mutex mutex_;
  std::unique_ptr<tsl::ProfilerSession> session_;
  tensorflow::profiler::XSpace xspace_;
  std::vector<std::string> errors_ ABSL_GUARDED_BY(mutex_);
  const libkineto::Config& config_;
  std::set<libkineto::ActivityType> activity_types_;
  std::unique_ptr<XProfCallbackHandler> callback_handler_;
  std::string run_dir_ ABSL_GUARDED_BY(mutex_);
  std::optional<std::string> worker_rank_ ABSL_GUARDED_BY(mutex_);
  uint64_t start_time_ns_ = 0;
  std::vector<libkineto::ResourceInfo> resource_infos_;
};

// TPU implementation of libkineto::IActivityProfiler.
// It is responsible for creating TpuKinetoProfilerSession instances.
class TpuProfiler : public libkineto::IActivityProfiler {
 public:
  const std::string& name() const override {
    static const absl::NoDestructor<std::string> kName("tpu_profiler");
    return *kName;
  }
  const std::set<libkineto::ActivityType>& availableActivities()
      const override {
    // PRIVATEUSE1_RUNTIME represents the runtime API calls for the custom
    // device (TPU).
    static const absl::NoDestructor<std::set<libkineto::ActivityType>>
        activities({libkineto::ActivityType::PRIVATEUSE1_RUNTIME});
    return *activities;
  }
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override {
    return std::make_unique<TpuKinetoProfilerSession>(config, activity_types);
  }
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t /*ts_ms*/, int64_t /*duration_ms*/,
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override {
    return configure(activity_types, config);
  }
};

absl::Status UpdateProfileOptions(std::string_view custom_config,
                                  tensorflow::ProfileOptions& opts,
                                  std::string& out_run_dir,
                                  std::optional<std::string>& out_worker_rank);

absl::Status UpdateProfileOptions(std::string_view custom_config,
                                  tensorflow::ProfileOptions& opts,
                                  std::string& out_run_dir);

}  // namespace torch_tpu

#endif  // TORCH_TPU_INTERNAL_PROFILER_TPU_PROFILER_H_
