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

#include "torch_tpu/_internal/profiler/tpu_profiler_plugin.h"

#include <kineto/ActivityType.h>
#include <kineto/Config.h>
#include <kineto/IActivityProfiler.h>
#include <kineto/output_base.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

// Forward declarations of PrivateUse1ProfilerRegistry marked as weak.
// This allows compilation against Torch 2.10 (where these don't exist)
// and safe dynamic loading on Torch 2.10 runtimes (where symbols resolve to
// null), enabling runtime dispatch without custom compile-time flags.
//
// TODO(b/499296481): Remove this weak-linking workaround and revert to
// including the privateuse1_profiler.h header and using
// REGISTER_PRIVATEUSE1_PROFILER(TpuProfiler) once PyTorch 2.10 support is
// dropped and 2.12 is the minimum required version.
namespace torch::profiler::impl {
using PrivateUse1ProfilerFactory = std::function<  // STD_FUNCTION_OK
    std::unique_ptr<libkineto::IActivityProfiler>()>;

class PrivateUse1ProfilerRegistry {
 public:
  static __attribute__((weak)) PrivateUse1ProfilerRegistry& instance();
  __attribute__((weak)) void registerFactory(
      PrivateUse1ProfilerFactory factory);
};
}  // namespace torch::profiler::impl
#include "torch_tpu/_internal/profiler/xprof_callback_handler.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xla/tsl/platform/env.h"

namespace torch_tpu {

// The set of valid parameter names for configuring the TPU profiler.
constexpr std::string_view kDeviceTracerLevel = "device_tracer_level";
constexpr std::string_view kHostTracerLevel = "host_tracer_level";
constexpr std::string_view kPythonTracerLevel = "python_tracer_level";
constexpr std::string_view kRunDir = "run_dir";

// Represents a parsed custom configuration option.
struct ProfilerOption {
  std::string_view name;
  std::string_view value;
};

// Helper to parse uint32_t and apply to a setter method of ProfileOptions.
absl::Status ParseAndSetUint(
    tensorflow::ProfileOptions& opts, std::string_view key,
    std::string_view val,
    void (tensorflow::ProfileOptions::*setter)(uint32_t)) {
  uint32_t uint_val;
  if (!absl::SimpleAtoi(val, &uint_val)) {
    return TT_ERROR(error::kInvalidArgument)
           << "expected the value of parameter " << key
           << " to be an integer, got '" << val << "'";
  }
  (opts.*setter)(uint_val);
  return absl::OkStatus();
}

// Updates a single key-value option in the ProfileOptions.
absl::Status UpdateProfileOption(tensorflow::ProfileOptions& opts,
                                 std::string_view key, std::string_view val) {
  if (key == kDeviceTracerLevel) {
    return ParseAndSetUint(
        opts, key, val, &tensorflow::ProfileOptions::set_device_tracer_level);
  }
  if (key == kHostTracerLevel) {
    return ParseAndSetUint(opts, key, val,
                           &tensorflow::ProfileOptions::set_host_tracer_level);
  }
  if (key == kPythonTracerLevel) {
    return ParseAndSetUint(
        opts, key, val, &tensorflow::ProfileOptions::set_python_tracer_level);
  }
  return TT_ERROR(error::kInvalidArgument)
         << "expected the profiler option to be one of " << kDeviceTracerLevel
         << ", " << kHostTracerLevel << ", " << kPythonTracerLevel << ", or "
         << kRunDir << ", got '" << key << "'";
}

// Splits a single custom config item (e.g., "key:value") by the first
// occurrence of ':'. Leading and trailing whitespace is stripped from both the
// key and value.
absl::StatusOr<ProfilerOption> SplitConfigItem(std::string_view item) {
  size_t colon_pos = item.find(':');
  if (colon_pos == std::string_view::npos) {
    return TT_ERROR(error::kInvalidArgument)
           << "expected the config item to be in the 'key:value' format, "
           << "got '" << item << "'";
  }
  std::string_view key = absl::StripAsciiWhitespace(item.substr(0, colon_pos));
  std::string_view val = absl::StripAsciiWhitespace(item.substr(colon_pos + 1));
  return ProfilerOption{key, val};
}

// Parses the custom configuration string from Kineto. Leading and trailing
// whitespace is ignored around keys and values. The configuration string is
// expected to be a comma-separated list of key-value pairs, where keys and
// values are separated by ':' (e.g., "host_tracer_level:3"). If a valid
// "run_dir" is found, it is written to out_run_dir; all other valid profiler
// options update the ProfileOptions object in place.
absl::Status UpdateProfileOptions(std::string_view custom_config,
                                  tensorflow::ProfileOptions& opts,
                                  std::string& out_run_dir) {
  for (std::string_view item :
       absl::StrSplit(custom_config, ',', absl::SkipEmpty())) {
    absl::StatusOr<ProfilerOption> profiler_option = SplitConfigItem(item);
    if (!profiler_option.ok()) {
      return profiler_option.status();
    }
    const auto& [name, val] = *profiler_option;

    if (name == kRunDir) {
      out_run_dir = std::string(val);
      continue;
    }
    TT_RETURN_IF_ERROR(UpdateProfileOption(opts, name, val));
  }
  return absl::OkStatus();
}

namespace {

// Resolves the base output directory for profiling artifacts.
std::string GetBaseOutputDir(std::string_view run_dir) {
  const auto& env_output_dir_opt = GetEnvOnce<kTpuProfilerOutputDirEnvVar>();
  if (env_output_dir_opt.has_value() && !env_output_dir_opt->empty()) {
    return *env_output_dir_opt;
  }

  std::string base_dir = std::string(run_dir);

  const auto& env_test_tmpdir_opt = GetEnvOnce<kTestTmpdirEnvVar>();
  const auto& env_tmpdir_opt = GetEnvOnce<kTmpdirEnvVar>();

  std::string env_tmp;
  if (env_test_tmpdir_opt.has_value() && !env_test_tmpdir_opt->empty()) {
    env_tmp = *env_test_tmpdir_opt;
  } else if (env_tmpdir_opt.has_value() && !env_tmpdir_opt->empty()) {
    env_tmp = *env_tmpdir_opt;
  }

  if (base_dir.empty() || base_dir == "/tmp") {
    return !env_tmp.empty() ? env_tmp : "/tmp";
  }
  return base_dir;
}

// Helper to determine the output path for the XPlane file.
// It follows the TensorBoard XProf convention used in TensorFlow/XLA to save
// data under <run_dir>/plugins/profile/<timestamp>/<hostname>.xplane.pb.
// The <timestamp> format is YYYY_MM_DD_HH_MM_SS in the local time zone,
// matching XLA's behavior.
//
// The base directory is determined in order of priority:
// 1. TPU_PROFILER_OUTPUT_DIR environment variable (if set and non-empty).
// 2. The provided `run_dir` (if not empty and not "/tmp").
// 3. TEST_TMPDIR environment variable (if set).
// 4. TMPDIR environment variable (if set).
// 5. Default to "/tmp".
absl::StatusOr<std::string> GetXPlaneOutputPath(std::string_view run_dir) {
  std::string base_dir = GetBaseOutputDir(run_dir);

  absl::Time now = absl::Now();
  // Match %E4Y format used in
  // tensorflow/compiler/xla/tsl/profiler/rpc/client/save_profile.cc
  std::string timestamp =
      absl::FormatTime("%E4Y_%m_%d_%H_%M_%S", now, absl::LocalTimeZone());

  char hostname[1024];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    snprintf(hostname, sizeof(hostname), "localhost");
  }
  hostname[sizeof(hostname) - 1] = '\0';

  std::string profile_dir =
      tsl::io::JoinPath(base_dir, "plugins", "profile", timestamp);

  tsl::Env* env = tsl::Env::Default();
  absl::Status status = env->RecursivelyCreateDir(profile_dir);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to create directory: " << profile_dir
                    << " error: " << status;
    return status;
  }

  std::string file_name = absl::StrCat(hostname, ".xplane.pb");
  // Windows file names do not support colons.
  absl::StrReplaceAll({{":", "_"}}, &file_name);

  return tsl::io::JoinPath(profile_dir, file_name);
}

}  // namespace

TpuKinetoProfilerSession::TpuKinetoProfilerSession(
    const libkineto::Config& config,
    std::set<libkineto::ActivityType> activity_types)
    : config_(config), activity_types_(std::move(activity_types)) {}

void TpuKinetoProfilerSession::start() {
  absl::MutexLock lock(mutex_);
  if (session_ != nullptr) {
    ABSL_LOG(WARNING) << "TpuKinetoProfilerSession already started";
    return;
  }
  tensorflow::ProfileOptions opts = tsl::ProfilerSession::DefaultOptions();
  opts.set_device_type(tensorflow::ProfileOptions::TPU);

  TT_THROW_IF_ERROR(
      UpdateProfileOptions(config_.getCustomConfig(), opts, run_dir_));

  session_ = tsl::ProfilerSession::Create(opts);
  if (session_ == nullptr) {
    ABSL_LOG(ERROR) << "Failed to create Tpu ProfilerSession!";
    errors_.push_back("Failed to create Tpu ProfilerSession");
    status_ = libkineto::TraceStatus::ERROR;
  } else {
    ABSL_LOG(INFO) << "Successfully created Tpu ProfilerSession!";
    status_ = libkineto::TraceStatus::RECORDING;
    callback_handler_ = std::make_unique<XProfCallbackHandler>();
  }
}

void TpuKinetoProfilerSession::stop() {
  absl::MutexLock lock(mutex_);
  if (session_ != nullptr) {
    callback_handler_.reset();
    status_ = libkineto::TraceStatus::PROCESSING;
    absl::Status status = session_->CollectData(&xspace_);
    ABSL_LOG(INFO) << "TpuKinetoProfilerSession::stop() CollectData status: "
                   << status;
    ABSL_LOG(INFO)
        << "TpuKinetoProfilerSession::stop() Collected XSpace planes: "
        << xspace_.planes_size();
    if (!status.ok()) {
      errors_.push_back("Failed to collect TPU profiling data: " +
                        status.ToString());
      status_ = libkineto::TraceStatus::ERROR;
      session_.reset();
      return;
    }
    status_ = libkineto::TraceStatus::READY;

    std::string run_dir =
        !run_dir_.empty()
            ? run_dir_
            : std::string(tsl::io::Dirname(config_.activitiesLogFile()));

    absl::StatusOr<std::string> resolved_path = GetXPlaneOutputPath(run_dir);
    if (!resolved_path.ok()) {
      errors_.push_back("Failed to get XPlane output path: " +
                        resolved_path.status().ToString());
      ABSL_LOG(ERROR) << "Failed to get XPlane output path: "
                      << resolved_path.status();
    } else {
      std::string output_path = *resolved_path;
      std::ofstream f(output_path, std::ios::binary);
      ABSL_VLOG(1) << "Attempting to write XPlane to " << output_path;
      if (!f) {
        errors_.push_back("Failed to open XSpace output file: " + output_path);
        ABSL_LOG(ERROR) << "Failed to open XSpace output file: " << output_path;
      } else {
        if (xspace_.SerializeToOstream(&f)) {
          ABSL_LOG(INFO) << "Successfully wrote XPlane to " << output_path;
        } else {
          errors_.push_back("Failed to write XSpace to file: " + output_path);
          ABSL_LOG(ERROR) << "Failed to write XSpace to file: " << output_path;
        }
        f.close();
      }
    }
  }
  session_.reset();
}

void TpuKinetoProfilerSession::processTrace(libkineto::ActivityLogger& logger) {
  // TODO(b/499240330): implement this.
}

namespace {

struct TpuProfilerRegisterer {
  TpuProfilerRegisterer() {
    if (torch::profiler::impl::PrivateUse1ProfilerRegistry::instance !=
        nullptr) {
      torch::profiler::impl::PrivateUse1ProfilerRegistry::instance()
          .registerFactory(
              []() -> std::unique_ptr<libkineto::IActivityProfiler> {
                return std::make_unique<TpuProfiler>();
              });
    } else {
      ABSL_LOG(INFO) << "PyTorch PrivateUse1ProfilerRegistry not found. "
                        "Native TPU profiling will be disabled (this is "
                        "expected on PyTorch < 2.12).";
    }
  }
};

static TpuProfilerRegisterer registerer;

}  // namespace

}  // namespace torch_tpu
