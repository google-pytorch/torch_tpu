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

#include "csrc/internal/profiler/tpu_profiler_plugin.h"

#include <kineto/ActivityType.h>
#include <kineto/Config.h>
#include <kineto/GenericTraceActivity.h>
#include <kineto/IActivityProfiler.h>
#include <kineto/output_base.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "c10/core/Device.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "csrc/common/device_type.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/profiler/xprof_callback_handler.h"
#include "torch/csrc/profiler/api.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xla/tsl/platform/env.h"

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

namespace torch_tpu {

// The set of valid parameter names for configuring the TPU profiler.
constexpr std::string_view kDeviceTracerLevel = "device_tracer_level";
constexpr std::string_view kHostTracerLevel = "host_tracer_level";
constexpr std::string_view kPythonTracerLevel = "python_tracer_level";
constexpr std::string_view kRunDir = "run_dir";
constexpr std::string_view kWorkerRank = "worker_rank";

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

// Unescapes a string value by removing backslashes that escape other
// characters.
//
// Scheme:
// - Any character preceded by a backslash `\` is copied literally, and the
//   backslash is discarded.
// - This allows preserving literal double quotes `"` and backslashes `\`
//   that were escaped in Python.
//
// Examples:
// - `a \"nested\" quote` -> `a "nested" quote`
// - `some\\path` -> `some\path`
// - `a\,b` -> `a,b`
// Unescapes a string option value by resolving escape sequences.
//
// Scheme:
// - Only `\"` (escaped double quote) and `\\` (escaped backslash) are supported
//   escape sequences. The backslash is discarded and the character is copied
//   literally.
// - Any other escape sequence is considered a bug and returns an error.
//
// Examples:
// - `a \"nested\" quote` -> `a "nested" quote`
// - `some\\path` -> `some\path`
absl::StatusOr<std::string> UnescapeString(std::string_view s) {
  std::string result;
  result.reserve(s.size());
  bool escaped = false;
  for (char c : s) {
    if (escaped) {
      if (c != '\\' && c != '"') {
        return TT_ERROR(error::kInvalidArgument)
               << "Invalid escape sequence: \\" << std::string(1, c)
               << " (only \\\\ and \\\" are supported)";
      }
      result.push_back(c);
      escaped = false;
    } else if (c == '\\') {
      escaped = true;
    } else {
      result.push_back(c);
    }
  }
  if (escaped) {
    return TT_ERROR(error::kInvalidArgument) << "Trailing backslash in string";
  }
  return result;
}

// Parses the value and sets it in the advanced_configuration map.
// Strings must be enclosed in double quotes. Unquoted values are parsed as
// booleans or integers.
absl::Status SetAdvancedConfigValue(tensorflow::ProfileOptions& opts,
                                    std::string_view key,
                                    std::string_view val) {
  tensorflow::ProfileOptions::AdvancedConfigValue adv_val;

  if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
    // Strip outer quotes and unescape internal characters.
    absl::StatusOr<std::string> unescaped_or =
        UnescapeString(val.substr(1, val.size() - 2));
    if (!unescaped_or.ok()) {
      return unescaped_or.status();
    }
    adv_val.set_string_value(*unescaped_or);
  } else if (val == "true") {
    adv_val.set_bool_value(true);
  } else if (val == "false") {
    adv_val.set_bool_value(false);
  } else {
    int64_t int_val;
    if (absl::SimpleAtoi(val, &int_val)) {
      adv_val.set_int64_value(int_val);
    } else {
      return TT_ERROR(error::kInvalidArgument)
             << "expected the advanced option '" << key
             << "' to be a quoted string, boolean (true/false), or integer, "
             << "got '" << val << "'";
    }
  }

  (*opts.mutable_advanced_configuration())[std::string(key)] = adv_val;
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
  return SetAdvancedConfigValue(opts, key, val);
}

// Helper to find the next comma or colon separator while ignoring delimiters
// that reside inside quotes, and respecting backslash escaping.
size_t FindNextSeparator(std::string_view custom_config, char sep,
                         size_t start = 0) {
  bool in_quotes = false;
  bool escaped = false;
  for (size_t i = start; i < custom_config.size(); ++i) {
    if (escaped) {
      escaped = false;
      continue;
    }
    if (custom_config[i] == '\\') {
      escaped = true;
    } else if (custom_config[i] == '"') {
      in_quotes = !in_quotes;
    } else if (custom_config[i] == sep && !in_quotes) {
      return i;
    }
  }
  return std::string_view::npos;
}

// Helper to split a string by an escape-aware separator, ignoring delimiters
// that reside inside quotes, and returning a list of string slices.
std::vector<std::string_view> SplitConfig(std::string_view custom_config,
                                          char sep) {
  std::vector<std::string_view> items;
  size_t start = 0;
  while (start < custom_config.size()) {
    size_t next = FindNextSeparator(custom_config, sep, start);
    std::string_view item = custom_config.substr(start, next - start);
    if (!item.empty()) {
      items.push_back(item);
    }
    if (next == std::string_view::npos) break;
    start = next + 1;
  }
  return items;
}

// Splits a single custom config item (e.g., "key:value") by the first
// occurrence of ':'. Leading and trailing whitespace is stripped from both the
// key and value.
absl::StatusOr<ProfilerOption> SplitConfigItem(std::string_view item) {
  size_t colon_pos = FindNextSeparator(item, ':');
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
                                  std::string& out_run_dir,
                                  std::optional<std::string>& out_worker_rank) {
  out_worker_rank = std::nullopt;
  if (!std::all_of(custom_config.begin(), custom_config.end(), [](char c) {
        return absl::ascii_isascii(static_cast<unsigned char>(c));
      })) {
    return TT_ERROR(error::kInvalidArgument)
           << "custom_config contains non-ASCII characters (this is a known "
              "limitation): "
           << custom_config;
  }
  for (std::string_view item : SplitConfig(custom_config, ',')) {
    absl::StatusOr<ProfilerOption> profiler_option = SplitConfigItem(item);
    if (!profiler_option.ok()) {
      return profiler_option.status();
    }
    const auto& [name, val] = *profiler_option;

    if (name == kRunDir) {
      out_run_dir = std::string(val);
      continue;
    }
    if (name == kWorkerRank) {
      if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
        // Strip outer quotes and unescape internal characters.
        absl::StatusOr<std::string> unescaped =
            UnescapeString(val.substr(1, val.size() - 2));
        if (!unescaped.ok()) {
          return unescaped.status();
        }
        out_worker_rank = *std::move(unescaped);
      } else {
        out_worker_rank = std::string(val);
      }
      continue;
    }
    TT_RETURN_IF_ERROR(UpdateProfileOption(opts, name, val));
  }
  return absl::OkStatus();
}

absl::Status UpdateProfileOptions(std::string_view custom_config,
                                  tensorflow::ProfileOptions& opts,
                                  std::string& out_run_dir) {
  std::optional<std::string> dummy_worker_rank;
  return UpdateProfileOptions(custom_config, opts, out_run_dir,
                              dummy_worker_rank);
}

// Resolves the base output directory for profiling artifacts.
std::string GetProfilerBaseOutputDir(std::string_view run_dir) {
  const auto& env_output_dir_opt_public =
      GetEnvOnce<kTorchTpuProfilerOutputDirEnvVar>();
  const auto& env_output_dir_opt =
      env_output_dir_opt_public.has_value()
          ? env_output_dir_opt_public
          : GetEnvOnce<kTpuProfilerOutputDirEnvVar>();
  if (env_output_dir_opt.has_value() && !env_output_dir_opt->empty()) {
    return *env_output_dir_opt;
  }

  std::string base_dir = std::string(run_dir);
  const auto& env_tmpdir_opt = GetEnvOnce<kTmpdirEnvVar>();
  std::string env_tmp;
  if (env_tmpdir_opt.has_value() && !env_tmpdir_opt->empty()) {
    env_tmp = *env_tmpdir_opt;
  }

  if (base_dir.empty() || base_dir == "/tmp") {
    return !env_tmp.empty() ? env_tmp : "/tmp";
  }
  return base_dir;
}

namespace {

// Helper to determine the output path for the XPlane file.
// It follows the TensorBoard XProf convention used in TensorFlow/XLA to save
// data under <run_dir>/plugins/profile/<timestamp>/<hostname>.xplane.pb.
// The <timestamp> format is YYYY_MM_DD_HH_MM_SS in the local time zone,
// matching XLA's behavior.
//
// The base directory is determined in order of priority:
// 1. TORCH_TPU_PROFILER_OUTPUT_DIR environment variable (if set and non-empty)
// 2. The provided `run_dir` (if not empty and not "/tmp").
// 3. TMPDIR environment variable (if set).
// 4. Default to "/tmp".
absl::StatusOr<std::string> GetXPlaneOutputPath(
    std::string_view run_dir, std::optional<std::string_view> worker_rank) {
  std::string base_dir = GetProfilerBaseOutputDir(run_dir);

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

  std::string file_name =
      worker_rank.has_value()
          ? absl::StrCat(hostname, "_", *worker_rank, ".xplane.pb")
          : absl::StrCat(hostname, ".xplane.pb");
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
  worker_rank_ = std::nullopt;
  tensorflow::ProfileOptions opts = tsl::ProfilerSession::DefaultOptions();
  opts.set_device_type(tensorflow::ProfileOptions::TPU);
  opts.set_raise_error_on_start_failure(true);

  bool with_stack = config_.isWithStackEnabled();
  if (!with_stack && torch::autograd::profiler::profilerEnabled()) {
    with_stack = torch::autograd::profiler::getProfilerConfig().with_stack;
  }

  if (with_stack) {
    opts.set_python_tracer_level(1);
  }

  TT_THROW_IF_ERROR(UpdateProfileOptions(config_.getCustomConfig(), opts,
                                         run_dir_, worker_rank_));

  start_time_ns_ = absl::GetCurrentTimeNanos();

  session_ = tsl::ProfilerSession::Create(opts);
  if (session_ == nullptr) {
    ABSL_LOG(ERROR) << "Failed to create Tpu ProfilerSession!";
    errors_.push_back("Failed to create Tpu ProfilerSession");
    status_ = libkineto::TraceStatus::ERROR;
  } else if (!session_->Status().ok()) {
    absl::Status status = session_->Status();
    std::string err = std::string(status.message());
    ABSL_LOG(ERROR) << "Tpu ProfilerSession start failed: " << err;
    errors_.push_back(err);
    status_ = libkineto::TraceStatus::ERROR;
    session_.reset();
    TT_THROW_IF_ERROR(status);
  } else {
    ABSL_LOG(INFO) << "Successfully created Tpu ProfilerSession!";
    status_ = libkineto::TraceStatus::RECORDING;
    XProfCallbackHandler::Register();
    callback_handler_ = std::make_unique<XProfCallbackHandler>();
  }
}

namespace {

// Synchronizes all TPU devices and deferred operations before stopping trace
// collection. This uses the TPU device guard to synchronize all TPU devices,
// which has the same behavior as using the Python `torch.tpu.synchronize()`
// API for all TPU devices.
//
// Synchronizing both layers before CollectData(&xspace_) ensures trace
// completeness for asynchronous operations without requiring pjrt_state.
absl::Status SynchronizeTpuDevicesBeforeStop() {
  ABSL_VLOG(1) << "Synchronizing TPU devices before stopping profiler.";

  if (c10::impl::hasDeviceGuardImpl(torch_tpu::GetPrivateUse1DeviceType())) {
    const auto* guard =
        c10::impl::getDeviceGuardImpl(torch_tpu::GetPrivateUse1DeviceType());
    int device_count = 0;
    if (guard != nullptr) {
      device_count = guard->deviceCount();
    }
    ABSL_VLOG(1) << "Synchronizing " << device_count
                 << " addressable TPU device stream queues.";
    for (int i = 0; i < device_count; ++i) {
      guard->synchronizeDevice(static_cast<c10::DeviceIndex>(i));
    }
  }
  return absl::OkStatus();
}

}  // namespace

void TpuKinetoProfilerSession::stop() {
  absl::MutexLock lock(mutex_);
  if (session_ != nullptr) {
    // Stop recording PyTorch frontend events immediately before synchronizing
    // and collecting trace data so teardown operations are not captured.
    callback_handler_.reset();

    absl::Status sync_status = SynchronizeTpuDevicesBeforeStop();
    if (!sync_status.ok()) {
      ABSL_LOG(WARNING)
          << "Failed to synchronize TPU operations before stopping profiler: "
          << sync_status << "; proceeding to collect available trace data.";
    }

    status_ = libkineto::TraceStatus::PROCESSING;
    absl::Status status = session_->CollectData(&xspace_);
    ABSL_LOG(INFO) << "TpuKinetoProfilerSession::stop() CollectData status: "
                   << status;
    ABSL_LOG(INFO)
        << "TpuKinetoProfilerSession::stop() Collected XSpace planes: "
        << xspace_.planes_size();
    if (!status.ok()) {
      errors_.push_back(absl::StrCat("Failed to collect TPU profiling data: ",
                                     status.ToString()));
      status_ = libkineto::TraceStatus::ERROR;
      session_.reset();
      return;
    }
    if (worker_rank_.has_value()) {
      for (std::string& hostname : *xspace_.mutable_hostnames()) {
        absl::StrAppend(&hostname, "_", *worker_rank_);
      }
    }
    status_ = libkineto::TraceStatus::READY;

    std::string run_dir =
        !run_dir_.empty()
            ? run_dir_
            : std::string(tsl::io::Dirname(config_.activitiesLogFile()));

    absl::StatusOr<std::string> resolved_path =
        GetXPlaneOutputPath(run_dir, worker_rank_);
    if (!resolved_path.ok()) {
      errors_.push_back(absl::StrCat("Failed to get XPlane output path: ",
                                     resolved_path.status().ToString()));
      ABSL_LOG(ERROR) << "Failed to get XPlane output path: "
                      << resolved_path.status();
    } else {
      std::string output_path = *resolved_path;
      ABSL_VLOG(1) << "Attempting to write XPlane to " << output_path;
      absl::Status s =
          tsl::WriteBinaryProto(tsl::Env::Default(), output_path, xspace_);
      if (!s.ok()) {
        errors_.push_back(
            absl::StrCat("Failed to write XSpace to file: ", output_path));
        ABSL_LOG(ERROR) << "Failed to write XSpace to file: " << output_path
                        << ". Error: " << s;
      } else {
        ABSL_LOG(INFO) << "Successfully wrote XPlane to " << output_path;
      }
    }
  }
  session_.reset();
}

namespace {
std::optional<uint64_t> GetXProfSessionStartTimeNs(
    const tensorflow::profiler::XSpace& space) {
  for (const auto& plane : space.planes()) {
    if (plane.name() == "Task Environment") {
      int64_t start_time_metadata_id = -1;
      for (const auto& [id, meta] : plane.stat_metadata()) {
        if (meta.name() == "profile_start_time") {
          start_time_metadata_id = id;
          break;
        }
      }
      if (start_time_metadata_id != -1) {
        for (const auto& stat : plane.stats()) {
          if (stat.metadata_id() == start_time_metadata_id) {
            if (stat.value_case() ==
                tensorflow::profiler::XStat::kUint64Value) {
              return stat.uint64_value();
            } else if (stat.value_case() ==
                       tensorflow::profiler::XStat::kInt64Value) {
              return stat.int64_value();
            }
          }
        }
      }
    }
  }
  return std::nullopt;
}
}  // namespace

std::string SanitizeStatString(std::string_view s) {
  std::string result(s);
  std::replace(result.begin(), result.end(), '"', '\'');
  std::erase(result, '\n');
  std::erase(result, '\r');
  return result;
}

void AddActivityStringMetadata(libkineto::GenericTraceActivity& activity,
                               const std::string& key, std::string_view value) {
  activity.addMetadataQuoted(key, SanitizeStatString(value));
}

void TpuKinetoProfilerSession::processTrace(libkineto::ActivityLogger& logger) {
  auto start_time_ns = GetXProfSessionStartTimeNs(xspace_);
  for (auto& plane : *xspace_.mutable_planes()) {
    for (auto& line : *plane.mutable_lines()) {
      if (start_time_ns.has_value()) {
        int64_t line_start_time_ns =
            start_time_ns.value() + line.timestamp_ns();
        line.set_timestamp_ns(line_start_time_ns);
      }
    }
  }

  constexpr int64_t kMaxEvents = 1000000;
  int64_t processed_events = 0;

  for (auto& plane : *xspace_.mutable_planes()) {
    if (processed_events >= kMaxEvents) {
      break;
    }

    bool is_device =
        absl::StartsWith(plane.name(), "/device:") || plane.name() == "TPU";

    // For device planes, use the XPlane ID as the device ID.
    // For host planes, map them directly to the main process PID so they
    // visually merge into the PyTorch CPU process span in Chrome Tracing.
    int32_t device_id = is_device ? plane.id() : getpid();

    if (is_device) {
      libkineto::DeviceInfo device_info = {device_id, device_id,
                                           std::string(plane.name()),
                                           std::string(plane.name())};
      logger.handleDeviceInfo(device_info, 0);
    }

    libkineto::ActivityType type =
        is_device ? libkineto::ActivityType::CONCURRENT_KERNEL
                  : libkineto::ActivityType::CPU_OP;

    absl::flat_hash_map<int64_t, std::string_view> event_name_map;
    for (const auto& [id, metadata] : plane.event_metadata()) {
      std::string_view name = metadata.name();
      if (name.empty()) {
        name = metadata.display_name();
      }
      if (name.empty()) {
        name = "Unknown";
      }
      event_name_map[id] = name;
    }

    absl::flat_hash_map<int64_t, std::string_view> stat_name_map;
    for (const auto& [id, metadata] : plane.stat_metadata()) {
      stat_name_map[id] = metadata.name();
    }

    for (const auto& line : plane.lines()) {
      if (processed_events >= kMaxEvents) {
        break;
      }

      uint32_t thread_id = static_cast<uint32_t>(
          line.display_id() ? line.display_id() : line.id());

      libkineto::ResourceInfo resource_info = {thread_id, thread_id, device_id,
                                               std::string(line.name())};
      resource_infos_.push_back(resource_info);
      logger.handleResourceInfo(resource_info, 0);

      for (const auto& event : line.events()) {
        if (!event.has_offset_ps()) {
          continue;  // Skip stateless/aggregated events that don't have
                     // timeline
        }

        if (processed_events >= kMaxEvents) {
          ABSL_LOG(WARNING) << "Hit maximum event limit of " << kMaxEvents
                            << ", truncating trace parsing to avoid OOM.";
          break;
        }

        bool is_mirrored_kineto_event = false;
        std::vector<const tensorflow::profiler::XStat*> all_stats;

        auto meta_it = plane.event_metadata().find(event.metadata_id());
        size_t total_stats =
            event.stats().size() + (meta_it != plane.event_metadata().end()
                                        ? meta_it->second.stats().size()
                                        : 0);
        all_stats.reserve(total_stats);

        if (meta_it != plane.event_metadata().end()) {
          for (const auto& stat : meta_it->second.stats()) {
            all_stats.push_back(&stat);
          }
        }
        for (const auto& stat : event.stats()) {
          all_stats.push_back(&stat);
        }

        for (const auto* stat : all_stats) {
          auto it = stat_name_map.find(stat->metadata_id());
          if (it != stat_name_map.end() && it->second == "pt_correlation_id") {
            is_mirrored_kineto_event = true;
            break;
          }
        }

        if (!is_device && is_mirrored_kineto_event) {
          // Skip events that Kineto already natively captured on the host
          // (mirrored to XProf by xprof_callback_handler.cc) to avoid
          // duplicates.
          continue;
        }

        std::string_view activity_name = "Unknown";
        if (meta_it != plane.event_metadata().end()) {
          activity_name = meta_it->second.name();
          if (activity_name.empty()) {
            activity_name = meta_it->second.display_name();
          }
        }
        if (activity_name.empty()) activity_name = "Unknown";

        libkineto::GenericTraceActivity activity;
        activity.activityName = activity_name;
        activity.activityType = type;

        int64_t duration_ps = event.duration_ps();

        int64_t event_start_realtime_ns =
            line.timestamp_ns() + (event.offset_ps() / 1000);

        // Kineto PyTorch expects absolute time in NANOSECONDS.
        activity.startTime = event_start_realtime_ns;
        activity.endTime = event_start_realtime_ns + (duration_ps / 1000);

        activity.device = device_id;
        activity.resource = thread_id;
        activity.threadId = is_device ? 0 : thread_id;

        for (const auto* stat : all_stats) {
          std::string_view stat_name = "Unknown";
          if (auto it = stat_name_map.find(stat->metadata_id());
              it != stat_name_map.end()) {
            stat_name = it->second;
          }
          std::string stat_name_str(stat_name);

          switch (stat->value_case()) {
            case tensorflow::profiler::XStat::kStrValue:
              AddActivityStringMetadata(activity, stat_name_str,
                                        stat->str_value());
              break;
            case tensorflow::profiler::XStat::kInt64Value:
              activity.addMetadataQuoted(stat_name_str,
                                         std::to_string(stat->int64_value()));
              break;
            case tensorflow::profiler::XStat::kUint64Value:
              activity.addMetadataQuoted(stat_name_str,
                                         std::to_string(stat->uint64_value()));
              break;
            case tensorflow::profiler::XStat::kDoubleValue:
              activity.addMetadataQuoted(stat_name_str,
                                         std::to_string(stat->double_value()));
              break;
            case tensorflow::profiler::XStat::kBytesValue:
              // Kineto JSON exporter cannot handle raw binary data (invalid
              // UTF-8). Furthermore, Kineto has no visualizer for XLA HLO
              // graphs.
              activity.addMetadataQuoted(stat_name_str, "<binary_data>");
              break;
            case tensorflow::profiler::XStat::kRefValue: {
              if (auto ref_it = stat_name_map.find(stat->ref_value());
                  ref_it != stat_name_map.end()) {
                AddActivityStringMetadata(activity, stat_name_str,
                                          ref_it->second);
              }
              break;
            }
            default:
              break;
          }
        }

        logger.handleGenericActivity(activity);
        processed_events++;
      }
    }
    plane.Clear();  // Clear the proto after processing to reclaim memory
  }
  xspace_.Clear();
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
