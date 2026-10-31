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
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

#include "absl/log/absl_log.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/libtpu_version.h"
#include "csrc/common/process.h"
#include "csrc/common/pybind_error_utils.h"
#include "csrc/common/utils.h"
#include "pybind11/stl.h"

namespace torch_tpu {
namespace {

// Returns the Boolean value of the environment variable `name`, or
// `default_value` if the variable is unset or cannot be parsed as a Boolean.
//
// GetBooleanEnvOnce() takes the variable name as a template parameter, so a
// name given at run time (from Python) has to be dispatched here. Only the
// variables that Python code reads are listed; add more as needed.
//
// REQUIRES: `name` is one of them.
std::optional<bool> PyGetBooleanEnvOnce(std::string_view name,
                                        std::optional<bool> default_value) {
  std::optional<bool> value;
  if (name == kTorchTpuAllowLegacyDeviceApisEnvVar) {
    value = GetBooleanEnvOnce<kTorchTpuAllowLegacyDeviceApisEnvVar>();
  } else if (name == kTorchTpuEnableDebugChecksEnvVar) {
    value = GetBooleanEnvOnce<kTorchTpuEnableDebugChecksEnvVar>();
  } else if (name == kTorchTpuInternalEnableReassociateNormWeightsEnvVar) {
    value = GetBooleanEnvOnce<
        kTorchTpuInternalEnableReassociateNormWeightsEnvVar>();
  } else if (name == kTorchTpuInternalMaterializeCollectiveTensorsEnvVar) {
    value = GetBooleanEnvOnce<
        kTorchTpuInternalMaterializeCollectiveTensorsEnvVar>();
  } else if (name == kTorchTpuInternalStorageAwarePlaceholdersEnvVar) {
    value =
        GetBooleanEnvOnce<kTorchTpuInternalStorageAwarePlaceholdersEnvVar>();
  } else {
    ABSL_LOG(FATAL)  // CRASH_OK=TorchTPU bug
        << "Before reading Boolean environment variable " << name
        << " from Python, it must be added to the body of "
           "PyGetBooleanEnvOnce().";
  }
  return value.has_value() ? value : default_value;
}

// Like PyGetBooleanEnvOnce(), but for integer environment variables.
std::optional<int64_t> PyGetIntegerEnvOnce(
    std::string_view name, std::optional<int64_t> default_value) {
  std::optional<int64_t> value;
  if (name == kLocalRankEnvVar) {
    value = GetIntegerEnvOnce<int64_t, kLocalRankEnvVar>();
  } else if (name == kRankEnvVar) {
    value = GetIntegerEnvOnce<int64_t, kRankEnvVar>();
  } else if (name == kTorchTpuBasePortEnvVar) {
    value = GetIntegerEnvOnce<int64_t, kTorchTpuBasePortEnvVar>();
  } else if (name == kTorchTpuDevicesPerHostEnvVar) {
    value = GetIntegerEnvOnce<int64_t, kTorchTpuDevicesPerHostEnvVar>();
  } else if (name == kTorchTpuHandshakePortEnvVar) {
    value = GetIntegerEnvOnce<int64_t, kTorchTpuHandshakePortEnvVar>();
  } else if (name == kTorchTpuInternalHandshakeTimeoutSEnvVar) {
    value =
        GetIntegerEnvOnce<int64_t, kTorchTpuInternalHandshakeTimeoutSEnvVar>();
  } else if (name == kTorchcommRankEnvVar) {
    value = GetIntegerEnvOnce<int64_t, kTorchcommRankEnvVar>();
  } else if (name == kTorchcommSizeEnvVar) {
    value = GetIntegerEnvOnce<int64_t, kTorchcommSizeEnvVar>();
  } else if (name == kWorldSizeEnvVar) {
    value = GetIntegerEnvOnce<int64_t, kWorldSizeEnvVar>();
  } else {
    ABSL_LOG(FATAL)  // CRASH_OK=TorchTPU bug
        << "Before reading integer environment variable " << name
        << " from Python, it must be added to the body of "
           "PyGetIntegerEnvOnce().";
  }
  return value.has_value() ? value : default_value;
}

// Like PyGetBooleanEnvOnce(), but for string environment variables. Unlike the
// Boolean and integer variants, the value needs no parsing, so it is returned
// as is, including when it is the empty string.
std::optional<std::string> PyGetStringEnvOnce(
    std::string_view name, std::optional<std::string> default_value) {
  std::optional<std::string> value;
  if (name == kJobCompletionIndexEnvVar) {
    value = GetEnvOnce<kJobCompletionIndexEnvVar>();
  } else if (name == kMasterAddrEnvVar) {
    value = GetEnvOnce<kMasterAddrEnvVar>();
  } else if (name == kMasterPortEnvVar) {
    value = GetEnvOnce<kMasterPortEnvVar>();
  } else if (name == kNnodesEnvVar) {
    value = GetEnvOnce<kNnodesEnvVar>();
  } else if (name == kNodeRankEnvVar) {
    value = GetEnvOnce<kNodeRankEnvVar>();
  } else if (name == kTorchTpuSlicebuilderAddressesEnvVar) {
    value = GetEnvOnce<kTorchTpuSlicebuilderAddressesEnvVar>();
  } else if (name == kTorchTpuTopologyEnvVar) {
    value = GetEnvOnce<kTorchTpuTopologyEnvVar>();
  } else if (name == kTorchTpuXprofSessionIdEnvVar) {
    value = GetEnvOnce<kTorchTpuXprofSessionIdEnvVar>();
  } else if (name == kTpuWorkerHostnamesEnvVar) {
    value = GetEnvOnce<kTpuWorkerHostnamesEnvVar>();
  } else {
    ABSL_LOG(FATAL)  // CRASH_OK=TorchTPU bug
        << "Before reading string environment variable " << name
        << " from Python, it must be added to the body of "
           "PyGetStringEnvOnce().";
  }
  return value.has_value() ? value : default_value;
}

}  // namespace

TT_PYBIND11_MODULE(env, m) {
  // Resolve the process lineage at module import time and mirror the resulting
  // TORCH_TPU_INTERNAL_TOPMOST_PID into Python's os.environ, so that child
  // processes spawned from Python (e.g. via os.environ.copy()) inherit it.
  static_cast<void>(  // VOID_CAST_OK=only called for its side effect.
      HasTorchTpuAncestorProcess());
  if (const char* const topmost_pid =
          std::getenv(  // GETENV_OK=Internal implementation env var.
              kTorchTpuInternalTopmostPidEnvVar);
      topmost_pid != nullptr) {
    PyBindModule::Import("os").attr(
        "environ")[kTorchTpuInternalTopmostPidEnvVar] = topmost_pid;
  }
  m.attr("IS_INTERNAL_TORCH_TPU") = static_cast<bool>(TT_IS_INTERNAL_TORCH_TPU);
#if defined(NDEBUG)
  m.attr("TORCH_TPU_IS_OPTIMIZED_BUILD") = true;
#else
  m.attr("TORCH_TPU_IS_OPTIMIZED_BUILD") = false;
#endif
  m.def("has_torch_tpu_ancestor_process", &HasTorchTpuAncestorProcess,
        "Returns True if the current process inherited the PID of an "
        "ancestor TorchTPU process. "
        "Best-effort: launchers that don't load TorchTPU are invisible, so "
        "use it only where a wrong answer is harmless (e.g. log "
        "deduplication). See csrc/common/process.h.");
  m.def("set_libtpu_version", &SetLibtpuVersion, pybind11::arg("version"));
  m.def("get_libtpu_version", &GetLibtpuVersionString);
  m.def("reset_libtpu_version_for_testing", &ResetLibtpuVersionForTesting);
  m.def("get_enable_debug_checks", &GetEnableDebugChecks);
  m.def("get_bool_env_once", &PyGetBooleanEnvOnce, pybind11::arg("name"),
        pybind11::arg("default_value") = std::optional<bool>(),
        "Returns the Boolean value of the given environment variable, or "
        "`default_value` if it is unset or cannot be parsed as a Boolean.");
  m.def("get_int_env_once", &PyGetIntegerEnvOnce, pybind11::arg("name"),
        pybind11::arg("default_value") = std::optional<int64_t>(),
        "Returns the integer value of the given environment variable, or "
        "`default_value` if it is unset or cannot be parsed as an integer.");
  m.def("get_str_env_once", &PyGetStringEnvOnce, pybind11::arg("name"),
        pybind11::arg("default_value") = std::optional<std::string>(),
        "Returns the value of the given environment variable, or "
        "`default_value` if it is unset.");
}

}  // namespace torch_tpu
