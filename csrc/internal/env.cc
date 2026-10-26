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
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/libtpu_version.h"
#include "csrc/common/pybind_error_utils.h"
#include "csrc/common/utils.h"
#include "csrc/common/version.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace torch_tpu {
namespace {

template <typename T>
absl::StatusOr<T> CastVersionComponent(const pybind11::handle& obj,
                                       std::string_view version) {
  T value = 0;
  bool fits = false;
  try {
    value = obj.cast<T>();
    fits = (value >= 0);
  } catch (const pybind11::builtin_exception&) {
    fits = false;
  } catch (const pybind11::error_already_set&) {
    fits = false;
  }
  TT_RET_CHECK(fits, error::kInvalidArgument)
      << "version component '" << pybind11::str(obj).cast<std::string>()
      << "' overflows in '" << version << "'";
  return value;
}

// Parses a PEP 440 version string via Python's `packaging.version.Version` into
// a `torch_tpu::Version`.
absl::StatusOr<Version> ParsePep440Version(std::string_view version) {
  pybind11::object parsed;
  pybind11::tuple release;
  bool valid_pep440 = false;
  try {
    parsed = PyBindModule::Import("packaging.version").attr("Version")(version);
    release = parsed.attr("release").cast<pybind11::tuple>();
    valid_pep440 =
        (parsed.attr("epoch").cast<int>() == 0 && release.size() <= 4);
  } catch (const pybind11::error_already_set&) {
    valid_pep440 = false;
  }
  TT_RET_CHECK(valid_pep440, error::kInvalidArgument)
      << "expected a valid PEP 440 version string with at most 4 release "
         "components, got '"
      << version << "'";

  std::array<int, 4> parts = {0, 0, 0, 0};
  for (size_t i = 0; i < release.size(); ++i) {
    TT_ASSIGN_OR_RETURN(parts[i],
                        CastVersionComponent<int>(release[i], version));
  }

  int64_t dev = Version::kStable;
  pybind11::object dev_obj = parsed.attr("dev");
  if (!dev_obj.is_none()) {
    TT_ASSIGN_OR_RETURN(dev, CastVersionComponent<int64_t>(dev_obj, version));
  }
  return Version(parts[0], parts[1], parts[2], parts[3], dev);
}

void PySetLibtpuVersion(std::string_view version) {
  if (version.empty()) {
    SetLibtpuVersion(version, Version::Max());
    return;
  }
  TT_ASSIGN_OR_THROW(const Version parsed_version, ParsePep440Version(version));
  SetLibtpuVersion(version, parsed_version);
}

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
  m.attr("IS_INTERNAL_TORCH_TPU") = static_cast<bool>(TT_IS_INTERNAL_TORCH_TPU);
#if defined(NDEBUG)
  m.attr("TORCH_TPU_IS_OPTIMIZED_BUILD") = true;
#else
  m.attr("TORCH_TPU_IS_OPTIMIZED_BUILD") = false;
#endif
  m.def("set_libtpu_version", &PySetLibtpuVersion, pybind11::arg("version"));
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
