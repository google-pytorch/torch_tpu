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

#include <string>

#include "csrc/common/pybind_error_utils.h"
#include "csrc/common/warnings.h"
#include "pybind11/pybind11.h"

namespace torch_tpu {

// Exposes the TorchTPU warning categories of common/warnings.h to Python, so
// that Python and C++ warnings share the same message prefixes.
TT_PYBIND11_MODULE(warnings_impl, m) {
  pybind11::enum_<WarningCategory>(GetPyBindHandle(m), "WarningCategory")
      .value("TORCH_TPU", WarningCategory::kTorchTpu)
      .value("PERFORMANCE", WarningCategory::kPerformance)
      .value("UNOPTIMIZED_BUILD", WarningCategory::kUnoptimizedBuild)
      .value("TCMALLOC", WarningCategory::kTcMalloc)
      .value("CORRECTNESS", WarningCategory::kCorrectness)
      .value("API_STAGE", WarningCategory::kApiStage)
      .value("INTERNAL_API", WarningCategory::kInternalApi)
      .value("EXPERIMENTAL_API", WarningCategory::kExperimentalApi)
      .value("DEPRECATED_API", WarningCategory::kDeprecatedApi);
  m.def(
      "get_warning_prefix",
      [](WarningCategory category) {
        return std::string(GetWarningPrefix(category));
      },
      pybind11::arg("category"),
      "Returns the message prefix of warnings in the given category.");
  m.def("is_deprecation_warning", &IsDeprecationWarning,
        pybind11::arg("category"),
        "Returns whether warnings in the given category are emitted as "
        "`DeprecationWarning`s rather than `UserWarning`s.");
}

}  // namespace torch_tpu
