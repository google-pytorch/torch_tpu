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
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/tpu_aten_kernels.h"

namespace torch_tpu {

namespace py = pybind11;

PYBIND11_MODULE(execution_mode_impl, m) {
  py::enum_<EagerMode>(m, "EagerMode")
      .value("DEFER_AND_FUSE", EagerMode::kDeferAndFuse)
      .value("DEFER_NEVER", EagerMode::kDeferNever)
      .value("DEFER_NEVER_AND_LAUNCH_BLOCKING",
             EagerMode::kDeferNeverAndLaunchBlocking)
      .value("INTERNAL_DEFER_ALL", EagerMode::kInternalDeferAll)
      .export_values();

  m.def("_get_eager_mode", GetEagerMode);
  m.def("_set_eager_mode", SetEagerMode, py::arg("eager_mode"));
  m.def("_push_eager_mode", &PushContextState<EagerModeContextState>,
        "Internal push for `execution_eager_mode` context manager.");
  m.def("_pop_eager_mode", &PopContextState<EagerModeContextState>,
        "Internal pop for `execution_eager_mode` context manager.");

  m.def("_enable_cpu_fallback", EnableCpuFallback, py::arg("enabled"));
  m.def("_is_cpu_fallback_enabled", IsCpuFallbackEnabled);
}

}  // namespace torch_tpu
