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

#include "csrc/common/pybind_error_utils.h"
#include "csrc/experimental/precompiled_kernels/precompiled_kernels.h"
#include "pybind11/pybind11.h"

namespace torch_tpu {

namespace py = pybind11;

TT_PYBIND11_MODULE(_tpu_torch_precompiled_kernels, m) {
  m.def("load_kernels", &LoadPrecompiledKernels,
        py::call_guard<py::gil_scoped_release>(),
        py::doc("Loads the precompiled kernels into the compilation cache."));
}

}  // namespace torch_tpu
