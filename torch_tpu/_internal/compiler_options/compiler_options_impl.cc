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
#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/error_utils.h"

namespace torch_tpu {

namespace py = pybind11;

static void PushCompilerOptions(CompilerOptionOverrides options) {
  TT_THROW_IF_ERROR(PushCompilerOptionOverrides(std::move(options)));
}

PYBIND11_MODULE(compiler_options_impl, m) {
  // Pushes the XLA options onto the per-thread custom compiler options stack.
  m.def("push_compiler_options", &PushCompilerOptions, py::arg("options"));

  // Pops the XLA options from the per-thread custom compiler options stack.
  m.def("pop_compiler_options", []() { PopCompilerOptionOverrides(); });
}

}  // namespace torch_tpu
