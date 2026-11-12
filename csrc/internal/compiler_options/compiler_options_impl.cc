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

#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/pybind_error_utils.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace torch_tpu {

namespace py = pybind11;

static void PushCompilerOptions(CompilerOptionOverrides options) {
  TT_THROW_IF_ERROR(PushCompilerOptionOverrides(std::move(options)));
}

TT_PYBIND11_MODULE(compiler_options_impl, m) {
  // Pushes the XLA options onto the per-thread custom compiler options stack.
  m.def("push_compiler_options", &PushCompilerOptions, py::arg("options"));

  // Pops the XLA options from the per-thread custom compiler options stack.
  m.def("pop_compiler_options", []() { PopCompilerOptionOverrides(); });

  // Private snapshot used to preserve the submitting thread's custom compiler
  // options when compilation is offloaded to a shared worker thread.
  m.def("_get_compiler_options", &GetCompilerOptionOverrides,
        "Returns the merged custom compiler option overrides active on the "
        "calling thread, or an empty map if none are active.");
}

}  // namespace torch_tpu
