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

#include <exception>
#include <optional>
#include <string>
#include <string_view>

#include "ATen/core/ATen_fwd.h"
#include "absl/flags/flag.h"
#include "c10/util/Exception.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/pybind_error_utils.h"
#include "csrc/common/tpu_hardware.h"
#include "csrc/common/warnings.h"
#include "csrc/eager/device_gen_impl.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/python_context.h"
#include "csrc/pjrt/pjrt_state.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/pybind.h"  // IWYU pragma: keep, for at::Tensor mapping
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {
namespace py = pybind11;

namespace {

void ResetEagerState() {
  ResetRepeatedOpsHeuristicState();
  ClearAllStreams();
}

// Resets default device generators across both C++ and Python.
// Wipes the C++ DeviceGenerators singleton and clears the Python
// _DefaultGeneratorsProperty._cached_default_generators on
// TpuDeviceModule.default_generators so subsequent accesses re-initialize
// through C++.
void ResetDefaultDeviceGenerators() {
  PyResetDefaultDeviceGeneratorsForTesting();
  // Clear the lazily initialized default_generators cached tuple in Python.
  // The `_DefaultGeneratorsProperty` descriptor instance is stored in
  // `_device_module.__tt_api_stages__["default_generators"].value` (and bound
  // to `TpuDeviceModule.__dict__["default_generators"]` after first external
  // access).
  try {
    PyBindModule device_module =
        PyBindModule::Import("torch_tpu._internal.device._device_module");
    py::object prop =
        device_module.attr("__tt_api_stages__")["default_generators"].attr(
            "value");
    prop.attr("_cached_default_generators") = py::none();
  } catch (const std::exception&) {
    // Ignore Python errors if torch_tpu is not yet initialized.
  }
}

std::string_view PyGetMemoryKind(const at::Tensor& tensor) {
  TT_ASSIGN_OR_THROW(
      const DeviceBufferRef buffer,
      MaterializeAndReturn(tensor, MaterializationReason::kDebugMode));
  TT_ASSIGN_OR_THROW(const xla::PjRtBuffer* pjrt_buffer, buffer.AwaitBuffer());
  if (pjrt_buffer == nullptr || pjrt_buffer->memory_space() == nullptr) {
    return "unknown";
  }
  return pjrt_buffer->memory_space()->kind();
}

// Test fixtures used by pybind_error_utils_test to verify Python API error
// handling and error translation wrappers.
struct TestErrorClass {
  void ThrowTtErrorInMemberFunction() {
    TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Covered by pybind_error_utils_test.
        false, error::kInvalidArgument)
        << "class throwing invalid argument";
  }
};

void ThrowTtErrorInFreeFunction() {
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Covered by pybind_error_utils_test.
      false, error::kInvalidArgument)
      << "throwing invalid argument";
}

void ThrowTtErrorIndexError() {
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Covered by pybind_error_utils_test.
      false, error::kPythonIndexError)
      << "throwing index error";
}

// Test struct used by pybind_error_utils_test to verify correct forwarding of
// Python API binding.
struct TestMembersClass {
  int read_only_field = 100;
  int read_write_field = 42;
};

void EmitTorchWarn(const std::string& msg) {
  HANDLE_TH_ERRORS
  TORCH_WARN(msg);  // TORCH_WARN_OK=test_utility
  END_HANDLE_TH_ERRORS_PYBIND
}

void EmitTorchWarnOnce(const std::string& msg) {
  HANDLE_TH_ERRORS
  TORCH_WARN_ONCE(msg);  // TORCH_WARN_OK=test_utility
  END_HANDLE_TH_ERRORS_PYBIND
}

void EmitTorchWarnDeprecation(const std::string& msg) {
  HANDLE_TH_ERRORS
  TORCH_WARN_DEPRECATION(msg);  // TORCH_WARN_OK=test_utility
  END_HANDLE_TH_ERRORS_PYBIND
}

void EmitTtWarn(WarningCategory category, const std::string& msg) {
  HANDLE_TH_ERRORS
  TT_WARN(category, msg);
  END_HANDLE_TH_ERRORS_PYBIND
}

void EmitTtWarnOnce(WarningCategory category, const std::string& msg) {
  HANDLE_TH_ERRORS
  TT_WARN_ONCE(category, msg);
  END_HANDLE_TH_ERRORS_PYBIND
}

}  // namespace

// Internal testing utilities.
TT_PYBIND11_MODULE(testing, m) {
  // Python bindings for testing PyBindWrapWithErrorHandling.
  //
  // These bindings throw errors on different contexts, making sure we are
  // showing the correct prefix on error messages.
  //
  // See: torch_tpu/tests/pybind_error_utils_test.py
  m.def("throw_tterror_in_free_function", &ThrowTtErrorInFreeFunction);
  m.def("throw_tterror_index_error", &ThrowTtErrorIndexError);

  // Passes `call_guard<gil_scoped_release>()` as argumento to `def()`.
  // This is undefined behavior if we don't acquire the GIL before going through
  // PyTorch error handling, which calls into the Python C API.
  m.def("throw_tterror_with_call_guard", &ThrowTtErrorInFreeFunction,
        py::call_guard<py::gil_scoped_release>());

  PyBindClass<TestErrorClass>(m, "TestErrorClass")
      .def(py::init<>())
      .def("throw_tterror_in_member_function",
           [](TestErrorClass& self) { self.ThrowTtErrorInMemberFunction(); });

  PyBindClass<TestMembersClass>(m, "TestMembersClass")
      .def(py::init<>())
      .def_readonly("read_only_field", &TestMembersClass::read_only_field)
      .def_readwrite("read_write_field", &TestMembersClass::read_write_field);

  // Forces DynamicDispatchOp() to fail with the given message for ops whose
  // base name matches `op_base_name`. If `op_base_name` is empty, no op is
  // forced to fail.
  //
  // This is NOT accumulative. If you call this multiple times, only the last
  // call will take effect.
  m.def("set_op_dispatch_failure",
        internal::SetOpDispatchFailure,  //
        py::arg("op_base_name"), py::arg("failure_message"));
  m.def("reset_eager_state", ResetEagerState,
        "Resets the eager mode maintained state.");
  m.def("set_init_default_generator_failure",
        PySetInitDefaultGeneratorFailureForTesting, py::arg("failure_message"),
        "Forces InitDefaultGenerator to fail with the given message.");
  m.def("reset_default_device_generators", ResetDefaultDeviceGenerators,
        "Resets the default device generators state in both C++ and Python.");
  m.def("get_memory_kind", PyGetMemoryKind, py::arg("tensor"),
        "Returns the memory space kind of the given tensor's buffer.");
  m.def("clear_sticky_error", ClearStickyError,
        "Clears the global sticky error state.");
  m.def("has_sticky_error", HasStickyError,
        "Returns whether the global sticky error state is set.");
  m.def("emit_torch_warn", &EmitTorchWarn, py::arg("msg"),
        "Emits a TORCH_WARN warning from C++.");
  m.def("emit_torch_warn_once", &EmitTorchWarnOnce, py::arg("msg"),
        "Emits a TORCH_WARN_ONCE warning from C++.");
  m.def("emit_torch_warn_deprecation", &EmitTorchWarnDeprecation,
        py::arg("msg"), "Emits a TORCH_WARN_DEPRECATION warning from C++.");
  // Registers the `WarningCategory` type taken by the functions below.
  PyBindModule::Import("torch_tpu._internal.warnings_impl");
  m.def("emit_tt_warn", &EmitTtWarn, py::arg("category"), py::arg("msg"),
        "Emits a TT_WARN warning from C++.");
  m.def("emit_tt_warn_once", &EmitTtWarnOnce, py::arg("category"),
        py::arg("msg"), "Emits a TT_WARN_ONCE warning from C++.");
  m.def(
      "has_tpu_device_nodes", []() -> bool { return HasTpuDeviceNodes(); },
      "Returns whether a device node through which the TPU runtime could "
      "reach a TPU exists at the default paths.");
  m.def(
      "pjrt_init_attempted",
      []() -> bool { return PjrtBackend::GetInstance().InitAttempted(); },
      "Returns whether PjRt runtime initialization has been attempted.");
  m.def(
      "enable_mlir_tracebacks",
      [](std::optional<bool> enabled) {
        absl::SetFlag(&FLAGS_torch_tpu_internal_mlir_tracebacks, enabled);
      },
      py::arg("enabled"),
      "Sets FLAGS_torch_tpu_internal_mlir_tracebacks in C++ for testing.");
}

}  // namespace torch_tpu
