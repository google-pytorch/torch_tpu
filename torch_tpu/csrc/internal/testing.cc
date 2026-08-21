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

#include <string_view>

#include "ATen/core/ATen_fwd.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch/csrc/utils/pybind.h"  // IWYU pragma: keep, for at::Tensor mapping
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/pybind_error_utils.h"
#include "torch_tpu/csrc/eager/device_gen_impl.h"
#include "torch_tpu/csrc/eager/events_queue.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/repeated_ops_heuristic.h"
#include "torch_tpu/csrc/eager/structured_log_buffer.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {
namespace py = pybind11;

namespace {

void ResetEagerState() {
  ResetRepeatedOpsHeuristicState();
  ClearAllStreams();
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

}  // namespace

// Internal testing utilities.
PYBIND11_MODULE(testing, m) {
  auto mod_with_error_handling = PyBindWrapWithErrorHandling(m);

  // Python bindings for testing PyBindWrapWithErrorHandling.
  //
  // These bindings throw errors on different contexts, making sure we are
  // showing the correct prefix on error messages.
  //
  // See: torch_tpu/tests/pybind_error_utils_test.py
  mod_with_error_handling.def("throw_tterror_in_free_function",
                              &ThrowTtErrorInFreeFunction);
  mod_with_error_handling.def("throw_tterror_index_error",
                              &ThrowTtErrorIndexError);

  py::class_<TestErrorClass> py_test_class(m, "TestErrorClass");
  PyBindWrapWithErrorHandling(py_test_class)
      .def(py::init<>())
      .def("throw_tterror_in_member_function",
           [](TestErrorClass& self) { self.ThrowTtErrorInMemberFunction(); });

  // Forces DynamicDispatchOp() to fail with the given message for ops whose
  // base name matches `op_base_name`. If `op_base_name` is empty, no op is
  // forced to fail.
  //
  // This is NOT accumulative. If you call this multiple times, only the last
  // call will take effect.
  mod_with_error_handling.def("set_op_dispatch_failure",
                              internal::SetOpDispatchFailure,  //
                              py::arg("op_base_name"),
                              py::arg("failure_message"));
  mod_with_error_handling.def("reset_eager_state", ResetEagerState,
                              "Resets the eager mode maintained state.");
  mod_with_error_handling.def(
      "set_init_default_generator_failure",
      PySetInitDefaultGeneratorFailureForTesting, py::arg("failure_message"),
      "Forces InitDefaultGenerator to fail with the given message.");
  mod_with_error_handling.def(
      "reset_default_device_generators",
      PyResetDefaultDeviceGeneratorsForTesting,
      "Resets the default device generators singleton state.");
  mod_with_error_handling.def(
      "get_memory_kind", PyGetMemoryKind, py::arg("tensor"),
      "Returns the memory space kind of the given tensor's buffer.");
}

}  // namespace torch_tpu
