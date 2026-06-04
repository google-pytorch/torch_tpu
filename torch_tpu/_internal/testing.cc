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
#include "torch_tpu/eager/events_queue.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/repeated_ops_heuristic.h"
#include "torch_tpu/experimental/eager/materialize_new.h"

namespace torch_tpu {
namespace py = pybind11;

namespace {

void ResetEagerState() {
  ResetRepeatedOpsHeuristicState();
  ResetNewMaterializationState();
  ClearEventsQueue();
}

}  // namespace

// Internal testing utilities.
PYBIND11_MODULE(testing, m) {
  // Forces DynamicDispatchOp() to fail with the given message for ops whose
  // base name matches `op_base_name`. If `op_base_name` is empty, no op is
  // forced to fail.
  //
  // This is NOT accumulative. If you call this multiple times, only the last
  // call will take effect.
  m.def("set_op_dispatch_failure", internal::SetOpDispatchFailure,  //
        py::arg("op_base_name"), py::arg("failure_message"));
  m.def("reset_eager_state", ResetEagerState,
        "Resets the eager mode maintained state.");
}

}  // namespace torch_tpu
