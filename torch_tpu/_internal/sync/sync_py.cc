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
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "ATen/core/TensorBody.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type
#include "torch_tpu/_internal/sync/sync.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace torch_tpu {
namespace py = pybind11;

using BufferRefToVarMap = absl::flat_hash_map<DeviceBufferRef, std::string>;

namespace {

bool AllTensorsAreOnDevice(absl::Span<const at::Tensor> tensors) {
  for (const at::Tensor& tensor : tensors) {
    if (tensor.device().type() != GetPrivateUse1DeviceType()) {
      return false;
    }
  }
  return true;
}

void PySync(const std::vector<at::Tensor>& tensors, bool wait,
            const std::string& reason_str) {
  std::vector<at::Tensor> tpu_tensors;
  absl::Span<const at::Tensor> tensors_span = absl::MakeConstSpan(tensors);
  if (!AllTensorsAreOnDevice(tensors)) {
    for (const at::Tensor& tensor : tensors) {
      if (tensor.device().type() == GetPrivateUse1DeviceType()) {
        tpu_tensors.push_back(tensor);
      }
    }
    tensors_span = absl::MakeConstSpan(tpu_tensors);
  }

  MaterializationReason reason = MaterializationReason::kExplicitSync;
  if (reason_str == "autograd") {
    reason = MaterializationReason::kAutogradSync;
  }

  if (wait) {
    // To achieve non blocking H2D copies torch_tpu has to ensure the host data
    // is alive during the duration of the transfer.
    // To do this we hold onto the at::Tensor object backing the memory.
    // If `torch_tpu` holds the last reference to said at::Tensor it will need
    // access to the GIL in order to release the memory.

    // Deadlock Example:
    // Main/Python thread (executing this code)
    //   - blocks until tensors are ready
    //   - holds GIL
    // H2D thread
    //   - holds (last) reference to at::Tensor
    //   - performs H2D copy
    //   - attempts to acquire GIL to cleanup at::Tensor

    // H2D thread would wait forever for the GIL since the main thread would
    // hold onto it.
    // Thus we must release the GIL here before waiting for the H2D copy to
    // complete.

    py::gil_scoped_release release;
    // SynchronizeTensors is GetMaterialized + wait
    TT_THROW_IF_ERROR(SynchronizeTensors(tensors_span, reason));
  } else {
    // GetMaterialized returns DeviceBufferRefs for the views, but we don't
    // return them to Python.
    TT_THROW_IF_ERROR(GetMaterialized(tensors_span, reason));
  }
}

void PySyncAll(bool wait, const std::string& reason_str) {
  std::optional<py::gil_scoped_release> release;
  if (wait) {
    // See the comments in PySync for why we need to release the GIL.
    release.emplace();
  }
  MaterializationReason reason = MaterializationReason::kExplicitSync;
  if (reason_str == "autograd") {
    reason = MaterializationReason::kAutogradSync;
  }
  TT_THROW_IF_ERROR(SynchronizeAll(
      wait ? WaitOnExecution::kYes : WaitOnExecution::kNo, reason));
}

bool PyIsMaterialized(const at::Tensor& tensor) {
  TT_CHECK_THROW(tensor.device().type() == GetPrivateUse1DeviceType(),
                 error::kInvalidArgument)
      << "tensor is not on the PrivateUse1 device";
  TT_ASSIGN_OR_THROW(bool is_materialized, IsMaterialized(tensor));
  return is_materialized;
}

bool PyIsReady(const at::Tensor& tensor) {
  TT_CHECK_THROW(tensor.device().type() == GetPrivateUse1DeviceType(),
                 error::kInvalidArgument)
      << "tensor is not on the PrivateUse1 device";
  if (!PyIsMaterialized(tensor)) {
    return false;
  }
  TT_ASSIGN_OR_THROW(bool is_ready, IsReady(tensor));
  return is_ready;
}

std::vector<DeviceBufferRef> GetComputationBuffers(
    const std::vector<at::Tensor>& tensors) {
  std::vector<DeviceBufferRef> buffer_refs;
  buffer_refs.reserve(tensors.size());
  for (const at::Tensor& tensor : tensors) {
    TT_ASSIGN_OR_THROW(DeviceBufferRef buffer_ref,
                       GetBaseBufferFromAtTensor(tensor));
    buffer_refs.push_back(buffer_ref);
  }
  return buffer_refs;
}

BufferRefToVarMap GetComputationGraph(py::dict capture_names_from) {
  absl::flat_hash_map<DeviceBufferRef, std::string> buffer_ref_to_var;
  for (auto item : capture_names_from) {
    if (py::isinstance<py::str>(item.first) &&
        THPVariable_Check(item.second.ptr())) {
      auto maybe_buffer_ref =
          GetBaseBufferFromAtTensor(item.second.cast<at::Tensor>());
      if (maybe_buffer_ref.ok()) {
        buffer_ref_to_var[*maybe_buffer_ref] = item.first.cast<std::string>();
      }
    }
  }
  return buffer_ref_to_var;
}

// Returns a graphviz compatible representation of the computation graph of the
// given tensors. This might fail if the tensors are not on the PrivateUse1
// device.
std::string PyGetComputationGraphviz(const std::vector<at::Tensor>& tensors,
                                     py::dict capture_names_from) {
  ScopedPythonContextCapturer capturer(OpName::kCompileMlir);
  ScopedPythonContextProvider provider(
      ScopedPythonContextCapturer::GetContext());
  auto refs = GetComputationBuffers(tensors);
  auto refs_to_vars = GetComputationGraph(capture_names_from);
  TT_ASSIGN_OR_THROW(auto res_str, GetComputationGraphviz(refs, refs_to_vars));
  return res_str;
}

// Returns a MLIR representation of the computation graph of the given tensors.
// This might fail if the tensors are not on the PrivateUse1 device.
std::string PyGetComputationMlir(const std::vector<at::Tensor>& tensors) {
  ScopedPythonContextCapturer capturer(OpName::kCompileMlir);
  ScopedPythonContextProvider provider(
      ScopedPythonContextCapturer::GetContext());
  auto refs = GetComputationBuffers(tensors);
  TT_ASSIGN_OR_THROW(auto res_str, GetComputationMlir(refs));
  return res_str;
}

// Drains the structured-log event buffer and returns one dict per event.
// Each dict mirrors EagerEvent: name, ts_us, dur_us, device, cache_hit,
// aten_graph, stablehlo, reason.
py::list PyDrainStructuredLogBuffer() {
  auto events = StructuredLogBuffer::Get().Drain();
  py::list out;
  for (auto& ev : events) {
    py::dict d;
    d["name"] = ev.name;
    d["ts_us"] = ev.ts_us;
    d["dur_us"] = ev.dur_us;
    d["device"] = ev.device;
    d["cache_hit"] = ev.cache_hit;
    d["aten_graph"] = ev.aten_graph_payload;
    d["stablehlo"] = ev.mlir_payload;
    d["reason"] = std::string(MaterializationReasonToString(ev.reason));
    out.append(std::move(d));
  }
  return out;
}

bool PyStructuredLogEnabled() { return StructuredLogBuffer::Get().enabled(); }

}  // namespace

PYBIND11_MODULE(_tpu_torch_sync, m) {
  m.def("_synchronize_list", &PySync, py::arg("tensors"),
        py::arg("wait") = false, py::arg("reason") = std::string(""),
        py::doc("Forces materialization of a list of TPU tensors. "
                "`reason` is an optional tag (empty or 'autograd')."));
  m.def(
      "_synchronize_tensor",
      [](const at::Tensor& tensor, bool wait, const std::string& reason) {
        PySync({tensor}, wait, reason);
      },
      py::arg("tensor"), py::arg("wait") = false,
      py::arg("reason") = std::string(""),
      py::doc("Forces materialization of a single TPU tensor. "
              "`reason` is an optional tag (empty or 'autograd')."));
  m.def("_synchronize_all", &PySyncAll, py::arg("wait") = false,
        py::arg("reason") = std::string(""),
        py::doc("Forces a materialization of all TPU tensors, optionally "
                "waiting for them to be ready."));

  m.def("_is_materialized", &PyIsMaterialized, py::arg("tensor"),
        py::doc("Checks if the tensor has a materialized PjRtBuffer."));

  m.def(
      "_is_ready", &PyIsReady, py::arg("tensor"),
      py::doc("Checks if the tensor has completed execution and is ready to be "
              "copied to CPU."));

  m.def("_get_computation_graphviz", &PyGetComputationGraphviz,
        py::arg("tensors"), py::arg("capture_names_from"),
        py::doc("Returns a graphviz compatible representation of the "
                "computation graph of the given tensors."));

  m.def("_get_computation_mlir", &PyGetComputationMlir, py::arg("tensors"),
        py::doc("Returns a MLIR compatible representation of the "
                "computation graph of the given tensors."));

  m.def("_drain_structured_log_buffer", &PyDrainStructuredLogBuffer,
        py::doc("Drains buffered EagerEvent records and returns them as a "
                "list of dicts."));
  m.def("_structured_log_enabled", &PyStructuredLogEnabled,
        py::doc("Returns true if TORCH_TRACE was set at program startup."));
}

}  // namespace torch_tpu
