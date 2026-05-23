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

#include "ATen/core/TensorBody.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
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

void PySync(const std::vector<at::Tensor>& tensors, bool wait) {
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
    TT_THROW_IF_ERROR(SynchronizeTensors(tensors_span));
  } else {
    // GetMaterialized returns DeviceBufferRefs for the views, but we don't
    // return them to Python.
    TT_THROW_IF_ERROR(
        GetMaterialized(tensors_span, MaterializationReason::kExplicitSync));
  }
}

void PySyncAll(bool wait) {
  std::optional<py::gil_scoped_release> release;
  if (wait) {
    // See the comments in PySync for why we need to release the GIL.
    release.emplace();
  }
  TT_THROW_IF_ERROR(
      SynchronizeAll(wait ? WaitOnExecution::kYes : WaitOnExecution::kNo));
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

absl::StatusOr<std::vector<DeviceBufferRef>> GetComputationBuffers(
    const std::vector<at::Tensor>& tensors) {
  std::vector<DeviceBufferRef> buffer_refs;
  buffer_refs.reserve(tensors.size());
  for (const at::Tensor& tensor : tensors) {
    TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref, GetBaseBuffer(tensor));
    buffer_refs.push_back(buffer_ref);
  }
  return buffer_refs;
}

BufferRefToVarMap GetComputationGraph(py::dict capture_names_from) {
  absl::flat_hash_map<DeviceBufferRef, std::string> buffer_ref_to_var;
  for (auto item : capture_names_from) {
    if (py::isinstance<py::str>(item.first) &&
        THPVariable_Check(item.second.ptr())) {
      auto maybe_buffer_ref = GetBaseBuffer(item.second.cast<at::Tensor>());
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
  TT_ASSIGN_OR_THROW(auto refs, GetComputationBuffers(tensors));
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
  TT_ASSIGN_OR_THROW(auto refs, GetComputationBuffers(tensors));
  TT_ASSIGN_OR_THROW(auto res_str, GetComputationMlir(refs));
  return res_str;
}

}  // namespace

PYBIND11_MODULE(_tpu_torch_sync, m) {
  m.def("_synchronize_list", &PySync, py::arg("tensors"),
        py::arg("wait") = false,
        py::doc("Forces a materialization of a list of TPU tensors, optionally "
                "waiting for them to be ready."));
  m.def(
      "_synchronize_tensor",
      [](const at::Tensor& tensor, bool wait) { PySync({tensor}, wait); },
      py::arg("tensor"), py::arg("wait") = false,
      py::doc("Forces a materialization of single TPU tensor, optionally "
              "waiting for it to be ready."));
  m.def("_synchronize_all", &PySyncAll, py::arg("wait") = false,
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
}

}  // namespace torch_tpu
