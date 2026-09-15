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
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "c10/util/python_stub.h"
#include "csrc/common/device_type.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/pybind_error_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/internal/sync/sync.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type

namespace torch_tpu {
namespace py = pybind11;

using BufferRefToVarMap = absl::flat_hash_map<DeviceBufferRef, std::string>;

namespace {

at::Tensor UnwrapTensor(py::handle obj) {
  if (THPVariable_CheckExact(obj.ptr())) {
    return THPVariable_Unpack(obj.ptr());
  }
  if (py::hasattr(obj, "_local_tensor")) {
    return UnwrapTensor(obj.attr("_local_tensor"));
  }
  if (py::hasattr(obj, "elem")) {
    return UnwrapTensor(obj.attr("elem"));
  }
  if (THPVariable_Check(obj.ptr())) {
    return THPVariable_Unpack(obj.ptr());
  }
  throw py::type_error("expected torch.Tensor");  // For pybind11.
}

bool AllTensorsAreDenseOnDevice(absl::Span<const at::Tensor> tensors) {
  for (const at::Tensor& tensor : tensors) {
    if (tensor.device().type() != GetPrivateUse1DeviceType() ||
        tensor.is_sparse()) {
      return false;
    }
  }
  return true;
}

void PySyncSpanNoWait(absl::Span<const at::Tensor> tensors_span) {
  absl::InlinedVector<DeviceBufferRef, 16> buffer_refs;
  buffer_refs.reserve(tensors_span.size());
  bool has_deferred_or_placeholder = false;
  for (const at::Tensor& tensor : tensors_span) {
    TT_ASSIGN_OR_THROW(DeviceBufferRef buffer_ref, GetBaseBuffer(tensor));
    const auto& node = buffer_ref.device_buffer_list();
    if (node->is_deferred() || node->is_placeholder()) {
      has_deferred_or_placeholder = true;
    }
    buffer_refs.push_back(std::move(buffer_ref));
  }

  if (!has_deferred_or_placeholder) {
    return;
  }

  // Why we must release the GIL:
  //   Materialize() may wait on or contend for internal mutexes with
  //   background threads (such as async H2D copy threads or compilation
  //   workers). Background H2D threads hold references to host at::Tensor
  //   objects to keep memory alive during transfer; if a background thread
  //   drops the last reference to a tensor with a Python custom deleter or
  //   PyObject wrapper, it must acquire the GIL to clean it up. Holding the
  //   GIL here while blocking/waiting on those threads causes an AB-BA
  //   deadlock.
  //
  // Why it is safe to release the GIL:
  //   1. All input at::Tensor objects have already been converted into pure
  //      C++ DeviceBufferRef handles (`buffer_refs`) above.
  //   2. Materialize() operates exclusively on C++ DeviceBufferRef handles
  //      and PJRT/MLIR runtime structures without accessing any PyObject* or
  //      Python interpreter state.
  //   3. If Materialize() fails and TT_THROW_IF_ERROR throws a C++ exception,
  //      RAII stack unwinding destroys `release` and re-acquires the GIL
  //      before pybind11 catches the exception and converts it to Python.
  py::gil_scoped_release release;
  // Even though this is "no wait", we still wait until the execution has
  // started, so that is_materializing() is true for all buffers. We just don't
  // wait until they reach the full is_materialized() state.
  TT_THROW_IF_ERROR(Materialize(buffer_refs,
                                MaterializationReason::kExplicitSync,
                                MaterializationMode::kSplitGraph,
                                MaterializationBlockMode::kBlockUntilStarted));
}

void PySyncSpan(absl::Span<const at::Tensor> tensors, bool wait) {
  absl::InlinedVector<at::Tensor, 16> tpu_tensors;
  absl::Span<const at::Tensor> tensors_span = tensors;
  if (!AllTensorsAreDenseOnDevice(tensors)) {
    for (const at::Tensor& tensor : tensors) {
      if (tensor.is_sparse()) {
        const at::Tensor indices = tensor._indices();
        const at::Tensor values = tensor._values();
        if (indices.device().type() == GetPrivateUse1DeviceType()) {
          tpu_tensors.push_back(indices);
        }
        if (values.device().type() == GetPrivateUse1DeviceType()) {
          tpu_tensors.push_back(values);
        }
      } else if (tensor.device().type() == GetPrivateUse1DeviceType()) {
        tpu_tensors.push_back(tensor);
      }
    }
    tensors_span = absl::MakeConstSpan(tpu_tensors);
  }

  if (wait) {
    // Release the GIL to prevent deadlocks with background H2D transfer threads
    // that may hold the last reference to a Python-backed at::Tensor and need
    // to acquire the GIL to destroy it while this thread waits for transfers
    // to complete. This is safe because SynchronizeTensors operates purely on
    // C++ tensor/buffer structures without invoking Python C APIs, and stack
    // unwinding re-acquires the GIL before pybind11 translates any C++
    // exception.
    py::gil_scoped_release release;
    TT_THROW_IF_ERROR(SynchronizeTensors(tensors_span));
  } else {
    PySyncSpanNoWait(tensors_span);
  }
}

void PySync(const std::vector<at::Tensor>& tensors, bool wait) {
  PySyncSpan(tensors, wait);
}

// Fast-path C++ synchronization for a single PyObject* representing a standard
// `torch.Tensor` (`THPVariable`).
//
// Performance Rationale:
//   - Returns `false` if `ptr` is a Python wrapper subclass (such as `DTensor`
//     with `_local_tensor` or `elem` attributes) or a sparse COO tensor,
//     falling back to the `UnwrapTensor` / `PySyncSpan` path.
//   - Unpacks the underlying `at::Tensor` via `THPVariable_Unpack` without
//     constructing a temporary `pybind11::object` or incrementing C++ shared
//     pointer reference counts.
//   - When `wait == true`, immediately releases the GIL before invoking
//     `SynchronizeTensors` so background H2D/execution threads or PJRT host
//     callbacks are never blocked by the main thread holding the GIL.
//   - When `wait == false`, checks `node->is_deferred() ||
//   node->is_placeholder()`
//     before releasing the GIL or calling `Materialize`, avoiding GIL release
//     overhead when the tensor is already materializing/materialized.
bool TrySyncSingleTensorFast(PyObject* ptr, bool wait) {
  if (!THPVariable_CheckExact(ptr)) {
    return false;
  }
  const at::Tensor& t = THPVariable_Unpack(ptr);
  if (t.is_sparse()) {
    return false;
  }
  if (t.device().type() != GetPrivateUse1DeviceType()) {
    return true;
  }
  TT_ASSIGN_OR_THROW(DeviceBufferRef buffer_ref, GetBaseBuffer(t));
  const auto& node = buffer_ref.device_buffer_list();
  if (wait) {
    // If the underlying DeviceBufferList is already synchronized, return
    // immediately while holding the GIL to avoid py::gil_scoped_release and
    // future-await overhead. Note that `is_synchronized() == true` implies
    // `is_materialized() == true` because `is_synchronized_` is only set after
    // `Synchronize()` (which awaits materialization) succeeds.
    if (node->is_synchronized()) {
      return true;
    }
    py::gil_scoped_release release;
    if (node->is_materialized()) {
      TT_THROW_IF_ERROR(node->Synchronize());
    } else {
      TT_THROW_IF_ERROR(SynchronizeTensors(absl::MakeConstSpan(&t, 1)));
    }
    return true;
  }
  if (node->is_deferred() || node->is_placeholder()) {
    py::gil_scoped_release release;
    // Even though this is "no wait", we still wait until the execution has
    // started, so that is_materializing() is true for the single buffer. We
    // just don't wait until they reach the full is_materialized() state.
    TT_THROW_IF_ERROR(Materialize(
        absl::MakeConstSpan(&buffer_ref, 1),
        MaterializationReason::kExplicitSync, MaterializationMode::kSplitGraph,
        MaterializationBlockMode::kBlockUntilStarted));
  }
  return true;
}

// Fast-path C++ synchronization for a contiguous `PyObject* const*` array
// extracted directly from a Python `list` (`PyList_GET_ITEM`) or `tuple`
// (`PyTuple_GET_ITEM`).
//
// Performance Rationale:
//   - Avoids pybind11 `std::vector<at::Tensor>` type-caster conversion, which
//     would otherwise heap-allocate a `std::vector` and construct/destroy N
//     copied `at::Tensor` objects (each performing atomic refcount increments
//     on `c10::TensorImpl`).
//   - Instead, stores raw `const at::Tensor*` pointers into a stack-allocated
//     `absl::InlinedVector<const at::Tensor*, 16>` pointing directly into the
//     lifetime-guaranteed Python sequence elements.
bool TrySyncTensorArrayFast(PyObject* const* items, Py_ssize_t size,
                            bool wait) {
  absl::InlinedVector<const at::Tensor*, 16> tpu_tensor_ptrs;
  tpu_tensor_ptrs.reserve(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject* item = items[i];
    if (!THPVariable_CheckExact(item)) {
      return false;
    }
    const at::Tensor& t = THPVariable_Unpack(item);
    if (t.is_sparse()) {
      return false;
    }
    if (t.device().type() == GetPrivateUse1DeviceType()) {
      tpu_tensor_ptrs.push_back(&t);
    }
  }
  if (tpu_tensor_ptrs.empty()) {
    return true;
  }
  if (wait) {
    // Check if all TPU tensors in the sequence are already synchronized before
    // releasing the GIL (`is_synchronized()` implies `is_materialized()`).
    // When synchronizing multiple outputs from a single compiled execution,
    // they share a single DeviceBufferList, so after the first synchronization
    // this check succeeds without GIL release or future waits.
    bool all_synchronized = true;
    for (const at::Tensor* tensor_ptr : tpu_tensor_ptrs) {
      TT_ASSIGN_OR_THROW(DeviceBufferRef buffer_ref,
                         GetBaseBuffer(*tensor_ptr));
      const auto& node = buffer_ref.device_buffer_list();
      if (!node->is_synchronized()) {
        all_synchronized = false;
        break;
      }
    }
    if (all_synchronized) {
      return true;
    }
    py::gil_scoped_release release;
    TT_THROW_IF_ERROR(SynchronizeTensors(tpu_tensor_ptrs));
    return true;
  }
  absl::InlinedVector<DeviceBufferRef, 16> buffer_refs;
  buffer_refs.reserve(tpu_tensor_ptrs.size());
  bool has_deferred_or_placeholder = false;
  for (const at::Tensor* tensor_ptr : tpu_tensor_ptrs) {
    TT_ASSIGN_OR_THROW(DeviceBufferRef buffer_ref, GetBaseBuffer(*tensor_ptr));
    const auto& node = buffer_ref.device_buffer_list();
    if (node->is_deferred() || node->is_placeholder()) {
      has_deferred_or_placeholder = true;
    }
    buffer_refs.push_back(std::move(buffer_ref));
  }
  if (has_deferred_or_placeholder) {
    py::gil_scoped_release release;
    // Even though this is "no wait", we still wait until the execution has
    // started, so that is_materializing() is true for all buffers. We just
    // don't wait until they reach the full is_materialized() state.
    TT_THROW_IF_ERROR(
        Materialize(buffer_refs, MaterializationReason::kExplicitSync,
                    MaterializationMode::kSplitGraph,
                    MaterializationBlockMode::kBlockUntilStarted));
  }
  return true;
}

// Synchronizes the entire active TPU device when `synchronize(None, wait=...)`
// is called from Python. Caches the `DeviceGuardImplInterface*` pointer in a
// function-local static to avoid repeated registry lookups.
void PySynchronizeDevice(bool wait) {
  static const c10::impl::DeviceGuardImplInterface* const impl =
      c10::impl::getDeviceGuardImpl(GetPrivateUse1DeviceType());
  if (impl == nullptr) {
    throw py::value_error("tpu deviceguardimpl not found");  // For pybind11.
  }
  const c10::DeviceIndex device_index = impl->getDevice().index();
  py::gil_scoped_release release;
  if (wait) {
    impl->synchronizeDevice(device_index);
  } else {
    // Wait until all streams on the device start executing but don't wait for
    // them to finish.
    // This is equivalent to using torch.tpu.Event().record() on all streams,
    // and then throwing away the events.
    // This ensures that all live nodes reach the is_materializing() state but
    // does not wait for them to complete (is_materialized()).
    auto stream_futures =
        MaterializeDevice(device_index, MaterializationReason::kExplicitSync);
    for (auto& event_future : stream_futures) {
      TT_THROW_IF_ERROR(event_future.Await());
    }
  }
}

void PySynchronizeSequence(PyObject* ptr, bool wait) {
  const Py_ssize_t size =
      PyList_CheckExact(ptr) ? PyList_GET_SIZE(ptr) : PyTuple_GET_SIZE(ptr);
  if (size == 0) {
    return;
  }
  auto* const* items = PyList_CheckExact(ptr) ? &PyList_GET_ITEM(ptr, 0)
                                              : &PyTuple_GET_ITEM(ptr, 0);
  if (size == 1) {
    if (!TrySyncSingleTensorFast(items[0], wait)) {
      const at::Tensor t = UnwrapTensor(items[0]);
      PySyncSpan(absl::MakeConstSpan(&t, 1), wait);
    }
    return;
  }
  if (TrySyncTensorArrayFast(items, size, wait)) {
    return;
  }
  absl::InlinedVector<at::Tensor, 16> unwrapped;
  unwrapped.reserve(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    unwrapped.push_back(UnwrapTensor(items[i]));
  }
  PySyncSpan(unwrapped, wait);
}

// Unified C++ entry point for `torch_tpu.sync.synchronize(tensors, wait)`.
//
// Dispatches directly in C++ based on the exact Python object type:
//   - `None`: global device synchronization (`PySynchronizeDevice`).
//   - `list` / `tuple` of size 1: single-tensor fast path
//   (`TrySyncSingleTensorFast`).
//   - `list` / `tuple` of size > 1: zero-copy pointer array fast path
//     (`TrySyncTensorArrayFast`).
//   - Single `torch.Tensor`: single-tensor fast path
//   (`TrySyncSingleTensorFast`).
//   - Fallback: unwraps Tensor subclasses (e.g., `DTensor`) via `UnwrapTensor`.
void PySynchronize(py::handle tensors, bool wait) {
  if (tensors.is_none()) {
    PySynchronizeDevice(wait);
    return;
  }
  auto* ptr = tensors.ptr();
  if (PyList_CheckExact(ptr) || PyTuple_CheckExact(ptr)) {
    PySynchronizeSequence(ptr, wait);
    return;
  }
  if (py::isinstance<py::list>(tensors)) {
    const py::list tensor_list = py::reinterpret_borrow<py::list>(tensors);
    absl::InlinedVector<at::Tensor, 16> unwrapped;
    unwrapped.reserve(tensor_list.size());
    for (py::handle item : tensor_list) {
      unwrapped.push_back(UnwrapTensor(item));
    }
    PySyncSpan(unwrapped, wait);
    return;
  }
  if (!TrySyncSingleTensorFast(ptr, wait)) {
    const at::Tensor t = UnwrapTensor(tensors);
    PySyncSpan(absl::MakeConstSpan(&t, 1), wait);
  }
}

bool PyIsMaterializing(py::handle tensor_obj) {
  const at::Tensor tensor = UnwrapTensor(tensor_obj);
  TT_CHECK_THROW(tensor.device().type() == GetPrivateUse1DeviceType(),
                 error::kInvalidArgument)
      << "tensor is not on the PrivateUse1 device";
  TT_ASSIGN_OR_THROW(const bool is_materializing, IsMaterializing(tensor));
  return is_materializing;
}

bool PyIsMaterialized(py::handle tensor_obj) {
  const at::Tensor tensor = UnwrapTensor(tensor_obj);
  TT_CHECK_THROW(tensor.device().type() == GetPrivateUse1DeviceType(),
                 error::kInvalidArgument)
      << "tensor is not on the PrivateUse1 device";
  TT_ASSIGN_OR_THROW(const bool is_materializing, IsMaterializing(tensor));
  if (!is_materializing) {
    return false;
  }
  TT_ASSIGN_OR_THROW(const bool is_materialized, IsMaterialized(tensor));
  return is_materialized;
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

TT_PYBIND11_MODULE(_tpu_torch_sync, m) {
  m.def("synchronize", &PySynchronize, py::arg("tensors") = py::none(),
        py::arg("wait") = false,
        py::doc("Forces a materialization of one or more TPU tensors."));
  m.def("_synchronize_list", &PySync, py::arg("tensors"),
        py::arg("wait") = false,
        py::doc("Forces a materialization of a list of TPU tensors, optionally "
                "waiting for them to be ready."));
  m.def(
      "_synchronize_tensor",
      [](const at::Tensor& tensor, bool wait) {
        PySyncSpan(absl::MakeConstSpan(&tensor, 1), wait);
      },
      py::arg("tensor"), py::arg("wait") = false,
      py::doc("Forces a materialization of single TPU tensor, optionally "
              "waiting for it to be ready."));

  m.def("is_materializing", &PyIsMaterializing, py::arg("tensor"),
        py::doc(
            "Checks if the tensor has started materializing to a PjRtBuffer."));
  m.def("_is_materializing", &PyIsMaterializing, py::arg("tensor"),
        py::doc(
            "Checks if the tensor has started materializing to a PjRtBuffer."));

  m.def(
      "is_materialized", &PyIsMaterialized, py::arg("tensor"),
      py::doc("Checks if the tensor has completed execution and is ready to be "
              "copied to CPU."));
  m.def(
      "_is_materialized", &PyIsMaterialized, py::arg("tensor"),
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
