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

#include <cstdint>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "pybind11/attr.h"
#include "pybind11/cast.h"
#include "pybind11/gil.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch/extension.h"  // IWYU pragma: keep
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/experimental/batch_transfer/batch_transfer.h"

namespace torch_tpu {

namespace py = pybind11;

static void PyBatchTransferD2HSync(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,   // INT_VEC_OK
    const std::vector<int64_t>& dst_offsets_major_dim,   // INT_VEC_OK
    const std::vector<int64_t>& copy_sizes_major_dim) {  // INT_VEC_OK
  py::gil_scoped_release release;
  Dimensions src_offsets(src_offsets_major_dim.begin(),
                         src_offsets_major_dim.end());
  Dimensions dst_offsets(dst_offsets_major_dim.begin(),
                         dst_offsets_major_dim.end());
  Dimensions copy_sizes(copy_sizes_major_dim.begin(),
                        copy_sizes_major_dim.end());
  TT_THROW_IF_ERROR(BatchTransferD2HSync(src_arrs, dst_arrs, src_offsets,
                                         dst_offsets, copy_sizes));
}

static void PyBatchTransferH2DSync(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,   // INT_VEC_OK
    const std::vector<int64_t>& dst_offsets_major_dim,   // INT_VEC_OK
    const std::vector<int64_t>& copy_sizes_major_dim) {  // INT_VEC_OK
  py::gil_scoped_release release;
  Dimensions src_offsets(src_offsets_major_dim.begin(),
                         src_offsets_major_dim.end());
  Dimensions dst_offsets(dst_offsets_major_dim.begin(),
                         dst_offsets_major_dim.end());
  Dimensions copy_sizes(copy_sizes_major_dim.begin(),
                        copy_sizes_major_dim.end());
  TT_THROW_IF_ERROR(BatchTransferH2DSync(src_arrs, dst_arrs, src_offsets,
                                         dst_offsets, copy_sizes));
}

static TransferFuture PyBatchTransferD2H(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,   // INT_VEC_OK
    const std::vector<int64_t>& dst_offsets_major_dim,   // INT_VEC_OK
    const std::vector<int64_t>& copy_sizes_major_dim) {  // INT_VEC_OK
  py::gil_scoped_release release;
  Dimensions src_offsets(src_offsets_major_dim.begin(),
                         src_offsets_major_dim.end());
  Dimensions dst_offsets(dst_offsets_major_dim.begin(),
                         dst_offsets_major_dim.end());
  Dimensions copy_sizes(copy_sizes_major_dim.begin(),
                        copy_sizes_major_dim.end());
  TT_ASSIGN_OR_THROW(auto result,
                     BatchTransferD2H(src_arrs, dst_arrs, src_offsets,
                                      dst_offsets, copy_sizes));
  return result;
}

static TransferFuture PyBatchTransferH2D(
    const std::vector<at::Tensor>& src_arrs,
    const std::vector<at::Tensor>& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,   // INT_VEC_OK
    const std::vector<int64_t>& dst_offsets_major_dim,   // INT_VEC_OK
    const std::vector<int64_t>& copy_sizes_major_dim) {  // INT_VEC_OK
  py::gil_scoped_release release;
  Dimensions src_offsets(src_offsets_major_dim.begin(),
                         src_offsets_major_dim.end());
  Dimensions dst_offsets(dst_offsets_major_dim.begin(),
                         dst_offsets_major_dim.end());
  Dimensions copy_sizes(copy_sizes_major_dim.begin(),
                        copy_sizes_major_dim.end());
  TT_ASSIGN_OR_THROW(auto result,
                     BatchTransferH2D(src_arrs, dst_arrs, src_offsets,
                                      dst_offsets, copy_sizes));
  return result;
}

static void PyAwaitAll(std::vector<TransferFuture>& futures) {
  py::gil_scoped_release release;
  for (auto& f : futures) {
    f.Await();
  }
}

PYBIND11_MODULE(batch_transfer_impl, m) {
  py::class_<TransferFuture>(m, "TransferFuture")
      .def(py::init<>())
      .def("wait", &TransferFuture::Await,
           py::call_guard<py::gil_scoped_release>());

  m.def(
      "batch_transfer_d2h", &PyBatchTransferD2H,  //
      py::arg("src_arrs"),                        //
      py::arg("dst_arrs"),
      py::arg("src_offsets_major_dim") = std::vector<int64_t>{},  // INT_VEC_OK
      py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},  // INT_VEC_OK
      py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});  // INT_VEC_OK

  m.def(
      "batch_transfer_h2d", &PyBatchTransferH2D,  //
      py::arg("src_arrs"),                        //
      py::arg("dst_arrs"),
      py::arg("src_offsets_major_dim") = std::vector<int64_t>{},  // INT_VEC_OK
      py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},  // INT_VEC_OK
      py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});  // INT_VEC_OK

  m.def(
      "batch_transfer_d2h_sync", &PyBatchTransferD2HSync,  //
      py::arg("src_arrs"),                                 //
      py::arg("dst_arrs"),
      py::arg("src_offsets_major_dim") = std::vector<int64_t>{},  // INT_VEC_OK
      py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},  // INT_VEC_OK
      py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});  // INT_VEC_OK

  m.def(
      "batch_transfer_h2d_sync", &PyBatchTransferH2DSync,  //
      py::arg("src_arrs"),                                 //
      py::arg("dst_arrs"),
      py::arg("src_offsets_major_dim") = std::vector<int64_t>{},  // INT_VEC_OK
      py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},  // INT_VEC_OK
      py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});  // INT_VEC_OK

  m.def("await_all", &PyAwaitAll);
}

}  // namespace torch_tpu
