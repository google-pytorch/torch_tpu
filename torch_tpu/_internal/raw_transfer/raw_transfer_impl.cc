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

#include "torch_tpu/experimental/raw_transfer/raw_transfer.h"

#include <vector>

#include "torch/extension.h"  // IWYU pragma: keep
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace torch_tpu {

PYBIND11_MODULE(raw_transfer_impl, m) {
  py::class_<PjRtCopyFuture>(m, "PjRtCopyFuture")
      .def(py::init<>())
      .def("wait", &PjRtCopyFuture::Await, py::call_guard<py::gil_scoped_release>());


  m.def(
      "transfer_d2h_batch_async",
      [](const std::vector<at::Tensor>& src_arrs,
         const std::vector<at::Tensor>& dst_arrs,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim) {
        py::gil_scoped_release release;
        return transfer_d2h_batch_async(src_arrs, dst_arrs, src_offsets_major_dim,
                                        dst_offsets_major_dim, copy_sizes_major_dim);
      },
      py::arg("src_arrs"), py::arg("dst_arrs"),
      py::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});

  m.def(
      "transfer_h2d_batch_async",
      [](const std::vector<at::Tensor>& src_arrs,
         const std::vector<at::Tensor>& dst_arrs,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim) {
        py::gil_scoped_release release;
        return transfer_h2d_batch_async(src_arrs, dst_arrs, src_offsets_major_dim,
                                        dst_offsets_major_dim, copy_sizes_major_dim);
      },
      py::arg("src_arrs"), py::arg("dst_arrs"),
      py::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
  

  m.def(
      "transfer_d2h_batch",
      [](const std::vector<at::Tensor>& src_arrs,
         const std::vector<at::Tensor>& dst_arrs,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim) {
        py::gil_scoped_release release;
        transfer_d2h_batch(src_arrs, dst_arrs, src_offsets_major_dim,
                           dst_offsets_major_dim, copy_sizes_major_dim);
      },
      py::arg("src_arrs"), py::arg("dst_arrs"),
      py::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});

  m.def(
      "transfer_h2d_batch",
      [](const std::vector<at::Tensor>& src_arrs,
         const std::vector<at::Tensor>& dst_arrs,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim) {
        py::gil_scoped_release release;
        transfer_h2d_batch(src_arrs, dst_arrs, src_offsets_major_dim,
                           dst_offsets_major_dim, copy_sizes_major_dim);
      },
      py::arg("src_arrs"), py::arg("dst_arrs"),
      py::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});

  m.def("await_all", [](std::vector<PjRtCopyFuture>& futures) {
    py::gil_scoped_release release;
    for (auto& f : futures) {
      f.Await();
    }
  });
}

}  // namespace torch_tpu
