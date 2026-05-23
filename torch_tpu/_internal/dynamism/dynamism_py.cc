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

#include <cstdint>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type
#include "torch_tpu/_internal/dynamism/dynamism.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_types.h"

namespace torch_tpu {
namespace py = pybind11;

namespace {

at::Tensor PyMarkDynamic(const at::Tensor& tensor, int64_t dimension,
                         int64_t lower_bound, int64_t upper_bound) {
  TT_CHECK_THROW(tensor.device().type() == GetPrivateUse1DeviceType(),
                 error::kInvalidArgument)
      << "tensor is not on the PrivateUse1 device";
  TT_ASSIGN_OR_THROW(auto base_or_view_tensor,
                     // if tensor is a base, then return value is the same as
                     // the input tensor, otherwise returns a new view tensor.
                     MarkDynamic(tensor, dimension, lower_bound, upper_bound));
  return base_or_view_tensor;
}

std::vector<BoundedDynamicDimension> PyGetDynamismInfo(
    const at::Tensor& tensor) {
  TT_CHECK_THROW(tensor.device().type() == GetPrivateUse1DeviceType(),
                 error::kInvalidArgument)
      << "tensor is not on the PrivateUse1 device";
  auto maybe_dynamism_info = GetDynamismInfo(tensor);
  if (maybe_dynamism_info.ok()) {
    return std::vector<BoundedDynamicDimension>(maybe_dynamism_info->begin(),
                                                maybe_dynamism_info->end());
  }
  return {};
}

}  // namespace

PYBIND11_MODULE(_tpu_torch_dynamism, m) {
  py::class_<BoundedDynamicDimension>(m, "BoundedDynamicDimension")
      .def_readonly("dimension", &BoundedDynamicDimension::dimension)
      .def_readonly("lower_bound", &BoundedDynamicDimension::lower_bound)
      .def_readonly("upper_bound", &BoundedDynamicDimension::upper_bound);

  m.def("mark_dynamic", &PyMarkDynamic, py::arg("tensor"), py::arg("dimension"),
        py::arg("lower_bound"), py::arg("upper_bound"),
        py::doc("Marks a tensor's dimensions as dynamic and returns the "
                "original input tensor or a new view tensor."));
  m.def("get_dynamism_info", &PyGetDynamismInfo, py::arg("tensor"),
        py::doc(
            "Returns a dictionary with dynamism hints for the given tensor."));
}

}  // namespace torch_tpu
