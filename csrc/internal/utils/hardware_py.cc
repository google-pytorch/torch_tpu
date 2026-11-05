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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/pybind_error_utils.h"
#include "csrc/common/tpu_hardware.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"  // IWYU pragma: keep

namespace torch_tpu {
namespace py = pybind11;

namespace {

py::tuple ToPyTuple(const TpuSpec::Topology& dims) {
  py::tuple result(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    result[i] = dims[i];
  }
  return result;
}

TpuSpec::TopologyMap ParseTopologiesDict(const py::object& topologies) {
  TpuSpec::TopologyMap result;
  for (const auto& [key, value] : py::dict(topologies)) {
    result.emplace(key.cast<int64_t>(), value.cast<TpuSpec::Topology>());
  }
  return result;
}

py::object PyTpuSpecTopologies(const TpuSpec& self) {
  py::dict result;
  for (const auto& [device_count, dims] : self.topologies) {
    result[py::int_(device_count)] = ToPyTuple(dims);
  }
  return PyBindModule::Import("types").attr("MappingProxyType")(result);
}

py::tuple PyTpuSpecTopology(const TpuSpec& self,
                            std::optional<int64_t> world_size) {
  if (!world_size.has_value()) {
    world_size =
        PyBindModule::Import("torch_tpu._internal.utils._hardware_backend")
            .attr("get_tpu_device_count")()
            .cast<int64_t>();
  }
  const absl::StatusOr<TpuSpec::Topology> topology =
      self.GetTopology(world_size);
  if (!topology.ok()) {
    throw std::runtime_error(  // For pybind11.
        std::string(topology.status().message()));
  }
  return ToPyTuple(*topology);
}

py::object PyTpuSpecsByPciDeviceId() {
  py::dict result;
  for (const auto& [device_id, spec] : TpuSpec::SpecsByPciDeviceId()) {
    result[py::str(device_id)] =
        py::cast(spec, py::return_value_policy::reference);
  }
  return PyBindModule::Import("types").attr("MappingProxyType")(result);
}

std::optional<std::string> PyGetTpuTopology(std::optional<int64_t> world_size,
                                            std::string_view pci_devices_dir,
                                            std::string_view vfio_dir) {
  const absl::StatusOr<std::optional<std::string>> topology =
      GetTpuTopology(world_size, pci_devices_dir, vfio_dir);
  if (!topology.ok()) {
    throw std::runtime_error(  // For pybind11.
        std::string(topology.status().message()));
  }
  return *topology;
}

}  // namespace

TT_PYBIND11_MODULE(_hardware_backend, m) {
  PyBindClass<SparseCoreSpec>(m, "SparseCoreSpec")
      .def(py::init([](int64_t sparse_cores_per_device, int64_t simd_width) {
             return SparseCoreSpec{
                 .sparse_cores_per_device = sparse_cores_per_device,
                 .simd_width = simd_width,
             };
           }),
           py::arg("sparse_cores_per_device"), py::arg("simd_width"))
      .def_readonly("sparse_cores_per_device",
                    &SparseCoreSpec::sparse_cores_per_device)
      .def_readonly("simd_width", &SparseCoreSpec::simd_width)
      .def("__eq__",
           [](const SparseCoreSpec& self, const py::handle& other) {
             if (!py::isinstance<SparseCoreSpec>(other)) {
               return false;
             }
             return self == other.cast<const SparseCoreSpec&>();
           })
      .def("__hash__",
           [](const SparseCoreSpec& self) {
             return py::hash(
                 py::make_tuple(self.sparse_cores_per_device, self.simd_width));
           })
      .def("__repr__", [](const SparseCoreSpec& self) {
        return absl::StrCat("SparseCoreSpec(sparse_cores_per_device=",
                            self.sparse_cores_per_device,
                            ", simd_width=", self.simd_width, ")");
      });

  PyBindClass<TpuSpec>(m, "TpuSpec")
      .def(py::init([](std::string name, int64_t devices_per_chip,
                       int64_t hbm_bytes_per_chip, int64_t bf16_flops_per_chip,
                       const py::object& topologies,
                       std::optional<SparseCoreSpec> sparse_core_spec) {
             return TpuSpec{
                 .name = std::move(name),
                 .devices_per_chip = devices_per_chip,
                 .hbm_bytes_per_chip = hbm_bytes_per_chip,
                 .bf16_flops_per_chip = bf16_flops_per_chip,
                 .topologies = ParseTopologiesDict(topologies),
                 .sparse_core_spec = sparse_core_spec,
             };
           }),
           py::arg("name"), py::arg("devices_per_chip"),
           py::arg("hbm_bytes_per_chip"), py::arg("bf16_flops_per_chip"),
           py::arg("topologies"), py::arg("sparse_core_spec") = std::nullopt)
      .def_readonly("name", &TpuSpec::name)
      .def_readonly("devices_per_chip", &TpuSpec::devices_per_chip)
      .def_readonly("hbm_bytes_per_chip", &TpuSpec::hbm_bytes_per_chip)
      .def_readonly("bf16_flops_per_chip", &TpuSpec::bf16_flops_per_chip)
      .def_property_readonly("topologies", &PyTpuSpecTopologies)
      .def_readonly("sparse_core_spec", &TpuSpec::sparse_core_spec)
      .def_property_readonly(
          "hbm_bytes_per_device",
          [](const TpuSpec& self) { return self.hbm_bytes_per_device(); })
      .def_property_readonly(
          "bf16_flops_per_device",
          [](const TpuSpec& self) { return self.bf16_flops_per_device(); })
      .def("topology", &PyTpuSpecTopology, py::arg("world_size") = std::nullopt)
      .def("__eq__",
           [](const TpuSpec& self, const py::handle& other) {
             if (!py::isinstance<TpuSpec>(other)) {
               return false;
             }
             return self == other.cast<const TpuSpec&>();
           })
      .def("__hash__",
           [](const TpuSpec& self) {
             return py::hash(py::make_tuple(self.name, self.devices_per_chip,
                                            self.hbm_bytes_per_chip,
                                            self.bf16_flops_per_chip));
           })
      .def("__repr__", [](const TpuSpec& self) {
        return absl::StrCat("TpuSpec(name='", self.name,
                            "', devices_per_chip=", self.devices_per_chip,
                            ", hbm_bytes_per_chip=", self.hbm_bytes_per_chip,
                            ", bf16_flops_per_chip=", self.bf16_flops_per_chip,
                            ")");
      });

  m.attr("TpuSpec").attr("V5P") =
      py::cast(&TpuSpec::V5p(), py::return_value_policy::reference);
  m.attr("TpuSpec").attr("V5E") =
      py::cast(&TpuSpec::V5e(), py::return_value_policy::reference);
  m.attr("TpuSpec").attr("V6E") =
      py::cast(&TpuSpec::V6e(), py::return_value_policy::reference);
  m.attr("TpuSpec").attr("V7") =
      py::cast(&TpuSpec::V7(), py::return_value_policy::reference);
  m.attr("TpuSpec").attr("V8I") =
      py::cast(&TpuSpec::V8i(), py::return_value_policy::reference);
  m.attr("TpuSpec").attr("V8T") =
      py::cast(&TpuSpec::V8t(), py::return_value_policy::reference);

  m.def("get_tpu_specs_by_pci_device_id", &PyTpuSpecsByPciDeviceId);
  m.attr("TpuSpec").attr("specs_by_pci_device_id") =
      m.attr("get_tpu_specs_by_pci_device_id");
  m.def("get_tpu_spec", &TpuSpec::Get,
        py::arg("pci_devices_dir") = "/sys/bus/pci/devices",
        py::return_value_policy::reference);
  m.def("get_tpu_device_count", &GetTpuDeviceCount,
        py::arg("pci_devices_dir") = "/sys/bus/pci/devices",
        py::arg("vfio_dir") = "/dev/vfio");
  m.def("get_visible_tpu_device_count", &GetVisibleTpuDeviceCount,
        py::arg("pci_devices_dir") = "/sys/bus/pci/devices",
        py::arg("vfio_dir") = "/dev/vfio");
  m.def("get_tpu_topology", &PyGetTpuTopology,
        py::arg("world_size") = std::nullopt,
        py::arg("pci_devices_dir") = "/sys/bus/pci/devices",
        py::arg("vfio_dir") = "/dev/vfio");
}

}  // namespace torch_tpu
