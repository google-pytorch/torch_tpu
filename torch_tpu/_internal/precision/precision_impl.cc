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
#include <optional>

#include "torch/extension.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/ops/precision_context.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {

using Precision = mlir::stablehlo::Precision;

PYBIND11_MODULE(precision_impl, m) {
  pybind11::enum_<Precision>(m, "Precision")
      .value("DEFAULT",
             Precision::DEFAULT)  // EXPLICIT_PRECISION_OK=root usage
      .value("HIGH",
             Precision::HIGH)  // EXPLICIT_PRECISION_OK=root usage
      .value("HIGHEST",
             Precision::HIGHEST)  // EXPLICIT_PRECISION_OK=root usage
      .export_values();

  pybind11::class_<DotAlgorithmConfig>(m, "DotAlgorithmConfig")
      .def(pybind11::init<c10::ScalarType, int64_t, int64_t, int64_t, bool>(),
           pybind11::arg("accumulation_type"),
           pybind11::arg("lhs_component_count"),
           pybind11::arg("rhs_component_count"),
           pybind11::arg("num_primitive_operations"),
           pybind11::arg("allow_imprecise_accumulation"))
      .def_property_readonly("accumulation_type",
                             &DotAlgorithmConfig::accumulation_type)
      .def_property_readonly("lhs_component_count",
                             &DotAlgorithmConfig::lhs_component_count)
      .def_property_readonly("rhs_component_count",
                             &DotAlgorithmConfig::rhs_component_count)
      .def_property_readonly("num_primitive_operations",
                             &DotAlgorithmConfig::num_primitive_operations)
      .def_property_readonly("allow_imprecise_accumulation",
                             &DotAlgorithmConfig::allow_imprecise_accumulation);

  m.def("_get_dot_algorithm", &GetDotAlgorithm,
        "Internal get for context manager");
  m.def("_push_dot_algorithm", &PushContextState<DotAlgorithmContextState>,
        "Internal push for context manager");
  m.def("_pop_dot_algorithm", &PopContextState<DotAlgorithmContextState>,
        "Internal pop for context manager");

  m.def("_get_precision", &GetPrecision, "Internal get for context manager");
  m.def("_push_precision", &PushContextState<Precision>,
        "Internal push for context manager");
  m.def("_pop_precision", &PopContextState<Precision>,
        "Internal pop for context manager");
}

}  // namespace torch_tpu
