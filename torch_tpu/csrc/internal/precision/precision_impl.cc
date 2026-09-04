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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch_tpu/csrc/ops/precision_context.h"  // IWYU pragma: keep

namespace torch_tpu {

using Precision = mlir::stablehlo::Precision;
mlir::stablehlo::Precision PyUnsafeGetPrecision();

namespace {

Precision ToPrecision(pybind11::object obj) {
  if (pybind11::hasattr(obj, "value")) {
    return static_cast<Precision>(obj.attr("value").cast<int>());
  }
  return static_cast<Precision>(obj.cast<int>());
}

pybind11::object ToPyPrecision(Precision precision) {
  pybind11::object precision_cls =
      pybind11::module_::import("torch_tpu._internal.precision")
          .attr("Precision");
  return precision_cls(static_cast<int>(precision));
}

}  // namespace

PYBIND11_MODULE(precision_impl, m) {
  m.def(
      "_get_precision", []() { return ToPyPrecision(PyUnsafeGetPrecision()); },
      "Internal get for context manager");
  // The explicit arg name future-proofs the `substitute_in_graph` polyfill in
  // precision.py: torch compares parameter *names* when the original's
  // signature is introspectable. Today pybind11 exposes none (the check is
  // skipped); if that ever changes, "mode" must match the polyfill.
  m.def(
      "_push_precision",
      [](pybind11::object mode) { PyPushPrecision(ToPrecision(mode)); },
      pybind11::arg("mode"), "Internal push for context manager");
  m.def("_pop_precision", &PyPopPrecision, "Internal pop for context manager");
  m.def(
      "_set_global_precision",
      [](pybind11::object mode) { PySetGlobalPrecision(ToPrecision(mode)); },
      pybind11::arg("mode"), "Set global float32 matmul precision");
  m.def(
      "_get_global_precision",
      []() { return ToPyPrecision(PyGetGlobalPrecision()); },
      "Get global float32 matmul precision");
}

}  // namespace torch_tpu
