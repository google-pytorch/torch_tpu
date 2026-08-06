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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/error_utils.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"

namespace torch_tpu {
namespace py = pybind11;

namespace {

void PyEnterLayoutContext(
    const std::vector<int64_t>&  // INT_VEC_OK
        minor_to_major,
    const std::vector<std::vector<int64_t>>&  // INT_VEC_OK
        tiles,
    int64_t element_size_in_bits) {
  std::vector<xla::Tile> xla_tiles;
  xla_tiles.reserve(tiles.size());
  for (const auto& tile_dims : tiles) {
    xla_tiles.push_back(xla::Tile(tile_dims));
  }
  xla::Layout layout(minor_to_major, xla_tiles, element_size_in_bits);
  PushContextState<LayoutContextState>(
      std::make_shared<xla::Layout>(std::move(layout)));
}

void PyExitLayoutContext() { PopContextState<LayoutContextState>(); }

}  // namespace

PYBIND11_MODULE(annotations_py, m) {
  m.def("enter_layout_context", &PyEnterLayoutContext,
        py::arg("minor_to_major"), py::arg("tiles"),
        py::arg("element_size_in_bits"));
  m.def("exit_layout_context", &PyExitLayoutContext);
}

}  // namespace torch_tpu
