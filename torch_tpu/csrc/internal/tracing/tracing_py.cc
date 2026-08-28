// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Pybind exposure of torch_tpu's eager-mode StructuredLogBuffer for the
// Python tracing daemon at torch_tpu/_internal/tracing/__init__.py. The
// daemon drains buffered events on a 5s timer and emits tlparse-compatible
// artifacts via torch._logging.trace_structured.

#include <memory>
#include <string>
#include <utility>

#include "absl/time/time.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch_tpu/eager/structured_log_buffer.h"

namespace torch_tpu {
namespace py = pybind11;

namespace {

// Drains the StructuredLogBuffer and returns one dict per event plus a
// dropped-events counter. Each dict mirrors StructuredLogEvent fields.
py::dict PyDrainStructuredLogBuffer() {
  StructuredLogBuffer::DrainResult result =
      StructuredLogBuffer::GetInstance().Drain();
  py::list events;
  for (auto& ev_ptr : result.events) {
    const StructuredLogEvent& ev = *ev_ptr;
    py::dict d;
    d["name"] = ev.name;
    d["ts_us"] = absl::ToUnixMicros(ev.timestamp);
    d["dur_us"] = absl::ToInt64Microseconds(ev.duration);
    d["cache_hit"] = ev.cache_hit;
    d["compile_failed"] = ev.compile_failed;
    d["aten_graph"] = ev.aten_graph_payload;
    d["stablehlo"] = ev.mlir_payload;
    d["reason"] = std::string(ToString(ev.reason));
    d["chromium_payload"] = ev.chromium_payload;
    events.append(std::move(d));
  }
  py::dict out;
  out["events"] = std::move(events);
  out["dropped_since_last_drain"] = result.dropped_since_last_drain;
  return out;
}

bool PyStructuredLogEnabled() {
  return StructuredLogBuffer::GetInstance().enabled();
}

}  // namespace

PYBIND11_MODULE(_tpu_torch_tracing, m) {
  m.doc() =
      "Pybind exposure of torch_tpu's eager StructuredLogBuffer "
      "for the Python tracing daemon.";

  m.def("_drain_structured_log_buffer", &PyDrainStructuredLogBuffer,
        py::doc("Drains the buffer and returns "
                "{events: [<dict per event>], dropped_since_last_drain: int}. "
                "Each event dict contains: name, ts_us, dur_us, cache_hit, "
                "aten_graph, stablehlo, reason."));

  m.def("_structured_log_enabled", &PyStructuredLogEnabled,
        py::doc("Returns true if TORCH_TRACE was set at program startup."));
}

}  // namespace torch_tpu
