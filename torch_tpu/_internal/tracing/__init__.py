# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Eager-mode tlparse tracing for torch_tpu.

When TORCH_TRACE is set in the environment before `import torch`, the
C++ StructuredLogBuffer accumulates one StructuredLogEvent per compiled
materialization batch. Calling enable() starts a daemon that drains the
buffer on a timer, plus a final drain at process exit, and emits
tlparse-compatible artifacts via torch._logging.trace_structured.

Initialization is explicit: torch_tpu/_loader.py calls enable_if_requested() on
first device load.
"""

import atexit
from collections.abc import Callable, Mapping
import threading
from typing import Any

from absl import logging
from torch._logging import trace_structured
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.tracing import _tpu_torch_tracing

_FLUSH_INTERVAL_SECONDS = 5.0


def _const_thunk(value: Any) -> Callable[[], Any]:
  """Create a closure capturing the value."""
  return lambda: value


def _emit_chromium(event: Mapping[str, Any]) -> None:
  """Thread-safe: emit one chromium_event for an eager batch.

  Args:
    event: Must contain `chromium_payload` (string, JSON-serialized Perfetto
      X-event built in C++ at Push time).
  """
  payload = event.get("chromium_payload", "")
  if not payload:
    return
  trace_structured(
      "chromium_event",
      metadata_fn=dict,
      payload_fn=_const_thunk(payload),
      expect_trace_id=False,
  )


def _emit_event_artifacts(event: Mapping[str, Any]) -> None:
  """Thread-safe: emit aten_graph and stablehlo artifacts for one batch.

  Args:
    event: A mapping containing event details, including `name`, `cache_hit`,
      `reason`, `aten_graph`, and `stablehlo`.
  """
  artifacts = [
      ("torchtpu_eager_aten_graph", "aten_graph"),
      ("torchtpu_eager_stablehlo", "stablehlo"),
  ]
  for artifact_name, payload_key in artifacts:
    metadata = {
        "name": artifact_name,
        "encoding": "string",
        "op_name": event["name"],
        "cache_hit": event["cache_hit"],
        "compile_failed": event.get("compile_failed", False),
        "reason": event["reason"],
    }
    payload = event.get(payload_key, "")
    trace_structured(
        "artifact",
        metadata_fn=_const_thunk(metadata),
        payload_fn=_const_thunk(payload),
        expect_trace_id=False,
    )


def _flush() -> None:
  """Drain the C++ buffer and emit artifacts. Safe from any thread."""
  try:
    drained = _tpu_torch_tracing._drain_structured_log_buffer()  # pylint: disable=protected-access
  except Exception:  # pylint: disable=broad-except
    logging.error("torch_tpu tracing drain failed", exc_info=True)
    return
  events = drained.get("events", [])
  dropped = drained.get("dropped_since_last_drain", 0)
  if dropped > 0:
    logging.warning(
        "torch_tpu tracing dropped %d events since last drain", dropped
    )
  for event in events:
    _emit_chromium(event)
    _emit_event_artifacts(event)


class _FlushDaemon(threading.Thread):
  """Background thread that drains the buffer every _FLUSH_INTERVAL_SECONDS."""

  def __init__(self) -> None:
    super().__init__(name="torch_tpu-eager-trace-flush", daemon=True)
    self._stop_event = threading.Event()

  def run(self) -> None:
    while not self._stop_event.wait(_FLUSH_INTERVAL_SECONDS):
      _flush()
    # Final drain on the way out.
    _flush()

  def stop(self) -> None:
    self._stop_event.set()


_daemon: _FlushDaemon | None = None
# Protects _daemon during enable/atexit transitions.
_lock = threading.Lock()


def _atexit_flush() -> None:
  """Stop the daemon and let its run() loop do the final flush.

  Timeout matches the regular flush interval so in-flight emits get
  the same headroom as a normal tick.
  """
  global _daemon
  with _lock:
    daemon = _daemon
    _daemon = None
  if daemon is None:
    return
  daemon.stop()
  daemon.join(timeout=_FLUSH_INTERVAL_SECONDS)
  if daemon.is_alive():
    logging.warning(
        "torch_tpu tracing daemon did not exit within %.1fs; "
        "some events may not be flushed before process exit.",
        _FLUSH_INTERVAL_SECONDS,
    )


def enable_if_requested() -> None:
  """Initialize the tracing daemon if TORCH_TRACE was set at startup."""
  global _daemon
  if not _tpu_torch_tracing._structured_log_enabled():  # pylint: disable=protected-access
    return
  with _lock:
    if _daemon is not None:
      return
    tpu_torch_compile.push_enable_tracebacks(True)
    _daemon = _FlushDaemon()
    _daemon.start()
    atexit.register(_atexit_flush)
