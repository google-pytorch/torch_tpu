# Copyright 2025 Google LLC
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

When TORCH_TRACE is set at program startup, the C++ StructuredLogBuffer
accumulates one EagerEvent per compiled materialization batch. This module
drains that buffer on a 5-second timer (plus atexit) and emits
tlparse-compatible artifacts:

  - `chromium_event` per batch: Perfetto timeline entry with timing,
    cache_hit in args, reason in args.
  - `torchtpu_eager_aten_graph` artifact per batch: FX-style ATen graph
    text (empty payload on cache hit).
  - `torchtpu_eager_stablehlo` artifact per batch: StableHLO MLIR text
    (empty payload on cache hit).
  - `torchtpu_eager_sync` artifact per flush: Python stack of the flush
    call site, plus num_traversals = number of events drained this flush.

Artifact metadata carries `op_name`, `cache_hit`, and `reason` so tlparse
can group events per batch and present them in the summary page.

Emitting both artifacts on every batch (including hits with empty
payloads) is intentional: the tlparse subparser uses the stablehlo
artifact as the per-batch anchor, so suppressing it on hit would cause
hits to disappear from the summary page.
"""

import atexit
import json
import os
import threading
import traceback as _tb_mod
from typing import Any

from absl import logging
import torch._logging
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.sync import _tpu_torch_sync

_FLUSH_INTERVAL_SECONDS = 5.0
_PID = os.getpid()


def _emit_chromium(event: dict[str, Any]) -> None:
  """Emit a Chrome Trace Event 'complete' record for one batch."""
  perfetto = {
      "name": event["name"],
      "ph": "X",
      "ts": event["ts_us"],
      "dur": max(0, event["dur_us"]),
      "cat": "torchtpu_eager",
      "pid": _PID,
      "tid": 0,
      "args": {
          "device": event["device"],
          "cache_hit": event["cache_hit"],
          "reason": event["reason"],
      },
  }
  payload = json.dumps(perfetto)
  torch._logging.trace_structured(  # pylint: disable=protected-access
      "chromium_event",
      metadata_fn=dict,
      payload_fn=lambda p=payload: p,
      expect_trace_id=False,
  )


def _emit_event_artifacts(event: dict[str, Any]) -> None:
  """Emit aten_graph + stablehlo artifacts for one batch.

  Both are always emitted (payloads empty on hit) so every batch
  appears in the tlparse summary page. The tlparse subparser uses
  stablehlo as the per-batch anchor and reads op_name / cache_hit /
  reason from artifact metadata.

  Args:
    event: The event dict containing name, cache_hit, reason, etc.
  """
  name = event["name"]
  cache_hit = event["cache_hit"]
  reason = event["reason"]
  aten_graph = event.get("aten_graph", "")
  stablehlo = event.get("stablehlo", "")

  torch._logging.trace_structured(  # pylint: disable=protected-access
      "artifact",
      metadata_fn=lambda: {
          "name": "torchtpu_eager_aten_graph",
          "encoding": "string",
          "op_name": name,
          "cache_hit": cache_hit,
          "reason": reason,
      },
      payload_fn=lambda p=aten_graph: p,
      expect_trace_id=False,
  )

  torch._logging.trace_structured(  # pylint: disable=protected-access
      "artifact",
      metadata_fn=lambda: {
          "name": "torchtpu_eager_stablehlo",
          "encoding": "string",
          "op_name": name,
          "cache_hit": cache_hit,
          "reason": reason,
      },
      payload_fn=lambda p=stablehlo: p,
      expect_trace_id=False,
  )


def _emit_flush_sync(num_traversals: int) -> None:
  """Emit a single torchtpu_eager_sync artifact describing this flush tick.

  Payload is the Python stack of the flush caller so users can see
  whether the timer thread, atexit, or user code triggered the drain.

  Args:
    num_traversals: Number of events drained in this flush.
  """
  stack = "".join(_tb_mod.format_stack())
  torch._logging.trace_structured(  # pylint: disable=protected-access
      "artifact",
      metadata_fn=lambda: {
          "name": "torchtpu_eager_sync",
          "encoding": "string",
          "num_traversals": num_traversals,
      },
      payload_fn=lambda p=stack: p,
      expect_trace_id=False,
  )


def flush() -> None:
  """Drain the C++ buffer and emit artifacts. Safe from any thread."""
  try:
    events = _tpu_torch_sync._drain_structured_log_buffer()  # pylint: disable=protected-access
  except Exception as e:  # pylint: disable=broad-except
    # Never let a flush error tear down the daemon.
    logging.exception("Failed to drain structured log buffer: %s", e)
    return
  if not events:
    return
  _emit_flush_sync(len(events))
  for event in events:
    _emit_chromium(event)
    _emit_event_artifacts(event)


class _FlushDaemon(threading.Thread):

  def __init__(self):
    super().__init__(name="torch_tpu-eager-trace-flush", daemon=True)
    self._stop = threading.Event()

  def run(self):
    while not self._stop.wait(_FLUSH_INTERVAL_SECONDS):
      flush()

  def stop(self):
    self._stop.set()


_daemon = None
_started = False
_lock = threading.Lock()


def _atexit_flush():
  if _daemon is not None:
    _daemon.stop()
  flush()


def maybe_enable():
  """Initialize tracing if TORCH_TRACE was set at program startup.

  The C++ StructuredLogBuffer only enables itself if TORCH_TRACE was set
  at program startup (same semantics as torch._logging's LazyTraceHandler),
  so this function is a no-op when TORCH_TRACE was not set before
  `import torch`.
  """
  global _daemon, _started
  if _started:
    return
  if not _tpu_torch_sync._structured_log_enabled():  # pylint: disable=protected-access
    return
  with _lock:
    if _started:
      return
    # TODO: Re-enable tracebacks when C++ Traversal::ReadableString is updated
    # to robustly filter and print line-of-code (LoC) info.
    # tpu_torch_compile.push_enable_tracebacks(True)
    _daemon = _FlushDaemon()
    _daemon.start()
    atexit.register(_atexit_flush)
    _started = True


maybe_enable()
