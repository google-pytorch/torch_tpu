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

"""The relay agent: one per TPU VM, reading one mailbox.

Being the only reader of its own prefix is what keeps this loop free of
locking. The agent takes the oldest item, runs it, writes the result,
and moves on.

It also owns the truth about its chip. When the chip stops working the
agent deletes its own binding, which makes the worker holding it fail
its next check and go find a different chip. That deletion is the whole
alerting mechanism; there is no separate channel to keep in sync.
"""

from __future__ import annotations

import dataclasses
import threading
import time

from rbe_tpu.common import binding as binding_lib
from rbe_tpu.common import gcs
from rbe_tpu.common import health as health_lib
from rbe_tpu.common import protocol


@dataclasses.dataclass(frozen=True)
class Execution:
  """What running one item produced.

  `infra_reason` is set when the test never got to run at all, which is
  a different thing from the test running and failing. The chip is not
  implicated either way.
  """

  exit_code: int
  output: str = ""
  output_files: dict[str, bytes] = dataclasses.field(default_factory=dict)
  infra_reason: str = ""


class Agent:
  """Serves one TPU's mailbox."""

  def __init__(self, client: gcs.Client, tpu: str, runner, *, deps=None):
    """Builds an agent.

    Args:
      client: Storage client for the relay bucket.
      tpu: This VM's TPU name, which is also its mailbox prefix.
      runner: Callable taking a WorkItem and returning an Execution.
      deps: Optional AgentDeps overriding the clock and health monitor.
    """
    deps = deps or AgentDeps()
    self._client = client
    self._tpu = tpu
    self._runner = runner
    self._time_fn = deps.time_fn
    self._monitor = deps.monitor or health_lib.HealthMonitor(
        time_fn=deps.time_fn
    )
    self._registry = binding_lib.FleetRegistry(client, time_fn=deps.time_fn)
    self._agent_id = deps.agent_id or f"agent-{tpu}"
    self._last_heartbeat = 0.0

  @property
  def quarantined(self) -> bool:
    return self._monitor.quarantined

  def start(self) -> bool:
    """Probes the chip and publishes a first fleet entry.

    Returns:
      True when the chip is usable. A false result still publishes an
      entry, so operators and workers can see why this TPU is out.
    """
    healthy = self._monitor.check_at_boot()
    if not healthy:
      self._revoke_binding()
    self._publish(force=True)
    self._clear_orphan_markers()
    return healthy

  def poll_once(self) -> bool:
    """Runs one iteration of the loop.

    Returns:
      True when an item was processed, which tells the caller not to
      sleep before looking again.
    """
    was_quarantined = self._monitor.quarantined
    self._monitor.poll()
    if was_quarantined != self._monitor.quarantined:
      self._on_health_change()
    self._publish()
    if self._monitor.quarantined:
      return False
    item = self._next_item()
    if item is None:
      return False
    self._process(item)
    return True

  def run_forever(self, sleep_fn=time.sleep) -> None:
    """Serves the mailbox until the process is stopped."""
    self.start()
    stop_event = threading.Event()

    def _heartbeat_loop():
      while not stop_event.is_set():
        try:
          self._monitor.poll()
          if self._monitor.quarantined:
            self._on_health_change()
          self._publish(force=True)
        except Exception:
          pass
        stop_event.wait(protocol.AGENT_HEARTBEAT_SECONDS)

    heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    heartbeat_thread.start()
    try:
      while True:
        if not self.poll_once():
          sleep_fn(protocol.AGENT_POLL_SECONDS)
    finally:
      stop_event.set()

  def _publish(self, *, force: bool = False) -> None:
    """Heartbeats on a timer, carrying the current health state."""
    now = self._time_fn()
    if not force and now - self._last_heartbeat < (
        protocol.AGENT_HEARTBEAT_SECONDS
    ):
      return
    self._last_heartbeat = now
    state = (
        protocol.STATE_QUARANTINED
        if self._monitor.quarantined
        else protocol.STATE_HEALTHY
    )
    self._registry.publish(
        protocol.FleetEntry(
            tpu=self._tpu,
            agent_id=self._agent_id,
            updated_at=now,
            state=state,
            reason=self._monitor.reason,
            detector=self._monitor.detector,
            quarantine_count=self._monitor.quarantine_count,
            quarantined_until=self._monitor.quarantined_until,
        )
    )

  def _on_health_change(self) -> None:
    """Reacts to the chip going bad, or coming back."""
    if self._monitor.quarantined:
      self._revoke_binding()
    self._publish(force=True)

  def _revoke_binding(self) -> None:
    """Drops the binding so the holder stops sending work here."""
    binding_lib.revoke(self._client, self._tpu)

  def _next_item(self) -> protocol.WorkItem | None:
    """Returns the oldest valid item, discarding any that are stale."""
    names = self._client.list(protocol.work_prefix(self._tpu))
    for name in names:
      blob = self._client.get(name)
      if blob is None:
        continue
      item = protocol.WorkItem.decode(blob.json())
      if self._owns(item):
        return item
      # Written by a worker that has since lost this chip. Its action
      # has already been told to go elsewhere, so drop the item.
      self._discard(item.item_id)
    return None

  def _owns(self, item: protocol.WorkItem) -> bool:
    """Checks the item was written by whoever currently holds the chip."""
    blob = self._client.get(protocol.binding_path(self._tpu))
    if blob is None:
      return False
    return blob.generation == item.binding_generation

  def _discard(self, item_id: str) -> None:
    self._client.delete(protocol.work_path(self._tpu, item_id))
    self._client.delete(protocol.running_path(self._tpu, item_id))

  def _process(self, item: protocol.WorkItem) -> None:
    """Runs one item and reports on it."""
    if self._client.get(protocol.result_path(self._tpu, item.item_id)):
      # Finished before a crash, and the result is already published.
      self._discard(item.item_id)
      return
    attempts = self._record_attempt(item.item_id)
    if attempts > protocol.MAX_ITEM_ATTEMPTS:
      self._report(
          item,
          Execution(exit_code=1),
          started_at=self._time_fn(),
          infra_reason=f"gave up after {attempts - 1} attempts",
      )
      return
    started_at = self._time_fn()
    execution = self._run(item)
    reason = self._judge(execution)
    self._report(item, execution, started_at=started_at, infra_reason=reason)

  def _record_attempt(self, item_id: str) -> int:
    """Bumps and returns the attempt count for an item."""
    path = protocol.running_path(self._tpu, item_id)
    blob = self._client.get(path)
    attempts = blob.json().get("attempts", 0) + 1 if blob else 1
    payload = f'{{"attempts": {attempts}}}'.encode("utf-8")
    self._client.put(path, payload)
    return attempts

  def _run(self, item: protocol.WorkItem) -> Execution:
    try:
      return self._runner(item)
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
      return Execution(
          exit_code=1,
          output=f"relay agent error: {error}",
          infra_reason=f"agent could not run the item: {error}",
      )

  def _judge(self, execution: Execution) -> str:
    """Decides whether a failure was the chip's fault.

    Returns:
      A reason string when the run should not count against the test,
      else an empty string.
    """
    if execution.infra_reason:
      # Our own plumbing broke, not the chip. Keeping this away from the
      # health monitor matters: a bucket outage hits every agent at once
      # and would otherwise quarantine the whole fleet in one go.
      return execution.infra_reason
    if execution.exit_code == 0:
      self._monitor.observe_success()
      return ""
    was_quarantined = self._monitor.quarantined
    usable = self._monitor.observe_failure(execution.output)
    if was_quarantined != self._monitor.quarantined:
      self._on_health_change()
    if usable:
      return ""
    return self._monitor.reason or "chip is quarantined"

  def _report(self, item, execution, *, started_at, infra_reason) -> None:
    """Uploads outputs, then the result, then clears the mailbox.

    The result object goes last on purpose. The worker treats it as the
    signal that everything else is already in place, so writing it
    earlier would let a worker read a half-finished run.
    """
    names = []
    for name, data in sorted(execution.output_files.items()):
      self._client.put(
          protocol.output_path(self._tpu, item.item_id, name), data
      )
      names.append(name)
    result = protocol.Result(
        item_id=item.item_id,
        exit_code=execution.exit_code,
        started_at=started_at,
        finished_at=self._time_fn(),
        infra_failure=bool(infra_reason),
        infra_reason=infra_reason,
        outputs=names,
    )
    self._client.put(
        protocol.result_path(self._tpu, item.item_id), result.encode()
    )
    self._discard(item.item_id)

  def _clear_orphan_markers(self) -> None:
    """Drops attempt markers whose work item is already gone."""
    work = set(self._client.list(protocol.work_prefix(self._tpu)))
    for name in self._client.list(protocol.running_prefix(self._tpu)):
      item_id = name.rsplit("/", 1)[-1].removesuffix(".json")
      if protocol.work_path(self._tpu, item_id) not in work:
        self._client.delete(name)


@dataclasses.dataclass
class AgentDeps:
  """Injection points, kept in one object to stay under the arg limit."""

  time_fn: object = time.time
  monitor: object = None
  agent_id: str = ""
