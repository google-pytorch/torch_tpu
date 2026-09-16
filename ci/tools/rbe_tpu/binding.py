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

"""Pairing a worker with a TPU, once, at boot.

A worker takes a chip when it starts and keeps it. The claim is a single
`ifGenerationMatch=0` write, which Cloud Storage resolves atomically, so
there is no agreement protocol and nothing to coordinate. After boot the
hot path does no locking at all: every action that lands on a worker
already knows which chip it owns.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import random
import time

from ci.tools.rbe_tpu import gcs
from ci.tools.rbe_tpu import protocol


@dataclasses.dataclass(frozen=True)
class BindingCheck:
  """Whether a worker may still dispatch to its chip, and why not."""

  ok: bool
  reason: str = ""


class FleetRegistry:
  """Reads and writes the self-registration entries agents publish."""

  def __init__(self, client: gcs.Client, *, time_fn=time.time):
    self._client = client
    self._time_fn = time_fn

  def publish(self, entry: protocol.FleetEntry) -> None:
    """Writes one agent's entry. The agent is its only writer."""
    self._client.put(protocol.fleet_path(entry.tpu), entry.encode())

  def read(self, tpu: str) -> protocol.FleetEntry | None:
    """Returns one agent's entry, or None when it has never registered."""
    blob = self._client.get(protocol.fleet_path(tpu))
    if blob is None:
      return None
    return protocol.FleetEntry.decode(blob.json())

  def names(self) -> list[str]:
    """Returns every TPU that has ever registered, healthy or not.

    Used to tell "all the chips are busy, keep waiting" apart from
    "there is no chip here that could ever serve me".
    """
    return [
        protocol.tpu_from_fleet_path(path)
        for path in self._client.list(protocol.FLEET_PREFIX)
    ]

  def available(self) -> list[str]:
    """Returns the TPUs that are heartbeating and not quarantined.

    Reading the registry rather than asking the Compute API keeps the
    worker's permissions down to this one bucket, and means the live
    fleet describes itself instead of having to be discovered by zone.

    Returns:
      TPU names, in arbitrary order.
    """
    now = self._time_fn()
    paths = self._client.list(protocol.FLEET_PREFIX)
    if not paths:
      return []

    def _check(path: str) -> str | None:
      blob = self._client.get(path)
      if blob is None:
        return None
      entry = protocol.FleetEntry.decode(blob.json())
      if entry.is_available(now):
        return entry.tpu
      return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
      return [tpu for tpu in executor.map(_check, paths) if tpu]


class Binder:
  """Holds one worker's claim on one TPU."""

  def __init__(self, client: gcs.Client, worker_id: str, *, time_fn=time.time):
    self._client = client
    self._worker_id = worker_id
    self._time_fn = time_fn
    self._registry = FleetRegistry(client, time_fn=time_fn)
    self.tpu = ""
    self.generation = 0

  @property
  def bound(self) -> bool:
    return bool(self.tpu)

  def bind(self, exclude=()) -> str:
    """Tries once to take any free chip.

    Candidates are shuffled. Walking them in a fixed order would send
    all 28 workers at the same chip first and serialise the whole fleet
    through a run of failed preconditions.

    Args:
      exclude: TPUs to skip, normally ones this caller already failed
        on. Without this a retry can land straight back on the chip
        that just broke and spend every attempt there.

    Returns:
      The TPU name taken, or an empty string when none was free.
    """
    candidates = [
        tpu for tpu in self._registry.available() if tpu not in exclude
    ]
    random.shuffle(candidates)
    for tpu in candidates:
      if self._try_take(tpu):
        return tpu
    return ""

  def wait_for_binding(self, timeout: float, sleep_fn=time.sleep, exclude=()):
    """Retries `bind` until it succeeds or the deadline passes.

    An unbound worker cannot refuse work: the RBE bot agent registers
    with the scheduler whatever our startup script thinks. So an action
    that lands here waits rather than failing. A chip being down should
    cost throughput, not a red test on a healthy change.

    Args:
      timeout: Seconds to keep trying.
      sleep_fn: Injected for tests.
      exclude: TPUs to skip.

    Returns:
      The TPU name taken, or an empty string on timeout.
    """
    deadline = self._time_fn() + timeout
    delay = 1.0
    while True:
      tpu = self.bind(exclude=exclude)
      if tpu:
        return tpu
      known = self._registry.names()
      if known and all(name in exclude for name in known):
        # Every chip here has already been tried. Waiting cannot help.
        # An empty registry is different: the agents have not booted
        # yet, and that is worth waiting out.
        return ""
      if self._time_fn() >= deadline:
        return ""
      sleep_fn(delay)
      delay = min(delay * 2, 15.0)

  def _try_take(self, tpu: str) -> bool:
    """Claims one chip, stealing it when the previous claim has expired."""
    path = protocol.binding_path(tpu)
    blob = self._client.get(path)
    if blob is None:
      return self._write_claim(tpu, 0)
    existing = protocol.Binding.decode(blob.json())
    if not existing.is_expired(self._time_fn()):
      return False
    # Steal against the generation we just read, not blindly. Two
    # workers spotting the same dead claim: one wins, one gets a 412.
    return self._write_claim(tpu, blob.generation)

  def _write_claim(self, tpu: str, expected_generation: int) -> bool:
    now = self._time_fn()
    binding = protocol.Binding(
        tpu=tpu,
        worker_id=self._worker_id,
        bound_at=now,
        expires_at=now + protocol.BINDING_TTL_SECONDS,
    )
    try:
      blob = self._client.put(
          protocol.binding_path(tpu),
          binding.encode(),
          if_generation_match=expected_generation,
      )
    except gcs.PreconditionFailed:
      return False
    self.tpu = tpu
    self.generation = blob.generation
    return True

  def refresh(self) -> bool:
    """Extends the claim. Returns False when the claim has been lost.

    A lost claim means the agent revoked it, which is how a bad chip
    tells its worker to stop. The caller must stop dispatching.
    """
    if not self.bound:
      return False
    now = self._time_fn()
    binding = protocol.Binding(
        tpu=self.tpu,
        worker_id=self._worker_id,
        bound_at=now,
        expires_at=now + protocol.BINDING_TTL_SECONDS,
    )
    try:
      blob = self._client.put(
          protocol.binding_path(self.tpu),
          binding.encode(),
          if_generation_match=self.generation,
      )
    except gcs.PreconditionFailed:
      self._forget()
      return False
    self.generation = blob.generation
    return True

  def check(self) -> BindingCheck:
    """Confirms the chip is still ours and still healthy.

    A worker runs this immediately before posting each work item, and
    again every few seconds while waiting for a result. It costs two
    reads against a test that averages two minutes, and it is what
    stops an action from ever being sent to a chip already known bad.

    It doubles as the keep-alive. Folding the refresh in here avoids a
    background thread, and means a worker that has gone quiet lets its
    claim lapse so the chip returns to the pool.

    Returns:
      Whether dispatch may proceed, and the reason when it may not.
    """
    if not self.bound:
      return BindingCheck(False, "not bound")
    blob = self._client.get(protocol.binding_path(self.tpu))
    if blob is None:
      self._forget()
      return BindingCheck(False, "binding revoked")
    if blob.generation != self.generation:
      self._forget()
      return BindingCheck(False, "binding taken over")
    agent = self._check_agent()
    if not agent.ok:
      return agent
    return self._keep_alive(protocol.Binding.decode(blob.json()))

  def _keep_alive(self, current: protocol.Binding) -> BindingCheck:
    """Extends the claim when it is getting close to its expiry."""
    age = self._time_fn() - current.bound_at
    if age < protocol.BINDING_REFRESH_SECONDS:
      return BindingCheck(True)
    if not self.refresh():
      return BindingCheck(False, "binding lost while renewing it")
    return BindingCheck(True)

  def _check_agent(self) -> BindingCheck:
    """Confirms the agent on our chip is alive and not quarantined."""
    entry = self._registry.read(self.tpu)
    if entry is None:
      return BindingCheck(False, "agent never registered")
    now = self._time_fn()
    if not entry.is_fresh(now):
      return BindingCheck(False, "agent heartbeat is stale")
    if entry.state != protocol.STATE_HEALTHY:
      return BindingCheck(False, f"chip quarantined: {entry.reason}")
    return BindingCheck(True)

  def release(self) -> None:
    """Gives the chip back. Safe to call when the claim is already gone."""
    if not self.bound:
      return
    try:
      self._client.delete(
          protocol.binding_path(self.tpu),
          if_generation_match=self.generation,
      )
    except gcs.PreconditionFailed:
      pass
    self._forget()

  def _forget(self) -> None:
    self.tpu = ""
    self.generation = 0


def revoke(client: gcs.Client, tpu: str) -> bool:
  """Tears down a chip's binding from the agent's side.

  This is the alert. An agent that has decided its chip is bad deletes
  the binding, which makes the holder's next refresh or pre-dispatch
  check fail and sends it to look for a different chip.

  Args:
    client: Storage client.
    tpu: The TPU whose binding should go away.

  Returns:
    True when a binding was actually removed.
  """
  return client.delete(protocol.binding_path(tpu))
