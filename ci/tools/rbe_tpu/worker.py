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

"""The relay worker: runs inside an RBE action, hands work to a TPU.

The worker holds a chip from the moment it boots, so dispatching is
mostly bookkeeping: post an item, wait for the result, exit with its
status.

The part worth reading is what happens when the chip goes bad. The
worker checks its binding before posting and again while waiting, and a
revoked binding sends the action to a different chip instead of failing
it. A broken TPU costs a little throughput, not a red test.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import os
import tarfile
import time

from ci.tools.rbe_tpu import binding as binding_lib
from ci.tools.rbe_tpu import gcs
from ci.tools.rbe_tpu import protocol

# How often the worker re-checks its binding while waiting for a result.
# Without this a revocation would not be noticed until the item's own
# timeout expired, which for a long test is several minutes of waiting
# on a chip that is already gone.
BINDING_CHECK_SECONDS = 15.0

# Grace on top of the item's timeout, covering the agent's own overhead.
RESULT_DEADLINE_SLACK_SECONDS = 60.0


@dataclasses.dataclass(frozen=True)
class Dispatch:
  """The outcome of trying to get one command run on a chip."""

  result: protocol.Result | None
  tpu: str = ""
  failure: str = ""

  @property
  def ok(self) -> bool:
    return self.result is not None


@dataclasses.dataclass
class WorkerDeps:
  """Injection points, kept in one object to stay under the arg limit."""

  time_fn: object = time.time
  sleep_fn: object = time.sleep
  log_fn: object = None
  bind_wait_seconds: float = protocol.BIND_WAIT_TIMEOUT_SECONDS


class Worker:
  """Dispatches actions from one RBE worker to its bound TPU."""

  def __init__(self, client: gcs.Client, worker_id: str, *, deps=None):
    deps = deps or WorkerDeps()
    self._client = client
    self._time_fn = deps.time_fn
    self._sleep_fn = deps.sleep_fn
    self._bind_wait = deps.bind_wait_seconds
    self._log_fn = deps.log_fn or (lambda _message: None)
    self.binder = binding_lib.Binder(client, worker_id, time_fn=deps.time_fn)

  def dispatch(self, request: "Request") -> Dispatch:
    """Runs one command on a chip, moving to another if this one dies.

    Args:
      request: What to run, and with what inputs.

    Returns:
      The agent's result, or a description of why there is none.
    """
    failures: list[str] = []
    tried: set[str] = set()
    for _ in range(protocol.MAX_DISPATCH_ATTEMPTS):
      if not self._ensure_chip(tried):
        failures.append("no healthy chip was available")
        break
      tpu = self.binder.tpu
      tried.add(tpu)
      outcome = self._attempt(request)
      if outcome.ok and not outcome.result.infra_failure:
        return outcome
      reason = outcome.failure or outcome.result.infra_reason or "chip fault"
      failures.append(f"{tpu}: {reason}")
      self._log_fn(f"moving to another chip after {tpu}: {reason}")
      self.binder.release()
    return Dispatch(None, failure="; ".join(failures))

  def _ensure_chip(self, tried) -> bool:
    """Makes sure we hold a chip that is fit to use and not already tried."""
    if (
        self.binder.bound
        and self.binder.tpu not in tried
        and self.binder.check().ok
    ):
      return True
    self.binder.release()
    tpu = self.binder.wait_for_binding(
        self._bind_wait,
        sleep_fn=self._sleep_fn,
        exclude=tried,
    )
    if not tpu:
      return False
    return self.binder.check().ok

  def _attempt(self, request: "Request") -> Dispatch:
    """Posts one item to the chip we hold and waits for its result."""
    tpu = self.binder.tpu
    item = protocol.WorkItem(
        item_id=protocol.new_item_id(self._time_fn()),
        tpu=tpu,
        binding_generation=self.binder.generation,
        argv=list(request.argv),
        workdir=request.workdir,
        timeout_seconds=request.timeout_seconds,
        env=dict(request.env),
        layers=list(request.layers),
        submitted_at=self._time_fn(),
    )
    self._client.put(protocol.work_path(tpu, item.item_id), item.encode())
    return self._await_result(item)

  def _await_result(self, item: protocol.WorkItem) -> Dispatch:
    """Polls for the result, bailing early if the chip is revoked."""
    deadline = (
        self._time_fn() + item.timeout_seconds + RESULT_DEADLINE_SLACK_SECONDS
    )
    started = self._time_fn()
    next_check = started + BINDING_CHECK_SECONDS
    path = protocol.result_path(item.tpu, item.item_id)
    while self._time_fn() < deadline:
      blob = self._client.get(path)
      if blob is not None:
        return Dispatch(protocol.Result.decode(blob.json()), tpu=item.tpu)
      if self._time_fn() >= next_check:
        check = self.binder.check()
        if not check.ok:
          self._abandon(item)
          return Dispatch(None, tpu=item.tpu, failure=check.reason)
        next_check = self._time_fn() + BINDING_CHECK_SECONDS
      self._sleep_fn(_poll_delay(self._time_fn() - started))
    self._abandon(item)
    return Dispatch(None, tpu=item.tpu, failure="timed out waiting for result")

  def _abandon(self, item: protocol.WorkItem) -> None:
    """Withdraws an item so a later agent does not start it."""
    self._client.delete(protocol.work_path(item.tpu, item.item_id))

  def upload_layers(self, directories) -> list[str]:
    """Packs directories into content-addressed blobs.

    Naming each layer after its own digest means the large, unchanging
    one is uploaded once and every later action just references it.

    Args:
      directories: Paths to pack, one layer each, in extraction order.

    Returns:
      Blob object names to put in a work item.
    """
    names = []
    for directory in directories:
      data = _tar_bytes(directory)
      digest = hashlib.sha256(data).hexdigest()
      path = protocol.blob_path(digest)
      if self._client.get(path) is None:
        self._client.put(path, data)
      names.append(path)
    return names


@dataclasses.dataclass(frozen=True)
class Request:
  """One command to run on a chip."""

  argv: list[str]
  workdir: str = ""
  timeout_seconds: float = 900.0
  env: dict[str, str] = dataclasses.field(default_factory=dict)
  layers: list[str] = dataclasses.field(default_factory=list)


def _poll_delay(elapsed: float) -> float:
  """Polls fast early for short tests, then backs off for the long tail."""
  if elapsed < protocol.WORKER_POLL_FAST_WINDOW_SECONDS:
    return protocol.WORKER_POLL_FAST_SECONDS
  return protocol.WORKER_POLL_SLOW_SECONDS


def _tar_bytes(directory: str) -> bytes:
  """Packs a directory into a deterministic, uncompressed tar.

  Determinism is what makes the digest stable: the same tree must
  produce the same bytes, or the content-addressed blob is uploaded
  again on every run.

  Args:
    directory: Directory to pack.

  Returns:
    The tar archive.
  """
  buffer = io.BytesIO()
  with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as tar:
    tar.dereference = True
    for path in sorted(_walk(directory)):
      try:
        info = tar.gettarinfo(path, arcname=os.path.relpath(path, directory))
      except OSError:
        continue
      info.mtime = 0
      info.uid = 0
      info.gid = 0
      info.uname = ""
      info.gname = ""
      if info.isreg():
        try:
          with open(path, "rb") as handle:
            tar.addfile(info, handle)
        except OSError:
          continue
      elif info.isdir():
        tar.addfile(info)
  return buffer.getvalue()


def _walk(directory: str) -> list[str]:
  paths = []
  for root, dirnames, filenames in os.walk(directory):
    dirnames.sort()
    paths.extend(os.path.join(root, name) for name in dirnames)
    paths.extend(os.path.join(root, name) for name in filenames)
  return paths
