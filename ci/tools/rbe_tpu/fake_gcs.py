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

"""An in-memory stand-in for Cloud Storage.

The binding protocol is only safe if generation preconditions behave the
way Cloud Storage says they do, so this fake models them precisely and the
tests assert against it. It is locked because the interesting tests run
several binders against one bucket at the same time.
"""

from __future__ import annotations

import threading

from ci.tools.rbe_tpu import gcs
from ci.tools.rbe_tpu import health


class FakeClient(gcs.Client):
  """A bucket held in a dict, with real generation semantics."""

  def __init__(self):
    self._lock = threading.Lock()
    self._objects: dict[str, gcs.Blob] = {}
    self._next_generation = 1
    self.calls: list[tuple[str, str]] = []
    # Maps an object name to a one-shot callable run inside the lock just
    # before a put is applied, which is how tests wedge a competing writer
    # into the gap between a read and its compare-and-swap.
    self.before_put: dict[str, object] = {}

  def get(self, name: str) -> gcs.Blob | None:
    with self._lock:
      self.calls.append(("get", name))
      return self._objects.get(name)

  def put(
      self, name: str, data: bytes, *, if_generation_match: int | None = None
  ) -> gcs.Blob:
    hook = self.before_put.pop(name, None)
    if hook is not None:
      hook()
    with self._lock:
      self.calls.append(("put", name))
      existing = self._objects.get(name)
      current = existing.generation if existing else 0
      if if_generation_match is not None and if_generation_match != current:
        raise gcs.PreconditionFailed(
            f"put {name}: generation {current} != {if_generation_match}"
        )
      blob = gcs.Blob(
          name=name, data=data, generation=self._next_generation
      )
      self._next_generation += 1
      self._objects[name] = blob
      return blob

  def delete(
      self, name: str, *, if_generation_match: int | None = None
  ) -> bool:
    with self._lock:
      self.calls.append(("delete", name))
      existing = self._objects.get(name)
      if existing is None:
        if if_generation_match not in (None, 0):
          raise gcs.PreconditionFailed(f"delete {name}: object is gone")
        return False
      if (
          if_generation_match is not None
          and if_generation_match != existing.generation
      ):
        raise gcs.PreconditionFailed(
            f"delete {name}: generation {existing.generation}"
            f" != {if_generation_match}"
        )
      del self._objects[name]
      return True

  def list(self, prefix: str) -> list[str]:
    with self._lock:
      self.calls.append(("list", prefix))
      return sorted(n for n in self._objects if n.startswith(prefix))

  def count(self, verb: str, name_prefix: str = "") -> int:
    """Counts recorded calls, for tests that assert on request volume.

    Args:
      verb: One of get, put, delete, list.
      name_prefix: Optional object-name prefix filter.

    Returns:
      How many matching calls have been made.
    """
    return sum(
        1
        for call_verb, name in self.calls
        if call_verb == verb and name.startswith(name_prefix)
    )


class FakeClock:
  """A clock the tests move by hand."""

  def __init__(self, now: float = 1_700_000_000.0):
    self._now = now
    self.slept = 0.0

  def time(self) -> float:
    return self._now

  def sleep(self, seconds: float) -> None:
    self.slept += seconds
    self._now += seconds

  def advance(self, seconds: float) -> None:
    self._now += seconds


class StubProbe:
  """A probe the test flips between healthy and broken."""

  def __init__(self, healthy: bool = True):
    self.healthy = healthy
    self.calls = 0
    self.detail = "stub detail"

  def __call__(self) -> health.ProbeResult:
    self.calls += 1
    return health.ProbeResult(self.healthy, "stub", self.detail)
