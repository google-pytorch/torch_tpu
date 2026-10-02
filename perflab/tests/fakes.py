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

"""Shared test doubles for perflab tests.

These are stateful fakes for the boundaries of the perflab harness (the wall
clock, the XProf profiler, the MLCompass exporter) plus small real objects
(a tiny `BenchmarkSpec`, a real `XSpace` proto) so that tests can assert on
observable results instead of on mock interactions.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from perflab import export as export_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib

from tsl.profiler.protobuf import xplane_pb2


class FakeClock:
  """A manually advanced clock that stands in for `time.perf_counter`.

  Reading the clock never advances it, so measured durations depend only on
  what the code under test does between reads (e.g. a `CountingStep` that
  advances the clock), not on how many times the clock is read.
  """

  def __init__(self, start: float = 0.0):
    self.now = start
    self.sleeps: list[float] = []

  def __call__(self) -> float:
    return self.now

  def advance(self, seconds: float) -> None:
    if seconds < 0:
      raise ValueError(f"a monotonic clock cannot go back {seconds}s")
    self.now += seconds

  def sleep(self, seconds: float) -> None:
    """Drop-in for `time.sleep` that records the delay and advances time."""
    self.sleeps.append(seconds)
    self.advance(seconds)


class CountingStep:
  """A fake `Stepper` whose n-th step returns "out-n".

  Attributes:
    calls: Number of times the step function has run.
    raise_at: If set, the step raises RuntimeError on this call number.
    clock: Optional FakeClock to advance while stepping.
    step_seconds: Duration of each call, in order; calls beyond the end of the
      sequence take no time.
    init_seconds: Duration of `pre_warmup_init`.
    pre_warmup_init_calls: Number of `pre_warmup_init` calls.
    post_warmup_hook_calls: Number of `post_warmup_hook` calls.
  """

  def __init__(
      self,
      raise_at: int | None = None,
      clock: FakeClock | None = None,
      step_seconds: Sequence[float] = (),
      init_seconds: float = 0.0,
  ):
    self.calls = 0
    self.raise_at = raise_at
    self.clock = clock
    self.step_seconds = step_seconds
    self.init_seconds = init_seconds
    self.pre_warmup_init_calls = 0
    self.post_warmup_hook_calls = 0

  def _advance(self, seconds: float) -> None:
    if self.clock is not None:
      self.clock.advance(seconds)

  def __call__(self):
    self.calls += 1
    if self.calls <= len(self.step_seconds):
      self._advance(self.step_seconds[self.calls - 1])
    if self.raise_at is not None and self.calls == self.raise_at:
      raise RuntimeError("failure")
    return f"out-{self.calls}"

  def get_step_fn(self):
    return self

  def compile(self, **kwargs):
    pass

  def pre_warmup_init(self):
    self.pre_warmup_init_calls += 1
    self._advance(self.init_seconds)

  def post_warmup_hook(self):
    self.post_warmup_hook_calls += 1


class RecordingOps:
  """A fake `DeviceOps` that records events and simulates recompiles.

  Attributes:
    events: "await" / "reset" in the order they happened.
    awaited: Every value passed to `await_result`, in order.
    reset_count: Number of peak-memory resets.
    peak_mb: Value reported by `peak_memory_mb`.
    recompile_on_calls: `await_result` call numbers that count as a recompile.
    session_id_lookups: Number of device-time lookups by xprof session id.
  """

  def __init__(self, peak_mb=None, recompile_on_calls=None):
    self.events = []
    self.awaited = []
    self.reset_count = 0
    self.peak_mb = peak_mb
    self.recompile_on_calls = recompile_on_calls or set()
    self.calls = 0
    self.session_id_lookups = 0
    self._recompiles = 0

  def await_result(self, out):
    self.events.append("await")
    self.awaited.append(out)
    self.calls += 1
    if self.calls in self.recompile_on_calls:
      self._recompiles += 1

  def reset_peak_memory(self):
    self.events.append("reset")
    self.reset_count += 1

  def peak_memory_mb(self):
    return self.peak_mb

  def get_device_time(self, session_id=None, expected_steps=None):
    del session_id, expected_steps
    self.session_id_lookups += 1
    return -1.0

  def compile_count(self):
    return self._recompiles


class FakeXprofSession:
  """A stateful stand-in for `xprof_adapter.XprofSession`.

  Attributes:
    session_id: The id embedded in the returned XProf URL.
    xspace: The XSpace returned when the session ends.
    start_error: If set, `start_session` raises it.
    accepts_session_id: If False, `start_session` rejects a `session_id` keyword
      like older xprof builds do.
    returns_xspace: If False, `end_session_and_get_xspace_and_url` is
      unavailable, like older xprof builds.
    start_kwargs: The keyword arguments of the successful `start_session` call,
      or None if the session was never started.
    ended: Whether the session has been ended.
  """

  def __init__(
      self,
      *,
      session_id: str = "fake_session",
      xspace: xplane_pb2.XSpace | None = None,
      start_error: Exception | None = None,
      accepts_session_id: bool = True,
      returns_xspace: bool = True,
  ):
    self.session_id = session_id
    self.xspace = xspace
    self.start_error = start_error
    self.accepts_session_id = accepts_session_id
    self.returns_xspace = returns_xspace
    self.start_kwargs: dict[str, Any] | None = None
    self.ended = False

  def start_session(self, **kwargs) -> None:
    if self.start_error is not None:
      raise self.start_error
    if not self.accepts_session_id and "session_id" in kwargs:
      raise TypeError("unexpected keyword argument 'session_id'")
    if self.start_kwargs is not None:
      raise RuntimeError("session already started")
    self.start_kwargs = kwargs

  def _end(self) -> None:
    if self.start_kwargs is None:
      raise RuntimeError("session was never started")
    if self.ended:
      raise RuntimeError("session already ended")
    self.ended = True

  def end_session_and_get_xspace_and_url(
      self,
  ) -> tuple[xplane_pb2.XSpace | None, str]:
    if not self.returns_xspace:
      raise AttributeError("end_session_and_get_xspace_and_url")
    self._end()
    return self.xspace, f"http://xprof/?session_id={self.session_id}"

  def end_session_and_get_session_id(self) -> str:
    self._end()
    return self.session_id


class FakeXprofSessionFactory:
  """Replaces the `xprof_adapter.XprofSession` class in tests.

  Each call hands out the next queued session, or a new
  `FakeXprofSession(**default_kwargs)` once the queue is empty.

  Attributes:
    created: Every session handed out, in order.
  """

  # TODO(wan): revisit DECISION (b/568592811): a FakeXprofClient for
  # `XprofAnalysisClient` is left out until a test here needs one.

  def __init__(self, *sessions: FakeXprofSession, **default_kwargs: Any):
    self._queued = list(sessions)
    self._default_kwargs = default_kwargs
    self.created: list[FakeXprofSession] = []

  def __call__(self) -> FakeXprofSession:
    if self._queued:
      session = self._queued.pop(0)
    else:
      session = FakeXprofSession(**self._default_kwargs)
    self.created.append(session)
    return session

  @property
  def started(self) -> list[FakeXprofSession]:
    return [s for s in self.created if s.start_kwargs is not None]


def make_xspace(
    xla_module_durations_ps: Mapping[str, Sequence[int]],
) -> xplane_pb2.XSpace:
  """Builds a real XSpace with one "XLA Modules" line per plane.

  Args:
    xla_module_durations_ps: Maps a plane name (e.g. "/device:TPU:0" or
      "/host:CPU") to the durations, in picoseconds, of its XLA module events.

  Returns:
    The XSpace proto.
  """
  xspace = xplane_pb2.XSpace()
  for plane_name, durations_ps in xla_module_durations_ps.items():
    line = xspace.planes.add(name=plane_name).lines.add(name="XLA Modules")
    offset_ps = 0
    for duration_ps in durations_ps:
      line.events.add(offset_ps=offset_ps, duration_ps=duration_ps)
      offset_ps += duration_ps
  return xspace


class RecordingExporter:
  """Stands in for `export.export`, the MLCompass upload boundary.

  Attributes:
    exported: Every `BenchmarkData` that would have been uploaded, in order.
  """

  def __init__(self):
    self.exported: list[export_lib.BenchmarkData] = []

  def __call__(self, data: export_lib.BenchmarkData) -> None:
    self.exported.append(data)


def tiny_spec(
    name: str = "tiny_linear", **overrides: Any
) -> registry_lib.BenchmarkSpec:
  """Returns a real forward-only `BenchmarkSpec` over a 4x4 `nn.Linear`.

  The model and inputs are created on the CPU in float32, so the spec is cheap
  enough for end-to-end harness tests.

  Args:
    name: The spec name.
    **overrides: Other `BenchmarkSpec` fields, e.g. `compile_config`.
  """

  def factory(ctx):
    del ctx  # The tiny model always runs on the CPU.
    return torch.nn.Linear(4, 4), (torch.randn(2, 4),), {}

  fields = dict(
      name=name,
      factory=factory,
      stepper=step_lib.StepperType.FORWARD,
      dtype=target_lib.DType.FP32,
  )
  fields.update(overrides)
  return registry_lib.BenchmarkSpec(**fields)
