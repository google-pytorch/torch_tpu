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

"""Utilities for measuring a torch step function."""

from collections.abc import Callable, Iterator
import contextlib
import dataclasses
import gc
import random
import time
import urllib.parse

from absl import flags
from absl import logging
import numpy as np
from torch_tpu._internal.benchmarks import xprof_adapter
from perflab import device_ops as device_ops_lib
from perflab import metrics
from perflab import step_lib

MAX_WARMUP_STEPS = flags.DEFINE_integer(  # PYTHON_FLAG_OK=benchmark
    "max_warmup_steps", 20, "Maximum number of warmup steps.", lower_bound=0
)
MIN_WARMUP_STEPS = flags.DEFINE_integer(  # PYTHON_FLAG_OK=benchmark
    "min_warmup_steps", 10, "Minimum number of warmup steps.", lower_bound=0
)
POST_WARMUP_STEPS = flags.DEFINE_integer(  # PYTHON_FLAG_OK=benchmark
    "post_warmup_steps", 10, "Number of post-warmup steps.", lower_bound=0
)
XPROF_STEPS = flags.DEFINE_integer(  # PYTHON_FLAG_OK=benchmark
    "xprof_steps",
    2,
    "Number of steps to collect xprof for after warmup.",
    lower_bound=0,
)
# The number of untimed steps run between the pre-measurement garbage
# collection and peak-memory reset and the timed post-warmup steps. After that
# pause, fast host-bound steps run about 1.15x, 1.05x and 1.02x slower than
# steady state for three steps, then settle. A fixed count runs the same steps
# on every rank of a distributed run and in every run.
_POST_WARMUP_UNTIMED_STEPS = 4


def _do_post_warmup() -> bool:
  return POST_WARMUP_STEPS.value > 0


# Used to disable some of the checks if the user
# just wants a single warmup run.
def _is_warmup_only() -> bool:
  return (
      MIN_WARMUP_STEPS.value == 1
      and POST_WARMUP_STEPS.value == 0
      and MAX_WARMUP_STEPS.value == 1
  )


# This limit is enough to measure two steps for fsdp benchmark which is
# currently the longest running benchmark. Without the limit, the
# process uploading profile can fail health checks and die.
_PROFILE_MAX_BYTES = 500 * 1024 * 1024  # 500 MB


def _session_id_from_url(url: str | None) -> str | None:
  """Extracts the `session_id` query parameter from an xprof URL."""
  if not url:
    return None
  parsed = urllib.parse.urlparse(url)
  session_ids = urllib.parse.parse_qs(parsed.query).get("session_id")
  if session_ids:
    return session_ids[0]
  # Some xprof URLs carry the id as the final path segment.
  tail = parsed.path.rstrip("/").rsplit("/", 1)[-1]
  return tail or None


# xprof derives the user-visible session id from the request's session id,
# generating a random one only when the caller leaves it unset. Left unset,
# benchmarks that run under one test target end up sharing a session, so their
# XProf links all point at the same merged page. Supplying our own id per
# session keeps each benchmark's profile separate. The harness only runs
# single-host benchmarks, so there is no multi-host session to keep grouped.
_MAX_UINT64 = (1 << 64) - 1


def _new_xprof_session_id() -> int:
  """Returns an id that is unique across benchmarks, processes, and hosts."""
  return random.SystemRandom().randrange(1, _MAX_UINT64)


# External background profilers or concurrent profiling sessions may collect
# traces for up to 10 seconds while holding the process-wide
# tsl::profiler::ProfilerLock. Retrying with a fresh XprofSession instance
# allows the benchmark to wait out any transient lock contention and still
# capture its own XSpace, device time, and XProf URL.
#
# The retry is a single, late attempt rather than frequent polling: xprof's
# start_session holds the GIL while it waits for the session manager's mutex,
# and an external session holds that mutex while it stops the Python tracer,
# which needs the GIL. A start that lands in that window deadlocks the process
# permanently, and polling is likely to land in it as soon as the external
# capture ends. Sleeping releases the GIL so the external session can finish.
# A start can still deadlock if it happens to land in that window.
# TODO(b/571245040): Poll again once start_session releases the GIL.
_XPROF_START_MAX_ATTEMPTS = 2
_XPROF_START_RETRY_DELAY_SECONDS = 30.0
_ANOTHER_SESSION_ACTIVE_ERROR = "Another profiling session active"


class XprofContext:
  """A context manager for Xprof sessions that logs the session URL.

  Attributes:
    name: A name for the xprof session, used in logging.
    enable_xprof: Whether to enable xprof profiling. If False, the context
      manager does nothing.
    trace_only_host: Whether to trace only the host.
    session: The xprof_adapter.XprofSession object if xprof is enabled,
      otherwise None.
    session_id: The ID of the xprof session after it has ended, if xprof was
      enabled. Otherwise None.
    xspace: The XSpace captured by this session, if xprof was enabled and the
      trace could be retrieved. This holds only this process's own trace, which
      is why device time is derived from it rather than by re-fetching by
      session id.
    overhead_seconds: The wall time spent starting and ending the session.
    clock: Returns the current time in seconds; used to measure
      `overhead_seconds`.
  """

  def __init__(
      self,
      *,
      name: str,
      enable_xprof: bool,
      trace_only_host: bool = False,
      clock: Callable[[], float] = time.perf_counter,
  ):
    self.name = name
    self.enable_xprof = enable_xprof
    self.trace_only_host = trace_only_host
    self.clock = clock
    self.session = None
    self.session_id = None
    self.xspace = None
    self.overhead_seconds = 0.0

  def _start_session(
      self,
      *,
      host_trace_level: int,
      enable_python_tracer: bool,
      kwargs: dict[str, str],
  ):
    """Creates and starts a new XprofSession."""
    session = xprof_adapter.XprofSession()
    try:
      session.start_session(
          host_trace_level=host_trace_level,
          enable_python_tracer=enable_python_tracer,
          host_cpu_profile=True,
          response_max_bytes=_PROFILE_MAX_BYTES,
          session_id=_new_xprof_session_id(),
          **kwargs,
      )
    except TypeError:
      # Older xprof builds do not accept an explicit session id. Sessions
      # may then be shared between benchmarks, which affects only the URLs;
      # the reported device time still comes from this process's own XSpace.
      logging.warning(
          "xprof start_session does not accept session_id; xprof links may"
          " be shared between benchmarks.",
          exc_info=True,
      )
      session.start_session(
          host_trace_level=host_trace_level,
          enable_python_tracer=enable_python_tracer,
          host_cpu_profile=True,
          response_max_bytes=_PROFILE_MAX_BYTES,
          **kwargs,
      )
    return session

  def __enter__(self):
    if self.enable_xprof:
      start_time = self.clock()
      kwargs = {}
      host_trace_level = 3
      enable_python_tracer = True
      if self.trace_only_host:
        kwargs["trace_mode"] = "TRACE_ONLY_HOST"
        host_trace_level = 1
        enable_python_tracer = False
      for attempt in range(1, _XPROF_START_MAX_ATTEMPTS + 1):
        try:
          self.session = self._start_session(
              host_trace_level=host_trace_level,
              enable_python_tracer=enable_python_tracer,
              kwargs=kwargs,
          )
          break
        except Exception as e:  # pylint: disable=broad-except
          if (
              _ANOTHER_SESSION_ACTIVE_ERROR not in str(e)
              or attempt == _XPROF_START_MAX_ATTEMPTS
          ):
            raise
          logging.warning(
              "xprof session for %s blocked by another active profiling"
              " session (attempt %d/%d); retrying in %.1fs: %s",
              self.name,
              attempt,
              _XPROF_START_MAX_ATTEMPTS,
              _XPROF_START_RETRY_DELAY_SECONDS,
              e,
          )
          time.sleep(_XPROF_START_RETRY_DELAY_SECONDS)
      self.overhead_seconds += self.clock() - start_time
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    # End the session and log the URL if it was started. If there is an
    # exception, we let the exception propagate.
    if not self.session:
      return

    start_time = self.clock()
    # Retrieving the XSpace together with the URL ends the session exactly
    # once and keeps the trace local to this process. Fall back to just the
    # session id if the combined call is unavailable, accepting that device
    # time will then be unavailable rather than wrong.
    try:
      xspace, url = self.session.end_session_and_get_xspace_and_url()
      self.xspace = xspace
      self.session_id = _session_id_from_url(url)
    except (AttributeError, TypeError, ValueError):
      logging.warning(
          "end_session_and_get_xspace_and_url unavailable; falling back to"
          " session id only for %s. Device time will be reported as"
          " unavailable rather than risking another benchmark's value.",
          self.name,
          exc_info=True,
      )
      self.session_id = self.session.end_session_and_get_session_id()
    self.overhead_seconds += self.clock() - start_time

    logging.info(
        "%s xprof URL: http://xprof/?session_id=%s",
        self.name,
        self.session_id,
    )


def _calculate_avg_device_time(
    device_ops: device_ops_lib.DeviceOps,
    xspace,
    xprof_steps: int,
) -> float:
  """Calculates the average device time in seconds across profiling steps.

  Args:
    device_ops: The device operations to use.
    xspace: The XSpace captured by this process's own xprof session. Using the
      local XSpace, rather than re-fetching by session id, is what keeps one
      benchmark from reporting another benchmark's device time when xprof
      assigns them a shared session id.
    xprof_steps: Number of steps that were profiled.

  Returns:
    The average device time per step in seconds, or -1.0 if unavailable.
  """
  if xspace is None or xprof_steps <= 0:
    return -1.0
  if not hasattr(device_ops, "get_device_time_from_xspace"):
    return -1.0
  total_device_time = device_ops.get_device_time_from_xspace(
      xspace, expected_steps=xprof_steps
  )
  if total_device_time == -1.0:
    return -1.0
  return total_device_time / xprof_steps


class PostWarmupRecompileError(AssertionError):
  """Raised when a recompilation happens during the post warmup loop."""


class _ProfilingTime:
  """Accumulates the wall time spent only on profiling.

  measure() leaves it out of the e2e wall time, so that runs with and without
  xprof report comparable wall times.
  """

  def __init__(self) -> None:
    self.seconds = 0.0


@contextlib.contextmanager
def gc_disabled() -> Iterator[None]:
  """Disable cyclic GC for the duration of the block, restoring prior state on exit.

  Cyclic collection landing mid-loop adds a latency spike to measurements. This
  defers
  the garbage collection to outside the measurement loop. Refcounting still runs
  inside the block, so acyclic garbage is still freed.

  NOTE: this does not call gc.collect(). The pre-timing collect must happen
  before
  the peak-memory reset, which is outside this scope of this context manager.
  """
  was_enabled = gc.isenabled()
  gc.disable()
  try:
    yield
  finally:
    if was_enabled:
      gc.enable()


def _get_warmup_overhead(timings: np.ndarray, num_warmup_steps: int) -> float:
  """Calculates the warmup overhead in seconds from the timings of the warmup runs.

  Args:
    timings: An array with size max warmup steps that contains the time it took
      to run individual steps.
    num_warmup_steps: The length of the timings array to use for warmup overhead
      calculations.

  Returns:
    The warmup overhead in seconds.
  """

  if _is_warmup_only():
    return timings[0]

  if len(timings) < num_warmup_steps:
    raise RuntimeError(
        "Timings array is smaller than the number of warmup steps. Timings"
        f" array has length {len(timings)}, but num_warmup_steps is"
        f" {num_warmup_steps}"
    )

  return (
      np.sum(timings[:num_warmup_steps])
      - timings[num_warmup_steps - 1] * num_warmup_steps
  )


def _warmup_run(
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    name: str,
    enable_xprof: bool = False,
    profiling_time: _ProfilingTime | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> metrics.WarmupRunResult:
  """Runs the model to warmup the caches.

  The number of iteration is controlled by the flags MIN_WARMUP_STEPS and
  MAX_WARMUP_STEPS.

  The benchmark function compilations have not stabilized if the number of cache
  misses is not the same for two consecutive steps. If the benchmark function
  compilations have not stabilized after MAX_WARMUP_STEPS runs, then the
  warmup result is not calculated and RuntimeError is raised.

  Args:
    stepper: The stepper to run.
    device_ops: The device operations to use.
    name: The name of the benchmark.
    enable_xprof: Whether to enable xprof profiling.
    profiling_time: If set, accumulates the wall time spent starting and ending
      the xprof session.
    clock: Returns the current time in seconds; used to time each step.

  Returns:
    A WarmupRunResult instance containing the number of warmup steps, the time
    taken for the first step, and the warmup overhead in seconds.
  """
  timings = np.zeros(MAX_WARMUP_STEPS.value, dtype=np.float64)
  compile_count = np.zeros(MAX_WARMUP_STEPS.value, dtype=np.int64)
  num_warmup_steps = None

  # gc is explicitly disabled below to prevent GC pauses from affecting the
  # warmup measurements. Collect garbage once before the warmup loop.
  gc.collect()

  with gc_disabled(), XprofContext(
      name="warmup_run",
      enable_xprof=enable_xprof,
      trace_only_host=True,
      clock=clock,
  ) as warmup_run_context:
    # The first step is timed from before get_step_fn(): the compilation
    # belongs to it whether a stepper compiles on the first call or ahead of
    # it in get_step_fn() (e.g. the single-trace training stepper).
    start_time = clock()
    step_fn = stepper.get_step_fn()
    for step in range(MAX_WARMUP_STEPS.value):
      if step > 0:
        start_time = clock()
      out = step_fn()
      device_ops.await_result(out)
      timings[step] = clock() - start_time
      # Outside the timed region: fail instead of timing a failed step.
      device_ops.check_result(out)
      del out
      # E.g. reset kv cache to its post-prefill state.
      stepper.post_warmup_hook()
      compile_count[step] = device_ops.compile_count()

      if (
          step > 0
          and step >= MIN_WARMUP_STEPS.value - 1
          and compile_count[step] == compile_count[step - 1]
      ):
        num_warmup_steps = step + 1
        break

  if profiling_time is not None:
    profiling_time.seconds += warmup_run_context.overhead_seconds

  if num_warmup_steps is None:
    if _is_warmup_only():
      num_warmup_steps = 1
    else:
      raise RuntimeError(
          "Benchmark function compilations have not stabilized after"
          f" {MAX_WARMUP_STEPS.value} warmup runs."
      )

  logging.info("Warmup Timings for %s: %s", name, timings)
  logging.info("Warmup compile counts for %s: %s", name, compile_count)
  warmup_session_xprof_url = None
  if enable_xprof and warmup_run_context.session_id:
    warmup_session_xprof_url = (
        f"http://xprof/?session_id={warmup_run_context.session_id}"
    )

  return metrics.WarmupRunResult(
      num_warmup_steps=num_warmup_steps,
      first_step_time_seconds=float(timings[0]),
      warmup_overhead_seconds=float(
          _get_warmup_overhead(timings, num_warmup_steps)
      ),
      warmup_session_xprof_url=warmup_session_xprof_url,
  )


def _post_warmup_run(
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    name: str,
    enable_xprof: bool = False,
    profiling_time: _ProfilingTime | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> metrics.PostWarmupRunResult:
  """Runs the model once after the warmup is complete.

  Args:
    stepper: The stepper to run.
    device_ops: The device operations to use.
    name: The name of the benchmark.
    enable_xprof: Whether to enable xprof profiling.
    profiling_time: If set, accumulates the wall time of the profiled steps,
      including starting and ending their xprof session.
    clock: Returns the current time in seconds; used to time each step.

  Returns:
    A PostWarmupRunResult instance containing the average step time and peak
    device memory usage.
  """

  timings = np.zeros(POST_WARMUP_STEPS.value, dtype=np.float64)
  xprof_timings = np.zeros(XPROF_STEPS.value, dtype=np.float64)
  # gc is explicitly disabled below to prevent GC pauses from affecting the
  # measurements. Collect garbage once before the timed loop.
  gc.collect()
  device_ops.reset_peak_memory()

  compile_count_before = device_ops.compile_count()

  step_fn = stepper.get_step_fn()

  def _run_single_step(
      expected_compile_count: int,
      *,
      run_post_hook: bool = True,
      trace_step_num: int | None = None,
  ) -> float:
    trace_ctx = (
        xprof_adapter.TraceMe("post_warmup_step", step_num=trace_step_num)
        if trace_step_num is not None
        else contextlib.nullcontext()
    )
    with trace_ctx:
      start_time = clock()
      out = step_fn()
      device_ops.await_result(out)
      elapsed = clock() - start_time
    # Outside the timed region: fail instead of timing a failed step.
    device_ops.check_result(out)
    del out
    if run_post_hook:
      stepper.post_warmup_hook()

    step_compile_count = device_ops.compile_count()
    if step_compile_count != expected_compile_count:
      raise PostWarmupRecompileError(
          "Recompilation happened inside the post warmup loop. Expected"
          f" {expected_compile_count}, got {step_compile_count}"
      )
    return elapsed

  def _run_single_event_timed_step(expected_compile_count: int) -> float:
    device_elapsed = device_ops.measure_step_device_time(step_fn)
    stepper.post_warmup_hook()

    step_compile_count = device_ops.compile_count()
    if step_compile_count != expected_compile_count:
      raise PostWarmupRecompileError(
          "Recompilation happened inside the post warmup loop. Expected"
          f" {expected_compile_count}, got {step_compile_count}"
      )
    return device_elapsed

  with gc_disabled():
    # A pause of a few tenths of a second in the step stream, such as a full
    # garbage collection, lets the host's caches lose the step's working set,
    # so that host-bound steps run slower for a few steps. Run a fixed number
    # of untimed steps first, so that the timed steps start warm. They count
    # towards the peak memory, as the timed steps do.
    for _ in range(_POST_WARMUP_UNTIMED_STEPS):
      _run_single_step(compile_count_before)
    for step in range(POST_WARMUP_STEPS.value):
      timings[step] = _run_single_step(compile_count_before)

  memory_usage = device_ops.peak_memory_mb()

  post_warmup_run_session_xprof_url = None
  xspace = None
  avg_device_time = -1.0
  if device_ops.supports_event_device_time() and XPROF_STEPS.value > 0:
    profiling_start_time = clock()
    event_device_timings = np.zeros(XPROF_STEPS.value, dtype=np.float64)
    with gc_disabled():
      for xprof_step in range(XPROF_STEPS.value):
        event_device_timings[xprof_step] = _run_single_event_timed_step(
            compile_count_before
        )
    if profiling_time is not None:
      profiling_time.seconds += clock() - profiling_start_time
    valid_event_timings = event_device_timings[event_device_timings >= 0.0]
    if valid_event_timings.size == XPROF_STEPS.value:
      avg_device_time = float(np.mean(valid_event_timings))
    logging.info(
        "Post Warmup Event Device Timings for %s: %s",
        name,
        event_device_timings,
    )
  elif enable_xprof and XPROF_STEPS.value > 0:
    profiling_start_time = clock()
    with gc_disabled():
      with XprofContext(
          name="post_warmup_run",
          enable_xprof=enable_xprof,
          trace_only_host=False,
          clock=clock,
      ) as xprof_context:
        for xprof_step in range(XPROF_STEPS.value):
          xprof_timings[xprof_step] = _run_single_step(
              compile_count_before,
              run_post_hook=(xprof_step < XPROF_STEPS.value - 1),
              trace_step_num=xprof_step,
          )
      stepper.post_warmup_hook()
    if profiling_time is not None:
      profiling_time.seconds += clock() - profiling_start_time

    xspace = xprof_context.xspace
    if xprof_context.session_id:
      post_warmup_run_session_xprof_url = (
          f"http://xprof/?session_id={xprof_context.session_id}"
      )
    avg_device_time = _calculate_avg_device_time(
        device_ops, xspace, XPROF_STEPS.value
    )

  logging.info("Post Warmup Timings for %s: %s", name, timings)
  if enable_xprof and not device_ops.supports_event_device_time():
    logging.info("Post Warmup Xprof Timings for %s: %s", name, xprof_timings)

  return metrics.PostWarmupRunResult(
      post_warmup_step_time_seconds=float(np.mean(timings)),
      peak_device_memory_mb=memory_usage,
      post_warmup_run_session_xprof_url=post_warmup_run_session_xprof_url,
      average_post_warmup_device_time_seconds=avg_device_time,
  )


def measure(
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    name: str,
    enable_xprof: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> metrics.PerformanceMetrics:
  """Warms up and measures a stepper.

  Args:
    stepper: The stepper to measure.
    device_ops: The device operations to use.
    name: The name of the benchmark, used in logs.
    enable_xprof: Whether to profile the warmup and extra post-warmup steps.
    clock: Returns the current time in seconds. Tests inject a fake clock to
      control the measured durations.

  Returns:
    The measured metrics. The e2e wall time leaves out the wall time spent only
    on profiling, so that runs with and without xprof report comparable values.
  """
  result_kwargs = {}
  profiling_time = _ProfilingTime()
  start_time = clock()

  stepper.pre_warmup_init()

  # Collect once, then freeze the survivors (the model, its inputs, the
  # imported modules, ...). They live through the measurement, and leaving
  # them out of the later collections keeps the post-warmup collection from
  # traversing the whole heap. That traversal takes ~0.4s, evicts the host's
  # caches and pauses the step stream, which made the first timed steps of
  # host-bound models up to ~2.5x slower than the last warmup steps.
  gc.collect()
  gc.freeze()
  try:
    result_kwargs |= dataclasses.asdict(
        _warmup_run(
            stepper,
            device_ops,
            name=name,
            enable_xprof=enable_xprof,
            profiling_time=profiling_time,
            clock=clock,
        )
    )

    if _do_post_warmup():
      result_kwargs |= dataclasses.asdict(
          _post_warmup_run(
              stepper,
              device_ops,
              name=name,
              enable_xprof=enable_xprof,
              profiling_time=profiling_time,
              clock=clock,
          )
      )
  finally:
    gc.unfreeze()

  return metrics.PerformanceMetrics(
      e2e_wall_time_seconds=(clock() - start_time - profiling_time.seconds),
      **result_kwargs,
  )
