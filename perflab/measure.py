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

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import dataclasses
import gc
import math
import random
import resource
import time
import urllib.parse

from absl import flags
from absl import logging
import numpy as np
from torch_tpu._internal.benchmarks import xprof_adapter
from perflab import common
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
    "post_warmup_steps",
    10,
    "Minimum number of timed post-warmup steps. 0 skips the post-warmup run.",
    lower_bound=0,
)
POST_WARMUP_MIN_SECONDS = flags.DEFINE_float(  # PYTHON_FLAG_OK=benchmark
    "post_warmup_min_seconds",
    2.0,
    "Minimum timed wall time of the post-warmup steps. More than"
    " --post_warmup_steps steps run until their step times add up to at least"
    " this, at most --post_warmup_max_steps steps.",
    lower_bound=0.0,
)
POST_WARMUP_MAX_STEPS = flags.DEFINE_integer(  # PYTHON_FLAG_OK=benchmark
    "post_warmup_max_steps",
    2000,
    "Maximum number of timed post-warmup steps run to reach"
    " --post_warmup_min_seconds. --post_warmup_steps steps always run.",
    lower_bound=0,
)
XPROF_STEPS = flags.DEFINE_integer(  # PYTHON_FLAG_OK=benchmark
    "xprof_steps",
    2,
    "Number of steps to collect xprof for after warmup.",
    lower_bound=0,
)
PIPELINED_STEPS = flags.DEFINE_bool(  # PYTHON_FLAG_OK=benchmark
    "pipelined_steps",
    True,
    "Whether to also time post-warmup steps dispatched back to back, without"
    " waiting for each step's result, for the pipelined step time and"
    " throughput. Only for steppers that support it (the TorchTPU forward and"
    " training steppers; not decode or torchax, which keep the throughput of"
    " the synchronized steps). If false, those steppers omit throughput"
    " metrics.",
)

# The number of untimed steps run between the pre-measurement garbage
# collection and peak-memory reset and the timed post-warmup steps. After that
# pause, fast host-bound steps run about 1.15x, 1.05x and 1.02x slower than
# steady state for three steps, then settle. A fixed count runs the same steps
# on every rank of a distributed run and in every run.
_POST_WARMUP_UNTIMED_STEPS = 4
# Number of isolated single-step peak-memory probes run after a long
# synchronized timing window (> _PEAK_MEMORY_SHORT_WINDOW_MAX_STEPS) and
# strictly before pipelined steps. Taking the median across isolated resets
# decouples peak_device_memory_mb from rare host preemption stalls that deepen
# async execution queues across multi-hundred-step windows. Short windows run 1
# isolated step instead so slow models do not pay a 5-step wall-time tax.
_PEAK_MEMORY_PROBE_STEPS = 5
_PEAK_MEMORY_SHORT_WINDOW_MAX_STEPS = 15
_SYNC_WINDOW_PEAK_MEMORY_KEY = "sync_window_peak_device_memory_mb"
_ALLOCATED_MEMORY_DRIFT_KEY = "allocated_memory_drift_mb"
# Windows of up to this many timed steps log every step time at the default
# verbosity.
_MAX_LOGGED_TIMINGS = 50


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
      kwargs: Mapping[str, str],
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
    capture_xprof: bool | None = None,
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
    capture_xprof: Whether this process opens the xprof session. Defaults to
      `enable_xprof`; ignored when `enable_xprof` is False.

  Returns:
    A WarmupRunResult instance containing the number of warmup steps, the time
    taken for the first step, and the warmup overhead in seconds.
  """
  capture_xprof = enable_xprof and capture_xprof is not False
  timings = np.zeros(MAX_WARMUP_STEPS.value, dtype=np.float64)
  compile_count = np.zeros(MAX_WARMUP_STEPS.value, dtype=np.int64)
  num_warmup_steps = None

  # gc is explicitly disabled below to prevent GC pauses from affecting the
  # warmup measurements. Collect garbage once before the warmup loop.
  gc.collect()
  # As in `_post_warmup_run`, the first agreement compiles its collective (on
  # TPU), so run one before the compile counts that the loop compares. It also
  # aligns the processes, so the first step's time leaves out their setup skew.
  device_ops.agree_on_step_count(0)

  with gc_disabled(), XprofContext(
      name="warmup_run",
      enable_xprof=capture_xprof,
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

      # SPMD invariant: the short-circuited `step > 0 and step >= MIN - 1`
      # depends only on `step` and flags, so every process reaches the
      # agreement at the same steps. Never put a process-local condition
      # before it.
      if (
          step > 0
          and step >= MIN_WARMUP_STEPS.value - 1
          and _all_processes_agree(
              device_ops, bool(compile_count[step] == compile_count[step - 1])
          )
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
  if capture_xprof and warmup_run_context.session_id:
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


def _timings_summary(timings: np.ndarray) -> str:
  """Returns the count, total and distribution of step times, in seconds."""
  p10, median, p90 = np.percentile(timings, [10, 50, 90])
  return (
      f"count={timings.size} total={np.sum(timings):.6g}"
      f" mean={np.mean(timings):.6g} median={median:.6g}"
      f" min={np.min(timings):.6g} p10={p10:.6g} p90={p90:.6g}"
      f" max={np.max(timings):.6g}"
  )


def _all_processes_agree(
    device_ops: device_ops_lib.DeviceOps, value: bool
) -> bool:
  """Returns whether `value` is True in every process of the benchmark.

  Reuses the step count agreement, which returns the largest count proposed:
  each process proposes 1 if its `value` is False and 0 if it is True, so the
  agreed count is 0 only if `value` is True in every process.

  Args:
    device_ops: The device operations, used to agree across processes.
    value: This process's value.
  """
  return device_ops.agree_on_step_count(0 if value else 1) == 0


def _run_timed_window(
    run_steps: Callable[[int], Sequence[float]],
    device_ops: device_ops_lib.DeviceOps,
    *,
    initial_steps: int | None = None,
) -> np.ndarray:
  """Runs timed steps until the window is long enough, and returns their times.

  The window has at least --post_warmup_steps steps, and more until their times
  add up to --post_warmup_min_seconds, up to --post_warmup_max_steps steps. A
  few fast steps would otherwise make a short and noisy window, and miss slower
  steady states such as a GPU clocking down under sustained load.

  The steps run in rounds: the first round runs `initial_steps` steps, and each
  further round runs as many more as the mean step time so far says are needed
  to reach --post_warmup_min_seconds. The processes of a distributed benchmark
  agree on the total step count of every round, including the first, so that
  they all run the same steps.

  Args:
    run_steps: Runs the given number of steps and returns the time of each, in
      seconds.
    device_ops: The device operations, used to agree on the step count.
    initial_steps: The number of steps of the first round, between
      --post_warmup_steps and --post_warmup_max_steps. Defaults to
      --post_warmup_steps.

  Returns:
    The times of the timed steps, in seconds.
  """
  min_steps = POST_WARMUP_STEPS.value
  max_steps = max(min_steps, POST_WARMUP_MAX_STEPS.value)
  target_steps = min_steps
  if initial_steps is not None:
    target_steps = min(max_steps, max(min_steps, initial_steps))
  timings: list[float] = []
  while True:
    target_steps = device_ops.agree_on_step_count(target_steps)
    if target_steps <= len(timings):
      return np.array(timings, dtype=np.float64)
    timings.extend(run_steps(target_steps - len(timings)))
    timed_seconds = sum(timings)
    target_steps = len(timings)
    if timed_seconds < POST_WARMUP_MIN_SECONDS.value:
      mean_step_seconds = max(timed_seconds / len(timings), 1e-9)
      target_steps = min(
          max_steps,
          max(
              len(timings) + 1,
              math.ceil(POST_WARMUP_MIN_SECONDS.value / mean_step_seconds),
          ),
      )


def _release_device_memory(device_ops: device_ops_lib.DeviceOps) -> None:
  """Frees what a failed step left behind."""
  gc.collect()
  device_ops.release_cached_memory()


def _run_pipelined_window(
    step_fn: step_lib.StepFn,
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    expected_compile_count: int,
    synchronized_step_seconds: float,
    clock: Callable[[], float],
) -> tuple[float, int]:
  """Times steps dispatched back to back and returns the time per step.

  Unlike the synchronized steps, which each wait for their result, these steps
  are dispatched without waiting, so the host dispatches the next steps while
  the device runs the previous ones, as a training or serving loop does. Each
  round of steps waits only for its last step's result. Each step's output is
  released before the next step is dispatched, so that at most one step's
  output is alive at a time, as in the synchronized steps. Its work is started
  first (`launch_result`), as a backend that defers work would otherwise drop
  the work of an output that is released unread and not run the step at all.

  The window grows in rounds like the synchronized window (see
  `_run_timed_window`), until the pipelined steps add up to
  --post_warmup_min_seconds. The first round is sized from the synchronized
  step time, so that steps that pipelining does not speed up take one round.

  Args:
    step_fn: The step function.
    stepper: The stepper, whose post-warmup hook runs after each step.
    device_ops: The device operations to use.
    expected_compile_count: The compile count that the steps must not change.
    synchronized_step_seconds: The median synchronized step time.
    clock: Returns the current time in seconds; used to time each round.

  Returns:
    The mean time per step in seconds, and the number of steps.

  Raises:
    PostWarmupRecompileError: If the steps recompiled.
  """

  def run_steps(steps: int) -> list[float]:
    out = None
    # Start from an idle device, so that no earlier work is timed.
    device_ops.await_result(None)
    start_time = clock()
    for _ in range(steps):
      # Release the previous step's output first: holding it while the next
      # step allocates its own would double the peak memory of the outputs.
      out = None
      out = step_fn()
      device_ops.launch_result(out)
      stepper.post_warmup_hook()
    device_ops.await_result(out)
    elapsed = clock() - start_time
    device_ops.check_result(out)
    del out
    _check_no_recompile(
        device_ops, expected_compile_count, "the pipelined post warmup steps"
    )
    return [elapsed / steps] * steps

  initial_steps = None
  if synchronized_step_seconds > 0:
    initial_steps = math.ceil(
        POST_WARMUP_MIN_SECONDS.value / synchronized_step_seconds
    )
  timings = _run_timed_window(
      run_steps, device_ops, initial_steps=initial_steps
  )
  if not timings.size:
    return 0.0, 0
  return float(np.mean(timings)), int(timings.size)


def _check_no_recompile(
    device_ops: device_ops_lib.DeviceOps,
    expected_compile_count: int,
    phase: str = "the post warmup loop",
) -> None:
  """Raises `PostWarmupRecompileError` if `compile_count()` changed."""
  step_compile_count = device_ops.compile_count()
  if step_compile_count != expected_compile_count:
    raise PostWarmupRecompileError(
        f"Recompilation happened inside {phase}. Expected"
        f" {expected_compile_count}, got {step_compile_count}"
    )


def _run_single_step(
    step_fn: step_lib.StepFn,
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    expected_compile_count: int,
    clock: Callable[[], float],
    run_post_hook: bool = True,
    trace_step_num: int | None = None,
    phase: str = "the post warmup loop",
) -> float:
  """Runs one synchronized step and returns its wall time in seconds."""
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
  _check_no_recompile(device_ops, expected_compile_count, phase)
  return elapsed


def _measure_sync_step_time(
    step_fn: step_lib.StepFn,
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    name: str,
    expected_compile_count: int,
    clock: Callable[[], float],
) -> tuple[float, int, dict[str, float]]:
  """Measures median synchronized step latency and timed-window diagnostics.

  Args:
    step_fn: The step function.
    stepper: The stepper, whose post-warmup hook runs after each step.
    device_ops: The device operations to use.
    name: The name of the benchmark, used in logs.
    expected_compile_count: The compile count that the steps must not change.
    clock: Returns the current time in seconds.

  Returns:
    `(median_step_time_seconds, num_timed_steps, diagnostics)`, where
    `diagnostics` records sync-window `mean`, `p10`, `p90`, `max`, the
    process-wide (`resource.RUSAGE_SELF`) involuntary context switch delta
    (`ru_nivcsw`), and (when supported by `device_ops`) the timed-window peak
    memory (`sync_window_peak_device_memory_mb`) and live allocation drift
    (`allocated_memory_drift_mb`).
  """
  # gc is explicitly disabled below to prevent GC pauses from affecting the
  # measurements. Collect garbage and reset peak memory once before the window.
  gc.collect()
  device_ops.reset_peak_memory()
  alloc_before = device_ops.allocated_memory_mb()
  with gc_disabled():
    # A pause of a few tenths of a second in the step stream, such as a full
    # garbage collection, lets the host's caches lose the step's working set,
    # so that host-bound steps run slower for a few steps. Run a fixed number
    # of untimed steps first, so that the timed steps start warm. They count
    # towards the sync-window peak memory, as the timed steps do.
    for _ in range(_POST_WARMUP_UNTIMED_STEPS):
      _run_single_step(
          step_fn,
          stepper,
          device_ops,
          expected_compile_count=expected_compile_count,
          clock=clock,
      )
    ru_nivcsw_before = resource.getrusage(resource.RUSAGE_SELF).ru_nivcsw
    timings = _run_timed_window(
        lambda steps: [
            _run_single_step(
                step_fn,
                stepper,
                device_ops,
                expected_compile_count=expected_compile_count,
                clock=clock,
            )
            for _ in range(steps)
        ],
        device_ops,
    )
    ru_nivcsw_after = resource.getrusage(resource.RUSAGE_SELF).ru_nivcsw

  sync_window_peak_mb = device_ops.peak_memory_mb()
  gc.collect()
  alloc_after = device_ops.allocated_memory_mb()

  if not timings.size:
    return 0.0, 0, {}

  logging.info(
      "Post Warmup Timings summary for %s: %s", name, _timings_summary(timings)
  )
  # Log lines are truncated past a few KB, so long windows only log every step
  # time at a higher verbosity.
  if timings.size <= _MAX_LOGGED_TIMINGS:
    logging.info("Post Warmup Timings for %s: %s", name, timings)
  else:
    logging.vlog(
        1,
        "Post Warmup Timings for %s: %s",
        name,
        " ".join(f"{t:.6g}" for t in timings),
    )

  median_step_time = float(np.median(timings))
  p10, p90 = np.percentile(timings, [10, 90])
  diagnostics = {
      "post_warmup_step_time_mean_seconds": float(np.mean(timings)),
      "post_warmup_step_time_p10_seconds": float(p10),
      "post_warmup_step_time_p90_seconds": float(p90),
      "post_warmup_step_time_max_seconds": float(np.max(timings)),
      "ru_nivcsw": float(ru_nivcsw_after - ru_nivcsw_before),
  }
  if sync_window_peak_mb is not None:
    diagnostics[_SYNC_WINDOW_PEAK_MEMORY_KEY] = float(sync_window_peak_mb)
  if alloc_before is not None and alloc_after is not None:
    diagnostics[_ALLOCATED_MEMORY_DRIFT_KEY] = float(alloc_after - alloc_before)
  return median_step_time, int(timings.size), diagnostics


def _measure_peak_memory(
    step_fn: step_lib.StepFn,
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    expected_compile_count: int,
    num_synchronized_steps: int,
    clock: Callable[[], float],
) -> float | None:
  """Measures steady-state single-step peak device memory across isolated steps.

  Runs strictly before pipelined steps, resetting the peak-memory counter before
  each step, reading `peak_memory_mb()` strictly before
  `stepper.post_warmup_hook()`, and returning the median peak across the probe
  steps. Short synchronized windows (`<= _PEAK_MEMORY_SHORT_WINDOW_MAX_STEPS`)
  run 1 isolated step; longer windows run `_PEAK_MEMORY_PROBE_STEPS` steps.
  Because `num_synchronized_steps` comes from `agree_on_step_count()`, all ranks
  run the same number of probe steps so distributed collectives stay in
  lockstep.

  Note: Per-step isolation requires backend `reset_peak_memory()` support; on
  backends whose runtime cannot reset peak stats (e.g. JAX PJRT without
  `reset_memory_stats`), each probe read repeats the process high-water mark.

  Args:
    step_fn: The step function.
    stepper: The stepper, whose post-warmup hook runs after each step peak read.
    device_ops: The device operations to use.
    expected_compile_count: The compile count that the steps must not change.
    num_synchronized_steps: Agreed number of timed synchronized steps completed.
    clock: Returns the current time in seconds.

  Returns:
    Median single-step peak device memory in MB across the probe steps, or None
    if `device_ops.peak_memory_mb()` is unsupported on any probe step.
  """
  probe_steps = (
      1
      if num_synchronized_steps <= _PEAK_MEMORY_SHORT_WINDOW_MAX_STEPS
      else _PEAK_MEMORY_PROBE_STEPS
  )
  probe_peaks: list[float | None] = []
  for probe_idx in range(probe_steps):
    # _measure_sync_step_time already collected garbage right before returning,
    # so skip a back-to-back collection before the first probe step.
    if probe_idx > 0:
      gc.collect()
    device_ops.reset_peak_memory()
    with gc_disabled():
      _run_single_step(
          step_fn,
          stepper,
          device_ops,
          expected_compile_count=expected_compile_count,
          clock=clock,
          run_post_hook=False,
          phase="the peak-memory probe",
      )
    probe_peaks.append(device_ops.peak_memory_mb())
    stepper.post_warmup_hook()
    _check_no_recompile(
        device_ops, expected_compile_count, "the peak-memory probe"
    )
  if not probe_peaks or any(p is None for p in probe_peaks):
    return None
  return float(np.median(np.asarray(probe_peaks, dtype=np.float64)))


def _measure_pipelined_step_time(
    step_fn: step_lib.StepFn,
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    name: str,
    expected_compile_count: int,
    synchronized_step_seconds: float,
    num_synchronized_steps: int,
    clock: Callable[[], float],
) -> tuple[float, int]:
  """Measures back-to-back pipelined step time, or (-1.0, 0) if skipped/OOM."""
  if (
      not PIPELINED_STEPS.value
      or not stepper.supports_pipelined_steps
      or num_synchronized_steps <= 0
  ):
    return -1.0, 0
  # Best effort: the synchronized results stand even if the pipelined steps
  # run out of memory, e.g. because the device's runtime keeps more steps'
  # buffers in flight.
  try:
    with gc_disabled():
      pipelined_step_time, num_pipelined_steps = _run_pipelined_window(
          step_fn,
          stepper,
          device_ops,
          expected_compile_count=expected_compile_count,
          synchronized_step_seconds=synchronized_step_seconds,
          clock=clock,
      )
  except Exception as e:  # pylint: disable=broad-except
    if not common.is_out_of_memory(e):
      raise
    logging.warning(
        "Pipelined post warmup steps for %s ran out of memory; not"
        " reporting the pipelined step time nor the throughput: %s",
        name,
        e,
    )
    del e
    _release_device_memory(device_ops)
    return -1.0, 0

  # Only the whole pipelined window is timed, not its steps.
  logging.info(
      "Post Warmup pipelined step time for %s: %.6gs over %d steps in"
      " %.6gs (synchronized: %.6gs)",
      name,
      pipelined_step_time,
      num_pipelined_steps,
      pipelined_step_time * num_pipelined_steps,
      synchronized_step_seconds,
  )
  return pipelined_step_time, num_pipelined_steps


def _measure_event_device_time(
    step_fn: step_lib.StepFn,
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    name: str,
    expected_compile_count: int,
    profiling_time: _ProfilingTime | None,
    clock: Callable[[], float],
) -> float:
  """Measures per-step device time via backend device events (e.g. CUDA).

  Args:
    step_fn: The step function.
    stepper: The stepper, whose post-warmup hook runs after each step.
    device_ops: The device operations to use.
    name: The name of the benchmark, used in logs.
    expected_compile_count: The compile count that the steps must not change.
    profiling_time: If set, accumulates wall time spent in this profiling phase.
    clock: Returns the current time in seconds.

  Returns:
    Average active device execution time in seconds across `--xprof_steps`
    event-timed steps, or `-1.0` if skipped or unavailable.
  """
  num_steps = XPROF_STEPS.value
  if num_steps <= 0:
    return -1.0
  profiling_start_time = clock()
  event_device_timings = np.zeros(num_steps, dtype=np.float64)
  with gc_disabled():
    for event_step in range(num_steps):
      device_elapsed = device_ops.measure_step_device_time(step_fn)
      stepper.post_warmup_hook()
      _check_no_recompile(device_ops, expected_compile_count)
      event_device_timings[event_step] = device_elapsed
  if profiling_time is not None:
    profiling_time.seconds += clock() - profiling_start_time
  avg_device_time = -1.0
  valid_event_timings = event_device_timings[event_device_timings >= 0.0]
  if valid_event_timings.size == num_steps:
    avg_device_time = float(np.mean(valid_event_timings))
  logging.info(
      "Post Warmup Event Device Timings for %s: %s",
      name,
      event_device_timings,
  )
  return avg_device_time


def _measure_xprof_device_time(
    step_fn: step_lib.StepFn,
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    name: str,
    expected_compile_count: int,
    enable_xprof: bool,
    capture_xprof: bool,
    profiling_time: _ProfilingTime | None,
    clock: Callable[[], float],
) -> tuple[float, str | None]:
  """Measures per-step device time and trace URL via an XProf capture window.

  Args:
    step_fn: The step function.
    stepper: The stepper, whose post-warmup hook runs after each step.
    device_ops: The device operations to use.
    name: The name of the benchmark, used in logs.
    expected_compile_count: The compile count that the steps must not change.
    enable_xprof: Whether all ranks run the XProf-profiled steps.
    capture_xprof: Whether this rank opens the XProf session and emits TraceMe
      scopes.
    profiling_time: If set, accumulates wall time spent in this profiling phase.
    clock: Returns the current time in seconds.

  Returns:
    A tuple of `(average_device_time_seconds, post_warmup_xprof_url)`, where
    unavailable device time is `-1.0` and an uncaptured URL is `None`.
  """
  num_steps = XPROF_STEPS.value
  if not enable_xprof or num_steps <= 0:
    return -1.0, None

  xprof_timings = np.zeros(num_steps, dtype=np.float64)
  post_warmup_run_session_xprof_url = None
  avg_device_time = -1.0
  profiling_start_time = clock()
  with gc_disabled():
    with XprofContext(
        name="post_warmup_run",
        enable_xprof=capture_xprof,
        trace_only_host=False,
        clock=clock,
    ) as xprof_context:
      # Align all ranks to host barrier-RPC granularity before each profiled
      # step (both after rank 0 starts XProf and after rank 0's slower
      # Python-traced post-step work) without dispatching a TPU collective into
      # the XProf trace.
      for xprof_step in range(num_steps):
        device_ops.host_barrier()
        xprof_timings[xprof_step] = _run_single_step(
            step_fn,
            stepper,
            device_ops,
            expected_compile_count=expected_compile_count,
            clock=clock,
            run_post_hook=(xprof_step < num_steps - 1),
            trace_step_num=xprof_step if capture_xprof else None,
        )
    stepper.post_warmup_hook()
    _check_no_recompile(device_ops, expected_compile_count)
  if profiling_time is not None:
    profiling_time.seconds += clock() - profiling_start_time

  if capture_xprof:
    if xprof_context.session_id:
      post_warmup_run_session_xprof_url = (
          f"http://xprof/?session_id={xprof_context.session_id}"
      )
    avg_device_time = _calculate_avg_device_time(
        device_ops, xprof_context.xspace, num_steps
    )

  logging.info("Post Warmup Xprof Timings for %s: %s", name, xprof_timings)
  return avg_device_time, post_warmup_run_session_xprof_url


def _post_warmup_run(
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    name: str,
    enable_xprof: bool = False,
    profiling_time: _ProfilingTime | None = None,
    clock: Callable[[], float] = time.perf_counter,
    capture_xprof: bool | None = None,
) -> metrics.PostWarmupRunResult:
  """Executes the post-warmup measurement phases in strict isolation.

  Runs Phases 2-5 sequentially after warmup completes:
    - Phase 2 (`_measure_sync_step_time`): untimed cache-warming steps followed
      by synchronized timed steps (median latency) and window diagnostics.
    - Phase 3 (`_measure_peak_memory`): isolated single-step peak-memory probe,
      executed strictly before pipelined steps and reading `peak_memory_mb()`
      before `stepper.post_warmup_hook()`.
    - Phase 4 (`_measure_pipelined_step_time`): back-to-back pipelined
      throughput window for steppers that support it.
    - Phase 5 (`_measure_event_device_time` / `_measure_xprof_device_time`):
      per-step active device time excluded from `e2e_wall_time_seconds`, with
      host-only rank synchronization before each XProf-profiled step.

  Args:
    stepper: The stepper to run.
    device_ops: The device operations to use.
    name: The name of the benchmark.
    enable_xprof: Whether to run the profiled steps. Must be the same on every
      rank: all ranks run the profiled steps in lockstep.
    profiling_time: If set, accumulates the wall time of the profiled steps,
      including starting and ending their xprof session.
    clock: Returns the current time in seconds; used to time each step.
    capture_xprof: Whether this process opens the xprof session and emits the
      TraceMe step scopes. Defaults to `enable_xprof`; ignored when
      `enable_xprof` is False.

  Returns:
    A PostWarmupRunResult instance containing the median step time, steady-state
    peak device memory usage, pipelined throughput time, and device time.
  """

  capture_xprof = enable_xprof and capture_xprof is not False
  # The first agreement on a step count compiles its collective (on TPU), so
  # run one before reading the compile count that the steps must not change.
  device_ops.agree_on_step_count(POST_WARMUP_STEPS.value)

  compile_count_before = device_ops.compile_count()
  step_fn = stepper.get_step_fn()

  sync_step_time, num_sync_steps, diagnostics = _measure_sync_step_time(
      step_fn,
      stepper,
      device_ops,
      name=name,
      expected_compile_count=compile_count_before,
      clock=clock,
  )

  # Measure single-step steady-state peak memory strictly before pipelined
  # steps, which can keep several steps' outputs in flight.
  # _measure_sync_step_time populates _ALLOCATED_MEMORY_DRIFT_KEY iff
  # DeviceOps.allocated_memory_mb() is supported (uniform across ranks),
  # skipping the probe on CPU without an extra device sync; TorchAx on TPU
  # reports live allocations without reset_memory_stats and therefore still runs
  # 1-5 probe steps returning the process high-water mark.
  if (
      _PEAK_MEMORY_PROBE_STEPS > 0
      and num_sync_steps > 0
      and _ALLOCATED_MEMORY_DRIFT_KEY in diagnostics
  ):
    memory_usage = _measure_peak_memory(
        step_fn,
        stepper,
        device_ops,
        expected_compile_count=compile_count_before,
        num_synchronized_steps=num_sync_steps,
        clock=clock,
    )
  else:
    memory_usage = diagnostics.get(_SYNC_WINDOW_PEAK_MEMORY_KEY)

  pipelined_step_time, num_pipelined_steps = _measure_pipelined_step_time(
      step_fn,
      stepper,
      device_ops,
      name=name,
      expected_compile_count=compile_count_before,
      synchronized_step_seconds=sync_step_time,
      num_synchronized_steps=num_sync_steps,
      clock=clock,
  )

  # The throughput comes from the pipelined steps, which overlap the host's
  # dispatch with the device's execution as a training or serving loop does.
  # A stepper that supports them but has no pipelined time (disabled by
  # --pipelined_steps, or out of memory) reports no throughput rather than a
  # synchronized one. Steppers that do not support them yet (decode, torchax)
  # keep the throughput of the synchronized steps.
  if pipelined_step_time > 0:
    throughput_step_time = pipelined_step_time
  elif stepper.supports_pipelined_steps:
    throughput_step_time = None
  elif num_sync_steps > 0:
    throughput_step_time = sync_step_time
  else:
    throughput_step_time = None

  if device_ops.supports_event_device_time():
    avg_device_time = _measure_event_device_time(
        step_fn,
        stepper,
        device_ops,
        name=name,
        expected_compile_count=compile_count_before,
        profiling_time=profiling_time,
        clock=clock,
    )
    post_warmup_run_session_xprof_url = None
  else:
    avg_device_time, post_warmup_run_session_xprof_url = (
        _measure_xprof_device_time(
            step_fn,
            stepper,
            device_ops,
            name=name,
            expected_compile_count=compile_count_before,
            enable_xprof=enable_xprof,
            capture_xprof=capture_xprof,
            profiling_time=profiling_time,
            clock=clock,
        )
    )

  return metrics.PostWarmupRunResult(
      post_warmup_step_time_seconds=sync_step_time,
      num_post_warmup_steps=num_sync_steps,
      pipelined_step_time_seconds=pipelined_step_time,
      num_pipelined_steps=num_pipelined_steps,
      throughput_step_time_seconds=throughput_step_time,
      peak_device_memory_mb=memory_usage,
      post_warmup_run_session_xprof_url=post_warmup_run_session_xprof_url,
      average_post_warmup_device_time_seconds=avg_device_time,
      diagnostics=diagnostics or None,
  )


def measure(
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    name: str,
    enable_xprof: bool = False,
    clock: Callable[[], float] = time.perf_counter,
    capture_xprof: bool | None = None,
) -> metrics.PerformanceMetrics:
  """Warms up and measures a stepper.

  Args:
    stepper: The stepper to measure.
    device_ops: The device operations to use.
    name: The name of the benchmark, used in logs.
    enable_xprof: Whether to profile the warmup and extra post-warmup steps.
      Must be the same on every rank: all ranks run the profiled steps in
      lockstep so collectives inside the step function do not deadlock.
    clock: Returns the current time in seconds. Tests inject a fake clock to
      control the measured durations.
    capture_xprof: Whether this process opens the xprof sessions and emits the
      TraceMe step scopes, e.g. only rank 0 of a distributed run. Defaults to
      `enable_xprof`; ignored when `enable_xprof` is False.

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
            capture_xprof=capture_xprof,
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
              capture_xprof=capture_xprof,
          )
      )
  finally:
    gc.unfreeze()

  return metrics.PerformanceMetrics(
      e2e_wall_time_seconds=(clock() - start_time - profiling_time.seconds),
      **result_kwargs,
  )
