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

import contextlib
import dataclasses
import gc
import time
from typing import Iterator

from absl import flags
from absl import logging
import numpy as np
from examples.benchmarks.e2e.harness import device_ops as device_ops_lib
from examples.benchmarks.e2e.harness import metrics
from examples.benchmarks.e2e.harness import step_lib
from torch_tpu._internal.benchmarks import xprof_adapter

MAX_WARMUP_STEPS = flags.DEFINE_integer(
    "max_warmup_steps", 20, "Maximum number of warmup steps.", lower_bound=0
)
MIN_WARMUP_STEPS = flags.DEFINE_integer(
    "min_warmup_steps", 10, "Minimum number of warmup steps.", lower_bound=0
)
POST_WARMUP_STEPS = flags.DEFINE_integer(
    "post_warmup_steps", 10, "Number of post-warmup steps.", lower_bound=0
)
XPROF_STEPS = flags.DEFINE_integer(
    "xprof_steps",
    2,
    "Number of steps to collect xprof for after warmup.",
    lower_bound=0,
)


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
  """

  def __init__(
      self, *, name: str, enable_xprof: bool, trace_only_host: bool = False
  ):
    self.name = name
    self.enable_xprof = enable_xprof
    self.trace_only_host = trace_only_host
    self.session = None
    self.session_id = None

  def __enter__(self):
    if self.enable_xprof:
      self.session = xprof_adapter.XprofSession()
      kwargs = {}
      host_trace_level = 3
      enable_python_tracer = True
      if self.trace_only_host:
        kwargs["trace_mode"] = "TRACE_ONLY_HOST"
        host_trace_level = 1
        enable_python_tracer = False
      self.session.start_session(
          host_trace_level=host_trace_level,
          enable_python_tracer=enable_python_tracer,
          host_cpu_profile=True,
          response_max_bytes=_PROFILE_MAX_BYTES,
          **kwargs,
      )
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    # End the session and log the URL if it was started. If there is an
    # exception, we let the exception propagate.
    if self.session:
      self.session_id = self.session.end_session_and_get_session_id()
      logging.info(
          "%s xprof URL: http://xprof/?session_id=%s",
          self.name,
          self.session_id,
      )


class PostWarmupRecompileError(AssertionError):
  """Raised when a recompilation happens during the post warmup loop."""


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
      name="warmup_run", enable_xprof=enable_xprof, trace_only_host=True
  ) as warmup_run_context:
    step_fn = stepper.get_step_fn()
    for step in range(MAX_WARMUP_STEPS.value):
      start_time = time.perf_counter()
      out = step_fn()
      device_ops.await_result(out)
      timings[step] = time.perf_counter() - start_time
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
      first_step_time_seconds=timings[0],
      warmup_overhead_seconds=_get_warmup_overhead(timings, num_warmup_steps),
      warmup_session_xprof_url=warmup_session_xprof_url,
  )


def _post_warmup_run(
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    name: str,
    enable_xprof: bool = False,
) -> metrics.PostWarmupRunResult:
  """Runs the model once after the warmup is complete.

  Args:
    stepper: The stepper to run.
    device_ops: The device operations to use.
    name: The name of the benchmark.
    enable_xprof: Whether to enable xprof profiling.

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

  def _run_single_step(expected_compile_count: int) -> float:
    start_time = time.perf_counter()
    out = step_fn()
    device_ops.await_result(out)
    elapsed = time.perf_counter() - start_time
    stepper.post_warmup_hook()

    step_compile_count = device_ops.compile_count()
    if step_compile_count != expected_compile_count:
      raise PostWarmupRecompileError(
          "Recompilation happened inside the post warmup loop. Expected"
          f" {expected_compile_count}, got {step_compile_count}"
      )
    return elapsed

  with gc_disabled():
    for step in range(POST_WARMUP_STEPS.value):
      timings[step] = _run_single_step(compile_count_before)

  post_warmup_run_session_xprof_url = None
  if enable_xprof and XPROF_STEPS.value > 0:
    with gc_disabled(), XprofContext(
        name="post_warmup_run",
        enable_xprof=enable_xprof,
        trace_only_host=False,
    ) as xprof_context:
      for xprof_step in range(XPROF_STEPS.value):
        xprof_timings[xprof_step] = _run_single_step(compile_count_before)

    if xprof_context.session_id:
      post_warmup_run_session_xprof_url = (
          f"http://xprof/?session_id={xprof_context.session_id}"
      )

  memory_usage = device_ops.peak_memory_mb()

  logging.info("Post Warmup Timings for %s: %s", name, timings)
  logging.info("Post Warmup Xprof Timings for %s: %s", name, xprof_timings)

  return metrics.PostWarmupRunResult(
      post_warmup_step_time_seconds=float(np.mean(timings)),
      peak_device_memory_mb=memory_usage,
      post_warmup_run_session_xprof_url=post_warmup_run_session_xprof_url,
  )


def measure(
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    name: str,
    enable_xprof: bool = False,
) -> metrics.PerformanceMetrics:
  result_kwargs = {}
  start_time = time.perf_counter()

  stepper.pre_warmup_init()

  result_kwargs |= dataclasses.asdict(
      _warmup_run(
          stepper,
          device_ops,
          name=name,
          enable_xprof=enable_xprof,
      )
  )

  if _do_post_warmup():
    stepper.post_warmup_hook()
    result_kwargs |= dataclasses.asdict(
        _post_warmup_run(
            stepper,
            device_ops,
            name=name,
            enable_xprof=enable_xprof,
        )
    )

  return metrics.PerformanceMetrics(
      e2e_wall_time_seconds=time.perf_counter() - start_time,
      **result_kwargs,
  )
