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

MAX_WARMUP_STEPS = flags.DEFINE_integer(
    "max_warmup_steps", 20, "Maximum number of warmup steps.", lower_bound=0
)
MIN_WARMUP_STEPS = flags.DEFINE_integer(
    "min_warmup_steps", 10, "Minimum number of warmup steps.", lower_bound=0
)
POST_WARMUP_STEPS = flags.DEFINE_integer(
    "post_warmup_steps", 10, "Number of post-warmup steps.", lower_bound=0
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

  if not num_warmup_steps:
    raise RuntimeError(
        "Benchmark function compilations have not stabilized after"
        f" {MAX_WARMUP_STEPS.value} warmup runs. num_warmup_steps was"
        f" {num_warmup_steps}. Consider increasing the number of warmup steps."
    )
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
    step_fn: step_lib.StepFn,
    device_ops: device_ops_lib.DeviceOps,
    *,
    name: str,
) -> metrics.WarmupRunResult:
  """Runs the model to warmup the caches.

  The number of iteration is controlled by the flags MIN_WARMUP_STEPS and
  MAX_WARMUP_STEPS.

  The benchmark function compilations have not stabilized if the number of cache
  misses is not the same for two consecutive steps. If the benchmark function
  compilations have not stabilized after MAX_WARMUP_STEPS runs, then the
  warmup result is not calculated and RuntimeError is raised.

  Args:
    run_step: The function to run.
    device_ops: The device operations to use.
    name: The name of the benchmark.

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

  with gc_disabled():
    for step in range(MAX_WARMUP_STEPS.value):
      start_time = time.perf_counter()
      out = step_fn()
      device_ops.await_result(out)
      timings[step] = time.perf_counter() - start_time
      compile_count[step] = device_ops.compile_count()

      if (
          step > 0
          and step >= MIN_WARMUP_STEPS.value - 1
          and compile_count[step] == compile_count[step - 1]
      ):
        num_warmup_steps = step + 1
        break

  logging.info("Warmup Timings for %s: %s", name, timings)
  logging.info("Warmup compile counts for %s: %s", name, compile_count)

  return metrics.WarmupRunResult(
      num_warmup_steps=num_warmup_steps,  #  pyrefly: ignore[bad-argument-type]
      first_step_time_seconds=timings[0],
      warmup_overhead_seconds=_get_warmup_overhead(timings, num_warmup_steps),  # pyrefly: ignore[bad-argument-type]
  )


def _post_warmup_run(
    step_fn: step_lib.StepFn,
    device_ops: device_ops_lib.DeviceOps,
    *,
    name: str,
) -> metrics.PostWarmupRunResult:
  """Runs the model once after the warmup is complete.

  Args:
    run_step: The function to run.
    device_ops: The device operations to use.
    name: The name of the benchmark.

  Returns:
    A PostWarmupRunResult instance containing the average step time and peak
    device memory usage.
  """

  timings = np.zeros(POST_WARMUP_STEPS.value, dtype=np.float64)

  # gc is explicitly disabled below to prevent GC pauses from affecting the
  # measurements. Collect garbage once before the timed loop.
  gc.collect()
  device_ops.reset_peak_memory()

  compile_count_before = device_ops.compile_count()

  with gc_disabled():
    for step in range(POST_WARMUP_STEPS.value):
      start_time = time.perf_counter()
      out = step_fn()
      device_ops.await_result(out)
      elapsed = time.perf_counter() - start_time
      timings[step] = elapsed

      step_compile_count = device_ops.compile_count()
      if step_compile_count != compile_count_before:
        raise PostWarmupRecompileError(
            "Recompilation happened inside the post warmup loop. Expected"
            f" {compile_count_before}, got {step_compile_count}"
        )

  memory_usage = device_ops.peak_memory_mb()

  logging.info("Post Warmup Timings for %s: %s", name, timings)

  return metrics.PostWarmupRunResult(
      post_warmup_step_time_seconds=np.mean(timings),
      peak_device_memory_mb=memory_usage,
  )


def measure(
    stepper: step_lib.Stepper,
    device_ops: device_ops_lib.DeviceOps,
    *,
    name: str,
) -> metrics.PerformanceMetrics:
  result_kwargs = {}
  start_time = time.perf_counter()

  stepper.pre_warmup_init()

  result_kwargs |= dataclasses.asdict(
      _warmup_run(
          stepper.get_step_fn(),
          device_ops,
          name=name,
      )
  )

  if _do_post_warmup():
    stepper.post_warmup_hook()
    result_kwargs |= dataclasses.asdict(
        _post_warmup_run(
            stepper.get_step_fn(),
            device_ops,
            name=name,
        )
    )

  return metrics.PerformanceMetrics(
      e2e_wall_time_seconds=time.perf_counter() - start_time,
      **result_kwargs,
  )
