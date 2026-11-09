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

"""Tests for measure()."""

import gc
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from perflab import device_utils
from perflab import measure as measure_lib
from perflab import metrics as metrics_lib
from perflab.tests import fakes
from tests import seed_test_utils

_ONE_MS_IN_PS = 10**9
_CONTENDED = "UNAVAILABLE: Another profiling session active."


def setUpModule():
  # Most tests count the steps that run. They run no untimed post-warmup steps;
  # MeasureUntimedStepsTest sets them.
  patcher = mock.patch.object(measure_lib, "_POST_WARMUP_UNTIMED_STEPS", 0)
  patcher.start()
  unittest.addModuleCleanup(patcher.stop)


def _run_measure(
    step,
    ops,
    steps,
    min_warmup,
    max_warmup=None,
    xprof_steps=None,
    name="test",
    enable_xprof=False,
    clock=None,
) -> metrics_lib.PerformanceMetrics:
  flag_kwargs = {
      "min_warmup_steps": min_warmup,
      "max_warmup_steps": (
          max_warmup if max_warmup is not None else min_warmup + 10
      ),
      "post_warmup_steps": steps,
  }
  if xprof_steps is not None:
    flag_kwargs["xprof_steps"] = xprof_steps
  with flagsaver.flagsaver(**flag_kwargs):
    return measure_lib.measure(
        step,
        ops,
        name=name,
        enable_xprof=enable_xprof,
        clock=clock or fakes.FakeClock(),
    )


class XspaceDeviceTimeOps(fakes.RecordingOps):
  """RecordingOps that parses device time from a real XSpace."""

  def get_device_time_from_xspace(self, xspace, expected_steps=None):
    return device_utils.get_total_device_time_from_xspace(
        xspace, expected_steps=expected_steps
    )


class _EventTimingOps(fakes.RecordingOps):
  """RecordingOps that simulates event-timed step device time, as on CUDA."""

  def __init__(
      self, timings=(), recompile_on_calls=None, fail_check_on_calls=None
  ):
    super().__init__(
        recompile_on_calls=recompile_on_calls,
        fail_check_on_calls=fail_check_on_calls,
    )
    self._timings = list(timings)

  def supports_event_device_time(self) -> bool:
    return True

  def measure_step_device_time(self, step_fn):
    out = step_fn()
    self.await_result(out)
    self.check_result(out)
    return self._timings.pop(0) if self._timings else 0.010


class _EventLoggingXprofSession(fakes.FakeXprofSession):
  """FakeXprofSession that logs when it starts and ends to `events`."""

  def __init__(self, events, label, **kwargs):
    super().__init__(session_id=f"{label}_id", **kwargs)
    self._events = events
    self._label = label

  def start_session(self, **kwargs):
    super().start_session(**kwargs)
    self._events.append(f"{self._label}_xprof_start")

  def end_session_and_get_xspace_and_url(self):
    self._events.append(f"{self._label}_xprof_end")
    return super().end_session_and_get_xspace_and_url()


class _XprofTestCase(seed_test_utils.RepeatableTest):
  """Base class that swaps the XProf profiler for stateful fakes."""

  def install_xprof(
      self, factory: fakes.FakeXprofSessionFactory
  ) -> fakes.FakeXprofSessionFactory:
    # TODO(wan): revisit DECISION (b/568592811): the profiler boundary is
    # swapped at `xprof_adapter.XprofSession` rather than through a new
    # `session_factory=` parameter threaded through measure() and XprofContext.
    self.enter_context(
        mock.patch.object(measure_lib.xprof_adapter, "XprofSession", factory)
    )
    return factory

  def install_sleep(self) -> fakes.FakeClock:
    clock = fakes.FakeClock()
    self.enter_context(
        mock.patch.object(measure_lib.time, "sleep", clock.sleep)
    )
    return clock


class MeasureGcTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.assertTrue(gc.isenabled(), "test precondition: GC starts enabled")
    self.addCleanup(gc.enable)

  def test_disabled_inside_restored_after(self):
    self.assertTrue(gc.isenabled())
    with measure_lib.gc_disabled():
      self.assertFalse(gc.isenabled())
    self.assertTrue(gc.isenabled())

  def test_restored_on_exception(self):
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=benchmark
      with measure_lib.gc_disabled():
        self.assertFalse(gc.isenabled())
        raise RuntimeError("boom")
    self.assertTrue(gc.isenabled(), "must restore even when the block raises")

  def test_prior_disabled_state_honoured(self):
    """If GC was already off on entry, leave it off -- don't blindly enable."""
    gc.disable()
    try:
      with measure_lib.gc_disabled():
        self.assertFalse(gc.isenabled())
      self.assertFalse(
          gc.isenabled(), "must not enable GC that was off on entry"
      )
    finally:
      gc.enable()

  def test_gc_disabled_during_timed_loop(self):
    ops = fakes.RecordingOps()
    seen = []

    class GcTestStep(fakes.CountingStep):

      def __call__(self):
        seen.append(gc.isenabled())
        return super().__call__()

    _run_measure(GcTestStep(), ops, steps=3, min_warmup=2)
    # GC is off during both warmup and post warmup loops.
    self.assertEqual(seen[:2], [False, False])
    self.assertEqual(seen[2:], [False, False, False])

  def test_gc_reenabled_after_normal_run(self):
    _run_measure(
        fakes.CountingStep(), fakes.RecordingOps(), steps=2, min_warmup=2
    )
    self.assertTrue(gc.isenabled())

  def test_gc_reenabled_when_step_raises(self):
    # 2 warmup, then explode on 1st timed step
    step = fakes.CountingStep(raise_at=3)
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=benchmark
      _run_measure(step, fakes.RecordingOps(), steps=3, min_warmup=2)
    self.assertTrue(gc.isenabled(), "GC must be restored even on failure")


class MeasureGcFreezeTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.assertEqual(gc.get_freeze_count(), 0, "precondition: nothing frozen")
    self.addCleanup(gc.unfreeze)

  def test_heap_frozen_during_measure_and_unfrozen_after(self):
    freeze_counts = []

    class FreezeTrackingStep(fakes.CountingStep):

      def __call__(self):
        freeze_counts.append(gc.get_freeze_count())
        return super().__call__()

    _run_measure(
        FreezeTrackingStep(), fakes.RecordingOps(), steps=2, min_warmup=2
    )

    self.assertLen(freeze_counts, 4)
    self.assertTrue(all(count > 0 for count in freeze_counts))
    self.assertEqual(gc.get_freeze_count(), 0)

  def test_heap_unfrozen_when_step_raises(self):
    step = fakes.CountingStep(raise_at=3)
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=benchmark
      _run_measure(step, fakes.RecordingOps(), steps=3, min_warmup=2)
    self.assertEqual(gc.get_freeze_count(), 0)


class MeasureUntimedStepsTest(seed_test_utils.RepeatableTest):

  def _measure(self, step_seconds, untimed_steps, steps=3, ops=None):
    clock = fakes.FakeClock()
    step = fakes.CountingStep(clock=clock, step_seconds=step_seconds)
    with mock.patch.object(
        measure_lib, "_POST_WARMUP_UNTIMED_STEPS", untimed_steps
    ):
      m = _run_measure(
          step,
          ops or fakes.RecordingOps(),
          steps=steps,
          min_warmup=2,
          clock=clock,
      )
    return step, m

  def test_runs_fixed_untimed_steps_whatever_their_timings(self):
    # The same steps whether the steps settle at once, later or never.
    for step_seconds in (
        [10.0, 1.0],
        [10.0, 1.0, 3.0, 2.0, 1.0],
        [10.0, 1.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05],
    ):
      with self.subTest(step_seconds=step_seconds):
        step, _ = self._measure(step_seconds, untimed_steps=4)
        self.assertEqual(step.calls, 2 + 4 + 3)

  def test_untimed_steps_are_not_timed(self):
    # Warmup: 10, 1. Untimed: 3, 2. Timed: 1, 1, 1.
    _, m = self._measure([10.0, 1.0, 3.0, 2.0, 1.0], untimed_steps=2)
    self.assertEqual(m.post_warmup_step_time_seconds, 1.0)

  def test_peak_reset_happens_before_untimed_steps(self):
    ops = fakes.RecordingOps()
    self._measure([10.0, 1.0], untimed_steps=4, steps=2, ops=ops)
    # 2 warmup steps, then the reset, then 4 untimed and 2 timed steps.
    reset_at = ops.events.index("reset")
    self.assertEqual(ops.events[:reset_at].count("await"), 2)
    self.assertEqual(ops.events[reset_at:].count("await"), 6)


class MeasureSequenceTest(seed_test_utils.RepeatableTest):

  def test_run_step_called_warmup_plus_steps_times(self):
    step, ops = fakes.CountingStep(), fakes.RecordingOps()
    _run_measure(step, ops, steps=4, min_warmup=2)
    self.assertEqual(step.calls, 6)

  def test_run_step_called_warmup_plus_steps_times_with_recompile(self):
    step = fakes.CountingStep()
    ops = fakes.RecordingOps(recompile_on_calls={1, 2, 3, 4})
    _run_measure(step, ops, steps=4, min_warmup=2)
    # Compilation stablises on 5th warmup step + 4 timed steps = 9 steps
    self.assertEqual(step.calls, 9)

  def test_peak_reset_happens_after_warmup_and_before_timed_loop(self):
    step, ops = fakes.CountingStep(), fakes.RecordingOps(recompile_on_calls={2})
    _run_measure(step, ops, steps=4, min_warmup=2)

    reset_at = ops.events.index("reset")
    awaits_before = ops.events[:reset_at].count("await")
    awaits_after = ops.events[reset_at:].count("await")

    self.assertEqual(ops.reset_count, 1, "peak must be reset exactly once")
    self.assertEqual(awaits_before, 3, "reset must come after ALL warmup steps")
    self.assertEqual(awaits_after, 4, "reset must come before ALL timed steps")

  def test_await_result_called_once_per_step_including_warmup(self):
    step, ops = fakes.CountingStep(), fakes.RecordingOps()
    _run_measure(step, ops, steps=4, min_warmup=2)
    self.assertLen(ops.awaited, 6)

  def test_await_result_receives_the_steps_output_untouched(self):
    """The pass-through contract.

    torchax's block_until_ready needs the actual value; the runner must not
    inspect, unwrap, or replace it.
    """
    step, ops = fakes.CountingStep(), fakes.RecordingOps()
    _run_measure(step, ops, steps=2, min_warmup=2)
    self.assertEqual(ops.awaited, ["out-1", "out-2", "out-3", "out-4"])

  def test_stepper_hooks_run_around_warmup(self):
    step = fakes.CountingStep()
    _run_measure(step, fakes.RecordingOps(), steps=2, min_warmup=2)
    self.assertEqual(step.pre_warmup_init_calls, 1)
    # Once per warmup step and once per timed step.
    self.assertEqual(step.post_warmup_hook_calls, 2 + 2)

  def test_previous_step_output_released_before_next_step_and_hook(self):
    """Ensures warmup and timed loops do not hold 2x step outputs in memory."""
    live_outputs = 0
    max_live_outputs_in_step = 0
    max_live_outputs_in_hook = 0

    class _TrackedOutput:

      def __init__(self) -> None:
        nonlocal live_outputs
        live_outputs += 1

      def __del__(self) -> None:
        nonlocal live_outputs
        live_outputs -= 1

    class _TrackingOps(fakes.RecordingOps):

      def await_result(self, out) -> None:
        del out
        self.events.append("await")
        self.calls += 1

      def check_result(self, out) -> None:
        del out
        self.events.append("check")

    class _OutputLifetimeStep:

      def get_step_fn(self):
        def step():
          nonlocal max_live_outputs_in_step
          out = _TrackedOutput()
          max_live_outputs_in_step = max(max_live_outputs_in_step, live_outputs)
          return out

        return step

      def compile(self, **kwargs):
        pass

      def pre_warmup_init(self):
        pass

      def post_warmup_hook(self):
        nonlocal max_live_outputs_in_hook
        max_live_outputs_in_hook = max(max_live_outputs_in_hook, live_outputs)

    _run_measure(_OutputLifetimeStep(), _TrackingOps(), steps=3, min_warmup=3)
    self.assertEqual(max_live_outputs_in_step, 1)
    self.assertEqual(max_live_outputs_in_hook, 0)
    self.assertEqual(live_outputs, 0)


class MeasureCheckResultTest(_XprofTestCase):

  def test_check_result_follows_every_await(self):
    step, ops = fakes.CountingStep(), fakes.RecordingOps()
    _run_measure(step, ops, steps=4, min_warmup=2)

    steps = [e for e in ops.events if e in ("await", "check")]
    self.assertEqual(steps, ["await", "check"] * 6)
    self.assertEqual(ops.checked, ops.awaited)

  def test_check_result_runs_once_per_profiled_step(self):
    step, ops = fakes.CountingStep(), fakes.RecordingOps()
    self.install_xprof(fakes.FakeXprofSessionFactory())
    with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
      measure_lib._post_warmup_run(
          step, ops, name="test_post_warmup", enable_xprof=True
      )

    # 3 timed steps + 2 profiled steps.
    self.assertEqual(ops.checked, ["out-1", "out-2", "out-3", "out-4", "out-5"])

  def test_failed_warmup_step_raises(self):
    # A failed asynchronous execution must fail the benchmark instead of
    # being timed as if the step had run.
    ops = fakes.RecordingOps(fail_check_on_calls={1})
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        RuntimeError, "execution failed for out-1"
    ):
      _run_measure(fakes.CountingStep(), ops, steps=3, min_warmup=2)

  def test_failed_timed_step_raises(self):
    ops = fakes.RecordingOps(fail_check_on_calls={4})
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        RuntimeError, "execution failed for out-4"
    ):
      _run_measure(fakes.CountingStep(), ops, steps=3, min_warmup=2)
    self.assertTrue(gc.isenabled())

  def test_check_result_is_not_timed(self):
    clock = fakes.FakeClock()

    class SlowCheckOps(fakes.RecordingOps):

      def check_result(self, out):
        clock.advance(100.0)
        super().check_result(out)

    m = _run_measure(
        fakes.CountingStep(clock=clock, step_seconds=1.0),
        SlowCheckOps(),
        steps=3,
        min_warmup=2,
        clock=clock,
    )

    self.assertEqual(m.first_step_time_seconds, 1.0)
    self.assertEqual(m.post_warmup_step_time_seconds, 1.0)


class MeasureRecompileGuardTest(seed_test_utils.RepeatableTest):

  def test_no_recompiles_finds_min_warmup_steps(self):
    m = _run_measure(
        fakes.CountingStep(), fakes.RecordingOps(), steps=3, min_warmup=2
    )
    self.assertEqual(m.num_warmup_steps, 2)

  def test_recompile_during_timed_loop_raises(self):
    # min_warmup=2 =>calls 1,2 are warmup; 3,4,5 are timed. Recompile on call 4.
    ops = fakes.RecordingOps(recompile_on_calls={4})
    with self.assertRaises(  # ASSERT_RAISES_OK=benchmark
        measure_lib.PostWarmupRecompileError
    ):
      _run_measure(
          fakes.CountingStep(),
          ops,
          steps=3,
          min_warmup=2,
          name="llama_compiled",
      )

  def test_recompiles_during_warmup_are_fine(self):
    ops = fakes.RecordingOps(recompile_on_calls={1, 2})  # both during warmup
    m = _run_measure(fakes.CountingStep(), ops, steps=3, min_warmup=3)
    self.assertEqual(m.num_warmup_steps, 3)

  def test_gc_reenabled_when_recompile_guard_fires(self):
    ops = fakes.RecordingOps(recompile_on_calls={4})
    with self.assertRaises(  # ASSERT_RAISES_OK=benchmark
        measure_lib.PostWarmupRecompileError
    ):
      _run_measure(fakes.CountingStep(), ops, steps=3, min_warmup=2)
    self.assertTrue(gc.isenabled())

  def test_recompiles_not_stabilized_during_warmup_raises(self):
    # Set max_warmup=3. Recompile on every warmup step (calls 1, 2, 3).
    ops = fakes.RecordingOps(recompile_on_calls={1, 2, 3})
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        RuntimeError,
        "Benchmark function compilations have not stabilized",
    ):
      _run_measure(
          fakes.CountingStep(), ops, steps=3, min_warmup=2, max_warmup=3
      )


class MeasureMetricsTest(_XprofTestCase):

  def test_peak_memory_passed_through(self):
    m = _run_measure(
        fakes.CountingStep(),
        fakes.RecordingOps(peak_mb=512.0),
        steps=2,
        min_warmup=2,
    )
    self.assertEqual(m.peak_device_memory_mb, 512.0)

  def test_peak_memory_none_when_backend_cannot_report(self):
    m = _run_measure(
        fakes.CountingStep(),
        fakes.RecordingOps(peak_mb=None),
        steps=2,
        min_warmup=2,
    )
    self.assertIsNone(m.peak_device_memory_mb)

  def test_timing_metrics_come_from_the_clock(self):
    clock = fakes.FakeClock()
    step = fakes.CountingStep(
        clock=clock,
        init_seconds=1.0,
        # Warmup stabilizes after min_warmup=3 steps, then 3 timed steps.
        step_seconds=[10.0, 2.0, 2.0, 2.0, 3.0, 4.0],
    )
    m = _run_measure(
        step, fakes.RecordingOps(), steps=3, min_warmup=3, clock=clock
    )

    self.assertEqual(m.first_step_time_seconds, 10.0)
    # Post warmup timings: 2.0, 3.0, 4.0 -> mean = 3.0
    self.assertEqual(m.post_warmup_step_time_seconds, 3.0)
    # Warmup timings: 10.0, 2.0, 2.0. Stabilized time is 2.0.
    # overhead = (10.0 + 2.0 + 2.0) - (2.0 * 3) = 8.0
    self.assertEqual(m.warmup_overhead_seconds, 8.0)
    # The end-to-end time also covers pre_warmup_init.
    self.assertEqual(m.e2e_wall_time_seconds, 1.0 + 14.0 + 9.0)

  def test_first_step_time_includes_compilation_in_get_step_fn(self):
    clock = fakes.FakeClock()
    step = fakes.CountingStep(
        clock=clock, step_seconds=1.0, compile_seconds=100.0
    )
    m = _run_measure(
        step, fakes.RecordingOps(), steps=3, min_warmup=2, clock=clock
    )

    self.assertEqual(m.first_step_time_seconds, 101.0)
    # Warmup timings: 101.0, 1.0 -> overhead = 102.0 - 1.0 * 2 = 100.0
    self.assertEqual(m.warmup_overhead_seconds, 100.0)
    self.assertEqual(m.post_warmup_step_time_seconds, 1.0)

  def test_e2e_wall_time_excludes_profiling(self):
    def run(enable_xprof):
      clock = fakes.FakeClock()
      self.install_xprof(
          fakes.FakeXprofSessionFactory(clock=clock, overhead_seconds=5.0)
      )
      return _run_measure(
          fakes.CountingStep(clock=clock, step_seconds=1.0),
          fakes.RecordingOps(),
          steps=3,
          min_warmup=2,
          xprof_steps=2,
          enable_xprof=enable_xprof,
          clock=clock,
      )

    profiled = run(enable_xprof=True)
    unprofiled = run(enable_xprof=False)

    # 2 warmup steps + 3 post-warmup steps of 1 s each.
    self.assertEqual(unprofiled.e2e_wall_time_seconds, 5.0)
    self.assertEqual(profiled.e2e_wall_time_seconds, 5.0)
    self.assertEqual(profiled.first_step_time_seconds, 1.0)

  def test_timing_metrics_emit_milliseconds(self):
    perf_metrics = metrics_lib.PerformanceMetrics(
        e2e_wall_time_seconds=12.123456789,
        first_step_time_seconds=1.123456789,
        warmup_overhead_seconds=0.987654321,
        post_warmup_step_time_seconds=0.001234567,
        average_post_warmup_device_time_seconds=0.000987654,
    )
    metric_map = perf_metrics.metric_map()
    self.assertAlmostEqual(
        metric_map["first_step_time_ms"], 1123.456789, places=6
    )
    self.assertAlmostEqual(
        metric_map["warmup_overhead_ms"], 987.654321, places=6
    )
    self.assertAlmostEqual(
        metric_map["post_warmup_step_time_ms"], 1.234567, places=6
    )
    self.assertAlmostEqual(
        metric_map["average_post_warmup_device_time_ms"], 0.987654, places=6
    )

  def test_negative_durations_are_converted_to_milliseconds(self):
    # The warmup overhead is negative when the last warmup step is slower than
    # the average warmup step.
    metric_map = metrics_lib.PerformanceMetrics(
        warmup_overhead_seconds=-0.25
    ).metric_map()

    self.assertEqual(metric_map["warmup_overhead_seconds"], -0.25)
    self.assertEqual(metric_map["warmup_overhead_ms"], -250.0)

  def test_unavailable_device_time_is_not_exported(self):
    # Without a profile, e.g. on GPU, device time stays at its -1.0 default.
    metric_map = metrics_lib.PerformanceMetrics(
        post_warmup_step_time_seconds=0.001234567
    ).metric_map()

    self.assertNotIn("average_post_warmup_device_time_seconds", metric_map)
    self.assertNotIn("average_post_warmup_device_time_ms", metric_map)

  def test_unmeasured_host_compilation_memory_is_not_exported(self):
    # measure() does not measure it, so perflab benchmarks never set it.
    metric_map = metrics_lib.PerformanceMetrics(
        post_warmup_step_time_seconds=0.001234567
    ).metric_map()

    self.assertNotIn("peak_host_compilation_memory_mb", metric_map)

  def test_metric_map_keys(self):
    """The exported metric names are what the benchmark dashboards query."""
    perf_metrics = metrics_lib.PerformanceMetrics(
        e2e_wall_time_seconds=10.0,
        num_warmup_steps=3,
        first_step_time_seconds=2.0,
        warmup_overhead_seconds=1.0,
        post_warmup_step_time_seconds=0.5,
        peak_device_memory_mb=1024.0,
        average_post_warmup_device_time_seconds=0.25,
        peak_host_compilation_memory_mb=512.0,
    )

    self.assertCountEqual(
        perf_metrics.metric_map(),
        [
            "num_warmup_steps",
            "first_step_time_seconds",
            "first_step_time_ms",
            "warmup_overhead_seconds",
            "warmup_overhead_ms",
            "post_warmup_step_time_seconds",
            "post_warmup_step_time_ms",
            "average_post_warmup_device_time_seconds",
            "average_post_warmup_device_time_ms",
            "peak_device_memory_mb",
            "peak_host_compilation_memory_mb",
        ],
    )


class XprofContextTest(
    _XprofTestCase,
    parameterized.TestCase,  # ABSLTEST_OK=seeded by RepeatableTest
):

  def test_xprof_disabled(self):
    xprof = self.install_xprof(fakes.FakeXprofSessionFactory())
    with measure_lib.XprofContext(
        name="test_disabled", enable_xprof=False
    ) as ctx:
      self.assertIsNone(ctx.session)
    self.assertIsNone(ctx.session_id)
    self.assertIsNone(ctx.xspace)
    self.assertEmpty(xprof.created)

  @parameterized.named_parameters(
      dict(
          testcase_name="full_trace",
          trace_only_host=False,
          expected_config=dict(
              host_trace_level=3,
              enable_python_tracer=True,
              host_cpu_profile=True,
              response_max_bytes=measure_lib._PROFILE_MAX_BYTES,
          ),
      ),
      dict(
          testcase_name="host_only_trace",
          trace_only_host=True,
          expected_config=dict(
              host_trace_level=1,
              enable_python_tracer=False,
              host_cpu_profile=True,
              response_max_bytes=measure_lib._PROFILE_MAX_BYTES,
              trace_mode="TRACE_ONLY_HOST",
          ),
      ),
  )
  def test_trace_config(self, trace_only_host, expected_config):
    xprof = self.install_xprof(fakes.FakeXprofSessionFactory())
    with measure_lib.XprofContext(
        name="trace", enable_xprof=True, trace_only_host=trace_only_host
    ):
      pass
    (session,) = xprof.started
    config = dict(session.start_kwargs)
    config.pop("session_id")
    self.assertEqual(config, expected_config)

  def test_xprof_enabled_returns_session_id_and_xspace_after_exit(self):
    xspace = fakes.make_xspace({"/device:TPU:0": [_ONE_MS_IN_PS]})
    xprof = self.install_xprof(
        fakes.FakeXprofSessionFactory(session_id="session_123", xspace=xspace)
    )
    with measure_lib.XprofContext(
        name="test_session", enable_xprof=True
    ) as ctx:
      (session,) = xprof.started
      self.assertIs(ctx.session, session)
      self.assertFalse(session.ended)
      self.assertIsNone(ctx.session_id)
      self.assertIsNone(ctx.xspace)

    self.assertTrue(session.ended)
    self.assertEqual(ctx.session_id, "session_123")
    self.assertIs(ctx.xspace, xspace)

  def test_xprof_enabled_exception_propagates_and_session_ends(self):
    xprof = self.install_xprof(
        fakes.FakeXprofSessionFactory(session_id="session_err")
    )
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
      with measure_lib.XprofContext(name="err_trace", enable_xprof=True) as ctx:
        raise ValueError("test error")

    self.assertTrue(xprof.created[0].ended)
    self.assertEqual(ctx.session_id, "session_err")

  def test_xprof_falls_back_when_xspace_api_unavailable(self):
    xprof = self.install_xprof(
        fakes.FakeXprofSessionFactory(
            session_id="fallback_id", returns_xspace=False
        )
    )
    with measure_lib.XprofContext(name="fallback", enable_xprof=True) as ctx:
      pass

    self.assertTrue(xprof.created[0].ended)
    self.assertEqual(ctx.session_id, "fallback_id")
    # Without an XSpace, device time is reported as unavailable rather than
    # being resolved through a possibly-shared session id.
    self.assertIsNone(ctx.xspace)

  def test_each_session_gets_a_distinct_session_id(self):
    """Distinct ids are what stop benchmarks sharing one XProf link."""
    xprof = self.install_xprof(fakes.FakeXprofSessionFactory())
    for _ in range(25):
      with measure_lib.XprofContext(name="s", enable_xprof=True):
        pass

    ids = [session.start_kwargs["session_id"] for session in xprof.started]
    self.assertLen(ids, 25)
    self.assertLen(set(ids), 25)
    for session_id in ids:
      self.assertIsInstance(session_id, int)
      # Must fit in the uint64 the xprof API expects.
      self.assertGreater(session_id, 0)
      self.assertLess(session_id, 1 << 64)

  def test_start_session_falls_back_when_session_id_unsupported(self):
    xprof = self.install_xprof(
        fakes.FakeXprofSessionFactory(
            session_id="legacy", accepts_session_id=False
        )
    )
    with measure_lib.XprofContext(name="legacy", enable_xprof=True) as ctx:
      pass

    (session,) = xprof.started
    self.assertNotIn("session_id", session.start_kwargs)
    self.assertTrue(session.ended)
    self.assertEqual(ctx.session_id, "legacy")

  def test_start_session_retries_with_fresh_session_when_another_session_active(
      self,
  ):
    xspace = fakes.make_xspace({"/device:TPU:0": [_ONE_MS_IN_PS]})
    contended = fakes.FakeXprofSession(start_error=RuntimeError(_CONTENDED))
    active = fakes.FakeXprofSession(session_id="recovered_id", xspace=xspace)
    xprof = self.install_xprof(fakes.FakeXprofSessionFactory(contended, active))
    sleeper = self.install_sleep()

    with measure_lib.XprofContext(name="contended", enable_xprof=True) as ctx:
      self.assertIs(ctx.session, active)

    self.assertEqual(xprof.created, [contended, active])
    self.assertEqual(
        sleeper.sleeps, [measure_lib._XPROF_START_RETRY_DELAY_SECONDS]
    )
    self.assertFalse(contended.ended)
    self.assertTrue(active.ended)
    self.assertEqual(ctx.session_id, "recovered_id")
    self.assertIs(ctx.xspace, xspace)

  def test_start_session_raises_after_exhausting_retries(self):
    xprof = self.install_xprof(
        fakes.FakeXprofSessionFactory(start_error=RuntimeError(_CONTENDED))
    )
    sleeper = self.install_sleep()

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        RuntimeError, "Another profiling session active"
    ):
      with measure_lib.XprofContext(name="contended", enable_xprof=True):
        pass

    self.assertLen(xprof.created, measure_lib._XPROF_START_MAX_ATTEMPTS)
    self.assertEmpty(xprof.started)
    self.assertLen(sleeper.sleeps, measure_lib._XPROF_START_MAX_ATTEMPTS - 1)

  def test_start_session_does_not_retry_other_errors(self):
    xprof = self.install_xprof(
        fakes.FakeXprofSessionFactory(start_error=RuntimeError("other"))
    )
    sleeper = self.install_sleep()

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        RuntimeError, "other"
    ):
      with measure_lib.XprofContext(name="broken", enable_xprof=True):
        pass

    self.assertLen(xprof.created, 1)
    self.assertEmpty(sleeper.sleeps)


class MeasureWarmupXprofTest(_XprofTestCase):

  def test_warmup_with_xprof_enabled(self):
    step, ops = fakes.CountingStep(), fakes.RecordingOps()
    xprof = self.install_xprof(
        fakes.FakeXprofSessionFactory(session_id="warmup_sess_id")
    )

    with flagsaver.flagsaver(min_warmup_steps=2, max_warmup_steps=10):
      result = measure_lib._warmup_run(
          step, ops, name="test_warmup", enable_xprof=True
      )

    (session,) = xprof.started
    # Warmup traces only the host, so it does not perturb device timings.
    self.assertEqual(session.start_kwargs["trace_mode"], "TRACE_ONLY_HOST")
    self.assertTrue(session.ended)
    self.assertEqual(
        result.warmup_session_xprof_url,
        "http://xprof/?session_id=warmup_sess_id",
    )
    self.assertEqual(step.calls, 2)

  def test_warmup_with_xprof_disabled(self):
    step, ops = fakes.CountingStep(), fakes.RecordingOps()
    xprof = self.install_xprof(fakes.FakeXprofSessionFactory())
    with flagsaver.flagsaver(min_warmup_steps=2, max_warmup_steps=10):
      result = measure_lib._warmup_run(
          step, ops, name="test_warmup", enable_xprof=False
      )
    self.assertEmpty(xprof.created)
    self.assertIsNone(result.warmup_session_xprof_url)
    self.assertEqual(step.calls, 2)


class MeasurePostWarmupXprofTest(_XprofTestCase):

  def test_post_warmup_with_xprof_enabled_runs_additional_profiling_steps(self):
    step, ops = fakes.CountingStep(), fakes.RecordingOps()
    xprof = self.install_xprof(
        fakes.FakeXprofSessionFactory(session_id="post_warmup_sess_id")
    )

    with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
      result = measure_lib._post_warmup_run(
          step, ops, name="test_post_warmup", enable_xprof=True
      )

    (session,) = xprof.started
    self.assertNotIn("trace_mode", session.start_kwargs)
    self.assertTrue(session.ended)
    self.assertEqual(
        result.post_warmup_run_session_xprof_url,
        "http://xprof/?session_id=post_warmup_sess_id",
    )
    # RecordingOps cannot parse an XSpace, so device time is unavailable.
    self.assertEqual(result.average_post_warmup_device_time_seconds, -1.0)
    # 3 non-profiling steps + 2 profiling steps = 5 steps total
    self.assertEqual(step.calls, 5)

  def test_post_warmup_with_xprof_device_time_calculated(self):
    # Two profiled steps, 20 ms of XLA modules each, on the busiest device.
    xspace = fakes.make_xspace({
        "/device:TPU:0": [20 * _ONE_MS_IN_PS, 20 * _ONE_MS_IN_PS],
        "/device:TPU:1": [5 * _ONE_MS_IN_PS, 5 * _ONE_MS_IN_PS],
        "/host:CPU": [1000 * _ONE_MS_IN_PS],
    })
    self.install_xprof(fakes.FakeXprofSessionFactory(xspace=xspace))
    step, ops = fakes.CountingStep(), XspaceDeviceTimeOps()

    with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
      result = measure_lib._post_warmup_run(
          step, ops, name="test_post_warmup", enable_xprof=True
      )

    self.assertAlmostEqual(result.average_post_warmup_device_time_seconds, 0.02)

  def test_post_warmup_device_time_ignores_shared_session_id(self):
    """Device time must come from this process's XSpace, not the session id.

    xprof can hand the same session id to several benchmark processes. Looking
    the trace up by id then returns whichever of them was slowest, so the
    session-id path must not be consulted at all.
    """
    xspace = fakes.make_xspace(
        {"/device:TPU:0": [20 * _ONE_MS_IN_PS, 20 * _ONE_MS_IN_PS]}
    )
    self.install_xprof(
        fakes.FakeXprofSessionFactory(
            session_id="shared_sess_id", xspace=xspace
        )
    )
    step, ops = fakes.CountingStep(), XspaceDeviceTimeOps()

    with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
      result = measure_lib._post_warmup_run(
          step, ops, name="test_post_warmup", enable_xprof=True
      )

    self.assertEqual(ops.session_id_lookups, 0)
    self.assertAlmostEqual(result.average_post_warmup_device_time_seconds, 0.02)

  def test_post_warmup_with_xprof_disabled_does_not_run_profiling_steps(self):
    step, ops = fakes.CountingStep(), fakes.RecordingOps()
    xprof = self.install_xprof(fakes.FakeXprofSessionFactory())
    with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
      result = measure_lib._post_warmup_run(
          step, ops, name="test_post_warmup", enable_xprof=False
      )
    self.assertEmpty(xprof.created)
    self.assertIsNone(result.post_warmup_run_session_xprof_url)
    # Only 3 non-profiling steps
    self.assertEqual(step.calls, 3)

  def test_post_warmup_recompile_during_profiling_steps_raises(self):
    # Calls 1, 2, 3 are non-profiling steps. Call 4 is 1st profiling step.
    step = fakes.CountingStep()
    ops = fakes.RecordingOps(recompile_on_calls={4})
    xprof = self.install_xprof(fakes.FakeXprofSessionFactory())
    with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
      with self.assertRaises(  # ASSERT_RAISES_OK=benchmark
          measure_lib.PostWarmupRecompileError
      ):
        measure_lib._post_warmup_run(
            step, ops, name="test_post_warmup", enable_xprof=True
        )
    # The profiling session is still closed when the guard fires.
    self.assertTrue(xprof.created[0].ended)

  def test_full_measure_with_xprof_enabled(self):
    step, ops = fakes.CountingStep(), fakes.RecordingOps()
    xprof = self.install_xprof(
        fakes.FakeXprofSessionFactory(
            fakes.FakeXprofSession(session_id="warmup_id"),
            fakes.FakeXprofSession(session_id="post_warmup_id"),
        )
    )

    m = _run_measure(
        step, ops, steps=3, min_warmup=2, xprof_steps=2, enable_xprof=True
    )

    # 2 warmup + 3 post warmup non-profiling + 2 post warmup profiling = 7 steps
    self.assertEqual(step.calls, 7)
    self.assertTrue(all(session.ended for session in xprof.created))
    self.assertEqual(
        m.warmup_session_xprof_url, "http://xprof/?session_id=warmup_id"
    )
    self.assertEqual(
        m.post_warmup_run_session_xprof_url,
        "http://xprof/?session_id=post_warmup_id",
    )

  def test_post_warmup_with_event_device_time_when_xprof_disabled(self):
    step, ops = fakes.CountingStep(), _EventTimingOps([0.012, 0.018])
    xprof = self.install_xprof(fakes.FakeXprofSessionFactory())
    with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
      result = measure_lib._post_warmup_run(
          step, ops, name="test_cuda_event_time", enable_xprof=False
      )
    self.assertEmpty(xprof.created)
    self.assertIsNone(result.post_warmup_run_session_xprof_url)
    # 3 E2E wall-time steps + 2 event-timed steps = 5 steps total.
    self.assertEqual(step.calls, 5)
    self.assertEqual(ops.checked, ["out-1", "out-2", "out-3", "out-4", "out-5"])
    self.assertAlmostEqual(
        result.average_post_warmup_device_time_seconds, 0.015
    )

  def test_post_warmup_with_event_device_time_when_xprof_enabled(self):
    step, ops = fakes.CountingStep(), _EventTimingOps([0.012, 0.018])
    xprof = self.install_xprof(fakes.FakeXprofSessionFactory())
    with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
      result = measure_lib._post_warmup_run(
          step, ops, name="test_cuda_event_time_xprof_flag", enable_xprof=True
      )
    # Event timing replaces the post-warmup XProf session.
    self.assertEmpty(xprof.created)
    self.assertEqual(step.calls, 5)
    self.assertAlmostEqual(
        result.average_post_warmup_device_time_seconds, 0.015
    )

  def test_e2e_wall_time_excludes_event_device_time_profiling(self):
    clock = fakes.FakeClock()
    m = _run_measure(
        fakes.CountingStep(clock=clock, step_seconds=1.0),
        _EventTimingOps([0.012, 0.018]),
        steps=3,
        min_warmup=2,
        xprof_steps=2,
        enable_xprof=False,
        clock=clock,
    )

    # 2 warmup steps + 3 post-warmup steps of 1 s each; 2 event-timed steps
    # are excluded from e2e_wall_time_seconds.
    self.assertEqual(m.e2e_wall_time_seconds, 5.0)
    self.assertAlmostEqual(m.average_post_warmup_device_time_seconds, 0.015)

  def test_post_warmup_recompile_during_event_timed_steps_raises(self):
    step, ops = fakes.CountingStep(), _EventTimingOps(recompile_on_calls={4})
    with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
      with self.assertRaises(  # ASSERT_RAISES_OK=benchmark
          measure_lib.PostWarmupRecompileError
      ):
        measure_lib._post_warmup_run(
            step, ops, name="test_cuda_recompile", enable_xprof=False
        )

  def test_post_warmup_hook_per_step_and_final_xprof_hook_outside_session(
      self,
  ):
    events = []

    class HookTrackingStep(fakes.CountingStep):

      def __call__(self):
        events.append("step")
        return super().__call__()

      def pre_warmup_init(self) -> None:
        events.append("pre_warmup_init")
        super().pre_warmup_init()

      def post_warmup_hook(self) -> None:
        events.append("post_warmup_hook")
        super().post_warmup_hook()

    class SessionTrackingOps(fakes.RecordingOps):

      def await_result(self, out) -> None:
        events.append("await")
        super().await_result(out)

      def check_result(self, out) -> None:
        events.append("check")
        super().check_result(out)

    self.install_xprof(
        fakes.FakeXprofSessionFactory(
            _EventLoggingXprofSession(events, "warmup"),
            _EventLoggingXprofSession(events, "post_warmup"),
        )
    )
    _run_measure(
        HookTrackingStep(),
        SessionTrackingOps(),
        steps=2,
        min_warmup=2,
        xprof_steps=2,
        enable_xprof=True,
    )

    # 2 warmup steps + 2 post-warmup steps + 2 profiled steps = 6 steps, so
    # post_warmup_hook must run exactly 6 times (never twice between warmup and
    # post-warmup), and the final profiled step's post_warmup_hook must run
    # AFTER post_warmup_xprof_end so it is not captured in XSpace.
    self.assertEqual(events.count("step"), 6)
    self.assertEqual(events.count("post_warmup_hook"), 6)
    xprof_start_idx = events.index("post_warmup_xprof_start")
    xprof_end_idx = events.index("post_warmup_xprof_end")
    hooks_inside_xprof = events[xprof_start_idx:xprof_end_idx].count(
        "post_warmup_hook"
    )
    hooks_after_xprof = events[xprof_end_idx:].count("post_warmup_hook")
    self.assertEqual(hooks_inside_xprof, 1)
    self.assertEqual(hooks_after_xprof, 1)


if __name__ == "__main__":
  absltest.main()
