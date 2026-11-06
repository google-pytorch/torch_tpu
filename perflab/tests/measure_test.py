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
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from perflab import measure as measure_lib
from perflab import metrics as metrics_lib
from tests import seed_test_utils


class CountingStep:

  def __init__(self, raise_at=None):
    self.calls = 0
    self.raise_at = raise_at

  def __call__(self):
    self.calls += 1
    if self.raise_at is not None and self.calls == self.raise_at:
      raise RuntimeError("failure")
    return f"out-{self.calls}"

  def get_step_fn(self):
    return self

  def compile(self, **kwargs):
    pass

  def pre_warmup_init(self):
    pass

  def post_warmup_hook(self):
    pass


class RecordingOps:

  def __init__(
      self, peak_mb=None, recompile_on_calls=None, fail_check_on_calls=None
  ):
    self.events = []
    self.awaited = []
    self.checked = []
    self.fail_check_on_calls = fail_check_on_calls or set()
    self.reset_count = 0
    self.peak_mb = peak_mb
    self.recompile_on_calls = recompile_on_calls or set()
    self.calls = 0
    self._recompiles = 0

  def await_result(self, out):
    self.events.append("await")
    self.awaited.append(out)
    self.calls += 1
    if self.calls in self.recompile_on_calls:
      self._recompiles += 1

  def check_result(self, out):
    self.events.append("check")
    self.checked.append(out)
    if len(self.checked) in self.fail_check_on_calls:
      raise RuntimeError(f"execution failed for {out}")

  def reset_peak_memory(self):
    self.events.append("reset")
    self.reset_count += 1

  def peak_memory_mb(self):
    return self.peak_mb

  def get_device_time(self, session_id=None):
    return -1.0

  def supports_event_device_time(self) -> bool:
    return False

  def measure_step_device_time(self, step_fn):
    out = step_fn()
    self.await_result(out)
    self.check_result(out)
    return -1.0

  def compile_count(self):
    return self._recompiles


class _EventTimingOps(RecordingOps):
  """Fake RecordingOps that simulates event-timed step device time."""

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


class _FailingXprofSession:
  """Fake XprofSession whose start always fails with the given error."""

  def __init__(self, error):
    self._error = error

  def start_session(self, **kwargs):
    del kwargs
    raise self._error

  def end_session_and_get_xspace_and_url(self, *args, **kwargs):
    raise AssertionError("A session that never started must not be ended.")

  def end_session_and_get_session_id(self, *args, **kwargs):
    raise AssertionError("A session that never started must not be ended.")


class FailingXprofSessionFactory:
  """Fake XprofSession class whose sessions never start.

  By default, start fails because another profiling session is active.
  """

  def __init__(self, error=None):
    self.created = 0
    self._error = error or RuntimeError(
        "UNAVAILABLE: Another profiling session active."
    )

  def __call__(self):
    self.created += 1
    return _FailingXprofSession(self._error)


class RecordingSleep:
  """Fake time.sleep that records the requested durations without sleeping."""

  def __init__(self):
    self.durations = []

  def __call__(self, seconds):
    self.durations.append(seconds)


class FakeClock:
  """Fake time.perf_counter that only advances when told to."""

  def __init__(self):
    self.now = 0.0

  def __call__(self):
    return self.now

  def advance(self, seconds):
    self.now += seconds


class TimedStep(CountingStep):
  """Step that takes `step_seconds` on `clock`.

  If `compile_seconds` is set, the first `get_step_fn()` call takes that long,
  as the single-trace training stepper traces and compiles in `get_step_fn()`.
  """

  def __init__(self, clock, step_seconds=1.0, compile_seconds=0.0):
    super().__init__()
    self._clock = clock
    self._step_seconds = step_seconds
    self._compile_seconds = compile_seconds

  def __call__(self):
    self._clock.advance(self._step_seconds)
    return super().__call__()

  def get_step_fn(self):
    self._clock.advance(self._compile_seconds)
    self._compile_seconds = 0.0
    return self


class _SlowXprofSession:
  """Fake XprofSession whose start and end each take `seconds` on `clock`."""

  def __init__(self, clock, seconds):
    self._clock = clock
    self._seconds = seconds

  def start_session(self, **kwargs):
    del kwargs
    self._clock.advance(self._seconds)

  def end_session_and_get_xspace_and_url(self):
    self._clock.advance(self._seconds)
    return None, "http://xprof/?session_id=slow"


class SequencedStep(CountingStep):
  """Step whose n-th call takes `step_seconds[n]` (else the last) on `clock`."""

  def __init__(self, clock, step_seconds):
    super().__init__()
    self._clock = clock
    self._step_seconds = list(step_seconds)

  def __call__(self):
    self._clock.advance(
        self._step_seconds[min(self.calls, len(self._step_seconds) - 1)]
    )
    return super().__call__()


def _run_measure(
    step,
    ops,
    steps,
    min_warmup,
    max_warmup=None,
    xprof_steps=None,
    name="test",
    enable_xprof=False,
    rewarm_max_steps=0,
) -> metrics_lib.PerformanceMetrics:
  # The re-warm steps are off by default, so that the step counts of the tests
  # don't depend on the (real or fake) step timings.
  flag_kwargs = {
      "min_warmup_steps": min_warmup,
      "max_warmup_steps": (
          max_warmup if max_warmup is not None else min_warmup + 10
      ),
      "post_warmup_steps": steps,
      "post_warmup_rewarm_max_steps": rewarm_max_steps,
  }
  if xprof_steps is not None:
    flag_kwargs["xprof_steps"] = xprof_steps
  with flagsaver.flagsaver(**flag_kwargs):
    return measure_lib.measure(step, ops, name=name, enable_xprof=enable_xprof)


class MeasureGcTest(seed_test_utils.RepeatableTest):

  def setUp(self):
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
    ops = RecordingOps()
    seen = []

    class GcTestStep:

      def get_step_fn(self):
        def step():
          seen.append(gc.isenabled())
          return "out"

        return step

      def compile(self, **kwargs):
        pass

      def pre_warmup_init(self):
        pass

      def post_warmup_hook(self):
        pass

    _run_measure(GcTestStep(), ops, steps=3, min_warmup=2)
    # GC is off during both warmup and post warmup loops.
    self.assertEqual(seen[:2], [False, False])
    self.assertEqual(seen[2:], [False, False, False])

  def test_gc_reenabled_after_normal_run(self):
    _run_measure(CountingStep(), RecordingOps(), steps=2, min_warmup=2)
    self.assertTrue(gc.isenabled())

  def test_gc_reenabled_when_step_raises(self):
    step = CountingStep(raise_at=3)  # 2 warmup, then explode on 1st timed step
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=benchmark
      _run_measure(step, RecordingOps(), steps=3, min_warmup=2)
    self.assertTrue(gc.isenabled(), "GC must be restored even on failure")


class MeasureGcFreezeTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.assertEqual(gc.get_freeze_count(), 0, "precondition: nothing frozen")
    self.addCleanup(gc.unfreeze)

  def test_heap_frozen_during_measure_and_unfrozen_after(self):
    freeze_counts = []

    class FreezeTrackingStep(CountingStep):

      def __call__(self):
        freeze_counts.append(gc.get_freeze_count())
        return super().__call__()

    _run_measure(FreezeTrackingStep(), RecordingOps(), steps=2, min_warmup=2)

    self.assertLen(freeze_counts, 4)
    self.assertTrue(all(count > 0 for count in freeze_counts))
    self.assertEqual(gc.get_freeze_count(), 0)

  def test_heap_unfrozen_when_step_raises(self):
    step = CountingStep(raise_at=3)
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=benchmark
      _run_measure(step, RecordingOps(), steps=3, min_warmup=2)
    self.assertEqual(gc.get_freeze_count(), 0)


class MeasureRewarmTest(seed_test_utils.RepeatableTest):

  def _measure(self, step_seconds, rewarm_max_steps, steps=3, ops=None):
    clock = FakeClock()
    step = SequencedStep(clock, step_seconds)
    with mock.patch.object(measure_lib.time, "perf_counter", new=clock):
      m = _run_measure(
          step,
          ops or RecordingOps(),
          steps=steps,
          min_warmup=2,
          rewarm_max_steps=rewarm_max_steps,
      )
    return step, m

  def test_rewarm_runs_until_step_time_settles(self):
    # Warmup: 10, 1. Re-warm: 3, 2, 1.5, 1.1, 1.06 (within 5% of 1.1). Timed:
    # 1, 1, 1.
    step, m = self._measure(
        [10.0, 1.0, 3.0, 2.0, 1.5, 1.1, 1.06, 1.0], rewarm_max_steps=10
    )
    self.assertEqual(step.calls, 2 + 5 + 3)
    self.assertEqual(m.post_warmup_step_time_seconds, 1.0)

  def test_rewarm_runs_two_steps_when_already_warm(self):
    step, m = self._measure([10.0, 1.0], rewarm_max_steps=10)
    self.assertEqual(step.calls, 2 + 2 + 3)
    self.assertEqual(m.post_warmup_step_time_seconds, 1.0)

  def test_rewarm_runs_past_a_stall(self):
    # A stalled re-warm step is followed by one that is much faster.
    step, m = self._measure(
        [10.0, 1.0, 1.0, 50.0, 1.0, 1.0], rewarm_max_steps=10
    )
    self.assertEqual(step.calls, 2 + 4 + 3)
    self.assertEqual(m.post_warmup_step_time_seconds, 1.0)

  def test_rewarm_stops_at_max_steps(self):
    step, m = self._measure([10.0, 1.0, 8.0, 4.0, 2.0, 1.0], rewarm_max_steps=2)
    self.assertEqual(step.calls, 2 + 2 + 3)
    self.assertAlmostEqual(m.post_warmup_step_time_seconds, 4.0 / 3)

  def test_no_rewarm_with_zero_max_steps(self):
    step, m = self._measure([10.0, 1.0, 3.0, 1.0], rewarm_max_steps=0)
    self.assertEqual(step.calls, 2 + 3)
    self.assertAlmostEqual(m.post_warmup_step_time_seconds, 5.0 / 3)

  def test_peak_reset_happens_before_rewarm(self):
    ops = RecordingOps()
    self._measure([10.0, 1.0, 3.0, 1.0], rewarm_max_steps=10, steps=2, ops=ops)
    # 2 warmup steps, then the reset, then 3 re-warm and 2 timed steps.
    reset_at = ops.events.index("reset")
    self.assertEqual(ops.events[:reset_at].count("await"), 2)
    self.assertEqual(ops.events[reset_at:].count("await"), 5)


class MeasureSequenceTest(seed_test_utils.RepeatableTest):

  def test_run_step_called_warmup_plus_steps_times(self):
    step, ops = CountingStep(), RecordingOps()
    _run_measure(step, ops, steps=4, min_warmup=2)
    self.assertEqual(step.calls, 6)

  def test_run_step_called_warmup_plus_steps_times_with_recompile(self):
    step, ops = CountingStep(), RecordingOps(recompile_on_calls={1, 2, 3, 4})
    _run_measure(step, ops, steps=4, min_warmup=2)
    # Compilation stablises on 5th warmup step + 4 timed steps = 9 steps
    self.assertEqual(step.calls, 9)

  def test_peak_reset_happens_after_warmup_and_before_timed_loop(self):
    step, ops = CountingStep(), RecordingOps(recompile_on_calls={2})
    _run_measure(step, ops, steps=4, min_warmup=2)

    reset_at = ops.events.index("reset")
    awaits_before = ops.events[:reset_at].count("await")
    awaits_after = ops.events[reset_at:].count("await")

    self.assertEqual(ops.reset_count, 1, "peak must be reset exactly once")
    self.assertEqual(awaits_before, 3, "reset must come after ALL warmup steps")
    self.assertEqual(awaits_after, 4, "reset must come before ALL timed steps")

  def test_await_result_called_once_per_step_including_warmup(self):
    step, ops = CountingStep(), RecordingOps()
    _run_measure(step, ops, steps=4, min_warmup=2)
    self.assertLen(ops.awaited, 6)

  def test_await_result_receives_the_steps_output_untouched(self):
    """The pass-through contract.

    torchax's block_until_ready needs the actual value; the runner must not
    inspect, unwrap, or replace it.
    """
    step, ops = CountingStep(), RecordingOps()
    _run_measure(step, ops, steps=2, min_warmup=2)
    self.assertEqual(ops.awaited, ["out-1", "out-2", "out-3", "out-4"])

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

    class _TrackingOps(RecordingOps):

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


class MeasureCheckResultTest(seed_test_utils.RepeatableTest):

  def test_check_result_follows_every_await(self):
    step, ops = CountingStep(), RecordingOps()
    _run_measure(step, ops, steps=4, min_warmup=2)

    steps = [e for e in ops.events if e in ("await", "check")]
    self.assertEqual(steps, ["await", "check"] * 6)
    self.assertEqual(ops.checked, ops.awaited)

  def test_check_result_runs_once_per_profiled_step(self):
    step, ops = CountingStep(), RecordingOps()
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      mock_session_cls.return_value.end_session_and_get_xspace_and_url.return_value = (
          None,
          "http://xprof/?session_id=sess_id",
      )
      with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
        measure_lib._post_warmup_run(
            step, ops, name="test_post_warmup", enable_xprof=True
        )

    # 3 timed steps + 2 profiled steps.
    self.assertEqual(ops.checked, ["out-1", "out-2", "out-3", "out-4", "out-5"])

  def test_failed_warmup_step_raises(self):
    # A failed asynchronous execution must fail the benchmark instead of
    # being timed as if the step had run.
    ops = RecordingOps(fail_check_on_calls={1})
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        RuntimeError, "execution failed for out-1"
    ):
      _run_measure(CountingStep(), ops, steps=3, min_warmup=2)

  def test_failed_timed_step_raises(self):
    ops = RecordingOps(fail_check_on_calls={4})
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        RuntimeError, "execution failed for out-4"
    ):
      _run_measure(CountingStep(), ops, steps=3, min_warmup=2)
    self.assertTrue(gc.isenabled())

  def test_check_result_is_not_timed(self):
    clock = [0.0]

    class TimedStep(CountingStep):

      def __call__(self):
        clock[0] += 1.0
        return super().__call__()

    class SlowCheckOps(RecordingOps):

      def check_result(self, out):
        clock[0] += 100.0
        super().check_result(out)

    with mock.patch.object(
        measure_lib.time, "perf_counter", side_effect=lambda: clock[0]
    ):
      m = _run_measure(TimedStep(), SlowCheckOps(), steps=3, min_warmup=2)

    self.assertEqual(m.first_step_time_seconds, 1.0)
    self.assertEqual(m.post_warmup_step_time_seconds, 1.0)


class MeasureRecompileGuardTest(seed_test_utils.RepeatableTest):

  def test_no_recompiles_finds_min_warmup_steps(self):
    m = _run_measure(CountingStep(), RecordingOps(), steps=3, min_warmup=2)
    self.assertEqual(m.num_warmup_steps, 2)

  def test_recompile_during_timed_loop_raises(self):
    # min_warmup=2 => calls 1,2 are warmup; 3,4,5 are timed. Recompile on call 4.
    ops = RecordingOps(recompile_on_calls={4})
    with self.assertRaises(  # ASSERT_RAISES_OK=benchmark
        measure_lib.PostWarmupRecompileError
    ):
      _run_measure(
          CountingStep(),
          ops,
          steps=3,
          min_warmup=2,
          name="llama_compiled",
      )

  def test_recompiles_during_warmup_are_fine(self):
    ops = RecordingOps(recompile_on_calls={1, 2})  # both during warmup
    m = _run_measure(CountingStep(), ops, steps=3, min_warmup=3)
    self.assertEqual(m.num_warmup_steps, 3)

  def test_gc_reenabled_when_recompile_guard_fires(self):
    ops = RecordingOps(recompile_on_calls={4})
    with self.assertRaises(  # ASSERT_RAISES_OK=benchmark
        measure_lib.PostWarmupRecompileError
    ):
      _run_measure(CountingStep(), ops, steps=3, min_warmup=2)
    self.assertTrue(gc.isenabled())

  def test_recompiles_not_stabilized_during_warmup_raises(self):
    # Set max_warmup=3. Recompile on every warmup step (calls 1, 2, 3).
    ops = RecordingOps(recompile_on_calls={1, 2, 3})
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        RuntimeError,
        "Benchmark function compilations have not stabilized",
    ):
      _run_measure(CountingStep(), ops, steps=3, min_warmup=2, max_warmup=3)


class MeasureMetricsTest(seed_test_utils.RepeatableTest):

  def test_peak_memory_passed_through(self):
    m = _run_measure(
        CountingStep(),
        RecordingOps(peak_mb=512.0),
        steps=2,
        min_warmup=2,
    )
    self.assertEqual(m.peak_device_memory_mb, 512.0)

  def test_peak_memory_none_when_backend_cannot_report(self):
    m = _run_measure(
        CountingStep(),
        RecordingOps(peak_mb=None),
        steps=2,
        min_warmup=2,
    )
    self.assertIsNone(m.peak_device_memory_mb)

  @mock.patch("time.perf_counter")
  def test_timing_metrics_are_populated_correctly(self, mock_perf_counter):
    mock_perf_counter.side_effect = [
        0.0,  # e2e measure start
        0.0,
        10.0,  # warmup step 0
        10.0,
        12.0,  # warmup step 1
        12.0,
        14.0,  # warmup step 2 (stabilizes here because min_warmup=3 and compile_counts match)
        14.0,
        16.0,  # post warmup step 0
        16.0,
        19.0,  # post warmup step 1
        19.0,
        23.0,  # post warmup step 2
        23.0,  # e2e measure end
    ]
    m = _run_measure(
        CountingStep(),
        RecordingOps(),
        steps=3,
        min_warmup=3,
    )

    self.assertEqual(m.first_step_time_seconds, 10.0)
    # Post warmup timings: 2.0, 3.0, 4.0 -> mean = 3.0
    self.assertEqual(m.post_warmup_step_time_seconds, 3.0)
    # Warmup timings: 10.0, 2.0, 2.0. Stabilized time is 2.0.
    # overhead = (10.0 + 2.0 + 2.0) - (2.0 * 3) = 8.0
    self.assertEqual(m.warmup_overhead_seconds, 8.0)

  def test_first_step_time_includes_compilation_in_get_step_fn(self):
    clock = FakeClock()
    step = TimedStep(clock, step_seconds=1.0, compile_seconds=100.0)
    with mock.patch.object(measure_lib.time, "perf_counter", new=clock):
      m = _run_measure(step, RecordingOps(), steps=3, min_warmup=2)

    self.assertEqual(m.first_step_time_seconds, 101.0)
    # Warmup timings: 101.0, 1.0 -> overhead = 102.0 - 1.0 * 2 = 100.0
    self.assertEqual(m.warmup_overhead_seconds, 100.0)
    self.assertEqual(m.post_warmup_step_time_seconds, 1.0)

  def test_e2e_wall_time_excludes_profiling(self):
    def run(enable_xprof):
      clock = FakeClock()
      with (
          mock.patch.object(measure_lib.time, "perf_counter", new=clock),
          mock.patch.object(
              measure_lib.xprof_adapter,
              "XprofSession",
              new=lambda: _SlowXprofSession(clock, seconds=5.0),
          ),
      ):
        return _run_measure(
            TimedStep(clock),
            RecordingOps(),
            steps=3,
            min_warmup=2,
            xprof_steps=2,
            enable_xprof=enable_xprof,
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


class XprofContextTest(seed_test_utils.RepeatableTest):

  def test_xprof_disabled(self):
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      with measure_lib.XprofContext(
          name="test_disabled", enable_xprof=False
      ) as ctx:
        self.assertIsNone(ctx.session)
        self.assertIsNone(ctx.session_id)
      self.assertIsNone(ctx.session)
      self.assertIsNone(ctx.session_id)
      mock_session_cls.assert_not_called()

  def test_xprof_enabled_default(self):
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      mock_session = mock_session_cls.return_value
      sentinel_xspace = mock.MagicMock()
      mock_session.end_session_and_get_xspace_and_url.return_value = (
          sentinel_xspace,
          "http://xprof/?session_id=session_123",
      )

      with measure_lib.XprofContext(
          name="test_session", enable_xprof=True
      ) as ctx:
        self.assertEqual(ctx.session, mock_session)
        mock_session_cls.assert_called_once()
        mock_session.start_session.assert_called_once_with(
            host_trace_level=3,
            enable_python_tracer=True,
            host_cpu_profile=True,
            response_max_bytes=measure_lib._PROFILE_MAX_BYTES,
            session_id=mock.ANY,
        )
        self.assertIsNone(ctx.session_id)
        self.assertIsNone(ctx.xspace)

      mock_session.end_session_and_get_xspace_and_url.assert_called_once()
      self.assertEqual(ctx.session_id, "session_123")
      self.assertIs(ctx.xspace, sentinel_xspace)

  def test_xprof_enabled_trace_only_host(self):
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      mock_session = mock_session_cls.return_value
      mock_session.end_session_and_get_xspace_and_url.return_value = (
          mock.MagicMock(),
          "http://xprof/?session_id=session_host_only",
      )

      with measure_lib.XprofContext(
          name="host_trace", enable_xprof=True, trace_only_host=True
      ) as ctx:
        mock_session.start_session.assert_called_once_with(
            host_trace_level=1,
            enable_python_tracer=False,
            host_cpu_profile=True,
            response_max_bytes=measure_lib._PROFILE_MAX_BYTES,
            session_id=mock.ANY,
            trace_mode="TRACE_ONLY_HOST",
        )

      mock_session.end_session_and_get_xspace_and_url.assert_called_once()
      self.assertEqual(ctx.session_id, "session_host_only")

  def test_xprof_enabled_exception_propagates_and_session_ends(self):
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      mock_session = mock_session_cls.return_value
      mock_session.end_session_and_get_xspace_and_url.return_value = (
          mock.MagicMock(),
          "http://xprof/?session_id=session_err",
      )

      with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
        with measure_lib.XprofContext(
            name="err_trace", enable_xprof=True
        ) as ctx:
          raise ValueError("test error")

      mock_session.end_session_and_get_xspace_and_url.assert_called_once()
      self.assertEqual(ctx.session_id, "session_err")

  def test_xprof_falls_back_when_xspace_api_unavailable(self):
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      mock_session = mock_session_cls.return_value
      mock_session.end_session_and_get_xspace_and_url.side_effect = (
          AttributeError("unsupported")
      )
      mock_session.end_session_and_get_session_id.return_value = "fallback_id"

      with measure_lib.XprofContext(name="fallback", enable_xprof=True) as ctx:
        pass

      mock_session.end_session_and_get_session_id.assert_called_once()
      self.assertEqual(ctx.session_id, "fallback_id")
      # Without an XSpace, device time is reported as unavailable rather than
      # being resolved through a possibly-shared session id.
      self.assertIsNone(ctx.xspace)

  def test_each_session_gets_a_distinct_session_id(self):
    """Distinct ids are what stop benchmarks sharing one XProf link."""
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      mock_session = mock_session_cls.return_value
      mock_session.end_session_and_get_xspace_and_url.return_value = (
          mock.MagicMock(),
          "http://xprof/?session_id=x",
      )

      for _ in range(25):
        with measure_lib.XprofContext(name="s", enable_xprof=True):
          pass

      ids = [
          call.kwargs["session_id"]
          for call in mock_session.start_session.call_args_list
      ]
      self.assertLen(ids, 25)
      self.assertLen(set(ids), 25)
      for session_id in ids:
        self.assertIsInstance(session_id, int)
        # Must fit in the uint64 the xprof API expects.
        self.assertGreater(session_id, 0)
        self.assertLess(session_id, 1 << 64)

  def test_start_session_falls_back_when_session_id_unsupported(self):
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      mock_session = mock_session_cls.return_value

      def start_session(**kwargs):
        if "session_id" in kwargs:
          raise TypeError("unexpected keyword argument 'session_id'")

      mock_session.start_session.side_effect = start_session
      mock_session.end_session_and_get_xspace_and_url.return_value = (
          mock.MagicMock(),
          "http://xprof/?session_id=legacy",
      )

      with measure_lib.XprofContext(name="legacy", enable_xprof=True) as ctx:
        pass

      self.assertEqual(mock_session.start_session.call_count, 2)
      self.assertNotIn(
          "session_id", mock_session.start_session.call_args_list[1].kwargs
      )
      self.assertEqual(ctx.session_id, "legacy")

  def test_start_session_retries_with_fresh_session_when_another_session_active(
      self,
  ):
    contended_session = mock.MagicMock()
    contended_session.start_session.side_effect = RuntimeError(
        "UNAVAILABLE: Another profiling session active."
    )
    active_session = mock.MagicMock()
    sentinel_xspace = mock.MagicMock()
    active_session.end_session_and_get_xspace_and_url.return_value = (
        sentinel_xspace,
        "http://xprof/?session_id=recovered_id",
    )

    with (
        mock.patch.object(
            measure_lib.xprof_adapter,
            "XprofSession",
            side_effect=[contended_session, active_session],
        ) as mock_session_cls,
        mock.patch.object(measure_lib.time, "sleep") as mock_sleep,
    ):
      with measure_lib.XprofContext(name="contended", enable_xprof=True) as ctx:
        self.assertIs(ctx.session, active_session)

      self.assertEqual(mock_session_cls.call_count, 2)
      mock_sleep.assert_called_once_with(
          measure_lib._XPROF_START_RETRY_DELAY_SECONDS
      )
      contended_session.end_session_and_get_xspace_and_url.assert_not_called()
      active_session.end_session_and_get_xspace_and_url.assert_called_once()
      self.assertEqual(ctx.session_id, "recovered_id")
      self.assertIs(ctx.xspace, sentinel_xspace)

  def test_start_session_raises_after_exhausting_retries(self):
    session_factory = FailingXprofSessionFactory()
    sleeper = RecordingSleep()

    with (
        mock.patch.object(
            measure_lib.xprof_adapter, "XprofSession", new=session_factory
        ),
        mock.patch.object(measure_lib.time, "sleep", new=sleeper),
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
          RuntimeError, "Another profiling session active"
      ):
        with measure_lib.XprofContext(name="contended", enable_xprof=True):
          pass

    self.assertEqual(
        session_factory.created, measure_lib._XPROF_START_MAX_ATTEMPTS
    )
    self.assertEqual(
        sleeper.durations, [measure_lib._XPROF_START_RETRY_DELAY_SECONDS]
    )

  def test_start_session_other_errors_still_raise(self):
    session_factory = FailingXprofSessionFactory(
        error=RuntimeError("INVALID_ARGUMENT: bad trace mode")
    )
    with (
        mock.patch.object(
            measure_lib.xprof_adapter, "XprofSession", new=session_factory
        ),
        mock.patch.object(measure_lib.time, "sleep", new=RecordingSleep()),
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
          RuntimeError, "bad trace mode"
      ):
        with measure_lib.XprofContext(name="bad", enable_xprof=True):
          pass

    self.assertEqual(session_factory.created, 1)


class MeasureWarmupXprofTest(seed_test_utils.RepeatableTest):

  def test_warmup_with_xprof_enabled(self):
    step, ops = CountingStep(), RecordingOps()
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      mock_session = mock_session_cls.return_value
      mock_session.end_session_and_get_xspace_and_url.return_value = (
          mock.MagicMock(),
          "http://xprof/?session_id=warmup_sess_id",
      )

      with flagsaver.flagsaver(min_warmup_steps=2, max_warmup_steps=10):
        result = measure_lib._warmup_run(
            step, ops, name="test_warmup", enable_xprof=True
        )

      mock_session.start_session.assert_called_once_with(
          host_trace_level=1,
          enable_python_tracer=False,
          host_cpu_profile=True,
          response_max_bytes=measure_lib._PROFILE_MAX_BYTES,
          session_id=mock.ANY,
          trace_mode="TRACE_ONLY_HOST",
      )
      mock_session.end_session_and_get_xspace_and_url.assert_called_once()
      self.assertEqual(
          result.warmup_session_xprof_url,
          "http://xprof/?session_id=warmup_sess_id",
      )
      self.assertEqual(step.calls, 2)

  def test_warmup_with_xprof_disabled(self):
    step, ops = CountingStep(), RecordingOps()
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      with flagsaver.flagsaver(min_warmup_steps=2, max_warmup_steps=10):
        result = measure_lib._warmup_run(
            step, ops, name="test_warmup", enable_xprof=False
        )
      mock_session_cls.assert_not_called()
      self.assertIsNone(result.warmup_session_xprof_url)
      self.assertEqual(step.calls, 2)


class MeasurePostWarmupXprofTest(seed_test_utils.RepeatableTest):

  def test_post_warmup_with_xprof_enabled_runs_additional_profiling_steps(self):
    step, ops = CountingStep(), RecordingOps()
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      mock_session = mock_session_cls.return_value
      mock_session.end_session_and_get_xspace_and_url.return_value = (
          mock.MagicMock(),
          "http://xprof/?session_id=post_warmup_sess_id",
      )

      with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
        result = measure_lib._post_warmup_run(
            step, ops, name="test_post_warmup", enable_xprof=True
        )

      mock_session.start_session.assert_called_once_with(
          host_trace_level=3,
          enable_python_tracer=True,
          host_cpu_profile=True,
          response_max_bytes=measure_lib._PROFILE_MAX_BYTES,
          session_id=mock.ANY,
      )
      mock_session.end_session_and_get_xspace_and_url.assert_called_once()
      self.assertEqual(
          result.post_warmup_run_session_xprof_url,
          "http://xprof/?session_id=post_warmup_sess_id",
      )
      self.assertEqual(result.average_post_warmup_device_time_seconds, -1.0)
      # 3 non-profiling steps + 2 profiling steps = 5 steps total
      self.assertEqual(step.calls, 5)

  def test_post_warmup_with_xprof_device_time_calculated(self):
    sentinel_xspace = mock.MagicMock()

    class DeviceTimeOps(RecordingOps):

      def get_device_time_from_xspace(self, xspace, expected_steps=None):
        if xspace is sentinel_xspace and expected_steps == 2:
          return 0.04
        return -1.0

    step, ops = CountingStep(), DeviceTimeOps()
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      mock_session = mock_session_cls.return_value
      mock_session.end_session_and_get_xspace_and_url.return_value = (
          sentinel_xspace,
          "http://xprof/?session_id=post_warmup_sess_id",
      )

      with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
        result = measure_lib._post_warmup_run(
            step, ops, name="test_post_warmup", enable_xprof=True
        )

      self.assertAlmostEqual(
          result.average_post_warmup_device_time_seconds, 0.02
      )

  def test_post_warmup_device_time_ignores_shared_session_id(self):
    """Device time must come from this process's XSpace, not the session id.

    xprof can hand the same session id to several benchmark processes. Looking
    the trace up by id then returns whichever of them was slowest, so the
    session-id path must not be consulted at all.
    """
    sentinel_xspace = mock.MagicMock()

    class SharedSessionOps(RecordingOps):

      def __init__(self):
        super().__init__()
        self.session_id_lookups = 0

      def get_device_time(self, session_id=None, expected_steps=None):
        # Would return another benchmark's (much larger) device time.
        self.session_id_lookups += 1
        return 10.0

      def get_device_time_from_xspace(self, xspace, expected_steps=None):
        del expected_steps
        return 0.04 if xspace is sentinel_xspace else -1.0

    step, ops = CountingStep(), SharedSessionOps()
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      mock_session = mock_session_cls.return_value
      mock_session.end_session_and_get_xspace_and_url.return_value = (
          sentinel_xspace,
          "http://xprof/?session_id=shared_sess_id",
      )

      with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
        result = measure_lib._post_warmup_run(
            step, ops, name="test_post_warmup", enable_xprof=True
        )

      self.assertEqual(ops.session_id_lookups, 0)
      self.assertAlmostEqual(
          result.average_post_warmup_device_time_seconds, 0.02
      )

  def test_post_warmup_with_xprof_disabled_does_not_run_profiling_steps(self):
    step, ops = CountingStep(), RecordingOps()
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
        result = measure_lib._post_warmup_run(
            step, ops, name="test_post_warmup", enable_xprof=False
        )
      mock_session_cls.assert_not_called()
      self.assertIsNone(result.post_warmup_run_session_xprof_url)
      # Only 3 non-profiling steps
      self.assertEqual(step.calls, 3)

  def test_post_warmup_recompile_during_profiling_steps_raises(self):
    # Calls 1, 2, 3 are non-profiling steps. Call 4 is 1st profiling step.
    step, ops = CountingStep(), RecordingOps(recompile_on_calls={4})
    with mock.patch.object(measure_lib.xprof_adapter, "XprofSession"):
      with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
        with self.assertRaises(  # ASSERT_RAISES_OK=benchmark
            measure_lib.PostWarmupRecompileError
        ):
          measure_lib._post_warmup_run(
              step, ops, name="test_post_warmup", enable_xprof=True
          )

  def test_full_measure_with_xprof_enabled(self):
    step, ops = CountingStep(), RecordingOps()
    warmup_session = mock.MagicMock()
    warmup_session.end_session_and_get_session_id.return_value = "warmup_id"
    post_warmup_session = mock.MagicMock()
    post_warmup_session.end_session_and_get_session_id.return_value = (
        "post_warmup_id"
    )

    with mock.patch.object(
        measure_lib.xprof_adapter,
        "XprofSession",
        side_effect=[warmup_session, post_warmup_session],
    ):
      m = _run_measure(
          step, ops, steps=3, min_warmup=2, xprof_steps=2, enable_xprof=True
      )

      # 2 warmup + 3 post warmup non-profiling + 2 post warmup profiling = 7 steps
      self.assertEqual(step.calls, 7)
      self.assertEqual(
          m.warmup_session_xprof_url, "http://xprof/?session_id=warmup_id"
      )
      self.assertEqual(
          m.post_warmup_run_session_xprof_url,
          "http://xprof/?session_id=post_warmup_id",
      )

  def test_post_warmup_with_event_device_time_when_xprof_disabled(self):
    step, ops = CountingStep(), _EventTimingOps([0.012, 0.018])
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
        result = measure_lib._post_warmup_run(
            step, ops, name="test_cuda_event_time", enable_xprof=False
        )
      mock_session_cls.assert_not_called()
      self.assertIsNone(result.post_warmup_run_session_xprof_url)
      # 3 E2E wall-time steps + 2 event-timed steps = 5 steps total.
      self.assertEqual(step.calls, 5)
      self.assertEqual(
          ops.checked, ["out-1", "out-2", "out-3", "out-4", "out-5"]
      )
      self.assertAlmostEqual(
          result.average_post_warmup_device_time_seconds, 0.015
      )

  def test_post_warmup_with_event_device_time_when_xprof_enabled(self):
    step, ops = CountingStep(), _EventTimingOps([0.012, 0.018])
    with mock.patch.object(
        measure_lib.xprof_adapter, "XprofSession"
    ) as mock_session_cls:
      with flagsaver.flagsaver(post_warmup_steps=3, xprof_steps=2):
        result = measure_lib._post_warmup_run(
            step, ops, name="test_cuda_event_time_xprof_flag", enable_xprof=True
        )
      mock_session_cls.assert_not_called()
      self.assertEqual(step.calls, 5)
      self.assertAlmostEqual(
          result.average_post_warmup_device_time_seconds, 0.015
      )

  def test_e2e_wall_time_excludes_event_device_time_profiling(self):
    clock = FakeClock()
    with mock.patch.object(measure_lib.time, "perf_counter", new=clock):
      m = _run_measure(
          TimedStep(clock, step_seconds=1.0),
          _EventTimingOps([0.012, 0.018]),
          steps=3,
          min_warmup=2,
          xprof_steps=2,
          enable_xprof=False,
      )

    # 2 warmup steps + 3 post-warmup steps of 1 s each; 2 event-timed steps
    # are excluded from e2e_wall_time_seconds.
    self.assertEqual(m.e2e_wall_time_seconds, 5.0)
    self.assertAlmostEqual(m.average_post_warmup_device_time_seconds, 0.015)

  def test_post_warmup_recompile_during_event_timed_steps_raises(self):
    step, ops = CountingStep(), _EventTimingOps(recompile_on_calls={4})
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

    class HookTrackingStep(CountingStep):

      def __call__(self):
        events.append("step")
        return super().__call__()

      def pre_warmup_init(self) -> None:
        events.append("pre_warmup_init")

      def post_warmup_hook(self) -> None:
        events.append("post_warmup_hook")

    class SessionTrackingOps(RecordingOps):

      def await_result(self, out) -> None:
        events.append("await")
        super().await_result(out)

      def check_result(self, out) -> None:
        events.append("check")
        super().check_result(out)

    def _end_warmup_session():
      events.append("warmup_xprof_end")
      return None, "http://xprof/?session_id=warmup_id"

    def _start_post_warmup_session(**_):
      events.append("post_warmup_xprof_start")

    def _end_post_warmup_session():
      events.append("post_warmup_xprof_end")
      return None, "http://xprof/?session_id=post_warmup_id"

    warmup_session = mock.MagicMock()
    warmup_session.end_session_and_get_xspace_and_url.side_effect = (
        _end_warmup_session
    )
    post_warmup_session = mock.MagicMock()
    post_warmup_session.start_session.side_effect = _start_post_warmup_session
    post_warmup_session.end_session_and_get_xspace_and_url.side_effect = (
        _end_post_warmup_session
    )

    with mock.patch.object(
        measure_lib.xprof_adapter,
        "XprofSession",
        side_effect=[warmup_session, post_warmup_session],
    ):
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
