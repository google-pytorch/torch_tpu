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
from examples.benchmarks.e2e.harness import measure as measure_lib
from examples.benchmarks.e2e.harness import metrics as metrics_lib
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

  def __init__(self, peak_mb=None, recompile_on_calls=None):
    self.events = []
    self.awaited = []
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

  def reset_peak_memory(self):
    self.events.append("reset")
    self.reset_count += 1

  def peak_memory_mb(self):
    return self.peak_mb

  def get_device_time(self, session_id=None):
    return -1.0

  def compile_count(self):
    return self._recompiles


def _run_measure(
    step,
    ops,
    steps,
    min_warmup,
    max_warmup=None,
    xprof_steps=None,
    name="test",
    enable_xprof=False,
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
    with self.assertRaises(RuntimeError):
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
    with self.assertRaises(RuntimeError):
      _run_measure(step, RecordingOps(), steps=3, min_warmup=2)
    self.assertTrue(gc.isenabled(), "GC must be restored even on failure")


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


class MeasureRecompileGuardTest(seed_test_utils.RepeatableTest):

  def test_no_recompiles_finds_min_warmup_steps(self):
    m = _run_measure(CountingStep(), RecordingOps(), steps=3, min_warmup=2)
    self.assertEqual(m.num_warmup_steps, 2)

  def test_recompile_during_timed_loop_raises(self):
    # min_warmup=2 => calls 1,2 are warmup; 3,4,5 are timed. Recompile on call 4.
    ops = RecordingOps(recompile_on_calls={4})
    with self.assertRaises(measure_lib.PostWarmupRecompileError):
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
    with self.assertRaises(measure_lib.PostWarmupRecompileError):
      _run_measure(CountingStep(), ops, steps=3, min_warmup=2)
    self.assertTrue(gc.isenabled())

  def test_recompiles_not_stabilized_during_warmup_raises(self):
    # Set max_warmup=3. Recompile on every warmup step (calls 1, 2, 3).
    ops = RecordingOps(recompile_on_calls={1, 2, 3})
    with self.assertRaisesRegex(
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

    default_perf_metrics = metrics_lib.PerformanceMetrics(
        post_warmup_step_time_seconds=0.001234567
    )
    self.assertEqual(
        default_perf_metrics.metric_map()["average_post_warmup_device_time_ms"],
        -1.0,
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

      with self.assertRaises(ValueError):
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
        with self.assertRaises(measure_lib.PostWarmupRecompileError):
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


if __name__ == "__main__":
  absltest.main()
