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

  def compile_count(self):
    return self._recompiles


def _run_measure(
    step, ops, steps, min_warmup, max_warmup=None, name="test"
) -> metrics_lib.PerformanceMetrics:
  with flagsaver.flagsaver(
      min_warmup_steps=min_warmup,
      max_warmup_steps=max_warmup
      if max_warmup is not None
      else min_warmup + 10,
      post_warmup_steps=steps,
  ):
    return measure_lib.measure(step, ops, name=name)


class MeasureGcTest(absltest.TestCase):

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


class MeasureSequenceTest(absltest.TestCase):

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


class MeasureRecompileGuardTest(absltest.TestCase):

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


class MeasureMetricsTest(absltest.TestCase):

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


if __name__ == "__main__":
  absltest.main()
