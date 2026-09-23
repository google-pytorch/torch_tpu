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

"""Unit tests for chip health detection."""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

import fake_gcs
from rbe_tpu.common import health
from rbe_tpu.common import protocol


class StubProbe:
  """A probe the test flips between healthy and broken."""

  def __init__(self, healthy=True):
    self.healthy = healthy
    self.calls = 0
    self.detail = "stub detail"

  def __call__(self) -> health.ProbeResult:
    self.calls += 1
    return health.ProbeResult(self.healthy, "stub", self.detail)


class ClassifyOutputTest(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def test_it_spots_a_runtime_that_never_reached_the_chip(self):
    text = "E0915 ... Failed to initialize TPU: no chips\n"
    self.assertEqual(health.classify_output(text), "Failed to initialize TPU")

  def test_it_spots_a_missing_libtpu(self):
    text = "ImportError: libtpu.so: cannot open shared object file"
    self.assertIn("libtpu.so", health.classify_output(text))

  def test_it_spots_a_device_that_will_not_open(self):
    self.assertEqual(
        health.classify_output("Failed to open /dev/vfio/0: EBUSY"),
        "Failed to open /dev/vfio",
    )

  def test_an_ordinary_assertion_failure_is_not_infrastructure(self):
    text = "AssertionError: tensors differ at index 3\nFAILED tests/x.py"
    self.assertEqual(health.classify_output(text), "")

  def test_out_of_memory_is_not_infrastructure(self):
    """A big test legitimately exhausts HBM. Never quarantine for it."""
    text = "RESOURCE_EXHAUSTED: Failed to allocate 8GB on device 0"
    self.assertEqual(health.classify_output(text), "")

  def test_a_slow_test_is_not_infrastructure(self):
    text = "DEADLINE_EXCEEDED: operation took longer than 600s"
    self.assertEqual(health.classify_output(text), "")

  def test_empty_output_is_not_infrastructure(self):
    self.assertEqual(health.classify_output(""), "")


class DeviceNodeProbeTest(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def test_it_is_happy_when_a_node_is_present(self):
    with mock.patch.object(
        health.glob, "glob", side_effect=[["/dev/accel0"], []]
    ):
      result = health.device_node_probe()
    self.assertTrue(result.healthy)
    self.assertEqual(result.detector, "device_nodes")

  def test_it_is_unhappy_when_every_node_is_gone(self):
    with mock.patch.object(health.glob, "glob", return_value=[]):
      result = health.device_node_probe()
    self.assertFalse(result.healthy)
    self.assertIn("no device nodes", result.detail)

  def test_it_only_checks_presence_and_never_opens_the_device(self):
    """A running test holds the chip, so a failed open is not a fault."""
    opened = []
    with mock.patch.object(health.glob, "glob", return_value=["/dev/accel0"]):
      with mock.patch.object(health.os, "open", side_effect=opened.append):
        health.device_node_probe()
    self.assertEqual(opened, [])


class CommandProbeTest(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def test_exit_zero_means_healthy(self):
    completed = mock.Mock(returncode=0, stdout="chip ok", stderr="")
    with mock.patch.object(health.subprocess, "run", return_value=completed):
      result = health.command_probe("true")
    self.assertTrue(result.healthy)
    self.assertEqual(result.detail, "chip ok")

  def test_a_nonzero_exit_means_broken(self):
    completed = mock.Mock(returncode=1, stdout="", stderr="chip is wedged")
    with mock.patch.object(health.subprocess, "run", return_value=completed):
      result = health.command_probe("false")
    self.assertFalse(result.healthy)
    self.assertEqual(result.detail, "chip is wedged")

  def test_a_hung_probe_counts_as_broken(self):
    error = health.subprocess.TimeoutExpired("probe", 60)
    with mock.patch.object(health.subprocess, "run", side_effect=error):
      result = health.command_probe("sleep 999")
    self.assertFalse(result.healthy)
    self.assertIn("timed out", result.detail)

  def test_a_probe_that_will_not_start_counts_as_broken(self):
    with mock.patch.object(
        health.subprocess, "run", side_effect=OSError("no such file")
    ):
      result = health.command_probe("/nope")
    self.assertFalse(result.healthy)
    self.assertIn("no such file", result.detail)

  def test_the_env_var_selects_the_command(self):
    with mock.patch.dict(
        health.os.environ, {health.PROBE_COMMAND_ENV: "my-probe --flag"}
    ):
      with mock.patch.object(
          health, "command_probe", return_value=health.ProbeResult(True, "x")
      ) as probe:
        health.default_probe()
    probe.assert_called_once_with("my-probe --flag")

  def test_without_the_env_var_it_checks_device_nodes(self):
    with mock.patch.dict(health.os.environ, {}, clear=True):
      with mock.patch.object(
          health,
          "device_node_probe",
          return_value=health.ProbeResult(True, "d"),
      ) as probe:
        health.default_probe()
    probe.assert_called_once_with()


class HealthMonitorTestCase(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def setUp(self):
    super().setUp()
    self.clock = fake_gcs.FakeClock()
    self.probe = StubProbe()
    self.monitor = health.HealthMonitor(self.probe, time_fn=self.clock.time)

  def fail_until_probed(self, text="Failed to initialize TPU"):
    """Feeds enough infrastructure-looking failures to trigger a probe."""
    for _ in range(protocol.INFRA_FAILURES_BEFORE_PROBE):
      usable = self.monitor.observe_failure(text)
    return usable


class BootCheckTest(HealthMonitorTestCase):

  def test_a_working_chip_starts_in_service(self):
    self.assertTrue(self.monitor.check_at_boot())
    self.assertFalse(self.monitor.quarantined)

  def test_a_broken_chip_never_enters_service(self):
    self.probe.healthy = False
    self.assertFalse(self.monitor.check_at_boot())
    self.assertTrue(self.monitor.quarantined)

  def test_it_records_what_found_the_problem(self):
    self.probe.healthy = False
    self.probe.detail = "no device nodes"
    self.monitor.check_at_boot()
    self.assertEqual(self.monitor.detector, "stub")
    self.assertEqual(self.monitor.reason, "no device nodes")


class FailureObservationTest(HealthMonitorTestCase):

  def test_one_suspicious_failure_does_not_trigger_a_probe(self):
    self.monitor.observe_failure("Failed to initialize TPU")
    self.assertEqual(self.probe.calls, 0)

  def test_two_in_a_row_does(self):
    self.fail_until_probed()
    self.assertEqual(self.probe.calls, 1)

  def test_an_ordinary_test_failure_never_probes(self):
    for _ in range(10):
      self.monitor.observe_failure("AssertionError: nope")
    self.assertEqual(self.probe.calls, 0)
    self.assertFalse(self.monitor.quarantined)

  def test_a_clean_run_resets_the_streak(self):
    self.monitor.observe_failure("Failed to initialize TPU")
    self.monitor.observe_success()
    self.monitor.observe_failure("Failed to initialize TPU")
    self.assertEqual(self.probe.calls, 0)

  def test_an_unrelated_failure_also_resets_the_streak(self):
    self.monitor.observe_failure("Failed to initialize TPU")
    self.monitor.observe_failure("AssertionError: nope")
    self.monitor.observe_failure("Failed to initialize TPU")
    self.assertEqual(self.probe.calls, 0)

  def test_the_probe_has_the_final_say(self):
    """Suspicious output plus a healthy chip must not quarantine."""
    self.probe.healthy = True
    usable = self.fail_until_probed()
    self.assertTrue(usable)
    self.assertFalse(self.monitor.quarantined)

  def test_a_confirmed_bad_chip_is_pulled(self):
    self.probe.healthy = False
    usable = self.fail_until_probed()
    self.assertFalse(usable)
    self.assertTrue(self.monitor.quarantined)

  def test_a_cleared_probe_lets_the_streak_start_over(self):
    self.fail_until_probed()
    self.assertEqual(self.probe.calls, 1)
    self.fail_until_probed()
    self.assertEqual(self.probe.calls, 2)


class QuarantineDurationTest(HealthMonitorTestCase):

  def quarantine(self):
    self.probe.healthy = False
    self.monitor.poll()
    self.probe.healthy = True

  def test_the_first_hold_is_the_minimum(self):
    self.quarantine()
    self.assertEqual(
        self.monitor.quarantined_until,
        self.clock.time() + protocol.QUARANTINE_MIN_SECONDS,
    )

  def test_a_repeat_offender_is_held_twice_as_long(self):
    self.quarantine()
    self.recover()
    self.clock.advance(health.IDLE_PROBE_SECONDS)
    start = self.clock.time()
    self.quarantine()
    self.assertEqual(
        self.monitor.quarantined_until,
        start + protocol.QUARANTINE_MIN_SECONDS * 2,
    )

  def test_the_hold_is_capped(self):
    self.monitor.quarantine_count = 40
    self.probe.healthy = False
    start = self.clock.time()
    self.monitor.poll()
    self.assertEqual(
        self.monitor.quarantined_until,
        start + protocol.QUARANTINE_MAX_SECONDS,
    )

  def recover(self):
    """Drives the monitor back to healthy."""
    self.clock.advance(protocol.QUARANTINE_MAX_SECONDS)
    for _ in range(protocol.PROBES_TO_RECOVER):
      self.clock.advance(health.IDLE_PROBE_SECONDS)
      self.monitor.poll()
    assert not self.monitor.quarantined


class RecoveryTest(HealthMonitorTestCase):

  def setUp(self):
    super().setUp()
    self.probe.healthy = False
    self.monitor.poll()
    self.probe.healthy = True

  def advance_and_poll(self, times):
    for _ in range(times):
      self.clock.advance(health.IDLE_PROBE_SECONDS)
      self.monitor.poll()

  def test_one_good_probe_is_not_enough(self):
    self.clock.advance(protocol.QUARANTINE_MIN_SECONDS)
    self.advance_and_poll(1)
    self.assertTrue(self.monitor.quarantined)

  def test_enough_good_probes_bring_it_back(self):
    self.clock.advance(protocol.QUARANTINE_MIN_SECONDS)
    self.advance_and_poll(protocol.PROBES_TO_RECOVER)
    self.assertFalse(self.monitor.quarantined)
    self.assertEqual(self.monitor.reason, "")

  def test_a_long_hold_blocks_recovery_even_with_clean_probes(self):
    # A chip that has broken repeatedly is held for 120 * 2^3 seconds,
    # which outlasts the probes, so the hold is what keeps it out.
    monitor = health.HealthMonitor(self.probe, time_fn=self.clock.time)
    monitor.quarantine_count = 3
    self.probe.healthy = False
    monitor.poll()
    self.probe.healthy = True
    for _ in range(protocol.PROBES_TO_RECOVER + 2):
      self.clock.advance(health.IDLE_PROBE_SECONDS)
      monitor.poll()
    self.assertTrue(monitor.quarantined)

  def test_on_a_first_offence_the_probes_are_what_take_the_time(self):
    # Worth pinning: three probes 60s apart is 180s, which is already
    # longer than the 120s minimum hold, so shortening the probe
    # interval would make the hold start to bite.
    probe_time = protocol.PROBES_TO_RECOVER * health.IDLE_PROBE_SECONDS
    self.assertGreater(probe_time, protocol.QUARANTINE_MIN_SECONDS)

  def test_a_relapse_restarts_the_count(self):
    self.clock.advance(protocol.QUARANTINE_MIN_SECONDS)
    self.advance_and_poll(protocol.PROBES_TO_RECOVER - 1)
    self.probe.healthy = False
    self.advance_and_poll(1)
    self.probe.healthy = True
    self.advance_and_poll(protocol.PROBES_TO_RECOVER - 1)
    self.assertTrue(self.monitor.quarantined)

  def test_recovering_does_not_forget_the_history(self):
    self.clock.advance(protocol.QUARANTINE_MIN_SECONDS)
    self.advance_and_poll(protocol.PROBES_TO_RECOVER)
    self.assertEqual(self.monitor.quarantine_count, 1)


class IdlePollTest(HealthMonitorTestCase):

  def test_it_probes_the_first_time_it_is_asked(self):
    self.monitor.poll()
    self.assertEqual(self.probe.calls, 1)

  def test_it_does_not_probe_again_straight_away(self):
    self.monitor.poll()
    self.clock.advance(health.IDLE_PROBE_SECONDS - 1)
    self.monitor.poll()
    self.assertEqual(self.probe.calls, 1)

  def test_it_probes_again_once_the_interval_has_passed(self):
    self.monitor.poll()
    self.clock.advance(health.IDLE_PROBE_SECONDS)
    self.monitor.poll()
    self.assertEqual(self.probe.calls, 2)

  def test_a_skipped_poll_still_reports_the_current_state(self):
    self.probe.healthy = False
    self.monitor.poll()
    self.clock.advance(1)
    self.assertFalse(self.monitor.poll())


if __name__ == "__main__":
  unittest.main()
