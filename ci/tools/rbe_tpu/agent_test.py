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

"""Unit tests for the TPU-side relay agent."""

from __future__ import annotations

import pathlib
import sys
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from ci.tools.rbe_tpu import agent as agent_lib
from ci.tools.rbe_tpu import binding as binding_lib
from ci.tools.rbe_tpu import fake_gcs
from ci.tools.rbe_tpu import health as health_lib
from ci.tools.rbe_tpu import protocol
from ci.tools.rbe_tpu.fake_gcs import StubProbe

TPU = "tpu-a"


class StubRunner:
  """Stands in for actually running a test on the chip."""

  def __init__(self, exit_code=0, output="", files=None):
    self.exit_code = exit_code
    self.output = output
    self.files = files or {}
    self.error = None
    self.calls = []
    self.on_run = None

  def __call__(self, item) -> agent_lib.Execution:
    self.calls.append(item)
    if self.on_run is not None:
      self.on_run()
    if self.error is not None:
      raise self.error
    return agent_lib.Execution(
        exit_code=self.exit_code,
        output=self.output,
        output_files=dict(self.files),
    )


class AgentTestCase(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):
  """A bucket, a clock, a controllable probe and a stub runner."""

  def setUp(self):
    super().setUp()
    self.client = fake_gcs.FakeClient()
    self.clock = fake_gcs.FakeClock()
    self.probe = StubProbe()
    self.monitor = health_lib.HealthMonitor(
        self.probe, time_fn=self.clock.time
    )
    self.runner = StubRunner()
    self.agent = agent_lib.Agent(
        self.client,
        TPU,
        self.runner,
        deps=agent_lib.AgentDeps(
            time_fn=self.clock.time, monitor=self.monitor
        ),
    )
    self.registry = binding_lib.FleetRegistry(
        self.client, time_fn=self.clock.time
    )

  def bind_a_worker(self, worker_id="worker-1") -> binding_lib.Binder:
    """Registers the TPU and gives a worker a live claim on it."""
    self.registry.publish(
        protocol.FleetEntry(
            tpu=TPU, agent_id="agent", updated_at=self.clock.time()
        )
    )
    binder = binding_lib.Binder(
        self.client, worker_id, time_fn=self.clock.time
    )
    binder.bind()
    return binder

  def post(self, binder, *, item_id="item-1", generation=None):
    """Writes a work item the way a worker would."""
    item = protocol.WorkItem(
        item_id=item_id,
        tpu=TPU,
        binding_generation=(
            binder.generation if generation is None else generation
        ),
        argv=["run-the-test"],
        workdir="/tmp/work",
        timeout_seconds=900.0,
        submitted_at=self.clock.time(),
    )
    self.client.put(protocol.work_path(TPU, item_id), item.encode())
    return item

  def result_for(self, item_id="item-1") -> protocol.Result | None:
    blob = self.client.get(protocol.result_path(TPU, item_id))
    return protocol.Result.decode(blob.json()) if blob else None

  def fleet_entry(self) -> protocol.FleetEntry:
    return self.registry.read(TPU)


class StartTest(AgentTestCase):

  def test_a_healthy_chip_registers_and_reports_in(self):
    self.assertTrue(self.agent.start())
    entry = self.fleet_entry()
    self.assertEqual(entry.state, protocol.STATE_HEALTHY)
    self.assertEqual(entry.tpu, TPU)

  def test_a_broken_chip_registers_as_quarantined(self):
    self.probe.healthy = False
    self.assertFalse(self.agent.start())
    entry = self.fleet_entry()
    self.assertEqual(entry.state, protocol.STATE_QUARANTINED)
    self.assertEqual(entry.detector, "stub")

  def test_a_broken_chip_hands_back_any_binding_it_had(self):
    binder = self.bind_a_worker()
    self.probe.healthy = False
    self.agent.start()
    self.assertIsNone(self.client.get(protocol.binding_path(TPU)))
    self.assertFalse(binder.check().ok)

  def test_it_clears_an_attempt_marker_with_no_work_behind_it(self):
    self.client.put(protocol.running_path(TPU, "ghost"), b'{"attempts": 1}')
    self.agent.start()
    self.assertIsNone(self.client.get(protocol.running_path(TPU, "ghost")))

  def test_it_keeps_a_marker_whose_work_item_is_still_queued(self):
    binder = self.bind_a_worker()
    self.post(binder, item_id="live")
    self.client.put(protocol.running_path(TPU, "live"), b'{"attempts": 1}')
    self.agent.start()
    self.assertIsNotNone(self.client.get(protocol.running_path(TPU, "live")))


class ProcessItemTest(AgentTestCase):

  def setUp(self):
    super().setUp()
    self.agent.start()
    self.binder = self.bind_a_worker()

  def test_it_runs_a_queued_item(self):
    item = self.post(self.binder)
    self.assertTrue(self.agent.poll_once())
    self.assertEqual([c.item_id for c in self.runner.calls], [item.item_id])

  def test_it_reports_the_exit_status(self):
    self.runner.exit_code = 7
    self.post(self.binder)
    self.agent.poll_once()
    self.assertEqual(self.result_for().exit_code, 7)

  def test_it_clears_the_mailbox_when_it_is_done(self):
    self.post(self.binder)
    self.agent.poll_once()
    self.assertIsNone(self.client.get(protocol.work_path(TPU, "item-1")))
    self.assertIsNone(self.client.get(protocol.running_path(TPU, "item-1")))

  def test_it_uploads_outputs_before_the_result(self):
    """The result is the worker's signal that everything else landed."""
    self.runner.files = {"stdout": b"hello", "stderr": b""}
    self.post(self.binder)
    self.agent.poll_once()
    puts = [name for verb, name in self.client.calls if verb == "put"]
    result_at = puts.index(protocol.result_path(TPU, "item-1"))
    stdout_at = puts.index(protocol.output_path(TPU, "item-1", "stdout"))
    self.assertLess(stdout_at, result_at)

  def test_it_lists_what_it_uploaded(self):
    self.runner.files = {"stdout": b"hello", "stderr": b"oops"}
    self.post(self.binder)
    self.agent.poll_once()
    self.assertEqual(self.result_for().outputs, ["stderr", "stdout"])

  def test_it_takes_the_oldest_item_first(self):
    self.post(self.binder, item_id="0000000002-b")
    self.post(self.binder, item_id="0000000001-a")
    self.agent.poll_once()
    self.assertEqual(self.runner.calls[0].item_id, "0000000001-a")

  def test_it_does_nothing_when_the_mailbox_is_empty(self):
    self.assertFalse(self.agent.poll_once())
    self.assertEqual(self.runner.calls, [])

  def test_a_runner_that_blows_up_is_reported_not_swallowed(self):
    self.runner.error = RuntimeError("tar is missing")
    self.post(self.binder)
    self.agent.poll_once()
    result = self.result_for()
    self.assertEqual(result.exit_code, 1)
    self.assertTrue(result.infra_failure)
    self.assertIn("tar is missing", result.infra_reason)

  def test_our_own_plumbing_breaking_never_costs_us_the_chip(self):
    """A bucket outage hits all 28 agents; it must not empty the fleet."""
    self.runner.error = RuntimeError("blob fetch failed")
    probes_before = self.probe.calls
    for index in range(10):
      self.post(self.binder, item_id=f"item-{index}")
      self.agent.poll_once()
    self.assertFalse(self.agent.quarantined)
    self.assertEqual(self.probe.calls, probes_before)
    self.assertIsNotNone(self.client.get(protocol.binding_path(TPU)))


class StaleItemTest(AgentTestCase):

  def setUp(self):
    super().setUp()
    self.agent.start()
    self.binder = self.bind_a_worker()

  def test_it_drops_an_item_from_a_worker_that_lost_the_chip(self):
    self.post(self.binder, generation=self.binder.generation + 999)
    self.assertFalse(self.agent.poll_once())
    self.assertEqual(self.runner.calls, [])
    self.assertIsNone(self.client.get(protocol.work_path(TPU, "item-1")))

  def test_it_drops_an_item_when_nobody_holds_the_chip(self):
    self.post(self.binder)
    binding_lib.revoke(self.client, TPU)
    self.assertFalse(self.agent.poll_once())
    self.assertEqual(self.runner.calls, [])

  def test_it_still_serves_the_current_holder_after_a_handover(self):
    self.post(self.binder, item_id="old")
    binding_lib.revoke(self.client, TPU)
    successor = self.bind_a_worker("worker-2")
    self.post(successor, item_id="new")
    self.agent.poll_once()
    self.assertEqual([c.item_id for c in self.runner.calls], ["new"])


class CrashRecoveryTest(AgentTestCase):

  def setUp(self):
    super().setUp()
    self.agent.start()
    self.binder = self.bind_a_worker()

  def test_it_does_not_rerun_an_item_that_already_has_a_result(self):
    self.post(self.binder)
    done = protocol.Result(
        item_id="item-1", exit_code=0, started_at=0.0, finished_at=1.0
    )
    self.client.put(protocol.result_path(TPU, "item-1"), done.encode())
    self.agent.poll_once()
    self.assertEqual(self.runner.calls, [])
    self.assertIsNone(self.client.get(protocol.work_path(TPU, "item-1")))

  def test_it_counts_attempts_across_restarts(self):
    self.post(self.binder)
    self.agent.poll_once()
    blob = self.client.get(protocol.running_path(TPU, "item-1"))
    self.assertIsNone(blob)

  def test_it_gives_up_on_an_item_that_keeps_killing_the_agent(self):
    self.post(self.binder)
    self.client.put(
        protocol.running_path(TPU, "item-1"),
        f'{{"attempts": {protocol.MAX_ITEM_ATTEMPTS}}}'.encode("utf-8"),
    )
    self.agent.poll_once()
    self.assertEqual(self.runner.calls, [])
    result = self.result_for()
    self.assertTrue(result.infra_failure)
    self.assertIn("gave up", result.infra_reason)

  def test_it_still_runs_an_item_below_the_attempt_cap(self):
    self.post(self.binder)
    self.client.put(
        protocol.running_path(TPU, "item-1"), b'{"attempts": 1}'
    )
    self.agent.poll_once()
    self.assertEqual(len(self.runner.calls), 1)


class ChipGoesBadTest(AgentTestCase):
  """The path the health signal exists for."""

  def setUp(self):
    super().setUp()
    self.agent.start()
    self.binder = self.bind_a_worker()
    self.runner.exit_code = 1
    self.runner.output = "Failed to initialize TPU: no chips visible"

  def run_items(self, count):
    for index in range(count):
      self.post(self.binder, item_id=f"item-{index}")
      self.agent.poll_once()

  def test_one_bad_run_does_not_pull_the_chip(self):
    self.run_items(1)
    self.assertFalse(self.agent.quarantined)
    self.assertIsNotNone(self.client.get(protocol.binding_path(TPU)))

  def test_a_confirmed_bad_chip_revokes_its_own_binding(self):
    self.probe.healthy = False
    self.run_items(protocol.INFRA_FAILURES_BEFORE_PROBE)
    self.assertTrue(self.agent.quarantined)
    self.assertIsNone(self.client.get(protocol.binding_path(TPU)))

  def test_the_worker_finds_out_on_its_very_next_check(self):
    self.probe.healthy = False
    self.run_items(protocol.INFRA_FAILURES_BEFORE_PROBE)
    check = self.binder.check()
    self.assertFalse(check.ok)
    self.assertEqual(check.reason, "binding revoked")

  def test_it_says_so_in_the_fleet_entry(self):
    self.probe.healthy = False
    self.probe.detail = "no device nodes"
    self.run_items(protocol.INFRA_FAILURES_BEFORE_PROBE)
    entry = self.fleet_entry()
    self.assertEqual(entry.state, protocol.STATE_QUARANTINED)
    self.assertEqual(entry.reason, "no device nodes")
    self.assertEqual(entry.detector, "stub")

  def test_the_failing_run_is_blamed_on_the_chip_not_the_test(self):
    self.probe.healthy = False
    self.run_items(protocol.INFRA_FAILURES_BEFORE_PROBE)
    result = self.result_for(f"item-{protocol.INFRA_FAILURES_BEFORE_PROBE - 1}")
    self.assertTrue(result.infra_failure)

  def test_a_quarantined_agent_stops_taking_work(self):
    self.probe.healthy = False
    self.run_items(protocol.INFRA_FAILURES_BEFORE_PROBE)
    before = len(self.runner.calls)
    successor = self.bind_a_worker("worker-2")
    self.post(successor, item_id="later")
    self.assertFalse(self.agent.poll_once())
    self.assertEqual(len(self.runner.calls), before)

  def test_a_healthy_probe_keeps_the_chip_in_service(self):
    """Suspicious output on a working chip must not cost us a chip."""
    self.probe.healthy = True
    self.run_items(protocol.INFRA_FAILURES_BEFORE_PROBE + 2)
    self.assertFalse(self.agent.quarantined)
    self.assertIsNotNone(self.client.get(protocol.binding_path(TPU)))
    self.assertFalse(self.result_for("item-0").infra_failure)

  def test_an_ordinary_failing_test_is_reported_as_a_failing_test(self):
    self.runner.output = "AssertionError: expected 3, got 4"
    self.run_items(5)
    self.assertFalse(self.agent.quarantined)
    result = self.result_for("item-0")
    self.assertEqual(result.exit_code, 1)
    self.assertFalse(result.infra_failure)


class RecoveryTest(AgentTestCase):

  def test_a_chip_that_comes_back_rejoins_the_fleet(self):
    self.probe.healthy = False
    self.agent.start()
    self.assertEqual(self.fleet_entry().state, protocol.STATE_QUARANTINED)
    self.probe.healthy = True
    self.clock.advance(protocol.QUARANTINE_MIN_SECONDS)
    for _ in range(protocol.PROBES_TO_RECOVER):
      self.clock.advance(health_lib.IDLE_PROBE_SECONDS)
      self.agent.poll_once()
    self.assertFalse(self.agent.quarantined)
    self.assertEqual(self.fleet_entry().state, protocol.STATE_HEALTHY)

  def test_it_does_not_take_its_old_binding_back(self):
    """Recovery makes the chip bindable again; it does not un-revoke."""
    self.bind_a_worker()
    self.probe.healthy = False
    self.agent.start()
    self.probe.healthy = True
    self.clock.advance(protocol.QUARANTINE_MIN_SECONDS)
    for _ in range(protocol.PROBES_TO_RECOVER):
      self.clock.advance(health_lib.IDLE_PROBE_SECONDS)
      self.agent.poll_once()
    self.assertIsNone(self.client.get(protocol.binding_path(TPU)))


class HeartbeatTest(AgentTestCase):

  def test_starting_publishes_straight_away(self):
    self.agent.start()
    self.assertIsNotNone(self.fleet_entry())

  def test_it_does_not_republish_on_every_pass(self):
    self.agent.start()
    before = self.client.count("put", protocol.FLEET_PREFIX)
    for _ in range(20):
      self.agent.poll_once()
    self.assertEqual(self.client.count("put", protocol.FLEET_PREFIX), before)

  def test_it_republishes_once_the_interval_passes(self):
    self.agent.start()
    before = self.client.count("put", protocol.FLEET_PREFIX)
    self.clock.advance(protocol.AGENT_HEARTBEAT_SECONDS)
    self.agent.poll_once()
    self.assertEqual(
        self.client.count("put", protocol.FLEET_PREFIX), before + 1
    )

  def test_the_heartbeat_is_well_inside_the_staleness_window(self):
    self.assertLess(
        protocol.AGENT_HEARTBEAT_SECONDS * 2, protocol.FLEET_STALE_SECONDS
    )


if __name__ == "__main__":
  unittest.main()
