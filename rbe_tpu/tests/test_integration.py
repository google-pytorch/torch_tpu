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

"""Real agents against real workers, through one in-memory bucket.

The unit tests put a fake on one side of every exchange, so they cannot
catch the two halves disagreeing about the protocol. Here both sides are
the production classes and only storage is faked.

Agents are stepped from the worker's sleep function, which keeps the
whole thing deterministic: no threads, no wall clock, and the test
decides exactly when each side gets to act.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

import fake_gcs
from fake_gcs import StubProbe
from rbe_tpu.agent import agent as agent_lib
from rbe_tpu.common import health as health_lib
from rbe_tpu.common import protocol
from rbe_tpu.worker import worker as worker_lib


class Chip:
  """One TPU: a probe we can break and a runner we can steer."""

  def __init__(self, name):
    self.name = name
    self.probe = StubProbe()
    self.exit_code = 0
    self.output = ""
    self.ran = []

  def run(self, item) -> agent_lib.Execution:
    self.ran.append(item.item_id)
    return agent_lib.Execution(
        exit_code=self.exit_code,
        output=self.output,
        output_files={"stdout": self.output.encode("utf-8")},
    )

  def break_chip(self, output="Failed to initialize TPU"):
    """Makes the chip fail the way a dead one does."""
    self.probe.healthy = False
    self.exit_code = 1
    self.output = output

  def fix_chip(self):
    self.probe.healthy = True
    self.exit_code = 0
    self.output = ""


class Fleet:
  """A bucket, a clock, and some number of agents that get stepped."""

  def __init__(self, names):
    self.client = fake_gcs.FakeClient()
    self.clock = fake_gcs.FakeClock()
    self.chips = {name: Chip(name) for name in names}
    self.agents = {}
    for name, chip in self.chips.items():
      monitor = health_lib.HealthMonitor(chip.probe, time_fn=self.clock.time)
      self.agents[name] = agent_lib.Agent(
          self.client,
          name,
          chip.run,
          deps=agent_lib.AgentDeps(time_fn=self.clock.time, monitor=monitor),
      )

  def start(self):
    for agent in self.agents.values():
      agent.start()

  def tick(self, seconds):
    """Advances time, then lets every agent take a turn."""
    self.clock.advance(seconds)
    for agent in self.agents.values():
      agent.poll_once()

  def make_worker(self, worker_id):
    return worker_lib.Worker(
        self.client,
        worker_id,
        deps=worker_lib.WorkerDeps(time_fn=self.clock.time, sleep_fn=self.tick),
    )


class HappyPathTest(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def setUp(self):
    super().setUp()
    self.fleet = Fleet(["tpu-a"])
    self.fleet.start()
    self.worker = self.fleet.make_worker("worker-1")

  def test_a_passing_test_comes_back_passing(self):
    outcome = self.worker.dispatch(worker_lib.Request(argv=["run"]))
    self.assertTrue(outcome.ok)
    self.assertEqual(outcome.result.exit_code, 0)
    self.assertFalse(outcome.result.infra_failure)

  def test_the_command_actually_reaches_the_chip(self):
    self.worker.dispatch(worker_lib.Request(argv=["run"]))
    self.assertEqual(len(self.fleet.chips["tpu-a"].ran), 1)

  def test_a_failing_test_is_reported_as_a_failing_test(self):
    self.fleet.chips["tpu-a"].exit_code = 4
    self.fleet.chips["tpu-a"].output = "AssertionError: 1 != 2"
    outcome = self.worker.dispatch(worker_lib.Request(argv=["run"]))
    self.assertTrue(outcome.ok)
    self.assertEqual(outcome.result.exit_code, 4)
    self.assertFalse(outcome.result.infra_failure)

  def test_outputs_make_it_back(self):
    self.fleet.chips["tpu-a"].output = "the test said this"
    outcome = self.worker.dispatch(worker_lib.Request(argv=["run"]))
    self.assertEqual(outcome.result.outputs, ["stdout"])
    blob = self.fleet.client.get(
        protocol.output_path("tpu-a", outcome.result.item_id, "stdout")
    )
    self.assertEqual(blob.data, b"the test said this")

  def test_the_mailbox_is_empty_afterwards(self):
    self.worker.dispatch(worker_lib.Request(argv=["run"]))
    self.assertEqual(self.fleet.client.list(protocol.work_prefix("tpu-a")), [])
    self.assertEqual(
        self.fleet.client.list(protocol.running_prefix("tpu-a")), []
    )

  def test_the_worker_keeps_its_chip_across_actions(self):
    self.worker.dispatch(worker_lib.Request(argv=["one"]))
    first = self.worker.binder.tpu
    self.worker.dispatch(worker_lib.Request(argv=["two"]))
    self.assertEqual(self.worker.binder.tpu, first)
    self.assertEqual(len(self.fleet.chips["tpu-a"].ran), 2)


class ManyWorkersTest(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def setUp(self):
    super().setUp()
    self.fleet = Fleet([f"tpu-{i:02d}" for i in range(4)])
    self.fleet.start()

  def test_each_worker_ends_up_on_its_own_chip(self):
    workers = [self.fleet.make_worker(f"worker-{i}") for i in range(4)]
    for worker in workers:
      worker.dispatch(worker_lib.Request(argv=["run"]))
    chips = [w.binder.tpu for w in workers]
    self.assertCountEqual(chips, list(self.fleet.chips))

  def test_no_chip_runs_two_actions_at_once(self):
    workers = [self.fleet.make_worker(f"worker-{i}") for i in range(4)]
    for worker in workers:
      worker.dispatch(worker_lib.Request(argv=["run"]))
    for chip in self.fleet.chips.values():
      self.assertEqual(len(chip.ran), 1)

  def test_a_fifth_worker_finds_nothing_free(self):
    workers = [self.fleet.make_worker(f"worker-{i}") for i in range(4)]
    for worker in workers:
      worker.dispatch(worker_lib.Request(argv=["run"]))
    spare = self.fleet.make_worker("worker-spare")
    self.assertEqual(spare.binder.bind(), "")

  def test_a_spare_worker_waits_rather_than_failing_the_action(self):
    workers = [self.fleet.make_worker(f"worker-{i}") for i in range(4)]
    for worker in workers:
      worker.dispatch(worker_lib.Request(argv=["run"]))
    spare = worker_lib.Worker(
        self.fleet.client,
        "worker-spare",
        deps=worker_lib.WorkerDeps(
            time_fn=self.fleet.clock.time,
            sleep_fn=self.fleet.tick,
            bind_wait_seconds=10.0,
        ),
    )
    started = self.fleet.clock.time()
    outcome = spare.dispatch(
        worker_lib.Request(argv=["run"], timeout_seconds=5.0)
    )
    self.assertFalse(outcome.ok)
    self.assertGreaterEqual(self.fleet.clock.time() - started, 10.0)

  def test_an_idle_worker_lets_its_chip_go_back_to_the_pool(self):
    """Claims lapse without use, so nobody sits on a chip they forgot."""
    fleet = Fleet(["tpu-only"])
    fleet.start()
    holder = fleet.make_worker("worker-holder")
    holder.dispatch(worker_lib.Request(argv=["run"]))
    spare = fleet.make_worker("worker-spare")
    self.assertEqual(spare.binder.bind(), "")
    fleet.clock.advance(protocol.BINDING_TTL_SECONDS + 1)
    fleet.tick(0)
    self.assertEqual(spare.binder.bind(), "tpu-only")

  def test_a_busy_worker_keeps_its_chip_past_the_ttl(self):
    """The keep-alive has to actually fire, or a run loses its chip."""
    fleet = Fleet(["tpu-only"])
    fleet.start()
    holder = fleet.make_worker("worker-holder")
    deadline = fleet.clock.time() + protocol.BINDING_TTL_SECONDS * 2
    while fleet.clock.time() < deadline:
      self.assertTrue(holder.dispatch(worker_lib.Request(argv=["run"])).ok)
      fleet.tick(30.0)
    self.assertEqual(holder.binder.tpu, "tpu-only")


class BadChipTest(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):
  """The scenario the health signal exists for, end to end."""

  def setUp(self):
    super().setUp()
    self.fleet = Fleet(["tpu-a", "tpu-b"])
    self.fleet.start()
    self.worker = self.fleet.make_worker("worker-1")
    # Pin the worker to tpu-a so the shuffle cannot decide the test.
    self.fleet.client.delete(protocol.fleet_path("tpu-b"))
    self.assertEqual(self.worker.binder.bind(), "tpu-a")
    self.fleet.agents["tpu-b"].start()

  def run_until_moved(self, request=None):
    request = request or worker_lib.Request(argv=["run"])
    return self.worker.dispatch(request)

  def test_a_chip_that_dies_does_not_turn_into_a_red_test(self):
    self.fleet.chips["tpu-a"].break_chip()
    outcome = self.run_until_moved()
    for _ in range(protocol.INFRA_FAILURES_BEFORE_PROBE):
      outcome = self.run_until_moved()
    self.assertTrue(outcome.ok)
    self.assertEqual(outcome.result.exit_code, 0)
    self.assertEqual(outcome.tpu, "tpu-b")

  def test_the_dead_chip_takes_itself_out_of_the_fleet(self):
    self.fleet.chips["tpu-a"].break_chip()
    for _ in range(protocol.INFRA_FAILURES_BEFORE_PROBE + 1):
      self.run_until_moved()
    entry = self.fleet.client.get(protocol.fleet_path("tpu-a"))
    self.assertEqual(
        protocol.FleetEntry.decode(entry.json()).state,
        protocol.STATE_QUARANTINED,
    )

  def test_it_hands_its_binding_back(self):
    self.fleet.chips["tpu-a"].break_chip()
    for _ in range(protocol.INFRA_FAILURES_BEFORE_PROBE + 1):
      self.run_until_moved()
    self.assertIsNone(self.fleet.client.get(protocol.binding_path("tpu-a")))

  def test_nothing_else_is_ever_sent_to_it(self):
    self.fleet.chips["tpu-a"].break_chip()
    for _ in range(protocol.INFRA_FAILURES_BEFORE_PROBE + 1):
      self.run_until_moved()
    sent_before = len(self.fleet.chips["tpu-a"].ran)
    for _ in range(5):
      self.run_until_moved()
    self.assertEqual(len(self.fleet.chips["tpu-a"].ran), sent_before)

  def test_the_other_chip_picks_up_the_slack(self):
    self.fleet.chips["tpu-a"].break_chip()
    for _ in range(protocol.INFRA_FAILURES_BEFORE_PROBE + 3):
      self.run_until_moved()
    self.assertGreater(len(self.fleet.chips["tpu-b"].ran), 0)

  def test_a_chip_broken_before_anyone_binds_is_never_used(self):
    fleet = Fleet(["tpu-x", "tpu-y"])
    fleet.chips["tpu-x"].break_chip()
    fleet.start()
    worker = fleet.make_worker("worker-1")
    outcome = worker.dispatch(worker_lib.Request(argv=["run"]))
    self.assertTrue(outcome.ok)
    self.assertEqual(outcome.tpu, "tpu-y")
    self.assertEqual(fleet.chips["tpu-x"].ran, [])


class ChipComesBackTest(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def test_a_repaired_chip_rejoins_and_takes_work_again(self):
    fleet = Fleet(["tpu-a"])
    fleet.chips["tpu-a"].break_chip()
    fleet.start()
    worker = fleet.make_worker("worker-1")
    self.assertFalse(
        worker.dispatch(
            worker_lib.Request(argv=["run"], timeout_seconds=5.0)
        ).ok
    )

    fleet.chips["tpu-a"].fix_chip()
    fleet.clock.advance(protocol.QUARANTINE_MIN_SECONDS)
    for _ in range(protocol.PROBES_TO_RECOVER):
      fleet.tick(health_lib.IDLE_PROBE_SECONDS)

    outcome = worker.dispatch(worker_lib.Request(argv=["run"]))
    self.assertTrue(outcome.ok)
    self.assertEqual(outcome.tpu, "tpu-a")


if __name__ == "__main__":
  unittest.main()
