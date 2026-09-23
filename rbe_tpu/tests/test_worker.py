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

"""Unit tests for the RBE-side relay worker."""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import unittest

import fake_gcs
from rbe_tpu.common import binding as binding_lib
from rbe_tpu.common import protocol
from rbe_tpu.worker import worker as worker_lib


class Responder:
  """Answers work items the way a real agent would.

  It is driven from the worker's sleep function, which is what lets a
  test decide exactly when the reply lands. It also heartbeats, because
  a real agent does and the fleet would otherwise go stale as the test
  advances the clock.
  """

  def __init__(self, client, clock, tpus):
    self.client = client
    self.clock = clock
    self.tpus = list(tpus)
    self.exit_code = 0
    self.infra_reason = ""
    self.answer_after = 1
    self.silent_tpus = set()
    self.quarantined_tpus = set()
    self.ticks = 0
    self.answered = []

  def __call__(self, seconds):
    self.clock.advance(seconds)
    self.ticks += 1
    self.heartbeat()
    if self.ticks >= self.answer_after:
      self.answer()

  def heartbeat(self):
    for tpu in self.tpus:
      quarantined = tpu in self.quarantined_tpus
      entry = protocol.FleetEntry(
          tpu=tpu,
          agent_id=f"agent-{tpu}",
          updated_at=self.clock.time(),
          state=(
              protocol.STATE_QUARANTINED
              if quarantined
              else protocol.STATE_HEALTHY
          ),
          reason="chip is dead" if quarantined else "",
      )
      self.client.put(protocol.fleet_path(tpu), entry.encode())

  def quarantine(self, tpu):
    """Takes a chip out the way an agent would: revoke, then publish."""
    self.quarantined_tpus.add(tpu)
    self.silent_tpus.add(tpu)
    binding_lib.revoke(self.client, tpu)
    self.heartbeat()

  def answer(self):
    for tpu in self.tpus:
      if tpu in self.silent_tpus:
        continue
      for name in self.client.list(protocol.work_prefix(tpu)):
        blob = self.client.get(name)
        if blob is None:
          continue
        item = protocol.WorkItem.decode(blob.json())
        self._reply(tpu, item)

  def _reply(self, tpu, item):
    result = protocol.Result(
        item_id=item.item_id,
        exit_code=self.exit_code,
        started_at=self.clock.time(),
        finished_at=self.clock.time(),
        infra_failure=bool(self.infra_reason),
        infra_reason=self.infra_reason,
    )
    self.client.put(protocol.result_path(tpu, item.item_id), result.encode())
    self.client.delete(protocol.work_path(tpu, item.item_id))
    self.answered.append(item)


class WorkerTestCase(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def setUp(self):
    super().setUp()
    self.client = fake_gcs.FakeClient()
    self.clock = fake_gcs.FakeClock()
    self.logs = []

  def register(self, *tpus, state=protocol.STATE_HEALTHY):
    registry = binding_lib.FleetRegistry(self.client, time_fn=self.clock.time)
    for tpu in tpus:
      registry.publish(
          protocol.FleetEntry(
              tpu=tpu,
              agent_id=f"agent-{tpu}",
              updated_at=self.clock.time(),
              state=state,
              reason="chip is dead" if state != protocol.STATE_HEALTHY else "",
          )
      )

  def make_worker(self, responder, worker_id="worker-1"):
    return worker_lib.Worker(
        self.client,
        worker_id,
        deps=worker_lib.WorkerDeps(
            time_fn=self.clock.time,
            sleep_fn=responder,
            log_fn=self.logs.append,
        ),
    )

  def responder(self, *tpus):
    return Responder(self.client, self.clock, tpus)


class DispatchTest(WorkerTestCase):

  def setUp(self):
    super().setUp()
    self.register("tpu-a")
    self.responder_obj = self.responder("tpu-a")
    self.worker = self.make_worker(self.responder_obj)
    self.request = worker_lib.Request(argv=["run-the-test"])

  def test_it_runs_the_command_and_brings_back_the_status(self):
    self.responder_obj.exit_code = 3
    outcome = self.worker.dispatch(self.request)
    self.assertTrue(outcome.ok)
    self.assertEqual(outcome.result.exit_code, 3)
    self.assertEqual(outcome.tpu, "tpu-a")

  def test_it_binds_on_the_way_in(self):
    self.worker.dispatch(self.request)
    self.assertEqual(self.worker.binder.tpu, "tpu-a")

  def test_it_keeps_the_chip_after_a_good_run(self):
    """Boot-time binding only pays off if success does not release."""
    self.worker.dispatch(self.request)
    generation = self.worker.binder.generation
    self.worker.dispatch(self.request)
    self.assertEqual(self.worker.binder.generation, generation)

  def test_the_item_carries_the_generation_it_was_written_under(self):
    """This is what lets the agent reject a deposed worker's items."""
    self.worker.dispatch(self.request)
    item = self.responder_obj.answered[0]
    live = self.client.get(protocol.binding_path("tpu-a"))
    self.assertEqual(item.binding_generation, live.generation)
    self.assertEqual(item.binding_generation, self.worker.binder.generation)

  def test_it_passes_the_command_through(self):
    request = worker_lib.Request(
        argv=["a", "b"], workdir="/w", env={"K": "V"}, layers=["blobs/x"]
    )
    self.worker.dispatch(request)
    item = self.responder_obj.answered[0]
    self.assertEqual(item.argv, ["a", "b"])
    self.assertEqual(item.workdir, "/w")
    self.assertEqual(item.env, {"K": "V"})
    self.assertEqual(item.layers, ["blobs/x"])

  def test_it_fails_cleanly_when_there_is_no_chip_at_all(self):
    self.client.delete(protocol.fleet_path("tpu-a"))
    self.responder_obj.tpus = []
    outcome = self.worker.dispatch(self.request)
    self.assertFalse(outcome.ok)
    self.assertIn("no healthy chip", outcome.failure)

  def test_it_will_not_use_a_quarantined_chip(self):
    self.responder_obj.quarantine("tpu-a")
    outcome = self.worker.dispatch(self.request)
    self.assertFalse(outcome.ok)
    self.assertIn("no healthy chip", outcome.failure)


class MovesToAnotherChipTest(WorkerTestCase):
  """What the health signal is for, seen from the worker's side."""

  def setUp(self):
    super().setUp()
    self.register("tpu-a", "tpu-b")
    self.responder_obj = self.responder("tpu-a", "tpu-b")
    self.worker = self.make_worker(self.responder_obj)
    self.request = worker_lib.Request(argv=["run-the-test"])

  def pin_to(self, tpu):
    """Forces the first binding, so the shuffle cannot decide the test."""
    other = "tpu-b" if tpu == "tpu-a" else "tpu-a"
    self.client.delete(protocol.fleet_path(other))
    self.assertEqual(self.worker.binder.bind(), tpu)
    self.register(other)

  def test_an_infrastructure_verdict_sends_the_action_elsewhere(self):
    self.responder_obj.infra_reason = "chip is quarantined"
    self.worker.dispatch(self.request)
    used = {item.tpu for item in self.responder_obj.answered}
    self.assertEqual(used, {"tpu-a", "tpu-b"})

  def test_it_reports_a_real_result_once_a_good_chip_takes_it(self):
    self.pin_to("tpu-a")
    self.responder_obj.infra_reason = "chip is quarantined"
    original = self.responder_obj._reply

    def reply(tpu, item):
      if tpu == "tpu-b":
        self.responder_obj.infra_reason = ""
      original(tpu, item)

    self.responder_obj._reply = reply
    outcome = self.worker.dispatch(self.request)
    self.assertTrue(outcome.ok)
    self.assertFalse(outcome.result.infra_failure)
    self.assertEqual(outcome.tpu, "tpu-b")

  def test_it_gives_up_after_a_bounded_number_of_chips(self):
    self.responder_obj.infra_reason = "chip is quarantined"
    outcome = self.worker.dispatch(self.request)
    self.assertFalse(outcome.ok)
    self.assertLessEqual(
        len(self.responder_obj.answered), protocol.MAX_DISPATCH_ATTEMPTS
    )

  def test_a_revoked_binding_mid_wait_moves_the_action(self):
    self.pin_to("tpu-a")
    self.responder_obj.silent_tpus = {"tpu-a"}
    original_sleep = self.responder_obj.__call__

    def sleep_fn(seconds):
      original_sleep(seconds)
      if self.responder_obj.ticks == 3:
        self.responder_obj.quarantine("tpu-a")

    self.worker._sleep_fn = sleep_fn  # noqa: SLF001 - drives the fake clock
    outcome = self.worker.dispatch(self.request)
    self.assertTrue(outcome.ok)
    self.assertEqual(outcome.tpu, "tpu-b")

  def test_it_notices_the_revocation_without_waiting_out_the_timeout(self):
    self.pin_to("tpu-a")
    self.responder_obj.silent_tpus = {"tpu-a"}
    original_sleep = self.responder_obj.__call__

    def sleep_fn(seconds):
      original_sleep(seconds)
      if self.responder_obj.ticks == 3:
        self.responder_obj.quarantine("tpu-a")

    self.worker._sleep_fn = sleep_fn  # noqa: SLF001 - drives the fake clock
    started = self.clock.time()
    request = worker_lib.Request(argv=["x"], timeout_seconds=900.0)
    self.worker.dispatch(request)
    elapsed = self.clock.time() - started
    self.assertLess(elapsed, worker_lib.BINDING_CHECK_SECONDS * 3)

  def test_it_withdraws_the_item_it_gave_up_on(self):
    self.responder_obj.silent_tpus = {"tpu-a", "tpu-b"}
    self.worker.dispatch(worker_lib.Request(argv=["x"], timeout_seconds=5.0))
    for tpu in ("tpu-a", "tpu-b"):
      self.assertEqual(self.client.list(protocol.work_prefix(tpu)), [])

  def test_a_silent_agent_does_not_hang_the_action_forever(self):
    self.responder_obj.silent_tpus = {"tpu-a", "tpu-b"}
    outcome = self.worker.dispatch(
        worker_lib.Request(argv=["x"], timeout_seconds=5.0)
    )
    self.assertFalse(outcome.ok)
    self.assertIn("timed out", outcome.failure)

  def test_the_failure_names_every_chip_it_tried(self):
    self.responder_obj.silent_tpus = {"tpu-a", "tpu-b"}
    outcome = self.worker.dispatch(
        worker_lib.Request(argv=["x"], timeout_seconds=5.0)
    )
    self.assertIn("tpu-a", outcome.failure)
    self.assertIn("tpu-b", outcome.failure)

  def test_it_does_not_wait_around_once_every_chip_is_spent(self):
    """With both chips tried, the third attempt must not stall."""
    self.responder_obj.silent_tpus = {"tpu-a", "tpu-b"}
    started = self.clock.time()
    self.worker.dispatch(worker_lib.Request(argv=["x"], timeout_seconds=5.0))
    elapsed = self.clock.time() - started
    self.assertLess(elapsed, protocol.BIND_WAIT_TIMEOUT_SECONDS)


class PollDelayTest(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def test_it_polls_fast_while_a_short_test_could_still_finish(self):
    self.assertEqual(
        worker_lib._poll_delay(0.0), protocol.WORKER_POLL_FAST_SECONDS
    )

  def test_it_backs_off_for_the_long_tail(self):
    elapsed = protocol.WORKER_POLL_FAST_WINDOW_SECONDS + 1
    self.assertEqual(
        worker_lib._poll_delay(elapsed), protocol.WORKER_POLL_SLOW_SECONDS
    )


class UploadLayersTest(WorkerTestCase):

  def setUp(self):
    super().setUp()
    self.worker = self.make_worker(self.clock.sleep)
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)

  def make_tree(self, name, contents):
    root = os.path.join(self.temp.name, name)
    os.makedirs(root, exist_ok=True)
    for filename, data in contents.items():
      path = os.path.join(root, filename)
      os.makedirs(os.path.dirname(path), exist_ok=True)
      with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    return root

  def test_it_names_a_layer_after_its_contents(self):
    root = self.make_tree("a", {"f.txt": "hello"})
    names = self.worker.upload_layers([root])
    self.assertEqual(len(names), 1)
    self.assertTrue(names[0].startswith("blobs/"))

  def test_the_same_tree_always_gets_the_same_name(self):
    first = self.make_tree("a", {"f.txt": "hello"})
    second = self.make_tree("b", {"f.txt": "hello"})
    self.assertEqual(
        self.worker.upload_layers([first]),
        self.worker.upload_layers([second]),
    )

  def test_a_changed_tree_gets_a_different_name(self):
    first = self.make_tree("a", {"f.txt": "hello"})
    second = self.make_tree("b", {"f.txt": "goodbye"})
    self.assertNotEqual(
        self.worker.upload_layers([first]),
        self.worker.upload_layers([second]),
    )

  def test_it_uploads_an_unchanged_layer_only_once(self):
    """The big shared layer is the reason this is content-addressed."""
    root = self.make_tree("a", {"f.txt": "hello"})
    self.worker.upload_layers([root])
    uploads = self.client.count("put", "blobs/")
    self.worker.upload_layers([root])
    self.assertEqual(self.client.count("put", "blobs/"), uploads)

  def test_it_keeps_the_layers_in_the_order_it_was_given(self):
    first = self.make_tree("a", {"f.txt": "one"})
    second = self.make_tree("b", {"f.txt": "two"})
    names = self.worker.upload_layers([first, second])
    self.assertEqual(names, self.worker.upload_layers([first, second]))
    self.assertNotEqual(names[0], names[1])

  def test_nested_files_make_it_in(self):
    root = self.make_tree("a", {"sub/deep.txt": "x"})
    names = self.worker.upload_layers([root])
    blob = self.client.get(names[0])
    self.assertIn(b"sub/deep.txt", blob.data)


if __name__ == "__main__":
  unittest.main()
