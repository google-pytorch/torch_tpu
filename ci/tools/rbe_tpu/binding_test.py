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

"""Unit tests for the binding protocol."""

from __future__ import annotations

import pathlib
import sys
import threading
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from ci.tools.rbe_tpu import binding
from ci.tools.rbe_tpu import fake_gcs
from ci.tools.rbe_tpu import gcs
from ci.tools.rbe_tpu import protocol


class BindingTestCase(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):
  """Shared fixture: an empty bucket and a clock the test drives."""

  def setUp(self):
    super().setUp()
    self.client = fake_gcs.FakeClient()
    self.clock = fake_gcs.FakeClock()
    self.registry = binding.FleetRegistry(
        self.client, time_fn=self.clock.time
    )

  def register(self, tpu, *, state=protocol.STATE_HEALTHY, age=0.0):
    """Publishes a fleet entry as if an agent had heartbeat `age` ago."""
    self.registry.publish(
        protocol.FleetEntry(
            tpu=tpu,
            agent_id=f"agent-{tpu}",
            updated_at=self.clock.time() - age,
            state=state,
            reason="chip is unhappy" if state != protocol.STATE_HEALTHY else "",
        )
    )

  def make_binder(self, worker_id="worker-1"):
    return binding.Binder(self.client, worker_id, time_fn=self.clock.time)


class FleetRegistryTest(BindingTestCase):

  def test_it_lists_a_healthy_agent(self):
    self.register("tpu-a")
    self.assertEqual(self.registry.available(), ["tpu-a"])

  def test_it_hides_an_agent_that_stopped_heartbeating(self):
    self.register("tpu-a", age=protocol.FLEET_STALE_SECONDS + 1)
    self.assertEqual(self.registry.available(), [])

  def test_it_keeps_an_agent_that_is_merely_slow(self):
    self.register("tpu-a", age=protocol.FLEET_STALE_SECONDS - 1)
    self.assertEqual(self.registry.available(), ["tpu-a"])

  def test_it_hides_a_quarantined_agent(self):
    self.register("tpu-a", state=protocol.STATE_QUARANTINED)
    self.assertEqual(self.registry.available(), [])

  def test_it_reads_back_the_quarantine_reason(self):
    self.register("tpu-a", state=protocol.STATE_QUARANTINED)
    entry = self.registry.read("tpu-a")
    self.assertEqual(entry.state, protocol.STATE_QUARANTINED)
    self.assertEqual(entry.reason, "chip is unhappy")

  def test_it_returns_none_for_an_unknown_tpu(self):
    self.assertIsNone(self.registry.read("tpu-nope"))


class BindTest(BindingTestCase):

  def test_it_takes_a_free_chip(self):
    self.register("tpu-a")
    binder = self.make_binder()
    self.assertEqual(binder.bind(), "tpu-a")
    self.assertTrue(binder.bound)

  def test_it_records_the_generation_it_claimed_at(self):
    self.register("tpu-a")
    binder = self.make_binder()
    binder.bind()
    blob = self.client.get(protocol.binding_path("tpu-a"))
    self.assertEqual(binder.generation, blob.generation)

  def test_it_finds_nothing_when_the_fleet_is_empty(self):
    self.assertEqual(self.make_binder().bind(), "")

  def test_it_skips_a_chip_somebody_else_holds(self):
    self.register("tpu-a")
    self.register("tpu-b")
    first = self.make_binder("worker-1")
    second = self.make_binder("worker-2")
    first.bind()
    self.assertEqual(second.bind(), "tpu-b" if first.tpu == "tpu-a" else "tpu-a")

  def test_it_refuses_when_every_chip_is_held(self):
    self.register("tpu-a")
    self.make_binder("worker-1").bind()
    self.assertEqual(self.make_binder("worker-2").bind(), "")

  def test_it_steals_a_claim_that_has_expired(self):
    self.register("tpu-a")
    abandoned = self.make_binder("worker-gone")
    abandoned.bind()
    self.clock.advance(protocol.BINDING_TTL_SECONDS + 1)
    self.register("tpu-a")
    self.assertEqual(self.make_binder("worker-2").bind(), "tpu-a")

  def test_it_leaves_a_claim_that_is_still_inside_its_ttl(self):
    self.register("tpu-a")
    self.make_binder("worker-1").bind()
    self.clock.advance(protocol.BINDING_TTL_SECONDS - 1)
    self.register("tpu-a")
    self.assertEqual(self.make_binder("worker-2").bind(), "")

  def test_it_will_not_bind_to_a_quarantined_chip(self):
    self.register("tpu-a", state=protocol.STATE_QUARANTINED)
    self.assertEqual(self.make_binder().bind(), "")

  def test_it_will_not_bind_to_a_chip_whose_agent_is_dead(self):
    self.register("tpu-a", age=protocol.FLEET_STALE_SECONDS + 1)
    self.assertEqual(self.make_binder().bind(), "")

  def test_it_loses_the_race_gracefully(self):
    """A competing claim landing between our read and our write."""
    self.register("tpu-a")
    loser = self.make_binder("worker-slow")
    winner = self.make_binder("worker-fast")
    path = protocol.binding_path("tpu-a")
    self.client.before_put[path] = winner.bind
    self.assertEqual(loser.bind(), "")
    self.assertEqual(winner.tpu, "tpu-a")

  def test_it_loses_a_steal_race_gracefully(self):
    """Two workers spotting the same expired claim: one wins."""
    self.register("tpu-a")
    self.make_binder("worker-gone").bind()
    self.clock.advance(protocol.BINDING_TTL_SECONDS + 1)
    self.register("tpu-a")
    loser = self.make_binder("worker-slow")
    winner = self.make_binder("worker-fast")
    self.client.before_put[protocol.binding_path("tpu-a")] = winner.bind
    self.assertEqual(loser.bind(), "")
    self.assertEqual(winner.tpu, "tpu-a")

  def test_every_worker_gets_its_own_chip(self):
    """The property that matters: 28 workers, 28 chips, no doubling up."""
    fleet = [f"tpu-{i:02d}" for i in range(28)]
    for tpu in fleet:
      self.register(tpu)
    binders = [self.make_binder(f"worker-{i}") for i in range(28)]
    barrier = threading.Barrier(len(binders))

    def run(b):
      barrier.wait()
      b.bind()

    threads = [threading.Thread(target=run, args=(b,)) for b in binders]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

    taken = [b.tpu for b in binders if b.bound]
    self.assertCountEqual(taken, fleet)

  def test_surplus_workers_come_away_empty_rather_than_sharing(self):
    self.register("tpu-a")
    self.register("tpu-b")
    binders = [self.make_binder(f"worker-{i}") for i in range(6)]
    barrier = threading.Barrier(len(binders))

    def run(b):
      barrier.wait()
      b.bind()

    threads = [threading.Thread(target=run, args=(b,)) for b in binders]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

    taken = sorted(b.tpu for b in binders if b.bound)
    self.assertEqual(taken, ["tpu-a", "tpu-b"])


class WaitForBindingTest(BindingTestCase):

  def test_it_returns_as_soon_as_a_chip_frees_up(self):
    binder = self.make_binder()
    attempts = []

    def sleep_fn(seconds):
      attempts.append(seconds)
      self.clock.advance(seconds)
      if len(attempts) == 2:
        self.register("tpu-a")

    self.assertEqual(
        binder.wait_for_binding(600.0, sleep_fn=sleep_fn), "tpu-a"
    )

  def test_it_gives_up_at_the_deadline(self):
    binder = self.make_binder()
    self.assertEqual(
        binder.wait_for_binding(10.0, sleep_fn=self.clock.sleep), ""
    )

  def test_it_backs_off_instead_of_hammering(self):
    binder = self.make_binder()
    delays = []

    def sleep_fn(seconds):
      delays.append(seconds)
      self.clock.advance(seconds)

    binder.wait_for_binding(60.0, sleep_fn=sleep_fn)
    self.assertEqual(delays[:4], [1.0, 2.0, 4.0, 8.0])
    self.assertTrue(all(d <= 15.0 for d in delays))

  def test_it_gives_up_at_once_when_every_chip_is_spent(self):
    self.register("tpu-a")
    binder = self.make_binder()
    delays = []
    result = binder.wait_for_binding(
        600.0, sleep_fn=delays.append, exclude={"tpu-a"}
    )
    self.assertEqual(result, "")
    self.assertEqual(delays, [])

  def test_it_still_waits_when_no_agent_has_registered_yet(self):
    """Workers and agents boot together; an empty fleet is not final."""
    binder = self.make_binder()
    delays = []

    def sleep_fn(seconds):
      delays.append(seconds)
      self.clock.advance(seconds)
      if len(delays) == 2:
        self.register("tpu-a")

    self.assertEqual(
        binder.wait_for_binding(600.0, sleep_fn=sleep_fn), "tpu-a"
    )


class RefreshTest(BindingTestCase):

  def setUp(self):
    super().setUp()
    self.register("tpu-a")
    self.binder = self.make_binder()
    self.binder.bind()

  def test_it_pushes_the_expiry_out(self):
    self.clock.advance(protocol.BINDING_REFRESH_SECONDS)
    self.assertTrue(self.binder.refresh())
    blob = self.client.get(protocol.binding_path("tpu-a"))
    renewed = protocol.Binding.decode(blob.json())
    self.assertEqual(
        renewed.expires_at,
        self.clock.time() + protocol.BINDING_TTL_SECONDS,
    )

  def test_it_tracks_the_new_generation(self):
    before = self.binder.generation
    self.binder.refresh()
    self.assertNotEqual(self.binder.generation, before)
    blob = self.client.get(protocol.binding_path("tpu-a"))
    self.assertEqual(self.binder.generation, blob.generation)

  def test_it_fails_after_the_agent_revokes_the_binding(self):
    binding.revoke(self.client, "tpu-a")
    self.assertFalse(self.binder.refresh())
    self.assertFalse(self.binder.bound)

  def test_it_fails_after_somebody_else_takes_over(self):
    binding.revoke(self.client, "tpu-a")
    self.make_binder("worker-2").bind()
    self.assertFalse(self.binder.refresh())

  def test_it_does_nothing_when_unbound(self):
    self.binder.release()
    self.assertFalse(self.binder.refresh())


class CheckTest(BindingTestCase):

  def setUp(self):
    super().setUp()
    self.register("tpu-a")
    self.binder = self.make_binder()
    self.binder.bind()

  def test_it_passes_when_everything_is_in_order(self):
    self.assertTrue(self.binder.check().ok)

  def test_it_catches_a_revoked_binding(self):
    binding.revoke(self.client, "tpu-a")
    check = self.binder.check()
    self.assertFalse(check.ok)
    self.assertEqual(check.reason, "binding revoked")
    self.assertFalse(self.binder.bound)

  def test_it_catches_a_takeover(self):
    binding.revoke(self.client, "tpu-a")
    self.make_binder("worker-2").bind()
    check = self.binder.check()
    self.assertFalse(check.ok)
    self.assertEqual(check.reason, "binding taken over")

  def test_it_catches_a_quarantined_chip(self):
    self.register("tpu-a", state=protocol.STATE_QUARANTINED)
    check = self.binder.check()
    self.assertFalse(check.ok)
    self.assertIn("quarantined", check.reason)
    self.assertIn("chip is unhappy", check.reason)

  def test_it_catches_an_agent_that_stopped_heartbeating(self):
    self.clock.advance(protocol.FLEET_STALE_SECONDS + 1)
    check = self.binder.check()
    self.assertFalse(check.ok)
    self.assertEqual(check.reason, "agent heartbeat is stale")

  def test_it_catches_an_agent_that_vanished_entirely(self):
    self.client.delete(protocol.fleet_path("tpu-a"))
    check = self.binder.check()
    self.assertFalse(check.ok)
    self.assertEqual(check.reason, "agent never registered")

  def test_it_reports_unbound_rather_than_crashing(self):
    self.binder.release()
    self.assertEqual(self.binder.check().reason, "not bound")

  def test_it_costs_two_reads(self):
    """One for the binding, one for the agent's heartbeat."""
    before = self.client.count("get")
    self.binder.check()
    self.assertEqual(self.client.count("get") - before, 2)


class ReleaseTest(BindingTestCase):

  def test_it_frees_the_chip_for_the_next_worker(self):
    self.register("tpu-a")
    first = self.make_binder("worker-1")
    first.bind()
    first.release()
    self.assertEqual(self.make_binder("worker-2").bind(), "tpu-a")

  def test_it_forgets_the_local_state(self):
    self.register("tpu-a")
    binder = self.make_binder()
    binder.bind()
    binder.release()
    self.assertFalse(binder.bound)
    self.assertEqual(binder.generation, 0)

  def test_it_does_not_delete_a_claim_that_is_no_longer_ours(self):
    self.register("tpu-a")
    stale = self.make_binder("worker-1")
    stale.bind()
    binding.revoke(self.client, "tpu-a")
    successor = self.make_binder("worker-2")
    successor.bind()
    stale.release()
    blob = self.client.get(protocol.binding_path("tpu-a"))
    self.assertIsNotNone(blob)
    self.assertEqual(
        protocol.Binding.decode(blob.json()).worker_id, "worker-2"
    )

  def test_it_is_safe_to_call_twice(self):
    self.register("tpu-a")
    binder = self.make_binder()
    binder.bind()
    binder.release()
    binder.release()


class RevokeTest(BindingTestCase):

  def test_it_reports_whether_a_binding_was_there(self):
    self.register("tpu-a")
    self.make_binder().bind()
    self.assertTrue(binding.revoke(self.client, "tpu-a"))
    self.assertFalse(binding.revoke(self.client, "tpu-a"))

  def test_it_leaves_the_chip_bindable_again(self):
    self.register("tpu-a")
    self.make_binder("worker-1").bind()
    binding.revoke(self.client, "tpu-a")
    self.assertEqual(self.make_binder("worker-2").bind(), "tpu-a")


class GenerationSemanticsTest(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):
  """The fake is the contract, so pin its behaviour down directly."""

  def setUp(self):
    super().setUp()
    self.client = fake_gcs.FakeClient()

  def test_create_if_absent_succeeds_once(self):
    self.client.put("a", b"1", if_generation_match=0)
    with self.assertRaises(gcs.PreconditionFailed):
      self.client.put("a", b"2", if_generation_match=0)

  def test_matching_the_current_generation_succeeds(self):
    blob = self.client.put("a", b"1", if_generation_match=0)
    self.client.put("a", b"2", if_generation_match=blob.generation)

  def test_a_stale_generation_is_rejected(self):
    stale = self.client.put("a", b"1", if_generation_match=0)
    self.client.put("a", b"2", if_generation_match=stale.generation)
    with self.assertRaises(gcs.PreconditionFailed):
      self.client.put("a", b"3", if_generation_match=stale.generation)

  def test_generations_do_not_repeat_after_a_delete(self):
    first = self.client.put("a", b"1", if_generation_match=0)
    self.client.delete("a")
    second = self.client.put("a", b"2", if_generation_match=0)
    self.assertNotEqual(first.generation, second.generation)

  def test_no_precondition_always_writes(self):
    self.client.put("a", b"1")
    self.client.put("a", b"2")
    self.assertEqual(self.client.get("a").data, b"2")

  def test_deleting_a_missing_object_reports_false(self):
    self.assertFalse(self.client.delete("nope"))

  def test_deleting_at_a_stale_generation_is_rejected(self):
    stale = self.client.put("a", b"1", if_generation_match=0)
    self.client.put("a", b"2", if_generation_match=stale.generation)
    with self.assertRaises(gcs.PreconditionFailed):
      self.client.delete("a", if_generation_match=stale.generation)

  def test_listing_is_prefix_scoped_and_sorted(self):
    for name in ("b/2", "a/1", "b/1"):
      self.client.put(name, b"")
    self.assertEqual(self.client.list("b/"), ["b/1", "b/2"])


if __name__ == "__main__":
  unittest.main()
