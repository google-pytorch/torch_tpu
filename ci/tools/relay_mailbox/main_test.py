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

"""Unit tests for the two entry points."""

from __future__ import annotations

import io
import pathlib
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from ci.tools.relay_mailbox import agent_main
from ci.tools.relay_mailbox import binding as binding_lib
from ci.tools.relay_mailbox import config
from ci.tools.relay_mailbox import fake_gcs
from ci.tools.relay_mailbox import protocol
from ci.tools.relay_mailbox import worker_main


class ConfigTest(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def test_a_missing_setting_is_an_error_not_a_guess(self):
    with self.assertRaises(config.ConfigError):
      config.require({}, config.BUCKET_ENV)

  def test_an_empty_setting_counts_as_missing(self):
    with self.assertRaises(config.ConfigError):
      config.require({config.BUCKET_ENV: "   "}, config.BUCKET_ENV)

  def test_it_trims_whitespace(self):
    value = config.require({config.BUCKET_ENV: " b "}, config.BUCKET_ENV)
    self.assertEqual(value, "b")

  def test_the_worker_id_can_be_set_explicitly(self):
    self.assertEqual(
        config.worker_id({config.WORKER_ID_ENV: "w-7"}), "w-7"
    )

  def test_otherwise_it_makes_one_up(self):
    self.assertTrue(config.worker_id({}))


class AgentMainTest(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def setUp(self):
    super().setUp()
    self.client = fake_gcs.FakeClient()

  def test_it_needs_a_bucket(self):
    self.assertEqual(agent_main.main(env={}), 2)

  def test_it_needs_a_tpu_name(self):
    env = {config.BUCKET_ENV: "b"}
    with mock.patch.object(config, "bucket_client", return_value=self.client):
      self.assertEqual(agent_main.main(env=env), 2)

  def test_it_will_not_guess_the_tpu_name_from_the_hostname(self):
    """A wrong guess would have an agent serve someone else's mailbox."""
    env = {config.BUCKET_ENV: "b"}
    with self.assertRaises(config.ConfigError):
      agent_main.build(env, client=self.client)

  def test_it_builds_an_agent_for_the_named_tpu(self):
    env = {config.BUCKET_ENV: "b", config.TPU_ENV: "tpu-a"}
    agent = agent_main.build(env, client=self.client)
    agent.start()
    registry = binding_lib.FleetRegistry(self.client)
    self.assertEqual(registry.names(), ["tpu-a"])


class WorkerMainConfigTest(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def test_no_layers_configured_means_no_layers(self):
    self.assertEqual(worker_main.layer_dirs({}), [])

  def test_layers_are_split_on_colons(self):
    env = {config.LAYER_DIRS_ENV: "/a:/b"}
    self.assertEqual(worker_main.layer_dirs(env), ["/a", "/b"])

  def test_empty_entries_are_dropped(self):
    env = {config.LAYER_DIRS_ENV: "/a::/b:"}
    self.assertEqual(worker_main.layer_dirs(env), ["/a", "/b"])

  def test_the_layer_order_is_kept(self):
    env = {config.LAYER_DIRS_ENV: "/base:/test"}
    self.assertEqual(worker_main.layer_dirs(env), ["/base", "/test"])

  def test_the_timeout_falls_back_to_the_default(self):
    self.assertEqual(
        worker_main.timeout_seconds({}), worker_main.DEFAULT_TIMEOUT_SECONDS
    )

  def test_the_timeout_can_be_set(self):
    env = {config.TIMEOUT_ENV: "42"}
    self.assertEqual(worker_main.timeout_seconds(env), 42.0)

  def test_a_nonsense_timeout_is_rejected_rather_than_ignored(self):
    with self.assertRaises(config.ConfigError):
      worker_main.timeout_seconds({config.TIMEOUT_ENV: "soon"})


class WorkerMainRunTest(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def setUp(self):
    super().setUp()
    self.client = fake_gcs.FakeClient()
    self.env = {config.BUCKET_ENV: "b", config.WORKER_ID_ENV: "w-1"}

  def test_with_no_command_it_explains_itself(self):
    self.assertEqual(worker_main.main(argv=[], env=self.env), 2)

  def test_with_no_chip_it_fails_the_action(self):
    with mock.patch.object(config, "bucket_client", return_value=self.client):
      with mock.patch.object(
          worker_main.worker_lib.Worker, "dispatch"
      ) as dispatch:
        dispatch.return_value = worker_main.worker_lib.Dispatch(
            None, failure="tpu-a: timed out"
        )
        self.assertEqual(worker_main.run(["test"], self.env, self.client), 1)

  def test_it_exits_with_the_tests_own_status(self):
    outcome = self._fake_outcome(exit_code=5)
    with mock.patch.object(
        worker_main.worker_lib.Worker, "dispatch", return_value=outcome
    ):
      self.assertEqual(worker_main.run(["test"], self.env, self.client), 5)

  def test_it_replays_what_the_test_printed(self):
    outcome = self._fake_outcome(exit_code=0, stdout=b"hello from the chip")
    stdout = io.StringIO()
    with mock.patch.object(
        worker_main.worker_lib.Worker, "dispatch", return_value=outcome
    ):
      with mock.patch.object(sys, "stdout", stdout):
        worker_main.run(["test"], self.env, self.client)
    self.assertEqual(stdout.getvalue(), "hello from the chip")

  def _fake_outcome(self, *, exit_code, stdout=b""):
    item_id = "item-1"
    outputs = []
    if stdout:
      outputs.append("stdout")
      self.client.put(
          protocol.output_path("tpu-a", item_id, "stdout"), stdout
      )
    result = protocol.Result(
        item_id=item_id,
        exit_code=exit_code,
        started_at=0.0,
        finished_at=1.0,
        outputs=outputs,
    )
    return worker_main.worker_lib.Dispatch(result, tpu="tpu-a")


class PackRunfilesLayerTest(unittest.TestCase):

  def test_includes_csrc_test_binaries_but_excludes_common_solibs(self):
    temp = tempfile.TemporaryDirectory()
    self.addCleanup(temp.cleanup)
    root = pathlib.Path(temp.name)

    cpp_bin = root / "_main" / "csrc" / "api" / "tensor_buffer_test_bin"
    cpp_bin.parent.mkdir(parents=True, exist_ok=True)
    cpp_bin.write_bytes(b"ELF-binary")

    common_so = root / "_main" / "csrc" / "common" / "libpywrap_torch_tpu_common.so"
    common_so.parent.mkdir(parents=True, exist_ok=True)
    common_so.write_bytes(b"huge-solib")

    data = worker_main.pack_runfiles_layer(root)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
      names = tar.getnames()

    self.assertIn("_main/csrc/api/tensor_buffer_test_bin", names)
    self.assertNotIn("_main/csrc/common/libpywrap_torch_tpu_common.so", names)


if __name__ == "__main__":
  unittest.main()
