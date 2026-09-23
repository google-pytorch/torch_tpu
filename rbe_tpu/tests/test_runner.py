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

"""Unit tests for materialising and running work items."""

from __future__ import annotations

import io
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import fake_gcs
from rbe_tpu.common import gcs
from rbe_tpu.common import protocol
from rbe_tpu.common import runner as runner_lib
from rbe_tpu.worker import worker as worker_lib


class RunnerTestCase(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def setUp(self):
    super().setUp()
    self.client = fake_gcs.FakeClient()
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = os.path.join(self.temp.name, "relay")
    self.runner = runner_lib.LocalRunner(self.client, self.root)

  def make_layer(self, files) -> str:
    """Uploads a directory as a layer and returns its blob name."""
    source = tempfile.mkdtemp(dir=self.temp.name)
    for name, data in files.items():
      path = os.path.join(source, name)
      os.makedirs(os.path.dirname(path), exist_ok=True)
      with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    worker = worker_lib.Worker(self.client, "w")
    return worker.upload_layers([source])[0]

  def make_item(self, argv, *, layers=(), workdir="", timeout=60.0, env=None):
    return protocol.WorkItem(
        item_id="item-1",
        tpu="tpu-a",
        binding_generation=1,
        argv=list(argv),
        workdir=workdir,
        timeout_seconds=timeout,
        env=env or {},
        layers=list(layers),
    )


class ExecuteTest(RunnerTestCase):

  def test_it_reports_a_clean_exit(self):
    result = self.runner(self.make_item(["true"]))
    self.assertEqual(result.exit_code, 0)

  def test_it_reports_a_failing_exit(self):
    result = self.runner(self.make_item(["false"]))
    self.assertEqual(result.exit_code, 1)

  def test_it_captures_stdout_and_stderr_separately(self):
    item = self.make_item(["sh", "-c", "echo out; echo err >&2"])
    result = self.runner(item)
    self.assertEqual(result.output_files["stdout"], b"out\n")
    self.assertEqual(result.output_files["stderr"], b"err\n")

  def test_the_combined_output_has_both(self):
    item = self.make_item(["sh", "-c", "echo out; echo err >&2"])
    result = self.runner(item)
    self.assertIn("out", result.output)
    self.assertIn("err", result.output)

  def test_it_passes_the_environment_through(self):
    item = self.make_item(
        ["sh", "-c", "echo $RELAY_TEST_VAR"], env={"RELAY_TEST_VAR": "set"}
    )
    self.assertEqual(self.runner(item).output_files["stdout"], b"set\n")

  def test_it_keeps_the_ambient_environment_too(self):
    item = self.make_item(["sh", "-c", "echo $PATH"])
    self.assertNotEqual(self.runner(item).output_files["stdout"], b"\n")

  def test_a_command_that_does_not_exist_is_an_infrastructure_fault(self):
    result = self.runner(self.make_item(["/no/such/binary"]))
    self.assertEqual(result.exit_code, 1)
    self.assertTrue(result.infra_reason)

  def test_a_test_that_overruns_is_the_tests_own_problem(self):
    """A slow test must not be blamed on the chip."""
    item = self.make_item(["sleep", "30"], timeout=0.2)
    result = self.runner(item)
    self.assertEqual(result.exit_code, 1)
    self.assertFalse(result.infra_reason)
    self.assertIn("killed after", result.output)

  def test_it_truncates_a_flood_of_output(self):
    size = runner_lib.MAX_CAPTURED_BYTES + 5000
    item = self.make_item(
        ["sh", "-c", f"head -c {size} /dev/zero | tr '\\0' 'x'"],
        timeout=60.0,
    )
    result = self.runner(item)
    self.assertEqual(
        len(result.output_files["stdout"]), runner_lib.MAX_CAPTURED_BYTES
    )


class LayerTest(RunnerTestCase):

  def test_it_unpacks_a_layer_into_the_run_directory(self):
    layer = self.make_layer({"hello.txt": "world"})
    item = self.make_item(["cat", "hello.txt"], layers=[layer])
    self.assertEqual(self.runner(item).output_files["stdout"], b"world")

  def test_later_layers_land_on_top_of_earlier_ones(self):
    base = self.make_layer({"f.txt": "base"})
    top = self.make_layer({"f.txt": "top"})
    item = self.make_item(["cat", "f.txt"], layers=[base, top])
    self.assertEqual(self.runner(item).output_files["stdout"], b"top")

  def test_it_unpacks_nested_paths(self):
    layer = self.make_layer({"a/b/c.txt": "deep"})
    item = self.make_item(["cat", "a/b/c.txt"], layers=[layer])
    self.assertEqual(self.runner(item).output_files["stdout"], b"deep")

  def test_it_downloads_a_layer_only_once(self):
    """The shared layer is big; re-fetching it rebuilds the funnel."""
    layer = self.make_layer({"f.txt": "x"})
    self.runner(self.make_item(["true"], layers=[layer]))
    gets = self.client.count("get", "blobs/")
    self.runner(self.make_item(["true"], layers=[layer]))
    self.assertEqual(self.client.count("get", "blobs/"), gets)

  def test_a_missing_layer_raises_so_the_action_can_move(self):
    item = self.make_item(["true"], layers=["blobs/deadbeef"])
    with self.assertRaises(gcs.Error):  # ASSERT_RAISES_OK=CI tool
      self.runner(item)

  def test_a_half_written_download_is_not_reused(self):
    layer = self.make_layer({"f.txt": "x"})
    cache = pathlib.Path(self.root) / "layers"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / (layer.rsplit("/", 1)[-1] + ".partial")).write_bytes(b"junk")
    item = self.make_item(["cat", "f.txt"], layers=[layer])
    self.assertEqual(self.runner(item).output_files["stdout"], b"x")

  def test_it_refuses_a_layer_that_tries_to_escape(self):
    """A layer naming ../ must not write outside the run directory."""
    self.client.put(protocol.blob_path("evil"), _traversal_tar())
    item = self.make_item(["true"], layers=[protocol.blob_path("evil")])
    with self.assertRaises(  # ASSERT_RAISES_OK=CI tool
        runner_lib.tarfile.TarError
    ):
      self.runner(item)

  def test_an_absolute_path_is_made_relative_not_honoured(self):
    """The data filter strips the leading slash rather than refusing."""
    target = os.path.join(self.temp.name, "escaped-abs.txt")
    self.client.put(protocol.blob_path("abs"), _tar_with_member(target))
    item = self.make_item(["true"], layers=[protocol.blob_path("abs")])
    self.runner(item)
    self.assertFalse(os.path.exists(target))


def _tar_with_member(name: str) -> bytes:
  buffer = io.BytesIO()
  with tarfile.open(fileobj=buffer, mode="w") as tar:
    payload = b"pwned"
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))
  return buffer.getvalue()


def _traversal_tar() -> bytes:
  return _tar_with_member("../escaped.txt")


class CleanupTest(RunnerTestCase):

  def test_it_clears_the_run_directory_afterwards(self):
    layer = self.make_layer({"f.txt": "x"})
    self.runner(self.make_item(["true"], layers=[layer]))
    runs = pathlib.Path(self.root) / "runs"
    self.assertEqual(list(runs.glob("*")) if runs.exists() else [], [])

  def test_it_clears_up_even_when_the_command_blows_up(self):
    self.runner(self.make_item(["/no/such/binary"]))
    runs = pathlib.Path(self.root) / "runs"
    self.assertEqual(list(runs.glob("*")) if runs.exists() else [], [])

  def test_it_keeps_the_layer_cache(self):
    layer = self.make_layer({"f.txt": "x"})
    self.runner(self.make_item(["true"], layers=[layer]))
    cache = pathlib.Path(self.root) / "layers"
    self.assertNotEqual(list(cache.glob("*")), [])


class RunsFromWorkdirTest(RunnerTestCase):

  def test_it_honours_an_explicit_working_directory(self):
    elsewhere = tempfile.mkdtemp(dir=self.temp.name)
    with open(os.path.join(elsewhere, "there.txt"), "w") as handle:
      handle.write("found")
    item = self.make_item(["cat", "there.txt"], workdir=elsewhere)
    self.assertEqual(self.runner(item).output_files["stdout"], b"found")


class TimeoutPlumbingTest(RunnerTestCase):

  def test_the_items_timeout_is_what_gets_enforced(self):
    item = self.make_item(["true"], timeout=12.5)
    with mock.patch.object(
        runner_lib.subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 0, b"", b""),
    ) as run:
      self.runner(item)
    self.assertEqual(run.call_args.kwargs["timeout"], 12.5)


if __name__ == "__main__":
  unittest.main()
