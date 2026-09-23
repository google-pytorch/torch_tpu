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

"""Tests the unified `environment` module for single-host and multi-host."""

import contextlib
import io
import os
from unittest import mock

from absl.testing import absltest
from torch_tpu._internal import env
from torch_tpu._internal import env_utils
from torch_tpu._internal.distributed.launchers import environment
from torch_tpu._internal.utils import hardware
from tests import seed_test_utils


@contextlib.contextmanager
def _mock_env(values: dict[str, str]):
  """Makes the environment look exactly like `values` to the launcher."""

  def get_str_env_once(name, default_value=None):
    return os.environ.get(name, default_value)

  def get_int_env_once(name, default_value=None):
    return int(os.environ[name]) if name in os.environ else default_value

  with contextlib.ExitStack() as stack:
    stack.enter_context(mock.patch.dict(os.environ, values, clear=True))
    stack.enter_context(
        mock.patch.dict(env_utils._resolved_values, {}, clear=True)  # pylint: disable=protected-access
    )
    stack.enter_context(
        mock.patch.object(env, "get_str_env_once", side_effect=get_str_env_once)
    )
    stack.enter_context(
        mock.patch.object(env, "get_int_env_once", side_effect=get_int_env_once)
    )
    yield


class EnvironmentTest(seed_test_utils.RepeatableTest):

  def test_singlehost_auto_detection(self):
    with _mock_env({}):
      with (
          mock.patch.object(hardware, "get_tpu_topology", return_value="2x2x1"),
          mock.patch.object(hardware, "get_tpu_device_count", return_value=4),
      ):
        environment.set_tpu_launch_env()
        self.assertEqual(os.environ.get("TORCH_TPU_TOPOLOGY"), "2x2x1")
        addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES", "")
        self.assertLen(addresses.split(","), 4)
        self.assertTrue(
            all(addr.startswith("localhost:") for addr in addresses.split(","))
        )

  def test_single_worker_hostname_inferred_as_singlehost(self):
    with _mock_env({"TPU_WORKER_HOSTNAMES": "localhost"}):
      with (
          mock.patch.object(hardware, "get_tpu_topology", return_value="2x2x2"),
          mock.patch.object(hardware, "get_tpu_device_count", return_value=8),
      ):
        environment.set_tpu_launch_env()
        self.assertEqual(os.environ.get("TORCH_TPU_TOPOLOGY"), "2x2x2")
        self.assertIsNone(os.environ.get("NNODES"))
        addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES", "")
        self.assertLen(addresses.split(","), 8)
        self.assertTrue(
            all(addr.startswith("localhost:") for addr in addresses.split(","))
        )

  def test_single_worker_hostname_with_job_completion_index_uses_multihost(
      self,
  ):
    with _mock_env(
        {"TPU_WORKER_HOSTNAMES": "localhost", "JOB_COMPLETION_INDEX": "0"}
    ):
      environment.set_tpu_launch_env(nproc_per_node=8)
      self.assertEqual(os.environ.get("NNODES"), "1")
      self.assertEqual(os.environ.get("NODE_RANK"), "0")
      self.assertEqual(os.environ.get("MASTER_ADDR"), "localhost")
      self.assertEqual(os.environ.get("MASTER_PORT"), "29500")
      addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES", "")
      self.assertEqual(
          addresses.split(","),
          [f"localhost:{8070 + i}" for i in range(8)],
      )

  def test_multihost_auto_detection_from_env(self):
    with _mock_env(
        {"TPU_WORKER_HOSTNAMES": "host1,host2", "JOB_COMPLETION_INDEX": "0"}
    ):
      environment.set_tpu_launch_env()

      addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES")
      self.assertIsNotNone(addresses)
      address_list = addresses.split(",")
      self.assertLen(address_list, 16)
      self.assertEqual(os.environ.get("NNODES"), "2")
      self.assertEqual(os.environ.get("NODE_RANK"), "0")
      self.assertEqual(os.environ.get("MASTER_ADDR"), "host1")
      self.assertEqual(os.environ.get("MASTER_PORT"), "29500")

  def test_print_launch_env_singlehost(self):
    with _mock_env({}):
      with (
          mock.patch.object(hardware, "get_tpu_topology", return_value="2x2x1"),
          mock.patch.object(hardware, "get_tpu_device_count", return_value=4),
      ):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
          environment._print_launch_env()  # pylint: disable=protected-access
        lines = dict(
            line.split("=", 1) for line in buf.getvalue().strip().splitlines()
        )
        self.assertEqual(lines.get("TORCH_TPU_TOPOLOGY"), '"2x2x1"')
        self.assertEqual(lines.get("WORLD_SIZE"), "4")
        self.assertIn("TORCH_TPU_SLICEBUILDER_ADDRESSES", lines)


if __name__ == "__main__":
  absltest.main()
