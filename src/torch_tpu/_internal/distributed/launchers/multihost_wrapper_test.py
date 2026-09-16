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

"""Tests the multihost_wrapper to check all mandatory variables are set."""

import contextlib
import os
from unittest import mock

from absl.testing import absltest
from torch_tpu._internal import env
from torch_tpu._internal import env_utils
from torch_tpu._internal.distributed.launchers import multihost_wrapper
from tests import seed_test_utils


@contextlib.contextmanager
def _mock_env(values: dict[str, str]):
  """Makes the environment look exactly like `values` to the wrapper.

  `multihost_wrapper` reads string and integer variables through
  `torch_tpu._internal.env`, whose getters memoize on first read, and writes
  them through `torch_tpu._internal.env_utils`, which memoizes too. Patching
  `os.environ` alone would therefore only take effect in whichever test happens
  to run first. This patches all of them so that each test sees a clean
  environment.

  Args:
    values: The environment variables to make visible.

  Yields:
    None. `os.environ` holds `values` for the duration, so a test can assert on
    what the wrapper wrote.
  """

  def get_str_env_once(name, default_value=None):
    return values.get(name, default_value)

  def get_int_env_once(name, default_value=None):
    return int(values[name]) if name in values else default_value

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
class MultiHostTestLauncherTest(seed_test_utils.RepeatableTest):

  def test_prepare_tpu_environment_multihost(self):
    with _mock_env(
        {"TPU_WORKER_HOSTNAMES": "host1,host2", "JOB_COMPLETION_INDEX": "0"}
    ):
      multihost_wrapper.prepare_tpu_environment()

      addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES")
      self.assertIsNotNone(addresses)
      address_list = addresses.split(",")
      self.assertLen(address_list, 16)

      expected = [
          "host1:8070",
          "host1:8071",
          "host1:8072",
          "host1:8073",
          "host1:8074",
          "host1:8075",
          "host1:8076",
          "host1:8077",
          "host2:8070",
          "host2:8071",
          "host2:8072",
          "host2:8073",
          "host2:8074",
          "host2:8075",
          "host2:8076",
          "host2:8077",
      ]
      self.assertEqual(address_list, expected)

  def test_prepare_tpu_environment_distributed_config(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "JOB_COMPLETION_INDEX": "1",
    }):
      multihost_wrapper.prepare_tpu_environment()

      self.assertEqual(os.environ.get("NNODES"), "2")
      self.assertEqual(os.environ.get("NODE_RANK"), "1")
      self.assertEqual(os.environ.get("MASTER_ADDR"), "host1")
      self.assertEqual(os.environ.get("MASTER_PORT"), "29500")

  def test_prepare_tpu_environment_multihost_custom_port(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "h1",
        "TORCH_TPU_BASE_PORT": "9000",
        "JOB_COMPLETION_INDEX": "0",
    }):
      multihost_wrapper.prepare_tpu_environment()

      addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES")
      self.assertIsNotNone(addresses)
      address_list = addresses.split(",")
      self.assertLen(address_list, 8)

      for i in range(8):
        self.assertEqual(address_list[i], f"h1:{9000 + i}")

  def test_prepare_tpu_environment_custom_devices_per_host(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "h1",
        "JOB_COMPLETION_INDEX": "0",
        "TORCH_TPU_DEVICES_PER_HOST": "4",
    }):
      multihost_wrapper.prepare_tpu_environment()

      addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES")
      self.assertIsNotNone(addresses)
      address_list = addresses.split(",")
      self.assertLen(address_list, 4)
      for i in range(4):
        self.assertEqual(address_list[i], f"h1:{8070 + i}")

  def test_prepare_tpu_environment_preserve_master_addr_and_port(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "JOB_COMPLETION_INDEX": "1",
        "MASTER_ADDR": "host_override",
        "MASTER_PORT": "12345",
    }):
      multihost_wrapper.prepare_tpu_environment()

      self.assertEqual(os.environ.get("MASTER_ADDR"), "host_override")
      self.assertEqual(os.environ.get("MASTER_PORT"), "12345")

  def test_prepare_tpu_environment_preserve_slicebuilder_addresses(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "h1",
        "JOB_COMPLETION_INDEX": "0",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": "h1:1234",
    }):
      multihost_wrapper.prepare_tpu_environment()

      self.assertEqual(
          os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES"), "h1:1234"
      )

  def test_prepare_tpu_environment_missing_worker_hostnames(self):
    with _mock_env({"JOB_COMPLETION_INDEX": "0"}):
      with self.assertRaisesRegex(ValueError, "No TPU_WORKER_HOSTNAMES found."):
        multihost_wrapper.prepare_tpu_environment()

  def test_prepare_tpu_environment_missing_completion_index(self):
    with _mock_env({"TPU_WORKER_HOSTNAMES": "h1"}):
      with self.assertRaisesRegex(
          ValueError, "No JOB_COMPLETION_INDEX found in environment."
      ):
        multihost_wrapper.prepare_tpu_environment()


if __name__ == "__main__":
  absltest.main()
