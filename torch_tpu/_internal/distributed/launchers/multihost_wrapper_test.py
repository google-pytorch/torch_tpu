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

import os
from unittest import mock

from absl.testing import absltest
from torch_tpu._internal.distributed.launchers import multihost_wrapper


class MultiHostTestLauncherTest(absltest.TestCase):

  def test_prepare_tpu_environment_multihost(self):
    with mock.patch.dict(
        os.environ,
        {"TPU_WORKER_HOSTNAMES": "host1,host2", "JOB_COMPLETION_INDEX": "0"},
    ):
      # Clear existing if any
      if "TORCH_TPU_SLICEBUILDER_ADDRESSES" in os.environ:
        del os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"]

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
    env = {
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "JOB_COMPLETION_INDEX": "1",
    }
    with mock.patch.dict(os.environ, env, clear=True):
      multihost_wrapper.prepare_tpu_environment()

      self.assertEqual(os.environ.get("NNODES"), "2")
      self.assertEqual(os.environ.get("NODE_RANK"), "1")
      self.assertEqual(os.environ.get("MASTER_ADDR"), "host1")
      self.assertEqual(os.environ.get("MASTER_PORT"), "29500")

  def test_prepare_tpu_environment_multihost_custom_port(self):
    with mock.patch.dict(
        os.environ,
        {
            "TPU_WORKER_HOSTNAMES": "h1",
            "TORCH_TPU_BASE_PORT": "9000",
            "JOB_COMPLETION_INDEX": "0",
        },
    ):
      # Clear existing if any
      if "TORCH_TPU_SLICEBUILDER_ADDRESSES" in os.environ:
        del os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"]

      multihost_wrapper.prepare_tpu_environment()

      addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES")
      self.assertIsNotNone(addresses)
      address_list = addresses.split(",")
      self.assertLen(address_list, 8)

      for i in range(8):
        self.assertEqual(address_list[i], f"h1:{9000 + i}")

  def test_prepare_tpu_environment_custom_devices_per_host(self):
    env = {
        "TPU_WORKER_HOSTNAMES": "h1",
        "JOB_COMPLETION_INDEX": "0",
        "TORCH_TPU_DEVICES_PER_HOST": "4",
    }
    with mock.patch.dict(os.environ, env, clear=True):
      multihost_wrapper.prepare_tpu_environment()

      addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES")
      self.assertIsNotNone(addresses)
      address_list = addresses.split(",")
      self.assertLen(address_list, 4)
      for i in range(4):
        self.assertEqual(address_list[i], f"h1:{8070 + i}")

  def test_prepare_tpu_environment_preserve_master_addr_and_port(self):
    env = {
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "JOB_COMPLETION_INDEX": "1",
        "MASTER_ADDR": "host_override",
        "MASTER_PORT": "12345",
    }
    with mock.patch.dict(os.environ, env, clear=True):
      multihost_wrapper.prepare_tpu_environment()

      self.assertEqual(os.environ.get("MASTER_ADDR"), "host_override")
      self.assertEqual(os.environ.get("MASTER_PORT"), "12345")

  def test_prepare_tpu_environment_missing_worker_hostnames(self):
    with mock.patch.dict(os.environ, {"JOB_COMPLETION_INDEX": "0"}, clear=True):
      with self.assertRaisesRegex(ValueError, "No TPU_WORKER_HOSTNAMES found."):
        multihost_wrapper.prepare_tpu_environment()

  def test_prepare_tpu_environment_missing_completion_index(self):
    with mock.patch.dict(
        os.environ, {"TPU_WORKER_HOSTNAMES": "h1"}, clear=True
    ):
      with self.assertRaisesRegex(
          ValueError, "No JOB_COMPLETION_INDEX found in environment."
      ):
        multihost_wrapper.prepare_tpu_environment()


if __name__ == "__main__":
  absltest.main()
