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
import io
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

  `multihost_wrapper` reads variables through `torch_tpu._internal.env`, whose
  getters memoize on first read, and writes them through
  `torch_tpu._internal.env_utils`, which memoizes too. Patching `os.environ`
  alone would therefore only take effect in whichever test happens to run first.
  This patches all of them so that each test sees a clean environment.

  Args:
    values: The environment variables to make visible.

  Yields:
    None. `os.environ` holds `values` for the duration, so a test can assert on
    what the wrapper wrote.
  """

  def get_str_env_once(name, default_value=None):
    if name in os.environ:
      return os.environ[name]
    return values.get(name, default_value)

  def get_int_env_once(name, default_value=None):
    if name in os.environ:
      return int(os.environ[name])
    return int(values[name]) if name in values else default_value

  def get_bool_env_once(name, default_value=None):
    raw_value = os.environ.get(name, values.get(name))
    if raw_value is not None:
      val = str(raw_value).lower()
      if val in ("1", "true", "yes", "on"):
        return True
      if val in ("0", "false", "no", "off"):
        return False
      return default_value
    return default_value

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
    stack.enter_context(
        mock.patch.object(
            env, "get_bool_env_once", side_effect=get_bool_env_once
        )
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

  def test_parse_topology(self):
    self.assertEqual(multihost_wrapper.parse_topology("4x4x4"), (4, 4, 4))
    self.assertEqual(multihost_wrapper.parse_topology("2,2,1"), (2, 2, 1))
    self.assertEqual(multihost_wrapper.parse_topology("2,2,1,2"), (2, 2, 1))
    self.assertEqual(multihost_wrapper.parse_topology("2x4"), (2, 4, 1))
    self.assertEqual(multihost_wrapper.parse_topology("4"), (4, 1, 1))

  def test_get_slice_host_coordinates_1d(self):
    # 1D topology "4" -> (4, 1, 1), with (2, 2, 1) chips_per_host:
    # Bx = 4 // 2 = 2, By = 1, Bz = 1 -> 2 hosts along X.
    hosts = multihost_wrapper.get_slice_host_coordinates("4")
    self.assertEqual(hosts, [(0, 0, 0), (1, 0, 0)])

  def test_get_slice_host_coordinates_2d(self):
    # 2D topology "2x4" -> (2, 4, 1), with (2, 2, 1) chips_per_host:
    # Bx = 1, By = 2, Bz = 1 -> 2 hosts along Y.
    hosts = multihost_wrapper.get_slice_host_coordinates("2x4")
    self.assertEqual(hosts, [(0, 0, 0), (0, 1, 0)])

  def test_get_slice_host_coordinates_3d(self):
    # 3D topology "4x4x4" -> (4, 4, 4), with (2, 2, 1) chips_per_host:
    # Bx = 2, By = 2, Bz = 4 -> 16 hosts.
    # Sorted reverse-lexicographically: Z-major, Y-intermediate, X-minor.
    hosts = multihost_wrapper.get_slice_host_coordinates("4x4x4")
    self.assertLen(hosts, 16)
    expected = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (0, 0, 2),
        (1, 0, 2),
        (0, 1, 2),
        (1, 1, 2),
        (0, 0, 3),
        (1, 0, 3),
        (0, 1, 3),
        (1, 1, 3),
    ]
    self.assertEqual(hosts, expected)

  def test_get_slice_host_coordinates_3d_single_host(self):
    # 3D topology "2,2,1" -> 1 host (0, 0, 0).
    hosts = multihost_wrapper.get_slice_host_coordinates("2,2,1")
    self.assertEqual(hosts, [(0, 0, 0)])

  def test_get_slice_host_coordinates_4d(self):
    # 4D topology "4,4,4,2" -> parsed to (4, 4, 4), 16 hosts.
    hosts_4d = multihost_wrapper.get_slice_host_coordinates("4,4,4,2")
    hosts_3d = multihost_wrapper.get_slice_host_coordinates("4x4x4")
    self.assertEqual(hosts_4d, hosts_3d)

    # 4D topology "2,2,1,2" -> parsed to (2, 2, 1), 1 host.
    hosts_single = multihost_wrapper.get_slice_host_coordinates("2,2,1,2")
    self.assertEqual(hosts_single, [(0, 0, 0)])

  def test_calculate_topological_node_rank_linearization_3d(self):
    topo = "4x4x4"
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank([0, 0, 0], topo), 0
    )
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank([2, 0, 0], topo), 1
    )
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank([0, 2, 0], topo), 2
    )
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank([2, 2, 0], topo), 3
    )
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank([0, 0, 1], topo), 4
    )
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank([2, 0, 1], topo), 5
    )
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank([0, 2, 1], topo), 6
    )
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank([2, 2, 1], topo), 7
    )
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank([2, 2, 3], topo), 15
    )

  def test_calculate_topological_node_rank_various_topologies(self):
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank([0, 0, 0], "2,2,1"), 0
    )
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank([1, 1, 0], "2,2,1"), 0
    )
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank(
            [2, 2, 0, 1], "4,4,4,2"
        ),
        3,
    )
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank([0, 2, 0], "2x4x1"), 1
    )

  def test_calculate_topological_node_rank_fallback(self):
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank(
            [], "4x4x4", fallback_index="2"
        ),
        2,
    )
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank(
            None, "4x4x4", fallback_index=3
        ),
        3,
    )
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank(
            [0, 0, 0], "", fallback_index="1"
        ),
        1,
    )

  def test_calculate_topological_node_rank_out_of_bounds_fallback(self):
    # Chip coordinates out of bounds: host (5, 0, 0) does not exist in 4x4x4.
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank(
            [10, 0, 0], "4x4x4", fallback_index="7"
        ),
        7,
    )
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank(
            [10, 0, 0], "4x4x4", fallback_index=8
        ),
        8,
    )

    # Negative coordinates out of bounds
    self.assertEqual(
        multihost_wrapper.calculate_topological_node_rank(
            [-1, 0, 0], "4x4x4", fallback_index="4"
        ),
        4,
    )

  def test_prepare_tpu_environment_with_coordinates(self):
    env_vars = {
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "JOB_COMPLETION_INDEX": "5",
        "PJRT_DEVICE": "TPU",
    }
    with (
        _mock_env(env_vars),
        mock.patch(
            "torch_tpu._internal.distributed.tpu_distributed.local_device_coords",
            return_value=[2, 0, 0],
        ),
    ):
      multihost_wrapper.prepare_tpu_environment()
      self.assertEqual(os.environ.get("NODE_RANK"), "1")

  def test_prepare_tpu_environment_with_coordinates_and_no_job_completion_index(
      self,
  ):
    env_vars = {
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "PJRT_DEVICE": "TPU",
    }
    with (
        _mock_env(env_vars),
        mock.patch(
            "torch_tpu._internal.distributed.tpu_distributed.local_device_coords",
            return_value=[0, 2, 0],
        ),
    ):
      multihost_wrapper.prepare_tpu_environment()
      self.assertEqual(os.environ.get("NODE_RANK"), "2")

  def test_prepare_tpu_environment_coordinate_lookup_failure_fallback(self):
    env_vars = {
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "JOB_COMPLETION_INDEX": "3",
        "PJRT_DEVICE": "TPU",
    }
    with (
        _mock_env(env_vars),
        mock.patch(
            "torch_tpu._internal.distributed.tpu_distributed.local_device_coords",
            side_effect=RuntimeError("no device"),
        ),
    ):
      multihost_wrapper.prepare_tpu_environment()
      self.assertEqual(os.environ.get("NODE_RANK"), "3")

  def test_print_shell_exports_stdout_format(self):
    env_vars = {
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "JOB_COMPLETION_INDEX": "1",
    }
    with _mock_env(env_vars):
      multihost_wrapper.prepare_tpu_environment()
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        multihost_wrapper.print_shell_exports()
      output = buf.getvalue()

      self.assertIn('export TORCH_TPU_TOPOLOGY="4x4x4"\n', output)
      self.assertIn('export NODE_RANK="1"\n', output)
      self.assertIn('export NNODES="2"\n', output)
      self.assertIn('export MASTER_ADDR="host1"\n', output)
      self.assertIn('export MASTER_PORT="29500"\n', output)
      self.assertIn("export TORCH_TPU_SLICEBUILDER_ADDRESSES=", output)
      self.assertIn("export TORCH_TPU_XPROF_SESSION_ID=", output)

  def test_print_shell_exports_always_prints_node_rank(self):
    with _mock_env({"NODE_RANK": "3"}):
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        multihost_wrapper.print_shell_exports()
      output = buf.getvalue()
      self.assertIn('export NODE_RANK="3"\n', output)


if __name__ == "__main__":
  absltest.main()
