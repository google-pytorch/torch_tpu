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

# pylint: disable=protected-access

import contextlib
import io
import os
import subprocess
import sys
from unittest import mock

from absl.testing import absltest
from torch.tpu.distributed import environment  # pyrefly: ignore[missing-import]
from torch_tpu._internal import env
from torch_tpu._internal import env_utils
from torch_tpu._internal.distributed import environment as internal_environment
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
        mock.patch.dict(env_utils._resolved_values, {}, clear=True)
    )
    stack.enter_context(
        mock.patch.object(env, "get_str_env_once", side_effect=get_str_env_once)
    )
    stack.enter_context(
        mock.patch.object(env, "get_int_env_once", side_effect=get_int_env_once)
    )
    yield


def _coords_process(
    coords: list[int], prefix_lines: str = ""
) -> subprocess.CompletedProcess[str]:
  """Returns a finished child process that reported `coords` on stdout."""
  return subprocess.CompletedProcess(
      args=[],
      returncode=0,
      stdout=f"{prefix_lines}TORCH_TPU_LOCAL_DEVICE_COORDS={coords}\n",
      stderr="",
  )


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
          internal_environment._print_launch_env()
        lines = dict(
            line.split("=", 1) for line in buf.getvalue().strip().splitlines()
        )
        self.assertEqual(lines.get("TORCH_TPU_TOPOLOGY"), '"2x2x1"')
        self.assertEqual(lines.get("WORLD_SIZE"), "4")
        self.assertIn("TORCH_TPU_SLICEBUILDER_ADDRESSES", lines)
        self.assertEqual(lines.get("NODE_RANK"), '"0"')

  def test_print_launch_env_always_prints_node_rank(self):
    with _mock_env({"NODE_RANK": "3"}):
      with (
          mock.patch.object(hardware, "get_tpu_topology", return_value="2x2x1"),
          mock.patch.object(hardware, "get_tpu_device_count", return_value=4),
      ):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
          internal_environment._print_launch_env()
        self.assertIn('NODE_RANK="3"\n', buf.getvalue())

  def test_print_launch_env_multihost_with_coordinates(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "JOB_COMPLETION_INDEX": "1",
    }):
      with (
          mock.patch.object(hardware, "get_tpu_device_count", return_value=4),
          mock.patch.object(
              subprocess, "run", return_value=_coords_process([2, 0, 0])
          ),
      ):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
          internal_environment._print_launch_env()
      output = buf.getvalue()
      self.assertIn('TORCH_TPU_TOPOLOGY="4x4x4"\n', output)
      self.assertIn('NODE_RANK="1"\n', output)
      self.assertIn('NNODES="2"\n', output)
      self.assertIn('MASTER_ADDR="host1"\n', output)
      self.assertIn('MASTER_PORT="29500"\n', output)
      self.assertIn("TORCH_TPU_SLICEBUILDER_ADDRESSES=", output)
      self.assertIn("TORCH_TPU_XPROF_SESSION_ID=", output)


class TopologyTest(seed_test_utils.RepeatableTest):

  def test_parse_topology(self):
    self.assertEqual(internal_environment._parse_topology("4x4x4"), (4, 4, 4))
    self.assertEqual(internal_environment._parse_topology("2,2,1"), (2, 2, 1))
    self.assertEqual(internal_environment._parse_topology("2,2,1,2"), (2, 2, 1))
    self.assertEqual(internal_environment._parse_topology("2x4"), (2, 4, 1))
    self.assertEqual(internal_environment._parse_topology("2X4"), (2, 4, 1))
    self.assertEqual(internal_environment._parse_topology("4"), (4, 1, 1))

  def test_get_slice_host_coordinates_1d(self):
    # 1D topology "4" -> (4, 1, 1), with (2, 2, 1) chips_per_host:
    # Bx = 4 // 2 = 2, By = 1, Bz = 1 -> 2 hosts along X.
    hosts = internal_environment._get_slice_host_coordinates("4")
    self.assertEqual(hosts, [(0, 0, 0), (1, 0, 0)])

  def test_get_slice_host_coordinates_2d(self):
    # 2D topology "2x4" -> (2, 4, 1), with (2, 2, 1) chips_per_host:
    # Bx = 1, By = 2, Bz = 1 -> 2 hosts along Y.
    hosts = internal_environment._get_slice_host_coordinates("2x4")
    self.assertEqual(hosts, [(0, 0, 0), (0, 1, 0)])

  def test_get_slice_host_coordinates_3d(self):
    # 3D topology "4x4x4" -> (4, 4, 4), with (2, 2, 1) chips_per_host:
    # Bx = 2, By = 2, Bz = 4 -> 16 hosts.
    # Sorted reverse-lexicographically: Z-major, Y-intermediate, X-minor.
    hosts = internal_environment._get_slice_host_coordinates("4x4x4")
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
    hosts = internal_environment._get_slice_host_coordinates("2,2,1")
    self.assertEqual(hosts, [(0, 0, 0)])

  def test_get_slice_host_coordinates_4d(self):
    # 4D topology "4,4,4,2" -> parsed to (4, 4, 4), 16 hosts.
    hosts_4d = internal_environment._get_slice_host_coordinates("4,4,4,2")
    hosts_3d = internal_environment._get_slice_host_coordinates("4x4x4")
    self.assertEqual(hosts_4d, hosts_3d)

    # 4D topology "2,2,1,2" -> parsed to (2, 2, 1), 1 host.
    hosts_single = internal_environment._get_slice_host_coordinates("2,2,1,2")
    self.assertEqual(hosts_single, [(0, 0, 0)])

  def test_calculate_topological_node_rank_linearization_3d(self):
    topo = "4x4x4"
    for coords, expected_rank in [
        ([0, 0, 0], 0),
        ([2, 0, 0], 1),
        ([0, 2, 0], 2),
        ([2, 2, 0], 3),
        ([0, 0, 1], 4),
        ([2, 0, 1], 5),
        ([0, 2, 1], 6),
        ([2, 2, 1], 7),
        ([2, 2, 3], 15),
        # Every chip on a host maps to that host's rank.
        ([3, 3, 3], 15),
    ]:
      with self.subTest(coords=coords):
        self.assertEqual(
            internal_environment._calculate_topological_node_rank(coords, topo),
            expected_rank,
        )

  def test_calculate_topological_node_rank_various_topologies(self):
    self.assertEqual(
        internal_environment._calculate_topological_node_rank(
            [0, 0, 0], "2,2,1"
        ),
        0,
    )
    self.assertEqual(
        internal_environment._calculate_topological_node_rank(
            [1, 1, 0], "2,2,1"
        ),
        0,
    )
    self.assertEqual(
        internal_environment._calculate_topological_node_rank(
            [2, 2, 0, 1], "4,4,4,2"
        ),
        3,
    )
    self.assertEqual(
        internal_environment._calculate_topological_node_rank(
            [0, 2, 0], "2x4x1"
        ),
        1,
    )
    # Missing trailing coordinates default to 0.
    self.assertEqual(
        internal_environment._calculate_topological_node_rank([2, 2], "4x4x4"),
        3,
    )

  def test_calculate_topological_node_rank_fallback(self):
    self.assertEqual(
        internal_environment._calculate_topological_node_rank(
            [], "4x4x4", fallback_index="2"
        ),
        2,
    )
    self.assertEqual(
        internal_environment._calculate_topological_node_rank(
            None, "4x4x4", fallback_index=3
        ),
        3,
    )
    self.assertEqual(
        internal_environment._calculate_topological_node_rank(
            [0, 0, 0], "", fallback_index="1"
        ),
        1,
    )
    self.assertEqual(
        internal_environment._calculate_topological_node_rank(
            [0, 0, 0], None, fallback_index="1"
        ),
        1,
    )

  def test_calculate_topological_node_rank_out_of_bounds_fallback(self):
    # Chip coordinates out of bounds: host (5, 0, 0) does not exist in 4x4x4.
    self.assertEqual(
        internal_environment._calculate_topological_node_rank(
            [10, 0, 0], "4x4x4", fallback_index="7"
        ),
        7,
    )
    self.assertEqual(
        internal_environment._calculate_topological_node_rank(
            [10, 0, 0], "4x4x4", fallback_index=8
        ),
        8,
    )

    # Negative coordinates out of bounds
    self.assertEqual(
        internal_environment._calculate_topological_node_rank(
            [-1, 0, 0], "4x4x4", fallback_index="4"
        ),
        4,
    )

  def test_calculate_topological_node_rank_invalid_topology_fallback(self):
    self.assertEqual(
        internal_environment._calculate_topological_node_rank(
            [0, 0, 0], "4x4xfour", fallback_index="6"
        ),
        6,
    )
    self.assertEqual(
        internal_environment._calculate_topological_node_rank(
            [0, 0, 0], ",", fallback_index="6"
        ),
        6,
    )


class TopologicalNodeRankTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enterContext(
        mock.patch.object(hardware, "get_tpu_device_count", return_value=4)
    )

  def test_set_tpu_launch_env_with_coordinates(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "JOB_COMPLETION_INDEX": "5",
    }):
      with mock.patch.object(
          subprocess, "run", return_value=_coords_process([2, 0, 0])
      ):
        environment.set_tpu_launch_env()
      self.assertEqual(os.environ.get("NODE_RANK"), "1")

  def test_set_tpu_launch_env_with_coordinates_and_no_job_completion_index(
      self,
  ):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
    }):
      with mock.patch.object(
          subprocess, "run", return_value=_coords_process([0, 2, 0])
      ):
        environment.set_tpu_launch_env()
      self.assertEqual(os.environ.get("NODE_RANK"), "2")

  def test_set_tpu_launch_env_coordinate_lookup_failure_fallback(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "JOB_COMPLETION_INDEX": "3",
    }):
      with mock.patch.object(
          subprocess,
          "run",
          side_effect=subprocess.CalledProcessError(1, "python"),
      ):
        environment.set_tpu_launch_env()
      self.assertEqual(os.environ.get("NODE_RANK"), "3")

  def test_set_tpu_launch_env_empty_coordinates_fallback(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "JOB_COMPLETION_INDEX": "3",
    }):
      with mock.patch.object(
          subprocess, "run", return_value=_coords_process([])
      ):
        environment.set_tpu_launch_env()
      self.assertEqual(os.environ.get("NODE_RANK"), "3")

  def test_set_tpu_launch_env_out_of_bounds_coordinates_fallback(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "JOB_COMPLETION_INDEX": "7",
    }):
      with mock.patch.object(
          subprocess, "run", return_value=_coords_process([10, 0, 0])
      ):
        environment.set_tpu_launch_env()
      self.assertEqual(os.environ.get("NODE_RANK"), "7")

  def test_set_tpu_launch_env_without_tpu_devices_skips_query(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "JOB_COMPLETION_INDEX": "3",
    }):
      with (
          mock.patch.object(hardware, "get_tpu_device_count", return_value=0),
          mock.patch.object(subprocess, "run") as run,
      ):
        environment.set_tpu_launch_env()
      run.assert_not_called()
      self.assertEqual(os.environ.get("NODE_RANK"), "3")

  def test_set_tpu_launch_env_without_topology_skips_query(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "JOB_COMPLETION_INDEX": "3",
    }):
      with mock.patch.object(subprocess, "run") as run:
        environment.set_tpu_launch_env()
      run.assert_not_called()
      self.assertEqual(os.environ.get("NODE_RANK"), "3")

  def test_set_tpu_launch_env_ignores_auto_detected_topology(self):
    # The auto-detected topology describes this host alone, not the slice, so
    # it must not be used to rank hosts.
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "JOB_COMPLETION_INDEX": "1",
    }):
      with (
          mock.patch.object(
              hardware, "get_tpu_topology", return_value="2,2,1,2"
          ),
          mock.patch.object(subprocess, "run") as run,
      ):
        environment.set_tpu_launch_env()
      run.assert_not_called()
      self.assertEqual(os.environ.get("NODE_RANK"), "1")
      self.assertEqual(os.environ.get("TORCH_TPU_TOPOLOGY"), "2,2,1,2")

  def test_set_tpu_launch_env_keeps_pinned_node_rank(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "NODE_RANK": "1",
    }):
      with mock.patch.object(subprocess, "run") as run:
        environment.set_tpu_launch_env()
      run.assert_not_called()
      self.assertEqual(os.environ.get("NODE_RANK"), "1")

  def test_set_tpu_launch_env_singlehost_skips_query(self):
    with _mock_env({
        "TORCH_TPU_TOPOLOGY": "2,2,1",
    }):
      with (
          mock.patch.object(hardware, "get_tpu_topology", return_value="2,2,1"),
          mock.patch.object(subprocess, "run") as run,
      ):
        environment.set_tpu_launch_env()
      run.assert_not_called()
      self.assertIsNone(os.environ.get("NODE_RANK"))


class QueryLocalDeviceCoordsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enterContext(
        mock.patch.object(hardware, "get_tpu_device_count", return_value=4)
    )

  def test_without_tpu_devices_returns_none(self):
    with _mock_env({}):
      with (
          mock.patch.object(hardware, "get_tpu_device_count", return_value=0),
          mock.patch.object(subprocess, "run") as run,
      ):
        self.assertIsNone(internal_environment._query_local_device_coords())
      run.assert_not_called()

  def test_runs_this_module_in_child_process(self):
    with _mock_env({"FOO": "bar"}):
      with mock.patch.object(
          subprocess, "run", return_value=_coords_process([1, 2, 3, 0])
      ) as run:
        self.assertEqual(
            internal_environment._query_local_device_coords(), [1, 2, 3, 0]
        )
      run.assert_called_once()
      (cmd,), kwargs = run.call_args
      self.assertEqual(
          cmd,
          [
              sys.executable,
              "-m",
              "torch_tpu._internal.distributed.environment",
              "--print_local_device_coords",
          ],
      )
      self.assertTrue(kwargs["check"])
      self.assertTrue(kwargs["capture_output"])
      self.assertGreater(kwargs["timeout"], 0)
      self.assertEqual(kwargs["env"]["FOO"], "bar")
      self.assertEqual(
          kwargs["env"]["PYTHONPATH"].split(os.pathsep),
          [p for p in sys.path if p],
      )

  def test_ignores_other_stdout_lines(self):
    with _mock_env({}):
      with mock.patch.object(
          subprocess,
          "run",
          return_value=_coords_process(
              [2, 0, 1], prefix_lines="Registered Python module for 'tpu'.\n"
          ),
      ):
        self.assertEqual(
            internal_environment._query_local_device_coords(), [2, 0, 1]
        )

  def test_child_failure_returns_none(self):
    for error in [
        subprocess.CalledProcessError(1, "python"),
        subprocess.TimeoutExpired("python", 300),
        OSError("no python"),
    ]:
      with self.subTest(error=type(error).__name__):
        with _mock_env({}):
          with mock.patch.object(subprocess, "run", side_effect=error):
            self.assertIsNone(internal_environment._query_local_device_coords())

  def test_missing_coordinates_line_returns_none(self):
    with _mock_env({}):
      with mock.patch.object(
          subprocess,
          "run",
          return_value=subprocess.CompletedProcess(
              args=[], returncode=0, stdout="unrelated\n", stderr=""
          ),
      ):
        self.assertIsNone(internal_environment._query_local_device_coords())

  def test_malformed_coordinates_line_returns_none(self):
    with _mock_env({}):
      with mock.patch.object(
          subprocess,
          "run",
          return_value=subprocess.CompletedProcess(
              args=[],
              returncode=0,
              stdout="TORCH_TPU_LOCAL_DEVICE_COORDS=[1, 2\n",
              stderr="",
          ),
      ):
        self.assertIsNone(internal_environment._query_local_device_coords())

  def test_print_local_device_coords(self):
    with mock.patch(
        "torch_tpu._internal.distributed.tpu_distributed.local_device_coords",
        return_value=[2, 0, 1],
    ):
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        internal_environment._print_local_device_coords()
    self.assertEqual(
        buf.getvalue(), "TORCH_TPU_LOCAL_DEVICE_COORDS=[2, 0, 1]\n"
    )


if __name__ == "__main__":
  absltest.main()
